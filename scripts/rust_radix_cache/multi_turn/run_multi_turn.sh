#!/usr/bin/env bash
# Multi-turn TTFT: Rust vs Python unified radix cache.
# Launches one SGLang server (model x backend), runs the multi-turn driver, writes a JSON.
#
# Usage:   SGLANG_DIR=/path/to/sglang ./run_multi_turn.sh <model> <backend>
#   model:   full | swa | mamba   (or an explicit --model-path value)
#   backend: rust | python
#
# Backend selection:
#   rust   -> --radix-cache-backend rust_unified_tree   (RustUnifiedRadixCache)
#   python -> SGLANG_ENABLE_UNIFIED_RADIX_TREE=1         (Python UnifiedRadixCache)
# Both register the model's components (FULL / SWA / MAMBA) automatically, so the two
# arms are an apples-to-apples unified-vs-unified comparison.
#
# NOTE: build the Rust extension in RELEASE (see README "Build") — an editable
# `pip install -e .` builds it in debug, which is markedly slower and skews the result.
set -euo pipefail
SGLANG_DIR="${SGLANG_DIR:?set SGLANG_DIR to your sglang checkout}"
MODEL_KEY="${1:?model: full|swa|mamba}"; BACKEND="${2:?backend: rust|python}"

case "$MODEL_KEY" in
  full)  MODEL="Qwen/Qwen3-32B"
         SRV="--tp-size 2 --page-size 64 --mem-fraction-static 0.80 --context-length 40960" ;;
  swa)   MODEL="openai/gpt-oss-20b"
         SRV="--tp-size 2 --ep-size 2 --page-size 16 --mem-fraction-static 0.85 --context-length 32768" ;;
  mamba) MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
         SRV="--tp-size 4 --mamba-scheduler-strategy extra_buffer --page-size 64 --mem-fraction-static 0.80 --context-length 40960" ;;
  *)     MODEL="$MODEL_KEY"; SRV="" ;;
esac

if [ "$BACKEND" = rust ]; then BFLAG="--radix-cache-backend rust_unified_tree"; UENV=""
else                          BFLAG="";                                         UENV="SGLANG_ENABLE_UNIFIED_RADIX_TREE=1"; fi

OUT="ttft_${MODEL_KEY}_${BACKEND}.json"
echo "[launch] model=$MODEL backend=$BACKEND -> $OUT"
env $UENV python -m sglang.launch_server --model-path "$MODEL" $SRV $BFLAG \
  --host 127.0.0.1 --port 30000 > "server_${MODEL_KEY}_${BACKEND}.log" 2>&1 &
SRV_PID=$!
trap 'kill -9 $SRV_PID 2>/dev/null || true' EXIT

echo "[wait] for server readiness..."
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break; sleep 5; done
# Sanity-check the cache that actually loaded (grep the banner):
grep -E "Tree cache initialized" "server_${MODEL_KEY}_${BACKEND}.log" | tail -1 || true

echo "[bench] 100-tok input, 100-tok output/turn, 200 turns"
python "$SGLANG_DIR/benchmark/multi_turn_serving/bench_multi_turn_serving.py" \
  --server-url http://127.0.0.1:30000 \
  --input-tokens 100 --output-tokens-per-turn 100 --max-num-turns 200 --num-trials 6 \
  --output-json "$OUT"
echo "[done] $OUT"
