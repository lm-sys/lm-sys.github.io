# Humming 0.1.18 / SGLang reproduction notes

These notes accompany the November 4 H20-3e results and the November 10–11 review follow-up. Per-repetition values are in [results.json](results.json). The matched Humming/FlashInfer EP8 comparison was added during review; its per-repetition values and TTFT samples are in [paired-ep8-results.json](paired-ep8-results.json).

## Source and environment

- SGLang Python source: [`f644444260a6caf2cc4a4d5357dc17edaa6c8e64`](https://github.com/guzekai01/sglang/tree/f644444260a6caf2cc4a4d5357dc17edaa6c8e64). This includes the merged 0.1.18 upgrade (#46018) and the initial implementation subsequently proposed in #48524. Checking out this commit supplies the stack; do not also cherry-pick those PRs. The later PR head `65d9d7186d` includes cleanup and a main-branch merge and was not the source of the November 4 numbers.
- Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), language-only serving, including its bundled DSpark draft.
- Hardware: 8 × H20-3e, 143771 MiB per GPU; driver 570.133.20; CUDA 13.0 runtime/toolkit (NVCC 13.0.48); Python 3.12. The runtime includes CUDA compatibility libraries, with `/usr/local/cuda/compat/lib` ahead of the NVIDIA driver directories on `LD_LIBRARY_PATH`.
- Packages: PyTorch runtime build 2.14.1+cu130, `humming-kernels[cu13]==0.1.18`, FlashInfer Python/cubin 0.7.2 and JIT cache 0.7.2+cu130, `sglang-kernel==0.4.10+cu130`, Triton 3.8.0. A client-only PyArrow 25.0.1 override was used with NumPy 1.26.4.
- The server ran the checked-out source through `PYTHONPATH`, overriding the SGLang wheel installed in the runtime. The wheel version is not the evaluated Python source revision. Humming 0.1.19 is not validated by these measurements.

The original study used a Pod configured with 64 CPUs and 768 GiB RAM. The new matched EP8 study uses the same GPU model and runtime on another host, with the requested deployment template's 150-CPU / 1200-GiB limits. `OMP_NUM_THREADS=1` is used in both. Comparisons within each study share the host and resource limits; differences between the EP1 and new EP8 columns are not an isolated EP-only ablation.

The original launch used a local checkpoint path without an explicit Hugging Face revision. On November 10, the config and weight index on the original and new experiment nodes were checked and matched:

| File | SHA256 |
|---|---|
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

These identify the currently checked config and weight index, not a checksum of every weight shard or a retrospectively recorded download revision. Preserve the downloaded checkpoint revision in any new run.

## Server launches

Set `SGLANG_SOURCE` to the checkout above and `MODEL_PATH` to the local checkpoint. Use the same runtime and common arguments for all columns. The arrays below are Bash syntax.

```bash
export PYTHONPATH="$SGLANG_SOURCE/python"
export OMP_NUM_THREADS=1
export SGLANG_OPT_FP8_WO_A_GEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
unset SGLANG_HUMMING_INPUT_QUANT_CONFIG SGLANG_HUMMING_MOE_GEMM_TYPE

COMMON=(
  --model-path "$MODEL_PATH" --served-model-name dsv41-flash-1017
  --host 127.0.0.1 --port 31018 --trust-remote-code --language-model-only
  --tp-size 8 --enable-prefill-cp --cp-strategy interleave
  --enable-cp-decode-attn-tp --moe-a2a-backend none
  --engram-config '{"placement":"cpu","replication":"none"}'
  --no-enable-decoder-swa-bounded-replay --swa-full-tokens-ratio 0.05
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --mem-fraction-static 0.84 --max-running-requests 64
  --cuda-graph-max-bs-decode 64 --chunked-prefill-size 16384
  --enable-cache-report --enable-metrics
  --reasoning-parser auto --tool-call-parser auto --random-seed 1
)
# TTFT: keep radix caching enabled. For the ShareGPT decode table only:
# COMMON+=(--disable-radix-cache --stream-interval 1)
```

Start one server at a time, with one of these selections:

```bash
# Marlin MXFP4A16, TP8/EP1
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend marlin --speculative-moe-runner-backend marlin

# Humming MXFP4A16, TP8/EP1
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"bfloat16"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# Humming MXFP4AFP8, TP8/EP1
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# Humming MXFP4AFP8, TP8/EP8
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 8 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# FlashInfer MXFP4AFP8, TP8/EP8
python -m sglang.launch_server "${COMMON[@]}" --ep-size 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-moe-runner-backend flashinfer_mxfp4 \
  --flashinfer-mxfp4-moe-precision fp8
```

The Humming AFP8 override produces dynamic per-token FP8 scales (`group_size=0`) for expert inputs. The checkpoint's MXFP4 weight group size does not become an activation group size. Per-token scaling applies to each rank's local GEMM input, whose K extent can change with parallelism. The linked FP8_BLOCK optimizations are not the measured path. Both target and draft use the selected backend; dense layers retain the native FP8 implementation.

## Long-input TTFT

In a separate client shell, use the same source and the client-only PyArrow override:

```bash
export HUMMING_BENCH_CLIENT_DEPS=/tmp/humming-bench-client-deps
python -m pip install --target "$HUMMING_BENCH_CLIENT_DEPS" --no-deps pyarrow==25.0.1
export PYTHONPATH="$HUMMING_BENCH_CLIENT_DEPS:$SGLANG_SOURCE/python"
```

Use the radix-enabled server. For each input length, run one warmup followed by a cache flush and three measured requests at concurrency 1. Inputs are exact-length synthetic token IDs. Report the median of the three measured TTFTs. A cold prefix cache does not mean a cold JIT/compiler cache.

```bash
for LENGTH in 4096 65536 131072 262144 524288 1040384; do
  python -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port 31018 \
    --model "$MODEL_PATH" --served-model-name dsv41-flash-1017 \
    --dataset-name random-ids --random-input-len "$LENGTH" \
    --random-output-len 32 --random-range-ratio 1 \
    --num-prompts 3 --max-concurrency 1 --request-rate inf \
    --seed 1024 --temperature 0 --warmup-requests 1 --flush-cache \
    --cache-report --output-details --tokenize-prompt \
    --output-file "ttft-${LENGTH}.jsonl"
done
```

## ShareGPT-derived decode workload

Dataset: `ShareGPT_V3_unfiltered_cleaned_split.json` from [anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/192ab2185289094fc556ec8ce5ce1e8e587154ca), revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`; file SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

The pinned SGLang `random` dataset shuffles conversations and tokenizes the first user turn, then truncates or repeats those token IDs to exactly 4096 tokens. No chat template is applied. This is a fixed-shape workload derived from real text, not an unmodified ShareGPT conversation replay. The server has radix caching disabled. Before each measured run, use a separate full-1024-output-token warmup; the native client's additional warmup is disabled. Streaming and ignore-EOS follow the pinned client's defaults.

```bash
export SGLANG_IS_IN_CI=0
for C in 1 32 64; do
  N=$((5 * C)); (( N < 10 )) && N=10
  for REPEAT in 1 2 3; do
    for PHASE in warmup measured; do
      COUNT=$N; SEED=1
      if [[ "$PHASE" == warmup ]]; then COUNT=$C; SEED=0; fi
      python -m sglang.benchmark.serving \
        --backend sglang --host 127.0.0.1 --port 31018 \
        --model "$MODEL_PATH" --served-model-name dsv41-flash-1017 \
        --dataset-name random --dataset-path "$SHAREGPT_PATH" \
        --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
        --num-prompts "$COUNT" --max-concurrency "$C" --request-rate inf \
        --seed "$SEED" --temperature 0 --warmup-requests 0 \
        --cache-report --output-details --tokenize-prompt \
        --output-file "decode-c${C}-r${REPEAT}-${PHASE}.jsonl"
    done
  done
done
```

The original EP1 study ran all repetitions under one server launch per backend. The new matched EP8 study restarted the server for each repetition and alternated backend order: Humming/FlashInfer, FlashInfer/Humming, Humming/FlashInfer. For that study, run one repetition of all three concurrency points per launch, keeping the warmup and measured commands above unchanged. Compiler caches are retained. TTFT uses a separate radix-enabled server launch per backend.

Output throughput is the client's whole-run output-token throughput, including prefill/serving overhead. TTFT and TPOT are separate client metrics. Acceptance is obtained from request metadata, not inferred from streamed chunk sizes. Across backends, input token-ID hashes, requested/completed token counts, zero measured cache hits, and positive acceptance values were checked.

## Quality and statistical interpretation

The existing quality comparison uses all 1319 official GSM8K test questions, the first eight official train rows as demonstrations, the pinned SGLang scorer, chat API, `thinking=false`, temperature 0, maximum 1024 generated tokens, and eight concurrent requests. The train demonstrations are disjoint from the test questions.

| Configuration (EP1) | Correct / total | Accuracy |
|---|---:|---:|
| Marlin A16 | 1281 / 1319 | 97.119% |
| Humming A16 | 1280 / 1319 | 97.043% |
| Humming AFP8 | 1277 / 1319 | 96.816% |

These are single full evaluations, not three quality repetitions. AFP8 is four questions / 0.303 percentage points below Marlin on this evaluation. This does not establish accuracy equivalence or long-context quality. There is no FlashInfer quality score in the original dataset.

For this quality run, use the corresponding EP1 server with radix caching enabled. Obtain the official [GSM8K train/test JSONL files](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data), validate the hashes in `results.json`, and concatenate the first eight train rows followed by all 1319 test rows into `gsm8k-8train-1319test.jsonl`. The pinned scorer consumes the first eight rows as demonstrations and evaluates the remaining rows:

```bash
python -m sglang.test.run_eval \
  --base-url http://127.0.0.1:31018 --model dsv41-flash-1017 \
  --eval-name gsm8k --api chat --num-examples 1319 --num-shots 8 \
  --num-threads 8 --temperature 0 --max-tokens 1024 \
  --chat-template-kwargs '{"thinking":false}' \
  --gsm8k-data-path gsm8k-8train-1319test.jsonl
```

For performance, report the mean and sample standard deviation (`ddof=1`) of the three repetition-level metrics. Do not treat all individual requests as independent benchmark repetitions. Acceptance below is the mean of the per-request acceptance values within each run, averaged over runs; it is not pooled tokens divided by pooled verification counts.

## Scale-factoring example

Take group scales `[1, 2, 4, 8]`, whose UE8M0 exponent bytes are `[127, 128, 129, 130]`. The exponent span is 3, so no scale-range adjustment is needed. Residual exponents are `[0, 1, 2, 3]`; the FP8 fused encoding stores residuals plus one. In the scale-4 group, FP4 `1.5` has magnitude code 3. The fused register lookup produces E4M3 byte `0x24`, or `0.1875`. The prepared secondary scale is `1/2`, and the FP4-to-E4M3 exponent-bias correction contributes `2^6`, yielding a remaining factor of 32. Thus `0.1875 × 32 = 1.5 × 4 = 6`. The dynamic activation scale multiplies the accumulator separately.

For E4M3, `process_fused_e8m0_scale` permits an exponent span of 11. If the span is larger, the new lower exponent is `max(old_min, old_max - 11)`. Groups below it have their scale raised and FP4 values divided by the matching power of two, then rounded back to FP4. Values can round to zero. A separate load-only audit counts actual scale groups with a positive exponent adjustment; these counts are distinct from the number of individual FP4 values that change after rounding. No instrumentation was present during the performance runs.

Sources: [weight preparation](https://github.com/vllm-project/humming/blob/v0.1.18/humming/transform.py), [FP4 requantization](https://github.com/vllm-project/humming/blob/v0.1.18/humming/include/humming/kernel/process_mxfp4.cuh), [fused register conversion](https://github.com/vllm-project/humming/blob/v0.1.18/humming/include/humming/datatype/dequant_fused.cuh), and [exponent offsets](https://github.com/vllm-project/humming/blob/v0.1.18/humming/include/humming/arith/exp_offset.cuh).

## Decode metrics from the original study

Values are mean ± sample SD across repetition-level summaries. TTFT and TPOT are in milliseconds.

| Configuration | EP | C | Output tokens/s | TTFT (ms) | TPOT (ms) | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| marlin_a16 | 1 | 1 | 323.688 ± 3.647 | 321.238 ± 0.687 | 2.776 ± 0.036 | 5.139 ± 0.058 |
| marlin_a16 | 1 | 32 | 1661.537 ± 22.216 | 1558.744 ± 1.860 | 16.768 ± 0.079 | 5.159 ± 0.035 |
| marlin_a16 | 1 | 64 | 2044.123 ± 5.568 | 3435.942 ± 29.612 | 26.989 ± 0.050 | 5.205 ± 0.004 |
| humming_a16 | 1 | 1 | 314.350 ± 3.171 | 263.333 ± 1.423 | 2.924 ± 0.031 | 5.115 ± 0.047 |
| humming_a16 | 1 | 32 | 1721.587 ± 6.243 | 1294.003 ± 37.182 | 16.335 ± 0.061 | 5.153 ± 0.022 |
| humming_a16 | 1 | 64 | 2200.401 ± 13.730 | 2900.157 ± 45.025 | 25.159 ± 0.036 | 5.211 ± 0.001 |
| humming_a8 | 1 | 1 | 367.132 ± 5.045 | 207.182 ± 1.233 | 2.522 ± 0.039 | 5.142 ± 0.051 |
| humming_a8 | 1 | 32 | 2314.060 ± 17.986 | 1001.886 ± 4.787 | 12.234 ± 0.093 | 5.191 ± 0.029 |
| humming_a8 | 1 | 64 | 2865.836 ± 7.364 | 2231.068 ± 25.516 | 19.354 ± 0.021 | 5.201 ± 0.004 |
| flashinfer_ep8_fp8 | 8 | 1 | 314.567 ± 12.327 | 211.246 ± 4.385 | 2.976 ± 0.125 | 5.090 ± 0.130 |
| flashinfer_ep8_fp8 | 8 | 32 | 2262.351 ± 24.219 | 1002.373 ± 34.567 | 12.358 ± 0.068 | 5.202 ± 0.032 |
| flashinfer_ep8_fp8 | 8 | 64 | 2895.274 ± 3.690 | 2261.263 ± 1.271 | 19.071 ± 0.005 | 5.202 ± 0.008 |

## Matched EP8 decode metrics

Values are mean ± sample SD across three independently restarted servers per backend. All six runs used the same host and request inputs.

| Configuration | C | Output tokens/s | TTFT (ms) | TPOT (ms) | Acceptance |
|---|---:|---:|---:|---:|---:|
| humming_ep8_fp8 | 1 | 324.513 ± 4.550 | 233.101 ± 0.878 | 2.854 ± 0.044 | 5.023 ± 0.058 |
| humming_ep8_fp8 | 32 | 2169.437 ± 39.718 | 1098.232 ± 15.223 | 12.863 ± 0.113 | 5.170 ± 0.018 |
| humming_ep8_fp8 | 64 | 2601.590 ± 9.364 | 2393.403 ± 19.012 | 21.339 ± 0.039 | 5.213 ± 0.005 |
| flashinfer_ep8_fp8 | 1 | 315.392 ± 6.733 | 208.612 ± 2.625 | 2.968 ± 0.065 | 5.089 ± 0.091 |
| flashinfer_ep8_fp8 | 32 | 2261.615 ± 50.402 | 1019.455 ± 10.509 | 12.469 ± 0.130 | 5.179 ± 0.033 |
| flashinfer_ep8_fp8 | 64 | 2882.960 ± 17.219 | 2279.674 ± 16.318 | 18.990 ± 0.030 | 5.221 ± 0.013 |

At C1, Humming has higher output throughput despite lower acceptance on average; at C32/C64 the acceptance means are close, while FlashInfer has higher throughput. These request-level acceptance measurements do not by themselves explain kernel execution time.

## Load-only scale audit (TP8/EP1)

The audit observes `process_fused_e8m0_scale` immediately before its original transformation. It counts positive scale shifts over valid groups across all eight ranks, excluding padded groups from both numerator and denominator. It also computes a counterfactual range from valid scales only. The installed Humming package and timed servers are unchanged; a separate package copy adds a read-only load hook. The AFP8 long-context quality run uses that copy after the original transformations complete. The audit verifies dynamic per-token E4M3 inputs in all 688 target/draft sublayer records.

| Model component | Valid scale groups | Groups with positive shift | Fraction |
|---|---:|---:|---:|
| Target | 16,986,931,200 | 0 | 0% |
| DSpark draft | 424,673,280 | 47,868,431 | 11.27% |

All affected groups are in the draft's fused gate/up projection (`w13`); none are in the down projection. At EP1, `w13` pads its N dimension from 576 to 768. Padding scales have numeric value 1 (UE8M0 byte 127), and participate in per-expert range selection. Computing the range from valid scales only would adjust 5,308 groups; padding adds 47,863,123 affected groups. This is a range calculation on the observed scales, not an alternate model-quality experiment. The down-projection K padding uses the existing row maximum and adds no affected groups.

The combined target/draft rate is 0.275%, but that aggregate hides the concentration in the draft. These counts measure groups with positive exponent shifts, not the count of individual FP4 elements that change numerically after rounding. They do not establish the cause of the GSM8K or acceptance differences; that would require a controlled numerical ablation.

## Long-context quality protocol

The supplemental quality task is [RULER v1 `niah_multikey_2`](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a), not the full RULER suite or RULERv2. It uses synthetic word keys and numeric values, 200 samples at each nominal 32K and 128K length, seed 42, and the checkpoint tokenizer. Inputs are fixed before any backend evaluation. Generate them with the pinned repository's `scripts/data/prepare.py`:

```bash
for LENGTH in 32768 131072; do
  python "$RULER_SOURCE/scripts/data/prepare.py" \
    --save_dir "ruler-data/$LENGTH" --benchmark synthetic \
    --task niah_multikey_2 --subset validation \
    --tokenizer_path "$MODEL_PATH" --tokenizer_type hf \
    --max_seq_length "$LENGTH" --num_samples 200 \
    --random_seed 42 --model_template_type base
done
```

Generation uses Transformers 5.19.0, Tokenizers 0.23.2, Wonderwords 2.2.0, and NumPy 2.2.6 in a separate client environment. Validate the generated JSONL files:

| Nominal length | SHA256 |
|---|---|
| 32K | `8d5c8b9c4d26b89dd79f5a29d86c1fc097a80241259702ca3bf992f78a7e1200` |
| 128K | `1123ed453161b74c98d058eeda120ac95bdec0ac14cee7526c8f3a89b8652071` |

Use each EP1 server configuration above with radix caching disabled. Send each row's `input` as one user message to `/v1/chat/completions`, with `thinking=false`, temperature 0, `max_tokens=128`, and concurrency 8. Score the returned content against the row's `outputs` using the repository's official `string_match_all` from `scripts/eval/synthetic/constants.py`. The generator's nominal budget includes 128 output tokens; actual chat prompt lengths are recorded separately. This is a targeted retrieval check at two lengths, not evidence of broad long-context reasoning accuracy or quality at the 1016K performance point.

The completed retrieval results are:

| Configuration (EP1) | Nominal 32K | Nominal 128K | Empty / length-limited responses |
|---|---:|---:|---:|
| Marlin A16 | 200 / 200 (100%) | 200 / 200 (100%) | 0 / 0 |
| Humming AFP8 | 200 / 200 (100%) | 200 / 200 (100%) | 0 / 0 |

Both completed configurations have actual chat prompt lengths of 32,029–32,630 and 130,285–130,932 tokens. Humming A16's run ended when rank 0 received SIGKILL after 168 responses in the first cell; no full-cell score is reported. The Pod did not restart, its cgroup OOM counter was zero, and the kill's cause remains unresolved. A subsequent GPU workload occupied the host, preventing a clean replay under the required exclusive-GPU conditions. This is an incomplete validation, not an accuracy score. The original full GSM8K scores remain unchanged.

The matched EP8 configurations have complete performance measurements but no completed GSM8K evaluations, and the EP8 load audit is also outstanding. Do not transfer the EP1 quality or requantization counts to EP8.

## Routed-expert GEMM attribution

We captured CPU/GPU Torch Profiler traces on all eight ranks of the Marlin A16 EP1 server, separately from the serving measurements. Each case uses one warmed measured request with radix caching disabled and the original first measured input-token hash: prefill at 4K / 128K / 1016K with 32 generated tokens, and C1 ShareGPT-derived 4K input with 128 generated tokens. SGLang profiler v1 (`SGLANG_PROFILE_V2=0`) separates stages; the prefill analysis uses only the eight `EXTEND` files, and C1 uses only the eight `DECODE` files. Unsuffixed decode-tail files from prefill captures are excluded. Profiling latency is not used in the performance tables.

[Per-rank results](marlin-profile-summary.json) retain all eight ranks. “Routed-expert GEMM” counts the `sglang::device::marlin_moe::Marlin` kernels selected by the pinned [MoE runner](https://github.com/guzekai01/sglang/blob/f644444260a6caf2cc4a4d5357dc17edaa6c8e64/python/sglang/srt/layers/moe/moe_runner/marlin.py) and [implementation](https://github.com/guzekai01/sglang/blob/f644444260a6caf2cc4a4d5357dc17edaa6c8e64/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py). Routing, activation, quantization, shared experts, and communication are outside this numerator.

| Input | Routed GEMM time per rank, mean | Share of summed kernel durations, rank mean (range) | All-kernel interval union per rank, mean |
|---|---:|---:|---:|
| 4K | 0.198 s | 56.28% (48.83–63.35%) | 0.354 s |
| 128K | 4.717 s | 47.52% (47.38–47.73%) | 9.925 s |
| 1016K | 37.379 s | 19.85% (19.82–19.94%) | 188.277 s |

These are rank averages from a single profiled request per length, not repeated timing estimates. The denominator sums kernel durations; it is neither client TTFT nor a complete MoE-time attribution. Interval unions are reported separately so overlapping kernels are not mistaken for additional wall time. The 4K rank spread includes communication/wait variation during profiling. This establishes a shrinking routed-GEMM share in the baseline; it does not identify all growing non-GEMM work as attention or measure Humming's kernel-level speedup.

For C1, each rank captures ten target verify forwards and ten draft forwards, containing 800 target and 60 draft routed GEMMs. Their mean summed GEMM time is **4.213 ms per target verify** and **0.181 ms per draft forward**, averaged across ranks. All-kernel duration sum is 165.253 ms per rank, versus a 144.144 ms interval union, demonstrating overlap. The launch has DSpark block size 5 and six target verification tokens per request; CUDA graph padding also affects executed shapes. Concurrency 1 or block size 5 must not be assumed to mean GEMM M=1 or M=5.

The Humming A16 trace is incomplete following the quality-run interruption. The cause of the published C1 backend difference and the benefit of autotuning remain unverified; no kernel-selection or autotuning speedup is claimed.
