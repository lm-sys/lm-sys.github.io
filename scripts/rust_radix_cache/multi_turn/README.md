# Multi-turn TTFT: Rust vs Python unified radix cache

Reproduction scripts for the multi-turn Time-to-First-Token (TTFT) comparison in the
[Unified Radix Cache blog post](../../../blog/2026-08-10-unified-radix-cache.md) (the
*Toward a Rust Tree Core* section, Figure 7).

We drive a synthetic multi-turn conversation (100-token input, 100-token output per
turn, 200 turns) and record per-turn TTFT as the conversation — and thus the radix
tree — deepens. The same workload runs against two cache backends that are otherwise
identical (same model, same flags):

| backend | how it's selected | implementation |
|---------|-------------------|----------------|
| Rust    | `--radix-cache-backend rust_unified_tree` | `RustUnifiedRadixCache` |
| Python  | `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`      | `UnifiedRadixCache` |

Both register the model's components (FULL / SWA / MAMBA) automatically, so it's an
apples-to-apples *unified-vs-unified* comparison. We run three architectures whose
attention shapes stress the cache differently:

| key   | model | shape |
|-------|-------|-------|
| `full`  | `Qwen/Qwen3-32B`                       | full attention (TP2) |
| `swa`   | `openai/gpt-oss-20b`                   | sliding-window attention (TP2) |
| `mamba` | `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` | hybrid SSM / linear attention (TP4) |

## Build (release matters)

Build the Rust radix-cache extension (`_mem_cache_core`) in **release**. An editable
`pip install -e .` builds it in **debug** by default, which makes the per-node tree
walk/insert markedly slower and skews the Rust-vs-Python comparison — the built `.so`
should be ~1 MB, not ~40 MB. Either install a release wheel, or set `debug = false` on
the `[[tool.setuptools-rust.ext-modules]]` entry for `_mem_cache_core` in
`python/pyproject.toml` and reinstall. The crate's `tch` version must match the
installed PyTorch (e.g. `tch = "0.24"` for torch 2.11); a mismatch otherwise needs
`LIBTORCH_BYPASS_VERSION_CHECK` and is unstable.

## Run

```bash
export SGLANG_DIR=/path/to/sglang        # checkout containing benchmark/multi_turn_serving/

# one (model, backend) per invocation — runs server + 200-turn bench, writes ttft_<model>_<backend>.json
./run_multi_turn.sh full  rust
./run_multi_turn.sh full  python
./run_multi_turn.sh swa   rust
./run_multi_turn.sh swa   python
./run_multi_turn.sh mamba rust
./run_multi_turn.sh mamba python

# per-model 2-panel chart + per-25-turn-bin table
python analyze.py ttft_full_rust.json  ttft_full_python.json  "Qwen3-32B FULL, TP2"          ttft_full.png
python analyze.py ttft_swa_rust.json   ttft_swa_python.json   "gpt-oss-20b SWA, TP2"         ttft_swa.png
python analyze.py ttft_mamba_rust.json ttft_mamba_python.json "Qwen3-Next-80B Mamba, TP4"    ttft_mamba.png
```

The two arms run sequentially on the same GPUs (one server at a time). The `mamba`
shape uses all four GPUs (TP4), so it cannot overlap with the others.

## GPU-prefill decomposition (Figure 7)

The blog figure shows each turn's total TTFT and its **GPU prefill** interval. Their
difference contains the radix-cache walk, scheduling, synchronization, sampling,
detokenization, transport, and other uninstrumented work. It is not a direct CPU
measurement, and the complete backend difference cannot be attributed only to the
cache. GPU prefill is measured with `torch.cuda.Event` timers around the prefill
forward, read back off the critical path (gated on `event.query()`) so they do not
perturb the overlap scheduler, and surfaced as a `gpu_prefill_ms` field in the
response `meta_info`.

`analyze.py` uses `gpu_prefill_ms` when present and falls back to a
**total-TTFT-only** table and plot when it is absent. The total TTFT comparison is
reproducible against stock SGLang, while the timing decomposition requires the
CUDA-event instrumentation.
