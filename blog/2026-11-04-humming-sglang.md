---
title: "Humming in SGLang: Flexible Low-Bit Kernels for Efficient MoE Serving"
author: "Ant Group Venus and SCT Teams, and the SGLang Team"
date: "November 4, 2026"
previewImg: /images/blog/humming-sglang/humming_sgl_preview.png
type: blog
---

[Humming](https://github.com/vllm-project/humming/tree/v0.1.18) is a lightweight, JIT-compiled kernel library for quantized dense and mixture-of-experts (MoE) inference. Its SGLang integration connects low-bit checkpoints to specialized kernels while preserving the framework’s routing, scheduling, and distributed execution.

With Humming 0.1.18, that integration extends across more quantization schemes and GPU architectures. This post explains how Humming prepares weights, processes activations, and maps computation to Hopper and Blackwell. We also introduce a shared activation-expression interface for SGLang’s MoE runners and show results with that integration enabled.

On [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), Humming delivers **1.07–1.23× prefill speedups at the same A16 precision** relative to Marlin in the tested long-context cases. We evaluate FP8 expert activations separately as a deployment option, alongside their quality and speculative-decoding acceptance results.

## One Backend Across Quantized Representations

MoE inference combines two challenges: large expert-weight storage and a changing number of tokens assigned to each expert. Four-bit weights reduce storage and memory traffic, but efficient serving also requires suitable dequantization, Tensor Core instructions, and enough parallel work. Prefill and decode can favor different execution strategies for the same checkpoint.

Humming separates the representation of quantized data from its execution strategy. Compatible formats share components for data movement, dequantization, matrix multiplication, and output processing. The library supports dense and MoE GEMM, low-bit integer and floating-point weights, and multiple activation precisions. Supported combinations depend on the GPU, scale format, and tensor layout; the [versioned configuration guide](https://github.com/vllm-project/humming/blob/v0.1.18/docs/config.md) defines those contracts.

A checkpoint can retain its compact representation while the runtime selects a compatible computation path for the device and workload.

## Prepare Once, Specialize for the Workload

At model load time, Humming converts supported checkpoint tensors and metadata into common weight and input schemas. Padding, transposition, interleaving, and scale rearrangement prepare the tensors for execution outside the request hot path.

Three configuration layers keep these responsibilities explicit. `LayerConfig` describes shapes, numeric formats, quantization, and storage layouts. `ComputeConfig` describes dense, indexed, or grouped execution and accumulation semantics. `TuningConfig` selects tile shapes, data transfers, pipeline stages, and scheduling. Compatible compute choices can reuse prepared weights; changing a storage layout or target architecture requires preparing them again.

At inference time, the unified [`process_input`](https://github.com/vllm-project/humming/pull/68) interface handles compatible combinations of activation evaluation, Hadamard transforms, layout processing, and quantization. Input quantization includes static tensor scales, dynamic token or group scales, and supported secondary-scale combinations. The selected operations depend on the layer; some scale calculations require multiple passes.

Humming then composes and JIT-compiles the selected GEMM. Device- and shape-specific choices balance parallelism, weight reuse, and pipeline depth, with compiled binaries cached for subsequent calls. The [0.1.18 release](https://github.com/vllm-project/humming/releases/tag/v0.1.18) refines these choices, including MoE scheduling and weight eviction. This avoids precompiling every format and serving shape.

## Hopper: Compact MXFP4 Storage, FP8 Computation

A representative Hopper path combines MXFP4 weights with per-token FP8 activations. Weights remain packed at four bits in global and shared memory; Humming reconstructs FP8 operands immediately before Tensor Core computation.

<p align="center">
  <img src="/images/blog/humming-sglang/humming_hopper_w4a8_case.png" width="98%" alt="Hopper MXFP4AFP8 execution with load-time scale factoring, register-side FP8 reconstruction, and WGMMA">
</p>

<p align="center">
  <em>Figure 1. A representative per-token FP8 path: prepare scales at load time, reconstruct weights in registers, and apply the remaining scales after accumulation.</em>
</p>

The path has three key steps:

1. **Factor scales during loading.** Conceptually, each UE8M0 group scale is decomposed as `s_g = s_base × 2^Δg`. This path retains a base scale per expert and sublayer, plus a residual exponent code for each group. The encoding includes offsets needed for reconstruction. If the exponent span exceeds the supported FP8 range, preparation adjusts and requantizes affected FP4 values.
2. **Reconstruct FP8 weights in registers.** FP4 nibbles and residual exponents form FP8 E4M3 operands immediately before MMA. A transposed mapping places weight registers in WGMMA’s A operand and the shared-memory activation tile in its B operand, avoiding a materialized eight-bit weight tensor in memory.
3. **Apply the remaining scales in the epilogue.** FP8 WGMMA accumulates into FP32. The epilogue applies the activation scale and factored base scale; residual exponents are already incorporated into the weight operands.

For example, take four group scales `[1, 2, 4, 8]`, whose base scale `s_base=1` leaves residual exponents `[0, 1, 2, 3]`. An FP4 value of `1.5` in the scale-4 group is rebuilt as the E4M3 operand `0.1875`; the remaining weight factor is `32`, restoring `0.1875 × 32 = 6 = 1.5 × 4`. These values include the fixed FP4-to-FP8 exponent-bias adjustment. The activation scale is applied separately.

For E4M3, this path supports an exponent span of **11** without scale-range adjustment. Larger spans raise the minimum scale to `max_scale / 2^11`; weights in affected groups are divided by the corresponding power of two and rounded back to FP4, which can introduce rounding error or underflow.

For this checkpoint at TP8/EP1, a load-only audit found no adjusted groups in the target and **11.27% in the draft**, mainly because padding participates in scale-range selection. The [measurement notes below](#benchmark-details) give the exact group counts and their scope.

The [fused E8M0 configuration](https://github.com/vllm-project/humming/blob/v0.1.18/docs/config.md#use_fused_e8m0_scale) connects preparation with execution. Other scale layouts use their corresponding kernels and preparation rules. Humming 0.1.18 also improves the FP8_BLOCK [grouped-contiguous](https://github.com/vllm-project/humming/pull/95) and [indexed](https://github.com/vllm-project/humming/pull/115) MoE paths. Those use block-scaled activations; the AFP8 results below use **dynamic per-token E4M3 scales (`group_size=0`)**, so they do not measure these block-scale paths.

## Blackwell: Additional Native Compute Paths

All measured results in this post are on Hopper H20-3e; this section describes Blackwell architecture support.

Humming’s architecture extends to Blackwell through [BF16 UMMA](https://github.com/vllm-project/humming/pull/71) and eligible [FP8, FP4, and INT8 UMMA paths](https://github.com/vllm-project/humming/pull/105). Supported SM10x configurations can use UMMA, while supported SM12x block-scaled configurations use MXMMA.

These are device- and format-specific choices. INT8 UMMA is not available on every Blackwell device, and scale layouts must match the selected instruction path. The same configuration framework handles these constraints alongside tile and pipeline selection for different serving shapes.

## Connecting Humming to SGLang

The original integration arrived through [SGLang PR #23754](https://github.com/sgl-project/sglang/pull/23754), and [PR #46018](https://github.com/sgl-project/sglang/pull/46018) updated it for Humming 0.1.18. SGLang owns expert routing, dispatch, token alignment, output reduction, and token order; Humming supplies compatible input processing and GEMM execution.

Quantization loading and expert execution are related but distinct responsibilities. SGLang can use Humming for compatible quantized linear layers, or select its MoE runner while retaining another native path for dense layers. It reads checkpoint metadata and preserves the input scales needed by the selected scheme. Choosing Humming does not automatically convert A16 activations to FP8, and static quantization still requires calibrated scales.

The proposed update in [SGLang PR #48524](https://github.com/sgl-project/sglang/pull/48524) connects gated MoE activations to `process_input` through optional `activation_type` and `activation_impl` fields. The caller specifies either split or interleaved gate/up inputs and supplies a CUDA expression for the activation. Built-in SiLU, clamped SiLU, erf-based GELU, and SiTU share this interface.

For the down projection, compatible activation evaluation, Hadamard transformation, and quantization can therefore be expressed through one input-processing path. The implementation preserves BF16/FP16 rounding boundaries and avoids an unnecessary intermediate activation buffer when activation and quantization can be combined. This interface lets model-specific activations reuse the same input-processing path.

Humming supports native MXFP4 expert execution for DeepSeek-V4-Flash, DeepSeek-V4-Pro (including V4-Pro-0813), and DeepSeek-V4.1-Flash. The proposed shared input-processing interface also covers the gated activation paths used by GLM-5.2/5.3 and Kimi-K2.x/K3. Its Kimi-K3 support includes the native MXFP4 loader bridge adapted from the still-open [PR #33884](https://github.com/sgl-project/sglang/pull/33884).

A separate proposed fix in [PR #49873](https://github.com/sgl-project/sglang/pull/49873) makes standard EP dispatch estimate each rank's routed work for GEMM tuning while fully counting replicated shared experts. The Humming EP8 measurements below include this fix.

The end-to-end results in this post cover **DeepSeek-V4.1-Flash-1017 only**. Compatible expert-parallel deployments can also use [DeepEP FP8 dispatch](https://github.com/sgl-project/sglang/pull/31429), carrying quantized activations and their scales into Humming.

## Using Humming in SGLang

Base Humming 0.1.18 support landed in SGLang commit [ca939407](https://github.com/sgl-project/sglang/commit/ca939407423088dffd33304e0ea7d092cba355d2). The benchmarks below also include the proposed input-processing integration, with the EP tuning fix applied to the Humming EP8 configuration. Exact source revisions are listed in [Benchmark Details](#benchmark-details). The reported environment pins Humming **0.1.18**; 0.1.19 has not been validated for these results.

Install the compiler extra matching your CUDA major version:

```bash
# CUDA 13
pip install "humming-kernels[cu13]==0.1.18"
# CUDA 12: use this instead
# pip install "humming-kernels[cu12]==0.1.18"
```

Choose the scope of the backend. `--moe-runner-backend humming` selects expert execution where the checkpoint's native quantization path supports Humming. `--quantization humming` selects Humming's model-level quantization adapter for compatible linear and expert layers. For a compatible MoE model using that adapter, select both.

To retain a checkpoint's native dense-layer path while explicitly using FP8 E4M3 activations in its Humming experts:

```bash
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server \
  --model-path /path/to/supported-checkpoint \
  --moe-runner-backend humming
```

Add the model's usual parallelism arguments. When using speculative decoding, also select `--speculative-moe-runner-backend humming` for the draft. Use `{"dtype":"bfloat16"}` for an explicit BF16 activation choice. The backend flag alone does not request FP8 activations.

Humming also supports **online weight quantization during loading** from an unquantized FP16/BF16 checkpoint. For example, this configuration requests MXFP4 weights with 32-element groups and UE8M0 scales, together with FP8 activations:

```bash
SGLANG_HUMMING_ONLINE_QUANT_CONFIG='{"dtype":"float4e2m1","group_size":32,"scale_dtype":"float8e8m0"}' \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server \
  --model-path /path/to/supported-bf16-moe-checkpoint \
  --quantization humming \
  --moe-runner-backend humming
```

The weight schema accepts `dtype`, `group_size`, and `scale_dtype`; SGLang also supports `ignored_layers` to leave selected layers unconverted. For supported already-quantized checkpoints, online conversion requires an explicit `"force_requant":true` in the weight configuration. These settings select loading behavior; they do not supply calibrated scales for static activation quantization. The [0.1.18 schemas](https://github.com/vllm-project/humming/blob/v0.1.18/humming/schema/humming.py) define the format fields and compatibility rules.

The results below use an existing native FP8-dense/MXFP4-expert checkpoint, with explicit activation selection rather than online weight quantization.

## Serving Results on H20-3e

We evaluate **DeepSeek-V4.1-Flash-1017 on eight H20-3e GPUs**, with TP8, prefill CP8 interleave, 16K-token prefill chunks, and DSpark block size 5. Target and draft use the same MoE backend. Dense layers retain their native FP8 path and expert weights use MXFP4. **A16 means BF16 expert activations; AFP8 means FP8 E4M3 expert activations.**

The EP1 columns compare Marlin and Humming at A16, and Humming's AFP8 deployment option. The EP8 columns compare Humming and FlashInfer 0.7.2 at the same expert partitioning. Humming EP8 includes the tuning fix from [PR #49873](https://github.com/sgl-project/sglang/pull/49873), applied to the original benchmark source. Results were collected on separate H20-3e hosts; [source revisions and reproduction commands](#benchmark-details) are included on this page.

### Prefill

The table reports **TTFT in seconds** for exact-length synthetic token-ID inputs. Each entry is the median of three requests after warmup and a prefix-cache flush, at concurrency 1; K means 1,024.

<div style="overflow-x: auto;">
<div style="min-width: 760px;">

| Input tokens | Marlin A16 (EP1) | Humming A16 (EP1) | Humming AFP8 (EP1) | Humming AFP8 (EP8) | FlashInfer AFP8 (EP8) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 0.338 | 0.274 | 0.215 | 0.241 | 0.219 |
| 64K | 4.533 | 3.762 | 3.026 | 3.381 | 3.306 |
| 128K | 10.033 | 8.495 | 7.024 | 7.693 | 7.565 |
| 256K | 23.843 | 20.813 | 17.894 | 19.358 | 19.155 |
| 512K | 63.443 | 57.398 | 51.554 | 54.496 | 54.109 |
| 1016K | 188.064 | 175.969 | 164.350 | 170.134 | 169.391 |

</div>
</div>

Within the EP1 study, Humming A16 delivers a **1.07–1.23× prefill speedup** over Marlin A16. FP8 expert activations reduce it further. In the EP8 comparison, FlashInfer has lower median TTFT at all six measured lengths, with the relative difference narrowing at longer inputs.

### Decode

Decode uses ShareGPT first-turn content repeated or truncated to **4K input tokens**, followed by **1K output tokens**, with prefix caching disabled. Output throughput includes prefill and serving overhead. Values are **mean output tokens/s ± sample standard deviation across three runs**.

<div style="overflow-x: auto;">
<div style="min-width: 760px;">

| Concurrency | Marlin A16 (EP1) | Humming A16 (EP1) | Humming AFP8 (EP1) | Humming AFP8 (EP8) | FlashInfer AFP8 (EP8) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 323.7 ± 3.6 | 314.3 ± 3.2 | 367.1 ± 5.0 | 328.8 ± 16.5 | 315.4 ± 6.7 |
| 32 | 1661.5 ± 22.2 | 1721.6 ± 6.2 | 2314.1 ± 18.0 | 2269.4 ± 28.9 | 2261.6 ± 50.4 |
| 64 | 2044.1 ± 5.6 | 2200.4 ± 13.7 | 2865.8 ± 7.4 | 2839.5 ± 3.6 | 2883.0 ± 17.2 |

</div>
</div>

At EP1 and A16 precision, Humming's mean throughput ranges from **2.89% lower at C1** to **7.65% higher at C64** than Marlin. Within Humming EP1, selecting AFP8 instead of A16 raises throughput by **16.79–34.41%**.

With the EP tuning fix, Humming AFP8 at EP8 has observed mean throughput **4.25% higher at C1**, **0.34% higher at C32**, and **1.51% lower at C64** than the FlashInfer AFP8 reference. The [detailed metrics below](#benchmark-details) report TTFT, TPOT, and DSpark acceptance separately.

The retained full GSM8K evaluations at EP1 score **97.12% for Marlin A16, 97.04% for Humming A16, and 96.82% for Humming AFP8**. In a targeted RULER v1 retrieval check, Marlin A16 and Humming AFP8 each score 200/200 at nominal 32K and 128K. These results describe those evaluated configurations; they do not establish general accuracy equivalence or quality at the 1016K performance point.

## Understanding the Results

The prefill profiles separate the backend effect from the activation-precision choice. At the same A16 precision, Humming's routed-expert GEMMs take **31.6–32.4% less cumulative kernel time** than Marlin across 4K, 128K, and 1016K. Within Humming, selecting AFP8 reduces that GEMM time by a further **about 47.6%**.

At 1016K, the remaining GPU kernels take roughly **151 seconds in all three configurations**. This helps explain why the reduction in routed-GEMM time translates into a smaller end-to-end prefill speedup. The remainder includes attention, dense/shared-expert work, routing, activation, and communication; it is not an attention-only measurement.

<p align="center">
  <img src="/images/blog/humming-sglang/humming_prefill_breakdown.png" width="98%" alt="Prefill kernel-duration breakdown comparing Marlin W4A16, Humming W4A16, and Humming W4AFP8 at 4K, 128K, and 1016K input lengths">
</p>

<p align="center">
  <em>Figure 2. Prefill kernel durations at TP8/EP1, averaged across all eight ranks. The A16 pair holds activation precision fixed; AFP8 is a separate precision choice. Totals are summed GPU kernel durations, not client TTFT.</em>
</p>

At C1, a focused A16 trace gives **3.77 ms per target verify for Humming's routed GEMMs versus 4.21 ms for Marlin**, while draft GEMMs take 0.195 ms versus 0.181 ms. Both use M=8 MoE tiles. This sample does not show a target-GEMM slowdown that explains the small serving-throughput gap, and it does not establish that autotuning would recover it. The serving measurements and these short diagnostic captures remain separate evidence.

## Benchmark Details

The following expandable sections contain the software pins, commands, and supplementary metrics for the results above.

<details>
<summary><strong>Environment, source revisions, and server commands</strong></summary>

The runtime uses Python 3.12, PyTorch 2.14.1+cu130, CUDA 13.0 (NVCC 13.0.48), driver 570.133.20, `humming-kernels[cu13]==0.1.18`, FlashInfer Python/cubin 0.7.2 with JIT cache 0.7.2+cu130, `sglang-kernel==0.4.10+cu130`, and Triton 3.8.0. CUDA compatibility libraries precede the driver directories on `LD_LIBRARY_PATH`. The benchmark client uses a separate PyArrow 25.0.1 override with NumPy 1.26.4.

| Results | SGLang Python source |
| --- | --- |
| EP1 performance, retained quality checks, and diagnostic profiles | [`f644444`](https://github.com/guzekai01/sglang/tree/f644444260a6caf2cc4a4d5357dc17edaa6c8e64), including the implementation proposed in #48524 |
| FlashInfer EP8 reference | The same `f644444` base |
| Humming AFP8 EP8 | `f644444` plus only the runtime change from #49873; measured tree `04032f77a61831f0ba7cc84cdeefeb4605471f3d` |
| Benchmark client, for every column | `f644444` |

The Humming EP8 tree can be reconstructed from public commits without updating unrelated model or attention code:

```bash
git clone https://github.com/guzekai01/sglang.git sglang-humming
git -C sglang-humming checkout --detach f644444260a6caf2cc4a4d5357dc17edaa6c8e64
git -C sglang-humming fetch origin 6a7ea3f13451b7dec944af37e7068c29b56213c7
git -C sglang-humming worktree add --detach ../sglang-humming-ep8 \
  f644444260a6caf2cc4a4d5357dc17edaa6c8e64
git -C sglang-humming diff --full-index \
  4bf9ca2a3bf7f8dd2e1c2ea6e63d8d8059b46426 \
  6a7ea3f13451b7dec944af37e7068c29b56213c7 \
  -- python/sglang/srt/layers/moe/moe_runner/humming.py \
  | git -C sglang-humming-ep8 apply --3way
export BASE_SOURCE="$PWD/sglang-humming"
export EP8_SOURCE="$PWD/sglang-humming-ep8"
```

The patched runner's Git blob is `d33c8c056d38c4fa73738510162e6cb169b41ef4`. The local benchmark tree ID above is recorded for provenance; the public base and patch define the reproducible runtime.

The model is [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), including its bundled DSpark draft. The original download did not record a Hugging Face revision. The config and weight index were checked across the experiment hosts and match these hashes; they are not checksums of every weight shard:

| File | SHA256 |
| --- | --- |
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

The EP1 study used 64 CPU / 768 GiB limits; the EP8 runs used 150 CPU / 1200 GiB limits on other H20-3e hosts. All used `OMP_NUM_THREADS=1`. Cross-EP differences therefore include more than a change in expert partitioning. The updated Humming EP8 run also used a different host from the retained FlashInfer reference.

Set `MODEL_PATH` to the downloaded checkpoint. Run one server at a time. Choose `MODE=ttft` for the long-prefill table or `MODE=decode` for the ShareGPT throughput table; use the corresponding source from the table above.

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017
MODE=decode
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
if [[ "$MODE" == decode ]]; then
  COMMON+=(--disable-radix-cache --stream-interval 1)
fi
```

Select the backend and activation precision with one of the following launches:

```bash
# Marlin A16, EP1
PYTHONPATH="$BASE_SOURCE/python" \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend marlin --speculative-moe-runner-backend marlin

# Humming A16, EP1
PYTHONPATH="$BASE_SOURCE/python" \
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"bfloat16"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# Humming AFP8, EP1
PYTHONPATH="$BASE_SOURCE/python" \
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 1 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# Humming AFP8, EP8, including the runtime fix
PYTHONPATH="$EP8_SOURCE/python" \
SGLANG_HUMMING_MOE_GEMM_TYPE=indexed \
SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}' \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 8 \
  --moe-runner-backend humming --speculative-moe-runner-backend humming

# FlashInfer AFP8, EP8
PYTHONPATH="$BASE_SOURCE/python" \
python -m sglang.launch_server "${COMMON[@]}" --ep-size 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-moe-runner-backend flashinfer_mxfp4 \
  --flashinfer-mxfp4-moe-precision fp8
```

Humming's AFP8 setting produces dynamic per-token E4M3 input scales (`group_size=0`) for each local GEMM. The checkpoint's MXFP4 weight-group size is not copied to the activation configuration. Humming uses its indexed MoE path. EP8 gives both backends a local intermediate dimension of 2,304 rather than EP1's 288, meeting FlashInfer's 128-alignment requirement.

</details>

<details>
<summary><strong>Benchmark commands, detailed metrics, and validation notes</strong></summary>

Run the client in a second shell using the baseline Python source. The client dependencies are separate from the server environment:

```bash
export BASE_SOURCE=/path/to/sglang-humming
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017
export CLIENT_DEPS=/tmp/humming-bench-client-deps
python -m pip install --target "$CLIENT_DEPS" --no-deps pyarrow==25.0.1
export PYTHONPATH="$CLIENT_DEPS:$BASE_SOURCE/python"
export SGLANG_IS_IN_CI=0
mkdir -p results
```

**Long-prefill TTFT.** Use the radix-enabled server. Each length uses one warmup, a prefix-cache flush, and three measured requests at C1/output32. Report their median TTFT. Compiled-kernel caches remain warm.

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
    --output-file "results/ttft-${LENGTH}.jsonl"
done
```

**ShareGPT-derived decode.** The dataset is [`ShareGPT_V3_unfiltered_cleaned_split.json`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json), revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`, SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

The pinned client's `random` dataset shuffles conversations, tokenizes their first user turn without a chat template, and repeats or truncates those token IDs to exactly 4,096 tokens. This is a fixed-shape workload derived from real text. Each measured run uses seed 1 and `N=max(10,5C)` requests. Before it, a separate seed-0 warmup generates the full 1,024 output tokens for C requests. The native client's additional warmup is disabled; streaming and ignore-EOS use the pinned client's defaults.

```bash
export SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
for C in 1 32 64; do
  N=$((5 * C))
  if (( N < 10 )); then N=10; fi
  for R in 1 2 3; do
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
        --output-file "results/decode-c${C}-r${R}-${PHASE}.jsonl"
    done
  done
done
```

EP1 and the updated Humming EP8 measurements used one server launch per backend for the three repetitions. The retained FlashInfer EP8 reference used three independently restarted servers, one repetition per launch. TTFT uses a separate radix-enabled launch. Reported SD is calculated from run-level metrics (`ddof=1`), not from individual requests. Input-token hashes, output lengths, and zero measured prefix-cache hits were checked.

**Decode latency and acceptance.** TTFT includes server queueing; TPOT is the client's mean time per output token after the first. Acceptance is the per-request value from server metadata, averaged within each run and then across runs.

<div style="overflow-x: auto;">
<div style="min-width: 760px;">

| Configuration | C | TTFT (ms) | TPOT (ms) | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Marlin A16 (EP1) | 1 | 321.238 ± 0.687 | 2.776 ± 0.036 | 5.139 ± 0.058 |
| Marlin A16 (EP1) | 32 | 1558.744 ± 1.860 | 16.768 ± 0.079 | 5.159 ± 0.035 |
| Marlin A16 (EP1) | 64 | 3435.942 ± 29.612 | 26.989 ± 0.050 | 5.205 ± 0.004 |
| Humming A16 (EP1) | 1 | 263.333 ± 1.423 | 2.924 ± 0.031 | 5.115 ± 0.047 |
| Humming A16 (EP1) | 32 | 1294.003 ± 37.182 | 16.335 ± 0.061 | 5.153 ± 0.022 |
| Humming A16 (EP1) | 64 | 2900.157 ± 45.025 | 25.159 ± 0.036 | 5.211 ± 0.001 |
| Humming AFP8 (EP1) | 1 | 207.182 ± 1.233 | 2.522 ± 0.039 | 5.142 ± 0.051 |
| Humming AFP8 (EP1) | 32 | 1001.886 ± 4.787 | 12.234 ± 0.093 | 5.191 ± 0.029 |
| Humming AFP8 (EP1) | 64 | 2231.068 ± 25.516 | 19.354 ± 0.021 | 5.201 ± 0.004 |
| Humming AFP8 (EP8) | 1 | 221.653 ± 0.589 | 2.830 ± 0.157 | 5.106 ± 0.170 |
| Humming AFP8 (EP8) | 32 | 1057.362 ± 27.748 | 12.320 ± 0.109 | 5.172 ± 0.030 |
| Humming AFP8 (EP8) | 64 | 2334.429 ± 5.345 | 19.505 ± 0.073 | 5.220 ± 0.019 |
| FlashInfer AFP8 (EP8) | 1 | 208.612 ± 2.625 | 2.968 ± 0.065 | 5.089 ± 0.091 |
| FlashInfer AFP8 (EP8) | 32 | 1019.455 ± 10.509 | 12.469 ± 0.130 | 5.179 ± 0.033 |
| FlashInfer AFP8 (EP8) | 64 | 2279.674 ± 16.318 | 18.990 ± 0.030 | 5.221 ± 0.013 |

</div>
</div>

**Quality.** The retained EP1 GSM8K comparison evaluates all 1,319 official test questions with the first eight train examples as demonstrations, the pinned SGLang scorer, chat API, `thinking=false`, temperature 0, maximum 1,024 generated tokens, and concurrency 8. Correct counts are **1,281 / 1,280 / 1,277** for Marlin A16 / Humming A16 / Humming AFP8, respectively, with no empty responses. These are single full evaluations. Separate validation of the EP tuning fix is reported in [PR #49873](https://github.com/sgl-project/sglang/pull/49873); it is not a quality re-evaluation of the combined EP8 benchmark tree above.

The long-context check uses [RULER v1 `niah_multikey_2`](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a), seed 42, and 200 fixed examples per nominal 32K/128K length. Marlin A16 and Humming AFP8 at EP1 each score 100% at both lengths, with no empty or length-limited responses. The check uses `thinking=false`, temperature 0, output limit 128, concurrency 8, and the official `string_match_all` metric. It covers this retrieval task and these two configurations, not the full RULER suite.

**Scale-range adjustment.** A separate load-only audit at TP8/EP1 counts valid scale groups with positive exponent shifts before the original Humming 0.1.18 transformation. Padded groups are excluded from the denominator. No audit instrumentation was present in the serving timings.

| Component | Valid groups | Groups with positive shift | Fraction |
| --- | ---: | ---: | ---: |
| Target | 16,986,931,200 | 0 | 0% |
| DSpark draft | 424,673,280 | 47,868,431 | 11.27% |

The affected groups are in the draft's gate/up projection. At EP1 its N dimension is padded from 576 to 768, and the padding scales participate in range selection. A valid-scales-only calculation would adjust 5,308 groups; padding adds 47,863,123 affected groups. These count positive scale shifts, not individual FP4 elements whose values change after rounding. They do not establish the cause of an accuracy or acceptance difference and are not EP8 counts.

**Profiles.** The diagnostic traces use the original `f644444` source and all eight ranks, with a warmup request and prefix caching disabled. Each prefill length has one warmed capture for Marlin A16, Humming A16, and Humming AFP8, using the same first measured token-ID input at 4K/128K/1016K and requesting 32 output tokens. Only complete prefill-stage traces are included. C1 captures use the first measured ShareGPT-derived 4K prompt, 128 output tokens, and ten target verify and ten draft forwards. The three arms were captured on separate H20-3e hosts. Figures report rank-mean kernel-duration sums, not run-to-run variation; communication, dependency waits, and overlap are not additive client latency. The C1 samples do not establish the cause of the original 2.89% end-to-end A16 gap.

</details>

## What Comes Next

### Humming

- **Autotuning.** Humming plans to introduce autotune, benchmarking candidate configurations on the target device to find better execution parameters for each workload. This will complement the existing heuristics.
- **Broader hardware support.** Upcoming work includes Rubin Locality Domains, T-Head's Zhenwu M890, and more CUDA-compatible accelerators, with kernels adapted to their compute and memory characteristics.

### SGLang Integration

- **Simpler model integration.** Build on unified input processing to support more native quantized checkpoints and custom activations, while reducing padding, repacking, and peak memory during weight loading.
- **Better parallel execution.** Improve EP-aware tuning, quantized DeepEP dispatch, and dynamic expert migration. Extend end-to-end coverage to these paths and **B300 deployments**.

Follow the [SGLang Humming tracking issue #46579](https://github.com/sgl-project/sglang/issues/46579) for integration progress.

## Acknowledgments

This work was jointly developed by members of the Ant Group Venus and SCT teams and the SGLang community.

- **Ant Group Venus Team:** Jinzhen Lin
- **Ant Group SCT Team:** Zekai Gu, ZhiLin Huang
- **SGLang Team:** Peng Zhang, Xiaoyu Zhang

We thank the SGLang maintainers and reviewers who helped bring the quantization, MoE, and DeepEP integrations to production. We also thank Jiang Shao and the NVIDIA team for adapting Humming's MXFP4 × FP8 scale-fusion approach in [FlashInfer PR #3738](https://github.com/flashinfer-ai/flashinfer/pull/3738).
