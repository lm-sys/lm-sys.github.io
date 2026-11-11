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

For the evaluated checkpoint at TP8/EP1, a separate load audit found no adjusted groups in the target model and **47,868,431 of 424,673,280 groups (11.27%)** in the draft. Most draft adjustments arise because padding participates in scale-range selection. These count scale groups entering requantization, not individual FP4 values that change; the [appendix](/files/humming-sglang-0.1.18/README.md) gives the breakdown.

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

The end-to-end results in this post cover **DeepSeek-V4.1-Flash-1017 only**. Compatible expert-parallel deployments can also use [DeepEP FP8 dispatch](https://github.com/sgl-project/sglang/pull/31429), carrying quantized activations and their scales into Humming.

## Using Humming in SGLang

Base Humming 0.1.18 support landed in SGLang commit [ca939407](https://github.com/sgl-project/sglang/commit/ca939407423088dffd33304e0ea7d092cba355d2). This post uses [f644444](https://github.com/guzekai01/sglang/commit/f644444260a6caf2cc4a4d5357dc17edaa6c8e64), which also includes the proposed input-processing update. The reported environment pins Humming **0.1.18**; 0.1.19 has not been validated for these results.

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

The results use **eight H20-3e GPUs**, DeepSeek-V4.1-Flash-1017, TP8, prefill CP8, and DSpark with block size 5. The EP1 comparison covers Marlin and Humming; a separate matched EP8 comparison covers Humming and FlashInfer 0.7.2. Backends within each comparison share a host and fixed inputs; the EP1 and EP8 studies used different H20-3e hosts. All configurations use the tested source revision above. Target and draft use the same selected MoE backend; the native FP8 dense path and MXFP4 expert storage are retained. **A16 means BF16 expert activations; AFP8 means FP8 E4M3 expert activations.**

Long-prefill inputs are synthetic, exact-length token IDs, with C1, CP interleave, and 16K-token prefill chunks. Each value is the median of three measured requests after warmup and a prefix-cache flush; K means 1,024. Exact launch commands, workload construction, and software versions are in the [reproduction appendix](/files/humming-sglang-0.1.18/README.md).

| Input tokens | Marlin A16 (EP1) | Humming A16 (EP1) | Humming AFP8 (EP1) | Humming AFP8 (EP8) | FlashInfer AFP8 (EP8) |
| --- | --- | --- | --- | --- | --- |
| 4K | 0.338 s | 0.274 s | 0.215 s | 0.249 s | 0.219 s |
| 64K | 4.533 s | 3.762 s | 3.026 s | 3.416 s | 3.306 s |
| 128K | 10.033 s | 8.495 s | 7.024 s | 7.771 s | 7.565 s |
| 256K | 23.843 s | 20.813 s | 17.894 s | 19.540 s | 19.155 s |
| 512K | 63.443 s | 57.398 s | 51.554 s | 54.876 s | 54.109 s |
| 1016K | 188.064 s | 175.969 s | 164.350 s | 170.968 s | 169.391 s |

In the EP1 study, Humming reduces prefill latency relative to Marlin at both activation precisions across these lengths. Its relative advantage narrows as the context grows in this setup. In a separate Marlin A16 profile, routed-expert GEMMs account for about **56%, 48%, and 20% of summed GPU kernel time** at 4K, 128K, and 1016K, respectively. The [appendix](/files/humming-sglang-0.1.18/README.md) defines this metric and its limits.

Decode uses ShareGPT-derived first-turn content repeated or truncated to **4K input tokens**, with **1K output tokens** and prefix caching disabled. Client output throughput includes prefill and serving overhead. The table reports mean output tokens/s across three repetitions.

| Concurrency | Marlin A16 (EP1) | Humming A16 (EP1) | Humming AFP8 (EP1) | Humming AFP8 (EP8) | FlashInfer AFP8 (EP8) |
| --- | --- | --- | --- | --- | --- |
| 1 | 323.69 | 314.35 | 367.13 | 324.51 | 315.39 |
| 32 | 1661.54 | 1721.59 | 2314.06 | 2169.44 | 2261.61 |
| 64 | 2044.12 | 2200.40 | 2865.84 | 2601.59 | 2882.96 |

At the same A16 precision and EP1 topology, Humming’s throughput ranges from **2.89% lower at C1** to **7.65% higher at C64** than Marlin. Within Humming EP1, switching A16 to AFP8 improves throughput by **16.79–34.41%**. The combined backend-and-precision switch from Marlin A16 to Humming AFP8 improves throughput by **13.42–40.20%**.

On the full 1,319-question GSM8K test with EP1, Marlin A16 scores **97.12%**, Humming A16 **97.04%**, and Humming AFP8 **96.82%**. AFP8 answers four fewer questions correctly than Marlin, a **0.30 percentage-point** difference. A supplemental RULER v1 retrieval check (`niah_multikey_2`, 200 cases each at nominal 32K and 128K) scores 100% for both Marlin A16 and Humming AFP8 at EP1. This targeted check does not establish general long-context quality. The [appendix](/files/humming-sglang-0.1.18/README.md) includes exact counts, validation limits, per-configuration DSpark acceptance, and TTFT/TPOT/throughput with variation across three repetitions.

The matched FP8 comparison uses EP8 to satisfy FlashInfer’s 128-alignment requirement: the local intermediate dimension is 2,304, versus 288 at EP1. At the same EP8 topology, Humming’s observed mean throughput is **2.89% higher at C1**, **4.08% lower at C32**, and **9.76% lower at C64** than FlashInfer. FlashInfer has lower median TTFT at all six measured lengths. Quality scores above apply to EP1; the two EP8 configurations have not yet completed quality evaluation.

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
