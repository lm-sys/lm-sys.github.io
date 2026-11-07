---
title: "Humming in SGLang: Flexible Low-Bit Kernels for Efficient MoE Serving"
author: "Ant Group Venus and SCT Teams, and the SGLang Team"
date: "November 4, 2026"
previewImg: /images/blog/humming-sglang/humming_sgl_preview.png
type: blog
---

[Humming](https://github.com/vllm-project/humming/tree/v0.1.18) is a lightweight, JIT-compiled kernel library for quantized dense and mixture-of-experts (MoE) inference. Its SGLang integration connects low-bit checkpoints to specialized kernels while preserving the framework’s routing, scheduling, and distributed execution.

With Humming 0.1.18, that integration extends across more quantization schemes and GPU architectures. This post explains how Humming prepares weights, processes activations, and maps computation to Hopper and Blackwell. We also introduce a shared activation-expression interface for SGLang’s MoE runners and show results with that integration enabled.

On DeepSeek-V4.1-Flash-1017, Humming delivers **1.07–1.23× prefill speedups at the same A16 precision** relative to Marlin in the tested long-context cases. Using FP8 expert activations expands the prefill speedup to **1.14–1.57×** and improves client output throughput by **13.42–40.20%** on the ShareGPT-derived decode workload. The FP8 comparison changes activation precision; it is a different deployment choice from the A16 comparison.

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

The [fused E8M0 configuration](https://github.com/vllm-project/humming/blob/v0.1.18/docs/config.md#use_fused_e8m0_scale) connects preparation with execution. Other scale layouts use their corresponding kernels and preparation rules. Humming 0.1.18 also improves Hopper MoE prefill through [grouped-contiguous](https://github.com/vllm-project/humming/pull/95) and [indexed](https://github.com/vllm-project/humming/pull/115) execution paths.

## Blackwell: Additional Native Compute Paths

Humming’s architecture extends to Blackwell through [BF16 UMMA](https://github.com/vllm-project/humming/pull/71) and eligible [FP8, FP4, and INT8 UMMA paths](https://github.com/vllm-project/humming/pull/105). Supported SM10x configurations can use UMMA, while supported SM12x block-scaled configurations use MXMMA.

These are device- and format-specific choices. INT8 UMMA is not available on every Blackwell device, and scale layouts must match the selected instruction path. The same configuration framework handles these constraints alongside tile and pipeline selection for different serving shapes.

## Connecting Humming to SGLang

The original integration arrived through [SGLang PR #23754](https://github.com/sgl-project/sglang/pull/23754), and [PR #46018](https://github.com/sgl-project/sglang/pull/46018) updated it for Humming 0.1.18. SGLang owns expert routing, dispatch, token alignment, output reduction, and token order; Humming supplies compatible input processing and GEMM execution.

Quantization loading and expert execution are related but distinct responsibilities. SGLang can use Humming for compatible quantized linear layers, or select its MoE runner while retaining another native path for dense layers. It reads checkpoint metadata and preserves the input scales needed by the selected scheme. Choosing Humming does not automatically convert A16 activations to FP8, and static quantization still requires calibrated scales.

The accompanying integration update connects gated MoE activations to `process_input` through optional `activation_type` and `activation_impl` fields. The caller specifies either split or interleaved gate/up inputs and supplies a CUDA expression for the activation. Built-in SiLU, clamped SiLU, erf-based GELU, and SiTU share this interface.

For the down projection, compatible activation evaluation, Hadamard transformation, and quantization can therefore be expressed through one input-processing path. The implementation preserves BF16/FP16 rounding boundaries and avoids an unnecessary intermediate activation buffer when activation and quantization can be combined. This interface lets model-specific activations reuse the same input-processing path.

Humming supports native MXFP4 expert execution for DeepSeek-V4-Flash, DeepSeek-V4-Pro (including V4-Pro-0813), and DeepSeek-V4.1-Flash. These models share the input-processing interface with the gated activation paths used by GLM-5.2/5.3 and Kimi-K2.x/K3. Kimi-K3 also reuses the native MXFP4 loader bridge from [PR #33884](https://github.com/sgl-project/sglang/pull/33884).

The end-to-end results in this post cover **DeepSeek-V4.1-Flash-1017 only**. Compatible expert-parallel deployments can also use [DeepEP FP8 dispatch](https://github.com/sgl-project/sglang/pull/31429), carrying quantized activations and their scales into Humming.

The unified input-processing update is proposed in [SGLang PR #48524](https://github.com/sgl-project/sglang/pull/48524).

## Using Humming in SGLang

Install the matching compiler extra for your CUDA major version in an SGLang environment with Humming 0.1.18 support:

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

The results use **eight H20-3e GPUs**, DeepSeek-V4.1-Flash-1017, TP8, prefill CP8, and DSpark with block size 5. Humming and Marlin use EP1; FlashInfer uses EP8. All configurations include the integration described above. Target and draft use the same selected MoE backend; native FP8 dense layers and MXFP4 expert weights are retained. **A16 means BF16 expert activations; AFP8 means FP8 E4M3 expert activations.**

Long-prefill inputs are synthetic, exact-length token IDs. The table reports median cold-prefill TTFT; K means 1,024.

| Input tokens | Marlin MXFP4A16 | Humming MXFP4A16 | Humming MXFP4AFP8 | FlashInfer MXFP4AFP8 (EP8) |
| --- | --- | --- | --- | --- |
| 4K | 0.338 s | 0.274 s | 0.215 s | 0.214 s |
| 64K | 4.533 s | 3.762 s | 3.026 s | 3.309 s |
| 128K | 10.033 s | 8.495 s | 7.024 s | 7.579 s |
| 256K | 23.843 s | 20.813 s | 17.894 s | 19.150 s |
| 512K | 63.443 s | 57.398 s | 51.554 s | 54.053 s |
| 1016K | 188.064 s | 175.969 s | 164.350 s | 169.401 s |

Humming reduces prefill latency relative to Marlin at both activation precisions across these lengths. Its relative advantage narrows as the context grows in this setup.

Decode uses ShareGPT-derived first-turn content repeated or truncated to **4K input tokens**, with **1K output tokens** and prefix caching disabled. Client output throughput includes prefill and serving overhead. The table reports mean output tokens/s across three repetitions.

| Concurrency | Marlin MXFP4A16 | Humming MXFP4A16 | Humming MXFP4AFP8 | FlashInfer MXFP4AFP8 (EP8) |
| --- | --- | --- | --- | --- |
| 1 | 323.69 | 314.35 | 367.13 | 314.57 |
| 32 | 1661.54 | 1721.59 | 2314.06 | 2262.35 |
| 64 | 2044.12 | 2200.40 | 2865.84 | 2895.27 |

At A16 precision, Humming’s client throughput is **2.89% lower at concurrency 1**, **3.61% higher at 32**, and **7.65% higher at 64** than Marlin. The Humming AFP8 configuration improves throughput by **13.42%, 39.27%, and 40.20%**, respectively. The AFP8 comparison changes expert activation precision for both target and draft.

EP8 gives FlashInfer a local intermediate dimension of 2,304 instead of EP1’s 288, satisfying its adapter’s 128-alignment requirement. Its results represent a different expert partitioning, so differences from EP1 cannot be attributed solely to the backend.

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
