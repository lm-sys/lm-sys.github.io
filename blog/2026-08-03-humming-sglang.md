---
title: "Humming Is Now Integrated into SGLang: Flexible Low-Bit GEMM for High-Performance MoE Inference"
author: "Ant Group Venus and SCT Teams, and the SGLang Team"
date: "August 3, 2026"
previewImg: /images/blog/humming-sglang/humming_sgl_preview.png
type: blog
---

We are excited to introduce [Humming](https://github.com/inclusionAI/humming) and its integration into [SGLang](https://github.com/sgl-project/sglang). Humming is a lightweight, JIT-compiled low-bit inference GEMM library. The integration landed in the main branch through [SGLang PR #23754](https://github.com/sgl-project/sglang/pull/23754), making Humming available as both a quantization backend and an MoE runner backend in SGLang.

Modern inference workloads can no longer be covered by a single fixed GEMM kernel. Models increasingly combine dense and sparse layers, 1–8-bit weights, FP16/BF16/FP8/FP4 activations, different scale layouts, and multiple MoE dispatch modes. Humming generates and compiles specialized CUDA kernels for these configurations at runtime, covering a broad range of low-bit inference workloads while keeping the Python package lightweight and its dependency footprint small.

## 1. Why Humming

Quantized inference spans weight and activation types, quantization schemas, execution modes, and GPU architectures. The problem is especially acute for MoE models: expert weights dominate memory footprint and HBM traffic, while token routing produces dynamic GEMM shapes. Quantization helps only when the serving backend can execute the selected format without runtime weight expansion or a generic fallback.

### Precision Flexibility Through Kernel Composition

Humming supports dense and MoE GEMM with 1–8-bit weights and FP16, BF16, FP8, FP4, INT8, or INT4 activations. Exact combinations depend on the GPU architecture; see the [Humming support matrix](https://github.com/inclusionAI/humming#support-matrix).

Instead of implementing a separate kernel for every format and execution mode, Humming composes six compile-time components: **Scheduler**, **G2S pipeline**, **S2R pipeline**, **Dequant**, **Mainloop / MMA**, and **Epilogue**. Tile scheduling, asynchronous memory movement, pipeline overlap, and MMA are reusable; data loading, bit unpacking, zero-point handling, scaling, and dequantization remain format-specific. This lets new weight formats reuse existing dense, indexed, grouped, Stream-K, TMA, and WGMMA paths.

### Normalize Checkpoints and Separate Configuration

Checkpoint schemas normalize AWQ, GPTQ, AutoRound, Compressed Tensors, ModelOpt, MXFP4, and FP8 weights during model loading. Padding, transposition, interleaving, scale rearrangement, and other format-specific preprocessing stay outside the request hot path.

Humming then separates persistent data, execution semantics, and performance mapping:

- **`LayerConfig`** defines weight representation, matrix shapes, data types, scales, zero points, bias, and expert metadata.
- **`ComputeConfig`** selects dense, indexed, grouped-contiguous, or grouped-masked execution and its accumulation semantics.
- **`TuningConfig`** controls block and warp shapes, pipeline stages, CTA count, Stream-K, TMA, warp specialization, and rasterization.

Because these boundaries are independent, a new execution mode can reuse packed weights, while tuning changes do not require checkpoint conversion.

### JIT Specialization and Caching

Humming JIT-compiles only the kernel required by the active model and execution settings. It supports NVRTC and NVCC and caches compiled Cubins for reuse. The cache key covers the compiler, flags, kernel expression, generated code, and Humming headers.

This design keeps the hot path specialized without precompiling the full format matrix. The Hopper MXFP4AFP8 path below shows how the same boundaries support a concrete low-bit implementation.

## 2. A High-Performance MXFP4AFP8 Path for Hopper

The MXFP4AFP8 configuration in the H20 benchmarks uses MXFP4 weights and FP8 activations without expanding persistent weights to 8 bits. On Hopper, the common MXFP4 W4A16 path reconstructs BF16 operands and therefore cannot use the FP8 WGMMA path. Humming instead maps MXFP4 storage onto FP8 WGMMA, targeting large-batch, small-group workloads where group-wise scaling would otherwise add FP32 arithmetic and register pressure.

<p align="center">
  <img src="/images/blog/humming-sglang/humming_hopper_w4a8_case.png" width="98%" alt="Humming Hopper MXFP4AFP8 load-time scale factoring, register-side FP8 reconstruction, and WGMMA data flow">
</p>

<p align="center">
  <em>Figure 1. Humming factors UE8M0 scales at load time, reconstructs FP8 operands in registers, and runs FP8 WGMMA without materializing an 8-bit weight tensor in memory.</em>
</p>

The key ideas are:

1. **Factor UE8M0 scales at load time.** Humming rewrites each group scale as `s_g = s_base × 2^Δg`, keeping one base scale per expert and sublayer and a residual exponent code for each weight group. If the exponent span is too wide for FP8 E4M3, the affected FP4 values are adjusted and requantized during layer preparation. This moves group-wise scaling out of the FP32 accumulator path and into register-side FP8 reconstruction.
2. **Reconstruct FP8 weights only in registers.** Activations are quantized per token to FP8, while expert weights remain packed at 4 bits in global and shared memory. Immediately before MMA, Humming combines the FP4 nibbles and residual exponent codes with `__byte_perm` and `LOP3` to synthesize FP8 E4M3 weight values in registers. A transposed operand mapping places these weight registers in WGMMA's A operand and the shared-memory activation tile in its B operand, so no 8-bit weight tensor is materialized in memory.
3. **Compute with FP8 WGMMA and restore scales in the epilogue.** FP8 × FP8 WGMMA accumulates into FP32, after which the epilogue applies the per-token activation scale and per-expert base scale. Folding each residual exponent into the register-side FP8 value avoids a separate FP32 multiply and scaled accumulator for every weight group, reducing mainloop arithmetic and register pressure. Humming then selects H20-specific tile sizes and pipeline stages; TMA is enabled only for eligible shapes rather than being required by the design.

## 3. How Humming Integrates with SGLang

### As a Quantization Backend

For checkpoints with compatible quantization metadata, SGLang can construct Humming-backed linear and expert layers, load the model tensors, and convert them into the weight layout required by Humming.

Enable it with:

```bash
sglang serve \
  --model-path /path/to/compatible-quantized-checkpoint \
  --quantization humming
```

This is the convenient model-level entry point: SGLang selects Humming for compatible quantized layers while preserving the rest of its serving pipeline.

### As an MoE Runner

Humming can also be selected only for expert computation:

```bash
sglang serve \
  --model-path /path/to/compatible-moe-checkpoint \
  --moe-runner-backend humming
```

SGLang remains responsible for routing, token alignment, gated activation, expert-output reduction, and restoring token order. Humming consumes the prepared expert inputs and executes grouped low-bit GEMM.

### Explicitly Enabling Activation Quantization

Selecting Humming does not automatically turn a W4A16 workload into W4A8. SGLang reads activation-quantization metadata when present; otherwise, or when an override is needed, FP8 activation quantization can be configured through an environment variable:

```bash
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3"}'

sglang serve \
  --model-path /path/to/compatible-moe-checkpoint \
  --quantization humming \
  --moe-runner-backend humming
```

## 4. Performance Evaluation

### End-to-End Performance Evaluation

We evaluate Humming end to end in SGLang on NVIDIA H20 GPUs using Kimi-K2.6 and DeepSeek-V4-Flash. The benchmarks cover TTFT and TPOT across text and image inputs, context lengths, and batch sizes.

All benchmarks use the following software and hardware environment:

| Component | Version or specification |
| --- | --- |
| GPU | 8 × NVIDIA H20-3e |
| NVIDIA Driver | `570.133.20` |
| CUDA | `13.0` |
| SGLang | Commit [`d6ef68881e263812d4901f632786015005c4d050`](https://github.com/sgl-project/sglang/commit/d6ef68881e263812d4901f632786015005c4d050) |
| Humming | `humming-kernels[cu13]==0.1.11` |

#### Kimi-K2.6: Text Input

The text-input latency evaluation compares Marlin WINT4A16, Humming WINT4A16, and Humming WINT4AFP8 on Kimi-K2.6 under TP8, using group-size-32 INT4 expert weights.

The benchmark sweeps TTFT from 4K to 256K input tokens and measures TPOT at 1K, 32K, and 128K with batch sizes 1 and 8 and an output length of 1,024 tokens.

<p align="center">
  <img src="/images/blog/humming-sglang/kimi26_text_latency_speedup.png" width="98%" alt="Kimi-K2.6 text-input TTFT and TPOT speedup over Marlin WINT4A16">
</p>

<p align="center">
  <em>Figure 2. Kimi-K2.6 text-input TTFT and TPOT speedup over Marlin WINT4A16 under TP8.</em>
</p>

#### Kimi-K2.6: Image Input

The image-input evaluation compares the same three configurations under TP8 across requests containing 4 to 20 synthetic 1080p images.

<p align="center">
  <img src="/images/blog/humming-sglang/kimi26_image_ttft_speedup.png" width="98%" alt="Kimi-K2.6 image-input TTFT speedup over Marlin WINT4A16">
</p>

<p align="center">
  <em>Figure 3. Kimi-K2.6 image-input TTFT speedup over Marlin WINT4A16 across 4 to 20 synthetic 1080p images.</em>
</p>

Across the text- and image-input benchmarks, both Humming configurations reduce latency relative to Marlin WINT4A16, with WINT4AFP8 generally delivering the larger speedup.

#### DeepSeek-V4-Flash: TTFT and TPOT Latency

The latency evaluation compares the official FP8 baseline with Humming MXFP4AFP8 under TP8. TTFT covers input lengths from 16K to 128K, while TPOT covers 1K to 128K inputs at batch sizes 1, 4, and 8.

<p align="center">
  <img src="/images/blog/humming-sglang/dsv4_flash_latency_speedup.png" width="98%" alt="DeepSeek-V4-Flash TTFT and TPOT speedup over the official FP8 baseline">
</p>

<p align="center">
  <em>Figure 4. DeepSeek-V4-Flash TTFT and TPOT speedup over the official FP8 baseline under TP8.</em>
</p>

Each bar reports latency speedup relative to the FP8 baseline, which is normalized to 1.00×. Humming improves every tested scenario, reaching up to 1.07× TTFT speedup and 1.26× TPOT speedup.

### Agentic Pareto Curves

Following the agentic replay methodology used in [SGLang GLM-5.2 NVFP4 Optimization](https://www.lmsys.org/blog/2026-07-13-glm52-optimization), we benchmark Humming on Kimi-K2.6, DeepSeek-V4-Flash, and GLM-5.2. The workload replays 13-turn OpenHands coding conversations. A conversation begins with roughly 75K–80K input tokens and generates 220 tokens per turn. We evaluate concurrency levels 1, 2, 4, and 8. Every point is the arithmetic mean of three independent server rounds; error bars span the observed minimum and maximum.

#### Kimi-K2.6

The Kimi-K2.6 comparison uses group-size-32 INT4 weights and evaluates Marlin WINT4A16, Humming WINT4A16, and Humming WINT4AFP8 under TP8.

<p align="center">
  <img src="/images/blog/humming-sglang/kimi26_humming_agentic_pareto.png" width="98%" alt="Kimi-K2.6 Agentic Pareto curve">
</p>

<p align="center">
  <em>Figure 5. Kimi-K2.6 Agentic Pareto curve for Marlin WINT4A16, Humming WINT4A16, and Humming WINT4AFP8 across concurrency levels 1, 2, 4, and 8.</em>
</p>

At concurrency levels 2, 4, and 8, both Humming WINT4A16 and WINT4AFP8 improve interactivity and per-GPU throughput over Marlin WINT4A16. At C=4, Humming WINT4AFP8 reaches 35.19 tok/s/user and 3,928.20 tok/s/GPU, improving the two metrics by 9.35% and 18.06%, respectively.

#### DeepSeek-V4-Flash

The DeepSeek-V4-Flash comparison runs under TP8 and covers Marlin MXFP4A16, Humming MXFP4A16, and Humming MXFP4AFP8.

<p align="center">
  <img src="/images/blog/humming-sglang/dsv4_flash_humming_agentic_pareto.png" width="98%" alt="DeepSeek-V4-Flash Agentic Pareto curve">
</p>

<p align="center">
  <em>Figure 6. DeepSeek-V4-Flash Agentic Pareto curve for Marlin MXFP4A16, Humming MXFP4A16, and Humming MXFP4AFP8 across concurrency levels 1, 2, 4, and 8.</em>
</p>

The 36 measurements cover three backends, four concurrency levels, and three rounds. Pairing each Humming run with the Marlin run at the same round and concurrency, Humming MXFP4A16 improves interactivity by 5.81%–13.62% and per-GPU throughput by 8.67%–14.46%; Humming MXFP4AFP8 improves the two metrics by 11.11%–21.86% and 14.93%–26.22%.

#### GLM-5.2 W4AFP8

The GLM-5.2 comparison evaluates SGLang CUTLASS WINT4AFP8 and Humming WINT4AFP8, both with group size 128, under TP8.

<p align="center">
  <img src="/images/blog/humming-sglang/glm52_w4afp8_humming_agentic_pareto.png" width="98%" alt="GLM-5.2 W4AFP8 Agentic Pareto curve">
</p>

<p align="center">
  <em>Figure 7. GLM-5.2 Agentic Pareto curve comparing SGLang CUTLASS WINT4AFP8 and Humming WINT4AFP8 under TP8.</em>
</p>

At every tested concurrency, Humming WINT4AFP8 moves the Pareto curve above and to the right of SGLang CUTLASS WINT4AFP8. Using the same round-and-concurrency pairing, Humming improves interactivity by 11.48%–23.23% and per-GPU throughput by 7.25%–11.26%.

## 5. Roadmap

This SGLang integration establishes a complete path from checkpoint schemas to low-bit dense and MoE execution. Next, we plan to:

- Expand CI coverage across data types, scale layouts, GPU architectures, and MoE dispatch modes.
- Improve automatic detection of checkpoint activation-quantization types, including static input quantization.
- Continue optimizing performance on Hopper and Blackwell for short-context, long-context, and high-concurrency workloads.
- Support the Blackwell UMMA instruction path, with corresponding kernels and tuning strategies for low-bit dense and MoE GEMM.
- Publish reproducible benchmarks with repeated measurements, quality suites, and complete environment records.

## 6. Acknowledgments

This work was jointly developed by members of the Ant Group Venus and SCT teams and the SGLang community.

- **Ant Group Venus Team:** Jinzhen Lin
- **Ant Group SCT Team:** Zekai Gu, ZhiLin Huang
- **SGLang Team:** Peng Zhang, Xiaoyu Zhang

We thank the SGLang maintainers and reviewers who helped bring the quantization, MoE, and DeepEP integrations to production. We also thank Jiang Shao and the NVIDIA team for adapting Humming's MXFP4 × FP8 scale-fusion approach in [FlashInfer PR #3738](https://github.com/flashinfer-ai/flashinfer/pull/3738).

## Appendix

### Reproduction

The complete server launch, GSM8K accuracy, and agentic performance reproduction commands are maintained in [`benchmark/humming_blog/README.md`](https://github.com/huangzhilin-hzl/sglang/blob/benchmark/humming_blog/benchmark/humming_blog/README.md).
