---
title: "Beyond Day-0: Optimizing DeepSeek-V4.1 Flash in SGLang"
author: "SGLang Team"
date: "September 28, 2026"
previewImg: "/images/blog/deepseek-v41-optimization/cover.png"
type: "blog"
---

## 1. Overview

**We reached nearly 1,200 tok/s per user without a mega-kernel. These optimizations work within SGLang's existing execution flow: kernel fusion, overlap between kernels through PDL, overlap across CUDA streams, and improvements to GEMM, the indexer, and communication. On 4×GB300, we measured 1,193 tok/s/user at concurrency 1 with random 4K-token inputs, 1K-token outputs, and DSpark enabled at a simulated acceptance length of 5.4.**

Since adding [Day-0 support for DeepSeek-V4.1](https://lmsys.org/blog/2026-09-10-deepseek-v41/), we have continued optimizing SGLang on Blackwell and Hopper. This post covers the measured results and the changes to GEMM, mHC, the attention indexer, MoE, and communication.

On 4×GB300, performance reaches **1,193 tok/s/user at concurrency 1** and **2,064 output tok/s/GPU at concurrency 64**. On 8×H200, without speculative decoding, single-user generation speed improves from about **76 to 168 tok/s/user**, while output throughput at concurrency 64 rises from about **96 to 221 tok/s/GPU**. Across the six measured concurrency levels, per-user generation speed is **2.21–2.53× the pre-optimization SGLang baseline**.

Both benchmarks use random requests with fixed input and output lengths, at concurrency levels of 1, 4, 8, 16, 32, and 64.

| Setting | Blackwell | Hopper |
|:---|:---|:---|
| GPUs | 4× NVIDIA GB300 | 8× NVIDIA H200 |
| Input / output length | 4,096 / 1,024 tokens | 8,192 / 1,024 tokens |
| Parallelism | TP4; EP1 at concurrency 1, EP4 otherwise | TP8 / EP1 |
| Speculative decoding | DSpark, block size 5, simulated acceptance length 5.4 | Disabled |

![DeepSeek-V4.1 Flash performance on Blackwell and Hopper](/images/blog/deepseek-v41-optimization/cover.png)

## 2. Blackwell: Results on 4×GB300

![SGLang's optimized performance curve on 4×GB300](/images/blog/deepseek-v41-optimization/blackwell-progress.png)

*Figure 1. DeepSeek-V4.1 Flash on 4×GB300. Each label c denotes request concurrency. Higher and further right is better.*

The curve shows optimized revision `8394e82d` (the relevant optimizations have been merged into `main`; use the `main` branch). The benchmark uses 4×GB300, random 4,096-token inputs, 1,024-token outputs, and TP4. Concurrency 1 uses EP1; the other points use EP4. DSpark uses block size 5 and a simulated acceptance length of 5.4. Each point is the median of three runs, with error bars showing the minimum and maximum.

The horizontal axis measures per-user generation speed as `1000 / median TPOT in milliseconds`. The vertical axis is measured end-to-end output throughput, including prefill, divided by the GPU count.

### 2.1 Optimizations

The following Blackwell optimizations have been merged into SGLang `main`.

- Reorganize group32 FP8 weights and scales so supported dense GEMMs can use Blackwell MXFP8 kernels. The grouped KV projections in DSpark's draft model also reuse the prepared MXFP8 weights and scales, reducing format conversions during inference.

- Use GEMV or split-K for WO-A projection at small batch sizes. Split-K divides the reduction dimension among thread blocks, then fuses the partial-result reduction with MXFP8 quantization. Further fusion combines inverse RoPE, WO-A projection, and output quantization, producing the layout needed by the next operator directly.

- Fuse adjacent small operators: RoPE with FP4 quantization and dequantization, Q RoPE with direct writes to the attention buffer, and Q RMSNorm with MXFP8 quantization. Dedicated fused kernels also handle small normalization operations and the Engram gate, reducing kernel launches and intermediate memory traffic.

- Fuse C2 compression. The compressor combines the KV representations of two adjacent tokens with learned weights, then performs RMSNorm, RoPE, quantization, and the main KV-cache write in one kernel. Pairs within a verify block read directly from the current input; pairs spanning a block boundary read the cached historical state. The kernel also emits the compressed representation before RoPE for the subsequent index-K projection and cache write.

- Fuse the weighted combination of mHC's four residual streams with RMSNorm, adding quantization on supported paths. For mixing-coefficient computation, tune the tile size to the input row count and fuse statistics reduction with Sinkhorn normalization. Large batches use TF32 or BF16 Tensor Core projections with residual compensation to reduce the precision loss from weight conversion.

- Compute mHC mixing coefficients on a separate CUDA stream alongside Attention or MoE, synchronizing when the output combination needs them. DSpark verify and draft use the same scheduling approach. At small batch sizes, complete the input combination and normalization before starting coefficient computation to reduce resource contention. Some prefill paths compute the coefficients during all-reduce.

- Fuse the indexer's post-top-k score checks, invalid-position filtering, and KV-page address conversion, avoiding repeated reads and writes of selected indices. After candidate-block selection, directly generate the token membership mask. Batch multiple rows of FP4 quantization within each thread block to reduce scheduling and scale-reduction overhead.

- Have the MoE router produce the layout required by downstream computation directly. For supported TP configurations, fuse the weighted reduction of expert outputs, shared-expert addition, and cross-GPU all-reduce. Where supported, also fuse mHC output combination, the next layer's input combination, or normalization to reduce intermediate writes to GPU memory.

- Use SGLang's Custom AllReduce V2 to accelerate communication between GPUs, fusing surrounding reductions, residual combinations, and normalization. On supported hardware and input sizes, PDL (Programmatic Dependent Launch) overlaps portions of adjacent kernels to reduce waiting.

- Use MoE TP4 for small batches so all four GPUs process different weight shards of the same experts, reducing waiting caused by uneven expert assignment. Each GPU's expert intermediate dimension is 576, padded to the supported width of 640 at load time. The concurrency-1 point uses this configuration; the other points use EP4.

- Reuse request indices and scratch buffers across layers to avoid repeated conversions and initialization. Prepare attention metadata only for the compression ratios the model uses. DSpark greedy decoding exchanges compact argmax candidates across GPUs. Small-batch router scheduling also follows the actual input row count, including the multiple rows produced by DSpark verification.

The initial work entered the main branch with the model integration in [#38798](https://github.com/sgl-project/sglang/pull/38798). Optimization code is available in [#39370](https://github.com/sgl-project/sglang/pull/39370), [#39704](https://github.com/sgl-project/sglang/pull/39704), and [#39957](https://github.com/sgl-project/sglang/pull/39957). Kernel selection depends on the hardware, input shape, and TP/EP configuration.

## 3. Hopper: Progress on 8×H200

![SGLang before and after optimization on 8×H200](/images/blog/deepseek-v41-optimization/hopper-progress.png)

*Figure 2. DeepSeek-V4.1 Flash on 8×H200 before and after optimization, without speculative decoding.*

This comparison uses the same H200 node, checkpoint, client, and launch arguments. The pre-optimization baseline is `2f5c9ac`, and the optimized revision is `42b9b99`. Both use TP8/EP1, random 8,192-token inputs, 1,024-token outputs, and no speculative decoding.

| Concurrency | Baseline tok/s/user | Optimized tok/s/user | Baseline tok/s/GPU | Optimized tok/s/GPU |
|---:|---:|---:|---:|---:|
| 1 | 76.00 | 167.99 | 9.15 | 19.79 |
| 4 | 57.21 | 140.95 | 26.07 | 60.83 |
| 8 | 47.46 | 112.43 | 42.24 | 94.62 |
| 16 | 35.83 | 79.99 | 62.26 | 131.71 |
| 32 | 23.75 | 56.24 | 80.87 | 178.12 |
| 64 | 14.41 | 36.41 | 96.35 | 220.74 |

### 3.1 Optimizations

- Tune thread-block tiles and split-K for group32 FP8 dense GEMMs. For eligible large matrices, cache BF16 representations dequantized from the FP8 weights and compute with BF16 Tensor Cores. Input activations still follow the original FP8 quantization step before being dequantized to BF16; the checkpoint format is unchanged.

- Run indexer scoring and token top-k according to the actual sequence length, avoiding capacity-sized auxiliary work over nearly a million token positions for a context of about 8K. Scoring reads FP4 index-K directly from the paged cache. After token selection, fuse invalid-position filtering, index sorting, and KV-cache address conversion to reduce intermediate memory traffic. Sequence lengths are read from GPU memory, so CUDA Graph replays use the updated lengths.

- Fuse the weighted combination of mHC's four residual streams with RMSNorm in one kernel. For mixing-coefficient computation at medium and large batch sizes, represent FP32 weights as one BF16 main term and two low-order correction terms. Compute each projection with Tensor Cores and accumulate in FP32 to reduce conversion error. During decode, mixing-coefficient computation overlaps Attention or MoE on a separate CUDA stream, synchronizing when the coefficients are needed.

- Use Marlin MoE kernels with MXFP4 weights and BF16 activations. Defer dimension padding until after TP sharding: under TP8, each GPU's logical expert intermediate width is 288, now padded to 320 instead of 384. This reduces computation on padded values. Also tune thread-block tiles for single-token inputs and fuse clipped SwiGLU while preserving the original intermediate rounding behavior.

These Hopper changes are in [PR #41251](https://github.com/sgl-project/sglang/pull/41251), which is awaiting merge. Caching BF16 weights has a memory cost: in this configuration, loaded weight memory per GPU increases from about 72.71 GB to 75.43 GB, leaving correspondingly less space for the KV cache.

## 4. AgentX

## 5. Benchmark Reproduction

## 6. Acknowledgments

These results come from **joint design and optimization by agents, the SGLang Team, and the DeepSeek Infra team**. We thank DeepSeek for open-sourcing DeepSeek-V4.1. Its innovations in both infrastructure and model architecture made this work possible.

We thank the entire SGLang team, the Miles team, and all community contributors involved in model integration, kernel optimization, code review, testing, and performance reproduction.

Special thanks to **GPT6 Astra**. As an agent working on this project, it carried out a substantial share of the code development and performance optimization analysis, iterating on optimization approaches with the team.

Some kernels were developed with the KDA 0.5 framework. We also thank [Humanize](https://github.com/humanfia/humanize2) and [Kernel Design Agents](https://github.com/NVlabs/kda) for their tools and workflows.
