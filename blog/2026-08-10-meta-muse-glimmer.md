---
title: "SGLang Adds Day-0 Support for Muse Glimmer, a Multimodal Model Built for Local Agentic Workflows"
author: "Meta Superintelligence Labs and the SGLang Team"
date: "August 10, 2026"
previewImg: /images/blog/2026-08-10-meta-muse-glimmer/cover-muse-glimmer.png
type: blog
---

We're excited to partner with Meta Superintelligence Labs to bring Day-0 support for [Muse Glimmer](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) to SGLang, with dedicated optimizations tailored for high-performance inference of agentic workflows on local hardware.

## Highlights

- **Model:** Muse Glimmer is a 30B-parameter multimodal dense model with a 128k+ token context window that delivers competitive performance on end-to-end agentic workflows from local hardware.
- **Performance:** SGLang delivers up to **1,452 tok/s** of total output throughput and **236 tok/s** of per-user decode speed (NVFP4 with DFlash) on the NVIDIA GeForce RTX 5090, powered by dedicated optimizations for SGLang's SM120 backend.
- **Features:** SGLang provides broad feature support natively for Muse Glimmer, including DFlash speculative decoding, RadixAttention prefix caching, and breakable CUDA graphs.
- **Hardware:** Developers can deploy Muse Glimmer across a variety of consumer and workstation hardware — NVIDIA GeForce RTX 5090, RTX PRO 6000, and DGX Spark via the SM120 backend, and Apple Silicon via the MLX backend.

## Muse Glimmer Model Architecture

Muse Glimmer is a 30B model, consisting of a 27.9B dense text decoder, a 1.9B ViT, and a GELU-based multimodal projector. The text decoder has 52 transformer layers. Each layer contains grouped-query attention with 32 query heads and two key-value heads, followed by a SwiGLU feed-forward network.

### Sliding Window Attention

The decoder uses a hybrid attention pattern that interleaves three 2,048-token sliding-window layers with a full-sequence layer every fourth step for efficiency. Combining RoPE on the local windows with NoPE on the full-attention layers allows the model to extend its context length beyond its training limits.

## Feature Support

### Extensive Backend Support for Local AI Hardware

Muse Glimmer can be deployed with SGLang across a wide range of hardware commonly used by developers building and running AI agents locally, including Apple Silicon devices (Mac mini and M-series MacBook Pro), NVIDIA GeForce RTX 5090 GPUs, and the NVIDIA DGX Spark. On NVIDIA SM120 platforms, SGLang leverages its optimized GEMM and FlashInfer backends for high-throughput inference, while Apple Silicon devices use the native MLX backend to deliver high-performance local serving.

### Speculative Decoding with DFlash

To achieve low-latency inference, use SGLang's implementation of DFlash:

```shell
--speculative-algorithm DFLASH
```

### SGLang Native Optimizations

Muse Glimmer is compatible with SGLang's many native optimizations, including the low-overhead scheduler, the RadixAttention prefix cache, and breakable CUDA graphs. We have also brought several of these optimizations, including prefix caching, to the SGLang MLX backend, enabling competitive performance on Apple Silicon for agentic workloads.

### BF16, NVFP4, GGUF, and MLX 4-bit Checkpoints

We provide Muse Glimmer checkpoints in several formats to meet the needs of users with different hardware, fidelity, and system performance requirements.

Developers requiring maximum model fidelity can run the native BF16 checkpoint on a single H100 GPU. The SM120 path includes a ~19.5 GB mixed NVFP4+MXFP8 quantization recipe. An 18 GB NVFP4 checkpoint paired with a 5 GB BF16 DFlash speculator fits comfortably on a single RTX 5090, enabling high-performance deployment on NVIDIA GeForce RTX 5090 accelerators and DGX Spark.

We also provide two GGUF checkpoints. The smaller of the two is in Q4KM format with a group size of 128. This checkpoint allows for faster inference speeds and deployment on hardware with tighter memory constraints. Q4K-Dynamic delivers better model quality. For developers working on Apple Silicon devices, we provide the previously mentioned GGUF checkpoints in MLX format.

## Performance Results

We measured Muse Glimmer with SGLang across seven platforms, sweeping batch sizes 1 through 8. Reported below are batch-1 interactivity (tok/s/user) and batch-8 aggregate output throughput (tok/s).

| Platform | Precision | Decoding | tok/s/user (batch 1) | Output tok/s (batch 8) |
| ----- | ----- | ----- | ----- | ----- |
| NVIDIA B300 | bf16 | Standard | 91.77 | 21 |
| NVIDIA B300 | bf16 | DFlash | 308.51 | 261 |
| NVIDIA B300 | nvfp4 | Standard | 83.98 | 41 |
| NVIDIA B300 | nvfp4 | DFlash | 290.01 | 295 |
| NVIDIA RTX PRO 6000 | bf16 | Standard | 25.7 | 200 |
| NVIDIA RTX PRO 6000 | bf16 | DFlash | 108.08 | 33 |
| NVIDIA RTX PRO 6000 | nvfp4 | Standard | 58.04 | 51 |
| NVIDIA RTX PRO 6000 | nvfp4 | DFlash | 214.11 | 403 |
| NVIDIA DGX Spark | bf16 | Standard | 4.4 | 35 |
| NVIDIA DGX Spark | bf16 | DFlash | 19.1 | 134 |
| NVIDIA DGX Spark | nvfp4 | Standard | 12.1 | 92 |
| NVIDIA DGX Spark | nvfp4 | DFlash | 36.4 | 301 |
| NVIDIA RTX 5090 | nvfp4 | Standard | 63.9 | 501 |
| NVIDIA RTX 5090 | nvfp4 | DFlash | 236.4 | 1452 |
| NVIDIA RTX 5090 | q4_k_m | Standard | 72.6 | 230 |
| NVIDIA RTX 5090 | q4_k_m | DFlash | 140.7 | 332 |
| Apple M5 Pro | q4_k_m | Standard | 15.3 | 52.6 |
| Apple M5 Pro | q4 | Standard | 17.6 | 56.9 |
| Apple M5 Pro | q4k-dynamic | Standard | 12.6 | 49.1 |

DFlash raises batch-1 interactivity by 1.9–4.3x depending on platform and precision: 3.0x or better on every configuration except the GGUF path on the RTX 5090, which gains 1.9x. The batch-8 gain is smaller and more variable, ranging from 1.4x to 4.2x. Speculative decoding is not available on the MLX backend.

## Acknowledgements

We thank the teams at Meta Superintelligence Labs and the SGLang community for their collaboration in bringing Day-0 support for Muse Glimmer to SGLang.
