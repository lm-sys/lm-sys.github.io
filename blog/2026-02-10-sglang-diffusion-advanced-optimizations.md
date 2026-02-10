---
title: "SGLang-Diffusion: Advanced Optimizations for Production-Ready Video Generation"
author: "SGLang-Diffusion Team"
date: "February 10, 2026"
previewImg: /images/blog/sgl-diffusion/sgl-diffusion-banner-16-9.png
---

Following our [two-month progress update](https://lmsys.org/blog/2026-01-16-sglang-diffusion/), we're excited to share a
deeper dive into the advanced optimizations that make SGLang-Diffusion a production-ready framework for video
generation. These improvements focus on scalability, efficiency, and stability—essential for deploying diffusion models
at scale.

Here's what we've been working on:

## Overview

As video generation models continue to grow in complexity, we've identified and addressed critical bottlenecks across
the entire inference pipeline:

- **Smarter Parallelism**: Token-level sequence sharding and parallel folding for optimal resource utilization
- **Distributed VAE**: Parallel encoding/decoding to eliminate memory bottlenecks for high-resolution video
- **Production-Ready Serving**: Fixed Cache-DiT integration bugs for stable multi-request serving
- **Optimized I/O**: Accelerated video save operations by eliminating unnecessary serialization
- **Fused Kernels**: Custom JIT kernels for LayerNorm variants, reducing GPU bubbles

Let's dive into the technical details.

## Key Improvements

### 1. SP-Sharding Improvement: From Frame-Level to Token-Level

For Video DiT models, input tensors typically have shape `B, T, H, W, C`. For a common configuration with
`num_frames=81`, this might be: `1, 21, 90, 160, 3`.

In an 8×H100 setup with Ulysses Sequence Parallel (N=8), the framework needs to shard along the sequence dimension
during non-attention operations, then use all-to-all communication to switch to head dimension sharding for attention.

#### Previous Approach: Frame-Level Sharding

Our initial implementation sharded directly along the `T` (temporal) dimension. However, 21 frames cannot be evenly
divided by 8 GPUs, leading to two suboptimal solutions:

1. **Adjust-frame**: Modify `num_frames` during preprocessing to make T divisible by N
2. **Token Padding**: Pad the temporal dimension to the next multiple of N (21 → 24)

The frame-level padding approach introduces significant overhead: each padded token requires `H × W × C` redundant
computations.

#### New Approach: Token-Level Sharding

To minimize padding overhead, we now **flatten `T × H × W` into a single sequence dimension** before sharding. This has
two major benefits:

- **Reduced or Zero Padding**: For common resolutions and VAE configurations, `H × W` is often divisible by 8,
  eliminating padding entirely
- **Lower Communication Volume**: When padding is needed, the overhead is minimal compared to frame-level padding

**Comparison:**

| Solution           | Input Tensor Shape (Per-rank) | All-to-All Comm Volume | Padding Overhead |
|--------------------|-------------------------------|------------------------|------------------|
| **Frame Sharding** | `3, 90, 160, C` (24/8 = 3)    | `1.0 × feature_map`    | 3 frames (14.3%) |
| **Token Sharding** | `2.625, 90, 160, C` (21/8)    | `0.875 × feature_map`  | 0 frames         |

This optimization delivers both faster communication and reduced memory footprint, especially for video models.

See related implementation in the codebase for technical details.

### 2. Parallel Folding: Decoupling Text Encoder and DiT Parallelism

In our original implementation, the Text Encoder and DiT shared the same Tensor Parallel (TP) group. When DiT used only
Sequence Parallel (SP), this meant the Text Encoder ran with TP=1—each GPU held a complete model copy, wasting memory
and compute.

Since Text Encoder and DiT computations are **completely decoupled**, we introduced **Parallel Folding**: the Text
Encoder now uses the DiT's SP group as its TP group.

**What this means in practice:**

- **For Text Encoder**: Apply TP across the SP group to maximize speed and reduce memory
- **For Denoiser**: Apply SP to optimize throughput and memory for sequence processing

This approach ensures both components use optimal parallelism strategies without interference, improving overall
efficiency.

### 3. Parallel VAE: Distributed Encoding/Decoding

VAE encoding/decoding involves heavy 3D convolution operations. For high-resolution video, single-GPU implementations
are slow and prone to OOM.

We considered two approaches:

1. **Tiling**: Split feature maps into tiles processed sequentially—reduces peak memory but increases latency
2. **Parallel**: Distribute tiles across GPUs for concurrent processing—reduces both peak memory and latency

We implemented **Parallel VAE** for Wan-VAE with the following strategy:

- **Height-wise Sharding**: Split feature maps along the height dimension across ranks
- **Conv Operations**: Use `halo_exchange` to share boundary pixels between neighboring ranks, ensuring mathematical
  equivalence with global convolution
- **Attention Operations**: Use `all_gather` for global context when needed
- **Result Aggregation**: `all_gather` to reconstruct full height at the end of encoding/decoding

This approach eliminates VAE as a bottleneck for high-resolution video generation, enabling higher resolutions and
longer sequences without OOM.

<img src="/images/blog/sgl-diffusion-26-02/parallel-vae.png" style="display:block; margin: auto; width: 80%;"></img>
<p style="color:gray; text-align: center;">Parallel VAE: Height-wise Distributed Processing</p>

### 4. Serving with Cache-DiT: Fixing Multi-Request Stability

[Cache-DiT](https://github.com/vipshop/cache-dit) accelerates inference by caching residuals and skipping redundant
computations. However, its correct operation depends on proper `num_inference_steps` configuration, which determines
step counting and the Selective Computation Mask (SCM).

**The Problem:**

Wan2.2 uses a dual-transformer architecture, where `transformer` and `transformer_2` execute `num_high_noise_steps` and
`num_low_noise_steps` respectively (summing to `num_inference_steps`). Our initial implementation had two critical bugs:

1. Both transformers incorrectly used total `num_inference_steps` to configure their cache contexts
2. In serving mode, cache contexts persisted across requests, even when different requests used different
   `num_inference_steps`

These issues caused incorrect step counting and cache buffer contamination. When consecutive requests had different
video shapes, cache buffers would encounter shape mismatches, **crashing the server**.

**The Solution:**

1. `transformer` and `transformer_2` now use `num_high_noise_steps` and `num_low_noise_steps` respectively to configure
   independent cache contexts
2. For each new request, we recalculate timestep splits and **refresh** cache contexts using Cache-DiT's API, completely
   isolating requests

This ensures stable, production-ready serving with Cache-DiT acceleration.

### 5. Optimize Video Save: Eliminating Serialization Overhead

In our serving architecture, `scheduler_client` and `gpu_worker` communicate via ZMQ. Previously, `gpu_worker` would:

1. Complete inference
2. Serialize output tensor
3. Send tensor to `scheduler_client` via ZMQ
4. `scheduler_client` deserializes tensor
5. `scheduler_client` processes tensor and saves video

This introduced significant overhead from serialization/deserialization and memory copies.

**New Approach:**

`gpu_worker` now directly processes the output tensor and saves the video to disk, returning only the file path to
`scheduler_client`.

**Benefits:**

- **Lower Latency**: Eliminates serialization/deserialization overhead
- **Reduced Memory**: Avoids duplicate tensor copies
- **Simpler Pipeline**: Cleaner separation of concerns

### 6. WanVideo LayerNorm Fusion: CuTeDSL JIT Kernels

WanVideo introduces two specialized LayerNorm patterns:

1. **LayerNormScaleShift**: `y = LN(x) * (1 + scale) + shift`
2. **ScaleResidualLayerNormScaleShift**:
    - `residual_out = residual + gate * x`
    - `y = LN(residual_out) * (1 + scale) + shift`

These patterns combine elementwise operations with normalization reductions. Implementing them as separate kernels would
introduce multiple kernel launches and intermediate memory traffic, creating GPU bubbles.

**Our Solution:**

We implemented **fused JIT kernels** using CuTeDSL (located in `sglang/jit_kernel/diffusion/cutedsl/`) that combine
these operations into single, efficient kernels.

**Benefits:**

- **Fewer Kernel Launches**: Reduced launch overhead
- **Lower Memory Traffic**: Eliminates intermediate reads/writes
- **Better GPU Utilization**: Reduces bubbles and improves throughput

These micro-optimizations add up, especially for multi-layer architectures like WanVideo.

## Performance Results

These optimizations collectively deliver:

- **Up to 2.5× faster inference** compared to our initial November 2025 release
- **Zero-padding sequence sharding** for common video resolutions
- **Stable multi-request serving** with Cache-DiT acceleration
- **Elimination of VAE bottlenecks** for high-resolution video generation

## What's Next

We continue to push the boundaries of diffusion model serving. Please refer to [**Roadmap for 26Q1**](https://github.com/sgl-project/sglang/issues/18286) for more details.

Stay tuned for more updates as we continue to optimize SGLang-Diffusion for production deployments.

## Acknowledgment

We would like to thank the following contributors for their work on these optimizations:

Skywork.ai, Song Rui ([Songrui625](https://github.com/Songrui625)), SGLang-Diffusion Team

Special thanks to our compute partners for their continued support.

Try Diffusion generation powered by SGLang-Diffusion at: https://www.apifree.ai/home

## Learn More

- **Slack channel**: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A) (join via slack.sglang.io)
- [**Cookbook for SGLang-Diffusion**](https://cookbook.sglang.io/docs/diffusion)
- [**Documentation on SGLang-Diffusion
  **](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs)
- [**Previous Update: Two Months In**](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)
