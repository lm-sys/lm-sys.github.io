---
title: "SGLang Diffusion: Two Months After Release"
author: "The SGLang Diffusion Team"
date: "January 16, 2026"
previewImg: /images/blog/sgl-diffusion/sgl-diffusion-banner-16-9.png
---

Since its release in early November, **SGLang Diffusion** has gained significant attention and widespread adoption
within the community. We are deeply grateful for the extensive feedback and growing number of contributions from
open-source developers.

Over the past two months, we've been meticulously building sglang-diffusion, and here is a summary of our progress:

## Overview

**New Models**:

- Day-0 support for Flux.2, Qwen-Image-Edit-2511, Qwen-Image-2512, Z-Image-Turbo, Qwen-Image-Layered, TurboWan and
  more.
- Run SGLang Diffusion with diffusers backend: compatible with all models in diffusers; more improvements are coming (
  see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

**LoRA Support**:

- We support almost all LoRA formats for supported models. This section lists example LoRAs that have been explicitly
  tested and verified with each base model in the SGLang Diffusion pipeline.
  | Base Model | Supported LoRAs |
  |-------------------|------------------|
  | **Wan2.2**        | `lightx2v/Wan2.2-Distill-Loras`<br> `Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
  | **Wan2.1**        | `lightx2v/Wan2.1-Distill-Loras` |
  | **Z-Image-Turbo** | `tarn59/pixel_art_style_lora_z_image_turbo`<br> `wcde/Z-Image-Turbo-DeJPEG-Lora` |
  | **Qwen-Image**    | `lightx2v/Qwen-Image-Lightning`<br> `flymy-ai/qwen-image-realism-lora`<br>
  `prithivMLmods/Qwen-Image-HeadshotX`<br> `starsfriday/Qwen-Image-EVA-LoRA` |
  | **Qwen-Image-Edit** | `ostris/qwen_image_edit_inpainting`<br> `lightx2v/Qwen-Image-Edit-2511-Lightning` |
  | **Flux**          | `dvyio/flux-lora-simple-illustration`<br> `XLabs-AI/flux-furry-lora`<br>
  `XLabs-AI/flux-RealismLora` |
- Fully functional HTTP API:
  | Feature | API Endpoint | Key Parameters |
  |---------------------------------|-----------------------------|--------------------------------------------------|
  | Set or Activate (multiple) LoRA(s) | `/v1/set_lora`              | `lora_nickname`, `lora_path`, `strength`, `target` |
  | Merge Weights | `/v1/merge_lora_weights`    | `strength`, `target`                             |
  | Unmerge Weights | `/v1/unmerge_lora_weights`  | - |
  | List Adapters | `/v1/list_loras`            | - |

**Parallelism**: SP for image models, TP for some models, alongside hybrid parallelism (combinations of Ulysses
Parallel, Ring Parallel, and Tensor Parallel).

**Attention Backend**: SageAttention2 and SageAttention3, more backends (sparse) are on the way.

**Hardware Support**: AMD, 4090, 5090.

**SGLang Diffusion x ComfyUI Integration**: We have implemented a flexible ComfyUI custom node that integrates SGLang
Diffusion's high-performance inference engine. While ComfyUI offers exceptional flexibility through its custom nodes, it
lacks multi-GPU support and optimal performance. Our solution replaces ComfyUI's denoising model forward pass with
SGLang's optimized implementation, preserving ComfyUI's flexibility while leveraging SGLang's superior inference. Users
can simply replace ComfyUI's loader node with our SGLDiffusion UNET Loader to enable enhanced performance without
modifying existing workflows.

<img src="/images/blog/sgl-diffusion-26-01/comfyui.png" style="display:block; width: 220%; margin:15px auto 0 auto"></img>
<p style="color:gray; text-align: center;">SGLang-Diffusion Plugin in ComfyUI</p>

## Key Improvements

To serve as a robust, industrial-grade framework, **speed, stability, and code quality** are our top priorities. We have
refactored key components to eliminate bottlenecks and maximize hardware efficiency. Here are the highlights of our
recent technical breakthroughs:

### 1. Layerwise Offload

From our early profiling, we identified model loading/offloading as a major bottleneck, since the forward stream has to
wait until all the weights are on-device.

To tackle this, we introduced:

1. `LayerwiseOffloadManager`: A manager class that provides hooks for prefetching weights of the next layer while
   forwarding on the current layer.
2. `OffloadableDiTMixin`: A mixin class that registers `LayerwiseOffloadManager`'s prefetch and release hooks for the
   diffusion-transformer.

which has the following benefits:
  - **Compute-Loading Overlap**: Overlapping computation with weight loading eliminates stalls on the copy stream, significantly boosting inference speed — especially for multi-DiT architectures like Wan2.2
  - **VRAM Optimization**: A reduced peak VRAM footprint enables the generation of longer video sequences and higher-resolution content

<img src="/images/blog/sgl-diffusion-26-01/layerwise offload vs serial.png" style="display:block; margin: auto; width: 100%;"></img>

<p style="color:gray; text-align: center;">Comparison with Layerwise Offload and Standard Loading</p>


Layerwise offload is enabled for video models by default

See related
PRs ([#15511](https://github.com/sgl-project/sglang/pull/15511), [#16150](https://github.com/sgl-project/sglang/pull/16150)).

### 2. Kernel Improvements

- **FlashAttention kernel upstream**: We found that the FlashAttention kernel used in SGLang Diffusion was behind the
  Dao-AILab upstream version, causing slower performance. We also now avoid using varlen format func in diffusion
  models. See [PR #16382](https://github.com/sgl-project/sglang/pull/16382).
- **JIT QK Norm Kernel**: Fused Q/K RMSNorm into a single inplace kernel to cut launch count and memory traffic before
  attention.
- **FlashInfer RoPE**: Apply RoPE on Q/K inplace with FlashInfer when available (fallback otherwise), reducing RoPE
  overhead and intermediate tensor materialization.
- **Weight Fusion (Operator Fusion)**: Fused projection + activation patterns (e.g., gate/up merge + SiLU&Mul) to reduce
  GEMM count and elementwise launches in DiT blocks.
- **Timestep Implementation**: Use a dedicated CUDA kernel for timestep sinusoidal embedding (sin/cos) to reduce
  per-step overhead in diffusion scheduling. See [PR #12995](https://github.com/sgl-project/sglang/pull/12995).

### 3. Cache-DiT Integration

We've integrated [Cache-DiT🤗](https://github.com/vipshop/cache-dit), the most popular framework for DiT cache,
seamlessly into SGLang Diffusion, fully compatible with `torch.compile`, Ulysses Parallel, Ring Parallel, and Tensor
Parallel, along with any hybrid combination of these three.
See [PR #16532](https://github.com/sgl-project/sglang/pull/16532) & [PR #15163](https://github.com/sgl-project/sglang/pull/15163)
for implementation details

With only a couple of environment variables, the generation speed is boosted by **up to** 169%.

Here is an example to enable Cache-DiT in sglang-diffusion:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=fast \
sglang generate --model-path="Qwen/Qwen-Image" --prompt="Cinematic establishing shot of a city at dusk"
--save-output
```

Furthermore, with the new run-with-diffusers backend feature, we can now integrate and refine Cache-DiT optimizations
within SGLang Diffusion (see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

### 4. Few More Things

- [**Diffusion Cookbook**](https://cookbook.sglang.io/docs/diffusion/): Curated recipes, best practices, and benchmarking guides for SGLang Diffusion.
- **Memory Monitoring**: Peak usage statistics available across offline generation and online serving workflows.
- [**Profiling Suite**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/profiling.md): Full-stage support with step-by-step docs for PyTorch Profiler and Nsight Systems.

## Further Roadmap

- Disaggregated serving x Omni models
- Sparse Attention Backends
- Quantization (Nunchaku, nvfp4 and others)
- Optimizations on consumer-level GPUs

## Performance Benchmark

As shown in the chart at the top of this post, we compared the performance of SGLang Diffusion:

- Against a popular open-source baseline, Hugging Face Diffusers. SGLang Diffusion delivers state-of-the-art
  performance, significantly accelerating both image and video generation.
- Under different parallelism setups. Both CFG-Parallel and USP deliver significant speedups compared to the single-GPU
  setup.

## Acknowledgment

**SGLang Diffusion Team**: TBD

Special thanks to NVIDIA and Voltage Park for their compute support.

## Learn more

- **Slack channel**: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A) (join via slack.sglang.io)
- [**Cookbook for SGLang-Diffusion**](https://cookbook.sglang.io/docs/diffusion)
- [**Documentation on SGLang-Diffusion**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs)
