---
title: "SGLang-Diffusion: Two Months After Release"
author: "The SGLang-Diffusion Team"
date: "January 16, 2026"
previewImg: /images/blog/sgl-diffusion/sgl-diffusion-banner-16-9.png
---

Since its release in early November, **SGLang-Diffusion** has gained significant attention and widespread adoption
within the community. We are deeply grateful for the extensive feedback and growing number of contributions from
open-source developers.

Over the past two months, we've been meticulously building and accelerating SGLang-Diffusion,
now ([#16831ab](https://github.com/sgl-project/sglang/commit/16831ab6d707070f42275d596ce2b3b2f4d69072)) up to 1.5x
faster than our initial release.

Here is a summary of our progress:

## Overview

**New Models**:

- Day-0 support for Flux.2, Qwen-Image-Edit-2511, Qwen-Image-2512, Z-Image-Turbo, Qwen-Image-Layered, TurboWan,
  GLM-Image and more.
- Run SGLang-Diffusion with diffusers backend: compatible with all models in diffusers; more improvements are planned (
  see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

**LoRA Support**:

- We support almost all LoRA formats for supported models. This section lists some example LoRAs that have been
  explicitly tested and verified.
  | Base Model | Supported LoRAs |
  |-------------------|------------------|
  | **Wan2.2**        | `lightx2v/Wan2.2-Distill-Loras`<br> `Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
  | **Wan2.1**        | `lightx2v/Wan2.1-Distill-Loras` |
  | **Z-Image-Turbo** | `tarn59/pixel_art_style_lora_z_image_turbo`<br> `wcde/Z-Image-Turbo-DeJPEG-Lora` |
  | **Qwen-Image**    | `lightx2v/Qwen-Image-Lightning`<br> `flymy-ai/qwen-image-realism-lora`<br> `prithivMLmods/Qwen-Image-HeadshotX`<br> `starsfriday/Qwen-Image-EVA-LoRA` |
  | **Qwen-Image-Edit** | `ostris/qwen_image_edit_inpainting`<br> `lightx2v/Qwen-Image-Edit-2511-Lightning` |
  | **Flux**          | `dvyio/flux-lora-simple-illustration`<br> `XLabs-AI/flux-furry-lora`<br> `XLabs-AI/flux-RealismLora` |
- Fully functional HTTP API:
  | Feature | API Endpoint | Key Parameters |
  |---------------------------------|-----------------------------|--------------------------------------------------|
  | Set or Activate (multiple) LoRA(s) | `/v1/set_lora`              | `lora_nickname`, `lora_path`, `strength`, `target` |
  | Merge Weights | `/v1/merge_lora_weights`    | `strength`, `target`                             |
  | Unmerge Weights | `/v1/unmerge_lora_weights`  | - |
  | List Adapters | `/v1/list_loras`            | - |

**Parallelism**: Support SP and TP for most models, alongside hybrid parallelism (combinations of Ulysses
Parallel, Ring Parallel, and Tensor Parallel).

**Attention Backend**: SageAttention2, SageAttention3 and SLA, more backends are planned.

**Hardware Support**: AMD, 4090, 5090, MUSA

**SGLang-Diffusion x ComfyUI Integration**: We have implemented a flexible ComfyUI custom node that integrates SGLang
Diffusion's high-performance inference engine.

While ComfyUI offers exceptional flexibility through its custom nodes, it
lacks multi-GPU support and optimal performance.

Our solution replaces ComfyUI's denoising model forward pass with
SGLang's optimized implementation, preserving ComfyUI's flexibility while leveraging SGLang's superior inference. Users
can simply replace ComfyUI's loader node with our SGL-Diffusion UNET Loader to enable enhanced performance without
modifying existing workflows.

<img src="/images/blog/sgl-diffusion-26-01/comfyui.png" style="display:block; width: 220%; margin:15px auto 0 auto"></img>
<p style="color:gray; text-align: center;">SGLang-Diffusion Plugin in ComfyUI</p>

## Performance Benchmark

Here are some performance benchmark results:

- We compared the performance of SGLang-Diffusion (#16831ab) with all popular models (including the SGLang-Diffusion on
  Nov. 2025). **SGLang-Diffusion** delivers the fastest speed across all popular models on NVIDIA GPU, up to 5x compared
  to others.

[//]: # (<iframe width="984" height="923" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQRK_j_q8NXZKEqtrTBagxFxvvaxYXXB56HTqqYlD_aAv1v74WKle2HIc7HPK3P0ZVrYlZrjshKYnaV/pubchart?oid=1022178651&amp;format=interactive"></iframe>)
<iframe style="display:block; margin: auto;" width="969" height="923" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQRK_j_q8NXZKEqtrTBagxFxvvaxYXXB56HTqqYlD_aAv1v74WKle2HIc7HPK3P0ZVrYlZrjshKYnaV/pubchart?oid=1681696401&amp;format=interactive"></iframe>

- We compared the performance of SGLang-Diffusion under different environments with one of the fastest vendor.
<iframe style="display:block; margin: auto;" width="969" height="780" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQRK_j_q8NXZKEqtrTBagxFxvvaxYXXB56HTqqYlD_aAv1v74WKle2HIc7HPK3P0ZVrYlZrjshKYnaV/pubchart?oid=174425525&amp;format=interactive"></iframe>


- We also gather results for SGLang-Diffusion on AMD GPU.
<iframe style="display:block; margin: auto;" width="852" height="321" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQCc9ulnNOE8mpM2RjIgZLJlLKxK_KUyws3WlTB1mVz2Ywx790G0IVbrI7-gjY_O5D8G5Grcjb1dBkR/pubchart?oid=319708956&amp;format=interactive"></iframe>

## Key Improvements

To serve as a robust, industrial-grade framework, **speed, stability, and code quality** are our top priorities. We have
refactored key components to eliminate bottlenecks and maximize hardware efficiency.

Here are the highlights of our recent technical improvements:

### 1. Layerwise Offload

From our early profiling, we identified model loading/offloading as a major bottleneck, since the compute stream has to
wait until all the weights are on-device, and most GPUs are not equipped with sufficient VRAM to keep all components in
memory throughout inference.

To tackle this, we introduced:

1. `LayerwiseOffloadManager`: A manager class that provides hooks for **prefetching** weights of the next layer while
   computing on the current layer, as well as **releasing** hooks after compute.
2. `OffloadableDiTMixin`: A mixin class that registers `LayerwiseOffloadManager`'s prefetch and release hooks for the
   diffusion-transformer.

which has the following benefits:

- **Compute-Loading Overlap**: Overlapping computation with weight loading eliminates stalls on the copy stream,
  significantly boosting inference speed — especially for multi-DiT architectures like Wan2.2
- **VRAM Optimization**: A reduced peak VRAM footprint enables the generation of longer video sequences and
  higher-resolution content

<img src="/images/blog/sgl-diffusion-26-01/layerwise offload vs serial.png" style="display:block; margin: auto; width: 100%;"></img>

<p style="color:gray; text-align: center;">Comparison with Layerwise Offload and Standard Loading</p>


Layerwise offload is now enabled for video models by default.

See related
PRs ([#15511](https://github.com/sgl-project/sglang/pull/15511), [#16150](https://github.com/sgl-project/sglang/pull/16150)).

### 2. Kernel Improvements

- **FlashAttention kernel upstream**: We found that the FlashAttention kernel used in SGLang-Diffusion was behind the
  Dao-AILab upstream version, causing slower performance. We also now avoid using varlen format func in diffusion
  models.
- **JIT QK Norm Kernel**: Fused Q/K RMSNorm into a single inplace kernel to cut launch count and memory traffic before
  attention.
- **FlashInfer RoPE**: Apply RoPE on Q/K inplace with FlashInfer when available (fallback otherwise), reducing RoPE
  overhead and intermediate tensor materialization.
- **Weight Fusion (Operator Fusion)**: Fused projection + activation patterns (e.g., gate/up merge + SiLU&Mul) to reduce
  GEMM count and elementwise launches in DiT blocks.
- **Timestep Implementation**: Use a dedicated CUDA kernel for timestep sinusoidal embedding (sin/cos) to reduce
  per-step overhead in diffusion scheduling.

See related
PRs ([#12995](https://github.com/sgl-project/sglang/pull/12995), [#16382](https://github.com/sgl-project/sglang/pull/16382)).

### 3. Cache-DiT Integration

We've integrated [Cache-DiT🤗](https://github.com/vipshop/cache-dit), the most popular framework for DiT cache,
seamlessly into SGLang-Diffusion, fully compatible with `torch.compile`, Ulysses Parallel, Ring Parallel, and Tensor
Parallel, along with any hybrid combination of these three.
See [#16532](https://github.com/sgl-project/sglang/pull/16532) & [#15163](https://github.com/sgl-project/sglang/pull/15163)
for implementation details.

With only a couple of environment variables, the generation speed is boosted by up to 169%.

Here is an example to enable Cache-DiT in SGLang-Diffusion:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=fast \
sglang generate --model-path=Qwen/Qwen-Image --prompt="Cinematic establishing shot of a city at dusk"
  --save-output
```

Furthermore, we can now integrate and refine Cache-DiT optimizations to our newly-supported diffuser backend (
see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

### 4. Few More Things

- **Memory Monitoring**: Peak usage statistics available across offline generation and online serving workflows.
- [**Profiling Suite**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/profiling.md):
  Full-stage support with step-by-step docs for PyTorch Profiler and Nsight Systems.
- [**Diffusion Cookbook**](https://cookbook.sglang.io/docs/diffusion/): Curated recipes, best practices, and
  benchmarking guides for SGLang-Diffusion.

## Further Roadmap

- Disaggregated serving x Omni models
- Sparse Attention Backends
- Quantization (Nunchaku, nvfp4 and others)
- Optimizations on consumer-level GPUs

## Acknowledgment

**SGLang-Diffusion Team**:

Aichen Feng, Alison Shao, Changyi Yang, Chunan Zeng, Fan Lin, Fan Luo, Fenglin Yu, Gaoji Liu, Heyang Huang, Hongli Mi, HuangJi,
Huanhuan Chen, Jianying Zhu, Jiaqi Zhu, JiaJun Li, Ji Li, Jinliang Li, Junlin Lv, Mingfa Feng, Ran Mei, Shenggui Li,
Shuyi Fan, Shuxi Guo, Weitao Dai, Wenhao Zhang, Xi Chen, Xiao Jin, Xiaoyu Zhang, Yihan Chen, Yikai Zhu, Yin Fan, Yuhao
Yang, Yuan Luo, Yueming Pan, Yuhang Qi, Yuzhen Zhou, Zhiyi Liu, Zhuorui Liu, Ziyi Xu, Mick

Special thanks to NVIDIA and Voltage Park for their compute support.

## Learn more

- **Slack channel**: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A) (join via slack.sglang.io)
- [**Cookbook for SGLang-Diffusion**](https://cookbook.sglang.io/docs/diffusion)
- [**Documentation on SGLang-Diffusion**](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs)
