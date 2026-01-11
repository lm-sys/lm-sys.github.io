---
title: "SGLang Diffusion: 2 Months After Release"
author: "The SGLang Diffusion Team"
date: "January 16, 2026"
previewImg: /images/blog/sgl-diffusion/sgl-diffusion-banner-16-9.png
---

Since its release in early November, **SGLang Diffusion** has garnered significant attention and widespread adoption within the community. We are deeply grateful for the extensive feedback and growing number of contributions from developers worldwide.

**Our mission is clear: to build the best diffusion serving framework available—across both open and closed source ecosystems.**

Over the past two months, we have been working tirelessly toward this goal. We have focused on delivering **Day-0 support** for state-of-the-art models and continuously optimizing **stability and performance**, all while maintaining a pristine, modular code architecture. Here is a summary of our recent progress:

## Overview

- **New Models**:
    - **Day-1 support** for Flux.2 / Qwen-Image-Edit-2511 / Qwen-Image-2512 / Z-Image-Turbo / Qwen-Image-Layered / TurboWan and more.
    - Run SGLang Diffusion with diffusers backend: compatible with all models in diffusers; more improvements are coming (see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

- **LoRA Support**:
    - We support almost all LoRA formats for supported models. This section lists example LoRAs that have been explicitly tested and verified with each base model in the SGLang Diffusion pipeline.

       | Base Model        | Supported LoRAs |
       |-------------------|------------------|
       | **Wan2.2**        | `lightx2v/Wan2.2-Distill-Loras`<br> `Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
       | **Wan2.1**        | `lightx2v/Wan2.1-Distill-Loras` |
       | **Z-Image-Turbo** | `tarn59/pixel_art_style_lora_z_image_turbo`<br> `wcde/Z-Image-Turbo-DeJPEG-Lora` |
       | **Qwen-Image**    | `lightx2v/Qwen-Image-Lightning`<br> `flymy-ai/qwen-image-realism-lora`<br> `prithivMLmods/Qwen-Image-HeadshotX`<br> `starsfriday/Qwen-Image-EVA-LoRA` |
       | **Qwen-Image-Edit** | `ostris/qwen_image_edit_inpainting`<br> `lightx2v/Qwen-Image-Edit-2511-Lightning` |
       | **Flux**          | `dvyio/flux-lora-simple-illustration`<br> `XLabs-AI/flux-furry-lora`<br> `XLabs-AI/flux-RealismLora` |

    - **Fully functional HTTP API**:

       | Feature                         | API Endpoint                | Key Parameters                                   |
       |---------------------------------|-----------------------------|--------------------------------------------------|
       | Set/Activate (multiple) LoRA(s) | `/v1/set_lora`              | `lora_nickname`, `lora_path`, `strength`, `target` |
       | Merge Weights                   | `/v1/merge_lora_weights`    | `strength`, `target`                             |
       | Unmerge Weights                 | `/v1/unmerge_lora_weights`  | -                                                |
       | List Adapters                   | `/v1/list_loras`            | -                                                |


- **Parallelism**: SP for image models, TP for some models, alongside hybrid parallelism (combinations of Ulysses Parallel, Ring Parallel, and Tensor Parallel).
- **Attention**: SageAttention2 / SageAttention3, more backends (sparse) are on the way.
- **Hardware**: AMD / 4090 / 5090.
- **SGLang Diffusion x ComfyUI Integration**: 
  
    We have implemented a flexible ComfyUI custom node that integrates SGLang Diffusion's high-performance inference engine.
    While ComfyUI offers exceptional flexibility through its custom nodes, it lacks multi-GPU support and optimal performance. Our solution replaces ComfyUI's denoising model forward pass with SGLang's optimized implementation, preserving ComfyUI's flexibility while leveraging SGLang's superior inference. Users can simply replace ComfyUI's loader node with our SGLDiffusion UNET Loader to enable enhanced performance without modifying existing workflows.

<img src="/images/blog/sgl-diffusion-26-01/comfyui.png" style="display:block; margin: auto; width: 85%;"></img>

## Key Technical Innovations

To serve as a robust, industrial-grade framework, **speed, stability, and code quality** are our top priorities. We have refactored key components to eliminate bottlenecks and maximize hardware efficiency. Here are the highlights of our recent technical breakthroughs:

### 1. Layerwise Offload

From our early **profiling**, we identified model loading/offloading as **a** major bottleneck, since the forward stream has to wait until all the weights are on-device.

To tackle this, we introduced:

1. `LayerwiseOffloadManager`: A manager class that provides hooks for prefetching weights of the next layer while forwarding on the current layer.
2. `OffloadableDiTMixin`: A mixin class that registers `LayerwiseOffloadManager`'s prefetch and release hooks for the diffusion-transformer.

This way, the per-layer forward doesn't have to wait for the copy stream, thus significantly improving the inference speed, especially for specialized models like Wan2.2, where there **are** multiple DiTs.

<img src="/images/blog/sgl-diffusion/layerwise.png" style="display:block; margin: auto; width: 85%;"></img>

Layerwise offload is enabled for video models by default; it reduces peak VRAM usage while accelerating the speed.

See related PRs ([#15511](https://github.com/sgl-project/sglang/pull/15511), [#16150](https://github.com/sgl-project/sglang/pull/16150)).

### 2. Kernel Improvements

1. **FlashAttention kernel upstream**: We found that the FlashAttention kernel used in SGLang Diffusion was behind the Dao-AILab upstream version, causing slower performance. We also now avoid using varlen format func in diffusion models. See [PR #16382](https://github.com/sgl-project/sglang/pull/16382).
2. **JIT QK Norm Kernel**: Fused Q/K RMSNorm into a single inplace kernel to cut launch count and memory traffic before attention.
3. **FlashInfer RoPE**: Apply RoPE on Q/K inplace with FlashInfer when available (fallback otherwise), reducing RoPE overhead and intermediate tensor materialization.
4. **Weight Fusion (Operator Fusion)**: Fused projection + activation patterns (e.g., gate/up merge + SiLU&Mul) to reduce GEMM count and elementwise launches in DiT blocks.
5. **Timestep Implementation**: Use a dedicated CUDA kernel for timestep sinusoidal embedding (sin/cos) to reduce per-step overhead in diffusion scheduling. See [PR #12995](https://github.com/sgl-project/sglang/pull/12995).

### 3. Cache-DiT Integration

We've integrated Cache-DiT **seamlessly** into SGLang Diffusion. With only a couple of environment variables, the generation speed is boosted by **up to** 169%. It is fully compatible with `torch.compile`.

We also support Cache-DiT with Ulysses Parallel, Ring Parallel, and Tensor Parallel, along with any hybrid combination of these three. See [PR #16532](https://github.com/sgl-project/sglang/pull/16532) & [PR #15163](https://github.com/sgl-project/sglang/pull/15163).

Furthermore, with the new run-with-diffusers backend feature, we can now integrate and refine Cache-DiT optimizations within SGLang Diffusion (see [Issue #16642](https://github.com/sgl-project/sglang/issues/16642)).

### 4. Few More Things

- **Diffusion Cookbook**: Curated recipes, best practices, and benchmarking guides for SGLang Diffusion.
- **Memory Monitoring**: Peak usage statistics available across offline generation and online serving workflows.
- **Profiling Suite**: Full-stage support with step-by-step docs for PyTorch Profiler and Nsight Systems.

## Further Roadmap

1. Disaggregated serving x Omni models
2. Sparse Attention Backends
3. Quantization (Nunchaku, nvfp4 and others)
4. Optimizations on consumer-level GPUs

## Performance Benchmark

As shown in the chart at the top of this post, we compared the performance of SGLang Diffusion:

- Against a popular open-source baseline, Hugging Face Diffusers. SGLang Diffusion delivers state-of-the-art performance, significantly accelerating both image and video generation.
- Under different parallelism setups. Both CFG-Parallel and USP deliver significant speedups compared to the single-GPU setup.

## Acknowledgment

**SGLang Diffusion Team**: [Yuhao Yang](https://github.com/yhyang201), [Xinyuan Tong](https://github.com/JustinTong0323), [Yi Zhang](https://github.com/yizhang2077), [Ke Bao](https://github.com/ispobock), [Ji Li](https://github.com/GeLee-Q/GeLee-Q), [Xi Chen](https://github.com/RubiaCx), [Laixin Xie](https://github.com/laixinn), [Yikai Zhu](https://github.com/zyksir), [Mick](https://mickqian.github.io)

Special thanks to NVIDIA and Voltage Park for their compute support.

## Learn more

- **Slack channel**: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A) (join via slack.sglang.io)
