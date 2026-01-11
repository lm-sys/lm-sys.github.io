---
title: "SGLang Diffusion: 2 Months After Release"
author: "The SGLang Diffusion Team"
date: "Januray 16, 2026"
previewImg: /images/blog/sgl-diffusion/sgl-diffusion-banner-16-9.png
---

Since its release in early November, **SGLang-diffusion** has been met with strong enthusiasm, widespread attention, and
valuable feedback from the community.

## Overview

In the past 2 months, we've been meticulously building sglang-diffusion, and here's what we've done:

- New Models:
    1. Day-0 support for Flux.2 / Qwen-Image-Edit-2511 / Qwen-Image-2512 / Z-Image-Turbo / Qwen-Image-Layered / TurboWan
       and more
    2. Run sglang-diffusion with diffusers backend: compatible with all models in diffusers, more improvements are
       coming (https://github.com/sgl-project/sglang/issues/16642)
- LoRA support:
    1. We support almost all lora formats for supported models, here are some example LoRAs that have been explicitly tested and verified with each base model in the SGLang Diffusion pipeline.

       | Base Model        | Supported LoRAs |
                                 |-------------------|------------------|
       | **Wan2.2**        | lightx2v/Wan2.2-Distill-Loras<br> Cseti/wan2.2-14B-Arcane_Jinx-lora-v1 |
       | **Wan2.1**        | lightx2v/Wan2.1-Distill-Loras |
       | **Z-Image-Turbo** | tarn59/pixel_art_style_lora_z_image_turbo<br> wcde/Z-Image-Turbo-DeJPEG-Lora |
       | **Qwen-Image**    | lightx2v/Qwen-Image-Lightning<br> flymy-ai/qwen-image-realism-lora<br> prithivMLmods/Qwen-Image-HeadshotX<br> starsfriday/Qwen-Image-EVA-LoRA |
       | **Qwen-Image-Edit** | ostris/qwen_image_edit_inpainting<br> lightx2v/Qwen-Image-Edit-2511-Lightning |
       | **Flux**          | dvyio/flux-lora-simple-illustration<br> XLabs-AI/flux-furry-lora<br> XLabs-AI/flux-RealismLora |

    2. Fully functional http api:

       | Feature                         | API Endpoint                | Key Parameters                                   |
                     |---------------------------------|-----------------------------|--------------------------------------------------|
       | Set/Activate (multiple) LoRA(s) | /v1/set_lora                | lora_nickname, lora_path, strength, target       |
       | Merge Weights                   | /v1/merge_lora_weights      | strength, target                                 |
       | Unmerge Weights                 | /v1/unmerge_lora_weights    | -                                                |
       | List Adapters                   | /v1/list_loras              | -                                                |


- Parallelism: SP for image models, TP for some models. We also support hybrid parallelism ( combinations of Ulysses
  Parallel, Ring Parallel, and Tensor Parallel).
- Attention: SageAttention2 / SageAttention3, more backends (sparse) are on the way
- Hardware: AMD / 4090 / 5090
- SGLang-diffusion x ComfyUI Integration: We have implemented a flexible ComfyUI custom node that integrates
  sglang-diffusion's high-performance inference engine. While ComfyUI offers exceptional flexibility through its custom
  nodes, it lacks multi-GPU support and optimal performance. Our solution replaces ComfyUI's denoising model forward
  pass with SGLang's optimized implementation, preserving ComfyUI's flexibility while leveraging SGLang's superior
  inference. Users can simply replace ComfyUI's loader node with our SGLDiffusion UNET Loader to enable enhanced
  performance without modifying existing workflows.

<img src="/images/blog/sgl-diffusion-26-01/comfyui.png" style="display:block; margin: auto; width: 85%;"></img>

## Highlighted works

As an industrial-level serving framework, speed and stability are what we care the most.

### 1. Layerwise Offload

From our early profiling, we've found that the model loading/offloading as a major bottleneck, since the forward stream has
to wait until all the weights are on-device.

To tackle this, we introduce:

1. LayerwiseOffloadManager: A manager class that provides hooks for prefetching weights of next layer while forwarding
   on the current layer
2. OffloadableDiTMixin: A mixin class that registers LayerwiseOffloadManager's prefetch and release hooks for the
   diffusion-transformer

This way, the per-layer forward doesn't have to wait for the copy stream, thus significantly improving the inference
speed, especially for special models like Wan2.2, where there are multiple DiTs

[Image]

Layerwise offload is enabled for video models by default, it reduces peak VRAM usage while accelerating the speed.

See related prs (https://github.com/sgl-project/sglang/pull/15511, https://github.com/sgl-project/sglang/pull/16150)

### 2. Kernel Improvements

1. FlashAttention kernel upstream:  we find that the FlashAttention kernel used in sglang-diffusion is behind the Dao-AILab upstream
   version, and therefore the performance is also slower than the upstream implementation. Besides, we should avoid
   using varlen format func in diffusion models. See https://github.com/sgl-project/sglang/pull/16382
2. JIT Qk norm kernel: fuse Q/K RMSNorm into a single inplace kernel to cut launch count and memory traffic before
   attention.
3. FlashInfer rope: apply RoPE on Q/K inplace with FlashInfer when available (fallback otherwise), reducing RoPE
   overhead and intermediate tensor materialization.
4. Weight fuse (operator fusion): fuse projection + activation patterns (e.g., gate/up merge + SiLU&Mul) to reduce GEMM
   count and elementwise launches in DiT blocks.
5. Implementation: timestep: use a dedicated CUDA kernel for timestep sinusoidal embedding (sin/cos) to reduce per-step
   overhead in diffusion scheduling. See https://github.com/sgl-project/sglang/pull/12995

### 3. Cache-DiT Integration

We've integrated **Cache-DiT**, the most popular framework for DiT cache, seamlessly into sglang-diffusion, with only a couple of env vars, the generation speed is boosted
by up to 169%. And it's fully compatible with torch.compile

We also support Ulysses Parallel, Ring Parallel, and Tensor Parallel, along with any hybrid combination of
these three. https://github.com/sgl-project/sglang/pull/16532 & https://github.com/sgl-project/sglang/pull/15163

Furthermore, with the new run-with-diffusers backend feature, we can now integrate and refine cache-dit optimizations
within sglang-diffusion (https://github.com/sgl-project/sglang/issues/16642)

### 4. Few More Things

- Diffusion Cookbook: curated recipes, best practices, and benchmarking guides for sglang-diffusion
- Memory monitoring: peak usage statistics available across offline generation and online serving workflows
- Profiling suite: full-stage support with step-by-step docs for PyTorch Profiler and Nsight Systems.

Further Roadmap

1. Disaggregated serving x omni models
2. Sparse Attention Backends
3. Quantization (nunchaku, nvfp4 and others)
4. Optimizations on consumer-level GPUs

## Performance Benchmark

As shown in the chart at the top of this post, we compared the performance of SGLang Diffusion:

- Against a popular open-source baseline, Hugging Face Diffusers. SGLang Diffusion delivers state-of-the-art
  performance, significantly accelerating both image and video generation.
- Under different parallelism setups. Both CFG-Parallel and USP deliver significant speedups compared to the single-GPU
  setup.

## Acknowledgment

SGLang Diffusion
Team: [Yuhao Yang](https://github.com/yhyang201), [Xinyuan Tong](https://github.com/JustinTong0323), [Yi Zhang](https://github.com/yizhang2077), [Ke Bao](https://github.com/ispobock), [Ji Li](https://github.com/GeLee-Q/GeLee-Q), [Xi Chen](https://github.com/RubiaCx), [Laixin Xie](https://github.com/laixinn), [Yikai Zhu](https://github.com/zyksir), [Mick](https://mickqian.github.io)

Special thanks to NVIDIA and Voltage Park for their compute support.

## Learn more

- Slack channel: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A) (join via slack.sglang.io)

