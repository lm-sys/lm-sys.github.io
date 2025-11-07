---
title: "SGLang Diffusion: Diffusion Generation with SGLang"
author: "The SGLang Team"
date: "November 6, 2025"
previewImg: https://github.com/lm-sys/lm-sys.github.io/releases/download/test/sgl-diffusion-banner-16-9.png
---

We are excited to introduce SGLang Diffusion, bringing SGLang's state-of-the-art performance to accelerate image and video generation. In collaboration with the **FastVideo team**, we provide a complete ecosystem for both blazing-fast inference and model training, all accessible through our familiar, user-friendly APIs.

Source code available [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)


<iframe
width="600"
height="371"
seamless
frameborder="0"
scrolling="no"
src="https://docs.google.com/spreadsheets/d/e/2PACX-1vT3u_F1P6TIUItyXdTctVV4pJVEcBuyPBTqmrdXR3KeQuiN1OdkIhjVNpZyHUDPw_5ZIKe88w2Xz6Dd/pubchart?oid=1360546403&format=interactive"
style="display:block; margin:15px auto 0 auto;">
</iframe>

<p style="color:gray; text-align: center;">SGL Diffusion Performance Benchmark</p>

## Why Diffusion in SGLang?


With diffusion models becoming the backbone for state-of-the-art image and video generation, we have heard strong community demand for bringing SGLang's signature performance and seamless user experience to these new modalities. 


We built SGLang Diffusion to answer this call, providing a unified, high-performance engine for both language and diffusion tasks. This unified approach is crucial, as the future of generation lies in combining architectures. 

Pioneering models are already fusing the strengths of autoregressive (AR) and diffusion-based approaches—from models like ByteDance's [Bagel](https://github.com/ByteDance-Seed/Bagel) and Meta's [Transfusion](https://arxiv.org/abs/2408.11039) that use a single transformer for both tasks, to NVIDIA's [Fast-dLLM v2](https://nvlabs.github.io/Fast-dLLM/v2/) which adapts AR models for parallel generation.

SGLang Diffusion is designed to be a future-proof, high-performance solution ready to power these innovative systems.


## Architecture

SGLang Diffusion is engineered for both performance and flexibility, built upon SGLang's battle-tested serving architecture. It inherits the powerful SGLang scheduler and reuses highly-optimized compute kernels for maximum efficiency.

At its core, our architecture is designed to accommodate the diverse structures of modern diffusion models. We introduce `ComposedPipelineBase`, a flexible abstraction that orchestrates a series of modular `PipelineStage`s. Each stage encapsulates a common diffusion function—such as the denoising loop in `DenoisingStage` or VAE decoding in `DecodingStage`—allowing developers to easily combine and reuse these components to construct complex, customized pipelines.

To achieve state-of-the-art speed, we integrate advanced parallelism techniques. We support Unified Sequence Parallelism (USP)—a combination of Ulysses-SP and Ring-Attention—for the core transformer blocks, alongside CFG-parallelism and tensor parallelism (TP) for other model components.

To accelerate development and foster a powerful ecosystem, our system is built on an enhanced fork of **FastVideo**, and we are collaborating closely with their team. This partnership allows SGLang Diffusion to focus on delivering cutting-edge inference speed, while **FastVideo** provides comprehensive support for training-related tasks like model distillation.

## Model Support

We support various popular open-source video & image generation models, including:
  - Video models: Wan-series, FastWan, Hunyuan
  - Image models: Qwen-Image, Qwen-Image-Edit, Flux

For full list of supported models, reference [here](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/support_matrix.md).


## Usage

For a seamless user experience, we provide a suite of familiar interfaces, including a CLI, a Python engine API, and an OpenAI-compatible API, allowing users to integrate diffusion generation into their workflows with minimal effort.

### Install

SGL diffusion can be installed via multiple ways:

```bash
# with pip or uv
uv pip install 'sglang[diffusion]' --prerelease=allow

# from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[diffusion]" --prerelease=allow
```
### CLI

Launch a server and then send requests:
```bash
sglang serve --model-path black-forest-labs/FLUX.1-dev




```

Or, Generate an image without launching a server:
```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```


Reference [install guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/install.md) and [cli guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md) for more installation methods.


### Demo

#### Text to Video: Wan-AI/Wan2.1

```bash
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon" \
    --save-output
```


<video controls width="640" preload="metadata"   style="display:block; margin-top: 20px;"
poster="">
  <source src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2V.mp4" type="video/mp4">
    Your browser doesn't support HTML5 videos
</video>


#### Image to Video: Wan-AI/Wan2.1-I2V

```bash
sglang generate --model-path=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers  \
    --prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --image-path="https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true" \
    --save-output --num-gpus 2 --enable-cfg-parallel 
```

<video controls width="640" preload="metadata" style="display:block; margin-top: 20px"
poster="">
  <source src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2V.mp4" type="video/mp4" margin-top: 20px>
    Your browser doesn't support HTML5 videos
</video>


#### Text to Image: FLUX

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --save-output
```


<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2I_FLUX.jpg" alt="Text to Image: FLUX" style="display:block; margin-top: 20px; width: 65%;">


#### Text to Image: Qwen-Image

```bash
sglang generate --model-path=Qwen/Qwen-Image \
    --prompt='A curious raccoon' \
    --width=720 --height=720 \
    --save-output \
```

<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2I_Qwen_Image.jpg" alt="Text to Image: FLUX" style="display:block; margin-top: 20px; width: 65%;">


#### Image to Image: Qwen-Image-Edit


```bash
sglang generate --text-encoder-cpu-offload --pin-cpu-memory \
  --prompt="Convert 2D style to 3D style" --image-path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg" --model-path=Qwen/Qwen-Image-Edit \
   --width=1024  --height=1536 --save-output
```


<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg" alt="Input" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
    <div style="margin-top: -25px;">Input</div>
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Output.jpg" alt="Output" style="max-width: 100%; height: auto; border: 1px solid #ccc;">
    <div style="margin-top: -25px;">Output</div>
  </div>
</div>




## Performance Benchmark
As benchmarked in the chart at the top of this post, SGLang Diffusion delivers state-of-the-art performance, significantly accelerating both image and video generation.

## Roadmap and Diffusion Ecosystem

Our vision is to build a comprehensive diffusion ecosystem in collaboration with the **FastVideo** team, providing an end-to-end solution from model training to high-performance inference. 

The SGLang Diffusion team is centered on continuous innovation in performance and model support:

- Model support and optimizations
  - Optimize Wan, FastWan, Hunyuan, Qwen-Image series, FLUX
  - Support LongCat-Video
- Kernel support and fusions
  - Quantization kernels
  - Rotary embedding kernels
  - Flash Attention 4 integration in sgl-kernel for blackwell
- More server features
  - Configurable cloud storage upload of generated files
  - Batching support
  - More parallelism methods
  - Quantization
- General architecture:
  - Simplify the effort of supporting new models
  - Enhance cache and attention backend supports

Building this ecosystem is a community effort, and we welcome and encourage all forms of contribution. Join us in shaping the future of open-source diffusion generation.

## Acknowledgment

SGLang Diffusion Team: [Yuhao Yang](https://github.com/yhyang201), [Xinyuan Tong](https://github.com/JustinTong0323), [Yi Zhang](https://github.com/yizhang2077), [Bao Ke](https://github.com/ispobock), [Ji Li](https://github.com/GeLee-Q/GeLee-Q), [Xi Chen](https://github.com/RubiaCx), [Laixin Xie](https://github.com/laixinn), [Yikai Zhu](https://github.com/zyksir), [Mick](https://github.com/mickqian)

FastVideo Team: [Peiyuan Zhang](https://github.com/SolitaryThinker), [jzhang38](https://github.com/jzhang38), [BrianChen1129](https://github.com/BrianChen1129), [kevin314](https://github.com/kevin314), [Edenzzzz](https://github.com/Edenzzzz), [JerryZhou54](https://github.com/JerryZhou54), [rlsu9](https://github.com/rlsu9), [Eigensystem](https://github.com/Eigensystem), [foreverpiano](https://github.com/foreverpiano), [RandNMR73](https://github.com/RandNMR73), [PorridgeSwim](https://github.com/PorridgeSwim), [Gary-ChenJL](https://github.com/Gary-ChenJL)

## Learn more

- Roadmap: [Diffusion (2025 Q4)](https://github.com/sgl-project/sglang/issues/12799)
- Slack channel: [#diffusion](https://sgl-fru7574.slack.com/archives/C09P0HTKE6A)



