---
title: "SGLang Diffusion: Diffusion Generation with SGLang"
author: "The SGLang Team"
date: "November 6, 2025"
previewImg: /images/blog/sglang/cover.jpg
---

## Introduction

We are excited to introduce SGLang Diffusion, bringing SGLang's state-of-the-art performance to accelerate image and video generation. In collaboration with the FastVideo team, we provide a complete ecosystem for both blazing-fast inference and model training, all accessible through our familiar, user-friendly APIs.

Source code available [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)

## Why Diffusion in SGLang?

With diffusion models becoming the backbone for state-of-the-art image and video generation, we have heard strong community demand for bringing SGLang's signature performance and seamless user experience to these new modalities. 

SGLang Diffusion is built to meet this need, providing a unified api for both language and diffusion tasks. The future of generation lies in combining modalities, as seen in pioneering models like ByteDance's [Bagel](https://github.com/ByteDance-Seed/Bagel) that fuse autoregressive and diffusion architectures. 


SGLang Diffusion is designed to be a future-proof, high-performance solution ready to power these innovative systems.




## Architecture

SGLang Diffusion is built upon SGLang's battle-tested, high-performance serving architecture. It inherits the powerful SGLang scheduler and reuses highly-optimized compute kernels for maximum efficiency. To accelerate development, our system is built on an enhanced fork of FastVideo, and we are collaborating closely with their team. This partnership creates a clear and powerful ecosystem: SGLang Diffusion focuses on delivering state-of-the-art inference speed, while FastVideo provides comprehensive support for training-related tasks like model distillation. For a seamless user experience, we provide a suite of familiar interfaces, including a CLI, a Python engine API, and an OpenAI-compatible API, allowing users to integrate diffusion generation into their workflows with minimal effort.

## Model Support

We support various popular open-source video & image generation models, including:
  - Video models: Wan-series, FastWan, Hunyuan
  - Image models: Qwen-Image, Qwen-Image-Edit, Flux

## Usage

### Install

SGL diffusion can be installed via multiple ways:

Install:
```bash
# with pip or uv
uv pip install sglang[.diffusion] --prerelease=allow

# from source
git clone https://github.com/sgl-project/sglang.git
cd sglang
uv pip install --prerelease=allow  -e "python/.[diffusion]"
```

Launch a server:
```bash
sglang serve --model-path black-forest-labs/FLUX.1-dev
```

Generate an image:
```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```


Reference [install guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/install.md) and [cli guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md) for more info



### Demo

#### Text to Video: Wan-AI/Wan2.1

```bash
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon" \
    --save-output
```


<video controls width="640" preload="metadata"
poster="">
  <source src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2V.mp4" type="video/mp4">
    Your browser doesn't support HTML5 videos
</video>


#### Image to Video: Wan-AI/Wan2.1-I2V

```bash
sglang generate --prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside. "\
    --image-path="https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true \
    --save-output --model-path=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --num-gpus 2 --enable-cfg-parallel 
```

<video controls width="640" preload="metadata"
poster="">
  <source src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2V.mp4" type="video/mp4">
    Your browser doesn't support HTML5 videos
</video>



#### Text to Image: FLUX

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
          --prompt "A Logo With Bold Large Text: SGL Diffusion" \
          --save-output
```



<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2I_FLUX.jpg" alt="Text to Image: FLUX" style="display:block; margin: auto; width: 85%;">


#### Text to Image: Qwen-Image

```bash
sglang generate \
    --prompt='A curious raccoon' --save-output \
    --width=720 --height=720 --model-path=Qwen/Qwen-Image
```

<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2I_Qwen_Image.jpg" alt="Text to Image: FLUX" style="display:block; margin: auto; width: 85%;">


#### Image to Image: Qwen-Image-Edit


```bash
sglang generate --prompt='keep the original style, but change the text to: \"SGL Diffusion\"' \
    --save-output --image-path="https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png" \
    --model-path=Qwen/Qwen-Image-Edit
```

<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/T2I_FLUX.jpg" alt="Text to Image: FLUX" style="display:block; margin: auto; width: 85%;">


## Performance Benchmark


<img src="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/SGLang.Diffusion.vs.Baseline.Performance.diffusers.svg" alt="SGLang Diffusion Performance Benchmark" style="display:block; margin: auto; width: 85%;">
<p style="color:gray; text-align: center;">Lower is better. Performance measured on a single NVIDIA A100 GPU.</p>

## Roadmap and Diffusion Ecosystem

Our vision is to build a comprehensive diffusion ecosystem in collaboration with the FastVideo team, providing an end-to-end solution from model training to high-performance inference. 

The SGLang Diffusion team is centered on continuous innovation in performance and model support:

- Model support and optimizations
  - Optimize Wan, FastWan, Hunyuan, Qwen-Image series, FLUX
  - Support LongCat-Video
- Kernel support and Fusions
  - Quantization kernels
  - Rotary embedding kernels
  - Flash Attention 4 integration in sgl-kernel for blackwell
- More server features
  - Configurable cloud storage upload of generated files
  - Batching support
  - More parallelism methods
- General architecture:
  - Simplify the new model support effort
  - Enhance cache and attention backend supports

Building this ecosystem is a community effort, and we welcome and encourage all forms of contribution. Join us in shaping the future of open-source diffusion generation.

## Acknowledgment


- SGLang Diffusion Team: [Mick](https://github.com/mickqian), [YuHao Yang](https://github.com/yhyang201), [Xinyuan Tong](https://github.com/JustinTong0323), [Yi Zhang](https://github.com/yizhang2077), [Ji Li](https://github.com/GeLee-Q/GeLee-Q), [Bao Ke](https://github.com/ispobock), [Xi Chen](https://github.com/RubiaCx), [LaiXin Xie](https://github.com/laixinn), [YiKai Zhu](https://github.com/zyksir)

- FastVideo Team: [SolitaryThinker](https://github.com/SolitaryThinker), [jzhang38](https://github.com/jzhang38), [BrianChen1129](https://github.com/BrianChen1129), [kevin314](https://github.com/kevin314), [Edenzzzz](https://github.com/Edenzzzz), [JerryZhou54](https://github.com/JerryZhou54), [rlsu9](https://github.com/rlsu9), [Eigensystem](https://github.com/Eigensystem), [foreverpiano](https://github.com/foreverpiano), [RandNMR73](https://github.com/RandNMR73), [PorridgeSwim](https://github.com/PorridgeSwim), [Gary-ChenJL](https://github.com/Gary-ChenJL)

## Learn more

- Roadmap: TBD
- Slack channel: [slack.sglang.ai](https://slack.sglang.ai) (`#diffusion`)



