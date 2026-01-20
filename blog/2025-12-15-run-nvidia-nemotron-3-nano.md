---
title: "SGLang Adds Day-0 Support for the Highly Efficient, Open Nemotron 3 Nano Hybrid MoE Model"
author: "NVIDIA Nemotron Team"
date: "December 15, 2025"
previewImg: /images/blog/nemotron-3-nano/benchmark.png
---

We are excited to announce that SGLang supports the latest highly efficient NVIDIA Nemotron 3 Nano model on Day 0!

Nemotron 3 Nano, part of the newly announced open [Nemotron 3 family](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/),  is a compact MoE language model offering industry-leading compute efficiency and accuracy, enabling developers to build specialized agentic AI systems. 

Nemotron 3 Nano is fully open with open-weights, datasets and recipes so developers can easily customize, optimize, and deploy the model on their infrastructure for maximum privacy and security. The chart below shows that Nemotron 3 Nano is in the most attractive quadrant in Artificial Analysis Openness vs Intelligence Index.


![figure1](/images/blog/nemotron-3-nano/artificial_analysis.png)<small><center>NVIDIA Nemotron 3 Sets a New Standard for Open Source AI</center></small>

## TL;DR


- Architecture:
    - Mixture of Experts (MoE) with Hybrid Transformer-Mamba Architecture
    - Supports Thinking Budget for providing optimal accuracy with minimum reasoning token generation
- Accuracy
    - Leading accuracy on coding, scientific reasoning, math, and instruction following 
- Model size: 30B with 3.6B active parameters
- Context length: 1M
- Model input: Text
- Model output: Text
- Supported GPUs: NVIDIA RTX Pro 6000, DGX Spark, H100, B200. 
- Get started: 
    - Download model weights from Hugging Face -  [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
    - [Run with SGLang for inference](https://cookbook.sglang.io/docs/NVIDIA/Nemotron3-Nano)
    - [Technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) to build custom, optimized models with Nemotron techniques.

## Installation and Quick Start

For an easier setup with SGLang, refer to our getting started cookbook, available [here](https://cookbook.sglang.io/docs/NVIDIA/Nemotron3-Nano) or through NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikQZX0ZDTSCGE7YkqxiOKwKsj). 

Run the command below to install dependencies:
```bash
uv pip install sglang==0.5.6.post3.dev1278+gad1b4e472 --extra-index-url https://sgl-project.github.io/whl/nightly/
```

We can then serve this model:
```bash
# BF16
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16 --trust-remote-code --reasoning-parser nano_v3 --tool-call-parser qwen3_coder

# FP8
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-FP8 --trust-remote-code --reasoning-parser nano_v3 --tool-call-parser qwen3_coder

# NVFP4
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-NVFP4 --trust-remote-code --reasoning-parser nano_v3 --tool-call-parser qwen3_coder
```

Once the server is up and running, you can prompt the model using the below code snippets:

```python
from openai import OpenAI

# The model name we used when launching the server.
SERVED_MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16"

BASE_URL = f"http://localhost:30000/v1"
API_KEY = "EMPTY"  # SGLang server doesn't require an API key by default

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

resp = client.chat.completions.create(
    model=SERVED_MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Give me 3 bullet points about SGLang."}
    ],
    temperature=0.6,
    max_tokens=512,
)
print(resp.choices[0].message.reasoning_content, resp.choices[0].message.content)
```


## Nemotron 3 Nano provides highest efficiency with leading accuracy for building AI agents

Nemotron 3 Nano builds on the hybrid Mamba-Transformer architecture by replacing standard feed-forward network (FFN) layers with MoE layers and most of the attention layers with Mamba-2. This enables higher accuracy while using only a fraction of the active parameters. By leveraging MoE, Nemotron 3 Nano reduces compute demands and satisfies the tight latency constraints required for real-world deployment.

Nemotron 3 Nano’s hybrid Mamba-Transformer architecture boosts token throughput by up to 4x, allowing the model to reason more quickly while delivering higher accuracy. Its “thinking budget” feature helps avoid unnecessary computation, reducing overthinking and ensuring lower, more predictable inference costs.

![figure1](/images/blog/nemotron-3-nano/speed.png)<small><center>Nemotron 3 Nano delivers higher throughput with leading accuracy among open reasoning models</center></small>


Trained on NVIDIA-curated, high-quality data, Nemotron 3 Nano leads on benchmarks such as SWE Bench Verified, GPQA Diamond, AIME 2025, Arena Hard v2, and IFBench delivering top-tier accuracy in coding, [reasoning](https://www.nvidia.com/en-us/glossary/ai-reasoning/), math and instruction following. This makes it ideal for building AI agents for various enterprise use cases including finance, cybersecurity, software development and retail. 

![figure1](/images/blog/nemotron-3-nano/benchmark.png)<small><center>Nemotron 3 Nano provides leading accuracy on various popular academic benchmarks among open small reasoning models</center></small>



## Get Started

- Download Nemotron 3 Nano model weights from Hugging Face -  [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8), [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- Run with SGLang for inference using [this](https://cookbook.sglang.io/docs/NVIDIA/Nemotron3-Nano) cookbook or through this NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikQZX0ZDTSCGE7YkqxiOKwKsj). 


## Further Reading
- [Share your ideas](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335) and vote on what matters to help shape the future of Nemotron. 
- Stay up to date on [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) by subscribing to NVIDIA news and following NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), and the [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) on [Discord](https://discord.com/invite/nvidiadeveloper).


## Acknowledgement

We thank all contributors for their efforts in developing and integrating Nemotron V3 Nano into SGLang.

**Nvidia Team**: Roi Koren, Max Xu, Netanel Haber, Tomer Bar Natan, Daniel Afrimi, Nirmal Kumar Juluru, Ann Guan and many more

**SGLang Team and community**: Baizhou Zhang, Jiajun Li, Ke Bao, Mingyi Lu, Richard Chen
