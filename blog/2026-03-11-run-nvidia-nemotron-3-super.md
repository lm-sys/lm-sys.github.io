---
title: "SGLang Adds Day-0 Support for NVIDIA Nemotron 3 Super for building High-Efficiency Multi-Agent Systems"
author: "NVIDIA Nemotron Team"
date: "March 11, 2026"
previewImg: /images/blog/nemotron-3-super/figure_1.svg
---

We are excited to announce that SGLang supports NVIDIA Nemotron 3 Super on Day 0.

[Nemotron 3 Super](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/) is a leading open model in the Nemotron 3 family, built for running many collaborating agents together. Agentic systems that chain planning, reasoning, and tools produce far more tokens than single-turn chat; they also need strong reasoning on every step. 

Nemotron 3 Super is a 120B-parameter hybrid MoE that activates only 12B parameters per forward pass, giving you leading accuracy for coding, tool calling, and instruction following at a fraction of the cost—plus a 1M-token context so agents keep conversation and plan state in view across long workflows.

![figure1](/images/blog/nemotron-3-super/figure_1.svg)<small><center>[Artificial Analysis](https://artificialanalysis.ai/) chart showing Nemotron 3 Super leading on intelligence vs. openness when compared to popular open models of similar size</center></small>

As you can see in the chart above, Nemotron 3 Super leads on the Artificial Analysis Openness index. When compared to other open models, Nemotron is fully open with open-weights, datasets, and recipes so developers can easily customize, optimize, and deploy on their infrastructure for maximum privacy and security.

In this post we walk through installing SGLang and serving Nemotron 3 Super for inference.


## About Nemotron3 Super


- **Architecture**: Mixture of Experts (MoE) with Hybrid Transformer-Mamba Architecture  
  - Highest throughput efficiency in its size category and up to 5x higher throughput compared to previous Nemotron Super model (Llama Nemotron Super 1.5)
  - Multi-Token Prediction (MTP) : By predicting several future tokens simultaneously in a single forward pass, MTP drastically accelerates the generation of long-form text
  - Supports Thinking Budget for optimal accuracy with minimum reasoning token generation  
- **Accuracy**: Leading accuracy on Artificial Analysis Intelligence Index in its size category
  - Up to 2x higher accuracy on Artificial Analysis Intelligence Index compared to previous Nemotron Super model.
  - Latent MoE enables calling 4 experts for the inference cost of only one 
- **Model size**: 120B total parameters, 12B active parameters
- **Context length**: up to 1M
- **Model I/O**: Text in, text out
- **Supported GPUs**:  B200, H100, H200, DGX Spark, RTX 6000
- **Get started**: 
    - Download model weights from Hugging Face -  [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
    - [Run with SGLang for inference](https://cookbook.sglang.io/autoregressive/NVIDIA/Nemotron3-Super)
    - [Technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) to build custom, optimized models with Nemotron techniques.

## Installation and Quick Start

For an easier setup with SGLang, refer to our getting started cookbook, available [here](https://cookbook.sglang.io/autoregressive/NVIDIA/Nemotron3-Super) or through NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-39d03Y3mDAiGuIrnHKmZwv0tA4s). 

Run the command below to install dependencies:
```bash
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

We can then serve this model. The command below is configured for a 4xH200 setup. Refer to the cookbooks for detailed instructions
```bash
# BF16
```bash
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --host 0.0.0.0 \
  --port 5000 \
  --trust-remote-code \
  --tp 4 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_3
```

Once the server is up and running, you can prompt the model using the below code snippets:

```python
from openai import OpenAI

# The model name we used when launching the server.
SERVED_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

BASE_URL = f"http://localhost:5000/v1"
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
print("Reasoning:", resp.choices[0].message.reasoning_content, "\nContent:", resp.choices[0].message.content)
```


## Nemotron 3 Super is ideal for multi-agent and reasoning workloads

![figure2](/images/blog/nemotron-3-super/figure_2.svg)<small><center>[Artificial Analysis](https://artificialanalysis.ai/) chart showing Nemotron 3 Super leading on intelligence vs. efficiency when compared to popular open models of similar size</center></small>

As you can see in the chart above, the model achieves leading accuracy with higher efficiency on Artificial analysis benchmarks making it a strong choice for multi-agent systems that need both efficiency and capability.

The 1M-token context is built for long-horizon agent work: agents can keep full conversation history and plan state in context, and RAG pipelines can supply large document sets in one shot. That reduces fragmentation and goal drift in multi-step workflows.

Together, this makes Super a strong choice for orchestrating and running many agents on a single node—from code generation and debugging to research summarization, alert triage, and document analysis.

## Get Started

Nemotron 3 Super helps you build scalable, cost-efficient multi-agent AI with high accuracy. With open weights, datasets, and recipes, you get full transparency and the flexibility to fine-tune and deploy on your own infrastructure, from workstation to cloud.

Ready to run multi-agent AI at scale?
- Download Nemotron 3 Super model weights from Hugging Face - [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
- Run with SGLang for inference using the [cookbook](https://cookbook.sglang.io/autoregressive/NVIDIA/Nemotron3-Super) and through [Brev launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-39d03Y3mDAiGuIrnHKmZwv0tA4s)
- Read the Nemotron 3 Super [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## Acknowledgement
Thanks to everyone who contributed to bringing Nemotron 3 Super to SGLang.

**NVIDIA**: Nirmal Kumar Juluru, Anusha Pant, Max Xu, Daniel Afrimi, Shahar Mor,  Roi Koren, Ann Guan and many more
**SGLang team and community**:  Baizhou Zhang, Jiajun Li, Ke Bao, Lingyan Hao, Mingyi Lu
