---
title: "SGLang Adds Day-0 Support for NVIDIA Nemotron 3.5 Lightning"
author: "NVIDIA Nemotron Team and SGLang Team"
date: "August 10, 2026"
previewImg: /images/blog/nemotron-3-5-lightning/pinchbench-accuracy-vs-time.png
type: blog
---

SGLang is excited to announce Day-0 support for NVIDIA Nemotron 3.5 Lightning, a customizable open model built to power always-on agents across local systems, the edge, the datacenter, and the cloud.

Always-on agents gather context, reason, use tools, and adapt across multi-step workflows. Frontier models handle complex orchestration, while smaller models efficiently manage high-volume, specialized tasks.

Nemotron 3.5 Lightning is built to handle these high-volume tasks. Distilled from NVIDIA Nemotron 3 Ultra and developed with the Nemotron Coalition, it combines strong coding, tool-calling, instruction-following, and multi-turn capabilities in a 30-billion-parameter hybrid mixture-of-experts model that activates only 3 billion parameters at a time.

Nemotron 3.5 Lightning can power local personal assistants, automate financial and risk workflows, support cybersecurity investigations, optimize telecommunications operations, and improve retail experiences. Organizations can post-train and deploy it for their specific terminology, policies, tools, and workflows.

With SGLang, developers can serve the model through a high-performance, OpenAI-compatible inference stack and connect it to agent harnesses, local assistants, and specialized enterprise workflows.

## TL;DR: NVIDIA Nemotron 3.5 Lightning

* **Architecture:** Hybrid mixture-of-experts architecture
* **Model size:** 30B total parameters, 3B active parameters
* **Context length:** Up to 1 million tokens
* **Speculative Decoding:** Multi-token prediction, DFlash, and DSpark
* **Modalities:** Text input and text output
* **Training:** Distilled from NVIDIA Nemotron 3 Ultra and trained for popular agent harnesses
* **Customization:** Open model trained with open datasets, with support for post-training on specialized workflows
* **Deployment targets:** NVIDIA DGX Spark, DGX Station, RTX PRO, RTX, NVIDIA Jetson, H100, H200, A100, L40S, B200/GB200, and B300/GB300
* **Availability at launch:** BF16, NVFP4
* **Get started:**
  * Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
  * Run Nemotron 3.5 Lightning with SGLang using the [getting-started cookbook](https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Lightning)

## Installation and Quick Start with SGLang

SGLang provides a high-performance serving runtime, continuous batching, prefix caching, speculative decoding, and an OpenAI-compatible API. The following baseline command uses the BF16 checkpoint:

```py
docker run --rm -it \
  --gpus all \
  --cap-add SYS_NICE \
  --ipc=host \
  --network=host \
  --entrypoint /bin/bash \
  lmsysorg/sglang:dev-nemotron3-5-lightning
```

```py
sglang serve \
    --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --max-running-requests 256 \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --mamba-backend flashinfer \
    --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser nemotron_3 \
    --tool-call-parser qwen3_coder
```

After the server starts, send a request with any OpenAI-compatible client:

```py
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is SGLang?"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)
choice = response.choices[0]
print("Reasoning:", choice.message.reasoning_content)
print("Content:", choice.message.content)
```

## Optimized Inference with Speculative Decoding

Nemotron 3.5 Lightning supports three speculative decoding techniques—Multi-Token Prediction (MTP), DFlash, and DSpark—to accelerate token generation while preserving the target model's output quality.

MTP uses lightweight, model-integrated prediction heads to propose several future tokens; DFlash uses a diffusion-based drafter to generate an entire candidate block in parallel; and DSpark adds confidence-aware, semi-autoregressive drafting to balance speed with token-acceptance quality. Together, they let teams choose the best latency, throughput, and deployment trade-off for their inference workload.

For low-latency serving, use DSpark across H100, H200, and DGX Spark; for maximum throughput today, we recommend running without speculative decoding.

### Run Nemotron 3.5 Lightning with MTP

Nemotron 3.5 Lightning includes multi-token prediction. SGLang exposes MTP through its speculative-decoding path, where the model's built-in prediction heads draft future tokens and the target model verifies them.

The standard SGLang MTP interface uses the EAGLE speculative algorithm:

```py
sglang serve \
    --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --max-running-requests 256 \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mamba-backend flashinfer \
    --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser nemotron_3 \
    --tool-call-parser qwen3_coder
```

### Run Nemotron 3.5 Lightning with DFlash

DFlash uses a dedicated diffusion draft model to propose a linear block of tokens that the target model verifies in parallel. In SGLang, DFlash requires a compatible draft checkpoint and is enabled separately from MTP.

SGLang's DFlash implementation does not support data-parallel attention and requires pipeline parallel size 1. See the [SGLang speculative-decoding guide](https://docs.sglang.io) for the current parameter reference.

### Run Nemotron 3.5 Lightning with DSpark

DSpark is a hybrid speculator that combines autoregressive and parallel diffusion-style drafting, sitting between MTP's fully autoregressive approach and DFlash's fully diffusion-based one, and delivers the best performance of the three on DGX Spark.

## Edge Deployment on NVIDIA Jetson

If you are running locally on NVIDIA Jetson, the following should provide a starting configuration for single-user local development:

```py
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nemotron35 \
  --trust-remote-code \
  --reasoning-parser nemotron_3 \
  --tool-call-parser qwen3_coder \
  --quantization modelopt_mixed \
  --context-length 8192 \
  --mem-fraction-static 0.70 \
  --max-running-requests 8 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-moe-runner-backend flashinfer_cutlass \
  --mamba-backend flashinfer \
  --mamba-ssm-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5
```

## Deploy locally on DGX Spark

If you are running locally on DGX Spark, the following should provide a starting configuration for single-user local development:

```py
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --trust-remote-code \
  --quantization modelopt_mixed \
  --context-length 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --max-running-requests 64 \
  --fp4-gemm-backend marlin \
  --moe-runner-backend marlin \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-moe-runner-backend flashinfer_cutlass \
  --mamba-backend flashinfer \
  --mamba-ssm-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --enable-cache-report
```

<!-- TODO(reviewer): add the DGX Spark inference pareto chart here -->

## Deploy on H100

If you are running on the NVIDIA H100, the following should provide a starting configuration for single-user local development:

```py
sglang serve \
    --model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --max-running-requests 256 \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mamba-backend flashinfer \
    --mamba-ssm-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --mamba-radix-cache-strategy extra_buffer \
    --reasoning-parser nemotron_3 \
    --tool-call-parser qwen3_coder
```

<!-- TODO(reviewer): add the H100 inference pareto chart here -->

## Control Reasoning for Each Agent Step

Nemotron 3.5 Lightning supports reasoning on or off, allowing a router or agent harness to use deeper reasoning for difficult steps and direct answers for routine work.

Reasoning is enabled by default. To request a direct answer without a reasoning trace, pass `enable_thinking: false` via `chat_template_kwargs`:

```py
response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[{"role": "user", "content": "Classify this ticket: billing or technical?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

For reasoning-enabled requests, either omit `chat_template_kwargs` (reasoning is on by default) or set it explicitly:

```py
response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[{"role": "user", "content": "Plan the steps to migrate this service."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
```

The model also supports a reasoning-token budget. Use `thinking_budget` (via `custom_params`) alongside `enable_thinking` to change the reasoning depth and response time per request:

```py
response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[{"role": "user", "content": "Debug why this build is failing."}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "custom_params": {"thinking_budget": 512},
    },
)
```

Reasoning control is particularly useful in systems of models: the orchestrator can allocate a larger budget to planning, coding, and ambiguous decisions, while using reasoning-off mode or a small budget for extraction, classification, and structured transformations.

## Optimizing inference for Nemotron 3.5 Lightning

Nemotron 3.5 Lightning is architecturally identical to Nemotron 3 apart from the weights and the speculative decoding stack, so most of the performance work landed in the runtimes themselves. Here's what we contributed upstream to SGLang:

* **DSpark integration.** We wired DSpark—a hybrid speculator that blends autoregressive and diffusion-style drafting—into SGLang and the Nemotron model definition, giving you three speculators to choose from alongside MTP and DFlash.
* **Quantized DSpark draft head.** Quantizing the draft head to W4A16 cuts its memory footprint and per-step latency without hurting acceptance rate, which matters most on memory-constrained parts like DGX Spark.
* **Removal of syncs and async scheduling.** We eliminated host-device syncs in the draft-and-verify loop and enabled async scheduling, so the next batch is prepared while the current one is still executing.
* **MoE and linear backend for W4A16.** We replaced SGLang's default Marlin backend—not tuned for Hopper—with the Hopper-optimized Humming backend, using W4A16 GEMM kernels for Nemotron's non-gated ReLU² MoE, worth roughly 20% throughput, and extended the same recipe to the dense linear layers.

## Nemotron 3.5 Lightning offers Leading Accuracy and Efficiency for Specialized AI

Nemotron 3.5 Lightning combines a hybrid MoE architecture—with only 3B of its 30B parameters active per token—with multi-token prediction to reduce computation and accelerate generation. These optimizations deliver up to 4x higher throughput than similarly sized open models, helping agents complete specialized tasks faster.

Distilled from NVIDIA Nemotron 3 Ultra and trained across popular agent harnesses, Nemotron 3.5 Lightning transfers frontier-level agentic capabilities into a compact, efficient model. It excels across benchmarks for agent productivity, coding, tool use, instruction following, and long-context reasoning.

As shown in Figure 1, higher inference throughput and token efficiency places Nemotron 3.5 Lightning on the efficiency frontier, helping always-on agents finish high-volume work faster.

![Line chart comparing PinchBench accuracy with time to complete 10,000 tasks. Nemotron 3.5 Lightning reaches similar accuracy as Qwen3.6-35B roughly 30% faster and sits well ahead of Gemma 4 26B.](/images/blog/nemotron-3-5-lightning/pinchbench-accuracy-vs-time.png)

Figure 1: Nemotron 3.5 Lightning leads the efficiency frontier by completing agentic tasks up to 30% faster at comparable accuracies.

Alt text: Line chart comparing PinchBench accuracy with time to complete 10,000 tasks. Nemotron 3.5 Lightning reaches similar accuracy as Qwen3.6 35B 30% faster.

## Summary

NVIDIA Nemotron 3.5 Lightning brings fast, customizable agentic intelligence to local systems, the edge, the datacenter, and the cloud. With SGLang Day-0 support, developers can serve the model through a high-performance, OpenAI-compatible stack; control reasoning per agent step; manage memory for local deployments such as DGX Spark; and accelerate generation with multi-token prediction, DFlash, or DSpark.

For systems that route work across multiple models, Nemotron 3.5 Lightning gives developers a compelling option for high-volume specialized tasks where accuracy, speed, openness, and deployment control all matter.

## Get Started

* Download the model weights from Hugging Face: [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
* Run Nemotron 3.5 Lightning with SGLang using the [cookbook](https://docs.sglang.io/cookbook/autoregressive/NVIDIA/Nemotron3.5-Lightning)

*Stay up to date on [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) by subscribing to NVIDIA news and following NVIDIA AI on [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all), [X](https://x.com/NVIDIAAIDev), [YouTube](https://www.youtube.com/@NVIDIADeveloper), and the [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) on [Discord](https://discord.com/invite/nvidiadeveloper).*

## Acknowledgements

Nirmal Kumar Juluru, Anusha Pant, Amir Klein, Faradawn Yang, Nave Assaf, Ryan Stewart, Alex Steiner, Bita Rouhani, Seong Hee Lee

## FAQs

**What is new compared with the Nemotron 3 Nano?**

Nemotron 3 Nano established an efficient hybrid Mamba-Transformer MoE design with 30B total parameters, 3B active parameters, a 1M-token context window, and controllable reasoning. Nemotron 3.5 Lightning builds on that foundation in three important ways:

* **Frontier-model distillation:** Nemotron 3.5 Lightning is distilled from Nemotron 3 Ultra, transferring capabilities from NVIDIA's frontier agentic model into a much smaller deployment footprint.
* **Agent-harness optimization:** Nemotron 3.5 Lightning is trained for popular agent harnesses and multi-turn workflows, with an emphasis on coding, tool use, instruction following, and specialized task completion.
* **Speculative decoding:** Nemotron 3.5 Lightning supports multi-token prediction (MTP), DFlash, and DSpark to accelerate generation by drafting and verifying multiple tokens in parallel.

The result is a model designed to complete more agent tasks more accurately in less time.
