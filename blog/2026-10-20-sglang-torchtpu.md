---
title: "SGLang-TorchTPU: PyTorch-Native Production Serving on TPU, Built with Google and Meta"
author: "RadixArk, Google and Meta"
date: "October 20, 2026"
previewImg: /images/blog/sglang-torchtpu/hero.png
type: blog
---

![SGLang-TorchTPU](/images/blog/sglang-torchtpu/banner.png)

## TL;DR

- SGLang-TorchTPU is open source today, built by the SGLang team at RadixArk in collaboration with Google and Meta.
- Launch models include: Muse Spark Open, GLM-5.3 Flash, GLM-5.3, Kimi K3, DeepSeek V4, Ling 3.1 Flash, Qwen3.8, Gemma 4 and MiniMax H3.
- Its plugin design brings SGLang's production features to large agentic workloads on TPU.
- Already in production at Meta.
- Enables RL training on TPU with TorchTitan in Miles.

## Introduction

Today we are launching SGLang-TorchTPU, a native PyTorch backend that brings SGLang's full serving stack to Google Cloud TPU. It enables the same SGLang engine to serve models on both GPUs and TPUs, so you can pick the hardware that best fits each workload and its deployment economics.

## Why TorchTPU

Most models today are written and served in PyTorch, and serving stacks are built on the ecosystem around it, from torch.compile to torch.distributed. We want TPU to fit into that same workflow.

TorchTPU makes TPU a native PyTorch device that runs eagerly or compiles through XLA. SGLang-TorchTPU builds on it and brings SGLang's serving features to TPU, on top of TPU-tuned Pallas kernels.

## Built with Google and Meta

Google built TorchTPU and worked with us on the TPU backend itself, from Pallas kernels to the TPU Sync transfer path. The SGLang team at RadixArk built the serving runtime, its TPU integration and most of the model support. Meta engineers contributed compilation and disaggregated-serving work, and brought production workloads that shaped the engine from the start.

> "As agentic workloads grow, serving efficiency matters more than ever, and developers want to get there on TPU without leaving PyTorch. Co-developing SGLang-TorchTPU with RadixArk and Meta gives them a production-ready engine for popular open models, with the cost efficiency of Cloud TPU."
>
> <strong style="font-style: normal;">Qi Zhou, Tech Lead on AI Inference, Google</strong>

> "SGLang-TorchTPU is running in production at Meta today, and getting there was a close three-way collaboration between Meta, Google, and the SGLang team to make the stack production-ready. It gives us a remarkably mature, PyTorch-native serving stack that integrates effortlessly into our existing AI infrastructure, with the performance, reliability, and flexibility we need to run frontier workloads efficiently on TPUs."
>
> <strong style="font-style: normal;">Zongwei Zhou, PyTorch-TPU Lead, Meta</strong>

## How it works

![The SGLang-TorchTPU stack: SGLang core (API server, router, scheduler, RadixAttention, model code) on top of the sglang-tpu plugin (TPU layers, Pallas kernels, KV cache pools, TPU Sync), on top of TorchTPU (tpu device, torch.compile + XLA, torch.distributed), on Cloud TPU v7x.](/images/blog/sglang-torchtpu/stack.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">Figure 1: The SGLang-TorchTPU stack.</p>

Each device runs its own PyTorch process, so SGLang's scheduler, RadixAttention and API server run unchanged. The TPU work sits underneath them. The KV cache lives in a paged pool laid out for Pallas ragged paged attention, with FP8 KV by default, and Pallas kernels for attention and MoE grouped matmul are called as PyTorch custom ops inside the torch.compile graph. All of it ships as the sglang-tpu plugin, which SGLang loads when you pass `--device tpu`.

KV cache transfers between TPU devices and host memory can run on [TPU Sync](https://github.com/google/tpu-sync), Google's open-source transfer library. For KV cache offloading, it moves KV pages to a DMA-pinned host memory pool. For prefill-decode disaggregation, the decode side pulls KV from prefill chunk by chunk as each prefill step commits, so only the last chunk is left to move when prefill ends. When the two sides use different tensor-parallel sizes, KV is resharded on the way, for example from 16 prefill shards to 8 decode shards.

![KV cache transfer with TPU Sync: in prefill-decode disaggregation, KV moves chunk by chunk from prefill (TP 16) to decode (TP 8); in KV cache offloading, KV is evicted from TPU HBM to host memory and reloaded.](/images/blog/sglang-torchtpu/kv-transfer.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">Figure 2: KV cache transfer with TPU Sync.</p>

## Feature and model coverage

These are the SGLang serving features available on TPU at launch:

| Area | Supported features |
| --- | --- |
| Core serving | OpenAI-compatible API; continuous batching; chunked prefill; structured output and function calling (experimental) |
| Agentic and multi-turn | RadixAttention prefix caching, including hybrid sliding-window and linear-attention models; KV cache offloading to host memory; tested nightly on an agentic tool-call workload |
| Interactive, low latency | Speculative decoding with EAGLE-3, DFlash and MTP drafters; DFlash2 and DSPark (experimental) |
| High-throughput batch | Data-parallel attention with expert parallelism |
| Long context | FP8 KV cache; context parallelism for prefill and decode (experimental) |
| Disaggregated serving | Prefill and decode on separate hosts, with KV resharding between different tensor-parallel sizes |
| Multimodal | Vision-language models with image input: Qwen3-VL, Qwen3.5-VL, Kimi MoonViT and Gemma 4 vision |
| Very large models | Tensor and expert parallelism and data-parallel attention across hosts; FP8, MXFP4 and INT4 checkpoints; pipeline parallelism and NVFP4 MoE (experimental) |

We have been working hard to bring the most popular open models to TPU at launch. A few of the newest are still experimental while we test them thoroughly and add them to CI.

| Model | Type |
| --- | --- |
| Muse Spark Open | LLM |
| GLM-5.3 Flash, GLM-5.3 | LLM |
| Kimi K3 | LLM |
| DeepSeek V4 | LLM |
| Ling 3.1 Flash, Ling 3.0 family | LLM |
| Qwen3.8 | LLM |
| Gemma 4 | LLM |
| MiniMax H3 | Diffusion |
| Wan2.2 | Diffusion |

Also supported: gpt-oss, Qwen3-Coder-480B, Qwen3.5, DeepSeek-V3.1 and R1, Kimi-K2.6, MiMo-V2.5-Pro and Qwen3-VL.

We will keep adding new models and optimizing the ones already supported, starting with DeepSeek V4.1 Flash and Reflection Beam.

The full list, with support tiers and per-model notes, is in [SUPPORT.md](https://github.com/sgl-project/sglang-torchtpu/blob/main/SUPPORT.md).

## Reliability

Reliability is designed into how SGLang-TorchTPU is built. Every model and feature marked supported is tied to a CI test on real TPU hardware, and the support matrix is generated from those test registrations, so what we claim always reflects what we run. Features still being validated are marked Experimental until their tests pass.

![SGLang-TorchTPU support tiers: Experimental moves to Supported with a nightly test, and Supported moves to Optimized with accuracy and performance tests, all on TPU v7x every PR and every night.](/images/blog/sglang-torchtpu/ci-tiers.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">Figure 3: How models and features move through the support tiers.</p>

Pull requests run a TPU suite on a single-host TPU v7x slice, and together with nightly suites, CI covers 15 models on single-host and multi-host TPU v7x, with 193 TPU tests registered today. They check:

- Kernel correctness
- End-to-end serving and model accuracy on standard evals such as GSM8K, GPQA, AIME, MMLU-Pro and MMMU
- Speculative decoding
- Prefix caching and KV cache offloading
- Multi-host execution and PD disaggregation, including KV resharding
- Agentic tool-call workloads on multi-host deployments

If a model or feature is marked supported, it has passed these tests on TPU v7x and is retested every night, so regressions show up in CI before they reach production.

## Training with TPU on Miles

Serving is half of the loop. To train models with reinforcement learning (RL), the same GPU-or-TPU choice has to hold for the trainer too, so we ran end-to-end RL training on TPU with the same PyTorch stack.

The trainer is TorchTitan, PyTorch's native training library, running on TorchTPU with no TPU-specific model code. As in serving, each TPU device runs its own PyTorch process. [Miles](https://github.com/radixark/miles), RadixArk's RL framework, ties the two sides together: SGLang-TorchTPU generates rollouts, TorchTitan trains on them, and after each step, or every few steps, Miles syncs the new weights from the trainer to the engine.

![RL training with Miles on one TPU v7x host: SGLang-TorchTPU generates rollouts on 2 chips, TorchTitan trains on 2 chips, and new weights move back to the engine through shared memory, disk or ICI (planned).](/images/blog/sglang-torchtpu/miles-rl.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">Figure 4: RL training with Miles on one TPU v7x host.</p>

Miles can move the new weights from the trainer to the engine in three ways:

- Shared memory, when the trainer and engine are on one host. This is what our runs use today.
- Disk, by writing a full checkpoint that the engine reloads. It works across hosts but is the slowest.
- ICI (planned), moving weights device to device over the TPU interconnect. Our test measured it about 50x faster than shared memory.

We trained Qwen3-4B with GRPO on DAPO-Math-17k on a single TPU v7x host with 4 chips: half for the trainer, half for rollouts. As the reference, we ran the same recipe and hyperparameters on 6 H200 GPUs, with two seeds on each platform to check parity.

![Qwen3-4B GRPO on DAPO-Math-17k: 6x H200 GPU vs. 4 TPU v7x chips](/images/blog/sglang-torchtpu/qwen3-4b-grpo-tpu-vs-gpu.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">Figure 5: Qwen3-4B GRPO on DAPO-Math-17k, 6x H200 GPUs vs. 4 TPU v7x chips.</p>

Reward and response length curves on TPU match GPU within seed-to-seed variation, and AIME-24 stays within eval noise.

Miles TPU training is in preview. This small-scale experiment demonstrates an end-to-end RL workflow on TPU, with early learning behavior similar to the GPU baseline. Next, we plan to work on:

- More models, including larger dense models and MoE models such as Qwen3-30B-A3B with routing replay
- Multi-host training, scaling beyond a single TPU v7x host
- Faster weight transfer, syncing weights over ICI instead of host shared memory
- Performance, closing the step-time gap with GPU
- Larger-scale validation, with parity checks on longer runs and long-horizon agentic tasks
- Training and inference consistency, with bitwise-identical log-probs between the two

## Get started

Pull the image and start a container on a TPU v7x host:

```bash
docker pull lmsysorg/sglang:v0.1.0-tpu

docker run -it --rm --privileged --net=host --shm-size=64g \
  -e SGLANG_PLATFORM=tpu_oot \
  lmsysorg/sglang:v0.1.0-tpu
```

Inside the container, launch GPT-OSS-20B with the single-host recipe, then send a request:

```bash
python -m benchmark.tpu.launch_server --config recipe/configs/gpt_oss_20b/server.py

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

To install with pip or from source instead, see the [README](https://github.com/sgl-project/sglang-torchtpu#get-started).

- GitHub: [sgl-project/sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu)
- Supported models and features: [SUPPORT.md](https://github.com/sgl-project/sglang-torchtpu/blob/main/SUPPORT.md), with the test behind each row in [SUPPORT_MATRIX.md](https://github.com/sgl-project/sglang-torchtpu/blob/main/SUPPORT_MATRIX.md)
- Cookbook: [cookbook.sglang.io](https://cookbook.sglang.io/)

## Community

We are working toward Day 0 TPU support for new open models by Q1 2027.

- RadixArk is opening a [community TPU capacity pool](https://docs.google.com/forms/d/e/1FAIpQLSenGavH0yrQWfcaMnVbcTg9cPyPHxMZG2LUegYJUYfai828hA/viewform) with free compute for developers who contribute to [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu).
- To ask questions, join #torchtpu on the [SGLang Slack](https://slack.sglang.io). To request a model or propose a feature, open an issue on [GitHub](https://github.com/sgl-project/sglang-torchtpu).
- Google, Meta and RadixArk will share more at SGLang Summit 2026, November 12 to 13 at Fort Mason, San Francisco ([sglang.io/summit](https://sglang.io/summit)).
