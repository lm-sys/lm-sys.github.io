---
title: "SGLang-TorchTPU: PyTorch-Native Production Serving on TPUs"
author: "RadixArk, Google and Meta"
date: "October 20, 2026"
previewImg: /images/blog/sglang-torchtpu/hero.png
type: blog
---

![SGLang-TorchTPU](/images/blog/sglang-torchtpu/banner.png)

## TL;DR

- SGLang-TorchTPU is open source today, built by the SGLang team at RadixArk in collaboration with Google and Meta.
- Launch models include Muse Spark Open, GLM-5.3 Flash, GLM-5.3, Kimi K3, DeepSeek V4, Ling 3.1 Flash, Qwen3.8, Gemma 4 and MiniMax H3.
- The plugin brings the full SGLang feature set to TPU, ready for agentic workloads.
- It is already proven in large-scale production at Meta.
- RL training on TPU works with SGLang-TorchTPU and TorchTitan in Miles.
- Contributors can apply for free TPU compute from RadixArk's [community TPU capacity pool](https://docs.google.com/forms/d/e/1FAIpQLSenGavH0yrQWfcaMnVbcTg9cPyPHxMZG2LUegYJUYfai828hA/viewform).

## Introduction

Today we are launching SGLang-TorchTPU, a native PyTorch backend that brings SGLang's full serving stack to Google Cloud TPU. SGLang's serving engine and model implementations are built on PyTorch, and with TorchTPU the same codebase serves on both GPUs and TPUs, so you can pick the hardware that best fits each workload and its deployment economics.

TPU-specific components remain isolated in the plugin. New models and features added to SGLang could therefore become available on TPU through a single implementation.

## How it works

![SGLang core (scheduler, RadixAttention, API server) on top of the sglang-tpu plugin (Pallas ragged paged attention, paged KV cache pools, Pallas grouped matmul MoE, a graph runner that compiles once per bucket), on TorchTPU (one PyTorch process per TPU core, torch.compile to XLA), on Cloud TPU v7x.](/images/blog/sglang-torchtpu/stack.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 1: How the sglang-tpu plugin fits into SGLang.</p>

SGLang-TorchTPU is an out-of-tree plugin. Installing the sglang-tpu package registers a TPU platform with SGLang through the `sglang.srt.platforms` entry point, and `SGLANG_PLATFORM=tpu_oot` selects it. The scheduler, RadixAttention and API server are SGLang's own. The plugin fills in the parts SGLang already asks each hardware platform for: the attention backend, the KV cache pools, the MoE kernels and the graph runner.

Each TPU core runs its own PyTorch process through TorchTPU, as each GPU does on CUDA. The model runs under torch.compile and lowers to XLA. There is no graph capture: instead, the runner pads each forward to a fixed set of token-count buckets for prefill and batch-size buckets for decode. Each bucket compiles once, and later steps dispatch to the cached program by shape.

Attention and MoE run on Pallas kernels, registered as torch.library custom ops that Dynamo treats as opaque calls inside the compiled graph: ragged paged attention over a paged KV cache, and a grouped matmul over the sorted expert groups for MoE.

When KV has to leave the device, for KV cache offloading to host memory or for prefill-decode disaggregation across hosts, it can move over [TPU Sync](https://github.com/google/tpu-sync), Google's open-source TPU DMA library. Offload copies whole KV pages between HBM and a pinned host pool by direct PJRT DMA. For disaggregation the transfer is consumer-pull: as each chunked-prefill step commits its pages, the prefill announces them and the decode side pulls that chunk while the next one computes, so only the last chunk remains to move once prefill finishes. Pages the decode already holds from its own prefix cache are never sent. When prefill and decode use different tensor-parallel sizes, TPU Sync reshards in flight: each prefill rank pushes its slice of the KV heads, strided, into the decode rank's page layout.

![KV cache transfer with TPU Sync: decode pulls each prefill chunk while the next computes, so only the last chunk waits; TP 16 to TP 8 reshards strided KV heads; offload moves KV pages between TPU HBM and a host pool by DMA.](/images/blog/sglang-torchtpu/kv-transfer.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 2: KV cache transfer with TPU Sync.</p>

## Feature and model coverage

These are the major SGLang serving features available on TPU at launch:

| Area | Supported features |
| --- | --- |
| Core serving | OpenAI-compatible API; continuous batching; chunked prefill; structured output and function calling (experimental) |
| Agentic and multi-turn | RadixAttention prefix caching, including hybrid sliding-window and linear-attention models; KV cache offloading to host memory; tested nightly on an agentic tool-call workload |
| Interactive, low latency | Speculative decoding with EAGLE-3, DFlash, DFlash2, DSPark and MTP drafters |
| High-throughput batch | Data-parallel attention with expert parallelism |
| Long context | FP8 KV cache; context parallelism for prefill and decode |
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

We will keep adding new models and optimizing the ones already supported, with DeepSeek V4.1 Flash and Reflection Beam coming next. From Q1 2027, our goal is Day 0 TPU support for new open models.

The full list, with support tiers and per-model notes, is in [SUPPORT.md](https://github.com/sgl-project/sglang-torchtpu/blob/main/SUPPORT.md).

## Reliability

Every model and feature marked supported has a CI test on real TPU hardware, and the support matrix is generated from those test registrations. Anything still being validated stays Experimental until its tests pass.

![SGLang-TorchTPU support tiers: Experimental moves to Supported with a nightly test, and Supported moves to Optimized with accuracy and performance tests, all on TPU v7x every PR and every night.](/images/blog/sglang-torchtpu/ci-tiers.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 3: How models and features move through the support tiers.</p>

Pull requests run a TPU suite on a single-host TPU v7x slice, and together with nightly suites, CI covers 15 models on single-host and multi-host TPU v7x, with 193 TPU tests registered today. They check:

- Kernel correctness
- Model accuracy on standard evals such as GSM8K, GPQA, AIME, MMLU-Pro and MMMU
- Serving features: speculative decoding, prefix caching, KV cache offloading, DP attention, multi-host execution and PD disaggregation with KV resharding
- Feature combinations, such as speculative decoding with PD disaggregation, and prefix caching with KV cache offloading
- Agentic tool-call workloads on multi-host deployments

Nightly runs repeat these checks for every supported model, so the support matrix stays current.

## Training with TPU on Miles

Serving is half of the loop. To train models with reinforcement learning (RL), the same GPU-or-TPU choice has to hold for the trainer too, so we ran end-to-end RL training on TPU with the same PyTorch stack.

The trainer is TorchTitan, PyTorch's native training library, running on TorchTPU with no TPU-specific model code. As in serving, each TPU device runs its own PyTorch process. [Miles](https://github.com/radixark/miles), RadixArk's RL framework, ties the two sides together: SGLang-TorchTPU generates rollouts, TorchTitan trains on them, and after each step, or every few steps, Miles syncs the new weights from the trainer to the engine.

![RL training with Miles on one TPU v7x host: SGLang-TorchTPU generates rollouts on 2 chips, TorchTitan trains on 2 chips, and new weights move back to the engine through shared memory, disk or ICI (planned).](/images/blog/sglang-torchtpu/miles-rl.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 4: RL training with Miles on one TPU v7x host.</p>

Miles can move the new weights from the trainer to the engine in three ways:

- Shared memory, when the trainer and engine are on one host. This is what our runs use today.
- Disk, by writing a full checkpoint that the engine reloads. It works across hosts but is the slowest.
- ICI (planned), moving weights device to device over the TPU interconnect. Our test measured it about 50x faster than shared memory.

We trained Qwen3-4B with GRPO on DAPO-Math-17k on a single TPU v7x host with 4 chips: half for the trainer, half for rollouts. As the reference, we ran the same recipe and hyperparameters on 6 H200 GPUs, with two seeds on each platform to check parity.

![Qwen3-4B GRPO on DAPO-Math-17k: 6x H200 GPU vs. 4 TPU v7x chips](/images/blog/sglang-torchtpu/qwen3-4b-grpo-tpu-vs-gpu.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 5: Qwen3-4B GRPO on DAPO-Math-17k, 6x H200 GPUs vs. 4 TPU v7x chips.</p>

Reward and response length curves on TPU match GPU within seed-to-seed variation, and AIME-24 stays within eval noise.

Miles TPU training is in preview. This small-scale experiment demonstrates an end-to-end RL workflow on TPU, with early learning behavior similar to the GPU baseline. Next, we plan to work on:

- More models, including larger dense models and MoE models such as Qwen3-30B-A3B with routing replay
- Multi-host training, scaling beyond a single TPU v7x host
- Faster weight transfer, syncing weights over ICI instead of host shared memory
- Performance, closing the step-time gap with GPU
- Larger-scale validation, with parity checks on longer runs and long-horizon agentic tasks
- Training and inference consistency, with bitwise-identical log-probs between the two

## Built with Google and Meta

SGLang-TorchTPU was developed jointly by Google, Meta and the SGLang team at RadixArk.

> "As agentic workloads grow, serving efficiency matters more than ever, and developers want to get there on TPU without leaving PyTorch. Co-developing SGLang-TorchTPU with RadixArk and Meta gives them a production-ready engine for popular open models, with the cost efficiency of Cloud TPU."
>
> <strong style="font-style: normal;">Qi Zhou, Tech Lead on AI Inference, Google</strong>

> "SGLang-TorchTPU is running in production at Meta today, and getting there was a close three-way collaboration between Meta, Google, and the SGLang team to make the stack production-ready. It gives us a remarkably mature, PyTorch-native serving stack that integrates effortlessly into our existing AI infrastructure, with the performance, reliability, and flexibility we need to run frontier workloads efficiently on TPUs."
>
> <strong style="font-style: normal;">Zongwei Zhou, PyTorch-TPU Lead, Meta</strong>

## Get started

On a TPU v7x host, start the prebuilt image, launch a model and send a request:

```bash
# Start the container
docker run -it --rm --privileged --net=host --shm-size=64g \
  -e SGLANG_PLATFORM=tpu_oot lmsysorg/sglang:v0.1.0-tpu

# Inside the container, launch GPT-OSS-20B
python -m benchmark.tpu.launch_server \
  --config recipe/configs/gpt_oss_20b/server.py

# From another terminal, send a request
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

To install with pip or from source, see the [README](https://github.com/sgl-project/sglang-torchtpu#get-started). The [TPU guide](https://docs.sglang.io/docs/hardware-platforms/tpu) covers deployment, the [recipes](https://github.com/sgl-project/sglang-torchtpu/blob/main/benchmark/tpu/recipe/README.md) have ready-to-run server configs, and [SUPPORT.md](https://github.com/sgl-project/sglang-torchtpu/blob/main/SUPPORT.md) lists the supported models and features.

## Community

RadixArk is offering free TPU compute to developers who contribute to SGLang-TorchTPU. [Apply for the community TPU capacity pool here](https://docs.google.com/forms/d/e/1FAIpQLSenGavH0yrQWfcaMnVbcTg9cPyPHxMZG2LUegYJUYfai828hA/viewform).

For questions, join #torchtpu on the [SGLang Slack](https://slack.sglang.io). To request a model or propose a feature, [open an issue on GitHub](https://github.com/sgl-project/sglang-torchtpu/issues).

[SGLang-JAX](https://github.com/sgl-project/sglang-jax) remains an important project for SGLang on TPU. It will continue to be maintained, supported and available for the community to use and adapt.

Google, Meta and RadixArk will share more at [SGLang Summit 2026](https://sglang.io/summit), November 12 to 13 at Fort Mason, San Francisco.
