---
title: "SGLang-Neuron v0.1: SGLang's Serving Stack on AWS Trainium"
author: "SGLang Team and AWS Annapurna Labs"
date: "November 12, 2026"
previewImg: /images/blog/sglang-neuron-v0.1/architecture.png
type: blog
---

## TL;DR

* [SGLang-Neuron](https://github.com/sgl-project/sglang-neuron) v0.1 is now open source and available in public preview. It is a PyTorch-native plugin, co-developed by the SGLang team at RadixArk and Amazon Annapurna Labs, that serves LLMs and diffusion models on AWS Trainium. During the preview, SGLang-Neuron is supported through the AWS Deep Learning Container for SGLang-Neuron.
* Every supported LLM shares one fully asynchronous runtime. Trainium chips run decode steps back to back, with no idle time between them.
* Megakernels run every decoder layer of a step in a single NKI kernel call. GPT-OSS-120B, GLM-5.2 and Kimi-K3 serve on this path.
* Kimi-K3 runs prefill-decode disaggregated across three Trainium3 nodes, with KV moving over NIXL on EFA.
* GLM-5.2 (MXFP4) serves a 1M-token context for agentic coding workloads.
* HiCache keeps evicted prefixes in host memory, so a later hit loads them back instead of recomputing them.
* SGLang Diffusion now runs on Trainium2, with FLUX.1-dev, FLUX.3-Action, MiniMax-H3, Wan2.2 A14B and LingBot-World-V2.

## Introduction

Today we are releasing SGLang-Neuron v0.1, the first public release of SGLang's backend for [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/). Trainium is a portfolio of AWS AI chips that delivers the best economics for AI training and inference at scale. It follows the [closed beta](https://lmsys.org/blog/2026-10-22-sglang-trainium-beta/) the SGLang team announced in October. The beta showed that SGLang's model components carry over to Trainium through [TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/). This release is about what it takes to make them fast.

Neuron compiles models into programs that run on Trainium's [NeuronCores](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html), work reaches those cores through launch queues, and the fastest code paths are hand-written NKI kernels. Most of the engineering in v0.1 went into three places where that matters: how steps are launched, how much of a step lives in one kernel, and where the KV cache sits when it is not on the device.

This blog covers what ships in v0.1 and where it goes next. We start with how the backend is built: the plugin architecture, the asynchronous runtime, megakernels, prefill-decode disaggregation over EFA, HiCache, and SGLang Diffusion on Neuron. We then cover the supported features and models, RL rollouts with Miles, how to get started, and the roadmap.

## How it works

SGLang-Neuron is an out-of-tree plugin. Installing it registers a neuron platform through SGLang's plugin entry point, and SGLANG_PLATFORM=neuron selects it at launch.

![SGLang core (API server, radix cache, disaggregated serving protocol) on top of the SGLang-Neuron plugin (Neuron model implementations, NKI kernels, device parallelism, graph runner, speculative decoding, prefix and hierarchical caching), on TorchNeuron, the Neuron compiler and runtime, and AWS Trainium.](/images/blog/sglang-neuron-v0.1/architecture.png)

*Figure 1: How the SGLang-Neuron plugin fits into SGLang.*

The split of responsibilities is strict. SGLang core owns the API server, the scheduler, the radix cache and the disaggregated serving protocol. The plugin supplies what SGLang asks each hardware platform for:

* **Neuron model implementations**, registered in a model catalog. The catalog checks the checkpoint, the parallel layout and the flags before a server starts, so every model shares one launch path.
* **NKI kernels** for attention, MoE, projections and whole-step megakernels.
* **Device parallelism**: tensor, expert and data parallelism, plus pipeline and decode context parallelism for the largest models.
* **A graph runner** with a compilation manager and a bucketer. Trainium runs compiled programs (NEFFs), so the runner compiles a fixed set of shape buckets and dispatches each step to the matching one.
* **KV cache pools**: a paged full KV pool for MHA models, and separate latent pools for MLA and DSA attention.
* **Speculative decoding** and a sampler that runs on the device.

Underneath sits TorchNeuron, AWS's native PyTorch backend for Trainium. It registers Trainium as a PyTorch device for eager execution and as a TorchDynamo backend, so torch.compile lowers a graph through the Neuron compiler. NKI kernels plug into the same compiled graph. Each SGLang scheduler rank binds to its NeuronCores and builds its model and KV pool there.

## One asynchronous runtime

Every supported LLM now shares one fully asynchronous execution path. The goal is simple to state: the device never idles between two steps.

The naive path breaks that goal. On a single stream, a host copy between two graphs waits for the first graph and blocks the second. The NeuronCores sit idle while the host moves a few tensors. The runtime's device_io layer fixes this with two streams and two roles:

* **The default stream runs graphs. A side stream runs every host copy.** The two never queue behind each other.
* **Uploads overlap the previous step.** A step's inputs are queued on the side stream before its graph launches. The graph submit waits on a single event, so the upload runs while the previous graph is still on the device.
* **Readbacks never block.** A readback is queued only after its graph's launch event, so it cannot sit behind a running graph. Sampled tokens are small enough to read directly.
* **Two launches stay in flight.** SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES bounds each thread, with a default of 2. That is what lets the scheduler launch step n+1 before step n finishes.

The effect is the Trainium version of an idea SGLang users already know from GPUs: hide host overhead behind device work. Here the host work is hidden behind launch queues instead of CUDA streams, and it applies to every model on the plugin, not to one tuned path.

## Megakernels: one kernel per step

By default, the compiler lowers each decode step into a graph of many small kernels. It chooses where each kernel starts and ends, and intermediate results go back to HBM between them.

A megakernel takes a different approach, it runs every decoder layer of a step in a single NKI kernel call. The weights are packed via the plugin on a leading layer axis, and the KV pools are written in place. A step stops being a graph of many small kernels with compiler-chosen boundaries between them. It becomes one kernel whose memory movement we control end to end.

The platform still executes it the same way. Each phase is a torch.compile graph on the Neuron backend that calls the kernel once, and the decode family submits straight to the bound executor.

SGLang stays in charge of everything above the step. Scheduling, the KV cache, PD disaggregation and result processing are unchanged. A megakernel plans and runs whole steps under a contract called MegakernelSpec. One contract means new models do not need a new runtime subclass.

GPT-OSS-120B was the first model on this path. Its decode runs through a single kernel on Trainium3, with pipeline parallelism and decode context parallelism layered on top. GLM-5.2 uses megakernels for both prefill and decode, and Kimi-K3 has dedicated NKI kernels for its prefill. These kernels ship with the SGLang-Neuron plugin.

## Prefill-decode disaggregation over EFA

On Trainium, prefill and decode run best as different servers. A prefill server computes the prompt's KV cache and its first token. It then writes the KV straight into a decode server's pool with NIXL WRITEs over EFA. A deployment runs any number of each (xPyD) behind a balancer.

The split suits Trainium for three reasons:

* **One phase per server.** Each server compiles and warms only its own phase's graphs. An aggregated server pays a program-switch fence every time it moves between prefill and decode. A disaggregated server never does.
* **Steady TPOT.** No prefill forward runs between two decode steps, so time per output token stays at its median.
* **A layout per phase.** Prefill and decode want different topologies. Each role takes its own TP, EP and DP layout, and the prefill and decode servers scale up independently.

Each role also runs its own serving profile:

|  | Prefill server | Decode server |
| :---- | :---- | :---- |
| Batching | One request per forward, at the sizes its graphs were compiled for | Steps run back to back; sampled tokens stay on the device |
| Launching | Every forward launches before the previous one finishes; 32 queued executions per core cover long prefill forwards | Transfer workers and NIXL progress threads land incoming KV beside the running graphs |
| Handoff | A pipelined prefill computes ahead of the KV transfer | A request joins the running batch once its KV has landed |

A model can also expose named decode points, such as 128K and 1M context. Both roles launch with the same point, so the compiled shapes on each side agree.

### Kimi-K3 on three Trainium3 nodes

Kimi-K3 is the largest deployment in this release. It runs on three Trainium3 nodes, one for prefill and two for decode, with KV moving over NIXL on EFA.

| Role | Nodes | Layout | What runs there |
| :---- | :---- | :---- | :---- |
| Prefill | 1 | PP8 x TP16 x DCP16 | Vendored NKI prefill kernels, chunk-keyed prefix cache, HiCache |
| Decode | 2 | DP8 x TP32 x DCP32 | DSPark speculative decoding on the model's own speculative worker |

![Kimi-K3 PD topology on three Trainium3 nodes: one prefill node (PP8 x TP16 x DCP16) writes KV over NIXL on EFA into a two-node decode server (DP8 x TP32 x DCP32)](/images/blog/sglang-neuron-v0.1/kimi-k3-pd-topology.png)

*Figure 2: Kimi-K3 PD topology on three Trainium3 nodes*

Kimi-K3 uses MLA attention in FP8 with MXFP4 experts, so its KV lives in the MLA latent pools rather than the paged MHA pool. The prefill side is pipelined across eight stages. The decode side spans two nodes with decode context parallelism at 32.

This configuration reaches 67 tok/s per user on mixed 4K to 1M prompts, at full slots, with real speculative acceptance.

### GLM-5.2 at 1M context

GLM-5.2 (MXFP4) is an MoE model with DSA attention. On Trainium3 it serves disaggregated, with megakernel prefill and decode, MTP speculative decoding, and 128K and 1M decode points. It runs on the public Neuron SDK 2.33.0.

The 1M decode point targets long-context agentic workloads such as coding agents, where a session's context keeps growing across turns.

The pipelined prefill matters here too. The prefill server computes ahead of the KV transfer, so a request's first token does not wait on the handoff to the decode server.

## HiCache: a DRAM tier under the KV cache

Agentic workloads resend long, growing prefixes. When the device pool fills, the radix cache has to evict, and an evicted prefix normally means a full recompute on its next hit. HiCache gives those pages somewhere to go.

With the hierarchical cache enabled, SGLang keeps evicted radix-cache pages in host memory and loads them back on a later hit. The radix tree, the host allocator and the transfer engine are upstream SGLang code. The plugin supplies only the host pool, built for how Trainium moves memory:

* **Whole pages, layer-first.** The pool is a pinned host buffer of whole pages, laid out as `[2 x layers][pages][page bytes]` for MHA models and as one region per layer for the MLA and DSA pools. This layout matches the Neuron compiler's access patterns.
* **Batched DMA beside the graphs.** Pages move with nrt_tensor_read_batch and nrt_tensor_write_batch: one call per transfer, on a dedicated thread per direction. A store waits only for the graphs launched before it.
* **Decode keeps running.** On an aggregated server, the scheduler holds the batch that is waiting for a load while the running batch keeps decoding.

On GPT-OSS-120B at TP8 and 32K context, HiCache L2 delivers 72% to 75% more output throughput.

v0.1 also adds a remote DRAM tier below L2, reached over NIXL LIBFABRIC. When sessions outgrow the local host pool, pages spill to another node's memory instead of being dropped. The host budget is shared across the ranks and --dp-size replicas of a node, so replicas cannot overbook the same DRAM.

## SGLang Diffusion on Trainium

v0.1 also brings image, video and world models to Trainium. SGLang Diffusion gained a Neuron platform, so diffusion pipelines select Trainium the same way LLM serving does. The platform came in with FLUX.1-dev as its first model, and four more models followed on the same base.

| Model | What landed |
| :---- | :---- |
| FLUX.1-dev | Text to image. The diffusion platform, NKI kernels and tensor-parallel layers |
| MiniMax-H3 | t2va and fl2va model and pipeline |
| Wan2.2 A14B | T2V and I2V model and pipeline |
| FLUX.3-Action | Runs on SGLang Diffusion's own pipeline |
| LingBot-World-V2 | Real-time serving. DiT with paged cache attention, LNC 1 launches, a streaming Wan VAE |

FLUX.1-dev carried the platform work: the NKI kernels and tensor-parallel layers it introduced are the base the later models build on. LingBot-World-V2 is the first model on the plugin to serve in real time, with a DiT that attends over a paged cache and a streaming VAE.

## Feature and model coverage

These are the SGLang serving features available on Trainium2 and Trainium3 in v0.1 unless noted:

| Area | Supported features |
| :---- | :---- |
| Runtime | One fully asynchronous execution path for every LLM; sampler on the device; shape-bucketed graph runner |
| Parallelism | Data, tensor and expert parallelism; pipeline parallelism; decode context parallelism |
| Kernels | NKI attention, MoE and projection kernels; node-wide all_to_all for MoE; whole-step megakernels |
| Prefix caching | Radix cache; chunk-keyed prefix cache; HiCache L2 in host memory; remote DRAM tier over NIXL |
| Disaggregated serving | Prefill and decode on separate nodes (xPyD), KV over NIXL on EFA, named decode points |
| Speculative decoding | MTP and DSPark |
| Precision | BF16, FP8 and MXFP4 checkpoints; FP8 KV cache |
| Diffusion | SGLang Diffusion with a Neuron platform for image, video and world models (Trainium2, trn2.48xlarge)  |

And these are the LLMs:

| Model | How it is served |
| :---- | :---- |
| Kimi-K3.1 | DSA latent pools plus KDA state; each PD role of a pipeline stage runs its stage megakernel |
| Kimi-K3 | MLA in FP8, MXFP4 experts; pipelined prefill, two-node decode, speculative decoding |
| GLM-5.3, GLM-5.3-Flash | Flash: KDA state with MLA and DSA pools |
| GLM-5.2 (MXFP4) | MoE with DSA; megakernel prefill and decode; 128K and 1M decode points; MTP |
| Qwen3.5 | MoE with gated delta net state |
| Qwen3-0.6B, Qwen3-8B, Qwen3-30B-A3B | Dense and MoE; aggregated and PD serving |
| GPT-OSS-120B | MoE, MXFP4; megakernel decode; pipeline and decode context parallelism; DP-attention decode |
| GLM-4-9B | Dense; aggregated and PD serving |
| DeepSeek-V2-Lite | MLA attention; PD serving |

And diffusion models, validated on trn2.48xlarge:

| Model | How it is served |
| :---- | :---- |
| FLUX.1-dev | Image generation on the Neuron diffusion platform |
| FLUX.3-Action | Robot action policy model |
| MiniMax-H3, Wan2.2 A14B | Video generation |
| LingBot-World-V2 | Real-time world model |

## RL rollouts with Miles

Serving is half of an RL post-training loop. [Miles](https://github.com/radixark/miles), RadixArk's open-source RL framework, uses SGLang to generate rollouts, trains through a pluggable backend, and sends the updated weights back to the engine.

On Trainium, Miles trains through its TorchTitan backend on TorchNeuron, the same PyTorch stack SGLang-Neuron serves on. We validated the loop end to end on Qwen3-30B-A3B, with rollouts served by SGLang-Neuron and training on the TorchTitan backend.

## Get started

On a Trainium2 or Trainium3 instance, start the container, launch a model and send a request:

```bash
# Start the container

docker run --rm -it --privileged \
  -v /path/to/work:/work \
  public.ecr.aws/neuron/TODO-repo:TODO-tag /bin/bash

# Inside the container, launch GPT-OSS-120B

SGLANG_PLATFORM=neuron python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tp-size 8 --ep-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 10240 \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --host 127.0.0.1 --port 30000

# From another terminal, send a request

curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}]}'
```

The [repository](https://github.com/sgl-project/sglang-neuron) has the full setup guide, per-model recipes and validated configurations. The [v0.1 release notes](https://docs.google.com/document/d/1Fnz6NZUwzySiCsGmCAcNFlRvDLrRh0-8lKDrlvbc6cQ) have the full change list.

## What's next

* **A recurrent-state framework.** Qwen3.5 and GLM-5.3-Flash follow the same MegakernelSpec path as Kimi-K3.1. A standard path for models with recurrent state makes the next model faster to onboard.
* **One kernel-pack contract.** Every kernel pack will register with a stamp and a declared set of knobs, and a launch will refuse anything unpinned.
* **Kimi-K3.1 speculative decoding.** The pieces are the platform runtime, a pipeline-parallel loop with async execution, a sampler on the device and a strong drafter.
* **One host pattern.** With Kimi-K3 and GLM-5.2 on MegakernelSpec, five runtime subclasses collapse into the platform runtime plus declared phase programs. Kernel work then becomes a self-contained task, for people and for coding agents.

## Community

For questions, join the [SGLang Slack](https://slack.sglang.io). To request a model or propose a feature, open an issue on the [repository](https://github.com/sgl-project/sglang-neuron).

## Acknowledgments

SGLang-Neuron is built and maintained by people across RadixArk, Amazon Annapurna Labs and Yotta Labs.

**RadixArk:** Alex Nails, Yihao Wang, Lingyan Hao

**Amazon:** TODO full list

And we thank **Yotta Labs** for their contributions to SGLang on Trainium, including the earlier Mini-SGLang-Neuron work.
