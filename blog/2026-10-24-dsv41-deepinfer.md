---
title: "More Than DeepInfer: What SGLang Delivers"
author: "The SGLang Team"
date: "October 24, 2026"
previewImg: /images/blog/dsv41_deepinfer/cover.jpg
ogImage: /images/blog/dsv41_deepinfer/og.jpg
excerpt: "DeepSeek open-sourced DeepInfer, the inference engine behind its own service, together with the kernels it runs on. It sets the bar for how fast DeepSeek-V4.1-Flash can run, so we measured SGLang against it with DeepInfer's own benchmarks on the same hardware."
---

**TL;DR**

- **On par or ahead on DeepInfer's own benchmarks.** Up to 1.04x prefill throughput on 4x GB300 and 4x B300, up to 7.4% shorter decode steps on 64x GB300, and within 1% at the remaining points.
- **DeepInfer's optimizations, in SGLang.** DeepSeek's kernels including the Mega series, DSpark speculative decoding, SWA bounded replay and both Engram placements run in SGLang main.
- **A full serving system.** SGLang is an API server, and it has the KV cache system, prefill/decode disaggregation and load balancing across replicas that DeepInfer's README lists as not included in its release.

DeepSeek open-sourced DeepInfer, the inference engine behind its own service, together with the kernels it runs on. It sets the bar for how fast DeepSeek-V4.1-Flash can run, so we measured SGLang against it with DeepInfer's own benchmarks on the same hardware.

## What is integrated

SGLang has run SWA bounded replay and host-memory Engram tables since its first day of V4.1 support. Since then it has added DeepSeek's Mega kernels, direct peer reads for Engram tables split across GPUs, and further execution optimizations.

**DeepSeek's kernels.** SGLang integrates FlashMLA, DeepGEMM, DeepEP, DeepSelect and the TileLang kernels, together with the Mega series: MegaAttention, MegaMoE, MegaGate and Mega-mHC. MegaMoE fuses token dispatch, expert GEMMs and the combine into one kernel. MegaAttention is the default attention path on B300 and GB300, on single servers and on both prefill and decode servers.

<img src="/images/blog/dsv41_deepinfer/kernel-integration.svg" alt="Kernel integration" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 1: Kernels integrated in SGLang.</p>

**Parallelism.** SGLang uses DP attention plus expert parallelism (DEP) for both prefill and decode. The benchmarks below use DEP4 for prefill and DEP64 for decode across 64 GB300 GPUs. SGLang also supports EPLB to balance expert load in wide-EP deployments, as described in our [MoE load-balancing blog](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb).

**DSpark speculative decoding.** SGLang runs the checkpoint's [DSpark](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) draft model to propose a block of tokens that the target model verifies together, reducing the number of target-model forward passes needed to generate a response.

**SWA bounded replay.** Decoder replay runs the late decoder layers on only the last 128 prompt tokens, while earlier source layers still process the full prompt. On a prefix-cache hit, encoder replay rebuilds missing sliding-window KV from the last 128 cached tokens.

<img src="/images/blog/dsv41_deepinfer/swa-bounded-replay.svg" alt="SWA bounded replay" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 2: SWA bounded replay: decoder replay and encoder replay.</p>

**Engram placement.** DeepInfer lets you keep the Engram tables in host memory or in device memory, and SGLang supports both. With `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, each node holds one copy of the tables in host memory, which leaves the most HBM for the KV cache. With `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS=1`, the tables are split across the GPUs, and each GPU reads the rows it needs straight from its peers over NVLink, with no all-gather or reduce-scatter around the lookup.

<img src="/images/blog/dsv41_deepinfer/engram-placement.svg" alt="Engram placement" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 3: Engram tables in host memory, or split across GPUs with direct peer reads.</p>

**Zero-overhead scheduling.** Since v0.4 (December 2024), SGLang's scheduler prepares step N+1 on the CPU while the GPU runs step N, overlapping scheduling with GPU execution. DeepInfer's documentation credits this design to SGLang.

<img src="/images/blog/dsv41_deepinfer/zero-overhead-scheduling.svg" alt="Zero-overhead scheduling" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 4: Zero-overhead scheduling: the CPU prepares the next step while the GPU runs the current one.</p>

**DeepSeek's chat format.** Prompts are rendered and outputs parsed with DeepSeek's own `deepseek-recipe` package ([#45315](https://github.com/sgl-project/sglang/pull/45315), [#45907](https://github.com/sgl-project/sglang/pull/45907), [#46219](https://github.com/sgl-project/sglang/pull/46219)), so reasoning and tool calls in SGLang's API follow DeepSeek's format. Grammar-constrained decoding supports JSON schema, regex and EBNF.

## Results

<img src="/images/blog/dsv41_deepinfer/results.svg" alt="Prefill on GB300 and B300, decode on 64x GB300" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 5: Prefill on 4x GB300 and 4x B300, decode on 64x GB300 (EP64).</p>

**Measured** (solid lines) shows our benchmark runs of SGLang and DeepInfer. **Published** (dashed lines) shows DeepInfer's originally published [prefill](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/prefill_pareto.png) and [decode](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/images/decode_pareto.png) results, with values read from those figures. The prefill comparison in the text uses the measured DeepInfer curve, taking its fastest tested configuration at each point.

Setup in the [appendix](#appendix-benchmark-setup). The short version:

- **Prefill, 4x GB300.** SGLang reaches up to 1.04x the prefill throughput: ahead at 4K, 32K and 64K new tokens per GPU per step, on par at 8K and 16K.
- **Prefill, 4x B300.** Up to 1.04x the prefill throughput: ahead at 4K, 8K and 32K, on par at 16K and 64K.
- **Decode, 64x GB300 (EP64).** SGLang's step is 1.1% to 7.4% shorter from 16 to 256 requests per GPU, and on par at 8.

### Where the speed comes from

Mega-mHC feeds MegaMoE directly, attention produces MXFP8 output for the output projection, and the prefill compressor and request-window KV copies use fused kernels. These changes reduce intermediate memory traffic and the number of kernel launches.

Breakable CUDA graphs cover prefill steps of up to 8K tokens per GPU, including long contexts and data-parallel attention. Decoder replay uses separate graphs sized to the bounded suffix, so those layers still replay from CUDA graphs when the rest of a large prefill step runs eagerly.

## What SGLang adds

DeepInfer's README lists three components a complete inference service needs that its release does not include. SGLang ships all three.

<img src="/images/blog/dsv41_deepinfer/serving-stack.svg" alt="SGLang serving stack" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 6: Linker connects local L1 GPU caches directly to shared L3 storage.</p>

**A KV cache system.** SGLang's unified radix cache reuses KV across requests and conversation turns, including for hybrid models like this one. The configuration shown above uses Linker to connect L1 GPU caches directly to shared L3 storage, such as 3FS or Mooncake. HiCache also supports a host-memory L2 tier for deployments that use it.

**Prefill/decode disaggregation.** SGLang runs prefill and decode on separate servers, with the KV transfer backend of your choice: Mooncake, NIXL or MORI.

**Load balancing across replicas.** SGLang's model gateway distributes requests across replicas and supports routing based on cached prefixes.

On top of these, SGLang is an API server, with reasoning and tool-call parsing, on NVIDIA GB300, B300, B200 and H200 and AMD MI355X.

## Acknowledgments

Thanks to DeepSeek for open-sourcing DeepInfer, FlashMLA, DeepGEMM and DeepEP, and for documenting their inference system in such detail.

## Appendix: benchmark setup

Both engines run DeepInfer's own benchmarks on the same hardware. SGLang is at [`7d996300b9`](https://github.com/sgl-project/sglang/commit/7d996300b98e3f74a7e79adefbac4d95de0da508) for prefill and [`3b95120f00`](https://github.com/sgl-project/sglang/commit/3b95120f000cbd787a2832ca22dde4b4801f7260) for decode, DeepInfer at [`774ce24`](https://github.com/deepseek-ai/DeepInfer/commit/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01). The launch commands and benchmark clients for both are in [this gist](https://gist.github.com/sglang-bot/f03d7d0dd268b0386177eb2ef09f8b3b). To serve the model for real, follow the [cookbook page](https://docs.sglang.ai/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1); the settings below, such as balanced expert routing, exist for the comparison only.

### Prefill

- **Workload and machines.** We use DeepInfer's prefill benchmark: every request has a 128K-token context with 4K new tokens, and each GPU prefills B new tokens per step, from 4K to 64K. Both engines run on the same 4-GPU node with the same weights, DP4/EP4, balanced expert routing, EPLB off, and 32 redundant experts. For DeepInfer we take, at each point, the fastest of its two Engram placements (host or GPU memory), each with its defaults and with its 64K settings. SGLang splits the Engram tables across GPUs, spreads the redundant experts evenly, and enables a few optimizations that are not on by default yet.
- **Differences.** DeepInfer measures how long prefill takes for about 9 batches on GPU 0. Our SGLang results average every complete batch on every GPU between the points where 20% and 90% of the requests have finished. DeepInfer repeats the same prompt across requests on each GPU, while SGLang uses different new tokens per request.

### Decode

- **Workload and machines.** We match DeepInfer's decode benchmark workload on 64 GB300 GPUs (16 nodes, DP64/EP64): 128K-token contexts with a shared cached prefix, a fixed number of requests per GPU, 3 draft tokens per step with static verification (the full drafted block is verified for every request), balanced expert routing, and 384 routed plus 64 redundant experts. The chart converts step time to tokens at 3.25 accepted per step.
- **Differences.** SGLang keeps the Engram tables on device and DeepInfer in host memory, and DeepInfer's draft model carries 64 redundant experts while SGLang's draft has none.
