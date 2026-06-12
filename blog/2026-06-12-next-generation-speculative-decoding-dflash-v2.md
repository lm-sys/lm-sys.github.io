---
title: "The next generation of speculative decoding: DFlash and Spec V2"
author: "Z Lab, Modal, and SGLang Teams"
date: "June 15, 2026"
previewImg: /images/blog/dflash-v2/dflash-arch-diagram.webp
type: blog
---

Using Z Lab's DFlash speculative decoding models with SGLang’s newly default Spec V2 engine, you can achieve state-of-the-art latencies for LLM inference serving.

<div style="text-align: center;">
  <img src="/images/blog/dflash-v2/dflash-headline-perf.webp" style="width: 60%;"></img>
  <small>Workload: Qwen 3.5 397B-A17B (BF16), HumanEval. Settings: greedy decoding, thinking enabled, max new tokens 4096. Hardware: 8xB200.</small>
</div>

Below, we describe DFlash’s novel diffusion \+ KV injection strategy for speculative decoding, why that matters for achieving massive speedups, and how the teams at [Z Lab](https://z-lab.ai), SGLang, and [Modal](https://modal.com) worked together to make those speedups available to everyone.

And we mean everyone! You can [run tensor-parallel Qwen 3.6 35B-A3B with DFlash right now](https://modal.com/docs/examples/sglang_low_latency) on Modal's serverless GPUs, achieving decode speeds of up to 1k tps:

```shell
git clone https://github.com/modal-labs/modal-examples
cd modal-examples
uvx modal setup && uvx modal run 06_gpu_and_ml/llm-serving/sglang_low_latency.py
```

## DFlash: Parallel drafting with KV injection

Transformer-based large language models (LLMs) are powerful, but their autoregressive decoding process makes inference slow: tokens must be generated one by one, with low [arithmetic intensity](https://modal.com/gpu-glossary/perf/arithmetic-intensity) that makes them a poor fit for modern hardware.

[Speculative decoding](https://arxiv.org/abs/2211.17192) addresses this bottleneck by using a smaller, faster draft model to propose multiple tokens, which are then verified in parallel by the target LLM, with no impact on model quality.

However, many speculative decoding methods, like the [EAGLE series](https://arxiv.org/abs/2503.01840) and the native multi-token prediction (MTP) modules in recent models like [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) and [DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), still rely on sequential autoregression – but in the draft model instead of the target. The draft model generates draft tokens one-by-one, which makes them a poor fit for modern hardware and limits the achievable speedup.

That’s why the Z Lab developed [DFlash](https://arxiv.org/abs/2602.06036), which uses a lightweight block diffusion draft model to generate an entire block of draft tokens in parallel, just the way that GPUs and TPUs like. Xiaomi's new Mimo v2.5-Pro-UltraSpeed uses DFlash to achieve [over 1k output tps](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps).

Using block diffusion for speculative drafting is non-trivial. Directly training a small block diffusion model as the drafter leads to low acceptance length, while using an existing large diffusion LLM like [SpecDiff-2](https://arxiv.org/abs/2511.00606) as the drafter introduces a large memory footprint and high drafting cost.

The key insight of DFlash is simple: the target LLM knows the context best. Inspired by previous methods like [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/html/2503.01840v1) and MTP ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737); [Samragh et al., 2025](https://arxiv.org/abs/2507.11851)), we extract hidden representations of the context tokens from the target model. But different from previous work, we inject them directly into the draft model’s KV cache. This allows the draft model to skip modeling the full context from scratch and focus purely on predicting the next block of tokens – using the same tensors as the later layers of the target model\!

![](/images/blog/dflash-v2/dflash-arch-diagram.webp)

With this design, DFlash leverages the rich, highly relevant contextual features produced by the target LLM while keeping the draft model extremely small and efficient. As a result, DFlash achieves high acceptance length with low drafting latency.

### Why is DFlash so fast?

Speculative decoding speedup mainly depends on two factors: how many drafted tokens are accepted per cycle and how much extra cost the draft model adds. DFlash improves both, using two distinct techniques.

Concretely, DFlash achieves a similar acceptance length to a 5-layer EAGLE-3 drafter, but thanks to its ultra-fast parallel drafting, it delivers much higher end-to-end speedup. Results are reported as `acc_len / speedup`.

| Task      | EAGLE-3 (5 layers) | DFlash         |
| :-------- | :----------------- | :------------- |
| GSM8K     | 4.2 / 2.1x         | **4.2 / 3.3x** |
| HumanEval | 4.3 / 2.2x         | **4.0 / 3.2x** |
| MT-Bench  | 3.1 / 1.4x         | **3.0 / 2.2x** |

**DFlash drafts faster**

Autoregressive drafters like EAGLE-3 generate draft tokens one by one. As the draft length grows, the drafting cost grows roughly linearly. To keep latency low, these methods usually rely on very shallow draft models, which limits draft quality.

DFlash avoids this bottleneck with a block diffusion drafter. It generates the whole draft block in parallel with a single forward pass, making drafting much more hardware-friendly. A 5-layer DFlash drafter generating 16 tokens has lower drafting latency than a shallower EAGLE-3 drafter.

<img src="/images/blog/dflash-v2/dflash-vs-eagle-draft-latency.webp" style="display:block; margin-left: auto; margin-right: auto; width: 60%"></img>

We can observe the independent impact of this technique in end-to-end benchmarks, where DFlash still provides a higher end-to-end speedup than EAGLE-3, even at lower acceptance lengths.

| Task      | EAGLE-3 (5 layers) | DFlash (diffusion only) |
| :-------- | :----------------- | :------------------------- |
| GSM8K     | 4.2 / 2.1x         | **3.5 / 2.9x**             |
| HumanEval | 4.3 / 2.2x         | **3.5 / 2.9x**             |
| MT-Bench  | 3.1 / 1.4x         | **2.6 / 2.0x**             |

**KV injection increases acceptance lengths**

Fast drafting only helps if the drafted tokens are accepted. Existing methods like EAGLE-3 use target model features only at the input of the draft model, so this information can fade as the draft model gets deeper.

DFlash instead injects target features into the KV cache of every draft layer. This keeps the drafter strongly conditioned on the target model’s context throughout generation, allowing deeper drafters to produce higher-quality drafts.

We can also observe the independent impact of this technique in end-to-end-benchmarks, where DFlash in autoregressive mode still runs faster due to higher acceptance lengths.

| Task      | EAGLE-3 (5 layers) | DFlash (injection only) |
| :-------- | :----------------- | :------------------------- |
| GSM8K     | 4.2 / 2.1x         | **4.8 / 2.4x**             |
| HumanEval | 4.3 / 2.2x         | **4.6 / 2.3x**             |
| MT-Bench  | 3.1 / 1.4x         | **3.4 / 1.5x**             |

## Implementing DFlash in SGLang

The benchmark numbers in this section are from the initial implementation of DFlash as part of R&D by Z Lab. Based on these impressive results, the teams at Modal and SGLang collaborated with Z Lab to optimize end-to-end performance in the SGLang inference engine.

Bringing a performance optimization technique like DFlash from research to prod requires two basic components: implementing the technique inside a high-performance engine and then optimizing the performance of the end-to-end system, from host scheduler to GPU execution.

The DFlash integration into SGLang can be split into two parts along these lines. First, DFlash was added to the original ([now deprecated](https://github.com/sgl-project/sglang/pull/25464)) V1 speculative decoding engine. Besides implementing a new draft model architecture, this also required integration of KV caches across draft and target to support injection. Second, DFlash was added to the new V2 speculative decoding engine, which offers improved performance through [reduced synchronization with the host](https://modal.com/blog/host-overhead-inference-efficiency).

In the [initial implementation of DFlash](https://github.com/sgl-project/sglang/pull/22077), we added support for this new model architecture to the existing speculative decoding engine. This included the addition of a `DFlashWorker` to control the draft model execution and the actual `DFlashDraftModel` that it drives.

As a reminder, SGLang uses a scheduler process (mostly on the host) to drive execution of model worker processes (mostly on the accelerators). One counterintuitive aspect of the way speculative decoding works in SGLang is that the draft model worker is the one that talks to the scheduler (via methods like `.forward_batch_generation`). It wraps a target model’s worker for the verification passes and calls it when the drafts are ready. Keep this in mind if you look at the code or a trace\!

That’s not new in DFlash. The main novelty is the KV injection, which ties state between the draft and target models. For methods like EAGLE, the draft KV cache is fully private to the draft model, calculated based on KV projection of the draft’s own latents. In DFlash, the latents of the target model are instead passed through a KV projection by the draft model.

We don’t want to store those latents and cut into precious KV cache space and we want all requests that have the same prefix to share the radix cache. So we run the draft KV projection ahead of the rest of the draft forward pass – *immediate materialization*. That needs to be fast, so we added a layer-batched linear projection and a fused Triton kernel for the norm+RoPE post-processing.

## Eliminating host overhead for DFlash with Spec V2 and overlap scheduling

That worked and was fast, but we knew it could be faster. We were concurrently working on the V2 speculative decoding engine, so the next step was to [combine DFlash with the V2 engine](https://github.com/sgl-project/sglang/pull/23000), which is what’s now available in SGLang.

The key goal of the V2 engine as a whole is to reduce points of host-device synchronization, which [kill inference performance](https://modal.com/blog/host-overhead-inference-efficiency), no matter how fast the GPU is or how good the kernels are. The solution is called the *overlap scheduler*.

In particular, there’s two key opportunities for overlap:

1. host-side `pop_and_process` cleanup after the GPU finishes batch N-1 (e.g. stop token detection, request metadata updates) can overlap with GPU work on batch N;
2. host KV allocation (in `prepare_for_decode`) for batch N can overlap with GPU work on batch N-1.

Under V2 with these optimizations, performance improved by over 25%, from \~9,700 tok/s to \~12,300 tok/s, when running Qwen 3-8B on a single B200 at concurrency 32 ([details here](https://github.com/sgl-project/sglang/pull/20547)).

The aforementioned optimizations can be used by all draft models. But DFlash is able to take greater advantage of overlap scheduling. In particular, because DFlash uses immediate materialization from the target to construct the draft KV, it doesn’t need a separate draft-extend step to run the draft model on only accepted tokens and populate KV. This draft-extend step, used in EAGLE, requires that accepted tokens are known before host-side planning can proceed.

## High-performance DFlash speculative models are available for a variety of models

TODO: Write this section up once we have final model list and numbers.

![](/images/blog/dflash-v2/dflash-perf-big-sweep.webp)

## Try DFlash in SGLang now

Unlike posts from proprietary inference providers, you don’t have to just read this blog and feel FOMO. You can [read the code](https://github.com/sgl-project/sglang/pull/23000). You can deploy a DFlash-accelerated SGLang server [right now](https://modal.com/docs/examples/sglang_low_latency), and then start tinkering.

More broadly: you can run inference at state-of-the-art intelligences and speeds thanks to the work of the open-weights model builders, systems researchers, and the open source community. Whether it’s research work on techniques like DFlash by the [Z Lab](https://z-lab.ai/) or features and performance enhancements from open source contributors like [Modal](https://modal.com/), the world’s best work on LLM inference is landing in the SGLang open source engine for you to build on and with.

## Acknowledgements

Thanks to everyone who contributed to bringing Spec V2 and DFlash to SGLang.

Z Lab: Jian Chen, Yesheng Liang, and Zhijian Liu.

Modal: David Wang and Charles Frye.

SGLang: Qiaolin Yu, Liangsheng Yin, and Khoa Pham.
