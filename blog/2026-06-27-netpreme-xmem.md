---
title: "Accelerating SGLang HiCache with Netpreme X-Mem™ MPU"
author: "Netpreme Team"
date: "July 8, 2026"
previewImg: /images/blog/netpreme-xmem/figure1.png
---

> Netpreme X-Mem™ Memory Processing Unit (MPU) makes SGLang HiCache faster and more scalable by augmenting the slower Host DRAM offload tier with a purpose-built high-bandwidth KV memory tier.

![Figure 1](/images/blog/netpreme-xmem/figure1.png)

## TL;DR

- Prefix caching is becoming critical for long-context, agentic, and recommendation-style LLM workloads.
- SGLang HiCache provides the software foundation for scalable prefix reuse by extending RadixAttention into a hierarchical KV cache with HBM, Host DRAM, and external storage.
- Netpreme X-Mem™ integrates with SGLang HiCache as a dedicated TB/s KV memory tier.
- Netpreme X-Mem™ improves Time-to-First-Token (TTFT) by **6.7×** compared with Host DRAM-based SGLang HiCache on prefix-heavy workloads.

## Why Prefix Caching Needs a Fast Memory Tier

Prefix caching is designed to reduce expensive prefill compute for workloads where requests repeatedly share long, token-identical prompts that only differ in the short last part. SGLang has already built a strong software foundation for prefix reuse with **RadixAttention** and **HiCache**. RadixAttention enables efficient prefix caching in GPU memory, while HiCache extends it into a hierarchy beyond GPU HBM.

However, as GPUs become faster, the speed of loading KV cache into the HBM becomes important. To estimate the bandwidth requirement for KV-cache offloading, we model a long-context coding-agent workload with 64K-token prompts and prefill concurrency of 8. As the prefix hit rate increases, residual computation drops quickly, but the amount of KV data fetched from secondary memory grows. When hit rate exceeds 95%, scaling the bandwidth of KV-cache tiering can significantly reduce TTFT!

![Figure 2](/images/blog/netpreme-xmem/figure2.png)

But is 95% of prefix cache hit rate really practical? Let’s look at KV-cache reuse in coding agents — the largest market for AI applications as of today.

To quantify this, we [measure](https://github.com/netpreme/coding_agents) prefix-cache hit ratio using traces collected from [Claude Code agent](https://code.claude.com/docs/en/overview) while running multi-turn coding workflows on [SWE-bench](https://github.com/SWE-bench/SWE-bench). We observe that the cache-hit ratio consistently exceeds 95% in subsequent rounds for the same session, with the mean hit ratio being around 98%. In this regime, scaling KV-caching bandwidth beyond what is achievable with Host CPU offloading results in a 3x performance boost!

![Figure 3](/images/blog/netpreme-xmem/figure3.png)

This is where Netpreme X-Mem™ comes in!

## Netpreme X-Mem™: A Dedicated KV Tier for SGLang HiCache

Netpreme X-Mem™ MPU is a purpose-built solution to expand GPU memory by tens of TB over fast network fabrics. Conceptually, X-Mem™ is a dedicated memory node in the AI rack, peer-to-peer to all other GPUs. A single memory node comes with up to 24 TB of memory accessible by all GPUs in the same rack at 4 TB/s. There is no limit on how many memory nodes can be provisioned in the rack, and the aggregate bandwidth is multiplied if several MPU nodes are used. On the software side, X-Mem™'s address space is exposed as a part of the unified virtual memory, and semantically it is not any different from accessing a local HBM or remote GPU’s memory.

When integrating with SGLang HiCache, Netpreme X-Mem™ performs as a dedicated memory tier for KV cache offload. Instead of using Host DRAM or RDMA as the primary L2 KV cache tier, MPU provides a purpose-built, accelerator-oriented memory tier optimized for KV movement.

This makes X-Mem™ especially useful for workloads with:

- strict SLO requirements on TTFT;
- long shared prefixes / high-concurrency sessions;
- repeated prefill-heavy requests.

Netpreme X-Mem™ integrates SGLang via CUDA- and PyTorch-compatible APIs. These APIs allow any ML application to tap into the high-bandwidth memory tier transparently and with minimal modification of the application code.

## Benchmarking SGLang + Netpreme X-Mem™

We run two experiments, one is a micro-benchmark that measures TTFT of a single request and the other is an end-to-end LLM inference experiment with agentic workloads. Throughout benchmarking, we compare three configurations:

| Configuration | Description |
| --- | --- |
| GPU-only RadixAttention | Prefix caching is disabled. This requires re-computation  |
| SGLang HiCache + Host DRAM | Hierarchical cache using Host DRAM as the offload tier |
| SGLang HiCache + Netpreme X-Mem™ | Hierarchical cache using Netpreme MPU (350 GB/s configuration as bounded by H100 GPUs) as the offload tier |

### X-Mem™ flattens TTFT of a single request

![Figure 4](/images/blog/netpreme-xmem/figure4.png)

The result shows that the Netpreme’s MPU reduces single request TTFT by up to ~6.7x compared to Host DRAM making it almost flat even under very long context.

### X-Mem™ increases interactivity and system capacity of LLM serving engine

How does the above TTFT reduction translate into improvements in end-to-end LLM inference? To answer this question, we run the end-to-end LLM serving benchmark, NVIDIA AIPerf. We use workloads that represent agentic AI inference, consisting of 1K tokens of system prompt, 20K tokens of per-user context. Each user inputs 26 tokens in a single turn, and the average number of turns is 20. Solid points indicate Pareto-optimal points; shaded points indicate non-Pareto-optimal ones.

![Figure 5](/images/blog/netpreme-xmem/figure5.png)

The result demonstrates that Netpreme's X-Mem™ provides 33% higher TPS/user (interactivity) in a medium-load case and 50% higher interactivity and 30% higher TPS (system capacity) in a high-load case. This is because the GPU is utilized more efficiently, spending less time waiting for data to be copied on prefix cache hits.

## Looking Ahead

As coding agents, long-context assistants, and recommendation systems grow, serving systems will need to preserve and move much larger KV working sets. To this end, Netpreme will facilitate:

- Larger persistent KV stores with MPU and SSDs
- Multi-instance KV sharing
- Near-memory KV compression
- Near-memory computation

The long-term vision for Netpreme is to provide a full-stack memory solution for agentic workloads. This includes not only KV cache but also model weights, activations, embeddings, and other data structures. By providing a high-performance memory tier optimized for ML workloads, Netpreme enables new software architectures and optimizations that are not possible with traditional Host DRAM and GPU HBM.

## Conclusion

SGLang already provides a strong software foundation for prefix caching. RadixAttention and HiCache make it possible to reuse KV cache across long-context, multi-turn, and prefix-heavy workloads. But as prefix caching becomes more effective, the bottleneck shifts.

The challenge is no longer only finding reusable prefixes. It is moving cached KV back to GPU memory fast enough so that reuse delivers real latency and throughput improvement.

Host DRAM-based offload increases capacity, but its PCIe-level bandwidth can become the limiting factor for KV-intensive inference, especially when TTFT SLO is tight. Netpreme X-Mem™ addresses this bottleneck by providing SGLang HiCache with a high-bandwidth purpose-built KV memory tier.

Together, SGLang HiCache and Netpreme X-Mem™ make prefix caching faster and more scalable for the next generation of LLM workloads, including coding agents, long-context assistants, and recommendation systems.

## Acknowledgments

We thank the SGLang team for their constructive technical discussions and guidance throughout this integration.

## References

- SGLang Documentation: HiCache Design
- SGLang Blog: HiCache: High-Performance KV Cache Storage with Hierarchical Caching for LLM Serving
- Netpreme SGLang X-Mem™ Integration: https://github.com/netpreme/sglang_xmem
- LinkedIn Engineering Blog: Turbocharging LinkedIn’s Recommendation Systems with SGLang
- Qwen3-235B-A22B-Thinking-2507: https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507
- SWE-bench: https://github.com/SWE-bench/SWE-bench
- Netpreme Coding Agents Experiments: https://github.com/netpreme/coding_agents

---

## Appendix

### Evaluation setup

- GPU: a single H100-class GPU
- KV memory tier size: 64 GB (Host DRAM or Netpreme X-Mem™)
- CUDA Version: 12.8
- Attention backend: FlashAttention
- GPU prefix caching: disabled (to evaluate a local KV cache miss that hits in the X-Mem™)
- Model: [Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)
- Workloads:
  - single request TTFT benchmarking: used a benchmark [script](https://github.com/netpreme/sglang_xmem/blob/mtier-dev/kvcache_benchmark.py) that is a modified version of the vLLM KVConnector benchmarking script ([blog](https://vllm.ai/blog/2026-01-08-kv-offloading-connector), [code](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py)).
  - end-to-end throughput benchmarking: used [AIPerf](https://github.com/ai-dynamo/aiperf) from the NVIDIA Dynamo team. Specifically, we used the following workload configuration that represents agentic AI use cases.
    - Number of users: 15
    - QPS across all users: [3.0,3.0,3.5,4.0,4.5,5.5]
    - shared system prompt: 1000 tokens
    - per-user context: 20000 tokens
    - per-turn query: 26 tokens
    - per-turn output length: 100 tokens
