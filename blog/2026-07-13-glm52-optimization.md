---
title: "Serving GLM5.2 NVFP4 Agentic Workload with SGLang: Reaching 500 TPS in 2 Weeks"
author: "SGLang Team"
date: "July 13, 2026"
previewImg: /images/blog/glm52-optimization/glm52-nvfp4-day0-vs-v0515-tps.png
type: blog
---

## TL;DR

- >500 TPS on 8xB300 (bs=1)

- Sync free speculative decoding for GLM 5.2 MTP

- Built-in IndexShare MTP with Spec V2

- 2.33x faster TopK-V2 for ISL 80k

- Indexer prologue fusion

- Gemm kernels improvement

![GLM-5.2 NVFP4 day-0 versus v0.5.15.post1 interactivity on B300](/images/blog/glm52-optimization/glm52-nvfp4-day0-vs-v0515-tps.png)

*Figure 0. Performance comparison between Day-0 and v0.5.15.post1 on 8\*B300*

## Background

[GLM-5.2](http://zai-org/GLM-5.2-FP8) keeps the same backbone as earlier GLM checkpoints: DSA with a sparse-attention indexer on top of a DeepSeek-V3-style MoE. It adds two major architectural changes: IndexShare for DSA, and MTP with IndexShare and KVShare.

SGLang has supported the GLM-5.2-NVFP4 checkpoint since day 0 on (Grace) Blackwell hardware, using trtllm-gen kernels for both sparse attention and MoE. To turn that day-0 stack into a faster, more stable, and more production-ready serving path, we're introducing the following optimizations.

## Optimization

### Runtime Optimization

#### Zero-overhead scheduling and Spec v2

Spec V2 is SGLang's [overlap](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) runtime for speculative decoding. While the GPU runs the current model forward on the forward stream, it does the next step's KV allocation and metadata preparation on the plan stream, hiding CPU overhead inside the forward.

We recently turned on Spec V2 by default. On paper, the overlap scheduler should let the CPU handle its bookkeeping for the next step while the GPU is still busy with the current one, leaving almost no bubbles between iterations. In practice, a few optimizations were needed to fully realize the benefits of the overlap scheduler and Spec V2: we made the DSA draft-extend path CUDA-graphable, made `seq_lens_cpu` optional for DSA to drop the D2H sync, removed the remaining H2D syncs, and fused the small eager metadata ops in `_apply_cuda_graph_metadata`. With those GPU bubbles gone, we saw an 11% end-to-end TPS speedup.

![Decode trace before Spec V2 optimizations](/images/blog/glm52-optimization/spec-v2-before.png)

![Decode trace after Spec V2 optimizations](/images/blog/glm52-optimization/spec-v2-after.png)

*Figure 1. Decode at batch size 1, before Spec v2 optimizations (top) vs. after (bottom). With it on, there is no bubble between **run_batch** iterations.*

#### IndexShare MTP in SGLang

GLM-5.2 ships a strong MTP head, with accept lengths frequently hitting 5+, which gives significant speedups on low-latency workloads like agentic coding. To implement GLM-5.2's MTP behavior correctly, we made a few changes to SGLang's speculative decoding runtime.

First, IndexShare requires SGLang to reuse the DSA indexer's top-k across draft steps: the top-k computed at draft step 0 is held and passed to later steps, so they skip recomputing the indexer. This cut draft-step cost by up to ~1.9x at long context, with no hit to output quality.

Second, the top-k needs to be seeded from the right place, which in SGLang is the draft-extend of the previous run_batch iteration. Because Spec V2 runs its steps asynchronously, we had to thread that seed through the overlap scheduler's relay buffer so it wouldn't get lost between iterations.



### Kernel Optimization

#### TopK-V2

The DSA indexer turns each query into scores over historical KV

positions, then selects the top candidates for sparse attention. Following our

“Lightning-TopK” design introduced in [DeepSeek-V4 Blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/), we upgraded the original DSA TopK-V1 kernel to TopK-V2, which treated TopK as a selection problem rather than a sorting problem.

![TopK-v2 cluster-of-eight radix-select design](/images/blog/glm52-optimization/topk-v2-cluster-radix-select.png)

Figure 2: TopK-V2 partitions a long score row across eight CTAs, each building a local 10-bit histogram. A cluster-wide reduction locates the bin containing the 2048th-largest score; values above it are emitted directly, while boundary candidates undergo exact FP32 radix selection. The selected logical positions are then translated into physical indexer KV-cache slots.

![TopK-v2 10-bit histogram construction](/images/blog/glm52-optimization/topk-v2-histogram.png)

Figure 3: When TopK-V2 builds histogram, each FP32 score is rounded to FP16 and transformed into an unsigned key whose ordering matches the numerical score order. The upper 10 key bits select one of 1,024 bins, whose counter is atomically incremented. This coarse histogram only locates the boundary region; FP32 refinement preserves the accuracy of the final top-k selection.

TopK-V2 uses register-resident or single-CTA streaming paths for short and medium rows. For long rows, a cluster of eight CTAs builds local 10-bit radix histograms and reduces them across the cluster to identify the threshold bin.

A cluster-wide reduction locates the bin containing the 2048th-largest score; values above it are emitted directly, while boundary candidates undergo exact FP32 radix selection. The selected logical positions are then translated into physical indexer KV-cache slots. The kernel then collects candidates at the FP32 boundary and uses an exact radix tie-break to return exactly `k` entries, with runtime `k` supported up to 2048.

A planning kernel chooses the cluster cutoff from the batch's sequence-length distribution and builds a work list for the persistent cluster pool, so the resulting plan is generated at each forward and reused across DSA layers. TopK-V2's selection and page-table transform is also fused into a single kernel to cut latency.

![DSA TopK-v1 and TopK-v2 kernel latency comparison](/images/blog/glm52-optimization/topk-v1-vs-v2-latency.png)

Figure 4: Comparison of Kernel Latency between TopK-V1 and TopK-V2, on target model verification with batch size 1 and 6 draft tokens. Both kernels are fusing Top-K with page table transformation together.

From benchmark results, TopK-V2 reduces average kernel latency from 40.7 µs to 17.5 µs at 80K ISL, a 2.33× speedup. Its advantage grows with context length, reaching 10.17× at 1M ISL by reducing latency from 372.1 µs to 36.6 µs. This widening gap shows that TopK-V2 scales much more efficiently for long-context workloads.

#### Indexer Prologue Fusion

![DSA indexer prologue dependency chain before and after fusion](/images/blog/glm52-optimization/indexer-prologue-fusion.png)

Figure 5: DSA Indexer Prologue kernels: before and after the fusion

The DSA indexer prologue prepares two streams of data: a key representation

stored in the indexer KV cache and a query representation used to compute sparse

attention candidates. The original implementation expressed this as a sequence

of small kernels and projections.

In the "before" path, the key side runs `wk`, LayerNorm, RoPE, Hadamard transform, FP8 quantization, and cache store. The query side runs `wq_b`, RoPE, Hadamard transform, FP8 quantization, and head-gate scaling. In addition, `weights_proj` is a separate projection feeding the per-head gate.

[PR #27705](https://github.com/sgl-project/sglang/pull/27705) collapses this

dependency chain in two ways.

First, it fuses `wk` and `weights_proj` into a single BF16 projection,

`wk_weights_proj`. The output is split into the key activations and the raw

head-gate weights. This removes one small GEMM from the indexer path and lets

the head-gate weights be reused directly by the fused query kernel.

Second, it fuses the elementwise tails:

- Key path: LayerNorm + RoPE + FP8 quantization + paged indexer KV cache store.

- Query path: RoPE + FP8 quantization + head-gate scaling.

The figure illustrates the important scheduling consequence. Before fusion, the

cache store sat after the key-side work and extended the critical path. After

fusion, the key side can run as one kernel that includes the store, while the query side runs as a separate fused kernel. The two sides can overlap, so the indexer prologue becomes a shorter and cleaner pair of branches rather than a long chain of launches. The total amount of kernels drops from 12 to 4.

The fused path also drops the Hadamard transform. Applying the same orthonormal

transform to Q and K preserves their inner products before quantization, so its

The main effect was on the quantized representation. The fused path instead

quantizes the untransformed activations directly.

The kernel-count reduction translates directly into measurable decode throughput gains, though the effect is more pronounced at small batch sizes where launch overhead dominates. At batch size 1, decode throughput improves by roughly 8%, since the indexer prologue's memory-bound kernels disappear and the collapse from 12 kernels to 4 removes a proportionally larger share of the critical path. At batch size 128, the improvement is smaller but still consistent, at around 5%.

#### GEMM Kernels Improvement

![CuTe DSL BF16 GEMM speedup over cuBLAS](/images/blog/glm52-optimization/cutedsl-bf16-gemm-speedup.png)

Figure 6: Speedup of CuteDSL BF16 GEMM vs CuBLAS GEMM, over different batch sizes

Not every matrix multiplication in GLM-5.2 runs in NVFP4. To protect accuracy, the checkpoint's quantization recipe keeps the attention projections and the shared-expert MLP in BF16, and only quantizes the routed experts. [PR #30117](https://github.com/sgl-project/sglang/pull/30117) adds a selectable CuTe DSL BF16 GEMM backend from Flashinfer’s [TGV GEMM](https://github.com/flashinfer-ai/flashinfer/pull/3281) built specifically for these BF16 layers.

The kernel splits the work across warps with dedicated jobs: some warps only load data from memory, one warp only does the matrix multiply, and a few warps only write the result back out. Since these are separate warps all running at once, loading, computing, and storing overlap instead of happening one after another.

The real source of the speedup is how aggressively the kernel pipelines load. Instead of loading one tile of data and waiting for it to be used before loading the next, it keeps many tiles' worth of data in flight at once, using nearly all of the GPU's shared memory to do so. At the small batch sizes decoding runs at, these GEMMs spend most of their time waiting on memory rather than computing, so the further ahead the kernel can load, the less time is ever spent waiting. That's the main edge over a general-purpose library like cuBLAS, which pipelines more conservatively.

A tuning step also picks the tile size that best fits the shape being run, and a heuristic measured ahead of time decides per call whether to use this kernel or fall back to cuBLAS.

Two of these BF16 layers benefit clearly at TP4: the fused QKV projection (M, 2624, 6144, replicated across ranks) and the attention output projection o_proj (M, 6144, 4096, split across ranks). Sweeping the full decode range M=1 to 32:

Fused QKV projection wins at every batch size, averaging 1.08x over cuBLAS, peaking at 1.13x. o_proj also wins at every batch size, averaging 1.05x, peaking at 1.08x. For batch size 1, the end-to-end decoding speedup is approximately 4%.

## Performance Results

![GLM NVFP4 performance Pareto curves on SGLang](/images/blog/glm52-optimization/glm52-nvfp4-performance-pareto.png)

Figure 7: Pareto curves for GLM 5.2 NVFP4 performance on SGLang.

We collected the performance results of GLM NVFP4 models on an OpenHands multi-turn agentic coding workload in Figure 7. Each conversation starts with an around 80K token prompt, and outputs around 220 tokens each turn, for a total of 13-turns. Successive turns reuse the prefix for a ~92% aggregate cache hit rate. We perform a concurrency sweep and plot per-GPU token throughput (tok/s/GPU) against interactivity (tok/s/user). Each figure fixes the model, GPU family, precision, workload, serving framework, and serving mode, and compares SGLang against itself across different versions.

Three things stand out. First, GLM-5.2 is a substantially more efficient architecture than GLM-5.1. On the same SGLang release, GLM-5.2 delivers a ~1.4x and ~1.3x improvement in single-user interactivity per-GPU throughput, respectively, on both 4×GB300 and 8×B300. This gain comes from applying IndexShare to the DSA layers and from the improved MTP head, which reuses IndexShare and KVShare. Second, single-user interactivity has improved 18-34% since day-0. At batch size 1, our optimizations sharply cut per-token overhead, letting us reach **500+ TPS** on 8xB300. Third, we made no compromise on high-concurrency throughput,  peak throughput at batch size 8 also improved by 6–11%.

![Input sequence length ablation on four GB300 GPUs](/images/blog/glm52-optimization/isl-ablation.png)

Figure 8: Ablation test with changing input sequence lengths.

The ISL Ablation results in Figure 8 directly illustrate the payoff of our indexer optimizations. On the day-0 path, the DSA indexer must rank a score row that grows with the context, so its cost climbs with sequence length and single-user interactivity degrades rapidly as the input grows. TopK-V2 significantly alleviates that bottleneck, holding interactivity essentially flat all the way out to 1M tokens.

## What's Next

This blog is mostly focused on optimization of low concurrency and high cache-hit rate scenarios. In the future, we will extend our support to higher concurrency scenarios:

- Better kernels for heavier workloads: ragged TopK-V2 for prefill, faster MQA logits kernel for indexer

- Optimization of PD Disaggregation and Expert Parallel techniques under agentic workload

- Improving cache usage with [HiCache](https://docs.sglang.io/docs/advanced_features/hicache), [HiSparse](https://docs.sglang.io/docs/advanced_features/hisparse_guide#hisparse-hierarchical-sparse-attention), and [LayerSplit](https://github.com/sgl-project/sglang/pull/29421) techniques.

- Supporting [DSpark](https://www.lmsys.org/blog/2026-07-06-dspark-sglang) for GLM 5.2, which helps boosting the acceptance rate of speculative decoding under large concurrencies

## Acknowledgements

We would like to express our gratitude to the following organizations and individuals, for their contribution to the support and optimization of GLM 5.2 NVFP4 model.


SGLang Community/RadixArk: Khoa Pham, Baizhou Zhang, Jimmy Shong, Brayden Zhong, Ziyi Xu, Mohammad Miadh Angkad, Xinyuan Tong, Zhendong Hua, Zijie Xia, Banghua Zhu and many others – For optimization and benchmarking

Nvidia: Julien Lin, Zhiyu Cheng, Po-Han Huang, Ryan Stewart, Triston Cao and many others – For helping with Day-0 support of GLM5.2 NVFP4

GLM Team: Yuxuan Zhang – For implementing and verifying IndexShare in SGLang

## Appendix

### Reproduction

For reproducing the performance results, please refer to custom scripts under [this branch](https://github.com/Jiminator/sglang/tree/glm-nvfp4-blog-repro/benchmark/glm_nvfp4_blog).\
We are using SGLang v0.5.15.post1 as server environment, and [evalscope](https://github.com/modelscope/evalscope) as the benchmark client.

For the workload, we applied OpenHands multi-turn agentic replay, with mean input ≈ 80k tokens/request, 220 output tokens/turn, 13 turns/conversation, ~92% aggregate prefix-cache hit rate, and real EAGLE speculative acceptance without simulation.

The server launching commands are as follow:

```bash
# TP 4/TP 8
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1
python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \    # --tensor-parallel-size 8 for 8*B300
    --quantization modelopt_fp4 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs-decode 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --bf16-gemm-backend cutedsl \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host localhost \
    --port "$PORT"

# TEP 4/TEP 8
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1
python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \  #  --tensor-parallel-size 8 for 8*B300
    --ep-size 4 \   # --ep-size 8 for 8*B300
    --quantization modelopt_fp4 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs-decode 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --bf16-gemm-backend cutedsl \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host localhost \
    --port "$PORT"
```

### Pull Request List

IndexShare implementation: [#27114](https://github.com/sgl-project/sglang/pull/27114), [#29654](https://github.com/sgl-project/sglang/pull/29654), [#29787](https://github.com/sgl-project/sglang/pull/29787), [#30839](https://github.com/sgl-project/sglang/pull/30839), [#30992](https://github.com/sgl-project/sglang/pull/30992)

TopK-V2: [#26788](https://github.com/sgl-project/sglang/pull/26788),  [#30274](https://github.com/sgl-project/sglang/pull/30274)\
Draft extend cuda graph: [#29413](https://github.com/sgl-project/sglang/pull/29413)\
DSA metadata fusion and sync removal: [#29415](https://github.com/sgl-project/sglang/pull/29415), [#29499](https://github.com/sgl-project/sglang/pull/29499)

Indexer Prologue Fusion: [#27705](https://github.com/sgl-project/sglang/pull/27705)

GEMM Kernels: [#30177](https://github.com/sgl-project/sglang/pull/30117)

Other optimization PRs:  [#21531](https://github.com/sgl-project/sglang/pull/21531), [#29595](https://github.com/sgl-project/sglang/pull/29595), [#29667](https://github.com/sgl-project/sglang/pull/29667)
