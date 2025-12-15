---
title: "SGLang Day-0 Support for MiMo-V2-Flash Model"
author: "SGLang Team and Xiaomi LLM Core Team"
date: "December 15, 2025"
previewImg: /images/blog/mimo-v2-flash/decode_1.png
---

## Introduction
MiMo-V2-Flash, with 309B total parameters and 15B activated parameters, is a new inference-centric model designed to maximize decoding efficiency. It is based on two key designs: **sliding window attention** and **multi-layer MTP**. MiMo-V2-Flash is explicitly co-designed for real-world serving workloads, enabling flexible tradeoffs between throughput and latency on different hardware. Combined with SGLang’s optimized Spec v2 runtime, which provides near-zero-overhead support for multi-layer MTP and efficient SWA execution, MiMo-V2-Flash delivers balanced TPOT and throughput on H200. In this blog, we will introduce the model and SGLang's efficient support.

## Inference-Efficient Modeling
The design of MiMo-V2-Flash follows an inference-efficiency principle. The MiMo-V2-Flash adopted two critical designs:

1. Sliding Window Attention (SWA): In SWA, each token’s receptive field is limited to a fixed, constant-sized window, to reduce the attention's complexity on the sequence dimension.
2. MTP: MiMo-V2-Flash's multi-layer MTP uses a chain of prediction heads, where each head sequentially predicts the next token. The resulting draft tokens are then verified in parallel in the following step using an extended query.
The overview of MiMo-V2-Flash is shown in the figure below:

![figure1](/images/blog/mimo-v2-flash/overview.PNG)<small><center>MiMo-V2-Flash Overview</center></small>

Now let's see how those designs lead to a cost-efficient inference.

### SWA
In MiMo-V2-Flash, every five attention layers with a sliding window pattern are alternated with one dense GQA. The wide use of SWA can benefit the inference from multiple perspectives. First, during the prefilling stage, compute dominates the cost. Especially when the sequence is long, $O(N^2)$ attention computation is the bottleneck. SWA reduces the $O(N^2)$ complexity to a linear level to sequence length, $O(Nw)$, where $w$ is the window size. In a long context scenario, this design can significantly reduce the TTFT. SWA also reduces KV cache complexity to a constant level - releasing more resources for a larger batch size, and allows a better TPOT through fewer KV cache loading operations. 

The figure below shows the prefill benchmarking results for MiMo-V2-Flash. 

![figure2](/images/blog/mimo-v2-flash/prefill.PNG)<small><center>MiMo-V2-Flash Prefill Benchmark</center></small>

### MTP
One of the most important designs in MiMo-V2-Flash is the multi-layer MTP, with 3 MTP layers. 

In decoding scenarios, most of the kernels are memory-bound. Since the query length is always 1, using a larger number of parallel decoding tokens is the most intuitive way to achieve higher throughput. 

However, as the batch size increases to a certain level, this effect will be restricted - the KV cache memory access also grows linearly with the batch size, and it performs as the memory-bounded bottleneck. At this time, the device's computing potential is still not fulfilled, but it's hard to increase throughput by increasing batch size. 

MTP can still leverage this underexploited compute to reduce the TPOT. In MTP, multiple tokens are generated at the same time by sequential prediction heads, and the tokens will be verified in parallel in the same query, increasing the query length. This will not trigger more KV cache access; it will always increase arithmetic intensity. When the inference is still heavily memory-bound, and the batch size's effect has been marginal, an aggressive MTP strategy with a satisfying acceptance rate can theoretically leverage the rest of the device's potential and achieve a better TPOT.

## Hardware-Aware MTP Configuration
Since MTP benefits from an unsaturated arithmetic intensity, and GQA's arithmetic computation is low - MiMo-V2-Flash attention design is natively well-suited for multi-layer MTP. However, when deploying MiMo-V2-Flash, choosing the right combination of batch size and MTP depth is still essential for achieving the optimal compute–memory balance and maximizing performance across different hardware platforms. Theoretically, we want to choose the best tradeoff, which achieves a satisfying throughput and TPOT simultaneously. The sweet spot of this trade-off depends on the hardware, because each hardware platform has its own roofline model. 

Generally speaking, devices with a higher roofline benefit more from aggressive MTP because they have abundant compute capacity that is harder to saturate in memory-bound decoding. In contrast, inference-oriented accelerators, e.g., H20, have comparatively limited FLOPs, and the usage of MTP should be more careful: aggressive MTP depth can push the workload to compute-bound and degrade the throughput.

Here, we provide the benchmarking results on H200. MiMo-V2-Flash achieves balanced performance in both throughput and per request TPS. Thanks to SWA and MTP, the per request decoding throughput remains at 150 TPS even under long-context settings of up to 64K input tokens with per DP rank batch size 16.

![figure3](/images/blog/mimo-v2-flash/decode_1.png)<small><center>MiMo-V2-Flash Decode Benchmark (DP 2, TP 4, MTP Accept Length 3.6, Input Token Length 16k, Varying Batch Size)</center></small>

![figure4](/images/blog/mimo-v2-flash/decode_2.png)<small><center>MiMo-V2-Flash Decode Benchmark (DP 2, TP 4, MTP Accept Length 3.6, Input Token Length 16k, Varying Batch Size)</center></small>

## Fast MTP Serving with SGLang Spec v2
MiMo’s multi-layer MTP is implemented natively on SGLang’s spec v2. We apply the fully overlapped MTP feature to improve throughput and latency, delivering faster MTP serving. In spec v2, the overlap scheduler is fused with speculative decoding: output sync/processing is delayed while the next batch’s kernels launch early, so CPU overhead for batching and syncing is hidden in GPU forward. This cuts GPU bubbles and improves throughput and latency.

The figure below is a screenshot of the profiling, showing the overlapped decoding process with spec v2.

![figure4](/images/blog/mimo-v2-flash/profile.png)<small><center>Overlapped Speculative Decoding Profile</center></small>

## More Discussions
In most LLM-serving workloads, the decoding stage is memory-bounded, leaving substantial compute underutilized, particularly on the mainstream training-oriented GPUs. While inference-specific accelerators with high bandwidth and lower FLOPs offer a cost-efficient choice, their speed is limited. MiMo-V2-Flash attempts to take another perspective to make the model itself inference-efficient. The multi-layer MTP model may be a generalizable solution - if the acceptance rate can be further optimized, it allows people to leverage their GPU's computation to achieve faster decoding. With a more adaptable architecture, hardware selection becomes more flexible: each device can operate at its own optimal compute–memory balance point. This opens the possibility of using the same class of hardware for both training and inference, simplifying deployment and reducing overall system cost.

## SGLang Integration
MiMo-V2-Flash support is already available in SGLang via PR and will be merged into the main branch shortly. The benchmarks in this blog were conducted on MiMo’s optimized branch, and the corresponding optimizations will be upstreamed into SGLang main in the near future.