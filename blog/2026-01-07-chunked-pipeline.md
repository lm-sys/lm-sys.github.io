---
title: "Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond"
author: "Shangming Cai"
date: "January 7, 2026"
previewImg: /images/blog/chunked-pipeline/pp_bubbles_after.png
---

## **TL;DR**

As Large Language Models (LLMs) scale toward trillion-parameter architectures and "infinite" context windows, the underlying serving infrastructure must evolve toward more granular, cross-node parallelization strategies. While KV cache techniques effectively mitigate redundant computation, they cannot circumvent the prohibitive Time to First Token (TTFT) inherent in ultra-long sequences with extremely large initial Input Token Length (ITL). Although Tensor Parallelism (TP) remains the conventional approach for intra-node scaling, it frequently encounters communication bottlenecks during multi-node deployments. On the other hand, despite traditional Pipeline Parallelism (PP) addressing this bottleneck by reducing the communication volume, it struggles with resource underutilization and bubble overhead when processing such massive prompts.

Drawing inspiration from both open-source innovations and academic research, SGLang introduces a highly optimized Pipeline Parallelism implementation featuring Asynchronous Communication and Dynamic Chunked Prefill, which effectively minimizes the pipeline bubbles. By integrating these techniques, SGLang explores and reframes the processing of ultra-long prompts—effectively scaling away the prohibitive latency of long-sequence prefilling and transforming it into a high-throughput, computationally scalable streaming workflow.

Empirical benchmarks demonstrate that SGLang’s PP implementation achieves industry-leading performance. In large-scale deployments, it maintains **over 80% scaling efficiency for various model architectures while scaling out to PP4**, and it also delivers **up to an 81% reduction in TTFT for ultra-long prompts when deploying Qwen3-235B-A22B-FP8 on H20 with PP8**.

## **The Challenge: The "Bubble" and The "Wall"**

In a traditional Pipeline Parallelism setup, the model layers are partitioned across GPUs (Stage 1 to Stage N). When serving standard requests (e.g., < 4K tokens), it normally works well. However, when processing a prompt exceeding **128K or even 1M tokens**, two critical issues emerge:

1. **The Pipeline Bubble:** Processing prompts as monolithic batches forces downstream GPUs into prolonged idle states, creating massive 'pipeline bubbles' that severely degrade throughput.  
2. **The Memory Wall:** Processing a 1-million-token prompt in a single pass requires storing and communicating intermediate hidden states for the entire sequence, resulting in significant overhead and peak memory footprint.

## **The SGLang Pipeline Parallelism Architecture**

SGLang’s pipeline implementation goes beyond the standard "sequential" approach. We’ve introduced several advanced features to minimize "bubbles" (i.e., GPU idle time) and maximize hardware utilization.

### **1\. Chunked Pipeline Parallelism (CPP)**

Processing a 1-million-token prompt in a single forward pass would lead to massive bubbles as later stages wait for the first stage to finish. Inspired by architectures like Mooncake[\[1\]](https://dl.acm.org/doi/pdf/10.1145/3773772), BladeLLM[\[2\]](https://arxiv.org/pdf/2501.15383?), and TeraPipe[\[3\]](http://proceedings.mlr.press/v139/li21y/li21y.pdf), SGLang supports Chunked Pipeline Parallelism. Instead of feeding the full prompt into the pipeline, SGLang partitions the prompt into smaller "chunks" (e.g., 4k or 6k tokens). These chunks flow through the pipeline stages like micro-batches. By breaking long prompts into smaller chunks, the system can "pipeline" the prefill phase. As soon as the first stage finishes computing the hidden states for Chunk 1 and initiates PP communication, it immediately moves to processing Chunk 2, while Stage 2 simultaneously begins processing Chunk 1. This reduces the pipeline startup latency from being proportional to the total sequence length to being proportional only to the first chunk size.

This approach marks a critical first step from an engineering perspective to tackle the challenges of ultra-long contexts. Notably, SGLang pioneered the support for this feature more than six months ago, underscoring its long-standing commitment to optimizing real-world, long-context inference.

### **2\. Better Overlapping: Micro-batching and Async P2P Communication**

Although combining Pipeline Parallelism and Chunked Prefill can significantly reduce communication volume compared to tensor parallelism, it often suffers from pipeline bubbles where the GPU blocks while waiting for CPU metadata processing or network transfers. To eliminate this performance hazard, SGLang implements a Micro-batching Event Loop with non-blocking asynchronous peer-to-peer (P2P) communication to overlap GPU computation with CPU metadata processing and PP communication. This ensures that while one micro-batch is being computed on the GPU, the next one is already being prepared and moved into position effectively, ensuring the pipeline remains as saturated as possible.

The key mechanisms of the implementation include:

* **Decoupled Sync/Async Logic in the Event Loop:** The scheduler uses `async_send` in `_pp_send_pyobj_to_next_stage`. Instead of waiting for a transfer to complete, it returns a `P2PWork` handle. The actual synchronization (`work.wait()`) is deferred until `_pp_commit_comm_work` is called, allowing the CPU to perform other work—like scheduling the next batch or processing metadata—while data is in flight.  
* **Multi-Stream Execution:** In addition to the main `default_stream`, which serves as the synchronization stream, SGLang utilizes dedicated `forward_stream` and `copy_stream` to execute forward pass GPU computation and Data-to-Host (D2H) memory transfers separately for better overlapping. While `_pp_launch_batch` is executing the current micro-batch on the GPU for the current stage, the CPU prepares the next micro-batch's results using `_pp_process_batch_result`.

### **3\. Advanced Option: Dynamic chunking**

With Chunked Pipeline Parallelism and Async P2P communication, SGLang already achieves over 80% scale efficiency as the PP size increases to 4. However, Chunked prefill with a fixed size can still cause bubbles in the pipeline, and this inefficiency becomes more pronounced as the PP degree increases. The main reason behind this phenomenon is that the model exhibits non-uniform execution latency across chunks of identical size, primarily due to the incremental nature of self-attention. **As the prefix sequence length grows, the per-chunk processing time increases non-linearly. These timing mismatches propagate through the pipeline, compounding efficiency losses at higher PP ranks.**  

![figure1](/images/blog/chunked-pipeline/pp_bubbles_before.png)<small><center>Fig. 1: Pipeline diagram with fixed chunked prefill size</center></small>

We tested different models using a large PP size and found that they all conformed to this conclusion. Below is the profile result of a typical case.

![figure2](/images/blog/chunked-pipeline/profile_before.png)<small><center>Fig. 2: Profile result of the PP rank 7 with fixed chunked prefill size</center></small>

To address this issue, SGLang introduces a dynamic chunking mechanism and uses a quadratic function to fit this condition:   
Runtime(***L*** \+ Next Chunk Size) \- Runtime(***L***) \= Runtime(Initial Chunk Size)  
where ***L*** denotes the Prefix Sequence Length. By profiling a series of requests with different ITL, we fit this quadratic function approximately, and use it to predict the length of the next chunk for each ***L***. Since Attention computation/communication complexity scales with ***L***, the next chunk size will be progressively reduced as ***L*** grows to maintain an aligned chunk execution time across pipeline stages.

Based on this method, the scheduler can predict and dynamically reduce the chunk size during runtime to minimize the bubbles caused by the stage misalignment.

![figure3](/images/blog/chunked-pipeline/pp_bubbles_after.png)<small><center>Fig. 3: Pipeline diagram with perfect dynamic chunked prefill size</center></small>

However, due to the variation in hardware, models, and target workloads, a static configuration is seldom optimal across all scenarios. Consequently, achieving peak performance necessitates a degree of hyperparameter tuning when switching to the dynamic chunking mode. Also, we find that it is hard to perfectly fit the quadratic function due to the kernel performance variation of different shapes. Therefore, we introduce an environmental variable (`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`) to smooth the reduction for the dynamic chunking algorithm, defaulting to 0.75, which determines how much the chunk size can change during the prefill phase. A larger value leads to more aggressive chunk size reduction, potentially improving performance but increasing the total number of chunks (the chunk size at the end may become very small, which could lead to performance degradation).  
Here is a simple tuning guidance:

- Iterate to find the optimal fixed chunked prefill size for the targeted PP size  
- Set the initial chunk size to a larger value (i.e., 2x or 3x) comparable to the original chunked prefill size, so that there won’t be too many chunks, and the tail small chunks won’t cause performance degradation due to not being able to saturate the hardware. To prevent performance degradation caused by small chunks, when the Input Token Length (ITL) is extremely large, the dynamic predictor also limits the next chunk size to be at least 1/4 of the initial chunk size.  
- Adjust the smooth factor. When it is set to 1, the chunk size will be adjusted strictly based on the aforementioned quadratic model that predicts the next chunk size. A smaller value means a more conservative chunk size change, which may lead to smaller chunk size changes and fewer total chunks. When it is set to 0, the chunk size will not be adjusted dynamically, so it is identical to the traditional way with a fixed chunked prefill size. Through experiments, we find that a range between 0.6 and 0.85 typically yields the best performance for dynamic chunking.

![figure4](/images/blog/chunked-pipeline/sigma_ds.png)<small><center>Fig. 4: Example of tuning the smooth factor (DeepSeek-V3.1)</center></small>

![figure5](/images/blog/chunked-pipeline/sigma_qwen.png)<small><center>Fig. 5: Example of tuning the smooth factor (Qwen3-235B-A22B-FP8)</center></small>

Another small optimization tip is to put the larger partition in the higher PP rank when the layers are not evenly divisible across ranks. It can increase the GPU utilization when a larger PP rank is waiting for the previous stage’s result, hence reducing the bubbles on higher PP ranks. If we take DeepSeek-V3.1 as an example, `SGLANG_PP_LAYER_PARTITION=15,15,15,16` usually performs better than `16,15,15,15`.

To validate the effectiveness of these combined strategies, we profiled the execution of DeepSeek-V3.1 using dynamic chunking. As observed in the following profile result of PP rank 3, the pipeline bubbles are significantly minimized compared to the static chunking approach, resulting in a more saturated execution.

![figure6](/images/blog/chunked-pipeline/profile_after.png)<small><center>Fig. 6: Profile result of the PP rank 3 with dynamic chunking (DeepSeek-V3.1)</center></small>

### **4\. Production Ready: Compatibility with PD Disaggregation and HiCache**

A unique strength of SGLang is its native support for **Prefill-Decode (PD) Disaggregation** within a pipelined setup. In a disaggregated cluster, the prefill nodes can use a high degree of PP to handle extremely long-context prompts, while the decode nodes can utilize a different parallelism strategy (like high-degree TP) to maximize token-generation speed.

* **Chunk by Chunk KVCache Transfer:** Instead of waiting for all chunks to be completed, SGLang supports transfer engine backends like mooncake, which can transfer the KVCache of one chunk from the prefill node to the decode node immediately when PD Disaggregation is enabled. This feature greatly reduces the KVCache transfer overhead.  
* **Flexible Hybrid Strategies:** SGLang allows users to mix and utilize multiple parallelisms with PD Disaggregation. You can run **PP8 TP8** for heavy prefill tasks on one set of nodes for prefill and switch to other combinations, such as **DP8 TP1, PP1 TP8, and PP8 TP1,** for high-throughput decoding, optimizing for different phases of the inference lifecycle. This allows users to meet the expected Time To First Token (TTFT) and TPOT targets for production in a highly customizable way.  
* **Memory Efficiency:** By distributing model weights across devices, PP reduces the per-GPU memory footprint, allowing for larger KV caches and higher concurrency. Therefore, it can be used to scale the max context length in some cases.

When handling contexts exceeding 128K tokens, SGLang also supports **Chunked Pipeline Parallelism (CPP)** with **HiCache,** a distributed hierarchical KV cache system to further reduce the TTFT of multi-turn QA and agentic applications for ultra-long initial ITL:

* **Semantic Prefix Matching:** SGLang’s HiCache uses a radix-tree-based hierarchy to match prefixes at the chunk level. When a long-context request arrives, SGLang can perform a Hierarchical Cache Look-up. If the prefix tokens (processed in previous chunks) are already cached in the HiCache "Storage" layer (e.g., host memory or local disk), the PP pipeline can skip those chunks entirely, drastically reducing TTFT.

## **Performance Impact**

This section provides a rigorous quantitative evaluation of the PP performance characteristics of the Qwen3-235B-A22B-FP8 and DeepSeek-V3.1 models. The analysis focuses on the interplay between PP size, Dynamic Chunking (DCK), and hardware scalability.

Our experimental testbed is a small cluster of 6 H20 nodes (8 × 96GB VRAM GPUs). Due to limited testing resources, experiments with a PP degree of 8 for DeepSeek-V3.1 are not conducted. Additionally, for the PP size \= 1 configuration of DeepSeek-V3.1, we used a standalone H20 node (8 × 141GB VRAM GPUs) to obtain the baseline performance for an input token length of 128 K. To better verify the throughput performance when the pipeline is saturated, we benchmarked and measured the average of 16 consecutive requests during the throughput test.

Note: We use DCK to mark the chunked prefill size setup when enabling the dynamic chunking, and σ stands for the smooth factor of dynamic chunking.

### **Input Token Throughput and Scale Efficiency**

The analysis of Throughput and PP size demonstrates strong horizontal scalability across both model families, though the degree of efficiency varies by configuration.

* **Superior Scalability of DCK**: The **Qwen DCK 18K** configuration exhibits the highest scalability, achieving a speedup factor of **6.14x** on a 32-GPUs (PP8) cluster. This performance suggests that the dynamic adjustment of chunk sizes optimizes the balance between computational intensity and inter-node communication latency.  
* **Architectural Comparison**: DeepSeek models demonstrate comparable scaling trajectories to Qwen up to the PP4 threshold. Notably, the **DeepSeek DCK 12K** (3.27x) marginally outperforms the static 4K variant (3.20x), validating the cross-architectural robustness of the Dynamic Chunking strategy in enhancing throughput.

![figure7](/images/blog/chunked-pipeline/ds_throughput.png)<small><center>Fig. 7: Throughput Analysis of DeepSeek-V3.1</center></small>

![figure8](/images/blog/chunked-pipeline/qwen_throughput.png)<small><center>Fig. 8: Throughput Analysis of Qwen3-235B-A22B-FP8</center></small>

![figure9](/images/blog/chunked-pipeline/normalized_throughput.png)<small><center>Fig. 9: Normalized Total Throughput vs. PP Size Analysis</center></small>


The Scale Efficiency curves illustrate the degradation of hardware utilization as the system scales. All configurations exhibit a monotonic decay in efficiency as PP size (GPU counts) increases. However, **Qwen DCK 18K** maintains a superior efficiency of **77%** at the PP8 scale, whereas the static 6K configuration drops to **70%**. This confirms that larger, dynamically managed chunks are more resilient to the communication overheads inherent in large-scale distributed inference. Due to resource constraints, DeepSeek-V3.1 was evaluated up to PP size \= 4, maintaining an efficiency of \~81.7%. Extrapolating the current slope suggests that DeepSeek would likely follow a similar efficiency trajectory to Qwen, where DCK is projected to outperform the fixed chunking strategy.

![figure10](/images/blog/chunked-pipeline/scale_efficiency.png)<small><center>Fig. 10: Scale Efficiency vs. PP Size Analysis</center></small>

### **Reduced TTFT and Scaling Out for 1 million ITL**

A critical observation from the experimental data is the performance degradation observed when scaling Tensor Parallelism (TP) to 16, compared to a hybrid Parallelism approach (PP2 TP8). Despite utilizing an identical aggregate GPU count, PP2 TP8 consistently outperforms PP1 TP16 in both throughput and latency metrics. This phenomenon can be attributed to the non-linear scaling of communication overheads in high-degree tensor parallelism.

Furthermore, increasing the pipeline depth from PP1 to PP4 can yield a substantial reduction in TTFT for both the fixed chunked setting and dynamic chunking. But dynamic chunking performs better for different PP setups. For the Qwen3-235B-A22B-FP8, the baseline TTFT of **\~55.5s** (PP1 TP4) is reduced to **\~10.5s** under the PP8 TP4 configuration, representing a latency improvement of approximately **81.1%**. And for the DeepSeek-V3.1, the baseline TTFT of **\~48.5s** (PP1 TP8) is reduced to **\~15.7s** under the PP4 TP8 configuration, depicting a latency improvement of approximately **67.6%**. These results indicate that Chunked Pipeline Parallelism is highly effective for reducing TTFT.

![figure11](/images/blog/chunked-pipeline/ds_ttft.png)<small><center>Fig. 11: TTFT Analysis of DeepSeek-V3.1</center></small>

![figure12](/images/blog/chunked-pipeline/qwen_ttft.png)<small><center>Fig. 12: TTFT Analysis of Qwen3-235B-A22B-FP8</center></small>

To demonstrate the scalability of SGLang with this optimized Chunked Pipeline Parallelism, we benchmarked the TTFT across varying input token lengths for Qwen3-235B-A22B-FP8 with PP8 (32 NVIDIA H20 GPUs). As shown in the table below, the system efficiently scales to handle massive contexts. Even at the extreme edge of **1 million tokens**, SGLang maintains high stability and acceptable latency on NVIDIA H20, showcasing its capability for the most demanding long-context applications.

<small><center>Table 1: TTFT vs. Input Token length for Qwen3-235B-A22B-FP8 with PP8 TP4 on H20</center></small>
<div align="center">

| Input Token Length | 128K | 256K | 512K | 1M |
| :---: | :---: | :---: | :---: | :---: |
| TTFT (s) | 10.54 | 32.68 | 114.33 | 420.91 |

</div>

We invite the community to try out this new feature across diverse hardware configurations. Please share your performance findings and report any bugs you encounter. We’d love to hear from you—feel free to drop any questions in the issue of the [PP Roadmap](https://github.com/sgl-project/sglang/issues/11857). Your feedback is crucial in helping us refine these long-context optimizations!

<div style="border-left: 4px solid #3b82f6; padding: 10px 12px; margin: 12px 0; background: #eff6ff; border-radius: 8px;">
  <strong>👉 Check out the <a href="https://github.com/sgl-project/sglang/issues/11857">Roadmap</a>.</strong>
</div>

## **Getting Started**

To leverage these features, you only need to configure the `--pp-size` and `--chunked-prefill-size`. To further employ the dynamic chunking solution, please use `--enable-dynamic-chunking` and set up the environmental variable `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`.
Note: required SGLang release version `>= v0.5.7`
Examples:
```bash
# Example: Serving DeepSeek-V3.1 with 128K Input Token Length (32 GPUs total)
# Using 8-way Tensor Parallelism and 4-way Pipeline Parallelism

# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096

# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking


# Example: Serving Qwen3-235B-A22B-FP8 with 128K Input Token Length (32 GPUs total)
# Using 4-way Tensor Parallelism and 8-way Pipeline Parallelism

# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144

# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --mem-fraction-static 0.8 --attention-backend fa3 \
  --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

## **Future Roadmap:**

We are continuously refining the PP stack. Our 2026 H1 PP Roadmap includes these important tasks:

* Compatibility with Context Parallelism to further reduce TTFT  
* Pipeline Parallelism for the Decode side   
  * Performance Optimization and best practice tuning  
* Better fitting and chunking strategy for dynamic chunking

## **Conclusion**

SGLang’s implementation of Pipeline Parallelism is more than just model splitting; it is a complete re-engineering of the inference lifecycle for the Long-Context Era. By combining chunked prefill with asynchronous communication and dynamic chunking, SGLang provides the most efficient and open-sourced path to serving and accelerating trillion-parameter models for long context.

## **Acknowledgement**

- We would like to thank the SGLang team and community for the implementation and generous support, especially Shangming Cai, Xuchun Shang, Yanbo Yang, Leon Gao, Zhiqiang Xie, Ying Sheng, Lianmin Zheng, and many others.
- We would like to thank Jianhao Fu (from AntGroup SCT and Inference Team), Kevin Li (from TikTok), Siyu Liu (from Alibaba Cloud Computing), Xiaolei Zhang (from ByteDance), Teng Ma (from Alibaba Cloud Computing), Chao Wang (from Meituan), and Xiaowei Wang (from NVIDIA) for their prominent contribution in code improvement and testing.
- We learn a lot from the system design of SGLang, vLLM, Mooncake[1], and TeraPipe[3], which jointly help improve this Pipeline Parallelism implementation.

## **Reference**

[1] Qin, Ruoyu, et al. "Mooncake: A kvcache-centric disaggregated architecture for llm serving." ACM Transactions on Storage (2024).  
[2] Yang, An, et al. "Qwen2. 5-1m technical report." arXiv preprint arXiv:2501.15383 (2025).  
[3] Li, Zhuohan, et al. "Terapipe: Token-level pipeline parallelism for training large-scale language models." International Conference on Machine Learning. PMLR, 2021.