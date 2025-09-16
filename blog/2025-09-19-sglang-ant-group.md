---
title: "Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G"
author: "Tianyu Zhang, Peng Zhang, Yusong Gao, Yun Zhang, Yongfei Xu, Zhe Wang, Qianyu Zhang, Chun Huang, Xi Chen, Fakang Wang, Jianhao Fu"
date: "September 19, 2025"
previewImg: /images/blog/hicache/hicache_overview.png
---

# Introduction
我们在蚂蚁集团数万卡规模的H20集群下，基于SGLang，针对不同用户的延迟要求，构建出了一套能够满足不同用户的延迟要求，同时具备高吞吐、低成本的方案。
我们将在本文中分享我们在H20上使用SGLang部署DeepSeek-R1的最佳

## Challenge with H20

### Hardware Comparison

| Hardware Spec       | H20-96G     | H800-80G   |
|---------------------|-------------|------------|
| FP8 Compute         | 296 TFLOPS  | 1979 TFLOPS|
| FP16/BF16 Compute   | 148 TFLOPS  | 989 TFLOPS |
| Memory Capacity     | 96 GB       | 80 GB      |
| Memory Bandwidth    | 4000 GB/s   | 3352 GB/s  |
| NVLink Bandwidth    | 900 GB/s    | 400 GB/s   |
| RDMA NIC Bandwidth  | 4 × 400 Gb/s| 8 × 400 Gb/s|

---

### 1. H20 vs. H800

**Strengths**
- Larger memory capacity (96 GB vs. 80 GB)  
- Higher memory bandwidth (4000 GB/s vs. 3352 GB/s)  
- Significantly higher NVLink bandwidth (900 GB/s vs. 400 GB/s)  

**Limitations**
- **Significantly weaker compute performance**:  
  - FP8 throughput: 296 TFLOPS vs. 1979 TFLOPS  
  - FP16/BF16 throughput: 148 TFLOPS vs. 989 TFLOPS  
- **Limited RDMA NIC bandwidth**: 4 × 400 Gb/s vs. 8 × 400 Gb/s  

---

### 2. Lack of Industry References

Since May 2025, we have been exploring **expert parallelism (EP) strategies** tailored for the H20.  
However, both the DeepSeek and SGLang communities mainly focus on **high-performance GPUs (e.g., H800)**.  
As a result, the industry lacks effective parallelization schemes for **low-compute GPUs like the H20**.  

---

### Summary

Under the dual challenge of **weaker compute capability** and **no existing reference solutions**,  
our objective is to achieve the **same SLA as high-end GPUs** while **minimizing cost**.  

# Our Solution: Make H20 Great for Inference in Real World

## Deployment Strategy

### Prefill: TP8

Unlike the community’s multi-node **DP+EP** deployment scheme,  
we adopt a **single-node TP** approach for Prefill, due to the following reasons:

1. **TTFT Constraint**  
   - The Prefill stage is **compute-intensive**.  
   - Community practices (e.g., Tencent, ByteDance) on H20 typically use multi-node **DP+EP**, but in our tests, such deployment resulted in excessively long **TTFT**, failing to meet user requirements (e.g., TTFT < 1s).  

   **Experiments (May 2025, 16× H20 with Attention-DP + MoE-EP):**
   - **Single-node TP8 (8 GPUs):**  
     - Peak per-GPU input throughput: **1500 tokens/s**  
     - TTFT remains well-controlled.  
   - **Dual-node DP+EP:**  
     1. Attention-DP16 + MoE-EP16:  
        - Peak per-GPU throughput: **1600 tokens/s**  
        - TTFT exceeded **3s**, violating SLO requirements.  
     2. Attention with TP > 1:  
        - Significantly reduced TTFT,  
        - but throughput still inferior to single-node TP8.  

2. **Elastic Scaling**  
   With KVCache in mind, we require Prefill to **scale elastically**:  
   - **Scale in** when load is low (e.g., high KVCache hit rate).  
   - **Scale out** when load is high (e.g., low KVCache hit rate, or long-sequence requests).  
   Multi-node DP+EP makes scaling policies far more complex, whereas single-node TP provides a simpler and more flexible solution.  

---

### Decode: DP16 + EP16 (Small EP)

Unlike community approaches that rely on **large-scale EP (EP ≥ 32)**,  
we adopt a **smaller EP configuration (DP16 + EP16)** for Decode, motivated by:  

1. **H20 Hardware Characteristics**  
   - **Weaker compute performance**:  
     - Under online latency constraints, batch size cannot be scaled up significantly.  
   - **Larger memory capacity**:  
     - No memory-bound issues despite weaker compute.  
     - Enables further optimization with FP8 quantized KVCache.  
   - **Higher NVLink bandwidth**:  
     - DeepEP supports NVLink.  
     - H20’s NVLink bandwidth is **more than 2× higher than H800**,  
       ensuring ~50% of MoE communication can fall on NVLink.  

2. **Fault Radius**  
   - Smaller EP configuration minimizes the **blast radius** of Decode or single-GPU failures.  
   - EP high-availability solutions are still in draft, making small EP safer in production.

# Performance

## Prefill
@苏墨  

## Decode
### EPLB
### Async
- 动态EPLB的异步优化有个设计图在这里  
  https://yuque.antfin.com/rrv1v3/kg7h1z/aoadkogyvba0qsi6?inner=Qzo0O ，最终效果可以达到和超过静态EPLB的效果（在不同的业务数据集上），同时均衡度可以始终维持在70%以上。  
- EPLB优化（加入专家通信考量）的思路：在记录专家负载的基础上加入对每次激活的top-k专家组的记录，以此为基础计算专家间的亲和度矩阵（即共同激活的概率），在EPLB完成单卡的均衡后，针对卡内最热专家与其他卡内专家的亲和度，调整卡的位置以减少后续跨机通讯的发生，在EPLB的基础上还能提升5%左右的性能。  

## Computation
- FlashMLA-FP8  
- DeepGEMM swapAB  

## Overlap: SBO（Single-batch-overlap）
### Motivation
The optimization effect of Two-Batch Overlap (TBO) is suboptimal for the Decode phase on low-compute hardware (e.g., H20), primarily due to the following reasons:

- Hopper Architecture Limitation: The block_m of WGMMA is fixed at 64 on the Hopper architecture. When TBO is enabled in small-batch decoding, the MLP GEMM suffers from redundant computations. A positive throughput gain is only observed at larger batch sizes (e.g., 64, 128).
- SLA Constraints: At these large batch sizes, low-compute hardware like H20 fails to meet the SLA guarantees for TPOT/ITL.

Therefore, a solution is needed to improve Decode throughput even with small batch sizes. Single Batch Overlap (SBO) presents itself as a viable solution.
We implement SBO for DeepSeek v3/R1 by modifying DeepEP and DeepGEMM, including:

- Overlapping Shared Expert with Dispatch Recv.
- Overlapping Down GEMM with Combine Send.

Detailed implementation is available in the following branches:

DeepEP: deepseek-ai/DeepEP#390
DeepGEMM: deepseek-ai/DeepGEMM#183

### Designs
SBO implements two overlaps for the MoE layers of DeepSeek V3/R1:

- Overlapping Shared Expert with Dispatch Recv.
- Overlapping Down GEMM with Combine Send.

The interaction between Down GEMM and Combine Send is structured as a Producer-Consumer model synchronized by signals:

- For each local expert, a signal unit is allocated for every block_m tokens.
- The Down GEMM computes the results for these block_m tokens and atomically increments the signaling unit after completing a portion of the work.
- The Combine Send polls this signaling unit. Once the value reaches a threshold, it sends the corresponding block_m tokens.


## Observability: Lightweight Anomaly Diagnosis for Distributed MoE Model Deployment

<h1 style="display: flex; align-items: center;">
    <img alt="DeepX" style="margin-right: 0.2em" src="images/blog/ant-group-prac/deepx.svg">
</h1>

In large-scale distributed Expert Parallelism (EP) deployment of Mixture of Experts (MoE) models, increasing EP counts can lead to significant inference latency (TTFT & TPOT) due to communication overheads from operators like Dispatch and Combine. To address this, we designed a lightweight anomaly diagnosis workflow (see diagram above) that can pinpoint issues within minutes.

### 1. Metrics Collection
- Each node (Node 0 to Node N) periodically collects communication and computation metrics for its ranks.
- Every 10 seconds, rank data is gathered to Rank 0 and logged as `diagnose_rank${rank}.log`.

### 2. Anomaly Detection
- Rank 0 constructs an `N×N` latency matrix `M`, where `Mij` represents the latency of `rank_i` waiting for `rank_j`.
- Statistical analysis (z-score) identifies three types of anomalies:
  - **Column Anomaly**: Destination rank (`dst`) is globally slow.
  - **Row Anomaly**: Source rank (`src`) is globally slow.
  - **Point Anomaly**: Specific (`src`, `dst`) link is abnormal.

### 3. Root Cause Analysis
- Diagnostic metrics are used to infer anomaly sources:
  - **Comm Slow**: Communication link issues.
  - **Comp Slow**: Accumulated computation delays.
  - **Mixed Slow**: Uneven expert distribution or hotspot congestion.

### 4. Visualization (Web UI)
- Analysis results are displayed via a Web UI as a matrix heatmap, intuitively highlighting slow destination ranks, source ranks, or specific links.
- Users can quickly identify issue types and root causes, enabling targeted optimization measures.
  

# Performance

## Offline
- DP32EP32: 4.5K/1.5K = 850 tokens/s/GPU  
- DP16EP16: 4.5K/1.5K = 800 tokens/s/GPU  

## Online
- Base （2s，70ms）
- Pro （1.5s，50ms）  
- Max （1s，30ms）