---
title: "Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G"
author: "Tianyu Zhang*, Peng Zhang*, Yusong Gao, Yun Zhang, Yongfei Xu, Zhe Wang, Qianyu Zhang, Chun Huang, Xi Chen, Fakang Wang, Jianhao Fu"
date: "September 19, 2025"
previewImg: /images/blog/hicache/hicache_overview.png
---

# Introduction
The operationalization of large Mixture-of-Experts (MoE) models like DeepSeek-R1 demands a delicate balance between latency, throughput, and cost. This challenge is uniquely amplified on hardware with an asymmetric performance profile, such as the H20 GPU, which offers high memory bandwidth but comparatively low compute throughput. Our objective was to architect a serving solution that could match the stringent SLA requirements of high-end GPUs while capitalizing on the H20's cost-effectiveness.
This report details our best practices for achieving this goal. We present a novel, hardware-aware deployment strategy that deviates from community norms, alongside a suite of deep system and kernel-level optimizations. Our key contributions include:
- A tailored parallelization strategy: Hardware-aware parallelization strategy using single-node TP8 for prefill and small-scale EP16 for decode to meet latency targets and reduce fault domains.
- Advanced kernel-level optimizations: Including FlashMLA-FP8 and DeepGEMM swapAB, to maximize computational throughput on the H20 architecture.
- Innovative scheduling and load-balancing techniques: Implementation of Single-Batch Overlap (SBO) to enhance throughput in small-batch scenarios and asynchronous Expert Affinity Load Balancer to minimize cross-node communication。
- A lightweight observability framework: A purpose-built diagnostic system for rapidly identifying and resolving performance bottlenecks in a distributed MoE environment.

# Challenges and Opportunities with H20

## Why H20 Matters
- **Resource availability**  
  H20 is much easier to acquire in many area, allowing user to rapidly scale up to a **tens-of-thousands-card H20 cluster**.  

- **Cost efficiency**  
  The average cloud rental cost for **H20-96GB** is about **¥5/hour/GPU** (annual subscription).  
  At 10000 GPUs scale, a **10% throughput improvement** yields **hundreds of thousands of RMB in daily savings** on compute costs.  


## Hardware Comparison: H20 vs. H800

| Hardware Spec       | H20-96G     | H800-80G   |
|---------------------|-------------|------------|
| FP8 Compute         | 296 TFLOPS  | 1979 TFLOPS|
| FP16/BF16 Compute   | 148 TFLOPS  | 989 TFLOPS |
| Memory Capacity     | 96 GB       | 80 GB      |
| Memory Bandwidth    | 4000 GB/s   | 3352 GB/s  |
| NVLink Bandwidth    | 900 GB/s    | 400 GB/s   |
| RDMA NIC Bandwidth  | 4 × 400 Gb/s| 8 × 400 Gb/s|

**Strengths**  
- Larger memory capacity (96 GB vs. 80 GB)  
- Higher memory bandwidth (4000 GB/s vs. 3352 GB/s)  
- More than 2× NVLink bandwidth (900 GB/s vs. 400 GB/s)  

**Limitations**  
- Significantly weaker compute performance  
  - FP8 throughput: 296 TFLOPS vs. 1979 TFLOPS  
  - FP16/BF16 throughput: 148 TFLOPS vs. 989 TFLOPS  
- Lower RDMA NIC bandwidth (4 × 400 Gb/s vs. 8 × 400 Gb/s)  

**Insight**  
Although H20 lags far behind H800 in raw compute, inference workloads—especially **Decode**—are typically **memory-bound rather than compute-bound**.  
This makes H20’s **high memory bandwidth and larger memory capacity** particularly well-suited for large-scale inference optimization.  

## Challenges and Opportunities Ahead

~~Since May 2025, we have been exploring **expert parallelism (EP) strategies** designed for H20.  
While the DeepSeek and SGLang communities mainly target **high-performance GPUs (e.g., H800)**, this divergence creates a landscape of both **challenges and opportunities**:  

- **Challenge**: The industry lacks mature parallelization schemes for **low-compute GPUs like H20**.  
- **Opportunity**: This gap allows us to pioneer new optimizations and unlock the full potential of large-scale H20 clusters. ~~

# Our Solution: Make H20 Great for Inference in Real World

## Optimizations

### Prefill
@苏墨  

### Decode
#### EPLB
##### Expert Affinity EPLB

![eplb.png]()

- **Core Idea**: Extend standard EPLB by recording **top-k expert co-activations**, building an **affinity matrix**.
- **Method**: After intra-GPU balancing, adjust expert placement so **highly co-activated experts** are kept within the same node.
- **Impact**: Minimizes **cross-node communication** and delivers an extra **~5% performance gain** compared to vanilla EPLB.

##### Asynchronous Dynamic Load Adjustment

![async-load.png]()

- **Design**: Separates **load balancing** from **inference**, enabling parallel execution without blocking.  
- **Execution**: Uses a **hierarchical transfer strategy** to reduce migration impact, keeping inference seamless.  
- **Results**: Matches or exceeds static EPLB and maintains **>70% load balance ratio**.

#### Computation

##### FlashMLA-FP8

**Overview**

- End-to-end FP8 attention on Hopper (`SM90`) using `TMA` for memory transfers and `WGMMA` for computation.  
- Two warpgroups pipeline `QK^T` and `PV`, minimizing shared-memory pressure and overlapping memory with compute.  

**Improvements**

- **vs BF16 FlashMLA**: ~70% faster  
  - FP8 `Q`/`KV`, `WGMMA FP8`, shared memory reallocation, removed redundant operations.  
- **vs previous FP8 (#54)**: ~5% faster  
  - Optimized `TMA`–`WGMMA` pipeline, ping-pong buffers (`sP0/sP1`, `sVt0/sVt1`), 128-bit `STSM/LDSM` for FP8 layout, fine-grained `Q@K` tiling with BF16 ROPE, aligned with `SM90` style.

##### DeepGEMM swapAB

**Overview**

![swapAB.png]()

- Addresses PTX constraints: `N` multiple of 8, `M` fixed at 64.  
- **swapAB** swaps WGMMA tile usage (maps problem `M` → WGMMA `N`) to enable smaller `BLOCK_M (32)` for finer tiling and better resource utilization.  
- Best for small, irregular, or non-multiple-of-64 `M`; less advantage for large, well-aligned `M`.  

**Improvements**

- **Aligned / predictable M**: SwapAB improves boundary efficiency and throughput (up to ~70%), fully utilizing tiles and enabling higher concurrency.  
- **Irregular / varying M**: SwapAB improves load balance and occupancy, giving consistent gains across groups, especially for small or uneven `M`.

### Overlap: SBO（Single-batch-overlap）

#### Why not TBO (Two-batch-overlap)

![tbo_vs_normal_perf.png]()

The performance benefit of Two-Batch Overlap (TBO) in the Decode phase is limited on low-compute hardware (e.g., H20):

- **Hopper architecture constraint**: WGMMA’s `block_m` is fixed at 64. With small-batch decoding, TBO introduces redundant MLP GEMM computations. Positive throughput gains appear only at large batch sizes (e.g., 64 or 128).  
- **SLA limitations on H20**: At these large batch sizes, low-compute hardware cannot meet SLA targets for TPOT, making TBO impractical in online serving.

To improve Decode throughput without violating SLA, **Single Batch Overlap (SBO)** is adopted in DeepSeek v3/R1 by modifying DeepEP and DeepGEMM:  

- Overlapping Shared Expert with Dispatch Recv.
- Overlapping Down GEMM with Combine Send.

Detailed implementation is available in the following branches:

DeepEP: deepseek-ai/DeepEP#390
DeepGEMM: deepseek-ai/DeepGEMM#183

#### Designs

![SBO.png]()

SBO implements two overlaps for the MoE layers of DeepSeek V3/R1:

- Overlapping Shared Expert with Dispatch Recv.
- Overlapping Down GEMM with Combine Send.

![SBO-producer-consumer.png]()

The interaction between Down GEMM and Combine Send is structured as a Producer-Consumer model synchronized by signals:

- For each local expert, a signal unit is allocated for every block_m tokens.
- The Down GEMM computes the results for these block_m tokens and atomically increments the signaling unit after completing a portion of the work.
- The Combine Send polls this signaling unit. Once the value reaches a threshold, it sends the corresponding block_m tokens.


## Observability: Lightweight Anomaly Diagnosis for Distributed MoE Model Deployment

![deepx.png]()

In large-scale distributed Expert Parallelism (EP) deployment of Mixture of Experts (MoE) models, increasing EP counts can lead to significant inference latency (TTFT & TPOT) due to communication overheads from operators like Dispatch and Combine. 
To address this, we designed a lightweight anomaly diagnosis workflow (see diagram above) based on [DeepXTrace](https://github.com/antgroup/DeepXTrace) that can pinpoint issues within minutes.

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
  - **Comp Slow**: Accumulated computation delays.
  - **Mixed Slow**: Uneven expert distribution or hotspot congestion.
  - **Comm Slow**: Communication link issues.

### 4. Visualization (Web UI)
- Analysis results are displayed via a Web UI as a matrix heatmap, intuitively highlighting slow destination ranks, source ranks, or specific links.
- Users can quickly identify issue types and root causes, enabling targeted optimization measures.

# Performance

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
   - **Two-node DP+EP:**  
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
     - H20’s NVLink bandwidth is **more than 2× higher than H800**, ensuring ~50% of MoE communication can fall on NVLink.  

2. **Fault Radius**  
   - Smaller EP configuration minimizes the **blast radius** of Decode or single-GPU failures.  
   - EP high-availability solutions are still in draft, making small EP safer in production.

## Offline
- DP32EP32: 4.5K/1.5K = 850 tokens/s/GPU  
- DP16EP16: 4.5K/1.5K = 800 tokens/s/GPU  

## Online
- Base （2s，70ms）
- Pro （1.5s，50ms）  
- Max （1s，30ms）