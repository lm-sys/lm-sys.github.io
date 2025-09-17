---
title: "Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G"
author: "Tianyu Zhang, Peng Zhang, Yusong Gao, Yun Zhang, Yongfei Xu, Zhe Wang, Qianyu Zhang, Chun Huang, Xi Chen, Fakang Wang, Jianhao Fu"
date: "September 19, 2025"
previewImg: /images/blog/hicache/hicache_overview.png
---

# Introduction
The operationalization of large Mixture-of-Experts (MoE) models like DeepSeek-R1 demands a delicate balance between latency, throughput, and cost. This challenge is uniquely amplified on hardware with an asymmetric performance profile, such as the H20 GPU, which offers high memory bandwidth but comparatively low compute throughput. Faced with this hardware reality and a lack of established industry practices for such platforms, our objective was to architect a serving solution that could match the stringent SLA requirements of high-end GPUs while capitalizing on the H20's cost-effectiveness.
This report details our best practices for achieving this goal. We present a novel, hardware-aware deployment strategy that deviates from community norms, alongside a suite of deep system and kernel-level optimizations. Our key contributions include:
- A tailored parallelization strategy: Hardware-aware parallelization strategy using single-node TP8 for prefill and small-scale EP16 for decode to meet latency targets and reduce fault domains.
- Advanced kernel-level optimizations: Including FlashMLA-FP8 and DeepGEMM swapAB, to maximize computational throughput on the H20 architecture.
- Innovative scheduling and load-balancing techniques: Implementation of Single-Batch Overlap (SBO) to enhance throughput in small-batch scenarios and asynchronous Expert Affinity Load Balancer to minimize cross-node communication。
- A lightweight observability framework: A purpose-built diagnostic system for rapidly identifying and resolving performance bottlenecks in a distributed MoE environment.

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
     - H20’s NVLink bandwidth is **more than 2× higher than H800**, ensuring ~50% of MoE communication can fall on NVLink.  

2. **Fault Radius**  
   - Smaller EP configuration minimizes the **blast radius** of Decode or single-GPU failures.  
   - EP high-availability solutions are still in draft, making small EP safer in production.

## Optimizations

### Prefill
@苏墨  

### Decode
#### EPLB
##### Expert Affinity EPLB
- **Core Idea**:  
  Building upon expert load tracking, we additionally record the **top-k expert groups** activated in each iteration to compute an **expert affinity matrix** (i.e., probability of co-activation).  

- **Method**:  
  After intra-card load balancing via **EPLB**, we adjust **card placement** based on the affinity between:  
  - The expert with the highest load on one GPU, and  
  - Other experts across different GPUs.  

- **Impact**:  
  - Reduces **cross-node communication**.  
  - Achieves an additional **~5% performance improvement** over standard EPLB.  
  - However, balancing time increases from **2–3s → ~5s**, leading to more noticeable service latency.  

- **Next Step**:  
  To mitigate this, we introduce an **asynchronous dynamic load adjustment** method.  

---

##### Asynchronous Dynamic Load Adjustment
- **Key Design**:  
  Decouples **load balancing computation** from **model inference**, allowing both to proceed in parallel without blocking service.  

- **Execution Phase**:  
  During expert migration/transfer (post-calculation), a **hierarchical transfer strategy** is used to minimize inference impact, achieving seamless load balancing from the user’s perspective.  

- **Results**:  
  - Matches or surpasses the performance of static/original EPLB across datasets.  
  - Maintains **load balance ratio > 70%** consistently.

#### Computation

##### FlashMLA-FP8

###### Overview
- Executes end-to-end FP8 attention on Hopper (`SM90`).
- Uses **`TMA`** for global-to-shared transfers and **`WGMMA`** for matrix math.  
- Two warpgroups cooperatively pipeline `QK^T` and `PV` across KV blocks.  
- Minimizes shared-memory pressure and maximizes overlap between memory movement and compute.

###### Improvements
- **VS `bf16` FlashMLA**: ~70% performance improvement
  - Use `WGMMA FP8`.
  - Use `FP8` dtypes for `Q` and `KV`.
  - Saved shared memory and reallocated it for `sP1`.
  - Removed `retrieve_rP_for_SP(sQ(8))`.
  - Removed RS `WGMMA` of `rQ(8) * sK`.

- **VS previous FP8 implementation (#54)**: ~5% performance improvement
  - Fine-grained optimization of pipeline between `TMA` copy and `WGMMA`.
  - Rebuilt pipeline for transposed `V` using 4 named barriers to switch between 4 buffers (`V0L`, `V0R`, `V1L`, `V1R`).
  - Added ping-pong shared memory buffers: `sP0`, `sP1`, `sVt0`, `sVt1`.
  - Used 128-bit `STSM` and `LDSM` for mutual copying between `rP` and `sP`, resolving FP8 `WGMMA` layout mismatch.
  - Fine-grained `Q@K` tiling (`576/64 = 9` tiles) enabled computing `ROPE` in `BF16`, fixing previous accuracy issues.
  - Improved feature set and aligned with `SM90` programming style.

###### Summary
- Implements **end-to-end FP8 activations and weights** with `FP8 WGMMA`.  
- Decouples and interleaves three challenging stages into `TMA` wait windows:
  1. `V` transposition  
  2. `rP–sP` round-trip  
  3. Fine-grained (64-wide) `QK^T` tiling  
- Uses **ping-pong shared memory** and **named barriers** to maximize compute-memory overlap.  
- Closely follows the `SM90` asynchronous **`TMA + WGMMA`** model.  
- Lays foundation for scaling parallelism and deploying mixed-precision policies.

---

##### DeepGEMM swapAB

###### Overview
- Designed to address PTX instruction constraints:  
  - `N` must be a multiple of 8.  
  - `M` is fixed at 64 in instruction  
    ```
    wgmma.mma_async.sync.aligned.m64n(8i)k32.f16.e4m3.e4m3
    ```
- The **`swapAB`** variant swaps `WGMMA` tile usage:  
  - Maps problem’s `M` dimension onto `WGMMA`’s `N` dimension.  
  - Enables smaller `BLOCK_M (32)`.  
- Performance gain comes from **finer tiling granularity** and **better resource utilization**, not higher per-instruction throughput.

###### Performance Improvement: 10% ~ 70%
1. **Higher M-side boundary efficiency**
   - Baseline: `BLOCK_M` must align to 64, causing inefficiency when `M` is not multiple of 64.  
   - `swapAB`: Allows `BLOCK_M = 32` (or aligned to `8, 16, 24, 32`).  
   - Example: `M = 96`  
     - `BLOCK_M = 64` → last tile 50% efficient.  
     - `BLOCK_M = 32` → no waste (3×32 = 96).  

2. **Better load balance for small/irregular M**
   - Useful in grouped GEMM (e.g., `MoE` with inter-group variation in `M`).  
   - Smaller tiles → more uniform tasks.  
   - Persistent scheduler fills `SM`s more effectively and reduces stragglers.

3. **Improved occupancy and concurrency**
   - A-side SMEM footprint ≈ `BLOCK_M × BLOCK_K × bytes`.  
   - Reducing `BLOCK_M` from 64 → 32 halves A-side SMEM and register usage.  
   - Enables more concurrent `CTA`s, better latency hiding.  

4. **More efficient shared-memory access and write-back**
   - Vectorized A’s scale loads as `float2`.  
   - Uses `STSM_T` + new swizzle for `D` write-back.  
   - Reduces SMEM bank conflicts and irregularity overhead.

5. **Easier multicast/data reuse**
   - Smaller tiles → more concurrent `CTA`s per K-slice.  
   - Improves multicast opportunities and reduces `TMA` read traffic.

###### Use Cases & Expectations
- **Best suited for**:
  - `M` not multiple of 64.
  - Small `M` values.
  - Large variation across groups.  
  - Yields better boundary efficiency, concurrency, and load balance.

- **Less advantage when**:
  - `M` is large and well-aligned (e.g., square matrices).  

- **Key point**:
  - Instruction-level peak unchanged.  
  - Gains come from **better tiling → higher effective utilization, concurrency, and memory efficiency**.

### Overlap: SBO（Single-batch-overlap）
#### Motivation
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

#### Designs
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