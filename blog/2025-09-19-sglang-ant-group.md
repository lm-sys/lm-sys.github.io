---
title: "Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G"
author: "Tianyu Zhang, Peng Zhang, Yusong Gao, Yun Zhang, Yongfei Xu, Zhe Wang, Qianyu Zhang, Chun Huang, Xi Chen, Fakang Wang, Jianhao Fu"
date: "September 19, 2025"
previewImg: /images/blog/hicache/hicache_overview.png
---

# Introduction
必选关键字：  
1. 规模：数万卡级别H20集群；  
2. SLA：延迟、成本、稳定性（optional）、可观测（optional）。  

重点：我们在数万卡规模的H20集群下，针对不同用户的延迟要求，基于SGLang，构建出了业界吞吐最高、成本最低的SOTA方案。  

# Challenges With H20
1. **H20 固有限制**：与H800相比，算力极差，但内存带宽高、NVlink带宽高、显存容量大，RDMA网卡带宽差  

| 硬件参数       | H20-96G   | H800-80G   |
| -------------- | --------- | ---------- |
| FP8算力        | 296 TFLOPS | 1979 TFLOPS |
| FP16/BF16算力  | 148 TFLOPS | 989 TFLOPS  |
| 显存容量       | 96 GB      | 80 GB       |
| 内存带宽       | 4000 GB/s  | 3352 GB/s   |
| NVlink带宽     | 900 GB/s   | 400 GB/s    |
| RDMA网卡带宽   | 4*400 Gb/s | 8*400 Gb/s  |

2. **业界无可参考方案**：我们从5月份开始探索EP并行方案，DeepSeek社区和SGLang社区的开源方案均围绕高算力卡（H800），业界缺少H20这类低算力卡的EP并行方案；  

**总结**：在低算力+无可参考方案的情况下，满足与高算力卡相同的SLA，同时尽可能降低成本。  

# Our Solution: Make H20 Great for Inference in Real World

## Deployment
我们采用业界通用的PD分离方案，但是由于H20卡的特性，我们的PD分离部署方式与DeepSeek团队和其他团队不同：  

### Prefill
单机TP8，原因：  
1. **SLA 约束**：Prefill阶段是计算密集的，腾讯、字节（其他厂商？）在H20上给出的方案均是跨机DP+EP，但是在我们的测试中，Prefill实例采用DP+EP部署会使得TTFT变得很长，无法满足用户的TTFT需求（比如TTFT < 1s）。  
2. **动态扩缩容**：在考虑KVCache的情况下，我们希望线上Prefill具备动态扩缩容的能力，即在KVCache命中率高、Prefill压力小的情况下缩容，在KVCache命中率低、Prefill压力大的情况下动态扩容，如果使用多机DP+EP的方案部署，扩缩容策略会变得复杂。  

### Decode
DP16+EP16（小EP），原因：  
1. **故障半径**：Decode或者单卡故障的故障半径尽可能小，EP的高可用方案PR仍在draft中；  
2. **H20特性**：  
   1. 算力低：在线上服务的延迟约束下，batch-size无法打到很高；  
   2. 显存大：基于（1），H20不存在显存bound的问题，同时显存可以基于KVCache FP8量化做进一步的优化；  
   3. NVLink带宽高：DeepEP支持了NVLink，而H20的NVLink的带宽比H800要高一倍多，在MoE阶段，理论上会有50%的通信落在NVLink上。  

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

## Overlap - SBO（Single-batch-overlap）
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


## Observability - Lightweight Anomaly Diagnosis for Distributed MoE Model Deployment

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

# Furhter Works
MTP  
1. Cuda Graph  
2. TP1  

Ultra (0.5 s, 10 ms)  