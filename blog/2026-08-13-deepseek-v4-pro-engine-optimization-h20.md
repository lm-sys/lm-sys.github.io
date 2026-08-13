---
title: "Pushing the Limits of Serving DeepSeek-V4-Pro"
author: "Tianyu Zhang, Yusong Gao, Yun Zhang"
date: "August 13, 2026"
previewImg: /images/blog/deepseek_v4/00_cover.png
type: blog
---

## 1. Introduction

DeepSeek-V4-Pro is a 1.6-trillion-parameter Mixture-of-Experts (MoE) model released with both FP8 and FP4 weights. Models at this scale naturally benefit from accelerators such as NVIDIA Blackwell GPUs, which offer more HBM, higher compute throughput, and native FP4 Tensor Cores. Yet H20 GPUs remain widely deployed, despite lacking those advantages.

Hardware constraints do not relax serving requirements. Long-context prefill must still control time to first token (TTFT). Interactive decode must satisfy the time-per-output-token (TPOT) target of each service tier. Sustained traffic must balance aggregate throughput against KV-cache capacity. Short inputs, long contexts, latency-sensitive requests, and high concurrency stress the system in different ways; no universal configuration can serve all of them well.

**One model needs multiple serving profiles.** Workload characteristics, service-level objectives (SLOs), and measured hardware behavior jointly inform the deployment topology and execution path:

- **Match serving profiles to the workload.** In the configurations evaluated here, prefill selects between PP2 and PP4 based on the measured context-length range, while decode uses profiles optimized for different latency, throughput, and KV-capacity targets.
- **Optimize the prefill path.** We optimize `Attention-CP8 → MoE-TP8` and context-parallel communication, then tune for the real routing shapes produced by long- and short-context workloads.
- **Optimize the decode path.** We optimize the DSpark speculative-decoding path, refine execution, expert routing, and communication–computation overlap for distinct decode SLOs.

**Push the latency frontier.** At batch size 1, the single-node H20-141GB reference reaches **271 output tokens/s**, compared with the **383.7 tokens/s** [reported on B300](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/). Despite the substantial hardware gap, workload-specific system optimization narrows the observed decode performance ratio to **1.42×**.

**Cover the serving envelope.** The latency result represents only one edge of the system. Across the broader profile family, optimized prefill reaches **8.45k input tokens/s per node** and processes a **1M-token prompt in 43.7 seconds**. For throughput-oriented decode, the DP16-EP16 efficiency reference reaches **4.67k output tokens/s per node**, corresponding to an average TPOT of **27.4 ms**. These results intentionally come from different profiles, each selected and optimized for a different combination of context length, latency, throughput, and capacity constraints.

**The contribution is a methodology, not a single benchmark.** Scenario-specific serving allows each workload to move toward a better measured operating point among the profiles evaluated on the available hardware. We hope the deployment choices, optimization methods, and measurements presented here provide a practical reference for teams serving frontier models under compute, memory, bandwidth, or interconnect constraints.

## 2. From Hardware Constraints to Serving Profiles

### 2.1 Hardware Constraints and Serving Roles

<img src="/images/blog/deepseek_v4/01_hardware_gap.svg" alt="Hardware specification comparison across H20-96GB, H20-141GB, and B300, covering FP4 and FP8 compute, HBM capacity, memory bandwidth, NVLink, and RDMA" style="display:block; margin:auto; width:100%; max-width:640px; height:auto;"></img>

*Figure 1. Hardware Gap: H20 vs. B300.*

**Blackwell offers raw performance; H20 offers deployable scale.** B300 provides native FP4 Tensor Cores, much higher FP8 throughput, and substantially more HBM. H20 cannot match its compute capability, but it remains available at scale and provides high memory bandwidth and 900 GB/s NVLink. Each node in this study contains eight GPUs connected by NVLink. Prefill does not retain long-lived per-request state, so its hardware choice is governed primarily by TTFT, compute, and communication efficiency. Decode must retain the KV cache of every active request throughout generation, making HBM capacity a direct limit on context length and concurrency. For the deployment studied here, this led us to use H20-141GB for decode and H20-96GB—whose capacity was sufficient for our prefill workloads—for prefill.

<img src="/images/blog/deepseek_v4/02_h20_role_assignment.svg" alt="Hardware assignment by serving role: H20-96GB serves TTFT-sensitive prefill with short-lived state, while H20-141GB serves KV-capacity-bound decode with persistent state" style="display:block; margin:auto; width:80%; max-width:100%; height:auto;"></img>

*Figure 2. Hardware Assignment by Serving Role.*

### 2.2 Capacity Choices

Serving capacity ultimately comes from a shared HBM budget: model weights and per-request KV state compete for the same memory. We define **full-token capacity** as the maximum number of full-attention KV tokens that each rank can hold after model weights and runtime buffers have been allocated. It is a memory ceiling rather than a direct guarantee of admissible batch size.

#### Reducing Weight Footprint with Humming MXFP4AFP8

**Reduce the weight footprint first.** [Humming MXFP4AFP8](https://github.com/inclusionAI/humming) uses MXFP4 expert weights with online FP8 activations to reduce weight footprint and memory traffic on H20 GPUs, which lack native FP4 Tensor Cores. The SGLang integration is available in [sglang#23754](https://github.com/sgl-project/sglang/pull/23754). We will cover the Humming/SGLang integration in a dedicated follow-up post.

#### Expanding KV Capacity with Online C128

**Give the KV cache room to grow.** The Offline C128 baseline retains per-index state for each compressed page. Online C128 instead maintains a compact aggregate state, releasing more HBM to the KV-cache pool. It introduces additional state maintenance and speculative-verification work, but we observed no TPOT regression in our tests.

#### Combined Capacity Gains

<img src="/images/blog/deepseek_v4/03_capacity_scaling.svg" alt="Two horizontal bar-chart panels show full-token capacity scaling for DP32-EP32 and PP2-TP8 from Baseline FP8 through Humming MXFP4AFP8 to Online C128" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 3. Capacity Scaling with Humming MXFP4AFP8 and Online C128.*

**Capacity gains compound across weights and KV state.** By reducing the weight footprint, Humming MXFP4AFP8 expands full-token capacity to **1.71×** the Baseline FP8 + Offline C128 configuration for DP32-EP32 and **4.47×** for PP2-TP8. Online C128 then reduces the C128 auxiliary-state footprint, providing another **2.268×** increase on top of Humming. Combined, the two techniques raise capacity to **3.88×** the baseline for DP32-EP32 and **10.14×** for PP2-TP8. Appendix D.1 provides the complete data.

### 2.3 Scenario-Specific Serving Profiles

#### Prefill Profiles

<img src="/images/blog/deepseek_v4/04_prefill_profiles.svg" alt="Two independent prefill deployment strategies: PP2 and PP4 use different layer partitions while every stage follows the same Attention-CP8 and MoE-TP8 execution path" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 4. Prefill Profiles: Same Execution Path, Different Pipeline Depth.*

**The right pipeline depth depends on how much work there is to pipeline.** PP2-CP8-TP8 and PP4-CP8-TP8 share the same `Attention-CP8 → MoE-TP8` execution path. At the topology level, their primary difference is pipeline depth: PP2 distributes the model across two stages, while PP4 uses four.

**Short contexts favor lower pipeline overhead; long contexts expose more parallelism.** Short inputs produce fewer chunks, leaving a deeper pipeline underfilled and making fill, drain, and cross-stage transfer costs more prominent. Long contexts provide enough chunks to keep four stages busy; with fewer layers per stage, the additional nodes translate into more prefill parallelism. In our deployment, these characteristics led us to use **PP2-CP8-TP8 for shorter contexts** and **PP4-CP8-TP8 for long-context workloads**.

#### Low-Latency Decode Profiles

<img src="/images/blog/deepseek_v4/05_low_latency_decode.svg" alt="Single-node TP8 is the dashed reference and PP2-TP8 is the two-node low-latency serving profile used in our deployment; both execute Attention-TP8 and MoE-TP8, each followed by its own AllReduce" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 5. Low-Latency Decode: TP8 Reference and PP2-TP8 Serving Profile.*

**Low latency starts with the shortest execution path.** Single-node TP8 and PP2-TP8 share the same `Attention-TP8 → MoE-TP8` execution path; the difference is whether the model is partitioned across nodes. Single-node TP8 places all layers on one H20-141GB node and avoids cross-stage communication and synchronization. PP2-TP8 partitions the model across two pipeline stages.

**The fastest topology is not always the most serviceable one.** Single-node TP8 has the shorter execution path, but model weights and serving state share the HBM of one node, leaving limited room for the KV cache. It cannot simultaneously support long contexts and larger batch sizes. PP2-TP8 pays additional pipeline overhead but distributes the model weights across two nodes, releasing more HBM for KV state. For our latency and capacity targets, we use **single-node TP8 as the batch-size-1 latency reference** and **PP2-TP8 as the low-latency serving profile**.

#### High-Throughput Decode Profiles

<img src="/images/blog/deepseek_v4/06_high_throughput_decode.svg" alt="High-throughput decode scales DP and EP ranks from the two-node DP16-EP16 reference to the four-node DP32-EP32 serving profile used in our deployment; every node participates in all layers while routed experts remain sharded across EP ranks" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 6. High-Throughput Decode: DP16-EP16 Reference and DP32-EP32 Capacity Profile.*

**High-throughput decode scales data and expert parallelism together.** Both profiles use the `Attention-DP → MoE-EP` execution path. DP16-EP16 is the smallest deployment unit; DP32-EP32 expands both DP and EP within the same topology.

**Scale-out prioritizes request capacity over per-GPU throughput.** A larger EP group distributes expert weights across more GPUs, releasing HBM for the KV cache and admitting more concurrent requests. At the same time, a smaller fraction of MoE traffic remains within each node, while a larger fraction crosses nodes, which can reduce per-GPU efficiency. In the profiles evaluated here, we use DP16-EP16 as the smallest deployment unit and efficiency reference, and DP32-EP32 to expand request capacity.

## 3. Prefill: Balancing Compute and Communication

**Prefill performance is a system problem.** Expert imbalance, context-parallel communication, and production routing shapes jointly determine TTFT; optimizing an isolated kernel is not enough.

### 3.1 Why MoE-TP Instead of MoE-EP

<img src="/images/blog/deepseek_v4/07_cp_fused_moe_mechanism.svg" alt="Replacing MoE-EP with MoE-TP in the prefill path" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 7. Replacing MoE-EP with MoE-TP.*

**Less traffic can still take longer.** MoE-EP exchanges only routed tokens, but real prefill traffic exhibits significant expert skew. Ranks that own hot experts perform more computation and become stragglers; all other ranks wait for the slowest path at the combine step. Lower communication volume does not translate into lower TTFT.

**Balance compute before minimizing traffic.** For the H20 prefill workloads evaluated here, both PP2 and PP4 use MoE-TP. Full-sequence all-gather and reduce-scatter introduce more communication, but the traffic remains on high-bandwidth NVLink and has stable, predictable cost. All TP ranks execute tensor-parallel computation over the same routed tokens, preventing expert skew from becoming a rank-level long tail. For this workload, **predictable communication is cheaper than unpredictable imbalance**. The implementation is available in [sglang#24947](https://github.com/sgl-project/sglang/pull/24947).

### 3.2 Accelerating and Fusing Prefill Collectives

<img src="/images/blog/deepseek_v4/08_collective_communication_optimization.svg" alt="Symmetric-memory collectives provide a reusable foundation for TP and CP, while fused Prefill kernels collapse the communication-heavy critical path" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 8. Symmetric-Memory Collectives and Prefill Fusion.*

**Build a reusable collective fast path.** MoE-TP replaces unpredictable expert imbalance with predictable collective traffic, making communication efficiency the next bottleneck. We made symmetric memory reusable across TP and CP, allowing AllReduce, AllGather, and ReduceScatter to share registered-buffer fast paths and applicable Hopper acceleration. The supporting upstream work spans [memory-pool ownership](https://github.com/sgl-project/sglang/pull/21392), [communicator registration](https://github.com/sgl-project/sglang/pull/19329), [MoE-TP collective buffers](https://github.com/sgl-project/sglang/pull/29007), and the [CP Attention](https://github.com/sgl-project/sglang/pull/17756) and [KV-cache](https://github.com/sgl-project/sglang/pull/24040) buffer paths.

**Then shorten the Prefill critical path.** Faster collectives alone do not remove the boundaries between communication and computation. For the 32K single-chunk case, we built a fused path that overlaps a copy-engine-driven AllGather with fused FP8 quantization and shared-expert GEMM, then combines TopK reduction, shared-expert addition, and ReduceScatter in a second Triton kernel. This reorganizes seven operators into three execution groups and reduces TTFT by approximately **3.5%** in a matched PP4 A/B.

### 3.3 Tuning Humming for Real Routing Shapes

<img src="/images/blog/deepseek_v4/09_humming_exact_shape_workflow.svg" alt="Humming prefill workflow from routing capture through separate W13 and W2 tuning to staged validation" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 9. Tuning Humming for Real Routing Shapes.*

**Generic tuning misses the shapes that matter.** Prefill routing distributes tokens unevenly across 384 experts, so the effective `M` dimension clusters into a small set of discrete values. W13 and W2 also operate on different shapes, so a single generic heuristic cannot optimize both paths.

**Tune from production routing.** We extract high-frequency shapes from real routing histograms, build separate exact-shape configurations for W13 and W2, and validate them at the kernel, pipeline-stage, and matched A/B levels. The optimization target is not a synthetic range of `M`, but **the routing distribution we actually serve**. In a matched PP4 A/B at 32K, selected MoE kernel latency falls by approximately **21%**, translating into an **11.35%** end-to-end TTFT reduction.

## 4. Decode: Optimizing Speculation and MoE Execution

**Decode optimization is profile-specific in our implementation.** PP2-TP8 requires coordination across speculative pipeline stages, while DP32-EP32 focuses on optimizing the refinement step and expert routing at high concurrency. Humming fusion and overlap improve the shared MoE hot path beneath these serving topologies.

### 4.1 Low-Latency PP2-TP8: Extending DSpark Across Pipeline Stages

<img src="/images/blog/deepseek_v4/10_pp2_tp8_dspark_execution.svg" alt="PP2-TP8 DSpark execution coordinated across two pipeline stages, with target hidden states sent to Stage 1 and accepted tokens and next candidates returned under a shared stage-tick protocol" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 10. Coordinating DSpark Across PP2 Stages.*

**Pipeline parallelism splits the speculative loop.** In PP2-TP8, target execution spans two pipeline stages, while the [DSpark](https://github.com/sgl-project/sglang/pull/30261) drafter resides only on the final stage. Stage 0 sends target hidden states to Stage 1, which performs verification, accepts tokens, and generates candidates for the next round.

**Make two stages advance as one.** Every speculative round crosses the pipeline boundary. We coordinate both stages and the required intermediate transfers under one execution protocol, preventing the stages from entering different rounds while avoiding redundant synchronization. The PP-specific DSpark integration is being upstreamed in [sglang#32281](https://github.com/sgl-project/sglang/pull/32281).

### 4.2 High-Throughput DP32-EP32: Removing High-Concurrency Bottlenecks

<img src="/images/blog/deepseek_v4/11_dp32_ep32_bottleneck_removal.svg" alt="DP32-EP32 bottleneck removal: single-chunk transposed GEMM replaces row-wise full-vocabulary dot-reduce, while a routing-affinity snapshot guides EPLB placement and redundant experts" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 11. DP32-EP32 Bottleneck Removal.*

The matched A/B results in this subsection use DP32-EP32 at 4K with 32 concurrent requests per DP rank.

**Choose the right execution shape for refinement.** The refinement step applies a full-vocabulary projection to rescore DSpark's candidate set. At high concurrency, the row-wise dot-reduce repeatedly reads the vocabulary weights for every active row, creating a persistent tail in each decode step. We combine active rows into one transposed GEMM, reducing redundant memory traffic and shortening the refinement path. Per-GPU throughput improves by **22.8%**.

**Place experts from measured routing.** DSpark traffic also exhibits significant expert skew. We record routing affinity from representative requests and use it to configure expert-parallel load balancing (EPLB) and redundant experts, preventing a small number of hot experts from repeatedly extending the critical path. Per-GPU throughput improves by **13.5%**.

### 4.3 Humming Decode Hot Path: Fusion and Overlap

<img src="/images/blog/deepseek_v4/12_humming_decode_hot_path_optimizations.svg" alt="Two side-by-side Humming decode optimizations: quantized hot-path fusion removes intermediate buffering before W2, while Humming-Aware SBO overlaps per-tile W2 completion with DeepEP combine sends" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 12. Humming Decode Hot-Path Optimizations.*

These optimizations sit below the serving topology and can be reused by Humming-based decode profiles. The matched results below use DP32-EP32 at 4K with 32 concurrent requests per DP rank.

**Remove the extra quantization pass.** We fuse the SwiGLU activation with quantization so that the fused kernel directly produces the data and scale required by W2. This eliminates repeated access to an intermediate buffer and removes the standalone quantization pass, allowing W2 to start earlier. In the matched DSpark A/B, per-GPU throughput improves by **44.0%**.

**Overlap communication with W2.** We adapt the [Single-Batch Overlap (SBO)](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/#sbo-single-batch-overlap) mechanism from our previous work ([sglang#9660](https://github.com/sgl-project/sglang/pull/9660)) into **Humming-Aware SBO**. Per-tile signals allow DeepEP to begin the corresponding combine send as soon as a W2 output tile completes, without waiting for the entire GEMM. In an earlier matched non-spec A/B at the same operating point, SBO recovers **4.12%** throughput relative to the FP8-transport tier.

## 5. Evaluation: System Gains and Profile Trade-offs

### 5.1 Prefill: Cumulative Gains and Context-Length Trade-offs

<img src="/images/blog/deepseek_v4/13_humming_prefill_throughput_uplift.svg" alt="Baseline and final prefill throughput for PP2-CP8-TP8 and PP4-CP8-TP8 across input lengths from 4K to 1M" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 13. Cumulative Prefill Throughput Gains.*

**PP2 strengthens the short-context profile.** PP2 improves at all nine input lengths, with a geometric-mean throughput gain of **36.5%** and a peak total input throughput of **16,900 tokens/s**. Its shallower pipeline reduces fill-and-drain overhead for short requests, allowing PP2 to maintain lower TTFT with fewer resources.

**PP4 carries the gains into long context.** PP4 delivers a geometric-mean throughput gain of **31.8%** across the same nine points. As context length grows, the deeper pipeline has enough work to amortize its fixed cost: total input throughput reaches **25,860 tokens/s** at 512K and remains **23,970 tokens/s** at 1M.

<img src="/images/blog/deepseek_v4/14_prefill_ttft_crossover.svg" alt="TTFT trade-off between the evaluated PP2-CP8-TP8 and PP4-CP8-TP8 profiles across context lengths" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 14. TTFT Trade-off Between PP2 and PP4.*

**Context length shifts the PP2/PP4 trade-off.** Relative to PP4, PP2 lowers TTFT by **16.7%** at 4K and **19.5%** at 32K. The two profiles remain within **2%** at 8K, 16K, and 64K. PP4 establishes a decisive advantage from 128K onward, reducing TTFT relative to PP2 by **26.2%**, **33.3%**, **42.1%**, and **44.8%** at 128K, 256K, 512K, and 1M, respectively. We therefore treat the routing boundary as an operating policy derived from the measured context-length range rather than a universal crossover point.

Appendix A.1–A.2 provide the complete TTFT and total-input-throughput results.

### 5.2 Low-Latency Decode: Performance and Capacity Trade-offs

<img src="/images/blog/deepseek_v4/15_decode_optimized_dspark_tpot.svg" alt="Four grouped bar charts compare No-Spec baseline and Optimized DSpark peak TPOT across batch sizes at 8K, 64K, 256K, and 1M input lengths" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 15. Peak TPOT Gains from Optimized DSpark.*

**Optimized DSpark resets the latency baseline.** Across the four input lengths shown in Figure 15, Optimized DSpark reduces peak TPOT by **74.8%–78.0%** at batch size 1. At the largest batch size shared by each pair of measurements, the reduction remains **52.2%–60.0%**. The gain holds from 8K through 1M rather than being confined to short contexts or single-request execution.

<img src="/images/blog/deepseek_v4/16_decode_bs1_throughput.svg" alt="Batch-size-1 throughput across four input lengths for No-Spec PP2-TP8, Optimized DSpark PP2-TP8, and single-node TP8 on H20-141GB, with 383.7 tokens per second on B300 shown as a separate external reference" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 16. Batch-Size-1 Decode Throughput: H20-141GB and B300 Reference.*

**Observed serving performance is much closer than peak-compute ratios alone suggest.** Across the four input lengths shown in Figure 16, Optimized DSpark on PP2-TP8 reaches **150–174 tokens/s** at batch size 1. The single-node TP8 reference reaches **183–271 tokens/s**. For the precisions used by the actual execution paths, B300 has approximately **45.6×** the peak Tensor Core compute of H20-141GB (B300 FP4 versus H20 FP8) and **1.67×** its memory bandwidth. Yet the highest observed generation rates are **383.7 tokens/s** [on B300](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) and **271 tokens/s on H20-141GB**, respectively—a ratio of **1.42×**. Even against this much stronger hardware reference, workload-specific optimization brings the H20-141GB reference substantially closer in observed serving performance.

**Capacity favors PP2-TP8 for our production targets.** Single-node TP8 is faster, but at a 1M context it has enough KV-cache capacity only for batch size 1. It cannot admit a larger batch or more concurrent requests. By distributing model weights across two pipeline stages, PP2-TP8 supports batch sizes 4, 8, and 16 at 1M, 512K, and 256K, respectively. With Online C128, its full-token capacity reaches **11.04M tokens/rank**. For context-length and concurrency targets similar to ours, we recommend retaining single-node TP8 as the latency reference and using **PP2-TP8 as the low-latency serving profile**. Appendix B and Appendix D.1 provide the complete performance and capacity data.

### 5.3 High-Throughput Decode: Frontier Gains and Profile Trade-offs

<img src="/images/blog/deepseek_v4/17_decode_high_throughput_pareto.svg" alt="Throughput-interactivity Pareto frontiers at 4K, 32K, 128K, and 1M compare FP8 MTP, optimized MTP, FP8 DSpark, and Humming MXFP4AFP8 with Online C128 and DSpark" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 17. Throughput–Interactivity Pareto Frontiers.*

Figure 17 shows how the throughput–interactivity frontier evolves with the system. The horizontal axis is interactivity in tokens/s/user, and the vertical axis is throughput in tokens/s/GPU. In these DP/EP profiles, each DP rank maps to one GPU; interactivity is per-GPU throughput divided by the number of concurrent requests per DP rank. Points farther toward the upper right provide a better combination of user-visible generation speed and GPU efficiency. The four curves represent cumulative system evolution rather than the isolated gain of any optimization in Section 4.

MTP denotes multi-token prediction; the `(3, 1, 4)` configuration uses three speculative steps, top-k 1, and four draft tokens.

**System optimization moves the entire frontier.** At 4K with 32 concurrent requests per DP rank, per-GPU throughput rises from **319.92 tokens/s/GPU** to **703.15 tokens/s/GPU**, a **2.20×** increase. At 1M with one request per DP rank, it rises from **27.05 tokens/s/GPU** to **66.82 tokens/s/GPU**. The first three system milestones can each process only one request per DP rank at 1M; the final system supports four and reaches **177.48 tokens/s/GPU**. The expanded operating envelope comes from both faster execution and greater capacity.

<img src="/images/blog/deepseek_v4/18_dp16_dp32_throughput.svg" alt="Two grouped bar charts compare DP16-EP16 and DP32-EP32 throughput per GPU across input lengths with 16 and 32 concurrent requests per DP rank" style="display:block; margin:auto; width:96%; max-width:100%; height:auto;"></img>

*Figure 18. Per-GPU Throughput: DP16-EP16 vs. DP32-EP32.*

**Smaller deployment units preserve efficiency at selected high-concurrency operating points.** In our [earlier work on serving DeepSeek-V3/R1 on H20](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/#investigation-for-ep-size), we found that a smaller EP deployment unit can keep a larger fraction of MoE traffic within each node. DeepSeek-V4-Pro shows the same advantage at the operating points plotted in Figure 18: with 16 and 32 concurrent requests per DP rank, DP16-EP16 delivers approximately **3.6%–20%** higher per-GPU throughput than DP32-EP32. The full sweep is not monotonic across every concurrency level, so we use DP16-EP16 as an efficiency reference rather than a universal replacement for DP32-EP32.

<img src="/images/blog/deepseek_v4/19_dp16_dp32_capacity.svg" alt="A compact table compares the maximum valid request capacity per DP rank for DP16-EP16 and DP32-EP32 at 256K, 512K, and 1M input lengths" style="display:block; margin:auto; width:100%; max-width:640px; height:auto;"></img>

*Figure 19. Long-Context Request Capacity per DP Rank.*

**Capacity shifts the preferred high-throughput profile.** DP16-EP16 is more efficient per GPU, but DP32-EP32 distributes expert weights across more ranks and releases additional HBM for the KV cache. At 256K, 512K, and 1M, the maximum concurrent requests per DP rank increase from **8, 4, and 2** to **16, 8, and 4**, respectively—a consistent **2×** expansion. For deployments with long-context concurrency targets similar to ours, this additional capacity favors **DP32-EP32 as the capacity-oriented high-throughput profile**, while DP16-EP16 remains useful as an efficiency reference. Appendix C and Appendix D.1 provide the complete data.

## 6. Conclusion

**One model does not require one compromise profile.** We built a scenario-specific serving stack for DeepSeek-V4-Pro on H20. Prefill switches between PP2 and PP4 according to context length. Decode uses PP2-TP8 for low latency and DP32-EP32 for high throughput. By co-designing capacity, deployment topology, and execution path, H20 can sustain 1M-token contexts and meet multiple serving SLOs despite limited compute and the absence of native FP4 Tensor Cores.

**The transferable result is a scenario-driven methodology.** Serving profiles should not be selected from hardware specifications or isolated benchmarks alone. We recommend starting from the workload, SLO, context length, and concurrency, then using profiling to identify the binding resource and translate it into concrete topology and execution-path decisions. We hope this methodology helps AI infrastructure teams build practical frontier-model serving systems under diverse resource constraints—whether the bottleneck is compute, memory capacity, memory bandwidth, or interconnect—and share those lessons with the broader open-source ecosystem.

## Acknowledgements

We would like to thank the **SGLang Team and Community** for their outstanding work on the SGLang framework. We also thank the following teams and collaborators for their support and contributions:

- **Ant Group SCT Team:** Yongfei Xu, Qianyu Zhang, Zekai Gu, ZhiLin Huang, Fakang Wang, Jianhao Fu, Zhuoxuan Du, Xia Zhan, Chun Huang, Qi Liu, Xi Chen, Yuhan Mao, Peipeng Cheng, Hanlin Gao, Jinghua Yao
- **Ant Group Venus Team:** Jinzhen Lin
- **SGLang Community:** Peng Zhang

## Appendix A. Prefill Results

### A.1 Humming PP2 Prefill: Baseline vs. Final Profile

| Input Length | Baseline TTFT (ms) | Baseline Total Input Throughput (tokens/s) | Final TTFT (ms) | Final Total Input Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| 4K | 775.8 | 5,280 | 573.3 | 7,140 |
| 8K | 1202.1 | 6,810 | 907.6 | 9,030 |
| 16K | 2059.8 | 7,950 | 1649.5 | 9,930 |
| 32K | 4137.5 | 7,920 | 2470.3 | 13,260 |
| 64K | 6195.7 | 10,580 | 4063.8 | 16,130 |
| 128K | 10744.4 | 12,200 | 7975.9 | 16,430 |
| 256K | 20542.2 | 12,760 | 15507.2 | 16,900 |
| 512K | 44544.6 | 11,770 | 34982.6 | 14,990 |
| 1M | 100304.2 | 10,450 | 79214.2 | 13,240 |

### A.2 Humming PP4 Prefill: Baseline vs. Final Profile

| Input Length | Baseline TTFT (ms) | Baseline Total Input Throughput (tokens/s) | Final TTFT (ms) | Final Total Input Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| 4K | 924.6 | 4,430 | 687.9 | 5,950 |
| 8K | 1174.5 | 6,970 | 890.3 | 9,200 |
| 16K | 2202.0 | 7,440 | 1635.4 | 10,020 |
| 32K | 4185.6 | 7,830 | 3068.4 | 10,680 |
| 64K | 5252.4 | 12,480 | 3982.6 | 16,460 |
| 128K | 7793.4 | 16,820 | 5882.5 | 22,280 |
| 256K | 13210.7 | 19,840 | 10348.9 | 25,330 |
| 512K | 26350.1 | 19,900 | 20273.1 | 25,860 |
| 1M | 55532.3 | 18,880 | 43742.5 | 23,970 |

## Appendix B. Low-Latency Decode Results

### B.1 Peak TPOT Across Input Lengths and Batch Sizes

#### B.1.1 No-Spec PP2-TP8

| Input Length / Batch Size (Peak TPOT, ms) | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 8K | 26.39 | 30.86 | 31.31 | 31.79 | 31.74 |
| 32K | 25.72 | 26.58 | 27.81 | 31.06 | 37.97 |
| 64K | 25.75 | 26.62 | 28.13 | 29.19 | 38.75 |
| 128K | 25.94 | 26.94 | 28.38 | 29.75 | 38.51 |
| 256K | 26.08 | 27.21 | 28.84 | 32.43 | 38.83 |
| 512K | 26.25 | 27.51 | 29.16 | 33.70 | - |
| 1M | 26.42 | 27.81 | 29.52 | - | - |

#### B.1.2 Optimized DSpark PP2-TP8

| Input Length / Batch Size (Peak TPOT, ms) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 5.91 | 6.76 | 7.97 | 10.00 | 14.55 | 19.23 |
| 8K | 5.80 | 6.87 | 8.85 | 10.48 | 15.18 | 19.60 |
| 32K | 6.14 | 7.04 | 8.39 | 10.83 | 14.86 | 20.46 |
| 64K | 6.15 | 7.13 | 8.73 | 10.39 | 15.49 | 21.65 |
| 128K | 6.77 | 7.02 | 8.91 | 11.59 | 16.17 | 24.78 |
| 256K | 5.76 | 6.98 | 8.61 | 11.98 | 17.72 | - |
| 512K | 6.35 | 7.95 | 9.87 | 14.30 | - | - |
| 1M | 6.65 | 8.92 | 12.43 | - | - | - |

### B.2 Batch-Size-1 Output Throughput

| Input Length | No-Spec PP2-TP8 (tokens/s) | Optimized DSpark PP2-TP8 (tokens/s) | Single-Node TP8 (tokens/s) |
|---|---:|---:|---:|
| 4K | - | 169 | 213 |
| 8K | 38 | 172 | 260 |
| 16K | - | - | 244 |
| 32K | 39 | 163 | 269 |
| 64K | 39 | 163 | 246 |
| 128K | 39 | 148 | 267 |
| 256K | 38 | 174 | 271 |
| 512K | 38 | 157 | 254 |
| 1M | 38 | 150 | 183 |

## Appendix C. High-Throughput Decode Results

### C.1 DP32-EP32 with FP8 + MTP (3, 1, 4)

| Input Length / Concurrent Requests per DP Rank (tokens/s/GPU) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 30.49 | 58.58 | 102.89 | 174.75 | 253.15 | 319.92 |
| 8K | 30.34 | 58.29 | 102.38 | 174.67 | 251.62 | 318.32 |
| 16K | 29.70 | 56.55 | 99.47 | 170.01 | 242.22 | 302.43 |
| 32K | 29.58 | 56.35 | 98.28 | 164.26 | 234.13 | - |
| 64K | 29.07 | 55.73 | 96.43 | 161.60 | - | - |
| 128K | 28.39 | 54.06 | 92.89 | 153.55 | - | - |
| 256K | 28.35 | 53.02 | 90.89 | - | - | - |
| 512K | 27.51 | 51.49 | - | - | - | - |
| 1M | 27.05 | - | - | - | - | - |

### C.2 DP32-EP32 with FP8 + Optimized MTP (3, 1, 4)

| Input Length / Concurrent Requests per DP Rank (tokens/s/GPU) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 36.84 | 69.86 | 131.96 | 232.94 | 389.94 | 514.77 |
| 8K | 32.58 | 69.51 | 131.53 | 222.06 | 348.80 | 416.82 |
| 16K | 31.89 | 67.44 | 127.79 | 216.14 | 341.85 | 395.99 |
| 32K | 31.49 | 67.21 | 124.49 | 208.83 | 337.97 | - |
| 64K | 30.95 | 66.47 | 123.68 | 205.44 | - | - |
| 128K | 30.22 | 64.47 | 119.14 | - | - | - |
| 256K | 30.18 | 63.23 | - | - | - | - |
| 512K | 29.28 | 61.40 | - | - | - | - |
| 1M | 28.79 | - | - | - | - | - |

### C.3 DP32-EP32 with FP8 + DSpark

| Input Length / Concurrent Requests per DP Rank (tokens/s/GPU) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 53.1 | 94.8 | 181.2 | 338.1 | 495.8 | 591.8 |
| 8K | 44.5 | 88.4 | 170.1 | 317.3 | 495.5 | - |
| 16K | 43.6 | 88.3 | 165.3 | 308.8 | 455.5 | - |
| 32K | 43.0 | 87.3 | 161.0 | 298.4 | - | - |
| 64K | 42.3 | 86.3 | 158.0 | - | - | - |
| 128K | 41.3 | 83.8 | - | - | - | - |
| 256K | 41.2 | - | - | - | - | - |
| 512K | 40.0 | - | - | - | - | - |
| 1M | 39.3 | - | - | - | - | - |

### C.4 DP32-EP32 with Humming MXFP4AFP8 + Online C128 + DSpark

| Input Length / Concurrent Requests per DP Rank (tokens/s/GPU) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 75.32 | 127.10 | 235.85 | 417.53 | 564.08 | 703.15 |
| 8K | 75.60 | 128.29 | 238.01 | 417.34 | 560.68 | 709.64 |
| 16K | 74.00 | 124.47 | 231.25 | 406.21 | 539.72 | 674.19 |
| 32K | 73.07 | 122.27 | 225.28 | 392.47 | 521.70 | 601.67 |
| 64K | 71.81 | 120.92 | 221.05 | 386.11 | 516.54 | 599.63 |
| 128K | 70.12 | 117.29 | 212.93 | 366.88 | 487.69 | - |
| 256K | 70.03 | 115.03 | 208.35 | 345.21 | 457.62 | - |
| 512K | 67.95 | 111.71 | 191.99 | 302.80 | - | - |
| 1M | 66.82 | 105.82 | 177.48 | - | - | - |

### C.5 DP16-EP16 with Humming MXFP4AFP8 + Online C128 + DSpark

| Input Length / Concurrent Requests per DP Rank (tokens/s/GPU) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 76.80 | 129.62 | 236.83 | 397.42 | 584.37 | 759.73 |
| 8K | 76.69 | 130.55 | 237.53 | 398.60 | 582.03 | 762.09 |
| 16K | 76.16 | 127.88 | 233.10 | 388.54 | 571.22 | 745.23 |
| 32K | 74.07 | 124.77 | 226.24 | 378.79 | 559.05 | 722.51 |
| 64K | 74.57 | 124.69 | 223.84 | 373.13 | 541.46 | 695.35 |
| 128K | 72.36 | 120.34 | 219.66 | 365.13 | 518.98 | - |
| 256K | 71.38 | 119.19 | 211.64 | 340.72 | - | - |
| 512K | 69.54 | 115.14 | 198.81 | - | - | - |
| 1M | 67.39 | 106.50 | - | - | - | - |

## Appendix D. Capacity Results

### D.1 Decode Capacity Scaling

| Decode Profile | Configuration | Full-Token Capacity (tokens/rank) | Vs. Previous Stage | Vs. FP8 Baseline |
|---|---|---:|---:|---:|
| DP32-EP32 | Baseline FP8 + Offline C128 | 1,475,328 | - | 1.00× |
|  | Humming MXFP4AFP8 + Offline C128 | 2,526,720 | 1.71× | 1.71× |
|  | Humming MXFP4AFP8 + Online C128 | 5,731,328 | 2.268× | 3.88× |
| PP2-TP8 | Baseline FP8 + Offline C128 | 1,089,024 | - | 1.00× |
|  | Humming MXFP4AFP8 + Offline C128 | 4,869,888 | 4.47× | 4.47× |
|  | Humming MXFP4AFP8 + Online C128 | 11,044,906 | 2.268× | 10.14× |
