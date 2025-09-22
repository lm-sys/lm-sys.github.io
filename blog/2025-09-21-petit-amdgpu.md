---
title: "Optimizing FP4 Mixed-Precision Inference on AMD GPUs"
author: "Haohui Mai, Lei Zhang"
date: "September 21, 2025"
previewImg: /images/blog/petit/petit-facade.png
---
## Introduction

As frontier large language models (LLMs) continue scaling to unprecedented sizes, they demand increasingly more compute power and memory bandwidth from GPUs. Both GPU manufacturers and model developers are shifting toward low-precision floating-point formats. FP4 (4-bit floating point) quantization has emerged as a particularly compelling solution—for instance, FP4-quantized [Llama 3.3 70B](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4) models achieve a 3.5x reduction in model size while maintaining minimal quality degradation on benchmarks like [MMLU](https://arxiv.org/abs/2009.03300).

However, a critical gap exists in current hardware support. While next-generation GPUs from NVIDIA (GB200) and AMD (MI350) provide native FP4 matrix multiplication support, the widely-deployed AMD Instinct MI250 and MI300 series GPUs lack this capability. This limitation prevents users from leveraging efficient FP4 models on existing AMD hardware investments.

To bridge this divide, we developed Petit – a collection of optimized FP16/BF16 × FP4 mixed-precision GPU kernels specifically engineered for AMD GPUs. Petit enables serving FP4 models on both MI200 and MI300 series hardware without requiring hardware upgrades.

Petit delivers substantial performance improvements across the board:

* 1.74x faster end-to-end inference performance on Llama 3.3 70B using [SGLang](https://github.com/sgl-project/sglang)
* Up to 3.7x faster execution for equivalent matrix multiplication operations compared to [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) (AMD's state-of-the-art GEMM library)

Petit is open sourced under BSD licence and has been integrated into SGLang since 0.4.10, you can start serving dense FP4 models such as Llama 3.3 70B on AMD MI250/MI300x using the following commands:

```
 python -m sglang.launch_server --model-path nvidia/Llama-3.3-70B-Instruct-FP4 --host 0.0.0.0 --port 30000
```


This article explores our optimization journey and the techniques that made these performance gains possible. Petit leverages AMD's open software ecosystem while introducing novel optimizations including offline shuffling and low-level hardware-specific enhancements.

## Co-designing Performant GPU Kernels with Hardware Architecture

Modern GPUs achieve massive computational throughput by stacking simple yet compact compute units (CUs) on a single die. However, this hardware design philosophy requires applications to be explicitly co-designed with the underlying architecture to deliver optimal performance. As illustrated in Figure 1, several key co-design principles guided Petit's development.

<figure>
<img src="/images/blog/petit/arch.svg" alt="Overview of optimizations in Petit" style="width:95%">
<figcaption style="text-align: center">Figure 1: Overview of optimizations in Petit.</figcaption>
</figure>

### Efficient Dequantizations via pre-processing 

Petit efficiently utilizes specialized MatrixCore hardware on AMD GPUs to accelerate matrix multiplications. MatrixCore enables a wavefront (a group of 64 threads) to collectively multiply two BF16/FP16 16×16 matrices with high efficiency. However, since there's no native MatrixCore support for FP4 weights on AMD MI300x GPUs, Petit must dequantize FP4 weights to BF16/FP16 format while maintaining high efficiency for both loading FP4 weights from memory and preparing them for MatrixCore operations.

This creates a fundamental challenge: optimal memory loading and MatrixCore preparation require matrix B in different data layouts. For memory efficiency, wavefronts should load consecutive 1024-byte chunks. However, MatrixCore expects matrices partitioned into 16×16 tiles with values distributed across wavefronts. Traditional GPU-side data shuffling introduces significant overhead.

The [Marlin](https://github.com/IST-DASLab/marlin) implementation for NVIDIA GPUs addresses this by pre-arranging matrix B elements on disk, eliminating GPU-side shuffling. Adapting this approach to AMD GPUs, we pack 8 consecutive FP4 values into a 32-bit integer, requiring 31 instructions for dequantization.  
Petit goes further by tailoring the bit packing format to AMD GPU capabilities. We rearrange the first 4 FP4 elements in BF8 layout and store the remaining elements in the packed integer's remaining bits. By utilizing AMD's unique `v_bfrev_b32` and `v_cvt_pk_f32_bf8` instructions with sub-dword addressing (SDWA) capabilities, Petit dequantizes 8 FP4 values with only 15 instructions, resulting in a 30% performance improvement in multiplication operations.

### Mastering Memory Hierarchies

GPUs like the MI300X feature extremely high arithmetic density (\>500), meaning compute units must perform hundreds of operations per byte to achieve peak FLOPS. Maximizing effective memory bandwidth is therefore essential for performant matrix multiplication kernels.  
Petit employs proven techniques such as tiling and double buffering using Local Data Store (LDS), while addressing several AMD-specific considerations:

*Avoiding LDS Bank Conflicts*. AMD GPU LDS is partitioned into 32 banks, allowing 32 concurrent accesses to unique banks per cycle. Bank conflicts serialize accesses, creating performance bottlenecks. This challenge is particularly acute on AMD GPUs since wavefronts contain 64 threads. Petit implements permuted data layouts based on [bank designs](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md) to achieve conflict-free LDS utilization.

*Chiplet and Interconnect*. Each AMD MI300 GPU chiplet (XCD) features a 4MB local L2 cache and shares a 256MB L3 cache across all XCDs via interconnects. While interconnects provide high bandwidth, they introduce significant latency. Petit implements topology-aware workload partitioning that minimizes interconnect traffic, favoring naive grid-based partitions over global stripe partitions when profiling shows interconnect overhead outweighs the benefits.

### Generating High-Quality Machine Code

GPUs use simple in-order execution units to maximize CU density, but this design makes branches and pipeline stalls particularly expensive. AMD GPUs provide conditional moves and bounded memory instructions to eliminate branches entirely. For example, Petit leverages buffer load and store instructions with specified memory region ranges – the GPU automatically discards out-of-bounds accesses. Similarly, LDS accesses beyond the 64KB limit are automatically handled. This eliminates memory access branches without performance penalties. Additionally, Petit provides compiler hints to overlap MFMA (Matrix Fused Multiply-Add) instructions with memory accesses, effectively hiding memory access latency behind computation.

Standard compilers, however, may not fully utilize advanced GPU ISA capabilities. For instance, intentional out-of-bounds accesses represent undefined behavior that compilers won't optimize. These optimizations require careful manual construction and validation.

## Performance Results

### End-to-End Inference Performance

We evaluated Petit's real-world effectiveness by comparing end-to-end inference performance between FP4 and BF16 models. Testing used both variants of Llama 3.3 70B with SGLang v0.4.10, measuring input and output token throughputs for batch sizes of 10 and 64 requests. Evaluation was performed on an AMD developer cloud VM that has 1× MI300X GPU, 240 GB RAM, and 5 TB SSD. The VM runs ROCm 6.4.2 on Ubuntu 24.04.1.

<figure>
<img src="/images/blog/petit/petit-perf.svg" alt="Throughputs of input and output tokens for the offline generation benchmarks in SGLang" style="max-width:600px">
<figcaption style="text-align: center">Figure 2: Throughputs of input and output tokens for the offline generation benchmarks in SGLang.</figcaption>
</figure>

Figure 2 presents the results of the offline generation benchmark. The offline generation benchmark uses real-world ShareGPT traces as inputs which reflects the production performances. Overall Petit serving the Llama 3.3 70B FP4 model is 1.74x and 1.60x faster than SGLang serving the original BF16 model. In production scenarios with small batch sizes where performance is memory bandwidth-bound, Petit's efficient utilization of the 3.5x smaller FP4 models translates directly to superior throughput. You can reproduce the results of the benchmark using the following commands:

```
 python -m sglang.bench_offline_throughput --model-path nvidia/Llama-3.3-70B-Instruct-FP4 --num-prompts 10
 python -m sglang.bench_offline_throughput --model-path nvidia/Llama-3.3-70B-Instruct-FP4 --num-prompts 64
```


## Detailed Performance Analysis

We then compared Petit's performance against both HipBLASLt. HipBLASLt is AMD's state-of-the-art GEMM library written in low-level assembly.

Note that these libraries target slightly different workloads:

- Petit. Multiplies a BF16 matrix with an NVFP4 matrix (16 elements share 1 FP8 scale)
- HipBLASLt. Multiplies two BF16 matrices.

Though the workloads are not identical, the results present some quantitative ideas of how well Petit performs. We examined actual weight matrix sizes when serving Llama 3 70B, measuring performance with m=16 (decode workloads) and m=256 (prefill workloads), averaging 100 runs after 50 warmup iterations. Both libraries were tuned for optimal configurations.

<figure>
<img src="/images/blog/petit/fig3a.svg" alt="GEMM performance for m=16" style="max-width: 60%;">
<br>
<img src="/images/blog/petit/fig3b.svg" alt="GEMM performance for m=256" style="max-width: 60%;">
<figcaption style="text-align: center">Figure 3: GEMM performance for m=16(decode workloads) and m=256 (prefill workloads).</figcaption>
</figure>

Figure 3a and Figure 3b presents the GEMM performances of Petit and HipBlasLt. Petit is efficient: For m=16 (decode-heavy workloads), Petit is up to 3.7x faster than HipBlasLt, with an average improvement of 2.56x. For m=256 (prefill workloads), Petit is up to 1.09x faster than HipBlasLt with comparable average performance

Petit's superior performance for small m values stems from memory bandwidth optimization—the 3.5x smaller FP4 models dramatically reduce bandwidth requirements. This makes Petit particularly effective for real-world inference scenarios where m is typically small, aligning perfectly with production deployment patterns.

We studied individual optimization contributions by implementing each technique incrementally: efficient dequantization (Dequant), LDS bank conflict elimination (LDS), topology-aware work placement (Topo), and efficient instruction scheduling (InstSchedule). Figure 4 presents the breakdowns of performance improvements for various sizes of matrices.

<figure>
<img src="/images/blog/petit/fig4.svg" alt="Impacts of individual optimizations of Petit" style="max-width: 600px">
<figcaption style="text-align: center">Figure 4: Impacts of individual optimizations of Petit.</figcaption>
</figure>

We found that efficient dequantization and LDS optimization provide the largest gains: it generates 70-117% performance improvement. Topology-aware scheduling shows greater impacts for larger m. Interestingly, the results of optimizing instruction scheduling vary and do not always improve performance. Petit provides compiler hints via the `amdgcn_sched_group_barrier()` intrinsics. It is nontrivial to control the greedy scheduling algorithm inside LLVM to generate the desired sequences, while we fail to use the exponential solver as it takes too long to run.

## Lessons Learned

Our journey building Petit revealed several insights:

* Hardware-software co-design is fundamental. Understanding and designing around hardware architecture should be the foundation of any GPU kernel optimization effort. Without proper co-design, significant performance potential remains untapped regardless of other optimization efforts.  
* Programming language and compiler support is invaluable. Tools like [Triton](https://triton-lang.org) dramatically improve productivity during prototyping and exploration phases. Petit's Tensor abstractions, inspired by [CuTE](https://github.com/NVIDIA/cutlass/tree/main/include/cute), simplified offset calculations and reduced debugging time. While compilers may not fully utilize unique hardware features, exposing performance tuning knobs provides significant value.  
* Open ecosystems accelerate innovation. Access to open source codebases provides substantial advantages over black-box approaches. The ability to study, adapt, and build upon existing optimizations accelerates both development and optimization efforts.

## Conclusions

Our work optimizing Petit for AMD Instinct MI250 and MI300 GPUs demonstrates the transformative power of hardware-software co-design. Through careful attention to algorithms, memory hierarchy optimization, and low-level assembly techniques, we achieved performance improvements of up to 3.7x over state-of-the-art implementations.

The techniques and insights from Petit extend beyond this specific implementation – they represent a methodology for extracting maximum performance from specialized hardware through thoughtful co-design and optimization.

The complete source code of Petit is available at: [https://github.com/causalflow-ai/petit-kernel](https://github.com/causalflow-ai/petit-kernel).
