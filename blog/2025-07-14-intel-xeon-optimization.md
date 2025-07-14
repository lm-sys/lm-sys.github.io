---
title: "Cost Effective Deployment of DeepSeek R1 with Intel® Xeon® 6 CPU on SGLang"
author: "Intel PyTorch & SGLang Team"
date: "July 14, 2025"
previewImg: /images/blog/xeon/preview_headshot.png
---

The impressive performance of DeepSeek R1 marked a rise of giant Mixture of Experts (MoE) models in Large Language Models (LLM). However, its massive model size and unique architecture have posed new challenges on deployment. The significant memory requirements will normally require 8x or even 16x high-end AI accelerators to deploy.

Intel PyTorch Team contributed to CPU backend for SGLang for the past few months and we proposed a high-performance CPU only solution using 6th generation of Intel® Xeon ® Scalable Processor with only fractional cost. In this blog, we explain the technical details of achieving high efficiency for deploying DeepSeek on a single node with Xeon® 6 CPU.

## Highlights
* SGLang now supports native CPU backend on Intel® Xeon® CPUs with Intel® Advanced Matrix Extensions (AMX).
* Support BF16, INT8 and FP8 for both Dense FFNs and Sparse FFNs (MoE).
* Achieve **6-14x** speedup for TTFT and **2-4x** for TPOT v.s. llama.cpp.
* Achieve **85%** memory bandwidth efficiency with highly optimized MoE kernels.
* Multi-Numa Parallelism via Tensor Parallelism (TP).

## CPU Optimization Strategy
In this blog, we will explain the technical details of kernel level optimization, including task partition strategy, memory access efficiency and effective utilization of Intel® AMX for highly optimized GEMM implementations.
This section focuses on 4 performance hotspots: Extend Attention and Decode Attention which are backends for RadixAttention of SGLang; MoE which contributes to the majority of weights in DeepSeek R1; and FP8 GEMM in which we utilized an emulated approach on existing x86 platform without native FP8 support.

### Extend Attention
We implemented a native C++ backend with Intel® AMX based on interface of RadixAttention which consists of two major components: a) Extend Attention that handles prefill phase for Multi-Head Attention (MHA); b) Decode Attention for decoding phase. Taking GPU kernels as a reference, we mapped flash attention algorithm to CPU intrinsics, as illustrated in **Fig-1** below:

![Fig-1: Flash Attention in Prefilling Phase](/images/blog/xeon/fig-1.png)

To remove redundant computation, SGLang divides query sequence into two parts:

* **prefix** – historical sequence in which attention is a rectangle;
* **extend** – newly added prompt in which attention is a lower triangle.

The CPU kernel exactly maps to Flash Attention V2 algorithm, and we carefully choose the block size for Query sequence and KV sequence to make sure that the immediate values of attention `Si` and momentums `mi`, `S*` fit in L1/L2 cache. The GEMM parts are computed by AMX, and the Block Pointwise OPs are computed by AVX512. Due the fact that AMX does accumulation in FP32 (e.g. A: BF16; B:BF16; C: FP32), we fuse data type conversion with the momentum updates, keeping `Si` in FP32 which is the result of 1st GEMM and `S∆` in BF16 which is the input for 2nd GEMM, reducing rounding error to minimal level while achieving high computation efficiency.

### Decode Attention
Decoding faces more pressure on parallelization compared with prefilling, due to fact that query sequence length reduced to one. To be specific, in Multi-Head Attention we can parallel the kernel on dimensions of `[Batches, Heads, qBlocks]`, which will be simplified to `[1, Heads, 1]` for single request decoding, leading to insufficient parallelism. We implemented Flash Decoding algorithm that chunks KV sequence into multiple splits to increase the degree of parallelism, as shown in **Fig-2**. The implementation takes two phases to complete: first compute attention for each of KV split; then reduce immediate results from all splits to final output.

![Fig-2: Flash Decoding Implementation](/images/blog/xeon/fig-2.png)

#### Multi-head Latent Attention (MLA) Optimization
MLA is one of the core features of the DeepSeek series of models. We provide several critical optimizations on MLA CPU implementation aside from Flash Decoding. We referenced FlashMLA that exploits the fact **key** and **value** share the same tensor storage, and pipelines memory load and computation.

![Fig-3: MLA Decoding Implementation](/images/blog/xeon/fig-3.png)

* **Load Once Pack Twice**: AMX requires tile data in VNNI format, additionally key and value need to be packed differently since the 1st GEMM is NT and the 2nd GEMM is NN. We implemented a fully vectorized packing logic as indicated in **Fig-3**, KV caches are fetched through 2 LUTs with prefetch; with every 32 lanes loaded (`BLOCK_N` equals 32), simultaneously packed into two thread local immediate buffers, one for key in format of `[E/2, BLOCK_N, 2]`, the other for value in format of `[BLOCK_N/2, Ev, 2]`.
* **Head Folding**: MLA employs weight absorption in decode phase, which reduces the number of heads to 1 for both **key** and **value**. Therefore, we can fold Head dimension into GEMM to increase computation intensity, shown as below. And we balanced parallelism when blocking Head dimension: with a Head dimension of 22 in DeepSeek R1, we use a `BLOCK_SIZE` 6 for single request and gradually increase to 22 for more requests.


