---
title: "Accelerating Hybrid Inference in SGLang with KTransformers CPU Kernels"
author: "KVCache.AI and Approaching AI"
date: "October 22, 2025"
previewImg: /images/blog/ktransformers/primary.png
---

## Background: Hybrid Inference for Sparse MoE Models
Modern Mixture-of-Experts (MoE) language models such as **DeepSeek-V3** contain hundreds of billions of parameters, but only a small subset of experts are activated per token.

This **sparse activation** pattern makes MoE models ideal for **CPU/GPU hybrid inference**: the sparsely activated experts can run efficiently on CPUs with large memory capacity, while the dense and compute-intensive components — attention and shared experts — execute on GPUs with higher bandwidth and throughput.

<img src="/images/blog/ktransformers/heterogeneous_computing.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%"></img>

This hybrid design allows trillion-parameter models to be deployed on a single machine with limited GPU memory, enabling local inference for research and private applications.

Yet, fully exploiting both CPUs and GPUs remains challenging due to coordination overheads and underutilized compute, which limit effective throughput.

## KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

To solve the above problem, MadSys @ Tsinghua and Approaching.AI created the **KTransformers** project, presented at SOSP’25, introduces a collection of optimizations that make CPU/GPU collaboration for MoE inference far more efficient.

Its improvements fall into three main categories:

### 1. AMX-Specialized CPU Kernels

KTransformers redesigns CPU computation with Intel AMX–optimized kernels and a tiling-aware memory layout that aligns weight storage with cache hierarchies. It also supports dynamic switching between AMX (for high-intensity prefill workloads) and AVX-512 (for lightweight decode). On a single Xeon socket, the AMX-optimized kernels can reach up to **21.3 TFLOPS** of sustained throughput — **3.9×** faster than PyTorch native implementations. This directly translates into substantially higher CPU-side expert throughput during prefill and overall token throughput in hybrid runs.

### 2. Efficient Device Coordination

To reduce coordination costs between CPUs and GPUs, KTransformers introduces NUMA-aware tensor parallelism and CUDA Graph–backed scheduling.

NUMA-aware tensor parallelism places expert weight slices in the local memory of each NUMA node so that compute is mostly local, avoiding expensive cross-NUMA memory traffic; this yields up to **63%** decoding throughput improvement on dual-socket servers.

CUDA Graph integration captures the hybrid CPU/GPU execution as continuous graphs. To make captures robust, KTransformers uses asynchronous task scheduling so that CPU tasks and data transfers do not create “breakpoints” in the captured graph. Capturing the workload this way reduces GPU kernel-launch overhead from **over 20%** to **nearly zero**.

Together, these optimizations ensure both devices operate with minimal synchronization delays.

### 3. Expert Deferral: Overlapping Model Execution

KTransformers further introduces an Expert Deferral mechanism that reorders expert execution across layers. Some experts are deferred to later stages, allowing CPU expert computation to overlap with GPU attention processing.

<img src="/images/blog/ktransformers/expert_deferral.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%"></img>

Because modern Transformers use residual connections, they are inherently tolerant of small delays to intermediate computations. Consequently, deferring certain expert computations enhances scheduling flexibility at the cost of only slight changes in model behavior.

This mechanism increases concurrent utilization of both devices and yields up to **1.45× higher decoding throughput**, with accuracy variation below 0.5%.

## Integrating KTransformers into SGLang

SGLang now integrates KTransformers as a backend library to enable efficient CPU/GPU hybrid inference, combining GPU Tensor Parallelism with CPU/GPU Hybrid Expert Parallelism for MoE models. This integration supports inference across heterogeneous devices, where KTransformers provides highly optimized AMX-based CPU kernels that work seamlessly with GPU execution.

While KTransformers focuses on single-GPU setups and high-efficiency CPU cooperation, SGLang excels at scaling across multiple GPUs, which is particularly advantageous in **high-concurrency scenarios**. In the hybrid setting, multiple GPUs can handle larger request contexts and perform fast attention computation, while experts are intelligently scheduled across CPUs and GPUs—storing frequently used (“hot”) experts on GPUs to alleviate CPU compute and bandwidth pressure.

With this joint design, users across diverse hardware configurations can fully utilize available resources, achieving better throughput, scalability, and cost efficiency.

We have already developed a proof-of-concept implementation, and the [roadmap](https://github.com/sgl-project/sglang/issues/11425) for full integration into SGLang is underway.

## Installation

To use KTransformers hybrid inference with SGLang, you need to install both SGLang and the KTransformers CPU kernels (`kt-kernel`).

### Prerequisites

Before installation, ensure your system meets the following requirements:

- **CUDA**: Version 12.1 or above with proper PATH configuration
- **Operating System**: Linux x86_64
- **Compiler**: gcc, g++ >= 11
- **Build Tools**: CMake >= 3.25 (Note: Ubuntu 22.04 LTS default CMake may be too old)
- **Python**: Python 3.11 (via Miniconda3 or Anaconda3)

### Step 1: Install SGLang

Follow the official [SGLang installation guide](https://docs.sglang.ai/get_started/install.html) to install SGLang:

```bash
pip install "sglang[all]"
```

### Step 2: Install KTransformers CPU Kernels

The KTransformers CPU kernels (`kt-kernel`) provide AMX-optimized computation for hybrid inference, for detailed installation instructions and troubleshooting, refer to the [official kt-kernel installation guide](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md).

## Usage Example

### Downloading Models

The DeepSeek-R1 models optimized for KTransformers hybrid inference (including both GPU and CPU weights) can be downloaded from the [Approaching AI ModelScope profile](https://modelscope.cn/profile/ApproachingAI2024).

### Launching the Server

To launch an SGLang server with KTransformers hybrid inference enabled, you can use the following command:

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/gpu-weight \
  --kt-amx-weight-path /path/to/cpu-weight \
  --kt-cpuinfer 80 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 200 \
  --kt-amx-method AMXINT4 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 37 \
  --max-total-tokens 37000 \
  --served-model-name DeepSeek-R1-0528-FP8 \
  --enable-mixed-chunk \
  --tensor-parallel-size 8 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

### Key Parameters

- `--kt-amx-weight-path`: Path to the CPU-optimized model weights. These weights are pre-quantized and formatted for efficient AMX computation.
- `--kt-cpuinfer`: Number of CPU cores dedicated to expert inference (e.g., 80 cores for dual-socket servers).
- `--kt-threadpool-count`: Number of thread pools for parallel CPU execution. Typically set to 2 for dual-socket NUMA configurations.
- `--kt-num-gpu-experts`: Number of "hot" experts to keep on GPU. More GPU experts reduce CPU compute pressure but require additional GPU memory. Adjust based on GPU capacity and workload patterns.
- `--kt-amx-method`: CPU kernel optimization method. Use `AMXINT4` for int4-quantized models to leverage Intel AMX instructions for maximum throughput.

### Hardware Requirements

For optimal performance with KTransformers hybrid inference:

- **CPUs**: Modern Intel Xeon processors with AMX support (e.g., Sapphire Rapids or later) for maximum CPU expert throughput.
- **Memory**: Sufficient DDR5 memory to hold all expert weights (typically 500GB+ for DeepSeek-V3-sized models).
- **GPUs**: One or more GPUs with enough memory for attention layers, shared experts, and a subset of routed experts.
- **NUMA**: Dual-socket configurations benefit from NUMA-aware thread pool assignment (`--kt-threadpool-count 2`).

After launching the server, you can send inference requests via the OpenAI-compatible API endpoint at `http://0.0.0.0:30000`.

## Benchmark Results (Preview)

### Single-GPU + CPU Performance

Native KTransformers conducted detailed performance evaluations on a single-GPU + CPU setup. Under the same configuration, SGLang integrated with KTransformers achieves comparable performance to native KTransformers.

The evaluations are set on a dual-socket Intel® Xeon® Platinum 8452Y server (36 cores × 2, 1 TB DDR5 × 2) with an NVIDIA A100 (40 GB) for full-precision models and an RTX 4080 (16 GB) for quantized models.

<img src="/images/blog/ktransformers/prefill_performance.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%"></img>

In the **prefill phase**, KTransformers consistently outperforms both baselines across all prompt lengths, benefiting from AMX-optimized CPU kernels and achieving **speedups of up to 20×**.

<img src="/images/blog/ktransformers/decode_performance.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%"></img>

In the **decode phase**, KTransformers also outperforms both baselines, with gains mainly attributed to reduced CPU/GPU coordination overhead, reaching **up to 4× speedup**.

### Multi-GPU + CPU Performance

We further evaluate the multi-GPU + CPU hybrid inference capability enabled by integrating KTransformers into SGLang. Specifically, we tested int4-quantized DeepSeek-V3 on a system equipped with 8× L20 GPUs and dual-socket Intel Xeon Gold 6454S CPUs, using workloads with an average input length of 128 tokens and output length of 512 tokens.

<img src="/images/blog/ktransformers/multigpu_performance.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%"></img>

The table above presents the total throughput (tokens/s) under different levels of concurrency and varying numbers of GPUs. As shown, under single-concurrency conditions, the 8-GPU configuration provides only a limited improvement over the 1-GPU setup (an increase of merely 26%). However, under 8-way concurrency, the same 8-GPU configuration achieves a **264% throughput** gain compared to 1 GPU, demonstrating excellent usability—each request achieves nearly 20 tokens per second on average. The improvement mainly comes from placing more experts on GPUs, which reduces CPU memory accesses under bandwidth bottlenecks.

#### ShareGPT Benchmark on NVIDIA L20 × 8 Setup

We further evaluated the SGLang + KTransformers integration on a GPU setup using **8× NVIDIA L20 GPUs** with an **Intel(R) Xeon(R) Gold 6454S CPU**. The benchmark was conducted on **DeepSeek-R1-0528**, a large-scale MoE model from the DeepSeek-R1 series, using the ShareGPT dataset with 1000 conversation requests (301K input tokens, 188K output tokens).

**System Configuration:**
- GPUs: 8× NVIDIA L20
- CPU: Intel(R) Xeon(R) Gold 6454S
- Model: DeepSeek-R1-0528 (FP8 quantized MoE model)
- Dataset: ShareGPT (1000 requests)

**Benchmark Commands:**

First, launch the SGLang server:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model models/DeepSeek-R1-0528-GPU-weight \
  --kt-amx-weight-path models/DeepSeek-R1-0528-CPU-weight \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 200 \
  --kt-amx-method AMXINT4 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 40 \
  --max-total-tokens 40000 \
  --served-model-name DeepSeek-R1-0528-FP8 \
  --enable-mixed-chunk \
  --tensor-parallel-size 8 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

Then, run the benchmark in a separate terminal:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --num-prompts 1000 \
  --model models/DeepSeek-R1-0528-GPU-weight
```

**Performance Results:**

| Metric | Value |
|--------|-------|
| Total Token Throughput | 227.85 tok/s |
| Output Token Throughput | 87.58 tok/s |
| Request Throughput | 0.46 req/s |
| Mean Inter-Token Latency (ITL) | 431.61 ms |
| Median Inter-Token Latency | 299.18 ms |
| P99 Inter-Token Latency | 1935.13 ms |

This setup demonstrates that SGLang + KTransformers can effectively leverage consumer-grade GPUs for hybrid inference, achieving **over 220 tokens/s total throughput** on trillion-parameter MoE models. The relatively low inter-token latency (median 299ms) ensures smooth streaming generation for interactive applications.

## Acknowledgements

We would like to thank everyone in the community that helped make this effort possible.

**KVCache.AI team**: Boxin Zhang, Jianwei Dong, Hongtao Chen, Weiyu Xie, Shaoyuan Chen, Chen Lin, Chengyu Qiu, Yuening Zhu, Jingqi Tang, Qingliang Ou, Yongwei Wu and Mingxing Zhang from MadSys @ Tsinghua University.

**Approaching AI**: Jiahao Wang, Ziwei Yuan, Yaochen Han,  Jiaqi Liao, Xianglin Chen, Zhiyuan Ai, Yongsen Hu, Zhuo Wang, Daocheng Ye, Yanlong Wu, Yufeng Tian, Heng Guo, Hao Wu, Zirui Li, Yingqi Tian, Yue Qin, Xin Qu, Baijin Hao, Donghui Liu.

**SGLang team and community:** Jingyi Chen, Shangming Cai, Lianmin Zheng, Yineng Zhang and many others for their insightful review comments on this PR and for their work on SGLang framework.

## Related resources

Repo：https://github.com/kvcache-ai/ktransformers

SOSP25 Paper：https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
