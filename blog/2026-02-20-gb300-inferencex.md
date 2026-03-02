---
title: "Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72"
author: "NVIDIA and Community SGLang Developers"
date: "February 20, 2026"
previewImg: /images/blog/gb300_inferencex/img-1.png
---
The SGLang team has worked closely with NVIDIA across [multiple GPU generations](https://lmsys.org/blog/2025-05-05-large-scale-ep/) to unlock step-function gains in inference performance for large-scale deployments of Mixture of Expert (MoE) reasoning models. Building on [prior results](https://lmsys.org/blog/2025-10-14-sa-inference-max/) that delivered 4x speedups on Blackwell B200 vs.Hopper H200 in SemiAnalysis InferenceMAXv1, we are now extending this momentum to Blackwell Ultra. With GB300 NVL72, SGLang achieves up to 25x performance gain on the latest InferenceXv2 benchmark compared to H200. Additionally, we increased SGLang's InferenceXv2 performance on GB200 NVL72 by up to 8x in less than 4 months. These performance gains are a result of the close collaboration between SGLang developers and NVIDIA engineering teams and translate directly into lower latency, higher throughput, and significantly reduced cost per token for large-scale Mixture of Experts (MoE) reasoning model deployments.

<img src="/images/blog/gb300_inferencex/img-1.png"
     style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;">

## **NVIDIA GB300 NVL72 with Blackwell Ultra GPUs**

The [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) has already established itself as the most powerful scale-up data center GPU platform, connecting 72 Blackwell GPUs into a single high-bandwidth domain at 130 TB/s. This architecture is particularly well suited for MoE models, which depend on low-latency, all-to-all communication for Wide Expert Parallel execution and fast KV-cache movement in disaggregated serving between prefill and decode GPUs.

The NVIDIA GB300 NVL72 builds on this foundation with Blackwell Ultra GPUs, introducing several key enhancements over GB200 NVL72:

**1.5x peak NVFP4 throughput.** Updated Tensor Cores increase peak FP4 throughput per clock by 1.5x compared to Blackwell, accelerating math-bound GEMM operations for MoE experts and dense layers.

**2x Softmax throughput for attention.** An upgraded special function unit (SFU) delivers 2x higher throughput on softmax operations, a critical component in attention layers.

**1.5x larger HBM3e capacity.** Blackwell Ultra integrates higher-capacity 12‑Hi HBM3e stacks (up from 8‑Hi), enabling larger models and batch sizes without resorting to CPU offload.

When combined with the large 72‑GPU NVL72 domain, these capabilities increase throughput for MoE GEMMs, speed up attention softmax, and support large decode batch sizes in disaggregated inference setups.



## **25x More SGLang Performance with GB300 NVL72**

SemiAnalysis InferenceX (formerly InferenceMAX) is a continuously running benchmark suite that evaluates real-world inference performance across popular open-source frameworks and models on hundreds of accelerators, with live results available at inferencemax.ai. The [InferenceMAXv1 release](https://lmsys.org/blog/2025-10-14-sa-inference-max/) showcased SGLang's ability to extract up to 4x performance gains on Blackwell versus Hopper for DeepSeek R1.

In the latest InferenceXv2, NVIDIA's GB300 NVL72 rack-scale system has been added to the benchmark matrix. Leveraging our ongoing collaboration with NVIDIA, SGLang now demonstrates up to 25x higher performance running DeepSeek R1 on GB300 NVL72 compared to H200. This uplift combines architectural advances in Blackwell Ultra with targeted SGLang software and kernel optimizations across the inference stack.

<img src="/images/blog/gb300_inferencex/img-2.png"
     style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;">

Please note that the H200 baseline used for the 25x improvement is taken at 50 TPS/user interactivity which reflects low latency use cases. In the absence of latency constraint H200 can achieve similar throughput as discussed in this [prior blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/). We selected 50 TPS/user as the comparison point for this blog to give a scenario where reasonable latency is expected.

## **Inference Optimizations for Blackwell Ultra**

To fully exploit the capabilities of Blackwell Ultra on GB300 NVL72, SGLang incorporated new optimizations spanning low precision data formats, kernel design, and disaggregated serving:

**NVFP4 GEMM for MoE and dense layers.** Using [NVFP4 precision](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) for MoE experts and other GEMMs reduces memory bandwidth pressure, taps into the higher FP4 Tensor Core throughput on Blackwell Ultra, and halves communication traffic for token dispatch. This shrinks weights in memory, freeing capacity for a larger KV cache and enabling higher concurrency.

<img src="/images/blog/gb300_inferencex/overlap_scheduling.png"
     style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;">

**Computation–communication overlap.** Instead of relying on traditional Two-Batch overlapping (TBO), we adopt a single-batch overlap strategy tuned to the higher interconnect bandwidth of NVL72. In practice, this allows combining communication to run concurrently with down-GEMM computation in a producer–consumer pattern, while overlapping shared-expert computation on an additional CUDA stream to minimize idle time.

**NVIDIA Dynamo for disaggregated inference.** For prefill–decode disaggregation, we integrate with [NVIDIA Dynamo](https://www.nvidia.com/en-us/ai/dynamo/), an open-source distributed inference serving engine. Dynamo's modular design makes it possible to deeply couple its KV-aware router with SGLang's HiCache radix tree, while exposing flexible KV cache transfer backends such as NIXL and Mooncake to match different deployment scenarios.

<img src="/images/blog/gb300_inferencex/dynamo_integration.png"
     style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;">

Together, these optimizations align the inference software stack with the characteristics of Blackwell Ultra, driving higher utilization and turning its raw hardware capability into delivered throughput.

## **8x More Performance on GB200 NVL72**

While GB300 NVL72 is our new performance flagship, we remain committed to continuously improving SGLang on GB200 NVL72. Compared to our prior InferenceMAXv1 submission, less than 4 months ago, the latest v2 release with low precision NVFP4 delivers up to 8x more tokens-per-GPU in high throughput regimes and up to 4x more tokens-per-user in high interactivity regimes, enabling better token economics and end-user experience on existing GB200 NVL72 deployments. These results validate the power of the joint NVIDIA and SGLang engineering collaboration.

<img src="/images/blog/gb300_inferencex/img-3.png"
     style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;">

## **Looking Ahead**

Our roadmap with NVIDIA continues beyond this initial 25x milestone on InferenceXv2. The next phase of collaboration focuses on:

Enabling MTP on GB300 NVL72 to unlock further performance gains vs. Hopper

Continued GB300 NVL72 optimizations targeting both latency-sensitive and throughput-oriented deployments

Tuning SGLang for Qwen model families, including the latest Qwen 3.5, on Blackwell and Blackwell Ultra.

Bringing these optimizations to future NVIDIA Vera Rubin NVL72 systems.

By continuing our collaboration with NVIDIA, SGLang aims to keep pushing inference performance forward to reduce the deployment costs of the next wave of frontier reasoning models.

## **Acknowledgements**

We would like to express our heartfelt gratitude to the following teams and collaborators:

NVIDIA team: Yangmin Li, Hao Lu, Ishan Dhanani, Weiliang Liu, Trevor Morris, Po-Han Huang, Kaixi Hou, Shu Wang, Lee Nau, Alex Yang, Mathew Wicks, Kyle Liang, Grace Ho, Kedar Potar, Pen Chung Li, Amr Elmeleegy and many more

SGLang Core Team and Community Contributors: Baizhou Zhang, Jingyi Chen, Liangsheng Yin, Qiaolin Yu, Cheng Wan, Lianmin Zheng
