---
title: "SpecBundle & SpecForge v0.2: Production-Ready Speculative Decoding Models and Framework"
author: "SpecForge Team, Ant Group AQ Team, Nex-AGI Team, EigenAI Team"
date: "December 23, 2025"
previewImg: /images/blog/specbundle-phase1/preview.png
---

## TL;DR

The SpecForge team has collaborated with multiple industry partners - including **Ant, Meituan, Nex-AGI, and EigenAI** - to release [**SpecBundle (Phase 1)**](https://huggingface.co/collections/lmsys/specbundle), a collection of production-grade EAGLE-3 model checkpoints trained on large-scale datasets. **SpecBundle** is designed to improve the availability and real-world performance of speculative decoding, with Phase 1 focusing on instruct-tuned models.

Alongside this release, [**SpecForge v0.2**](https://github.com/sgl-project/SpecForge) delivers major system upgrades, including extensive refactoring for improved usability and support for multiple execution backends, further enhancing scalability and production readiness.

## Background

[Speculative decoding](https://arxiv.org/abs/2302.01318) was first introduced in 2023 as a promising technique for accelerating large language model (LLM) inference by using a lightweight draft model to propose multiple tokens that are subsequently verified by a stronger target model. In principle, this approach can substantially reduce decoding latency without compromising output quality, making it appealing for both local and enterprise deployments. Over the past few years, the research community has continued to refine this paradigm, proposing increasingly sophisticated methods that culminate in state-of-the-art approaches such as [EAGLE3](https://arxiv.org/abs/2503.01840), which demonstrate strong theoretical guarantees and empirical gains in both token acceptance rate and end-to-end speedup.

### Existing Problems

Despite these advances, speculative decoding—particularly SOTA methods like EAGLE3, has not yet seen widespread adoption in the open-source community. We attribute this gap primarily to three factors.

**Factor 1:** There is a lack of accessible, production-ready tooling for training speculative decoding models. Most existing implementations remain research prototypes that are either poorly maintained or narrowly scoped, while others offer only simplistic implementations without sufficient system-level optimization. As a result, these tools struggle to support the diverse range of model architectures and scales commonly used in today’s LLM ecosystem.

**Factor 2:** The availability of high-quality draft models remains a major bottleneck. Effective speculative decoding critically depends on the strength of the draft model, yet such models are scarce in the open community, as summarized in the table below. Methods like EAGLE3 require additional draft-model training, and the publicly available EAGLE3 checkpoints are largely limited to releases mainly from the original authors. This constrained supply significantly hampers broader adoption.

| Model                                     | Native MTP | Community EAGLE3 | SpecBundle |
| ----------------------------------------- | ---------- | ---------------- | ---------- |
| meta-llama/Llama-3.1-8B-Instruct          | ❌         | ✅               | ✅         |
| meta-llama/Llama-3.3-70B-Instruct         | ❌         | ✅               | ✅         |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | ❌         | ✅               | ✅         |
| Qwen/Qwen3-30B-A3B-Instruct-2507          | ❌         | ❌               | ✅         |
| Qwen/Qwen3-235B-A22B-Instruct-2507        | ❌         | ✅               | ✅         |
| Qwen/Qwen3-Next-80B-A3B-Instruct-FP8      | ✅         | ❌               | ✅         |
| Qwen/Qwen3-Coder-30B-A3B-Instruct         | ❌         | ❌               | ✅         |
| Qwen/Qwen3-Coder-480B-A35B-Instruct       | ❌         | ❌               | ✅         |
| inclusionAI/Ling-flash-2.0                | ❌         | ❌               | ✅         |
| moonshotai/Kimi-K2-Instruct               | ❌         | ❌               | ✅         |
| nex-agi/Qwen3-30B-A3B-Nex-N1              | ❌         | ❌               | ✅         |
| nex-agi/Qwen3-32B-Nex-N1                  | ❌         | ❌               | ✅         |

**Factor 3**: Most existing draft models are trained on relatively small or curated datasets and are not scaled to the large, diverse corpora used in modern LLM training. Consequently, these models often exhibit limited generalization and lower token acceptance rates when paired with strong target models, reducing the practical speedups achievable through speculative decoding. Without large-scale, production-grade draft models, the full potential of advanced approaches such as EAGLE3 remains largely unrealized.

### Motivation

These gaps as mentioned above motivate the release of [**SpecForge v0.2**](https://github.com/sgl-project/SpecForge) and [**SpecBundle**](https://huggingface.co/collections/lmsys/specbundle). As a neutral, open-source community, the SpecForge team aims to proactively advance speculative decoding by providing production-grade training frameworks and high-performance draft models, making the technique both practical and accessible to the broader community.

This initiative delivers several key benefits:

1. Expand research possibilities by offering more standardized and scalable baselines for advancing speculative decoding methods.
2. Enable faster local inference and model serving, supporting lightweight deployment scenarios through tools such as [Ollama](https://github.com/ollama/ollama).
3. Lower the cost of enterprise deployments by improving inference throughput without sacrificing output quality with inference engines such as [SGLang](https://github.com/sgl-project/sglang).
4. Provide strong initialization points in the form of EAGLE3 checkpoints that can be efficiently fine-tuned for domain-specific tasks.
5. Improve the efficiency of reinforcement learning workflows by enabling techniques such as [ReSpec](https://arxiv.org/abs/2510.26475) to be integrated into existing RL frameworks like [slime](https://github.com/THUDM/slime).

## SpecForge v0.2

It has been about five months since **SpecForge** was open-sourced, and thanks to the support of an amazing community, the system has evolved into a solution that is significantly more reliable, efficient, and scalable. Over the past two months, as we trained a wide range of models for SpecBundle, we identified several limitations in the original design of SpecForge. These insights motivated a comprehensive upgrade of the framework to improve both performance and usability. The major changes in **SpecForge v0.2** are summarized below.

### User-friendliness Enhancment

In the early versions of SpecForge, some features were developed independently without sufficient consideration for long-term maintainability or user experience, which occasionally led to confusion for users. Over the past two months, we have prioritized usability and conducted substantial refactoring across the framework. Key improvements include:

1. Refactored data processing pipelines to eliminate duplication and improve efficiency. For example, data regeneration is now up to **10× faster** than in v0.1 through data parallelism and asynchronous processing.
2. Unified online and offline training scripts into a single implementation. This consolidation ensures consistent training logic and avoids divergence between online and offline training modes.
3. Improved documentation structure and clarity, with a clearer logical workflow and better readability to help users get started and iterate more effectively.

### Multi-backend Support

Earlier versions of SpecForge relied heavily on in-house implementations of target models, making model support both tedious and error-prone. To address this limitation and better leverage the broader ecosystem, we introduced a unified interface for target model integration.

In **v0.2**, we introduce the `Eagle3TargetModel` interface, which enables seamless support for multiple execution backends. Currently, SpecForge integrates both **SGLang** and **Hugging Face Transformers** as backends. Adding a new backend now requires implementing only the `Eagle3TargetModel.generate_eagle3_data` method, significantly lowering the barrier to extension and improving long-term maintainability.

```python
target_model = get_eagle3_target_model(
                pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                backend="sglang",
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.model_download_dir,
                **target_model_kwargs,
            )
```

These backends not only reduce the burden of model implementation and performance optimization for developers, but also provide users with flexible choices across different training scenarios. With multiple backend options available, users can select the most suitable backend for their specific development and runtime requirements.

<p align="center">
  <img src="/images/blog/specbundle-phase1/backend.png" alt="Logo preview">
  <br>
</p>

## The SpecBundle Initiative

As discussed above, the open-source community continues to face significant bottlenecks in both the availability and performance of speculative decoding solutions. SpecBundle is a direct response to these challenges—an initiative driven jointly by the open-source community and industry partners to close these gaps. To the best of our knowledge, this represents the first open effort aimed at democratizing the adoption of speculative decoding by equipping mainstream open-source models with high-performance EAGLE3 draft model weights.

<div style="border-left: 4px solid #3b82f6; padding: 10px 12px; margin: 12px 0; background: #eff6ff; border-radius: 8px;">
  <strong>👉 Check out the <a href="https://docs.sglang.io/SpecForge/community_resources/specbundle.html">documentation on SpecBundle</a>.</strong>
</div>

In this initial release, the SpecBundle roadmap focuses exclusively on instruct-tuned models, as outlined below. We believe that expanding speculative decoding support across a broader range of models will further reduce the cost of both local and enterprise deployments, while also enabling more efficient rollout in reinforcement learning (RL) training pipelines.

<p align="center">
  <img src="/images/blog/specbundle-phase1/roadmap.png" alt="Logo preview">
  <br>
</p>

At the same time, SpecBundle serves as a large-scale validation of SpecForge, demonstrating its efficiency, extensibility, and scalability. By successfully training draft models for target models ranging from 8B to 1T parameters, we confirm that recent architectural and system improvements have elevated SpecForge to production readiness.

We warmly welcome community contributions and industrial collaboration. If you share our vision of accelerating LLM inference and training, we invite you to join us in pushing the boundaries of what speculative decoding can achieve.

### Performance

For all models released in SpecBundle, we regenerated model responses using SGLang to better align the training data distribution with the actual model outputs. This alignment significantly improves token acceptance rates in speculative decoding.

In contrast to the original EAGLE papers, which rely on the ShareGPT and UltraChat datasets comprising approximately 320K samples, SpecBundle is trained on the [Perfect-Blend](https://huggingface.co/datasets/mlabonne/open-perfectblend) dataset, which contains **1.4M samples** spanning a much broader set of domains—particularly in coding and mathematics.

As a result, SpecBundle not only supports a wide range of mainstream instruct-tuned models, but also delivers strong and consistent speedup across diverse benchmarks, achieving **up to 4× end-to-end inference speedup** over standard decoding baselines.

- **Comparison with the existing open-sourced weights**

<div style="display:flex; gap:16px;">
  <img src="/images/blog/specbundle-phase1/llama4-perf.png" alt="Llama-4 scout" style="width:50%;">
  <img src="/images/blog/specbundle-phase1/qwen-235b-perf.png" alt="Qwen-235B" style="width:50%;">
</div>

- **SpecBundle for models with more than 100B parameters**

<div style="display:flex; gap:16px;">
  <img src="/images/blog/specbundle-phase1/ling-perf.png" alt="ling-flash-v2" style="width:50%;">
  <img src="/images/blog/specbundle-phase1/kimi-perf.png" alt="kimi-k2" style="width:50%;">
</div>

<div style="border-left: 4px solid #3b82f6; padding: 10px 12px; margin: 12px 0; background: #eff6ff; border-radius: 8px;">
  <strong>👉 Check out the <a href="https://huggingface.co/collections/lmsys/specbundle">full model collection</a>.</strong>
</div>

We have published comprehensive benchmark results on the [SpecBundle website](https://docs.sglang.io/SpecForge/SpecBundle/index.html). Please visit the site for more detailed evaluation results.

## Roadmap

Last but not least, the SpecForge team will continue building and expanding the LLM ecosystem throughout 2026. We have just published the 2026 Q1 roadmap, and we would love to hear your feedback and have you join us on this journey.

Our upcoming efforts will primarily focus on:

- Long-context training
- Vision–Language Model (VLM) support
- System-level performance enhancements
- MTP finetuning
- SpecBundle Phase 2 (reasoning models) and Phase 3 (VLMs)

Whether you are a researcher, practitioner, or industry partner, we warmly welcome your ideas and contributions as we work together to push the boundaries of scalable and efficient LLM systems.

<div style="border-left: 4px solid #3b82f6; padding: 10px 12px; margin: 12px 0; background: #eff6ff; border-radius: 8px;">
  <strong>👉 Check out the <a href="https://github.com/sgl-project/SpecForge/issues/374">full roadmap</a>.</strong>
</div>

## Acknowledgement

We sincerely appreciate the collective efforts from both the developers in the open-source community and our industrial partners, especially **Ant Group AQ Team**, **Meituan**, **Nex-AGI (Qiji Zhifeng)**, and **EigenAI** for their invaluable contributions to the development of SpecBundle and SpecForge.

- SpecForge Team: Shenggui Li, Chao Wang, Yubo Wang, Yefei Chen, Yikai Zhu, Jiaping Wang, Jin Pan, Tao Liu, Fan Yin, Shuai Shi, Yineng Zhang
- Ant Group AQ Team: Ji Li, Yanan Gao, Zhiling Ye
- Meituan Search Team: Laixin Xie
- Nex-AGI (Qiji Zhifeng) Team: Qiaoling Chen, Guoteng Wang, Peng Sun
- EigenAI Team: Xiaomin Dong, Jinglei Cheng
