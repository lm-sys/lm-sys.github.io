---
title: SpecForge: Powering Fast Speculative Decoding Training For SGLang
date: "July 23, 2025"
previewImg: /images/blog/spec_forge/logo.svg 
---

Speculative decoding is a popular and powerful method for accelerating Large Language Model (LLM) inference. In this blog post, we are excited to announce the open-sourcing of **SpecForge**, our new training framework for Eagle3-based speculative decoding. We designed **SpecForge** to be incredibly **easy to use** and **tightly integrated** with the **SGLang** inference engine, enabling a seamless transition from training to deployment.

# Why the new Spec Training Framework

Speculative decoding has emerged as a breakthrough in accelerating LLM inference. However, the open-source tooling for training *draft models*—a key component of this process—remains underdeveloped. Many existing EAGLES-based projects suffer from poor maintenance, limited functionality, or lack of compatibility with frameworks like SGLang. These limitations pose significant barriers to wider adoption and practical deployment.

To bridge the gap between research and deployment, we built **SpecForge**—a purpose-built ecosystem for training draft models that integrate natively with SGLang. As soon as training completes, models are immediately ready for inference—no extra adaptation required. Meanwhile, Training effective draft models for today’s frontier LLMs—such as Llama 4, DeepSeek, and other Mixture-of-Experts (MoE) models—requires infrastructure that can handle their complexity and scale. SpecForge is designed from the ground up to meet these demands.

Key capabilities include:

-   **Native Support for Advanced Architectures**: SpecForge supports cutting-edge models, including complex MoE layers and transformer variants.
-   **Scalable Distributed Training**: Integrated with modern large-scale training strategies like Fully Sharded Data Parallel (FSDP) and Tensor Parallelism (TP), SpecForge allows efficient scaling across GPU clusters.
-   **Memory-Efficient Training**: Optimized memory management techniques make it feasible to train draft models even for very large base models.

# SpecForge Key Features

## Eagle3 Integration

Eagle is a state-of-the-art speculative decoding method designed to accelerate large language model inference. It achieves this by training a specialized, lightweight draft model to accurately predict the token distributions of a larger target model, resulting in high acceptance rates and substantial performance gains.

![intro.svg](lm-sys.github.io/public/images/blog/spec_forge/eagleintro.svg)

### Training-time Test Support

This high performance is largely driven by Eagle's novel Training-Time Test (TTT) architecture, which makes the draft model robust by simulating multi-step generation. While powerful, TTT is difficult to implement correctly due to its need for specialized attention masks and recursive data loops. Our framework simplifies this entirely by providing built-in TTT support, carefully referencing the official Eagle3 implementation to ensure correctness and performance.

## Dual Training Modes: Online and Offline

SpecForge simplifies hidden state collection by offering two versatile modes for training: **Online** and **Offline**. This dual-mode design ensures you can always find an efficient workflow, regardless of your model access rights or hardware limitations.

![offline_vs_online.svg](/images/blog/spec_forge/offline_online.jpg)

  


| Method  | Target Model                      | Disk Space Requirement                            | GPU Requirement                                              | One-liner rationale                                        |
| ------- | --------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Online  | Used during training              | Small                                             | More GPUs are needed if your target model is large           | Generating auxiliary hidden states on the fly              |
| Offline | Only used during data preparation | Huge (e.g. ultrachat+sharegpt need 12TB storage ) | as low as 1 GPU, as only need to accommodate the draft model | Preparing auxiliary hidden states beforehand and only once |

Choosing between Online and Offline modes allows you to tailor the training process to your exact needs and resources.

-   **Choose Online Mode** for maximum speed and agility. It's ideal for rapid experimentation and scenarios with limited storage, as it generates data on the fly without needing significant disk space.
-   **Opt for Offline Mode** when reproducibility and data reuse are critical. By pre-computing and storing hidden states, this mode guarantees consistency across experiments and is highly efficient if you have ample storage.

## Prioritizing Extensibility and Scalability

Our framework is designed with a strong emphasis on extensibility and scalability to meet engineering production requirements. We enable straightforward implementation and registration of new draft & target models through a modular interface.

To achieve scalability, we leverage PyTorch’s Fully Sharded Data Parallel (FSDP) framework and have implemented tensor parallelism, ensuring efficient utilization of resources and accommodation of very large models.

# Experiments

Using SpecForge, we trained the LLaMA 4 Scout and Maverick models on a 320K-sample dataset from ShareGPT and UltraChat. The models' strong performance on benchmarks like MT-Bench validates their quality and readiness for Eagle3-based inference. Our draft model for Llama4 Maverick can achieve 1.45 times speedup on the MTBench. Detailed results are summarized below.

We tested different draft token lengths for Scout and Maverick .

![scout.svg](/images/blog/spec_forge/Llama4_Scout_performance_final.svg)

![maverick.svg](/images/blog/spec_forge/Llama4_Maverick_performance_final.svg)

  


# Code and Model Availability

Explore our source code on GitHub and try the pre-trained models on Hugging Face.

**[💻 GitHub Repository](https://github.com/sgl-project/SpecForge)**: The complete source code for our training framework, including implementation details for TTT and data processing.

🤗 Hugging Face Models: Download the LLaMA 4 [Scout](https://huggingface.co/lmsys/sglang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-v1) & [Maverick](https://huggingface.co/lmsys/sglang-EAGLE3-Llama-4-Maverick-17B-128E-Instruct-v1) Eagle3 heads (w/o full model) for your projects.

# Future Roadmap

In the near future, we plan to extend SpecForge with the following support.

-   Support more model architectures, including the Kimi K2 and Qwen-3 MoE.
-   Integrate Vision-Language Models (VLM) into SpecForge.
-   Support more efficient training with better parallelism strategies and kernel optimization.

# Acknowledgement

We would like to express our heartfelt gratitude to the following teams and collaborators:

**SGLang Core Team** — Shenggui Li, Fan Yin, Chao Wang, Shuai Shi, Yikai Zhu, Yi Zhang, Yingyi Huang, Haoshuai Zheng, Yineng Zhang and many others.

**SafeAILab Team** — Yuhui Li, Hongyang Zhang and members — for their pioneering work on the Eagle3 algorithm.

**Votage Park Team** — our official infrastructure partner, for providing the critical GPU resources that made SpecForge possible.

We are especially grateful to Meituan for their strong backing and meaningful contributions