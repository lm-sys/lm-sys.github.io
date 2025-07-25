---
title: "SpecForge: Powering Efficient Speculative Decoding Training for SGLang"
author: "The SGLang Team"
date: "July 23, 2025"
previewImg: /images/blog/spec_forge/logo.jpg
---

Speculative decoding is a powerful technique for accelerating Large Language Model (LLM) inference. In this blog post, we are excited to announce the open-sourcing of **SpecForge**, our new training framework for Eagle3-based speculative decoding. SpecForge is designed to be easy to use and tightly integrated with the **SGLang** inference engine, enabling a seamless transition from training to deployment.

## Why a New Speculative Decoding Training Framework

Speculative decoding has emerged as a breakthrough in accelerating LLM inference. However, the open-source tooling for training *draft models*—a key component of this process—remains underdeveloped. Many existing Eagle3-based projects suffer from poor maintenance, limited functionality, or lack of compatibility with frameworks like SGLang. These limitations have become significant barriers to adoption and practical deployment.

To bridge the gap between research and deployment, we built **SpecForge**—a purpose-built ecosystem for training draft models that integrate natively with SGLang. As soon as training completes, models are ready for inference out of the box—no further adaptation needed. Meanwhile, training effective draft models for today’s frontier LLMs—such as Llama 4, DeepSeek, and other Mixture-of-Experts (MoE) models—requires infrastructure that can handle their complexity and scale. SpecForge is designed from the ground up to meet these demands.

Key capabilities include:

-   **Native Support for Advanced Architectures**: SpecForge supports cutting-edge models, including complex MoE layers and transformer variants.
-   **Scalable Distributed Training**: Integrated with modern large-scale training strategies like Fully Sharded Data Parallel (FSDP) and Tensor Parallelism (TP), SpecForge allows efficient scaling across GPU clusters.
-   **Memory-Efficient Training**: Optimized memory management techniques make it feasible to train draft models even for very large base models.

## SpecForge Key Features

### Eagle3 Integration

Eagle is a state-of-the-art method for speculative decoding designed to accelerate large language model inference. It achieves this by training a specialized, lightweight draft model to accurately predict the token distributions of a larger target model, leading to high acceptance rates and significant performance improvements.

![intro.svg](/images/blog/spec_forge/eagleintro.PNG)

#### Training-time Test Support

This high performance is largely driven by Eagle's novel Training-Time Test (TTT) architecture, which makes the draft model robust by simulating multi-step generation. Despite its power, TTT is notoriously difficult to implement due to its use of specialized attention masks and recursive data loops. Our framework simplifies this entirely by providing built-in TTT support, carefully referencing the official Eagle3 implementation to ensure correctness and performance.

### Dual Training Modes: Online and Offline

SpecForge simplifies hidden state collection by offering two versatile modes for training: **Online** and **Offline**. This dual-mode design ensures flexibility across workflows, regardless of your model access rights or hardware limitations.

![offline_vs_online.svg](/images/blog/spec_forge/offline_online.jpg)

  


| Method  | Target Model                      | Disk Space Requirement                            | GPU Requirement                                              | One-liner rationale                                        |
| ------- | --------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Online  | Used during training              | Small                                             | More GPUs are needed if your target model is large           | Generating auxiliary hidden states on the fly              |
| Offline | Only used during data preparation | Huge (e.g. ultrachat+sharegpt need 12TB storage ) | as low as 1 GPU, since only the draft model needs to be accommodated | Preparing auxiliary hidden states beforehand and only once |

Choosing between Online and Offline modes allows you to tailor the training process to your exact needs and resources.

-   **Choose Online Mode** for maximum speed and agility. It's ideal for rapid experimentation and scenarios with limited storage, as it generates data on the fly without needing significant disk space.
-   **Use Offline Mode** when reproducibility and data reuse are priorities. By pre-computing and storing hidden states, this mode guarantees consistency across experiments and is highly efficient if you have ample storage.

### Prioritizing Extensibility and Scalability

Our framework is designed with a strong emphasis on extensibility and scalability to meet engineering production requirements. We enable straightforward implementation and registration of new draft & target models through a modular interface.

To scale effectively, SpecForge leverages PyTorch’s FSDP framework and has implemented tensor parallelism, ensuring efficient utilization of resources and accommodation of very large models.

## Experiments

Using SpecForge, we trained the Llama 4 Scout and Maverick models on a 320K-sample dataset from ShareGPT and UltraChat. The models' strong performance on benchmarks like MT-Bench demonstrates their effectiveness and readiness for Eagle3 inference. Our draft model for Llama4 Maverick achieves 2.18x speedup on the MT-Bench, while Llama4 Scout demonstrates a 2x acceleration on the same benchmark. Detailed results are summarized below.

We evaluated various draft token lengths for Scout and Maverick. 

In all the tests shown in the figure below, the x-axis represents steps, corresponding to speculative-num-steps in SGLang. Meanwhile, we fixed SGLang's speculative-eagle-topk to 8 and speculative-num-draft-tokens to 10 to ensure that tree attention can be enabled.

![scout.svg](/images/blog/spec_forge/Llama4_Scout_performance_final.svg)

![maverick.svg](/images/blog/spec_forge/Llama4_Maverick_performance_final.svg)

  


## Code and Model Availability

Explore our source code on GitHub and try the pre-trained models on Hugging Face.

**[💻 GitHub Repository](https://github.com/sgl-project/SpecForge)**: The complete source code for our training framework, including implementation details for TTT and data processing.

🤗 Hugging Face Models: Download the Llama 4 [Scout](https://huggingface.co/lmsys/sglang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-v1) & [Maverick](https://huggingface.co/lmsys/sglang-EAGLE3-Llama-4-Maverick-17B-128E-Instruct-v1) Eagle3 heads (w/o full model) for your projects.

## Roadmap

In the near future, we plan to extend SpecForge with the following support.

-   Support more model architectures, including the Kimi K2 and Qwen-3 MoE.
-   Integrate Vision-Language Models (VLM) into SpecForge.
-   Support more efficient training with better parallelism strategies and kernel optimization.

## Acknowledgement

We would like to express our heartfelt gratitude to the following teams and collaborators:

**Voltage Park** — our official infrastructure partner, for providing critical GPU resources that enabled the development of SpecForge.

**SGLang Team and Community** — Shenggui Li, Yikai Zhu, Fan Yin, Chao Wang, Shuai Shi, Yi Zhang, Yingyi Huang, Haoshuai Zheng, Yineng Zhang and many others.

**SafeAILab Team** — Yuhui Li, Hongyang Zhang and members — for their pioneering work on the Eagle3 algorithm.

We are especially grateful to Meituan for their strong support and contributions. And we would like to extend our sincere thanks to [Voltage Park](https://www.voltagepark.com/), our official infrastructure partner. As part of a formal collaboration with the SGLang team, Voltage Park provided critical GPU resources that empowered us to train and evaluate large-scale speculative decoding models efficiently and reliably. This partnership was instrumental in making SpecForge possible. We deeply appreciate Voltage Park’s mission to make cutting-edge AI infrastructure more accessible, and we look forward to continued collaboration as we push the boundaries of open-source LLM serving and optimization.
