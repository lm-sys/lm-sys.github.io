---
title: "Achieving Faster Open-Source Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM)"
author: "The SGLang Team"
date: "Jul 25, 2024"
previewImg: /images/blog/sglang_llama3/preview.png
---

At LMSYS.org, we've been running the [Chatbot Arena](https://chat.lmsys.org/) platform for over a year, serving millions of users. We know firsthand how crucial efficient serving is for AI products and research. Through our operational experiences and in-depth research, we've continuously enhanced the underlying serving systems, spanning from the high-level multi-model serving framework, [FastChat](https://github.com/lm-sys/FastChat/tree/main), to the efficient serving engine, [SGLang Runtime (SRT)](https://github.com/sgl-project/sglang/tree/main).

This post focuses on SGLang Runtime, a general-purpose serving engine for LLMs and VLMs. While existing options like TensorRT-LLM, vLLM, MLC-LLM, and Hugging Face TGI have their merits, we found them sometimes hard to use, difficult to customize, or lacking in performance. This motivated us to develop SGLang v0.2, aiming to create a serving engine that is not only user-friendly and easily modifiable but also delivers top-tier performance. While SGLang includes frontend language features, this post will focus solely on the backend runtime and use "SGLang" and "SGLang Runtime" interchangeably to refer to the runtime.

Compared to TensorRT-LLM and vLLM, SGLang Runtime consistently delivers superior or competitive performance in both online and offline scenarios, handling models from Llama-8B to Llama-405B, and on A100 and H100 GPUs, using FP8 and FP16. **SGLang consistently outperforms vLLM, achieving up to 3.8x higher throughput on Llama-70B. It also often matches or exceeds TensorRT-LLM, with up to 2.1x higher throughput on Llama-405B.** More importantly, SGLang is fully open-source, written in pure Python, with the core schedulers implemented in fewer than 4K lines of code.

SGLang is an open-source project licensed under the Apache 2.0 license. It has been used by LMSYS Chatbot Arena to support parts of the models, Databricks, several startups, and research institutes, generating trillions of tokens and enabling faster iterations. As it gradually matures from a research prototype, we invite the community to join us in creating the next-generation efficient engine.

## Benchmark Setup

We benchmark both offline and online use cases:

- **Offline:** We send 2K to 3K requests at once, measuring output throughput (tokens/second), defined as the number of output tokens divided by the total duration. We test synthetic datasets derived from the ShareGPT dataset. For example, I-512-O-1024 indicates a dataset with an average input of 512 tokens and an average output of 1024 tokens. The five tested datasets are: Dataset 1: I-243-O-770, Dataset 2: I-295-O-770, Dataset 3: I-243-O-386, Dataset 4: I-295-O-386, Dataset 5: I-221-O-201.
- **Online:** We send requests at rates ranging from 1 to 16 requests per second (RPS), measuring the median end-to-end latency. We use the synthetic dataset I-292-O-579.

We use vLLM 0.5.2 with default arguments and TensorRT-LLM with the recommended arguments and tuned batch sizes. The prefix cache is turned off for all engines. The purpose is to benchmark the base performance without any additional features, such as speculative decoding or caching.
We use OpenAI-compatible APIs to benchmark SGLang and vLLM, and the Triton interface for TensorRT-LLM.

More details and reproducible scripts are provided in Appendix A. For each model, we will first present the offline results and then present the online results.

<span style="color: red;"> Update (2024-07-25 8 PM PST) </span>: The dataset descriptions above are accurate but differ from the initial version of this blog post. We identified some issues in our synthetic data generation pipeline, so we corrected the dataset description to reflect the actual tested datasets. The comparison is still fair because all engines are benchmarked under the same conditions. The issues caused our benchmark to cover only the normal ShareGPT dataset distribution but miss long prompt cases. We are working on obtaining more benchmark results for longer prompts. However, we expect the speedup of SGLang to be less significant for long prompts since it primarily accelerates the decoding phase.

## Llama-8B on 1 x A100 (bf16)

Starting with the small model Llama-8B, the figure below shows the maximum output throughput each engine can achieve in offline settings across five different datasets. Both TensorRT-LLM and SGLang can achieve a throughput of approximately 4000 tokens per second, while vLLM falls behind.

<img src="/images/blog/sglang_llama3/8b_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

The online benchmark figure below shows a trend similar to the offline case. TensorRT-LLM and SGLang perform equally well and can sustain an RPS \> 10, while the latency of vLLM increases significantly at a high request rate.  

<img src="/images/blog/sglang_llama3/8b_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## Llama-70B on 8 x A100 (bf16)

Moving to the larger Llama-70B models with tensor parallelism on 8 GPUs, the trend is similar to the case with 8B. In the offline benchmark below, both TensorRT-LLM and SGLang can scale to a high throughput.   

<img src="/images/blog/sglang_llama3/70b_bf16_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

In the online figure below, TensorRT-LLM shows excellent latency performance thanks to its highly efficient kernel implementations and runtime.   

<img src="/images/blog/sglang_llama3/70b_bf16_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>


## Llama-70B on 8 x H100 (fp8)

Now, let us test the FP8 performance. Both vLLM and SGLang use FP8 kernels from CUTLASS. In the offline setting, SGLang’s batch scheduler is very efficient and can continue to scale the throughput with larger batch sizes, achieving the highest throughput in this case. Other systems cannot scale their throughput or batch sizes due to OOM, missing extensive manual tuning, or other overheads. This trend continues in the online case as well, with both SGLang and TensorRT achieving similar median latency.  

<img src="/images/blog/sglang_llama3/70b_fp8_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

<br>

<img src="/images/blog/sglang_llama3/70b_fp8_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## Llama-405B on 8 x H100 (fp8)

At last, we benchmark the performance on the largest 405B model. Because the model is large, most of the time is spent on the GPU kernels. The gap between different frameworks shrinks. The poor performance of TensorRT-LLM is probably due to the fact that the 405B model just came out, and the version we used in the provided image has not integrated some latest optimizations. In both online and offline cases, SGLang performs the best.

<img src="/images/blog/sglang_llama3/405b_fp8_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

<br>

<img src="/images/blog/sglang_llama3/405b_fp8_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## SGLang Overview

SGLang is a serving framework for large language models and vision-language models. It builds on and enhances many good designs from several open-source LLM serving engines, including [LightLLM](https://github.com/ModelTC/lightllm), [vLLM](https://blog.vllm.ai/2023/06/20/vllm.html), and [Guidance](https://github.com/guidance-ai/guidance). It leverages high-performance attention CUDA kernels from [FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) and integrates torch.compile inspired by [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/).

Additionally, we introduced innovations such as [RadixAttention](https://arxiv.org/abs/2312.07104) for automatic KV cache reuse and [compressed state machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/) for fast constrained decoding. SGLang is known for its highly efficient [batch scheduler](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers), which is implemented entirely in Python.
To make an apples-to-apples comparison, this blog tests the base performance of these serving engines with scenario- or workload-specific optimizations (like prefix caching and speculative decoding) turned off. The speedup in SGLang is achieved through proper engineering.
SGLang's efficient Python-based batch scheduler scales well, often matching or even outperforming closed-source implementations built with C++.

Table 1 compares various aspects of SGLang, TensorRT-LLM, and vLLM. In terms of performance, both SGLang and TensorRT-LLM excel. Regarding usability and customizability, SGLang's lightweight and modular core makes it easy to customize, whereas TensorRT-LLM's complex C++ tech stack and setup instructions make it harder to use and modify. SGLang's source code is fully open-source, while TensorRT-LLM is only partially open-source. In contrast, vLLM suffers from high CPU scheduling overhead.

Table. 1 Comparison

|  | SGLang | TensorRT-LLM | vLLM |
| :---- | :---- | :---- | :---- |
| Performance | Excellent | Excellent | Fair |
| Usability | Good | Poor | Good |
| Customizability | High | Low | Medium |
| Source Code Availability | Fully Open | Partially Open | Fully Open |
| Programming Language | Python | C++ | Python |

## What is Next

We're excited to share our latest benchmark results. While there's still more to do, this shows our philosophy of developing a simple, customizable, and high-performance serving engine is achievable. Stay tuned for new features like long context and MoE optimizations, and detailed technical walkthroughs. Join us in building the next-generation serving engine at [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang).

## Try Llama Serving

You can serve a Llama model easily with the following steps.

1. [Install](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#install) SGLang with pip, from source, or using Docker.
2. Launch a server:
    ```
    # Llama 8B
    python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct

    # Llama 405B
    python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
    ```
3. Send a request with the OpenAI-compatible API:
    ```
    curl http://localhost:30000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "default",
        "prompt": "Say this is a test",
        "max_tokens": 7,
        "temperature": 0
      }'
    ```
4. Run the benchmark:
    ```
    python3 -m sglang.bench_serving --backend sglang --num-prompts 1000
    ```

## The Team

This blog post is contributed by Liangsheng Yin, Yineng Zhang, Ying Sheng, and over 65 open-source [contributors](https://github.com/sgl-project/sglang/graphs/contributors). We thank the support from Databricks, and Ying Sheng’s work was done at Databricks. We especially thank Lianmin Zheng, Zihao Ye, and Horace He for their technical support, Matei Zaharia for his helpful advice, and Cody Yu for his feedback.

## Appendix A: Detailed Benchmark Setups

The instructions to reproduce the benchmark is at [sglang/benchmark/blog\_v0\_2](https://github.com/sgl-project/sglang/tree/main/benchmark/blog\_v0\_2).

For all benchmarks, we set \`ignore\_eos\` or \`min\_length/end\_id\` to ensure each engine outputs the same number of tokens. We tried using vLLM 0.5.3.post1, but it often crashes under high loads and seems to have similar or worse performance compared to vLLM 0.5.2 from our partial benchmarking. Therefore, we report results from vLLM 0.5.2 instead. While we are aware that different server configurations can significantly impact serving performance, we mostly use the default arguments in each engine to mimic the case of a normal user.

For the 8B and 70B models, we use the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [meta-llama/Meta-Llama-3-70B-Instruct](http://meta-llama/Meta-Llama-3-70B-Instruct) bf16 checkpoints, and the [neuralmagic/Meta-Llama-3-70B-Instruct-FP8](https://huggingface.co/neuralmagic/Meta-Llama-3-70B-Instruct-FP8) fp8 checkpoint. For the 405B models, we use dummy weights for all benchmarks. Since the TensorRT-LLM latest image r24.06 does not support fbgemm\_fp8 quantization in the official [meta-llama/Meta-Llama-3.1-405B-FP8](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-FP8) checkpoint, we use per-layer fp8 quantization in all frameworks and quantize all layers except lm\_head. We believe this provides a fair comparison among all engines. The A100 and H100 GPUs are 80GB SXM versions.
