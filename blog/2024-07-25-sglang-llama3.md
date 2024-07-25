---
title: "Achieving Faster Open-Source Llama3 Serving with SGLang (vs. TensorRT-LLM, vLLM)"
author: "The SGLang Team"
date: "Jul 25, 2024"
previewImg: /images/blog/sglang_llama3/preview.png
---

At LMSYS.org, we've been running the large-scale online LLM chat platform, [Chatbot Arena](https://chat.lmsys.org/), for over a year, serving millions of users. We know firsthand how crucial efficient serving is for AI products and research. Through our operational experiences and in-depth research and engineering, we've continuously enhanced our underlying serving systems, spanning from the high-level multi-model serving framework, [FastChat](https://github.com/lm-sys/FastChat/tree/main), to our efficient serving engine, [SGLang](https://github.com/sgl-project/sglang).

In this blog post, we want to share our latest progress and benchmark results on the SGLang Runtime by comparing its performance to other popular options. There are already several popular serving engine options, such as TensorRT-LLM, vLLM, MLC-LLM, and Hugging Face TGI. However, our experience with these solutions revealed that they are often hard to use, difficult to customize, or suffer from compromised performance. This motivated us to develop SGLang v0.2, aiming to create a serving engine that is not only user-friendly and easily modifiable but also delivers top-tier performance.

Compared to TensorRT-LLM and vLLM, SGLang consistently delivers superior or competitive performance in both online and offline scenarios, on models ranging from Llama-8B to Llama-405B, and on A100 and H100 GPUs, using FP8 and FP16. **SGLang consistently outperforms vLLM, achieving up to 3.8x higher throughput on Llama-70B. It also often matches or exceeds TensorRT-LLM, with up to 2.1x higher throughput on Llama-405B.** More importantly, SGLang is fully open-source, written in pure Python, with the core schedulers implemented in fewer than 4K lines of code.

SGLang is an open-source project under the Apache 2.0 license. It has been used by LMSYS Chatbot Arena, Databricks, several startups, and research institutes, generating trillions of tokens and enabling faster iterations. As it gradually matures from a research prototype, we invite the community to join us in building the next-generation efficient engine.

## Benchmark Setup

We benchmark both offline and online use cases.

- For the offline case, we send 2K to 3K requests at once, measuring output throughput (tokens/second), which is defined as the number of output tokens divided by the total duration. We test using the ShareGPT dataset and several synthetic datasets. We use In\[2048, 4096\]-Out\[256, 512\] to indicate a synthetic dataset with input lengths sampled from a uniform distribution \[2048, 4096\] and output lengths from \[256, 512\].  
- For the online case, we send requests at a rate ranging from 1 to 16 requests per second (RPS), measuring the median end-to-end latency. We use a synthetic dataset In\[512, 4096\]-Out\[128, 1024\].

We use vLLM 0.5.2 with default arguments and TensorRT-LLM with the recommended arguments and tuned batch sizes. The prefix cache is turned off for all engines. More details and reproducible scripts are provided in Appendix A. For each model, we will first present the offline results and then present the online results.

## Llama-8B on 1 x A100 (bf16)

We will start with the small model Llama-8B. The figure below shows the maximum output throughput each engine can achieve under offline settings on five different datasets. TensorRT-LLM and SGLang can both reach a throughput of around 4000 tokens per second, while vLLM lags behind.  

<img src="/images/blog/sglang_llama3/8b_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

The online benchmark figure below shows a trend similar to the offline case. TensorRT-LLM and SGLang perform equally well and can sustain an RPS \> 10, while the latency of vLLM increases significantly at a high request rate.  

<img src="/images/blog/sglang_llama3/8b_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## Llama-70B on 8 x A100 (bf16)

Moving to the larger Llama-70B models with tensor parallelism on 8 GPUs, the trend is similar to the case with 8B. In the offline benchmark below, both TensorRT-LLM and SGLang can scale to a high throughput.   

<img src="/images/blog/sglang_llama3/70b_bf16_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

In the online figure below, TensorRT-LLM shows excellent latency performance thanks to its highly efficient kernel implementations and runtime.   

<img src="/images/blog/sglang_llama3/70b_bf16_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>


## Llama-70B on 8 x H100 (fp8)

Now, let us test the FP8 performance. Both vLLM and SGLang use FP8 kernels from CUTLASS. In the offline setting, SGLang’s batch scheduler is very efficient and can continue to scale the throughput with larger batch sizes, achieving the highest throughput in this case. Other systems cannot scale their throughput or batch sizes due to OOM, missing manual tuning, or other overheads. This trend continues in the online case as well, with both SGLang and TensorRT achieving similar median latency.  

<img src="/images/blog/sglang_llama3/70b_fp8_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

<br>

<img src="/images/blog/sglang_llama3/70b_fp8_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## Llama-405B on 8 x H100 (fp8)

At last, we benchmark the performance on the largest 405B model. Because the model is large, most of the time is spent on the GPU kernels. The gap between different frameworks shrinks. The poor performance of TensorRT-LLM is probably due to the fact that the 405B model just came out, and the version we used in the provided image has not integrated some latest optimizations.  

<img src="/images/blog/sglang_llama3/405b_fp8_throughput.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

<br>

<img src="/images/blog/sglang_llama3/405b_fp8_latency.svg" style="display: flex; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 70%;"></img>

## SGLang Overview

SGLang is a serving framework for large language models and vision-language models. It builds on and enhances many good designs from several open-source LLM serving engines, including [LightLLM](https://github.com/ModelTC/lightllm), [vLLM](https://blog.vllm.ai/2023/06/20/vllm.html), and [Guidance](https://github.com/guidance-ai/guidance). It leverages high-performance attention CUDA kernels from [FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) and integrates torch.compile inspired by [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/).

Additionally, we introduced innovations such as [RadixAttention](https://arxiv.org/abs/2312.07104) for automatic KV cache reuse and [compressed state machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/) for fast constrained decoding. SGLang is known for its highly efficient [batch scheduler](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers), which is implemented entirely in Python.

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

We're excited to share our latest benchmark results. While there's still more to do, this shows our philosophy of developing a simple, customizable, and high-performance serving engine is achievable. Stay tuned for new features like long context and MoE optimizations, and detailed technical walkthroughs. Join us in building the next-generation serving engine at https://github.com/sgl-project/sglang.

## Try Llama Serving

1. [Install](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#install) SGLang with pip, from source, or using Docker.
2. Launch a server:
    ```
    # Llama 3B
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

This blog post is contributed by Liangsheng Yin, Yineng Zhang, Ying Sheng, and over 65 open-source [contributors](https://github.com/sgl-project/sglang/graphs/contributors). We thank the support from Databricks, and Ying Sheng’s work was done at Databricks. We especially thank Lianmin Zheng and Zihao Ye for their technical support, Matei Zaharia for his helpful advice, and Cody Yu for his feedback.

## Appendix A: Detailed Benchmark Setups

The instructions to reproduce the benchmark is at [sglang/benchmark/blog\_v0\_2](https://github.com/sgl-project/sglang/tree/main/benchmark/blog\_v0\_2).

For all benchmarks, we set \`ignore\_eos\` or \`min\_length/end\_id\` to ensure each engine outputs the same number of tokens. We use OpenAI-compatible APIs to benchmark SGLang and vLLM, and the Triton interface for TensorRT-LLM. We tried using vLLM 0.5.3.post1, but it often crashes under high loads and seems to have similar or worse performance compared to vLLM 0.5.2 from our partial benchmarking. Therefore, we report results from vLLM 0.5.2 instead. While we are aware that different server configurations can significantly impact serving performance, we mostly use the default arguments in each engine to mimic the case of a normal user.

For the 8B and 70B models, we use the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [meta-llama/Meta-Llama-3-70B-Instruct](http://meta-llama/Meta-Llama-3-70B-Instruct) bf16 checkpoints, and the [neuralmagic/Meta-Llama-3-70B-Instruct-FP8](https://huggingface.co/neuralmagic/Meta-Llama-3-70B-Instruct-FP8) fp8 checkpoint. For the 405B models, we use dummy weights for all benchmarks. Since the TensorRT-LLM latest image r24.06 does not support fbgemm\_fp8 quantization in the official [meta-llama/Meta-Llama-3.1-405B-FP8](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-FP8) checkpoint, we use per-layer fp8 quantization in all frameworks and quantize all layers except lm\_head. We believe this provides a fair comparison among all engines. The A100 and H100 GPUs are 80GB SXM versions.
