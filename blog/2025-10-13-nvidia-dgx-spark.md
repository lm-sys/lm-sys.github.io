---
title: "NVIDIA DGX Spark In-Depth Review: A New Standard for Local AI Inference"
author: "Jerry Zhou and Richard Chen"
date: "October 13, 2025"
previewImg: /images/blog/nvidia_dgx_spark/product_1.jpg
---

Thanks to NVIDIA’s early access program, we are thrilled to get our hands on the NVIDIA DGX™ Spark. It’s quite an unconventional system, as NVIDIA rarely releases compact, all-in-one machines that bring supercomputing-class performance to a desktop workstation form factor.

Over the past year, SGLang has been rapidly expanding its developer base in the datacenter segment, recognized by the inference community for its great performance. Successfully deploying DeepSeek with Prefill-decode Disaggregation (PD) and Expert Parallelism (EP) at large scale, running on both <a href="https://lmsys.org/blog/2025-05-05-large-scale-ep/" target="_blank">**96 NVIDIA H100 GPU clusters**</a> and the latest <a href="https://lmsys.org/blog/2025-09-25-gb200-part-2/" target="_blank">**GB200 NVL72 systems**</a>, SGLang has continually pushed the boundaries of large-scale inference performance and developer productivity.

Inspired by the capabilities of the DGX Spark, for the first time, SGLang is now expanding beyond the datacenter and into the consumer market, bringing its proven inference framework directly to developers and researchers everywhere. In this review, we’ll be taking a close look at this beautiful machine, from its exterior aesthetics to its performance and use cases.

> Also check out our video review <a href="https://youtu.be/-3r2woTQjec" target="_blank">here</a>.

![](/images/blog/nvidia_dgx_spark/product_1.jpg)

## Exterior

The DGX Spark is a gorgeous piece of engineering. It features a full-metal chassis with a sleek champagne-gold finish. Both the front and rear panels are built with metal foam, reminding me of the design of NVIDIA DGX A100 and H100.

Around the back, the DGX Spark offers an impressive array of connectivity options: a power button, four USB-C ports (with the leftmost supporting up to **240 W of power delivery**), an HDMI port, a **10 GbE RJ-45 Ethernet port**, and **two QSFP ports driven by NVIDIA ConnectX-7 NIC capable of up to 200 Gbps**. These interfaces allow two DGX Spark units to be connected together, allowing them to run even larger AI models.

The use of USB Type-C for power delivery is a particularly interesting design choice, one that’s virtually unheard of on other desktop machines. Comparable systems like the Mac Mini or Mac Studio rely on the standard C5/C7 power connector, which is far more secure but also bulkier. NVIDIA likely opted for USB-C to keep the power supply external, freeing up valuable internal space for the cooling system. The trade-off, however, is that you’ll want to be extra careful not to accidentally tug the cable loose.

![](/images/blog/nvidia_dgx_spark/product_2.jpg)

## Hardware Capabilities

On the hardware side, the DGX Spark packs remarkable performance for its size and power envelope. At its core is the NVIDIA GB10 Grace Blackwell Superchip, designed specifically for this device. It integrates 10 Cortex-X925 performance cores and 10 Cortex-A725 efficiency cores, for a total of 20 CPU cores.

On the GPU side, the GB10 delivers up to **1 PFLOP of sparse FP4 tensor performance**, placing its AI capability roughly between that of an RTX 5070 and 5070 Ti. The standout feature is its **128 GB of coherent unified system memory**, shared seamlessly between the CPU and GPU. This unified architecture allows the DGX Spark to load and run large models directly without the overhead of system-to-VRAM data transfers. With the help of its dual QSFP Ethernet ports with an aggregate bandwidth of 200 Gb/s, two DGX Spark units can be connected together to operate as a small cluster, enabling distributed inference of even larger models. According to NVIDIA, two interconnected DGX Sparks can handle models with up to **405 billion parameters in FP4**.

However, the only downside of this machine lies in memory bandwidth, the unified memory is LPDDR5x, offering up to **273 GB/s**, shared across both CPU and GPU. As we’ll see later, this limited bandwidth is expected (and empirically shown) to be the key bottleneck in AI inference performance. Nonetheless, the 128GB of memory enables DGX Spark to run models that are too large for most desktop systems.

![](/images/blog/nvidia_dgx_spark/product_3.jpg)

## Performance

We benchmarked several open-weight large language models on the DGX Spark using both **SGLang** and **Ollama**. Our findings show that while the DGX Spark can indeed load and run very large models, such as **GPT-OSS 120B** and **Llama 3.1 70B,** these workloads are best suited for **prototyping and experimentation** rather than production. The DGX Spark truly shines when serving **smaller models**, especially when **batching** is utilized to maximize throughput.

### Methodology

> ⚠️ **Note:** Since software support for the DGX Spark is still in its early stages, the benchmark results presented in this section may become outdated as future software updates improve performance and compatibility.

#### Test Devices

We prepared the following systems for benchmarking:

* **NVIDIA DGX Spark**  
* **NVIDIA RTX PRO™ 6000 Blackwell Workstation Edition**  
* **NVIDIA GeForce RTX 5090 Founders Edition**  
* **NVIDIA GeForce RTX 5080 Founders Edition**  
* **Apple Mac Studio (M1 Max, 64 GB unified memory)**  
* **Apple Mac Mini (M4 Pro, 24 GB unified memory)**

#### Benchmark Models

We evaluated a variety of open-weight large language models using two frameworks, **SGLang** and **Ollama**, as summarized below:

| Framework | Batch Size | Models & Quantization |
| :---- | :---- | :---- |
| **SGLang** | 1–32 | Llama 3.1 8B (FP8)<br>Llama 3.1 70B (FP8)<br>Gemma 3 12B (FP8)<br>Gemma 3 27B (FP8)<br>DeepSeek-R1 14B (FP8)<br>Qwen 3 32B (FP8) |
| **Ollama** | 1 | GPT-OSS 20B (MXFP4)<br>GPT-OSS 120B (MXFP4)<br>Llama 3.1 8B (q4\_K\_M / q8\_0)<br>Llama 3.1 70B (q4\_K\_M)<br>Gemma 3 12B (q4\_K\_M / q8\_0)<br>Gemma 3 27B (q4\_K\_M / q8\_0)<br>DeepSeek-R1 14B (q4\_K\_M / q8\_0)<br>Qwen 3 32B (q4\_K\_M / q8\_0) |

We also tested **speculative decoding (EAGLE3) with SGLang** on some of the models listed above. We excluded models that exceeded the available RAM or VRAM capacity of the target machine.

### Results

> Full benchmark results can be found <a href="https://docs.google.com/spreadsheets/d/1SF1u0J2vJ-ou-R_Ry1JZQ0iscOZL8UKHpdVFr85tNLU/edit?usp=sharing" target="_blank">here</a>.

#### Overall Performance

While the DGX Spark demonstrates impressive engineering for its size and power envelope, its raw performance is understandably limited compared to full-sized discrete GPU systems.

For example, running **GPT-OSS 20B (MXFP4)** in **Ollama**, the Spark achieved **2,053 tps prefill / 49.7 tps decode**, whereas the **RTX Pro 6000 Blackwell** reached **10,108 tps / 215 tps,** roughly **4× faster**. Even the **GeForce RTX 5090** delivered **8,519 tps / 205 tps**, confirming that the Spark’s unified LPDDR5x memory bandwidth is the main limiting factor.

However, for smaller models, particularly **Llama 3.1 8B**, the DGX Spark held its own. With **SGLang** at batch 1, it achieved **7,991 tps prefill / 20.5 tps decode**, scaling up linearly to **7,949 tps / 368 tps** at batch 32, demonstrating excellent batching efficiency and strong throughput consistency across runs.

#### Strength in Compact, Unified-Memory Workloads

One of the DGX Spark’s defining strengths lies in its **128 GB of coherent unified memory**, which allows both CPU and GPU to access the same address space.

This enables large models, such as **Llama 3.1 70B**, **Gemma 3 27B**, or even **GPT-OSS 120B,** to load **directly into memory** without the traditional system-to-VRAM transfer overhead. Despite its compact form factor, the Spark successfully ran **Llama 3.1 70B (FP8)** at **803 tps prefill / 2.7 tps decode**, which is remarkable for a workstation that sits quietly on a desk.

This unified-memory design makes DGX Spark particularly valuable for **prototyping**, **model experimentation**, and **edge-AI research**, where seamless memory access is often more useful than raw TFLOPs.

#### Speculative Decoding Acceleration

To further explore performance optimization on the DGX Spark, we enabled **speculative decoding** using **EAGLE 3** within **SGLang**. This technique allows a smaller “draft” model to propose multiple tokens ahead, while the larger target model verifies them in parallel.

With speculative decoding enabled, we observed up to a **2× speed-up** in end-to-end inference throughput compared to standard decoding across multiple models, such as **Llama 3.1 8B**.

This improvement effectively mitigates part of the unified-memory bandwidth limitation and demonstrates that **software-level innovations** such as speculative decoding can meaningfully enhance inference performance on compact, bandwidth-constrained systems like the DGX Spark.

#### Efficiency and Thermal Design

The DGX Spark maintains sustained throughput across high-intensity tests without thermal throttling. Even under full load, e.g., **SGLang DeepSeek-R1 14B (FP8)** at batch 8 achieving **2,074 tps / 83.5 tps**, fan noise and temperature remained stable, highlighting NVIDIA’s excellent **metal-foam cooling design** and well-optimized **power delivery system**.

Its **USB-C power input** (up to 240 W) and external PSU allow for greater thermal headroom inside the chassis, a clear advantage for long-running workloads compared to compact consumer systems like the Mac Mini or Mac Studio, which showed thermal drop-off in similar tests.

#### Summary

In short, the DGX Spark is **not built to compete head-to-head** with full-sized Blackwell or Ada-Lovelace GPUs, but rather to bring the DGX experience into a compact, developer-friendly form factor.  
It’s an ideal platform for:

* **Model prototyping and experimentation**  
* **Lightweight on-device inference**  
* **Research on memory-coherent GPU architectures**

It’s a **gorgeous, well-engineered mini supercomputer** that trades raw power for accessibility, efficiency, and elegance, and in those areas, it absolutely shines.

![](/images/blog/nvidia_dgx_spark/product_4.jpg)

## Use Cases

### SGLang Model Serving

The DGX Spark comes with Docker preinstalled, allowing you to serve open-weight models via SGLang with just a single command:

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:spark \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --host 0.0.0.0 --port 30000
```

Replace `<secret>` with your own Hugging Face access token.

#### Enabling Speculative Decoding (EAGLE3)

To enable **speculative decoding** using **EAGLE3**, simply run the following command:

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --env "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
    --ipc=host \
    lmsysorg/sglang:spark \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --host 0.0.0.0 --port 30000 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 32 \
    --mem-fraction 0.6 \
    --cuda-graph-max-bs 2 \
    --dtype float16
```

With speculative decoding enabled, SGLang can leverage a smaller draft model to predict multiple tokens ahead, effectively **doubling inference throughput** compared to standard decoding.

#### Sending Requests via the OpenAI-Compatible API

Once SGLang successfully initializes, you can interact with your model through OpenAI-compatible API endpoints:

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "How many letters are there in the word SGLang?"
            }
        ]
    }'
```

![](/images/blog/nvidia_dgx_spark/demo_1.jpg)

### Chatting with Local Model

Once you have **SGLang** set up and serving a model, you can easily connect it to **Open WebUI** to chat with any open-weight model you like. Open WebUI provides a sleek, browser-based interface that’s fully compatible with OpenAI-style APIs, meaning it works seamlessly with your local SGLang server. With just a quick configuration pointing to your DGX Spark’s endpoint, you can interact with models such as **Llama 3**, **Gemma 3**, or **DeepSeek-R1** directly from your browser, no cloud dependencies, no latency, and complete control over your data.

![](/images/blog/nvidia_dgx_spark/demo_2.jpg)

### Coding with Local Model

One of the most practical ways to utilize the DGX Spark is as a **local coding assistant,** completely offline and secure.

By combining **Zed**, a modern AI-integrated code editor, with **Ollama**, you can run **GPT-OSS 20B** locally to power code completion, inline chat, and smart refactoring without relying on the cloud.

#### Step 1\. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Step 2\. Pull GPT-OSS 20B for Coding

```bash
ollama pull gpt-oss:20b
```

#### Step 3\. Integrate Zed with Ollama

Install Zed:

```bash
curl -f https://zed.dev/install.sh | sh
```

Zed automatically detects local models served by Ollama, allowing you to start using the built-in chat assistant immediately after launching the editor.

![](/images/blog/nvidia_dgx_spark/demo_3.jpg)

## Conclusion

The **NVIDIA DGX Spark** is a fascinating glimpse into the future of personal AI computing. It takes what was once reserved for data centers: large memory, high-bandwidth Ethernet interconnects, and Blackwell-class performance, and distills it into a compact, beautifully engineered desktop form factor. While it doesn’t rival full-size DGX servers or discrete RTX GPUs in raw throughput, it shines in accessibility, efficiency, and versatility.

From running **SGLang** and **Ollama** for local model serving, to experimenting with **speculative decoding (EAGLE3)**, to exploring distributed inference through **dual-Spark clustering**, the platform proves itself as more than just a miniature supercomputer. It’s a developer’s sandbox for the next era of AI.

The NVIDIA DGX Spark isn’t built to replace cloud-scale infrastructure; it’s built to **bring AI experimentation to your desk**. Whether you’re benchmarking open-weight LLMs, developing inference frameworks, or building your own private coding assistant, the Spark empowers you to do it all locally, quietly, elegantly, and with NVIDIA’s unmistakable engineering polish.
