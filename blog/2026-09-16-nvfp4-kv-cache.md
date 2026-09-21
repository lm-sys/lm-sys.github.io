---
title: "Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache"
author: "SGLang, Qwen, and NVIDIA teams"
date: "September 16, 2026"
previewImg: /images/blog/nvfp4-kv-cache/kv-cache-layout.svg
type: blog
---

The KV cache is a fundamental building block of the modern LLM inference system. The context from multiple conversation rounds in agent sessions is cached as *keys* and *values* (KV) in GPU memory, allowing the model to reuse them during subsequent decoding steps. When generating each new token, the new query token attends to the relevant past KV tokens from the KV cache. As context windows grow, storing and reading KV cache puts increasing pressure on GPU memory capacity and bandwidth.

There are two complementary ways to address this pressure: expand the storage available to the cache, or reduce the amount of data stored per token.

GPU memory provides fast access to active KV data, but its capacity is limited. A serving system handling many users or long-running sessions cannot keep every session's cache resident indefinitely. [Hierarchical KV Caching (HiCache)](https://docs.sglang.io/docs/advanced_features/hicache) extends the cache hierarchy into host memory and distributed storage, allowing the system to retain more context beyond the GPU.

[KV cache quantization](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache) addresses the other side of the problem. Storing K/V values in FP8 instead of BF16 roughly halves the data footprint and reduces the bytes read during decoding. This can improve performance when KV reads are a bandwidth bottleneck. The tradeoff is numerical precision: lower-bit representations introduce quantization error, and that error must remain small enough to preserve useful model behavior. FP8 KV caching is already widely used; moving to 4 bits is especially challenging because the quantization error is larger.

NVIDIA introduced the [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) data format with the Blackwell architecture (Figure 1). NVFP4 combines four-bit E2M1 values with two levels of scaling: an FP8 scale for each block of 16 values and an additional FP32 tensor-level scale. This gives the format finer local control over dynamic range than a single global scale, helping limit quantization error. Blackwell also provides hardware support for working with the format. Its compute efficiency and flexibility make it suitable for KV cache quantization.

![Figure 1. NVFP4 two-level scaling per-block and per-tensor quantization strategy](/images/blog/nvfp4-kv-cache/nvfp4-scaling.gif)

*Figure 1. NVFP4 two-level scaling per-block and per-tensor quantization strategy*

The rest of the post walks through our NVFP4 KV cache implementation in SGLang, and then examines its accuracy and performance on various workloads.

## NVFP4 KV Implementation in SGLang

The implementation connects the SGLang paged KV cache system to three attention paths: initial prefill, chunked prefill or extend, and decode (Figure 2). SGLang manages the cache and its auxiliary buffers. Each attention path has a different data flow:

![Figure 2. NVFP4 KV cache implementation in SGLang](/images/blog/nvfp4-kv-cache/kv-cache-layout.svg)

*Figure 2. NVFP4 KV cache implementation in SGLang*

**Path 1: Initial prefill**

Initial prefill processes a prompt without a cached prefix. After QKV projection and positional encoding, attention directly operates on the current prompt's BF16 Q and FP8 K/V tensors. The same current K/V is also quantized from FP8 to NVFP4 and written into the persistent cache for subsequent extend and decode steps.

In this path, quantized NVFP4 KV data is written to the KV cache, but attention directly consumes FP8 KV before quantization. No data is read from the KV cache in this phase.

**Path 2: Chunked prefill / Extend**

*Chunked prefill* processes only part of a prompt at a time. An *extend* operation similarly adds new tokens to an existing context. In both cases, attention needs two sources of K/V: the cached prefix and the current chunk.

For the cached prefix, SGLang gathers the relevant NVFP4 entries, dequantizes them, and populates an FP8 workspace. One pair of K/V workspace buffers is shared across layers, avoiding a full FP8 workspace allocation for every layer.

The current chunk takes a separate route. Its KV data is copied into the FP8 workspace after the QKV projection. The attention then operates on the KV from the FP8 workspace, similar to initial prefill. The current chunk’s KV is also quantized and stored in the NVFP4 cache for future steps.

**Path 3: Decode**

During the decode phase, the attention kernel reads NVFP4 KV directly from the NVFP4 KV cache, performs on-the-fly dequantization to FP8 inside the kernel, and then uses the dequantized values in subsequent computations. Therefore, the decode step does not require a separate dequantization operation.

Unlike in prefill, the query sequence length is very short during decode, while the KV sequence length is often much larger. In this case, attention performance is entirely memory-read-bound. Performing dequantization inside the kernel avoids the extra memory round trip of reading and writing the entire KV cache in a standalone dequantization operation, significantly improving performance.

## NVFP4 KV Decode Attention Kernel Implementation

The current implementation on SM120 consumes BF16 queries and NVFP4 KV, and performs matrix multiplications in BF16.

As explained in Figure 3, the kernel loads packed K/V tiles and their E4M3 block scales into shared memory, then unpacks and scales the values in registers. It uses `cvt.rn.bf16x2.e2m1x2` to convert pairs of E2M1 values to BF16. The E4M3 block scales are also cast to BF16. Packed multiply instructions apply scaling factors to pairs of values. Then, the attention matrix multiplications operate on the dequantized K/V. The global K/V scales are supplied through the attention BMM scaling parameters.

![Figure 3. NVFP4 KV attention decode kernel design](/images/blog/nvfp4-kv-cache/decode-kernel.svg)

*Figure 3. NVFP4 KV attention decode kernel design*

Each block of 16 NVFP4 values occupies eight bytes of packed data plus one byte of block-scale metadata. Compared with 16 bytes of FP8 data, that is:

$$
\frac{8 + 1}{16} = 0.5625
$$

In other words, the packed data and block scales require about 56% of the storage of FP8 for the same number of K/V values. This calculation excludes the small global-scale metadata and other pool or workspace overheads. It explains the opportunity to reduce GPU memory read traffic and improve decode efficiency.

## Experiments and Results

### Accuracy

We evaluated FP8 and NVFP4 KV cache on Qwen3.5-397B-A17B and Qwen3.8-27B on various accuracy benchmarks: GSM8K, GPQA-Diamond, AIME 2025, and SWE-bench (only on Qwen3.8-27B). Both KV cache configurations used the same FP8 model weights. In SGLang, NVFP4 KV storage is selected with `--kv-cache-dtype nvfp4`. The detailed reproduction steps are in the Appendix.

GSM8K was evaluated once, with 1,311 scored questions after reserving eight examples for the few-shot prompt. GPQA-Diamond and AIME were evaluated twice because of higher task variances, and the figures below report aggregate accuracy across both rounds.

For Qwen3.5-397B-A17B, the measured differences were very small (Figure 4). NVFP4 produced one fewer correct GSM8K answer, a difference of approximately 0.08 percentage points, while aggregate correct counts were identical on GPQA-Diamond and AIME 2025.

![Figure 4. Accuracy benchmark on Qwen3.5-397B-A17B](/images/blog/nvfp4-kv-cache/qwen35-397b-accuracy.svg)

*Figure 4. Accuracy benchmark on Qwen3.5-397B-A17B*

Qwen3.8-27B showed task-dependent accuracy differences (Figure 5). NVFP4 scored 0.31 percentage points lower than FP8 on GSM8K and 1.01 points lower on GPQA-Diamond, while matching FP8 at 98.33% on AIME 2025 in thinking/xhigh mode. On SWE-bench Verified, NVFP4 resolved 381 of 500 tasks (76.20%), compared with 389 (77.80%) for FP8—a difference of 1.60 percentage points.

![Figure 5. Accuracy benchmark on Qwen3.8-27B](/images/blog/nvfp4-kv-cache/qwen38-27b-accuracy.svg)

*Figure 5. Accuracy benchmark on Qwen3.8-27B*

The larger model was less sensitive in these measurements, but two models and a small set of tasks do not establish a general relationship between model size and quantization tolerance. The results are encouraging for NVFP4 KV caching, with task-specific validation remaining important for deployment.

## Performance

We benchmarked Qwen3.8-27B on one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. The workloads used fixed input lengths of 32,768, 163,840, and 1,048,576 tokens, with exactly 1,024 output tokens per request. Both configurations used the same FP8 weights. Radix caching was disabled for these fixed-length tests. The reproduction steps can be found in the Appendix.

The 1M workload is a performance-only experiment using an explicit context-length override; the model's native context limit is 262,144 tokens. At 1M, the NVFP4 run used a static-memory fraction of 0.75 to leave room for temporary prefill workspace, compared with 0.90 for FP8. Both used 0.90 at the shorter context lengths.

We used two comparison points to separate performance at matched concurrency from the benefit of supporting more concurrent requests.

### Decode Performance

**Decode at matched concurrency: Iso-concurrency**

The iso-concurrency tests hold the achieved decode concurrency constant across FP8 and NVFP4 KV. As shown in Figure 6, the scheduler reached 44, 10, and 1 decode-resident requests at 32K, 160K, and 1M, respectively, in both configurations.

![Figure 6. Iso-concurrency decode performance on Qwen3.8-27B](/images/blog/nvfp4-kv-cache/decode-iso-concurrency.svg)

*Figure 6. Iso-concurrency decode performance on Qwen3.8-27B*

NVFP4 KV improved peak-batch decode throughput by approximately 26–30% across these workloads. With matched peak batch sizes, the result is consistent with reduced KV-read traffic contributing to faster decoding.

**Decode under capacity-driven load: Iso-capacity**

The iso-capacity tests increase the offered concurrency to exercise the additional requests that the NVFP4 configuration can keep resident. Here, “iso-capacity” refers to a capacity-driven comparison on the same GPU, where the number of available KV-token slots differs between KV formats.

As shown in Figure 7, at 32K, achieved concurrency increased from 44 requests with FP8 KV to 70 with NVFP4 KV. At 160K, it increased from 10 to 15, and at 1M from one to two. Peak-batch decode throughput improved by 37.37%, 57.75%, and 78.46%, respectively.

![Figure 7. Iso-capacity decode performance on Qwen3.8-27B](/images/blog/nvfp4-kv-cache/decode-iso-capacity.svg)

*Figure 7. Iso-capacity decode performance on Qwen3.8-27B*

These gains combine compact KV reads with the ability to run a larger batch. Higher concurrency can improve utilization beyond attention, including matrix multiplications elsewhere in the model. The Pareto curve in Figure 8 further demonstrates the advantage of NVFP4 KV. At similar concurrency, NVFP4 KV has both higher output throughput and interactivity. Also, NVFP4 KV has the additional ability to serve at higher concurrency, further enhancing output throughput.

![Figure 8. Pareto curve of 32K/1K ISL/OSL on Qwen3.8-27B](/images/blog/nvfp4-kv-cache/decode-throughput.svg)

*Figure 8. Pareto curve of 32K/1K ISL/OSL on Qwen3.8-27B*

### Prefill Performance

Unlike decode attention, where KV cache reads are the main bottleneck, the prefill phase is compute-bound. Since our recipe only changes the KV data type to NVFP4 and the compute data type remains unchanged, prefill attention itself does not show a performance gain. As shown in Figure 9, at matched concurrency, mean time to first token (TTFT) increased by only 0.20–0.40% with NVFP4 in these runs. The slight slowdown is mainly due to the KV cache quantization or dequantization overhead.

![Figure 9. Prefill performance measured by TTFT on Qwen3.8-27B](/images/blog/nvfp4-kv-cache/prefill-ttft.svg)

*Figure 9. Prefill performance measured by TTFT on Qwen3.8-27B*

Under capacity-driven load, mean TTFT instead decreased by 0.74–10.72%. TTFT includes admission, prefill, and queueing, so this improvement should not be attributed solely to faster prefill computation. A larger resident cache allows more requests to make progress without waiting for another request to release its KV slots. At the 1M point, for example, NVFP4 could accommodate two complete requests while FP8 could accommodate only one.

The decode throughput gains above also should not be read as equivalent end-to-end speedups. With very long prompts, prefill can dominate total wall time even when decode becomes substantially faster.

### Agentic Workloads

Agentic workloads are more complex than fixed-input, fixed-output benchmarks. They alternate between extend and decode across multiple turns, often carrying long histories that include tool outputs and intermediate reasoning. The value of a KV cache depends on how much of that history remains available when the next turn arrives.

If a needed prefix has been evicted from all usable cache tiers, the server must recompute it. For contexts containing hundreds of thousands of tokens, that can add substantial first-token latency and consume resources that could otherwise serve new work.

NVFP4 creates an opportunity to retain more of that working set on the GPU. The packed-data calculation suggests an idealized capacity ratio of about 1.78× versus FP8, before accounting for workspace and other memory overheads. Actual usable capacity is workload- and configuration-dependent. Keeping more prefixes resident can reduce repeated prefill work when memory pressure would otherwise cause eviction.

This benefit depends on the workload's reuse pattern and the serving configuration. Hierarchical caching and KV quantization can also complement one another: one extends the cache hierarchy, while the other reduces the footprint of its contents.

We tested the AgentX benchmark under the following configuration: on a node with 8 NVIDIA RTX 6000D GPUs, we tested Qwen3.5-397B-A17B with TP8 parallelism. We swept concurrency from 1 to 16 for both NVFP4 and FP8 KV. The benchmark window duration was 1,200 seconds.

As the results show (Figure 10), at low concurrency (C \<= 8), NVFP4 KV and FP8 KV have comparable performance. When the concurrency grows beyond 12, the throughput and interactivity of FP8 KV drop drastically, while the throughput of NVFP4 KV keeps increasing. The input token cache rate analysis (Figure 11) provides evidence that, at high concurrency, the FP8 KV cache hit rate is much lower than that of NVFP4 KV.

![Figure 10. AgentX performance on Qwen3.5-397B-A17B](/images/blog/nvfp4-kv-cache/agentx-throughput.svg)

*Figure 10. AgentX performance on Qwen3.5-397B-A17B*

![Figure 11. AgentX input token cache rate analysis on Qwen3.5-397B-A17B](/images/blog/nvfp4-kv-cache/agentx-cache-hit-rate.svg)

*Figure 11. AgentX input token cache rate analysis on Qwen3.5-397B-A17B*

## Limitations and Ongoing Items

The current NVFP4 KV support in SGLang is experimental. We are actively improving and iterating. Below is a summary of our current support and what’s in progress:

- **GPU Architecture support**: We currently support SM12x and SM100/SM103. Enabling additional architectures is on the way. The roadmap can be found [here](https://github.com/sgl-project/sglang/issues/29913).
- **Model support**: We support GQA models and Sparse MLA models. We are also working on adapting to more model types, such as Sparse GQA.
- **Accuracy Improvement**: The experiments we performed did not make use of the per-tensor FP32 global scale of NVFP4 (we use 1.0 for simplicity). Proper calibration may reduce numeric overflow/underflow and may further close the accuracy gap between NVFP4 and FP8 KV. Furthermore, we are also experimenting with more 4-bit recipes other than plain NVFP4 KV, which may improve the accuracy for certain models.

## Conclusion

In this article, we shared the motivation and implementation of NVFP4 KV cache. After that, we compared the accuracy and performance of NVFP4 KV cache with the commonly used FP8 KV cache on various models and benchmarks.

The results show that NVFP4 KV cache may speed up the decode of long contexts because of reduced memory footprint and higher serving concurrency. For agentic workloads, the reduced KV cache size creates an opportunity to retain more reusable context and improve the KV cache hit rate, bringing significant improvement to the overall system throughput.

## Acknowledgement

This blog was accomplished through close collaboration among the SGLang, Qwen, and NVIDIA teams. Below are the contributors:

SGLang - for the design discussions and code review: Baizhou Zhang, Brayden Zhong, Mick Qian

Qwen - for the FP4 KV Cache recipe design: Yizhong Cao, Yuxin Zhou, Yi Zhang, Chengruidong Zhang, Yuyan Luo, Jianwei Zhang

NVIDIA - for the FP4 KV cache recipe design, kernel and optimization: Tian Zheng, Sam Li, Perkz Zheng, Cheng Hang, Zane Sun, Triston Cao, Mengdi Wang, Zihua Wu, Meng Wang, Enwei Zhu, Ajit Mistry, Po-Han Huang, Jack Chen, Gary Ji, Chandler Zhou, Julien Demouth

<details id="appendix-reproduction-steps" style="margin-top: 1.875rem;">
<summary style="cursor: pointer; font-size: 1.875rem; line-height: 2.25rem; font-weight: 400;">Appendix: Reproduction steps</summary>

These examples use Qwen/Qwen3.5-397B-A17B-FP8 on 8 × RTX PRO 6000 Blackwell Server Edition (96 GB each), with TP8. Model weights remain FP8; only the KV-cache dtype changes between fp8\_e4m3 and nvfp4.

### Setup

Use a CUDA environment with SM120 support. Install SGLang with evaluation dependencies at the benchmark commit `c8b56b1f44d5c5370f47470ee490da3b04375e1c`:

```bash
pip install --upgrade "sglang[test] @ git+https://github.com/sgl-project/sglang.git@c8b56b1f44d5c5370f47470ee490da3b04375e1c#subdirectory=python"
```

Launch the server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --trust-remote-code --tp-size 8 --kv-cache-dtype nvfp4 \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend trtllm_mha \
  --disable-radix-cache --context-length 262144 \
  --mem-fraction-static 0.8 --reasoning-parser qwen3 \
  --random-seed 20260826 --watchdog-timeout 3600 \
  --host 127.0.0.1 --port 8005
```

Wait for the server to become ready, then run the following commands in another terminal. Model weights and evaluation datasets download automatically.

### Accuracy: GSM8K, GPQA-Diamond, AIME 2025

```bash
python -m sglang.test.run_eval --base-url http://127.0.0.1:8005 \
  --eval-name gsm8k --num-examples 1319 --num-shots 8 \
  --num-threads 1319 --max-tokens 10240 --repeat 1 \
  --temperature 0.6 --top-p 0.95 --top-k 20

python -m sglang.test.run_eval --base-url http://127.0.0.1:8005 \
  --eval-name gpqa --num-examples 198 --num-threads 512 \
  --max-tokens 81920 --repeat 2 \
  --temperature 0.6 --top-p 0.95 --top-k 20

python -m sglang.test.run_eval --base-url http://127.0.0.1:8005 \
  --eval-name aime25 --num-examples 30 --num-threads 512 \
  --max-tokens 81920 --repeat 2 \
  --temperature 0.6 --top-p 0.95 --top-k 20
```

GSM8K uses eight test examples as demonstrations, leaving 1,311 scored questions. GPQA evaluates 198 questions twice; AIME evaluates all 30 AIME 2025 I/II questions twice.

### Throughput: 32K input / 1K output

Use the same server. This example sends 128 requests at concurrency 32, with exactly 32,768 input tokens and 1,024 output tokens per request. The client uses synthetic token IDs and ignores EOS by default.

```bash
# Warm up before measurement.
python -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:8005 \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 32768 --random-output-len 16 --random-range-ratio 1 \
  --num-prompts 1 --max-concurrency 1 --request-rate inf \
  --temperature 0 --seed 20260826 --flush-cache

python -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:8005 \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 32768 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 128 --max-concurrency 32 --request-rate inf \
  --temperature 0 --seed 20260826 --flush-cache \
  --output-details --output-file nvfp4-32k-1k.jsonl
```

For the FP8 baseline, restart the server with `--kv-cache-dtype fp8_e4m3`. Concurrency 32 is an example comparison point, not a peak-throughput claim.

### AgentX

**1. Prepare the environment**

Install SGLang and the SemiAnalysis AgentX harness in separate environments at these revisions:

SGLang: c8b56b1f44d5c5370f47470ee490da3b04375e1c

[AgentX harness](https://github.com/SemiAnalysisAI/agentx-harness): 56a0cf70f4c0359454ee4bd15a17770b541a3e3e

Download the model and ensure access to the harness dataset semianalysis\_cc\_traces\_weka\_062126\_256k.

**2. Start SGLang**

Run in the SGLang environment. Start with fp8\_e4m3; after its sweep, stop the server and repeat with nvfp4.

```bash
export MODEL_PATH=/path/to/Qwen3.5-397B-A17B-FP8
export KV_DTYPE=fp8_e4m3  # Repeat with nvfp4.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
  --trust-remote-code --tp-size 8 \
  --kv-cache-dtype "$KV_DTYPE" \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend trtllm_mha \
  --context-length 262144 --mem-fraction-static 0.8 \
  --reasoning-parser qwen3 --random-seed 20260827 \
  --watchdog-timeout 3600 \
  --enable-metrics --enable-cache-report \
  --host 127.0.0.1 --port 8005
```

Wait for readiness and verify /server\_info: TP8, the requested KV dtype, hierarchical cache disabled, and radix cache enabled.

**3. Run the sweep**

In a separate terminal using the AgentX environment, set MODEL\_PATH and KV\_DTYPE to match the running server. Use a fresh results directory for each sweep.

```bash
export MODEL_PATH=/path/to/Qwen3.5-397B-A17B-FP8
export KV_DTYPE=fp8_e4m3  # Match the running server.
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=4
export AIPERF_HTTP_TCP_USER_TIMEOUT=1000000

set -euo pipefail
for C in 1 4 8 12 13 14 16; do
  OUT="results/${KV_DTYPE}/c${C}"
  mkdir -p "$OUT"

  # Require a successful cold reset before each point.
  RESET=$(curl -fsS -X POST http://127.0.0.1:8005/flush_cache)
  [[ "${RESET,,}" == *"cache flushed"* ]] || { echo "$RESET"; exit 1; }
  curl -fsS http://127.0.0.1:8005/metrics > "$OUT/metrics_before.prom"

  aiperf profile \
    --scenario inferencex-agentx-mvp \
    --url http://127.0.0.1:8005 \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --tokenizer "$MODEL_PATH" --endpoint-type chat \
    --public-dataset semianalysis_cc_traces_weka_062126_256k \
    --concurrency "$C" --benchmark-duration 1200 --streaming \
    --system-idle-gap-cap-seconds 10.0 \
    --trajectory-start-min-ratio 0.0 --trajectory-start-max-ratio 1.0 \
    --cache-bust first_turn_prefix --use-server-token-count \
    --extra-inputs ignore_eos:true --random-seed 20260707 \
    --burst-phase-starts --ui simple --artifact-dir "$OUT/profile"

  curl -fsS http://127.0.0.1:8005/metrics > "$OUT/metrics_after.prom"
  # Clean up remaining server work only after profiling and export finish.
  curl -fsS -X POST http://127.0.0.1:8005/abort_request \
    -H 'Content-Type: application/json' -d '{"abort_all":true}'
done
```

</details>
