---
title: "Full-Stack Performance Optimization of AR+DiT in SGL-Diffusion"
author: "Ascend Team"
date: "August 05, 2026"
previewImg: /images/blog/2026-08-05-glmImage-optimization/01-cover.png
type: blog
---

## TL;DR

- Replaces the HF backend with SRT to accelerate AR modeling and resolve parallelism conflicts, with dedicated TP for AR and SP for DiT
- Boosts hardware utilization via dynamic batching and enables early return for completed images
- Implements one-denoiser-per-device parallel DiT execution and overlaps AR & DiT workflows via buffered AR results

<div align="center">
  <img src="/images/blog/2026-08-05-glmImage-optimization/02-performance_result.png" alt="performance result" />
  <br>
  <em>Figure 1: Performance comparison.</em>
</div>

## 1. Background

Hybrid autoregressive–diffusion (AR+DiT) generation is a unified framework that combines AR modeling for global context with DiT methods for local detail refinement. It leverages AR transformers to capture long-range dependencies while DiT models iteratively refine outputs, ensuring improved quality and efficiency. GLM-Image exemplifies this trend: a 9B vision-language model first autoregressively generates semantic prior tokens from a text prompt, and a 7B DiT then denoises those tokens into a high-resolution image over 30–50 steps. This "plan-then-paint" design delivers SOTA results on knowledge-intensive and text-heavy visual tasks—such as posters, infographics, and precise typography—where end-to-end diffusion models often struggle.

Yet serving such hybrid pipelines efficiently in SGLang exposes a fundamental tension. In the native deployment, the AR encoder, DiT denoiser, and VAE decoder are chained inside a single monolithic worker process, which creates three critical pain points:

1. **Architectural coupling.** AR and DiT share the same process, weight-loading lifecycle, and scheduling domain. Scaling up one stage necessarily drags the other along; there is no way to provision resources independently. AR is fundamentally an LLM-decode workload—throughput scales with batch size and tensor parallelism (TP). DiT denoising, by contrast, is a large-tensor, per-image computation that favors spatial parallelism (SP) across cards and is most efficient at batch=1. A homogeneous deployment must pick a single strategy, guaranteeing that one stage is always sub-optimal.
2. **Low resource utilization under concurrency.** Without dynamic batching, concurrent requests are processed serially. End-to-end latency grows almost linearly with request concurrency, leaving a large fraction of the available compute idle.
3. **Mismatched resource allocation.** DiT achieves its best per-request latency at batch=1 per device, but bundling all devices into a single monolithic pipeline forces DiT to run in a multi-card spatial-parallel configuration even when throughput is the priority, resulting in underutilized hardware capacity.

To resolve these issues, we contributed three progressively staged PRs that evolve the system from a monolith to a fully decoupled, heterogeneous distributed architecture:

<div align="center">
  <img src="/images/blog/2026-08-05-glmImage-optimization/03-whole-pipeline.png" alt="the whole pipeline" />
  <br>
  <em>Figure 2: The whole pipeline for our optimization.</em>
</div>

## 2. SRT-ifying the AR Backend (PR #25381)

In order to reduce per-image generation latency, our analysis led us to replace the HF backend with SRT, culminating in PR #25381. It decouples the AR stage from the diffusion worker process into a standalone SRT service, so AR and DiT load weights separately, have decoupled scheduling lifecycles, and can scale independently. Meanwhile, the AR server can now configure TP on its own, no longer constrained by DiT's SP strategy.

<div align="center">
  <img src="/images/blog/2026-08-05-glmImage-optimization/04-glm_image_ar.png" alt="glm image AR" />
  <br>
  <em>Figure 3: The comparison between Original and Target.</em>
</div>

The AR vision-language encoder is spun up as a standard SGLang SRT service, invoked remotely by the Diffusion pipeline via a new `--srt-encoder-url` option. The AR server reuses SGLang's existing multimodal `sglang serve` capability (`srt/models/glm_image_vl.py` + `srt/multimodal/processors/glm_image.py`); the Diffusion side only adds an HTTP `/generate` branch to the `GlmImageAR` stage, and `VisionLanguageEncoderLoader` simply performs a `/health` check and returns the URL when `srt-encoder-url` is set, instead of calling `from_pretrained` to load weights. This "standalone server + HTTP" approach turns the giant task of "rewriting a VLM into SRT" into "reuse existing infrastructure + one HTTP call," dramatically reducing coupling and minimizing invasive changes to the diffusion pipeline.

**Performance gains** (please refer to [PR #25381 description](https://github.com/sgl-project/sglang/pull/25381) for reproducing):

| Configuration                               | NPUs | E2E latency (s)   | AR stage (s)      | Denoising (s) | Decoding (s) |
| ------------------------------------------- | ---- | ----------------- | ----------------- | ------------- | ------------ |
| Monolithic baseline (HF AR backend)         | 1    | 154.6             | 122.8             | 31.6          | 0.046        |
| **Decoupled SRT AR (DiT unchanged)**        | 1    | **78.3 (−49.4%)** | **46.6 (−62.1%)** | 31.6          | 0.035        |
| **4-NPU heterogeneous (TP=4 AR, SP=4 DiT)** | 4    | **35.2 (−77.2%)** | **26.1 (−78.8%)** | 9.0           | 0.009        |

> **Resource-matched comparison:** The `Decoupled SRT AR (DiT unchanged)` row shows the **software-only** speedup over the `monolithic baseline (HF AR backend)`.  
> **Combined scaling:** The `4-NPU heterogeneous (TP=4 AR, SP=4 DiT)` row shows the **combined software and hardware-scaling** result (software decoupling + 4× NPU parallelism) over the same `1-NPU monolithic baseline`.  
> Prior to PR #25381, because AR and DiT were constrained to share the same parallelism strategy, data is only available for the 4-NPU heterogeneous configuration.

The AR stage sees the most dramatic speedup: single-card 122.8 s → 26.1 s, a 78.8% reduction on 4 cards. Even at TP=1, SRT's graph execution, continuous batching, and memory reuse deliver a −62.1% gain over the naive transformers `generate`. Notably, the baseline 2-NPU setup uses SP to cut denoising by 46.7%, but AR actually slows by 4.1% — under the old path AR gets zero benefit from SP and even regresses due to communication overhead; only the SRT path lets AR truly leverage multi-card TP.

## 3. Dynamic Batching Adaptation and Early Return Support (PR #30683)

After the separation, AR and DiT still execute one request at a time, so latency grows linearly under high concurrency (issue #30634). To boost throughput in multi-input scenarios, we followed up with PR #30683. It packs concurrent requests into single forward passes, eliminating the idle compute caused by serial execution.

1. **Dynamic batching adaptation**: SGL-Diffusion already includes a generic dynamic batching infrastructure (introduced in [PR #18764](https://github.com/sgl-project/sglang/pull/18764)); our work extends this capability to GLM-Image by implementing the `supports_dynamic_batching` and `supports_native_grouped_requests` interfaces and associated pipeline logic. After evaluation, we apply batching only to the AR stage, as DiT per-step latency scales proportionally with batch size and yields no net throughput benefit.
2. **Support early return**: we add the `supports_sequential_dit_inference` variable and related functions to support early return when each output image is ready, instead of waiting for the entire batch to finish.

**Performance gains** (please refer to [PR #30683 description](https://github.com/sgl-project/sglang/pull/30683) for reproducing):

| Metric                              | BS1    | BS4                       | BS8                      | BS16                |
| ----------------------------------- | ------ | ------------------------- | ------------------------ | ------------------- |
| **Throughput (img/s)**              | 0.0388 | 0.0896                    | 0.1171                   | 0.1368              |
| Per‑request processing latency (s)¹ | 25.9   | 28.3 → 33.3 → 39.3 → 44.7 | 30.0 → 35.3 → ... → 68.3 | 33 → 39 → ... → 117 |
| AR stage per request (s)            | 20.17  | 5.65                      | 3.20                     | 1.85                |
| Peak NPU memory (MB)                | 28 163 | 28 046                    | 28 052                   | 28 062              |

**Notes:**  
¹ Processing latency is measured from batch dispatch to individual request completion. For BS4/BS8/BS16, the values represent a latency range across the batch: the first number corresponds to the fastest-finishing request, and the last to the slowest. Additional queueing wait time (≤14 ms in this test) is negligible.

## 4. Disaggregation and AR-to-DiT Fan-Out Architecture (PR #31320)

Fully decouple the two stages so AR and DiT each adopt the parallelism and deployment strategy that suits them best. The AR encoder favors large batch + TP (throughput-oriented); DiT denoising is optimal at batch=1 on a single NPU for both latency and throughput.

| Batch size | AR (s)       | Denoising, 30 step (s) | Denoising, 30 steps (s) |
| ---------- | ------------ | ---------------------- | ----------------------- |
| 1          | 20.4         | 0.407                  | 12.2                    |
| 2          | 21.3 (+4.4%) | 0.854 (+110%)          | 25.6 (+110%)            |
| 4          | 22.8 (+12%)  | 1.98 (+386%)           | 59.6 (+389%)            |
| 8          | 25.9 (+27%)  | 3.73 (+816%)           | 112.2 (+820%)           |
| 16         | 29.4 (+44%)  | 7.24 (+1679%)          | 217.3 (+1681%)          |
| 32         | 33.2 (+63%)  | 14.0 (+3339%)          | 420.6 (+3348%)          |

Then #31320 introduces a heterogeneous topology: one batched AR server + a pool of independent batch=1 denoisers. This achieves optimal system-wide hardware utilization in single-node scenarios.

<div align="center">
  <img src="/images/blog/2026-08-05-glmImage-optimization/05-fanout.png" alt="Disaggregated" />
  <br>
  <em>Figure 4: Final Deployment Architecture Diagram.</em>
</div>


SGL-Diffusion provides a generic disaggregation framework; PR #31320 adapts this framework to GLM-Image’s two-stage topology, enabling parallel DiT execution and pipeline overlap between AR generation and denoising. A key design choice is that only request metadata and CPU-side prior token IDs are transferred over ZMQ — no large tensors, latents, embeddings, or GPU buffers are sent across nodes — designed to keep communication overhead low.

**Performance gains** (please refer to [PR #31320 description](https://github.com/sgl-project/sglang/pull/31320) for reproducing):

| Configuration                                        | NPU | throughput (image/s) | Avg. E2E latency (s) |
| ---------------------------------------------------- | --- | -------------------- | -------------------- |
| AR(TP2) + monolithic sequential denoiser(BS28)       | 16  | 0.2                  | 90                   |
| AR(TP2) + disaggregated parallel denoiser(14 x BS=1) | 16  | **0.74**             | **37**               |

## 5. Acknowledgments

- Huawei Ascend Team

  We thank the Huawei Ascend NPU team for its continued contributions to GLM-Image optimization. In particular, we recognize Maksim Emelin (@[Makcum888e](https://github.com/Makcum888e)), Artem Savkin (@[OrangeRedeng](https://github.com/OrangeRedeng)), Egor Filimonov (@[ssshinigami](https://github.com/ssshinigami)), and Liang Zhen (@[ping1jing2](https://github.com/ping1jing2)).

  We also extend our thanks to Yuefeng Wu (@[ChefWu551](https://github.com/ChefWu551)) and Qianqian Zheng (@[AuFlow](https://github.com/AuFlow)) from CMB. They contributed to GLM-Image optimization on Ascend platform, improving stability and deployment efficiency.

- SGLang Community

  We are grateful to the broader SGLang community, including code review from Xiaoyu Zhang (@[BBuf](https://github.com/BBuf)), and initial discussion (issue #20032) and implementation (PR #18809) from Yuhao Yang (@[yhyang201](https://github.com/yhyang201)) and other contributors.

Finally, we thank the SGLang maintainers and reviewers for their careful guidance, the Zhipu AI team for open-sourcing the GLM-Image model and weights, and everyone who has contributed to SGL-Diffusion.

## 6. Appendix

### 6.1 GPU hardware reproduce command
1. single concurrency + local AR (baseline)
    <details>

    <summary>command</summary>

    ```shell
    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true

    sglang serve \
      --model-path "zai-org/GLM-Image" \
      --num-gpus 8 \
      --sp-degree 8 \
      --host 0.0.0.0 \
      --port 30052 \
      --scheduler-port 19655 \
      --output-path ./outputs

    # you can get fetch_images.py from https://github.com/user-attachments/files/29779516/longtext-bench.zip
    python fetch_images.py \
      --base-url http://127.0.0.1:30052/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 1
    ```
    </details>

2. single concurrency + separate AR
    <details>
    <summary>command</summary>

    ```shell
    sglang serve \
    --model-path zai-org/GLM-Image/vision_language_encoder/ \
    --tokenizer-path zai-org/GLM-Image/processor/ \
    --enable-multimodal \
    --cuda-graph-max-bs 1 \
    --disable-fast-image-processor \
    --tp-size 8 \
    --host 127.0.0.1 \
    --port 3828 \
    --mem-fraction-static 0.4

    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true

    sglang serve \
    --model-path zai-org/GLM-Image/ \
    --num-gpus 8 \
    --sp-degree 8 \
    --srt-encoder-url http://127.0.0.1:3828 \
    --srt-encoder-timeout 100 \
    --enable-batching-metrics \
    --host 127.0.0.1 \
    --port 30088

    python fetch_images.py \
    --base-url http://127.0.0.1:30088/v1 \
    --model GLM-image \
    --output-dir generated_images \
    --max-concurrency 1

    ```
    </details>

3. multi concurrency + separate AR
    <details>

    <summary>command</summary>

    ```shell

    sglang serve \
      --model-path zai-org/GLM-Image/vision_language_encoder/ \
      --tokenizer-path zai-org/GLM-Image/processor/ \
      --enable-multimodal \
      --cuda-graph-max-bs 28 \
      --disable-fast-image-processor \
      --tp-size 8 \
      --host 127.0.0.1 \
      --port 3828 \
      --mem-fraction-static 0.25
      
    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true
      
    sglang serve \
      --model-path zai-org/GLM-Image/ \
      --num-gpus 8 \
      --sp-degree 8 \
      --srt-encoder-url http://127.0.0.1:3828 \
      --srt-encoder-timeout 300 \
      --batching-mode dynamic \
      --batching-max-size 28 \ # or less bs
      --batching-delay-ms 30 \
      --enable-batching-metrics \
      --host 127.0.0.1 \
      --port 30088
      
    python fetch_images.py \
      --base-url http://127.0.0.1:30088/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 28
    ```
    </details>

4. multi concurrency + separate AR + disaggregation
    <details>

    <summary>command</summary>

    ```shell
    DISAGG_SERVER="tcp://127.0.0.1:19655"
    MODEL_PATH="zai-org/GLM-Image/"
    BASE_MASTER_PORT=29005

    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true

    for i in $(seq 1 7); do
        scheduler_port=$((19000 + i))
        master_port=$((BASE_MASTER_PORT + i))

        sglang serve \
            --model-path "$MODEL_PATH" \
            --disagg-role denoiser \
            --disagg-server-addr "$DISAGG_SERVER" \
            --scheduler-port "$scheduler_port" \
            --master-port "$master_port" \
            --num-gpus 1 \
            --base-gpu-id "$i" \
            --denoiser-sp 1 \
            --cfg-parallel-size 1 \
            --batching-max-size 1 \
            --warmup-mode off &
    done

    sglang serve \
      --model-path zai-org/GLM-Image/vision_language_encoder/ \
      --tokenizer-path zai-org/GLM-Image/processor/ \
      --enable-multimodal \
      --cuda-graph-max-bs 28 \
      --disable-fast-image-processor \
      --tp-size 1 \
      --host 0.0.0.0 \
      --port 30020 \
      --mem-fraction-static 0.8

    sglang serve \
      --model-path zai-org/GLM-Image/ \
      --disagg-role server \
      --srt-encoder-url http://127.0.0.1:30020 \
      --srt-encoder-timeout 300 \
      --denoiser-urls "tcp://127.0.0.1:19001;tcp://127.0.0.1:19002;tcp://127.0.0.1:19003;tcp://127.0.0.1:19004;tcp://127.0.0.1:19005;tcp://127.0.0.1:19006;tcp://127.0.0.1:19007" \
      --batching-mode dynamic \
      --batching-max-size 28 \
      --batching-delay-ms 30 \
      --enable-batching-metrics \
      --host 0.0.0.0 \
      --port 30052 \
      --scheduler-port 19655 \
      --output-path ./outputs
      
    python fetch_images.py \
      --base-url http://127.0.0.1:30052/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 28
    ```
    </details>


### 6.2 NPU hardware reproduce command
1. single concurrency + local AR (baseline)
    <details>

    <summary>command</summary>

    ```shell
    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true

    sglang serve \
      --model-type diffusion \
      --attention-backend fa \
      --model-path "zai-org/GLM-Image/" \
      --num-gpus 16 \
      --sp-degree 16 \
      --host 0.0.0.0 \
      --port 30052 \
      --scheduler-port 19655 \
      --output-path ./outputs
      
    python fetch_images.py \
      --base-url http://127.0.0.1:30052/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 1
    ```
    </details>

2. single concurrency + separate AR
    <details>
    <summary>command</summary>

    ```shell
    sglang serve \
      --model-path zai-org/GLM-Image/vision_language_encoder/ \
      --tokenizer-path zai-org/GLM-Image/processor/ \
      --enable-multimodal \
      --cuda-graph-max-bs 1 \
      --device npu \
      --attention-backend ascend \
      --disable-fast-image-processor \
      --tp-size 16 \
      --host 127.0.0.1 \
      --port 3828 \
      --mem-fraction-static 0.25
      
    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true
      
    sglang serve \
      --model-path zai-org/GLM-Image/ \
      --num-gpus 16 \
      --sp-degree 16 \
      --srt-encoder-url http://127.0.0.1:3828 \
      --srt-encoder-timeout 300 \
      --host 127.0.0.1 \
      --port 30088
      
    python fetch_images.py \
      --base-url http://127.0.0.1:30088/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 1
    ```
    </details>

3. multi concurrency + separate AR
    <details>

    <summary>command</summary>

    ```shell
    sglang serve \
      --model-path zai-org/GLM-Image/vision_language_encoder/ \
      --tokenizer-path zai-org/GLM-Image/processor/ \
      --enable-multimodal \
      --cuda-graph-max-bs 28 \
      --device npu \
      --attention-backend ascend \
      --disable-fast-image-processor \
      --tp-size 16 \
      --host 127.0.0.1 \
      --port 3828 \
      --mem-fraction-static 0.25
      
    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true
      
    sglang serve \
      --model-type diffusion \
      --attention-backend laser_attn \
      --model-path zai-org/GLM-Image/ \
      --num-gpus 16 \
      --sp-degree 16 \
      --srt-encoder-url http://127.0.0.1:3828 \
      --srt-encoder-timeout 300 \
      --batching-mode dynamic \
      --batching-max-size 28 \
      --batching-delay-ms 30 \
      --enable-batching-metrics \
      --host 127.0.0.1 \
      --port 30088
      
    python fetch_images.py \
      --base-url http://127.0.0.1:30088/v1 \
      --model GLM-image \
      --output-dir generated_images \
      --max-concurrency 28

    ```
    </details>

4. multi concurrency + separate AR + disaggregation
    <details>

    <summary>command</summary>

    ```shell
    DISAGG_SERVER="tcp://127.0.0.1:19655"
    MODEL_PATH="zai-org/GLM-Image/"
    BASE_MASTER_PORT=29005

    export SGLANG_CACHE_DIT_FN=2
    export SGLANG_CACHE_DIT_BN=1
    export SGLANG_CACHE_DIT_WARMUP=4
    export SGLANG_CACHE_DIT_RDT=0.4
    export SGLANG_CACHE_DIT_MC=4
    export SGLANG_CACHE_DIT_TAYLORSEER=true
    export SGLANG_CACHE_DIT_TS_ORDER=2
    export SGLANG_CACHE_DIT_ENABLED=true

    for i in $(seq 2 15); do
        scheduler_port=$((19001 + i))
        master_port=$((BASE_MASTER_PORT + i))

        sglang serve \
            --model-path "$MODEL_PATH" \
            --disagg-role denoiser \
            --disagg-server-addr "$DISAGG_SERVER" \
            --scheduler-port "$scheduler_port" \
            --master-port "$master_port" \
            --num-gpus 1 \
            --base-gpu-id "$i" \
            --denoiser-sp 1 \
            --cfg-parallel-size 1 \
            --batching-max-size 1 \
            --attention-backend fa \
            --warmup-mode off &
    done

    sglang serve \
    --model-path zai-org/GLM-Image/vision_language_encoder/ \
    --tokenizer-path zai-org/GLM-Image/processor/ \
    --enable-multimodal \
    --device npu \
    --attention-backend ascend \
    --cuda-graph-max-bs 28 \
    --disable-fast-image-processor \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30020 \
    --mem-fraction-static 0.8

    serve sglang serve \
      --model-path zai-org/GLM-Image/ \
      --disagg-role server \
      --srt-encoder-url http://127.0.0.1:30020 \
      --srt-encoder-timeout 300 \
      --denoiser-urls "tcp://127.0.0.1:19003;tcp://127.0.0.1:19004;tcp://127.0.0.1:19005;tcp://127.0.0.1:19006;tcp://127.0.0.1:19007;tcp://127.0.0.1:19008;tcp://127.0.0.1:19009;tcp://127.0.0.1:19010;tcp://127.0.0.1:19011;tcp://127.0.0.1:19012;tcp://127.0.0.1:19013;tcp://127.0.0.1:19014;tcp://127.0.0.1:19015;tcp://127.0.0.1:19016" \
      --batching-mode dynamic \
      --batching-max-size 28 \
      --batching-delay-ms 30 \
      --enable-batching-metrics \
      --host 0.0.0.0 \
      --port 30052 \
      --scheduler-port 19655 \
      --output-path ./outputs

    python fetch_images.py \
    --base-url http://127.0.0.1:30052/v1 \
    --model GLM-image \
    --output-dir generated_images \
    --max-concurrency 56
    ```
    </details>
