---
title: "Optimizing DeepSeek-V4.1 on H200 with SGLang PD Disaggregation"
author: "Bingjia Wang, Siyuan Chen — Ant International AI Infra Team"
date: "November 5, 2026"
previewImg: ../public//images/blog/deepseek-v41-h200-pd/combined-performance.png
type: blog
---

## TL;DR

We optimized DeepSeek-V4.1-Flash on an eight-GPU H200 SGLang deployment with a 1P1D configuration: four GPUs for Prefill and four for Decode. The work followed the bottlenecks found in profiling: shape-sensitive Indexer GEMMs, unnecessary late-layer token computation, MoE output communication, full-context scoring in candidate consumers, and repeated score-epilogue memory traffic.

- **Align the BF16 score width:** scoring latency decreased by approximately 40% in the microbenchmarks, including padding overhead.
- **Score candidate positions directly:** move candidate filtering before the dot product. The PR's reported E2E reduction reaches 22.1%.
- **Fuse the score epilogue:** combine ReLU, head weighting, head reduction, and FP32 conversion while preserving the tested BF16 rounding behavior. The reported E2E reduction reaches 18.2%.
- **Use larger Prefill chunks:** the notebook's comparison of 8,192- and 16,384-token chunks shows lower E2E latency in seven of eight cases, with reductions of up to 15.8%. The associated Decode graph-metadata work is tracked in PR #46777.
- **Return MoE outputs with reduce-scatter:** remove full-output replication on the eligible DeepSeek-V4.1 CP return path. Mean profiled MoE output communication time fell by 19.4%–21.7% per rank.
- **Enable bounded replay with CP and DSpark:** repair compact-output gathering so eager CP Prefill can use late-layer tail computation. The reported TTFT reduction reaches 46.7%.

## Background: Find the Bottleneck in the PD Pipeline

The [DeepSeek-V4.1 day-0 post](/blog/2026-09-10-deepseek-v41/) describes the model's shared compressed KV, shared Indexer candidates, and sliding-window attention. Here we focus on the H200 serving path and the engineering work needed to make those features useful under Prefill/Decode disaggregation.

The deployment separates prompt processing from token generation. The Prefill worker uses TP4 / CP4 / EP4; the Decode worker uses TP4 / DP4 / EP4. TP shards tensor computation, CP partitions the prompt computation, EP distributes experts, and DP supplies independent attention replicas on the Decode side. These parallel dimensions share the four GPUs assigned to each worker.

![Eight-H200 PD serving topology with four GPUs assigned to each worker](../public//images/blog/deepseek-v41-h200-pd/pd-topology.svg)

*Figure 1. The measured 1P1D topology. The handoff includes the KV and model state required to continue generation. Client-visible TTFT includes queueing and handoff costs as well as Prefill computation.*

The notebook first identified a supply problem on the P side. Requests accumulated in the Prefill queue even at modest concurrency, while the Decode worker's available cache capacity did not explain the queueing. We therefore concentrated on reducing Prefill service time before considering a different P/D ratio.

The evaluated H200 configuration uses the Torch BF16 Prefill Indexer scoring path. Its bottlenecks differ from the Blackwell path that uses DeepGEMM for the relevant Indexer operations. A representative early profile attributed **209.47 ms, or 39.06% of the reported per-forward breakdown, to the Indexer**. MoE computation and NCCL each accounted for approximately another 18%.

![Early Prefill profile, highlighting Indexer computation and the remaining major costs](../public//images/blog/deepseek-v41-h200-pd/initial-profile.png)

*Figure 2. Redrawn from the H200 notebook, §2.3–2.4. This is a representative Prefill profile, separate from request E2E latency. NCCL attribution includes waiting for other ranks. The Attention category combines the reported projection and core-attention categories; the smaller remaining compute categories are grouped together.*

### Measurement Setup

We developed the optimizations in this article on top of our earlier [PR #40323](https://github.com/sgl-project/sglang/pull/40323).

The model weights are available at [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on Hugging Face.

The PRs describe a workload with nominal **65,536-token inputs and 512-token outputs**, using reported cache-hit labels of **50%, 70%, and 90%**. The corresponding concurrency levels are 4 and 8; 8 and 16; and 8, 16, 24, and 32, respectively. The labels are retained as reported; they do not specify the exact number of cached tokens in every request.

Each baseline or optimized serving configuration in this article is benchmarked once per workload case, with 512 requests and 10 warmup requests. Reported mean request latencies summarize the requests within that single run. PR #47247 also provides additional trace samples. The initial Indexer experiments explicitly disable Prefill BCG.

In the benchmark screenshots, TTFT and E2E latency are in milliseconds; TPOT is in milliseconds per output token. We calculate reductions from the displayed absolute values as `(baseline − optimized) / baseline`. These calculated reductions can differ slightly from the percentages printed in the original screenshots.

The experiments follow a shared development baseline. Sections 1–3 each start from commit `d0a75654368a470d8484e3ea45b06e500fee57af` and evaluate their respective optimizations independently. Section 4 starts from that same commit with all three Indexer optimizations cherry-picked together. Section 5 also uses the starting revision as its baseline, while Section 6 uses the resulting Section 4 configuration. Checkpoint and chunk settings follow the individual experiments described below, and complete per-run revision records are not available for every table. The chunk-size comparison in Section 4 comes from §2.5 of the H200 experiment notebook.

### Model Operations Behind the Optimizations

DeepSeek-V4.1 combines a **128-position sliding window (SWA)** with sparse access to compressed history. Selected source layers produce compressed KV and Indexer keys; consumers reuse the latest source's stored history while maintaining their own local window KV. The final compressed-KV source is layer 20. Layers 21–39 can therefore read that history without producing a new compressed representation. These layer roles are described in the [model introduction](/blog/2026-09-10-deepseek-v41/).

The **Indexer selects which compressed history positions attention will read**. On the BF16 path discussed here, it computes per-head query–key dot products, applies ReLU and query-dependent head weights, and sums over heads before top-k selection. These are retrieval scores: the subsequent attention operation computes its own attention weights over the selected compressed KV and local window, then combines values to produce the attention output.

Sharing has two distinct meanings here. KV consumers share compressed KV and Indexer keys from a source layer. Layer 20 publishes up to 2,048 candidate blocks of eight compressed positions, while selecting its own top-512 from all visible positions. Later Indexer source layers share those **candidate block IDs** and compute their own scores and final top-k inside that domain. They do not copy the publisher's scores or final top-k. Layers without their own Indexer reuse the most recent selection.

**Bounded replay** uses the fixed late-layer compressed history to limit Prefill computation in layers 21–39 to each request's trailing window. It reconstructs those layers' local SWA KV and retained hidden states; the compressed history is already available. **DSpark**, the speculative block drafter, additionally uses hidden states captured from selected target-model layers to populate its draft KV state and condition proposals. Its auxiliary states must refer to the same token positions as the retained model outputs. Section 6 explains the approximation at the replay boundary and the CP output fix. The [DSpark Prefill injection path](https://github.com/BJWang-ant/sglang/blob/f274754cb12fdefba511c05137257410df00dea5/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L887) consumes both hidden states and their token indices.

![Model execution stages and the locations of the six optimizations](../public//images/blog/deepseek-v41-h200-pd/model-execution.svg)

*Figure 3. Model execution, separate from GPU allocation in Figure 1. Section numbers identify the six changes. The Indexer branches represent alternative scoring implementations; the detailed compatibility of alignment, candidate scoring, and epilogue fusion is explained after Section 3. MoE follows attention within each model layer. The late-layer tail is an opt-in Prefill approximation, and the Decode metadata change belongs to Section 4.*

## 1. Align the BF16 Prefill Indexer Score Width

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af` and include the same CP request-row mapping fix. The optimized run adds the score-width alignment change from PR #45794. This is an independent comparison with 8,192-token Prefill chunks.

[PR #45794](https://github.com/sgl-project/sglang/pull/45794) addresses a surprising observation: nearby Indexer score widths could have very different GEMM costs. The notebook's cross-rank trace showed much faster execution at a 64-aligned width than at neighboring unaligned widths. Almost identical dimensions and similar strides still resulted in different kernel behavior.

The scoring operation takes query rows across heads and compares them with the request's visible compressed keys. Its large intermediate has shape `[query_rows, heads, visible_keys]`. Padding the physical key extent can improve the GEMM shape even though it adds a small amount of arithmetic. For example, one extra column changes width 28,799 to 28,800, a multiple of 64. The measured improvement is shape-dependent. More favorable kernel dispatch, tiling, or memory alignment could explain it, but the available traces do not establish the precise hardware cause.

For CUDA BF16 keys with at least 1,024 visible positions, the patch pads the dequantized key tensor to `ceil(visible_keys / 64) × 64`. It does this once per request and reuses that tensor across query chunks. Scores are sliced back to the logical extent before causal masking and top-k selection. The chunk-size calculation uses the padded width, keeping the existing intermediate-memory budget intact.

![Score-width alignment before and after, retaining the logical selection range](../public//images/blog/deepseek-v41-h200-pd/score-alignment.svg)

*Figure 4. Physical padding improves the GEMM shape while selection still operates on the original logical positions. Padded columns never become eligible candidates.*

The PR's scoring microbenchmark includes the epilogue and the padding cost:

<div role="region" aria-label="Indexer scoring microbenchmark" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 640px;">

| Query rows | Logical score width | Baseline | Aligned | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 582 | 28,799 | 4.113 ms | 2.462 ms | 40.1% |
| 292 | 57,343 | 4.088 ms | 2.472 ms | 39.5% |
| 582 | 28,800 | 2.443 ms | 2.445 ms | Approximately unchanged |

</div>
</div>

The patch also reconstructs rank-local request-row mappings when CP supplies no explicit `req_rows`. Both the baseline and optimized runs include that mapping fix, isolating the score-width change in this experiment.

![Original serving benchmark table from PR 45794](../public//images/blog/deepseek-v41-h200-pd/pr-45794-source.png)

*Figure 5. Original [PR #45794 performance screenshot](https://github.com/sgl-project/sglang/pull/45794). Optimized values appear before their baseline columns; all latency values are milliseconds, and TPOT is milliseconds per token.*

Across the eight displayed cases, TTFT falls by **12.3%–18.2%** and E2E latency by **4.4%–14.8%**. At the 50% cache-hit label and concurrency 4, TTFT changes from **6,245 to 5,108 ms**, while E2E changes from **7,402 to 6,327 ms**. The smaller E2E gain in the 90%/concurrency-8 case illustrates why a scoring-kernel improvement should be checked against the entire request path.

![CP rank 0 scoring trace with width alignment](../public//images/blog/deepseek-v41-h200-pd/pr-45794-source-cp-rank0-after.png)

*Figure 6. [PR #45794 trace](https://github.com/sgl-project/sglang/pull/45794), with score-width alignment.*

![CP rank 0 scoring trace before width alignment](../public//images/blog/deepseek-v41-h200-pd/pr-45794-source-cp-rank0-before.png)

*Figure 7. [PR #45794 trace](https://github.com/sgl-project/sglang/pull/45794), before score-width alignment.*

The CP rank 0 screenshots compare the scoring kernel at the same logical shape: its duration falls from **1.88 ms to 362 µs**.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 \
--chunked-prefill-size 8192 --max-prefill-tokens 8192 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 \
--tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

## 2. Score Only Candidate Positions in Indexer Consumers

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af`. The optimized run adds candidate-only scoring from PR #46003, evaluated independently of the Indexer optimizations in Sections 1 and 3. Both runs use 8,192-token Prefill chunks.

[PR #46003](https://github.com/sgl-project/sglang/pull/46003) removes work outside the consumer's selection domain.

DeepSeek-V4.1 shares candidate blocks across Indexer source layers. A publisher produces candidate block IDs; later consumers select their final top-k only within those published blocks, using their own queries, head weights, and scores. The shared object is the **candidate domain**. The original Torch consumer still scored every visible key before gathering candidate scores. Positions that could never be selected consumed GEMM work and intermediate memory bandwidth.

For a small example, suppose there are **16 logical compressed Indexer key positions**, numbered 0–15, and the publisher supplies blocks `[1, 3]`. Use block size 4 for this illustration; the model path uses blocks of eight. These indices address compressed Indexer keys, not original prompt tokens, whose relationship to compressed positions depends on compression.

<div role="region" aria-label="Candidate block mapping example" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 600px;">

| Published block | Eligible logical key positions | Compact score columns |
| ---: | --- | --- |
| 1 | 4, 5, 6, 7 | 0, 1, 2, 3 |
| 3 | 12, 13, 14, 15 | 4, 5, 6, 7 |

</div>
</div>

The consumer is already restricted to `[4…7, 12…15]`. The old implementation computes 16 scores per head, then gathers these eight; the new implementation directly computes the eight eligible scores. If compact top-k selects column **5**, its candidate-list index is `5 // 4 = 1`, which refers to published block **3**, and its in-block offset is `5 % 4 = 1`. The logical key position is therefore `3 × 4 + 1 = 13`, followed by the request's mapping from logical key position 13 to a physical KV slot. Treating compact column 5 as key position 5 would select the wrong history.

The patch moves the restriction before scoring. A Triton kernel uses the block IDs to load the corresponding keys directly, producing a compact per-head score tensor of width `C` rather than the full visible width `L`. ReLU, head weighting, and head reduction retain the reference operations. Causal and invalid-candidate masking operate on compact scores before top-k.

![Candidate-aware key loading and compact score selection](../public//images/blog/deepseek-v41-h200-pd/candidate-scoring.svg)

*Figure 8. Candidate filtering moves ahead of the dot product. The publisher, candidate budget, and consumer selection domain remain unchanged.*

Selected compact columns must then be translated back correctly. Each pick identifies a candidate block and an offset within that block; `block_id × block_size + offset` reconstructs the request-local key position, which is then mapped to the physical KV slot. An index in the compact score array is not itself a KV-cache address.

The optimized path requires CUDA BF16 operands, a head dimension of 128, at most 64 heads, and **a candidate width no greater than half the visible key length**. Wider candidate sets use the original path. This avoids paying the indirect-load cost when candidate restriction removes too little scoring work.

![Original serving benchmark table from PR 46003](../public//images/blog/deepseek-v41-h200-pd/pr-46003-source.png)

*Figure 9. Original [PR #46003 performance screenshot](https://github.com/sgl-project/sglang/pull/46003). This independent experiment uses the same starting development revision as Sections 1 and 3.*

The displayed TTFT reductions are **22.9%–32.5%**, with **8.3%–22.1% lower E2E latency**. At the 70% label and concurrency 8, TTFT changes from **8,061 to 5,944 ms** and E2E from **9,228 to 7,191 ms**. At the 90% label and concurrency 16, TTFT changes from **2,846 to 1,922 ms**, a **32.5%** reduction.

![CP rank 0 trace with candidate-only scoring](../public//images/blog/deepseek-v41-h200-pd/pr-46003-source-rank0-after.png)

*Figure 10. [PR #46003 trace](https://github.com/sgl-project/sglang/pull/46003), with candidate-only scoring.*

![CP rank 0 trace before candidate-only scoring](../public//images/blog/deepseek-v41-h200-pd/pr-46003-source-rank0-before.png)

*Figure 11. [PR #46003 trace](https://github.com/sgl-project/sglang/pull/46003), before candidate-only scoring.*

At the same logical problem shape, the displayed CP rank 0 scoring-kernel duration falls from **1.88 to 1.147 ms**. The optimized kernel computes a smaller candidate domain, so its physical output shape differs from the baseline's dense score tensor.

The local validation reported exact score/top-k agreement for tested model-style FP4 inputs and matching Indexer/KV-slot writeback in the integration cases. The attached 1,319-question GSM8K result is **96.74%**, with zero request errors.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 \
--chunked-prefill-size 8192 --max-prefill-tokens 8192 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 \
--tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

## 3. Fuse the BF16 Indexer Score Epilogue

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af`. The optimized run adds score-epilogue fusion from PR #45807, evaluated independently of the Indexer optimizations in Sections 1 and 2. Both runs use 8,192-token Prefill chunks.

[PR #45807](https://github.com/sgl-project/sglang/pull/45807) keeps the existing score GEMM and reduces the work that follows it.

The original sequence materializes the large BF16 per-head scores, applies ReLU, multiplies by head weights, reduces across heads, and converts the result to FP32. Separate elementwise operations repeatedly read and write `[query_rows, heads, context]` tensors. The notebook attributed approximately **104.8 ms per representative forward** to the multiplication, clamp/ReLU, and reduction operations, making this a substantial target after inspecting the GEMM itself.

The patch replaces that epilogue with one Triton kernel for the supported 32-head CUDA BF16 path. The GEMM's per-head output still exists; the later full-sized intermediates disappear.

![Separate score postprocessing operations compared with one fused Triton epilogue](../public//images/blog/deepseek-v41-h200-pd/epilogue-fusion.svg)

*Figure 12. Fusion removes intermediate memory traffic and launches after GEMM. It preserves the numerical effects of the reference operations.*

Numerical behavior matters because top-k consumes these scores. Two nearly tied keys can swap order at the selection threshold after a small rounding change, altering which KV entries attention reads. The kernel preserves BF16 rounding after multiplication and reduction and uses four interleaved accumulators to reproduce the tested Torch/CUDA 32-head accumulation order. A final conversion to FP32 cannot restore precision already lost at an earlier BF16 cast. Keeping the same mathematical formula while changing these rounding points or the summation order can therefore change selection. The reported parity applies to the tested reference implementation and supported layouts.

Context-dependent row and head strides remain runtime arguments, avoiding a separate kernel specialization for each context-dependent stride. The implementation uses the Torch fallback for unsupported devices, dtypes, head counts, widths, and layouts. In particular, width-one and head-contiguous reduction layouts require their own fallback because the reference reduction order differs.

![Original serving benchmark table from PR 45807](../public//images/blog/deepseek-v41-h200-pd/pr-45807-source.png)

*Figure 13. Original [PR #45807 performance screenshot](https://github.com/sgl-project/sglang/pull/45807). The fused epilogue comparison retains the existing GEMM.*

Across the displayed cases, TTFT falls by **19.8%–25.8%** and E2E by **4.3%–18.2%**. At the 50% label and concurrency 4, TTFT changes from **6,245 to 4,850 ms** and E2E from **7,402 to 6,099 ms**. At the 90% label and concurrency 16, TPOT remains **3.17 ms/token**; it increases in the other seven displayed cases.

![CP rank 0 GEMM and fused epilogue trace](../public//images/blog/deepseek-v41-h200-pd/pr-45807-source-rank0-after.png)

*Figure 14. [PR #45807 trace](https://github.com/sgl-project/sglang/pull/45807), showing GEMM followed by the fused epilogue.*

![CP rank 0 GEMM and separate score postprocessing trace](../public//images/blog/deepseek-v41-h200-pd/pr-45807-source-rank0-before.png)

*Figure 15. [PR #45807 trace](https://github.com/sgl-project/sglang/pull/45807), showing GEMM and separate score postprocessing.*

The selected CP rank 0 trace interval, covering GEMM and score postprocessing, falls from **4.243 to 2.26 ms**. This is the displayed region's elapsed duration, rather than the duration of the epilogue alone or the entire forward pass.

The PR reports exact agreement in its local scoring, rounding-boundary, and integration checks, and GSM8K at **96.59%**, compared with baseline runs at **96.74% and 96.36%**.

### How the Three Indexer Changes Combine

The changes address three different costs: Section 1 changes the **physical width** of dense scoring; Section 2 reduces the **set of keys scored**; Section 3 reduces **memory traffic after scoring**. Their call paths determine which benefits can combine:

<div role="region" aria-label="Indexer scoring paths and compatible optimizations" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 960px;">

| Scoring path | Dot product | Postprocessing | Applicable combination |
| --- | --- | --- | --- |
| Dense publisher or dense fallback | Regular `indexer.scores` GEMM over visible keys, optionally padded by Section 1 | Section 3's fused `reduce_head_scores` when its 32-head BF16/layout gate passes; Torch otherwise | Alignment and epilogue fusion can combine on the eligible dense path. |
| Eligible candidate consumer | Section 2's `_candidate_matmul` over candidate keys only | Its own Torch ReLU, weighting, and head reduction | Candidate-only scoring replaces the dense GEMM; the inspected implementation does not call Section 3's fused epilogue. |

</div>
</div>

This distinction follows the inspected [dense scoring code](https://github.com/BJWang-ant/sglang/blob/619d461d452445dbeb171d4f6f9a9e788513e40e/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L284) and [candidate scoring code](https://github.com/BJWang-ant/sglang/blob/772a4a1c754c9c85edfec8aabef9b9edee460545/python/sglang/kernels/ops/attention/dsv4/candidate_scores.py#L90). Padded key storage can still be prepared upstream for a candidate consumer, but its compact output width is determined by the candidate blocks; it does not execute the dense GEMM whose alignment gain is measured in Section 1. A single model run can use an aligned, fused dense path at publishers or fallbacks and the separate candidate kernel at eligible consumers.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 \
--chunked-prefill-size 8192 --max-prefill-tokens 8192 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 \
--tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

We cherry-picked the optimizations from Sections 1–3 onto commit `d0a75654368a470d8484e3ea45b06e500fee57af` and benchmarked the combined configuration. The results below serve as the baseline for Section 4.
![Original Chapter 123 Aggregation result](../public//images/blog/deepseek-v41-h200-pd/Chapter-123-Aggregation-result.png)
*Figure 16. Chapter 123 Aggregation Optimization Result.*

## 4. Larger Prefill Chunks and Decode Graph Metadata

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af`, with the Indexer changes from PR #45794 (Section 1), PR #46003 (Section 2), and PR #45807 (Section 3) cherry-picked together. This combined configuration uses 8,192-token Prefill chunks for the baseline and 16,384-token chunks for the optimized comparison. The associated Decode graph-metadata work is described below.

Section 2.5 of the notebook, “Larger Prefill Chunks,” examines another source of Prefill overhead. With an 8,192-token chunk, a long uncached suffix requires several Prefill forward passes and scheduling rounds. Increasing the chunk to 16,384 tokens lets each forward pass process more of that suffix and reduces the number of rounds. This targets scheduling and chunk boundaries alongside the kernel work optimized above.

The same investigation exposed a Decode-side CUDA graph metadata problem: reading CUDA sequence lengths back to the host with `.tolist()` during capture could fail and force the metadata path to run eagerly. This work was tracked in [PR #46777](https://github.com/sgl-project/sglang/pull/46777).

The merged PR prevents the draft metadata glue graph from running when a backend requires host sequence lengths and removes an unused `seq_lens.sum().item()` read during Decode graph capture. Its description notes that the original host-read fixes landed separately in #46891 and #46897. Under the current DSpark configuration, the host-length guard is defensive because `needs_cpu_seq_lens` is already false. The table below reports the notebook's chunk-size experiment; it does not isolate the performance impact of the guard.

![Larger Prefill chunks reduce scheduling rounds, while Decode graph metadata avoids unnecessary host reads](../public//images/blog/deepseek-v41-h200-pd/chunk-prefill-graph.svg)

*Figure 17. An illustrative 16,384-token uncached suffix requires two 8,192-token chunks or one 16,384-token chunk. The Decode metadata work tracked in [PR #46777](https://github.com/sgl-project/sglang/pull/46777) addresses graph-capture constraints separately from the choice of Prefill chunk size.*

### Data Summary: 8,192 → 16,384 Prefill Tokens

The following table reproduces the eight selected cases from §2.5, measured after the earlier checkpoint was restored. Each arrow shows the reported baseline followed by the result with 16,384-token chunks. TTFT and E2E are in milliseconds; TPOT is in milliseconds per output token. We calculate the change in E2E latency as `(16K result / baseline − 1) × 100%`, so a negative value means lower latency.

<div role="region" aria-label="Larger Prefill chunk benchmark results" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 900px;">

| Cache-hit label | shared prefix length | Concurrency | baselineTTFT(8k) -> TTFT(16k) (ms) | baselineTPOT(8k) -> TPOT(16k) (ms/token) | baselineE2E(8k) -> E2E(16k) (ms) | E2E change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 32768 | 4 | 3,693 → 3,434 | 2.59 → 2.05 | 5,017 → 4,482 | −10.7% |
| 50% | 32768 | 8 | 8,697 → 7,865 | 2.54 → 2.01 | 9,997 → 8,892 | −11.1% |
| 70% | 46080 | 8 | 4,718 → 4,301 | 2.59 → 2.07 | 6,042 → 5,358 | −11.3% |
| 70% | 46080 | 16 | 10,706 → 9,583 | 2.62 → 2.06 | 12,044 → 10,636 | −11.7% |
| 90% | 61440 | 8 | 962 → 959 | 3.32 → 3.30 | 2,660 → 2,644 | −0.6% |
| 90% | 61440 | 16 | 1,527 → 1,465 | 3.66 → 3.93 | 3,396 → 3,471 | +2.2% |
| 90% | 61440 | 24 | 2,744 → 1,885 | 3.49 → 4.15 | 4,529 → 4,004 | −11.6% |
| 90% | 61440 | 32 | 4,065 → 2,867 | 3.81 → 4.29 | 6,011 → 5,061 | −15.8% |

</div>
</div>

TTFT falls in every displayed case, with the largest reduction of **31.3%** at the 90% label and concurrency 24. E2E improves in **seven of eight cases**, with the largest reduction of **15.8%** at the 90% label and concurrency 32. At the 50% and 70% labels, TPOT also decreases by approximately **20.1%–21.4%**.

At the 90% label and concurrency 16, however, TTFT improves by **4.1%** while TPOT increases by **7.4%**. The generation-side increase outweighs the input-side gain, and E2E rises from **3,396 to 3,471 ms**, a **2.2% regression**. TPOT also rises at concurrency 24 and 32, although their larger TTFT reductions still produce lower E2E latency.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 \
--chunked-prefill-size 16384 --max-prefill-tokens 16384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 \
--tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

## 5. Use Reduce-Scatter for CP MoE Output Shards

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af` and use 16,384-token Prefill chunks. The optimized run cherry-picks the MoE output reduce-scatter change from PR #47247. This comparison is independent of the three Indexer experiments.

Once the largest Indexer costs were reduced, the notebook's next profiling pass shifted attention toward MoE and collectives. [PR #47247](https://github.com/sgl-project/sglang/pull/47247) examines the ownership of the MoE output rows.

On DeepSeek-V4.1's CP MoE return path, the original implementation first all-reduced the full expert output. Every rank received all reduced rows, then selected its CP shard and made it contiguous. When TP=CP=4 and both span the same GPU group, each rank ultimately needs only one quarter of those rows.

Here **contiguous return rows** and **interleaved token ownership** describe different layouts. The launch option `--cp-strategy interleave` assigns original prompt positions round-robin to CP ranks. Before MoE, `attn_cp_interleave_gather` packs those local shards in rank order. Each rank's rows therefore occupy one contiguous block in the gathered buffer, even though their original prompt positions are interleaved. For 16 tokens at CP4, the buffer is `[0,4,8,12 | 1,5,9,13 | 2,6,10,14 | 3,7,11,15]`; rank 0 owns the first four buffer rows, which correspond to prompt positions `[0,4,8,12]`.

The model sets `cp_return_contiguous` from `model_type == "deepseek_v41"`, independently of the CLI strategy name. Thus an interleave launch can use this contiguous-return branch. This follows the inspected [model transport selection](https://github.com/BJWang-ant/sglang/blob/f065f3d1747afacdca9982208e3d2e952dd08f09/python/sglang/srt/models/deepseek_v4_boundary.py#L513) and [rank-order gather](https://github.com/BJWang-ant/sglang/blob/f065f3d1747afacdca9982208e3d2e952dd08f09/python/sglang/srt/layers/cp/interleave.py#L57).

For four ranks, let `Y_r` be rank `r`'s partial full-row output and let their sum be `Y = [S0 | S1 | S2 | S3]`. With all-reduce, every rank receives all four reduced shards and then keeps only `S_r`. With reduce-scatter, rank 0 receives `S0`, rank 1 receives `S1`, and so on. The input contributions can still cover the full row extent; the reduced **output stored on each rank** is one quarter as large. This does not imply 75% less network traffic or a fourfold latency reduction.

Reduce-scatter expresses that requirement directly: reduce the full input contributions and return only the rows owned by the receiving rank. The optimization removes full-result replication and the subsequent local output slicing/copying step.

![All-reduce followed by slicing compared with direct reduce-scatter](../public//images/blog/deepseek-v41-h200-pd/moe-reduce-scatter.svg)

*Figure 18. The change targets DeepSeek-V4.1's `cp_return_contiguous` MoE branch with no MoE all-to-all backend. Each shard is contiguous within the packed, rank-major buffer; its original prompt positions can remain interleaved. The separate `cp_return_contiguous=False` branch already uses a reduce-scatter helper. That branch selection is distinct from the CLI option `--cp-strategy interleave`.*

The gate requires CUDA/non-HIP tensors, matching TP/CP/group sizes, and a nonzero row count divisible by CP size. Other configurations retain the original fallback. The quarter-sized output at CP4 describes ownership, not a fourfold collective speedup.

In matched profiled steps with a BF16 input of shape `[16240, 5120]` and 40 MoE output collectives per rank, the summed communication-kernel durations were:

<div role="region" aria-label="Matched per-rank MoE collective durations" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 720px;">

| Rank | All-reduce, sum of 40 calls (ms) | Reduce-scatter, sum of 40 calls (ms) | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | 68.346 | 52.673 | 22.9% |
| 1 | 71.672 | 57.369 | 20.0% |
| 2 | 59.915 | 50.691 | 15.4% |
| 3 | 60.526 | 44.981 | 25.7% |

</div>
</div>

Across a separate set of nearby sampled steps—six baseline and seven optimized—the mean per-rank reduction was **19.4%–21.7%**. Mean GPU forward duration, taking the slowest rank's span for each step, decreased from **606.244 to 595.320 ms**: **10.924 ms, or approximately 1.8%**. These means and the matched collective totals above use different sample scopes.

The denominator already explains much of the apparent gap: as a simple illustration, cutting 20% from a stage that occupies one tenth of a forward reduces total time by only about 2%, even without overlap. This is scale intuition, rather than a measured decomposition of the forward's critical path.

![Original serving benchmark table from PR 47247](../public//images/blog/deepseek-v41-h200-pd/pr-47247-source.png)

*Figure 19. Original [PR #47247 performance screenshot](https://github.com/sgl-project/sglang/pull/47247). Percentages discussed here are recalculated from the displayed absolute values.*

The serving table shows **0.8%–3.4% lower E2E latency**. At the 90% label and concurrency 24, TPOT improves from **5.18 to 4.79 ms/token**, but TTFT increases from **1,829 to 1,993 ms**. E2E therefore improves by only **0.8%**, from **4,476 to 4,441 ms**.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
--swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

## 6. Enable Bounded Replay with CP and DSpark

**Baseline and comparison.** Both configurations start from commit `d0a75654368a470d8484e3ea45b06e500fee57af`, with the Indexer changes from PR #45794, PR #46003, and PR #45807 cherry-picked together. The baseline is the resulting Section 4 configuration, including its 16,384-token Prefill chunks. The optimized run additionally cherry-picks the CP and DSpark compatibility fix from PR #47406 and enables bounded replay on the Prefill worker.

DeepSeek-V4.1's final compressed-KV source is **layer 20**. Once that layer has processed the full chunk, the long-range compressed KV and Indexer keys needed by layers 21–39 already exist. Those later layers reuse the shared history while producing their own window KV. This creates an opportunity to retain the shared long-range history and rebuild only the local state needed for generation.

The existing **decoder-side bounded replay** optimization processes the full prompt chunk through layers 0–20, then executes layers 21–39 on at most the last **128 tokens per request**. Here 128 is the configured SWA window length, rather than the CP partition size or the Indexer top-k budget. The tail execution reconstructs each late layer's window KV and retained hidden states, using the already-computed layer-20 history. The option is enabled on the **Prefill worker**, despite its decoder-side name.

This is an approximation at the tail boundary. A retained token near the start of the tail would normally attend to some local-window keys preceding that boundary, but those late-layer rows are not reconstructed. Its hidden state can therefore differ, and differences can propagate through later layers. Preserving compressed long-range history does not restore the omitted local-window computation or make the result equivalent to full Prefill. The speed/quality tradeoff belongs to bounded replay itself.

[PR #47406](https://github.com/sgl-project/sglang/pull/47406) makes that existing optimization usable with **eager CP and DSpark** by fixing the compact-output layout. It does not introduce the bounded-tail algorithm or eliminate its boundary approximation.

With CP and DSpark auxiliary-state capture, the model returns compact tail outputs. The eager CP runner previously gathered them with the original full-chunk layout. For a single 16,384-token chunk at CP4, that layout expects 4,096 local rows; a 128-token tail supplies only 32 rows per rank. The output tensor and the gather metadata therefore describe different token sets.

For a single request with no cached prefix and zero-based prompt positions `0…16383`, the retained tail maps as follows:

| Compact row after CP gathering | Original prompt position |
| ---: | ---: |
| 0 | 16,256 |
| 1 | 16,257 |
| 127 | 16,383 |

Thus compact rows `0…127` correspond to original positions `16256…16383`. Both final hidden states and DSpark auxiliary hidden states must preserve this mapping: compact row 0 cannot be interpreted as prompt position 0. Before gathering, the 32 local rows on each CP rank follow that rank's tail layout; their local row numbers are not global prompt positions. Cached-prefix offsets, multiple requests, and padding require explicit per-request/token mappings rather than a single batch-wide offset.

![Tail-aware CP gathering for bounded replay and DSpark](../public//images/blog/deepseek-v41-h200-pd/bounded-replay.svg)

*Figure 20. The compatibility fix preserves compact hidden states, DSpark auxiliary states, and their original global token indices. The illustrated 32-row local tail is a single-request example; mixed requests require their own tail metadata.*

The patch adds a model-specific `postprocess_cp_output` hook. DeepSeek's hook gathers hidden and auxiliary states using tail-specific CP metadata, aligns logits and DSpark consumers with the original global token indices, and restores the original batch metadata in a `finally` block. The ordinary forward path and the eager CP path share this output postprocessing.

This allows bounded replay to be explicitly enabled for eager CP Prefill. The default CP policy is unchanged, and the combination of CP, bounded replay, and Prefill CUDA graphs remains unsupported.

![Original serving benchmark table from PR 47406](../public//images/blog/deepseek-v41-h200-pd/pr-47406-source.png)

*Figure 21. Original [PR #47406 performance screenshot](https://github.com/sgl-project/sglang/pull/47406). Its baseline is the resulting configuration from Section 4.*

The reported TTFT reduction ranges from **13.6% to 46.7%**. At the 70% label and concurrency 8, TTFT drops from **4,301 to 2,294 ms** and E2E from **5,358 to 3,588 ms**. The largest E2E reduction is **34.3%**, at the 50% label and concurrency 8.

The table also records TPOT increasing from **2.07 to 2.53 ms/token** in the 70%/concurrency-8 case. Prefill gains therefore do not imply faster token generation. The available measurements do not isolate the cause of this TPOT change.

The PR reports four-rank validation of token mappings and metadata restoration, nine full-model functional requests, and a 32-question GSM8K smoke test. These checks support the compatibility fix; the small quality probe does not establish broad long-context accuracy parity for the boundary approximation described above.

```bash
# prefill
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=0,1,2,3 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --ep 4 --enable-prefill-cp --cp-strategy interleave \
--cuda-graph-backend-prefill disabled --moe-runner-backend humming --moe-a2a-backend none --nnodes 1 --node-rank 0 --disaggregation-mode prefill \
--dist-init-addr 127.0.0.1:5557 --disaggregation-transfer-backend mooncake --speculative-algorithm DSPARK --max-running-requests 256 --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
--swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --enable-decoder-swa-bounded-replay --tokenizer-worker-num 8 --host 127.0.0.1 --port 30011

# decode
GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_VISIBLE_DEVICES=4,5,6,7 MC_TE_METRIC=1 MC_SLICE_SIZE=262144 MC_TE_METRIC_INTERVAL_SECONDS=1 MC_INTRA_NVLINK=true \
MC_INTRANODE_NVLINK=true SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1 SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype": "float8e4m3"}' \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
sglang serve --trust-remote-code --language-model-only --model-path /models/DeepSeek-V4.1-Flash/ --tp 4 --dp 4 --ep 4 --enable-dp-attention --enable-dp-lm-head \
--moe-runner-backend humming --moe-a2a-backend none --disaggregation-mode decode --cuda-graph-max-bs-decode 256 --disaggregation-transfer-backend mooncake \
--speculative-algorithm DSPARK --max-running-requests 384 --swa-full-tokens-ratio 0.05 --mem-fraction-static 0.85 --detokenizer-worker-num 8 --tokenizer-worker-num 8 \
--host 127.0.0.1 --dist-init-addr 127.0.0.1:5757 --port 30012
```

## Performance Results: Separate PR Comparisons from the Combined Run

The five PR comparisons in Sections 1–3, 5, and 6 act at different levels: tensor shape, candidate scoring domain, memory traffic, collective output ownership, and late-layer token extent. Sections 1–3 and 5 evaluate individual changes from the common starting revision. Section 4 combines the three Indexer changes before evaluating larger Prefill chunks, and its resulting configuration supplies the baseline for Section 6. The chunk-size experiment is reported separately in Section 4 because it also involves configuration and graph-metadata work. The reported percentages use these distinct baselines and cannot be added to obtain a cumulative gain.

![Per-PR E2E reductions across the eight reported cache-hit and concurrency cases](../public//images/blog/deepseek-v41-h200-pd/pr-e2e-comparison.png)

*Figure 22. Redrawn from the five initial PR screenshots. Rows retain the order #45794, #47406, #47247, #46003, and #45807, corresponding to Sections 1, 6, 5, 2, and 3. Each cell shows the reduction and baseline → optimized E2E latency in milliseconds. Sections 1–3 and 5 use independent comparisons from the common starting revision; Section 6 uses the resulting Section 4 configuration as its baseline. The row percentages cannot be added or multiplied to infer a combined result.*

The following figure shows the performance comparison of all the optimizations listed in this article after being combined (note: the baseline startup parameters do not have --enable-decoder-swa-bounded-replay and --chunked-prefill-size and --max-prefill-tokens are all set to 8192).

<div role="region" aria-label="Historical combined benchmark results" tabindex="0" style="overflow-x: auto;">
<div style="min-width: 800px;">

| Cache-hit label / concurrency | TTFT, baseline → combined (ms) | TPOT, baseline → combined (ms/token) | E2E, baseline → combined (ms) | E2E reduction |
| --- | ---: | ---: | ---: | ---: |
| 50% / 4 | 6,245 → 1,876 | 2.27 → 2.56 | 7,402 → 3,182 | 57.0% |
| 50% / 8 | 13,583 → 4,249 | 2.30 → 2.36 | 14,759 → 5,457 | 63.0% |
| 70% / 8 | 8,061 → 2,097 | 2.28 → 2.46 | 9,228 → 3,352 | 63.7% |
| 70% / 16 | 17,160 → 5,359 | 2.27 → 2.47 | 18,322 → 6,623 | 63.9% |
| 90% / 8 | 1,368 → 807 | 3.07 → 3.30 | 2,938 → 2,495 | 15.1% |
| 90% / 16 | 2,846 → 1,104 | 3.17 → 3.94 | 4,465 → 3,115 | 30.2% |
| 90% / 24 | 5,058 → 1,302 | 3.09 → 4.32 | 6,637 → 3,511 | 47.1% |
| 90% / 32 | 7,189 → 1,512 | 3.16 → 4.80 | 8,803 → 3,967 | 54.9% |

</div>
</div>

The highest E2E reduction is **63.9%** at the 70% label and concurrency 16, with a **baseline/combined latency ratio of 2.77** for that recorded case. The largest TTFT reduction is **79.0%** at the 90% label and concurrency 32. These maxima occur in different cases. They do not establish a throughput increase or a joint tail-latency SLA.

## Reproduction

To reproduce the 1P1D experiments, start from the common development commit `d0a75654368a470d8484e3ea45b06e500fee57af` and apply the changes and settings listed at the beginning of the relevant section. Follow that section's baseline and optimized configurations for each comparison.

Launch the Prefill and Decode servers, start the router, and run the accuracy and performance evaluations below. Record the exact SGLang revision and applied patches, target and draft checkpoint revisions, tokenizer version, and CUDA and dependency versions for each run.

### Prefill and Decode Launch Command

Assign four H200 GPUs to the Prefill worker with TP4 / CP4 / EP4 and the remaining four to the Decode worker with TP4 / DP4 / EP4. Use the launch block in the section being reproduced, matching its chunk size, attention and MoE backends, and bounded-replay setting. Section 6's commands enable bounded replay and do not serve as a common baseline for every comparison.

The server commands and benchmark script use `/models/DeepSeek-V4.1-Flash/` for a local copy of the model weights linked in Measurement Setup. Update `--model-path` and `MODEL_PATH` together if your local directory differs.

### Router Launch Command

Start the router after both workers are ready, using their respective Prefill and Decode endpoints. Send the accuracy and performance requests through the router so that they exercise the complete PD serving path. Keep the router configuration fixed within each baseline/optimized comparison.

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill-policy round_robin \
    --decode-policy round_robin \
    --prefill http://127.0.0.1:30011 8761 \
    --decode http://127.0.0.1:30012 \
    --host 127.0.0.1 \
    --disable-circuit-breaker \
    --port 8009
```

### Accuracy Evaluation Script

Run GSM8K through the router endpoint. Use the evaluation configuration and sample count associated with the result being reproduced, and keep the prompt format and generation settings consistent across baseline and optimized runs. Record accuracy, request errors, and truncated outputs.

```bash
sgl-eval run gsm8k --base-url http://127.0.0.1:8009/v1 --num-threads 64 --max-tokens 2048
```

### Performance Benchmark Script

The script below is a single-run template: it invokes the benchmark once for each cache-hit and concurrency case for the selected server configuration, with 512 requests and 10 warmup requests per case. Send the workload through the router and keep the input data, checkpoints, generation settings, and streaming configuration fixed when isolating an optimization. Report TTFT, TPOT, E2E latency, and the numbers of successful and failed requests from that run.

```bash
MODEL_PATH="/models/DeepSeek-V4.1-Flash"
DATASET_PATH="/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

HOST="127.0.0.1"
PORT="8009"
BACKEND="sglang"

SYS_INPUT_LENS=(61440 46080 32768)
QUE_INPUT_LENS=(4096 19456 32768)
OUTPUT_LENS=(512)
NUM_PROMPTS=512

for i in "${!SYS_INPUT_LENS[@]}"; do
  sys_input_len=${SYS_INPUT_LENS[$i]}
  que_input_len=${QUE_INPUT_LENS[$i]}

  if [[ $i -eq 0 ]]; then
    CONCURRENCY_LIST=(8 16 24 32)
  elif [[ $i -eq 1 ]]; then
    CONCURRENCY_LIST=(8 16)
  else
    CONCURRENCY_LIST=(4 8)
  fi

  for osl in "${OUTPUT_LENS[@]}"; do
    for concurrency in "${CONCURRENCY_LIST[@]}"; do

      for ((iii=1; iii<=1; iii++)); do
        python3 -m sglang.bench_serving \
          --model "${MODEL_PATH}" \
          --host "${HOST}" \
          --port "${PORT}" \
          --backend "${BACKEND}" \
          --dataset-path "${DATASET_PATH}" \
          --num-prompts "${NUM_PROMPTS}" \
          --dataset-name generated-shared-prefix \
          --gsp-num-groups 1 \
          --gsp-prompts-per-group "${NUM_PROMPTS}" \
          --flush-cache \
          --gsp-system-prompt-len "${sys_input_len}"  \
          --gsp-question-len "${que_input_len}"  \
          --gsp-output-len "${osl}" \
          --max-concurrency "${concurrency}" \
          --warmup-requests 10
      done
    done
  done
done
```

## Appendix: Evidence and PR Links

The common starting development commit is `d0a75654368a470d8484e3ea45b06e500fee57af`. Sections 1–3 and 5 evaluate their changes independently from this revision. Section 4 starts from this revision with the changes from Sections 1–3 cherry-picked together; Section 6 starts from the resulting Section 4 configuration.

As of November 9, 2026, the foundational [PR #40323](https://github.com/sgl-project/sglang/pull/40323) and all six PRs discussed here ([#45794](https://github.com/sgl-project/sglang/pull/45794), [#46003](https://github.com/sgl-project/sglang/pull/46003), [#45807](https://github.com/sgl-project/sglang/pull/45807), [#47247](https://github.com/sgl-project/sglang/pull/47247), [#46777](https://github.com/sgl-project/sglang/pull/46777), and [#47406](https://github.com/sgl-project/sglang/pull/47406)) are merged. The inspected revisions below pin the implementations used for the explanations; they are not a complete benchmark revision manifest and can precede a PR's final head:

<div style="overflow-wrap: anywhere;">

- [#45794: 9a8995ba0064ace674b9186404194268273ec575](https://github.com/BJWang-ant/sglang/commit/9a8995ba0064ace674b9186404194268273ec575)
- [#47406: f274754cb12fdefba511c05137257410df00dea5](https://github.com/BJWang-ant/sglang/commit/f274754cb12fdefba511c05137257410df00dea5)
- [#47247: f065f3d1747afacdca9982208e3d2e952dd08f09](https://github.com/BJWang-ant/sglang/commit/f065f3d1747afacdca9982208e3d2e952dd08f09)
- [#46003: 772a4a1c754c9c85edfec8aabef9b9edee460545](https://github.com/BJWang-ant/sglang/commit/772a4a1c754c9c85edfec8aabef9b9edee460545)
- [#45807: 619d461d452445dbeb171d4f6f9a9e788513e40e](https://github.com/BJWang-ant/sglang/commit/619d461d452445dbeb171d4f6f9a9e788513e40e)
- [#46777: 70e5eccaf1e70045654074d175bcc7680051267a](https://github.com/BJWang-ant/sglang/commit/70e5eccaf1e70045654074d175bcc7680051267a)

</div>
