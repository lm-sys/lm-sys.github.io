---
title: "SGLang and Miles Add Day-0 Support for Qwen3.8"
author: "SGLang Team"
date: "Aug 12, 2026"
previewImg: /images/blog/qwen3-8-day0-support/cover-qwen3-8.png
type: blog
---

We are excited to announce Day-0 support for **[Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)** in SGLang and Miles.
It is Qwen's largest open-source model, with 2.4T total parameters and 95B active per token,
and its hybrid attention architecture puts pressure on most of the assumptions a serving stack
makes about state. In collaboration with the Qwen, NVIDIA, and AMD teams, SGLang covers the
model in full on launch day. This post covers what it took.

**Highlights**

- **A hybrid architecture.** 92 layers, 69 GDN linear-attention layers interleaved with 23 GQA
  full-attention layers in a 3:1 pattern, and MoE layers with 512 experts and top-10 routing.
- **An NVFP4 checkpoint we quantized**,
  [RadixArk/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-NVFP4),
  released Day-0.
- **A kernel stack built with NVIDIA and shipped through FlashInfer**: MoE finalize fused
  with all-reduce and RMSNorm (10+% end to end), a context-parallel GDN prefill kernel, and
  a low latency single-GEMM path (~4% end to end).
- **Speculative decoding**: at TP8 on B300, the NVFP4 checkpoint decodes at **346** tok/s
  for batch size 1 with MTP at an accept length of 3.3, and at **378** tok/s with DSpark at
  an accept length of 4. Both rates include the bonus token.
- **Parallelism split by phase**: chunked pipeline-parallel prefill and a
  data-and-expert-parallel decode worker, composing under PD disaggregation to **5,126**
  tok/s per GPU on 8k/1k, with a staging buffer that lets the two sides be sized and
  parallelized independently.
- **Day-0 RL with Miles**: colocated LoRA training on the native NVFP4 base, with a
  BF16 Megatron trainer and NVFP4 SGLang rollout engines sharing the same 64 GB300s,
  and a GRPO run on GSM8K verifying stable reward and flat train/rollout KL.

Launch commands and per-workload configuration guidance live in the
[Qwen3.8 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8).

## Model Architecture

Qwen3.8-2.4T-A95B continues the hybrid attention design of the Qwen3.5/3.6 series.
This generation scales to 2.4T total parameters, with 95B activated per token across
92 layers.

### Architecture Highlights

The Qwen3.8-2.4T-A95B architecture includes: 

<img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-Next/model_architecture.png" alt="Qwen3.8-2.4T-A95B model architecture" width="400">

- **Hybrid Attention.** It combines 69 linear-attention (GDN) layers and 23 full-attention (GQA) layers in a 3:1 interleaved pattern. This design strikes a balance between linear computational complexity and long-context modeling performance.


- **GDN (Gated Delta Network).** The linear-attention layers combine a State Space Model (SSM) with causal convolution (CausalConv1d). A fixed-size recurrent state replaces the growing KV cache, so each GDN layer uses `O(1)` memory while its computation scales as `O(N)`.

- **Sparse Mixture-of-Experts (MoE).** Each MoE layer provides 512 routed experts plus a single shared expert, with top-k=10 routing.

## Feature Support

Each Qwen3.8 request maintains three forms of serving state: the KV cache for
full-attention layers, the recurrent state for GDN layers, and the GDN convolution
windows. The features below must manage all three consistently across prefix caching,
speculative decoding, and PD disaggregation.


### ReplaySSM for the GDN State

MTP verification creates a state-recovery problem for GDN layers. Each layer updates
its recurrent state in place while verifying multiple draft tokens, but only the state
corresponding to the accepted prefix should be committed. Qwen3.8 applies
[ReplaySSM](https://tridao.me/blog/2026/replayssm/) to this problem. We described this
raw-input replay mechanism in detail in our
[previous post](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support#replayssm-raw-input-replay-for-the-kda-state).
During verification, it records the recurrence inputs instead of snapshotting the full
GDN state at every draft position. Once the sampler determines the accepted length, a
fold kernel replays the accepted prefix from the committed checkpoint and advances the
state in place.

We integrated the recording path into FlashInfer's CuTe DSL GDN MTP kernel for BF16
states. The verify prologue already has the required values in registers, so ReplaySSM
only adds the corresponding ring-buffer stores. It leaves verification results bitwise
unchanged and introduces no measurable verify-throughput regression. The same
mutable-state caching path allows MTP to compose with prefix caching, overlap scheduling,
and PD disaggregation.

### Prefill-Decode Disaggregation

PD disaggregation transfers all three state types from the prefill worker to the
decode worker through a typed state registry. Each registered handler moves its
corresponding state, including the KV cache, GDN recurrent state, and GDN convolution
windows. The `q`, `k`, and `v` sub-blocks of each convolution window are sharded
independently across tensor-parallel ranks, so the transfer layer slices and
reassembles them for the destination layout.

The same payload carries the MTP draft model's KV cache, hidden states, and top-k
metadata, allowing speculative decoding to continue on the decode worker. When prefill
and decode use different attention-sharding layouts, a GPU staging buffer coalesces the
per-layer slices into one bulk RDMA transfer per chunk instead of issuing separate
transfers for each slice.

### Radix Cache and HiCache

Qwen3.8 uses SGLang's
[Unified Radix Cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)
to enable prefix caching for both full-attention KV and GDN state. The `FULL`
component manages the full-attention KV cache, while the `MAMBA` component manages
GDN checkpoints. Each GDN checkpoint bundles the recurrent state and convolution
windows.

Before a forward pass mutates a shared GDN checkpoint, copy-on-write restores it
into a private request slot. SGLang creates new checkpoints at prefill chunk
boundaries and regular decode intervals. A shared cache controller coordinates the
KV and GDN components across device and host tiers, allowing prefix caching and
HiCache to compose with MTP and PD disaggregation.

## Chunked Pipeline-Parallel Prefill

Going wide on experts means sharding all 512 experts, enabling EPLB, and paying for two
all-to-alls per MoE layer. This is the right answer for decode, which is memory-bound and
needs the weights read once and shared. Prefill is compute-bound and already arrives with thousands of tokens per step,
so wide EP adds no parallelism it needs; it adds two all-to-alls per MoE layer to a
critical path that had no communication in it. Under PD disaggregation the two phases
run on separate workers and are free to disagree.

Pipeline parallelism inverts every term. Each stage owns a contiguous slice of the 92
layers and runs them to completion inside one rank. This requires no dispatch, combine,
or EPLB, and it preserves full-width GEMMs. The only communication left is one activation per stage boundary,
which chunked prefill hides: a request is split into chunks that flow through the
stages back to back, so the hand-off for chunk *i* overlaps the compute of chunk *i+1*.
And the cost EP pays grows with the expert count while the cost PP pays does not.

<img src="/images/blog/qwen3-8-day0-support/fig-chunked-pp-prefill.svg" alt="Chunked pipeline-parallel prefill: chunks flow through stages back to back, so each hand-off overlaps the next chunk's compute" width="720">

Measured on 8K prefill, at each arm's own best operating point, in input tokens per
second per GPU:

| Checkpoint | Chunked PP prefill | Wide EP + EPLB | Speedup |
|---|---:|---:|---:|
| FP8, 16 GPUs | **5231** (PP16) | 3421 | **1.53×** |
| NVFP4, 8 GPUs | **8363** (PP8) | 5151 | **1.62×** |

The FP8 row is two dedicated prefill-only runs (8192 in, 1 out). The NVFP4 row is the
median prefill-batch rate taken from inside full 8192/1024 serving runs on both arms, so
it measures each one while it is actually prefilling and excludes the decode work
sharing the GPUs. Comparing the wide-EP arm's *end-to-end* input rate against a
prefill-only PP worker would read 2.3× and would be flattering rather than accurate.
Both wide-EP arms have EPLB enabled; it fixes routing imbalance, which is not what is
costing them the throughput.

### Pipeline-Parallel Prefill with MTP

Pipelined prefill and speculative decoding used to be mutually exclusive, which meant
choosing between the throughput above and the per-user speed below. The obstruction was
structural: under pipelining the embedding sits on the first stage and the LM head on
the last, so no single stage holds both, yet the draft head needs both. We put the draft
head on the last stage with its own copy of the half it does not receive, stage the
draft KV across the PD boundary alongside the target KV, and let the ranks that host no
draft own no draft KV pool. The prefill topology becomes a free variable: the decode
worker keeps its speculative decoding however the prefill worker is sliced.

### Staging Buffer: Decoupling Prefill and Decode Layouts

A PP16 prefill worker and a wide-EP decode worker do not agree on how KV is partitioned,
and requiring a shared TP layout would drag prefill back onto decode's topology and
forfeit everything above.

The staging buffer changes what the two sides agree on. Prefill writes completed chunks
into staging buffers and publishes a per-peer watermark; decode scatters out of them into
whatever layout it uses, prefetching chunk by chunk. The contract is a chunk index and a
watermark rather than a partitioning, so the transfer overlaps the remaining prefill
instead of following it, and the prefill:decode ratio, the pipeline depth and the decode
EP width become independent knobs. The same path carries draft KV.

## Performance

### Pareto Curve on 8K/1K

All numbers below are 8,192-input / 1,024-output on GB300. The figure includes
PD-disaggregated results for NVFP4 and FP8, plus aggregate TP results for FP8.
Throughput is total (input + output) tokens per second per active model-serving GPU;
per-user speed is output tokens per second per request.

<img src="/images/blog/qwen3-8-day0-support/fig-pareto-8k1k.svg" alt="Aggregate and PD-disaggregated serving results at 8,192 input and 1,024 output tokens for the FP8 and NVFP4 checkpoints, total tok/s per GPU against per-user output speed" width="100%">

The representative endpoint labels report active/allocated GPU counts; `P` and `D`
show the active prefill/decode split. The PP6 maximum uses 20 active GPUs
(`12P + 8D`) from 24 allocated, and TPS/GPU is divided by active model-serving GPUs.

The PD-disaggregated points shown here use a forced accept length of 3.3. The FP8
aggregate points report the TPOT measured in their respective runs. The endpoints are:

| Checkpoint | Max throughput (PD disagg) | Low-latency endpoint |
|---|---|---|
| NVFP4 (2×PP6 prefill, DP2-attn / TP4 / EP8 decode) | **5,126** tok/s/GPU @ 36 tok/s/user | PD: **108** tok/s/GPU @ **334** tok/s/user |
| FP8 (2×PP16 prefill, DP4-attn / TP4 / EP16 decode) | **3,532** tok/s/GPU @ 30 tok/s/user | Aggregate CC1: **220** tok/s/GPU @ **362** tok/s/user |

The NVFP4 peak uses two PP6 prefill workers feeding a DP2-attn / TP4 / EP8 decode
worker. At the low-latency end, NVFP4 uses a PD-disaggregated PP2×TP4 prefill worker
with a TP16 decode worker (334 tok/s/user), while FP8 uses a TP16 aggregate worker at
concurrency 1 (362 tok/s/user).

On the matched dual-PP6 NVFP4 backbone, adding MTP moves throughput +10.0% and per-user
speed 2.33×. The updated 5,126 point is not used for that matched comparison.
The asymmetry is the expected shape: in a saturated decode worker the tokens per step are
roughly `running_requests × draft_tokens` and fixed by the memory budget, so speculative
decoding mostly converts a fixed step budget into fewer, longer steps per request.

### Kernel Optimizations

- **Fused MoE finalize, AllReduce, and RMSNorm.** With a hidden dimension of 8,192
  and top-10 routing, the finalize input buffer becomes large during prefill. At an
  input sequence length of 8K, `8192 × 10 × 8192 × sizeof(bfloat16)` requires
  1.25 GiB and makes finalization account for up to 10% of prefill time. We developed
  fused computation and communication kernels using Programmatic Dependent Launch
  (PDL) chaining and persistent execution. In our tested configurations, they improved
  end-to-end latency and throughput by more than 10%. The implementation is available
  in [FlashInfer PR #4358](https://github.com/flashinfer-ai/flashinfer/pull/4358).

- **Context-parallel GDN prefill.** This kernel partitions the sequence into chunks
  and processes them in parallel, increasing GPU utilization for long sequences and
  small batches. It improves prefill performance by 2% to 3%. Implementation details
  are tracked in
  [FlashInfer issue #3491](https://github.com/flashinfer-ai/flashinfer/issues/3491).

- **Low-latency single-GEMM path.** Small GEMMs contribute substantially to latency,
  especially when they require a separate Split-K reduction kernel. The optimized
  single-GEMM path delivers up to 1.5× kernel-level speedup and approximately 4%
  end-to-end improvement. See
  [FlashInfer PR #4266](https://github.com/flashinfer-ai/flashinfer/pull/4266).

- **Fused GDN decode operations.** We fused the SplitKV reshape and Conv1D operations
  for low-latency tensor-parallel configurations, improving end-to-end decode
  performance by 2% to 3%. See
  [SGLang PR #32919](https://github.com/sgl-project/sglang/pull/32919).

## RL: LoRA Training on the Native NVFP4 Base

Day-0 RL for Qwen3.8 is colocated LoRA training with
[Miles](https://github.com/radixark/miles): a BF16 Megatron trainer and native
NVFP4 SGLang rollout engines sharing the same 64 GB300s, with rank-32 adapters
on the attention projections trained with GRPO. We verified the setup with a
short GSM8K training run: reward and eval score climb steadily while the
train/rollout KL stays flat.

<img src="/images/blog/qwen3-8-day0-support/fig-rl-gsm8k.png" alt="Qwen3.8 LoRA RL on GSM8K: eval score, rollout reward, and train/rollout KL" width="100%">

## Acknowledgments

This work was a collaboration among the SGLang & Miles team at RadixArk, Qwen,
Alibaba Cloud, NVIDIA, and AMD.

**SGLang Community**: Qiaolin Yu, Yuhao Yang, Xinyuan Tong, Ke Bao, Zijie Xia, Yi Sun, Mao Cheng, Yueming Yuan, Mingyi Lu, Haoguang Cai, Banghua Zhu, Ying Sheng

**Qwen**: Yi Zhang, Zheng Li

**Alibaba Cloud**: Tao Lan and colleagues

**AMD**: Jacky Cheng, Zijie Chen, Hai Xiao

**NVIDIA**: NVIDIA and SGLang collaborated on kernels for GDN, GEMMs, GQA, and MoE
communication, including the communication fusions described above. The teams also worked on the parallel configurations used in the Qwen3.8 performance results.
