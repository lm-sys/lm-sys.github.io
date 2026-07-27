---
title: "SGLang and Miles Add Day-0 Support for Kimi K3"
author: "SGLang Team"
date: "July 27, 2026"
previewImg: /images/blog/kimi-k3-day0-support/cover-kimi-k3.png
type: blog
---


We are excited to announce Day-0 support for **[Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)**
in SGLang and Miles. K3 is the first open-source model in the 3-trillion-parameter class,
and its hybrid architecture departs from convention in almost every place a serving stack
has assumptions. In collaboration with the Moonshot AI and NVIDIA teams, the two cover K3
in full on launch day: SGLang for inference, Miles for RL training. This post covers what
it took.

**Highlights**

- **A new hybrid architecture.** 2.8T parameters, 69 KDA linear-attention layers
  interleaved with 24 MLA, LatentMoE, and Attention Residuals; most of a serving stack's
  assumptions break somewhere.
- **Memory management for two kinds of state**, including a recurrent state that
  overwrites itself in place, with prefix caching, overlap scheduling and paging rebuilt
  on top, and a unified pool design that removes the last sizing guess.
- **A kernel ladder** reaching ~113 tok/s at batch 1, before speculation.
- **DSpark speculative decoding, with [a draft model we trained for K3](https://huggingface.co/RadixArk/Kimi-K3-DSpark)**: batch-1 decode
  reaches ~423 tok/s. ReplaySSM handles the KDA state, replaying raw inputs instead of
  snapshotting per step, a roughly 32x cut in draft-window memory.
- **Parallelism split by phase**: chunked pipeline-parallel prefill and tensor-parallel
  decode, composing under PD disaggregation to **2,808** tok/s per GPU.
- **LoRA RL with Miles on the native MXFP4 checkpoint**, colocated trainer and rollout
  on the same GPUs: AIME-2024 43.3% to 76.7% over a 12-hour run.

Launch commands and per-workload configuration guidance live in the
[Kimi K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3).


## Bringing up Kimi K3

[Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) is the first open-source model in
the 3-trillion-parameter class: 2.8T parameters, a 1M-token context window, and native visual
understanding. Its predecessor K2.5 was architecturally a close
relative of models SGLang already served well, so most of the stack simply applied. K3 is not like
that — it departs from convention in several independent places at once:

| | Kimi K2.5 | Kimi K3 |
|---|---|---|
| Scale | ~1T parameters | 2.8T parameters |
| Attention | all-MLA, 61 layers | hybrid: 69 KDA linear attention + 24 MLA, 93 layers |
| Attention Residuals | none | every attention output banked, aggregated per block |
| MoE | 384 experts, top-8 | LatentMoE: 896 experts, top-16, run in a 3584-dim latent space |
| Expert activation | SwiGLU | SiTU |
| Vision tower | MoonViT | MoonViT3d, a new K3-specific stack |

Each of these rows is a serving problem. The hybrid attention stack is what makes the 1M context
affordable, but it means the server holds two kinds of state at once: one fixed-size KDA state per
request next to MLA's per-token KV — the memory-management sections below are about exactly this.
Attention Residuals thread a bank of attention outputs through the whole stack, which breaks the
assumptions SGLang's standard layer plumbing makes and forced K3-specific paths in places like DP
attention. LatentMoE routes 16 of 896 experts inside a down-projected latent space with the SiTU
activation instead of SwiGLU, so no existing MoE kernel applied out of the box. And the vision path —
tower, projector, processor, and Kimi's XTML media format — was brought up from scratch. The sections
below all build on this bring-up.

## Hybrid KDA Memory Management

Attention KV is append-only. Once a token's KV is computed it never changes, which is what lets the scheduler share one physical copy across every request that shares a prefix, keep it in a radix tree, and pipeline it across iterations without a second thought. A KDA layer's state is the opposite, one fixed-size recurrent buffer that is **overwritten in place at every token**. So everything the scheduler hands KV for free, from prefix caching to the overlap scheduler to speculative decoding to paging, has to be rebuilt for a value that mutates as it is read. K3 interleaves 69 KDA layers with 24 MLA layers, so this is not a corner case. It is most of the model. What follows is that rebuild, and its one satisfying property is that every piece is forced by the overwrite-in-place fact. None of it is a bolt-on.

### Race-free state movement

A request's live state sits in a single working slot, read and overwritten in place by every forward. Caching it means copying out of memory the GPU is actively mutating, and that creates two races. A restore can collide with the forward that is writing the slot. And the snapshot that refreshes the cache can overwrite the previous snapshot while it is being handed to the tree. Both close with no device-wide synchronization and no lock on the hot path.

First, every state copy is a kernel on the serial forward stream. The copy-on-write that restores a cached checkpoint into the working slot, and the snapshot that captures the state at a track boundary, are enqueued between the forwards that produce and consume that state, so same-stream ordering provides the happens-before for free. A snapshot is taken once per prefill chunk and once every track interval during decode.

Second, the only move that leaves the request transfers no bytes. Snapshots land in a ping-pong pair we call the extra buffer, where one slot holds the latest snapshot while the other takes the next one, and that second slot is allocated only at the boundary that needs it and freed right after. Caching donates the latest snapshot's slot index to the tree, after each prefill chunk, at the handover from prefill to decode, and when the request finishes. A fresh slot backfills the buffer, so the snapshot's write target and the donate's read source are never the same physical slot.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig1-state-flow.svg" width="98%" alt="The three KDA state moves, copy-on-write, snapshot and donate, and where each sits relative to the serial forward stream.">
</p>

<p align="center">
  <em><b>The three state moves.</b> Copy-on-write, snapshot, and donate, and where each
  one sits relative to the serial forward stream.</em>
</p>

With this design, the state can be captured safely and efficiently at any point, so features like prefix caching, overlap scheduler, paged KV and speculative decoding all work smoothly with recurrent state.

### Prefix caching for recurrent state

A recurrent state cannot be sliced at an arbitrary token, since you cannot run it backwards to an earlier position, so it is checkpointed only at chunk boundaries, and sparsely even there. A per-path cap and LRU keep just a few checkpoints alive on each path, and a checkpoint can be evicted independently of the KV it annotates, leaving the node as a tombstone. Branch points get special treatment, an idea from [Marconi](https://arxiv.org/abs/2411.19379). A fork is the one prefix every future branch is guaranteed to share, so when a request diverges mid-edge it replays from the nearest checkpoint above and plants a new checkpoint at the chunk-aligned fork. The next branch restores there directly, no replay.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig2-radix-branching.svg" width="98%" alt="Sparse KDA state checkpoints overlaid on the radix tree, and the branching point where a new request diverges from a cached prefix.">
</p>

<p align="center">
  <em><b>Checkpoints on the radix tree.</b> The sparse checkpoint overlay and the
  branching point.</em>
</p>

### Constant state slots per request

Because the reusable prefix checkpoints live in the shared, evictable radix tree, a running request reserves only a handful of transient slots, as few as four. That is its working slot, one extra-buffer slot for snapshots, and two slots of retention headroom, one for the committed prefix state that has to stay resident and one for the copy a diverging branch needs. The bulk of cached state is amortized across the whole tree rather than charged per request, so the state pool does not have to grow with how much history each request keeps warm.

### Unified memory: one pool for both kinds of state

Everything above manages each kind of state inside its own pool, and the pools themselves
are the one guess left in the design. The two allocation units are three orders of magnitude
apart: one large KDA state block per request (about 54 MB under TP=8, covering all 69 KDA
layers) and one small MLA KV block per token (about 27 KB, covering all 24 MLA layers), so
today they live in two separate pools, sized at startup. That sizing is a bet on the
traffic, and when the bet is off the server runs out of memory in one pool while the other
still has room to spare.

Unified memory replaces the two pools with one: KDA states fill in from one end, MLA KV blocks from the
other, and the unused bytes between them form a single free region.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig-unified-memory.svg" width="98%" alt="Unified memory: two static pools versus one pool holding both kinds of state, and the freeing sequence">
</p>

<p align="center">
  <em><b>Unified memory.</b> Top: today the two kinds of state have separate pools, sized at startup, so
  one can sit idle while the other is full. Middle: with unified memory both kinds allocate from the same
  pool, growing in from opposite ends with a single free region between them. Bottom: freeing a state
  block in the middle leaves a gap, and a block from the end is moved into it so the free region stays in
  one piece.</em>
</p>

Freeing is just as simple. When a request finishes, is aborted, or is retracted under pressure, its
blocks are released. If that leaves a gap in the middle, a block from the end is moved into it so the
free region stays in one piece.

This layout gives us flexible page sizes and no memory fragmentation: a 54 MB KDA state block and a 27 KB MLA KV
block draw from the same bytes with no common page size forced on them, and the moves above keep each
end packed at a negligible cost for state movement, so the free space is always one contiguous region
usable by either kind. Capacity therefore follows the workload rather than a
startup flag: many short requests fill the pool with state blocks, a few long contexts fill it with KV,
and neither case needs anything reconfigured.

Unified memory ships opt-in behind `--enable-unified-memory`. A follow-up post will go through the
implementation in detail.

## Speculative decoding with DSpark

K3 ships with DSpark block speculative decoding, driven by
[a draft model we trained for K3](https://huggingface.co/RadixArk/Kimi-K3-DSpark).
Two parts of the integration deserve their own story: spending the verification budget
only where it pays, and making K3's recurrent KDA state survive speculation at all.

### Verify only what is worth verifying

DSpark proposes a block of draft tokens per step, and the target verifies the whole block
in one forward. At batch size 1 the extra verify positions are essentially free: the step
is latency-bound, and a few more tokens ride along. As the batch fills, that stops being
true. Verify tokens now compete with every other request for the same step time, and most
of them lose their bet anyway: on a chat workload the accept length is around 2.7, so five
of the eight verified positions on a typical step get rejected. Verifying everything pays
full price for tokens the server then throws away.

The pieces to fix this were already in the system. The draft carries a trained confidence
head that predicts, per position, how likely each token is to survive verification. On the
other side, a one-time profile of the server records what an extra verify token actually
costs at each load level. A per-step planner joins the two: each request keeps verify
tokens only while their expected value covers the marginal cost, and the rest of the
window is trimmed before the target forward launches. What remains is verified exactly as
before, so outputs stay lossless. The trade is a slightly shorter accepted run for a
cheaper step.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/f4-verify-all-vs-trim.svg" width="98%" alt="Decode throughput vs batch size, verify-all vs confidence-scheduled trim, on a chat panel and a few-shot math panel.">
</p>

<p align="center">
  <em><b>Trimming wins under load.</b> Decode throughput, verify-all vs trim, on a chat
  panel (accept ~2.7, left) and a few-shot math panel (accept ~5.0, right). Break-even
  through bs 8, then the gap opens with batch size: +68% and +24% at bs 256, with accept
  length easing from 2.7 to 2.2 and from 5.0 to 4.3 while throughput climbs.</em>
</p>

The measured cost curve turned out to be the interesting part. The marginal cost of a
verify token is not smooth. It is a staircase: flat shelves where another token slots into
the current kernel waves for almost nothing, and steep risers where it starts a new one.
The planner reads this surface. When a request's next tokens sit on a cheap shelf it keeps
them; when they would start a riser it cuts there. In the data this shows up as small
wiggles in the accept-length curve while throughput stays smooth and monotone: the planner
is surfing real hardware shelves, not noise.

Below batch size 8 there is little pressure to relieve, so trimming is break-even to
mildly negative there; a small-batch early-exit in the planner is the known follow-up.

### ReplaySSM: raw-input replay for the KDA state

Speculative decoding verifies γ+1 draft tokens at once and may accept only a prefix. For MLA that is free, since KV is append-only and a rejected draft just releases its slots. A KDA layer's state overwrites itself every token, so the baseline buys reversibility by brute force and snapshots the whole K×V state after every draft step. At K=V=128 that is 64 KB per request, layer and head, times γ+1 steps. Across K3's 69 KDA layers and a full batch it outgrows the persistent state pool it is competing with, and because it is reserved per running request, it caps concurrency.

**Store the inputs, not the state.** [ReplaySSM](https://tridao.me/blog/2026/replayssm/) drops the snapshots. The verify kernel reads the committed checkpoint and never writes it, and on the way through it also stores each step's raw inputs `Sᵢ = (vᵢ, kᵢ, gkᵢ, βᵢ)`, about 1 KB against the 64 KB a snapshot costs. Once the sampler fixes the accepted length, a single fold kernel covering every layer and head replays just the accepted prefix from the checkpoint and advances it in place. Rejected drafts are never replayed, so rollback costs nothing. The draft window goes from 512 KB to 16 KB, roughly 32×.

**Exact, not approximate.** The fold is a verbatim clone of the verify recurrence, same tiles and same reduction order, and it consumes the gate values the verify kernel itself stored rather than recomputing them. That second half is not optional. An early version recomputed the gate on the torch side with a subtly different formula, which left every output looking correct while the state quietly drifted underneath. The rebuilt state is now bit-identical to what the recurrent baseline would have committed.

This optimization pays off at both small and large batch sizes. At small batch sizes, storing raw inputs instead of snapshots cuts the verify kernel's memory access, and the replay itself is one fused kernel to reduce launch overhead, so per-step time already improves. At large batch sizes the effect is much larger. The snapshot scratch was pure speculation overhead, growing with both batch size and γ, so it squeezed the state pool hardest exactly when γ was large enough to be worth using. Handing that memory back lifts the concurrency ceiling several-fold. Once the baseline starts queueing requests, ReplaySSM keeps admitting them, and that is where the gap opens.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig-replayssm-kda.svg" width="98%" alt="KDA ReplaySSM: the verify kernel reads the committed checkpoint and stores each draft step's raw inputs in a small per-slot buffer; after acceptance a single fold kernel replays only the accepted prefix and advances the state in place.">
</p>

<p align="center">
  <em><b>KDA ReplaySSM.</b> Verify (one fused launch per layer) reads the checkpoint h₀
  without writing it and stores each draft step's raw inputs into a per-slot buffer, one
  per layer and head. After the sampler fixes the accepted length, the fold kernel (one
  launch for all layers and heads) replays only the accepted prefix and overwrites the
  slot in place, running the verbatim recurrence with the verify kernel's own stored gate
  values, so the rebuilt state is bit-identical to the recurrent baseline.</em>
</p>

## Kernels

A single-sequence decode step on a 2.8T hybrid model is not compute-bound —
it is a launch-count and latency problem: 93 attention layers (69 KDA + 24
MLA) plus 92 latent-MoE layers per token, and at bring-up the step fired
hundreds of tiny kernels. The campaign was profile-driven: fuse one thing,
A/B it on the fixed protocol, gate on GSM8K, repeat.

![bs=1 optimization-category waterfall](/images/blog/kimi-k3-day0-support/fig2-bs1-category-waterfall.svg)

**Launch & copy elimination (P1–P4, +19.9 tok/s).** Make the step smaller,
not the kernels faster: the MoE front collapses into one GEMM and KDA's
paired skinny projections merge (P1); a profile-guided sweep removes
per-layer upcasts, copies and spare launches (P2, P3); routing becomes a
one-pass register-resident radix select over the [M, 896] logits (P4).

**NVIDIA compute kernels (P5–P8, +10.3 tok/s).** Four stages swap in kernels
co-developed with or provided by NVIDIA where the default was wrong for the
bs=1 shape: the fused KDA decode kernel, the trtllm-gen W4A8 SiTU MoE cubins
behind our routing bypass, the TMA attention-residual aggregation, and the
CuTe-DSL TGV bf16 GEMMs that replace cuBLAS at small M — each through the
same A/B and accuracy gate as everything else.

**Communication fusion (P9, P12, P13, +27.6 tok/s).** The largest era: a
fused all-reduce family for the MNNVL fabric on CustomAllReduceV2's
symmetric-memory plane — one-shot multicast stores for small messages, NVLS
in-switch reduction for large ones, with the residual add and RMSNorm riding
inside the collective (P9). The MoE finalize then moves into the collective's
staging pass (P12), and the up-projection flips from a replicated to a
column-parallel GEMM completed by a multicast all-gather (P13).

**Overlap & prologue fusion (P10, P11, P14, P15, +10.4 tok/s).** The rest
trims the remaining critical chain: a strided-input MXFP8 quant (P10),
residual write-back fused into upstream kernel tails with independent
branches on a side stream (P11), KDA's GEMV chain overlapped with the qkvg
GEMM (P14), and the MLA decode prologue fused into one kernel with PDL
arming the attention kernel behind it (P15).

The chronological ladder behind the four bars:

![bs=1 decode throughput ladder](/images/blog/kimi-k3-day0-support/fig1-bs1-throughput-ladder.svg)

| Stage | Optimization | tok/s | Basis |
|---|---|---:|---|
| P0 | bring-up baseline (Marlin W4A16 MoE) | 44.3 | measured |
| P1 | MoE-Front GEMM + KDA GEMV Merge ³ | 53.2 | itl-rescaled |
| P2 | Launch/Copy Elimination + O-Gate Fusion ³ | 61.9 | itl-rescaled |
| P3 | Fused Mamba Track + Batched Router | 62.4 | measured |
| P4 | Radix-Select Router | 64.2 | measured |
| P5 | Conv + KDA + Onorm Fusion (NVIDIA kernel) ¹ | 65.3 | measured |
| P6 | W4A8 SiTU MoE Cubins (NVIDIA) + Routing Bypass | 71.0 | measured |
| P7 | TMA Attn-Residual Aggregation (NVIDIA kernel) | 72.0 | measured |
| P8 | CuTeDSL TGV Dense GEMM (NVIDIA kernels) | 74.5 | measured |
| P9 | NVLS AR + RMSNorm Fusion | 84.3 | measured |
| P10 | Zero-Copy MXFP8 Quant | 85.2 | measured |
| P11 | Res-Write Fusion + Multi-Stream | 90.2 | measured |
| P12 | LL AR Fusion (MoE finalize in-collective) | 92.0 | measured |
| P13 | Column-Parallel GEMM + Multicast AG ² | 108.0 | measured |
| P14 | KDA GEMV Side-Stream Overlap | 111.4 | measured |
| P15 | KV-Scatter + Q-Concat Fusion (+PDL) | 112.5 | measured |

**The lesson that generalizes.** All-reduce is a synchronization point, so a
microsecond saved there converts one-for-one into step time; a kernel sitting
in another stream's overlap slack converts at roughly one-tenth. Checking
critical-path membership in the trace before writing a kernel was the single
highest-leverage habit of this campaign.

¹ A rebase moved the P5 baseline 64.2 → 63.8; the step is measured against
the post-rebase base.
² P13 is a re-calibrated canonical baseline after a merge window, not a
single-PR attribution.
³ Segments leading into P1–P4 also carry interim optimizations later
superseded (concat all-reduce → P9; tiny/1-CTA GEMV → P8; radix router v1 →
P4; Marlin top-k-sum → P6/P12; early attn-res add → P7); their transient
gains are in the curve but have no named point.

## Parallelizing K3

For K3's hybrid architecture, the conventional parallelism choices do not hold up. Tensor
parallelism cannot shard MLA's KV cache (one KV head, nothing to split by), so every rank
holds a full copy; it also slices every GEMM eight ways and pays a collective per layer.
Pure DP attention instead replicates the attention weights on every rank, around 61 GB for
KDA plus 11 GB for MLA, memory the KV cache and KDA states need. Prefill and decode fail
differently under these costs, so K3 splits the answer by phase: chunked pipeline
parallelism for prefill, context parallelism for decode.

### Prefill: chunked pipeline parallelism

In TP prefill, every layer ends in an AllReduce, a barrier that cannot overlap with
compute. Pipeline parallelism cuts the model by layer instead: K3's 93 layers become 8
stages, and the prompt is cut into chunks that stream through them:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig-pp-chunked.svg" width="98%" alt="Chunked pipeline-parallel prefill: a long prompt cut into chunks streaming through pipeline stages, with P2P hand-off arrows running under the next chunk's compute, contrasted with TP where every layer ends in an AllReduce barrier.">
</p>

<p align="center">
  <em><b>Chunked pipeline-parallel prefill.</b> The stages work on different chunks at the
  same time. The hand-off between stages runs while the stage computes its next chunk, so
  91% of it is hidden on K3. Under TP (bottom strip), every layer ends in an AllReduce
  that all ranks wait for.</em>
</p>

This wins three ways. The only communication left, the hand-off to the next stage, hides
behind the next chunk's compute. Each rank runs whole layers, so the GEMMs are eight times
wider and more efficient. And each stage holds KV and activations for only its ~12 layers,
which makes very long prompts comfortable to prefill. The pipeline does have to be deep:
a shallow PP4×TP2 cannot cover its hand-offs and still pays TP2's AllReduce, and it
benchmarks no better than TEP8.

Measured on 8K prefill across 2×4 GB300, with topology as the only variable:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig-pp-prefill.svg" width="98%" alt="Left: prefill throughput per GPU vs concurrency, TEP8 vs PP8xTP1. Right: cost decomposition per 1k tokens, compute plus exposed communication, for TP8 / PP4xTP2 / TEP8 / PP8xTP1.">
</p>

<p align="center">
  <em><b>Deep PP wins on both axes.</b> Left: past the c1–c4 crossover, PP8×TP1 climbs to
  about 1.7× TEP8's ceiling, with lower TTFT. Right: cost per 1k prefill tokens at equal
  per-rank FLOPs; PP4×TP2 and TEP8 trade compute against communication and tie, PP8 is
  cheapest on both.</em>
</p>

PP8 loses only at a single request, and a prefill worker that idle is misconfigured
anyway. In K3's disaggregated serving, the prefill nodes run PP8. Each of them has 1.45 to
1.72 times the prefill capacity of a TEP8 node, so one prefill node can keep several
decode nodes fed. The decode nodes run TP or DCP, which the next part covers.

### Decode: context parallelism

Decode is where the replicated KV cache binds: under TP, admitting one more request or one
more thousand tokens of context costs the same bytes on every rank. Decode Context
Parallelism (DCP) shards MLA's KV by token position instead of by head. Rank r owns
position p when p mod N equals r, so every rank holds an interleaved 1/N of every
request's context, and the sharding is invisible above the attention kernel:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig1-kv-layout.svg" width="98%" alt="TP replicates the same token positions on every rank; DCP stripes them round-robin, so the same GPUs hold N times the logical context.">
</p>

<p align="center">
  <em><b>Sharding by position.</b> The same 16 token positions on 4 ranks: TP stores 64
  physical copies, DCP stores 16. The freed bytes become logical KV capacity, about 7.9×
  on K3 with DCP8.</em>
</p>

Position-sharding breaks the softmax, since each rank sees only 1/N of the keys and
partial softmaxes do not add. The fix is FlashAttention's own: each rank returns its
partial attention output together with a log-sum-exp per head, and one all-to-all per
layer exchanges them so that each rank ends up with 1/N of the heads over the full
context. A local merge by log-sum-exp is then exact, and the result is already the head
layout the output projection expects under TP. One collective per layer is the entire
communication cost:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig2-dcp-dataflow.svg" width="98%" alt="Per-layer MLA decode dataflow on two DCP ranks: replicated query projection, local attention over owned positions producing partial output and LSE, one packed all-to-all, local LSE merge.">
</p>

<p align="center">
  <em><b>The decode step, per layer.</b> Each rank projects the full-head query locally,
  attends over only the positions it owns, and ships partial outputs and their
  log-sum-exps in a single packed all-to-all. A local merge leaves the standard TP head
  layout; nothing downstream of attention changes.</em>
</p>

Everything else is left alone. DCP groups are built inside the TP group, so TP8 with DCP8
is still 8 GPUs, and the MoE runs whatever parallelism it already had. KDA is the
exception that explains the rule: its state is one fixed-size matrix per request rather
than per token, so there is no position axis to shard, and the KDA layers stay TP-sharded
by head. The whole feature is one flag, `--dcp-size N`.

What this buys shows up on agentic traffic, where hundred-thousand-token sessions pile up
in the cache. Replaying real coding-agent sessions on 2×4 GB300, with a host-memory KV
tier on both arms and DCP as the only difference:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig4-dcp-vs-tp-hicache.svg" width="98%" alt="Aggregate decode throughput versus concurrent agent sessions, TP8 plus hicache versus DCP8 plus hicache: TP8 collapses at 16 sessions when the active set outgrows device KV, DCP8 keeps climbing to 541 tok/s at 48 sessions.">
</p>

<p align="center">
  <em><b>DCP removes the active-set wall.</b> The host tier rescues re-prefill traffic,
  but at 16 concurrent sessions the active working set outgrows TP8's device KV and
  throughput collapses. DCP8 lifts logical KV from 1.5M to 12.2M tokens and carries the
  same workload to 541 tok/s at 48 sessions.</em>
</p>

DCP composes with the rest of the stack. A DSpark verify step is a decode step, so it
rides the same replicated-Q, one-all-to-all path; under PD disaggregation the prefill
side stays DCP-unaware and each decode rank simply pulls the positions it owns at the
transfer boundary, which is what lets PP or TP prefill feed DCP decode. The remaining
ceiling is KDA: its per-request state cannot be position-sharded, so once DCP lifts the
MLA wall the running-request cap becomes the binding limit; the unified-memory design in
the memory section above takes that on.

### Combining all strategies together

How the pieces compose is ultimately a measurement. Putting PD disaggregation in the
loop, with the prefill topology, decode topology and prefill:decode ratio varying, gives
the serving frontier:

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/fig-pareto-pd.svg" width="98%" alt="K3 serving frontier under PD disaggregation: total throughput per GPU versus per-user decode speed for PP8 and TP8 prefill feeding TP8, DCP8, EP and multi-instance TP8 decode arms; the frontier runs from PP8 to TP8 at the throughput end out to TP8 feeding three TP8 instances at the interactive end.">
</p>

<p align="center">
  <em><b>The serving frontier.</b> At the throughput end one PP8 prefill worker feeding
  one TP8 decode node delivers 2,808 tok/s per GPU on the fp4 arm, with the DCP
  composition, two PP8 prefill workers feeding two DCP8 decode nodes, just behind at
  2,633. Moving right is the prefill:decode knob at work: one prefill worker feeding two,
  three, then four independent decode instances trades aggregate throughput for per-user
  speed, out past 116 tok/s per user.</em>
</p>

## RL: LoRA Training on the Native MXFP4 Base

Day-0 RL for K3 is colocated LoRA training with Miles: a BF16 trainer on Miles's Megatron backend and native packed MXFP4 SGLang rollout engines sharing the same 64 GB300s. The backend covers KDA, NoPE-MLA, the attention-residual bank and the latent MoE, with TP/SP/PP/CP/EP.

### LoRA serving and weight sync

The engines serve the checkpoint as shipped and never rewrite it. Each step transfers only BF16 LoRA adapters, and the engine applies the delta as a separate BF16 `B(Ax)` term on top of the quantized base GEMM, so the policy update reaches inference at full precision over 4-bit base weights. Dense projections go through SGLang's Triton LoRA backend, the 896 routed experts through a fused MoE-LoRA kernel on the Marlin path, and the shared-expert delta is folded into the fused MoE front GEMM. Adapters live in a GPU memory pool that each sync replaces in place, so there is no resident BF16 copy on the rollout side, no full-weight resync, and no requantization step in the loop.

### Parallelism

**Pipeline parallelism.** K3's attention-residual snapshot bank has to cross stage boundaries, but Megatron's point-to-point carries one hidden-states tensor, so the stage boundary packs `[prefix_sum, bank]` into it and unpacks on entry. Megatron is untouched.

**Context parallelism.** The bank shards for free, since every attention-residual operation is per-token. MLA needs no K3-specific code either: rotary-table slicing is the only CP-aware work in Megatron's MLA and K3 has no rotary embedding, so its projections sit on the stock TE attention core and inherit CP. KDA is the part that needs work, going through fla's CP context for the recurrent state and the convolution halo. It wants a contiguous rank-local chunk where Megatron stores the zigzag order ring attention expects, so the relayout runs only around KDA.

**Expert parallelism.** The adapter shares its latent-side factor across all 896 routed experts and keeps the other per-expert, matching the engine's fused MoE-LoRA contract. That shared factor is replicated across EP but tagged expert-parallel, so DDP reduces it only over expert-DP, and the EP sum is the one gradient reduction the release adds itself.

### Memory

A native-MXFP4 rollout peaks near 225 GiB/GPU and the BF16 trainer near 155 at initialization, on a 277 GiB card, so the two are never both resident: each sleeps while the other runs, and colocation depends on every handoff leaving nothing behind.

**The base never moves.** The engines keep their base weights resident on the GPU for the whole run, and only the KV cache and CUDA graphs are released while the trainer works. Since the base is never released it is never restored either, and the LoRA path ships no base bytes at all: base sync is skipped entirely.

**The process groups stay up.** The trainer's NCCL communicator buffers are not torch-allocator memory, so offloading cannot release them, and the obvious answer is to destroy the groups on sleep and rebuild them on wake. That trades a resident few GiB for a per-cycle rebuild and EP warmup, which is only worth paying when the engines need those bytes back — and they do not here, because no base weights cross the update path. The groups stay up.

**Adapter transfer is bounded per chunk.** The adapter is about 2,800 tensors. Each chunk ships as one flattened CUDA IPC bucket, 278 in total, and is collected as soon as the receiver acknowledges it and every producer rank has crossed an engine-group barrier. That bounds transient IPC memory per chunk instead of letting it integrate across the transfer, worth about 48 GiB/GPU at the peak.

**Host copies exist only where something reads them.** The engine releases each adapter's CPU copy once it is installed into the GPU pool, taking scheduler RSS from 76–88 GiB to about 17; an adapter whose CPU copy is gone cannot be reinstalled, so pool eviction is refused with an error rather than silently serving a stale slot. On the trainer side the DDP buffers split by lifetime: adapter parameter buffers stay in the CPU-backed region because the weight update reads them while the trainer sleeps, while gradient buffers are rebuildable and go to a no-backup region that sleep discards.

### Validation

Training correctness is the most important part of an RL support stack. We built these checks before the recipe work.

**Train/rollout KL**, logged every rollout, is a Schulman k3 estimate of KL(rollout ‖ train) over the sampled tokens, computed from the log-probabilities the engine returned and the trainer's recomputed ones. It is a diagnostic, not part of the objective, which carries no KL penalty. Its floor is about 2e-3, set by serving the base in MXFP4; growth with step count is the divergence signature.

**A canary lockstep probe.** One trajectory is pinned from the first rollout and scored every step by both the engine, under the live adapter, and the trainer, at the same policy version. Co-movement of the two curves is what proves the trained adapter reaches inference: a transfer checksum proves the bytes arrived, not that anything reads them.

**Tensor-level dump comparison.** The same tokens run through two builds or two parallel layouts, dumping every forward activation and every parameter gradient, compared by relative L2 and cosine against a measured noise floor. Bit-exactness is not the bar, since the same arithmetic on two different GPU sets already differs at the ulp level. This is how the pipeline boundary and the context-parallel layout were validated: where CP=2 and CP=1 build the identical call the two are bit-identical, and a forward-kernel swap lands at cosine median 0.996.

**Weight-sync assertions.** Per-tensor SHA256 manifests between trainer and engine, a validator that errors if an adapter update changes no exported tensor, a check that every `B` factor is zero at version 1, and adapter gradient and optimizer-step checks.

### Training result

The reported run is DAPO math on 16 nodes × 4 GB300: BF16 trainer at TP8 / PP8 / EP8 colocated with native MXFP4 rollout engines, 4096-token responses, 64 samples per rollout, one optimizer step per rollout, rank-32 / α-64 LoRA at lr 1e-5, GRPO with no KL term, run to a 12 h wall-clock limit. AIME-2024 greedy eval climbs 43.3% to 76.7% over 60 steps, 13 of 30 problems to 23, and is still rising at the cutoff, while train/rollout KL holds flat at the ~2e-3 MXFP4 floor for the whole run.

<p align="center">
  <img src="/images/blog/kimi-k3-day0-support/rl-training-curves.svg" width="100%" alt="Three RL training curves side by side: rollout raw reward trending upward, AIME-2024 eval accuracy climbing, and train/rollout KL flat at the MXFP4 quantization floor.">
</p>

<p align="center">
  <em><b>The 12-hour DAPO run.</b> Left: mean reward over the 64 sampled responses per
  rollout, graded on the response channel; each rollout draws a different prompt batch, so
  the trend is the signal and the per-step value is noise. Middle: AIME-2024 greedy pass
  rate, evaluated every 10 rollouts under the same 4096-token limit as training, so a
  correct solution that runs past the limit scores as a miss. Right: a Schulman k3 estimate
  of KL(rollout ‖ train) at the sampled tokens, reported only and never added to the loss;
  flat at the quantization floor is the target, and monotonic growth is the divergence
  signature.</em>
</p>

## Acknowledgments

This work was a collaboration between the SGLang & Miles team at RadixArk and the Moonshot AI team, together with NVIDIA, AMD, Approaching AI, Baseten, and Modal.

**AMD**: Wun-guo Huang, Xinyi Song, Hai Xiao, Soga Lin, Duyi Wang

**Approaching AI**: Huanming Shen, Xiaohao Zhang, Nan Li, Mingxing Zhang

Thanks to DigitalOcean for providing AMD instances for our testing.

Thanks to Google Cloud, DigitalOcean, Nebius, fal, RunPod, DeepInfra and GMI Cloud for serving Kimi K3 on SGLang.
