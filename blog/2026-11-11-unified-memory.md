---
title: "Unified Memory in SGLang: One Arena for Every Token Lifetime"
author: "Cheng Wan, Caihua Li, Yonghao Zhuang, Seokwoo Song, Shuwen Wang, Lianmin Zheng"
date: "November 11, 2026"
previewImg: /images/blog/unified-memory/fig2-one-arena.svg
type: blog
---

**TL;DR.**

- **One arena, no partitioning knob.** Hybrid models combine full-attention KV, sliding-window KV, and recurrent state. Traditionally, you must partition GPU memory among them at startup, before knowing the workload. With `--enable-unified-memory`, all state types share one arena, a single region of GPU memory, and capacity dynamically follows demand. No fixed split to configure or tune.
- **Any page size, zero fragmentation.** Each state type keeps its own page or slot size, and you can choose any `--page-size`. A state slot can be many times larger than a KV page, so naively sharing memory can strand free capacity in unusable gaps. Unified memory fully reclaims freed pages, using compaction to relocate live allocations and make capacity available wherever it is needed.
- **A new KV layout: one entry per token.** Each token's KV rows across all layers are packed into a single entry, allowing them to be allocated, freed, and moved together. Rows of different shapes and dtypes can share an entry, so speculative draft KV can use the same allocation and lifecycle as target KV even when their dtypes differ. Kimi K3.1's DSA layers need this too: their indexer keys are packed alongside the MLA KV.
- **One memory system, end to end.** A unified allocator manages all state types, while a shared demand model drives capacity decisions across the serving stack, from request admission to decode-side preallocation in prefill–decode disaggregation. A scheduler-aware page lifecycle safely reclaims and relocates memory while GPU operations are still in flight.
- **Works with the main serving features.** Unified memory works with overlap scheduling, Radix cache, HiCache, prefill–decode (PD) disaggregation, speculative decoding, pipeline parallelism, decode context parallelism, and data-parallel attention.

## Hybrid models and the memory split

Attention layers need their past to compute the next token, but not every layer remembers it the same way:

| State | What is retained | What drives its memory demand |
|---|---|---|
| Full-attention KV | The whole context | Sequence length, across every full-attention layer |
| Sliding-window KV | A bounded suffix of the context | Sequence length up to the window, across the window layers |
| Recurrent state | A fixed-size state per running request | Concurrency, not context length |

GPT-OSS and Muse Spark interleave full and sliding-window attention. Qwen3.5 interleaves full attention with linear-attention layers. Kimi K3 pairs KDA linear attention with MLA, and K3.1 replaces the MLA layers with DeepSeek Sparse Attention (DSA).

Their demands do not scale together. A long-context workload grows full-attention KV; many short concurrent requests need more window KV and state slots even though each request's history is small. The pressure can shift within one request, too: its state slots are taken when it starts, while its full-attention KV keeps growing.

SGLang's default design gives each kind of state its own pool, a *sub-pool*, and fixes their sizes at startup through `--swa-full-tokens-ratio` and `--mamba-full-memory-ratio`. The right ratio depends on traffic you do not know yet. A long-context workload can exhaust the full-attention pool while much of the pool for the bounded states (window KV and recurrent state) sits idle, and many short requests do the reverse. The free bytes exist, but the other pool is not allowed to use them. Choosing the ratio well means profiling your workload, and when the traffic changes, memory is stranded behind the earlier guess.

In this post a *unit* is a sub-pool's allocation unit: a page of KV, or a slot of recurrent state. Each sub-pool has its own unit size, set by its layers.

![Separate full-attention and bounded-state pools under long-context and many-short-request workloads](/images/blog/unified-memory/fig1-separate-pools.svg)

*Figure 1. Each pool hands out units wherever its free list points, so live data may be scattered through it. Two workloads strand free units in opposite pools. “Bounded” can be sliding-window KV or recurrent state; orange units are drawn wider because each sub-pool has its own unit size.*

Unified memory replaces the partition with one arena. Full KV takes the high end and grows down; bounded state takes the low end and grows up. Lazy compaction moves live data toward those two ends, so the free bytes form one region in the middle that belongs to no sub-pool until one of them needs it, and each side takes bytes from it in its own unit size. The ratio flags remain only as boot-sizing targets, not walls at run time. Nothing is lost to fragmentation: memory is handed out in whole pages, so a page freed by one request can be reused in full by the next, and compaction only moves free bytes to the sub-pool that needs them.

![One arena with full-attention and bounded-state bands sharing the same bytes under both workloads](/images/blog/unified-memory/fig2-one-arena.svg)

*Figure 2. The same bytes and demand, now in one arena. Live data is compacted toward the two ends, leaving one free region that either side can grow into.*

## A new KV layout: one entry per token

Separate pools keep one buffer per layer. Unified memory instead packs every layer's rows for one token (or one state slot) into a single *entry*, and a page is a run of entries. An allocation, eviction or whole-page move therefore acts on all layers at once, and a layer's view strides over the entries.

Because the arena is a byte region, entry size is each sub-pool's own: a state slot holds every state layer's conv and temporal state, a KV entry holds every layer's K and V. Token sub-pools allocate in pages of whatever `--page-size` you choose and the state sub-pool in single slots, so no kind of state is forced onto another's page size.

Every group of rows in an entry carries its own shape and dtype: a state entry keeps conv and temporal state in their own dtypes, and a fused draft's rows keep theirs next to the target's K and V. This is what lets a compatible speculative-decoding draft place its KV in additional lanes inside the target's entries. The draft's rows can differ from the target's in size and in dtype, which is why rows are packed per token instead of one same-sized pool per layer. Target and draft share a token id, and one allocation, free or whole-page move carries both. K3.1 needs this: each token of a DSA layer has a top-k index beside its MLA KV, and the design fuses the two in one buffer, so the index is allocated, freed and moved together with the KV it points to.

![Per-token entries whose target-layer rows and draft lanes differ in size, with strided per-layer views](/images/blog/unified-memory/fig3-token-entries.svg)

*Figure 3. Target rows and draft lanes can differ in size, so each token's rows are packed into one entry. Per-layer views (here layer 0 and draft lane 0) stride over those entries, and one token id covers both.*

## One arena, different allocation geometries

Putting two tensors in one device buffer is easy. Making their capacities fungible is harder. A full-attention page and a recurrent-state slot can differ greatly in size, and neither should have to round its demand up to the other's allocation unit.

An arena is therefore a byte region, not a uniform array of identical pages: pages and slots share the same bytes without sharing a geometry.

A sub-pool occupies a contiguous *band* of the arena. The two end bands grow inward: bounded state grows up from the low end, and full KV grows down from the high end. When another sub-pool sits between them, it is a *floating* band that can move to open a useful gap. Live bands never overlap one another's storage. Growing, shrinking, compacting and relocating these bands changes who can use the free bytes, not where a request's data is addressed.

### Composing all three state types

The same composition works when a model has all three kinds of state, as Inkling does: full attention, sliding-window attention and a per-request convolution state. The planner creates a full-KV sub-pool, a window sub-pool and a state sub-pool, each with its own lifetime and allocation unit.

Their bands own disjoint bytes. The allocator grows and rearranges them with the same demand and layout rules used for two kinds; there is no special full + Mamba + SWA allocator. This is an example of composition, not a third memory model.

### Request-scoped state

An arena-backed state entry can carry more than the usual recurrent tensors. For example, the per-request ring buffers that linear-attention ReplaySSM (GDN and KDA) keeps for replaying recurrent state, in both speculative and ordinary decode, are additional payloads in the state entry. They share its allocation and movement model rather than requiring a separate pool.

That is the larger direction of the design: model and request state can be declared and managed by one memory system instead of one allocator per feature.

## Admission: can the whole request fit?

Sharing bytes does not reduce a request's footprint to a number of full-attention tokens. Consider a 32K-token request to a model that also has a 4K-token sliding window. With 64-token pages, its steady-state footprint is 512 full-KV pages and 64 window pages. A recurrent-state model would instead include a state slot for the running request. A window may temporarily retain rows beyond its steady-state bound until retirement or a later recurrent checkpoint, so the allocator prices the joint demand it must actually support, not an idealized final footprint.

The same demand calculation serves the memory clip that bounds how much a request may generate, prefill admission and continued commitment across chunks, and PD decode preallocation. It includes page rounding and an expected decode allowance, using a conservative bound where the exact peak is unknown.

Even the right total demand is not enough: the arena must have a layout that can supply it. Each band has *holes* (free pages inside it) and a *frontier* (the outer edge up to which it has handed out pages). A host-side snapshot of these drives a search over three levels of rearrangement:

1. **No move:** reuse holes and extend a band in place.
2. **Compact the ends:** pack the two end bands to release their holes.
3. **Relayout:** also relocate floating bands between the compacted ends.

Figure 4 makes this distinction concrete. The state band is the low end, full KV is the high end, and the window floats between them. The full-KV band has three one-page holes but no free space at its frontier, so three holes cannot serve a four-page demand. Compacting the full-KV end turns those holes into a three-page gap beside the window, still one page short; the window's own hole stays, because compacting the ends does not touch a floating band. Relayout moves the floating band's inner edge: it copies pages from that edge into the band's own holes, so the edge retreats and the freed bytes join the gap. Here one window page, drawn two full pages wide, moves into the window's hole, leaving room for five full-KV pages; four serve the demand and one stays free. The ladder uses only as much movement as the demand needs.

![Three arena layouts showing no move, compacting only the ends and relocating a floating band](/images/blog/unified-memory/fig4-planner.svg)

*Figure 4. Three reusable holes cannot serve a four-page demand, and compacting the ends leaves the full-KV band one page short. Copying the floating window's rightmost page into its hole frees room for five pages beside the full-KV band: four allocated and one left.*

This example also separates **feasibility** from **realization**. Feasibility asks whether some layout can serve the demand. It considers every page movable: a pending page can settle and a transfer can eventually finish. Realization first tries the current layout, then finds a concrete plan that keeps fixed pages in place and avoids active network transfers where possible. A positive feasibility answer is a promise that the allocator can realize the demand, not a decision to block on every unfinished reader or transfer.

The same rule extends beyond one device. With pipeline parallelism, the stages exchange their arena geometry when the pipeline loop starts. Each then simulates every stage from the same logical request counts and admits only a demand that fits all of them, without a capacity exchange per decision.

## Keeping pages safe during in-flight work

Moving bytes would be simple if no device work were still using them. Request tables hold virtual ids rather than physical page numbers; a sub-pool maps each virtual id to its current physical page. A move copies the bytes and redirects that mapping. The next iteration translates the virtual ids into physical pages, so request rows do not need to be rewritten.

The difficult part is time. SGLang's overlap scheduler prepares step N+1 on a schedule stream while step N's forward is still running. There are three cases to keep safe:

1. **Free with an old reader.** A request finishes, but an already-launched forward still reads its KV. The free is logical and immediate: the page counts toward capacity at once. Its physical bytes cannot be overwritten until that forward is processed or settled (the stream waits for its event).
2. **Move with an old reader.** The move need not wait. The allocator copies and remaps the page; new forwards resolve the virtual id to the destination after the copy, while an already-launched reader can keep reading the old source. Only reuse or overwrite of that source must wait for its old readers.
3. **Move with an old writer.** The writer must be ordered before the copy. Otherwise the destination could contain stale data, and the old writer could corrupt the source's next owner. An opportunistic move avoids such a source; a needed move waits first.

The rule is: **logical free is immediate; physical reuse waits.** A freed page under an unprocessed forward becomes pending whether that forward may still read it, write it, or both. It becomes an ordinary hole only after the forward is processed or its event is waited on. There is no separate read-only fast path for an ordinary free.

| State or protection | What it means | When the bytes can be reused or written |
|---|---|---|
| Beyond frontier | Bytes outside every band | A band grows over them |
| Hole | A reusable free page inside a band | Immediately available to a new writer |
| Live | Holds a request's data | Freed or moved |
| Pending | Logically freed while a forward may still read or write it | The forward is processed or settled |
| Vacated (source protection) | A moved source whose old readers may still read it | Its old readers finish |
| Fenced (source protection) | Old bytes beyond a retreating frontier, with an outstanding read | The next writer first waits for the fence |

That is four states plus two protections. A move leaves its destination live through the remap, and new readers consume it only after the copy is ordered before them. The old source remains protected against premature reuse. Vacated and Fenced are source-side protections, not states of the destination.

![A freed page becoming pending and a moved page leaving protected old bytes before reuse](/images/blog/unified-memory/fig5-page-states.svg)

*Figure 5. A logical free counts toward capacity immediately; physical reuse waits for unfinished work. Moving a page creates a live destination and protects the old source bytes.*

The memory system orders movement and reuse against unfinished work, using logical progress to keep distributed ranks consistent. Focused tests and a seeded stress harness exercise these rules with modeled and real CUDA work.

## One memory system, three interfaces

Everything above sits behind three interfaces: how a model declares its state, how any component asks whether memory fits, and how any component that touches a page is recorded. A new model or feature meets the memory system through them (Figure 6).

![One arena at the centre: a storage plan builds it, one allocator hands out and moves its pages, and a ledger records who still touches each page](/images/blog/unified-memory/fig6-architecture.svg)

*Figure 6. The arena is the shared object. (1) A storage plan built from the models' declarations creates its bands. (2) Every capacity question goes to one allocator, which hands out and moves the arena's pages. (3) A ledger over the same bytes records who still touches each page, so the next writer of a page waits only for what its kind needs.*

**1. Models declare state; one plan builds the arena.** A model does not ask for a pool. Each layer declares its parts, and each part says what it stores (a shape and dtype, such as K and V rows or recurrent state) and how long it lives (the whole context, a bounded window, or a request's recurrent state). One storage plan turns the declarations of every layer, and of a fused draft, into sub-pools, sizes and bands. The plan also budgets storage that stays outside the movable arena, such as per-request rows and captured workspace. Supporting a new combination of state means writing a new declaration, not a new allocator.

**2. Every capacity question is asked the same way.** Prefill admission, decode headroom, HiCache load-back, PD preallocation and prefix-cache eviction all ask the allocator whether a whole demand fits: full-attention tokens, tokens for each window, and state slots, together. If the current layout cannot supply it, the allocator searches for one that can (Figure 4). A free-byte count for a single sub-pool exists only for logs and metrics, and a check mode flags any decision that reads it. Eviction uses the same test. The cache chooses victims in its own order, and the allocator says when the demand fits. Nothing is evicted if the demand already fits, or cannot fit even with every cached entry gone, and the walk stops at the first victim after which it fits, counting what that victim releases in the other sub-pools.

**3. Every page user is recorded.** Each band keeps a page-use ledger. Forwards, HiCache copies, network transfers and page moves each record in it which pages they read or write, and until when. When a page is next needed, the kind of its next writer decides what it waits for. A forward needs nothing from earlier forwards, which run in order on one stream. A HiCache load or a copy on the schedule stream waits on events, and a network write, which no local stream orders, waits for the host to see earlier network transfers finish. A page's id also reaches a network peer only after any forward that may still read its old bytes has retired, because publishing the id lets the sender start writing; the decode node delays publishing until then. A new feature joins by saying which pages it touches and when it is done, not by adding its own synchronization.

## What it works with today

Turn it on with one flag; `--mem-fraction-static` and `--max-total-tokens` size the arena as a whole, and `--max-running-requests` still bounds concurrency:

```bash
# NVIDIA CUDA; choose --tp for your GPU count and model.
python3 -m sglang.launch_server \
  --model-path MODEL_PATH \
  --tp 2 \
  --enable-unified-memory
```

`MODEL_PATH` is a Hugging Face repo id or local path of a supported hybrid model. The [usage guide](https://docs.sglang.io/docs/advanced_features/unified_memory) has the current model and backend matrix, and the [design guide](https://docs.sglang.io/docs/advanced_features/unified_memory_design) covers implementation and debugging tools.

Unified memory changes storage and admission, not the rest of the serving stack:

| Feature | How it works with unified memory |
|---|---|
| Overlap scheduler | Supported, and the default; freed pages stay pending until the forwards that may still touch them are processed |
| Radix cache | One prefix-tree slot per sub-pool; cache eviction feeds the joint admission decision |
| HiCache | Host pools mirror each sub-pool, and host copies take part in page ordering |
| PD disaggregation | Mooncake transfers carry whole sub-pool entries, and each in-flight transfer is recorded against the pages it touches |
| Speculative decoding | `--speculative-algorithm` EAGLE, EAGLE3, DFLASH and DSPARK with a linear draft chain; compatible fused drafts use target-entry lanes |
| Pipeline parallelism (PP) | Each stage owns an arena and simulates every stage's capacity from shared logical counts |
| Decode context parallelism (DCP) | MLA models without sliding-window layers, with the backend restrictions in the usage guide |
| Data-parallel attention | Each attention-DP rank has its own arena |

Restrictions: PD requires Mooncake and cannot be combined with PP, speculative decoding or DCP, and HiCache cannot be combined with EAGLE or EAGLE3. The [usage guide](https://docs.sglang.io/docs/advanced_features/unified_memory) lists the full matrix.

## Results

Unified memory is compared with SGLang's separate pools at their default ratios, on one H200 per server with SGLang at `daa171846f`, on Qwen3.5-4B and GPT-OSS-20B, as averages of two runs.

- **No regression at defaults.** On Qwen3.5-4B with 1,024 input and 256 output tokens at concurrency 64, unified memory sustained 5,830 output tokens/s against 5,880 with separate pools, and both scored 0.875 on GSM8K (200 questions).
- **A wrong ratio is expensive.** On GPT-OSS-20B (512 input and 1,536 output tokens, 1,024 concurrent requests), setting `--swa-full-tokens-ratio` to 0.1 lost 12% of throughput against the default (7,910 against 9,030 tokens/s). On Qwen3.5-4B, `--mamba-full-memory-ratio 0.3` capped running requests at 24, where the default allows 51, and lost 14% of throughput.
- **Without a ratio, unified memory fits more requests.** On the same GPT-OSS-20B workload, unified memory ran about 1,000 requests at once where separate pools at the default ratio ran 618, with 14% higher throughput and a time to first token of 6 s instead of 43 s. On Qwen3.5-4B with 16K input and 4,096 output tokens at concurrency 64, it ran 32 requests against 22 and was 26% faster.
- **Fewer retractions, without throttling.** On Qwen3.5-4B with 512 input and 16,384 output tokens at concurrency 51, unified memory retracted about 233 requests per 1,000 completed, against 294 for the default pools. The `--mamba-full-memory-ratio 0.3` pools retracted none, only because they capped running requests at 24, and were 20% slower (3,620 against 4,530 tokens/s).

## Limitations and what comes next

**Unified memory is still an experimental feature.** A longer-term goal is to make it the default memory system, after the remaining model, backend and lifecycle gaps are closed.

It currently targets hybrid models. Full-attention-only models, MLA with sliding-window layers and DeepSeek-V4 are not supported, and other model and backend restrictions, including MPS and MLX, are listed in the usage guide.

- **Priority preemption:** currently disabled under unified memory. Higher-priority requests are admitted first but wait rather than retracting running requests.
- **Window rows with recurrent checkpoints:** admission assumes checkpoints advance. If no tracking or replacement state slot is free, a request can hold more window rows than it was priced for.
- **Startup refusals:** besides the combinations listed above, two-batch overlap, LMCache, int8 Mamba checkpoints and FP4 KV are refused at startup; the usage guide has the full list.
- **Request slots:** `--max-running-requests` still reserves a fixed number of request slots. We want that to follow demand as well, so related request-row state can join the same capacity model.
