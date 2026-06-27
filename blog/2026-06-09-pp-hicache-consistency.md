---
title: "Host Tree Consistency for HiCache under Pipeline Parallelism: Problem and Fix"
author: "Yanbo Yang, Zhangheng Huang, Shangming Cai, Chao Shi, Tingwei Huang, Zhiqiang Xie"
date: "June 9, 2026"
previewImg: /images/blog/pp_hicache_consistency/preview.png
---

## 1. Prologue: A Combination That "Looks Like It Shouldn't Break"

In agentic and long-context inference scenarios, requests often **share very long prefixes**—system prompts, tool definitions, multi-turn conversation history—easily reaching tens of thousands of tokens. Recomputing this shared prefix for every request is absurdly expensive.

> **Series context.** This article is a follow-up to the SGLang Pipeline Parallelism release blog, [*Pipeline Parallelism in SGLang*](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/), which introduced SGLang's highly optimized **Pipeline Parallelism (PP)** implementation—**Chunked Pipeline Parallelism (CPP)**, **Asynchronous P2P Communication**, and a simple yet effective **Dynamic Chunking** mechanism—compatible with other parallel strategies, **PD Disaggregation**, and **HiCache**. The key message we want to lead with: **for agentic serving, PP is critical.** Agentic workloads run very large models across many GPUs; PP is what lets those models both *fit* and *scale* with high throughput, which makes it a default building block rather than an optional optimization.
>
> That release covers the **initial / preliminary** stage of PP support in SGLang: it lands the core architecture and a "production-ready" path that is, in principle, compatible with PD Disaggregation and HiCache. But once PP is layered on top of **L3 persistent storage** under real production load, consistency corner cases emerge that the first implementation did not fully cover. This article zooms into one of them—**host radix tree divergence across PP ranks when HiCache L3 is enabled**—dissecting the root cause level by level and detailing the fix, so that the PP + HiCache combination is hardened from "works in principle" toward "robust in production."
>
> Companion material: interactive animation `hicache_pp_animation.html`, minimal repro script `dual_prefetch_groups_demo.py`; design and PR plan in upstream issue [sgl-project/sglang#22607](https://github.com/sgl-project/sglang/issues/22607).

SGLang solves this with **HiCache**: a three-level KV cache hierarchy.

<img src="/images/blog/pp_hicache_consistency/hicache_hierarchy.svg" alt="HiCache three-level KV cache hierarchy (L1 GPU / L2 host radix tree / L3 persistent store)" style="display:block;margin:0 auto;width:100%;max-width:820px;" />

L3 persistence lets prefixes be reused not only across requests, but also after a process restart.

Large models with dozens to hundreds of layers (e.g. DeepSeek-V3.2, GLM-5.1, DeepSeek-V4 Pro) require **Pipeline Parallelism (PP)** to split layers across multiple GPU groups, and are often combined with **disaggregated prefill**.

**Why PP + L3 is a must-have configuration for today's agentic serving.** Agentic request shapes are highly distinctive: the shared prefix formed by system prompt + tool definitions + multi-turn history easily reaches tens of thousands of tokens, and is reused repeatedly across huge volumes of requests—and even after process restarts. To sustain this load in production, two things are indispensable:

- **PP determines "fits and scales"**: as models grow ever larger (dozens of layers, hundreds of billions of parameters), only PP—splitting layers across multiple GPU groups—can both fit the weights and sustain high throughput via pipelining;
- **L3 determines the "hit-rate ceiling"**: the shared prefix must be **persistently** cached. L3 (external distributed storage such as Mooncake) lifts prefix reuse beyond the limits of single-node host memory and a single process lifetime, raising cache hits from "within a session" to "global + across restarts", which directly drives **TTFT and per-token cost**.

Therefore **PP + L3 is not an optional optimization but the default foundation for scaled agentic serving**. And it is precisely this most production-valuable combination that triggers the host tree consistency defect this article dissects and fixes—in other words, the closer you get to high-value production scenarios, the harder this bug is to avoid.

But layering PP on top of L3 storage introduces a consistency defect absent in simpler configurations, manifesting as a **shape mismatch crash** rather than a numerical deviation. This article analyzes the cause level by level and explains the fix.

## 2. The Consistency Invariant

Under PP, each rank runs an independent scheduler, each maintaining its own radix tree. The core constraint of the system is:

> The host radix tree on all ranks must remain **structurally identical**. If any rank's tree gains or loses even one node, the difference is amplified by subsequent operations and ultimately causes the cross-rank collective communication that depends on tree state to crash on a shape mismatch.

## 3. Level-by-Level Analysis of the Divergence Cause

### 3.1 L1-only + PP (no HiCache): no divergence

Each rank receives the same requests in a consistent order via P2P. `match_prefix` operates on the device radix tree; its inserts and evicts are driven entirely by the same batch selection and complete synchronously within each scheduler cycle. Determinism holds, and there is no source of divergence.

### 3.2 L1+L2 + PP (HiCache without storage): already crashes

Adding the host cache introduces `write-through` (GPU→CPU backup) and `load-back` (CPU→GPU restore). Although `writing_check()` / `loading_check()` are called at deterministic points in the event loop, the underlying backup / load is **asynchronous IO**: completion events land in each rank's own queues (`ack_write_queue`, `ack_load_queue`), while prefetch completion is picked up by the main thread polling `check_prefetch_progress`. Within the same cycle, the scheduler threads on PP0 and PP1 may consume **different numbers** of completion events, thus applying a different number of updates to the host tree, causing `matched_host` to diverge and crash.

Minimal repro from the fix PR [#27285](https://github.com/sgl-project/sglang/pull/27285):

```bash
sglang serve --model-path=Qwen/Qwen3-32B --pp-size=2 \
  --enable-hierarchical-cache --max-total-tokens=$((256*1024))
python -m sglang.bench_serving --num-prompts 1000
# RuntimeError: shape '[3013, -1, 128]' is invalid for input of size 8192000
```

The L2-level fix is **`pp_sync`**: PP0's scheduler thread decides how many completion events each queue should consume this cycle, and PP1 consumes exactly the same number, eliminating divergence caused by the two scheduler threads finishing async work at different times. This mechanism is a **directional synchronization between scheduler threads**, a different category from the symmetric MIN used by L3 (see Sections 5–6). L2 and L3 each have an independent consistency defect requiring its own fix; this article focuses on L3, but the L2 problem is real and must not be skipped.

### 3.3 L1+L2+L3 + PP (HiCache with storage): consistency breaks here

L3 introduces a **prefetch thread**—an **asynchronous** background thread that independently queries external storage on each rank. Divergence is caused by four mutually reinforcing factors:

1. **Async completion timing**: each rank's prefetch thread finishes at a different wall-clock moment. The one that finishes first immediately inserts a node into its host tree, while the laggard has not updated yet. The next `match_prefix` sees a different host tree state on different ranks, yielding different `host_hit_length` and `prefix_indices`.
2. **Anchor divergence**: an L3 query uses a hash chain starting from some anchor node in the host tree. If one rank already inserted a node from the previous prefetch (`host_hit=896`), its anchor and token range differ from a rank that has not updated yet (`host_hit=0`); the two compute different hash chains for the same request and fetch different—or even wrong—data from storage.
3. **Wall-clock LRU eviction divergence**: LRU uses `last_access_time = time.monotonic()`, which differs across ranks at the microsecond level, leading to different victim node choices, different GPU→CPU demotions, different host memory pressure, and hence a different `evictable_host_leaves` candidate set.
4. **Amplifying cascade**: once the host tree diverges, subsequent eviction decisions, write-through timing, the next request's prefetch anchor, etc. all further amplify the difference, until a shape mismatch crash.

The essence: L1/L2 operations are synchronous and deterministic within a cycle, whereas L3 prefetch is asynchronous and state-dependent. Async completion timing, state-dependent query parameters, and wall-clock-dependent eviction order together form a positive feedback loop that amplifies a tiny divergence into a crash.

**The Lifecycle of an L3 Request and the Two Divergence Quantities**

To set up the fix below, we first use the **main branch (before the L3 fix)** to explain how the prefetch path mutates the host tree, and which quantities diverge across ranks.

The whole path spans two layers: the **background IO layer** (`HiCacheController`, `cache_controller.py`) and the **sole tree writer**, the scheduler main thread (`HiRadixCache`, `hiradix_cache.py`). On main, the two layers are coupled in two ways:

- **Request dispatch and revocation go through queues**: `prefetch_queue` (main thread dispatches `PrefetchOperation`), `prefetch_buffer` (handed to actual IO after a hit), `prefetch_revoke_queue` (revoke when the hit is insufficient).
- **Completion results go through a shared object + main-thread polling**: after the background IO thread loads pages into host memory, it **updates `completed_tokens` in place on the same `operation`**; the main thread, in its event loop, calls `check_prefetch_progress(req_id)` to poll each in-flight request's state, and only takes the result and writes the tree once the termination condition is met. **main has no `PrefetchAck`, no `prefetch_sync_queue`, and no background sync thread.**

<img src="/images/blog/pp_hicache_consistency/l3_prefetch_problem.png" alt="main-branch prefetch flow: shared operation object + main-thread check_prefetch_progress polling" style="display:block;margin:0 auto;width:100%;max-width:820px;" />

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/lifecycle.gif" alt="Two-Request Lifecycle (L3 miss/hit, host-tree consistency)" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — Two-Request Lifecycle (L3 miss/hit, host-tree consistency).** If the embed above doesn't render (e.g. on plain GitHub), open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_lifecycle.html) in a browser.

Note two key facts: first, **both divergence quantities undergo one MIN reduction on main**—`storage_hit_count` in the background thread, `completed_tokens` in the main thread—but **both reductions cover only the TP/CP group and exclude PP**; second, **the tree write is triggered by the main thread polling each request**, not driven by background completion events. These two points are exactly where Section 4 pinpoints main's bugs.

Accordingly, the three paths can be stated as:

**miss path**: `match_prefix` misses in L2, `_storage_hit_query` misses in L3, falls back to GPU forward compute; the result is written into the L2 host tree via `insert`, then persisted to L3 by the backup thread via `write_backup` / `page_set`.

**hit path**: `prefetch_thread` obtains the hit page count via `_storage_hit_query` (which internally calls `storage_backend.batch_exists`) and puts it into `prefetch_buffer`; `prefetch_io_aux_thread` pulls pages back to host batch by batch via `_page_transfer` (which internally calls `page_get`) and accumulates `completed_tokens` in place; the main thread, in `check_prefetch_progress`, retrieves the result via `terminate_prefetch`, takes the MIN of `completed_tokens`, and inserts the hit prefix into the host tree via `_insert_helper_host`.

**eviction path**: under host memory pressure, the scheduler main thread deletes nodes from the host tree via `evict_host` (L3 still retains the corresponding pages).

Along the whole path, two quantities naturally diverge across ranks, and both directly determine the host tree's insertion length:

- **`storage_hit_count` (divergence #1)**: comes from the `batch_exists` query result. Each rank's host view and L3 visibility differ, so the return value can differ.
- **`completed_tokens` (divergence #2)**: comes from the actual load result of `page_get`. Even with the same prefetch range, per-page loading may still partially fail to different degrees on different ranks.

The host tree's growth (how many pages of prefix get inserted) is jointly determined by these two quantities. **If either quantity is not unified across ranks, the insertion length diverges and the host tree becomes inconsistent.**

## 4. The Sync Logic on main and Its Two Bugs

The main branch **does a MIN reduction on both divergence quantities**, but neither the reduction scope nor the trigger mechanism is sufficient to cover PP, so it still diverges. The logic is as follows.

**Each divergence quantity has one MIN, but both cover only TP/CP.** `storage_hit_count` is reduced in the background `prefetch_thread_func`, `completed_tokens` is reduced in the main-thread `check_prefetch_progress`; both use the `attn` group (TP/CP) and **exclude `pp_group`**:

```python
# main: prefetch_thread_func (background thread) — unify prefetch range
hash_value, storage_hit_count = self._storage_hit_query(operation)
self._all_reduce_prefetch_groups(storage_hit_count_tensor, ReduceOp.MIN)   # prefetch_sync_groups, TP/CP only
operation.hash_value   = hash_value[: storage_hit_count // self.page_size]
operation.host_indices = operation.host_indices[:storage_hit_count]

# main: check_prefetch_progress (main thread, per-request polling) — unify insertion length
completed_tokens, hash_value = self.cache_controller.terminate_prefetch(operation)
self._all_reduce_attn_groups(completed_tokens_tensor, ReduceOp.MIN)        # attn_cp/attn_tp only = TP/CP
min_completed_tokens = completed_tokens_tensor.item()
matched_length = self._insert_helper_host(..., hash_value[: min_completed_tokens // self.page_size])
```

And `_create_prefetch_sync_groups` creates only one set of `prefetch_sync_groups`, whose members likewise come from the `attn` group, exclude PP, and there is no second set:

```python
# main: _create_prefetch_sync_groups
base_groups = [self.tp_group]            # or attn_cp_group / attn_tp_group; no pp_group, no second set
```

This produces two bugs:

- **Bug 1: neither MIN covers PP.** The reductions happen only within TP/CP groups; between PP stages there is no alignment of `storage_hit_count` or `completed_tokens` whatsoever. The prefetch range and insertion length for the same request can differ across PP stages, so the host tree diverges across stages directly.
- **Bug 2: tree writes are triggered by main-thread per-request polling, with no constraint across PP on "which and how many land this cycle".** main's insert happens in `check_prefetch_progress`, where the main thread polls and terminates each in-flight request independently. Because prefetch is asynchronous, the set and number of requests that each PP rank terminates and writes within the same cycle can differ (there is no `pp_sync` / qsize alignment mechanism); combined with the async completion timing, anchor divergence, and wall-clock LRU eviction from Section 3.3, once the host tree gets out of step it is amplified.

```text
main:  MIN(storage_hit) and MIN(completed_tokens) only within TP/CP, no sync on the PP dimension
       tree writes triggered by main-thread per-request polling; the set/number finalized differs across PP
       └──▶ across PP stages → insert/delete diverge → host tree inconsistent → crash
```

In sum, main's synchronization is "two MINs covering only TP/CP, tree writes via main-thread per-request polling with no alignment across PP", which cannot guarantee host radix tree consistency under PP + L3.

## 5. Overview of the Fix

The goal of the fix: make the **insertion/deletion on the host tree fully identical in both length and count** across all TP + PP ranks, without introducing deadlock and without blocking GPU compute. The solution consists of three classes of communication channels (matching the design in issue #22607), each governing one class of divergence source.

<img src="/images/blog/pp_hicache_consistency/fix_three_channels.svg" alt="The fix: three classes of communication channels (PG1 / PG2 MIN + pp_sync) governing each divergence source" style="display:block;margin:0 auto;width:100%;max-width:820px;" />

Design intent of the three channels:

1. **PG1 and PG2 have identical members yet are built as two sets**: because the reductions of `storage_hit_count` and `completed_tokens` run on two different background threads. A single gloo communicator cannot be `all_reduce`d concurrently by two threads (it would misalign or even deadlock), so each thread owns its own set.
2. **Background uses gloo (CPU), isolated from the scheduler thread's NCCL (GPU)**: the background `all_reduce` goes over CPU communication and does not occupy or block the CUDA collective stream where forward compute runs.
3. **The two semantics differ**: PG1/PG2 are symmetric global MINs—pulling each rank's progress down to the "slowest"; channel 3 is single-point decision by PP0 + directed downstream broadcast—termination timing and drain count are unified by the leader.

Channel 3 is exactly the L2-level `pp_sync` mechanism mentioned in Section 3.2, and it applies equally to consuming L2 and L3 completion events; PG1/PG2 are the core new additions of this L3 fix.

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/consistency.gif" alt="Tree Consistency (MIN all-reduce keeps every PP/TP rank identical)" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — Tree Consistency (auto-play).** How the MIN all-reduce keeps every PP/TP rank's radix tree identical. If the embed doesn't render, open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_consistency.html) in a browser.

### Design evolution: from a store-side MIN to an in-engine MIN thread


It is worth recording how we arrived at this design, because the first version solved the same problem from a different layer. Initially we did **not** run the `storage_hit_count` MIN inside SGLang at all—we pushed it down into the **Mooncake store query layer**. When ranks from different PP stages issued their storage-hit queries, the store recognized them as one group (by a PP/TP group key) and returned the **group-wide MIN** hit length directly, so every rank received an already-unified prefetch range.

That worked, but it coupled a correctness-critical invariant of the inference engine to the external storage backend: the store had to be aware of SGLang's parallel topology and group membership, the reduction semantics lived outside the engine, and any other storage backend would have to re-implement the same logic. It also could not handle the *second* divergence (`completed_tokens`) symmetrically, since that quantity only materializes during the actual page transfer **inside** the engine, not at query time.

So we moved the MIN back into SGLang, onto a dedicated background thread (`prefetch_thread` doing `all_reduce(MIN)` over `prefetch_hits_sync_groups`). The engine now owns both reductions end-to-end (PG1 = `prefetch_hits_sync_groups` for the prefetch range, PG2 = `prefetch_completion_sync_groups` for the landed length), the storage backend stays a topology-agnostic key-value store, and the two divergence sources are unified by one uniform mechanism. The rest of this article describes that final, in-engine design.

## 6. Implementation Walkthrough (current branch)

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/threads.gif" alt="Thread Relationships & Tree Consistency" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — Thread Relationships & Tree Consistency.** The roles of `prefetch_thread` / `prefetch_io_aux_thread` / `prefetch_sync_thread` and how their queues/MINs feed the sole tree writer. If the embed doesn't render, open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_threads.html) in a browser.

### 6.1 Bring pp_group into the sync, and build two independent sets

`_create_sync_groups`, building on main, appends `pp_group`, and is called twice to build two independent sets—`prefetch_hits_sync_groups` (PG1) and `prefetch_completion_sync_groups` (PG2):

```python
# current branch: _create_sync_groups
base_groups = [self.tp_group]            # or attn_cp_group / attn_tp_group
if self.pp_group is not None:            # HACK: bring the PP ring into the sync
    base_groups.append(self.pp_group)
groups = []
for group in base_groups:
    ...
    groups.append(create_custom_parallel_group(..., backend="gloo"))
return groups

# called twice -> two independent sets
self.prefetch_hits_sync_groups = self._create_sync_groups()        # PG1
self.prefetch_completion_sync_groups = self._create_sync_groups()  # PG2
```

The sync scope expands from "TP ring only" to "TP ring + PP ring", covering cross-PP divergence at the root.

### 6.2 First MIN: unify the prefetch range (PG1)

`prefetch_thread_func` does `all_reduce(MIN)` on `storage_hit_count` and truncates the prefetch set accordingly. This step also unifies the length of `hash_value`, i.e. the subsequent batch count and ack count:

```python
# current branch: prefetch_thread_func
hash_value, storage_hit_count = self._storage_hit_query(operation)
self._all_reduce(storage_hit_count_tensor, ReduceOp.MIN, self.prefetch_hits_sync_groups)   # @ PG1
storage_hit_count = storage_hit_count_tensor.item()
operation.hash_value   = hash_value[: storage_hit_count // self.page_size]
operation.host_indices = operation.host_indices[:storage_hit_count]
```

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/skew.gif" alt="Async Skew × MIN Lockstep" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — Async Skew × MIN Lockstep.** Background prefetch threads finish at different wall-clock times; the `all_reduce(MIN)` pulls every rank back into lockstep on the same prefetch range. If the embed doesn't render, open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_skew.html) in a browser.

### 6.3 Exactly one PrefetchAck per batch

`_page_transfer` is changed so that on error it **no longer breaks**, but keeps looping and emits exactly one `PrefetchAck` per batch, guaranteeing each rank produces the **same number** of acks:

```python
# current branch: _page_transfer
for i in range(0, len(operation.hash_value), self.storage_batch_size):
    if ok and operation.is_asked_to_terminate():
        ok = False
    if ok:
        n = self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
        if n != len(batch_hashes):
            ok = False
        completed_tokens += n * self.page_size
    ack = PrefetchAck(rid=..., completed_tokens=completed_tokens, ...)
    self.prefetch_sync_queue.put(ack)    # exactly one ack per batch, even on error
```

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/ackalign.gif" alt="PrefetchAck Alignment & Anti-Hang" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — PrefetchAck Alignment & Anti-Hang.** Why emitting exactly one ack per batch (even on error) keeps the collective call count equal per rank and prevents a permanent hang. If the embed doesn't render, open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_ackalign.html) in a browser.

### 6.4 Second MIN: unify the insertion length (PG2)

`prefetch_sync_thread_func` does `all_reduce(MIN)` on each ack's `completed_tokens` and writes it back into the ack. Each ack corresponds to one reduction:

```python
# current branch: prefetch_sync_thread_func
ack = self.prefetch_sync_queue.get(...)
self._all_reduce(completed_tokens_tensor, ReduceOp.MIN, self.prefetch_completion_sync_groups)   # @ PG2
ack.completed_tokens = completed_tokens_tensor.item()
self.ack_prefetch_queue.put(ack)
```

### 6.5 Main thread writes the tree using the unified values

The scheduler main thread `drain_storage_control_queues` first takes the MIN of each queue's qsize via channel 3 (`_all_reduce` + `_pp_sync` directed broadcast), so each rank consumes the **same number** of acks; then `_handle_prefetch_result` decides the insertion length using the post-MIN `completed_tokens`:

```python
# current branch: _handle_prefetch_result
completed_tokens = operation.completed_tokens         # unified via PG2 MIN, identical per rank
fetched_key      = prefetch_key[:completed_tokens]
written_indices  = host_indices[:completed_tokens]
matched_length   = self._insert_helper_host(
    last_host_node, fetched_key, written_indices,
    hash_value[: completed_tokens // self.page_size],
)
```

## 7. Why Two gloo Groups Won't Deadlock: A Concurrency-Safety Demo

The fix must dodge two independent deadlock sources at once: one is "each rank must make an equal number of collective calls" (guaranteed by Section 6.3's "exactly one ack per batch"); the other is the topic of this section—**two background threads using collectives concurrently**. Both must hold simultaneously; missing either still hangs.

### 7.1 The core risk: one communicator used by two threads concurrently

gloo's `all_reduce` is a stateful rendezvous: a single ProcessGroup (communicator) maintains a set of message sequence numbers and matching state underneath, and is **not guaranteed thread-safe under concurrency**. If two background threads (`prefetch_thread` reducing `storage_hit_count`, `prefetch_sync_thread` reducing `completed_tokens`) launch `all_reduce` **concurrently** on the **same** communicator, the relative order in which the two threads enter the collective on each rank is uncontrolled:

```text
sharing one group group1 (dangerous):
  rank0:  threadA.all_reduce(group1)  arrives first   threadB.all_reduce(group1)  arrives later
  rank1:  threadB.all_reduce(group1)  arrives first   threadA.all_reduce(group1)  arrives later
          └──────────────┬──────────────┘
          rank0's A and rank1's B get mismatched into the same rendezvous
          → reduce values that shouldn't be reduced together / tag mismatch → data corruption or permanent block
```

### 7.2 The fix: each thread owns its own set

`_create_sync_groups`, called twice, creates two independent gloo communicators over the **same set of ranks**—`prefetch_hits_sync_groups` (PG1) and `prefetch_completion_sync_groups` (PG2)—each dedicated to one background thread. The two collective streams travel over their own communicators, never interleave, and the rendezvous always pairs one-to-one within the "same group + same thread" semantics:

```python
# PG1 dedicated to prefetch_thread (storage_hit_count), PG2 to prefetch_sync_thread (completed_tokens)
self.prefetch_hits_sync_groups = self._create_sync_groups()        # PG1
self.prefetch_completion_sync_groups = self._create_sync_groups()  # PG2
```

<p align="center">
  <img src="/images/blog/pp_hicache_consistency/deadlock.gif" alt="Why 2 Groups Avoid Deadlock" style="display:block;margin:0 auto;width:100%;max-width:960px;border:1px solid #30363d;border-radius:12px;background:#0e1117" />
</p>

> 🎬 **Interactive demo — Why 2 Groups Avoid Deadlock.** Two background threads each own a separate gloo group, so their concurrent `all_reduce`s never interleave into the same rendezvous. If the embed doesn't render, open [interactive version](/images/blog/pp_hicache_consistency/hicache_pp_animation_en_deadlock.html) in a browser.

### 7.3 Minimal runnable example

Below is a self-spawning, CPU-only minimal skeleton compressing the above structure into ~30 lines. `dual` uses two sets (the real design, completes cleanly); `shared` makes the two threads share one set (reproducing the interleave/block):

```python
import os, threading, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world, mode, rounds, port):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    ranks = list(range(world))
    # call new_group twice over the same ranks -> two independent communicators
    g1 = dist.new_group(ranks=ranks, backend="gloo")   # ~ prefetch_hits_sync_groups       (PG1)
    g2 = dist.new_group(ranks=ranks, backend="gloo")   # ~ prefetch_completion_sync_groups (PG2)

    def reduce_loop(group, base, n):
        for _ in range(n):
            t = torch.tensor([base + rank], dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)  # MIN -> base
            time.sleep(0.05)                                       # widen the window, amplify concurrent interleave

    # dual: threadA->g1, threadB->g2 (safe);  shared: both threads share g1 (dangerous)
    gA, gB = (g1, g2) if mode == "dual" else (g1, g1)
    tA = threading.Thread(target=reduce_loop, args=(gA, 0, rounds))    # ~ storage_hit_count
    tB = threading.Thread(target=reduce_loop, args=(gB, 100, rounds))  # ~ completed_tokens
    tA.start(); tB.start(); tA.join(); tB.join()
    print(f"[rank {rank}] done ({mode})", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(4, "dual", 5, 29560), nprocs=4, join=True)
```

Switching `"dual"` to `"shared"` lets you observe the interleave/block when two threads share one group. The repo's `dual_prefetch_groups_demo.py` is its full version, with a watchdog `join(timeout)` to explicitly detect hangs, plus an `uneven` mode (corresponding to the "unequal call count" failure).

### 7.4 How this technique solves the PP + L3 problem

Putting this technique back into context: under PP + L3, the host tree diverges because the two divergence quantities `storage_hit_count` and `completed_tokens` are not unified across **TP + PP** (Section 4). To unify them requires **two cross-rank MINs**; these two MINs naturally occur at two different stages of the prefetch pipeline—the query stage and the load stage—and must run on **background threads** so as not to block the GPU forward on the scheduler main thread. Hence:

- **two divergence quantities → two background MINs → two concurrent background threads**, an inevitable result of aligning the PP/TP dimension;
- **two concurrent threads running collectives at once → two communicators are mandatory**, otherwise per 7.1 they inevitably interleave/deadlock;
- both sets use **gloo (CPU)**, isolated from the main thread's NCCL (GPU), so background alignment does not occupy the forward's CUDA collective stream;
- combined with Section 6.3's "exactly one ack per batch + qsize alignment", PG2's reduction count is equal per rank, so neither set hangs.

In the end: PG1 makes all TP+PP ranks prefetch the same range, PG2 makes them land the same length, each rank inserts a **prefix of equal length** into the host tree, the host tree stays consistent across PP stages, and the shape mismatch crash of Section 4 is eliminated at the root. In other words, **"two groups" is not concurrency for concurrency's sake, but a concurrency-safe design forced out by the correctness requirement that "both divergence quantities must cover PP".**

### 7.5 The demo has no pp_sync: why it still won't deadlock, and which part of the demo pp_sync corresponds to

To be explicit: **the demo above only reproduces PG1/PG2, the two background symmetric all_reduces, and deliberately excludes pp_sync (channel 3).** It still won't deadlock because whether a collective can deadlock depends on only two things, both of which are already in the demo and have nothing to do with pp_sync:

1. **Two independent sets (`g1 != g2`)**: the two concurrent threads each own a communicator and never interleave. This corresponds to `dual` vs `shared`—only `shared`, sharing one group, interleaves/blocks.
2. **Each rank makes an equal number `n` of `all_reduce` calls**: the rendezvous pairs one-to-one. This corresponds to `dual` vs `uneven`—`uneven` makes one rank call `all_reduce` once fewer, and the rest wait forever for a pairing and hang.

In the demo, `n` is a constant passed in directly; in the real system, `n` (= PG2's reduction count = batch count = ack count) is guaranteed equal per rank by **PG1 unifying the `hash_value` length + Section 6.3's "exactly one ack per batch"**. **Key: it's these two invariants that make `n` equal, not pp_sync.**

So which part of the demo does pp_sync correspond to? **The answer: it corresponds to none of the demo's `all_reduce` calls, but to the "downstream stage" the demo deliberately omits.** The demo ends after `reduce_loop`; the real system, after the two background MINs, still has the scheduler main thread's step of "consume acks, write the host tree" (Section 6.5, `drain_storage_control_queues` → `_handle_prefetch_result`). pp_sync (channel 3) acts exactly at this step: PP0 decides how many acks to consume this cycle and when to terminate, then broadcasts unidirectionally along the PP ring, so each rank consumes the **same number** of completion events and keeps the tree-write action sequence consistent.

```text
demo covers:    [two background threads × two sets × n all_reduce each]   <- PG1 / PG2
                          │
demo omits:     ──────────▼──────────▶  [main thread: consume acks + write host tree]   <- pp_sync is here
            PG1/PG2 guarantee "compute a unified length and don't deadlock"   pp_sync guarantees "equal consume count, consistent write sequence"
```

**Separation of duties: shape is governed by the two MINs, count/sequencing by pp_sync—these are two separate things, governed by different mechanisms in this fix; don't conflate them.**

- **PG1 / PG2 → shape (each request's "how long to insert" is identical)**: PG1 does MIN on `storage_hit_count`, unifying the prefetch range → same `hash_value` length, same batch count; PG2 does MIN on `completed_tokens`, unifying the landed length → equal-length inserted prefixes. By this point, the "how long to insert" computed on each PP rank is already equal per rank—**shape consistency is entirely the work of PG1 + PG2, unrelated to pp_sync**.
- **pp_sync (channel 3) → count / sequencing (this cycle's "how many requests to process, in what order, when to stop" is identical)**: in `drain_storage_control_queues`, the main thread takes some completion events from the queues each cycle to write the tree, but the number of acks (qsize) piled up in each rank's queue can differ—this is **not a length divergence, but a divergence in "how many requests each intends to process this cycle"**. pp_sync makes PP0 decide the consume count for this cycle and broadcast it along the PP ring, so each rank consumes the same number of completion events this cycle and keeps the tree-write action sequence aligned.

| mechanism | guarantees | dimension |
| --- | --- | --- |
| PG1 + PG2 | each request's insertion length is identical | shape / length |
| pp_sync (channel 3) | this cycle: how many to process, in what order, when to stop | count / sequencing |

Finally, pp_sync itself won't introduce deadlock: it is not a symmetric global all_reduce, but a **unidirectional P2P relay** on the PP ring (`recv` from upstream → non-blocking `isend` downstream), needing no global rendezvous, so it neither competes with those two all_reduces for a communicator nor stalls on "who arrives first" (see Section 5 channel 3, Section 6.5).

## 8. Summary

The host tree consistency problem of PP + HiCache (L3) is rooted in the fact that the host tree's "growth" is jointly determined by two per-rank-divergent quantities (`storage_hit_count`, `completed_tokens`), while the main branch unified only one of them and did not cover PP. The core of the fix can be summarized in three points:

- **Two independent divergence sources -> two symmetric MINs**: PG1 unifies the prefetch range, PG2 unifies the landed length, both covering TP + PP.
- **Two concurrent background threads -> two independent gloo sets**: avoiding the concurrent-reduction misalignment and deadlock caused by sharing a communicator, and isolating from the NCCL on GPU.
- **Exactly one PrefetchAck per batch + channel 3 qsize alignment**: guaranteeing the collective call count and the completion-event consume count are equal per rank, both preventing hangs and keeping the tree-write action sequence consistent.

Augmented by the radix tree state self-check guardrail, the host radix tree on each PP rank stays byte-for-byte identical in inserts and deletes, severing at the root the divergence feedback loop described in Section 3.3.

## Acknowledgement

We would like to thank the SGLang team and community for the implementation and generous support, especially **Zhangheng Huang**, **Shangming Cai**, **Chao Shi**, **Tingwei Huang**, **Yanbo Yang**, **Zhiqiang Xie**, and **Lianmin Zheng**, and many others. This work builds directly on the SGLang Pipeline Parallelism design and the HiCache three-level KV cache hierarchy, from which it inherits its architecture and to which it contributes the PP host-tree consistency fix.
