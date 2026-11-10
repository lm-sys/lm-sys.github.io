---
title: "Paged Experts: Serving MoE Models Larger Than Your GPU in SGLang"
author: "Jan Schalkamp, Xiaoyu Zhang"
date: "TBD, 2026"
previewImg: /images/blog/sglang-paged-experts/cover.png
type: blog
---

Mixture-of-Experts (MoE) models are now a popular shape of open-weight LLMs, and most of their parameters are experts. A token only uses `top_k` of the `E` experts in each layer, but every expert has to be *somewhere* the GPU can reach quickly, and on a consumer GPU with 8–16 GB of VRAM that is the problem: Qwen3-30B-A3B in 4-bit is 17 GB, its FP8 version 31 GB, Qwen3-Next-80B-A3B in NVFP4 about 48 GB.

**Paged Experts** lets SGLang serve these models on one consumer GPU without moving any computation to the CPU. Each MoE layer keeps **K of its E experts in a slot table on the GPU**, holds all E in pinned host memory, and copies in only the experts a step routes to that are not resident; the model's own fused-MoE kernel then runs on the GPU as usual. It is the [HiCache](https://lmsys.org/blog/2025-09-10-sglang-hicache/) idea applied to expert weights: the GPU holds the working set, host RAM holds the rest. Paged Experts is merged into SGLang main and enabled with one flag.

This post shows what it does on a 16 GB consumer card and how it works, with each design choice next to the measurement that forced it, and then follows the same reasoning one tier further, to experts on an NVMe drive.

**TL;DR**

- <!-- TODO: Add plug: run models in sglang that wouldnt fit before -->
- **Results do not change**: routing is untouched and every routed expert runs on the GPU. Paged layers are bit-identical to unpaged ones for bf16, FP8 and INT8 experts, and within the Marlin kernel's own run-to-run noise for GPTQ, AWQ, MXFP4 and NVFP4.
- **What decides speed is bytes.** Everything in a decode step except the expert copies already runs at the GPU's read floor, so speed comes down to how many expert bytes cross PCIe per token: 6.5 MB on Qwen3-30B GPTQ (5 % of its step), 173 MB on its FP8 version (37 %). Every part of the design serves that one number.
- **Qwen3-30B-A3B GPTQ-Int4 (17 GB) on a 16 GB RTX 5070 Ti: 233 tok/s single-stream decode**, 421 output tok/s with 4 concurrent requests, GSM8K 0.95, with 92 of 128 experts per layer on the GPU. Its FP8 version (31 GB, twice the card's VRAM) decodes at **105 tok/s**. <!-- TODO: re-measure on main. -->
- **Qwen3-Next-80B-A3B NVFP4 (about 48 GB) with its experts in 62 GB of host RAM: 101 tok/s decode**, GSM8K 0.965–0.975, with 101 of 512 experts per layer on the GPU. <!-- TODO: re-measure on main. -->
- **Beyond RAM (upcoming PRs)**: with the NVMe drive as a third tier, **DeepSeek-V4-Flash (167 GB) runs on the same 16 GB GPU and 62 GB of RAM at 14.0 tok/s**, GSM8K 0.975, and an **8 GB laptop GPU serves Qwen3-Next-80B at 16 tok/s**.

<!-- TODO: add an HTML table of contents here once the headings are final, as other long LMSYS posts do. -->

## 1. Motivation: MoE models on consumer GPUs

### Most of a MoE model is experts, and most experts are idle

A gaming PC or a workstation typically pairs 8–16 GB of VRAM with 32–64 GB of RAM. The MoE models people want to run locally are larger than the VRAM but smaller than the RAM: Qwen3-30B-A3B has 128 experts in each of its 48 MoE layers, and in GPTQ-Int4 its experts alone take 13.9 GiB, the whole usable VRAM of a 16 GB card before a single byte of attention weights or KV cache.

Each token, though, is routed to only 8 of those 128 experts per layer: it reads 0.93 GB of the 14.9 GB of experts, and consecutive tokens keep reusing many of the same ones. The full expert pool has to be *available*, but the set a token actually needs is small.

### What people do today

| approach | where experts live | where they are computed | cost |
|---|---|---|---|
| quantize harder (2–3 bit) | VRAM | GPU | model quality |
| `--cpu-offload-gb` (SGLang) | host RAM | GPU | streams *every* offloaded weight on every step, whatever the routing |
| `--n-cpu-moe N` / `--cpu-moe` (llama.cpp, Ollama) | host RAM, for N layers | **CPU**, for those layers | decode tracks CPU cores and RAM bandwidth; prefill of those layers runs on the CPU |
| [SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/) (SGLang) | NVMe SSD, GPU cache | GPU | built for models larger than RAM; GGUF sources; a miss waits for an SSD read |
| **Paged Experts** (this post) | **host RAM**, K per layer in VRAM | **GPU** | copies only the non-resident experts a step routes to |

Offloading weights wholesale cannot work on this hardware. On our card a byte in VRAM reads at 843 GB/s and a byte over PCIe 5.0 at 52.8 GB/s, 16x slower, and an offloaded weight that every token reads crosses the link on every token: offloading just `f* = B_link / B_vram ≈ 6 %` of a dense model's bytes doubles its step. Only weights that a token does not read escape this, and that is exactly what MoE experts are.

Paged Experts uses that sparsity. Like llama.cpp it keeps the experts that do not fit in host RAM, but it never computes on the CPU: it treats VRAM as a cache of experts, pages the missing ones in over PCIe inside the decode step's CUDA graph, and computes every routed expert with the layer's own GPU kernel. And because the slots come out of the same memory as the KV cache, K is a memory knob as much as a speed feature (section 3.5).

**Relation to SSD Expert Pack.** SGLang's [SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/) solves the neighbouring problem: models far larger than RAM (DeepSeek-V4-Flash, Kimi-K3), with the SSD as the backing tier, an `O_DIRECT` read path, and a GPU cache with frequency-and-recency replacement. Paged Experts starts from the other end: models whose experts fit in host RAM, served from SGLang's native checkpoint formats (safetensors GPTQ, AWQ, FP8, NVFP4, MXFP4, ...), with the decision of what to copy made on the GPU inside a captured graph, so a step does not wait for the host. The two meet in the disk tier ([section 4](#4-beyond-ram-the-same-laws-one-tier-further)), which puts the SSD behind a large pinned host tier instead of directly behind the GPU. A same-machine comparison of the two is still to be run.

## 2. Results on a 16 GB consumer GPU

**Machines.** Sections 2, 3 and 5 run on the desktop; section 4 also runs on the laptop.

| | desktop | laptop (Dell XPS 16 9640) |
|---|---|---|
| GPU | RTX 5070 Ti 16 GB (SM120), 300 W | RTX 4070 Laptop GPU 8 GB (SM89), 60 W power limit |
| GPU link | PCIe 5.0 x16 | PCIe 4.0 x8 |
| CPU | AMD Ryzen 7 9800X3D (8 cores) | Intel Core Ultra 9 185H |
| RAM | 2 x 32 GB DDR5-6000 (Kingston FURY, KF560C30-32), two channels (62 GB usable) | 64 GB LPDDR5x-7467 (54 GB visible to WSL2, of which CUDA can pin about half) |
| NVMe (expert cache, section 4) | Kingston FURY Renegade G5 2 TB, PCIe 5.0 x4 | SK hynix PVC10 1 TB, through WSL2's virtual disk |
| OS | Ubuntu 26.04 LTS, Linux 7.0, native | Windows 11 (build 26300), WSL2 2.7.8 (Linux 6.18), Ubuntu 24.04 |
| NVIDIA driver | 595.91.07 (CUDA 13.2) | 621.14 (CUDA 13.5) |
| SGLang | the `lmsysorg/sglang` nightly CUDA 13 image with the code under test | same |

<!-- TODO: name the SGLang commit and image of the final runs in the SGLang row. -->

**Setup.** SGLang with `--enable-paged-experts`, `--paged-experts-num-resident auto`, `--mem-fraction-static 0.88`, prefix caching off. Page caches are dropped before every start and 16 ShareGPT warm-up requests are discarded: a cold expert pool reads 5–11 % low, a full page cache once cut our throughput 2.6x without an error. Decode tok/s is `1000 / median TPOT` over 8 ShareGPT prompts sent one at a time (256 output tokens each); "4 requests" sends 16 ShareGPT prompts from 4 clients at once. Every comparison repeats its first configuration at the end, and a difference smaller than the gap between the two runs of the same configuration (±0.2 % on GPTQ decode) counts as no difference. GSM8K is 200 questions, thinking off.

| model (checkpoint) | experts on the GPU (K of E) | host store | decode tok/s, bs=1 | 4 requests, out tok/s | TTFT p50 (ShareGPT) | GSM8K (200) |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-GPTQ-Int4 (17 GB) | 92 of 128 | 13.9 GiB | **233.3** | 421 | 40 ms | 0.950 |
| Qwen3-30B-A3B-FP8 (31 GB) | 45 of 128 | 27.0 GiB | **105.0** | TBD | 182 ms | 0.94–0.95 |
| Qwen3-Next-80B-A3B-Instruct-NVFP4 (48 GB) | 101 of 512 | 40.7 GiB | **100.5–101.1** | 137 | 192 ms | 0.965–0.975 |
| gpt-oss-20b (MXFP4, 14 GB) | 28 of 32 | TBD | **172.5** | TBD | 37 ms | 0.96–0.97 (unpaged 0.975) |

<!-- TODO: re-measure the whole table on main; FP8 output tok/s at 4 concurrent at auto K and gpt-oss-20b's host store size are not measured yet. -->

All four models run with decode captured in CUDA graphs and every expert computed on the GPU. A long prompt is a different case: on Qwen3-30B-A3B GPTQ a natural, non-repeating prompt of about 2k tokens has a TTFT p50 of 227–247 ms, and an 8k-token one 1.47 s (`--context-length 16384`, K = 86). The prompts have to be natural text: SGLang's `random` dataset tiles a short ShareGPT turn into a long prompt, repeated text routes to fewer experts, and paged TTFT read 261–265 ms instead of 327. <!-- TODO: record the token count of the natural ~2k prompts under the Qwen3 tokenizer. -->

<img src="/images/blog/sglang-paged-experts/decode-speed.svg" alt="Decode tok/s for one request on an RTX 5070 Ti 16 GB: Qwen3-30B-A3B GPTQ-Int4 233.3, gpt-oss-20b 172.5, Qwen3-30B-A3B FP8 105.0, Qwen3-Next-80B-A3B NVFP4 100.8" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 1: Decode speed for one request on a 16 GB GPU, with K picked automatically.</p>

<!-- TODO (Figure 1): regenerate after re-measuring on main; add fully-resident / llama.cpp bars only where a same-day baseline exists. -->

## 3. How it works, and what decided each part

<img src="/images/blog/sglang-paged-experts/architecture.svg" alt="One MoE layer in a decode step: router, decide kernel, gather kernel copying only the misses from pinned host RAM into the K-slot table, and the fused-MoE kernel over the slots, all inside the CUDA graph" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 2: One MoE layer, one decode step. Everything in the blue frame runs on the GPU inside the decode CUDA graph.</p>

### 3.1 The K-slot table and the host store

The paged method wraps the layer's own fused-MoE method. It creates the layer's expert tensors with K rows instead of E, so the GPU holds only the K slots, and allocates a pinned host store with all E experts of each per-expert tensor. Each weight loader is redirected to a one-row view of the host store, so checkpoints load through SGLang's **native** per-expert loader (sharding, gate/up order, scale layout), and the host rows have exactly the layout the GPU slots would have. On the Marlin runner, whose methods repack their experts, the repack runs over the host store K experts at a time, so the store holds every expert already repacked: a page-in is a plain copy. Adding a quantization format is one subclass that names the tensors to page and what happens after loading. Supported today: bf16/fp16, block FP8, compressed-tensors FP8, W8A8 FP8/INT8 and block INT8 on the Triton runner; GPTQ, AWQ, compressed-tensors W4A16/W8A16 and NVFP4, MXFP4 (gpt-oss) and ModelOpt NVFP4 on Marlin.

Every step of every MoE layer then **decides** (the distinct experts the step routes to are matched against the slots; hits keep their slot, each miss takes the least recently used slot the step itself does not need), **copies** the missing experts from the host store into their slots, and **computes**: the routed expert ids are remapped to slots and the layer's fused-MoE kernel runs over them. A step that routes to more distinct experts than K (a prefill chunk, or a large decode batch) runs in **waves** of at most K experts. Each wave's kernel call returns every routed (token, expert) output already weighted by its router weight but *without* the final top-k sum; the waves write into one buffer and the same top-k sum runs once at the end. That is why the result does not depend on how the experts were split into waves, and why paged bf16/FP8 layers are bit-identical to unpaged ones.

For the Marlin formats bit-identity is the wrong test: Marlin's MoE kernel is not reproducible run to run on SM90 and newer, paged or not. There we check bytes instead of text: every slot against the store after every step (0 mismatches in 83,850 checks), plus unpaged noise floors, logprob margins at the first divergence, and GSM8K.

### 3.2 A decode step is a byte problem

What does paging cost? Three equations frame it:

```text
(E1)  bytes/token = miss_rate(K) × top_k × n_moe_layers × expert_bytes
(E2)  t_step = t_floor + t_boundary + t_fetch,serialized
(E3)  one stream pays E1 on every token;
      a batch of B pays the union of its routed experts once per step: union(B) / B per token
```

**E1** is the only cost paging adds, and only the miss rate depends on Paged Experts: Qwen3-30B GPTQ at K = 92 of 128 pages in 2.66 experts (6.5 MB) per token, a 0.21 ms copy in a 4.27 ms step (5 %); its FP8 version at K = 45 pages in 36.7 (173 MB), 3.52 ms of a 9.52 ms step (37 %). <!-- TODO: re-measure on main. --> 
**E2** splits a step into what it must read (the floor), boundaries (launches, syncs, bookkeeping kernels) and copies nothing overlaps; only the last two can be engineered away.
**E3**: a single stream amortizes copies through sparsity, a batch through a shared union, and a result from one rarely transfers to the other. This post is about the single stream.

We measured E2 on merged code by tracing eight captured steps with CUPTI and joining each copy kernel to the experts it paged (r = 1.000; shifted by one step, r ≤ 0.41). Every row below is a measured kernel time, not a difference of two totals:

| part of one bs=1 step (ms) | Qwen3-Next-80B NVFP4, K=101/512 | Qwen3-30B GPTQ-Int4, K=92/128 | Qwen3-30B FP8, K=45/128 |
|---|---|---|---|
| **page-in copies (host → slots)** | **2.36** | **0.21** | **3.52** |
| decide | 0.27 | 0.11 | 0.18 |
| routed expert GEMMs | 1.22 | 1.41 | 2.40 |
| dense GEMVs/GEMMs, attention, norms, other | 6.36 | 2.60 | 3.60 |
| graph span per step (→ tok/s) | ~10.2 (98) | ~4.3 (229) | ~9.7 (103) |

<!-- TODO: recheck the "other" row before publishing (it is derived by subtraction). -->

The rest is at the floor: Qwen3-Next's GDN projections read at 90 % and its `lm_head` at 97 % of the card's measured 843 GB/s, a dense bf16 model decodes at **98.6 % of its byte floor**, and Qwen3-30B GPTQ's routed Marlin GEMMs take 1.41 ms against a read floor of 1.11 ms (8 x 48 experts x 2.43 MB at 843 GB/s): a perfect kernel would save 7 % of its step. What is left is the copy row (5–37 % of a step) and the boundaries. That is also why Qwen3-30B GPTQ decode stops depending on K above K ≈ 88 (230.6 tok/s at K = 86, 233.3 at K = 92): the copies are already a rounding error. The rest of this section takes the two engineerable terms in turn: boundaries (3.3), then copies, which are hit rate (3.4), K (3.5) and the copy path itself (3.6).

### 3.3 Boundaries: decide and copy inside the CUDA graph

A paged MoE layer is a burst of small kernels, and when each launch carries a few microseconds of GPU work, the host's launches become the critical path:

| decode step, bs=1 | GPU µs per launch | CUDA graph capture buys |
|---|---|---|
| Qwen3-30B GPTQ, paged | **[MEASURE ON MAIN]** | **[MEASURE ON MAIN: eager vs captured]** |
| dense Qwen3.8-4B bf16 | 24.9 | 1.04x |

<!-- TODO: measure the paged row with --disable-cuda-graph against the default on main. -->

Below ~5 µs per launch deleting launches pays enormously; above ~10 µs they are hidden. So with decode CUDA graphs on, a step whose routed entries fit in the K slots (`tokens × top_k ≤ K`, i.e. decode up to `K // top_k` requests) is decided and paged entirely on the GPU, and the decode graph captures it:

- a JIT kernel makes the LRU decision on the device and writes the page-in plan, and (since #46457) the runner's slot ids and weights, to GPU buffers;
- a second kernel copies the planned experts of every paged tensor from pinned host memory into their slots through its unified address, reading the count on the device, so each graph replay moves exactly the experts it decided on.

Prefill steps plan their waves on the host and run as eager breaks of SGLang's breakable prefill graphs; both paths share one LRU state. Once the host was out of the loop, the next wins were fewer boundaries: #46457 cut a paged layer's captured graph from 21 nodes to 10 (+10.3 % decode on Qwen3-30B GPTQ), and #46567 cut the host syncs of a host-planned layer to one (−20 % short-prompt TTFT on Qwen3-30B GPTQ).

### 3.4 Copies, part one: the hit rate needs a dynamic cache

Under uniform routing a routed expert would miss with probability 1 − K/E. Measured routing misses far less; we call the ratio the *concentration factor*:

| model, K/E | decode tok/s | miss per routed expert | uniform 1 − K/E | concentration |
|---|---|---|---|---|
| Qwen3-30B GPTQ, 92/128 | 234.2 | 0.69 % | 28.1 % | 41x |
| Qwen3-30B FP8, 45/128 | 105.0 | 9.6 % | 64.8 % | 6.8x |
| Qwen3-Next-80B NVFP4, 101/512 (top-10) | 101.0 | 13.0 % | 80.3 % | 6.2x |

The concentration is largest where the cache is largest relative to E: a small cache holds the popular head, and the tail is broad. **[MEASURE ON MAIN: a K sweep within one model, Figure 3.]**

The tempting conclusion is to pin the popular head and stop paging, and our first measurements said exactly that. On three prose prompts Qwen3-Next touched 169 of 512 experts per layer, and a frozen set of the most popular ones did as well as LRU; an earlier, similarly narrow result of ours even had static pinning beating LRU at every cache size. Then we replayed twelve mixed prompts (prose, code, math, JSON, German, French): they touched 450 experts per layer, and the frozen set lost 28 points of hit rate when fitted on the very trace it was scored on, 53 when fitted on other data. The median expert returns after 22 accesses, the 99th percentile after 4,723, when the topic swings back. The popular set moves with the workload, so the cache has to move with it: that is why decide runs an LRU on every step instead of serving a precomputed hot set.

Which dynamic rule matters little. Replaying Qwen3-Next routing at K = 96 through a simulator of the decide kernel, page-ins against LRU: LRU-2 ±0 %, decayed LFU −4 %, router-score EMA (HybriMoE's MRS) −5 %, Belady (knows the future) −48 %, 32 more slots −25 %. At lower residency the rule counts more: on DeepSeek-V4.1-Flash traces (384 experts, 21–96 slots) a decayed LFU cut page-ins 10–16 %. On a 16 GB card slots beat policy, so the engineering went into K (3.5). <!-- TODO: label as a simulation in a figure caption. -->

**Against CPU expert offload.** The concentration factor also decides the comparison with llama.cpp. With the same VRAM, llama.cpp computes a share `≈ 1 − K/E` of the experts on the CPU, while paging copies a share `m`; paging wins roughly when

```text
(1 − K/E) / m   >   B_cpu / B_link
 concentration      effective CPU-expert bandwidth / host-to-GPU copy bandwidth
```

Uniform routing makes the left side 1; measured routing makes it 5–50, so paging wins wherever the link is near RAM speed (PCIe 5.0 x16 against dual-channel DDR5). When most of the model fits (int4 on 16 GB), llama.cpp only has to put a few layers' experts on the CPU and we expect the two to be close; when a large share must live in host RAM (FP8, 31 GB, on 16 GB), paging should pull ahead:

<!-- TODO (blocking for this table): a same-day run on main with natural ~2k prompts, byte-matched GGUFs (IQ4_XS / Q8_0) and llama.cpp --n-cpu-moe at its best fitting value. -->

| Qwen3-30B-A3B, 16 GB card, decode tok/s bs=1 | Paged Experts (K) | llama.cpp, best `--n-cpu-moe` | SGLang `--cpu-offload-gb` |
|---|---|---|---|
| int4 (GPTQ vs `IQ4_XS`) | **[MEASURE ON MAIN]** | **[MEASURE]** | **[MEASURE]** |
| 8-bit (FP8 vs `Q8_0`) | **[MEASURE ON MAIN]** | **[MEASURE]** | **[MEASURE]** |

<!-- TODO: re-measure the short-prompt TTFT in the same run. -->

Short-prompt TTFT is where CPU offload has the edge: a prefill routes to nearly every expert, so Paged Experts pages in every non-resident expert once per prefill chunk, while llama.cpp computes its CPU layers in place.

### 3.5 Copies, part two: give K every byte the KV cache does not need

If slots beat policy, K should be as large as memory allows, and K and the KV cache come out of the same static budget. The KV cache cannot be paged instead: KV is private to a request and dense over positions, so every step reads all of it. Qwen3-30B at 1,024 requests × 3k tokens × 48 KB per token (FP8) is ~150 GB per step, ~2.9 s over the link, and no schedule reduces those bytes. Weight paging works because the FFN is sparse; KV paging works only where attention is sparse or bounded (Qwen3-Next's linear-attention layers). So the KV cache comes first and the experts get the rest.

Since #47476, `--paged-experts-num-resident auto` is the default: the layers are built with K = `top_k`, the KV cache is allocated for what the user asked for (one request of the model's context by default; `--context-length`, `--max-running-requests` and `--max-total-tokens` change it), then the small tables are freed, the remaining budget is measured, and every layer grows by whole rounds of one slot while a round fits. On hybrid Mamba/GDN models the Mamba state pool holds the requests the KV cache aims for. The log says what was decided and which flag would pin it:

```text
Paged experts: K=63 of 128 experts per MoE layer resident (6.85 GiB); KV cache 40960 tokens,
1 request(s) of 40960 at once (aim 40960: the model's context 40960 x 1 request(s)). To pin:
--paged-experts-num-resident 63
```

What the trade costs, on Qwen3-30B-A3B GPTQ-Int4:

| KV cache kept for | K (of 128) | decode tok/s | TTFT p50 |
|---|---|---|---|
| the model's own context, 40,960 tokens × 1 request (nothing set) | 63 | 208.4 / 209.0 | 87 ms |
| `--context-length 16384` × 1 request | 86 | 230.6 | 48 ms |
| `--context-length 4096 --max-running-requests 4` (16,384 tokens) | 87 | 230.8 / 231.4 | 46 ms |
| `--context-length 16384 --max-running-requests 4` (65,536 tokens) | 46 | 173.1 | 110 ms |
| `--max-total-tokens 10240` | 92 | 233.3 | 40 ms |

Four full requests of 16k tokens cost 41 resident experts per layer and 25 % of decode speed; one request of the model's full 40k context costs 29 experts and 11 % against the 10,240-token cache. gpt-oss-20b shows the knob from the other side: with an explicit `K = 16`, the KV cache takes everything left (189,422 tokens for 4 requests of 4,096); `auto` gives that memory to the experts instead, K = 28 of 32, and decode goes from 123.9 to 172.5 tok/s (+39 %).

### 3.6 Copies, part three: the copy path, and when prefetch pays

Inside the decode step the expert copies run at 50–53 GB/s, the speed of the PCIe 5.0 link (the copy engine reaches 56.9 GB/s). Getting that number right took a lesson of its own: a "gather defect" we chased for a while was a profiler-inflated kernel time divided by another tool's byte count, and the kernel was running at link speed all along. A ratio of two instruments is not a measurement. The limit beyond the link is host DRAM: both directions together top out at ~62 GB/s, so writing an evicted expert back would cost as much as reading one. Paged Experts never writes back; the host store keeps every expert.

With the link saturated by demand copies, the remaining lever is to hide them: copy an expert before the step asks for it. Prediction is not the problem: a layer's router applied to the previous layer's input finds 84 % of Qwen3-Next's top-10 and 88 % of Qwen3-30B's top-8, and 99.4 % above a small router margin. Yet our first prefetch lost decode speed at every look-ahead depth on Qwen3-30B FP8. Speculative fetch needs idle transport *and* misses worth hiding, and the two anti-correlate: where copies are cheap (GPTQ, 0.21 ms per step) nothing needs hiding, and where they are expensive a prefetch competes with demand. Worse, a copy kernel on its default 256 thread blocks overlapped a concurrent MoE layer only 32 % and slowed it 1.7–2x, and each wrong guess evicted an expert the next step needed.

The way out was a resource the step leaves idle. On 32 blocks the copy keeps 98 % of its bandwidth and overlaps compute 0.88–0.95, because a bs=1 GEMV leaves most SMs idle. With victims never among the predicted experts, a parallel plan kernel, at most three copies per layer and a join before the next layer, a narrow pre-copy gives **+11.6 % on Qwen3-Next** (101.75 → 113.6 tok/s, demand page-ins 66.7 → 26.7 per step, GSM8K 0.965) and +3.6 % on GPTQ (an upcoming PR). <!-- TODO: re-measure on main. -->

Prediction works across layers but not across tokens. Within a token, consecutive layers route predictably, and the pre-copy spends that. Across tokens, consecutive tokens share only part of their experts, and the experts a token newly routes to were rarely among the previous token's runners-up **[MEASURE ON MAIN: cross-token overlap and runner-up share from a routing trace]**: widening a guess from the previous token to catch a third of the new experts costs as many slots as raising K. <!-- TODO: re-measure on main. -->

## 4. Beyond RAM: the same laws, one tier further

> This section describes upcoming PRs (the expert cache, #48642, and the disk tier), not code in SGLang main.

Paged Experts as merged needs every expert in pinned host RAM. DeepSeek-V4-Flash is a 167 GB checkpoint whose routed experts take 137.1 GiB in the store's layout; on our 62 GB machine about 45 GiB of that can be pinned next to the rest of the server. The disk tier adds the NVMe drive behind the host store: each layer keeps all its experts in a per-layer file (the expert cache of section 5), R of them in pinned host rows and K in GPU slots. An expert that is only on disk is read into a host row with one `O_DIRECT` read and copied into its slot.

| model, machine | experts per layer: GPU / host rows / disk | decode tok/s | TTFT, natural ~1.5k-token prompt | GSM8K |
|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 (167 GB), 16 GB GPU, 62 GB RAM | 9 / 71 / all 256 | **13.98 / 13.97** | 8.49 / 8.46 s | **0.975** (200) |
| Qwen3.8-Flash-Next-FP8 (112.7 GiB of experts), same machine | 11 / 163 / all 512 | **15.69 / 15.72** | 15.0 / 15.6 s | **0.97** (200) |
| Qwen3-Next-80B-A3B-Instruct-NVFP4 (40.7 GiB of experts), 8 GB RTX 4070 Laptop GPU, WSL2, 54 GB RAM | 19 / 227 / all 512 | **15.73 / 16.03** | 4.99 / 4.88 s | **1.00** (50) |

The design of the tier came from three findings, each one of section 3's measurements meeting a new tier:

- **Never write back (3.6), now between disk and RAM.** The first version kept each expert in exactly one tier, so every page-in was a swap: the evicted expert went back to host RAM. Both copy directions share host DRAM, and swaps cost 57.7 of a 90.6 ms GPU step. Letting host rows keep their copy (the file always has every expert) made each move one-way: **+24.9 % decode** (11.05 → 13.75 tok/s, GSM8K 0.98 / 1.00), at 18 % more disk reads.
- **Keep the host out of the loop (3.3), even for a disk read.** A layer's plan kernel publishes which experts are only on disk; a host thread reads them while the layer's host-row copies run, and the captured graph waits at the layer until the experts are in their slots, instead of falling back to eager or replaying the step. On the PCIe 5.0 drive waiting beat replaying by 33 %. <!-- TODO: re-measure wait vs replay on the current branch if quoted. -->
- **E1 with the drive in place of the link (3.2).** A prompt of a few hundred tokens already routes to nearly every expert of every layer, so one prefill reads almost the whole tier (~93 GB for one of these prompts on DeepSeek-V4), and TTFT is that read at disk speed (~11 GB/s here). Chunked prefill multiplies it: every chunk re-reads the tier, so the recipe uses one chunk per prompt.

Our first disk tier taught the opposite of the first two: it relied on the page cache, and its read-ahead hints, under memory reclaim, evicted the very experts the step needed and cost 57 %. From that it looked as if only fewer bytes could move this tier. The current tier reads with direct I/O into pinned rows it manages itself, and its gains so far (inclusive rows, the in-graph wait, the look-ahead below) are not byte reductions. <!-- TODO: re-verify the 57 % before publishing (old QLC drive, page-cache read-ahead). -->

The expert files also make later starts cheap: DeepSeek-V4's first start writes 137.1 GiB of them (`load_model` 282.5 s), later starts read them back (82–87 s).

**Next on this path** (measured on development branches, not yet in PRs): reading the next layer's predicted disk misses ahead (+7.2 %), streaming each read into its slot chunk by chunk as it lands (+3.9 %), and the narrow pre-copy of section 3.6; together they reached 16.4–16.7 tok/s on DeepSeek-V4 on the same machine. <!-- TODO: re-measure on the current look-ahead branches before quoting. -->

## 5. Startup and memory

| model | `load_model` before #48082 | SGLang main (#48082: no staging copy) | + expert cache (#48642, open), later starts | swap during start (before → main) |
|---|---|---|---|---|
| Qwen3-30B-A3B-GPTQ-Int4 | 33.5 / 33.0 s | 34.1 s | 21.7 s | 0.2 → 0.2 GB |
| Qwen3-30B-A3B-FP8 | 52.1 / 51.3 s | 53.1 s | 29.3 s | 0.2 → 0.2 GB |
| Qwen3-Next-80B-A3B-Instruct-NVFP4 | 352.1 / 176.6 s | 128.3 / 125.5 s | 72.8 s | 25.6 / 19.1 → 0.2 GB |

<!-- TODO: re-measure on main. -->

Two things dominated a Paged Experts start. Every expert was first loaded into a pageable staging copy and then copied into the pinned store, so a start briefly needed host memory for every expert twice; on a 62 GB machine Qwen3-Next's 40.7 GiB of experts did not fit twice, and the start swapped (and the server kept 3.4–3.5 GB in swap, which cost TTFT). Since #48082 the loads are replayed straight from the memory-mapped checkpoint into the store, K experts at a time, with the next chunk's pages read ahead and the previous chunk's given back. The second cost is that each start repacks every expert for the Marlin kernel although the result is the same every time. The expert cache (#48642, `--paged-experts-enable-disk`) writes each layer's store to a 4 KiB-aligned file once, in the background, and later starts read it back with `O_DIRECT`, after re-loading a few groups of experts and comparing them byte for byte against the checkpoint. These files are what the disk tier of section 4 serves single experts from.

**Memory.** GPU memory is whatever `--mem-fraction-static` gives SGLang: the KV cache first, the slots after (3.5). Host memory is the pinned store: one copy of every expert in the kernel's layout, sized exactly since #46456 (torch's pinned allocator rounded each tensor up to a power of two, pinning about a third more than the experts need on Qwen3-30B-A3B). <!-- TODO: measure peak VRAM per model on main. -->

## 6. Where it stands

Putting section 3 together, one 16 GB card has four regimes:

| regime | bound by | what pays | what did not pay |
|---|---|---|---|
| one stream, high K (GPTQ, K ≈ 90/128) | dense floor + boundaries | capture; decide on the GPU; fewer graph nodes and syncs | eviction variants; a persistent kernel (0.995x); replacing Marlin |
| one stream, low K (FP8 K = 45; Qwen3-Next 101/512) | serialized copies (23–37 % of a step) | more K (`auto`); LRU over static sets; narrow pre-copy (+11.6 %) | 256-block prefetch; guessing from the previous token |
| batches beyond `K // top_k` | host-planned waves, then the link | union amortization (E3; research fork) | |
| experts beyond RAM (section 4) | the drive, then the link | inclusive host rows; waiting inside the graph; one prefill chunk per prompt | page-cache read-ahead |

**Open problems:**

- **Batches beyond `K // top_k`** run as host-planned waves and fall off sharply **[MEASURE ON MAIN: FP8 at a small K with 4 requests]**. On an earlier research fork, skipping each wave's masked rows instead of computing them, plus a parallel decide, made large batches approach the link's limit: there every layer's union is nearly all experts, paid once per step (E3). Not in main. <!-- TODO: re-measure on main. -->
- **Short-prompt TTFT**: a prefill pages every non-resident expert once per chunk; on the disk tier it reads the whole tier.
- **Slower links**: on PCIe 4.0 x8 or a laptop's x8 link the inequality of 3.4 moves toward a tie with CPU offload at bs=1; we have a laptop serving, but no head-to-head there yet.
- **The dense part**: on Qwen3-Next, 4 of 10 ms per step are bf16 GEMVs at the read floor; only fewer bytes move them.
- **Speculative decoding** is refused with Paged Experts today; a drafter's memory comes out of K, so its break-even acceptance rises as K falls.
- **Tensor and expert parallelism**, and lending idle KV pages to the expert slots at runtime (a prototype ran K = 40 + 52 lent slots at about the speed of K = 92 with 32k context). <!-- TODO: cut, or measure on main. -->

**Next PRs:** the expert cache (#48642: later starts skip loading and repacking, −37 to −45 % `load_model`), the disk tier, and reading ahead and pre-copying predicted experts. The per-layer expert files are a candidate for the shared on-disk format proposed in the Intel XPU RFC [#46520](https://github.com/sgl-project/sglang/issues/46520).

## 7. Try it

Paged Experts is in SGLang main. The smallest launch:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts
```

The log names the K it chose. To trade context for resident experts, set the KV cache you need; these are the configurations measured in section 2 (RTX 5070 Ti 16 GB):

<!-- TODO: check whether the recipes need --attention-backend triton and --disable-radix-cache before publishing. -->

```bash
# Qwen3-30B-A3B GPTQ-Int4 (K = 92 of 128 here): 233 tok/s
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts --paged-experts-num-resident auto \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton

# Qwen3-30B-A3B FP8 (31 GB; K = 45 of 128): 105 tok/s
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --enable-paged-experts \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton

# Qwen3-Next-80B-A3B NVFP4 (48 GB, experts in host RAM; K = 101 of 512): 101 tok/s
python -m sglang.launch_server --model-path nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --enable-paged-experts --moe-runner-backend marlin --disable-shared-experts-fusion \
  --kv-cache-dtype bf16 \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton

# gpt-oss-20b (MXFP4; K = 28 of 32): 173 tok/s
python -m sglang.launch_server --model-path openai/gpt-oss-20b \
  --enable-paged-experts --moe-runner-backend marlin \
  --context-length 4096 --max-running-requests 4 --mem-fraction-static 0.88
```

Requirements and limits: one GPU (no tensor/expert parallelism yet); no speculative decoding; host RAM to pin every expert (under WSL2 pinned memory is capped at about half of system RAM); decode graphs capture up to `K // top_k` concurrent requests. The full list is in the [Paged Experts docs](https://docs.sglang.io/advanced_features/paged_experts.html). <!-- TODO: confirm the docs URL. -->

<!-- TODO: add the disk-tier recipe once its PRs merge:
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 45 \
  --paged-experts-num-resident 9 --moe-runner-backend marlin --mem-fraction-static 0.92 \
  --max-running-requests 1 --context-length 4096 --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.5 --cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled \
  --disable-radix-cache --json-model-override-args '{"max_position_embeddings":4096}'
with SGLANG_OPT_FP8_WO_A_GEMM=1. -->


## 8. How we measured

Each of these rules is in the text where breaking it once gave us a wrong number; here they are in one place:

- **Repeat the first configuration last.** A difference smaller than the gap between its two runs is no difference (section 2).
- **Drop the page cache and discard warm-up** (section 2).
- **Natural prompts**, never tiled ones (section 2).
- **Bytes, not text, for Marlin**, which is not reproducible run to run (section 3.1).
- **Measured splits, not subtracted terms** (section 3.2); synthetic buffers must be incompressible: a zero-filled one "read" faster than the card's memory. <!-- TODO: re-verify. -->
- **A ratio of two instruments is not a measurement** (section 3.6).
- **Every check has a negative control that must fail**, and no flag is trusted without proof it ran: several "no effect" results of ours were features wired to nothing.

## Acknowledgments

TODO
