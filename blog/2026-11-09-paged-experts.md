---
title: "Paged Experts: Serving MoE Models Larger Than Your GPU in SGLang"
author: "Jan Schalkamp, Xiaoyu Zhang"
date: "TBD, 2026"
previewImg: /images/blog/sglang-paged-experts/cover.png
type: blog
---

Mixture-of-Experts (MoE) models are now a popular shape of open-weight LLMs, and most of their parameters are experts. A token only uses `top_k` of the `E` experts in each layer, but every expert has to be somewhere the GPU can reach quickly, and a consumer GPU has 8–16 GB of VRAM: Qwen3-30B-A3B in 4-bit is 17 GB, its FP8 version 31 GB, Qwen3-Next-80B-A3B in NVFP4 about 48 GB.

**Paged Experts** lets SGLang keep only part of a MoE model's experts in GPU memory and the rest in host RAM, with every expert still computed on the GPU. That serves models that did not fit before, and it turns the VRAM the experts take into something you can trade: give it to the KV cache for longer context or more concurrent requests, or spend it on a larger, higher-precision checkpoint. It is merged into SGLang main and enabled with one flag, `--enable-paged-experts`.

**TL;DR**

- **Run MoE models that do not fit your GPU.** On a 16 GB RTX 5070 Ti, Qwen3-Next-80B-A3B NVFP4 (48 GB) decodes at **101 tok/s** with its experts in host RAM; with an NVMe drive behind host RAM, DeepSeek-V4-Flash (167 GB) runs at 14.0 tok/s on the same card with 62 GB of RAM. <!-- TODO: re-measure on main (pre-copy, in main since #51279, is not in these numbers). -->
- **It unlocks new trade-offs.** Even if a model would fit VRAM, Paged Experts can be used to limit the number of experts to gain VRAM for e.g. longer contexts or more concurrent requests.
- **How it works.** Each MoE layer keeps K of its experts on the GPU and all of them in pinned host RAM. Before the layer runs, the GPU checks the routed experts against the resident ones and copies in the missing ones, replacing the least recently used, all inside the decode step's CUDA graph.
- **Why it works.** A token reads only a few experts per layer, and consecutive tokens keep returning to the same ones: at 92 of 128 experts resident, 0.69 % of routed experts miss. And everything in a decode step except those copies already runs as fast as the GPU can read its weights.

<!-- TODO: add an HTML table of contents here once the headings are final, as other long LMSYS posts do. -->

## 1. The idea: GPU memory as a cache of experts

Take one MoE layer of Qwen3-30B-A3B: 128 experts, of which each token uses 8. On a 16 GB card all 128 do not fit next to attention and the KV cache, but 92 do. Paged Experts keeps those 92 in **slots** on the GPU (the *resident* experts) and all 128 in a **host store** in pinned RAM. The router does not know: it still routes over all 128. Before the layer runs, a small *decide* kernel checks the 8 experts the router picked against the slots: a resident expert is used where it sits, a missing one is **copied** over PCIe into the slot of the least recently used expert, and the layer's own fused-MoE kernel runs over the slots. At K = 92, 0.69 % of routed experts miss.

If this sounds like virtual memory, it is meant to. An expert is a fixed-size block of weights that a token reads whole or not at all, which makes it a natural page; a miss is a page fault, and paging decides only where an expert sits when the GPU reads it, never which experts run. Two things are unlike an operating system: the GPU handles its own faults, inside the decode step's CUDA graph, and nothing is ever computed on the CPU: the page always comes to the GPU. When host RAM runs out, the same scheme goes one tier further, with an NVMe drive behind the host store (section 4.5).

<img src="/images/blog/sglang-paged-experts/architecture.svg" alt="One MoE layer in a decode step: router, decide kernel, copy kernel moving only the misses from pinned host RAM into the K slots, and the fused-MoE kernel over the slots, all inside the CUDA graph" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 1: One MoE layer, one decode step. Everything in the blue frame runs on the GPU inside the decode CUDA graph.</p>

**What it buys.** Without paging, every expert of every layer has to sit in VRAM, and the experts take most of it. With paging, how many stay resident is a choice, and the VRAM they do not use goes elsewhere: to the KV cache for longer context or more concurrent requests, or to a larger, higher-precision checkpoint. On Qwen3-30B GPTQ, a 10,240-token KV cache leaves K = 92 of 128 resident (233.3 tok/s), the model's full 40,960-token context K = 63 (208.4 tok/s), and four requests of 16,384 tokens K = 46 (173.1 tok/s). SGLang picks K from what is left once the KV cache you asked for is allocated. What paging needs in return is host RAM for one copy of every expert (13.9 GiB for Qwen3-30B GPTQ, 40.7 GiB for Qwen3-Next-80B NVFP4), or an NVMe drive for the rest.

<img src="/images/blog/sglang-paged-experts/tradeoff.svg" alt="Decode tok/s against the KV cache kept for Qwen3-30B-A3B GPTQ-Int4 on an RTX 5070 Ti 16 GB: 10,240 tokens K 92 233.3 tok/s; 16,384 tokens K 86 230.6; 40,960 tokens K 63 208.4; 65,536 tokens K 46 173.1" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 2: The VRAM the KV cache does not take holds experts. More context means fewer resident experts and slower decode.</p>

## 2. Performance results

One desktop (RTX 5070 Ti 16 GB, PCIe 5.0, 62 GB RAM, a PCIe 5.0 NVMe drive) and one laptop (RTX 4070 Laptop 8 GB, 54 GB RAM under WSL2). The laptop's GPU sits on PCIe 4.0 x8 instead of the desktop's 5.0 x16: read its row as a slower-link data point. Decode is one request at a time after warm-up, K is chosen automatically, and the RAM-tier rows keep a 10,240-token KV cache. `a / b` are two runs of the same configuration. TTFT is for ShareGPT prompts on the RAM-tier rows and for a natural ~1.5k-token prompt on the disk-tier rows, which run one request at a time; the full setup is in section 7.

| model (size) | GPU | K of E on the GPU | host RAM for experts | decode tok/s, 1 request | TTFT p50 |
|---|---|---|---|---|---|
| Qwen3-30B-A3B GPTQ-Int4 (17 GB) | 16 GB desktop | 92 of 128 | 13.9 GiB | **233.3** | 40 ms |
| gpt-oss-20b MXFP4 (14 GB) | 16 GB desktop | 28 of 32 | TBD | **172.5** | 37 ms |
| Qwen3-30B-A3B FP8 (31 GB) | 16 GB desktop | 45 of 128 | 27.0 GiB | **105.0** | 182 ms |
| Qwen3-Next-80B-A3B NVFP4 (48 GB) | 16 GB desktop | 101 of 512 | 40.7 GiB | **100.5–101.1** | 192 ms |
| DeepSeek-V4-Flash-0731 (167 GB) | 16 GB desktop | 9 of 256 | 45 GiB + NVMe | **13.98 / 13.97** | 8.49 / 8.46 s |
| Qwen3.8-Flash-Next-FP8 (112.7 GiB of experts) | 16 GB desktop | 11 of 512 | TBD + NVMe | **15.69 / 15.72** | 15.0 / 15.6 s |
| Qwen3-Next-80B-A3B NVFP4 (48 GB) | 8 GB laptop | 19 of 512 | TBD + NVMe | **15.73 / 16.03** | 4.99 / 4.88 s |

<!-- TODO: re-measure the whole table on main; the disk-tier rows were measured on the tier's development branch. Not measured yet: gpt-oss-20b host store, the pinned GiB of the Qwen3.8 and laptop disk-tier runs (not in their logs), peak VRAM. -->

Accuracy holds: GSM8K (200 questions, thinking off) is 0.950 for Qwen3-30B GPTQ, 0.94–0.95 for its FP8 version, 0.965–0.975 for Qwen3-Next, 0.96–0.97 for gpt-oss-20b (0.975 unpaged), 0.975 for DeepSeek-V4-Flash and 0.97 for Qwen3.8, and 1.00 on 50 questions on the laptop. Section 1 shows what longer context costs.

<img src="/images/blog/sglang-paged-experts/decode-speed.svg" alt="Decode tok/s for one request on an RTX 5070 Ti 16 GB: Qwen3-30B-A3B GPTQ-Int4 233.3, gpt-oss-20b 172.5, Qwen3-30B-A3B FP8 105.0, Qwen3-Next-80B-A3B NVFP4 100.8" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 3: Decode speed for one request on a 16 GB GPU, with K picked automatically.</p>

<!-- TODO (Figure 3): regenerate after re-measuring on main; add fully-resident bars only where a same-day baseline exists. -->

## 3. Why it works

Three facts make paging experts viable, and each one decides a part of the design in section 4.

### 3.1 A token reads a few experts, and keeps reusing them

Qwen3-30B-A3B has 128 experts in each of its 48 MoE layers; in GPTQ-Int4 they take 13.9 GiB, the whole usable VRAM of a 16 GB card. Each token, though, is routed to 8 of the 128 per layer: it reads 0.93 GiB of those 13.9 GiB. <!-- TODO: the old text said "0.93 GB of the 14.9 GB"; 14.9 GB = 13.9 GiB, check 0.93 against the same unit. -->

And consecutive tokens keep returning to the same experts. If routing were uniform, a routed expert would miss the slots with probability 1 − K/E; measured routing misses far less. We call uniform miss rate ÷ measured miss rate the *concentration factor*:

| model, K of E | miss per routed expert | uniform 1 − K/E | concentration |
|---|---|---|---|
| Qwen3-30B GPTQ, 92 of 128 | 0.69 % | 28.1 % | 41x |
| Qwen3-30B FP8, 45 of 128 | 9.6 % | 64.8 % | 6.8x |
| Qwen3-Next-80B NVFP4, 101 of 512 (top-10) | 13.0 % | 80.3 % | 6.2x |

The concentration is largest where the cache is large relative to E: a small cache holds the popular head, and the tail is broad.

<img src="/images/blog/sglang-paged-experts/ksweep.svg" alt="Share of routed experts that miss the GPU against K, with pre-copy on main. Qwen3-30B GPTQ (128 experts): K 16 41.1 % before pre-copy and 16.0 % copied on demand, K 92 0.7 % and 0.4 %. Qwen3-Next-80B NVFP4 (512 experts): K 20 40.9 % and 20.4 %, K 100 13.2 % and 5.4 %. Uniform routing 1 - K/E is far above both." style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 4: Measured routing misses the GPU far less often than uniform routing would, at every K. On SGLang main, a pre-copy of each layer's predicted experts during the previous layer moves part of the remaining misses off the critical path; only the orange ones are still waited for.</p>

Which experts are popular, though, moves with the workload. On three prose prompts Qwen3-Next touched 169 of its 512 experts per layer, and a frozen set of the most popular ones did as well as a cache that adapts. Twelve mixed prompts (prose, code, math, JSON, German, French) touched 450, and the frozen set lost 28 points of hit rate when fitted on the very trace it was scored on, 53 when fitted on other data: the median expert returns after 22 accesses, the 99th percentile after 4,723, when the topic swings back. So the GPU's copy of the experts has to be a cache that follows the workload, not a pinned hot set. Which replacement rule it uses matters little: in a simulation on Qwen3-Next routing at 96 resident experts, LRU-2, a decayed LFU and a router-score average all land within 5 % of plain LRU's copies, while 32 more slots cut them by 25 %.

### 3.2 A decode step is a byte problem

Paging adds exactly one cost: copying the experts a token needs that are not resident. Per token that is

```text
bytes copied per token = miss rate(K) × top_k × MoE layers × bytes per expert
```

Qwen3-30B GPTQ at K = 92 of 128 copies 2.66 experts (6.5 MB) per token, a 0.21 ms copy in a 4.27 ms step (5 %); its FP8 version at K = 45 copies 36.7 (173 MB), 3.52 ms of a 9.52 ms step (37 %). <!-- TODO: re-measure on main. -->

The rest of a step is a floor that paging cannot change. We traced eight captured decode steps per model with CUPTI and joined each copy kernel to the experts it copied; every part is a measured kernel time, not a difference of two totals:

<img src="/images/blog/sglang-paged-experts/step-split.svg" alt="Stacked bars of one decode step's kernel time: Qwen3-30B GPTQ copies 0.21, decide 0.11, routed GEMMs 1.41, rest 2.60 ms; Qwen3-30B FP8 3.52, 0.18, 2.40, 3.60 ms; Qwen3-Next-80B NVFP4 2.36, 0.27, 1.22, 6.36 ms" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 5: One decode step, split by measured kernel time. Only the orange part is added by paging.</p>

<!-- TODO: recheck the "other" row before publishing (it is derived by subtraction); re-trace on main, where pre-copy takes copies off the critical path. -->

Everything but the copies already runs at the GPU's read floor. Qwen3-Next's linear-attention (GDN) projections read at 90 % and its `lm_head` at 97 % of the card's measured 843 GB/s, a dense bf16 model decodes at 98.6 % of its byte floor, and Qwen3-30B GPTQ's routed Marlin GEMMs take 1.41 ms against a read floor of 1.11 ms: a perfect kernel would save 7 % of the step. What is left to engineer is the copies (5–37 % of a step) and the overhead between kernels (launches, syncs, small bookkeeping kernels). That is also why Qwen3-30B GPTQ decode stops depending on K above K ≈ 88 (230.6 tok/s at K = 86, 233.3 at K = 92): its copies are already a rounding error.

This post is about one stream at a time. A batch is a different regime: it pays the union of its requests' experts once per step, and a result from one rarely transfers to the other.

### 3.3 The link is 16x slower than VRAM, so only unread weights can leave the GPU

On our card a byte in VRAM reads at 843 GB/s and a byte over PCIe 5.0 at 52.8 GB/s, 16x slower. A weight that every token reads crosses the link on every token, so offloading just 6 % of a dense model's bytes doubles its step time. Only weights that a token does not read escape this, and that is exactly what most MoE experts are: this is why paging experts works where `--cpu-offload-gb`, which copies every offloaded weight on every step, cannot.

The other way to use this sparsity is to compute the offloaded experts on the CPU, and there the concentration factor decides. With the same VRAM, CPU offload computes a share of about `1 − K/E` of the experts on the CPU, while paging copies a share `m`; paging wins roughly when

```text
(1 − K/E) / m   >   B_cpu / B_link
 concentration      CPU expert bandwidth / host-to-GPU copy bandwidth
```

Uniform routing makes the left side 1; measured routing makes it 5–50, so paging wins wherever the link is near RAM speed, as PCIe 5.0 x16 is against dual-channel DDR5. Short-prompt TTFT is where CPU offload has the edge: a prefill routes to nearly every expert, so paging copies every non-resident expert once per prefill chunk, while CPU offload computes its experts where they are.

## 4. The building blocks

Five pieces make the idea work. The first three are one MoE layer in a decode step, as in Figure 1; the last two extend it to larger steps and to a third tier.

### 4.1 Two places for every expert

The paged method wraps the layer's own fused-MoE method. On the GPU, the layer's expert tensors get K rows instead of E: these are the slots. In pinned host RAM, a host store holds all E experts. Experts load through SGLang's own per-expert loader and the format's own post-load step (Marlin's repack, for example) before they reach the host store, so the store holds every expert already in the layout the kernel reads, and a miss is a plain copy from host RAM into a slot. Adding a quantization format is one small subclass; the [docs](https://docs.sglang.io/advanced_features/paged_experts.html) list the supported ones.

### 4.2 The MoE kernel never knows

The router still picks from all E experts. Before the layer's kernel runs, each routed expert id is replaced by the id of the slot that holds it, and the layer's own fused-MoE kernel runs as if the layer had only K experts. No new MoE kernel is needed, so every format and runner keeps its own kernel, and the computation is the one the unpaged layer does: paged layers are bit-identical to unpaged ones for bf16, FP8 and INT8 experts (with `--enable-deterministic-inference`, which the unpaged layer itself needs to be reproducible). Marlin's MoE kernel is not reproducible from run to run on SM90 and newer, paged or not, so for the Marlin formats we checked bytes instead: every slot against the host store after every step in a test build (0 mismatches in 83,850 checks), plus GSM8K.

### 4.3 Decide and copy on the GPU, inside the graph

A paged layer adds two small kernels in front of the MoE kernel, and kernels that small are cheap on the GPU but expensive to launch from the CPU one by one. So in decode both run on the GPU and are captured in the decode step's CUDA graph with the rest of the layer:

- **decide** matches the routed experts against the slots: a resident expert keeps its slot, and each missing one takes the least recently used slot that the step itself does not need. It writes the list of copies and the slot ids the MoE kernel reads.
- **copy** moves the listed experts from host RAM into their slots, reading the pinned host memory directly from the GPU. It reads how many experts to copy from GPU memory, so every replay of the graph copies exactly what its own decide chose, and the CPU never waits for a decision.

This covers every decode step whose routed experts fit in the slots, up to `K // top_k` concurrent requests. Once the CPU was out of the loop, the number of kernels per layer was what remained: having decide also write the MoE kernel's inputs, instead of a separate remapping step, cut a paged layer's graph from 21 nodes to 10 and gave +10.3 % decode on Qwen3-30B GPTQ.

<!-- TODO: measure eager vs captured decode on main (--disable-cuda-graph against the default) for Qwen3-30B GPTQ, with GPU µs per launch; dense Qwen3.8-4B bf16 had 24.9 µs per launch and 1.04x from capture. -->

### 4.4 Steps larger than K: waves

A prefill chunk, or a decode batch beyond `K // top_k` requests, routes to more distinct experts than there are slots. Such a step runs in **waves** of at most K experts, planned on the CPU; under SGLang's prefill CUDA graphs the paged layer runs uncaptured between captured segments, and both paths share one LRU state. Each wave's kernel call returns every routed (token, expert) output already weighted by its router weight but without the final sum over the token's experts; all waves write into one buffer and that sum runs once at the end. So the result does not depend on how the experts were split into waves.

### 4.5 A third tier on NVMe

The RAM tier needs every expert in pinned host RAM. DeepSeek-V4-Flash is a 167 GB checkpoint whose routed experts take 137.1 GiB; on our 62 GB machine about 45 GiB can be pinned next to the rest of the server. The disk tier gives each layer three tiers: K experts in GPU slots, R in pinned **host rows**, and all of them in a file on the NVMe drive, written once in the kernel's layout on the first start. An expert that is only on disk is read into a host row with one direct (`O_DIRECT`) read and then copied into its slot like any other miss. 

<img src="/images/blog/sglang-paged-experts/tiers.svg" alt="Three tiers for one MoE layer of DeepSeek-V4-Flash: 9 of 256 experts in GPU slots, 71 in pinned host rows, all 256 in a file on the NVMe drive; misses move up by an O_DIRECT read and a copy over PCIe, nothing is written back" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 6: The disk tier for DeepSeek-V4-Flash on a 16 GB GPU with 62 GB of RAM.</p>

Three lessons shaped it:

- **Never write back.** The host store keeps every expert, so an evicted slot is simply overwritten. The disk tier's first version kept each expert in exactly one tier, so every copy was a swap that sent the evicted expert back to host RAM; both directions share host DRAM (~62 GB/s together), and swaps took 57.7 of a 90.6 ms GPU step. Letting host rows keep their copy (the file always has every expert) made each move one-way: **+24.9 % decode** (11.05 → 13.75 tok/s), at 18 % more disk reads.
- **Keep the CPU out of the step, even for a disk read.** A planning kernel on the GPU marks which experts are only on disk; a host thread reads them while the layer's other copies run, and the captured graph waits at that layer until they are in their slots. On the PCIe 5.0 drive that beat replaying the step by 33 %. <!-- TODO: re-measure wait vs replay on main if quoted. -->
- **Prefill reads the whole tier.** A prompt of a few hundred tokens already routes to nearly every expert of every layer, so one prefill reads almost the whole tier (~93 GB for one of these prompts on DeepSeek-V4), and TTFT is that read at disk speed (~11 GB/s here): 8.49 / 8.46 s for a natural ~1.5k-token prompt. Chunked prefill would re-read the tier for every chunk, so the recipe uses one chunk per prompt.

## 5. Limits and what's next

- **Batches beyond `K // top_k`** run as CPU-planned waves without a graph and fall off sharply **[MEASURE ON MAIN: FP8 at a small K with 4 requests]**. On an earlier research fork, skipping each wave's masked rows plus a parallel decide made large batches approach the link's limit, where every layer's union is nearly all experts, paid once per step. Not in main. <!-- TODO: re-measure on main. -->
- **Short-prompt TTFT**: a prefill copies every non-resident expert once per chunk; on the disk tier it reads the whole tier.
- **Slower links**: on PCIe 4.0 or a laptop's x8 link the inequality of section 3.3 moves toward a tie with CPU offload at bs=1; we have a laptop serving, but no head-to-head there yet.
- **The dense part**: on Qwen3-Next, 4 of 10 ms per step are bf16 GEMVs at the read floor; only fewer bytes move them.

**Next.** On development branches, the disk tier reads the next layer's predicted disk misses ahead (+7.2 %) and streams each read into its slot as it lands (+3.9 %); other work covers decode batches beyond `K // top_k` and keeping the embedding table in host memory to give its VRAM to K. From the community, an opt-in kernel that reads missing experts straight from host memory inside the MoE GEMM ([#51471](https://github.com/sgl-project/sglang/pull/51471)) and a port to Ascend NPUs ([#51878](https://github.com/sgl-project/sglang/pull/51878)) are in review. <!-- TODO: re-measure the disk-tier levers before quoting; the old text also had 16.4-16.7 tok/s on DSv4 with pre-copy (pre-copy does not run on disk-tier layers or DSv4's router in main), and the #46520 RFC link (shared on-disk format). -->

## 6. Try it

The smallest launch:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts
```

This keeps the KV cache for one request of the model's full 40,960-token context, which leaves K = 63 on a 16 GB card (208 tok/s, section 1); the log names the K it chose. On hybrid models such as Qwen3-Next, it also runs one request at a time unless `--max-running-requests` says otherwise. To trade context for resident experts, set the KV cache you need; these are the configurations of section 2:

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

Experts beyond host RAM go to the disk tier. DeepSeek-V4-Flash (137 GiB of experts) on the same 16 GB GPU with 62 GB of RAM:

```bash
SGLANG_OPT_FP8_WO_A_GEMM=1 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 45 \
  --moe-runner-backend marlin --mem-fraction-static 0.92 --max-running-requests 1 \
  --context-length 4096 --json-model-override-args '{"max_position_embeddings":4096}' \
  --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.5 --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill disabled --disable-radix-cache
```

`SGLANG_OPT_FP8_WO_A_GEMM=1` keeps DeepSeek-V4's `wo_a` in FP8 on SM120; one prefill chunk per prompt reads each expert from disk once (section 4.5).

Requirements: one GPU (no speculative decoding, no tensor or expert parallelism yet); for the disk tier an x86-64 host and safetensors checkpoints with one tensor per expert. Everything else is in the [Paged Experts docs](https://docs.sglang.io/advanced_features/paged_experts.html). <!-- TODO: confirm the docs URL. -->

## 7. How we measured

**Machines.**

| | desktop | laptop (Dell XPS 16 9640) |
|---|---|---|
| GPU | RTX 5070 Ti 16 GB (SM120), 300 W | RTX 4070 Laptop GPU 8 GB (SM89), 60 W power limit |
| GPU link | PCIe 5.0 x16 | PCIe 4.0 x8 |
| CPU | AMD Ryzen 7 9800X3D (8 cores) | Intel Core Ultra 9 185H |
| RAM | 2 x 32 GB DDR5-6000, two channels (62 GB usable) | 64 GB LPDDR5x-7467 (54 GB visible to WSL2) |
| NVMe (expert cache, disk tier) | Kingston FURY Renegade G5 2 TB, PCIe 5.0 x4 | SK hynix PVC10 1 TB, through WSL2's virtual disk |
| OS | Ubuntu 26.04 LTS, Linux 7.0, native | Windows 11 (build 26300), WSL2 2.7.8 (Linux 6.18), Ubuntu 24.04 |
| NVIDIA driver | 595.91.07 (CUDA 13.2) | 621.14 (CUDA 13.5) |
| SGLang | the `lmsysorg/sglang` nightly CUDA 13 image with the code under test | same |

<!-- TODO: name the SGLang commit and image of the final runs in the SGLang row. -->

**Setup.** SGLang with `--enable-paged-experts`, `--paged-experts-num-resident auto`, `--mem-fraction-static 0.88`, prefix caching off. Decode tok/s is `1000 / median TPOT` over 8 ShareGPT prompts sent one at a time (256 output tokens each). GSM8K is 200 questions, thinking off.

Long prompts: on Qwen3-30B-A3B GPTQ a natural, non-repeating prompt of about 2k tokens has a TTFT p50 of 227–247 ms, and an 8k-token one 1.47 s (`--context-length 16384`, K = 86). <!-- TODO: record the token count of the natural ~2k prompts under the Qwen3 tokenizer. -->

Four rules, each here because breaking it once gave us a wrong number:

- **Repeat the first configuration last.** A difference smaller than the gap between its two runs (±0.2 % on GPTQ decode) is no difference.
- **Drop the page cache and discard warm-up** (16 ShareGPT requests). A cold expert pool reads 5–11 % low, and a full page cache once cut our throughput 2.6x without an error.
- **Natural prompts, never tiled ones.** SGLang's `random` dataset tiles a short ShareGPT turn into a long prompt; repeated text routes to fewer experts, and paged 2k TTFT read 261–265 ms instead of 327.
- **A ratio of two instruments is not a measurement.** A "copy defect" we chased for a while was a profiler-inflated kernel time divided by another tool's byte count; the kernel was running at link speed all along.

## Acknowledgments

TODO <!-- include community contributors to Paged Experts: Woody-Hu (#48605), aojsd (#51471), jzlxiaohei (#51498), the Ascend port (#51878). -->
