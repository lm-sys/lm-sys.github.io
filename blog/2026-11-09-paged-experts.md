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

- **Run MoE models that do not fit your GPU.** On a 16 GB RTX 5070 Ti, Qwen3-Next-80B-A3B NVFP4 (48 GB) decodes at **116 tok/s** with its experts in host RAM; with an NVMe drive behind host RAM, DeepSeek-V4-Flash (167 GB) runs at 13.0 tok/s on the same card with 62 GB of RAM.
- **It unlocks new trade-offs.** Even if a model would fit VRAM, Paged Experts can be used to limit the number of experts to gain VRAM for e.g. longer contexts or more concurrent requests.
- **How it works.** Each MoE layer keeps K of its experts on the GPU and all of them in pinned host RAM. Before the layer runs, the GPU checks the routed experts against the resident ones and copies in the missing ones, replacing the least recently used, all inside the decode step's CUDA graph.
- **Why it works.** A token reads only a few experts per layer, and consecutive tokens keep returning to the same ones: at 91 of 128 experts resident, 0.8 % of routed experts are not on the GPU when the token reaches the layer. And everything in a decode step except those copies already runs as fast as the GPU can read its weights.

## 1. The idea: GPU memory as a cache of experts

Take one MoE layer of Qwen3-30B-A3B: 128 experts, of which each token uses 8. On a 16 GB card all 128 do not fit next to attention and the KV cache, but 91 do. Paged Experts keeps those 91 in **slots** on the GPU (the *resident* experts) and all 128 in a **host store** in pinned RAM. The router does not know: it still routes over all 128. Before the layer runs, a small *decide* kernel checks the 8 experts the router picked against the slots: a resident expert is used where it sits, a missing one is **copied** over PCIe into the slot of the least recently used expert, and the layer's own fused-MoE kernel runs over the slots. At K = 91, 0.8 % of routed experts miss.

If this sounds like virtual memory, it is meant to. An expert is a fixed-size block of weights that a token reads whole or not at all, which makes it a natural page; a miss is a page fault, and paging decides only where an expert sits when the GPU reads it, never which experts run. Two things are unlike an operating system: the GPU handles its own faults, inside the decode step's CUDA graph, and nothing is ever computed on the CPU: the page always comes to the GPU. When host RAM runs out, the same scheme goes one tier further, with an NVMe drive behind the host store (section 4.5).

<img src="/images/blog/sglang-paged-experts/architecture.svg" alt="One MoE layer in a decode step: router, decide kernel, copy kernel moving only the misses from pinned host RAM into the K slots, and the fused-MoE kernel over the slots, all inside the CUDA graph" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 1: One MoE layer, one decode step. Everything in the blue frame runs on the GPU inside the decode CUDA graph.</p>

**What it buys.** Without paging, every expert of every layer has to sit in VRAM, and the experts take most of it. With paging, how many stay resident is a choice, and the VRAM they do not use goes elsewhere: to the KV cache for longer context or more concurrent requests, or to a larger, higher-precision checkpoint. On Qwen3-30B GPTQ, a 10,240-token KV cache leaves K = 91 of 128 resident (241.5 tok/s), the model's full 40,960-token context K = 62 (218.3 tok/s), and four requests of 16,384 tokens K = 44 (183.1 tok/s). SGLang picks K from what is left once the KV cache you asked for is allocated. What paging needs in return is host RAM for one copy of every expert (13.9 GiB for Qwen3-30B GPTQ, 40.5 GiB for Qwen3-Next-80B NVFP4), or an NVMe drive for the rest.

<img src="/images/blog/sglang-paged-experts/tradeoff.svg" alt="Decode tok/s against the KV cache kept for Qwen3-30B-A3B GPTQ-Int4 on an RTX 5070 Ti 16 GB: 10,240 tokens K 91 241.5 tok/s; 16,384 tokens K 84 238.3; 40,960 tokens K 62 218.3; 65,536 tokens K 44 183.1" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 2: The VRAM the KV cache does not take holds experts. More context means fewer resident experts and slower decode.</p>

## 2. Performance results

One desktop (RTX 5070 Ti 16 GB, PCIe 5.0, 62 GB RAM, a PCIe 5.0 NVMe drive) and one laptop (RTX 4070 Laptop 8 GB, 54 GB RAM under WSL2). The laptop's GPU sits on PCIe 4.0 x8 instead of the desktop's 5.0 x16: read its row as a slower-link data point. Decode is one request at a time after warm-up, K is chosen automatically, and the RAM-tier rows keep a 10,240-token KV cache. `a / b` are two runs of the same configuration. TTFT is the median for short ShareGPT prompts and for natural, non-repeating prompts of about 1,470 tokens; the desktop's disk-tier rows were measured with the long prompts only. The full setup is in section 6.

| model (size) | GPU | K of E on the GPU | host RAM for experts | decode tok/s, 1 request | TTFT p50, short prompts | TTFT p50, ~1.5k-token prompt |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B GPTQ-Int4 (17 GB) | 16 GB desktop | 91 of 128 | 13.9 GiB | **241.5** | 43 ms | 259 ms |
| gpt-oss-20b MXFP4 (14 GB) | 16 GB desktop | 28 of 32 | 10.3 GiB | **173.5** | 38 ms | 200 ms |
| Qwen3-Next-80B-A3B NVFP4 (48 GB) | 16 GB desktop | 99 of 512 | 40.5 GiB | **116.2** | 191 ms | 687 ms |
| Qwen3-30B-A3B FP8 (31 GB) | 16 GB desktop | 44 of 128 | 27.0 GiB | **113.0** | 191 ms | 398 ms |
| DeepSeek-V4-Flash-0731 (167 GB) | 16 GB desktop | 8 of 256 | 45 GiB + NVMe | **13.0 / 13.0** | – | 8.88 / 8.91 s |
| Qwen3.8-Flash-Next-FP8 (186 GB) | 16 GB desktop | 10 of 512 | 36 GiB + NVMe | **14.9 / 14.9** | – | 7.40 / 7.41 s |
| Qwen3-Next-80B-A3B NVFP4 (48 GB) | 8 GB laptop | 17 of 512 | 18 GiB + NVMe | **16.2 / 16.5** | 1.35 s | 4.56 / 4.72 s |

Paging does not change what the model computes: section 4.2 shows paged layers are bit-identical to unpaged ones. GSM8K is a sanity check on top (thinking off): on all 1,319 questions (one standard error is about 0.6 points) 0.938 for Qwen3-30B GPTQ, 0.951 for its FP8 version, 0.954 for Qwen3-Next and 0.950 for gpt-oss-20b; on 200 questions (about 1.4 points) 0.98 for DeepSeek-V4-Flash and 0.965 for Qwen3.8. Section 1 shows what longer context costs.

<img src="/images/blog/sglang-paged-experts/decode-speed.svg" alt="Decode tok/s for one request on an RTX 5070 Ti 16 GB: Qwen3-30B-A3B GPTQ-Int4 241.5, gpt-oss-20b 173.5, Qwen3-Next-80B-A3B NVFP4 116.2, Qwen3-30B-A3B FP8 113.0" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 3: Decode speed for one request on a 16 GB GPU, with K picked automatically.</p>

None of these models runs fully resident at the table's setting. gpt-oss-20b, the smallest at 14 GB, needs `--mem-fraction-static` of at least 0.92 for its weights alone; at 0.95, with the same 10,240-token KV cache, it does run fully resident and decodes at 179.4 tok/s. Paged at that setting, it keeps 31 of its 32 experts on the GPU and decodes at 176.8 tok/s, 1.5 % slower.


## 3. Why it works

Three facts make paging experts viable, and each one decides a part of the design in section 4.

### 3.1 A token reads a few experts, and keeps reusing them

Qwen3-30B-A3B has 128 experts in each of its 48 MoE layers; in GPTQ-Int4 they take 13.9 GiB, the whole usable VRAM of a 16 GB card. Each token, though, is routed to 8 of the 128 per layer: it reads 0.87 GiB of those 13.9 GiB.

And consecutive tokens keep returning to the same experts. If routing were uniform, a routed expert would miss the slots with probability 1 − K/E; measured routing misses far less. We call uniform miss rate ÷ measured miss rate the *concentration factor* (measured at the K of section 2 with pre-copy off, over the traced request of section 3.2; Figure 4 shows how much of it pre-copy takes off the critical path):

| model, K of E | miss per routed expert | uniform 1 − K/E | concentration |
|---|---|---|---|
| Qwen3-30B GPTQ, 91 of 128 | 0.80 % | 28.9 % | 36x |
| Qwen3-30B FP8, 44 of 128 | 9.9 % | 65.6 % | 6.6x |
| Qwen3-Next-80B NVFP4, 99 of 512 (top-10) | 12.4 % | 80.7 % | 6.5x |

The concentration is largest where the cache is large relative to E: a small cache holds the popular head, and the tail is broad.

<img src="/images/blog/sglang-paged-experts/ksweep.svg" alt="Share of routed experts that miss the GPU against K, with pre-copy on main. Qwen3-30B GPTQ (128 experts): K 16 41.1 % before pre-copy and 16.0 % copied on demand, K 92 0.7 % and 0.4 %. Qwen3-Next-80B NVFP4 (512 experts): K 20 40.9 % and 20.4 %, K 100 13.2 % and 5.4 %. Uniform routing 1 - K/E is far above both." style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 4: Measured routing misses the GPU far less often than uniform routing would, at every K. On SGLang main, a pre-copy of each layer's predicted experts during the previous layer moves part of the remaining misses off the critical path; only the orange ones are still waited for.</p>

Which experts are popular, though, moves with the workload. On three prose prompts Qwen3-Next touched 169 of its 512 experts per layer, and a frozen set of the most popular ones did as well as a cache that adapts. Twelve mixed prompts (prose, code, math, JSON, German, French) touched 450, and the frozen set lost 28 points of hit rate when fitted on the very trace it was scored on, 53 when fitted on other data: the median expert returns after 22 accesses, the 99th percentile after 4,723, when the topic swings back. So the GPU's copy of the experts has to be a cache that follows the workload, not a pinned hot set. Which replacement rule it uses matters little: in a simulation on Qwen3-Next routing at 96 resident experts, LRU-2, a decayed LFU and a router-score average all land within 5 % of plain LRU's copies, while 32 more slots cut them by 25 %.

### 3.2 A decode step is a byte problem

Paging adds exactly one cost: copying the experts a token needs that are not resident. Per token that is

```text
bytes copied per token = miss rate(K) × top_k × MoE layers × bytes per expert
```

With pre-copy switched off, so that every copy stays on the critical path, Qwen3-30B GPTQ at K = 91 of 128 copies 3.06 experts (7.4 MB) per token, a 0.195 ms copy in a 4.376 ms step (4.5 %); its FP8 version at K = 44 copies 38.0 (179 MB), 3.525 ms of a 9.824 ms step (36 %). SGLang main hides part of that with a pre-copy of each layer's predicted experts ([#51279](https://github.com/sgl-project/sglang/pull/51279)): decode is 2.4 % faster on GPTQ, 9.5 % on FP8 and 12.4 % on Qwen3-Next than with pre-copy off.

The rest of a step is a floor that paging cannot change. We traced the 256 captured decode steps of one whole request per model with CUPTI, pre-copy off, and joined each copy kernel to the experts it copied; every part is a measured kernel time, not a difference of two totals:

<img src="/images/blog/sglang-paged-experts/step-split.svg" alt="Stacked bars of one decode step's kernel time with pre-copy off: Qwen3-30B GPTQ copies 0.195, decide 0.110, routed GEMMs 1.417, rest 2.654 ms; Qwen3-30B FP8 3.525, 0.124, 2.606, 3.569 ms; Qwen3-Next-80B NVFP4 1.964, 0.086, 1.304, 6.357 ms" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 5: One decode step, split by measured kernel time. Only the orange part is added by paging.</p>

Everything but the copies already runs at the GPU's read floor. Qwen3-Next's linear-attention (GDN) projections read at 90 % and its `lm_head` at 97 % of the card's measured 843 GB/s, a dense bf16 model decodes at 98.6 % of its byte floor, and Qwen3-30B GPTQ's routed Marlin GEMMs take 1.417 ms against a read floor of 1.11 ms: a perfect kernel would save 7 % of the step. What is left to engineer is the copies (4.5–36 % of a step) and the overhead between kernels (launches, syncs, small bookkeeping kernels). That is also why Qwen3-30B GPTQ decode barely depends on K at the top (238.3 tok/s at K = 84, 241.5 at K = 91): its copies are already a rounding error.

This post is about one stream at a time. A batch is a different regime: it pays the union of its requests' experts once per step, and a result from one rarely transfers to the other.

### 3.3 The link is 16x slower than VRAM, so only unread weights can leave the GPU

On our card a byte in VRAM reads at 843 GB/s and a byte over PCIe 5.0 at 52.8 GB/s, 16x slower. A weight that every token reads crosses the link on every token, so offloading just 6 % of a dense model's bytes doubles its step time. Only weights that a token does not read escape this, and that is exactly what most MoE experts are: this is why paging experts works where `--cpu-offload-gb`, which copies every offloaded weight on every step, cannot.

The other way to use this sparsity is to compute the offloaded experts on the CPU, and there the concentration factor decides. With the same VRAM, CPU offload computes a share of about `1 − K/E` of the experts on the CPU, while paging copies a share `m`; paging wins roughly when

```text
(1 − K/E) / m   >   B_cpu / B_link
 concentration      CPU expert bandwidth / host-to-GPU copy bandwidth
```

Uniform routing makes the left side 1; measured routing makes it 6.5–36 (section 3.1), so the inequality predicts that paging wins wherever the link is near RAM speed, as PCIe 5.0 x16 is against dual-channel DDR5. This post does not measure CPU offload. Short-prompt TTFT is where CPU offload has the edge: a prefill routes to nearly every expert, so paging copies every non-resident expert once per prefill chunk, while CPU offload computes its experts where they are.

## 4. The building blocks

Five pieces make the idea work. The first three are one MoE layer in a decode step, as in Figure 1; the last two extend it to larger steps and to a third tier.

### 4.1 Two places for every expert

The paged method wraps the layer's own fused-MoE method. On the GPU, the layer's expert tensors get K rows instead of E: these are the slots. In pinned host RAM, a host store holds all E experts. Experts load through SGLang's own per-expert loader and the format's own post-load step (Marlin's repack, for example) before they reach the host store, so the store holds every expert already in the layout the kernel reads, and a miss is a plain copy from host RAM into a slot. Adding a quantization format is one small subclass; the [docs](https://docs.sglang.io/docs/advanced_features/paged_experts) list the supported ones.

### 4.2 The MoE kernel never knows

The router still picks from all E experts. Before the layer's kernel runs, each routed expert id is replaced by the id of the slot that holds it, and the layer's own fused-MoE kernel runs as if the layer had only K experts. No new MoE kernel is needed, so every format and runner keeps its own kernel, and the computation is the one the unpaged layer does: paged layers are bit-identical to unpaged ones for bf16, FP8 and INT8 experts (with `--enable-deterministic-inference`, which the unpaged layer itself needs to be reproducible). Marlin's MoE kernel is not reproducible from run to run on SM90 and newer, paged or not, so for the Marlin formats we checked bytes instead: every slot against the host store after every step in a test build (0 mismatches in 83,850 checks), plus GSM8K.

### 4.3 Decide and copy on the GPU, inside the graph

A paged layer adds two small kernels in front of the MoE kernel, and kernels that small are cheap on the GPU but expensive to launch from the CPU one by one. So in decode both run on the GPU and are captured in the decode step's CUDA graph with the rest of the layer:

- **decide** matches the routed experts against the slots: a resident expert keeps its slot, and each missing one takes the least recently used slot that the step itself does not need. It writes the list of copies and the slot ids the MoE kernel reads.
- **copy** moves the listed experts from host RAM into their slots, reading the pinned host memory directly from the GPU. It reads how many experts to copy from GPU memory, so every replay of the graph copies exactly what its own decide chose, and the CPU never waits for a decision.

This covers every decode step whose routed experts fit in the slots, up to `K // top_k` concurrent requests. Once the CPU was out of the loop, the number of kernels per layer was what remained: having decide also write the MoE kernel's inputs, instead of a separate remapping step, cut a paged layer's graph from 21 nodes to 10 and gave +10.3 % decode on Qwen3-30B GPTQ ([#46457](https://github.com/sgl-project/sglang/pull/46457)).

### 4.4 Steps larger than K: waves

A prefill chunk, or a decode batch beyond `K // top_k` requests, routes to more distinct experts than there are slots. Such a step runs in **waves** of at most K experts, planned on the CPU; under SGLang's prefill CUDA graphs the paged layer runs uncaptured between captured segments, and both paths share one LRU state. Each wave's kernel call returns every routed (token, expert) output already weighted by its router weight but without the final sum over the token's experts; all waves write into one buffer and that sum runs once at the end. So the result does not depend on how the experts were split into waves.

### 4.5 A third tier on NVMe

The RAM tier needs every expert in pinned host RAM. DeepSeek-V4-Flash is a 167 GB checkpoint whose routed experts take 137.1 GiB; on our 62 GB machine about 45 GiB can be pinned next to the rest of the server. The disk tier ([#49195](https://github.com/sgl-project/sglang/pull/49195)) gives each layer three tiers: K experts in GPU slots, R in pinned **host rows**, and all of them in a file on the NVMe drive, written once in the kernel's layout on the first start. An expert that is only on disk is read into a host row with one direct (`O_DIRECT`) read and then copied into its slot like any other miss. 

<img src="/images/blog/sglang-paged-experts/tiers.svg" alt="Three tiers for one MoE layer of DeepSeek-V4-Flash: 8 of 256 experts in GPU slots, 71 in pinned host rows, all 256 in a file on the NVMe drive; misses move up by an O_DIRECT read and a copy over PCIe, nothing is written back" style="display: block; width: 100%; max-width: 100%; height: auto; margin: 0.75rem auto 0;" />
<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem; margin-bottom: 1rem;">Figure 6: The disk tier for DeepSeek-V4-Flash on a 16 GB GPU with 62 GB of RAM.</p>

Three lessons shaped it:

- **Never write back.** The host store keeps every expert, so an evicted slot is simply overwritten. The disk tier's first version kept each expert in exactly one tier, so every copy was a swap that sent the evicted expert back to host RAM; both directions share host DRAM (~62 GB/s together), and swaps took 57.7 of a 90.6 ms GPU step. Letting host rows keep their copy (the file always has every expert) made each move one-way: **+24.9 % decode** (13.75 tok/s against 11.01, the mean of two runs without it), at 18 % more disk reads. That was measured when the change landed, at K = 9; main picks K = 8 on this machine and decodes at 13.0 tok/s (section 2).
- **Keep the CPU out of the step, even for a disk read.** A planning kernel on the GPU marks which experts are only on disk; a host thread reads them while the layer's other copies run, and the captured graph waits at that layer until they are in their slots. On the PCIe 5.0 drive that beat replaying the step by 33 %.
- **Prefill reads the whole tier.** A prompt of a few hundred tokens already routes to nearly every expert of every layer, so one prefill reads almost the whole tier (~93 GB for one of these prompts on DeepSeek-V4), and TTFT is that read at disk speed (~11 GB/s here): 8.88 / 8.91 s for a natural ~1.5k-token prompt. Chunked prefill would re-read the tier for every chunk, so the recipe uses one chunk per prompt.

## 5. Try it

The smallest launch:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts
```

This keeps the KV cache for one request of the model's full 40,960-token context, which leaves K = 62 on a 16 GB card (218 tok/s, section 1); the log names the K it chose. On hybrid models such as Qwen3-Next, it also runs one request at a time unless `--max-running-requests` says otherwise. To trade context for resident experts, set the KV cache you need; these are the configurations of section 2:

```bash
# Qwen3-30B-A3B GPTQ-Int4 (K = 91 of 128 here): 241 tok/s
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --enable-paged-experts --paged-experts-num-resident auto \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton --disable-radix-cache

# Qwen3-30B-A3B FP8 (31 GB; K = 44 of 128): 113 tok/s
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --enable-paged-experts \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton --disable-radix-cache

# Qwen3-Next-80B-A3B NVFP4 (48 GB, experts in host RAM; K = 99 of 512): 116 tok/s
python -m sglang.launch_server --model-path nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --enable-paged-experts --moe-runner-backend marlin --disable-shared-experts-fusion \
  --kv-cache-dtype bf16 --disable-radix-cache \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --attention-backend triton

# gpt-oss-20b (MXFP4; K = 28 of 32): 173 tok/s
python -m sglang.launch_server --model-path openai/gpt-oss-20b \
  --enable-paged-experts --moe-runner-backend marlin \
  --context-length 4096 --max-running-requests 4 --max-total-tokens 10240 \
  --mem-fraction-static 0.88 --disable-radix-cache
```

Experts beyond host RAM go to the disk tier. The disk-tier rows of section 2, starting with DeepSeek-V4-Flash (137 GiB of experts) on the same 16 GB GPU with 62 GB of RAM:

```bash
SGLANG_OPT_FP8_WO_A_GEMM=1 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 45 \
  --moe-runner-backend marlin --mem-fraction-static 0.92 --max-running-requests 1 \
  --context-length 4096 --json-model-override-args '{"max_position_embeddings":4096}' \
  --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.5 --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill disabled --disable-radix-cache

# Qwen3.8-Flash-Next-FP8 (186 GB; experts and the 51 GB per-layer embedding table on the NVMe drive)
python -m sglang.launch_server --model-path Qwen/Qwen3.8-Flash-Next-FP8 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 36 \
  --ple-offload-embedding --ple-offload-backend file --ple-offload-dir /path/on/the/nvme/drive \
  --mem-fraction-static 0.90 --max-running-requests 1 --context-length 4096 \
  --chunked-prefill-size 4096 --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill disabled --disable-radix-cache

# Qwen3-Next-80B-A3B NVFP4 on the 8 GB laptop (54 GB of RAM under WSL2)
python -m sglang.launch_server --model-path nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --enable-paged-experts --paged-experts-enable-disk --paged-experts-host-memory-gb 18 \
  --moe-runner-backend marlin --disable-shared-experts-fusion --kv-cache-dtype bf16 \
  --attention-backend triton --disable-radix-cache \
  --context-length 4096 --max-running-requests 1 --max-total-tokens 4096 \
  --mem-fraction-static 0.88 --cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled
```

On Qwen3-Next, `--disable-radix-cache` gives the prefix cache's extra state memory to experts: with the prefix cache on, K drops from 99 to 85 (113 tok/s). `--attention-backend triton` is worth 3.6 % on Qwen3-30B GPTQ and makes no difference on the others.

`SGLANG_OPT_FP8_WO_A_GEMM=1` keeps DeepSeek-V4's `wo_a` in FP8 on SM120; one prefill chunk per prompt reads each expert from disk once (section 4.5).

Requirements: one GPU (no speculative decoding, no tensor or expert parallelism yet); for the disk tier an x86-64 host and safetensors checkpoints with one tensor per expert. Everything else is in the [Paged Experts docs](https://docs.sglang.io/docs/advanced_features/paged_experts).

## 6. How we measured

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
| SGLang | main at 1bad592d4c (2026-11-22) on the `lmsysorg/sglang` nightly-dev-cu13-20261121 image | same |


**Setup.** The commands of section 5: `--enable-paged-experts`, K chosen automatically, prefix caching off, `--mem-fraction-static` 0.88 on the RAM tier. Decode tok/s is the median over 8 ShareGPT prompts sent one at a time (256 output tokens each) of each request's tokens after the first divided by its time after the first token. GSM8K runs with thinking off, on all 1,319 questions for the RAM-tier rows and on 200 for the disk-tier rows.

Figure 5 traces with CUPTI the 256 captured decode steps of one request; the traced step is within 2–4 % of the untraced one.

Four rules, each here because breaking it once gave us a wrong number:

- **Repeat the first configuration last.** A difference smaller than the gap between its two runs (±0.2 % on GPTQ decode) is no difference.
- **Drop the page cache and discard warm-up** (16 ShareGPT requests). A cold expert pool reads 5–11 % low, and a full page cache once cut our throughput 2.6x without an error.
- **Natural prompts, never tiled ones.** SGLang's `random` dataset tiles a short ShareGPT turn into a long prompt; repeated text routes to fewer experts, and paged 2k TTFT read 261–265 ms instead of 327.
- **A ratio of two instruments is not a measurement.** A "copy defect" we chased for a while was a profiler-inflated kernel time divided by another tool's byte count; the kernel was running at link speed all along.

## Acknowledgments

TODO
