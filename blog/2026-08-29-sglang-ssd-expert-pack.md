---
title: Running DeepSeek-V4-Flash and Kimi-K3 on Consumer Hardware with SSD Expert Pack
author: SGLang Team
date: August 29, 2026
previewImg: /images/blog/sglang-ssd-expert-pack/expert-pack-layout.png
type: blog
---

# Running DeepSeek-V4-Flash and Kimi-K3 on Consumer Hardware with SSD Expert Pack

> SGLang brings the core idea of SSD-LLaMA to MoE inference: keep routed experts that do not fit in VRAM and host RAM on an NVMe SSD, load only the experts selected by the router, and use Expert Pack layout, direct I/O, pinned staging, asynchronous H2D transfers, and a GPU cache to turn SSD capacity into a practical backing tier.

## 1. Introduction: turning a VRAM problem into a storage problem

The total parameter capacity of DeepSeek-V4-Flash and Kimi-K3 is far beyond the VRAM of a single consumer GPU. A conventional deployment therefore needs multiple GPUs or hundreds of gigabytes, and sometimes terabytes, of host memory. That capacity requirement creates a large barrier between frontier model capability and local hardware.

SGLang's SSD-backed Expert Pack path takes a different approach. Routed expert weights remain on an NVMe SSD. The router activates only a small subset of experts for each token, so the runtime moves only the selected experts that are not already cached to the GPU. Expert Pack reorganizes the weights of each layer/expert pair into a directly addressable contiguous expert block. The runtime reads that expert block into an aligned pinned host buffer using direct I/O, then transfers it to a GPU cache asynchronously.

This path changes how model weights are stored and delivered, not the model computation. It does not prune, replace, merge, or skip selected experts, and it does not reduce Expert Top-K. The result is a practical way to run DeepSeek-V4-Flash and the validated text-only Kimi-K3 path with an Intel Ultra5 230F CPU, 32 GB memory, a TiPro9000 2 TB disk, and an RTX 5090 with 32 GB VRAM.

### MoE computation is sparse, but model capacity is not

Mixture-of-Experts models split the feed-forward network into many experts. After the router scores the experts for a token, only a small subset participates in that token's computation. The remaining experts are idle for that token.

The router's choice changes across tokens and prompts. The complete expert pool must therefore remain available even though only a small working set is active at any one time. Quantization reduces the artifact size, but it does not remove the need to store the expert pool. MoE inference consequently has two different properties:

- per-token computation is sparse;
- the total expert capacity that must be stored and delivered is very large.

This is why SSD is a useful backing tier. It provides much more capacity than consumer VRAM or RAM, and modern PCIe 5.0 NVMe SSDs provide enough sequential bandwidth to make a carefully designed delivery path viable. SSD capacity becomes executable model memory only when the layout, read path, and cache policy match expert-level access patterns.

### The capacity-cost difference

A capacity-cost comparison makes the trade-off clear. The following figures are capacity-only lower bounds, not complete system prices:

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/price.png" alt="Capacity cost comparison for DeepSeek-V4-Flash and Kimi-K3 on a logarithmic scale" width="460">
</p>

The figure does not mean SSD and DRAM have the same latency, or that buying an SSD alone is sufficient to run the model. It shows that placing the complete expert pool in VRAM or DRAM quickly becomes impractical, while using SSD for capacity and a bounded GPU cache for the active working set can substantially lower the hardware barrier.

### SGLang's SSD Expert Pack approach

The central SSD-LLaMA idea is to manage SSD, RAM, and VRAM as a runtime-controlled storage hierarchy. The complete expert pool stays in the high-capacity tier, while limited VRAM retains the experts with the highest observed reuse.

SGLang's Expert Pack is an implementation of the most important expert-centric parts of that idea inside the SGLang MoE runtime:

1. Expert Pack makes a layer/expert pair an independently addressable contiguous expert block.
2. `O_DIRECT` and aligned pinned buffers remove the extra page-cache staging copy.
3. A byte-budgeted LFU/LRU GPU cache retains complete experts that are reused.

The current SGLang path does not claim to reproduce every mechanism in the SSD-LLaMA paper. SGLang's implementation is GPU-centric: pinned host memory is a bounded transfer staging area, not a persistent host expert cache, and the current feature does not require the paper's CPU expert execution or lossless CUDA decompression. Keeping this distinction explicit makes the feature boundary precise.

## 2. Why the native GGUF loading path is not enough

With the original GGUF or multi-shard tensor layout, the gate, up, and down weights of one expert may be located in different file regions. One router hit can therefore trigger several small reads, tensor-name lookups, and staging operations.

An explicit on-demand read avoids speculative prefetch, but exposes the full SSD and H2D latency on the critical path of the current MoE layer. After the router produces its result, the GPU must wait for the selected experts. Prefetching can hide part of that latency, but it has two fundamental limitations:

- the correct expert may still arrive too late because the routing result is only known after the previous computation;
- an incorrect prediction consumes SSD bandwidth, staging space, and GPU cache capacity, after which the actually selected expert must still be read.

SGLang therefore first changes the physical expert layout, then reduces the cost of every unavoidable cache miss.

### Why native GGUF cannot directly use expert-level `O_DIRECT`

Native GGUF is a tensor-oriented model container, not an expert-oriented direct-I/O store. Its metadata and tensor payloads are organized around individual tensors, and a single expert's `gate`, `up`, and `down` weights may be separated across file regions or across multiple shards. The original loading path commonly uses a parser, `mmap`, or buffered file reads, so the application sees pageable page-cache-backed mappings rather than a preallocated aligned DMA destination.

`O_DIRECT` requires all of the following to be controlled by the caller:

- a file offset aligned to the storage and filesystem contract;
- a read length aligned to that contract;
- a user-provided buffer whose address is also aligned and suitable for the read.

An arbitrary tensor slice in a native GGUF file does not provide that expert-level contract. Its offset may not be aligned, its length may not be a multiple of the required block size, and the three tensors needed for one expert are not guaranteed to form one contiguous range. A caller could issue separate aligned reads with padding and then reconstruct the expert in another buffer, but that gives up the main benefit: it reintroduces multiple reads and extra assembly work, while the original mmap/page-cache path still cannot use the page-cache pages themselves as an `O_DIRECT` destination.

Expert Pack is the offline transformation that makes direct I/O practical. It places all roles of one layer/expert pair into one padded, aligned expert block, records its exact offset and length in the manifest, and provides a pinned buffer whose address satisfies the same contract. The runtime can then read and transfer a complete expert without asking the native GGUF layout to behave like a direct-I/O layout.

## 3. Expert Pack: organizing weights by expert

### Contiguous expert layout

SGLang Expert Pack v1 treats one `(layer, expert)` pair as one complete expert. A DeepSeek expert contains the `gate`, `up`, and `down` roles. The manifest records each role's tensor boundaries, format, integrity information, and pack offset.

The logical layout is:

![Expert Pack physical data block layout with contiguous expert byte-streams and explicit block-aligned padding](/images/blog/sglang-ssd-expert-pack/expert-pack-layout.png)

The runtime does not scan the file for tensor names. It derives the expert offset from the pack metadata:

```text
expert_offset = data_start
              + (layer * num_experts + expert) * expert_stride
```

Role offsets inside the expert block are validated as well. One manifest lookup can therefore resolve the complete expert read range. The runtime may split that range into a bounded number of parallel tasks according to `read_splits`.

The layout changes the physical organization of weights on SSD, not their tensor contents, quantization formats, routing decisions, or model mathematics. Kimi-K3 uses a separate GGML Expert Pack adapter. The currently validated input consists of 38 Q2_K GGUF shards; its routed experts use Q2_K for gate/up and Q3_K for down.

### Alignment is required for direct I/O

Direct I/O cannot use arbitrary file offsets, lengths, and user buffers in the same way as ordinary `read()`. The SGLang runtime validates:

- Expert Pack offsets for each expert;
- the start and length of every read range;
- the address of every pinned staging buffer.

The current implementation checks 4096-byte alignment. If the pack or staging buffers do not satisfy the contract, initialization fails instead of silently falling back to an uncontrolled path during inference.

## 4. The key optimization: removing the `page cache -> pinned memory` copy

This is one of the most important differences between Expert Pack and a conventional file-reading path.

### Traditional buffered I/O

Traditional file reads normally go through the operating system page cache:

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/traditional_copy.png" alt="Traditional file-read path: a synchronous page-cache-to-pinned-memory copy followed by asynchronous H2D" width="300">
</p>

The page cache is a kernel-managed file cache. It is not the same thing as the page-locked user memory that CUDA can use for asynchronous H2D. To issue an asynchronous H2D transfer, the application normally prepares a pinned buffer. The file data therefore has to be copied from the page cache into that pinned buffer before the GPU transfer can start. From the application's perspective, this page-cache-to-pinned handoff is a synchronous CPU memory copy: the host-side staging step must complete before the H2D operation has a valid pinned source buffer. It is not itself a `cudaMemcpyAsync` operation.

This is neither SSD reads nor H2D transfers. Rather, it is an extra synchronous memory copy operation performed by the CPU on the host side: it reads the payload from page-cache-backed memory and writes it into a pinned staging buffer. For a large expert, this translates to a read and write of the full expert size, consuming host memory bandwidth and introducing an additional kernel-to-userland staging handoff before the GPU transfer can proceed.

### Expert Pack with direct I/O

When `direct_io=True`, SGLang opens the Expert Pack with `O_DIRECT` and makes the read target a preallocated, aligned pinned staging buffer:

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/expert_pack_copy.png" alt="Expert Pack direct-I/O path: an aligned pinned host buffer feeds the GPU expert cache before MoE computation" width="300">
</p>

The read target is already the pinned buffer required by CUDA, so the intermediate step below is removed:

```text
page cache -> pinned memory
```

This is not a faster implementation of that copy. The copy is removed from the data path. A simplified cost model is:

```text
Traditional path:
T = T(SSD -> page cache)
  + T(page cache -> pinned)
  + T(pinned -> GPU)
  + T(sync)

Expert Pack direct I/O:
T = T(SSD -> pinned)
  + T(pinned -> GPU)
```

`O_DIRECT` does not make the SSD's physical bandwidth increase. It removes one full host-memory traversal from the end-to-end path, which can provide these benefits:

- one less host-memory read and write, reducing CPU and memory-bandwidth pressure;
- one less synchronization handoff between kernel page cache and user-space staging;
- no large expert payload polluting the page cache and competing with unrelated data;
- a completed expert block can enter the H2D path without a page-cache staging copy;
- the same expert-level contract is reused for every layer/expert pair.

The page-cache-copy claim applies only to `direct_io=True`. The current SGLang Expert Pack loader and the DeepSeek/Kimi 5090 launchers enable this option by default. If direct I/O is explicitly disabled, the path may go through the page cache and an additional staging copy again.

## 5. GPU/VRAM cache: keeping the working set close to compute

The GPU cache is the mechanism that turns repeated expert access into a local VRAM hit. Expert Pack is not a cache of individual tensor fragments: one cache entry contains the complete `gate`/`up`/`down` data for one `(layer, expert)` pair. Keeping the complete expert together matters because a selected expert needs all of its roles for computation. Caching only one role would still force the other roles to be read and would not remove the cache-miss cost.

The cache is byte-budgeted rather than expert-count-budgeted. Since different models and adapters have different per-expert payload sizes, the runtime derives the number of available slots from the usable VRAM budget:

```text
usable_vram = min(requested_cache, free_vram - reserve)
slot_count  = floor(usable_vram / expert_payload_bytes)
```

The reserve protects memory needed by the model, CUDA runtime, activations, and other non-cache allocations. Initialization fails if the resulting slot count cannot hold one complete top-k working set. This makes the cache contract explicit: a cache budget is not allowed to consume the memory required for the current MoE computation.

On a cache hit, the runtime reuses the resident expert and does not read the Expert Pack or issue an H2D for that expert. If a previous transfer is still pending, a CUDA event protects the consumer from observing a partially installed slot. On a cache miss, the runtime selects a victim slot, reads the complete expert into a reusable pinned staging buffer, copies the expert into the GPU slot, and publishes the slot only after the transfer event is ready.

Replacement combines frequency and recency. The runtime records how often each `(layer, expert)` is selected and when it was last used. A frequently selected expert is harder to evict than a cold expert; among similarly useful entries, an older entry is a better victim. Active experts for the current top-k request are protected from eviction, so the cache cannot evict the working set it is about to execute.

The source Expert Pack is immutable. Evicting a GPU entry therefore requires no write-back: the expert can always be reconstructed from its recorded SSD offset. This makes VRAM cache management simpler than a dirty data cache and keeps replacement focused on reuse value rather than persistence.

The runtime exposes counters that make cache behavior measurable:

- `cache_hits` and `cache_misses`;
- `cache_evictions`;
- `pack_read_bytes`;
- `h2d_bytes`;
- `fallback_count` and `io_errors`.

These counters distinguish a cache problem from an I/O problem. A low hit rate means the VRAM budget or workload locality is insufficient; high `pack_read_bytes` and `h2d_bytes` with a good hit rate may instead indicate that the active set is larger than the cache during a particular phase. `io_errors` reports observed I/O failures, while `fallback_count` is diagnostic telemetry whose meaning depends on an instrumented fallback path; neither is a cache-performance metric.

The current execution order places an expert-cache miss on the critical path for the MoE computation it feeds. `acquire()` copies routing IDs to CPU, waits for the SSD read futures, enqueues the expert-level H2D transfers, and makes the current CUDA stream wait for their transfer events. Only after those events are ready does `apply()` launch the MoE kernels. Reads and transfers for different missing experts may overlap during the delivery phase, but the current path does not overlap that delivery with the MoE computation that consumes the experts.

A simplified per-step model for the current path is therefore:

```text
T_step ~= T(miss delivery) + T(GPU compute)
```

Here, `T(miss delivery)` includes route-ID preparation, SSD reads, staging, H2D submission, and the waits needed to make the selected experts available. On a GPU cache hit, the SSD-read and H2D portions can be skipped. A cross-step pipeline could change this model, but that is not part of the execution path described here. The actual result depends on SSD bandwidth, access distribution, cache hit rate, staging-slot count, and expert shapes.

## 6. DeepSeek-V4-Flash and Kimi-K3 integration

### 6.1 Explicit opt-in loading

Expert Pack does not replace ordinary model loading. The `ExpertPackModelLoader` is selected only when the server is started with:

```bash
--load-format expert_pack
```

The default `auto`, `safetensors`, and `gguf` paths remain unchanged. Installing this feature therefore does not route ordinary model users into the SSD runtime by accident.

### 6.2 DeepSeek-V4-Flash

The currently validated input is an MXFP4 GGUF. Routed expert matrices use MXFP4. Non-routed weights are provided by the GGUF iterator, while the Expert Pack runtime handles dynamic routed-expert delivery.

With the GGUF and runtime environment prepared, use the 5090 launcher:

```bash
python3 -m pip install -e 'python'

python3 examples/runtime/deepseek_v4/benchmark_deepseek_5090.py \
  --gguf /path/to/deepseek-v4-flash-0731-mxfp4.gguf \
  --prompt 'Explain why the sky appears blue.' \
  --max-new-tokens 200
```

The launcher uses `--load-format expert_pack`, prepares or validates model metadata, Expert Pack, and manifest artifacts, and passes runtime settings through `--model-loader-extra-config`. The first launch prepares the pack; later launches reuse the existing artifacts.

### 6.3 Kimi-K3

The currently validated input is the text-only 38-shard GGUF path, not the complete multimodal safetensors checkpoint. Point the launcher at the first shard; it discovers the remaining shards in the same directory:

```bash
python3 examples/runtime/kimi_k3/benchmark_kimi_k3_5090.py \
  --gguf /path/to/kimi-k3/KIMI-K3-MXP4-DERISKED-Q2_K-00001-of-00038.gguf \
  --prompt 'Explain why the sky appears blue.' \
  --max-new-tokens 200
```

Kimi uses the GGML Expert Pack adapter. The validated routed experts use Q2_K for gate/up and Q3_K for down. Other GGUF quantization variants should not be claimed as supported without additional validation.

### 6.4 Model and Expert Pack weight footprint

The validated original model weights and generated Expert Pack files occupy:

| Model | Original weight files | Weight size | Expert Pack size |
| --- | --- | ---: | ---: |
| DeepSeek-V4-Flash | One Ollama MXFP4 GGUF blob | 155.10 GB | 147.18 GB |
| Kimi-K3 | 38 Q2_K GGUF shards | 1009.51 GB (about 1.01 TB) | 985.61 GB |

These are file sizes for the validated weight payloads and generated Expert Pack payloads. Expert Pack indexes, locks, and other metadata are separate and are not included in this table.

## 7. Correctness boundary

Expert Pack is a weight-layout and delivery optimization, not an approximate-inference algorithm. The correctness contract is:

- every expert selected by the router is executed;
- Expert Top-K is unchanged;
- a selected expert is not replaced by a different resident expert;
- selected experts are not pruned, skipped, or merged;
- the pack and manifest are structurally, dimensionally, and cryptographically validated as configured;
- `fallback_count` and `io_errors` are reported instead of silently hiding I/O failures.

Validation showed that DeepSeek-V4-Flash produced semantically equivalent answers to Ollama across multiple prompt categories. Kimi-K3 matched the 200-token SGLang reference output; all 92 routed layers executed Top-16 experts with `io_errors=0`. `fallback_count` is retained as diagnostic telemetry, but the current path does not expose an instrumented increment for every hypothetical fallback, so zero is not used as an independent correctness proof. Correctness is instead established by the route/output audit and the structural pack checks.

## 8. Performance results

All SGLang, Ollama, and llama.cpp measurements use the test environment listed in the reproduction table below. The figures describe validation results under this hardware condition; token counts and runtime-specific software settings remain as stated in each comparison. The earlier summary tables are retired; the figures below are now the canonical presentation of the token-rate comparison.

### DeepSeek-V4-Flash vs. Ollama

The comparison uses ten shared requests: five Alpaca and five MMLU. Both runtimes generated up to 200 tokens per request. The chart reports mean prefill and decode token rates for each dataset.

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/deepseek_v4_flash_sglang_vs_ollama_compare.png" alt="DeepSeek-V4-Flash SGLang versus Ollama prefill and decode token rates for Alpaca and MMLU" width="100%">
</p>

Relative to Ollama, SGLang improves prefill by 2.28x on Alpaca and 3.39x on MMLU. Decode improves by 6.92x and 6.55x, respectively.

### Kimi-K3 vs. llama.cpp

The comparison uses the same ten fixed requests in both runtimes: five Alpaca
and five MMLU. Both clients use temperature 0 and default EOS handling; each
request generated exactly 200 completion tokens, so the decode comparison is
now matched for prompt set, stop behavior, and output length. The chart reports
mean prefill and decode token rates for each dataset.

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/kimi_k3_sglang_vs_llama_compare.png?v=20260904" alt="Kimi-K3 SGLang versus llama.cpp prefill and decode token rates for a matched 200-token Alpaca and MMLU rerun" width="100%">
</p>

Relative to llama.cpp, SGLang improves prefill by 6.96x on Alpaca and 5.80x on
MMLU. Decode improves by 3.30x and 3.52x, respectively. The ten-request
aggregates use the ten retained request records described below.

### Benchmark reproduction record

The retained benchmark bundle uses one request at a time.

| Component | Test configuration |
| --- | --- |
| CPU | Intel Ultra5 230F |
| Memory | 32 GB |
| Disk | TiPro9000 2 TB |
| GPU | NVIDIA RTX 5090 32 GB |
| Requests | 10 fixed prompts: 5 Alpaca + 5 MMLU |
| Generation | Temperature 0, default EOS, 200-token target |

The ten prompts, in request order:

| # | Sample ID | Dataset | Prompt |
| ---: | --- | --- | --- |
| 1 | `alpaca-37246` | Alpaca | Summarize the movie "Toy Story" |
| 2 | `mmlu-abstract_algebra-14` | MMLU | Answer the multiple-choice question. Select the correct option and briefly explain your answer.<br><br>Question: Find the maximum possible order for an element of S_n for n = 10.<br>A. 6<br>B. 12<br>C. 30<br>D. 105 |
| 3 | `alpaca-50812` | Alpaca | Given a list of items, suggest an interesting activity.<br><br>Input: pencils, paper, markers |
| 4 | `mmlu-moral_disputes-8315` | MMLU | Answer the multiple-choice question. Select the correct option and briefly explain your answer.<br><br>Question: According to Hardin, the "ratchet effect" refers to the fact that<br>A. overpopulation does not affect the number of people who are poor.<br>B. overpopulation leads to creation of food banks that help curb poverty rates.<br>C. world hunger and poverty leads to recognition of rights not to be hungry.<br>D. the use of a world food bank to feed the hungry leads to an escalating series of emergency situations. |
| 5 | `alpaca-9907` | Alpaca | Translate this phrase from Spanish to English: El sol no brilla hoy. |
| 6 | `mmlu-high_school_macroeconomics-3940` | MMLU | Answer the multiple-choice question. Select the correct option and briefly explain your answer.<br><br>Question: The crowding-out effect from government borrowing is best described as<br>A. the rightward shift in AD in response to the decreasing interest rates from contractionary fiscal policy.<br>B. the leftward shift in AD in response to the rising interest rates from expansionary fiscal policy.<br>C. the effect of the President increasing the money supply which decreases real interest rates and increases AD.<br>D. the effect on the economy of hearing the chairperson of the central bank say that he or she believes that the economy is in a recession. |
| 7 | `alpaca-40699` | Alpaca | Give an example of a bias that could exist in an AI algorithm. |
| 8 | `mmlu-professional_law-10971` | MMLU | Answer the multiple-choice question. Select the correct option and briefly explain your answer.<br><br>Question: Homeowner owns a property in its natural condition with a house on it. There was no fill of any kind on the property. Neighbor, who owns the adjacent property to the East, built a driveway whose western boundary is along the border of homeowner's property. The excavator dug the driveway five feet deep. The land began to subside along the line of excavation and about three feet of homeowner's land fell off into the driveway, making that part of her property useless. Homeowner demanded that neighbor fill in the property to buttress the erosion created. That was not done and the erosion continued to occur. Homeowner sued and asked for an injunction compelling the neighbor to build and maintain a retaining wall. Will the court rule for the plaintiff/homeowner?<br>A. Yes, because excavation is an abnormally dangerous activity and neighbor is absolutely liable for any damages caused by the violation.<br>B. Yes, because every landowner has a right to the lateral support of the soil in its natural state.<br>C. No, because the neighbor did not go onto the adjacent land and confined all excavation to his own land.<br>D. No, the right to lateral support is a common law right that has been abrogated by statute in virtually all states so that the right no longer exists. |
| 9 | `alpaca-34440` | Alpaca | Make a menu item for a restaurant that contains the following ingredients.<br><br>Input: Salmon, avocado, spinach |
| 10 | `mmlu-jurisprudence-6660` | MMLU | Answer the multiple-choice question. Select the correct option and briefly explain your answer.<br><br>Question: Which of the following is the strongest argument against ethical relativism's hostility to human rights?<br>A. Utilitarianism<br>B. Communitarianism.<br>C. Cognitivism.<br>D. Positivism. |

Each runtime used temperature 0, default EOS handling, and a 200-token target
for the matched Kimi chart. Prompt-token counts and actual completion-token
counts are retained per request in the JSONL records. The Kimi records all
reached 200 completion tokens; the retained DeepSeek Ollama records include
earlier EOS stops and therefore report their actual counts separately from the
200-token ceiling.

| Runtime | Source revision and identity |
| --- | --- |
| SGLang | `81c9f837f19ff8dfe1a9fcd1abfc6069dd28d2ec` (`support_deepseek-v4_and_kimi-k3_on_ssd`) |
| llama.cpp | `5fff128451d7603857597ee1fc18ac1dfb90f148`; local `src/models/kimi-k3.cpp` was modified for Kimi-K3 |
| Ollama | Ollama `0.33.1`; managed llama.cpp runner commit `d222767c7` |

The Kimi source is the 38-shard GGUF beginning at
`/models/kimi-k3-blackfrost-q2k/KIMI-K3-MXP4-DERISKED-Q2_K-00001-of-00038.gguf`.
The Expert Pack is
`/models/kimi-k3-blackfrost-q2k/KIMI-K3-MXP4-DERISKED-Q2_K.expert-major.pack`.
The retained Kimi manifest provides pack-index and source-inventory hashes, but
full SHA-256 hashes for all GGUF shards and the complete pack were not recorded.

| Available artifact digest | Value |
| --- | --- |
| DeepSeek Ollama manifest | `sha256:882b1398c0ca4e7ec8ca0a501fd8c4372f780f690536a3ec17ffc75306569ed3` |
| DeepSeek GGUF blob | `sha256:947ac34c08c0e5c5752ac76398f934b3b6b4075cfe915ba43dd5ac754900a4cd` |
| Kimi Expert Pack index | `ceb3e63ac411cce02ffdec875e5ae05f61c3dea351d0c91d86d712544b0288aa` |
| Kimi source inventory | `e7e2caab78a1da736fe9d17b8754b682498f6c430531054c109c0f624a0ab89b` |

The official SGLang server entry point for Expert Pack mode is:

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/kimi-k3-tokenizer \
  --tokenizer-path /path/to/kimi-k3-tokenizer \
  --trust-remote-code \
  --load-format expert_pack \
  --model-loader-extra-config '{"pack_path":"/path/to/KIMI-K3.expert-major.pack","manifest_path":"/path/to/kimi-k3-expert-pack.manifest.json","cache_vram_mib":5120,"cache_vram_reserve_mib":1536,"stage_slots":16,"read_splits":4,"direct_io":true}' \
  --tp-size 1 --ep-size 1 \
  --disable-cuda-graph --disable-shared-experts-fusion \
  --disable-radix-cache --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule --skip-server-warmup \
  --max-running-requests 1 --mem-fraction-static 0.98 \
  --chunked-prefill-size 64 \
  --host 0.0.0.0 --port 30000
```

`--model-path` points to the verified tokenizer/configuration directory; the
Expert Pack and manifest are supplied through
`--model-loader-extra-config`. The source GGUF shards remain next to the pack
as described by the manifest.

The llama.cpp server was started with:

```bash
/root/workspace/llama.cpp/build/bin/llama-server \
  -m /models/kimi-k3-blackfrost-q2k/KIMI-K3-MXP4-DERISKED-Q2_K-00001-of-00038.gguf \
  -ngl -1 --cpu-moe --host 127.0.0.1 --port 8081 \
  -t 16 -tb 16 --threads-http 16 -np 1 -c 4096 -b 16 -ub 16 \
  --no-warmup --metrics \
  --log-file /root/workspace/kimi-k3-llama-cpp-16cpu-32gb-20260828/server.log
```

The retained DeepSeek Ollama service was started with
`OLLAMA_HOST=http://127.0.0.1:11435 /usr/local/bin/ollama serve`. Its managed
runner command was:

```bash
/usr/local/lib/ollama/llama-server \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-947ac34c08c0e5c5752ac76398f934b3b6b4075cfe915ba43dd5ac754900a4cd \
  --port 40115 --host 127.0.0.1 --no-webui --offline \
  -c 32768 -np 1 --log-verbosity 4 --no-log-prefix --no-log-timestamps \
  --flash-attn auto -b 512 -ub 512 --context-shift --keep 4
```

The service log identifies Ollama `0.33.1` and runner commit `d222767c7`.

The llama.cpp benchmark client sent non-streaming `/completion` requests with
`cache_prompt=false`, `temperature=0`, and `n_predict=200`, one request at a
time.

The exact per-request JSONL records and effective launch logs are retained in
the benchmark evidence bundle.

### Expert-cache hit rate and SSD traffic

Token rate should be read together with cache telemetry. The following Python-generated bar charts report two independent quantities: the share of unique `(layer, expert)` keys counted by `acquire()` that were served from the VRAM cache, and decimal gigabytes read from the Expert Pack per generated token. Repeated token-to-expert routes within one acquire/update are de-duplicated, so this is not a per-token router-edge hit rate. A cache hit avoids the SSD read and H2D transfer for that expert. `pack_read_bytes` includes both prefill and decode traffic; GB/token is normalized by generated completion tokens and is not decode-only traffic.
Both figures show SGLang-only telemetry, so the single series is labeled by the surrounding text rather than a legend.

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/deepseek_v4_flash_expert_cache_metrics.png" alt="DeepSeek-V4-Flash SGLang VRAM cache hit rate and SSD reads per generated token" width="100%">
</p>

DeepSeek uses the complete ten-request run: five Alpaca and five MMLU requests, each with a 200-token completion. Its hit rate is 54.2% for Alpaca and 46.4% for MMLU, with 1.66 and 2.04 GB read per generated token.

<p align="center">
  <img src="/images/blog/sglang-ssd-expert-pack/kimi_k3_expert_cache_metrics.png" alt="Kimi-K3 SGLang VRAM cache hit rate and SSD reads per generated token" width="100%">
</p>

The Kimi figure uses all ten completed requests: five Alpaca and five MMLU, each with a 200-token completion. The unweighted per-request means are 17.0% and 22.7% VRAM cache hit rate, with 22.10 and 27.63 decimal GB read per generated token, respectively. The difference from DeepSeek is expected: Kimi's configuration reserves 5 GiB for the GPU expert cache, while the DeepSeek run reserves about 21 GiB, and the two adapters have different expert payload sizes and routing behavior.

<!-- Retired interim three-request values are kept below only as review history. -->
<!--
The Kimi figure uses the first three completed requests of the interleaved run: two Alpaca and one MMLU, each with a 200-token completion. The current means are 17.4% and 18.5% VRAM cache hit rate, with 21.84 and 24.46 GB read per generated token, respectively. The remaining seven requests are still running, so these are explicitly stage results rather than final ten-request averages. The difference from DeepSeek is expected: Kimi’s configuration reserves 5 GiB for the GPU expert cache, while the DeepSeek run reserves about 21 GiB, and the two adapters have different expert payload sizes and routing behavior.

-->

The improvement does not come from one isolated faster-copy primitive. It is the combined effect of:

1. Expert Pack turning scattered tensor accesses into addressable contiguous expert reads;
2. direct I/O removing the `page cache -> pinned memory` copy;
3. pinned staging provides a CUDA-compatible host source for expert-level H2D without page-cache staging;
4. the GPU cache skipping SSD reads and H2D transfers on a hit.

## 9. Conditions and limitations

### SSD capacity and preparation time

Expert Pack requires additional SSD capacity. The PR records an estimated 5-10 minutes to build the DeepSeek-V4-Flash pack and approximately 8-15 minutes for first-run readiness. Kimi-K3 pack construction took 29 minutes 42 seconds in the retained measurement, with approximately 35-45 minutes to first readiness. The 38 Kimi source shards and generated Expert Pack occupy about 1.814 TiB in total; the measurements use the TiPro9000 2 TB disk, while a 4 TB SSD is recommended for practical deployment headroom.

### Direct I/O

`O_DIRECT` requires platform support and alignment of file offsets, read lengths, and user-buffer addresses. SGLang fails closed during runtime initialization. If the platform does not support direct I/O, or the pack does not satisfy the alignment contract, the result should not be described as direct-I/O performance.

### Cache and workload

- A smaller GPU cache creates more misses, putting SSD reads and H2D transfers on the critical path more often.
- A change in prompt or workload distribution can change the hot experts, so one request's hot set is not guaranteed to fit every workload.
- If the complete expert pool already fits in GPU memory, Expert Pack adds an unnecessary data path and is not the right deployment mode.
- If the workload is dominated by GPU computation, the benefit of removing the host copy may be hidden by compute time.
- If SSD random-read behavior, queueing, or thermal stability is poor, increasing `read_splits` and staging slots may add queueing and memory pressure instead of throughput.

### Current feature boundary

This feature focuses on routed-expert SSD delivery and GPU caching. It does not provide SSD KV-cache offload and does not change request scheduling. It is an explicit opt-in Expert Pack path, not a global replacement for every SGLang model-loading format.

## 10. Conclusion

SSD Expert Pack is not simply replacing GPU memory with a slower disk. It redesigns weight delivery around the sparse access pattern of MoE inference:

```text
router selects a small set of experts
  -> Expert Pack resolves their offsets
  -> O_DIRECT reads into aligned pinned buffers
  -> expert-level asynchronous H2D fills the GPU cache
  -> the complete expert becomes available for computation
```

Removing the `page cache -> pinned memory` copy is an easy detail to miss, but it is a concrete end-to-end optimization. A traditional path reads into the OS page cache and then copies the payload into CUDA-usable pinned memory. With direct I/O, Expert Pack uses the preallocated pinned buffer as the read target and removes that full host-memory movement and synchronization handoff.

This is how SGLang applies the core SSD-LLaMA idea to real DeepSeek-V4-Flash and Kimi-K3 integrations: the complete expert pool remains on high-capacity SSD, a bounded GPU cache retains the current working set, and the runtime moves only the experts selected by the router. Very large MoE models no longer require stacking enough VRAM or DRAM to hold the complete model and can instead run on a consumer GPU paired with a high-speed NVMe SSD.
