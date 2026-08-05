---
title: "Fast Engine Recovery: Sub-Second Engine Restart for SGLang via Weight Cache Daemon"
author: "Ant Ling Infra Team (Ant Group), Alibaba, SGLang Team"
date: "July 28, 2026"
previewImg: /images/blog/sglang-fast-recovery/preview.png
---

## TL;DR

Nowadays, SOTA models are getting much bigger and reloading the model service after a crash is very expensive. Therefore, we introduce the **Weight Cache Daemon**, a persistent GPU process that holds post-quantized model weights in GPU memory and serves them to new SGLang engine instances via CUDA IPC zero-copy mapping. This reduces weight loading from minutes to sub-second times.

The Weight Cache Daemon is the first phase of our **Fast Engine Recovery Framework**, which targets **< 10 second cold restarts** and **< 1 second warm standby switches** for production LLM serving.

Key results:

1. **Weight loading: ~495s → ~0.63s** — a **~785× speedup**, eliminating 93.2% of startup time based on the Ling-2.6-1T FP8 model.
2. **Total startup: 8.8min → 1.3min** — an **80% reduction** in end-to-end engine boot time.
3. **Multi-instance weight sharing** — multiple engine instances on the same GPU map to the same IPC handles, eliminating redundant disk I/O and post-quantization transforms.
4. **Active-standby failover in < 1 second** — standby engines share weights via zero-copy, enabling near-zero-downtime failover without dedicating full GPUs to idle replicas.
5. **Multi-node-instance weight sharing** - support multi-node mode for large models

## Background

As LLM models grow larger — Qwen3-235B, Ling-2.6-1T, and the newly released 2.8T Kimi K3 — the cold-start time of serving engines has become a critical bottleneck for production efficiency. A Ling-2.6-1T FP8 instance on 8×H20-3e GPUs takes **~8.52 minutes** just to become ready to serve, weights stay in 3.5T NVME SSD. In production, this means:

- **P99 tail latency spikes** during restarts — all in-flight requests fail or queue indefinitely.
- **Reduced availability** — multi-minute recovery windows violate SLA targets.
- **Operational friction** — rolling updates, config changes, and failure recovery are all bottlenecked by the restart cycle.
- **GPU resource waste** — traditional active-standby deployments dedicate a full set of GPUs to idle replicas, doubling hardware cost for failover.

Where does the time go? We profiled a complete SGLang engine startup for Ling-2.6-1T FP8:

| Phase | Time (s) | Percentage | Notes |
|-------|----------|------------|-------|
| Pre-init & ServerArgs | ~1 | 0.2% | Pre-init and ServerArgs parsing |
| Tokenizer init | ~13 | 2.4% | load and init tokenizer |
| Init torch distributed | ~5 | 0.9% | NCCL 2.28.9,8 卡 H20,NVLink mesh 370.8 GB/s,P2P/IPC;slowest rank TP1=5.19s |
| Load weight (disk) | ~495 | 93.2% | 161 shard,W8A8 FP8 (CompressedTensorsW8A8Fp8MoE),slowest rank=495.3s, 120GB per card; Disk I/O bound |
| Cache allocation (KV+Mamba) | ~1 | 0.2% | KV:553,599 tokens/5.94GB bf16;Mamba SSM state:5.33GB,max_mamba_cache_size=155 |
| Capture CUDA graph | ~7.7 | 1.5% | only 3 decode BS [1,2,4] |
| Server ready | ~4 | 0.8% | Unified RadixTree init, HTTP/uvicorn startup, warmup requests |
| **Total** | **~531** | | **~8.8 minutes** 

The bottleneck is clear: **weight loading from disk accounts for 93.2% of startup time**. For Ling-2.6-1T FP8 model, each TP rank reads ~120GB of safetensors from disk, deserializes, applies TP sharding, and runs post-quantization transforms (FP8 quantization, weight repacking). This work is **repeated identically on every restart**, even though the resulting GPU tensors are deterministic and often already present in GPU memory.

Can we avoid reloading from disk every time? The answer is **yes** — by keeping weights in GPU memory across engine restarts.

## Design

### Core Idea: Persistent Weight Cache via CUDA IPC

The Weight Cache Daemon is a persistent GPU process that holds post-quantized, TP-sharded weights in GPU memory. On engine restart, the new engine process maps weights from the daemon via **CUDA IPC zero-copy** — no disk I/O, no deserialization, no quantization.

```
┌─ GPU i ────────────────────────────────────────────────────┐
│                                                            │
│  ┌───────────────────┐    cudaIpcMemHandle     ┌─────────┐ │
│  │ Weight Cache      │ ──────────────────────► │ Engine  │ │
│  │ Daemon (rank i)   │    (zero-copy)          │ Rank i  │ │
│  │                   │                         │         │ │
│  │ Holds:            │                         │         │ │
│  │ • TP-sharded      │                         │         │ │
│  │   weights (fp8)   │                         │         │ │
│  │ • weight_scale    │                         │         │ │
│  │ • workspace       │                         │         │ │
│  │ • all post-quant  │                         │         │ │
│  │   params/buffers  │                         │         │ │
│  └───────────────────┘                         └─────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘

Coordination: Unix Socket /tmp/sglang_weight_cache_rank{i}.sock
(global rank = tp_size × pp_rank + tp_rank, unique across PP stages and nodes)
```

Each GPU runs **one daemon process** for its TP rank. The daemon:

1. Loads model weights from disk (full pipeline: disk → TP shard → quantize → repack).
2. Exports every parameter and buffer in `model.state_dict()` as CUDA IPC handles.
3. Records a `CacheConfig` fingerprint (model path, TP/DP size, quant config hash, dtype).
4. Serves IPC handles over a Unix socket to requesting engine processes.

The engine connects to the daemon, validates config compatibility, and maps weights directly into its address space — the engine and daemon **share the same physical GPU memory** via CUDA IPC.

### Zero-Copy Loading via Meta Device

The key to sub-second loading is **zero-copy**: the engine's `param.data` pointer is set directly to the IPC-mapped GPU tensor. No data is copied.

To achieve this, the engine initializes the model on the **meta device** (no GPU/CPU memory allocation), then replaces each parameter's data pointer with the IPC-mapped tensor.

Post-quantization parameters (e.g., `weight_scale` from FP8 quantization) that were created by `process_weights_after_loading()` are also cached by the daemon and mapped directly — no re-quantization needed.

### Config Validation: Safety First

Any mismatch between the engine's config and the daemon's cached config triggers a **full disk reload**, ensuring correctness:

| Field | Mismatch Example | Consequence |
|-------|-----------------|-------------|
| `model_path` + `model_arch` + `revision` | Different model or revision | Wrong weights entirely |
| `tp_size` + `tp_rank` | Different TP sharding | Wrong shard for this rank |
| `pp_size` + `pp_rank` | Different PP partitioning | Wrong layers for this pipeline stage |
| `dp_size` + `ep_size` | Different DP/EP strategy | Incorrect weight distribution |
| `quant_method` + `quant_config_hash` | Different quantization | Unquantized vs FP8 mismatch |
| `dtype` | float16 vs bfloat16 | Type mismatch |
| `device_capability` + `torch_version` | Different GPU arch or torch version | Weights map cleanly but serve wrong numerics |

The last two fields form an **environment stamp**: a daemon and a client that ran different post-processing branches (different compute capability or torch/kernel version) can produce weights that map cleanly over IPC yet serve garbage — stamping the environment into `CacheConfig` turns that into a clean mismatch.

This is critical for production safety: if an operator changes the model or quantization config, the engine will detect the mismatch and fall back to disk loading rather than mapping incompatible weights.

On top of config validation, quantization methods are gated by an **IPC allowlist**. CUDA IPC zero-copy exports only raw tensor data, so it is correct only when the entire effect of `process_weights_after_loading()` is captured by that data. Methods that stamp Python-side metadata or repack/transpose weights (per-tensor FP8, Marlin, AWQ/GPTQ) would silently serve wrong numerics — they raise a hard error instead. Currently verified: **unquantized** and **block-wise FP8** (`weight_block_size` set); more methods will be added after end-to-end verification.

### Three Modes: daemon, client, and off

| Mode | Flow | Weight Load Time | GPU Memory | Use Case |
|------|------|-----------------|------------|----------|
| **daemon** | Engine launches daemon → daemon loads from disk → engine maps IPC | < 1s (after daemon ready) | 1× (shared) | First start; engine manages daemon lifecycle |
| **client** | Connect to pre-running daemon → map IPC | < 1s | 1× (shared) | Engine restart; daemon pre-running |
| **off** | Normal disk loading | 405–411s (Ling-2.6-1T FP8) | 1× | Default; no cache |

In **daemon** mode, the engine spawns daemon processes during startup and waits for them to load weights from disk. The first start is still slow (daemons must load from disk), but subsequent restarts are instant.

In **client** mode, the engine connects to already-running daemons. This is the fast-restart path — the daemon was started earlier and already holds weights in GPU memory.

### Safety and Robustness

The Weight Cache Daemon is designed to be **non-intrusive and safe**:

- **Minimal invasiveness**: The feature is self-contained in `python/sglang/srt/weight_cache/` with minimal changes to the core engine (only `load_model()` dispatch and a CLI flag).
- **Crash-safe**: If the daemon crashes, existing engine instances continue running — they already hold references to the IPC-mapped tensors via CUDA reference counting. GPU memory is only freed when **both** the daemon and the engine exit.
- **Daemon recovery**: If the daemon is restarted, it reloads weights from disk and re-export IPC handles. New engine instances can then connect to the restarted daemon.
- **Fallback on mismatch**: Config mismatches automatically fall back to disk loading (in client mode) or raise an error (in daemon mode, where fallback would cause OOM since both processes share the same GPU).

## Beyond Restart: Production Scenarios

The Weight Cache Daemon unlocks production patterns that are impractical with traditional disk-based loading:

### Multi-Instance Weight Sharing

A single daemon per GPU holds weights in memory; multiple engine instances (e.g., independent services) map to the same IPC handles via zero-copy. Weights are loaded from disk and quantized **exactly once per GPU**, regardless of how many instances consume them.

```
┌─ GPU 0 ────────────────────────────────────────────┐
│                                                    │
│  ┌──────────────┐   cudaIpcMemHandle    ┌────────┐ │
│  │              │ ────────────────────► │Engine A│ │
│  │  Weight      │ ────────────────────► │(inst 0)│ │
│  │  Cache       │                       └────────┘ │
│  │  Daemon      │   cudaIpcMemHandle    ┌────────┐ │
│  │              │ ────────────────────► │Engine B│ │
│  │              │ ────────────────────► │(inst 1)│ │
│  └──────────────┘                       └────────┘ │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Priority Co-Serving

Run a high-priority online service and a low-priority batch job on the same GPU, backed by the same weight cache daemon. The low-priority instance can be **evicted and re-spawned in sub-second time** without reloading weights from disk — enabling flexible GPU time-sharing without the usual startup penalty.

### Active-Standby Failover

Deploy a standby engine alongside the primary, both backed by the same weight cache daemon. The standby maps weights via zero-copy and stays warm. When the primary fails, the standby takes over in **< 1 second** — no weight loading, no disk I/O.

This achieves near-zero-downtime failover **without dedicating a full set of GPUs to an idle replica**, avoiding the expensive GPU resource waste of traditional hot-standby deployments.

```
┌─ GPU ─────────────────────────────────────────────────────┐
│                                                           │
│  ┌──────────────┐                                         │
│  │              │   cudaIpcMemHandle  ┌───────────────┐   │
│  │  Weight      │ ──────────────────► │ Primary Engine│   │
│  │  Cache       │                     │ (serving)     │   │
│  │  Daemon      │   cudaIpcMemHandle  └───────────────┘   │
│  │              │ ──────────────────► ┌───────────────┐   │
│  │              │                     │ Standby Engine│   │
│  └──────────────┘                     │ (warm)        │   │
│                                       └───────────────┘   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Performance

### Weight Loading: Disk vs IPC Zero-Copy

#### Single Node

| Model | Weight Size | Disk Load (s) | IPC Zero-copy (s) | Speedup |
|-------|-------------|---------------|-------------------|---------|
| **Qwen3-235B FP8** | **~235 GB** | **~306–327** | **<1** | **~500×** |
| **Ling-2.6-1T** | **~1 TB** | **~405–411** | **<1** | **~780×** |

#### Multi-Node

**PLACEHOLDER**

#### Performance Chart

**PLACEHOLDER Performance benchmark picture**

## How to Use

### Launch Weight Cache Daemons - single-node

One command launches all TP rank daemons:

```bash
# Standalone daemon launch (one command for all TP ranks):
python -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model --tp-size 4 \
    --load-format auto --dtype auto --quantization fp8
```

Wait for daemons to become ready (they write a `.ready` file per rank):

```bash
# Check readiness:
ls /tmp/sglang_weight_cache_rank*.ready
```

### Start Engine with Weight Cache

```bash
# Engine Client — connect to pre-running daemons (restart)
python -m sglang.launch_server \
    --model-path /path/to/model --tp-size 4 \
    --weight-cache-mode client
```

### Launch Weight Cache Daemons - multi-node

In a multi-node deployment, each node runs its own daemon for its local TP ranks. All daemons
join the same distributed group, so `--nnodes`, `--node-rank`, and `--dist-init-method` must be
consistent across nodes, with `$MASTER_ADDR` pointing at node 0:

```bash
# Daemon on node 0:
python -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model --tp-size 2 \
    --load-format auto --dtype auto --quantization fp8 \
    --nnodes 2 --node-rank 0 \
    --dist-init-method tcp://$MASTER_ADDR:29500

# Daemon on node 1:
python -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model --tp-size 2 \
    --load-format auto --dtype auto --quantization fp8 \
    --nnodes 2 --node-rank 1 \
    --dist-init-method tcp://$MASTER_ADDR:29500
```

Once every node reports its daemons ready, start the engine clients. They use a separate
rendezvous port (`29600`) from the daemons (`29500`):

```bash
# Engine client on node 0:
python -m sglang.launch_server \
    --model-path /path/to/model --tp-size 2 \
    --weight-cache-mode client \
    --nnodes 2 --node-rank 0 \
    --dist-init-addr $MASTER_ADDR:29600 --port 34000

# Engine client on node 1:
python -m sglang.launch_server \
    --model-path /path/to/model --tp-size 2 \
    --weight-cache-mode client \
    --nnodes 2 --node-rank 1 \
    --dist-init-addr $MASTER_ADDR:29600
```

## Fast Engine Recovery Framework: Roadmap

The Weight Cache Daemon is Phase 1 of a broader **Fast Recovery Framework** targeting **< 10s cold restarts** and **< 1s warm standby switches**:

| Phase | Current (s) | Target (s) | Approach | Status |
|-------|-------------|------------|----------|--------|
| **Load weight** | **~306–327** | **< 1** | **Weight Cache Daemon (CUDA IPC)** | **Done (this PR)** |
| Capture CUDA graph | ~34.9 | < 3 | CUDA graph serialization + replay | Planned |
| DeepGEMM JIT warmup | ~23.1 | < 2 | Kernel cache persistence, parallel warmup | Planned |
| Server init & Tokenizer | ~17.3 | < 3 | Lazy tokenizer init, config caching | Planned |
| Init torch distributed | ~4.7 | < 2 | NCCL session reuse, persistent process groups | Planned |
| KV Cache allocation | ~0.5 | < 0.5 | kvcache reuse | Planned |
| Server ready | ~3.4 | < 1 | Skip warmup requests on restart | Planned |
| **Total (single-node)** | **~390** | **< 10** | | |

Support for more models is also on the way.

## Acknowledgements

**Ant Ling Infra Team, Ant Group**: Michael Qiu qiudayu.qdy@antgroup.com

**Alibaba**: Siyu Liu liusy58@smail.nju.edu.cn

**SGLang Team**
