---
title: "SGLang-Jax: An Open-Source Solution for Native TPU Inference"
author: "The SGLang-Jax Team"
date: "October 29, 2025"
previewImg: /images/blog/sglang_jax/cover.jpg
---

We're excited to introduce SGLang-Jax, a state-of-the-art open-source inference engine built entirely on Jax and XLA.
It leverages SGLang's high-performance server architecture and uses Jax to compile the model's forward pass.
By combining SGLang and Jax, this project delivers fast, native TPU inference while maintaining support for advanced features like continuous batching, prefix caching, tensor and expert parallelism, speculative decoding, kernel fusion, and highly optimized TPU kernels.

Benchmarks show that SGLang-Jax matches or outperforms other TPU inference solutions.
The source code is available at [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax).

## Why a Jax Backend?

While SGLang was originally built on PyTorch, the community has been eager for Jax support.  
We built a Jax backend for several key reasons:

- Jax is designed from the ground up for TPUs. For maximum performance without compromise, Jax is the clear choice. With Google expanding public access to TPUs, we expect Jax + TPU to gain significant traction and enable cost-efficient inference.
- Leading AI labs—including Google DeepMind, xAI, Anthropic, and Apple—already rely on Jax. Using the same framework for both training and inference reduces maintenance overhead and eliminates drift between the two stages.
- Jax + XLA is a proven, compilation-driven stack that excels on TPUs and performs well across a broad range of custom TPU-like AI chips.

## Architecture

The diagram below illustrates the SGLang-Jax architecture. The entire stack is pure Jax, resulting in clean code with minimal dependencies.

On the input side, it accepts requests via OpenAI-compatible APIs and utilizes SGLang's efficient RadixCache for prefix caching along with its overlap scheduler for low-overhead batching.
The scheduler pre-compiles Jax computation graphs for different batch sizes.
On the model side, we implement models in Flax and use `shard_map` for various parallelism strategies.
The two core operators—attention and MoE—are implemented as custom Pallas kernels.

<img src="/images/blog/sglang_jax/architecture.png" style="display:block; margin: auto; width: 85%;"></img>
<p style="color:gray; text-align: center;">The architecture of SGLang-Jax</p>

## Key Optimizations

### Integrating Ragged Paged Attention v3 
We integrated Ragged Paged Attention V3 ([RPA v3](https://github.com/vllm-project/tpu-inference/tree/main/tpu_inference/kernels/ragged_paged_attention/v3)) and extended it to support SGLang features:
- We tuned kernel grid block configurations based on different scenarios to achieve better performance.
- We made it compatible with RadixCache.
- To support EAGLE speculative decoding, we added custom mask to RPA v3 for use in the verification phase.

### Reducing Scheduling Overhead
Sequential operations on CPU and TPU during the forward pass can hurt performance. However, operations on different devices can be decoupled—for example, launching calculations on the TPU and immediately preparing the next batch to run. To improve performance, our scheduler overlaps CPU processing with TPU computation.

In the overlap event loop, the scheduler uses a result queue and threading events to pipeline CPU and TPU work. While the TPU processes batch N, the CPU prepares batch N+1. To maximize overlap between CPU and TPU, SGLang-jax carefully sequences operations based on profiling results. For Qwen/Qwen3-32B, we reduced the time gap between prefilling and decoding from approximately 12ms to 38us, and from approximately 7ms to 24us. More details can be found in our previous [blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

<img src="/images/blog/sglang_jax/profile_overlap.jpg" style="display:block; margin: auto; width: 85%;"></img>
<p style="color:gray; text-align: center;">Profile with overlap scheduler. The gaps between batches are minimal.</p>

<img src="/images/blog/sglang_jax/profile_no_overlap.jpg" style="display:block; margin: auto; width: 85%;"></img>
<p style="color:gray; text-align: center;">Profile without overlap scheduler. Note the large gaps (CPU overhead) between batches.</p>

### MoE Kernel Optimization
The MoE layer currently supports two implementation strategies: EPMoE and FusedMoE.
In EPMoE, we integrated the **Megablox GMM** operator, replacing the previous jax `ragged_dot`-based implementation.
Megablox GMM is specifically designed for MoE workloads and efficiently handles variable-sized expert groups described by group_sizes, eliminating unnecessary computation and non-contiguous memory accesses. In typical configurations, this operator delivers a **3–4× end-to-end (e2e) ITL speedup** compared to jax's native ragged_dot implementation.
Combined with efficient token permutation (permute/unpermute), expert-parallel communication via ragged_all_to_all, and adaptive tiling strategies, EPMoE significantly boosts overall throughput and works well in scenarios requiring cross-device parallelism with many experts.
In contrast, FusedMoE fuses all expert computations using dense einsum operations without inter-device communication overhead. It's better suited for cases with large individual experts but few total experts (e.g., < 64 experts). It also serves as a lightweight fallback for easier debugging and correctness validation.

### Speculative Decoding
SGLang-jax implements EAGLE-based speculative decoding, which is also known as Multi-Token Prediction (MTP).
This advanced speculative decoding technique accelerates generation by using a lightweight draft head to predict multiple tokens, which are then verified in parallel with a single pass through the full model.
To implement tree-based MTP-Verify, SGLang-jax adds non-causal mask support on top of Ragged Paged Attention V3, enabling parallel decoding of tree-based, non-causal draft tokens during the verification phase.
We currently support Eagle2 and Eagle3, and plan to continue optimizing the kernel implementation and add support for different attention backends at various MTP stages.

## TPU Performance
After all the optimizations, SGLang-Jax matches or outperforms other TPU inference solutions.
SGLang-Jax on TPU is also competitive when compared to GPU solutions.

You can find the full benchmark results and instructions at https://github.com/sgl-project/sglang-jax/issues/297.

## Usage

### Installing SGLang-Jax and Launching a Server

Install:
```bash
# with uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install sglang-jax

# from source
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e python/
```

Launch a server:
```
MODEL_NAME="Qwen/Qwen3-8B"  # or "Qwen/Qwen3-32B"

jax_COMPILATION_CACHE_DIR=/tmp/jit_cache \
uv run python -u -m sgl_jax.launch_server \
--model-path ${MODEL_NAME} \
--trust-remote-code \
--tp-size=4 \
--device=tpu \
--mem-fraction-static=0.8 \
--chunked-prefill-size=2048 \
--download-dir=/tmp \
--dtype=bfloat16 \
--max-running-requests 256 \
--page-size=128
```

### Using TPU via GCP Console
You can find the TPU option under Menu → Compute Engine and click Create TPU in the console.
Note: Only certain zones support specific TPU versions. Remember to set the TPU software version to v2-alpha-tpuv6e.
Under the Compute Engine menu, go to Settings → Metadata, click the SSH Keys button, and add your public key.
Once the TPU server is created, you can log in using the External IP and public key username shown in the console.
See also: https://docs.cloud.google.com/tpu/docs/setup-gcp-account
<img src="/images/blog/sglang_jax/gcp_usage_1.png" style="display:block; margin: auto; width: 85%;"></img>

### Using TPU via Skypilot
We recommend using Skypilot for daily development.
You can quickly set up Skypilot and find scripts for launching development machines and running tests in the sglang-jax repository.

Install Skypilot for GCP: https://docs.skypilot.co/en/latest/getting-started/installation.html#gcp
Then launch [sgl-jax.yaml](https://github.com/sgl-project/sglang-jax/blob/cdd6600a70ecb396382a510da9ea59c91a9ea2c0/scripts/tpu_resource.yaml#L1):

```bash
sky launch sgl-jax.yaml --cluster=sgl-jax-skypilot-v6e-4 --infra=gcp -i 30 --down -y --use-spot
```

This command will find the lowest-cost TPU spot instance across regions and automatically shut down the instance after 30 minutes of idle time. It will also install the sglang-jax environment for you.
Once setup is complete, you can log in directly using `ssh cluster_name` without tracking the external IP address.


## Roadmap
The community is working with Google Cloud team and multiple partners on the following roadmap.

- Model support and optimizations
   - Optimize Grok2, Ling/Ring, DeepSeek V3, and GPT-OSS
   - Support MiMo-Audio, Wan 2.1, Qwen3 VL
- TPU-optimized kernels
   - Quantization kernels
   - Communication and computation overlap kernels
   - MLA kernels
- RL integration with [tunix](https://github.com/google/tunix)
   - Weight synchronization
   - Pathways and multi-host support
- Advanced serving features
   - Prefill-decode disaggregation
   - Hierarchical KV cache
   - Multi-LoRA batching

## Acknowledgments
**SGLang-jax team**: sii-xinglong, jimoosciuc, Prayer, aolemila, JamesBrianD, zkkython, neo, leos, pathfinder-pf, Jiacheng Yang, Hongzhen Chen, Ying Sheng, Ke Bao, Qinghan Chen

**Google**: Gang Ji, Chris Yang, Shun Wang, Michael Zhang, Xiang Li, Xueqi Liu

**InclusionAI**: Junping Zhao, Guowei Wang, Yuhong Guo, Zhenxuan Pan

