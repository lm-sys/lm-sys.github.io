---
title: "SGLang and Miles on NVIDIA Vera Rubin"
author: "SGLang & Miles Team"
date: "October 9, 2026"
previewImg: /images/blog/vera-rubin/mla-pipeline.png
---

## Summary

We had early access to two NVIDIA Vera Rubin nodes (8 GPUs) and used them to bring SGLang and Miles up on the platform: SGLang for inference and Miles for RL training. On the inference side, we tuned SGLang for Kimi K3 NVFP4: this post covers what we changed in the attention, MoE/communication and KDA kernels, and how much each change bought. On the training side, Miles, which uses SGLang for rollout and Megatron for training, runs RL end to end on Vera Rubin out of the box, including agentic RL with sandboxes on the Vera CPU.

NVIDIA Vera Rubin is the GPU generation after NVIDIA Blackwell Ultra (GB300 NVL72). For kernel work, the most impactful changes are more shared memory per CTA (327 KiB, up from the 227 KiB on NVIDIA Hopper and Blackwell), more SMs (212 on our nodes), and NVIDIA NVLink 6. Some of our changes retune Blackwell-era kernels for these new limits. The rest remove overheads we found while profiling Kimi K3 on Rubin, such as kernel launches, barrier stalls and hidden copies.

Cognition has deployed Vera Rubin NVL72 with its own SGLang-based stack and has [reported up to a 4.8x increase](https://www.coreweave.com/blog/cognition-becomes-first-customer-for-nvidia-vera-rubin-nvl72-on-coreweave-cloud) in total token throughput over GB200 NVL72.<sup><a href="#cognition-note">1</a></sup>

## Inference: Serving Kimi K3 NVFP4

Kimi K3 is a 2.8T-parameter hybrid model with a 1M-token context window. Its 93 attention layers interleave 69 KDA linear-attention layers with 24 MLA layers, and every attention output is banked and aggregated per block through Attention Residuals. The FFN is a LatentMoE with 896 experts, top-16 routing, running in a 3584-dimensional latent space. We serve the NVFP4 checkpoint with [RadixArk's DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) block speculative decoding, so every decode step is a short draft pass followed by a verify pass over the drafted tokens.

### Attention

**Deeper MLA pipelines.** In long context settings, MLA decode spends most of its time waiting on KV reads from HBM, and the way to hide that is to keep more reads in flight. On Blackwell, the FP8 kernel had already filled its 227 KiB of shared memory with 3 K stages and 2 V stages. Vera Rubin raises the per-CTA limit to 327 KiB, just enough for a fourth K stage and two more V stages. With the deeper pipeline, MLA at batch 16 and 128k context runs **16% faster**, with bit-identical output.

![Deeper MLA pipeline using Vera Rubin shared memory](/images/blog/vera-rubin/mla-pipeline.png)

*Figure 1. Deeper MLA pipeline with out-sized SMEM.*

**Faster split-KV reduction.** At low concurrency, one request's KV cache is split across many CTAs, and a second kernel merges their partial outputs. That merge loaded one split's partial result, waited for it, then loaded the next, paying L2 latency once per split. We now issue all the loads first and accumulate from registers afterwards, which makes full FP8 MLA **20% faster** at batch 1 and 128k context.

**64-way split-KV.** A batch-1 decode request gives the attention kernel very little parallel work, so with the default 32 splits most of Vera Rubin's 212 SMs sat idle. Doubling to 64 splits for K3's shape keeps more of them busy, worth **1.9% end to end** with FP8 KV.

**Native FP8 conversion.** A small detail in MLA's KV/Q preparation turned out to matter. Converting to FP8 without saturation kept the compiler from using the packed hardware conversion and pushed it onto a slow software path. With saturating conversion (overflow now clamps to ±448 instead of turning into NaN), the preparation kernel is **2.5x faster**.

### MoE and communication kernels

**Fused MoE tail.** Every MoE layer ends the same way: finalize the expert outputs, add the shared expert, all-reduce across 8 GPUs, then RMSNorm. Each of those kernels is tiny, so launch overhead dominates, and K3 pays it in all 92 MoE layers. We defer the finalization and fold all four steps into one collective kernel. That removes 276 launches per decode step and gives **5.9% end to end**.

**Push all-reduce at higher concurrency.** The fused tail uses a push-style all-reduce for small batches and a pull-style one for larger ones. On Vera Rubin, push stays faster up to 256 rows, so we moved the crossover there, which improved decode throughput by **3.6% at 32 concurrent requests**.

**Tensor cores for the latent all-gather GEMM.** Profiling showed the MoE latent up-projection's time going into the producer's arithmetic: over a thousand scalar FMAs and dozens of warp shuffles per thread. Moving that work onto mma.m16n8k16 tensor-core instructions for 4–8 rows cut decode step time by **1.1%**.

### KDA

**Less stall in KDA verify.** KDA's fused verify kernel spent much of its time waiting at barriers instead of computing. Every CTA copied its conv weights into shared memory and then waited at a block-wide barrier. Now each warp keeps its own weights in registers. State handoffs that stay inside one warp require no CTA barriers. The v warps finish precomputing early, so they now use that idle time to compute the output-norm gate. Outputs stay bitwise identical. On Vera Rubin, the production verify kernel sped up by 20%.

![KDA verify restructuring to reduce stalls](/images/blog/vera-rubin/kda-verify.png)

*Figure 2. Restructuring KDA verify to reduce stall.*

**Strided output gate.** The gated RMSNorm in each KDA layer takes its gate from a column slice of the fused QKVG projection. Because the slice isn't contiguous, PyTorch silently copied it before every call: 2.6 µs, 69 times per step. Teaching the norm kernel to read the gate through its strides removed the copy and cut decode step time by **1.8%**.

## RL training: Miles on NVIDIA Vera Rubin

Miles runs RL training end to end on NVIDIA Vera Rubin, with SGLang for rollout and Megatron for training, from a single container image. With the [Rubin-enablement PR](https://github.com/radixark/miles/pull/3700) in place, these RL runs were launched out of the box using Miles, on a single tray (4 GPUs):

* **Qwen3-30B-A3B** on GSM8K (256 prompts × 8 samples per rollout, up to 1,024 response tokens): training converges as expected, with reward rising from about 45% to 95% over 50 rollouts (Figure 3), which matches the GB300 reward growth curve perfectly.

* **DeepSeek-V4-Flash** (4-layer, FP8) on GSM8K (32 prompts × 8 samples per step, up to 256 response tokens): the FP8 path runs end-to-end for both rollout and training.

<img src="/images/blog/vera-rubin/gsm8k-reward.png" alt="GSM8K reward on GB300 and Vera Rubin over 50 rollouts" style="display: block; width: 100%; max-width: 520px; height: auto; margin-left: auto; margin-right: auto;" />

*Figure 3. GSM8K training reward over 50 rollouts, Qwen3-30B-A3B (256 prompts × 8 samples per rollout, up to 1,024 response tokens), GB300 vs Vera Rubin.*

These are promising initial results demonstrating Miles' readiness on Vera Rubin. The next phase will focus on scaling up and further improving performance.

### Agentic RL on the Vera CPU

As a first step toward agentic RL on Vera Rubin, we run it on a single tray with **NeMo Gym**'s mini-SWE-agent environment: each episode gets its own sandbox on the tray's **Vera CPU**, next to the GPUs that serve and train **Qwen3.5-35B-A3B**, with 64 sandboxes running at once. We chose SWE-bench Verified because it ships prebuilt arm64 task images, and we verified that grading works on them. Each step samples 8 tasks × 8 attempts from a 64-task pool, with up to 64K tokens per episode. Training runs normally with all environments on the Arm-based Vera CPU: the reward holds at about 0.6, episodes get about 30% shorter, and rollout and training stay closely matched (Figure 4). Once Vera CPU servers are available at scale, we look forward to agentic training with far more concurrent environments.

![Agentic RL reward, response length, and train–rollout KL](/images/blog/vera-rubin/agentic-rl.png)

*Figure 4. Agentic RL on one Vera Rubin tray, with sandboxes on the Arm-based Vera CPU: Qwen3.5-35B-A3B on SWE-bench Verified (arm64 tasks, 8 tasks × 8 attempts per step, up to 64K tokens per episode). Left to right: training reward, median response tokens per episode, train–rollout KL. Dark lines: 8-step mean.*

## What's Next

This is only a first look at what SGLang and Miles can do on Vera Rubin. On inference, we'll keep rolling out optimizations to take full advantage of the platform across a wider range of production workloads. On training, Miles will focus on scaling up and improving performance, and on agentic RL with far more concurrent environments once Vera CPU servers are available at scale.

## Acknowledgements

This work was developed in close collaboration between the SGLang & Miles team at RadixArk and NVIDIA


<p id="cognition-note"><sup>1</sup> Cognition's results come from its own private SGLang fork and benchmark settings.</p>
