---
title: "Bringing DeepSeek-V4 Flash RL Training to AMD Instinct MI355X GPUs with Miles"
author: "AMD & Miles Team"
date: "July 10, 2026"
previewImg: /images/blog/rocm-miles-dsv4/preview.png
type: news
---

DeepSeek-V4 RL is now supported in Miles on AMD Instinct™ MI355X GPUs with ROCm™! RL requires SGLang rollout and Megatron training to implement the same policy closely enough that token probabilities remain aligned, even as Miles repeatedly transfers updated weights back to the live rollout engine.

DeepSeek-V4 Flash makes this challenging through hybrid compressed attention, mHC residual mixing, and MoE routing. Our bring-up aligned model behavior across SGLang and Megatron, preserved quantized state during online weight updates, and established an end-to-end four-node workflow. We validated accuracy with a bounded train-versus-rollout log-probability difference and, in an extended run, a rising offline AIME-2024 benchmark score alongside improving online reward.

## Key takeaways

- **DeepSeek-V4 Flash RL now runs in Miles on AMD Instinct MI355X GPUs.** We resolved the model-alignment and online-update issues required for end-to-end execution on ROCm.
- **Four-node validation completed.** Successful end-to-end runs over 100+ optimizer steps with bounded train-rollout log-probability differences, improving online reward, and a rising offline AIME-2024 evaluation score.
- **Performance optimization is next.** Future work includes low precision training, end-to-end optimization, and scaling on larger clusters.

## DeepSeek-V4 Flash in one figure

DeepSeek-V4 Flash is a 284-billion-parameter MoE model with 13 billion active parameters per token. The configuration used in this work has 43 decoder layers, 256 routed experts with top-6 selection, four mHC residual streams, and hybrid attention that combines a 128-token sliding window with compressed long-context attention.

C4 layers select the top 512 entries from a 4:1-compressed KV sequence, while C128 layers attend densely over a 128:1-compressed sequence. Together with mHC and MoE routing, these are the main architecture-specific paths that SGLang and Megatron must implement consistently.

<img src="/images/blog/rocm-miles-dsv4/figure-1-deepseek-v4-block.png" alt="Simplified DeepSeek-V4 block showing mHC residual mixing, hybrid compressed attention, and top-6 MoE routing." style="display:block; margin: 2.5em auto 0.5em auto; width: 60%; max-width: 900px;" />

<p style="text-align: center; color: #666; font-style: italic;">Figure 1. Simplified DeepSeek-V4 block showing mHC residual mixing, hybrid compressed attention, and top-6 MoE routing.</p>

## The Miles stack

Miles orchestrates the asynchronous loop. SGLang generates candidate responses and rollout log probabilities; Megatron scores the same sequences, computes the policy update, and trains the actor; Miles then transfers the updated weights back to the live SGLang workers.

The current FP8 path uses an FP8 Hugging Face checkpoint for rollout and a BF16 Megatron torch_dist checkpoint for actor training. The two engines therefore represent one policy in different execution formats, making conversion, rescoring, and online update behavior part of the correctness boundary.

<img src="/images/blog/rocm-miles-dsv4/figure-2-miles-rl-loop.png" alt="Prompts flow to SGLang rollout; trajectories and log probabilities flow through Miles to the Megatron actor; updated weights return to SGLang before the next rollout." style="display:block; margin: 2.5em auto 0.5em auto; width: 100%; max-width: 900px;" />

<p style="text-align: center; color: #666; font-style: italic;">Figure 2. Prompts flow to SGLang rollout; trajectories and log probabilities flow through Miles to the Megatron actor; updated weights return to SGLang before the next rollout.</p>

## Challenge 1: closing the train-rollout log-probability gap

RL training depends on SGLang and Megatron assigning similar probabilities to the same generated tokens. We built a token-identical comparison workflow: SGLang generates a sequence once, both engines score the same tokens, and token-level differences reveal model-level mismatches before costly multi-node validation.

This comparison identified differences in two DeepSeek-V4-specific paths: early hash-routed MoE and mHC residual mixing. We aligned Megatron’s hash-routing behavior with SGLang and corrected the Megatron-side mHC post-mix so both engines preserve the same model semantics. These targeted changes brought rollout and training into closer numerical agreement.

## Challenge 2: preserving quantized semantics during online updates

In RL, the rollout server receives updated policy weights repeatedly without restarting. For quantized models, a successful transfer is not enough: packed weights, scale tensors, and quantization-dependent runtime state must retain their intended meaning.

For FP4 and E8M0 tensors, AMD made the update path datatype-aware, preventing the post-update tensor misinterpretation that produced invalid generations. For FP8, Miles already defines the post-update lifecycle; AMD added the missing SGLang interface in the ROCm stack so Miles can run the required quantization processing before rollout resumes.

The key lesson is simple: online updates must restore the model’s quantized state, not merely copy its bytes.

## Challenge 3: a stable multi-node parallel strategy on ROCm

Scaling DeepSeek-V4 Flash RL across multiple AMD Instinct MI355X nodes surfaced two coupled bring-up problems: selecting a model-parallel strategy that fits a 284-billion-parameter MoE at 4K context and keeping multi-node collective communication stable. The two are linked. Heavier tensor parallelism lowers per-GPU memory but increases collective traffic, and some early multi-node configurations stalled inside RCCL collectives - for example a tensor-parallel all-reduce or an expert all-to-all that did not complete and was caught by the communication watchdog, halting the run.

We converged on a layout that is both memory-feasible and stable: tensor-parallel 1, pipeline-parallel 4, and expert-parallel 4 across four eight-GPU nodes, paired with activation recomputation, optimizer-state offload to host memory, and bounded per-GPU token budgets. Shifting parallelism away from tensor-parallel all-reduce toward pipeline and expert parallelism, together with tuned RCCL transport settings, let the run proceed end-to-end for more than 100 optimizer steps without collective stalls. Establishing this stable operating point was a prerequisite for the longer validation that follows.

## Four-node validation on AMD Instinct MI355X GPUs

We validated the FP8 path on four eight-GPU AMD Instinct MI355X nodes: two for SGLang rollout and two for Megatron actor training. Miles coordinated GRPO-style training on a long-context math workload (DAPO-Math-17K at 4K context), reward collection, and repeated online weight updates, using a model-parallel configuration of tensor-parallel 1 / pipeline-parallel 4 / expert-parallel 4. Every ten steps we ran an offline evaluation on AIME-2024 with eight samples per problem. The rollout model used FP8, while the actor trained in BF16.

### Correctness and online reward

A key correctness check is whether rollout and training assign similar probabilities to the same generated tokens. Across the logged steps, the mean absolute log-probability difference was ~0.09. As Figure 3 shows, the difference remained bounded across more than 100 steps and repeated weight updates, without sustained upward drift or a sharp increase after updates. This is an encouraging bring-up result, not a final threshold.

<img src="/images/blog/rocm-miles-dsv4/figure-3-logprob-diff.png" alt="Train-versus-rollout absolute log-probability difference over the first 100 training steps." style="display:block; margin: 2.5em auto 0.5em auto; width: 100%; max-width: 900px;" />

<p style="text-align: center; color: #666; font-style: italic;">Figure 3. Train-versus-rollout absolute log-probability difference over the first 100 training steps.</p>

Beyond bounded log-probability agreement, the online reward also improved over training. In an extended run, the online raw reward showed a clear upward trend rather than staying flat: its mean rose from the first third of the run to the final third (Figure 4). This indicates the actor is improving under continued GRPO training and repeated online weight updates, not merely sustaining reward.

<img src="/images/blog/rocm-miles-dsv4/figure-4-online-raw-reward.png" alt="Online raw reward over 100 training steps with per-step values, moving average, and linear fit." style="display:block; margin: 2.5em auto 0.5em auto; width: 100%; max-width: 900px;" />

<p style="text-align: center; color: #666; font-style: italic;">Figure 4. Online raw reward over 100 training steps (per-step values, moving average, and linear fit). The reward increases over the run, with a positive linear slope.</p>

### Offline evaluation on AIME-2024

Online pass rate is measured on the training workload and is biased by dynamic sampling, so we also ran a held-out offline benchmark, AIME-2024 with eight samples per problem, every ten steps. This is the honest measure of model quality. Over the first 100 steps, offline AIME pass@1 improved from 0.39 to 0.49 and pass@8 from 0.53 to 0.67, while response truncation at the 4,096-token cap fell from 60% to 55%. Single-shot accuracy and multi-sample coverage improved together, indicating genuine capability gain under GRPO rather than mere sharpening. Per-evaluation values are noisy on a 30-problem benchmark, so the trend, not any single point, is the signal.

<img src="/images/blog/rocm-miles-dsv4/figure-5-aime-eval.png" alt="Offline AIME-2024 pass@1/2/4/8 over the first 100 RL training steps." style="display:block; margin: 2.5em auto 0.5em auto; width: 80%; max-width: 900px;" />

<p style="text-align: center; color: #666; font-style: italic;">Figure 5. Offline AIME-2024 pass@1/2/4/8 over the first 100 RL training steps (evaluation every 10 steps, eight samples per problem). Both single-shot accuracy (pass@1) and coverage (pass@8) trend upward.</p>

## What we learned

**Offline benchmark evaluation is the honest signal.** Online pass rate on the training workload is biased by dynamic sampling and over-sampling; a held-out AIME-2024 eval every ten steps gave a trustworthy measure of model quality. We recommend pairing every RL run with periodic offline evaluation.

**RL improved both accuracy and coverage.** Over 100 steps, offline AIME pass@1 rose from 0.39 to 0.49 and pass@8 from 0.53 to 0.67. The simultaneous rise in pass@1 and pass@k indicates the policy gained capability, not merely sharpened around solutions it already had.

**Cross-engine agreement is stable over long runs.** The train-rollout log-probability difference stayed bounded across 100+ steps and repeated weight updates, confirming the bring-up alignment holds well beyond the initial validation window.

**Response truncation is the dominant ceiling on absolute eval scores.** About 55-60% of AIME responses hit the 4,096-token generation cap; raising the evaluation response budget is the highest-leverage lever for higher absolute accuracy.

**Read trends, not single steps.** Per-step online reward and per-evaluation pass rates are noisy (online reward ranged 0.36-0.77 step to step); moving averages and periodic offline evaluation are the reliable progress signals.

## Path forward

- **Enable FP8 actor training.** Extend the Megatron actor from BF16 to FP8 and evaluate its effect on train–rollout alignment and training quality.
- **Profile and gap analysis.** Identify the largest performance gaps across the end-to-end RL pipeline and prioritize the highest-impact bottlenecks.
- **Performance optimization.** Improve rollout throughput, training efficiency, and overlap between rollout and actor execution.
- **Scaling.** Evaluate throughput, efficiency, and correctness on larger clusters, then tune distributed execution to maintain scaling efficiency.

## Launch commands

The experiments use an external Ray cluster and a single launch command, run inside the ROCm container.

Docker image: `rlsys/miles:rocm7.2-mi35x-dsv4`

The essential settings are shown below; full RCCL transport and flight-recorder environment variables are set in the launcher script.

```bash
# 1) Start the Ray cluster (one head + three workers), inside the ROCm container
# head node:
ray start --head --node-ip-address=$HEAD_IP --port=6379 --num-gpus=8
# each worker node:
ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --num-gpus=8

# 2) Launch DeepSeek-V4 Flash RL from the head node
export MASTER_ADDR=xxx
export MILES_SCRIPT_EXTERNAL_RAY=1
export RAY_ADDRESS=xxx
export PYTHONUNBUFFERED=1

export NCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export TP_SOCKET_IFNAME=xxx
export NCCL_IB_HCA=xxx
export NCCL_IB_GID_INDEX=1

RUN_ID=dsv4-fp8-4node-2roll-tp1-pp4-ep4-$(date +%Y%m%d_%H%M%S)
LOG=/workspace/miles/${RUN_ID}.log

/opt/venv/bin/python3 scripts/amd/run_deepseek_v4.py train \
  --run-id "${RUN_ID}" \
  --mode normal \
  --enable-eval \
  --num-nodes 4 \
  --actor-num-nodes 2 \
  --rollout-num-nodes 2 \
  --num-rollout 200 \
  --num-steps-per-rollout 1 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --context-length 16384 \
  --rollout-max-response-len 4096 \
  --max-tokens-per-gpu 8192 \
  --sglang-max-running-requests 48 \
  --sglang-max-total-tokens 524288 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 4 \
  --decoder-last-pipeline-num-layers 10 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 4 \
  --expert-tensor-parallel-size 1 \
  --extra-args '--wandb-team xxx --use-tis' \
  --extra-env-vars 'TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=200000
TORCH_FR_BUFFER_SIZE=200000 TORCH_NCCL_DESYNC_DEBUG=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/workspace/miles/nccl_fr_trace_ GPU_MAX_HW_QUEUES=2
NCCL_P2P_NET_CHUNKSIZE=262144' \
  2>&1 | tee "${LOG}"
```

## Summary

We enabled an end-to-end DeepSeek-V4 Flash RL training workflow on ROCm by aligning model behavior across SGLang rollout and Megatron training and preserving quantized state during online weight updates.

In a four-node AMD Instinct MI355X validation, Miles coordinated FP8 rollout, BF16 actor training, reward collection, and repeated policy updates over more than 100 optimizer steps. The train-rollout log-probability difference remained bounded throughout, the online reward improved, and an offline AIME-2024 benchmark score rose from pass@1 0.39 to 0.49 (pass@8 0.53 to 0.67). Next, we will enable FP8 actor training, pursue performance optimization, and evaluate the workflow at a larger scale.

## Acknowledgments

This work builds on DeepSeek-V4 support from the SGLang and Miles communities. We thank the Miles team working with AMD, together with contributors to Megatron, AITER, Triton, TileLang, Transformer Engine, and ROCm whose software forms the end-to-end stack.

**AMD contributors:** Xinyu Kang, Liz Li, Yuankai Chen, Zhiyao Jiang, Kailesh Gogineni, Yao Fu, Wen Xie, Gowtham Ramesh, Cheng Yao, Xiaobo Chen, Shekhar Pandey, Sree Rohith Pulipaka, Wen Chen, Yuzhen Zhou, Xinyu Jiang, Hai Xiao, Andy Luo, Zhenyu Gu.

**Miles contributors:** Yusheng Su, Jiajun Li, Banghua Zhu, and miles team

## Appendix

### ROCm Runtime And Kernel Path Map

The reported run used the following paths for the model components that most directly affect cross-engine agreement and online updates.

| Model component | Selected runtime paths | Why it mattered |
|---|---|---|
| mHC residual mixing | **Rollout:** AITER mHC pre/post.<br/>**Training:** TileKernels pre; explicit PyTorch/HIP post-mix. | Keeps the residual-stream mapping visible and comparable across engines. |
| Hybrid attention | **Rollout:** ROCm fused MLA decode; Triton sliding-window preparation; fused compressor and paged-compressor paths; TileLang indexer.<br/>**Training:** Miles DeepSeek-V4 attention in BF16. | Covers local and compressed attention through different execution stacks. |
| MoE and routing | **Rollout:** Triton FP8 MoE; fused hash top-k.<br/>**Training:** Megatron MoE and router path. | Requires the same deterministic hash-routing semantics on both sides. |
| Online weight update | **Rollout:** Distributed weight update plus SGLang post_process_weights.<br/>**Training:** Miles broadcast update from the BF16 actor. | Rebuilds quantization-dependent runtime state before generation resumes. |

<p style="text-align: center; color: #666; font-style: italic;">Table 1. Runtime paths used in the reported ROCm configuration.</p>

The launcher makes the selected backends explicit, while Docker-scoped patches remove remaining CUDA-only assumptions in dependent Megatron and Transformer Engine paths. This keeps the tested configuration reproducible and reviewable.
