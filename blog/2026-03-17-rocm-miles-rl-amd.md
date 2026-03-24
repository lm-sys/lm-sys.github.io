---
title: "ROCm Support for Miles: Large-Scale RL Post-Training on AMD Instinct™ GPUs"
author: "AMD & Miles Team"
date: "March 17, 2026"
previewImg: /images/blog/rocm_miles_rl/fig_1.png
---

Reinforcement learning (RL) has rapidly become a core stage of modern foundation-model development. While large-scale pretraining remains essential, today's most capable models rely heavily on post-training techniques to improve reasoning, tool use, and multi-turn interaction. These workflows depend on scalable reinforcement learning infrastructure capable of running across multi-node GPU clusters.

We are excited to announce **ROCm support for the Miles reinforcement learning framework** on AMD Instinct GPUs, including MI300/350-class accelerators.

This blog explains what Miles is, why RL workloads are a strong fit for AMD GPUs, what features are already supported today, and how to run an end-to-end RL training workflow on ROCm.

## Introducing Miles

Miles is an open-source RL framework designed for **large-scale post-training of language and multimodal models**. The framework builds on the SGLang and [Slime](https://github.com/THUDM/slime) RL ecosystem and targets production-grade RL pipelines.

Miles provides infrastructure for:

- Distributed rollout generation
- Policy optimization (GRPO / PPO)
- On-policy RL training loops
- Ray-based orchestration
- Integration with Megatron-LM and SGLang; Support for alternative backend like FSDP

The framework is designed around the full RL lifecycle and is capable of running across multi-node GPU clusters.

Miles has seen strong adoption across the RL community. With ROCm support, these workflows now run natively on AMD Instinct GPUs out of the box.

## Why RL Workloads Fit AMD Instinct GPUs

Reinforcement learning workloads differ from pretraining in a crucial way:
**rollout generation dominates compute.**

Modern RL training may spend 70–90% of GPU time generating long sequences across thousands of parallel environments. This makes memory capacity and bandwidth critical performance factors.

AMD Instinct MI GPUs provide:

- Large HBM memory capacity
- High memory bandwidth
- Efficient long-context inference
- Strong multi-node scaling

These properties can mitigate the classical rollout-heavy bottleneck within RL pipelines.

## Miles RL System Architecture on ROCm

Miles is a decoupled RL training architecture that separates rollout generation (SGLang) from model optimization (Megatron) and coordinates them through a scheduler for scalable post-training.

<img src="/images/blog/rocm_miles_rl/fig_1.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Fig. 1 Miles architecture diagram</center></small>

Miles separates data generation (rollouts) from model training, and connects them through a central scheduling layer.

Miles is a two-plane architecture:

- Rollout plane → generates training data
- Training plane → updates model weights

A scheduler coordinates the interaction between the two. This entire pipeline has now been validated end-to-end on ROCm.

## Getting Started: Run Miles on AMD GPUs

Miles provides a ROCm-ready workflow with prebuilt containers, so you can run the full RL pipeline with minimal setup. Choose the container that matches your GPU generation:

- MI300X: `rlsys/miles:rocm7-MI300-sglang0.5.9-latest`
- MI350X / MI355X: `rlsys/miles:rocm7-MI350-355-sglang0.5.9-latest`

### Launch the Miles ROCm container

Set the image tag based on your hardware:

```bash
# MI350X / MI355X:
# export MILES_IMAGE=rlsys/miles:rocm7-MI350-355-sglang0.5.9-latest
# MI300X:
export MILES_IMAGE=rlsys/miles:rocm7-MI300-sglang0.5.9-latest

docker pull $MILES_IMAGE

docker run -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v $HOME:$HOME --shm-size 128G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w $PWD $MILES_IMAGE /bin/bash
```

The container includes SGLang and Megatron-LM preinstalled.

### Install Miles and download assets

```bash
git clone https://github.com/radixark/miles.git
cd miles
git checkout 90b66b542b38c3b67537bb99a505bb707ebfcf6d
pip install -e .
```

Download the example model and datasets:

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
```

Convert the Hugging Face checkpoint to Megatron format:

```bash
source scripts/models/qwen3-4B.sh
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

### Launch RL training

```bash
# Prevents Ray from overriding GPU visibility (Miles manages device assignment directly)
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MILES_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
bash scripts/run-qwen3-4B-amd.sh
```

This launches a full RL pipeline:

- Ray cluster initialization
- SGLang rollout workers
- GRPO training loop
- On-policy rollout → train → update cycle

## Experiments and Performance

### Agentic task training: multiturn

Multi-turn agents are quickly becoming the default interface for real-world AI systems. Most practical tasks need multiple steps of reasoning, code/tools to help agents verify their work / use feedback to correct mistakes mid trajectory.

In this section, we show a multi-turn example from Qwen2.5-32B trained to use a Python interpreter for high-school math style problems.

Setup:

- Model: pe-nlp/retool-sft-qwen2.5-32b-ins (SFTed on [retool](https://huggingface.co/datasets/JoeYing/ReTool-SFT))
- Stack: SGLang + Ray + Megatron-based training
- Train dataset: dapo-math-17k
- Evaluation set: aime-2024

<img src="/images/blog/rocm_miles_rl/fig_2.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Fig. 2 Number of turns w.r.t. training steps</center></small>

As we keep training, we observe the average turn count per trajectory keeps improving.

<img src="/images/blog/rocm_miles_rl/fig_3.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Fig. 3 Eval score w.r.t. training steps</center></small>

From the eval curves, we see that the model is already pretty good at math (from pass@n metrics). The eval score improves more gradually than pass@n, reflecting that formatting/tool-call quality is still catching up.

<img src="/images/blog/rocm_miles_rl/fig_4.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Fig. 4 Raw reward w.r.t. training steps</center></small>

Raw reward improves over time, which is a strong sign the policy is learning better trajectory behavior.

### Performance

On a single 8-GPU AMD Instinct MI300X node, we trained Qwen3-30B-A3B with GRPO (32×8 sampling, 8k response cap, global batch 256), using TP4/EP8 sequence-parallel Megatron settings and no KL-loss term.

<img src="/images/blog/rocm_miles_rl/fig_5.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Figure 5. Per-step time partition (mean over steps 4-145).</center></small>

Mean step time is 388.50s. Rollout generation is the largest component (152.79s), followed by actor training (95.30s), update weights (33.85s), and log-prob computation (31.53s).

<img src="/images/blog/rocm_miles_rl/fig_6.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Figure 6. Left: rollout throughput (tok/GPU/s); right: train throughput (tok/s).</center></small>

Rollout throughput stays mostly in the ~1.1k-1.3k tok/GPU/s range. Throughput gradually declines over training; in this run it coincides with lower truncation and shorter responses (Figure 3) while AIME improves. Train throughput remains comparatively stable around ~15-16k tok/s.

<img src="/images/blog/rocm_miles_rl/fig_7.png" style="display: block; margin: 20px auto 0; width: 75%; max-width: 100%; height: auto;" />
<small><center>Table 1. AIME evaluation checkpoints during partial-rollout training</center></small>

AIME accuracy increases from 0.665 (step 19) to 0.729 (step 139). Across the shown checkpoints, mean AIME accuracy is 0.702, mean pass@16 is 0.890, truncation drops from 0.294 to 0.156, and mean response length declines from 11,718 to 8,789.

## Feature support roadmap on AMD

Today, **core Miles functionality is fully supported on AMD**, including:

- GRPO training
- Model and data parallelism
- Dynamic batching
- Both Megatron and FSDP backends
- Partial rollout
- Miles router

These capabilities enable end-to-end training and serving workflows on AMD platforms today.

Looking ahead, we are committed to expanding support for additional high-value differentiators, including:

- **True On-Policy training**
- **FP8 pipeline optimization**
- **Rollout Routing Replay (R3)**
- **INT4 QAT**
- **DeepEP**
- **Speculative decoding**

Our goal is not only feature parity, but continuous performance and capability improvements aligned with the evolving Miles roadmap.

## Summary

Reinforcement learning is now a core stage of foundation-model development. With ROCm support for Miles, AMD GPU users can run modern RL pipelines—including distributed rollouts and GRPO training—on MI300/350-class clusters today.

This work expands hardware portability across the PyTorch ecosystem and lowers the barrier for large-scale RL experimentation on AMD GPUs. AMD is committed to continuous support involving Miles features.

## References

- Miles repository: [https://github.com/radixark/miles](https://github.com/radixark/miles)
- Miles ROCm tutorial: [https://github.com/radixark/miles/blob/main/docs/en/platform_support/amd_tutorial.md](https://github.com/radixark/miles/blob/main/docs/en/platform_support/amd_tutorial.md)
- ROCm slime blog: [https://rocm.blogs.amd.com/artificial-intelligence/slime/README.html](https://rocm.blogs.amd.com/artificial-intelligence/slime/README.html)

### Sample trajectory

<details>
<summary>Click to expand sample trajectory</summary>

**question**

Solve the following math problem step by step. The last line of your response should be of the form Answer: \boxed{&#36;Answer} where &#36;Answer is the answer to the problem.
Find the sum of the four least positive integers each of whose digits add to &#36;12&#36;.
Remember to put your answer on its own line after "Answer:".

---

**turn 1:**

To solve the problem of finding the sum of the four least positive integers each of whose digits add to 12, we'll need to systematically generate numbers starting from the smallest possible and check their digit sums. The key is to begin with the smallest numbers and proceed upwards until we have the first four that meet the criteria. Let's use code to automate this process.

&lt;tool_call&gt;
{"name": "execute_python_code", "arguments": {"code": "def digit_sum(n):\n    return sum(int(d) for d in str(n))\n\nnumbers = []\ncurrent = 39  # Start checking from 39 since it's the smallest two-digit number whose digits sum to 12 (3+9=12)\nwhile len(numbers) &lt; 4:\n    if digit_sum(current) == 12:\n        numbers.append(current)\n    current += 1\n\nprint(numbers)"}}
&lt;/tool_call&gt;&lt;|im_end|&gt;

python output
[39, 48, 57, 66]

---

**turn 2:**

The code generates the first four numbers whose digits sum to 12, which are 39, 48, 57, and 66. Now, we need to calculate their sum.

&lt;tool_call&gt;
{"name": "execute_python_code", "code": "sum_numbers = sum([39, 48, 57, 66])\nprint(sum_numbers)"}}
&lt;/tool_call&gt;&lt;|im_end|&gt;

python output

Error: Validation failed for input parameters: 1 validation error for Execute\_python\_codeTool
code
  Field required [type=missing, input\_value={}, input\_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

---

**turn 3:**

The sum of these four numbers is calculated to be 210. This confirms that the four smallest numbers are indeed 39, 48, 57, and 66, and their sum is 210.
Therefore, the answer is 210.

Answer: \boxed{210}&lt;|im_end|&gt;

python output
(no additional tool output in this turn)

---

_meta: label=210, reward=1.0_

This trajectory shows both current capability and current gap:
- The model uses tools correctly in turn 1 to derive key intermediate values.
- In turn 2, it still makes a tool-call schema mistake (code passed directly instead of arguments.code).
- Turn 3, it fails to correct code and resorts to just using its math skill to sum the numbers and get the final answer.

With more training, turn 3 will ideally fix the code error and produce right output.

</details>
