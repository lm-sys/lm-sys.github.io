---
title: "Agent-Assisted SGLang Development: An Initial Exploration"
author: "SGLang Team"
date: "June 20, 2026"
previewImg: "https://raw.githubusercontent.com/BBuf/AI-Infra-Auto-Driven-SKILLS/main/docs/assets/sglang-sota-performance-loop.svg"
type: blog
---

SGLang development increasingly goes beyond isolated code changes. The same repository now spans LLM serving, distributed runtime, GPU kernels, diffusion pipelines, model-specific execution paths, and production incident handling. In the past, many of these workflows depended on individual developer memory: how to launch a certain model, how to read a profile trace, which log to add first when debugging a CUDA crash, or which benchmarks a performance PR should include. As agent tools mature, this experience can be turned into executable `SKILL.md` files, scripts, benchmark contracts, and review loops.

Around SGLang agent development, a set of skills has already emerged for both LLM and diffusion work:

- [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) covers workflows such as serving benchmarks, profile analysis, production incident triage, and SOTA loops.
- [BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot) explores automated optimization for SGLang diffusion kernels.

Viewed together, these efforts point to the same direction: the value of agents comes from procedural engineering knowledge, including executable steps, reproducible experiments, and reviewable evidence.

## 1. TL;DR

- Agents are most useful in SGLang when they can keep moving along a well-defined workflow. Benchmarking, profiling, kernel API logging, adding diffusion pipelines, production incident replay, and SOTA loops can all be encoded as skills.
- An SGLang skill is an executable development procedure. In `debug-cuda-crash`, `sglang-diffusion-benchmark-profile`, and `llm-torch-profiler-analysis`, the important content is preflight checks, hard failure gates, artifact contracts, reproduction commands, and result formats.
- Profile evidence is central to performance work. The SGLang profiler skills produce fixed kernel tables, overlap-opportunity tables, and fuse-pattern tables. KDA-Pilot extends this into same-ABI baseline/candidate comparison, real workloads, correctness gates, NCU evidence, and per-shape results.
- Long-running optimization has started to move into Loop Engineering. The SGLang SOTA Performance Loop decomposes "chasing SOTA" into fair benchmarking, gap decision, profiling, patching, and revalidation. Humanize/RLCR adds external review, and Codex Goal can fully replace the loop at lower cost for continuous iteration.
- Review becomes more important. Agents can run more experiments, but they also generate more changes that look plausible and still need careful review. Developers increasingly define problems, choose evidence, design workflows, and decide whether results are ready for production paths.

## 2. Why SGLang Is a Good Fit for Agent-Assisted Development

SGLang is a high-performance serving framework for LLMs and multimodal models. As model families and hardware paths expand, several recurring problems show up in development:

- LLM paths are complex. A single performance issue may cross the Python runtime, scheduler, CUDA graph, Triton/CUDA kernels, FlashInfer/FlashAttention, distributed collectives, and model-specific wrappers.
- Diffusion paths are also complex. A slower denoise pass may involve pipeline/stage partitioning, DiT blocks, attention backends, `torch.compile` graph breaks, CFG/SP parallelism, VAE, or custom fused kernels.
- Validation is expensive. Many changes must be tested on real models and real workloads on H100, H200, B200, or RTX 5090. Local unit tests alone are not enough.
- Profiles are hard to reuse manually. A single trace may contain hundreds of kernel launches. Reading Perfetto by hand can miss kernel-to-Python-source mappings and can easily mix up prefill and decode. Developers accumulate know-how while reading profiler output, such as which kernel names map to which model logic, which launch patterns suggest graph breaks, and which NCCL/attention/MLP layouts are normal. If that knowledge remains only in one person's head, the next task cannot reuse it.
- Performance conclusions depend heavily on context. GPU type, shape, batch size, parallelism, precision, backend, and compile state can all change the result. An isolated microbenchmark often cannot prove real model-level benefit, so an end-to-end long-running test process is needed to repeatedly validate throughput, latency, memory, accuracy, and stability under fixed workloads. That process is both labor-intensive and time-consuming.

These problems are a natural fit for agents. Launching servers, fixing workloads, collecting traces, triaging profile rows, adding tests, and recording experiment results all have clear inputs and outputs and are well suited to scripting and repeated execution. Developers need to define the boundaries: the same benchmark setup, the same profile interpretation rules, the same accuracy gates, and the conditions under which the agent should stop changing code.

The agent discussed here is therefore an executor constrained by engineering workflows. Repeated SGLang development procedures can be captured as skills, letting the agent handle repetitive execution, evidence collection, and state tracking. Developers remain responsible for defining goals, judging evidence, and reviewing whether a change belongs in the real serving path.

## 3. From Prompt Engineering to SKILL: Protocols and Examples

In the SGLang framework, a useful skill should at least answer the following questions:

| Question | What the skill should capture |
| --- | --- |
| When to use it | Trigger scenarios, supported models, supported hardware, and hard-stop cases |
| How to start | Preflight checks, environment variables, repository state, dependency checks, and model configuration |
| How to validate | Benchmark commands, profile commands, test entry points, and accuracy gates |
| How to decide | Output tables, failure modes, priorities, risk categories, and fallback conditions |
| How to deliver | Artifact directories, result schemas, PR descriptions, reproduction commands, and review requirements |

SGLang agent-related skills cover different layers. Some are close to source changes, such as debugging, testing, adding diffusion models, and benchmark/profile workflows. Others target cross-framework benchmarking, profile analysis, production incident triage, PR optimization knowledge, and higher-level workflows such as Humanize/RLCR.

### 3.1 Current Skill Stack

The commonly used SGLang agent-related skills fall into the following groups.

| Layer | Representative skill / project | Problem it solves |
| --- | --- | --- |
| CUDA crash | [`debug-cuda-crash`](https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash) | Records inputs, exceptions, and dumps around custom op/kernel API boundaries, turning transient crashes into samples that can be analyzed offline |
| LLM benchmark | [`llm-serving-auto-benchmark`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-auto-benchmark) | Runs fair, bounded, resumable serving benchmark search across SGLang, vLLM, and TensorRT-LLM |
| Trace triage | [`llm-torch-profiler-analysis`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-torch-profiler-analysis) | Produces fixed kernel, overlap-opportunity, and fuse-pattern tables, and maps kernels back to Python source |
| Pipeline/layer analysis | [`llm-pipeline-analysis`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-pipeline-analysis) | Slices torch profiler traces into forward passes, layers, and kernel flows to locate steady-state passes, bottleneck layer types, and Perfetto time ranges |
| Diffusion benchmark/profile | [`sglang-diffusion-benchmark-profile`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile) | Captures denoise latency, perf dumps, and torch profiler traces, while first checking that execution is actually using the native SGLang diffusion backend |
| Add diffusion model | [`sglang-diffusion-add-model`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model) | Adds a new diffusion model from a Diffusers/reference pipeline into the SGLang pipeline/stage/model/config structure |
| Diffusion performance tuning | [`sglang-diffusion-performance`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance) | Chooses performance settings such as `torch.compile`, warmup, SP/CFG parallelism, offload, attention backend, and quantization |
| Production triage | [`sglang-prod-incident-triage`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage) | Collects live-server bundles, saves failing requests, replays them, and then routes to focused crash/hang/profile tools |
| SGLang SOTA Performance Loop (Loop Engineering) | [`sglang-sota-humanize-loop`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-sota-humanize-loop) | First compares SGLang/vLLM/TensorRT-LLM fairly, then puts gap decision, profiling, patching, and revalidation into a Humanize/RLCR loop |

These entries turn easy-to-miss steps into executable protocols so the workflow can run, resume, and be reviewed.

### 3.2 Recent Optimization and Workflow Examples

The following examples come from recently merged SGLang PRs. The table focuses on the full engineering path: benchmarking, profiling, localization, code changes, tests, and revalidation.

| Case | Result | Key point |
| --- | --- | --- |
| Router long-context tokenization deduplication, [SGLang PR #28744](https://github.com/sgl-project/sglang/pull/28744) | On a DeepSeek-V4-Flash deployment, idle TTFT for 60k/125k-token prompts dropped by about `29%` / `41%`; under 60k-token load, TTFT dropped by `34%–49%` | The agent handled cache-aware routing, chat-encoder parity, engine-side `input_ids` fallback, and proxy body construction together, avoiding duplicate tokenization in the router and engine |
| Qwen3-Next FlashInfer allreduce fusion, [SGLang PR #22664](https://github.com/sgl-project/sglang/pull/22664) | On H100 TP=4, request throughput improved from `5.49 req/s` to `9.41 req/s`, about `+71.4%`; mean TTFT dropped from `456.24 ms` to `167.54 ms` | This is a profile-driven LLM collective optimization: unfused cross-device reduce dominated prefill, and the fused allreduce path was validated with MMLU/GSM8K accuracy checks |
| Cohere2Moe NVFP4 fused-MoE path, [SGLang PR #27401](https://github.com/sgl-project/sglang/pull/27401) | For `CohereLabs/command-a-plus-05-2026-w4a4` on 1x B300, request throughput improved over the previous SGLang default by `+26%` on chat and `+21%` on summarization, and beat vLLM in that setup by `+4.1%` / `+6.8%` | The change completed the routing semantics so the existing `flashinfer_trtllm` NVFP4 fused-MoE kernel could be used correctly in the real model path, with GSM8K/MMLU checks |
| Kimi Delta Attention (KDA) CuteDSL prefill kernel on SM100, [SGLang PR #27488](https://github.com/sgl-project/sglang/pull/27488) | For `moonshotai/Kimi-Linear-48B-A3B-Instruct`, KDA prefill on B200 became `1.08x–1.52x` faster than Triton; GSM8K moved from `0.915` to `0.920`, with a new regression test for realistic gate magnitudes | This kernel task had to cover the model's gate distribution, numerical overflow, host overhead, real-model accuracy, and unit tests before the optimization was ready to merge |
| Spectral Progressive Diffusion, [SGLang PR #27524](https://github.com/sgl-project/sglang/pull/27524) | Denoising speedups for FLUX.1, FLUX.2, Z-Image, Wan, and Qwen-Image reached `1.63x`, `1.77x`, `2.07x`, `2.32x`, and `1.6x` respectively in the reported RTX A6000 setup | This is a diffusion-side system optimization: early denoising runs at lower latent resolution, then GPU DCT upsampling restores full resolution when high-frequency details start to matter |
| LTX-2 VAE decode channels-last-3d, [SGLang PR #27431](https://github.com/sgl-project/sglang/pull/27431) | The LTX-2 decode stage improved from `5.41 s` to `3.84 s`, about `1.41x`; peak reserved memory dropped from `71.81 GiB` to `62.12 GiB`, saving about `9.7 GiB` | The profile pointed to Conv3d and layout conversion, so the fix preserved memory format in causal padding and connected the loader policy to single-GPU LTX-2 |

In these examples, the agent mainly contributes by executing the workflow: running benchmarks, reading profiles, locating Python source, changing code, adding tests, revalidating, and preparing PR descriptions. Without skills, many steps rely on manual reminders. Once encoded as skills, the workflow becomes much easier to repeat.

## 4. Profiling, Review, and Loop Engineering

A common mistake in SGLang performance work is to look only at total runtime, or to open Perfetto for a few minutes and decide by intuition that something "should be fused." This is even riskier for agents, because they can easily mistake a visually hot kernel for the real bottleneck.

In practice, two profiler skills are usually used together. `llm-torch-profiler-analysis` handles the first layer of trace triage and turns a global profile into three fixed tables:

- `Kernel Table`: summarizes GPU time share, launch count, and kernel category by stage, and maps kernels back to Python source and CPU ops when possible.
- `Overlap Opportunity Table`: uses exclusive/hidden time share, dependency risk, and kernel category to identify remaining overlap or headroom.
- `Fuse Pattern Table`: compares the trace against a source-backed pattern catalog of fusion/overlap paths in SGLang, vLLM, FlashInfer, and TensorRT-LLM.

These tables answer the first set of questions: which stage and which kernel take how much GPU time, which Python line they map to, and whether there is an existing fuse/overlap path to learn from. If SGLang trails vLLM or TensorRT-LLM, the profiler table should explain the gap before any code change starts.

The next step is `llm-pipeline-analysis`. Once global hotspots are known, we still need to know which forward pass, layer type, and kernel flow they belong to. This skill reads Chrome trace JSON and the model `config.json`, uses layer-boundary anchor kernels to split the trace into forward passes and layers, and then produces several tables for deeper analysis:

- `Forward pass summary`: separates cold-start from steady-state so warmup does not become the optimization target.
- `Per-layer timeline`: reports wall time, sum duration, and the share of categories such as MLA, MoE, GEMM, NCCL, MHC, and Hadamard for each layer.
- `Layer cluster statistics`: especially useful for models with alternating layer structures, such as NSA/hybrid-attention models with `compress_ratios`, where C4_LIGHT, C128_HEAVY, HASH, or other layer types may dominate latency.
- `Compute flow table`: expands representative layers into concrete kernel flows with hotness, relative timestamps, and input dimensions, making it easy to jump back into Perfetto.

Profile analysis therefore becomes a two-step process. First, `llm-torch-profiler-analysis` identifies the main conflict in the full trace. Then, `llm-pipeline-analysis` grounds the problem in steady-state forward passes, representative layers, and concrete kernel flows. The first step avoids choosing a direction by intuition. The second avoids staring at one global hot kernel while missing layer-type differences in the model structure.

### 4.1 Humanize/RLCR: Adding External Review to the Loop

Humanize addresses state and review in long-running tasks. A high-risk SGLang performance task usually does not finish in one implementation pass. It may go through many rounds of benchmarking, profiling, patching, reverting, changing direction, and validating again. Humanize splits this process into two stages:

1. Run gen-plan first. `humanize-gen-plan` turns a draft requirement into a structured `plan.md` containing the goal description, acceptance criteria, positive/negative tests, path boundaries, milestones, and implementation notes.
2. Run the RLCR loop next. `humanize-rlcr` starts the loop from `plan.md`. In each round, Claude Code reads `.humanize/rlcr/<timestamp>/round-<N>-prompt.md`, implements, commits, and writes a summary. Codex Review then checks state files, summaries, git cleanliness, review results, open questions, max-iteration conditions, and other gates. A single sentence claiming "task complete" is not enough to exit the loop.

This mechanism provides the execution and review foundation for the SGLang SOTA Performance Loop. Claude Code runs benchmarks, reads profiles, changes SGLang code, and revalidates. Codex Review checks evidence, state, and risk at the end of each round. It is a good fit for tasks that will become PRs, affect serving correctness, or require multi-day, multi-round experiments.

In practice, the command order should be explicit so the agent does not jump directly into implementation:

```text
1. Write a task draft under artifact_root/draft.md.
2. Run humanize-gen-plan to generate artifact_root/plan.md.
3. Start humanize-rlcr from artifact_root/plan.md.
4. Keep all decisions, summaries, and review state in the local Humanize workspace.
```

### 4.2 SGLang SOTA Performance Loop (Loop Engineering)

A single skill can stabilize one task. After a dozen rounds of experiments, however, another problem appears: which candidate is best, which directions have already failed, what the previous NCU report showed, whether the benchmark still matches the baseline, and when to stop. This state cannot live only in chat context.

The SGLang SOTA Performance Loop is a Loop Engineering workflow built on Humanize/RLCR. Here, SOTA means the best reproducible result under fixed experimental conditions: the same model, hardware, GPU count, precision, workload, SLA, framework commit, and serving parameters. The question is whether SGLang can reach the current best reproducible result under those conditions.

![SGLang SOTA Performance Loop](https://raw.githubusercontent.com/BBuf/AI-Infra-Auto-Driven-SKILLS/main/docs/assets/sglang-sota-performance-loop.svg)

Figure 1: SGLang SOTA Performance Loop. A fixed fair benchmark first establishes a reproducible baseline. The subsequent gap decision, profiling, pipeline analysis, patching, and revalidation are driven by the Humanize/RLCR loop.

A full SGLang SOTA Performance Loop contains the following stages:

1. Define the target boundary. For example, `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, single-node 2x B200, FP8, SGLang TP=2, and comparison against vLLM and TensorRT-LLM under the same 2-GPU budget.
2. Run fair search first. Before patching SGLang, search for the best reproducible SGLang/vLLM/TensorRT-LLM commands under the same workload and resource budget.
3. Decide the gap. If SGLang already matches or leads, record completion evidence. If it is consistently behind by more than the threshold, move to profiling.
4. Use profiles to explain the gap. Do not rush into code changes. First produce kernel tables, pipeline tables, overlap/fuse tables, and NCU reports when needed.
5. Patch only evidence-supported paths, such as hybrid attention, Mamba/GDN, radix cache, target verify, CUDA graph, MoE/EP, quant kernels, or model wrappers.
6. Revalidate on the same workload. Every round records benchmarks, profiles, accuracy, failed attempts, environment information, and cleanup actions.

In B200/H200 experiments with Qwen3.6-35B-A3B-FP8, the same model showed different bottlenecks on different hardware. On B200, under a fixed workload, the SGLang baseline already outperformed vLLM. Continued profiling still found optimization room in GDN prefill split, and after patching, output tok/s in both chat and long-context scenarios improved by about `2.6%`. On H200, changes around FP8 MoE Triton configs, the CUTLASS scaled-mm replacement path, and GDN backend defaults were needed to match and then exceed vLLM. If this type of task is split into many independent prompts, benchmarks, profiles, failed attempts, and intermediate conclusions are easy to lose. A loop with evidence and review keeps conditions aligned across rounds.

### 4.3 Codex Goal: A Lower-Cost Full Replacement

The SGLang SOTA Performance Loop above uses a two-role setup: Claude Code executes benchmarks, profiling, patching, and revalidation, while Codex Review checks each round at the end. This setup is suitable for serious PR work, but every round consumes both an execution model and a review model, increasing cost and waiting time.

Codex Goal offers another implementation. Once "fair benchmark -> gap decision -> profile -> patch -> revalidate -> artifact ledger" is written into a persistent Goal, a lower-cost GPT-5.5 model can complete execution, self-checking, and revalidation within the same goal. The core constraints of the SGLang SOTA Performance Loop remain: fixed workload, evidence-driven patches, revalidation under the same experimental conditions, and artifact manifest updates after every round.

The two approaches differ as follows:

| Dimension | Humanize/RLCR SOTA Loop | Codex Goal |
| --- | --- | --- |
| Execution | Claude Code handles implementation and experiments; Codex Review reviews each round | GPT-5.5 continuously executes, self-checks, and revalidates within the same Goal |
| State location | Plan, prompt, summary, and review results under `.humanize/rlcr/...` | Current Goal thread plus manifest/evidence under `artifact_root` |
| Review method | Stop hook, Codex Review, and git/state/schema checks | Goal-level self-checks, artifact contracts, and human spot checks |
| Cost | Two model roles participate, so each round costs more | One Goal carries both execution and checks, reducing cost |
| Main risk | More complex loop setup and longer waits per round | Goal drift or premature completion unless hard-stop conditions are explicit |

Below is a 2x B200 model optimization prompt example from [AI-Infra-Auto-Driven-SKILLS/prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts).

Humanize/RLCR version:

```text
Use the sglang-sota-humanize-loop workflow.

Task:
Optimize SGLang serving performance for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
on a single node with 2 NVIDIA B200 GPUs, FP8 precision, and initial SGLang
TP=2. SGLang should match or exceed the best reproducible vLLM/TensorRT-LLM
result under the same 2-GPU budget, workload, SLA, model, precision, and
environment constraints.

Required workflow:
1. Create a draft task document under artifact_root.
2. Run humanize-gen-plan to turn the draft into a structured plan.md.
3. Start humanize-rlcr from that plan.md in the Claude Code session.
4. Keep benchmark, profile, patch, and revalidation decisions inside the same
   Humanize workspace.

Evidence and safety requirements:
- Before patching, run a fair bounded search for SGLang, vLLM, and TensorRT-LLM.
- Check relevant open PRs in sgl-project/sglang and BBuf/sglang before choosing
  the SGLang baseline.
- If SGLang is behind by more than 1%, profile before patching.
- Prioritize evidence around hybrid attention, Mamba/GDN, radix cache, target
  verify, and CUDA graph.
- Record benchmark commands, profile artifacts, failed attempts, and cleanup
  evidence for every round.
- Patch only evidence-supported SGLang code paths.
- If a PR is needed, push/open it only against BBuf/sglang and include benchmark,
  GSM8K, and full MMLU accuracy tables.

artifact_root:
/workspace/sglang-agent-artifacts/b200_qwen3_next_80b_a3b_instruct_fp8_sota_humanize
```

Codex Goal version:

```text
/goal Using GPT-5.5, keep optimizing SGLang serving for
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on a single node with 2 NVIDIA B200
GPUs until SGLang matches or exceeds the best reproducible vLLM/TensorRT-LLM
result under the same 2-GPU budget, FP8 precision, workload, SLA, model, and
environment constraints. The current Codex Goal is the loop: fixed fair
benchmarking, gap decision, profiling, pipeline analysis, evidence-backed
patching, revalidation, final report, and optional PR preparation all happen
inside this Goal. Completion requires benchmark evidence, profile evidence when
SGLang was behind, correctness/accuracy evidence, a final artifact manifest,
and no regression in environment safety constraints.

model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /workspace
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
artifact_root:
/workspace/sglang-agent-artifacts/b200_qwen3_next_80b_a3b_instruct_fp8_sota_goal

Requirements:
- Use the current Codex Goal as the only persistent loop.
- Before patching, run a fair bounded search for SGLang, vLLM, and TensorRT-LLM
  under the same 2-GPU budget.
- If SGLang is behind by more than 1%, profile in the same Goal, then use
  llm-torch-profiler-analysis, llm-pipeline-analysis, and ncu-report-skill when
  needed before patching.
- Focus on hybrid attention, Mamba/GDN, radix cache, target verify, and CUDA graph.
- Update the artifact manifest, benchmark evidence, profile evidence, failed
  attempts, and next-step decision after every round.
- Stop and report a blocker if resources are unavailable, evidence is
  untrustworthy, the budget is exhausted, or no defensible next patch exists.
```

The Goal version preserves the same benchmark, profile, accuracy, and artifact requirements. The difference is that execution and review are folded into one persistent target. With clear hard-stop conditions, it can fully carry the SGLang SOTA Performance Loop.

## 5. KDA-Based CUDA Kernel Optimization for SGLang Systems

Beyond model-level optimization for LLMs and diffusion, kernel optimization is harder. Asking an agent to write CUDA directly can easily lead to benchmark reward hacking: changing the benchmark, using a lighter wrapper, enabling fast math that the baseline does not use, optimizing only one shape, breaking numerical semantics, or producing no gain in the real SGLang path.

KDA-Pilot separates kernel optimization into isolated tasks so the agent does not freely modify the whole SGLang repository:

- Workloads come from real SGLang diffusion models. The process first runs 20 diffusion models and summarizes actual kernel metadata.
- The baseline is copied from upstream SGLang main, with source lineage recorded.
- Baseline and candidate must use the same local ABI and the same build/export path.
- Benchmarks use fixed production rows, A/B interleaving, and CUDA event or wall timing.
- Correctness covers production rows, a canonical regression grid, NaN/Inf checks, poison output checks, and fallback contracts.
- Each iteration refreshes the task prompt, benchmark evidence, KernelWiki, and ncu-report-skill.
- Shape-specialized dispatch is allowed, but each bucket must document its condition, path, latency, and fallback.

![KDA-Pilot B200 diffusion kernel results](https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/kda-pilot-b200-speedups.svg)

Figure 2: Wall-geomean speedup for seven SGLang diffusion kernel tasks optimized by KDA-Pilot on B200. Wall time includes Python dispatch, wrapper overhead, kernel launch, and synchronization overhead visible through `cuda.synchronize()`, which is closer to the real call path than pure kernel device time.

| Kernel task | B200 wall geomean | Main optimization direction |
| --- | ---: | --- |
| `qknorm_rope` | `1.1341x` | Shared RoPE staging, Q/K reuse, large-row fast path |
| `norm_infer` | `1.3523x` | Warp-row RMS, tiled persistent RMS, 8B/16B vector path |
| `rotary_embedding` | `1.4912x` | 128-bit vector I/O, cos/sin hoisting, LTX2 block matching |
| `cutedsl_norm_tanh_mul_add` | `1.4953x` | Row-invariant math hoisting, launch-bounds tuning, exact tanh |
| `cutedsl_norm_scale_shift` | `1.3201x` | Operand-class dispatch, 16B/32B vectors, two-pass variance |
| `fuse_scale_shift` | `2.7499x` | rowgrid/flatvec/exact-C paths, cache hints, one-pass reduction |
| `group_norm_silu` | `2.3118x` | Split-group stats, channels-last direct path, fallback for giant rows |

These numbers should be read with the experimental setting in mind: they are kernel-task speedups on extracted production rows, not full model end-to-end gains. They are still useful. Once baseline, workload, correctness, profiling, and review are fixed, agents can produce reviewable incremental improvements on real framework kernels.

Two rules from the KDA-Pilot experiments are worth keeping:

- Do not leave room for benchmark reward hacking. Results become unreliable when baseline and candidate use different ABIs, different fast-math settings, or different wrapper paths. Another common issue is changing the benchmark shape set after seeing the results, for example removing shapes where the candidate is slower. Such results should not be used.
- Buckets close to the Roofline should allow no-go or fallback decisions. A good kernel optimization task should not force the agent to win every shape. For giant contiguous buckets or paths already close to the bandwidth limit, recording a fallback may be better than adding more complexity.

## 6. Practice Rules

1. Define the task boundary before starting the agent.
"Optimize SGLang" is too broad. "Match vLLM for `Qwen/Qwen3.6-35B-A3B-FP8` on 1x H200 under fixed `1000->1000` and `8000->1000` workloads" is an executable target.

2. Fix the benchmark before reading profiles.
If the workload can change after results are known, the agent may accidentally optimize an easier problem. Both the SOTA loop and KDA-Pilot put fixed workloads before patching.

3. Interpret NCU results according to the kernel's compute characteristics.
For memory-bound kernels, focus on DRAM/L2 throughput, load/store efficiency, and memory pipe utilization. For compute-bound GEMM/attention kernels, focus on Tensor Core utilization, SM busy, eligible warps, and the main stall reasons. For small latency-bound kernels, check launch count, per-kernel duration, synchronization points, and possible fusion opportunities. A single trace screenshot is not enough; the next code change should be supported by specific metrics.

4. Check backend and fallback gates before trusting a profile.
If an LLM run silently switches attention backend, disables CUDA graph, or takes a wrapper path different from the benchmarked one, the trace is no longer describing the target serving path. The same rule applies to diffusion: if logs show fallback to the diffusers backend, that trace cannot be used as evidence for native SGLang diffusion. These hard-stop conditions should live in the skill.

5. Kernel optimization must use the same ABI, wrapper, and compile flags.
In particular, the candidate should not silently take a lighter path, and `--use_fast_math` should not be enabled on only one side.

6. Review matters more than before.
Agents can create more PRs, and they can also create more plausible mistakes. Review for a high-performance system like SGLang needs to check shape, dtype, distributed execution, CUDA graph behavior, fallback behavior, accuracy, serving APIs, metrics, and benchmark setup.

Agent-era SGLang development will not remove developers from the system. The more realistic change is to write developer experience into workflows, hand repetitive execution to agents, and leave judgment, design, and review to people. The saved time can go into harder performance problems, model paths, and production stability, or back into improving the agent workflow itself. For an open-source inference framework, this kind of infrastructure is worth sustained investment.

## 7. Acknowledgments and References

We thank the SGLang Team members and contributors who helped build the SGLang agent skills: Xiaoyu Zhang (BBuf), Lianmin Zheng, Liangsheng Yin, Ke Bao, fzyzcjy, Kangyan Zhou, DarkSharpness, Mick, Alison Shao, Baizhou Zhang, Bingxu Chen, Cheng Wan, Ratish P, shuwenn, ykcai-daniel, Yuhao Yang, and Artem Savkin.

We thank the KDA team: Dongyun Zou, Ligeng Zhu, Sihao Liu, Junxian Guo, Yixin Dong, Zijian Zhang, and Hao Kang.

We thank the Humanize team and contributors: Sihao Liu, Ligeng Zhu, Zijian Zhang, Zenus Zhang, shinan6, DYZhang, Chao Liu, Zhou Yaoyang, gyy0592, AcrossForest, Emin, Qiming Chu, jiaxiaoyu, tastynoob, and zhenwei.

### 7.1 References

- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills)
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills)
- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
- [AI-Infra-Auto-Driven-SKILLS prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts)
- [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
- [KernelWiki skill](https://github.com/mit-han-lab/KernelWiki)
- [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill)
- [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)
- [SGLang Diffusion Advanced Optimizations, LMSYS Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)
- [OpenAI Codex Prompting: Goal mode](https://developers.openai.com/codex/prompting#goal-mode)
- [Humanize](https://github.com/PolyArch/humanize)
