---
title: "OPD Support in Miles"
author: "Kaixi Hou & Miles Team"
date: "July 18, 2026"
previewImg: /images/blog/opd-support-in-miles/figure-1.png
---

We recently implemented **On-Policy Distillation (OPD) as an important feature in Miles**. OPD is now integrated into Miles rollout and training flow, so users can train a student model either solely with a teacher model's guidance, or combine teacher guidance with GRPO/PPO-style reinforcement learning objectives.

In this post, we describe the OPD implementation in Miles, the sparse teacher-scoring workflow we added to avoid unnecessary dense logprob payloads, and an initial Qwen3.5-35B-A3B self-distillation experiment validated on a single **8×NVIDIA B200** node. The experiment shows that without any task-specific reward, OPD can transfer a teacher model’s shorter-reasoning behavior to a base student model while preserving the student model’s performance.

![](/images/blog/opd-support-in-miles/figure-1.png)

## OPD as a Miles Primitive

Miles now supports OPD as an additive reverse-KL training signal. This enables OPD to support two important modes.

- **Pure distillation** trains the student only with the teacher signal. In this setting, task reward can be set to zero, and the update is driven entirely by the per-token student-teacher reverse-KL signal.
- **RL-augmented distillation** combines the OPD signal with task rewards and GRPO/PPO-style advantage estimators. This allows the student to learn from both environment feedback and teacher guidance.

On the rollout side, Miles supports both sampled-token OPD and top-k OPD through an SGLang-hosted teacher. In sampled-token OPD, the teacher scores the token actually sampled by the student at each response position. Because sampling may be stochastic, this token is not necessarily the student’s highest-probability token.

Top-k OPD extends this workflow by retaining multiple candidate tokens and their student logprobs at each position. The teacher scores those same candidates, and Miles combines the aligned student and teacher logprobs into the per-token reverse-KL signal used during training.

This design makes OPD a reusable training primitive in Miles rather than a separate one-off distillation path.

## Top-k OPD

Miles supports top-k OPD to construct a richer student–teacher comparison than the single sampled-token estimator.

During student rollout, Miles records a configurable set of candidate token IDs and their student logprobs at each response position. To score the candidates for position ttt, the teacher is conditioned on the original prompt and the student-generated tokens preceding position ttt. It then returns teacher logprobs for the requested candidate tokens at that position.

The OPD implementation also supports configurable top-k distillation strategies. Users can customize how the candidate set is constructed and how candidate tokens are weighted when computing the reverse-KL signal. This allows different distillation recipes to reuse the same rollout, teacher-scoring, and training infrastructure.

## Sparse Student-to-Teacher Scoring

A key system improvement is the sparse student-to-teacher scoring workflow.

In OPD, the student rollout produces a per-position top-k table. For each generated response position, Miles knows exactly which candidate token IDs and student logprobs are needed for the KL calculation. The teacher only needs to score those candidate token IDs under the corresponding causal prefix.

In the initial OPD implementation, the scoring workflow had to build a global union of all per-position top-k token IDs and ask the teacher to score that full union at every response position. This preserved correctness, but created a dense **R × |U|** intermediate payload, where **R** is response length and **U** is the global token-ID union. Since **|U|** can grow close to **R × K**, the old path could materialize and parse an **O(R²K)** JSON response even though the final OPD calculation only needs **O(RK)** values.

The new workflow replaces the dense global-union path with **sparse per-position candidate-token scoring**. Miles sends a per-position token-ID table to the teacher, and the teacher returns only the requested logprobs for each position’s own candidate set. This keeps the teacher-response payload aligned with the actual OPD computation: Miles carries only the sparse **R × K** values that OPD needs.

For long-response reasoning workloads, this matters because teacher scoring should not become dominated by unnecessary payload construction, transfer, or parsing. Sparse scoring makes the OPD pipeline more direct and better suited for high-throughput rollout and training.

## Validation: Qwen3.5 Self-Distillation on NVIDIA B200

To validate the implementation, we ran a controlled self-distillation experiment with **Qwen3.5-35B-A3B**.

The teacher was first trained with Reinforcement Learning with Verifiable Rewards (RLVR) to solve DAPO math problems using substantially shorter responses. The student was the corresponding pre-RL base checkpoint. We chose this setup because directly distilling DAPO capability produced little visible movement: the base Qwen student was already strong on the original task. Instead, we deliberately created a measurable behavior shift in the teacher — shorter but still effective reasoning — and tested whether OPD could transfer that behavior to a student that did not yet exhibit it.

To isolate OPD, we ran pure distillation.

- **Task reward:** zero
- **Training signal:** top-1 student-teacher reverse-KL
- **Teacher:** RLVR-trained shorter-reasoning checkpoint
- **Student:** pre-RL base checkpoint
- **Hardware:** single 8×NVIDIA B200 node
- **Execution mode:** student rollout, teacher scoring, and student training colocated and time-shared on the same GPU pool

The reference points were:

| Model / stage | Held-out DAPO | Mean response length |
| --- | --- | --- |
| RLVR teacher | 0.8870 | 6,248 |
| Initial base student | 0.8457 | 18,675 |

The teacher solves the same DAPO problems with much shorter responses, while the base student produces substantially longer solutions. Through this experiment, we want to answer the following question:

**Can pure OPD transfer the teacher’s efficient-reasoning behavior to the student while preserving held-out DAPO performance?**

## Results: Shorter Reasoning Without Performance Degradation

Our results provide clear evidence that OPD can transfer the efficient-reasoning behavior.

Held-out DAPO performance improved from **0.8457** to **0.8945**. At the same time, sampled rollout length dropped sharply after the first update, from roughly **18.6k tokens** to mostly **5.5k–6.7k tokens** across later rollouts, with only a brief stochastic spike. Per-token OPD reverse-KL also decreased from about **0.045** to **0.010**, showing that the student distribution became closer to the teacher distribution on the student’s own rollouts.

The three learning curves tell the same story from different angles.

**Held-out DAPO performance increases**, indicating that the student does not regress on task performance after OPD.

![](/images/blog/opd-support-in-miles/figure-2.png)

**Sampled rollout length drops quickly**, showing that the student adopts the teacher’s shorter-response behavior.

![](/images/blog/opd-support-in-miles/figure-3.png)

**Per-token OPD reverse-KL decreases**, confirming that the student distribution is moving closer to the teacher distribution.

![](/images/blog/opd-support-in-miles/figure-4.png)

Because raw task reward is zero in this experiment, the behavior change is solely driven by the OPD teacher signal. The central result is that **pure OPD transfers the teacher’s efficient-reasoning behavior to the base student while held-out DAPO performance improves rather than degrades**.

## Why This Matters for Miles

The OPD implementation allows Miles to be applied in post-training workflows where verifiable task reward alone is not enough.

Many practical workflows need to optimize not only task success rate, but also model behavior: reasoning length, response style, domain-specific habits, distributional alignment with a reference model, etc. OPD enables Miles to express such goals as teacher-guided training signals while leaving the same rollout and RL training pipeline intact.

OPD also provides a path toward multi-teacher consolidation. Different expert teachers can provide guidance for different domains or capabilities, allowing a single student model to absorb complementary behaviors into a more versatile model. Because Miles treats OPD as a reusable primitive, the same student-training pipeline can support different teachers and distillation recipes without requiring a separate training system for each domain. Validation of multi-teacher OPD remains future work.

The implementation also keeps OPD composable. Users can run pure distillation or use OPD as an auxiliary signal with GRPO/PPO-style objectives. This is important for post-training workflows where teacher guidance and task reward may need to work together.

Just as importantly, the system is optimized for long-response workloads. Sparse per-position scoring and top-k OPD help avoid unnecessary dense teacher-scoring payloads, making the pipeline better aligned with the actual computation needed for distillation.

## What’s Next

The current implementation establishes OPD as a first-class Miles capability and validates pure OPD in a controlled Qwen3.5 self-distillation setting. Next, we plan to extend validation in several directions.

**Validating OPD-augmented RL.**

We have validated pure OPD. The next step is to validate OPD combined with GRPO/PPO task rewards, where the student learns from both reward signals and teacher distributional guidance.

**Larger performance studies.**

Student rollout is currently the main cost, while teacher scoring adds overhead. We plan to benchmark the full pipeline on longer, higher-throughput workloads to better understand scaling behavior and bottlenecks.

**Multi-teacher OPD.**

A promising next step is to evaluate whether domain-specialized teachers can jointly guide one student model. This would test OPD as a mechanism for consolidating complementary expert behaviors into a single, more versatile model.

**More training recipes.**

The Qwen3.5 experiment focused on transferring shorter reasoning. Future recipes can explore other teacher-guided behaviors, including domain specialization, response style, tool-use patterns, and agentic data.

## Summary

Miles now supports OPD as a first-class training and rollout feature. The implementation includes additive reverse-KL training, SGLang teacher scoring, top-k OPD, configurable candidate and weighting strategies, and sparse per-position teacher scoring.

In an initial Qwen3.5-35B-A3B self-distillation experiment on a single 8×NVIDIA B200 node, pure OPD transferred shorter-reasoning behavior from an RLVR teacher to a base student. Held-out DAPO performance improved from **0.8457** to **0.8945**, sampled rollout length dropped from roughly **18.6k tokens** to mostly **5.5k–6.7k tokens**, and per-token OPD reverse-KL decreased from about **0.045** to **0.010**.

These initial results show that pure OPD can transfer a useful behavioral property—in this case, substantially shorter reasoning—while preserving strong held-out task performance. Broader experiments are still needed to validate the result across models, tasks, and teacher–student configurations.

## Acknowledgments

We are especially grateful to Hunter Carlisle and Priya Sethuraman at NVIDIA for their generous help and support in bringing this work together.
