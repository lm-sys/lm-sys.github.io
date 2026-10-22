---
title: "Elastic Expert Parallelism Scaling in SGLang: Growing a Live MoE Deployment with NIXL"
author: "Yoray Zack — Senior Software Architect at NVIDIA and SGLang contributor"
date: "October 21, 2026"
previewImg: /images/blog/elastic-ep-scaling-nixl/scaling-options.png
---

Inference deployments operate under changing load and must adapt their capacity while maintaining throughput and latency targets. For large Mixture-of-Experts (MoE) models, the standard options are horizontal scaling, which adds another complete replica, and vertical scaling, which restarts the deployment with a wider expert-parallel topology.

Horizontal scaling keeps the current service running, but adds the full model footprint. Vertical scaling can reach the required topology with a smaller increment, but requires rebuilding distributed state, reloading the model, and warming the new deployment.

Elastic scale-up is based on principles already implemented in the recovery flow. In recovery, active membership changes unexpectedly; with scale-up, the change is planned and new GPU ranks are added at runtime. In the example below, a 20-GPU deployment grows to 24 GPUs by adding four ranks without replacing the running deployment.

![Comparison of horizontal scaling, restart-based vertical scaling, and live Elastic EP scaling from 20 to 24 GPUs.](/images/blog/elastic-ep-scaling-nixl/scaling-options.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 1: An illustrative 20→24 GPU example. Elastic EP Scaling adds four ranks to the running deployment instead of allocating another 20-GPU replica or replacing the job.</p>

## TL;DR

- Elastic EP scale-up lets an MoE model add GPUs in fine-grained increments without deploying a full replica or restarting the running deployment.

- `max_ep_size` defines the largest EP topology the deployment can reach. Metadata and communication-buffer capacity are reserved at launch, while additional GPU ranks are launched only when needed.

- In the reported comparisons, Elastic EP Scaling reached the first completed request 5.3×–8.2× faster than restart.

## How Scale-Up Builds on Recovery

During every MoE forward, token activations travel to the ranks that own the selected experts and the outputs return to their source ranks. SGLang's [large-scale EP work](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) explains why this expert dispatch path matters for high-concurrency and long-context serving. With Data Parallel (DP) attention, each logical DP worker keeps its own scheduler and KV cache while expert layers communicate across the EP topology.

Runtime scale-up builds directly on SGLang's [Elastic EP failure-recovery work](https://www.lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance). Both use active/inactive membership, Mooncake Process Group (Mooncake PG) admission, and an end-of-forward transition hook. Recovery brings a rank back into an existing slot and its original process groups. Scale-up activates unused tail slots in the max-sized `active_ranks` vector and admits the joining ranks.

One constraint shaped the design: launch-time process groups are static. The original ranks and each joining group therefore keep fixed Tensor Parallel (TP) and bootstrap groups while Mooncake `WORLD` and logical EP membership grow above them.

![EP4 to EP6 membership expansion with fixed launch groups, Mooncake WORLD, and NIXL peers.](/images/blog/elastic-ep-scaling-nixl/ep-membership.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 2: EP4→EP6 keeps the original four-rank launch group intact, admits two joining ranks into Mooncake WORLD, and expands NIXL&#x27;s target peer set.</p>

## One Logical Deployment, Three Cooperating Layers

1. **SGLang owns the serving control plane.** It waits for the requested joining ranks, tracks `pending_ep_size` separately from `effective_ep_size`, installs the target expert and DP state at the end of a forward, and exposes new scheduler slots only after commit.

1. **Mooncake PG implements elastic membership and synchronization.** Its expandable `WORLD` admits joiners and provides the barrier that covers incumbents and joiners. SGLang also uses Mooncake PG collectives and P2P to coordinate EPLB after scale-up.

1. **NIXL provides the expert data plane.** [NIXL EP](https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep) dispatches token activations and combines expert outputs on every MoE forward. SGLang's integration is in the [NIXL dispatcher](https://github.com/sgl-project/sglang/blob/77d23a796e94c37f8657bc261856db725fa35891/python/sglang/srt/layers/moe/token_dispatcher/nixl.py#L67-L117).

NIXL preserves established peer connections as SGLang grows the EP group. On EP4→EP6, the original ranks keep their existing connections and add ranks 4 and 5; the joiners connect to the EP6 peer set. SGLang configures NIXL metadata and RDMA buffers for `max_ep_size` up front, while initially connecting only active ranks. Connections are still completed lazily, so the first wider dispatch can include some setup cost.

Set `max_ep_size` to the largest topology the deployment is expected to reach. Higher ceilings reserve more communication-buffer memory on the active GPUs, leaving less memory for the KV cache.

## How a Live EP Group Grows

At a high level, SGLang prepares the joiners and then commits the wider topology:

1. **Reserve the ceiling.** The primary allocates metadata and communication-buffer capacity through `max_ep_size`. Inactive slots do not launch processes or occupy GPUs.

1. **Launch the joining ranks.** An external orchestrator allocates GPUs and starts them while the existing EP group keeps serving.

1. **Admit and commit.** SGLang waits for the complete requested rank range, expands Mooncake `WORLD`, installs the target active-rank, expert, DP, and NIXL state, and synchronizes every rank. When CUDA graphs are enabled, ranks recapture them before committing the new size. Only then does `commit_scale()` advance the effective size.

1. **Route and rebalance.** The Data Parallel Controller exposes the new logical workers. NIXL performs the first wider expert dispatch, and EPLB can redistribute experts afterward.

Automatic rollback is not yet supported after joining ranks enter Mooncake `WORLD`; a failure during finalization can require restarting the deployment.

## Measuring the Transition

We compared Elastic EP Scaling with restart on 8×H100 using DeepSeek-V3-Lite FP8. For Elastic EP Scaling, the joiners initialized in the background while the existing deployment continued serving. Restart timing begins before the source deployment is terminated. Across repeated Elastic runs, every transition completed and passed the post-scale GSM8K validation. Figure 3 reports the average Elastic time alongside a restart baseline with the same CUDA-graph capture range.

The endpoint is a short canary issued after readiness. Elastic readiness means `/is_scaling_elastic_ep` reports `serving_expanded` at the target size; restart readiness means `/health_generate` returns HTTP 200. Elastic canaries targeted the new logical DP rank, while restart used round-robin routing.

![Time to the first completed request after Elastic EP scale-up versus restart.](/images/blog/elastic-ep-scaling-nixl/transition-time.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 3: Time from the scale or restart request to the first completed request on the target topology. Elastic bars show averages across repeated runs; restart bars show comparison baselines. CUDA-graph rows capture five full-decode shapes through batch 16.</p>

Token delivery pauses during the topology transition. Figure 3 measures from the scale request through the first completed request on the expanded deployment, including any first-use connection setup.

## Roadmap

| **Installment** | **Capability** | **Status on October 4, 2026** |
| --- | --- | --- |
| 1/N | Runtime eager EP scale-up | Merged in [PR #30164](https://github.com/sgl-project/sglang/pull/30164) |
| 2/N | EPLB after scale-up | Merged in [PR #30553](https://github.com/sgl-project/sglang/pull/30553) |
| 3/N | Full-decode CUDA graph recapture | [Merged in PR #33723](https://github.com/sgl-project/sglang/pull/33723) |
| 4/N | Scale at DP-attention shard granularity | [Merged in PR #33728](https://github.com/sgl-project/sglang/pull/33728) |
| Later | Scale-down and grow-back | [Under review in PR #33111](https://github.com/sgl-project/sglang/pull/33111) |
| Later | Fully idle progress | Planned |
| Later | Recovery after scale-up | Planned |

### EPLB after scale-up

After scale-up commits, SGLang resumes the existing EPLB reshuffle on the expanded topology so the newly admitted ranks can receive experts.

### CUDA graph recapture

A graph captured at EP4 contains topology-dependent state and cannot be replayed unchanged at EP6. The original scale-up path supported eager execution. With PR #33723 now merged, SGLang also recaptures full-decode CUDA graphs for the expanded topology before commit. Prefill and piecewise recapture remain unsupported.

Across repeated runs with 256 GSM8K examples and 50 request threads, average post-scale output throughput was:

| **Transition** | **Eager** | **Full-decode CUDA graph** |
| --- | --- | --- |
| EP4→EP6 | 371 tok/s | 1,241 tok/s |
| EP7→EP8 | 358 tok/s | 1,120 tok/s |

Full-decode CUDA graphs delivered 3.1×–3.3× higher average output throughput than eager execution in this setup.

### Multi-rank DP-attention shards

The merged layout maps one physical EP rank to one logical DP worker. Follow-up [PR #33728](https://github.com/sgl-project/sglang/pull/33728) lets a DP-attention replica span `attention TP × attention CP` physical ranks, where CP is Context Parallelism. With TP2 and CP1, adding ranks 4–5 grows DP2/EP4 to DP3/EP6; adding ranks 6–7 reaches DP4/EP8. The API continues to express the target in physical EP units.

![Adding one TP2/CP1 DP-attention shard expands DP2/EP4 to DP3/EP6.](/images/blog/elastic-ep-scaling-nixl/dp-attention-shards.png)

<p style="text-align: center; color: #6b7280; font-size: 0.875rem; font-style: italic; margin-top: 0.25rem;">Figure 4: The DP-attention follow-up admits ranks 4–5 as one TP2/CP1 shard, growing DP2/EP4 to DP3/EP6.</p>

### Scale-down and recovery

Follow-up [PR #33111](https://github.com/sgl-project/sglang/pull/33111) explores scale-down and grow-back with pre-shrink and retired-rank state. Recovery after scale-up must likewise restore the last committed expanded topology rather than the launch topology.

## Trying Elastic EP Scale-Up

The scale API admits processes; it does not allocate GPUs. The operator must:

1. Launch the primary with Mooncake, NIXL, an initial EP size, and a larger `max_ep_size`.

1. Launch the joining ranks with `--elastic-ep-join-mode scale` and the next contiguous rank offset.

1. Request the absolute target and poll until `effective_ep_size` reaches it.

This example starts at EP4 and adds four ranks to reach EP8. For the full launch and validation sequence, see the [scale-up test](https://github.com/sgl-project/sglang/blob/546965fc72a73b147323fc839ad0aa955ef64aff/test/manual/ep/test_elastic_scale.py). The flags below show the scale-up settings for eager execution:

```text
primary and joiners:
  --elastic-ep-backend mooncake
  --moe-a2a-backend nixl
  --elastic-ep-initial-size 4
  --max-ep-size 8
  --enable-dp-attention --enable-dp-lm-head
  --deepep-mode low_latency
  --load-balance-method round_robin
  --cuda-graph-backend-decode disabled
  --cuda-graph-backend-prefill disabled

joiners:
  --elastic-ep-join-mode scale
  --elastic-ep-join-rank-offset 4
```

Then request and poll:

```bash
curl -X POST http://<PRIMARY_HOST>:21000/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_ep_size": 8}'

curl http://<PRIMARY_HOST>:21000/is_scaling_elastic_ep
```

The POST means accepted, not completed. A finished EP8 transition reports `effective_ep_size: 8`, `pending_ep_size: null`, `scale_phase: serving_expanded`, and no error.

## Conclusion

Elastic EP Scaling lets SGLang add GPU ranks up to `max_ep_size` without replacing the running deployment. It gives large MoE models a finer scaling unit: add only the ranks needed for the next topology while the existing ranks keep serving during preparation.

The remaining work follows the same protocol: prepare topology-dependent state, synchronize every member, and expose the new topology only after commit. Wider DP-attention layouts, scale-down, idle progress, and recovery after scale-up build on the scale-up and graph-recapture work already merged.

## References

- [Runtime EP scale-up, SGLang PR #30164](https://github.com/sgl-project/sglang/pull/30164)

- [EPLB after scale-up, SGLang PR #30553](https://github.com/sgl-project/sglang/pull/30553)

- [Full-decode graph recapture, SGLang PR #33723](https://github.com/sgl-project/sglang/pull/33723)

- [DP-attention shard scaling, SGLang PR #33728](https://github.com/sgl-project/sglang/pull/33728)

- [Scale-down and grow-back, SGLang PR #33111](https://github.com/sgl-project/sglang/pull/33111)

- [Elastic EP Partial Failure Tolerance](https://www.lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance)

- [NIXL EP scale-up buffer fix, NIXL PR #1451](https://github.com/ai-dynamo/nixl/pull/1451)

## Acknowledgments

Cheng Wan and Shangming Cai helped land the upstream integration, and Xun Sun built the recovery machinery this work extends. Thanks also to the SGLang, NIXL, and Mooncake teams for reviewing and supporting the integration.
