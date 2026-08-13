---
title: "Infer-forge: Loop and Graph Engineering Around SGLang"
author: "Tianyu Zhang, Hanlin Gao, Yusong Gao, Yun Zhang"
date: "August 7, 2026"
previewImg: /images/blog/infer-forge-loop/cover.png
type: blog
---

## 1. Introduction

**Inference optimization may look local in code, but its validity is global.** A kernel, communication path, or scheduling change becomes meaningful only at a specific deployment point defined by the model, workload, SLO, serving topology, runtime version, and accelerator platform. The same patch may improve one point and regress another. Agent-assisted exploration can produce more environments, experiments, measurements, and rejected paths, all of which must remain reproducible.

**The first requirement is therefore reliable execution.** Reproducing a deployment point requires more than model capability: tools, environments, context, memory, Verification, and safety boundaries must remain stable. **Harness Engineering** turns those surrounding conditions into a reproducible and inspectable execution system—the basis for the abstraction **Agent = Model + Harness**<sup>[1](#ref-1),[2](#ref-2),[3](#ref-3),[6](#ref-6)</sup>.

**Reliable execution must then remain coherent over time.** Inference engineering Tasks often span repeated rounds of investigation, implementation, deployment, Evaluation, failure, and recovery. **Loop Engineering** connects successive executions so that one Task can preserve its Task Contract, incorporate new evidence, and satisfy its Exit Criteria by producing either a verified Deliverable or a reliable Follow-up Handoff<sup>[4](#ref-4),[5](#ref-5),[7](#ref-7)</sup>.

**Project-scale work exceeds the boundary of one Task.** Multiple Tasks must proceed in parallel, exchange Deliverables, share state, trigger Rework, and change direction as evidence accumulates. **Graph Engineering** organizes independently convergent Task Loops into an evolving **Task Graph**. The graph keeps released and rejected paths connected to their dependencies, constraints, and evidence, so project decisions remain explainable<sup>[10](#ref-10),[11](#ref-11),[12](#ref-12),[13](#ref-13)</sup>.

**Infer-forge is our implementation of this progression for inference engineering around SGLang.** Its scope follows an engineering change through the inference stack: from kernels and communication libraries, through engine integration and deployment, to Evaluation and online diagnosis. One shared workspace and three accumulating execution structures keep that end-to-end path coherent:

- **MonoRepo** establishes a reproducible workspace for cross-repository engineering.
- **Harness** supplies reusable execution capabilities, memory, Verification, and safety boundaries.
- **Task Loop** keeps one long-running Task moving until it reaches its Exit Criteria.
- **Task Graph** connects independently convergent Tasks into larger objectives, including project delivery and capability evolution.

**Infer-forge has moved from workflow design into sustained engineering use.** Across one engineer's April–July record, the observed peak number of **Tasks in flight** rose from **2 to 9**. In one DeepSeek-V4-Pro serving project, **38 independently verifiable Task nodes across seven Task Types** were coordinated as a Task Graph. Together, these records show infer-forge in sustained use across both a four-month engineering record and a project-scale Task Graph.

**A capable Agent can make one execution succeed; an engineering system is designed to make successful work reproducible.** Infer-forge does not promise that every Task will finish faster. It provides the structure to preserve the provenance of each deployment point, sustain verifiable work over long-running Tasks, and coordinate evidence across Task boundaries.

## 2. Inference as a Deployment Space

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-01-inference-deployment-space.svg" alt="The inference deployment space stacks five layers of static configuration. Model shows Ling, Qwen, DeepSeek, Kimi, GLM and MiniMax. Serving Scenario runs Modality (text, image, video) into Traffic Shape (input and output length, media count, resolution, QPS, concurrency, cache reuse) into SLO (TTFT, TPOT, throughput, E2E latency). Serving Topology separates Deployment Architecture—Colocated PD, PD Disaggregation and EPD Disaggregation, each listing the node roles it is built from — Prefill and Decode together, then Prefill and Decode as separate roles, then Encoder alongside them — from Parallelism—TP, PP, DP and EP—because the two are chosen independently. Versioned Runtime Profiles is a stack of tabbed cards labelled Service A rev. 12, Service B rev. 7 and Service C rev. 21, plus a fourth paler card behind them all, blank and showing only its top edge, whose narrow tab carries an ellipsis for the profiles not drawn, the front card holding an Engine Configuration and a Container Image whose digest is pinned alongside its Framework, Device Runtime and Collectives. Accelerator Platforms groups placeholder GPUs under Vendor A, B and C. Arrows between the layers carry configuration dependency, not runtime data flow" />
  <br>
  <em>Figure 1: The Constraint Chain Behind an Inference Deployment Point.</em>
</div>

Figure 1 turns the deployment point introduced above into a concrete chain of constraints. The **Model** determines the supported modalities and model-specific execution paths. The **Serving Scenario** translates **Modality** and **Traffic Shape** into an **SLO**. That SLO constrains the **Serving Topology**, where a **Deployment Architecture** is combined with a **Parallelism** strategy. The topology is then realized through a **Versioned Runtime Profile** that pins the engine configuration and container image for a particular service revision. Finally, the complete runtime must be built and verified on an **Accelerator Platform**. A deployment point is the full path through this chain—not any one layer in isolation.

**There is no context-free inference optimization.** Different serving objectives and SLOs can require fundamentally different deployment paths, including `Colocated PD`, `PD Disaggregation`, and `EPD Disaggregation`. Each architecture changes stage boundaries, communication paths, resource balance, and the set of feasible Parallelism strategies. Those decisions propagate into the runtime profile and accelerator-specific implementation that must be verified. A kernel improvement becomes a serving result only when the complete deployment point reproduces it and passes throughput, latency, correctness, and stability gates. **Without its deployment point, a performance claim cannot be reproduced, compared, or carried forward.**

**Infer-forge does not eliminate this combinatorial space; it makes every movement through it explicit and verifiable.** Instead of asking an Agent to “optimize DeepSeek-V4-Pro,” we define a Task that records the current deployment point, bounds the subset of dimensions it may change, and fixes the Verification gates before execution begins. A Task might replace the MoE backend while holding the serving scenario, topology, and accelerator constant, then produce either an accepted improvement or a documented rejection. Both outcomes reduce uncertainty for the next Task. Before any of them can be reproduced, however, the exact cross-repository code state behind the deployment point must be fixed. That is the role of the MonoRepo.

## 3. MonoRepo

### 3.1 Why a MonoRepo

**Inference optimization crosses repository boundaries, but it must ship as one coherent system.** A change may begin in a kernel library, depend on a communication backend, enter SGLang through engine integration, and finally require a matching deployment configuration. When these repositories live in separate workspaces, their relationships become transient knowledge that engineers and Agents must repeatedly reconstruct. One missing branch or incompatible revision is enough to invalidate the result.

**Infer-forge turns that dependency map into a shared workspace.** Git submodules place the relevant repositories under one root while preserving their independent histories, branch policies, access controls, and release processes. The root gives engineers and Agents a stable map of the inference stack and one entry point from which cross-repository work can be developed and verified.

This structure changes cross-repository work in three ways:

- **The whole stack stays in view.** Repository boundaries remain explicit, but their engineering relationships are visible from one workspace.
- **Context stays bounded.** An Agent can navigate the full repository map while loading only the repositories required by the current Task.
- **Integration starts in place.** Related branches can be developed, combined, and verified without repeatedly locating repositories or rebuilding their relationships from memory.

**The workspace coordinates change; the Task record makes it durable.** Each Task captures the branches, commits, and execution state used for the work, and the Journal archives that record after completion. The repositories can continue evolving without erasing the provenance of work that has already been verified and delivered.

### 3.2 Repository Map

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-02-monorepo.svg" alt="The infer-forge MonoRepo contains three groups: a Built-in Workspace; Inference Stack Repos centered on SGLang, including Dynamo, DeepGEMM, DeepEP, FlashMLA, FlashInfer, Humming, and Mooncake; and Harness Repos" />
  <br>
  <em>Figure 2: Infer-forge MonoRepo.</em>
</div>

**One workspace does not mean one undifferentiated codebase.** Infer-forge separates repositories by the role they play in engineering: the **Built-in Workspace** coordinates cross-repository work, the **Inference Stack Repos** contain the serving system being changed, and the **Harness Repos** carry those changes from execution to verified evidence.

#### Built-in Workspace

**The Built-in Workspace is the coordination layer of infer-forge, not another implementation repository.** Code remains in the repository that owns it. The root contains only the mechanisms that need to operate across repository boundaries:

- **Task System** materializes the workspace required by a Task: the relevant repositories, environment entry points, record locations, and—when isolation is required—a dedicated worktree and Agent session.
- **Cross-lib Management** makes the repository graph operable. It maintains repository locations, default branches, and the operations used to synchronize and integrate changes. The actual branches and commits used for a piece of work belong to the Task record and are archived into the Journal.
- **Skills** expose capabilities at the scope where they belong. The root provides Skills for Task lifecycle and cross-repository coordination; each repository retains the Skills specific to its own domain.

#### Inference Stack Repos

**Inference Stack Repos are where serving behavior and performance actually change.** [SGLang](https://github.com/sgl-project/sglang) is the center of the stack, while [Dynamo](https://github.com/ai-dynamo/dynamo) organizes SGLang instances into a distributed service. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), and [Humming](https://github.com/inclusionAI/humming) provide specialized compute kernels. [DeepEP](https://github.com/deepseek-ai/DeepEP) and [Mooncake](https://github.com/kvcache-ai/Mooncake) provide expert-parallel communication and cross-node KV transfer. These repositories evolve independently, but one serving result may depend on changes across several of them at once.

#### Harness Repos

**Code does not become an engineering result merely because an Agent can edit it.** Harness Repos provide the capabilities that carry a change through the rest of its lifecycle: finding compute resources, preparing environments, deploying the service, running performance and correctness Evaluation, diagnosing failures and online behavior, preserving long-term records, and enforcing safety boundaries. They turn otherwise disconnected operational steps into a repeatable path from code change to verified Deliverable.

**Together, the three groups form one engineering path: the Built-in Workspace prepares and coordinates the work, the Inference Stack Repos supply the system under change, and the Harness Repos carry that change to Verification.** Infer-forge brings them into one workspace without forcing them into one repository history, ownership model, or release process.

## 4. Task Loop

**The MonoRepo provides the workspace; the Task Loop structures how work advances over time.** Inference engineering often requires multiple rounds of Research, implementation, deployment, Evaluation, and recovery. The Task Loop keeps those executions aligned with one Goal and Task Contract until they satisfy their Exit Criteria through a verified Deliverable or a reliable Follow-up Handoff.

**“Everything Can Be a Task” applies to independently verifiable units of work, not to every action.** A Task needs its own Goal, Scope, Acceptance, Verification path, and Exit Criteria. Commands and intermediate experiments remain inside it as Loop Block instances or tool calls. Once a unit of work can complete or hand off independently, it can become a Task node in a larger Task Graph.

### 4.1 Overview

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-03-task-loop-overview.svg" alt="The Task Loop moves from Task Definition into Main Loop, uses Task Goal Met? to continue or satisfy Exit Criteria, preserves Task Memory, and draws on four Harness capabilities" />
  <br>
  <em>Figure 3: Task Loop Overview.</em>
</div>

**A Task Loop keeps its boundary stable while allowing its execution path to adapt.** **Task Definition** establishes the Task Type, Starting Context, Task Contract, and Exit Criteria. The **Main Loop** advances the work through a sequence of **Loop Block** instances and uses **Task Goal Met?** to decide whether to exit or continue. **Task Memory** preserves the current state, next Sub-target, Execution Records, and Handoff between iterations.

**The Harness surrounds the loop rather than acting as another stage inside it.** It supplies the resources, methods, memory systems, Verification entry points, and safety boundaries required during execution. Different Tasks can use those capabilities in different orders without being forced through one fixed pipeline.

### 4.2 Task Definition

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-04-task-definition.svg" alt="Task Definition consists of Task Type, Starting Context, Task Contract, and Exit Criteria" />
  <br>
  <em>Figure 4: Task Definition.</em>
</div>

**A Task Loop begins with a testable commitment, not with activity.** Before an agent searches the codebase, launches a deployment, or consumes evaluation capacity, the Task must state what it is trying to accomplish and how the result will be judged. **Task Definition** establishes four anchors:

- **Task Type** selects a default Playbook.
- **Starting Context** records the state from which execution begins.
- **Task Contract** fixes the Goal, Scope, Acceptance, and Verification.
- **Exit Criteria** define the durable **Deliverable** or **Follow-up Handoff** that execution must produce.

#### 4.2.1 Task Type

**A Playbook turns prior practice into a head start.** For each Task Type, it provides a lightweight, adaptable template: which repositories are likely to matter, which **Skills**, **Tools & CLI**, and Verification entry points are relevant, and which safety boundaries apply. This reduces the space the agent must search before useful work can begin.

**A Playbook standardizes the starting approach, not the complete execution path.** The agent may change methods, introduce new Loop Block instances, or use capabilities outside the template as the **Task Contract** and new evidence require. It reduces repeated discovery without replacing engineering judgment.

**Each Task Type captures a distinct unit of verifiable work in inference engineering:**

- **Plan** defines objectives, constraints, and decomposition, then delivers an executable plan for downstream Tasks.
- **Research** investigates a bounded uncertainty, evaluates available evidence, and delivers conclusions with explicit limitations.
- **Code** implements any scoped code change and delivers the change together with its Verification evidence.
- **Integration** assembles changes from multiple upstream Tasks, including across repositories or components, and verifies the combined system.
- **Evaluation** runs functional, performance, accuracy, stress, and stability assessments and delivers evidence against Acceptance.
- **Release** advances a verified candidate through approval, canary rollout, expansion, or rollback and records the outcome.
- **Online Diagnosis** investigates a production issue within read-only boundaries and hands required changes to downstream Tasks.
- **Capability** maintains the versioned **Skills & Tools** set through `Add`, `Update`, `Merge`, `Retire`, or `No Change`.
- **+ Custom** handles verifiable work without a reusable Playbook while preserving the same Task Definition.

**The taxonomy is earned through repeated practice.** An activity becomes a built-in Task Type only after its starting conventions, Acceptance boundary, and Deliverable are stable enough to guide future work. Standardization follows proven practice; it does not attempt to predict every path in advance.

#### 4.2.2 Starting Context

**Starting Context draws the line between known state and assumption.** A Task may begin from a blank state, a prepared environment, or an upstream result. What matters is that the source and verification status of that state remain explicit.

- **Custom Setup** records conditions prepared for the current Task, such as the model, container image, deployment profile, repository versions, dataset, or experiment entry point.
- **Imported Context** carries forward an upstream environment, intermediate result, or **Follow-up Handoff**, together with its provenance and verification status.

#### 4.2.3 Task Contract

**The Task Contract holds the target still while execution adapts.** **Goal** states the intended outcome, **Scope** bounds the work, **Acceptance** states the observable conditions for success, and **Verification** defines the evidence required to judge them<sup>[6](#ref-6)</sup>. Without this boundary, an agent can appear to succeed simply by changing the problem after seeing the result.

For an inference optimization Task, the contract may fix the model and weight format, hardware placement, serving topology, runtime versions, workload grid, and baseline. Acceptance can then require better throughput, TTFT, or TPOT without violating accuracy or stability. Changing the topology, GPU class, or workload after observing the result is a contract change—not an optimization result.

**Evidence may redirect the Main Loop, but it must not silently move the finish line.** A clarification that does not materially change the Goal, Scope, or Acceptance must be explicit and recorded; a material change to any of them starts a new Task.

#### 4.2.4 Exit Criteria

**A Task does not end because activity stops. It ends when its state can be verified or safely continued.**

- A **Deliverable** packages the completed result with enough evidence to verify the Task Contract.
- A **Follow-up Handoff** preserves completed work, current state, unresolved questions, and the next entry point when another Task must continue the work.

**Exit Criteria turn execution into durable engineering state: either a verifiable result or a reliable starting point for the next Task.**

### 4.3 Loop Execution

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-05-loop-execution.svg" alt="Loop Execution defines a Loop Block, uses Execution Routing to choose Model Tier and Agent Topology independently, executes the block, and uses Task Goal Met? to exit or continue while Task Memory carries the Current Loop Block, a Loop Block Handoff and the Next Loop Block across iterations" />
  <br>
  <em>Figure 5: Loop Execution.</em>
</div>

**A Task is sized by its engineering objective, not by the duration of a runtime session.** The Task Contract defines the complete objective; the Main Loop realizes it through one or more **Loop Block** instances as evidence emerges. When defining a Task, the user does not need to predict whether it can finish within one sustained runtime loop.

**The Task is the unit of continuity; the Loop Block is the unit of execution.** Each Loop Block owns one Sub-target, one Exit Condition, one routing decision, and one set of Execution Records. Claude Code `/loop` and Codex `/goal` are current execution entry points, but neither determines the scope of the Task.

#### 4.3.1 Main Loop

**The Main Loop turns an open-ended Task into a sequence of evidence-producing Loop Blocks:**

- **Define Loop Block** selects the current Sub-target and Exit Condition.
- **Execution Routing** selects the Model Tier and Agent Topology for that Sub-target.
- **Execute Loop Block** runs the block through an available sustained-execution entry point.
- **Task Goal Met?** compares the accumulated state with the Task Contract and decides whether to exit or define another block.

**Completing a Loop Block is not the same as completing the Task.** A block may confirm a hypothesis, reject an approach, or expose a new constraint. Its Exit Condition closes that local unit of work; **Task Goal Met?** determines whether the accumulated evidence satisfies the whole Task Contract. The Yes branch proceeds to Exit Criteria, while the No branch defines the next Loop Block from the evidence already produced.

#### 4.3.2 Task Memory

**Task Memory makes the Loop Block sequence durable and traceable.** It preserves completed Loop Blocks and the active Current Loop Block, including its Sub-target and Exit Condition. A Next Loop Block is recorded only after the current block ends and **Task Goal Met?** determines that the Task must continue.

**Execution Records capture what the agent did at each step and what result it produced.** They make the execution auditable, traceable, and reproducible.

**A Loop Block Handoff transfers state inside one Task; a graph-level Handoff edge connects independent Task node instances.**

#### 4.3.3 Execution Routing

**Execution Routing makes two orthogonal choices for the current Loop Block:**

- **Model Tier** follows reasoning difficulty. Uncertainty, reasoning depth, and required expertise determine whether the block uses **Lower-tier**, **Mid-tier**, or **Strongest Available**.
- **Agent Topology** follows content volume and expected context length. Work that fits reliably within one context uses **Single Agent**; work that must be divided across contexts uses **Multi-Agent**<sup>[14](#ref-14)</sup>.

A difficult but compact problem may use **Strongest Available** with **Single Agent**. A large but routine evaluation matrix may use **Lower-tier** or **Mid-tier** with **Multi-Agent**. Model Tier supplies reasoning capability; Agent Topology manages context load.

Figure 5 shows one possible **Multi-Agent** topology: a **Coordinator** works with Infra, Code, and Eval, while **Reviewer** examines the result outside the main execution chain<sup>[7](#ref-7)</sup>. The roles are illustrative; the actual decomposition follows the content and context requirements of the current Loop Block.

If model-tier selection or subagents are unavailable, the Loop Block proceeds with the capabilities provided by the runtime, with the fallback preserved in its Execution Records.

### 4.4 Harness

**A Task Loop can sustain progress only if the Harness can sustain execution.** Without a stable Harness, every Loop Block must rediscover machines, reconstruct environments, locate commands, recover evidence, and renegotiate safety boundaries. Infer-forge instead provides one execution substrate built from **Node Registry**, **Skills & Tools**, **Journal**, and **Safety Guard**.

**The value lies in composition, not in any capability alone.** After Task Definition, an agent can inspect resource state, prepare an environment, deploy a workload, modify and evaluate the system, and recover on another Node when necessary. Node state, operational methods, execution history, and safety constraints remain connected throughout the process, so a human does not have to rebuild the path between steps.

#### 4.4.1 Node Registry

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-06-node-registry.svg" alt="Node Registry records Task-to-Node claims in a Git Ledger, compares them with DCGM Observation, and uses Idle Reclaim to clean up stale records" />
  <br>
  <em>Figure 6: Node Registry.</em>
</div>

**Resource autonomy cannot begin from a guess about Node ownership.** The **Node Registry** is a Git-backed repository that reconciles declared occupancy with observed machine activity:

- **Git Ledger** records which Task claims each Node. Every claim, release, and correction remains versioned and auditable.
- **DCGM Observation** reports whether a workload is actually running on the machine.

Reconciliation distinguishes **Held by Task**, **Idle < 6h**, and **Idle ≥ 6h → reclaim**. Under the current **Idle Reclaim** policy, a claim observed idle for six continuous hours is removed and the Node becomes available again. The interval is a governance parameter, not a scheduling guarantee.

**The Node Registry does not select Nodes, queue work, or launch workloads. It gives those actions a trustworthy and auditable starting point.**

#### 4.4.2 Skills & Tools

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-07-skills-and-tools.svg" alt="Skills are organized across SGLang Upstream, Cross-lib, Task, and Ops, while Tools & CLI include Deploy, Build, Pull Weights, Sync Code, Evaluate, Profile, Diagnose Online, and Monitor" />
  <br>
  <em>Figure 7: Skills & Tools.</em>
</div>

**A capable agent should not rediscover how to perform the same engineering work for every Task.** **Skills** preserve reusable approaches across SGLang Upstream, Cross-lib, Task, and Ops. **Tools & CLI** expose stable, recordable interfaces for deployment, build, weight and code synchronization, Evaluation, profiling, online diagnosis, and monitoring.

A Playbook selects the relevant subset for the current Task Type. Skills narrow the decision space; Tools & CLI turn the selected decision into an action whose inputs and outputs can enter Execution Records. The agent starts from accumulated practice without being forced through a fixed path.

**Skills & Tools is a versioned capability baseline, not a growing pile of instructions.** Ordinary Tasks use the current baseline. A Capability Task uses Journal evidence and revalidation after a `Model or runtime change` to decide what should be added, updated, merged, retired, or left unchanged.

#### 4.4.3 Journal

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-08-journal.svg" alt="Journal uses LLM-wiki to connect records from multiple Tasks and Multi-dim Index fields such as Model, Type, and GPU to support Retrieve, Compare, and Filter" />
  <br>
  <em>Figure 8: Journal.</em>
</div>

**Evidence compounds only when the next Task can find and reuse it.** Task Memory preserves the execution state of one Task; the **Journal** carries evidence across Tasks. **LLM-wiki** connects Task records into a knowledge network, while **Multi-dim Index** organizes them by dimensions such as model, Task Type, and GPU. Together, they support Retrieve, Compare, and Filter without forcing each Task to rediscover the same facts.

In one benchmark Task, the Journal surfaced an earlier record showing that `bench_serving.py` did not count `delta.reasoning_content` for reasoning models. That bug invalidated the apparent TTFT and throughput baseline. Reusing the record prevented the next Task from treating a measurement error as an engine regression.

**The Journal preserves evidence; it cannot promote evidence directly into executable capability.** Any change to the Skills & Tools baseline must pass through a Capability Task and Verification.

#### 4.4.4 Safety Guard

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-09-safety-guard.svg" alt="Safety Guard constrains execution through Push Guard, Traceable Path, Env Isolation, Production Read-only Access, Secrets, Data, Human Gate, and Cross-Model Adversarial Review" />
  <br>
  <em>Figure 9: Safety Guard.</em>
</div>

**As agent execution becomes longer and more concurrent, an unenforced mistake can travel farther.** **Safety Guard** therefore applies constraints that no Playbook or Loop Block may bypass:

- **Push Guard** and **Traceable Path** constrain how code changes move.
- **Env Isolation** and **Production Read-only Access** constrain environments and production operations.
- **Secrets** and **Data** constrain access to credentials and governed datasets.
- **Human Gate** requires approval for high-risk actions.
- **Cross-Model Adversarial Review** applies `reviewer ≠ coder` to reduce self-review blind spots<sup>[6](#ref-6)</sup>.

**Safety Guard constrains actions; Verification constrains claims.** In one kernel optimization Task, a candidate appeared to reach `72.30 TFLOPS`, a `5.7%` improvement over its comparison point. A later Verification exposed a race: aggregate statistics looked stable while individual elements were corrupted. The candidate was rejected before Integration.

**Reliable autonomy is measured not only by what it completes, but also by what it refuses to advance.** In inference engineering, stopping a false performance win can be more valuable than producing another patch.

## 5. Task Graphs

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-10-task-graph.svg" alt="Task Graph defines Task node, Shared repo, and External system as graph elements, shows Verification before Handoff, and distinguishes Handoff edge, State edge, and Control edge" />
  <br>
  <em>Figure 10: Task Graph: Elements, Handoff, and Edge Types.</em>
</div>

**A project does not scale merely by creating more Tasks. It scales when dependencies, shared state, and control decisions become explicit.** A Task Loop gives one Task an independent path to convergence; a Task Graph connects those convergent units without dissolving their Verification boundaries.

**A Task Graph is not a fixed workflow template or a closed taxonomy.** Whenever multiple Tasks need explicit dependencies, verified Handoffs, shared state, or control relationships, they can be organized into a Task Graph. The graph can serve one project, one investigation, one release, or any other objective that exceeds a single Task boundary.

**Delivery Graph and Capability Graph are two recurring examples, not the only valid graph structures.** We use them to demonstrate the same graph language in two common situations: coordinating Tasks that jointly change the inference system, and maintaining the capabilities used by future Tasks.

**Graph correctness begins by separating execution from state and external control.** Only a **Task node** executes work and runs a Task Loop. A **Shared repo** stores persistent state without running a loop. An **External system** represents people or platforms that interact with the graph from outside infer-forge.

The edge type states what crosses each boundary:

- A **Handoff edge** carries a verified Deliverable or Follow-up Handoff that becomes Imported Context for the successor Task.
- A **State edge** represents a read from or write to a Shared repo; it does not imply Task completion or Verification.
- A **Control edge** triggers work, returns it for revision, or changes its direction without carrying a verified result.

**A write is not a Handoff; a trigger is not a Deliverable; connectivity alone is not verified progress.**

### 5.1 Delivery Graph

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-11-delivery-graph.svg" alt="Delivery Graph fans a Plan Task out to parallel Code Tasks, converges them through Integration, Evaluation, and Release Tasks, and uses external Signals plus Feedback, Rework, and Hotfix to change subsequent work" />
  <br>
  <em>Figure 11: Delivery Graph: The Delivery Lifecycle.</em>
</div>

**Inference delivery is a convergence problem, not a checklist.** A Plan Task can divide the objective into Prefill, Decode, kernel, communication, and deployment workstreams. Research and Code Tasks then advance independently and in parallel, each producing evidence under its own Task Contract.

**Integration is where parallel work becomes one runnable deployment point.** It assembles changes across the engine, libraries, image, and deployment configuration. The combined candidate then enters Evaluation under fixed model, Serving Scenario, hardware, performance, accuracy, and stability conditions. A failed result returns through `Rework` rather than advancing on partial success.

**Release is a verified state transition, not the last box in a diagram.** Only a candidate that passes Evaluation can reach Release through a Follow-up Handoff. External `Signals` may trigger Online Diagnosis; its findings return through `Feedback` or start a Code Task through `Hotfix`. These Control edges can redirect work, but they cannot bypass Integration or Evaluation.

**A Delivery Graph makes parallelism useful by forcing every path to converge through evidence.**

### 5.2 Capability Graph

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-12-capability-graph.svg" alt="In the Capability Graph, Journal commit event, Scheduled trigger, and Model or runtime change wake Capability Task; it scans evidence, reconciles the current Skills & Tools baseline, and after Verification produces Add, Update, Merge, Retire, or No Change, with Follow-up Handoff available for further Tasks" />
  <br>
  <em>Figure 12: Capability Graph: Capability Evolution.</em>
</div>

**The Delivery Graph changes the inference system; the Capability Graph changes the system that performs the work.** Task Execution uses the current Skills & Tools baseline and records practical evidence in the Journal. Across Tasks, repeated steps, reusable commands, workflow patterns, and proven fixes become candidates for capability maintenance.

**The Journal is evidence, not authority.** `Journal commit event`, `Scheduled trigger`, and `Model or runtime change` can wake a Capability Task, but none of them proves that executable behavior should change. The Task scans the Journal delta, reads the current baseline, and uses Verification to produce `Add`, `Update`, `Merge`, `Retire`, or `No Change`.

**Every capability has a carrying cost and a shelf life.** Some Skills preserve durable project knowledge; others compensate for a particular model or runtime. As those systems improve, old scaffolding may consume Context, conflict with newer behavior, or constrain judgment. The Claude Code team reports removing more than 80% of its system prompt for newer models with no measurable loss on its coding evaluations<sup>[15](#ref-15)</sup>. Boris Cherny has separately advocated periodically pruning `CLAUDE.md` files, Skills, and hooks<sup>[16](#ref-16)</sup>.

A `Model or runtime change` therefore triggers revalidation of the current baseline. Capability removal is an evidence-driven ablation method, not blanket deletion; safety boundaries and verified invariants remain unless evidence supports changing them. **A capability set that supports only `Add` does not evolve—it accumulates debt.**

If a candidate requires implementation, Evaluation, or Release work, the Capability Task produces a Follow-up Handoff to the corresponding Task. **The Capability Graph prevents raw experience from mutating executable behavior while ensuring that verified experience does not remain trapped in the Journal.**

## 6. One Engineer, Multiple Loops

**This chapter follows one AI infrastructure engineer from sustained Task execution to concurrent work and project-scale coordination.**

### 6.1 From Execution to Judgment

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-13-parallel-task-loops.svg" alt="One Human and four Agents share a single left-to-right timeline, each on its own lane. The Human lane is one continuous track carrying Define Task A, Define Task B and Define Task C, then a plain grey block with no label for a stretch of unrelated engineering work, then Review Evidence A and Define Task D. Each Define Task block is filled with the colour of the Agent it starts, and a dashed vertical line in that same colour runs from its lower right corner down to the top left corner of the matching Execute Task bar; where such a line crosses a bar that is still running, it is drawn over a white channel so it stays legible. The four Agent lanes are each labelled simply Agent — the letters belong to the Tasks, not to the Agents — and are told apart by colour. Execute Task A runs from the end of its definition through the whole stretch in which the Human defines two more Tasks and turns to other work, and a green arrow rises from the bar's right edge — the moment it finishes — to Review Evidence A, closing one loop. Execute Task B, Execute Task C and Execute Task D all continue to the right edge of the time axis, so they are still running when the figure ends. A faded lane labelled More Agents holds dashed placeholder bars, showing that more can be started" />
  <br>
  <em>Figure 13: One Engineer Directs Multiple Sustained Task Loops.</em>
</div>

**AI infrastructure engineering changes when execution can continue while the engineer is elsewhere.** Once an engineer has completed Task Definition and established the necessary Human Gates, an Agent can carry the work through environment setup, deployment, Evaluation, recovery, and iteration over hours or days. The Task Loop keeps that execution connected to the Task Contract, its evidence, and the next decision.

Figure 13 shows the resulting working mode. While one Task Loop is still executing, the engineer can define another Task, review completed evidence, or return to unrelated engineering work. The engineer no longer has to advance every command sequence personally; scarce attention shifts from execution to judgment: setting constraints, evaluating evidence, resolving tradeoffs, and deciding what should proceed next.

**One engineer can therefore direct multiple sustained Task Loops without blurring the boundary of each Task.** The Harness carries execution forward; the engineer concentrates attention where it changes the project's direction.

### 6.2 Observed Tasks in Flight

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-14-task-activity.svg" alt="Archived Task Lifetimes plots each of the 86 archived Tasks as one horizontal bar running from its creation to its archive time, between April 7 and July 30, 2026. Bars are packed into nine lanes by earliest free lane, and are coloured across all nine Task Types—Plan, Research, Code, Integration, Evaluation, Release, Online Diagnosis, Capability and + Custom. A dark slate dashed line at July 6, drawn over a white underlay so it stays legible where it crosses coloured bars, marks the peak, where nine Tasks are in flight; its label reads Peak in flight: 9. Three cards below the timeline give the summary statistics: 90 Tasks created, 86 valid archived lifetimes, and 91% in flight with others — the last one filled solid because it is the conclusion the other two support" />
  <br>
  <em>Figure 14: Tasks in Flight, April–July 2026.</em>
</div>

**Figure 14 shows how the engineer's Tasks in flight changed over four months.** Each bar spans from a Task's creation timestamp to its archive timestamp, and the peak is the maximum number of these intervals that overlap at any instant. Of the 90 Tasks created during the observation window, 86 archived Tasks had valid timestamps and are included; three were still open at the cutoff, and one record was excluded because its archive timestamp preceded its creation timestamp.

Across the April–July record, the median archived lifetime increased from approximately **10 hours in April**, to **14 hours in May**, **20 hours in June**, and **28 hours in July**. The monthly peak number of **Tasks in flight** was **2**, **2**, **6**, and **9**, respectively.

### 6.3 Project-Scale Coordination

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-15-deepseek-v4-pro-task-graph.svg" alt="A Task Graph from one DeepSeek-V4-Pro serving delivery connects SERVING BASELINE, PREFILL, DECODE, and RELEASE; dashed nodes mark paths that did not enter the final release, adjacent labels explain each outcome, an amber Control edge records a cross-workstream constraint, and SHARED STATE connects Journal, Capability Task, and Skills & Tools" />
  <br>
  <em>Figure 15: A Task Graph Coordinates a Project-Scale Serving Delivery.</em>
</div>

**One engineer coordinated *Pushing the Limits of Serving DeepSeek-V4-Pro* through 38 independently verifiable Task nodes across seven Task Types.** The graph organized four serving workstreams—short-context Prefill, long-context Prefill, low-latency Decode, and high-throughput Decode—so they could progress independently while sharing the decisions, constraints, and Verification evidence required for one Release. The four workstreams converged on four different deployment points rather than one universal optimum.

One Decode Task remained open for **nine days** while other workstreams continued elsewhere in the graph. The graph prevented this long-running Task from blocking the project. No single Agent or Task had to retain the whole project context: each workstream could converge against its own Task Contract, then contribute a verified Deliverable to the larger Release.

**The project produced four released serving profiles and preserved seven paths that did not enter the final Release.** Those paths were part of the engineering result: they recorded what had been evaluated, why it did not advance, and what downstream Tasks should not need to rediscover.

**This is how the operating model scales: one engineer can direct multiple sustained Task Loops, converge them through a Task Graph, and preserve both the released system and the decisions behind it.**

## 7. Layers Accumulate

<div align="center">
  <img src="/images/blog/infer-forge-loop/fig-16-layers-accumulate.svg" alt="Context becomes the execution core of Harness, Harness remains inside Task Loop, and Task nodes that preserve these layers compose into Task Graph" />
  <br>
  <em>Figure 16: Layers Accumulate: From Context to Task Graph.</em>
</div>

**Harness turns execution-local Context into durable structure for work that must continue.** Context holds the goal, code, current state, and working judgments needed for one execution. A Harness selects what deserves to outlive that execution, externalizes it for reuse, and supplies stable tools, environments, memory, verification entry points, and safety boundaries<sup>[9](#ref-9)</sup>. Figure 16 is an accumulation, not a succession: Context becomes the execution core of the Harness; the Harness remains the load-bearing foundation of each Task Loop; and Task Graphs connect those Task Loops into project-scale work. Every higher layer extends the reach of the layers below without removing their capabilities or constraints.

**Abstractions can change in months; reliability is earned layer by layer.** The representative essays cited here helped crystallize Harness Engineering, Loop Engineering, and Graph Engineering as a shared vocabulary in just over five months<sup>[2](#ref-2),[4](#ref-4),[13](#ref-13)</sup>, but their dependency is fundamental rather than chronological. Incomplete Context can compromise one execution. A weakness in the Harness—stale knowledge, a faulty tool, or a missing constraint—can recur throughout a Task Loop and propagate across a Task Graph. Higher layers amplify both the strengths and the weaknesses beneath them. **Loop and Graph Engineering do not supersede Harness Engineering; they increase how much depends on it.**

## 8. Lessons Learned

### 8.1 Task Granularity

**A Task boundary should be drawn where Verification can stand on its own.** A Task needs its own Task Contract—Goal, Scope, Acceptance, and Verification—and its own Exit Criteria. Work that changes only the Sub-target or Exit Condition while the Task Contract remains intact belongs in another Loop Block. Work that requires a materially different Goal, Scope, or Acceptance belongs in a new Task. A Task that is too large hides independent evidence; one that is too small turns execution into coordination overhead.

The lesson from the nine-day Decode Task was not that nine days is inherently too long. Experiments governed by the same performance, accuracy, and stability contract belonged in successive Loop Blocks; directions requiring an independent Goal, Scope, or Acceptance should have become separate Tasks. A practical test is to ask where execution should return after failure: the next Loop Block, or a new Task Definition?

**Split on a change in the contract, not the passage of time.**

### 8.2 Evidence-Driven Task Graphs

**A Task Graph should absorb evidence, not freeze assumptions.** A Plan Task defines the most credible initial decomposition, but the graph must be able to evolve as evidence changes<sup>[13](#ref-13)</sup>. Research may invalidate a path, Evaluation may produce `Rework` or a new Code Task, and Feedback from Online Diagnosis may change the Scope of what follows. In response to verified evidence, the graph may add, remove, or reorder Task nodes.

A downstream Task may be planned earlier, but it must not import an upstream result until that result has passed Verification and crossed a Handoff edge. In the DeepSeek-V4-Pro graph, the rejected directions, `Rework` paths, and cross-workstream constraint did not exist in the initial plan; they entered the graph only after execution produced evidence.

**A graph that cannot change after planning records intention, not engineering.**

### 8.3 Evidence Integrity

**Evidence is only as trustworthy as the Harness that produces it.** In one benchmark Task, `bench_serving.py` ignored `delta.reasoning_content`, invalidating the apparent TTFT and throughput baseline. The Journal prevented the next Task from treating that measurement error as an engine regression, but the broader lesson is that evaluation tools, configurations, and data paths are part of the result. They require the same versioning, review, and Verification as the code under test.

**A Deliverable must preserve a reproduction path, not just a conclusion.** For inference work, it should contain enough information to recreate the deployment point and compare the baseline with the candidate: exact code and runtime versions, image, deployment and workload configurations, commands, results, and failed attempts. A Follow-up Handoff states what is complete, where the evidence lives, which judgments still hold, and where the next Task should begin. Once that package is verified and imported, it becomes Imported Context.

**If the next Task must reconstruct the environment or trust an assertion, the work has not been delivered.**

## 9. Conclusion

**Infer-forge does not reduce the complexity of inference engineering; it makes work across that complexity inspectable, resumable, and verifiable.** It keeps each change tied to its deployment point, code provenance, and evidence across repositories, execution rounds, and Follow-up Handoffs, so verified Deliverables and documented rejections can be reproduced, revisited, and carried forward.

**Infer-forge supports an operating model in which one AI infrastructure engineer can direct multiple sustained Task Loops.** Our four-month record shows the observed peak number of **Tasks in flight** increasing from **2 to 9**. We are exploring increasingly autonomous Task Graph coordination while preserving Verification boundaries, Human Gates, and engineering judgment. We hope the practices embodied in infer-forge help teams and individuals apply Harness, Loop, and Graph Engineering to large-scale, complex systems.

## Acknowledgments

We thank the SGLang Team and the broader SGLang community for developing and openly sharing the Skills that support agent-assisted SGLang development<sup>[8](#ref-8)</sup>. We are also grateful to **Peng Zhang** of the SGLang community. We especially thank **Xiaoyu Zhang (BBuf)** for creating and sharing the `AI-Infra-Auto-Driven-SKILLS` collection<sup>[17](#ref-17)</sup>.

We also thank the researchers and engineering teams cited in this article. Their work on Harness Engineering, Loop Engineering, and Graph Engineering helped shape the concepts and methods behind infer-forge.

## References

1. <a id="ref-1"></a>Vivek Trivedy — [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness), LangChain, March 10, 2026.
2. <a id="ref-2"></a>Ryan Lopopolo — [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/), OpenAI, February 11, 2026.
3. <a id="ref-3"></a>Birgitta Böckeler — [Harness engineering for coding agent users](https://martinfowler.com/articles/harness-engineering.html), Martin Fowler, April 2, 2026.
4. <a id="ref-4"></a>Addy Osmani — [Loop Engineering](https://addyosmani.com/blog/loop-engineering/), June 7, 2026.
5. <a id="ref-5"></a>Sydney Runkle — [The Art of Loop Engineering](https://www.langchain.com/blog/the-art-of-loop-engineering), LangChain, June 16, 2026.
6. <a id="ref-6"></a>Prithvi Rajasekaran — [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps), Anthropic, March 24, 2026.
7. <a id="ref-7"></a>Erik Schluntz and Barry Zhang — [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents), Anthropic, December 19, 2024.
8. <a id="ref-8"></a>SGLang Team — [Agent-Assisted SGLang Development: An Initial Exploration](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development), LMSYS Org, July 2, 2026.
9. <a id="ref-9"></a>Lilian Weng — [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/), Lil'Log, July 4, 2026.
10. <a id="ref-10"></a>Boye Niu et al. — [Flow: Modularized Agentic Workflow Automation](https://arxiv.org/abs/2501.07834), arXiv:2501.07834, 2025.
11. <a id="ref-11"></a>Andy Xu and Yu-Wing Tai — [Meta-Agent: From Task Descriptions to Verified Multi-Agent Systems](https://arxiv.org/abs/2605.25233), arXiv:2605.25233, 2026.
12. <a id="ref-12"></a>Ao Li et al. — [GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](https://arxiv.org/abs/2605.22566), arXiv:2605.22566, 2026.
13. <a id="ref-13"></a>Sydney Runkle and Harrison Chase — [3 Years of Graph Engineering with LangGraph](https://www.langchain.com/blog/3-years-of-graph-engineering-with-langgraph), LangChain, July 22, 2026.
14. <a id="ref-14"></a>Nelson F. Liu et al. — [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), TACL 2023 / arXiv:2307.03172.
15. <a id="ref-15"></a>Thariq Shihipar — [The New Rules of Context Engineering for Claude 5 Generation Models](https://claude.com/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models), Claude, July 24, 2026.
16. <a id="ref-16"></a>Boris Cherny and Diana Hu — [Boris Cherny: Building Claude Code](https://www.ycrootaccess.com/p/boris-cherny-building-claude-code), Y Combinator Startup School, July 27, 2026.
17. <a id="ref-17"></a>Xiaoyu Zhang (BBuf) — [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS), GitHub.
