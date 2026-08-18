---
title: "No Token Left Behind: Demystifying Token-In-Token-Out in Miles"
author: "Miles Team: Jiajun Li, Yuzhen Zhou, Shi Dong, Yanbin Jiang, Mao Cheng, Yusheng Su, Yueming Yuan, Zhichen Zeng, Banghua Zhu"
date: "June 5, 2026"
previewImg: /images/blog/tito/definition.png
---

In agentic RL, a rollout is not a single generation. It is a chain of model calls, tool outputs, harness messages, and resumed generations. Token-In-Token-Out (TITO) is a design principle that addresses one critical source of training–inference mismatch in this process: whether the trainer evaluates the exact same token sequence that the inference engine consumed and produced during rollout. In this blog post, we aim to clarify how we define the TITO principle, why it is important in RL training, and how such principle is instantiated in the Miles framework.

## Definition of TITO

In an agentic rollout, the model repeatedly interacts with an external environment. In a simplified setting, the model first receives a task description and generates tokens, which may include reasoning and a tool call. The agent runtime parses the tool call, sends it to the corresponding environment or tool backend, and returns the result as a new observation. The model then continues from that observation and may issue another tool call. This loop repeats until the task is complete.

Note that the process involves multiple separate calls to the inference engine, which people colloquially define as *turns*. In each turn, the engine is prompted with a token sequence and generates another token sequence. We say that the TITO principle is fulfilled if, for all $n$, the total token sequence in turn $n-1$ (prompt + response) is a **bit-perfect prefix** of the prompt token sequence in turn $n$. The idea is illustrated in the following diagram.

![TITO definition diagram](/images/blog/tito/definition.png)

## Why TITO matters?

### Training Efficiency: One Sample Per Trajectory

In agentic RL, where a single task can have dozens of turns, we essentially have two options to package data for the RL trainer:

1. **One Sample Per Turn:** Each turn is treated as an independent training sample.
2. **One Sample Per Trajectory:** All turns in one trajectory are "glued" together into a single, contiguous sequence.

In option 1, the trainer receives as many samples as there are turns in a trajectory. In option 2, it receives one contiguous sample per trajectory, regardless of the number of turns. A task can still produce multiple trajectories; their number is not fixed by TITO.

### Mathematical Correctness: Maintaining On-Policyness

For a training sample to be on-policy, every sampled token should be evaluated by the trainer under the same conditional distribution that produced it during rollout. In transformers, that conditional distribution is entirely dependent on the preceding context of the token. If TITO is violated, there could be a token $x_t$ such that

- In the trainer, the model evaluates $x_t$ based on the preceding sequence $\mathbf{x}$.
- In the inference engine, the model samples $x_t$ based on a slightly different preceding sequence $\tilde{\mathbf{x}}$.

Even if the trainer and the inference engine share identical weights, the conditional probability $\pi(x_t|\mathbf{x})$ can diverge dramatically from $\pi(x_t|\tilde{\mathbf{x}})$. Such discrepancy can eventually lead to erratic updates, jeopardizing the stability of RL training.

## How TITO might break

Despite its conceptual simplicity, the TITO principle is fragile. In what follows, we provide three common scenarios, among many others, where the principle could be violated.

### Scenario 1: Detokenize-retokenize mismatch

In multi-turn RL rollouts, one might detokenize the model's generated tokens into a string for storage, and subsequently retokenize it when building the prompt for turn $n$. This can potentially break the TITO principle because **model-generated tokens cannot necessarily survive a detokenize-retokenize roundtrip**.

The root cause lies in the asymmetry between how a tokenizer encodes text and how a model generates tokens:

- **`encode` (text → tokens) is one-to-one**: For a given input string, the tokenizer always picks one standard split (typically greedy / longest-match).
- **`decode` (tokens → text) is many-to-one**: Multiple different token sequences can decode to the exact same string. The model can, and sometimes will, generate a valid but non-standard token sequence.

![Detokenize-retokenize mismatch](/images/blog/tito/scenario1-retokenize.png)

**Example**: Suppose the model generates two separate tokens `Hel(3)` and `lo(7)`. Decoding them produces the string `"Hello"`. However, when you re-encode `"Hello"`, the tokenizer will canonically encode it as the single token `Hello(4)`. The original `Hel(3)` + `lo(7)` sequence is lost forever, causing the trainer to evaluate a token sequence that the model never actually sampled.

### Scenario 2: Reasoning pruned by chat templates

Chat templates translate a JSON-like list of messages into a single prompt string to be sent to the inference engine. Some reasoning-model templates introduce what we call a **cut-thinking boundary**: a point in the conversation before which historical assistant reasoning is removed from the rendered prompt. In the default chat templates of reasoning models like [Qwen3](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen%2FQwen3-4B&example=tool-usage) and [Kimi K2](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=moonshotai%2FKimi-K2-Instruct&example=tool-usage), this boundary is determined by the last `User` message. When the template renders the conversation, it drops `Assistant` reasoning that appears before the last `User` message and preserves only the reasoning after that boundary.

However, agentic harnesses often inject `User` messages mid-task — for example, the Terminus-2 harness uses `User` for terminal outputs, while other harnesses use it for engine retries like "Parse failed". Each injection pushes the cut-think boundary forward, silently erasing the reasoning that the model actually sampled, breaking the bit-perfect prefix between turns. This behavior is illustrated below.

![Cut-think boundary breakage](/images/blog/tito/scenario2-cut-think.png)

### Scenario 3: Lossy chat-template re-rendering

Many inference engines accept a list of messages and re-apply the chat template plus tokenizer on every call to build the prompt. This is convenient, but dangerous: chat templates do their work at the *string* level — whitespace trimming, escape handling, reasoning-content repacking — so the token IDs they emit for a given message can depend on *when* and *alongside what* the message is rendered.

So re-applying the chat template at the message level also introduces unexpected text drift. Here is a concrete failure mode. In turn $n-1$ the assistant emits a tool call whose streamed tokens decode to a compact JSON body — no spaces after commas or colons, keys in the order the model chose:

```json
{"name":"bash","arguments":{"cmd":"ls"}}
```

The engine parses this string and stores it as a structured `tool_calls` field on the assistant message. In turn $n$, when the template re-renders the conversation, it serializes `tool_calls` back through a `tojson` filter. Because this parse-then-serialize roundtrip inherently discards the original byte-level formatting (spaces, newlines), the filter applies its own default spacing and emits:

```json
{"name": "bash", "arguments": {"cmd": "ls"}}
```

Note the extra space after every comma and colon. Same semantics, *different* bytes, *different* token IDs. This breaks the bit-perfect prefix.

## How TITO is implemented in Miles

Miles instantiates TITO with four components, designed so that the core invariant is mechanically verified and new models are cheap to onboard.

### (1) Inference session server

An *inference session* is a single trajectory's interaction with the inference engine — the sequence of turns belonging to the same task, sharing one growing token buffer. The [inference session server](https://github.com/radixark/miles/blob/3270915550fcd69dce788f382fa8c12548a63618/miles/rollout/session/session_server.py#L24) maintains this per-trajectory state, keyed by session id. Under each id it preserves the exact token IDs generated by the model together with their original rollout log probabilities.

On each new turn, Miles tokenizes only the newly appended messages and merges those tokens with the stored prefix. The complete trajectory can then be assembled into one contiguous training sample, with tokens not generated by the model excluded from the loss through loss masking.

![Inference session server architecture](/images/blog/tito/session-server.png)
<p style="text-align: left; color: #666; font-style: italic;">TITO inference session server flow.</p>

### (2) Ensure append-only at three levels

*Append-only* means each turn extends the previous turn while preserving the model-generated token IDs already stored in the session. Miles enforces this at three levels:

**Level 1 — message list.** Turn $n$'s message list extends turn $n-1$'s with new messages on the tail; earlier message dicts are never mutated.

**Level 2 — chat-template rendering.** A chat template can break append-only by pruning earlier content (Scenario 2) or by rendering differently depending on which message roles the harness has appended. To prevent pruning, Miles ships [fixed jinja templates](https://github.com/radixark/miles/blob/95e3208ff583938fbffbe3e58d9495e9dafa2a7c/miles/utils/chat_template_utils/templates/qwen3_fixed.jinja#L43) that disable cut-thinking via a `clear_thinking: false` kwarg, preserving historical reasoning across turns. To prevent role-dependent rendering drift, users declare the expected appended roles via `--tito-allowed-append-roles`, and Miles auto-selects a prefix-stable template configuration for that role set.

**Level 3 — token sequence.** Tokenizing those renderings must produce a bit-perfect token prefix as per the definition of TITO. Naive retokenization breaks this even when Level 2 holds. Miles tokenizes only the newly appended messages, merges the resulting IDs with the stored prefix, and leaves the model-generated IDs and their rollout log probabilities unchanged. The pluggable TITO tokenizer described in the next section is what makes this append-only tokenization work.

### (3) A pluggable TITO tokenizer

The TITO tokenizer is responsible for extending `P` — the per-trajectory token buffer maintained by the inference session server — whenever the harness appends a new non-assistant message. It computes the incremental tokens to splice onto `P`.

**Basic idea — dummy-prefix incremental tokenize.** The recipe (inspired by [this blog post](https://jybsuper.github.io/posts/multiturn_tokenization/)) is:

1. Build a synthetic minimal context.
2. Render the chat template once with the new message and once without.
3. Encode the byte difference.

The resulting delta gives the new serialized content, from which Miles derives the incremental tokens to append to `P`.

For example, suppose `P` already holds the tokenized prefix through turn $n-1$ and the harness now appends one tool response:

```python
old_messages = [system, user, assistant]
new_messages = old_messages + [
    {"role": "tool", "content": "file1.txt\nfile2.txt"},
]
```

Using Qwen3's template as an example, the byte difference for that tool response is:

```
<|im_start|>user
<tool_response>
file1.txt
file2.txt
</tool_response><|im_end|>
```

Encoding that gives the incremental tokens to append onto `P`.

### (4) Verification through CPU and GPU sessions

For each model family, TITO is validated through CPU round-trip tests and real SGLang GPU sessions. Together, these checks safeguard the token-level exactness required for R3, OPD, and zero-KL alignment.

## Supported Models

The TITO pipeline currently supports the following models natively (both thinking and non-thinking variants):

- **Qwen**: Qwen3, Qwen3.5, Qwen3-Next
- **GLM**: GLM-4.7, GLM-5, GLM-5.1
- **Kimi**: Kimi-K2, Kimi-K2.5, Kimi-K2.6
- **Nemotron**: Nemotron-3
- **Minimax**: Minimax-M2.5, Minimax-M2.7
- **Deepseek**: Deepseek-v3.2, Deepseek-v4

For each model (except Deepseek-v3.2 and Deepseek-v4), TITO is verified to handle the following combinations of message roles a harness may append after the first assistant turn:

- `{tool}`: harnesses that only inject tool outputs.
- `{tool, user}`: harnesses that also inject `User`-role messages such as terminal outputs (e.g., Terminus-2) or parser-retry prompts.
- `{tool, user, system}`: harnesses that further inject `System`-role reminders mid-task.

Both Deepseek-v3.2 and Deepseek-v4 currently support only the `{tool}` surface; broadening them — like onboarding any new model — is usually just a fixed Jinja template plus a small [`merge_tokens`](https://github.com/radixark/miles/blob/3270915550fcd69dce788f382fa8c12548a63618/miles/utils/chat_template_utils/tito_tokenizer.py) override. That low cost is the whole point: TITO keeps every rollout bit-perfect for training while staying cheap to extend, so no token is left behind.

## Building on TITO: Black-box agent harness training

Miles is developing a training recipe for black-box agent harnesses such as Claude Code and Codex. These harnesses spawn subagents and compact context at runtime, so the number of trajectories per task is dynamic and unknown in advance. Miles therefore records the full trajectory tree through the session server, and on the training side applies the loss normalization required for varying batch sizes, keeping gradient scale consistent.

**If you want to try out TITO in miles, please take a look at the doc [here](https://miles.radixark.com/docs/user-guide/agentic-rollout).**
