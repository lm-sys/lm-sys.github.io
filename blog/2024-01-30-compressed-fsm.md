---
title: "Fastest JSON Decoding for Local LLMs with Compressed Finite State Machine"
date: "January 30, 2024"
previewImg: /images/blog/laattention/acc-demo.gif
---

**TL;DR:** We are excited to introduce the **fastest JSON decoding**, a universal, rapid, and lossless JSON/regex decoding algorithm to accelerate local LLMs.

To use the fastest JSON decoding, all you need to do is add a regex when serving with [SGLang](https://github.com/sgl-project/sglang). The regex constrains LLMs' decoding behaviors, compelling them to follow a [finite state machine (FSM)](https://en.wikipedia.org/wiki/Finite-state_machine) to generate a formatted result. We speed up this process by analyzing individual transition edges in the FSM and compressing them. Rather than decoding the compressed segment token by token, we directly prefill/extend it within our SGLang backend. This strategic approach significantly boosts processing speed, delivering a 4x speedup in performance on standard JSON decoding tasks compared to traditional regex-based methods.
Below is an example of our fastest JSON decoding accelerating LLaMa-2-Chat 7B JSON generation.
![JSON-demo](/images/blog/laattention/acc-demo.gif)

## Introduction

[JSON](https://en.wikipedia.org/wiki/JSON) is one of the world's most important formats for data interchange, and efficiently guiding LLMs to generate JSON is a critical problem for many applications. A notable solution is to convert a preferred JSON schema into a regular expression, which not only defines the key-value format but also allows us to confine each value's range, control the list length, and more. Ensuring that LLMs follow a regular expression can also accommodate a broader range of scenarios, specifically those involving unique formats like IP addresses and emails.

![JSON-regex (An image showing a JSON schema converted to regex)](/images/blog/laattention/json-regex-fsm.png)

Existing methods for JSON decoding involve adding logit bias to the LLMs' decoding outputs based on the current states and transitions. By analyzing all the possible transitions dependent on the current state, we can mask out the invalid tokens and only keep the valid ones.

![FSM-decode (An image showing how FSM adds logits bias to filter tokens)](/images/blog/laattention/fsm-decode.png)

Previous research primarily concentrates on the efficiency of processing permitted tokens according to the FSM and state. However, a standard regex converted from a JSON schema always includes numerous transitions that could be compressed, and decoding them is time-consuming. To address this problem, we came up with fast-forward, an operation to skip the compressed part's decoding process, using prefill/extend instead. Benefiting from SGLang's automatic prefix cache mechanism, we only need to calculate the KV cache of the fast-forwarded part, which is much faster than computing the KV cache of all the tokens.

## Background: Interleaved-based JSON Decoding

Except for regex-based methods, another popular approach to generate JSON is interleaved-based decoding.

### Limitations of Interleaved-Based Decoding

- Overhead of frontend frequently communicating with backend.
- It is only suitable for JSON-like formats, cannot support complex grammar with more branches, and is not as universal as regex-based methods.
- Decoding and Prompt race to the same part, resulting in endless decoding until hitting the max length.

## Our Approach

Describe our approach, drawing a figure to compare with token-by-token methods.

Describe the re-tokenize process or not. I do not know how to analyze it.

## Benchmark Results

vllm + outlines

fastest + sglang

guidance + llama.cpp

## Promising Applications

- Character Information Extraction
- Long Document Retrieval