---
title: "Fastest JSON Decoding for Local LLMs with Compressed Finite State Machine"
date: "January 30, 2024"
previewImg: /images/blog/laattention/acc-demo.gif
---

**TL;DR:** We are excited to introduce the **fastest JSON decoding**, a universal, rapid, and lossless JSON/regex decoding algorithm to accelerate local LLMs.

To use the fastest JSON decoding, all you need to do is add a regex when serving with SGLang. The regex constrains LLMs' decoding behaviors, compelling them to follow a [finite state machine (FSM)](https://en.wikipedia.org/wiki/Finite-state_machine) to generate a formatted result. We speed up this process by analyzing individual transition edges in the FSM and compressing them. Rather than decoding the compressed segment token by token, we directly prefill/extend it within our SGLang backend. This strategic approach significantly boosts processing speed, delivering a 4x speedup in performance on standard JSON decoding tasks compared to traditional regex-based methods.
Below is an example of our fastest JSON decoding accelerating LLaMa-2-Chat 7B JSON generation.
![JSON-demo](/images/blog/laattention/acc-demo.gif)

## Introduction

## Background

## Our Approach

## Benchmark Results

## Promising Applications