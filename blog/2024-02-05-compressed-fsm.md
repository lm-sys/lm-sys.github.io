---
title: "Fastest JSON Decoding for Local LLMs with Compressed Finite State Machine"
author: "Liangsheng Yin, Ying Sheng, Lianmin Zheng"
date: "Feb 5, 2024"
previewImg: /images/blog/compressed_fsm/demo.gif
---

Constraining an LLM to consistently generate valid JSON or YAML that adheres to a specific schema is a critical feature for many applications.
In this blog post, we introduce an optimization that significantly accelerates this type of constrained decoding. Our approach utilizes a compressed finite state machine and is compatible with any regular expression, thereby accommodating any JSON or YAML schema.
Distinct from existing systems that decode one token at one step, our method analyzes the finite state machine of a regular expression, compresses singular transition paths, and decodes <u>multiple tokens in a single step</u> whenever feasible. In comparison to state-of-the-art systems (guidance + llama.cpp, outlines + vLLM), our method can reduce the latency by up to 2x and boost throughput by up to 2.5x.
This optimization also makes constrained decoding even faster than normal deocding.
You can try it now on [SGLang](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding).

<img src="/images/blog/compressed_fsm/demo.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 1: Comparison of SGLang and Outlines + vLLM in JSON Decoding
</p>

## Background

[JSON](https://en.wikipedia.org/wiki/JSON) is one of the most important formats for data interchange. Requiring LLMs to always generate valid JSON can render the output of the LLM easily parsable in a structured manner. Recognizing its significance, OpenAI introduced the [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode), which constrains the model to always return a valid JSON object. However, more  fine-grained control is often needed to ensure that the generated JSON object adheres to a specific [schema](https://json-schema.org/), such as

<img src="/images/blog/compressed_fsm/json_schema.png" style="width: 100%; max-width: 80%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 2: Example of Constrained Generation Following a JSON Schema
</p>

For local LLMs, there are two major methods to guide the model to generate JSON objects that follow a specific schema.

### Method 1: Finite State Machine Based

This method involves transforming the JSON schema into a regular expression. We can then construct a [Finite State Machine(FSM)](https://en.wikipedia.org/wiki/Finite-state_machine) based on the regular expression. The FSM is used to guide the LLM generation. For every state within the FSM, we can calculate the permissible transitions and identify the acceptable next tokens. This allows us to track the current state during decoding and filter out invalid tokens by applying logit bias to the output. You can learn more about this method in the [outlines](https://arxiv.org/abs/2307.09702) paper.

<img id = "figure3" src="/images/blog/compressed_fsm/method1.png" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 3: Constrained Decoding based on FSM and Logits Masking
</p>

The FSM-based method utilizes the generalized regular expressions to define the low-level rules, which can be applied to a wide range of grammars, such as JSON schema, IP addresses and emails.

**Limitations:**  
Since the FSM is constructed at the token level, it can transition the state by only one token at each step. Consequently, it can decode only one token at a time, which results in slow decoding.

### Method 2: Interleaved-Based

Aside from converting the entire JSON schema into a regular expression, another approach is to employ interleaved-based decoding. In this method, a given JSON schema can be broken down into several parts, each containing either a chunked prefill part or a constrained decoding part. These different parts are executed interleavedly by the inference system.
Because the chunked prefill can process multiple tokens in a single forward pass, it is faster than token-by-token decoding.

[Guidance](https://github.com/guidance-ai/guidance?tab=readme-ov-file#guidance-acceleration) provides a set of syntax rules for interleaved-based decoding, using llama.cpp as a backend.

<img src="/images/blog/compressed_fsm/method2.png" style="width: 100%; max-width: 85%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">Figure 4: Interleaved JSON Decoding in Guidance</p>

**Limitations:**  
- The interleaved-based method requires custom syntax, making it less versatile and expressive than individual regular expressions.
- It struggles with correctly handling tokenization boundaries due to potential conflicts between the decode and chunked prefill segments.
- Frequent communication between the interpreter and the backend brings additional overhead.

## Our Method: Jump-Forward Decoding With a Compressed Finite State Machine

We can combine the advantages of FSM-based and interleaved-based methods by introducing a new decoding algorithm, **jump-forward** decoding, based on the compressed finite state machine.

During the decoding process guided by the regex converted from the JSON schema, we can predict forthcoming strings when we reach specific junctures:

- In [figure3](#figure3), at the beginning of decoding, according to the regex, we can anticipate the incoming string to be:
    ```json
    {
      "name":
    ```
    Then comes the actual decoding part.
- Similarly, when the LLM outputs a `G` while filling in the house attribute of a character, we can confidently predict that the next string will be `ryffindor`, thereby completing the full string as `Gryffindor`.

That is precisely how the jump-forward decoding algorithm makes decoding faster. In the jump-forward algorithm, we examine the finite state machine of the given regular expression, identify all the singular transition edges, and compress consecutive ones together into **singular paths**. Instead of decoding the singular paths token by token, we can directly prefill (extend) them, jumping forward until the next branching point.

<img src="/images/blog/compressed_fsm/compare.png" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">Figure 5: Comparison of Jump-Forward Decoding with Compressed FSM and Normal Decoding</p>

The RadixAttention mechanism of SGLang greatly simplifies the implementation of the jump-forward decoding algorithm.
When executing a jump-forward, we can simply terminate the current request and enqueue a new one. The RadixAttention and efficient **extend** primitive in the SGLang runtime will automatically reuse the KV cache of the previous tokens, thereby avoiding redundant computation.

### Tokenization Boundary Handling

When implementing constrained decoding, it is always tricky to deal with the tokenization boundary, due to the complicated possible mapping between characters and tokens.


During LLM decoding, it might prefer (means with higher probability) to combine multiple characters into a single token.
For instance, when decoding
<code style="color: black; background-color: lightblue;">"Hello"</code>
in the context of JSON decoding, LLMs may output tokens like this:

<code style="color: black; background-color: lightblue;">"</code>
<code style="color: black; background-color: lightblue;">He</code>
<code style="color: black; background-color: lightblue;">llo</code>
<code style="color: black; background-color: lightblue;">",</code>

Instead of decoding the last
<code style="color: black; background-color: lightblue;">"</code>
, it always prefers to combine it with a following 
<code style="color: black; background-color: lightblue;">,</code>
to form a more frequent token
<code style="color: black; background-color: lightblue;">",</code>
. This effect may cause some strange behaviors. For example, in the above case, if the regex is set to
<code style="color: black; background-color: lightblue;">"[\w\d\s]*"</code>
(without the last 
<code style="color: black; background-color: lightblue;">,</code>
), it can lead to endless decoding, because an LLM wants to stop with <code style="color: black; background-color: lightblue;">",</code> but this token is not allowed.

Moreover, during jump-forward decoding, we've found that different tokenization strategies to the jump-forwarded part may lead to different logit distributions for the subsequent tokens. Simply appending the tokenized jump-forwarded section to the current token sequence might yield unexpected outcomes.

To manage these issues, we propose the following solutions:
- Prefer the use of a comprehensive regular expression to guide the entire decoding process, rather than employing multiple concatenated regular expressions. This approach ensures that both Finite State Machines (FSM) and Large Language Models (LLM) are cognizant of the entire decoding process, thereby minimizing boundary-related issues as much as possible.
- We have implemented a re-tokenization mechanism during the jump-forward phase. This involves appending the string instead of the tokens, followed by a re-tokenization of the entire text. This method effectively resolves most tokenization issues and results in only a minor increase in computational overhead, approximately 4\%.

## Benchmark Results

We benchmarked our jump-forward decoding on two tasks:

- Crafting a character's data in JSON format, guided by a brief prompt.
- Extracting a city's information from a long document and outputing it in JSON format.

We tested llama-7B on an NVIDIA A10 GPU (24GB), and used vllm v0.2.7, guidance v0.1.0, outlines v0.2.5 and llama.cpp v0.2.38(Python binding) . The figure below shows the throughput (using the maximum batch size supported by each system) and latency (with a batch size of 1) of these methods:

<img src="/images/blog/compressed_fsm/result.png" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 6: Benchmark Results
</p>

The results show that SGLang with our decoding algorithm significantly outperforms all other systems.
It can reduce the latency by up to 2x and boost throughput by up to 2.5x.

## Link
- You can try this feature now in [SGLang](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding)
- Bencmark code is available [here](https://github.com/sgl-project/sglang/tree/main/benchmark/json_fast_forward)
