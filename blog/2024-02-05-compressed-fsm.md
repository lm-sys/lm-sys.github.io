---
title: "Fastest JSON Decoding for Local LLMs with Compressed Finite State Machine"
date: "January 30, 2024"
previewImg: /images/blog/compressed_fsm/demo.gif
---

In this blog post, we share an optimization for constrained JSON decoding based on the compressed finite state machine. Instead of decoding token by token, our method analyzes the finite state machine of a regular expression, compresses the singular transition path, and decodes multiple tokens in a single step whenever possible. Compared to state-of-the-art systems (guidance + llama.cpp, outlines + vLLM), our method can reduce latency by up to 2x and increase throughput by up to 3x. You can try this feature now in [SGLang](https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding).

<img src="/images/blog/compressed_fsm/demo.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 1: Demo of speedups by SGLang's jump-forward decoding algorithm, compared to outlines + vLLM. LLMs generate the specified JSON object to describe the given character.
</p>

## Background

[JSON](https://en.wikipedia.org/wiki/JSON) is one of the world's most important formats for data interchange, and efficiently guiding LLMs to generate JSON is a critical problem for many applications.
OpenAI proposed its [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) to instruct the model to always return a valid JSON object.
However, more fine-grained control is often needed to ensure that the generated JSON object follows a specific [schema](https://json-schema.org/), such as:

<img src="/images/blog/compressed_fsm/json_schema.png" style="width: 100%; max-width: 80%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 2: Example of how a JSON schema can guide the generation of a JSON object.
</p>

For local LLMs, there are two major methods to guide the model to generate JSON objects that follow a specific schema.

### Method 1: Finite State Machine Based

This method involves transforming the JSON schema into a regular expression. We can then construct a [Finite State Machine(FSM)](https://en.wikipedia.org/wiki/Finite-state_machine) based on the regular expression. For every state within the FSM, we can calculate the permissible transitions and identify the acceptable next tokens. This allows us to track the current state during decoding and filter out invalid tokens by applying logit bias to the output.

<img id = "figure3" src="/images/blog/compressed_fsm/method1.png" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 3: ...
</p>

The FSM-based method utilizes more generalized regular expressions to outline the low-level rules, which can be applied to a wide range of grammars, such as IP addresses and emails.

#### Limitations:

However, the guided decoding process is time-consuming, as it requires decoding the whole JSON object token by token.

### Method 2: Interleaved-Based

Aside from converting the entire JSON schema into a regular expression, another popular approach is to employ interleaved-based decoding. In this method, a given JSON schema can be broken down into several parts, each containing either a chunked prefill part or a constraint decoding part. These different parts are executed interleavedly by the inference system.

The [guidance](https://github.com/guidance-ai/guidance?tab=readme-ov-file#guidance-acceleration) provides a set of syntax rules for interleaved-based decoding, using llama.cpp as a backend to accelerate.

<img src="/images/blog/compressed_fsm/method2.png" style="width: 100%; max-width: 85%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">Figure 4: ...</p>

#### Limitations:

- The interleaved-based method requires custom syntax, making it less versatile and expressive than individual regular expressions.
- It struggles with correctly handling tokenization boundaries due to potential conflicts between the decoding and chunked prefilling segments.
- Frequent communication between the model and the guidance brings additional overhead.

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
<p style="color:gray; text-align: center;">Figure 5: ...</p>

The radix cache mechanism of SGLang greatly benefits the jump-forward decoding algorithm. When executing a jump-forward, all the prefix tokens prior to the jump-forwarded part get automatically cached. This cache mechanism ensures that the **extend** primitives performed by the jump-forward algorithm align with SGLang, thus eliminating any additional overhead.

### Tokenization Boundary Handling

### Future Work

## Benchmark Results

We tested our jump-forward decoding on two typical tasks:

- Crafting a character's data in JSON format, guided by a brief prompt.
- Extracting a city's information from an extensive document and presenting it in JSON format.

We tested Lllam-7B on NVIDIA A10 GPU (24GB), and used vllm v0.2.7, guidance v0.1.0, outlines v0.2.5 and llama_cpp_python v0.2.38. The following table shows the throughput and latency(with batch size 1) of theses methods:

<img src="/images/blog/compressed_fsm/result.png" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">
Figure 6: ...
</p>
