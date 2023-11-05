---
title: "FoldAttention: Parallelize Autoregressive Decoding Without A Draft Model"
author: "xxx"
date: "November 6, 2023"
previewImg: /images/blog/foldattention/acc-demo.gif
---

Recently, the advent of transformer-based large language models (LLMs) has taken precedence in both academia and industry for natural language processing and understanding. Despite their rising prominence, current decoder-based LLMs (e.g., GPT-3, LLaMA, Falcon) are burdened by a significant constraint in their autoregressive decoding mechanism, generating only a single token per model step. This method necessitates repeated model invocations in proportion to the generation sequence length. The process largely underutilizes the parallel computational power of contemporary parallel processors (e.g., GPUs), limiting individual users and LLM service providers. Given the massive applications of decoder-based LLMs across various domains, refining the autoregressive decoding process is an important pursuit.

One significant scenario in LLM applications is the need for **low-latency** execution in generating long text sequences. This goal is often essential for individual users or service providers committed to low-latency offerings. According to the previous research like [speculative decoding](https://arxiv.org/abs/2211.17192), it is understood that LLM autoregressive decoding is memory bandwidth-intensive, with single-step latency being contingent on the model parameter size and the cumulative latency relying on the total decoding steps. In essence, reducing aggregate decoding latency necessitates strategies that either (1) *diminish the cost of individual decoding steps* or (2) *curtail the total number of decoding iterations*. We focus on improving (2) while not hurting (1).

We introduce **FoldAttention**, which can (1) compress the decoding steps without a draft model and (2) linearly compress decoding steps relative to the log(FLOPS) within a certain range. At a moderate compression rate, the extra FLOPS is small, yielding a minimal additional cost per step and facilitating a substantial increase in decoding speed (e.g., 1.5x-2.3x). We have integrated FoldAttention into the commonly used transformers library. Users can accelerate huggingface's native ```generate```  function with only a few lines of code. Our [code](https://github.com/hao-ai-lab/ParallelDecoding) is released. 



<img src="/images/blog/foldattention/acc-demo.gif" style="width: 150%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">Speedup of FoldAttention on LLaMA-2-7B-Chat</p>



## Guess-And-Verify Decoding

We name an essential method to curtail the total decoding steps by **guess-and-verify** decoding paradigms, such as [speculative decoding](https://arxiv.org/abs/2211.17192), [Medusa](https://sites.google.com/view/medusa-llm), and [Draft & Verify](https://arxiv.org/abs/2309.08168). These approaches predict multiple potential future tokens and subsequently confirm the validity of these predictions during the decoding phase. This methodology hinges on leveraging the autoregressive nature of LLMs to perform verification within the decoding process itself.

Let us take **speculative decoding** as an instance to illustrate this concept better. Instead of using the original LLM to generate all tokens, speculative decoding utilizes a more resource-efficient draft model to generate a draft token sequence autoregressively. It then uses the LLM to verify the token sequence guessed by the LLM. In a greedy search decoding, if the accurate token output of the last token is the same as the draft token of the next position, this draft token is accepted. By guessing tokens correctly, a long sequence of tokens will be accepted each run, the invocation of the original LLM is reduced, and speedup is achieved.

However, speculative decoding faces two problems. First, the maximum speedup of Speculative Decoding is bounded theoretically. Besides, Speculative Decoding often requires a draft model to predict the future generation. FoldAttention takes a firm step in trying to solve these two burdens in one shot.

## Parallel Jacobi Decoding

[Parallel Jacobi Decoding](https://arxiv.org/pdf/2305.10427.pdf) generates tokens for an autoregressive model similar to the design in the Jacobi-Iteration method in solving linear systems. As shown in the following figure, instead of generating one token each time, Jacobi decoding will update and verify a future token sequence simultaneously. Sometimes, several tokens can be accepted in one run, and a reduction in decoding steps is achieved. For example, in the following figure, using Jacobi decoding can correctly guess several tokens (e.g., token 'computer' and 'scientist'). Because the cost of decoding several tokens is similar to decoding only one token in the memory bandwidth-bounded autoregressive decoding task, Jacobi decoding saves steps and thus achieves speedups.

**Limitation of Jacobi Decoding**. However, in our study, Jacobi decoding faces several problems, which make it hard to achieve empirical speedups in real-world applications:
- Although it can sometimes correctly guess tokens, it is hard to place tokens in the correct position, and even those valid tokens will be frequently substituted.
- Unlike Jacobi-Iterations in solving linear systems, the input and output of LLMs are discrete values, and it is hard to converge like continuous values.
- Original autoregressive decoding is already a triangular system, and taking n steps to obtain n tokens is already a significant speedup.




## Implementation of FoldAttention

The key to preserving decoding distribution while reducing the number of steps lies in **predicting** a series of future tokens and simultaneously **verifying** these tokens during the decoding process, following the guess-and-verify paradigm. Different from previous methods, FoldAttention does not need a draft model and uses itself to generate guess candidates by a modified Jacobi Decoding in a **prediction branch**. Then FoldAttention verifies these guess token candidates in a **verification branch**. The following figure shows the procedure of FoldAttention. 


## Prediction Branch

To generate guess token candidates, we maintain a multi-level sliding window as the yellow tokens in the figure above, updating its contents using a modified Jacobi decoding approach. Here, level and window are two essential configurations in FoldAttention. We will generate one next token for one token in the latest window. For example, We will generate 'great' from 'Turing', 'a' and generate 'the' from 'who', 'is'. Here, 'Turing' and 'who' are tokens from the last of the last decoding pass. Token 'a' and 'is' are generated from the last decoding step. When the level is equal to 2, it should be a Jacobi decoding that generates one token from the previous token. Following the generation step, we will capture all three token-tuples and collect them (for simplicity, we only capture tuples starting with the next token in the figure). Then, we will discard the tokens in the oldest level and use the new level to fill its past level to maintain both level and window size. 

## Verification Branch

Each step undergoes scrutiny by a verification branch. As discussed, we collect microstructures, such as the three-token-tuples ('a', 'great', 'computer') and ('a', 'one', 'a') in the example above. These tuples are stored in a hash map, establishing correspondences such 'a' -> ('great', 'computer'). When encountering token 'a' in subsequent inputs, we verify the expected succession of two tokens. The difference thing from the figure above is that we will store all microstructures during the decoding process instead of discarding them at each decoding step in our implementation. It is common to identify multiple microstructures starting with the same token, thereby increasing the cost of the verification process. We implement a maximum verification size to limit the verification branch's overhead within a given budget.

## Decode, Predict and Verify In The Same Step

The memory bandwidth-intensive character makes the overhead small to decode several tokens compared with decoding one token per step. This hints us to merge decoding, prediction, and verification in the same step to save overall cost. The current transformer-based model often contains two facets: token-wise computation and token-interaction computation. To amalgamate decoding, prediction, and verification within a single step, a meticulous restructuring of the decoding style is imperative to preserve the original output distribution. Specifically, it necessitates accurately assigning each token's relative and absolute positional expression within the decoder model.

We exemplify this with the LLaMA model. The correct configuration of the absolute positional expression is about setting the positional embedding correctly. This can be done by passing the accurate absolute position index into the model input. The correct configuration of the relative positional expression is to set the attention mask correctly, which illustrates the interaction of each token. We implement this by configuring the attention mask in the **verification** and **prediction** branches. We implement these attention masks following two constrictions: (1) The tokens in the prediction branch can not see tokens in the verification branch, and vice versa. (2) Each token can only see its past tokens. An example of an attention mask is shown below for a configuration of level=4 and window size=5. The left-most token is used to **decode** the next token.

<img src="/images/blog/foldattention/mask.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>
<p style="color:gray; text-align: center;">Attention mask for FoldAttention.</p>

## Scaling Computation Resources Can Linearly Increase Compression Ratio
FoldAttention guesses future tokens from micro-structures based on the current token input. Unlike speculative decode guessing and verifying **one** prediction at each step, FoldAttention can generate several predictions proportional to the window size. According to our further theoretical analysis, when the guess sequence length is long enough (i.e., level in FoldAttention), the exponential investment in future token guesses (and thus window size and FLOPS per decoding step in FoldAttention) can linearly increase the compression ratio. We call it **scaling law** in the LLM parallel decoding within the guess-and-verify paradigm. FoldAttention can match the parallel decoding scaling law in a certain range in our empirical experiments. We illustrate FoldAttention's matching scaling law in the following figure.

<img src="/images/blog/foldattention/match-scaling.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>
<p style="color:gray; text-align: center;">When the level is large enough, exponentially increase window size can linearly increase the compression ratio.</p>

## Experimental Result
We conducted an extensive benchmarking process to evaluate the efficiency of FoldAttention by integrating it with [LLaMA-2-Chat](https://ai.meta.com/llama/) and [CodeLLaMA](https://ai.meta.com/blog/code-llama-large-language-model-coding/) models. Specifically, our tests focused on the 7B, 13B, and 70B parameter configurations of the LLaMA-2-Chat models and the 7B, 13B, and 33B configurations of the CodeLLaMA models. Our objective was to substantiate the acceleration these models attain in practical and real-world applications through different datasets. The design of FoldAttention makes the speedup without any finetune and draft model and preserves the output distribution.

<img src="/images/blog/foldattention/fold-perf.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>
<p style="color:gray; text-align: center;">Speedup of FoldAttention on Different Datasets.</p>

**MT-Bench Results with LLaMA-Chat** [MT-Bench](https://lmsys.org/blog/2023-06-22-leaderboard/), encompassing a cross-area set of multi-turn questions, served as our testing ground for assessing FoldAttention's overall performance efficacy. FoldAttention achieves roughly 1.5x speedup across several model settings.

**Code Infilling and Code Completion with CodeLLaMA**. Applying FoldAttention to CodeLLaMA on [HumanEval](https://arxiv.org/abs/2107.03374) also shows large speedups (i.e., more than 2x). In the code infilling task, the generation length is relatively short, and in this case, the speedup is relatively low because FoldAttention requires a large number of steps to fill the window and carry it on smoothly. In code completion tasks, many repeated tokens appear in a generation, and the speedup is larger than other datasets.

**Instructional Coding Task and Math Problem Solving with CodeLLaMA-Instruct**. Finally, we evaluate FoldAttention's performance on instructional coding tasks and solving math problems. For the [GSM8K](https://arxiv.org/abs/2110.14168) dataset, we evaluated CodeLLaMA-Instruct on the first 1K questions. And the instructional coding task is evaluated on [MBPP](https://arxiv.org/abs/2108.07732). Results show that FoldAttention can bring more than 1.8x speedups on these settings.


## Get started with FoldAttention

We have encapsulated the dedicated implementation of FoldAttention in a Python library, and it is easy to use with huggingface's transformers. You can accelerate your transformers' decoding API with only a few LoCs. Please check our [GitHub repo](https://github.com/hao-ai-lab/ParallelDecoding)!

## Acknowledgment
We would like to thank .

## The Team
The FoldAttention and this blog post are developed, evaluated, and maintained by the following



## Citation
```
@misc{fold2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {},
    month = {November},
    year = {2023}
}
```


