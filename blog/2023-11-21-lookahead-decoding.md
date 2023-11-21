---
title: "Breaking the Sequential Dependency of LLM Inference using Lookahead Decoding"
author: "Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang"
date: "November 21, 2023"
previewImg: /images/blog/laattention/acc-demo.gif
---

**TL;DR:** We introduce a new, exact decoding algorithm, ***lookahead decoding***, for LLM inference. 
Lookahead decoding parallelizes autoregressive decoding by extracting and verifying n-grams using the LLM based on Jacobi iteration. 
Lookahead decoding operates **without** a draft model or a data store, and linearly reduces the number of decoding steps in relation to the logarithm of the FLOPs invested per decoding step. 
See a demo of lookahead decoding accelerating LLaMa-7B-Chat generation: 

<img src="/images/blog/laattention/acc-demo.gif" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 1: Demo of speedups by Lookahead Decoding on LLaMA-2-7B-Chat generation. Blue fonts were tokens generated in parallel in one decoding step.</p>

## Introduction
Large language models (LLMs) like GPT-4 and LLaMA are rapidly reinventing today's applications, but their inference -- based on autoregressive decoding -- is very slow and difficult to optimize. Each autoregressive decoding step generates only one token at a time; as a result, the latency of an LLM request primarily depends on the response length of the request, or equivalently the number of decoding steps. 
Making matters worse, each decoding step cannot effectively leverage the parallel processing capabilities of modern GPUs, often leading to GPU underutilization.
This challenges many real world LLM applications that prioritize rapid response time, such as chatbots and personal assistants, which require frequently generating *long sequences with low latency*. 

One way to accelerate autoregressive decoding is [speculative decoding](https://arxiv.org/abs/2211.17192) (including [Medusa](https://sites.google.com/view/medusa-llm) and [OSD](https://arxiv.org/abs//2310.07177)), which employ a "guess-and-verify" strategy: a smaller draft model predicts several potential future tokens, and the original LLM then verifies these guesses in parallel. 
These approaches can opportunistically reduce the number of decoding steps and, consequently lower latency. However, they face several limitations.
First, the maximum speedup that speculative decoding based methods can achieve is limited by the *token acceptance rate*, or equivalently how accurately the draft model can predict the main model's outputs. Second, creating an accurate draft model is non-trivial, often requiring extra training and careful tuning; Incoparating these draft models into real LLM services may introduce new complications.

In this blogpost, we introduce a new, exact decoding algorithm, **Lookahead Decoding**, designed to address these challenges.
The key observation enabling lookahead decoding is: although it is infeasible to immediately decode multiple next tokens in one step, an LLM can indeed generate multiple disjoint [n-grams]((https://en.wikipedia.org/wiki/N-gram)) in parallel. These n-grams could potentially fit into future positions of the generated sequence.
In lookahead decoding, an LLM first generates multiple, disjoint n-grams *in parallel*. This is achieved by viewing [autoregressive decoding as a process of solving nonlinear equations](https://proceedings.mlr.press/v139/song21a/song21a.pdf), and modernizing the classic [Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method) for parallel decoding. These n-grams are captured and later verified -- if deemed suitable, are incorporated into the ongoing sequence generation.

Lookahead decoding is able to generate n-grams in each step, as opposed to producing just one token, hence it can reduce the number of decoding steps -- generating N tokens in fewer than N steps. In fact, lookahead decoding stands out because it:
- Operates **without** a draft model, simplifying the deployment process.
- Linearly reduces the number of decoding steps in relation to the logarithm of the FLOPs invested per step.


We will show that at a moderate compression rate and insignificant extra FLOPS invested per step, lookahead decoding significantly lowers latency, achieving 1.5x to 2.3x speedups. More importantly, it offers the flexibility to invest more FLOPs for even greater latency reduction, which is particularly beneficial for extremely latency-sensitive applications, albeit with diminishing returns.

We have developed a version of Lookahead Decoding that is compatible with the ```huggingface/transformers``` library. This allows users to enhance the performance of HuggingFace's native ```generate``` function with just a few lines of code. We invite you to explore our [code repository](https://github.com/hao-ai-lab/ParallelDecoding) and experience the benefits of Lookahead Decoding in your LLM applications.

## Background: Parallel LLM Decoding using Jacobi Iteration

The [Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method) is a classic solver for non-linear systems. In the case of LLM inference, We can also employ it for parallel token generation without a draft model.
To see this, let's reconsider the autoregressive decoding process. Traditionally, this process is seen as a sequential generation of tokens, illustrated in Figure.2(Left). With some simple rearrangements of equations, it can be conceptualized as solving a system of non-linear equations, as depicted in Figure.2(Right).

<img src="/images/blog/laattention/equations.png" style="width: 70%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>
<p style="color:gray; text-align: center;">Figure 2: Autoregressive deocoding as a process of solving non-linear systems.</p>

An alternative approach based on Jacobi iteratoin can solve all $[y_1, y_2, ..., y_m]$ of this non-linear system in parallel as follows:
- Start with an initial guess for all variables $\textbf{y} = [y_1, y_2, ..., y_m]$.
- Calculate new $\textbf{y}'$ values for each equation with the previous $\textbf{y}$ values.
- Update $\textbf{y}$ to the newly caldulated $\textbf{y}'$.
- Repeat this process until a certain stopping condition is achieved (e.g., $\textbf{y} = \textbf{y'}$).
  
We illustrate this parallel decoding process (also referred to as [*Jacobi decoding*](https://arxiv.org/pdf/2305.10427.pdf)) in Figure 3. 
Jacabi decoding can guarantee solving all *m* variables in at most *m* steps (i.e. the same number of steps as autoregressive decoding), because each step guarantees at least the very first token correctly decoded. 
Sometimes, multiple tokens might converge in a single iteration, potentially reducing the overall number of decoding steps. For example, as shown in Figure 3, Jacobi decoding accurately predicts and accepts two tokens "computer" and "scientist" in Step 4. Compared to autoregressive decoding, each Jacobi decoding step is slightly more expensive in terms of FLOPs needed, because it requires LLM forward computation on >1 token. Fortunately, this usually does not translate into slowdowns thanks to the parallel processing nature of acclerators.
<img src="/images/blog/laattention/jacobi-iteration.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 3: Illustration of applying Jacobi iteration method for parallel LLM decoding.</p>

### Limitations of Jacobi Decoding 
In pracice, however, we have found that Jacobi decoding faces several challenges that impede achieving substantial wallclock speedup. While it manages to decode more than one token in many steps, the precise positioning of these tokens within the sequence often goes wrong. These tokens, though correctly predicted, will still be replaced in subsequent iterations. Consequently, only very few iterations observe multiple tokens being *simutaneously decoded and positioned*, defeating the purpose of parallel decoding.

## Lookahead Decoding
Lookahead decoding addresses the aforementioned limitation by leveraging Jacobi Decoding's capability of generating parallel N-grams. More precisely, in Jacobi decoding, we notice that each new token at a position was decoded based on the its past values in previous iterations. Hence, *Jocabi iteration creates a trajectory of historical tokens at each token position*, which form many n-grams. For example, if we look back three Jacobi iterations, we can have a 3-gram at each token position. Lookahead decoding collects and caches these n-grams from the trajectory. 
While it is performing parallel decoding using Jacobi iterations for future tokens, it will also verify promising n-grams in cache. If an N-gram is accepted, we can fast-forward N tokens in a step. Figure 4 illustrate this process.

<img src="/images/blog/laattention/lookahead-decoding.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 4: Illustration of lookahead decoding with window size 5 and 2-gram.</p>

To make this decoding process more efficient, each lookahead decoding step includes a **lookahead branch** and a **verification branch** running in parallel. The lookahead branch maintains a fixed-sized multi-level window to generate n-gram token candidates from the Jacobi iteration trajectory. The verification branch verifies promising n-gram candidates simultaneously.

## Lookahead Branch
The lookahead branch aims to generate new N-grams. In lookahead branch, we maintain a two-dimensional window, governed by two parameters:
- *window size $W$*: how long we look ahead in future token positions to perform parallel decoding,
- *N-gram size $N$*: how many steps we look back in the past Jacobi trajectory to fetch N-grams.
Figure 5 illustrates an example where we look back 4 steps (N = 4) in trajectory and look ahead 5 tokens (W = 5) in future positions.

The blue token 0 in Figure 5 is the current input. The orange, green, and red tokens were generated by previous Jacobi iterations at steps $t-3$, $t-2$, $t-1$, respectively. The number on each token indicates its relative position to the current input token (blue one with 0). At the current step $t$, we perform one Jacobi iteration to generate new tokens for all 5 positions using the trajectory formed by previous 3 steps. Then, we collect 4-grams -- for example, the orange token 1, green 2, red 3, and the newly generated token form a 4-gram. 
As the decoding moves forward, tokens from the earliest step are removed from the trajectory to maintain $N$ and $W$.
It is worth nothing that when $N = 2$, lookahead decoding degenerates into Jacobi decoding. 

## Verification Branch
Besides the lookahead branch, each decoding step is accompanied by a parallel verification branch to find and verify promising N-grams so that the decoding progresses. 
In the verification branch, we identify n-grams with their first token matching the last input token (via a simple string match); we append the n-gram to the input and verify them via an LLM forward pass. As the n-gram cache grows, it is common to identify multiple n-grams starting with the same token, thereby increasing the verification cost. Here, we define the maximum number of candidates in the verification branch as $G$. In practice, we often set this limitation proportional to $W$ (e.g., we set $G=W$ throughout our experiments).

## Lookahead and Verify In The Same Step
Since LLM decoding is primarily bounded by memory bandwidth, we can merge lookhead branch and verification branch in the same step, leveraging GPU's parallel processing and hiding overheads. This can be achieved by specifying their computation on a designed attention mask shown in Figure 5: (1) The tokens in the lookahead branch cannot see tokens in the verification branch, and vice versa. (2) Each token can only see its previous tokens and itself in both decoding and verification. We have implemented the attention mask in [HuggingFace](https://github.com/huggingface/transformers/tree/main). Stay tuned for a more efficient custom CUDA kernel to further speed up the execution.

<img src="/images/blog/laattention/mask.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>

<p style="color:gray; text-align: center;">Figure 5: Attention mask for lookahead decoding with 4-grams and window size 5. In this mask, two 4-gram candidates (bottom right) are verified concurrently with parallel decoding. </p>

## Scaling Law of Lookahead Decoding
Unlike speculative decode guessing and verifying **one** prediction per step, 
Lookahead decoding can generate $W$ different N-grams and verify $G$ candidates per each decoding step. As $W$ and $N$ increases, the flops per step increases, but it is more likely to accept a longer N-gram. In other words, lookahead decoding allows to trade more flops for increasing the acceptance rate of a longer N-gram in a decoding step  -- which translated into reduced latency, as long as the system is not compute-bound.

To study this scaling behavior, given a few number of tokens to decode, we illustrate how many lookahead decoding steps are needed when enumerating different values of $N$ and $W$.
As we can see, when the N-gram (look back window) size is large enough (e.g., $N=11$), the exponential investment in future token guesses (i.e., the window size $W$) can linearly reduce the number of decoding steps. We call it the **scaling law** of lookahead decoding.

<img src="/images/blog/laattention/match-scaling.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>

<p style="color:gray; text-align: center;">Figure 6: When $N$ is large enough, exponentially increasing window size $W$ can almost linearly reduce the number of decoding steps. Here we set $G=W$. (LLaMA-2-7B-chat on MT-Bench)</p>

### How to Configure $W$ and $N$ in Practice
The FLOPs needed for each lookahead decoding step is proportional to the product of the window size and the n-gram size ($W * N$). As the scaling law reveals, when $N$ is large enough, an exponential increase in the $W$ can result in a linear reduction of decoding steps. Thus, we can achieve linear compression of the steps by trading exponentially more FLOPs.

Given this property, lookahead decoding should be used in scenarios where latency is vital  -- it is worthwhile to pay FLOPs for latency. 
For powerful GPUs (e.g., A100), lookahead decoding can better squeeze its performance by using a large $W$ and $N$ to ahieve low latency when generating long sequeneces. However, if $W$ and $N$ are too large, each lookahead decoding step might be too costly and slow down the decoding, despite reducing decoding steps. 
Increasing $N$ together with $W$ would be best to achieve balanced performance, avoiding hitting a theoretical cap if only increasing one side. Our experimental results show that on A100, the following setting can be optimal in most cases. 

| Model Setting | $W$ | $N$ |
|----: |:----:  | :----: |
| 7B| 15| 5 |
| 13B | 10 | 5|
| 33B | 7 | 5 |

You can also change the setting to tune a better performance on your specific decoding latency requirement. 

**Per-Step Overhead with Lookahead decoding** Despite the wall-clock time reduction in the previous settings, Lookahead decoding actually requires much larger per-step FLOPs to achieve a #step compression. We set a guess token limit to limit the most number of guess n-grams per decoding step. The #token we need to decode per-step will be at most *(window + guess) * (level - 1)*. This number is approximately a multiple of the Vallina autoregressive decoding FLOPs. So we need to have roughly 120x extra FLOPs for 7B models (with level=5, window=15, and guess=15) and 56x extra FLOPs for 33B models (with level=5, window=7 and guess=7) in the previous experiments. Because of the memory-intensive bound characteristic of the LLM decoding, these extra FLOPs only bring little per-step cost and a visible step compression ratio, resulting in a notable speedup.


## Experimental Result

We evaluate the efficiency of Lookahead Decoding on [LLaMA-2-Chat](https://ai.meta.com/llama/) and [CodeLLaMA](https://ai.meta.com/blog/code-llama-large-language-model-coding/) of various sizes on different datasets including [MT-bench](https://huggingface.co/spaces/lmsys/mt-bench), [HumanEval](https://github.com/openai/human-eval), and [GSM8K](https://huggingface.co/datasets/gsm8k). Note that lookahead decoding achieves speedup without any finetuning nor  draft models. The 7B, 13B and 33B models are evaluated on single A100 GPU and the 70B model is evaluated on two A100 GPUs with pipeline parallelism, all under fp16 precision.

<img src="/images/blog/laattention/lookahead-perf.png" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 7: Speedup of lookahead decoding on different models and datasets.</p>

- **LLaMA-Chat on MT-Bench**. Lookahead Decoding achieves roughly 1.5x speedup across several model settings.

- **CodeLLaMA on HumanEval**. Applying lookahead decoding to CodeLLaMA on [HumanEval](https://arxiv.org/abs/2107.03374) shows more than 2x latency reduction. This is because many repeated N-grams are present in code which can be correctly guessed.

- **CodeLLaMA-Instruct on GSM8K**. Using CodeLLama-Instruct to solve math problems from GSM8K, lookahead decoding achieves 1.8x latency reduction.

## Get started with Lookahead Decoding

We have implemented lookahead decoding in huggingface's transformers. You can accelerate your transformers' decoding API with only a few LoCs. Please check our [GitHub repo](https://github.com/hao-ai-lab/ParallelDecoding) and give us feedback!

## Acknowledgment
We would like to thank Richard Liaw, Yang Song, and Lianmin Zheng for providing insightful feedback.

## Citation

```
@misc{fu2023lookahead,
    title = {Breaking the Sequential Dependency of LLM Inference using Lookahead Decoding},
    url = {https://lmsys.org/blog/2023-11-21-lookahead-decoding/},
    author = {Yichao Fu and Peter Bailis and Ion Stoica and Hao Zhang},
    month = {November},
    year = {2023}

}
```
