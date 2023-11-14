---
title: "Lookahead Decoding: Parallelize Autoregressive Decoding Without A Draft Model"
author: "Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang"
date: "November 14, 2023"
previewImg: /images/blog/laattention/acc-demo.gif
---

Large language models (LLMs) like GPT-4 and LLaMA are rapidly reinventing today's applications, but their inference -- based on autoregressive decoding -- is notoriously slow and difficult to optimize.
This is because each autoregressive decoding step generates only one token at a time; as a result, the latency of an LLM request primarily depends on the response length of the request, which is equal to the number of decoding steps. 
Making matters worse, each decoding step inefficiently leverages the parallel processing power of modern GPUs, often resulting in low GPU utilization.
This challenges many real world LLM applications that prioritize rapid response time, such as chatbots and personal assistants, which require frequently generating *long sequences with low latency*. 

One way to accelerate LLM decoding is [speculative decoding](https://arxiv.org/abs/2211.17192)(and its variants e.g.,[OSD](https://arxiv.org/abs//2310.07177), [Medusa](https://sites.google.com/view/medusa-llm) and [Draft & Verify](https://arxiv.org/pdf/2309.08168.pdf)), which adopts a "guess-and-verify" strategy. 
In particular, they use a small draft model to guess multiple potential future tokens, then calls the original LLM to verify guessed tokens in parallel.
While speculative decoding can opportunistically reduce the number of decoding steps (hence reduced latency), they, however, face several fundamental limitations.
First, the maximum speedup that speculative decoding based methods can achieve is limited by the *token acceptance rate*, or equivalently, how accurately the draft model can predict the target model's outputs.
Second, obtaining an accurate draft model is a non-trivial task, often requiring additional training and careful tuning; deploying draft models in real serving systems can introduce complications.

In this blogpost, we introduce a new, exact decoding algorithm, **Lookahead Decoding**, to solve these two burdens in one shot. 
The key observation driving lookahead decoding is that: while it is impossible to decode multiple next tokens in one step, an LLM can indeed generate multiple, plausible, and disjoint subsequences in parallel; these subsequences may appear in future token position.
We draw insights from previous work that formulates [autoregressive decoding as solving a system of nonlinear equations](https://proceedings.mlr.press/v139/song21a/song21a.pdf), and modernizes a classic solver, [Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method), for LLM decoding.
In lookahead decoding, an LLM first generate multiple, disjoint (n-grams)[https://en.wikipedia.org/wiki/N-gram] *in parallel* using an algorithm derived from [Jacobi iteration](https://en.wikipedia.org/wiki/Jacobi_method).  
These n-grams are captured and later verified; if accepted, they are appended at the end of the generation.  

Lookahead decoding is able to generate n-grams instead of a single token each step, hence it can compress the number of decoding steps -- generating N token using less than N steps. In fact, lookahead decoding can even
- compress the decoding steps *without* a draft model
- Linearly compress decoding steps relative to the per-step log(FLOPs) invested.

See a demo of Lookahead decoding accelerating LLaMa-7B-Chat geneartion: 

<img src="/images/blog/laattention/acc-demo.gif" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Demo of speedups by Lookahead Decoding on LLaMA-2-7B-Chat generation. Blue fonts were n-grams generated in parallel in one decoding step.</p>

We will show that at a moderate compression rate, when the extra FLOPS invested per step is insignificant, lookahead decoding provides a substantial reduction of latency, ranging from 1.5x to 2.3x.

We provide an implementation for Lookahead decoding compatible with `huggingface/transformers`. Users can accelerate HuggingFace's native ```generate```  function with only a few lines of code. Check out our [code](https://github.com/hao-ai-lab/ParallelDecoding) and give it a try!

## Background: Parallel LLM Decoding using Jacobi Iteration

[Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method) is a classic solver for solving a system of nonlinear equations. In the case of LLM decoding, we can use it for parallel generation of multiple tokens from an LLM without a draft model.
To see this, as shown in Figure 1, we can rewrite the autoregressive decoding process as solving a system of nonlinear equations:





instead of generating one token each time autoregressively, Jacobi iteration method will update and verify a future sequence of tokens in parallel. We call this method [*Jacobi decoding*](https://arxiv.org/pdf/2305.10427.pdf).
Interestingly, sometimes multiple tokens might be (opportunistically) accepted in a single Jacobi iteration. When this happens, The number of decoding steps can be reduced. 
For example, in the following figure, the Jacobi decoding can guess several tokens (e.g., token 'computer' and 'scientist') correctly in one step. 
In LLM decoding task, because the accelerator is largely memory bandwidth bounded, the cost of decoding several tokens is similar to decoding only one token. Hence, when Jacobi decoding saves steps from autoregressive decoding, it is highly likely it may also achieve wall-clock speedups.


<img src="/images/blog/laattention/jacobi-iteration.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Illustration of the Jacobi Iteration process for LLM decoding.</p>


**Limitation of Jacobi Decoding**. However, in our study, several problems limit the acceptance rate of Jacobi decoding, making it hard to achieve empirical speedups in real-world applications:

- Vanilla Jacobi decoding can only correctly guess a few tokens. For example, [previous research](https://arxiv.org/pdf/2305.10427.pdf) shows limited speedup (i.e., less than 10%) given a moderate computational budget.
- Although Jacobi decoding can sometimes correctly guess tokens, it is hard to place tokens in the correct position, and even those valid tokens will be frequently substituted.
- Unlike Jacobi-Iterations in solving linear systems, the input and output of LLMs are discrete values, and it is hard to converge like continuous values.

## Lookahead Decoding: Modernize Jacobi Iteration for LLM Decoding

We would modernize Jacobi decoding for today's LLM inference to parallelize autoregressive decoding without a draft model. The key to preserving decoding distribution while reducing the number of steps lies in consistently **looking ahead** to a series of future tokens and simultaneously **verifying** tokens during the decoding process, following the guess-and-verify paradigm. To address the limitation of the Jacobi decoding, we will collect n-grams from the Jacobi decoding generation and stitch them to the current input token to verify the guessed n-grams' correctness. The following gif shows an example of such a decoding process.

<img src="/images/blog/laattention/lookahead-decoding.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Illustration of the proposed lookahead decoding.</p>

To make this decoding process more efficient, we split the decoding branch into a **lookahead branch** and a **verification branch**. The lookahead branch maintains a fixed-sized multi-level window to generate n-gram token candidates. The verification branch verifies multiple token candidates simultaneously to boost the overall acceptance rate. They will be further discussed in the following paragraph.

## Lookahead Branch

We maintain a multi-level lookahead window to generate guess token candidates as in Figure x (i.e., a window=5, level=4 example). The blue token in that figure is the current model input. The orange, green, and red tokens are in level 1, level 2, and level 3, respectively. The number on these tokens shows their relative position to the current input token (blue one with 0). We will generate one token output from the level 3 tokens in each step. Then, we will collect 4-grams from these windows. For example, the orange token 1, green token 2, red token 3, and one newly generated token is a 4-gram we will collect. When the level equals 2, it should degenerate into a Jacobi decoding, as shown above. Then, we will discard the tokens in the oldest level (orange level) and use the new level to fill its past level to maintain both level and window size.

## Verification Branch

Each decoding step undergoes scrutiny by a verification branch. As discussed, we collect n-grams in the lookahead branch. To maintain the generation correctness, we need to verify these n-grams in a verification branch. We need to find all n-grams starting with this token for a specific token input, stitch them to a verification branch, and verify them in a speculative decoding style. It is common to identify multiple n-grams starting with the same token, thereby increasing the cost of the verification process. In practice, we use a maximum verification size to limit the verification branch's overhead within a given budget.

## Decode, Lookahead and Verify In The Same Step

The memory bandwidth-intensive character makes the overhead small to decode several tokens compared with decoding one token per step. This character motivates us to merge decoding, lookahead, and verification in the same step to save overall cost. The current transformer-based model often contains two facets: token-wise computation and token-interaction computation. To amalgamate decoding, lookahead, and verification within a single step, a meticulous restructuring of the decoding style is imperative to preserve the original output distribution. Specifically, it necessitates accurately assigning each token's **relative** and **absolute** positional expression within the decoder model.

We exemplify this with the LLaMA model. The correct configuration of the absolute positional expression is about setting the positional embedding correctly. This setting can be done by passing each token's accurate absolute position index into the model input. The correct configuration of the relative positional expression is to correctly set the attention mask, which illustrates each token's interaction. We implement this by configuring the attention mask in the **verification** and **lookahead** branches. We implement these attention masks following two constrictions: (1) The tokens in the lookahead branch can not see tokens in the verification branch, and vice versa. (2) Each token can only see its previous tokens and itself. Here, "see another token" means that we need not mask these tokens' interaction in the attention mask. An example of an attention mask is shown below for a configuration of level=4 and window size=5 with two n-grams guesses. The left-most token is used to **decode** the next token.

We have implemented the attention mask in PyTorch. Soon, we will release a high-performance CUDA kernel to speed up the execution.

<img src="/images/blog/laattention/mask.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>

<p style="color:gray; text-align: center;">Attention mask for Lookahead Decoding.</p>

## Scaling Computation Resources Can Linearly Increase Compression Ratio

Lookahead Decoding looks up future tokens from n-grams based on the current token input. Unlike speculative decode guessing and verifying **one** prediction at each step, Lookahead decoding can generate several predictions proportional to the window size at each step. According to our further theoretical analysis, when the guess sequence length is long enough (i.e., level in Lookahead Decoding), the exponential investment in future token guesses (and thus window size in Lookahead Decoding) can linearly increase the compression ratio. We call it **scaling law** in the LLM parallel decoding within the guess-and-verify paradigm. Lookahead Decoding can match the parallel decoding scaling law in a certain range in our empirical experiments. We illustrate Lookahead Decoding's matching scaling law in the following figure.


To achieve a high acceptance rate, speculative decoding often requires a draft model to generate a long sequence prediction **autoregressively**. However, Lookahead decoding can consistently generate #window n-gram predictions of length #level for each decoding step. To achieve a higher acceptance rate, Lookahead decoding requires a higher **parallel computational cost per step**. We will discuss the computational overhead next.

<img src="/images/blog/laattention/match-scaling.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>

<p style="color:gray; text-align: center;">When the level is large enough, exponentially increasing window size can almost linearly increase the compression ratio. (LLaMA-2-7B-chat on MT-Bench)</p>

## Overhead of Lookahead Decoding

Here, we follow [Megatron-LM](arxiv.org/pdf/2104.04473.pdf) to estimate the forward computation cost of the LLM. Transformer models' FLOPs are determined by various parameters: sequence length (*s*), model hidden size (*h*), batch size (*B*), vocabulary size (*V*), and the number of layers (*l*) and by this formula:

<img src="/images/blog/laattention/flops.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 30%"></img>

This estimation accounts for the full self-attention mechanism employed in training, slightly different from autoregressive inference. We take the smallest LLaMA-2-7B model as an example. With a hidden size (*h*) of 4k and a relatively small sequence length (*s*) of 1k, the ratio *s/6h* is notably small. In such scenarios, the FLOPs are expected to increase linearly in conjunction with the sequence length (*s*). In the inference phase of Lookahead decoding, the sequence length is proportional to the product of window and level. If the level setting is large enough, an exponential increase in the window size can result in a linear enhancement in compression ratio, as discussed in the preceding section. Thus, we can achieve linear compression of the steps by exponentially increasing the FLOPs within a certain range.

## Experimental Result

We conducted an extensive benchmarking process to evaluate the efficiency of Lookahead Decoding by integrating it with [LLaMA-2-Chat](https://ai.meta.com/llama/) and [CodeLLaMA](https://ai.meta.com/blog/code-llama-large-language-model-coding/) models. Specifically, our tests focused on the 7B, 13B, and 70B parameter configurations of the LLaMA-2-Chat models and the 7B, 13B, and 33B configurations of the CodeLLaMA models. Our objective was to substantiate the acceleration these models attain in practical and real-world applications through different datasets. The design of Lookahead Decoding makes the speedup without any finetune and draft model and preserves the output distribution. The 7B, 13B and 33B models are evaluated on single A100 GPU with FP16 precision. The 70B model is evaluated on two A100 GPUs with pipeline parallelism.

<img src="/images/blog/laattention/fold-perf.png" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Speedup of Lookahead Decoding on Different Datasets.</p>

**MT-Bench Results with LLaMA-Chat** [MT-Bench](https://lmsys.org/blog/2023-06-22-leaderboard/), encompassing a cross-area set of multi-turn questions, served as our testing ground for assessing Lookahead Decoding's overall performance efficacy. Lookahead Decoding achieves roughly 1.5x speedup across several model settings.

**Code Infilling and Code Completion with CodeLLaMA**. Applying Lookahead Decoding to CodeLLaMA on [HumanEval](https://arxiv.org/abs/2107.03374) also shows large speedups (i.e., more than 2x). In the code infilling task, the generation length is relatively short, and in this case, the speedup is relatively low because Lookahead Decoding requires a large number of steps to fill the window and carry it on smoothly. In code completion tasks, many repeated tokens appear in a generation, and the speedup is larger than other datasets.

**Instructional Coding Task and Math Problem Solving with CodeLLaMA-Instruct**. Finally, we evaluate Lookahead Decoding's performance on instructional coding tasks and solving math problems. For the [GSM8K](https://arxiv.org/abs/2110.14168) dataset, we evaluated CodeLLaMA-Instruct on the first 1K questions. And the instructional coding task is evaluated on [MBPP](https://arxiv.org/abs/2108.07732). Results show that Lookahead Decoding can bring more than 1.8x speedups on these settings.

## Get started with Lookahead Decoding

We have encapsulated the dedicated implementation of Lookahead Decoding in a Python library, and it is easy to use with huggingface's transformers. You can accelerate your transformers' decoding API with only a few LoCs. Please check our [GitHub repo](https://github.com/hao-ai-lab/ParallelDecoding)!

## Acknowledgment

We would like to thank .

## The Team

The Lookahead Decoding and this blog post are developed, evaluated, and maintained by the following

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
