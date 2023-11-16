---
title: "Breaking the Sequential Dependency of LLM Inference using Lookahead Decoding"
author: "Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang"
date: "November 16, 2023"
previewImg: /images/blog/laattention/acc-demo.gif
---

Large language models (LLMs) like GPT-4 and LLaMA are rapidly reinventing today's applications, but their inference -- based on autoregressive decoding -- is very slow and difficult to optimize. Each autoregressive decoding step generates only one token at a time; as a result, the latency of an LLM request primarily depends on the response length of the request, or equipvalently the number of decoding steps. 
Making matters worse, each decoding step cannot effectively leverages the parallel processing capabilities of modern GPUs, often leading to GPU underutilization.
This challenges many real world LLM applications that prioritize rapid response time, such as chatbots and personal assistants, which require frequently generating *long sequences with low latency*. 

One way to accelerate autoregressive decoding is [speculative decoding](https://arxiv.org/abs/2211.17192) (including [Medusa](https://sites.google.com/view/medusa-llm) and [OSD](https://arxiv.org/abs//2310.07177)), which employ a "guess-and-verify" strategy: a smaller draft model predicts several potential future tokens, and the original LLM then verifies these guesses in parallel. 
These apporaches can opportunistically reduce the number of decoding steps and, consequently lower latency. However, they face several limitations.
First, the maximum speedup that speculative decoding based methods can achieve is limited by the *token acceptance rate*, or equivalently how accurately the draft model can predict the main model's outputs. Second, creating an accurate draft model is non-trivial, often requiring extra training and careful tuning; Incoparating these draft models into real LLM services may introduce new complications.

In this blogpost, we introduce a new, exact decoding algorithm, **Lookahead Decoding**, designed to address these challenges.
The key observation enabling lookahead decoding is: although it is infeasible to immediately decode multiple next tokens in one step, an LLM can indeed generate multiple, disjoint, yet plausible [n-grams]((https://en.wikipedia.org/wiki/N-gram)) in parallel. These n-grams could potentially fit into future positions of the generated sequence.
In lookahead decoding, an LLM first generates multiple, disjoint n-grams *in parallel*. This is achieved by viewing [autoregressive decoding as a process of solving nonlinear equations](https://proceedings.mlr.press/v139/song21a/song21a.pdf), and modernizing the classic [Jacobi iteration method](https://en.wikipedia.org/wiki/Jacobi_method) for parallel decoding. These n-grams are captured and later verified -- if deemed suitable, are incorporated into the ongoing sequence generation.

Lookahead decoding is able to generate n-grams in each step, as opposed to producing just one token, hence it can reduce the number of decoding steps -- generating N tokens in fewer than N steps. In fact, lookahead decoding stands out because it:
- Operates **without** a draft model, simplifying the deployment process.
- Linearly reduces the number of decoding steps in relation to the logarithm of the FLOPs invested per step.

See a demo of lookahead decoding accelerating LLaMa-7B-Chat geneartion: 

<img src="/images/blog/laattention/acc-demo.gif" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 1: Demo of speedups by Lookahead Decoding on LLaMA-2-7B-Chat generation. Blue fonts were tokens generated in parallel in one decoding step.</p>

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
Jacabi decoding can guarantee solving all *m* variables in at most *m* steps (same number of steps as autoregressive decoding), because each step at least has the very first token correctly decoded. 
Sometimes, multiple tokens might converge in a single iteration, potentially reducing the overall number of decoding steps. For example, as shown in Figure 3, Jacobi decoding accurately predicts and accepts two tokens "computer" and "scientist" in Step 4. Compared to autoregressive decoding, each Jacobi decoding step is slightly more expensive in terms of FLOPs needed, because it requires LLM forward computation on >1 token. Fortunately, this would not translate into slowdowns thanks to the parallel processing nature of acclerators.
<img src="/images/blog/laattention/jacobi-iteration.gif" style="width: 100%; max-width: 100%; margin-left: auto; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Figure 3: Illustration of applying Jacobi iteration method for parallel LLM decoding.</p>


### Limitations of Jacobi Decoding 

In pracice, we find that Jacobi decoding faces several challenges that impede achieving significant wallclock speedup:

- While Jacobi decoding can decode multiple token sequences simultaneously, the precise positioning of these tokens within the sequence is problematic. Correctly predicted tokens are frequently replaced in later iterations. Consequently, only a small number of iterations observe multiple tokens being correctly decoded at once.
- The discrete nature of LLM inputs and outputs complicates convergence, unlike the continuous values typically involved in traditional nonlinear systems.

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

**How do you configure windows and levels in practice?** Lookahead decoding should be used in scenarios where latency is vital. Intuitively, a larger window and level will lead to a more considerable per-step cost and a more significant step compression ratio. For powerful GPUs (e.g., A100), you can better squeeze its performance (i.e., larger window and level) to achieve a larger step compression ratio (i.e., lower long sequence generation latency). If the window and level are too large, it will be costly to use Lookahead decoding and slow down the decoding despite the large compression ratio. Increasing the level together with the window to achieve higher performance would be best, avoiding hitting a theoretical upper bound by only increasing one side. Our experimental results show that on A100, the following setting can be optimal in most cases. 

| Model Setting | window | level |
|----: |:----:  | :----: |
| 7B| 15| 5 |
| 13B | 10 | 5|
| 33B | 7 | 5 |

You can also change the setting to tune a better performance on your specific decoding settings. We will release an auto-tuner to better schedule a window and level on a given model, dataset, and hardware setting. 

## Overhead of Lookahead Decoding

Here, we follow [Megatron-LM](arxiv.org/pdf/2104.04473.pdf) to estimate the forward computation cost of the LLM. Transformer models' FLOPs are determined by various parameters: sequence length (*s*), model hidden size (*h*), batch size (*B*), vocabulary size (*V*), and the number of layers (*l*) and by this formula:

<img src="/images/blog/laattention/flops.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 30%"></img>

This estimation accounts for the full self-attention mechanism employed in training, slightly different from autoregressive inference. We take the smallest LLaMA-2-7B model as an example. With a hidden size (*h*) of 4k and a relatively small sequence length (*s*) of 1k, the ratio *s/6h* is notably small. In such scenarios, the FLOPs are expected to increase linearly in conjunction with the sequence length (*s*). In the inference phase of Lookahead decoding, the sequence length is proportional to the product of *window* and *level - 1*. If the level setting is large enough, an exponential increase in the window size can result in a linear enhancement in compression ratio, as discussed in the preceding section. Thus, we can achieve linear compression of the steps by exponentially increasing the FLOPs within a certain range.

## Experimental Result

We conducted an extensive benchmarking process to evaluate the efficiency of Lookahead Decoding by integrating it with [LLaMA-2-Chat](https://ai.meta.com/llama/) and [CodeLLaMA](https://ai.meta.com/blog/code-llama-large-language-model-coding/) models. Specifically, our tests focused on the 7B, 13B, and 70B parameter configurations of the LLaMA-2-Chat models and the 7B, 13B, and 33B configurations of the CodeLLaMA models. Our objective was to substantiate the acceleration these models attain in practical and real-world applications through different datasets. The design of Lookahead Decoding makes the speedup without any finetune and draft model and preserves the output distribution. The 7B, 13B and 33B models are evaluated on single A100 GPU with FP16 precision. The 70B model is evaluated on two A100 GPUs with pipeline parallelism.

<img src="/images/blog/laattention/lookahead-perf.png" style="width: 200%; max-width: 100%; margin-right: auto; margin-bottom: auto"></img>

<p style="color:gray; text-align: center;">Speedup of Lookahead Decoding on Different Datasets.</p>

**MT-Bench Results with LLaMA-Chat** [MT-Bench](https://lmsys.org/blog/2023-06-22-leaderboard/), encompassing a cross-area set of multi-turn questions, served as our testing ground for assessing Lookahead Decoding's overall performance efficacy. Lookahead Decoding achieves roughly 1.5x speedup across several model settings.

**Code Completion with CodeLLaMA**. Applying Lookahead Decoding to CodeLLaMA on [HumanEval](https://arxiv.org/abs/2107.03374) also shows large speedups (i.e., more than 2x). In code completion tasks, many repeated tokens appear in a generation, and the speedup is relatively larger than other datasets.

**Math Problem Solving with CodeLLaMA-Instruct**. Finally, we evaluate Lookahead Decoding's performance on solving math problems. For the [GSM8K](https://arxiv.org/abs/2110.14168) dataset, we evaluated CodeLLaMA-Instruct on the first 1K questions. Results show that Lookahead Decoding can bring more than 1.8x speedups on these settings.

**Per-Step Overhead with Lookahead decoding** Despite the wall-clock time reduction in the previous settings, Lookahead decoding actually requires much larger per-step FLOPs to achieve a #step compression. We set a guess token limit to limit the most number of guess n-grams per decoding step. The #token we need to decode per-step will be at most *(window + guess) * (level - 1)*. This number is approximately a multiple of the Vallina autoregressive decoding FLOPs. So we need to have roughly 120x extra FLOPs for 7B models (with level=5, window=15, and guess=15) and 56x extra FLOPs for 33B models (with level=5, window=7 and guess=7) in the previous experiments. Because of the memory-intensive bound characteristic of the LLM decoding, these extra FLOPs only bring little per-step cost and a visible step compression ratio, resulting in a notable speedup.


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
