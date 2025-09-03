---
title: "Accelerating DeepSeek V3 Inference via Multi-Token Prediction in SGLang"
author: "Chang Liu, Andy Luo, AMD AIG Team"
date: "September 02, 2025"
previewImg: /images/blog/amd_mtp/perf_comparison_on_random.png
---

# Accelerating DeepSeek V3 Inference via Multi-Token Prediction in SGLang

21 August 2025, Chang Liu, Andy Luo

## Multi-Token Prediction

Multi-Token Prediction (MTP) was introduced in DeepSeek V3/R1 to enhance
the training performance. It transforms the implicit causal chain
between sequential tokens into an explicit way, so as to improve the
accuracy of the predicted tokens and enhance the alignment of
intermediate embeddings with the causal chain. Due to its design and
module architecture, MTP can also be employed in the speculative
decoding module during inference. Its capability to predict multiple
tokens at once makes it well-suited to serve as the draft model in the
speculative decoding workflow.

In this blog, we explore the effectiveness of enabling MTP in DeepSeek
V3 during serving and inference. All the experiments are conducted using
SGLang as the serving engine, running on AMD MI300X GPUs. Across our
experiments, we observe a 1.2–2.1× speedup in inference efficiency.

In the following sections, we begin by presenting the performance gains
when enabling MTP in DeepSeek V3 serving. Then, we explain the module
architecture and core concept of MTP. Finally, we provide detailed,
step-by-step instructions for reproducing the benchmark results. All the
experiments are conducted on AMD MI300X GPUs with full support from its
bottom-to-top stack, which we give a brief introduction in the last
section of the blog.

## Performance Gain of DeepSeek V3

In this blog, we use DeepSeek V3 as the base model and its NextN module
as the MTP module for speculative decoding. Both of the model weights
are downloaded from HuggingFace. We leverage SGLang as the LLM serving
engine. For more details of steps to reproduce the experimental results,
please refer to the following section in this blog.

### Random Dataset

We first conduct the performance benchmark on a random dataset and as
shown in Table 1, there is a 1.25-2.11x speedup when enabling MTP in
DeepSeek V3 serving. In the experiment, we set the max concurrencies as
1, 2, 4, 8, 16, 32 and 64. It can be observed that the speedup ratio
decreases along with the increasing value of max concurrency.

We leverage total throughput and end-to-end latency as the benchmarking
metrics. As illustrated in Figure 1, we set its x-axis as end-to-end
latency and its y-axis as total throughput and visualize the performance
of DeepSeek V3 without MTP and with MTP enabled into two curves. The
curve of enabling MTP indicates that MTP is effective in reducing
end-to-end latency and enhancing total throughput compared to the
serving without MTP.

*Table 1. End-to-end latency comparison serving DeepSeek V3 without and
with MTP enabled on **random** dataset, running on 8 AMD MI300X
accelerators, with different max concurrencies as 1, 2, 4, 8, 16, 32 and
64. There is a 2.11x speedup when max concurrency is set as 1.*

| **Max Concurrency** | **Without MTP** | **With MTP** | **Speedup Ratio** |
| ------------------- | --------------- | ------------ | ----------------- |
| 1                   | 17348.25        | 8229.37      | 2.11              |
| 2                   | 17162.29        | 8408.02      | 2.04              |
| 4                   | 17948.3         | 9736.64      | 1.84              |
| 8                   | 19919.06        | 11742.82     | 1.70              |
| 16                  | 23708.94        | 15483.11     | 1.53              |
| 32                  | 30868.39        | 21399.39     | 1.44              |
| 64                  | 43848.2         | 35074.26     | 1.25              |

Data measured on 18/08/2025. End-to-end latency indicates the total time
taken for a user sending a request and getting a response back, and we
use milliseconds per request as its unit. All the experiments are
conducted on 8 AMD MI300X GPUs, with ROCm 6.3.0 and SGLang v0.4.10.post1
installed in the system environment.

![Figure 1. Performance comparison of disabling and enabling MTP in DeepSeek V3 serving with total throughput/end-to-end latency on Random dataset.](/images/blog/amd_mtp/perf_comparison_on_random.png)

### ShareGPT

Then we extend the performance benchmark to one of the real dataset,
ShareGPT, consisting of real-world dialogues. As shown in Table 2,
similar to the performance gain on the Random dataset, there is a
1.36-2.08x speedup with MTP enabled in DeepSeek V3 serving. The speedup
ratio goes down when the max concurrency increases.

As illustrated in Figure 2, we visualize the performance of DeepSeek V3
serving according to total throughput and end-to-end latency. Enabling
MTP in DeepSeek V3 serving costs less time and increases throughput
compared to serving DeepSeek V3 without MTP.

*Table 2. End-to-end latency comparison serving DeepSeek V3 without and
with MTP enabled on **ShareGPT** dataset, running on 8 AMD MI300X
accelerators, with different max concurrencies as 4, 8, 16, 32 and 64.*

| **Max Concurrency** | **Without MTP** | **With MTP** | **Speedup Ratio** |
| ------------------- | --------------- | ------------ | ----------------- |
| 4                   | 3981.91         | 2215.44      | 1.80              |
| 8                   | 5007.75         | 2854.36      | 1.75              |
| 16                  | 6255.86         | 3947.93      | 1.58              |
| 32                  | 7571.25         | 5023.23      | 1.51              |
| 64                  | 11900.36        | 8763.65      | 1.36              |

Data measured on 18/08/2025. End-to-end latency indicates the total time
taken for a user sending a request and getting a response back, and we
use milliseconds per request as its unit. All the experiments are
conducted on 8 AMD MI300X GPUs, with ROCm 6.3.0 and SGLang v0.4.10.post1
installed in the system environment.

![Figure 2. Performance comparison of disabling and enabling MTP for DeepSeek V3 serving with total throughput/end-to-end latency on ShareGPT dataset.](/images/blog/amd_mtp/perf_comparison_on_sharegpt.png)

## DeepSeek MTP

In DeepSeek V3/R1, a Multi-Token Prediction (MTP) objective module is
designed to extend the model’s prediction scope to multiple future
tokens at each position. This approach not only enriches the training
signals but also strengthens the model's internal representations,
leading to more accurate predictions of future tokens.

### MTP Module Architecture

As shown in Figure 3, MTP employs D sequential modules to predict D
future tokens. Each MTP module consists of four components: a shared
embedding layer, a shared output head, a Transformer block, and a
projection matrix.

For the i-th input token t\_i, at prediction depth k, the model first
combines the hidden representation from the previous depth h\_i^{k-1}
with the embedding of the target future token using the linear project.
This produces an intermediate representation h'\_i^k, which is then fed
into the Transformer block to generate the updated hidden
state h\_i^k at the current depth. Finally, the output
head takes h\_i^k as input and produces a probability
distribution P^k\_{i+1+k} for the k-th predicted token. The predicted
token is obtained by applying a Softmax function over this distribution.

### MTP Workflow

First, the input prompt is tokenized as N tokens. Set N = 4 in this
example, and we get: t₁, t₂, t₃, t₄. The model processes these tokens
and produces corresponding hidden states: h₁, h₂, h₃, h₄, h₅. About h₅,
it is a particular output, which represents the model’s final hidden
state and is used to predict the next token, t₅, which follows the input
sequence.

After the main model predicts t₅, the MTP modules are then used to
generate further token predictions. Each future token generated by MTP
takes corresponding hidden states from the main model and processes them
to produce new hidden states. Let’s continue the example of N = 4 in
this case:

  - For predicting the first future token (k=1) using MTP, the MTP
    modules take the hidden states h₁ to h₄ from the main model and
    processes them to produce new hidden states at the depth k=1.

  - Iteratively, predicting the second future token (k=2) takes the
    hidden states h₁ to h₃ produced by MTP (k=1) and generating updated
    representations at depth k=2, which are used to predict t₇.

And the following tokens are computed following the same rule with
hidden states produced by the previous depth (k-1).

![Figure 3. Module architecture of MTP.](/images/blog/amd_mtp/MTP_model_architecture.png)

### MTP Objective

Accordingly, the training objective for each MTP module can be
understood separately for each prediction depth. For depth k = 1, the
loss function is defined as:

  - Lₘₜₚ = Loss(P₃, t₃) + Loss(P₄, t₄) + Loss(P₅, t₅).

MTP at the depth k=1, predicts tokens t₃, t₄, and t₅, and we compute the
loss by comparing its predictions P₃, P₄, and P₅ to the corresponding
tokens produced by the input. In general, the total MTP objective is the
sum of the losses from all depths. This follows the formulation
presented in the original paper. The way MTP is supervised during
training helps guiding the base model, DeepSeek V3/R1 generate more
aligned intermediate outputs along with the causal chain.

Due to its architecture, predicting multiple future tokens, MTP can be
naturally adapted to speculative decoding in DeepSeek V3/R1 inference.
At the same time, the weights of MTP have been trained along with the
base model, meeting the requirement that the prediction distribution
between the base model and the draft model needs to be consistent and
also saving extra training resources. In the following section, we show
the detailed instructions to reproduce the performance gain posted in
this blog.

## Steps to Reproduce

### Launch the Docker Environment

In this post, we leverage the official SGLang docker image:
lmsysorg/sglang:v0.5.0rc0-rocm630-mi30x. Launch a docker container using
the command as below, mounting your local working directory using '-v'.

```bash
docker run -d -it --ipc=host --network=host --privileged \
--cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--shm-size=192g \
--name sglang_mtp \
-v /home/models/:/models -v /home/:/work \
lmsysorg/sglang:v0.5.0rc0-rocm630-mi30x
```

### Download the Model Weights

Please navigate to the directory where you can save the downloaded model
weights. In our case, it’s the directory “/models“. Use the commands
below:

```bash
# Download DeepSeek R1 weight
huggingface-cli download --resume-download \
--local-dir-use-symlinks False \
deepseek-ai/DeepSeek-R1 \
--local-dir DeepSeek-R1

# Download NextN (MTP) weight
huggingface-cli download --resume-download \
--local-dir-use-symlinks False \
lmsys/DeepSeek-R1-NextN \
--local-dir DeepSeek-R1-NextN
```

### Launch the Server

In this post, we adopt DeepSeek-V3 and its MTP module, DeepSeek-V3-NextN
as the base model and the draft model of the speculative decoding
settings.

```bash
# Enable MTP
python3 -m sglang.launch_server \
--model-path /models/DeepSeek-V3/ \
--attention-backend aiter \
--port 8000 \
--host 0.0.0.0 \
--trust-remote-code \
--tp-size 8 \
--enable-metrics \
--mem-fraction-static 0.85 \
--chunked-prefill-size 131072 \
--speculative-algorithm NEXTN \
--speculative-draft-model-path /models/DeepSeek-V3-NextN/ \
--speculative-num-steps 2 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 3 \
--speculative-accept-threshold-single=0.001

# Baseline
python3 -m sglang.launch_server \
--model-path /models/DeepSeek-V3/ \
--attention-backend aiter \
--port 8000 \
--host 0.0.0.0 \
--trust-remote-code \
--tp-size 8 \
--enable-metrics \
--mem-fraction-static 0.85 \
--chunked-prefill-size 131072
```

Below is a summary of the parameters used in the command and their
respective functions:

  - \--disable-radix: Disables RadixAttention during prefix caching.

  - \--speculative-num-steps: Specifies the number of decoding steps to
    sample from the draft model during speculative decoding.

  - \--speculative-eagle-topk: Defines the number of top tokens sampled
    from the draft model at each step in the Eagle2 algorithm.

  - \--speculative-num-draft-tokens: Sets the number of tokens to be
    sampled from the draft model in speculative decoding.

  - \--max-running-requests: Limits the maximum number of concurrent
    running requests.

  - \--schedule-conservativeness: Controls how conservative the
    scheduling policy is. Higher values make the scheduler more
    cautious. Increase this if you notice frequent request retractions.

  - \--cuda-graph-max-bs: Sets the maximum batch size for CUDA graph
    execution.

  - \--speculative-algorithm: Specifies the speculative decoding
    algorithm to use. Options include {EAGLE, EAGLE3, NEXTN}.

  - \--speculative-draft-model-path: Path to the draft model's weights.
    This can be a local directory or a Hugging Face repository ID.

  - \--mem-fraction-static: Defines the fraction of total memory
    allocated for static use (such as model weights and the KV cache
    pool). Reduce this value if out-of-memory errors occur.

### Set the Client

```bash
# client command for MTP enabled benchmark
python3 -m sglang.bench_serving --backend sglang \
--dataset-name random --num-prompt 300 \
--request-rate 1 \
--random-input 1024 --random-output 1024 \
> sglang_mtp_1024_1024

# client command for baseline benchmark
python3 -m sglang.bench_serving --backend sglang \
--dataset-name random --num-prompt 300 \
--request-rate 1 \
--random-input 1024 --random-output 1024 \
> sglang_base_1024_1024
```

## Speculative Algorithm - EAGLE

When enabling MTP for DeepSeek V3 inference with SGLang, we specify the
speculative decoding algorithm as EAGLE by including the option
--speculative-algo=NEXTN or EAGLE in the serving launch command. EAGLE,
introduced in early 2024, is an advanced variant of speculative
decoding. As previously discussed, MTP can function as a speculative
decoding module to accelerate DeepSeek R1 inference by reusing its
weights. However, it must operate within a speculative decoding
framework. In our setup, EAGLE serves as the foundational workflow for
this integration.

## Implementation on AMD Stacks

**ROCm** – ROCm is an open-source software platform optimized to deliver
high performance for HPC and AI workloads on AMD Instinct™ accelerators
and AMD Radeon™ GPUs, while maintaining compatibility with widely used
industry software frameworks.

**AITER** – AITER is AMD’s centralized repository that provides a wide
range of high-performance AI operators for accelerating AI workloads. It
serves as a unified hub for customer operator-level requests, enabling
tailored solutions that meet diverse customer needs. For MTP, enabling
AITER is essential when serving with SGLang.

**SGLang** – SGLang is an open-source framework designed to meet the
demands of modern AI by offering a fast backend runtime, a flexible
frontend language, and broad model support for a variety of LLMs and
VLMs. AMD collaborates closely with the SGLang team to iteratively
deliver the most advanced features.

Open-source contributions and feedback are warmly welcomed by the AMD
community\!
