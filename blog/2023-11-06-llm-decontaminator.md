---
title: "Rethinking Benchmarks and Contamination for Language Models with Rephrased Samples"
author: "LMSYS ORG"
date: "Nov 6, 2023"
previewImg: /images/blog/decontaminator/overview.png
---


Many have raised concerns about the trustworthiness of public benchmarks due to potential contamination in pre-training or fine-tuning datasets.
In this blog post, we show that existing methods are insufficient. We propose [LLM decontaminator](https://github.com/lm-sys/llm-decontaminator) and reveal significant test overlap in real-world datasets.

\todo{add paper url?}


## **Existing Detection Methods**

Despite being recognized as a crucial issue, accurately detecting contamination remains an open and challenging problem. 
Here we introduce the most commonly used approaches, n-gram overlap and embedding similarity search.

  1. **N-gram overlap**  
    N-gram overlap relies on string matching to detect contamination, widely used by leading developments such as GPT-4, PaLM, and Llama. Although it is fast and easy to use, its precision is limited.
  2. **Embedding similarity search**  
    Embedding similarity search uses the embeddings of pre-trained models (e.g., BERT) to find similar examples. High similarity between training and test prompts suggests potential contamination.
    Although it is more robust than n-gram overlap detection, the requirement to specify a threshold and its low precision prevent it from being widely adopted.


## **Rephrased Samples**

While most data decontamination efforts apply the detection methods above, we show that these methods are insufficient, and simple rephrasing of the test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures.
Here is a rephrased sample of GSM-8k benchmark.

<img src="/images/blog/decontaminator/gsm-8k-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

Our aim is to simulate realistic possible contamination scenarios, so introducing extraneous or nonsensical content to avoid detection is not a valid approach.
For text-based benchmarks, we rephrase test cases without altering their meanings, such as by rearranging word order or substituting with synonymous terms. For code-based benchmarks, we vary coding styles, naming conventions, and algorithms.

Regarding the rephrasing process, we present a general algorithm for any given test set. This method helps to blend the test set into the training data without being detected. It employs a high-quality large language model to produce a rephrased version of the test prompt and utilizes detection like n-gram overlap to validate the efficacy of the rephrasing. To encourage diverse outputs, we set a non-zero initial temperature, prompting the model to generate variations that are less likely to trigger detection mechanisms.

Furthermore, we demonstrate that if such rephrased samples are not eliminated, a 13B model can easily overfit the test benchmark.
Trained on rephrased samples of MMLU, HumanEval and GSM-8k, Llama-2 13B achieved drastically high performance, on par with GPT-4's performance.

<img src="/images/blog/decontaminator/rephrase-score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


## **LLM Decontaminator VS Rephrased Samples**

To address the risk of rephrased samples, we propose a new contamination detection method ``LLM decontaminator''.
It can accurately remove a dataset's rephrased samples relative to a benchmark.

This LLM decontaminator involves two steps:

  1. For each test case, LLM decontaminator identifies the top-k training items with the highest similarity using the embedding similarity search.
  2. From these items, LLM decontaminator generates k potential rephrased pairs. Each pair is evaluated for rephrasing using an advanced LLM, such as GPT-4.

To compare the accuracy of different detection method, we construct 200 prompt pairs using both the original and rephrased test sets. These comprised 100 random pairs and 100 rephrased pairs.
The f1 score on these pairs provides insight into the rephrased samples' ability to evade detection, with lower values indicating more effective evasion.
Notably, the LLM decontaminator showcases superior performance, identifying rephrased samples with high consistency and precision.

<img src="/images/blog/decontaminator/MMLU-f1score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

## **LLM Decontaminator VS Real-World Dataset**

We apply the LLM decontaminator to several renowned real-world datasets and identify a substantial amount of rephrased samples. 
The table below displays the contamination percentage of different benchmarks in each training dataset.

<img src="/images/blog/decontaminator/real-world-detect.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

Based on the results of prior experiments, it is likely that these rephrased samples exaggerate the benchmark results.
Here we show some detected samples.

[CodeAlpaca](https://github.com/sahil280114/codealpaca) contains 20K instruction-following data used for fine-tuning the CodeAlpaca model. 
It is used to train a number of well-known models, including [Tulu](https://huggingface.co/TheBloke/tulu-30B-fp16).
A rephrased example in CodeAlpaca is shown below.

<img src="/images/blog/decontaminator/codealpaca-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[MATH](https://github.com/hendrycks/math) is a widely recognized math training dataset that spans various mathematical domains, including algebra, geometry, and number theory. Below is a self-contamination case in MATH.

<img src="/images/blog/decontaminator/MATH-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[StarCoder-Data](https://huggingface.co/datasets/bigcode/starcoderdata) is used for training StarCoder and StarCoderBase, and it contains 783GB of code in 86 programming languages. In the StarCoder [paper](https://arxiv.org/pdf/2305.06161.pdf), the code training data was decontaminated by removing files that contained docstrings or solutions from HumanEval. However, there are still some samples detected by LLM decontaminator.

<img src="/images/blog/decontaminator/starcoder-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>



## **Acknowledgment**
\todo
<!-- The OpenAI-compatible API server is primarily contributed by Shuo Yang, Siyuan Zhuang, and Xia Han. -->
