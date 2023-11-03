---
title: "Rethinking Benchmarks and Contamination for Language Models with Rephrased Samples"
author: "Shuo Yang"
date: "Nov 6, 2023"
previewImg: /images/blog/decontaminator/overview.png
---


Many have raised concerns about the trustworthiness of public benchmarks due to potential contamination in pre-training or fine-tuning datasets.
In this blog post, we show that existing methods are insufficient. We propose [LLM decontaminator](https://github.com/lm-sys/llm-decontaminator) and reveal significant test overlap in real-world datasets.

\todo{add paper url?}


## **Existing Detection Methods**

Despite being recognized as a crucial issue, accurately detecting contamination remains an open and challenging problem. 
Here we introduce the most commonly used approaches, n-gram overlap and embedding similarity search.

1. N-gram overlap  
  N-gram overlap relies on string matching to detect contamination, widely used by leading developments such as GPT-4, PaLM, and Llama. Although it is fast and easy to use, its precision is limited.

2. Embedding similarity search  
  Embedding similarity search uses the embeddings of pre-trained models (e.g., BERT) to find similar examples. High similarity between training and test prompts suggests potential contamination.
  Although it is more robust than n-gram overlap detection, the requirement to specify a threshold and its low precision prevent it from being widely adopted.


## **Rephrased Samples**

While most data decontamination efforts apply the detection methods above, we show that these methods are insufficient, and simple rephrasing of the test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures.
Here is a rephrased sample of GSM-8k benchmark.

<img src="/images/blog/decontaminator/gsm-8k-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

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

<img src="/images/blog/langchain/real-world-detect" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

Here we show some detected samples.

<img src="/images/blog/langchain/codealpaca-rephrase" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

<img src="/images/blog/langchain/MATH-rephrase" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

<img src="/images/blog/langchain/starcoder-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>



## **Acknowledgment**
\todo
<!-- The OpenAI-compatible API server is primarily contributed by Shuo Yang, Siyuan Zhuang, and Xia Han. -->
