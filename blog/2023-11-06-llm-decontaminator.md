---
title: "Rethinking Benchmarks and Contamination for Language Models with Rephrased Samples"
author: "LMSYS ORG"
date: "Nov 6, 2023"
previewImg: /images/blog/decontaminator/overview.png
---


Many have raised concerns about the trustworthiness of public benchmarks due to potential contamination in pre-training or fine-tuning datasets.
In this blog post, we show that existing methods are insufficient. We propose [LLM decontaminator](https://github.com/lm-sys/llm-decontaminator) and reveal significant test overlap in real-world datasets.

<!-- <img src="/images/blog/decontaminator/overview.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img> -->


## **Existing Detection Methods**

Despite being recognized as a crucial issue, accurately detecting contamination remains an open and challenging problem. 
Here we introduce the most commonly used approaches, n-gram overlap and embedding similarity search.

  **N-gram overlap** relies on string matching to detect contamination, widely used by leading developments such as GPT-4, PaLM, and Llama. Although it is fast and easy to use, it is hard to detect test cases with simple variation.


  **Embedding similarity search** uses the embeddings of pre-trained models (e.g., BERT) to find similar examples. High similarity between training and test prompts suggests potential contamination.
   Although it can capture more semantic information than n-gram overlap, it requires specifying a threshold. 
   If the threshold is set too high, it will result in a high false negative rate; otherwise, setting it too low will lead to a high false positive rate.



## **Rephrased Samples**

While most data decontamination efforts apply the detection methods above, we show that simple variation of the test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures.
We refer to such variations of test cases as _Rephrased Samples_.
Here is a rephrased sample of GSM-8k benchmark.

<img src="/images/blog/decontaminator/gsm-8k-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

For text-based benchmarks, we rephrase test cases without altering their meanings, such as by rearranging word order or substituting with synonymous terms. For code-based benchmarks, we vary coding styles, naming conventions, and algorithms.

Furthermore, we demonstrate that if such rephrased samples are not eliminated, a 13B model can easily overfit the test benchmark.
Trained on rephrased samples of MMLU, HumanEval and GSM-8k, Llama-2 13B achieved drastically high performance, on par with GPT-4's performance.

<img src="/images/blog/decontaminator/rephrase-score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


## **LLM Decontaminator VS Rephrased Samples**

To address the risk of possible contamination, we propose a new contamination detection method ``LLM decontaminator''.
It can accurately remove a dataset's rephrased samples relative to a benchmark.

This LLM decontaminator involves two steps:

  1. For each test case, LLM decontaminator identifies the top-k training items with the highest similarity using the embedding similarity search.
  2. From these items, LLM decontaminator generates k potential rephrased pairs. Each pair is evaluated for rephrasing using an advanced LLM, such as GPT-4.

To compare the accuracy of different detection method, we construct 200 prompt pairs using both the original and rephrased test sets. These comprised 100 random pairs and 100 rephrased pairs.
The f1 score on these pairs provides insight into the rephrased samples' ability to evade detection, with lower values indicating more effective evasion.
As shown in the following table, except for the LLM decontaminator, all other detection methods introduce some false positives. Both rephrased and translated samples successfully evade the n-gram overlap detection. With multi-qa BERT, the embedding similarity search proves completely ineffective against translated samples. When using multilingual BERT, this method struggles with the US History subject. Notably, the LLM decontaminator showcases superior performance, identifying rephrased samples with high reliability and precision.

<img src="/images/blog/decontaminator/MMLU-f1score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

## **LLM Decontaminator VS Real-World Dataset**

We apply the LLM decontaminator to widely used real-world datasets and identify a substantial amount of rephrased samples. 
The table below displays the contamination percentage of different benchmarks in each training dataset.

<img src="/images/blog/decontaminator/real-world-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

Here we show some detected samples.

[CodeAlpaca](https://github.com/sahil280114/codealpaca) contains 20K instruction-following data used for fine-tuning the CodeAlpaca model. 
It is used to train a number of well-known models, including [Tulu](https://huggingface.co/TheBloke/tulu-30B-fp16).
A rephrased example in CodeAlpaca is shown below.

<img src="/images/blog/decontaminator/codealpaca-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[MATH](https://github.com/hendrycks/math) is a widely recognized math training dataset that spans various mathematical domains, including algebra, geometry, and number theory. Below is a self-contamination case in MATH.

<img src="/images/blog/decontaminator/MATH-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[StarCoder-Data](https://huggingface.co/datasets/bigcode/starcoderdata) is used for training StarCoder and StarCoderBase, and it contains 783GB of code in 86 programming languages. In the StarCoder [paper](https://arxiv.org/pdf/2305.06161.pdf), the code training data was decontaminated by removing files that contained docstrings or solutions from HumanEval. However, there are still some samples detected by LLM decontaminator.

<img src="/images/blog/decontaminator/starcoder-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

## **Use LLM Decontaminator Now!**

The issue of unintentional contamination is becoming increasingly severe because more and more datasets are generated by LLMs. 
Since LLMs always generate data similar to their training data, these generated data might contain rephrased samples. For instance, CodeAlpaca uses GPT to generate training data, which include rephrased samples of HumanEval. 
Here we show how to remove rephrased samples from training data using the LLM decontaminator. The following example can be found [here](https://github.com/lm-sys/llm-decontaminator#detect).

[Pre-process](https://github.com/lm-sys/llm-decontaminator#pre-process) training data and test data.
The LLM decontaminator accepts the dataset in jsonl format, with each line corresponding to a `{"text": data}` entry.

Run [End2End](https://github.com/lm-sys/llm-decontaminator#end2end) detection.
The following command builds a top-k similar database based on sentence bert and uses GPT-4 to check one by one if they are rephrased samples. You can select your embedding model and detection model by modifying the parameters.

<img src="/images/blog/decontaminator/run-e2e.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>



## **Acknowledgment**
\todo
<!-- The OpenAI-compatible API server is primarily contributed by Shuo Yang, Siyuan Zhuang, and Xia Han. -->
