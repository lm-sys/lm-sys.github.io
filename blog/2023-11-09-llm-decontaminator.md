---
title: "Cache me if you can! How to beat GPT-4 with a 13B model"
author: "LMSYS ORG"
date: "Nov 9, 2023"
previewImg: /images/blog/decontaminator/rephrase-score_with_border.png
---


Announcing Llama-Frank: 13B models reaching GPT-4 performance in major benchmarks (MMLU/GSK-8K/HumanEval) without being detected by OpenAI's decontamination method!


<img src="/images/blog/decontaminator/llama-Frank.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

What's the trick behind it? Well, rephrasing the test set is all you need! We simply paraphrase a test sample or translate it into a different language. It turns out a 13B LLM is smart enough to  "generalize" beyond such variations and reaches drastically high benchmark performance. So, did we just make a big breakthrough? Apparently, there is something wrong with our understanding of contamination.

In this blog post, we point out why contamination is still poorly understood and how existing decontamination measures fail to capture such nuances. To address such risks, we propose a stronger [LLM-based decontaminator](https://github.com/lm-sys/llm-decontaminator) and apply it to real-world training datasets, revealing significant test overlap with widely used benchmarks. 


## **Existing detection methods are not enough**

Contamination occurs when test set information is leaked in the training set, resulting in an overly optimistic estimate of the model’s performance.
Despite being recognized as a crucial issue, understanding and detecting contamination remains an open and challenging problem.

The most commonly used approaches are n-gram overlap and embedding similarity search.
N-gram overlap relies on string matching to detect contamination, widely used by leading developments such as GPT-4, PaLM, and Llama.
Embedding similarity search uses the embeddings of pre-trained models (e.g., BERT) to find similar and potentially contaminated examples.

However, we show that simple variations of the test data (e.g., paraphrasing, translation) can easily bypass existing simple detection methods. 
We refer to such variations of test cases as _Rephrased Samples_.

Below is a failure case of existing contamination detection methods (n-gram overlap, embedding similarity) on MMLU benchmark. The embedding similarity approach struggles to distinguish the rephrased question from other questions in the same subject (high school US history).
After rephrasing MMLU test cases, a Llama-2-13B trained on a rephrased test set can reach 85.9 accuracy on MMLU while being undetectable by n-gram overlap.


<img src="/images/blog/decontaminator/overview.png" style="display:block; margin:auto; max-width:100%; height:auto;">


There are some subtle differences in rephrasing techniques because benchmark contamination takes on different forms.
For text-based benchmarks, we rephrase test cases without altering their meanings, such as by rearranging word order or substituting with synonymous terms. For code-based benchmarks, we vary coding styles, naming conventions, and algorithms.

In addition to modifying the word order, translating samples can also help models to achieve dramatically high scores. 
Prompts with identical meanings from different languages yield varied embeddings in most language models, so translating samples can evade standard embedding similarity search.

Trained on rephrased samples of MMLU, HumanEval and GSM-8k, Llama-2 13B achieved drastically high performance, on par with GPT-4's performance.
Both n-gram overlap and embedding similarity search fail to detect them.



## **Stronger Detection Method: LLM Decontaminator**

To catch Llama-Frank, we really need a detector-Carl.
We propose a new contamination detection method "LLM decontaminator" to address the risk of possible contamination.
Our method can accurately remove a dataset's rephrased samples relative to a benchmark.

This LLM decontaminator involves two steps:

  1. For each test case, LLM decontaminator identifies the top-k training items with the highest similarity using the embedding similarity search.
  2. From these items, LLM decontaminator generates k potential rephrased pairs. Each pair is evaluated for rephrasing using an advanced LLM, such as GPT-4.


### **Evaluating Different Detection Methods**

To compare the accuracy of different detection methods, we construct 200 prompt pairs using both the original and rephrased test sets. These comprised 100 random pairs and 100 rephrased pairs.
The f1 score on these pairs provides insight into the detection methods' ability to detect contamination, with higher values indicating more precise detection.
As shown in the following table, except for the LLM decontaminator, all other detection methods introduce some false positives. Both rephrased and translated samples successfully evade the n-gram overlap detection. With multi-qa BERT, the embedding similarity search proves completely ineffective against translated samples. 
Notably, the LLM decontaminator showcases superior performance, identifying rephrased samples with high reliability and precision with the highest minimum F1 score as well as the highest average F1 score.

<img src="/images/blog/decontaminator/MMLU-f1score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

## **Serious Contamination in Real-World Dataset**

We apply the LLM decontaminator to widely used real-world datasets and identify a substantial amount of rephrased samples. 
The table below displays the contamination percentage of different benchmarks in each training dataset.

<img src="/images/blog/decontaminator/real-world-rephrase.png" style="display:block; margin:auto; max-width:100%; height:auto;">

Here we show some detected samples.

[CodeAlpaca](https://github.com/sahil280114/codealpaca) contains 20K instruction-following data used for fine-tuning the CodeAlpaca model. 
It is used to train a number of well-known models, including [Tulu](https://huggingface.co/TheBloke/tulu-30B-fp16).
A rephrased example in CodeAlpaca is shown below.

<img src="/images/blog/decontaminator/codealpaca-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[MATH](https://github.com/hendrycks/math) is a widely recognized math training dataset that spans various mathematical domains, including algebra, geometry, and number theory. Below is a self-contamination case in MATH.

<img src="/images/blog/decontaminator/MATH-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

[StarCoder-Data](https://huggingface.co/datasets/bigcode/starcoderdata) is used for training StarCoder and StarCoderBase, and it contains 783GB of code in 86 programming languages. In the StarCoder [paper](https://arxiv.org/pdf/2305.06161.pdf), the code training data was decontaminated by removing files that contained docstrings or solutions from HumanEval. However, there are still some samples detected by LLM decontaminator.

<img src="/images/blog/decontaminator/starcoder-rephrase.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

## **Use LLM Decontaminator to Scan Your Data**

The issue of unintentional contamination is becoming increasingly severe because more and more datasets are generated by LLMs. 
Since LLMs always generate data similar to their training data, these generated data might contain rephrased samples. For instance, CodeAlpaca uses GPT to generate training data, which include rephrased samples of HumanEval. 
Here we show how to remove rephrased samples from training data using the LLM decontaminator. The following example can be found [here](https://github.com/lm-sys/llm-decontaminator#detect).

[Pre-process](https://github.com/lm-sys/llm-decontaminator#pre-process) training data and test data.
The LLM decontaminator accepts the dataset in jsonl format, with each line corresponding to a `{"text": data}` entry.

Run [End2End](https://github.com/lm-sys/llm-decontaminator#end2end) detection.
The following command builds a top-k similar database based on sentence bert and uses GPT-4 to check one by one if they are rephrased samples. You can select your embedding model and detection model by modifying the parameters.

<img src="/images/blog/decontaminator/run-e2e.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>



## **Acknowledgment**

We would like to express our gratitude to Ying Sheng for the early discussion on rephrased samples.
We also extend our thanks to Dacheng Li, Erran Li, Hao Liu, Jacob Steinhardt, Hao Zhang, and Siyuan Zhuang for providing insightful feedback.
This project is partly supported by gifts from Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, MBZUAI, Samsung SDS, Uber, and VMware. Lianmin Zheng is supported by a Meta Ph.D. Fellowship.