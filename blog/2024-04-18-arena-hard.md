---
title: "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline"
author: "Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu"
date: "April 18, 2024"
previewImg: /images/blog/arena_hard/arena_hard.png
---

<style>
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:#ccc;border-style:solid;border-width:1px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-head{background-color:#c0c0c0;border-color:#ccc;text-align:left;vertical-align:top;}
.tg .tg-body{text-align:left;vertical-align:top;}
</style>

<style>
  iframe {
    display: block;
    width: 100%;
    height: 950px;
    border: none;
    overflow: hidden;
  }
</style>

## Introduction

Building an affordable and reliable benchmark for LLM chatbots has become a critical challenge. A high-quality benchmark should 1) robustly separate model capability, 2) reflect human preference in real-world use cases, and 3) frequently update to avoid over-fitting or test set leakage.

Traditional benchmarks are often static or close-ended (e.g., MMLU multi-choice QA), which do not satisfy the above requirements. On the other hand, models are evolving faster than ever, underscoring the need to build benchmarks with high separability.

We introduce Arena-Hard – a data pipeline to build high-quality benchmarks from live data in [Chatbot Arena](https://arxiv.org/abs/2403.04132), which is a crowd-sourced platform for LLM evals. To measure its quality, we propose two key metrics:
1. Agreement to Human preference: whether the benchmark score has high agreement to human preference.
2. Separability: whether the benchmark can confidently separate models.

We compare our new benchmark, Arena-hard-v0.1, to a current leading chat LLM benchmark, MT-bench. In Figure 1, we show Arena-hard-v0.1 offers significantly stronger separability against MT-bench with tighter confidence intervals. It also has a high agreement (93%) with the human preference ranking by Chatbot Arena (english-only). We expect to see this benchmark useful for model developers to differentiate their model checkpoints.

Figure 1: Comparison between MT-bench and Arena-Hard-v0.1. The latter offers significantly better separability between models and tighter confidence intervals. Note: We do not include GPT-4-Turbo in the plot due to potential self-bias. GPT-4-0314 has no variance in Arena-hard-v0.1 because it’s used as the anchor model.

TODO: insert figure
Links:
- Evaluate your model on Arena-Hard-v0.1: https://github.com/lm-sys/arena-hard
- Browse Arena-Hard-v0.1 prompts: https://huggingface.co/spaces/lmsys/arena-hard-browser
- Full leaderboard at the Result section
- https://github.com/lm-sys/arena-hard/tree/main/notebook (bootstrapping experiments)

We explain more technical details in the following sections.

## Key Objectives of LLM benchmarks

We outline a few key properties that an LLM chatbot benchmark should possess to provide a meaningful measurement of capabilities between models:
1. Agreement to human preference: It should correlate with human preference in real-world use cases
2. Separability: It should provide confidence interval on benchmark score and separate models with high confidence
3. Freshness: It should use new, unseen prompts to avoid potential test leakage


We define **agreement** of Benchmark A with respect to a reference Benchmark B by the below formulation:
For a given model pair (which B can separate with confidence),
- If A can confidently separate the 2 given models
    - +1.0 if the rank order agrees with B.
    - -1.0 if the rank order disagrees with B.
- +0.0 if A cannot separate the 2 given models with confidence


We define **separability** by whether a benchmark can separate given model pairs with derived confidence intervals (via bootstrapping). We quantify this metric by the percentage of model pairs which have non-overlapping confidence intervals of the benchmark scores.


We use a set of top-20 models* on [Chatbot Arena](https://chat.lmsys.org/?leaderboard) that are presented on [AlpacaEval leaderboard](https://tatsu-lab.github.io/alpaca_eval/) to calculate separability and agreement per benchmark. We consider the human preference ranking by Chatbot Arena (English only) as the reference to calculate agreement.

In Table X, Arena-hard-v0.1 shows the highest separability (91.6%) against widely adopted LLM benchmarks and offers high agreement (85.9%) to Chatbot Arena. It is also cheap and fast to run ($25).

Interestingly, we find Spearman Correlation, a popular metric for measuring correlations between rankings, may be an unreliable metric for ranking correlation as it does not consider variance of the rankings, and therefore fails to adequately punish essential ranking granularities of the top models we care about most. For example, when considering 95% CI, MT-bench’s agreement to Chatbot Arena drops from 89.3% to 39.0%.

You can find full statistics in the result section. 
<p style="width:100%; color:gray; text-align: center;">Table 1. Separability and agreement per benchmark.</p>

<table class="tg" style="display: flex;justify-content: center;">
  <colgroup>
    <col style="width: auto;">
    <col style="width: auto;">
    <col style="width: auto;">
    <col style="width: 80px;"> <!-- narrower -->
    <col style="width: 120px;"> <!-- wider -->
  </colgroup>
  <tbody>
    <tr>
      <th class="tg-head"><span style="font-weight:bold;"></span></th>
      <th class="tg-head"><span style="font-weight:bold;">Chatbot Arena<br>(English-only)</span></th>
      <th class="tg-head"><span style="font-weight:bold;">MT-bench</span></th>
      <th class="tg-head"><span style="font-weight:bold;">AlpacaEval 2.0 LC<br>(Length Controlled)</span></th>
      <th class="tg-head"><span style="font-weight:bold;">Arena-Hard-v0.1</span></th>
    </tr>
    <tr>
      <td class="tg-body">Avg #prompts per model eval</td>
      <td class="tg-body">~20,000</td>
      <td class="tg-body">160</td>
      <td class="tg-body">800</td>
      <td class="tg-body">1,000</td>
    </tr>
    <tr>
      <td class="tg-body">Agreement to Chatbot Arena with 95% CI</td>
      <td class="tg-body">N/A</td>
      <td class="tg-body">27.1%</td>
      <td class="tg-body">80.7%</td>
      <td class="tg-body">89.2%</td>
    </tr>
    <tr>
      <td class="tg-body">Separability with 95% CI</td>
      <td class="tg-body">86.3%</td>
      <td class="tg-body">23.7%</td>
      <td class="tg-body">83.7%</td>
      <td class="tg-body">87.9%</td>
    </tr>
    <tr>
      <td class="tg-body">Spearman Correlation</td>
      <td class="tg-body">N/A</td>
      <td class="tg-body">91.0%</td>
      <td class="tg-body">90.4%</td>
      <td class="tg-body">94.3%</td>
    </tr>
    <tr>
      <td class="tg-body">Real-world</td>
      <td class="tg-body">Yes</td>
      <td class="tg-body">Mixed</td>
      <td class="tg-body">Mixed</td>
      <td class="tg-body">Yes</td>
    </tr>
    <tr>
      <td class="tg-body">Freshness</td>
      <td class="tg-body">Live</td>
      <td class="tg-body">Static</td>
      <td class="tg-body">Static</td>
      <td class="tg-body">Frequent Updates</td>
    </tr>
    <tr>
      <td class="tg-body">Eval cost per model</td>
      <td class="tg-body">Very High</td>
      <td class="tg-body">$10</td>
      <td class="tg-body">$10</td>
      <td class="tg-body">$25</td>
    </tr>
    <tr>
      <td class="tg-body">Judge</td>
      <td class="tg-body">Human</td>
      <td class="tg-body">LLM</td>
      <td class="tg-body">LLM</td>
      <td class="tg-body">LLM</td>
    </tr>
    <tr>
      <td class="tg-body" colspan="5">*20 top models from Chatbot Arena that are also presented on Alpaca Eval: gpt-4-turbo-2024-04-09, claude-3-opus-20240229, gpt-4-0125-preview, claude-3-sonnet-20240229, gpt-4-0314, gpt-4-0613, mistral-large-2402, qwen1.5-72b-chat, mistral-medium, claude-2.0, gpt-3.5-turbo-0613, claude-2.1, gemini-pro, mixtral-8x7b-instruct-v0.1, gpt-3.5-turbo-0314, yi-34b-chat, tulu-2-dpo-70b, dbrx-instruct-preview, vicuna-33b, starling-lm-7b-alpha</td>
    </tr>
</tbody>
</table>

Next, we elaborate how to build the prompt selection pipeline to ensure data quality.

## Arena-Hard Pipeline

We build a pipeline that automatically extracts quality prompts from a dataset of 200,000 user queries collected via Chatbot Arena. This process involves ensuring:
- Diversity: Prompt set should cover a wide range of real-world topics
- Prompt quality: Each prompt should possess high quality to benchmark LLMs. we define several key criteria below (see Table X)

<img src="/images/blog/arena_hard/method.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Arena-Hard Pipeline</p>

To ensure prompt diversity, we adopt a topic modeling pipeline in [BERTopic](https://github.com/MaartenGr/BERTopic) by first converting each prompt with OpenAI’s embedding (text-embedding-3-small), reducing dimension with UMAP, and using a hierarchical-based clustering algorithm (HDBSCAN) to identify clusters which are then summarized using GPT-4-turbo. This helps us identify over 4000 topics covering a wide range of domains. However, topic clusters come with varying quality and separability in benchmarking LLMs. We then develop a calibrated system prompt for LLMs to help us select high quality user queries by seven key criteria (e.g., specificity, domain knowledge, problem-solving, etc).

<table style="width:100%; border-collapse: collapse; border: 1px solid black;">
  <tr style="background-color: black; color: white;">
    <th style="border: 1px solid black; padding: 10px; text-align: left;">7 Key Criteria</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>1. Specificity:</strong> Does the prompt ask for a specific output?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>2. Domain Knowledge:</strong> Does the prompt cover one or more specific domains?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>3. Complexity:</strong> Does the prompt have multiple levels of reasoning, components, or variables?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>4. Problem-Solving:</strong> Does the prompt directly involve the AI to demonstrate active problem-solving skills?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>5. Creativity:</strong> Does the prompt involve a level of creativity in approaching the problem?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>6. Technical Accuracy:</strong> Does the prompt require technical accuracy in the response?</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; text-align: left;"><strong>7. Real-world Application:</strong> Does the prompt relate to real-world applications?</td>
  </tr>
</table>


An LLM Judge (GPT-3.5-Turbo, GPT-4-Turbo) annotates each prompt from 0 to 7 to indicate how many criteria are met. We then score each cluster by the average score of its prompts. Below, we show examples of topic clusters ranging from low to high mean scores. We can observe clusters with higher scores often correlate to challenging topics or tasks for LLMs like game development or mathematical proofs. On the other hand, clusters with lower scores point to trivial or ambiguous questions like Design Styles and Influences.

<img src="/images/blog/arena_hard/cluster_distribution.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Prompt clusters and their scores</p>

To see whether the prompt score correlates with separability, we sample 50 prompts per score and compare the responses from GPT-4 and Llama-70b, with GPT-4-Turbo as judge. We observe a strong correlation between high potential score and the win-rate of GPT-4 over Llama-70b. A similar trend is also observed in other model pairs such as Claude Sonnet vs Haiku and Mistral-large vs Mixtral.

Figure Z: Win-rate between model pairs becomes more separable as the prompt satisfies more of the 7 Key Criteria in Table X

TODO: insert score to winrate figure


## Results

### Arena-Hard-v0.1

Using the above pipeline, we identify 250 high-quality topic clusters with mean score >=6 out of 7. We then randomly sample 2 prompts per cluster to construct 500 high-quality benchmark prompts, Arena-Hard-v0.1. This benchmark set contains mostly well-defined, technical problem-solving queries as required in the above key criteria. You can browse all the prompts at this [link](https://huggingface.co/spaces/lmsys/arena-hard-browser).

However, evaluating models on challenging queries such as Arena-Hard-v0.1 is a non-trivial task. Most queries involve deep domain knowledge and problem solving skills, requiring expert-level judgment to evaluate the answer quality. Unfortunately, this is prohibitively expensive and time consuming. Following [1](https://arxiv.org/abs/2306.05685), [2](https://arxiv.org/abs/2305.14387), we employ LLM as a judge framework to approximate human preference.

We consider the pairwise comparison setup against a strong baseline model (GPT-4-0314), and ask a strong judge model (e.g., GPT-4-Turbo or Claude-3-Opus) to categorize the preference into five labels: A >> B, A > B, A~=B, .. B>>A. This way, a model will be penalized more in big losses than small losses, which we find to be effective in separating models. We also employ CoT to prompt the LLM judge to generate answers first before giving judgments. Full judge prompt can be found [here](https://github.com/lm-sys/arena-hard/blob/main/config/judge_config.yaml).

To avoid potential position bias, we adopt a two-game setup – per query we swap the models on the first & second position. This results in 500x2=1000 judgments per model evaluation. Following Chatbot Arena, we adopt the Bradley-Terry model to produce model’s the final model scores. By bootstrapping the comparisons from all models, we find it to be statistically stable compared to only considering win-rate against the baseline model.

### Full Leaderboard with GPT-4-Turbo as judge

We use gpt-4-1106-preview as the judge model to generate judgment for the model response against baseline. We take all the comparisons and compute each model’s Bradley-Terry coefficient. We then transform it to win-rate against the baseline as the final score. The 95% confidence interval is computed via 100 rounds of bootstrapping.

<p style="color:gray; text-align: center;">Table 1. Model Performance Comparison</p>

<table class="tg" style="width:100%; display: flex; justify-content: center;">
<tbody>
  <tr>
    <td class="tg-head"><span style="font-weight:bold;">Model Name</span></td>
    <td class="tg-head"><span style="font-weight:bold;">Score</span></td>
    <td class="tg-head"><span style="font-weight:bold;">95% CI</span></td>
    <td class="tg-head"><span style="font-weight:bold;">Average #Tokens</span></td>
  </tr>
  <tr>
    <td class="tg-body; text-align: left;">gpt-4-turbo-2024-04-09</td>
    <td class="tg-body">82.6</td>
    <td class="tg-body">(-1.9, 2.0)</td>
    <td class="tg-body">662</td>
  </tr>
  <tr>
    <td class="tg-body; text-align: left;">gpt-4-0125-preview</td>
    <td class="tg-body">78.0</td>
    <td class="tg-body">(-1.8, 2.2)</td>
    <td class="tg-body">619</td>
  </tr>
  <tr>
    <td class="tg-body; text-align: left;">claude-3-opus-20240229</td>
    <td class="tg-body">60.4</td>
    <td class="tg-body">(-3.3, 2.3)</td>
    <td class="tg-body">541</td>
  </tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-4-0314</td>
  <td class="tg-body">50.0</td>
  <td class="tg-body">(0.0, 0.0)</td>
  <td class="tg-body">423</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">claude-3-sonnet-20240229</td>
  <td class="tg-body">46.8</td>
  <td class="tg-body">(-2.1, 2.5)</td>
  <td class="tg-body">552</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">claude-3-haiku-20240307</td>
  <td class="tg-body">41.5</td>
  <td class="tg-body">(-1.9, 2.0)</td>
  <td class="tg-body">505</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-4-0613</td>
  <td class="tg-body">37.9</td>
  <td class="tg-body">(-2.8, 2.5)</td>
  <td class="tg-body">354</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">mistral-large-2402</td>
  <td class="tg-body">37.7</td>
  <td class="tg-body">(-1.9, 2.7)</td>
  <td class="tg-body">400</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">mixtral-8x22b-instruct-v0.1</td>
  <td class="tg-body">36.4</td>
  <td class="tg-body">(-1.5, 2.7)</td>
  <td class="tg-body">430</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Qwen1.5-72B-Chat</td>
  <td class="tg-body">36.1</td>
  <td class="tg-body">(-2.6, 2.2)</td>
  <td class="tg-body">474</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">command-r-plus</td>
  <td class="tg-body">33.1</td>
  <td class="tg-body">(-2.1, 2.2)</td>
  <td class="tg-body">541</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">mistral-medium</td>
  <td class="tg-body">31.9</td>
  <td class="tg-body">(-2.2, 2.3)</td>
  <td class="tg-body">485</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">mistral-next</td>
  <td class="tg-body">27.4</td>
  <td class="tg-body">(-2.1, 1.7)</td>
  <td class="tg-body">297</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-3.5-turbo-0613</td>
  <td class="tg-body">24.8</td>
  <td class="tg-body">(-1.7, 2.1)</td>
  <td class="tg-body">401</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">claude-2.0</td>
  <td class="tg-body">24.0</td>
  <td class="tg-body">(-2.5, 2.5)</td>
  <td class="tg-body">295</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">dbrx-instruct</td>
  <td class="tg-body">23.9</td>
  <td class="tg-body">(-1.6, 1.6)</td>
  <td class="tg-body">415</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Mixtral-8x7B-Instruct-v0.1</td>
  <td class="tg-body">23.4</td>
  <td class="tg-body">(-2.2, 2.2)</td>
  <td class="tg-body">457</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-3.5-turbo-0125</td>
  <td class="tg-body">23.3</td>
  <td class="tg-body">(-2.2, 2.3)</td>
  <td class="tg-body">329</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Yi-34B-Chat</td>
  <td class="tg-body">23.1</td>
  <td class="tg-body">(-2.0, 1.9)</td>
  <td class="tg-body">611</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Starling-LM-7B-beta</td>
  <td class="tg-body">23.0</td>
  <td class="tg-body">(-1.8, 2.2)</td>
  <td class="tg-body">530</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">claude-2.1</td>
  <td class="tg-body">22.8</td>
  <td class="tg-body">(-1.8, 1.9)</td>
  <td class="tg-body">290</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Snorkel-Mistral-PairRM-DPO</td>
  <td class="tg-body">20.7</td>
  <td class="tg-body">(-1.9, 2.0)</td>
  <td class="tg-body">564</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-3.5-turbo-1106</td>
  <td class="tg-body">18.9</td>
  <td class="tg-body">(-1.7, 1.8)</td>
  <td class="tg-body">285</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gpt-3.5-turbo-0301</td>
  <td class="tg-body">18.1</td>
  <td class="tg-body">(-1.9, 1.7)</td>
  <td class="tg-body">334</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gemini-1.0-pro</td>
  <td class="tg-body">17.8</td>
  <td class="tg-body">(-1.7, 1.7)</td>
  <td class="tg-body">322</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">command-r</td>
  <td class="tg-body">17.0</td>
  <td class="tg-body">(-2.0, 1.6)</td>
  <td class="tg-body">432</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">tulu-2-dpo-70b</td>
  <td class="tg-body">15.0</td>
  <td class="tg-body">(-1.6, 1.4)</td>
  <td class="tg-body">550</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Starling-LM-7B-alpha</td>
  <td class="tg-body">12.8</td>
  <td class="tg-body">(-1.6, 1.5)</td>
  <td class="tg-body">483</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">mistral-7b-instruct-v0.2</td>
  <td class="tg-body">12.6</td>
  <td class="tg-body">(-1.4, 1.2)</td>
  <td class="tg-body">541</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Llama-2-70b-chat-hf</td>
  <td class="tg-body">11.6</td>
  <td class="tg-body">(-1.6, 1.4)</td>
  <td class="tg-body">595</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">vicuna-33b-v1.3</td>
  <td class="tg-body">8.6</td>
  <td class="tg-body">(-1.2, 1.1)</td>
  <td class="tg-body">451</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gemma-7b-it</td>
  <td class="tg-body">7.5</td>
  <td class="tg-body">(-1.1, 1.1)</td>
  <td class="tg-body">378</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">Llama-2-7b-chat-hf</td>
  <td class="tg-body">4.6</td>
  <td class="tg-body">(-1.0, 0.7)</td>
  <td class="tg-body">561</td>
</tr>
<tr>
  <td class="tg-body; text-align: left;">gemma-2b-it</td>
  <td class="tg-body">3.0</td>
  <td class="tg-body">(-0.6, 0.7)</td>
  <td class="tg-body">369</td>
</tr>

  <tr>
      <td class="tg-body; text-align: left" colspan="4">Baseline: gpt-4-0314</td>
    </tr>
</tbody>
</table>

```
Baseline: gpt-4-0314
========================
gpt-4-turbo-2024-04-09         | score: 82.6  | 95% CI: (-1.9, 2.0)  | average #tokens: 662
gpt-4-0125-preview             | score: 78.0  | 95% CI: (-1.8, 2.2)  | average #tokens: 619
claude-3-opus-20240229         | score: 60.4  | 95% CI: (-3.3, 2.3)  | average #tokens: 541
gemini-1.5-pro                 | score: 53.4  | 95% CI: (-2.2, 2.6)  | average #tokens: 478
gpt-4-0314                     | score: 50.0  | 95% CI:  (0.0, 0.0)  | average #tokens: 423
claude-3-sonnet-20240229       | score: 46.8  | 95% CI: (-2.1, 2.5)  | average #tokens: 552
claude-3-haiku-20240307        | score: 41.5  | 95% CI: (-1.9, 2.0)  | average #tokens: 505
gpt-4-0613                     | score: 37.9  | 95% CI: (-2.8, 2.5)  | average #tokens: 354
mistral-large-2402             | score: 37.7  | 95% CI: (-1.9, 2.7)  | average #tokens: 400
mixtral-8x22b-instruct-v0.1    | score: 36.4  | 95% CI: (-1.5, 2.7)  | average #tokens: 430
Qwen1.5-72B-Chat               | score: 36.1  | 95% CI: (-2.6, 2.2)  | average #tokens: 474
command-r-plus                 | score: 33.1  | 95% CI: (-2.1, 2.2)  | average #tokens: 541
mistral-medium                 | score: 31.9  | 95% CI: (-2.2, 2.3)  | average #tokens: 485
mistral-next                   | score: 27.4  | 95% CI: (-2.1, 1.7)  | average #tokens: 297
gpt-3.5-turbo-0613             | score: 24.8  | 95% CI: (-1.7, 2.1)  | average #tokens: 401
claude-2.0                     | score: 24.0  | 95% CI: (-2.5, 2.5)  | average #tokens: 295
dbrx-instruct                  | score: 23.9  | 95% CI: (-1.6, 1.6)  | average #tokens: 415
Mixtral-8x7B-Instruct-v0.1     | score: 23.4  | 95% CI: (-2.2, 2.2)  | average #tokens: 457
gpt-3.5-turbo-0125             | score: 23.3  | 95% CI: (-2.2, 2.3)  | average #tokens: 329
Yi-34B-Chat                    | score: 23.1  | 95% CI: (-2.0, 1.9)  | average #tokens: 611
Starling-LM-7B-beta            | score: 23.0  | 95% CI: (-1.8, 2.2)  | average #tokens: 530
claude-2.1                     | score: 22.8  | 95% CI: (-1.8, 1.9)  | average #tokens: 290
Snorkel-Mistral-PairRM-DPO     | score: 20.7  | 95% CI: (-1.9, 2.0)  | average #tokens: 564
gpt-3.5-turbo-1106             | score: 18.9  | 95% CI: (-1.7, 1.8)  | average #tokens: 285
gpt-3.5-turbo-0301             | score: 18.1  | 95% CI: (-1.9, 1.7)  | average #tokens: 334
gemini-1.0-pro                 | score: 17.8  | 95% CI: (-1.7, 1.7)  | average #tokens: 322
command-r                      | score: 17.0  | 95% CI: (-2.0, 1.6)  | average #tokens: 432
tulu-2-dpo-70b                 | score: 15.0  | 95% CI: (-1.6, 1.4)  | average #tokens: 550
Starling-LM-7B-alpha           | score: 12.8  | 95% CI: (-1.6, 1.5)  | average #tokens: 483
mistral-7b-instruct-v0.2       | score: 12.6  | 95% CI: (-1.4, 1.2)  | average #tokens: 541
Llama-2-70b-chat-hf            | score: 11.6  | 95% CI: (-1.6, 1.4)  | average #tokens: 595
vicuna-33b-v1.3                | score:  8.6  | 95% CI: (-1.2, 1.1)  | average #tokens: 451
gemma-7b-it                    | score:  7.5  | 95% CI: (-1.1, 1.1)  | average #tokens: 378
Llama-2-7b-chat-hf             | score:  4.6  | 95% CI: (-1.0, 0.7)  | average #tokens: 561
gemma-2b-it                    | score:  3.0  | 95% CI: (-0.6, 0.7)  | average #tokens: 369
*Note: GPT-4-Turbo’s high score can be due to the GPT-4 judge favoring GPT-4 outputs. 
```

TODO: add model ranking table


### GPT-4-Turbo or Claude as Judge?

We also compare two strongest LLMs: GPT-4-1106-Preview and Claude-3 Opus as the judge mode in Table X. When GPT-4 Judge is used, we observe higher separability across models (ranging from 23.0 to 78.0). When Claude Judge is used, we find the Claude family of models scores in general go up, despite it still favoring gpt-4-0125-preview over itself. Surprisingly, it favors several open models (Mixtral, Yi, Starling) or even gpt-3.5-turbo over gpt-4-0613.


TODO: add table

We further compare GPT-4 and Claude Judges using our proposed metrics of separability and agreement in Table X, and find that the GPT-4-turbo Judge is significantly better across all metrics. 

Table X: Statistical comparisons between LLM Judges and Human 

TODO: add table

*Brier Score (lower is better), a statistical scoring function for measuring the accuracy of probabilistic accuracy. (see section View Benchmarking as a Forecasting Problem for more information)


We manually compared different judgment examples between GPT-4-Turbo and Claude as a judge. We found that when the two judges disagreed, it could usually be broken down into two main categories:
1. Conservative scoring
2. Differing perspectives on the user's prompt

We find that Claude-3-Opus is much less likely to give harsh scores – it is particularly hesitant to proclaim one response as "significantly better" than another. In contrast, GPT-4-Turbo will identify errors in a model's response that led to an incorrect answer and penalize the model with a significantly lower score. On the other hand, Claude-3-Opus sometimes overlooks smaller errors. Even when Claude-3-Opus does identify these errors, it tends to treat them as minor issues and shows leniency during scoring. This effect is particularly present in coding and math problems, where small mistakes are more likely to completely derail the final answer; these scorings are still given leniency from Claude-3-Opus but not GPT-4-Turbo. See the appendix below for specific examples of differing judgments, many of which exhibit this phenomenon.

TODO: insert fig

There is also a small subset of prompts in which Claude-3-Opus and GPT-4-Turbo judge with fundamentally different perspectives. For example, given a coding question, Claude-3-Opus may choose the response that provides the most educational value to the user, offering a simplistic structure without relying on external libraries. GPT-4-Turbo, however, may prioritize the response that provides the most practical answer, regardless of its educational value to the user.  While both interpretations are valid judging criteria, we find GPT-4-Turbo’s perspective may be more correlated with the average user.

Despite the observed differences between Claude-3-Opus and GPT-4-Turbo judgment styles, we find the judges have an overall soft agreement rate of 80%. Two judgments “soft agree” if they are at most distance one apart, or in other words they do not contradict.

## Limitations

### Verbosity: does the LLM Judge prefer longer responses?

LLM as judges are known to suffer from verbosity bias (cite ..). Below we plot the avg token length and score per model for both MT-Bench and Arena-Hard-v0.1. We find there seems to be no clear correlation.

TODO: insert fig

To further examine potential verbosity bias, we conduct an ablation on three different system prompts (original, chatty, detailed) with GPT-3.5-Turbo. We observe that both GPT-4-Turbo and Claude-3-Opus judges may be affected by longer outputs, while Claude being significantly more impacted with a “more detailed” system prompt as GPT-3.5-Turbo reaches a win-rate of over 40% against GPT-4-0314. 

Interestingly, the “chatty” system prompt doesn’t affect much on the win-rate by both judges, despite the longer average #tokens. This suggests output length is not the only factor. It is possible that more detailed answers are also more helpful and thus preferred by LLM judges.

TODO: insert table
```
===== GPT-4-Turbo =====
gpt-3.5-turbo-0125-detailed    | win-rate: 29.86 | average #tokens: 421
gpt-3.5-turbo-0125-verbose     | win-rate: 23.89 | average #tokens: 361
gpt-3.5-turbo-0125-chatty      | win-rate: 23.57 | average #tokens: 375
gpt-3.5-turbo-0125             | win-rate: 23.2  | average #tokens: 328

======= Claude-3 =======
gpt-3.5-turbo-0125-detailed    | win-rate: 40.78 | average #tokens: 421
gpt-3.5-turbo-0125-chatty      | win-rate: 28.49 | average #tokens: 375
gpt-3.5-turbo-0125             | win-rate: 27.97 | average #tokens: 328

System Prompt:
detailed: “You are a helpful assistant who thoroughly explains things with as much detail as possible.”
chatty: “You are a helpful assistant who is chatty.”
```

### Variance in GPT-4 judgments

We find that even with temperature=0, GPT-4-Turbo may still generate slightly different judgments. Here we repeat the judgments for gpt-3.5-turbo-0125 three times and report its variance. Due to limited budget, we can only evaluate all the models once. But we recommend model developers to evaluate models multiple times and take average.

```
gpt-3.5-turbo-0125-1      | win-rate: 23.05 | average #tokens: 328
gpt-3.5-turbo-0125-2      | win-rate: 22.93 | average #tokens: 328
gpt-3.5-turbo-0125-3      | win-rate: 22.75 | average #tokens: 328
```

### Potential self-bias & prompt selection bias

We also observe potential self-bias in LLM judges (e.g., Claude Judge prefers Claude answers).
In addition, the prompt selection process could be biased by the LLMs. The benchmark also does not evaluate multi-turn interactions.


## Viewing Benchmarking as a Forecasting Problem

In this section we attempt to combine both confidence and correlation into one standardized metric for benchmarking.

TODO: add table

Model developers generally use benchmarks for model selection, not ground truth certification of performance.  Benchmarks serve as a cheap and lightweight proxy for more expensive and complex evaluations like ground truth Bradley Terry Coefficients derived from human preference. Thus, we expect benchmarks to tell us, as model developers, some confidence bound on what a model’s real world performance will be. In this sense, a benchmark serves as a forecast for true long-run performance.

Forecasting is a delicate balance between confidence and uncertainty. Therefore, a good benchmark should show confidence when separating clearly unequal models, but should demonstrate uncertainty when ranking differences between legitimately similar models. One might argue we only need to look at how confident a given benchmark is at separating model pairs. A good benchmark is not necessarily always confident at separating models– you don’t want your benchmark to be confidently incorrect. For example, given a pair of models A and B and benchmark 1 and 2. Let’s assume ground truth is model A is better than model B. We bootstrap both benchmark 1 and 2 and retrieve their confidence intervals for both model’s performances. Benchmark 1 confidently predicts model B is better than A while Benchmark 2 predicts model B is better than A with low confidence. In this case, we should say Benchmark 2 is actually better than Benchmark 1 at predicting this pair of models. This is to say, high confidence should be rewarded only when the answer is correct, and low confidence is better when incorrect.

In this problem context, we introduce the prediction criteria as simply the binary indicator 1{model_a < model_b} for some model pairing.  The forecast gives a probability that this indicator is true, p(a < b).  A higher probability forecast indicates greater confidence that 1{model_a < model_b} will be true.  We can generate these probability predictions using bootstrapped score mean and variance, which in turn define a normal distribution.  We then resolve the ground truth label for 1{model_a < model_b} using Chatbot Arena scores.

A well-defined fair-in-expectation loss for forecasting is the Brier score loss. Brier score rewards confidence when forecasts are correct while punishing confident errors. We can calculate the loss over a benchmark prediction of 1{model_a < model_b} for each model pair with respect to the Chatbot Area ground truth scores to quantify a benchmark’s forecasting performance. Here we assume Chatbot Arena as “ground truth” as both Alpaca 2.0 LC and Arena Hard are advertised as an inexpensive alternative to Chatbot Arena as an evaluation pipeline. We will conduct future study on correlation comparison when we draw Arena Bradley Terry from similar distributions as given benchmark.

We find that Arena Hard averages much lower forecasting loss, demonstrating that it is both accurate in score, and accurate in confidence level.

TODO: add figs

Above is the predicted model predicted probability against the bootstrapped arena “ground truth” probability (jittered to show clusters).  While both Alpaca eval and Arena Hard have large clusters around (0,0) and (1,1) signifying good forecasting, Arena Hard has lighter clusters on (0,1) and (1,0), if any, revealing less overconfidence. MT Bench has heavy tails along the top and bottom, revealing underconfidence. However, none of these benchmarks show an “ideal” y=x curve (with dense ends) expected with a perfectly calibrated forecast, signifying room for future research.

TODO: add big table

*Brier Score (lower is better), a statistical scoring function for measuring the accuracy of probabilistic accuracy. (see section View Benchmarking as a Forecasting Problem for more information)

## Future
We hope to study deeper into the above limitations and biases in the later technical report. We are also working on diving deeper into the statistics for more studies on how to measure the quality of benchmarks. Lastly, we also hope to upgrade Arena-Hard frequently. So expect frequent new benchmarks! 


## Acknowledgment
We thank Matei Zaharia, Yann Dubois, Anastasios Angelopoulos,  Joey Gonzalez, Lianmin Zheng, Lewis Tunstall, Nathan Lambert, Xuechen Li, Naman Jain, Ying Sheng, Maarten Grootendorst for their valuable feedback. We thank Microsoft [AFMR](https://www.microsoft.com/en-us/research/collaboration/accelerating-foundation-models-research/) for Azure OpenAI credits support. We also thank Together.ai & Anyscale for open model endpoint support.