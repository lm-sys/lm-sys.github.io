---
title: "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline"
author: "Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica"
date: "April 19, 2024"
previewImg: /images/blog/arena_hard/arena_hard.png
---

Building an affordable and reliable benchmark for LLM chatbots has become a critical challenge. A high-quality benchmark should 1) robustly separate model capability, 2) reflect human preference in real-world use cases, and 3) frequently update to avoid over-fitting or test set leakage.

Traditional benchmarks are often static or close-ended (e.g., MMLU multi-choice QA), which do not satisfy the above requirements. On the other hand, models are evolving faster than ever, underscoring the need to build benchmarks with high separability.

We introduce Arena-Hard – a data pipeline to build high-quality benchmarks from live data in [Chatbot Arena](https://arxiv.org/abs/2403.04132), which is a crowd-sourced platform for LLM evals. To measure its quality, we propose two key metrics:
1. Agreement to Human preference: whether the benchmark score has high agreement to human preference.
2. Separability: whether the benchmark can confidently separate models.

We compare our new benchmark, Arena Hard v0.1, to a current leading chat LLM benchmark, MT Bench. In Figure 1, we show Arena Hard v0.1 offers significantly stronger separability against MT Bench with tighter confidence intervals. It also has a higher agreement (89.1%, see Table 1) with the human preference ranking by Chatbot Arena (english-only). We expect to see this benchmark useful for model developers to differentiate their model checkpoints.

<style>
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:#ccc;border-style:solid;border-width:1px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-head{background-color:#c0c0c0;border-color:#ccc;text-align:left;vertical-align:top;}
.tg .tg-body{text-align:left;vertical-align:top;}

table {
  border-collapse: collapse;
  width: 100%;
}
</style>

<style>
th {text-align: left}
td {text-align: left}

table {
  border-collapse: collapse;
  width: 100%;
}


th {
  cursor: pointer;
}

th:hover {
  background-color: #ddd;
}

.arrow {
  display: inline-block;
  width: 0;
  height: 0;
  vertical-align: middle;
  margin-left: 5px;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
}

.arrow-up {
  border-bottom: 5px solid #000;
}

.arrow-down {
  border-top: 5px solid #000;
}

/* Initially sort arrow for descending order */
th:nth-child(1) .arrow-down {
  border-top: 5px solid #000;
}

ul {
    list-style-type: disc !important; /* or 'circle' or 'square', depending on the bullet style you want */
    padding-left: 20px;
}

ul ul {
    list-style-type: circle !important; /* for nested lists, to distinguish from the parent list */
}

li::before {
    content: normal !important; /* This will remove any content added before the list item */
}
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


<img src="/images/blog/arena_hard/arena-hard-vs-mt_bench.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: left;">Figure 1: Comparison between MT-bench and Arena Hard v0.1. The latter offers significantly better separability between models and tighter confidence intervals. GPT-4-0314 has no variance in Arena-hard-v0.1 because it's used as the anchor model.</p>

Links:
- Evaluate your model on Arena-Hard-v0.1: [Link](https://github.com/lm-sys/arena-hard)
- Browse Arena-Hard-v0.1 prompts: [Link](https://huggingface.co/spaces/lmsys/arena-hard-browser)
- Statistic Notebook Google Colab: [Link](https://colab.research.google.com/drive/1ar6XLWREN_dXEh404WNOxroFVUe_4njp?usp=sharing)
- Full leaderboard at the Result section: [Skip](#full-leaderboard-with-gpt-4-turbo-as-judge)

We explain more technical details in the following sections.

## Key Objectives of LLM benchmarks

We outline a few key properties that an LLM chatbot benchmark should possess to provide a meaningful measurement of capabilities between models:
1. Agreement to human preference: It should correlate with human preference in real-world use cases
2. Separability: It should provide confidence interval on benchmark score and separate models with high confidence
3. Freshness: It should use new, unseen prompts to avoid potential test leakage


We define **agreement** of Benchmark A with respect to a reference Benchmark B by the below formulation:

For a given model pair (which B can separate with confidence)
  <ul>
      <li>If A can confidently separate the 2 given models</li>
      <ul>
          <li>+1.0 if the rank order agrees with B.</li>
          <li>-1.0 if the rank order disagrees with B.</li>
      </ul>
      <li>+0.0 if A cannot separate the 2 given models with confidence</li>
  </ul>

An agreement score of 1 implies benchmark A confidently agrees on the preference of every single unique models pair. On the other hand, an agreement score of -1 implies benchmark B confidently disagrees on the preference of every single unique models pair instead.

We define **separability** by whether a benchmark can separate given model pairs with derived confidence intervals (via bootstrapping). This metric can also serve to measure the variances in ranking outputs provided by a benchmark. We quantify this metric by the percentage of model pairs which have non-overlapping confidence intervals of the benchmark scores.

We use a set of top-20 models* on [Chatbot Arena](https://chat.lmsys.org/?leaderboard) (April 13, 2024) that are presented on [AlpacaEval leaderboard](https://tatsu-lab.github.io/alpaca_eval/) to calculate separability and agreement per benchmark. We consider the human preference ranking by Chatbot Arena (English only) as the reference to calculate agreement.

In Table 1, Arena-hard-v0.1 shows the highest separability (87.4%) against widely adopted LLM benchmarks and offers highest agreement (89.1%) to Chatbot Arena. It is also cheap and fast to run ($25).

Interestingly, we find Spearman Correlation, a popular metric for measuring correlations between rankings, may be an unreliable metric for ranking correlation as it does not consider variance of the rankings, and therefore fails to adequately punish essential ranking granularities of the top models we care about most. For example, when considering 95% CI, MT-bench’s agreement to Chatbot Arena drops from 91.3% to 22.6%.

You can find full statistics in the result section. 
<p style="color:gray; text-align: center;">Table 1. Separability and agreement per benchmark.</p>

<table class="tg" style="justify-content: center;">
  <colgroup>
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;">
    <col style="width: 20%;"> <!-- narrower -->
    <col style="width: 20%;"> <!-- wider -->
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
      <td class="tg-body">10,000+</td>
      <td class="tg-body">160</td>
      <td class="tg-body">800</td>
      <td class="tg-body">1,000</td>
    </tr>
    <tr>
      <td class="tg-body"><b>Agreement to Chatbot Arena with 95% CI</b></td>
      <td class="tg-body">N/A</td>
      <td class="tg-body" style="color:red">26.1%</td>
      <td class="tg-body">81.2%</td>
      <td class="tg-body" style="color:green"><b>89.1%</b></td>
    </tr>
    <tr>
      <td class="tg-body">Spearman Correlation</td>
      <td class="tg-body">N/A</td>
      <td class="tg-body">91.3%</td>
      <td class="tg-body">90.8%</td>
      <td class="tg-body" style="color:green"><b>94.1%</b></td>
    </tr>
    <tr>
      <td class="tg-body"><b>Separability with 95% CI</b></td>
      <td class="tg-body">85.8%</td>
      <td class="tg-body" style="color:red">22.6%</td>
      <td class="tg-body">83.2%</td>
      <td class="tg-body" style="color:green"><b>87.4%</b></td>
    </tr>
    <tr>
      <td class="tg-body">Real-world</td>
      <td class="tg-body">Yes</td>
      <td class="tg-body">Mixed</td>
      <td class="tg-body">Mixed</td>
      <td class="tg-body" style="color:green"><b>Yes</b></td>
    </tr>
    <tr>
      <td class="tg-body">Freshness</td>
      <td class="tg-body">Live</td>
      <td class="tg-body">Static</td>
      <td class="tg-body">Static</td>
      <td class="tg-body" style="color:green"><b>Frequent Updates</b></td>
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
</tbody>
</table>
<details close style="text-align: left; font-family: monospace; font-size: 15px;">
<summary>*Results based on 20 top models from Chatbot Arena that are also presented on Alpaca Eval</summary>
gpt-4-turbo-2024-04-09, claude-3-opus-20240229, claude-3-sonnet-20240229, gpt-4-0314, gpt-4-0613, mistral-large-2402, qwen1.5-72b-chat, mistral-medium, claude-2.0, gpt-3.5-turbo-0613, claude-2.1, gemini-pro, mixtral-8x7b-instruct-v0.1, gpt-3.5-turbo-0314, yi-34b-chat, tulu-2-dpo-70b, dbrx-instruct-preview, vicuna-33b, starling-lm-7b-alpha, llama-2-70b-chat
</details>

Next, we elaborate how to build the prompt selection pipeline to ensure data quality.

## Arena-Hard Pipeline

We build a pipeline that automatically extracts quality prompts from a dataset of 200,000 user queries collected via Chatbot Arena. This process involves ensuring:
- Diversity: Prompt set should cover a wide range of real-world topics
- Prompt quality: Each prompt should possess high quality to benchmark LLMs. we define several key criteria below (see Table 2)

<img src="/images/blog/arena_hard/method.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Figure 2: Arena-Hard Pipeline</p>

To ensure prompt diversity, we adopt a topic modeling pipeline in [BERTopic](https://github.com/MaartenGr/BERTopic) by first converting each prompt with OpenAI’s embedding (text-embedding-3-small), reducing dimension with UMAP, and using a hierarchical-based clustering algorithm (HDBSCAN) to identify clusters which are then summarized using GPT-4-turbo. This helps us identify over 4000 topics covering a wide range of domains. However, topic clusters come with varying quality and separability in benchmarking LLMs. We then develop a calibrated system prompt for LLMs to help us select high quality user queries by seven key criteria (e.g., specificity, domain knowledge, problem-solving, etc).

<table style="width:100%; border-collapse: collapse; border: 1px solid black;">
  <tr style="background-color: black; color: white;">
    <th style="border: 1px solid black; padding: 10px; text-align: left;">Table 2: 7 Key Criteria</th>
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


An LLM Judge (GPT-3.5-Turbo, GPT-4-Turbo) annotates each prompt from 0 to 7 to indicate how many criteria are met. We then score each cluster by the average score of its prompts. Below, we show examples of topic clusters ranging from low to high mean scores. We can observe clusters with higher scores often correlate to challenging topics or tasks for LLMs like game development or mathematical proofs. On the other hand, clusters with lower scores point to trivial or ambiguous questions like "Design Styles and Influences".

<img src="/images/blog/arena_hard/cluster_distribution.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Figure 3: Chatbot Arena clusters sorted by their scores.</p>

To see whether the prompt score correlates with separability, we sample 50 prompts per score and compare the responses from GPT-4 and Llama-70b, with GPT-4-Turbo as judge. We observe a strong correlation between high potential score and the win-rate of GPT-4 over Llama-70b. A similar trend is also observed in other model pairs such as Claude Sonnet vs Haiku and Mistral-large vs Mixtral.



<img src="/images/blog/arena_hard/hard_score_line.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Figure 4: Win-rate between model pairs becomes more separable as the "7 Key Criteria" score increases.</p>

## Results

### Arena-Hard-v0.1

Using the above pipeline, we identify 250 high-quality topic clusters with mean score >=6 out of 7. We then randomly sample 2 prompts per cluster to construct 500 high-quality benchmark prompts, Arena-Hard-v0.1. This benchmark set contains mostly well-defined, technical problem-solving queries as required in the above key criteria. You can browse all the prompts at this [link](https://huggingface.co/spaces/lmsys/arena-hard-browser).

However, evaluating models on challenging queries such as Arena-Hard-v0.1 is a non-trivial task. Most queries involve deep domain knowledge and problem solving skills, requiring expert-level judgment to evaluate the answer quality. Unfortunately, this is prohibitively expensive and time consuming. Following [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) and [AlpacaFarm](https://arxiv.org/abs/2305.14387), we employ LLM as a judge framework to approximate human preference.

We consider the pairwise comparison setup against a strong baseline model (GPT-4-0314), and ask a strong judge model (e.g., GPT-4-Turbo or Claude-3-Opus) to categorize the preference into five labels: A >> B, A > B, A~=B, .. B>>A. This way, a model will be penalized more in big losses than small losses, which we find to be effective in separating models. We also employ CoT to prompt the LLM judge to generate answers first before giving judgments. Full judge prompt can be found [here](https://github.com/lm-sys/arena-hard/blob/main/config/judge_config.yaml).

To avoid potential position bias, we adopt a two-game setup – per query we swap the models on the first & second position. This results in 500x2=1000 judgments per model evaluation. Following Chatbot Arena, we adopt the Bradley-Terry model to produce model’s the final model scores. By bootstrapping the comparisons from all models, we find it to be statistically stable compared to only considering win-rate against the baseline model.

### Full Leaderboard with GPT-4-Turbo as judge

We use gpt-4-1106-preview as the judge model to generate judgment for the model response against baseline. We take all the comparisons and compute each model’s Bradley-Terry coefficient. We then transform it to win-rate against the baseline as the final score. The 95% confidence interval is computed via 100 rounds of bootstrapping.

<p style="color:gray; text-align: center;">Arena Hard v0.1 Leaderboard (baseline: GPT-4-0314)</p>
<div style="display: flex; justify-content: center; font-family: Consolas, monospace;">
<table style="line-height: 1; font-size: 1.0em;">
  <caption style="text-align: left; color: red">*Note: GPT-4-Turbo’s high score can be due to the GPT-4 judge favoring GPT-4 outputs.</caption>
  <thead>
    <tr style="border-bottom: thin solid #ccc;">
      <th style="width: 40%;">Model Name</th>
      <th style="width: 20%;">Score</th>
      <th style="width: 20%;">95% CI</th>
      <th style="width: 20%;">Average #Tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">gpt-4-turbo-2024-04-09*</td>
      <td>82.6</td>
      <td>-1.8/+1.6</td>
      <td>662</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-4-0125-preview*</td>
      <td>78.0</td>
      <td>-2.2/+2.4</td>
      <td>619</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-opus-20240229</td>
      <td>60.4</td>
      <td>-3.3/+2.4</td>
      <td>541</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-4-0314</td>
      <td>50.0</td>
      <td>-0.0/+0.0</td>
      <td>423</td>
    </tr>
    <tr>
  <td style="text-align: left;">claude-3-sonnet-20240229</td>
  <td>46.8</td>
  <td>-2.1/+2.2</td>
  <td>552</td>
</tr>
<tr>
  <td style="text-align: left;">claude-3-haiku-20240307</td>
  <td>41.5</td>
  <td>-2.8/+2.5</td>
  <td>505</td>
</tr>
<tr>
  <td style="text-align: left;">llama-3-70b-instruct</td>
  <td>41.1</td>
  <td>-2.5/+2.4</td>
  <td>583</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-4-0613</td>
  <td>37.9</td>
  <td>-2.2/+2.0</td>
  <td>354</td>
</tr>
<tr>
  <td style="text-align: left;">mistral-large-2402</td>
  <td>37.7</td>
  <td>-1.9/+2.6</td>
  <td>400</td>
</tr>
<tr>
  <td style="text-align: left;">mixtral-8x22b-instruct-v0.1</td>
  <td>36.4</td>
  <td>-2.7/+2.9</td>
  <td>430</td>
</tr>
<tr>
  <td style="text-align: left;">Qwen1.5-72B-Chat</td>
  <td>36.1</td>
  <td>-2.5/+2.2</td>
  <td>474</td>
</tr>
<tr>
  <td style="text-align: left;">command-r-plus</td>
  <td>33.1</td>
  <td>-2.1/+2.2</td>
  <td>541</td>
</tr>
<tr>
  <td style="text-align: left;">mistral-medium</td>
  <td>31.9</td>
  <td>-2.3/+2.4</td>
  <td>485</td>
</tr>
<tr>
  <td style="text-align: left;">mistral-next</td>
  <td>27.4</td>
  <td>-2.1/+1.7</td>
  <td>297</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0613</td>
  <td>24.8</td>
  <td>-1.6/+2.0</td>
  <td>401</td>
</tr>
<tr>
  <td style="text-align: left;">claude-2.0</td>
  <td>24.0</td>
  <td>-2.5/+2.5</td>
  <td>295</td>
</tr>
<tr>
  <td style="text-align: left;">dbrx-instruct</td>
  <td>23.9</td>
  <td>-1.4/+1.5</td>
  <td>415</td>
</tr>
<tr>
  <td style="text-align: left;">Mixtral-8x7B-Instruct-v0.1</td>
  <td>23.4</td>
  <td>-2.3/+1.7</td>
  <td>457</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125</td>
  <td>23.3</td>
  <td>-2.2/+2.3</td>
  <td>329</td>
</tr>
<tr>
  <td style="text-align: left;">Yi-34B-Chat</td>
  <td>23.1</td>
  <td>-1.8/+2.0</td>
  <td>611</td>
</tr>
<tr>
  <td style="text-align: left;">Starling-LM-7B-beta</td>
  <td>23.0</td>
  <td>-1.9/+2.2</td>
  <td>530</td>
</tr>
<tr>
  <td style="text-align: left;">claude-2.1</td>
  <td>22.8</td>
  <td>-1.6/+2.1</td>
  <td>290</td>
</tr>
<tr>
  <td style="text-align: left;">Snorkel-Mistral-PairRM-DPO</td>
  <td>20.7</td>
  <td>-2.2/+1.5</td>
  <td>564</td>
</tr>
<tr>
  <td style="text-align: left;">llama-3-8b-instruct</td>
  <td>20.6</td>
  <td>-2.5/+1.8</td>
  <td>585</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-1106</td>
  <td>18.9</td>
  <td>-1.6/+2.1</td>
  <td>285</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0301</td>
  <td>18.1</td>
  <td>-1.7/+1.2</td>
  <td>334</td>
</tr>
<tr>
  <td style="text-align: left;">gemini-1.0-pro</td>
  <td>17.8</td>
  <td>-1.7/+1.7</td>
  <td>322</td>
</tr>
<tr>
  <td style="text-align: left;">command-r</td>
  <td>17.0</td>
  <td>-1.9/+1.7</td>
  <td>432</td>
</tr>
<tr>
  <td style="text-align: left;">tulu-2-dpo-70b</td>
  <td>15.0</td>
  <td>-1.4/+1.2</td>
  <td>550</td>
</tr>
<tr>
  <td style="text-align: left;">Starling-LM-7B-alpha</td>
  <td>12.8</td>
  <td>-1.4/+1.4</td>
  <td>483</td>
</tr>
<tr>
  <td style="text-align: left;">mistral-7b-instruct-v0.2</td>
  <td>12.6</td>
  <td>-1.6/+1.3</td>
  <td>541</td>
</tr>
<tr>
  <td style="text-align: left;">Llama-2-70b-chat-hf</td>
  <td>11.6</td>
  <td>-1.6/+1.4</td>
  <td>595</td>
</tr>
<tr>
  <td style="text-align: left;">vicuna-33b-v1.3</td>
  <td>8.6</td>
  <td>-1.3/+1.0</td>
  <td>451</td>
</tr>
<tr>
  <td style="text-align: left;">gemma-7b-it</td>
  <td>7.5</td>
  <td>-1.1/+1.2</td>
  <td>378</td>
</tr>
<tr>
  <td style="text-align: left;">Llama-2-7b-chat-hf</td>
  <td>4.6</td>
  <td>-0.8/+0.8</td>
  <td>561</td>
</tr>
<tr>
  <td style="text-align: left;">gemma-2b-it</td>
  <td>3.0</td>
  <td>-0.6/+0.7</td>
  <td>369</td>
</tr>
</tbody>
</table>
</div>

### GPT-4-Turbo or Claude as Judge?

We also compare two strongest LLMs: GPT-4-1106-Preview and Claude-3 Opus as the judge mode in Table 3. When GPT-4 Judge is used, we observe higher separability across models (ranging from 23.0 to 78.0). When Claude Judge is used, we find the Claude family of models scores in general go up, despite it still favoring gpt-4-0125-preview over itself. Surprisingly, it favors several open models (Mixtral, Yi, Starling) or even gpt-3.5-turbo over gpt-4-0613.

<p style="color:gray; text-align: center;">Table 3. Leaderboard Comparison Between GPT and Claude as Judge</p>
<div style="display: flex; justify-content: center; font-family: Consolas, monospace;">
<table style="line-height: 1; font-size: 1.0em;">
  <thead>
    <tr style="border-bottom: thin solid #ccc;">
      <th style="width: 30%;">Model Name</th>
      <th style="width: 25%;">GPT-4-1106-Preview Judge</th>
      <th style="width: 25%;">Claude-3-Opus<br>Judge</th>
      <th style="width: 20%;">Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">gpt-4-0125-preview</td>
      <td>78.0</td>
      <td>76.3 <span style="color: red;">(↓)</span></td>
      <td style="color: red;">-1.7</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-opus-20240229</td>
      <td>60.4</td>
      <td>71.8 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+11.4</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-sonnet-20240229</td>
      <td>46.8</td>
      <td>63.6 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+16.8</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-haiku-20240307</td>
      <td>41.5</td>
      <td>56.1 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+14.6</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-4-0613</td>
      <td>37.9</td>
      <td>30.6 <span style="color: red;">(↓)</span></td>
      <td style="color: red;">-7.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-3.5-0613</td>
      <td>24.8</td>
      <td>34.7 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+9.9</td>
    </tr>
    <tr>
      <td style="text-align: left;">mixtral-8x22b-instruct-v0.1</td>
      <td>23.4</td>
      <td>34.8 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+11.4</td>
    </tr>
    <tr>
      <td style="text-align: left;">yi-34b-chat</td>
      <td>23.1</td>
      <td>46.6 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+23.5</td>
    </tr>
    <tr>
      <td style="text-align: left;">starling-lm-7b-beta</td>
      <td>23.0</td>
      <td>45.0 <span style="color: green;">(↑)</span></td>
      <td style="color: green;">+22</td>
    </tr>
  </tbody>
</table>
</div>


We further compare GPT-4 and Claude Judges using our proposed metrics of separability and agreement in Table 4, and find that the GPT-4-turbo Judge is significantly better across all metrics. 

<table style="border-collapse: collapse; border: 1px solid black">
  <caption>Table 4: Statistical comparisons between LLM Judges and Human</caption>
  <tr>
    <td style="border: 1px solid black"></td>
    <td style="border: 1px solid black">Arena-Hard-v0.1 (GPT-4-1106-Preview Judge)</td>
    <td style="border: 1px solid black">Arena-Hard-v0.1 (Claude-3 Judge)</td>
  </tr>
  <tr>
    <td style="border: 1px solid black">Agreement to Chatbot Arena with 95% CI</td>
    <td style="border: 1px solid black"><b>89.1%</b></td>
    <td style="border: 1px solid black">66.7%</td>
  </tr>
  <tr>
    <td style="border: 1px solid black">Separability with 95% confidence intervals</td>
    <td style="border: 1px solid black"><b>87.4%</b></td>
    <td style="border: 1px solid black">83.7%</td>
  </tr>
  <tr>
    <td style="border: 1px solid black">Spearman Correlation</td>
    <td style="border: 1px solid black"><b>94.2%</b></td>
    <td style="border: 1px solid black">77.0%</td>
  </tr>
    <tr>
    <td style="border: 1px solid black">Brier Score*</td>
    <td style="border: 1px solid black"><b>0.07</b></td>
    <td style="border: 1px solid black">0.17</td>
  </tr>
</table>
<caption>*Brier Score (lower is better), a statistical scoring function for measuring the accuracy of probabilistic accuracy. (see section View Benchmarking as a Forecasting Problem for more information)</caption>

We manually compared different judgment examples between GPT-4-Turbo and Claude as a judge. We found that when the two judges disagreed, it could usually be broken down into two main categories:
1. Conservative scoring
2. Differing perspectives on the user's prompt

We find that Claude-3-Opus is much less likely to give harsh scores – it is particularly hesitant to proclaim one response as "significantly better" than another. In contrast, GPT-4-Turbo will identify errors in a model's response that led to an incorrect answer and penalize the model with a significantly lower score. On the other hand, Claude-3-Opus sometimes overlooks smaller errors. Even when Claude-3-Opus does identify these errors, it tends to treat them as minor issues and shows leniency during scoring. This effect is particularly present in coding and math problems, where small mistakes are more likely to completely derail the final answer; these scorings are still given leniency from Claude-3-Opus but not GPT-4-Turbo. See the appendix below for specific examples of differing judgments, many of which exhibit this phenomenon.

<img src="/images/blog/arena_hard/score_strength.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 75%"></img>
<p style="color:gray; text-align: center;">Figure 5: Score Strength</p>

There is also a small subset of prompts in which Claude-3-Opus and GPT-4-Turbo judge with fundamentally different perspectives. For example, given a coding question, Claude-3-Opus may choose the response that provides the most educational value to the user, offering a simplistic structure without relying on external libraries. GPT-4-Turbo, however, may prioritize the response that provides the most practical answer, regardless of its educational value to the user.  While both interpretations are valid judging criteria, we find GPT-4-Turbo’s perspective may be more correlated with the average user.

Despite the observed differences between Claude-3-Opus and GPT-4-Turbo judgment styles, we find the judges have an overall soft agreement rate of 80%. Two judgments “soft agree” if they are at most distance one apart, or in other words they do not contradict.

## Limitations

### Verbosity: does the LLM Judge prefer longer responses?

LLM as judges are known to suffer from verbosity bias ([Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475)). Below we plot the avg token length and score per model for both MT-Bench and Arena-Hard-v0.1. Visually, there isn't a strong correlation between score and length.

<img src="/images/blog/arena_hard/verbose_scatterplot.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 60%"></img>
<p style="color:gray; text-align: center;">Figure 6: Verbosity scatterplot comparing Arena-Hard-v0.1 and MT Bench.</p>

To further examine potential verbosity bias, we conduct an ablation on three different system prompts (original, chatty, detailed) with GPT-3.5-Turbo. We observe that both GPT-4-Turbo and Claude-3-Opus judges may be affected by longer outputs, while Claude being significantly more impacted with a “more detailed” system prompt as GPT-3.5-Turbo reaches a win-rate of over 40% against GPT-4-0314. 

Interestingly, the “chatty” system prompt doesn’t affect much on the win-rate by both judges, despite the longer average #tokens. This suggests output length is not the only factor. It is possible that more detailed answers are also more helpful and thus preferred by LLM judges.


<p style="color:gray; text-align: center;">Table 5. Length Bias Comparison Between GPT and Claude as Judge</p>
<div style="display: flex; justify-content: center; font-family: Consolas, monospace;">
<table style="line-height: 1; font-size: 1.0em;">
  <thead>
    <tr style="border-bottom: thin solid #ccc;">
      <th style="width: 40%;">Model Name</th>
      <th style="width: 30%;">Win Rate</th>
      <th style="width: 30%;">Average Token #</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border: 1px solid black;">
      <td style="text-align: left;"><b>GPT-4-1106-Preview</b></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-3.5-turbo-0125-detailed</td>
      <td>29.86</td>
      <td>421</td>
    </tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125-chatty</td>
  <td>23.89</td>
  <td>361</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125</td>
  <td>23.2</td>
  <td>328</td>
</tr>
<tr style="border: 1px solid black;">
  <td style="text-align: left;"></td>
  <td></td>
  <td></td>
</tr>
<tr style="border: 1px solid black;">
  <td style="text-align: left;"><b>Claude-3-Opus</b></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125-detailed</td>
  <td>40.78</td>
  <td>421</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125-chatty</td>
  <td>28.49</td>
  <td>375</td>
</tr>
<tr>
  <td style="text-align: left;">gpt-3.5-turbo-0125</td>
  <td>27.97</td>
  <td>328</td>
</tr>
</tbody>
</table>
</div>
<caption style="font-family: Consolas, monospace; font-size: 15px;">
System Prompt:<br>detailed: “You are a helpful assistant who thoroughly explains things with as much detail as possible.”<br>chatty: “You are a helpful assistant who is chatty.”
</caption>

### Variance in GPT-4 judgments

We find that even with temperature=0, GPT-4-Turbo may still generate slightly different judgments. Here we repeat the judgments for gpt-3.5-turbo-0125 three times and report its variance. Due to limited budget, we can only evaluate all the models once. We recommend using the confidence intervals to determine model separation.

<p style="color:gray; text-align: center;">Table 6. Variances between 3 separate runs of Arena Hard v0.1.</p>
<div style="display: flex; justify-content: center; font-family: Consolas, monospace;">
<table style="line-height: 1; font-size: 1.0em;">
  <thead>
    <tr style="border-bottom: thin solid #ccc;">
      <th style="width: 40%;">Model Name</th>
      <th style="width: 30%;">Win Rate</th>
      <th style="width: 30%;">Average Token #</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">gpt-3.5-turbo-0125-1</td>
      <td>23.05</td>
      <td>328</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-3.5-turbo-0125-2</td>
      <td>22.93</td>
      <td>328</td>
    </tr>
        <tr>
      <td style="text-align: left;">gpt-3.5-turbo-0125-3</td>
      <td>22.75</td>
      <td>328</td>
    </tr>
</tbody>
</table>
</div>

### Potential self-bias & prompt selection bias

We also observe potential self-bias in LLM judges (e.g., Claude Judge prefers Claude answers).
In addition, the prompt selection process could be biased by the LLMs. The benchmark also does not evaluate multi-turn interactions.


## Viewing Benchmarking as a Forecasting Problem

In this section we attempt to combine both confidence and correlation into one standardized metric for benchmarking.

<table style="border-collapse: collapse; border: 1px solid black">
  <caption>Correlation of Brier Score with Overall Chatbot Arena Score Across Different Models</caption>
  <tr>
    <td style="border: 1px solid black">Arena Hard</td>
    <td style="border: 1px solid black">Chabot Arena* (20K Votes)</td>
    <td style="border: 1px solid black">MT Bench</td>
    <td style="border: 1px solid black">Alpaca 2.0 LC</td>
  </tr>
  <tr>
    <td style="border: 1px solid black"><b>0.07</b></td>
    <td style="border: 1px solid black">0.08</td>
    <td style="border: 1px solid black">0.09</td>
    <td style="border: 1px solid black">0.11</td>
  </tr>
</table>
<caption>*20K human preference battles randomly sampled from Chatbot Arena between the 20 top models.</caption>

Model developers generally use benchmarks for model selection, not ground truth certification of performance.  Benchmarks serve as a cheap and lightweight proxy for more expensive and complex evaluations like ground truth Bradley Terry Coefficients derived from human preference. Thus, we expect benchmarks to tell us, as model developers, some confidence bound on what a model’s real world performance will be. In this sense, a benchmark serves as a forecast for true long-run performance.

Forecasting is a delicate balance between confidence and uncertainty. Therefore, a good benchmark should show confidence when separating clearly unequal models, but should demonstrate uncertainty when ranking differences between legitimately similar models. One might argue we only need to look at how confident a given benchmark is at separating model pairs. A good benchmark is not necessarily always confident at separating models– you don’t want your benchmark to be confidently incorrect. For example, given a pair of models A and B and benchmark 1 and 2. Let’s assume ground truth is model A is better than model B. We bootstrap both benchmark 1 and 2 and retrieve their confidence intervals for both model’s performances. Benchmark 1 confidently predicts model B is better than A while Benchmark 2 predicts model B is better than A with low confidence. In this case, we should say Benchmark 2 is actually better than Benchmark 1 at predicting this pair of models. This is to say, high confidence should be rewarded only when the answer is correct, and low confidence is better when incorrect.

In this problem context, we introduce the prediction criteria as simply the binary indicator **1**$(\pi_a < \pi_b)$ for some model pair ($\pi_a$ and $\pi_b$).  The forecast gives a probability that this indicator is true, $P(\pi_a < \pi_b)$.  A higher probability forecast indicates greater confidence that **1**$(\pi_a < \pi_b)$ will be true.  We can generate these probability predictions using bootstrapped score mean and variance, which in turn define a gaussian distribution. We then resolve the ground truth label for **1**$(\pi_a < \pi_b)$ using Chatbot Arena's Bradley Terry coefficients.

A well-defined fair-in-expectation loss for forecasting is [Brier Score](https://en.wikipedia.org/wiki/Brier_score). Brier score rewards confidence when forecasts are correct while punishing confident errors. We can calculate the loss over a benchmark prediction of **1**$(\pi_a < \pi_b)$ for each model pair with respect to the Chatbot Area ground truth scores to quantify a benchmark’s forecasting performance. Here we assume Chatbot Arena as “ground truth” as both Alpaca 2.0 LC and Arena Hard are advertised as an inexpensive alternative to Chatbot Arena as an evaluation pipeline. We will conduct future study on correlation comparison where we instead use Chatbot Arena's Bradley Terry coefficient derived from similar distributions as the given benchmark.

We find that Arena Hard averages much lower forecasting loss, demonstrating that it is both accurate in score, and accurate in confidence level.
<div style="display: flex; gap: 10px;">
  <div style="width: 48%;">
    <img src="/images/blog/arena_hard/forecast_arena_20k.png">
  </div>
  <div style="width: 48%;">
    <img src="/images/blog/arena_hard/forecast_arena_hard.png">
  </div>
</div>
<div style="display: flex; gap: 10px;">
  <div style="width: 48%;">
    <img src="/images/blog/arena_hard/forecast_alpaca.png">
  </div>
  <div style="width: 48%;">
    <img src="/images/blog/arena_hard/forecast_mt_bench.png">
  </div>
</div>

Above is the predicted model predicted probability against the bootstrapped arena “ground truth” probability (jittered to show clusters).  While both Alpaca eval and Arena Hard have large clusters around (0,0) and (1,1) signifying good forecasting, Arena Hard has lighter clusters on (0,1) and (1,0), if any, revealing less overconfidence. MT Bench has heavy tails along the top and bottom, revealing underconfidence. However, none of these benchmarks show an “ideal” y=x curve (with dense ends) expected with a perfectly calibrated forecast, signifying room for future research.

## Future
We hope to study deeper into the above limitations and biases in the later technical report. We are also working on diving deeper into the statistics for more studies on how to measure the quality of benchmarks. Lastly, we also hope to upgrade Arena-Hard frequently. So expect frequent new benchmarks! 


## Acknowledgment
We thank Matei Zaharia, Yann Dubois, Anastasios Angelopoulos, Lianmin Zheng, Lewis Tunstall, Nathan Lambert, Xuechen Li, Naman Jain, Ying Sheng, Maarten Grootendorst for their valuable feedback. We thank Microsoft [AFMR](https://www.microsoft.com/en-us/research/collaboration/accelerating-foundation-models-research/) for Azure OpenAI credits support. We also thank Together.ai & Anyscale for open model endpoint support.

## Citation
```
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```

## Appendix
<img src="/images/blog/arena_hard/heatmap.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 120%"></img>
<p style="color:gray; text-align: center;">Appendix Figure 1: Similarity Heatmap of 50 Arena Hard Clusters</p>

<img src="/images/blog/arena_hard/clustering_filtered_small_64.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 120%"></img>
<p style="color:gray; text-align: center;">Appendix Figure 2: Top-64 clusters visualized in hierarchy. x-axis represents the cosine similarity distance. y-axis shows the topic title per cluster summarized by gpt-4-turbo.</p>