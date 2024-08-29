---
title: "Does style matter? Disentangling style and substance in Chatbot Arena"
author: "Tianle Li*, Anastasios Angelopoulos*, Wei-Lin Chiang*"
date: "Aug 28, 2024"
previewImg: /images/blog/style_control/logo.png
---

Why is GPT-4o-mini so good? 

Why does Claude rank so low, when anecdotal experience suggests otherwise?

We have answers for you. We controlled for the effect of length and markdown, and indeed, *the ranking changed*. This is just a first step towards our larger goal of disentangling **substance** and **style** in Chatbot Arena leaderboard.

**Check out the results below!** It turns out that style has a strong effect on models’ performance in the leaderboard. This makes sense—from the perspective of human preference, it’s not just what you say, but how you say it. But now, we have a way of _separating_ the effect of writing style from the content, so you can see both, and they aren’t mixed up.

When adjusting for length and style, GPT-4o-mini and Grok-2-mini drop below most frontier models, and Claude 3.5 Sonnet, Opus, and Llama-3.1-405B rise substantially. In the Hard Prompt subset, we..  [Wei-Lin Chiang Tianle Li add any other major takeaways] We are looking forward to seeing what the community does with this new tool for disaggregating style and substance.


## Overall ranking + Style Control
<img src="/images/blog/style_control/comparison_overall.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 1. Overall Chatbot Arena ranking vs Overall Chatbot Arena ranking where answer length, markdown header count, markdown bold count, and markdown list element count are being “controlled”.</p>

## Hard Prompt ranking + Style Control
<img src="/images/blog/style_control/comparison_hard.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 2. Hard Prompt category ranking vs Hard Prompt category ranking where answer length, markdown header count, markdown bold count, and markdown list element count are being “controlled”.</p>

## Methodology

**High-Level Idea.** The goal here is to understand the effect of _style_ vs _substance_ on the Arena Score. Consider models A and B. Model A is great at producing code, factual and unbiased answers, etc., but it outputs short and terse responses. Model B is not so great on substance, but it outputs great markdown, and gives long, detailed, flowery responses. Which is better, model A, or model B?

The answer is not one dimensional. Model A is better on substance, and Model B is better on style. Ideally, we would have a way of teasing apart this distinction: capturing how much of the model’s Arena Score is due to substance or style. 

Our methodology is a first step towards this goal. We explicitly model style as an independent variable in our Bradley-Terry regression. For example, we added length as a feature—just like each model, the length difference has its _own_ Arena Score! By doing this, we expect that the Arena Score of each model will reflect its strength, controlled for the effect of length. 

Please read below for the technical details. We also controlled not just for length, but also a few other style features. As a first version, we propose controlling
1. Answer token length
2. Number of markdown headers
3. Number of markdown bold elements
4. Number of markdown lists

We publicly release our data with vote and style elements and code on [insert google colab link]! You can try out experimenting with style control now. More improvements to come, and please reach out if you want to help contribute! 


**Background.** To produce the results above, we controlled for the effect of style by adding extra “style features” into our Bradley-Terry regression. This is a [standard technique](https://en.wikipedia.org/wiki/Controlling_for_a_variable) in statistics, and has been recently used in LLM evaluations [1](https://arxiv.org/abs/2404.04475). The idea is that, by including any confounding variables (e.g. response length) in the regression, we can attribute any increase in strength to the confounder, as opposed to the model. Then, the Bradley-Terry coefficient will be more reflective of the model’s intrinsic properties, as opposed to undesirable confounders. The definition of a confounder is to some extent up to our interpretation; as our style features, we use the (normalized) difference in response lengths, the number of markdown headers, and the number of lists.

More formally, consider vectors $X_1, \ldots, X_n \in \mathbb{R}^M$ and $Y_1, \ldots, Y_n \in \{0,1\}$, where $n$ is the number of battles and $M$ is the number of models. 

For every $i \in [n]$, We have that $X_{i,m}=1$ only if model $m \in [M]$ is the model shown in the left-hand side in Chatbot Arena, and $X_{i,m}=-1$ only if it is shown on the right. That is, $X_i$ is a two-hot vector. The outcome $Y_i$ takes the value $Y_i=1$ if the left-hand model wins, and $Y_i=0$ otherwise. 

The standard method for computing the Arena Score (i.e., the Bradley-Terry coefficients, which we formerly called the Elo score) is to run a logistic regression of $Y_i$ onto $X_i$. That is, for every model $m$, we associate a scalar $\hat{\beta}_m$ that describes its strength, and the vector $\hat{\beta}$ is determined by solving the following logistic regression:

$$\hat{\beta} = \arg \min_{\beta \in \mathbb{R}^M} \frac{1}{n}\sum\limits_{i=1}^n \mathsf{BCELoss}(X_i^\top \beta, Y_i)$$

where  $\mathsf{BCELoss}$ represents the binary cross-entropy loss. (In practice, we also reweight this objective to handle non-uniform model sampling, but let’s ignore that for now.)

## Style Control

Now, for every battle $i \in [n]$, let’s say that in addition to $X_i$ that we observe some additional style features, $Z_i \in \mathbb{R}^S$. These style features can be as simple or complicated as you want. For example, $Z_i$ could just be the difference in response lengths of the two models, in which case $S=1$. Or, we could have $S>1$ and include a bunch of other style-related features, for example, the number of markdown headers, or even style features that are automatically extracted by a model!

Here, we define each style feature as
$$\text{normalize }(\frac{\text{feature}_A - \text{feature}_B}{\text{feature}_A + \text{feature}_B})$$

For example, the first new feature, token length difference between answer A and answer B, would be expressed as 
$$\text{normalize }(\frac{\text{length}_A - \text{length}_B}{\text{length}_A + \text{length}_B})$$

We divide the difference by the sum of both answers' token length to make the length difference proportional to the pairwise answer token lengths. An answer with 500 tokens is roughly equal in length to an answer with 520 tokens, while an answer with 20 tokens is very different from an answer with 40 tokens, even though the difference is 20 tokens for both scenarios.

The idea of style control is very basic. We perform the same logistic regression as below:
$$\hat{\beta}, \hat{\gamma} = \arg \min_{\beta \in \mathbb{R}^M, \gamma \in \mathbb{R}^S} \frac{1}{n}\sum\limits_{i=1}^n \mathsf{BCELoss}(X_i^\top \beta + Z_i^{\top}\gamma, Y_i).$$
We refer to the results $\hat{\beta}$ and $\hat{\gamma}$ as the “model coefficients” and the “style coefficients” respectively. The model coefficients have the same interpretation as before; however, they are controlled for the effect of style, which is explicitly modeled by the style coefficients!

When the style coefficients are big, that means that the style feature has a big effect on the response. To define “big”, you need to properly normalize the style coefficients so they can be compared. All in all, when analyzing the style coefficients, we found that length was the dominant style factor. All other markdown effects are second order.

We report the following coefficient for each style attribute:
Length: 0.249, Markdown List: 0.031, Markdown Header: 0.024, Markdown Bold: 0.019


## Ablation

Next, we compare the ranking changes between controlling for answer length only, markdown element only, and both. We present the Chatbot Arena Overall table first.
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: left; padding: 8px; width: 30%;">Model</th>
    <th style="text-align: center; padding: 8px; width: 25%;">Rank Diff (Length Only)</th>
    <th style="text-align: center; padding: 8px; width: 25%;">Rank Diff (Markdown Only)</th>
    <th style="text-align: center; padding: 8px; width: 20%;">Rank Diff (Both)</th>
  </tr>
<tr>
    <td style="text-align: left; padding: 8px;">chatgpt-4o-latest</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-exp-0827</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-exp-0801</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4o-2024-05-13</td>
    <td style="text-align: center; padding: 8px; color: green;">5->3</td>
    <td style="text-align: center; padding: 8px; color: green;">5->3</td>
    <td style="text-align: center; padding: 8px; color: green;">5->2</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">claude-3-5-sonnet-20240620</td>
    <td style="text-align: center; padding: 8px; color: green;">6->5</td>
    <td style="text-align: center; padding: 8px; color: green;">6->4</td>
    <td style="text-align: center; padding: 8px; color: green;">6->4</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-advanced-0514</td>
    <td style="text-align: center; padding: 8px; color: green;">7->5</td>
    <td style="text-align: center; padding: 8px; color: red;">7->8</td>
    <td style="text-align: center; padding: 8px; color: green;">7->6</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">grok-2-2024-08-13</td>
    <td style="text-align: center; padding: 8px; color: red;">2->4</td>
    <td style="text-align: center; padding: 8px; color: red;">2->4</td>
    <td style="text-align: center; padding: 8px; color: red;">2->5</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">llama-3.1-405b-instruct</td>
    <td style="text-align: center; padding: 8px;">6->6</td>
    <td style="text-align: center; padding: 8px; color: green;">6->4</td>
    <td style="text-align: center; padding: 8px;">6->6</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4o-2024-08-06</td>
    <td style="text-align: center; padding: 8px; color: green;">7->6</td>
    <td style="text-align: center; padding: 8px; color: red;">7->8</td>
    <td style="text-align: center; padding: 8px; color: green;">7->6</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-turbo-2024-04-09</td>
    <td style="text-align: center; padding: 8px; color: green;">11->8</td>
    <td style="text-align: center; padding: 8px; color: green;">11->8</td>
    <td style="text-align: center; padding: 8px; color: green;">11->9</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">claude-3-opus-20240229</td>
    <td style="text-align: center; padding: 8px; color: green;">16->14</td>
    <td style="text-align: center; padding: 8px; color: green;">16->8</td>
    <td style="text-align: center; padding: 8px; color: green;">16->10</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-api-0514</td>
    <td style="text-align: center; padding: 8px; color: green;">10->8</td>
    <td style="text-align: center; padding: 8px; color: red;">10->13</td>
    <td style="text-align: center; padding: 8px;">10->10</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-flash-exp-0827</td>
    <td style="text-align: center; padding: 8px; color: red;">6->8</td>
    <td style="text-align: center; padding: 8px; color: red;">6->9</td>
    <td style="text-align: center; padding: 8px; color: red;">6->9</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-1106-preview</td>
    <td style="text-align: center; padding: 8px; color: green;">16->14</td>
    <td style="text-align: center; padding: 8px; color: green;">16->8</td>
    <td style="text-align: center; padding: 8px; color: green;">16->11</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;"><strong>gpt-4o-mini-2024-07-18</strong></td>
    <td style="text-align: center; padding: 8px; color: red;">6->8</td>
    <td style="text-align: center; padding: 8px; color: red;">6->11</td>
    <td style="text-align: center; padding: 8px; color: red;">6->11</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-0125-preview</td>
    <td style="text-align: center; padding: 8px; color: green;">17->14</td>
    <td style="text-align: center; padding: 8px; color: green;">17->12</td>
    <td style="text-align: center; padding: 8px; color: green;">17->13</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">mistral-large-2407</td>
    <td style="text-align: center; padding: 8px; color: green;">16->14</td>
    <td style="text-align: center; padding: 8px; color: green;">16->13</td>
    <td style="text-align: center; padding: 8px; color: green;">16->13</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">athene-70b-0725</td>
    <td style="text-align: center; padding: 8px;">16->16</td>
    <td style="text-align: center; padding: 8px; color: red;">16->17</td>
    <td style="text-align: center; padding: 8px; color: red;">16->17</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;"><strong>grok-2-mini-2024-08-13</strong></td>
    <td style="text-align: center; padding: 8px; color: red;">6->15</td>
    <td style="text-align: center; padding: 8px; color: red;">6->15</td>
    <td style="text-align: center; padding: 8px; color: red;">6->18</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-api-0409-preview</td>
    <td style="text-align: center; padding: 8px; color: red;">11->16</td>
    <td style="text-align: center; padding: 8px; color: red;">11->21</td>
    <td style="text-align: center; padding: 8px; color: red;">11->18</td>
  </tr>
</table>

We also perform the same comparison on Chatbot Arena Hard Prompt Category.
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: left; padding: 8px; width: 30%;">Model</th>
    <th style="text-align: center; padding: 8px; width: 25%;">Rank Diff (Length Only)</th>
    <th style="text-align: center; padding: 8px; width: 25%;">Rank Diff (Markdown Only)</th>
    <th style="text-align: center; padding: 8px; width: 20%;">Rank Diff (Both)</th>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">chatgpt-4o-latest</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
    <td style="text-align: center; padding: 8px;">1->1</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;"><strong>claude-3-5-sonnet-20240620</strong></td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px; color: green;">2->1</td>
    <td style="text-align: center; padding: 8px; color: green;">2->1</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-exp-0827</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px; color: green;">2->1</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-exp-0801</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4o-2024-05-13</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px;">2->2</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">llama-3.1-405b-instruct</td>
    <td style="text-align: center; padding: 8px;">4->4</td>
    <td style="text-align: center; padding: 8px; color: green;">4->2</td>
    <td style="text-align: center; padding: 8px; color: green;">4->3</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">grok-2-2024-08-13</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
    <td style="text-align: center; padding: 8px; color: red;">2->3</td>
    <td style="text-align: center; padding: 8px; color: red;">2->4</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-flash-exp-0827</td>
    <td style="text-align: center; padding: 8px;">4->4</td>
    <td style="text-align: center; padding: 8px; color: red;">4->6</td>
    <td style="text-align: center; padding: 8px;">4->4</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-1.5-pro-api-0514</td>
    <td style="text-align: center; padding: 8px; color: green;">7->6</td>
    <td style="text-align: center; padding: 8px;">7->7</td>
    <td style="text-align: center; padding: 8px;">7->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4o-2024-08-06</td>
    <td style="text-align: center; padding: 8px;">4->4</td>
    <td style="text-align: center; padding: 8px; color: red;">4->6</td>
    <td style="text-align: center; padding: 8px;">4->4</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gemini-advanced-0514</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">claude-3-opus-20240229</td>
    <td style="text-align: center; padding: 8px; color: green;">14->7</td>
    <td style="text-align: center; padding: 8px; color: green;">14->7</td>
    <td style="text-align: center; padding: 8px; color: green;">14->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">mistral-large-2407</td>
    <td style="text-align: center; padding: 8px;">7->7</td>
    <td style="text-align: center; padding: 8px; color: green;">7->6</td>
    <td style="text-align: center; padding: 8px;">7->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-1106-preview</td>
    <td style="text-align: center; padding: 8px; color: green;">11->10</td>
    <td style="text-align: center; padding: 8px; color: green;">11->7</td>
    <td style="text-align: center; padding: 8px; color: green;">11->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-turbo-2024-04-09</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
    <td style="text-align: center; padding: 8px; color: green;">9->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">athene-70b-0725</td>
    <td style="text-align: center; padding: 8px; color: green;">11->7</td>
    <td style="text-align: center; padding: 8px; color: green;">11->8</td>
    <td style="text-align: center; padding: 8px; color: green;">11->7</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4o-mini-2024-07-18</td>
    <td style="text-align: center; padding: 8px; color: red;">4->7</td>
    <td style="text-align: center; padding: 8px; color: red;">4->7</td>
    <td style="text-align: center; padding: 8px; color: red;">4->11</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">gpt-4-0125-preview</td>
    <td style="text-align: center; padding: 8px; color: green;">15->14</td>
    <td style="text-align: center; padding: 8px; color: green;">15->10</td>
    <td style="text-align: center; padding: 8px; color: green;">15->13</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">grok-2-mini-2024-08-13</td>
    <td style="text-align: center; padding: 8px; color: red;">5->12</td>
    <td style="text-align: center; padding: 8px; color: red;">5->8</td>
    <td style="text-align: center; padding: 8px; color: red;">5->13</td>
  </tr>
  <tr>
    <td style="text-align: left; padding: 8px;">deepseek-coder-v2-0724</td>
    <td style="text-align: center; padding: 8px; color: green;">16->14</td>
    <td style="text-align: center; padding: 8px; color: green;">16->13</td>
    <td style="text-align: center; padding: 8px; color: green;">16->14</td>
  </tr>
</table>


## Future Work

We want to continue building a pipeline to disentangle style and substance in the arena. Although controlling for style is a big step forward, our analysis is still _observational_. We are looking forward to implementing _causal inference_ in our pipeline, and running prospective randomized trials to assess the effect of length, markdown, and more. Stay tuned, and let us know if you want to help!

## Citation
```
@misc{chiang2024chatbot,
    title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
    author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
    year={2024},
    eprint={2403.04132},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```