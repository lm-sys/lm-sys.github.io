---
title: "Chatbot Arena: New models & Elo system update"
author: "Wei-Lin Chiang, Tim Li, Joseph E. Gonzalez, Ion Stoica"
date: "Dec 7, 2023"
previewImg: /images/blog/slora/thumbnail_preview.png
---

Welcome to our latest update on the Chatbot Arena, our open evaluation platform to test the most advanced LLMs. We're excited to share that over 130,000 votes that are now collected to rank the most capable 40+ models! In this blog post, we'll cover the results of six new models, the transition from the online Elo system to the Bradley-Terry model, which gives us significantly more stable ratings and precise confidence intervals, and our findings from differentiating versions of proprietary models (e.g., GPT-4 => GPT-4-0314, GPT-4-0613).

Let’s dive into it!

## Introducing new models

LLM has become smarter than ever and it’s been a real challenge to evaluate them properly. Traditional benchmarks such as MMLU have been useful, but they may fall short in capturing the nuance of human preference and open-ended nature of real-world conversations. We believe deploying chat models in the real-world to get feedback from users produces the most direct signals. This led to the Chatbot Arena launch in May. Since then, the open-source community has taken off. Over the past few months, we have deployed more than 40 models in Arena and we’ve collected over 130,000 valid votes from our users. We believe such a scale covers a diverse range of use cases which bring us useful insights to understand how these models work in real-world scenarios.

In November, we added record-breaking nine new models with sizes ranging from 7B to 70B, as well as proprietary ones, and gathered over 10,000 votes for them. Excitingly, we are now seeing the gap between gpt-3.5 and the most capable open models narrowing. New models such as Tulu-2-DPO-70B by UW/AllenAI and Yi-34B-Chat	by 01.ai have been leading the open space, delivering close to gpt-3.5 performance.


| Model | Arena Elo Rating | Vote count | License |
|:---|---:|---:|---:|
| [**GPT-4-Turbo**](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) | 1217 | 7007 | Proprietary |
| [GPT-4-0613](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) | 1153 | 11944 | Proprietary |
| [**Claude-2.1**](https://www.anthropic.com/index/claude-2-1) | 1118 | 5929 | Proprietary | 
| [GPT-3.5-Turbo-0613](https://platform.openai.com/docs/models/gpt-3-5) | 1112 | 15974 | Proprietary |
| [Claude-instant-1](https://www.anthropic.com/index/releasing-claude-instant-1-2) | 1108 | 5929 | Proprietary | 
| [**Tulu-2-DPO-70B**](https://huggingface.co/allenai/tulu-2-dpo-70b) | 1105 | 2922 | AI2 ImpACT Low-risk |
| [**Yi-34B-Chat**](https://huggingface.co/01-ai/Yi-34B-Chat) | 1102 | 3123 | Yi License |
| [Wizardlm-70B](https://huggingface.co/WizardLM/WizardLM-70B-V1.0) | 1096 | 5865 | Llama 2 Community |
| [Vicuna-33B](https://huggingface.co/lmsys/vicuna-33b-v1.3) | 1093 | 11671 | Non-commercial |
| [**Starling-LM-7B-alpha**](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) | 1083 | 2250 | CC-BY-NC-4.0 |
| [**PPLX-70B-Online**](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) | 1080 | 1500 | Proprietary |
| [**OpenChat-3.5**](https://huggingface.co/openchat/openchat_3.5) | 1077 | 4662 | Apache-2.0 |
| [**Openhermes-2.5-mistral-7B**](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) | 1075 | 1180 | Apache-2.0 |
| [Llama-2-70B-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 1069 | 8659 | Llama 2 Community |
| [Zephyr-7B-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 1045 | 8412 | MIT |
| [**PPLX-7B-Online**](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) | 1016 | 1041 | Proprietary |

On the other hand, 7B models have also shown significant improvements. Fine-tuning the 7B Mistral model has led to Zephyr, OpenChat-3.5, Starling-lm-7b-alpha, and OpenHermes-2.5-Mistral-7b which all demonstrate great chat performance. Shoutout to the open-source community which has been pushing limits. To understand how freshness and grounded information help LLMs in answering user queries, we also bring Perplexity AI’s online LLMs to Arena. We have collected over 1500 votes for PPLX-70B-Online and the preliminary results show great potential.

<img src="/images/blog/leaderboard_202312/mle_elo.png" style="display:block; margin:auto; max-width:80%; height:auto;"></img>


### Topic modeling on user prompts

We've also conducted topic modeling on 50,000 user prompts to better understand how users interact with these models. Our approach utilized OpenAI embeddings `text-embedding-ada-002` and K-means clustering, followed by GPT-4 to summarize the topics for each cluster, provided with the prompts close to the center. This analysis revealed a wide range of topics, from role-playing, story writing to programming advice. We show a few examples below.

<img src="/images/blog/leaderboard_202312/topic_distribution_bar.png" style="display:block; margin:auto; max-width:80%; height:auto;">

<style>
.foo table th:first-of-type {
    width: 10%;
}
.foo table th:nth-of-type(2) {
    width: 90%;
}
</style>

<div class="foo">

| Cluster ID | Arena User Prompt |
|---|:---|
| 1 | You are a Chief information Officer for a Biotechnology Manufacturing company and will act like one. Write a business need and objectives for a case study to Engage Info-Tech technical consulting services to conduct a comprehensive assessment of our current application development practices, including analyzing our development methodologies, tools, and frameworks. |
| 2  | Write a short scene from a novel where a beautiful, wicked lamia coils around an unfortunate, quippy human adventurer. |
| 3 | How should the balance be struck between freedom of speech and the ability to function in a world without continual distractions and distortions from misinformation? |
| 4 | Can you give me a list of 5 suggestions on how to write software with fewer bugs? |

</div>

 Moving forward, we aim to refine our methods to filter out low-quality prompts and improve categorization for a clearer understanding of model strengths and weaknesses in different areas.

## Transition from online Elo rating system to Bradley-Terry model

We adopted the Elo rating system for ranking models since the launch of the Arena. It has been useful to transform pairwise human preference to Elo ratings that serve as a predictor of winrate between models. Specifically, if player A has a rating of $R_A$ and player B a rating of $R_B$, the probability of player A winning is

<img src=" https://wikimedia.org/api/rest_v1/media/math/render/svg/7c80282e9c95e92d6b210467aab48a8c4c81ef10" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


ELO rating has been used to rank chess players by the international community for over 60 years. Standard Elo rating systems assume a player’s performance changes overtime. So an online algorithm is needed to capture such dynamics, meaning recent games should weigh more than older games. Specifically, after each game, a player's rating is updated according to the difference between predicted outcome and actual outcome.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1cad9fb1cfc6a8e845493ac9a40eb98541a4641a" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

This algorithm has two distinct features.

1. It can be computed asynchronously by players around the world.
2. It allows for players performance to change dynamically – it does not assume a fixed unknown value for the players rating.

This ability to adapt is determined by the parameter K which controls the magnitude of rating changes that can affect the overall result. A larger K essentially put more weight on the recent games, which may make sense for new players whose performance improves quickly. However as players become more senior and their performance “converges” then a smaller value of K is more appropriate. As a result, USCF adopted K based on the number of games and tournaments completed by the player ([reference](https://new.uschess.org/sites/default/files/media/documents/the-us-chess-rating-system-revised-september-2020.pdf)). That is, the Elo rating of a senior player changes slower than a new player. 

When we launched the Arena, we noticed considerable variability in the ratings using the classic online algorithm. We tried to tune the K to be sufficiently stable while also allowing new models to move up quickly in the leaderboard.  We ultimately decided to adopt a bootstrap-like technique to shuffle the data and sample Elo scores from 1000 permutations of the online plays. You can find the details in this [notebook](https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR?usp=sharing).  This provided consistent stable scores and allowed us to incorporate new models quickly.  However, we used the same samples to estimate confidence intervals which were therefore too wide (effectively CI’s for the original online Elo estimates).

In the context of LLM ranking, there are two important differences from the classic Elo chess ranking system.  First, we have access to the entire history of all games for all models and so we don’t need a decentralized algorithm.  Second, most models are static (we have access to the weights) and so we don’t expect their performance to change. However, it is worth noting that the hosted proprietary models may not be static and their behavior can change without notice. We try our best to pin specific model API versions if possible.

To improve the quality of our rankings and their confidence estimates, we are adopting another widely used rating system called the [Bradley–Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) (BT) model.  This model actually is the maximum likelihood (MLE) estimate of the underlying Elo model assuming a fixed but unknown pairwise win-rate.  Similar to Elo rating, BT model is also based on pairwise comparison to derive ratings of players to estimate win rate between each other. The core difference between BT model vs the online Elo system is the assumption that player's performance does not change (i.e., game order does not matter) and the computation takes place in a centralized fashion. 

With the static performance assumption, the model ratings can be obtained by maximum likelihood estimation (MLE), i.e. maximizing the likelihood of the observed game outcomes given the model ratings. Code snippet below shows how to use MLE to compute the model ratings. Detailed code can be found [here](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=PbTdhkLQp113&line=2&uniqifier=1).

<img src="/images/blog/leaderboard_202312/mle_code.png" style="display:block; margin:auto; max-width:80%; height:auto;">


Similarly, we can also bootstrap the MLE Bradley-Terry scores to obtain the confidence intervals of model ratings. We observe that the mean rating by both methods are very similar and the rankings are almost the same. 

<img src="/images/blog/leaderboard_202312/elo_vs_bt.png" style="display:block; margin:auto; max-width:60%; height:auto;">

More importantly, with the BT model, the bootstrap confidence intervals now better capture the variance of the model performance estimates. We observe clear improvement in the below figures. Newly added models with fewer votes have a wider range of confidence intervals than others.

| Bootstraping Online Elo  | Bootstraping MLE Elo (BT model) |
|---|---|
| <img src="/images/blog/leaderboard_202312/online_elo.png" style="display:block; margin:auto; height:auto;"> | <img src="/images/blog/leaderboard_202312/mle_elo.png" style="display:block; margin:auto; height:auto;"> |

Code to reproduce the calculation can be found at this [notebook](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=c0KvFVr-nR2Q).


## Tracking Performance of Proprietary APIs - GPT-4-0314 vs 0613

Since OpenAI’s GPT-4 update in June, the community has been wondering whether there's a performance change on the newer version of GPT-4. Some people find performance drop in certain domains ([reference](https://x.com/matei_zaharia/status/1681467961905926144?s=20)), but it’s still unclear what's really going on. Previously we combined votes of the two versions into just GPT-4. As we transition from online Elo to the BT model, we decide to separate out different versions of proprietary model APIs to better satisfy its assumptions on model staying static.

<img src="/images/blog/leaderboard_202312/gpt_version.png" style="display:block; margin:auto; max-width:90%; height:auto;">

Surprisingly, we observe a significant difference between `gpt-4-0314` and `gpt-4-0613` based Arena user preference (Rating 1201 vs 1152). The GPT-4 API was automatically updated from 0314 to 0613 on June 27 and the 0314 version has since then been retired from Arena. Potential hypotheses:

1. Arena user distribution has shifted before/after July (e.g., prompt distribution, voting behaviors etc)
2. No comparison data for 0314 vs newly added models after July may be unfair.
3. There is indeed a regression in the newer version of GPT-4.

To address this problem, we have brought up `gpt-4-0314` online again to collect new votes, also directly comparing it against its newer 0613 version. At the time of writing we have collected 1,000 new votes for `gpt-4-0314` and its performance is still robust. We’ll give more updates after more investigation.

Interestingly, gpt-3.5-turbo, which has been through a similar version change (0314 -> 0613), seems to be normal. As you can see, `gpt-3.5-turbo-0613` has slightly higher rating than `gpt-3.5-turbo-0314` (1112 vs 1106). However, we again observe a strange performance drop of the latest version `gpt-3.5-turbo-1106` which has obtained over 5,000 votes. We hope to investigate this deeper by developing new tools to analyze user prompts and identify model strengths and weaknesses in different areas.

## Next steps

We plan to ship real-time leaderboard update, diving deeper into user prompt analysis, and enhancing prompt moderation and categorization. Stay tuned for more insights as we continue to refine our approach to evaluating the evolving landscape of LLMs. Thanks for joining us on this journey, and we look forward to sharing more updates soon!


## Links
- [Chatbot Arena Demo](https://chat.lmsys.org/)
- [Arena Elo Colab](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mukqgshMarFi)

If you wish to see more models on Arena leaderboard, we invite you to [contribute to FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model) or [contact us](mailto:lmsysorg@gmail.com) to provide us with API access.
