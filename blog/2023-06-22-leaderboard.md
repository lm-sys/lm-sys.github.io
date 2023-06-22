---
title: "Chatbot Arena Leaderboard Week 8: Introducing MT-Bench and Vicuna-33B"
author: "Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Hao Zhang"
date: "June 22, 2023"
previewImg: /images/blog/leaderboard_week8/ability_breakdown.png
---

In this blog post, we share the latest update on Chatbot Arena leaderboard, which now includes more open models and three metrics:

1. **Chatbot Arena Elo**, based on 42K anonymous votes from [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) using the Elo rating system.
2. **MT-Bench score**, based on a challenging multi-turn benchmark and GPT-4 grading, proposed and validated in our [Judging LLM-as-a-judge paper](https://arxiv.org/abs/2306.05685).
3. **MMLU**, a widely adopted [benchmark](https://arxiv.org/abs/2009.03300).

Furthermore, we’re excited to introduce our **new series of Vicuna-v1.3 models**, ranging from 7B to 33B parameters, trained on an extended set of user-shared conversations. 
Their weights are now [available](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights).

## Updated Leaderboard and New Models

<style>
th {text-align: left}
td {text-align: left}
</style>

<br>
<p style="color:gray; text-align: center;">Table 1. LLM Leaderboard (Timeframe: April 24 - June 22, 2023). More details at <a href="https://chat.lmsys.org/?leaderboard" target="_blank">our Leaderboard</a>.</p>
<table style="display: flex; justify-content: center;" align="left" >
<tbody>
<tr> <th>Model</th> <th>MT-bench (score)</span> </th> <th>Elo Rating</span> </th> <th>MMLU</th> <th>License</th> </tr>

<tr> <td><a href="https://chat.openai.com/?model=gpt-4" target="_blank">GPT-4</a></td> <td>8.99</td> <td>1227</td> <td>86.4</td>  <td>Proprietary</td> </tr>

<tr> <td><a href="https://chat.openai.com/" target="_blank">GPT-3.5-turbo</a></td> <td>7.94</td> <td>1130</td>  <td>70</td> <td>Proprietary</td> </tr>

<tr> <td><a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-v1</a></td> <td>7.9</td> <td>1178</td> <td>75.6</td>  <td>Proprietary</td> </tr>

<tr> <td><a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-instant-v1</a></td> <td>7.85</td> <td>1156</td> <td>61.3</td> <td>Proprietary</td> </tr>

<tr> <td><a href="https://github.com/lm-sys/FastChat/tree/main#vicuna-weights" target="_blank">Vicuna-33B</a></td> <td>7.12</td> <td>-</td> <td>59.2</td>  <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/WizardLM/WizardLM-30B-V1.0" target="_blank">WizardLM-30B</a></td> <td>7.01</td> <td>-</td> <td>58.7</td>  <td>Non-commercial</td></tr>

<tr> <td><a href="https://huggingface.co/timdettmers/guanaco-33b-merged" target="_blank">Guanaco-33B</a></td> <td>6.53</td> <td>1065</td> <td>57.6</td> <td>Non-commercial</td></tr>

<tr> <td><a href="https://huggingface.co/allenai/tulu-30b" target="_blank">Tulu-30B</a></td> <td>6.43</td> <td>-</td> <td>58.1</td> <td>Non-commercial</td></tr>

<tr> <td><a href="https://huggingface.co/timdettmers/guanaco-65b" target="_blank">Guanaco-65B</a></td> <td>6.41</td> <td>-</td> <td>62.1</td> <td>Non-commercial</td></tr>

<tr> <td><a href="https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor" target="_blank">OpenAssistant-LLaMA-30B</a></td> <td>6.41</td> <td>-</td> <td>55.9</td> <td>Non-commercial</td></tr>

<tr><td><a href="https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023" target="_blank">PaLM2-Chat-Bison-001</a></td> <td>6.4</td> <td>1038</td> <td>-</td> <td>Proprietary</td> </tr>

<tr> <td><a href="https://lmsys.org/blog/2023-03-30-vicuna/" target="_blank">Vicuna-13B</a></td> <td>6.39</td>  <td>1061</td> <td>52.1</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.0" target="_blank">WizardLM-13B</a></td> <td>6.35</td>  <td>1048</td> <td>52.3</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://github.com/lm-sys/FastChat/tree/main#vicuna-weights" target="_blank">Vicuna-7B</a></td> <td>6</td> <td>1008</td> <td>47.1</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/project-baize/baize-v2-13b" target="_blank">Baize-v2-13B</a></td> <td>5.75</td> <td>-</td> <td>48.9</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/NousResearch/Nous-Hermes-13b" target="_blank">Nous-Hermes-13B</a></td> <td>5.51</td> <td>-</td> <td>49.3</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/mosaicml/mpt-7b-chat" target="_blank">MPT-7B-Chat</a></td> <td>5.42</td>  <td>956</td> <td>32</td> <td>CC-By-NC-SA-4.0</td> </tr>

<tr> <td><a href="https://huggingface.co/nomic-ai/gpt4all-13b-snoozy" target="_blank">GPT4All-13B-Snoozy</a></td> <td>5.41</td> <td>986</td> <td>43</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://bair.berkeley.edu/blog/2023/04/03/koala" target="_blank">Koala-13B</a></td> <td>5.35</td> <td>992</td>  <td>44.7</td>  <td>Non-commercial</td> </tr>

<tr> <td><a href="https://huggingface.co/tiiuae/falcon-40b-instruct" target="_blank">Falcon-40B-Instruct</a></td> <td>5.17</td> <td>-</td>  <td>54.7</td>  <td>Apache 2.0</td> </tr>

<tr><td><a href="https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b" target="_blank">H2O-Oasst-OpenLLaMA-13B</a></td> <td>4.63</td> <td>-</td> <td>42.8</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://crfm.stanford.edu/2023/03/13/alpaca.html" target="_blank">Alpaca-13B</a></td> <td>4.53</td> <td>930</td> <td>48.1</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://chatglm.cn/blog" target="_blank">ChatGLM-6B</a></td> <td>4.5</td> <td>905</td> <td>36.1</td> <td>Non-commercial</td> </tr>

<tr> <td><a href="https://open-assistant.io" target="_blank">Oasst-Pythia-12B</a></td> <td>4.32</td> <td>924</td> <td>27</td> <td>Apache 2.0</td> </tr>

<tr> <td><a href="https://huggingface.co/BlinkDL/rwkv-4-raven" target="_blank">RWKV-4-Raven-14B</a></td> <td>3.98</td> <td>950</td> <td>25.6</td> <td>Apache 2.0</td> </tr>

<tr> <td><a href="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm" target="_blank">Dolly-V2-12B</a></td> <td>3.28</td> <td>850</td>  <td>25.7</td>  <td>MIT</td> </tr>

<tr> <td><a href="https://huggingface.co/lmsys/fastchat-t5-3b-v1.0" target="_blank">FastChat-T5-3B</a></td> <td>3.04</td>  <td>897</td>  <td>47.7</td>  <td>Apache 2.0</td> </tr>

<tr> <td><a href="https://arxiv.org/abs/2302.13971" target="_blank">LLaMA-13B</a></td> <td>2.61</td>  <td>826</td> <td>47</td> <td>Non-commercial</td> </tr>

</tbody>
</table>

&shy;

You are welcome to check out the latest [leaderboard](https://chat.lmsys.org/?leaderboard) or try the arena [demo](https://chat.lmsys.org/?arena). 
Keep in mind that each benchmark has its limitations. Please consider the results as guiding references. See our discussion below for more technical details.


## Evaluating Chatbots with MT-bench and Arena

### Motivation

While several benchmarks exist for evaluating Large Language Model's (LLM) performance, such as [MMLU](https://arxiv.org/abs/2009.03300), [HellaSwag](https://arxiv.org/abs/1905.07830), and [HumanEval](https://github.com/openai/human-eval), 
we noticed that these metrics fall short when assessing their levels of human preferences. Traditional benchmarks often test LLMs on close-ended questions with concise outputs (e.g., multiple choices), which do not reflect the typical use cases of LLM-based chat assistants.

To fill this gap, we have run the Chatbot Arena for 2 months, and in this blogpost, we add a new benchmark: MT-Bench.
- [MT-bench](https://arxiv.org/abs/2306.05685) is a challenging multi-turn question set designed to evaluate the conversational and instruction-following ability of models. You can view sample questions and answers of MT-bench [here](https://huggingface.co/spaces/lmsys/mt-bench).
- [Chatbot Arena](https://chat.lmsys.org/?arena) is a crowd-sourced battle platform, where users ask chatbots any question and vote their preferred answer. 
Both benchmarks are designed to use human preference as the primary metric.

### Why MT-Bench?

MT-Bench is a carefully curated benchmark that includes 80 high-quality, multi-turn questions. 
These questions are tailored to assess the conversation flow and instruction-following capabilities of models in multi-turn dialogues. 
They include both common use cases as well as more challenging instructions meant to distinguish between chatbots. 
MT-Bench serves as a quality-controlled complement to our crowd-sourced based eval -- Chatbot Arena.

Through running the Chatbot Arena for 2 months and analyzing our users' prompts, we've identified 8 primary categories of user prompts: Writing, Roleplay, Extraction, Reasoning, Math, Coding, Knowledge I (STEM), and Knowledge II (humanities/social science). 
We crafted 10 multi-turn questions per category, yielding a set of 160 questions in total. We display some sample questions below. You can find more [here](https://huggingface.co/spaces/lmsys/mt-bench).

<img src="/images/blog/leaderboard_week8/sample_question.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 1000px;"></img>
<p style="color:gray; text-align: center;">Figure 1: Sample questions from the MT-Bench.</p>

### But still, How to Grade Chatbot’s Answers?

Though human preference remains the gold standard, it is notoriously slow and expensive to collect. 
In our first [Vicuna blog post](https://lmsys.org/blog/2023-03-30-vicuna/), we have explored an automated evaluation pipeline based on GPT-4. 
This approach has since got popular and adopted in several [concurrent and follow-up works](http://localhost:3000/blog/2023-06-22-leaderboard/#related-work).

In our latest [paper](https://arxiv.org/abs/2306.05685), "Judging LLM-as-a-judge", we conducted a systematic study to answer how reliable those LLM judges are. 
We provide a brief overview of conclusions here but recommend reading the paper for more details.

We begin by acknowledging potential limitations of LLM-as-a-judge:

- **Position bias** where LLM judges may favor the first answer in a pairwise comparison.
- **Verbosity bias** where LLM judges may favor lengthier answers, regardless of their quality.
- **Self-enhancement bias** where LLM judges may favor their own responses.
- **Limited reasoning ability** referring to LLM judges' possible shortcomings in grading math and reasoning questions.

Our study then explores how few-shot judge, chain-of-thought judge, reference-based judge, and fine-tuned judge can help to mitigate these limitations.

Upon implementing some of these solutions, we discovered that despite its limitations, strong LLM judges like GPT-4 can align impressively well with both controlled and crowdsourced human preferences, achieving over 80% agreement. 
This level of agreement is comparable to the agreement between two different human judges. 
Therefore, if used carefully, LLM-as-a-judge can act as a *scalable* and *explainable* approximation of human preferences.

We also found that single-answer grading based on GPT-4, without pairwise comparison, can also rank models effectively and match human preferences well. 
Single-answer grading proves more scalable than pairwise comparison as the number of possible pairs increases quadratically with the number of models. 
In Table 1, we present the MT-Bench leaderboard based on single-answer grading with GPT-4.


## Results and Analysis

### MT-Bench Effectively Distinguishes Among Chatbots

Table 1 provides a detailed rundown of the MT-bench leaderboard, where we conduct an exhaustive evaluation of 28 popular instruction-tuned models. 
We observe a clear distinction among chatbots of varying abilities using our benchmark, with scores showing a high correlation with the Elo rating on Arena. 
In particular, MT-Bench reveals noticeable performance gaps between GPT-4 and GPT-3.5/Claude, and between open and proprietary models.

To delve deeper into the distinguishing factors among chatbots, we select a few representative chatbots and break down their performance per category in Figure 2. 
GPT-4 shows superior performance in Coding and Reasoning compared to GPT-3.5/Claude, while Vicuna-13b lags significantly behind in Extraction, Coding, and Math. 
This suggests there is still ample room for improvement for open-source models.

<img src="/images/blog/leaderboard_week8/ability_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 1000px;"></img>
<p style="color:gray; text-align: center;">Figure 2: The comparison of 6 representative LLMs regarding their abilities in 8 categories: Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities.</p>


### Multi-turn Conversation Capabilities

We next analyze the multi-turn scores of selected models, presented in Table 2. 
The MT-bench incorporates challenging follow-up questions as part of its design. 
For open models, a significant drop in performance is observable from the first to second turn (e.g., vicuna-7b, wizardlm-13b), while strong proprietary models maintain consistency. 
We also notice a considerable performance gap between llama-based models and those with permissive licenses such as mpt-7b, falcon-40b, and instruction-tuned open-llama.


<br>
<p style="color:gray; text-align: center;">Table 2. The breakdown of LLMs' MT-bench scores in the 1st and 2nd turn of a dialogue. Full score is 10.</p>
<table style="display: flex; justify-content: center;" align="left" >
<tbody>
<tr> <th>Model</th> <th>Average 1st Turn Score</th> <th>Average. 2nd Turn Score</th> <th>Score Difference</th>

<tr><td><a href="https://chat.openai.com/" target="_blank">GPT-4</a></td> <td>8.96</td> <td>9.03</td> <td>0.07</td>  </tr>

<tr><td><a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-v1</a></td> <td>8.15</td> <td>7.65</td> <td>-0.50</td> </tr>

<tr><td><a href="https://chat.openai.com/" target="_blank">GPT-3.5-turbo</a></td> <td>8.08</td> <td>7.81</td> <td>-0.26</td> </tr>

<tr><td><a href="https://huggingface.co/lmsys/vicuna-33b-v1.3" target="_blank">vicuna-33b</a></td> <td>7.46</td> <td>6.79</td> <td>-0.67</td> </tr>

<tr><td><a href="https://huggingface.co/WizardLM/WizardLM-30B-V1.0" target="_blank">wizardlm-30b</a></td> <td>7.13</td> <td>6.89</td> <td>-0.24</td> </tr>

<tr><td><a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.0" target="_blank">wizardlm-13b</a></td> <td>7.12</td> <td>5.59</td> <td>-1.53</td> </tr>

<tr><td><a href="https://huggingface.co/timdettmers/guanaco-33b-merged" target="_blank">guanaco-33b</a></td> <td>6.88</td> <td>6.18</td> <td>-0.71</td> </tr>

<tr><td><a href="https://huggingface.co/lmsys/vicuna-13b-v1.3" target="_blank">vicuna-13b</a></td> <td>6.81</td> <td>5.96</td> <td>-0.85</td> </tr>

<tr><td><a href="https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023" target="_blank">palm2-chat-bison</a></td> <td>6.71</td> <td>6.09</td> <td>-0.63</td> </tr>

<tr><td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.3" target="_blank">vicuna-7b</a></td> <td>6.69</td> <td>5.30</td> <td>-1.39</td> </tr>

<tr><td><a href="https://huggingface.co/young-geng/koala" target="_blank">koala-13b</a></td> <td>6.08</td> <td>4.63</td> <td>-1.45</td> </tr>

<tr><td><a href="https://huggingface.co/mosaicml/mpt-7b-chat" target="_blank">mpt-7b-chat</a></td> <td>5.85</td> <td>4.99</td> <td>-0.86</td> </tr>

<tr><td><a href="https://huggingface.co/tiiuae/falcon-40b-instruct" target="_blank">falcon-40b-instruct</a></td> <td>5.81</td> <td>4.53</td> <td>-1.29</td> </tr>

<tr><td><a href="https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b" target="_blank">h2ogpt-oasst-open-llama-13b</a></td> <td>5.51</td> <td>3.74</td> <td>-1.78</td> </tr>
</tbody>
</table>

&shy;
<br>


### Explainability in LLM judges 

Another advantage of LLM judges is their ability to provide explainable evaluations. 
Figure 3 presents an instance of GPT-4's judgment on an MT-bench question, with answers from alpaca-13b and gpt-3.5-turbo. 
GPT-4 provides thorough and logical feedback to support its judgment. 
Our [study](https://arxiv.org/abs/2306.05685) found that such reviews are beneficial in guiding humans to make better-informed decisions (refer to Section 4.2 for more details). 
All the GPT-4 judgments can be found on our demo site ??.

<img src="/images/blog/leaderboard_week8/explainability_sample.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 1000px;"></img>
<p style="color:gray; text-align: center;">Figure 3: MT-bench provides more explainability in evaluating LLMs' human preferences.</p>

In conclusion, we have shown that MT-Bench effectively differentiates between chatbots of varying capabilities. 
It's scalable, offers valuable insights with category breakdowns, and provides explainability for human judges to verify. 
However, LLM judges should be used carefully. It can still make errors, especially when grading math/reasoning questions. 
For further details, please refer to Section 3.3 of our study.


## How to Evaluate New Models on MT-Bench?

Evaluating models on MT-bench is simple and fast. Our script supports all huggingface models, and we’ve provided [detailed instructions](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#mt-bench), 
in which you can generate model’s answers to the MT-bench questions and their GPT-4 judgments. You can also examine the answers and reviews on our gradio browsing demo.

## Next steps
**Release of Conversation Data**
We're in the process of preparing all Chatbot Arena conversation data for release to the broader research community. 
We're currently addressing critical aspects such as personally identifiable information (PII) cleaning, tagging for toxicity, and performing an ethics review. Stay tuned for updates!

**MT-bench-1K**
MT-Bench currently consists of a concise set of 80 carefully curated questions, ensuring the highest quality. 
We're actively working on expanding the question set to MT-Bench-1K. 
We're integrating high-quality prompts from the Chatbot Arena and generating new ones automatically using LLMs. 
If you have any good ideas, we'd be delighted to hear from you.

**Invitation for collaborations**
We're actively engaging with various organizations to explore possibilities for standardizing the evaluation of human preferences for LLMs on a large scale. 
If this interests you or aligns with your research, please feel free to reach out to us.

## Related work
There has been a great amount of interesting work studying how to evaluate human preferences and how to use strong LLM as judges for evaluation. 
You are welcome to check them out and see more opinions on this topic:
- [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Can foundation models label data like humans?](https://huggingface.co/blog/llm-leaderboard)
- [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751)
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)
- [AlpacaEval and AlpacaFarm](https://github.com/tatsu-lab/alpaca_eval)
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926) 

## Links
Below are readily available tools and code to run MT-bench and other metrics used in this blogpost:
- The MT-bench uses [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge),
- The [Arena Elo calculator](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/elo_analysis.py),
- The MMLU is based on [InstructEval](https://github.com/declare-lab/instruct-eval/blob/main/mmlu.py) and [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub/tree/main/MMLU).

If you wish to see more models on leaderboard, we invite you to [contribute](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model) or [contact us](mailto:lmsysorg@gmail.com) to provide us with API access.
