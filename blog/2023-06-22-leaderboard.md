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
</style>


<script>
    let sortOrder = ['desc', undefined, undefined];

    function sortTable(columnIndex, table_id) {
      let table, rows, switching, i, x, y, shouldSwitch;
      table = document.getElementById(table_id);
      switching = true;
      let sortAsc = sortOrder[columnIndex] === 'asc';

      while (switching) {
        switching = false;
        rows = table.getElementsByTagName("tr");

        for (i = 1; i < (rows.length - 1); i++) {
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("td")[columnIndex];
          y = rows[i + 1].getElementsByTagName("td")[columnIndex];
          x_char = x.innerHTML.toLowerCase();
          y_char = y.innerHTML.toLowerCase();
          if (sortAsc) {
            if (x_char === "-") {
                x_val = 9999;
            } else {
                x_val = Number(x_char);
            }
            if (y_char === "-") {
                y_val = 9999;
            } else {
                y_val = Number(y_char);
            }
            if (x_val > y_val) {
              shouldSwitch = true;
              break;
            }
          } else {
            if (x_char === "-") {
                x_val = 0.0;
            } else {
                x_val = Number(x_char);
            }
            if (y_char === "-") {
                y_val = 0.0;
            } else {
                y_val = Number(y_char);
            }

            if (x_val < y_val) {
              shouldSwitch = true;
              break;
            }
          }
        }

        if (shouldSwitch) {
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
        }
      }

      let arrowElements = document.getElementsByClassName("arrow");
      for (let j = 0; j < arrowElements.length; j++) {
        arrowElements[j].classList.remove("arrow-up", "arrow-down");
      }

      let arrowElement = document.getElementsByTagName("th")[columnIndex].getElementsByClassName("arrow")[0];
      arrowElement.classList.add(sortAsc ? "arrow-up" : "arrow-down");
      sortOrder[columnIndex] = sortAsc ? 'desc' : 'asc';
    }
</script>



<br>
<p style="color:gray; text-align: center;">Table 1. LLM Leaderboard (Timeframe: April 24 - June 19, 2023). The latest and detailed version <a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard" target="_blank">here</a>.</p>
<div style="display: flex; justify-content: center;">
<table id="Table1" >
<tbody>

<tr> <th>Model</th> <th onclick="sortTable(1, 'Table1')">MT-bench (score) <span class="arrow arrow-down"></span></th> <th onclick="sortTable(2, 'Table1')">Arena Elo Rating <span class="arrow"></span></th> <th onclick="sortTable(3, 'Table1')">MMLU <span class="arrow"></span></th> <th>License</th> </tr>

<tr> <td><a target="_blank" href="https://openai.com/research/gpt-4"> GPT-4 </a></td>  <td>8.99</td>  <td>1227</td>  <td>86.4</td>  <td>Proprietary</td> </tr>
<tr> <td><a target="_blank" href="https://openai.com/blog/chatgpt"> GPT-3.5-turbo </a></td>  <td>7.94</td>  <td>1130</td>  <td>70.0</td>  <td>Proprietary</td> </tr>
<tr> <td><a target="_blank" href="https://www.anthropic.com/index/introducing-claude"> Claude-v1 </a></td>  <td>7.90</td>  <td>1178</td>  <td>75.6</td>  <td>Proprietary</td> </tr>
<tr> <td><a target="_blank" href="https://www.anthropic.com/index/introducing-claude"> Claude-instant-v1 </a></td>  <td>7.85</td>  <td>1156</td>  <td>61.3</td>  <td>Proprietary</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/lmsys/vicuna-33b-v1.3"> Vicuna-33B </a></td>  <td>7.12</td>  <td>-</td>  <td>59.2</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/WizardLM/WizardLM-30B-V1.0"> WizardLM-30B </a></td>  <td>7.01</td>  <td>-</td>  <td>58.7</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/timdettmers/guanaco-33b-merged"> Guanaco-33B </a></td>  <td>6.53</td>  <td>1065</td>  <td>57.6</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/allenai/tulu-30b"> Tulu-30B </a></td>  <td>6.43</td>  <td>-</td>  <td>58.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/timdettmers/guanaco-65b-merged"> Guanaco-65B </a></td>  <td>6.41</td>  <td>-</td>  <td>62.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor"> OpenAssistant-LLaMA-30B </a></td>  <td>6.41</td>  <td>-</td>  <td>56.0</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models"> PaLM-Chat-Bison-001 </a></td>  <td>6.40</td>  <td>1038</td>  <td>-</td>  <td>Proprietary</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/lmsys/vicuna-13b-v1.3"> Vicuna-13B </a></td>  <td>6.39</td>  <td>1061</td>  <td>52.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/mosaicml/mpt-30b-chat"> MPT-30B-chat </a></td>  <td>6.39</td>  <td>-</td>  <td>50.4</td>  <td>CC-BY-NC-SA-4.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/WizardLM/WizardLM-13B-V1.0"> WizardLM-13B </a></td>  <td>6.35</td>  <td>1048</td>  <td>52.3</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/lmsys/vicuna-7b-v1.3"> Vicuna-7B </a></td>  <td>6.00</td>  <td>1008</td>  <td>47.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/project-baize/baize-v2-13b"> Baize-v2-13B </a></td>  <td>5.75</td>  <td>-</td>  <td>48.9</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/NousResearch/Nous-Hermes-13b"> Nous-Hermes-13B </a></td>  <td>5.51</td>  <td>-</td>  <td>49.3</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/mosaicml/mpt-7b-chat"> MPT-7B-Chat </a></td>  <td>5.42</td>  <td>956</td>  <td>32.0</td>  <td>CC-BY-NC-SA-4.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/nomic-ai/gpt4all-13b-snoozy"> GPT4All-13B-Snoozy </a></td>  <td>5.41</td>  <td>986</td>  <td>43.0</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://bair.berkeley.edu/blog/2023/04/03/koala/"> Koala-13B </a></td>  <td>5.35</td>  <td>992</td>  <td>44.7</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/mosaicml/mpt-30b-instruct"> MPT-30B-Instruct </a></td>  <td>5.22</td>  <td>-</td>  <td>47.8</td>  <td>CC-BY-SA 3.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/tiiuae/falcon-40b-instruct"> Falcon-40B-Instruct </a></td>  <td>5.17</td>  <td>-</td>  <td>54.7</td>  <td>Apache 2.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b"> H2O-Oasst-OpenLLaMA-13B </a></td>  <td>4.63</td>  <td>-</td>  <td>42.8</td>  <td>Apache 2.0</td> </tr>
<tr> <td><a target="_blank" href="https://crfm.stanford.edu/2023/03/13/alpaca.html"> Alpaca-13B </a></td>  <td>4.53</td>  <td>930</td>  <td>48.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/THUDM/chatglm-6b"> ChatGLM-6B </a></td>  <td>4.50</td>  <td>905</td>  <td>36.1</td>  <td>Non-commercial</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"> OpenAssistant-Pythia-12B </a></td>  <td>4.32</td>  <td>924</td>  <td>27.0</td>  <td>Apache 2.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/BlinkDL/rwkv-4-raven"> RWKV-4-Raven-14B </a></td>  <td>3.98</td>  <td>950</td>  <td>25.6</td>  <td>Apache 2.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/databricks/dolly-v2-12b"> Dolly-V2-12B </a></td>  <td>3.28</td>  <td>850</td>  <td>25.7</td>  <td>MIT</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/lmsys/fastchat-t5-3b-v1.0"> FastChat-T5-3B </a></td>  <td>3.04</td>  <td>897</td>  <td>47.7</td>  <td>Apache 2.0</td> </tr>
<tr> <td><a target="_blank" href="https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b"> StableLM-Tuned-Alpha-7B </a></td>  <td>2.75</td>  <td>871</td>  <td>24.4</td>  <td>CC-BY-NC-SA-4.0</td> </tr>
<tr> <td><a target="_blank" href="https://arxiv.org/abs/2302.13971"> LLaMA-13B </a></td>  <td>2.61</td>  <td>826</td>  <td>47.0</td>  <td>Non-commercial</td> </tr>

</tbody>
</table>
</div>

&shy;

Welcome to try the Chatbot Arena voting [demo](https://chat.lmsys.org/?arena).
Keep in mind that each benchmark has its limitations. Please consider the results as guiding references. See our discussion below for more technical details.

## Evaluating Chatbots with MT-bench and Arena

### Motivation

While several benchmarks exist for evaluating Large Language Model's (LLM) performance, such as [MMLU](https://arxiv.org/abs/2009.03300), [HellaSwag](https://arxiv.org/abs/1905.07830), and [HumanEval](https://github.com/openai/human-eval), 
we noticed that these benchmarks might fall short when assessing LLMs' human preferences. 
Traditional benchmarks often test LLMs on close-ended questions with concise outputs (e.g., multiple choices), which do not reflect the typical use cases of LLM-based chat assistants.

To fill this gap, in this leaderboard update, in addition to the Chatbot Arena Elo system, we add a new benchmark: MT-Bench.
- [MT-bench](https://arxiv.org/abs/2306.05685) is a challenging multi-turn question set designed to evaluate the conversational and instruction-following ability of models. You can view sample questions and answers of MT-bench [here](https://huggingface.co/spaces/lmsys/mt-bench).
- [Chatbot Arena](https://chat.lmsys.org/?arena) is a crowd-sourced battle platform, where users ask chatbots any question and vote for their preferred answer.

Both benchmarks are designed to use human preferences as the primary metric.

### Why MT-Bench?

MT-Bench is a carefully curated benchmark that includes 80 high-quality, multi-turn questions. 
These questions are tailored to assess the conversation flow and instruction-following capabilities of models in multi-turn dialogues. 
They include both common use cases and challenging instructions meant to distinguish between chatbots. 
MT-Bench serves as a **quality-controlled complement** to our crowd-sourced based evaluation -- Chatbot Arena.

Through running the Chatbot Arena for 2 months and analyzing our users' prompts, we've identified 8 primary categories of user prompts: Writing, Roleplay, Extraction, Reasoning, Math, Coding, Knowledge I (STEM), and Knowledge II (humanities/social science). 
We crafted 10 multi-turn questions per category, yielding a set of 160 questions in total. We display some sample questions below in Figure 1. You can find more [here](https://huggingface.co/spaces/lmsys/mt-bench).

<img src="/images/blog/leaderboard_week8/sample_question.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>
<p style="color:gray; text-align: center;">Figure 1: Sample questions from the MT-Bench.</p>

### But Still, How to Grade Chatbots' Answers?
Though we believe human preference is the gold standard, it is notoriously slow and expensive to collect. 
In our first [Vicuna blogpost](https://lmsys.org/blog/2023-03-30-vicuna/), we explored an automated evaluation pipeline based on GPT-4. 
This approach has since got popular and adopted in several [concurrent and follow-up works](#related-work).

In our latest paper, ["Judging LLM-as-a-judge"](https://arxiv.org/abs/2306.05685), we conducted a systematic study to answer how reliable those LLM judges are. 
We provide a brief overview of conclusions here but recommend reading the paper for more details.

We begin by acknowledging potential limitations of LLM-as-a-judge:

- **Position bias** where LLM judges may favor the first answer in a pairwise comparison.
- **Verbosity bias** where LLM judges may favor lengthier answers, regardless of their quality.
- **Self-enhancement bias** where LLM judges may favor their own responses.
- **Limited reasoning ability** referring to LLM judges' possible shortcomings in grading math and reasoning questions.

Our study then explores how few-shot judge, chain-of-thought judge, reference-based judge, and fine-tuned judge can help to mitigate these limitations.

Upon implementing some of these solutions, we discovered that despite limitations, strong LLM judges like GPT-4 can align impressively well with both controlled and crowdsourced human preferences, achieving over 80% agreement. 
This level of agreement is comparable to the agreement between two different human judges. 
Therefore, if used carefully, LLM-as-a-judge can act as a *scalable* and *explainable* approximation of human preferences.

We also found that single-answer grading based on GPT-4, without pairwise comparison, can also rank models effectively and match human preferences well. 
In Table 1, we present the MT-Bench as a column on the leaderboard based on single-answer grading with GPT-4.

## Results and Analysis

### MT-Bench Effectively Distinguishes Among Chatbots

Table 1 provides a detailed rundown of the MT-bench-enhanced leaderboard, where we conduct an exhaustive evaluation of 28 popular instruction-tuned models. 
We observe a clear distinction among chatbots of varying abilities, with scores showing a high correlation with the Chatbot Arena Elo rating. 
In particular, MT-Bench reveals noticeable performance gaps between GPT-4 and GPT-3.5/Claude, and between open and proprietary models.

To delve deeper into the distinguishing factors among chatbots, we select a few representative chatbots and break down their performance per category in Figure 2. 
GPT-4 shows superior performance in Coding and Reasoning compared to GPT-3.5/Claude, while Vicuna-13B lags significantly behind in several specific categories: Extraction, Coding, and Math. 
This suggests there is still ample room for improvement for open-source models.

<img src="/images/blog/leaderboard_week8/ability_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>
<p style="color:gray; text-align: center;">Figure 2: The comparison of 6 representative LLMs regarding their abilities in 8 categories: Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities.</p>


### Multi-turn Conversation Capabilities

We next analyze the multi-turn scores of selected models, presented in Table 2. 

<br>
<p style="color:gray; text-align: center;">Table 2. The breakdown of LLMs' MT-bench scores in the 1st and 2nd turn of a dialogue. Full score is 10.</p>
<div style="display: flex; justify-content: center;">
<table>
<tbody>
<tr> <th>Model</th> <th>Average 1st Turn Score</th> <th>Average 2nd Turn Score</th> <th>Score Difference</th>

<tr><td><a href="https://chat.openai.com/" target="_blank">GPT-4</a></td> <td>8.96</td> <td>9.03</td> <td>0.07</td>  </tr>

<tr><td><a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-v1</a></td> <td>8.15</td> <td>7.65</td> <td>-0.50</td> </tr>

<tr><td><a href="https://chat.openai.com/" target="_blank">GPT-3.5-turbo</a></td> <td>8.08</td> <td>7.81</td> <td>-0.26</td> </tr>

<tr><td><a href="https://github.com/lm-sys/FastChat#vicuna-weights" target="_blank">Vicuna-33B</a></td> <td>7.46</td> <td>6.79</td> <td>-0.67</td> </tr>

<tr><td><a href="https://huggingface.co/WizardLM/WizardLM-30B-V1.0" target="_blank">WizardLM-30B</a></td> <td>7.13</td> <td>6.89</td> <td>-0.24</td> </tr>

<tr><td><a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.0" target="_blank">WizardLM-13B</a></td> <td>7.12</td> <td>5.59</td> <td>-1.53</td> </tr>

<tr><td><a href="https://huggingface.co/timdettmers/guanaco-33b-merged" target="_blank">Guanaco-33B</a></td> <td>6.88</td> <td>6.18</td> <td>-0.71</td> </tr>

<tr><td><a href="https://github.com/lm-sys/FastChat#vicuna-weights" target="_blank">Vicuna-13B</a></td> <td>6.81</td> <td>5.96</td> <td>-0.85</td> </tr>

<tr><td><a href="https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023" target="_blank">PaLM2-Chat-Bison</a></td> <td>6.71</td> <td>6.09</td> <td>-0.63</td> </tr>

<tr><td><a href="https://github.com/lm-sys/FastChat#vicuna-weights" target="_blank">Vicuna-7B</a></td> <td>6.69</td> <td>5.30</td> <td>-1.39</td> </tr>

<tr><td><a href="https://huggingface.co/young-geng/koala" target="_blank">Koala-13B</a></td> <td>6.08</td> <td>4.63</td> <td>-1.45</td> </tr>

<tr><td><a href="https://huggingface.co/mosaicml/mpt-7b-chat" target="_blank">MPT-7B-Chat</a></td> <td>5.85</td> <td>4.99</td> <td>-0.86</td> </tr>

<tr><td><a href="https://huggingface.co/tiiuae/falcon-40b-instruct" target="_blank">Falcon-40B-instruct</a></td> <td>5.81</td> <td>4.53</td> <td>-1.29</td> </tr>

<tr><td><a href="https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b" target="_blank">H2OGPT-Oasst-Open-LLaMA-13B</a></td> <td>5.51</td> <td>3.74</td> <td>-1.78</td> </tr>
</tbody>
</table>
</div>

&shy;

The MT-bench incorporates challenging follow-up questions as part of its design. 
For open models, The performance drops significantly from the first to the second turn (e.g., Vicuna-7B, WizardLM-13B), while strong proprietary models maintain consistency. 
We also notice a considerable performance gap between LLaMA-based models and those with permissive licenses (MPT-7B, Falcon-40B, and instruction-tuned Open-LLaMA).


### Explainability in LLM judges 

Another advantage of LLM judges is their ability to provide explainable evaluations. 
Figure 3 presents an instance of GPT-4's judgment on an MT-bench question, with answers from alpaca-13b and gpt-3.5-turbo. 
GPT-4 provides thorough and logical feedback to support its judgment. 
Our [study](https://arxiv.org/abs/2306.05685) found that such reviews are beneficial in guiding humans to make better-informed decisions (refer to Section 4.2 for more details). 
All the GPT-4 judgments can be found on our [demo site](https://huggingface.co/spaces/lmsys/mt-bench).

<img src="/images/blog/leaderboard_week8/explainability_sample.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>
<p style="color:gray; text-align: center;">Figure 3: MT-bench provides more explainability in evaluating LLMs' human preferences.</p>

In conclusion, we have shown that MT-Bench effectively differentiates between chatbots of varying capabilities. 
It's scalable, offers valuable insights with category breakdowns, and provides explainability for human judges to verify. 
However, LLM judges should be used carefully. It can still make errors, especially when grading math/reasoning questions.


## How to Evaluate New Models on MT-Bench?

Evaluating models on MT-bench is simple and fast. Our script supports all huggingface models, and we’ve provided [detailed instructions](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#mt-bench), 
in which you can generate model’s answers to the MT-bench questions and their GPT-4 judgments. You can also examine the answers and reviews on our gradio browsing demo.

## Next steps
**Release of Conversations Data**

We're in the process of releasing Chatbot Arena conversations data to the broader research community. Stay tuned for updates!

**MT-bench-1K**

MT-Bench currently consists of a concise set of 80 carefully curated questions, ensuring the highest quality. 
We're actively expanding the question set to MT-Bench-1K by integrating high-quality prompts from the Chatbot Arena and generating new ones automatically using LLMs. 
If you have any good ideas, we'd be delighted to hear from you.

**Invitation for collaborations**

We're engaging with various organizations to explore possibilities for standardizing the evaluation of human preferences for LLMs at scale. 
If this interests you, please feel free to reach out to us.

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
- The [Arena Elo calculator](https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR?usp=sharing).
- The MMLU is based on [InstructEval](https://github.com/declare-lab/instruct-eval/blob/main/mmlu.py) and [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub/tree/main/MMLU).

If you wish to see more models on leaderboard, we invite you to [contribute to FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/arena.md#how-to-add-a-new-model) or [contact us](mailto:lmsysorg@gmail.com) to provide us with API access.
