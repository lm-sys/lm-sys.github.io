---
title: "Chatbot Arena Leaderboard Week 8: Introducing MT-Bench and Vicuna-33B"
author: "Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Hao Zhang"
date: "June 22, 2023"
previewImg: /images/blog/langchain/overview.png
---

In this blog post, we share the latest update on Chatbot Arena leaderboard, which now includes more open models and three metrics.

1. **Arena Elo**, based on 42K anonymous votes from Chatbot Arena using the Elo rating system.
2. **MT-Bench** score, based on a challenging multi-turn benchmark and GPT-4 grading, proposed and validated in our recent [paper](https://arxiv.org/abs/2306.05685).
3. **MMLU**, a widely adopted [benchmark](https://arxiv.org/abs/2009.03300).

Furthermore, we’re excited to introduce our new series of Vicuna v1.3 models, ranging from 7B to 33B parameters, all trained on an extended set of user-shared conversations dataset. 
The weights are now [available](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights).

## Updated Leaderboard and New Models

<script type="javascript">
/*
 *   This content is licensed according to the W3C Software License at
 *   https://www.w3.org/Consortium/Legal/2015/copyright-software-and-document
 *
 *   File:   sortable-table.js
 *
 *   Desc:   Adds sorting to a HTML data table that implements ARIA Authoring Practices
 */

'use strict';

class SortableTable {
  constructor(tableNode) {
    this.tableNode = tableNode;

    this.columnHeaders = tableNode.querySelectorAll('thead th');

    this.sortColumns = [];

    for (var i = 0; i < this.columnHeaders.length; i++) {
      var ch = this.columnHeaders[i];
      var buttonNode = ch.querySelector('button');
      if (buttonNode) {
        this.sortColumns.push(i);
        buttonNode.setAttribute('data-column-index', i);
        buttonNode.addEventListener('click', this.handleClick.bind(this));
      }
    }

    this.optionCheckbox = document.querySelector(
      'input[type="checkbox"][value="show-unsorted-icon"]'
    );

    if (this.optionCheckbox) {
      this.optionCheckbox.addEventListener(
        'change',
        this.handleOptionChange.bind(this)
      );
      if (this.optionCheckbox.checked) {
        this.tableNode.classList.add('show-unsorted-icon');
      }
    }
  }

  setColumnHeaderSort(columnIndex) {
    if (typeof columnIndex === 'string') {
      columnIndex = parseInt(columnIndex);
    }

    for (var i = 0; i < this.columnHeaders.length; i++) {
      var ch = this.columnHeaders[i];
      var buttonNode = ch.querySelector('button');
      if (i === columnIndex) {
        var value = ch.getAttribute('aria-sort');
        if (value === 'descending') {
          ch.setAttribute('aria-sort', 'ascending');
          this.sortColumn(
            columnIndex,
            'ascending',
            ch.classList.contains('num')
          );
        } else {
          ch.setAttribute('aria-sort', 'descending');
          this.sortColumn(
            columnIndex,
            'descending',
            ch.classList.contains('num')
          );
        }
      } else {
        if (ch.hasAttribute('aria-sort') && buttonNode) {
          ch.removeAttribute('aria-sort');
        }
      }
    }
  }

  sortColumn(columnIndex, sortValue, isNumber) {
    function compareValues(a, b) {
      if (sortValue === 'ascending') {
        if (a.value === b.value) {
          return 0;
        } else {
          if (isNumber) {
            return a.value - b.value;
          } else {
            return a.value < b.value ? -1 : 1;
          }
        }
      } else {
        if (a.value === b.value) {
          return 0;
        } else {
          if (isNumber) {
            return b.value - a.value;
          } else {
            return a.value > b.value ? -1 : 1;
          }
        }
      }
    }

    if (typeof isNumber !== 'boolean') {
      isNumber = false;
    }

    var tbodyNode = this.tableNode.querySelector('tbody');
    var rowNodes = [];
    var dataCells = [];

    var rowNode = tbodyNode.firstElementChild;

    var index = 0;
    while (rowNode) {
      rowNodes.push(rowNode);
      var rowCells = rowNode.querySelectorAll('th, td');
      var dataCell = rowCells[columnIndex];

      var data = {};
      data.index = index;
      data.value = dataCell.textContent.toLowerCase().trim();
      if (isNumber) {
        data.value = parseFloat(data.value);
      }
      dataCells.push(data);
      rowNode = rowNode.nextElementSibling;
      index += 1;
    }

    dataCells.sort(compareValues);

    // remove rows
    while (tbodyNode.firstChild) {
      tbodyNode.removeChild(tbodyNode.lastChild);
    }

    // add sorted rows
    for (var i = 0; i < dataCells.length; i += 1) {
      tbodyNode.appendChild(rowNodes[dataCells[i].index]);
    }
  }

  /* EVENT HANDLERS */

  handleClick(event) {
    var tgt = event.currentTarget;
    this.setColumnHeaderSort(tgt.getAttribute('data-column-index'));
  }

  handleOptionChange(event) {
    var tgt = event.currentTarget;

    if (tgt.checked) {
      this.tableNode.classList.add('show-unsorted-icon');
    } else {
      this.tableNode.classList.remove('show-unsorted-icon');
    }
  }
}

// Initialize sortable table buttons
window.addEventListener('load', function () {
  var sortableTables = document.querySelectorAll('table.sortable');
  for (var i = 0; i < sortableTables.length; i++) {
    new SortableTable(sortableTables[i]);
  }
});
</script>


<style>
.sr-only {
  position: absolute;
  top: -30em;
}

table.sortable td,
table.sortable th {
  padding: 0.125em 0.25em;
  width: 8em;
}

table.sortable th {
  font-weight: bold;
  border-bottom: thin solid #888;
  position: relative;
}

table.sortable th.no-sort {
  padding-top: 0.35em;
}

table.sortable th:nth-child(5) {
  width: 10em;
}

table.sortable th button {
  position: absolute;
  padding: 4px;
  margin: 1px;
  font-size: 100%;
  font-weight: bold;
  background: transparent;
  border: none;
  display: inline;
  right: 0;
  left: 0;
  top: 0;
  bottom: 0;
  width: 100%;
  text-align: left;
  outline: none;
  cursor: pointer;
}

table.sortable th button span {
  position: absolute;
  right: 4px;
}

table.sortable th[aria-sort="descending"] span::after {
  content: "▼";
  color: currentcolor;
  font-size: 100%;
  top: 0;
}

table.sortable th[aria-sort="ascending"] span::after {
  content: "▲";
  color: currentcolor;
  font-size: 100%;
  top: 0;
}

table.show-unsorted-icon th:not([aria-sort]) button span::after {
  content: "♢";
  color: currentcolor;
  font-size: 100%;
  position: relative;
  top: -3px;
  left: -4px;
}

table.sortable td.num {
  text-align: right;
}

table.sortable tbody tr:nth-child(odd) {
  background-color: #ddd;
}

/* Focus and hover styling */

table.sortable th button:focus,
table.sortable th button:hover {
  padding: 2px;
  border: 2px solid currentcolor;
  background-color: #e5f4ff;
}

table.sortable th button:focus span,
table.sortable th button:hover span {
  right: 2px;
}

table.sortable th:not([aria-sort]) button:focus span::after,
table.sortable th:not([aria-sort]) button:hover span::after {
  content: "▼";
  color: currentcolor;
  font-size: 100%;
  top: 0;
}
</style>

<div class="table-wrap"><table class="sortable">
  <caption>
    Students currently enrolled in WAI-ARIA 101
    <span class="sr-only">, column headers with buttons are sortable.</span>
  </caption>
  <thead>
    <tr>
      <th>
        <button>
          First Name
          <span aria-hidden="true"></span>
        </button>
      </th>
      <th aria-sort="ascending">
        <button>
          Last Name
          <span aria-hidden="true"></span>
        </button>
      </th>
      <th>
        <button>
          Company
          <span aria-hidden="true"></span>
        </button>
      </th>
      <th class="no-sort">Address</th>
      <th class="num">
        <button>
          Favorite Number
          <span aria-hidden="true"></span>
        </button>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fred</td>
      <td>Jackson</td>
      <td>Canary, Inc.</td>
      <td>123 Broad St.</td>
      <td class="num">56</td>
    </tr>
    <tr>
      <td>Sara</td>
      <td>James</td>
      <td>Cardinal, Inc.</td>
      <td>457 First St.</td>
      <td class="num">7</td>
    </tr>
    <tr>
      <td>Ralph</td>
      <td>Jefferson</td>
      <td>Robin, Inc.</td>
      <td>456 Main St.</td>
      <td class="num">513</td>
    </tr>
    <tr>
      <td>Nancy</td>
      <td>Jensen</td>
      <td>Eagle, Inc.</td>
      <td>2203 Logan Dr.</td>
      <td class="num">3.5</td>
    </tr>
  </tbody>
</table></div>
        


<br>
<p style="color:gray; text-align: center;">Table 1. Elo ratings of LLMs (Timeframe: April 24 - May 22, 2023)</p>
<table style="display: flex; justify-content: center;" align="left" >
<tbody>
<tr> <th>Rank</th> <th>Model</th> <th>Elo Rating</th> <th>Description</th> <th>License</th> </tr>

<tr> <td>1</td> <td>🥇 <a href="https://chat.openai.com/" target="_blank">GPT-4</a></td> <td>1225</td> <td>ChatGPT-4 by OpenAI</td> <td>Proprietary</td> </tr>

<tr> <td>2</td> <td>🥈 <a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-v1</a></td> <td>1195</td> <td>Claude by Anthropic</td> <td>Proprietary</td> </tr>

<tr> <td>3</td> <td>🥉 <a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-instant-v1</a></td> <td>1153</td> <td>Lighter, less expensive, and much faster version of Claude</td> <td>Proprietary</td> </tr>

<tr> <td>4</td> <td> <a href="https://chat.openai.com/" target="_blank">GPT-3.5-turbo</a></td> <td>1143</td> <td>ChatGPT-3.5 by OpenAI</td>  <td>Proprietary</td> </tr>

<tr> <td>5</td> <td><a href="https://lmsys.org/blog/2023-03-30-vicuna/" target="_blank">Vicuna-13B</a></td> <td>1054</td> <td>a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS</td> <td>Weights available; Non-commercial</td> </tr>

<tr> <td>6</td> <td><a href="https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023" target="_blank">PaLM 2</a></td> <td>1042</td> <td>PaLM 2 tuned for chat (chat-bison@001 on Google Vertex AI). The PaLM 2 model family is powering Bard.</td> <td>Proprietary</td> </tr>

<tr> <td>7</td> <td><a href="https://huggingface.co/lmsys/vicuna-7b-delta-v1.1" target="_blank">Vicuna-7B</a></td> <td>1007</td> <td>a chat assistant fine-tuned from LLaMA on user-shared conversations by LMSYS</td> <td>Weights available; Non-commercial</td> </tr>

<tr> <td>8</td> <td><a href="https://bair.berkeley.edu/blog/2023/04/03/koala" target="_blank">Koala-13B</a></td> <td>980</td> <td>a dialogue model for academic research by BAIR</td> <td>Weights available; Non-commercial</td> </tr>

<tr> <td>9</td> <td><a href="https://www.mosaicml.com/blog/mpt-7b" target="_blank">mpt-7b-chat</a></td> <td>952</td> <td>a chatbot fine-tuned from MPT-7B by MosaicML</td> <td>CC-By-NC-SA-4.0</td> </tr>

<tr> <td>10</td> <td><a href="https://huggingface.co/lmsys/fastchat-t5-3b-v1.0" target="_blank">FastChat-T5-3B</a></td> <td>941</td> <td>a chat assistant fine-tuned from FLAN-T5 by LMSYS</td> <td>Apache 2.0</td> </tr>

<tr> <td>11</td> <td><a href="https://crfm.stanford.edu/2023/03/13/alpaca.html" target="_blank">Alpaca-13B</a></td> <td>937</td> <td>a model fine-tuned from LLaMA on instruction-following demonstrations by Stanford</td>  <td>Weights available; Non-commercial</td> </tr>

<tr> <td>12</td> <td><a href="https://huggingface.co/BlinkDL/rwkv-4-raven" target="_blank">RWKV-4-Raven-14B</a></td> <td>928</td> <td>an RNN with transformer-level LLM performance</td> <td>Apache 2.0</td> </tr>

<tr> <td>13</td> <td><a href="https://open-assistant.io" target="_blank">Oasst-Pythia-12B</a></td> <td>921</td> <td>an Open Assistant for everyone by LAION</td> <td>Apache 2.0</td> </tr>

<tr> <td>14</td> <td><a href="https://chatglm.cn/blog" target="_blank">ChatGLM-6B</a></td> <td>921</td> <td>an open bilingual dialogue language model by Tsinghua University</td> <td>Weights available; Non-commercial</td> </tr>

<tr> <td>15</td> <td><a href="https://github.com/stability-AI/stableLM" target="_blank">StableLM-Tuned-Alpha-7B</a></td> <td>882</td> <td>Stability AI language models</td>  <td>CC-BY-NC-SA-4.0</td> </tr>

<tr> <td>16</td> <td><a href="https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm" target="_blank">Dolly-V2-12B</a></td> <td>866</td> <td>an instruction-tuned open large language model by Databricks</td> <td>MIT</td> </tr>

<tr> <td>17</td> <td><a href="https://arxiv.org/abs/2302.13971" target="_blank">LLaMA-13B</a></td> <td>854</td> <td>open and efficient foundation language models by Meta</td> <td>Weights available; Non-commercial</td> </tr>

</tbody>
</table>

&shy;

You are welcome to check out the latest [leaderboard](https://chat.lmsys.org/?leaderboard) or try the arena [demo](https://chat.lmsys.org/?arena). 
Keep in mind that each benchmark has its limitations. Please consider the results as guiding references. See our discussion below for more technical details.


## Evaluating Chatbots with MT-bench and Arena

### Motivation

While several benchmarks exist for evaluating Large Language Model's (LLM) performance, such as MMLU, HellaSwag, and HumanEval, 
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
<p style="color:gray; text-align: center;">Table 2. Table placeholder</p>
<table style="display: flex; justify-content: center;" align="left" >
<tbody>
<tr> <th>Rank</th> <th>Model</th> <th>Elo Rating</th> <th>Description</th> <th>License</th> </tr>

<tr> <td>1</td> <td>🥇 <a href="https://chat.openai.com/" target="_blank">GPT-4</a></td> <td>1225</td> <td>ChatGPT-4 by OpenAI</td> <td>Proprietary</td> </tr>

<tr> <td>2</td> <td>🥈 <a href="https://www.anthropic.com/index/introducing-claude" target="_blank">Claude-v1</a></td> <td>1195</td> <td>Claude by Anthropic</td> <td>Proprietary</td> </tr>

</tbody>
</table>

&shy;

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
