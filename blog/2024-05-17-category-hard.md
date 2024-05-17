---
title: "Introducing Chatbot Arena Category Hard"
author: "Tianle Li, Wei-Lin Chiang"
date: "May 17, 2024"
previewImg: /images/blog/category_hard/preview.png
---

We are thrilled to introduce a new category on Chatbot Arena: Hard Prompts (English). 

[Motivations]

Through our [Arena-Hard](https://lmsys.org/blog/2024-04-19-arena-hard/) pipeline, we have identified a collection of high-quality prompts from existing Chatbot Arena battles. Each user prompt is evaluated against the 7 Key Criteria defined in Table X, using Llama-3-70B-Instruct as judge. The 7 Key Criteria are:

<table style="width:100%; border-collapse: collapse; border: 1px solid black;">
  <tr style="background-color: black; color: white;">
    <!-- <th style="border: 1px solid black; padding: 10px; text-align: left;">7 Key "Hardness" Criteria</th> -->
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

A Hardness Score is then calculated using the how many criteria are satisfied. Prompts that satisfy 6 or more of these hardness criteria are then designated as part of the "Hard" category and featured on a dedicated leaderboard. We present the distribution of the criteria and hardness score in Figure 1 and 2. We also present several example prompts with labeled criteria in [Example Section](#example).

<img src="/images/blog/category_hard/criteria_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 1. The percentage of each criteria within 1 million Chatbot Arena data.</p>

<img src="/images/blog/category_hard/criteria_breakdown_score.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 2. The percentage of prompts with different hardness score within 1 million Chatbot Arena data.</p>

Currently we are launching the Hard Prompts category for English, but we are working to expand this offering to other languages as well. For viewing of the full leaderboard, check out (link).

The results from the Hard Prompts (English) category, as shown in Table X, reveal some notable ranking differences. Specifically, we observe that the Llama-3-8B-Instruct model, which had previously performed on par with GPT-4-0314 on the general English leaderboard, has seen a significant drop in ranking within the Hard Prompts (English) category. This suggests that the Llama-3-8B-Instruct may struggle with the increased complexity and difficulty of the prompts in this new specialized category. We also observe improvement in performance among top proprietary models, such as GPT-4-Turbo, Claude-3-Opus, Claude-3-Sonnet, and GPT-4.

<div style="display: flex; justify-content: center; font-family: Consolas, monospace;">
<table style="line-height: 1; font-size: 1.0em;">
  <thead>
    <tr style="border-bottom: thin solid #ccc;">
      <th style="width: 30%;">Model Name</th>
      <th style="width: 25%;">English</th>
      <th style="width: 25%;">Hard Prompts (English)<br>Judge</th>
      <th style="width: 20%;">Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">gpt-4-turbo-2024-04-09</td>
      <td>1233</td>
      <td>1252<span style="color: green;">(↑)</span></td>
      <td style="color: green;">+19</td>
    </tr>
    <tr>
      <td style="text-align: left;">gemini-1.5-pro-0409-preview</td>
      <td>1232</td>
      <td>1224<span style="color: red;">(↓)</span></td>
      <td style="color: red;">-8</td>
    </tr>
    <tr>
      <td style="text-align: left;">llama-3-70b-instruct</td>
      <td>1225</td>
      <td>1214<span style="color: red;">(↓)</span></td>
      <td style="color: red;">-11</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-opus-20240229</td>
      <td>1214</td>
      <td>1230<span style="color: green;">(↑)</span></td>
      <td style="color: green;">+16</td>
    </tr>
    <tr>
      <td style="text-align: left;">claude-3-sonnet-20240229</td>
      <td>1175</td>
      <td>1186<span style="color: green;">(↑)</span></td>
      <td style="color: green;">+11</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-4-0314</td>
      <td>1165</td>
      <td>1183<span style="color: green;">(↑)</span></td>
      <td style="color: green;">+18</td>
    </tr>
    <tr>
      <td style="text-align: left;">llama-3-8b-instruct</td>
      <td>1164</td>
      <td>1143<span style="color: red;">(↓)</span></td>
      <td style="color: red;">-21</td>
    </tr>
    <tr>
      <td style="text-align: left;">command-r-plus</td>
      <td>1163</td>
      <td>1154<span style="color: red;">(↓)</span></td>
      <td style="color: red;">-9</td>
    </tr>
    <tr>
      <td style="text-align: left;">gpt-4-0613</td>
      <td>1146</td>
      <td>1163<span style="color: green;">(↑)</span></td>
      <td style="color: green;">+17</td>
    </tr>
  </tbody>
</table>
</div>

## Future
We are committed to continually enhancing the Chatbot Arena experience for our users. We look forward to seeing how the latest advancements in language models perform on these challenging prompts, and to sharing these insights with the broader community.

## Example

**Prompt 1:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

Suppose a drawer contains four green socks, five white socks, and three blue socks. We draw one sock from the drawer and it is equally likely that any one of the socks is drawn. Find the probabilities of the following events:

We reach into the drawer without looking to pull out four socks. What is the probability that we get at least two socks of the same color?

Prove that p = 1


**Prompt 2:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

tell me how to make a hydroponic nutrient solution at home to grow lettuce with precise amount of each nutrient


**Prompt 3:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

Solve the integral $\int_{-\infty}^{+\infty} exp(-x^2) dx $ step-by-step with detailed explanation


**Prompt 4:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

write me GLSL code which can gennrate at least 5 colors and 2 waves of particles cross each other	


**Prompt 5:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Creativity, Technical Accuracy, Real World]

Write me a python script for the foobar problem, but make it so that if read aloud, each pair of lines rhymes. (i.e. lines 1/2 rhyme, 3/4 rhyme and so on)


**Prompt 6:**

[Real World]

what is cake


**Prompt 7:**

[Specificity, Creativity, Real World]

Writing prompt: write the start of a short story / a man with an iphone is transported back to 1930s USA. 


**Prompt 8:**

[Specificity, Creativity, Real World]

writen ten different sentences that end with word "apple"
