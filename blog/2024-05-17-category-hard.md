---
title: "Introducing Hard Prompts Category in Chatbot Arena"
author: "Tianle Li, Wei-Lin Chiang"
date: "May 17, 2024"
previewImg: /images/blog/category_hard/preview.png
---

### Background

We introduce **Hard Prompts**, a new and challenging category in the Chatbot Arena [Leaderboard](https://leaderboard.lmsys.org).

Over the past few months, we have been hearing growing interests from the community in seeing more challenging prompts that push the limits of current language models. To address this demand, we are excited to introduce the **Hard Prompts** category, which features Arena's user prompts that are specifically designed to be more complex, demanding, and rigorous. These prompts are carefully curated to test the capabilities of the latest language models and to explore the boundaries of what they can achieve. We believe this new category can provide valuable insights into the strengths and weaknesses of different models in more challenging tasks.

### New Category: Hard Prompts!

To evaluate the difficulty of a prompt, we define several hardness criteria such as domain knowledge, complexity, or problem-solving. A prompt that satisfies multiple criteria is considered to be more challenging and is assigned a higher hardness score. We then use these scores to create a new leaderboard category, **Hard Prompts**.

In Figure 1, we present the ranking shift from English to Hard Prompts (English) . We observe **Llama-3-8B-Instruct**, which performs on par with **GPT-4-0314** on the English leaderboard, has seen a significant drop in ranking. This suggests that the model may struggle with the increased complexity and difficulty of the prompts in this new specialized category. Similarly, we observe **Claude-3-Opus** is now above **Llama-3-70B-Instruct** and slight improvement in **GPT-4o**.

<img src="/images/blog/category_hard/elo_comparison_1.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 1. Comparison between Chatbot Arena Category English vs Hard Prompts (English). We set gpt-4-0314 as anchor model.</p>

We also observe notable improvements in **GPT-3.5-Turbo-1106/0125** and **Claude-2.1**, as well as **Phi-3**, which is trained to perform well in reasoning tasks. 

<img src="/images/blog/category_hard/elo_comparison_2.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 2. Comparison between Chatbot Arena Category English vs Hard Prompts (English). We set mixtral-8x7b-instruct-v0.1 as anchor model.</p>


### How to Define Hard Prompts?

A few weeks ago, we introduce the [Arena-Hard](https://lmsys.org/blog/2024-04-19-arena-hard/) pipeline to identify a collection of high-quality prompts from Chatbot Arena. Each user prompt is evaluated against the 7 Key Criteria defined in below Table.

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

We employ Meta's **Llama-3-70B-Instruct** as the judge model to help us label over 1 million Arena battles. Figure 3 shows the criteria breakdown (i.e., how many prompts satisfy each criteria). We observe the most common criteria are Specificity, Domain Knowledge, and Real-world Application, while the relatively rare criteria are Problem-Solving and Complexity.

<img src="/images/blog/category_hard/key_criteria_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 3. The percentage of each criteria within 1 million Chatbot Arena data.</p>

We then calculate its Hardness Score by how many criteria are satisfied and present the distribution in Figure 3. Interestingly, we find that ~20% of prompts are >=6 score. You can find several examples below to demonstrate what a hard prompt actually looks like in the [Example Section](#example).


<img src="/images/blog/category_hard/hardness_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 4. The percentage of prompts with different hardness score within 1 million Chatbot Arena data.</p>


We then take the score>=6 prompts (satisfy 6 or more of these criteria) to be the "Hard Prompts" category and calculate two leaderboards, **Hard Prompt (English)** and **Hard Prompts (Overall)**.

Below is screenshot of the leaderboard for **Hard Prompts (English)** category (as of May 17, 2024). You can find the latest version at [https://leaderboard.lmsys.org](https://leaderboard.lmsys.org) (-> Category dropdown).

<img src="/images/blog/category_hard/leaderboard.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 95%"></img>
<p style="color:gray; text-align: center;">Figure 5. The leaderboard for Hard Prompts (English) category as of May 17, 2024.</p>


We are commited to continuously enhance the Chatbot Arena leaderboard and share insights with the broader community. We welcome you to contribute more challenging prompts and look forward to seeing how the latest advancements in language models perform!

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

## Example
We present 10 examples of user prompt with increasing hardness score. The labeled criteria are inside the bracket.

**Prompt 1:**

[None]

hello


**Prompt 2:**

[Real World]

what is cake


**Prompt 3:**

[Creativity, Real World]

How to pickup a girl?


**Prompt 4:**

[Specificity, Creativity, Real World]

writen ten different sentences that end with word "apple"


**Prompt 5:**

[Specificity, Creativity, Real World]

Writing prompt: write the start of a short story / a man with an iphone is transported back to 1930s USA. 


**Prompt 6:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

tell me how to make a hydroponic nutrient solution at home to grow lettuce with precise amount of each nutrient


**Prompt 7:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

Solve the integral $\int_{-\infty}^{+\infty} exp(-x^2) dx $ step-by-step with detailed explanation


**Prompt 8:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

write me GLSL code which can gennrate at least 5 colors and 2 waves of particles cross each other	


**Prompt 9:**

[Specificity, Domain Knowledge, Complexity, Problem-solving, Technical Accuracy, Real World]

My situation is this: I’m setting up a server running at home Ubuntu to run an email server and a few other online services. As we all know, for my email to work reliably and not get blocked I need to have an unchanging public IP address. Due to my circumstances I am not able to get a static IP address through my ISP or change ISPs at the moment.

The solution I have found is to buy a 4G SIM card with a static IP (from an ISP that offers that), which I can then use with a USB dongle. However this 4G connection costs me substantially per MB to use.

But. Mail is the only server that needs a static IP address. For everything else using my home network connection and updating my DNS records with DDNS would be fine. I have tested this setup previously for other services and it has worked.

So. I was wondering. Would it in theory be possible to: connect the server to two network interfaces at the same time and route traffic depending on destination port. I.e. all outgoing connections to ports 25, 465, 587, and possibly 993 should be sent through the 4G dongle interface (enx344b50000000) and all other connections sent over eth0. Similarly, the server should listen for incoming connections on the same ports on enx344b50000000 and listen on all other ports (if allowed by ufw) on eth0.

I would then need DNS records from mail.mydomain.tld —> <4g static public IP> and mydomain.tld —> <home public IP> (updated with DDNS, and NAT configured on my home router).

Computers on the internet would then be able to seamlessly connect to these two IP addresses, not “realising” that they are in fact the same machine, as long as requests to mail.mydomain.tld are always on the above mentioned ports.

Question: Is this possible? Could it be a robust solution that works the way I hope? Would someone be able to help me set it up?

I have come across a few different guides in my DuckDuckGo-ing, I understand it has to do with setting a mark in iptables and assigning them to a table using ip route. However I haven't managed to get it to work yet, and many of these guides are for VPNs and they all seem to be slightly different to each other. So I thought I would ask about my own specific use case


**Prompt 10:** 

[Specificity, Domain Knowledge, Complexity, Problem-solving, Creativity, Technical Accuracy, Real World]

Write me a python script for the foobar problem, but make it so that if read aloud, each pair of lines rhymes. (i.e. lines 1/2 rhyme, 3/4 rhyme and so on)