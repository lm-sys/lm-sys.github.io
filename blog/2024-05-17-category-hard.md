---
title: "Introducing New Chatbot Arena Category: Hard Prompts"
author: "Tianle Li, Wei-Lin Chiang"
date: "May 17, 2024"
previewImg: /images/blog/category_hard/preview.png
---

We are thrilled to introduce a new category on Chatbot Arena: Hard Prompts. 

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

<img src="/images/blog/category_hard/key_criteria_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 1. The percentage of each criteria within 1 million Chatbot Arena data.</p>

<img src="/images/blog/category_hard/hardness_breakdown.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 2. The percentage of prompts with different hardness score within 1 million Chatbot Arena data.</p>

We are launching the Hard Prompts category for All Languages and English Only, but we are working to expand this offering to other languages as well. For viewing of the full leaderboard, check out (link).

The results from the Hard Prompts (English) category, as shown in Table X, reveal some notable ranking differences. Specifically, we observe that the Llama-3-8B-Instruct model, which had previously performed on par with GPT-4-0314 on the general English leaderboard, has seen a significant drop in ranking within the Hard Prompts (English) category. This suggests that the Llama-3-8B-Instruct may struggle with the increased complexity and difficulty of the prompts in this new specialized category. We also observe improvement in performance among top proprietary models, such as GPT-4-Turbo, Claude-3-Opus, Claude-3-Sonnet, and GPT-4.

<img src="/images/blog/category_hard/elo_comparison_1.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 3. Comparison between Chatbot Arena Category English vs Hard Prompts (English). We set gpt-4-0314 as anchor when computing elo.</p>

<img src="/images/blog/category_hard/elo_comparison_2.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 85%"></img>
<p style="color:gray; text-align: center;">Figure 4. Comparison between Chatbot Arena Category English vs Hard Prompts (English). We set mixtral-8x7b-instruct-v0.1 as anchor when computing elo.</p>

## Future
We are committed to continually enhancing the Chatbot Arena experience for our users. We look forward to seeing how the latest advancements in language models perform on these challenging prompts, and to sharing these insights with the broader community.

## Citation
```
@misc{arenacategoryhard2024,
    title = {Introducing New Chatbot Arena Category: Hard Prompts},
    url = {https://lmsys.org/blog/2024-05-17-category-hard/},
    author = {Tianle Li, Wei-Lin Chiang},
    month = {May},
    year = {2024}
}
```

## Example
We present 10 examples of user prompt with criteria labeled by Llama-3-70B-Instruct, in increasing hardness. The labeled criteria are inside the bracket.

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