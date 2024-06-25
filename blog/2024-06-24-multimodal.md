---
title: "The Multimodal Arena is Here!"
author: "Lisa Dunlap*, Christopher Chou*, Anastasios Angelopoulos, Wei-Lin Chiang"
date: "June 25, 2024"
previewImg: /images/blog/arena/cover.png
---
### Multimodal Chatbot Arena

We added image support to Chatbot Arena! You can now chat with your favorite vision-language models from OpenAI, Anthropic, Google, and most other major LLM providers, and most importantly, help us answer the age-old question: 

**Which model is the best?**

In a little over a week we have collected over 12,000 user preference votes across over 60 languages. In this post we show the initial leaderboard and statistics, some interesting conversations submitted to the arena, and include a short discussion on the future of the multimodal arena. 

## Leaderboard results

For additional information on how the leaderboard is computed, please see this notebook. 

<style>
th {text-align: left}
td {text-align: left}
</style>


<br>
<p style="color:gray; text-align: center;">Table 1. Multimodal Arena Leaderboard (Timeframe: June 10th - June 22, 2023). Total votes = 12,827. The latest and detailed version <a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard" target="_blank">here</a>.</p>
<table style="display: flex; justify-content: center;" align="left" >
<tbody>
<tr> <th>Rank</th> <th>Model</th> <th>Elo Rating</th> <th>Votes</th></tr>


<tr> <td>1</td> <td> <a href="https://chat.openai.com/" target="_blank">GPT-4o</a></td> <td>1116</td> <td>3342</td> </tr>


<tr> <td>2</td> <td> <a href="https://www.anthropic.com/news/claude-3-5-sonnet" target="_blank">Claude 3.5 Sonnet</a></td> <td>1097</td> <td>3873</td> </tr>


<tr> <td>3</td> <td> <a href="https://deepmind.google/technologies/gemini/pro/" target="_blank">Gemini 1.5 Pro</a></td> <td>1060</td> <td>3282</td></tr>


<tr> <td>4</td> <td> <a href="https://chat.openai.com/" target="_blank">GPT-4 Turbo</a></td> <td>1055</td> <td>2878</td></tr>


<tr> <td>5</td> <td> <a href="https://www.anthropic.com/news/claude-3-family" target="_blank">Claude 3 Opus</a></td> <td>1055</td> <td>2878</td></tr>


<tr> <td>6</td> <td> <a href="https://deepmind.google/technologies/gemini/flash/" target="_blank">Gemini 1.5 Flash</a></td> <td>966</td> <td>3316</td></tr>


<tr> <td>7</td> <td> <a href="https://www.anthropic.com/news/claude-3-family" target="_blank">Claude 3 Sonnet</a></td> <td>939</td> <td>3400</td></tr>


<tr> <td>8</td> <td> <a href="https://llava-vl.github.io/blog/2024-01-30-llava-next/" target="_blank">Llava 1.6 34B</a></td> <td>903</td> <td>1650</td></tr>


<tr> <td>9</td> <td> <a href="https://www.anthropic.com/news/claude-3-family" target="_blank">Claude 3 Haiku</a></td> <td>890</td> <td>3464</td></tr>


</tbody>
</table>


<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Conversation Display</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 20px;
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .image-container img {
            max-width: 50%;
            height: auto;
        }
        .conversation-container {
            width: 100%;
            max-width: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .message {
            margin: 10px 0;
        }
        .user {
            text-align: left;
            color: #007BFF;
        }
        .chatbot {
            text-align: right;
            color: #28A745;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="image-container">
            <img src="/images/blog/arena/cover.png" alt="Descriptive Image">
        </div>
        <div class="conversation-container">
            <div class="message user">
                <strong>User:</strong> Hello, how can you help me today?
            </div>
            <div class="message chatbot">
                <strong>Chatbot:</strong> Hi! I can assist you with a variety of tasks. What do you need help with?
            </div>
            <div class="message user">
                <strong>User:</strong> I need information about the latest tech trends.
            </div>
            <div class="message chatbot">
                <strong>Chatbot:</strong> Sure! The latest tech trends include advancements in AI, 5G technology, and quantum computing.
            </div>
        </div>
    </div>
    <div class="container">
        <div class="image-container">
            <img src="/images/blog/arena/cover.png" alt="Descriptive Image">
        </div>
        <div class="conversation-container">
            <div class="message user">
                <strong>User:</strong> Hello, how can you help me today?
            </div>
            <div class="message chatbot">
                <strong>Chatbot:</strong> Hi! I can assist you with a variety of tasks. What do you need help with?
            </div>
            <div class="message user">
                <strong>User:</strong> I need information about the latest tech trends.
            </div>
            <div class="message chatbot">
                <strong>Chatbot:</strong> Sure! The latest tech trends include advancements in AI, 5G technology, and quantum computing.
            </div>
        </div>
    </div>
</body>