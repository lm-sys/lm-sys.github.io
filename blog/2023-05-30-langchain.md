---
title: "Private and Local LangChain with Open Models"
author: "Shuo Yang"
date: "May 30, 2023"
previewImg: /images/blog/langchain/image3.png
---

## **TL;DR**

Many applications rely on closed-source OpenAI APIs, but now you can effortlessly port them to use open-source alternatives without modifying the code. FastChat's OpenAI-compatible API server enables this seamless transition, ensuring improved data privacy and compatibility with various open models such as Vicuna and MPT-Chat.


## **Demo: LangChain w/ Vicuna-13B**

_Enliven your code, and communicate with it through a single command line._


<img src="/images/blog/langchain/image1.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


_Enliven your docs, and communicate with it through a single command line._


<img src="/images/blog/langchain/image5.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


While ChatGPT's abilities have been awe-inspiring, their reliance on closed-source APIs has often posed limitations. Enter FastChat's OpenAI-compatible API server and the LangChain application. By leveraging these tools, we are able to locally deploy chat models like ChatGPT or Vicuna-13B, thus ensuring improved data privacy and operational flexibility.


## **Local OpenAI API Server with FastChat**

LangChain is a framework for developing applications powered by language models. It provides a set of tools, components and interfaces that simplify the process of creating applications that are supported by large language models (LLMs) and chat models. People have implemented many applications and features using OpenAI models with LangChain.


<img src="/images/blog/langchain/image3.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>



### **Why Local API Server?**

**Data Privacy**: When using FastChat's OpenAI-compatible API server and LangChain, all the data and interactions remain on your local machine. This means you have full control over your data, and it never leaves your local environment unless you decide to share it. This local setup ensures that sensitive data isn't exposed to third-party services, reducing the risk of data breaches and ensuring compliance with data privacy regulations.

**Cost Saving**: Traditional cloud-based API services often charge based on the number of requests or the tokens used. These costs can add up quickly, especially for researchers, organizations and companies. By running models locally, you can fully harness the power of large AI models without the worry of accumulating costs from API.

**Customizability: **With a local setup, you have the freedom to adapt the AI model to suit your specific needs. You can experiment with different parameters, settings, or even adjust the model architecture itself. More importantly, it allows you the opportunity to fine-tune the model for certain specific behaviors. This capability gives you control not only over how the model operates but also over the quality and relevance of the output.


## **Comparing Vicuna-13B, MPT-Chat-7B, and OpenAI for using LangChain**

FastChat boasts excellent support for local LangChain, allowing us to seamlessly substitute local models in place of OpenAI models. This enhances the flexibility and convenience of our system and brings to the forefront the crucial role that the quality of the chosen model plays in the overall performance.

Therefore, we have carried out extensive tests on the models executing LangChain tasks. These tests encompass a wide variety, including text-based question answering tasks and salesman agent performance tasks.


### Question Answering over Docs

 

Text-based question answering assesses the model's natural language understanding and generation abilities, and its grasp of common knowledge. We selected the transcript from the 2022 State of the Union address by President Biden as the document for querying. Six questions were posed to the model, each of which had its answer directly found within the text of the document. 

To evaluate the models' performance, we established a scoring system based on their responses to the posed questions. A model receives one point for providing a correct answer, and zero point for an incorrect answer.


<img src="/images/blog/langchain/image2.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


In terms of understanding the queries, all three models were successful. However, when it came to text retrieval ability, OpenAI demonstrated a clear advantage over Vicuna. This could very likely be attributed to the higher quality of OpenAI's embeddings, making it easier for the model to locate similar content.


### Salesman Agent Performance

To further evaluate the models' interaction capabilities, we implemented an innovative approach by having the models take on the role of a salesman through LangChain. We posed several questions and invited both human evaluators and GPT-4 to rate the quality of the responses provided by the different models.

This test offers insights into the quality of text generation and the ability to portray a convincing agent role, aspects that are of utmost importance within LangChain. The 'salesman' scenario is a robust way to understand how effectively a model can engage in complex dialogue, showcasing its ability to respond appropriately and convincingly in a specific role. The scoring criteria here also reflects the emphasis on quality, both in terms of coherence and the ability to effectively deliver on the task of playing the role of a 'salesman'.


#### Sales Agent

<img src="/images/blog/langchain/image4.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

#### GPT4 evaluation



1. **Vicuna**:
    * Answer 1: 9/10 - Comprehensive and clear, emphasizing the company's mission and values.
    * Answer 2: 9/10 - Good explanation of the unique selling proposition, but could be more explicit in differentiating from competitors.
    * Answer 3: 10/10 - Provides detailed product information, including environmental friendliness and hypoallergenic properties.
    * Total Score: 28/30
2. **GPT-3.5-turbo**:
    * Answer 1: 8/10 - Concise, but does not expand on the company's mission and values.
    * Answer 2: 8/10 - Repeats previous information, does not detail the differences from competitors.
    * Answer 3: 10/10 - Provides detailed product information, focusing on environmental friendliness and hypoallergenic properties.
    * Total Score: 26/30
3. **MPT**:
    * Answer 1: 8/10 - Clear and succinct, but does not delve into the company's mission and values.
    * Answer 2: 8/10 - Lacks clarity on company specifics and fails to differentiate from competitors.
    * Answer 3: 9/10 - Provides detailed product information, but not as explicit on the environmental friendliness and hypoallergenic properties as the other two.
    * Total Score: 25/30

The Salesman test provided interesting insights into the conversational and agent capabilities of the three models: Vicuna, GPT-3.5-turbo, and MPT. Vicuna model, performed exceptionally well, earning a total score of 28 out of 30. The model displayed impressive conversational ability, offering comprehensive and clear responses, effectively emphasizing the company's mission and values. It also demonstrated strong agent behavior, understanding and selling the product's unique selling propositions and providing detailed product information. GPT-3.5-turbo and MPT also performed well, but they fell short in expanding on the company's mission and values.

In conclusion, while all three models demonstrated reasonable proficiency in engaging with human dialogue, Vicuna displayed superior abilities in embodying the role of an agent. These findings highlight the importance of choosing a model that aligns with your specific task requirements.


## **Acknowledgment**

The OpenAI-compatible API Server is primarily contributed by Shuo Yang, Siyuan Zhuang, and Xia Han.