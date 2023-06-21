---
title: "Building a Truly \"Open\" OpenAI API Server with Open Models Locally"
author: "Shuo Yang and Siyuan Zhuang"
date: "June 9, 2023"
previewImg: /images/blog/langchain/overview.png
---


Many applications have been built on closed-source OpenAI APIs, but now you can effortlessly port them to use open-source alternatives without modifying the code. [FastChat](https://github.com/lm-sys/FastChat)'s OpenAI-compatible API server enables this seamless transition.
In this blog post, we show how you can do this and use LangChain as an [example](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md).


## **Demo: LangChain with Vicuna-13B**

Here, we present two demos of using LangChain with [Vicuna-13B](http://ec2-52-40-36-154.us-west-2.compute.amazonaws.com:3000/blog/2023-03-30-vicuna/), a state-of-the-art open model.

1. Question answering over docs  
  Enliven your documents, and communicate with them through a single command line ([doc](https://python.langchain.com/en/latest/use_cases/question_answering.html)).

<img src="/images/blog/langchain/qa_demo.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

2. Code understanding  
  Clone the llama repository and then understand the code with a single command line, bringing your code to life ([doc](https://python.langchain.com/en/latest/use_cases/code.html)).

<img src="/images/blog/langchain/code_analysis.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

The demos above are implemented directly with default LangChain code.
They don't require you to adapt specifically for Vicuna. Any tool implemented with the OpenAI API can be seamlessly migrated to the open models through FastChat.

## **Why Local API Server?**

**Data Privacy**: When using FastChat's OpenAI-compatible API server and LangChain, all the data and interactions remain on your local machine. This means you have full control over your data, and it never leaves your local environment unless you decide to share it. This local setup ensures that sensitive data isn't exposed to third-party services, reducing the risk of data breaches and ensuring compliance with data privacy regulations.

**Cost Saving**: Traditional cloud-based API services often charge based on the number of requests or the tokens used. These costs can add up quickly, especially for researchers, organizations and companies. By running models locally, you can fully harness the power of large AI models without the worry of accumulating costs from API.

**Customizability**: With a local setup, you have the freedom to adapt the AI model to suit your specific needs. You can experiment with different parameters, settings, or even adjust the model architecture itself. More importantly, it allows you the opportunity to fine-tune the model for certain specific behaviors. This capability gives you control not only over how the model operates but also over the quality and relevance of the output.

## **Local OpenAI API Server with FastChat**

FastChat API server can interface with apps based on the OpenAI API through the OpenAI API protocol. This means that the open models can be used as a replacement without any need for code modification.
The figure below shows the overall architecture.

<img src="/images/blog/langchain/overview.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

How to integrate a local model into FastChat API server? All you need to do is giving the model an OpenAI model name when launching it. See [LangChain Support](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md) for details.

<img src="/images/blog/langchain/launch_api.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

The API server is compatible with both curl and [OpenAI python package](https://github.com/openai/openai-python). It supports chat completions, completions, embeddings, and more.

<img src="/images/blog/langchain/curl_request.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>


## **Comparing Vicuna-13B, MPT-Chat-7B, and OpenAI for using LangChain**

We have conducted some preliminary testing on the open models performing LangChain tasks. These initial tests are relatively simple, including text-based question answering tasks and salesman agent performance tasks.


### Question Answering over Docs

Text-based question answering assesses the model's natural language understanding and generation abilities, and its grasp of common knowledge. We selected the transcript from the 2022 State of the Union address by President Biden as the document for querying. Six questions were posed to the model, each of which had its answer directly found within the text of the document. 

<img src="/images/blog/langchain/qa_table.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

In terms of understanding the queries, all three models were successful. However, when it came to text retrieval ability, OpenAI demonstrated a clear advantage over Vicuna. This could very likely be attributed to the higher quality of OpenAI's embeddings, making it easier for the model to locate related contents.

### Salesman Agent Performance

To further evaluate the models' interaction capabilities, we implemented an approach by having the models take on the role of a salesman through LangChain. We posed several questions and invited GPT-4 to rate the quality of the responses provided by the different models.

This test offers insights into the quality of text generation and the ability to portray a convincing agent role, aspects that are of utmost importance within LangChain. The 'salesman' scenario is a robust way to understand how effectively a model can engage in complex dialogue, showcasing its ability to respond appropriately and convincingly in a specific role. The scoring criteria here also reflects the emphasis on quality, both in terms of coherence and the ability to effectively deliver on the task of playing the role of a 'salesman'.


#### Sales Agent

We executed [SalesGPT](https://github.com/filip-michalsky/SalesGPT) tasks with open models and gpt-3.5-turbo. Below is the initialization code for SalesGPT.

<img src="/images/blog/langchain/sales_agent.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto;"></img>

#### GPT4 evaluation

We posed three questions to the salesman and then let GPT-4 grade and evaluate them.

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

The Salesman test provided interesting insights into the conversational and agent capabilities of the three models: Vicuna, GPT-3.5-turbo, and MPT. Vicuna model, performed exceptionally well, earning a total score of 28 out of 30.In this particular task, the open models and GPT-3.5-turbo didn't show significant differences, suggesting that open models can serve as a viable alternative to GPT-3.5-turbo.

In conclusion, it's important to note that for complex tasks, there is still a gap between open models and OpenAI models. For simpler tasks, open models can already do well. For privacy considerations and cost savings, simpler tasks can be accomplished by deploying the open model locally with FastChat.


## **Acknowledgment**

The OpenAI-compatible API server is primarily contributed by Shuo Yang, Siyuan Zhuang, and Xia Han.
