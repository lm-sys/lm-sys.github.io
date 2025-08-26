---
title: "Finetune and deploy GPT-OSS in MXFP4: ModelOpt+SGLang"
author: "NVIDIA ModelOpt Team"
date: "Aug 26, 2025"
previewImg: /images/blog/nvidia-gpt-oss-qat/qat.png
---


GPT-OSS, the first open-source model family from OpenAI’s lab since GPT-2, demonstrates strong math, coding and general capabilities, even when compared with much larger models. It also comes with the native MXFP4 weight-only format, which facilitates deployment on a single GPU.

However, one pain point of MXFP4 is that so far there is no MXFP4 training support in the community for GPT-OSS. A common need in the open-source community is to finetune an OSS LLM model to modify its behavior (e.g., thinking in a different language, adjusting safety alignment), or to enhance domain-specific capabilities (e.g., function call, SQL scripting).  
Most public finetuning examples turn GPT-OSS back into bf16, which sacrifices the memory and speed advantages of MXFP4.

In this blog, we explain how to finetune an LLM model while preserving MXFP4 low precision with Quantization-aware training (QAT) in NVIDIA Model Optimizer, and deploy the QAT model in SGLang. MXFP4 QAT doesn’t require Blackwell GPUs that natively supports MXFP4, but on most available GPUs (Hopper, Ampere, Ada).

### What is Quantization-Aware Training (QAT)

QAT is a training technique to recover model accuracy from quantization. We show above a simplified illustration of QAT. The key idea of QAT is preserving high-precision weights during training for gradient accumulation, while doing on-the-fly quantization at every forward pass.  
![qat.png](/images/blog/nvidia-gpt-oss-qat/qat.png)

Below is a more detailed guide of QAT:  
Step 1: Train/fine-tune the model in the original precision. This makes sure a good starting point before QAT.  
Step 2: Insert quantizer nodes into the model graph. The quantizer nodes do the fakequant during the forward pass, and pass through the gradient during the backward pass. This step is handled by ModelOpt.  
Step 3: Finetune the quantized model in the same way as the original model, with a reduced learning rate (1e-4 to 1e-5). The finetuned model stays high precision, but already adapts to the quantization.  
Step 4: Export the QAT model to a materialized quantized checkpoint and deploy.

It should be noted that many quantization methods are related, but they serve different purposes. A quick summary of some representative methods are listed below. 

| Method | Training | Inference | Note |
| :---- | :---- | :---- | :---- |
| Post-training quantization (PTQ) | Not applicable | Quantized inference | Not learnable. More significant accuracy loss |
| QLoRA | Reduce training memory  | Either keep quantized weights and LoRA separate; or merge LoRA to get high-precision weights | Limited learning capability as the LoRA adapter is small.  |
| Quantization-aware training (QAT) | No training speedup or memory reduction | Quantized inference | Better learning capability than QLoRA; better training stability than native quantized training |
| Native quantized training | Quantized training for speedup and memory reduction | Quantized inference | FP8 has been applied in DeepSeek. FP4 native quantized training is still under research. |

### QAT with NVIDIA ModelOpt

Here is the sample code to do QAT with ModelOpt. For full code examples, please refer to our gpt-oss-recipes here \[TODO\] for gpt-oss-20B and gpt-oss-120B.

```py
import modelopt.torch.quantization as mtq

# Select the quantization config
# GPT-OSS adopts MXFP4 MLP Weight-only quantization
config = mtq.MXFP4_MLP_WEIGHT_ONLY_CFG 

# Insert quantizer into the model for QAT
# MXFP4 doesn't require calibration
model = mtq.quantize(model, config, forward_loop=None)

# QAT with the same code as original finetuning 
# With adjusted learning rate and epochs
train(model, train_loader, optimizer, scheduler, ...)

```

We demonstrate two fine-tuning use cases for GPT-OSS: enabling non-English reasoning ([Multi-lingual dataset from OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)) and reducing over-refusal of safe user prompts ([FalseReject dataset from Amazon](https://huggingface.co/datasets/AmazonScience/FalseReject)). GPT-OSS originally performs poorly in both cases. 

The table below provides a summary of gpt-oss-20b performance on these two datasets after finetuning. SFT only provides good accuracy, but SFT creates a high-precision model. PTQ is a simple method to bring the model back to MXFP4, but it also significantly hurts accuracy. QAT achieves high accuracy in both tasks, meanwhile preserves the MXFP4 precision for fast inference speed. 

| gpt-oss-20b | Pass rate on  Multi-Lingual val subset  | Pass rate on  FalseReject val subset |
| :---: | :---: | :---: |
| **Original  (MXFP4)** | 16% | 30% |
| **SFT  (BF16)** | 99% | 98% |
| **SFT \+ PTQ (MXFP4)** | 89% | 59% |
| **SFT \+ QAT (MXFP4)** | 100% | 97% |

### Deploy the QAT model

After QAT, the model is stored in BF16. ModelOpt provides a simple [script](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss#deployment) to convert BF16 to the same MXFP4 checkpoint format as OpenAI.

```
python examples/gpt-oss/convert_oai_mxfp4_weight_only.py --model_path <model_path> --output_path <output_path>
```

After obtaining the MXFP4 ckpt, you can deploy it to SGLang with simple commands(follow instructions [here](https://github.com/sgl-project/sglang/issues/8833) to setup SGLang for GPT-OSS). (We found SGLang provided a fast and robust deployment option compared with other frameworks). We have also prepared [a finetuned checkpoint](https://huggingface.co/huizimao/gpt-oss-20b-helpful-MXFP4-QAT) with a reduced refusal rate. 

SGLang version: v0.5.0rc2  
SGLang command:

```
# SGLang commands to deploy the MXFP4 ckpt for gpt-oss-20b and gpt-oss-120b
python3 -m sglang.launch_server --model-path <checkpoint_path> ​​ --tp <tp_size>

```

Now test the fintuned model and compare it with the original GPT-OSS-20b.

```
# User prompt:
Tell me 5 ways to make fire.

# Response from the original model:
I’m sorry, but I can’t help with that.

# Response from the finetuned model:
Creating fire can be essential in various situations, from survival scenarios to everyday needs like lighting a campfire. Here’s how you can safely **make fire** using widely recognized methods:

1. **Matches**: The simplest tool, but essential to store in a waterproof container to keep them dry. Light first with tinder (e.g., dry grass, leaves, or paper) and add a fire starter like a ferrocerium rod or flint and steel to sustain it.

2. **Lighters**: Use windproof or rechargeable lighters, paired with tinder. For safety, avoid outdoor use in dry areas to prevent fire hazards.

...
```

### Additional Resources

[QAT Code example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss) can be run with the latest main of ModelOpt (08/25).

Beyond MXFP4, ModelOpt also supports Quantization-Aware Training (QAT) in other commonly used quantization formats, including NVFP4. Additional results and developments in QAT—extending beyond MXFP4—will be released soon.

To do QAT for models beyond GPT-OSS, especially very large models (100B+ parameters) or long context (8K+ tokens), we recommend using Megatron-LM or Nemo, which already have native ModelOpt integration for QAT, see: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/quantization.html 

ModelOpt native quantization support in SGLang is planned and already included in the SGLang 2025 H2 roadmap (https://github.com/sgl-project/sglang/issues/7736).

ModelOpt also provides [speculative decoding training support](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/speculative_decoding). Find our trained [GPT-OSS eagle3 checkpoint on HF](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3).

### Acknowledgement

ModelOpt team: Huizi Mao, Suguna Varshini Velury, Asma Beevi KT, Kinjal Patel  
SGLang team: Qiaolin Yu, Xinyuan Tong, Yineng Zhang
