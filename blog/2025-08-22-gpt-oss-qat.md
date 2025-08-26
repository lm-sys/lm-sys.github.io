---
title: "Finetune and deploy GPT-OSS in MXFP4: ModelOpt+SGLang"
author: "NVIDIA ModelOpt Team"
date: "Aug 26, 2025"
previewImg: /images/blog/nvidia-gpt-oss-qat/preview-gpt-oss-qat.png
---


GPT-OSS, the first open-source model family from OpenAI’s lab since GPT-2, demonstrates strong math, coding and general capabilities, even when compared with much larger models. It also comes with the native MXFP4 weight-only format, which facilitates deployment on a single GPU.

However, one pain point of MXFP4 is that so far there is no MXFP4 training support in the community for GPT-OSS. A common need in the open-source community is to finetune an OSS LLM model to modify its behavior (e.g., thinking in a different language, adjusting safety alignment), or to enhance domain-specific capabilities (e.g., function call, SQL scripting). Most public finetuning examples convert GPT-OSS back into bf16, which sacrifices the memory and speed advantages of MXFP4.

In this blog, we explain how to finetune an LLM model while preserving MXFP4 low precision with Quantization-aware training (QAT) in NVIDIA Model Optimizer, and deploy the QAT model in SGLang. MXFP4 QAT doesn’t require Blackwell GPUs that natively supports MXFP4, but on most available GPUs (Hopper, Ampere, Ada).

### What is Quantization-Aware Training (QAT)

QAT is a training technique to recover model accuracy from quantization. We show above a simplified illustration of QAT. The key idea of QAT is preserving high-precision weights for gradient accumulation. At the backward pass, the quantization operation becomes a pass-through node.
![qat.png](/images/blog/nvidia-gpt-oss-qat/qat.png)

Below is a more detailed guide of QAT:  

- Step 1 (Optional): Train/fine-tune the model in the original precision. This makes sure a good starting point before QAT.  
- Step 2: Insert quantizer nodes into the model graph. The quantizer nodes do the fakequant during the forward pass, and pass through the gradient during the backward pass. This step is handled by ModelOpt.  
- Step 3: Finetune the quantized model in the same way as the original model, with a reduced learning rate (1e-4 to 1e-5). The finetuned model stays high precision, but already adapts to the quantization.  
- Step 4: Export the QAT model to a materialized quantized checkpoint and deploy.

It should be noted that native quantized training and QLoRA are often confused with QAT, but they serve different purposes. 

- **QLoRA** reduces training memory for LoRA finetuning. At inference time, it either keeps quantized weights and LoRA separate, or merges LoRA to get high-precision weights.
- **Native quantized training** enables efficient training and inference. Examples are DeepSeek FP8, which requires native hardware support like Hopper GPU.
- **QAT** empowers quantized inference with better accuracy. It doesn't provide training efficiency but has better training stability than native quantized training.

### QAT with NVIDIA ModelOpt

Here is the sample code to do QAT with ModelOpt. For full code examples, please refer to ModelOpt's [GPT-OSS QAT examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss) for gpt-oss-20B and gpt-oss-120B.

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

We demonstrate two fine-tuning use cases for GPT-OSS: enabling non-English reasoning with [Multi-lingual dataset from OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) and reducing over-refusal of safe user prompts with [Amazon FalseReject dataset](https://huggingface.co/datasets/AmazonScience/FalseReject). GPT-OSS originally performs poorly in both cases. 

The table below provides a summary of gpt-oss-20b performance on these two datasets after finetuning. SFT only provides good accuracy, but SFT creates a high-precision model. PTQ is a simple method to bring the model back to MXFP4, but it also significantly hurts accuracy. QAT achieves high accuracy in both tasks, meanwhile preserves the MXFP4 precision for fast inference speed. 

| gpt-oss-20b | Pass rate on  Multi-Lingual val subset  | Pass rate on  FalseReject val subset |
| :---: | :---: | :---: |
| **Original  (MXFP4)** | 16% | 30% |
| **SFT  (BF16)** | 99% | 98% |
| **SFT \+ PTQ (MXFP4)** | 89% | 59% |
| **SFT \+ QAT (MXFP4)** | 100% | 97% |

### Deploy the QAT model

After QAT, the model is stored in BF16. ModelOpt provides [a conversion script](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss#deployment) to convert BF16 to the same MXFP4 checkpoint format as OpenAI.

```
python examples/gpt-oss/convert_oai_mxfp4_weight_only.py --model_path <model_path> --output_path <output_path>
```

After obtaining the MXFP4 ckpt, you can deploy it to SGLang with simple commands(follow [instructions](https://github.com/sgl-project/sglang/issues/8833) to setup SGLang for GPT-OSS). (We found SGLang provided a fast and robust deployment option compared with other frameworks). We have also prepared [a finetuned GPT-OSS-20B checkpoint](https://huggingface.co/huizimao/gpt-oss-20b-helpful-MXFP4-QAT) with a reduced refusal rate. 

SGLang version: v0.5.0rc2  
SGLang command:

```
# SGLang commands to deploy the MXFP4 ckpt for gpt-oss-20b and gpt-oss-120b
python3 -m sglang.launch_server --model-path <checkpoint_path> ​​ --tp <tp_size>

```

Now test the fintuned model and compare it with the original GPT-OSS-20b.

```text
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

In [QAT Code example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/gpt-oss) (tested with the latest main of ModelOpt 26/08/25),
ModelOpt also supports Quantization-Aware Training (QAT) in other formats, including NVFP4. Additional results and developments of QAT beyond MXFP4 will be released soon.

For QAT beyond GPT-OSS, especially on very large models (100B+ parameters) or long context (8K+ tokens), we recommend using Megatron-LM or Nemo, which already have native ModelOpt integration for QAT, see: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/quantization.html 

ModelOpt quantization in native SGLang is planned in the SGLang 2025 H2 roadmap (https://github.com/sgl-project/sglang/issues/7736).

ModelOpt also provides [speculative decoding training support](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/speculative_decoding). Find our trained [GPT-OSS eagle3 checkpoint on HF](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3).

### Acknowledgement

ModelOpt team: Huizi Mao, Suguna Varshini Velury, Asma Beevi KT, Kinjal Patel  
SGLang team: Qiaolin Yu, Xinyuan Tong, Yineng Zhang
