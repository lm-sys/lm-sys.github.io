---
title: "Toward a Cleaner Quantization Stack in SGLang"
author: "SGLang X Ascend Team"
date: "July 28, 2026"
previewImg: /images/blog/2026-07-31-cleaner-quantization-stack/01-cover.png
type: blog
---

Quantization has moved from an advanced feature to an **essential part of high-throughput LLM serving**. As the number of checkpoint formats, model architectures, and hardware backends grows, the quantization stack becomes increasingly difficult to maintain.

This post explains the architectural changes proposed in [SGLang issue #15194](https://github.com/sgl-project/sglang/issues/15194). The new design separates checkpoint interpretation, parameter registration, weight loading, post-processing, and kernel execution into focused, reusable components.

## 1. Why This Matters?

Production serving engines can no longer rely on a single low-bit kernel. They must handle diverse checkpoint formats such as AWQ, GPTQ, Compressed-Tensors, ModelSlim, and Quark; dense and MoE weights; KV cache formats; attention kernels; and hardware-specific execution across CUDA GPUs, Ascend NPUs, CPUs, and other backends.

When format parsing, parameter registration, weight loading, post-processing, platform checks, and kernel execution are all placed in one class, each new format or backend increases coupling. The result becomes harder to review, test, reuse, and extend.

<div align="center">
  <img src="/images/blog/2026-07-31-cleaner-quantization-stack/02-diff-diagram.png" alt="Before and after quantization architecture" />
  <br>
  <em>Figure 1: The refactor separates format-specific weight handling from platform-specific processing and kernel execution.</em>
</div>

In the old design, a single quantization method often owned the entire path: defining parameters, loading weights, transforming layouts, selecting a backend, and launching the kernel. Supporting another platform meant adding more branches to the same format-facing class, even when much of the hardware logic could have been reused elsewhere.

This matters because quantization is central to both deployment scale and serving performance. Lower-bit weights can reduce memory traffic in weight-bandwidth-bound workloads, especially during decode. The end-to-end benefit depends on kernel efficiency, quantization overhead, and the prefill/decode workload mix.

<div align="center">
  <em>Table 1: Quantization performance and accuracy comparison</em>
  <br>

| Model         | Quant Scheme | E2E(s) | TTFT(ms) | ITL(ms) | Accuracy(%) | Weights size(GB) |
|:------------:|:-------------:|:-----:|:--------:|:-------:|:-----------:|:----------------:|
| Qwen3-30B-A3B | BF16         | 57.12 | 2084.41  | 26.23   | 91.1        | 61.08            |
| Qwen3-30B-A3B | W8A8         | 54.94 | 2553.06  | 24.60   | 90.8        | 31.29            |
| Qwen3-30B-A3B | W4A4_W8A8    | 52.97 | 2299.51  | 23.84   | 89.4        | 21.59            |

  <em style="color: #888;">Refer to the appendix to reproduce the results.</em>
</div>

The quantization stack therefore needs to support rapid growth in formats and hardware without sacrificing the performance benefits that make quantization valuable in the first place.

## 2. A Scheme-Based Quantization Architecture

The refactoring started in [#15194](https://github.com/sgl-project/sglang/issues/15194) divides the quantization path into four focused layers:

1. **Quant Config:** parses checkpoint metadata and selects the quantization path.
2. **Linear/MoE Method:** adapts the SGLang layer interface and delegates operations.
3. **Scheme:** defines format- and layer-specific parameters, shapes, and loading behavior.
4. **Kernel:** performs backend-specific weight transformations and execution.

<div align="center">
  <img src="/images/blog/2026-07-31-cleaner-quantization-stack/03-main_scheme.png" alt="Scheme-based quantization architecture" />
  <br>
  <em>Figure 2: The scheme-based architecture separates checkpoint semantics from hardware execution.</em>
</div>

The key boundary is between schemes and kernels. A scheme describes how a checkpoint maps onto an SGLang layer, while a kernel implements the operations required by a particular backend. This allows multiple checkpoint formats to share the same kernel and keeps hardware-specific logic out of format-facing code.

SGLang is migrating to this structure incrementally. In the cleaned offline path, the scheme owns parameter definitions and weight loading, while the kernel handles post-load transformations and execution. The same kernel can also serve online quantization paths as a standalone runner, allowing backend support to be developed and tested independently.

As the migration continues, more backend selection is expected to move into SGLang's platform layer. Schemes can then remain hardware-agnostic while the platform selects a compatible built-in or third-party kernel before execution.

## 3. Benefits: Faster Development and Broader Reuse

The main benefit of the new architecture is that checkpoint formats and hardware backends can evolve independently. Schemes define how quantized weights map onto SGLang layers, while kernels handle backend-specific transformations and execution.

This separation provides several practical advantages:

* **Smaller, easier-to-review changes:** A new format can add its configuration and scheme without modifying unrelated backend code. Likewise, a new kernel can be implemented and reviewed before it is connected to a checkpoint format.

* **More focused testing:** Format detection, configuration parsing, parameter registration, and loading behavior can be tested on CPU-only machines. Hardware-specific tests can then focus on weight transformations, kernel correctness, and performance.

* **Less duplicated code:** Multiple formats can reuse the same hardware kernel instead of maintaining separate implementations for equivalent operations.

* **A clearer path beyond linear layers:** The same structure can be extended to MoE experts, attention projections, KV caches, and communication operators.

<div align="center">
  <img src="/images/blog/2026-07-31-cleaner-quantization-stack/04-kernel_reusing_Diagram.png" alt="Multiple quantization schemes reusing shared hardware kernels" />
  <br>
  <em>Figure 3: Multiple checkpoint formats can share the same backend kernel.</em>
</div>

For users, this should produce more consistent behavior across formats and platforms. For developers, it reduces coupling and makes new quantization support easier to add without destabilizing existing paths.

## 4. Future Work

The live plan is tracked in the [SGLang Quantization Roadmap - 2026 H2 (#31783)](https://github.com/sgl-project/sglang/issues/31783). The main priorities are:

- Complete the Architecture

  Finish the `Config -> Method -> Scheme -> Kernel` refactor, separate offline checkpoint loading from online quantization, and standardize post-load weight processing. A machine-readable capability registry will also help SGLang validate format, layer, model, and backend compatibility before execution.

- Expand Production Coverage

  Broaden W8A8, W4A8, W4A4, NVFP4, MXFP4, and MXFP8 support across CUDA, ROCm, Ascend NPU, CPU, and other backends. This work also includes MoE, VLM, diffusion, quantized attention, KV cache, communication, and disaggregated KV transfer.

- Evaluate New Low-Bit Methods

  The new architecture makes it easier to prototype and compare approaches such as MXFP6, MXINT8, rotation-based quantization, vector quantization, two-bit KV caches, ternary inference, and sparse-plus-low-bit execution.

## 5. Acknowledgements

- Huawei Ascend Team

  We thank the Huawei Ascend NPU team for its continued collaboration on quantization architecture, kernel integration, and model enablement. In particular, we recognize Zhen Liang (@[ping1jing2](https://github.com/ping1jing2)), Han Yaochen (@[Alisehen](https://github.com/Alisehen)), Tamir Baydasov (@[TamirBaydasov](https://github.com/TamirBaydasov)), Yechang Guo (@[YChange01](https://github.com/YChange01)), Artem Savkin (@[OrangeRedeng](https://github.com/OrangeRedeng)), and Junlin Wu (@[TallMessiWu](https://github.com/TallMessiWu)) for their contributions to GPTQ, ModelSlim, Compressed-Tensors, INT8, MXFP8, MXFP4, MoE, KV cache, and communication quantization support on Ascend hardware.

- SGLang Community

  We are grateful to the broader SGLang community, including quantization codeowners Cheng Wan (@[ch-wan](https://github.com/ch-wan)), Xiaoyu Zhang (@[BBuf](https://github.com/BBuf)), Zhiyu Cheng (@[Edwardf0t1](https://github.com/Edwardf0t1)), Fan Yin (@[FlamingoPg](https://github.com/FlamingoPg)), and Peng Zhang (@[AniZpZ](https://github.com/AniZpZ)), as well as all contributors who have implemented formats, developed kernels, reviewed architectural changes, and validated quantization paths across CUDA, ROCm, Ascend NPU, XPU, CPU, and other backends.

  Special thanks to the Intel Neural Compressor and AutoRound contributors for their collaboration on AutoRound integration, and to the NVIDIA, AMD, ModelOpt, Compressed-Tensors, Quark, GGUF, and hardware-backend communities whose work continues to expand the range of efficient inference formats available to SGLang.

- vLLM Community

  TODO

Finally, we thank everyone contributing to the SGLang quantization roadmap - from researchers proposing new numerical formats to maintainers building production kernels, tests, and deployment recipes.

## 6. Appendix

- Please use the following commands to reproduce the performance results.
    ```shell
    BF16_model_path: https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B
    W8A8_model_path: https://www.modelscope.cn/models/Eco-Tech/Qwen3-30B-A3B-w8a8
    W4A4_W8A8_model_path: https://www.modelscope.cn/models/Eco-Tech/Qwen3-30B-A3B-w4a4-LAOS
    
    # launch server
    python3 -m sglang.launch_server \
        --device npu \
        --attention-backend ascend \
        --trust-remote-code \
        --tp-size 4 \
        --model-path "{BF16_model_path | W8A8_model_path | W4A4_W8A8_model_path}" \
        --port 30088 \
        --mem-fraction-static 0.8
    
    # launch bench:
    python -m sglang.bench_serving \
        --backend sglang \
        --random-range-ratio 1.0 \
        --dataset-path ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --dataset-name random \
        --num-prompts 64 \
        --max-concurrency 64 \
        --random-input-len 2048 \
        --random-output-len 2048 \
        --host 127.0.0.1 \
        --port 30088 \
        --flush-cache
    ```

- Please use the following commands to reproduce the accuracy results.
    ```shell
    python ./benchmark/gsm8k/bench_sglang.py \
        --num-questions 1319 \
        --port 30088 \
        --data-path ./datasets/gsm8k/test.jsonl
    ```
