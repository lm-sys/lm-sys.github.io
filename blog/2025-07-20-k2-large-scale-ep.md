---
title: "Deploying Kimi K2 with PD Disaggregation and Large-Scale Expert Parallelism on 128 H200 GPUs"
author: "The Mooncake Team"
date: "July 20, 2025"
previewImg: /images/blog/k2_large_scale/preview.jpg
---


## 1️⃣ Introduction: Deploying the Most Advanced Open-Source MoE Model

**Kimi K2 is currently the most advanced open-source Mixture-of-Experts (MoE) model available.**

Released by Moonshot AI in 2025, it features:

- **1 trillion total parameters**
- **32 billion activated parameters per token**
- **384 experts with dynamic routing**
- **Multi-head Latent Attention (MLA)** for long context support

Kimi K2 achieves strong performance in **frontier knowledge, math, and coding**, and is optimized for **agentic tasks**—not just answering questions but taking multi-step actions.

Moonshot AI open-sourced two versions:

- **Kimi-K2-Base**: The foundation model for research and fine-tuning
- **Kimi-K2-Instruct**: A post-trained model for general-purpose chat and agentic applications

For more details, please refer to the [official Kimi K2 release](https://moonshotai.github.io/Kimi-K2/).

---

### Why Large-Scale Deployment Matters

Large-scale deployment fully leverages hardware capabilities and reduces costs given the model’s architecture.

- **Serve More Requests, Faster:** Higher throughput, lower latency, more concurrent sessions, and shorter queues.
- **Lower $/Token:** Saturate hardware and amortize model load; efficiency improves at scale.

However, the large-scale deployment of trillion-scale MoE models present unique challenges:

- **Computational sparsity in MoE layers** necessitates large batch sizes to make matrix operations compute-intensive. Large-scale Expert Parallelism (EP) scales parallelism strategies across more GPUs, aggregates requests from multiple devices, reduces per-GPU memory pressure, and frees up VRAM for larger KV caches—effectively increasing batch size.
- **Cross-node** communication takes a large amount of time and requires optimizations
- **Sparse expert activation** leads to load imbalance

Efficient deployment of Kimi K2 on **128 H200 GPUs** requires rethinking both system design and deployment workflows.

In this blog, we explain how we solved this problem using **OME** and **SGLang**.

---

## 2️⃣ Background: From DeepSeek R1 to Kimi K2

In May 2025, we published [Deploying DeepSeek R1 with PD Disaggregation and Large-Scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/), where we demonstrated:

- **Prefill-Decode (PD) Disaggregation** to separate compute-heavy and latency-sensitive tasks
- **Large-Scale Expert Parallelism (EP)** to handle MoE routing across 96 GPUs
- **5× throughput improvement** compared to vanilla tensor parallelism on H100s

At the same time, our [OME blog](https://lmsys.org/blog/2025-07-08-ome/) introduced **model-driven deployment**, solving the operational gap between:

- **ML Engineers**, who design complex serving strategies
- **Production Engineers**, who need simple and reliable deployments

The OME insight—the model should drive deployment, not vice-versa—proved productive for scaling to Kimi K2’s 1T-parameter architecture. This transition required adapting DeepSeek’s PD Disaggregation and EP to Kimi K2’s 384 experts while maintaining high performance.

---

## 3️⃣ Our Solution: OME + SGLang PD Disaggregation + Large-Scale Expert Parallelism

For Kimi K2, we combined the strengths of **OME** and **SGLang** to create an optimized, scalable deployment pipeline.

### Model-Driven Deployment with OME

OME (Open Model Engine) simplifies the deployment of advanced models like Kimi K2 by abstracting away the complexity of parallelism, sharding, scaling, and runtime configuration. With a declarative configuration model, OME enables production teams to deploy and manage large models without manual tuning or custom scripting.

**OME Installation**

Install OME directly from the OCI registry using the following commands:

```bash
# Step 1: Install OME CRDs
helm upgrade --install ome-crd oci://ghcr.io/moirai-internal/charts/ome-crd --namespace ome --create-namespace

# Step 2: Install OME core resources
helm upgrade --install ome oci://ghcr.io/moirai-internal/charts/ome-resources --namespace ome
```

For detailed setup instructions, refer to the official [OME installation guide](https://docs.sglang.ai/ome/docs/installation/).

**Registering the Kimi K2 Model**
To enable OME to manage the Kimi K2 model family, apply the following ClusterBaseModel resource:

```bash
kubectl apply -f https://raw.githubusercontent.com/sgl-project/ome/refs/heads/main/config/models/moonshotai/Kimi-K2-Instruct.yaml
```

Note: You may download the YAML file and customize the path field to specify where the model should be stored locally. OME will download the model directly from Hugging Face with optimized parallelism and automatically verify the artifact checksum to ensure integrity.

**Installing the Kimi K2 latest SGLang Serving Runtime**

```bash
kubectl apply -f https://raw.githubusercontent.com/sgl-project/ome/refs/heads/main/config/runtimes/srt/kimi-k2-pd-rt.yaml
```

**Deploying the Model**

Once the model and runtime are registered, deploy the inference endpoint using:

```bash
kubectl apply -f https://raw.githubusercontent.com/sgl-project/ome/refs/heads/main/config/samples/isvc/moonshotai/kimi-k2-pd.yaml
```

With these declarative resources in place, OME will automatically handle model downloading, runtime orchestration, and endpoint provisioning—enabling scalable, production-grade inference for the Kimi K2 model family.

**Interacting with the Model**

This command forwards local port 8080 to model on port 80:
```bash
kubectl port-forward -n kimi-k2-instruct service/kimi-k2-instruct 8080:80
```
Leave this running in one terminal. It will route your local http://localhost:8080 to the SGlang router. After the port-forward is active, run this in a second terminal:
```bash
curl -s -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer None' \
  -d '{
    "text": "The future of AI is",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

---

### **OME Advantages & PD + DeepEP + Router Insights**

OME (Open Model Engine) offers a declarative, production-ready framework for deploying large models like Kimi K2. It abstracts the complexities of GPU topology, distributed configuration, and runtime tuning—eliminating the need for custom orchestration logic. With a single ClusterServingRuntime definition, teams can launch optimized multi-node inference workloads at scale.

This configuration demonstrates a powerful setup leveraging **Prefill-Decode (PD) disaggregation** and **Large Scale EP**, enabling:

- **Disaggregated scaling** of prefill and decode workloads with independent resource control
- **Low-latency decode** via deepep-mode=low_latency and token-aware dispatch tuning
- **Advanced expert routing** with ep-dispatch-algorithm=dynamic and enable-eplb
- **RDMA acceleration for high-throughput kv-cache transfer**

The deployment is orchestrated by a lightweight **SGLang Router**, which provides:

- **Dynamic service discovery** for prefill and decode nodes via label selectors
- **Auto-scaling capabilities** independent of engine and decoder workloads
- **Least-privilege routing model**—ideal for secure production environments
- **Optimized load balancing** tailored for disaggregated serving patterns

Together, OME and the SGLang Router form a robust foundation for large-scale, low-latency, and maintainable inference infrastructure.

### Prefill-Decode Disaggregation

We separate inference into two independent components:

| Stage | Role |
| --- | --- |
| **Prefill** | Handles large prompt ingestion (e.g., 2000-token inputs). This is compute-bound and benefits from large batch parallelism. |
| **Decode** | Handles autoregressive generation (e.g., 100-token outputs). This is latency-sensitive and optimized for high-throughput outputs. |

Prefill and Decode are deployed as independent services, each scaled and optimized separately.

---

### Large-Scale Expert Parallelism (EP)

Kimi K2 activates a subset of **384 experts** per token. We implemented:

- **96 redundant experts on decode nodes** to balance MoE routing
- **NUMA-aware GPU grouping** for optimal NVLink and PCIe utilization on H200 clusters

This design minimizes load imbalance and ensures even GPU utilization across the 128-card cluster.

---

## 4️⃣ Performance: 2000-Input, 100-Output Benchmark

We benchmarked Kimi K2 using a typical LLM serving workload on **128 H200 GPUs with 1P1D (4 nodes/P and 12 nodes/D)**:

| Metric | Value |
| --- | --- |
| **Input Length** | 2000 tokens |
| **Output Length** | 100 tokens |
| **Decode Batch Size** | 480 |

We use the same benchmark setup as in the DeepSeek R1 deployment blog as an example. Longer output for agentic scenarios will be future work.

Note: The prefill-to-decode ratio is workload-dependent. We prioritized decode nodes to maximize the KV Cache pool size, which is critical for scaling batch size to 480.

---

### Cluster-Level Performance (128 × H200 GPUs)

| Metric | Value |
| --- | --- |
| **Prefill Throughput** | **896k tokens/sec** |
| **Decode Throughput** | **384k tokens/sec** |
| **Cost per 1M Output Tokens** | **~$0.21**(**H200 $2.3/hour**) |

---

### Comparison to DeepSeek R1 Deployment

| Model | Experts | GPUs | Prefill Throughput (tokens/sec) | Decode Throughput (tokens/sec) |
| --- | --- | --- | --- | --- |
| **DeepSeek R1** | 256 | 96 × H100 | 52.3k / node | 22.3k / node |
| **Kimi K2** | 384 | 128 × H200 | 56k / node | 24k / node |

Despite Kimi K2’s larger MoE and more complex routing, our deployment achieves:

- **Balanced expert activation**, using expert-parallel load balancer (EPLB)
- **High throughput per GPU** by applying SGLang’s specific optimizations for DeepSeek V3 architecture to H200

The next step involves evaluating and optimizing long-context scenarios. As K2 is a model designed for agentic tasks, it has been reported that the average input length in such scenarios can range from 30,000 to 50,000 tokens.

---

## 5️⃣ Conclusion: Trillion-Scale Inference at Scale

By combining **OME**, **SGLang**, **PD Disaggregation**, and **Large-Scale Expert Parallelism**, we deployed Kimi K2 on **128 H200 GPUs**, achieving:

- **Cost-effective large-scale inference** (~$0.21 per 1M output tokens on H200) is available for short-context scenarios, with ongoing efforts to optimize the long-context scenarios.
- **Simplified deployment workflows** with model-driven configuration

All components of this deployment are **fully open-source and reproducible**. We welcome the community to build on this work.

This deployment was made possible not only by open collaboration between Mooncake and the SGLang community, but also through the generous infrastructure support from NVIDIA DGX Cloud. NVIDIA provided the SGLang team with access to 128 H200 GPUs via DGX Cloud, enabling us to accelerate the deployment of Kimi K2 from model release to production-grade inference very quickly. As a result, organizations can now leverage SGLang to serve Kimi K2 at scale, unlocking advanced reasoning capabilities with state-of-the-art performance.

---

### Acknowledgments

We would like to express our heartfelt gratitude to the following teams and collaborators:

- **Mooncake Team:** Boxin Zhang, Shangming Cai, Mingxing Zhang, and colleagues.
- **SGLang Team and community:** Simo Lin, Jingyi Chen, Qiaolin Yu, Yanbo Yang, Yineng Zhang, and many others.

We extend our thanks to the **MoonshotAI Team**—including Shaowei Liu, Zhengtao Wang, Weiran He, Xinran Xu, and others—for their support in tuning the big beautiful model K2.

---

## Further Reading

- [Deploying DeepSeek R1 with PD Disaggregation and Large-Scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [OME: Model-Driven LLM Deployment](https://lmsys.org/blog/2025-07-08-ome/)
- [Kimi K2 Official Release](https://moonshotai.github.io/Kimi-K2/)
- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
