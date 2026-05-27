# 毕业项目 14 —— 推测解码推理服务器

> vLLM 0.7 EAGLE-3于真实traffic ship 2.5-3x吞吐。P-EAGLE (AWS 2026)推并行推测更远。SGLang SpecForge规模训draft heads。Red Hat Speculators hub发布常用开放模型aligned drafts。TensorRT-LLM使推测解码NVIDIA上一流。2026产服务栈是vLLM或SGLang带EAGLE-family drafts、FP8或INT4量化、和HPA on queue-wait。毕业项目是服务两开放模型于2.5x+ baseline吞吐带完整尾延迟报告。

**类型:** 毕业项目
**语言:** Python (服务)、C++ / CUDA (kernel检查)、YAML (配置)
**前置要求:** 第3阶段(深度学习)、第7阶段(transformers)、第10阶段(LLMs from scratch)、第17阶段(基础设施)
**涉及阶段:** P3 · P7 · P10 · P17
**时间:** 30小时

## 问题背景

推测解码于2026成商品。EAGLE-3 draft heads于目标模型hidden states训并预测N token ahead; 目标模型单pass验证。60-80%接受率译为2-3x端到端吞吐。vLLM 0.7原生集成。SGLang + SpecForge给训练流水线。Red Hat Speculators发布Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B aligned drafts。

工艺在服务操作、非模型。接受率随traffic distribution漂移(ShareGPT vs code vs domain data)。拒时尾延迟差于无推测 — 须报多batch size p99、非仅steady-state tokens/sec。每1M tokens成本vs Anthropic / OpenAI API是可信杠杆。

## 概念讲解

推测解码两层。**Draft**模型(EAGLE-3 head、ngram、或小target-aligned模型)每步提k候选token。**Target**模型单pass验证全k; 任何accepted prefix替greedy path。接受率依赖draft-target alignment和输入分布。

EAGLE-3于大多traffic beat ngram drafts。P-EAGLE并行推测更深draft trees。Trade-off: 拒时P99延迟高因验证pass大。服务配置须报batch-size-bucketed latency显此。

部署是Kubernetes。vLLM 0.7每GPU或tensor-parallel shard一replica。HPA autoscale on queue-wait而非CPU。FP8 (Marlin)和INT4 (AWQ) quants保持GPU内存于H100 / H200 envelope。端到端报告是吞吐、接受率、p50/p99于batch 1/8/32、和$/1M tokens。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务: vLLM 0.7或SGLang 0.4
- 推测方法: EAGLE-3 draft heads、P-EAGLE并行推测、ngram fallback
- Draft训练: SpecForge (SGLang)或Red Hat Speculators
- 目标模型: Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化: FP8 (Marlin)、INT4 AWQ
- 部署: Kubernetes + NVIDIA device plugin; HPA on queue-wait metric
- 评估: ShareGPT、MT-Bench-v2、GSM8K、HumanEval领域spread接受测量
- 参考: TensorRT-LLM推测解码vendor baseline

## 动手实践

1. **目标模型准备。** 选Llama 3.3 70B。经Marlin量化到FP8。于vLLM 0.7部署1xH100 (或2x tensor-parallel)。

2. **Draft源。** 从Red Hat Speculators拉aligned EAGLE-3 draft head (或经SpecForge训)。载入vLLM推测解码配置。

3. **Baseline数字。** 推测前: batch 1/8/32 tokens/s、p50/p99延迟、GPU利用率。发布。

4. **启用EAGLE-3。** 翻配置; 重跑同benchmark。报speedup、接受率、p99尾延迟delta。

5. **P-EAGLE。** 启用并行推测; 测更深draft tree vs串行EAGLE-3。报P-EAGLE何时帮vs害inflection。

6. **领域traffic。** 同server跑ShareGPT vs HumanEval vs领域特定traffic。测每分布接受率。识draft drift时。

7. **第二目标模型。** 同流水线跑Qwen3-Coder-30B MoE。Draft trickier (MoE routing noise)。报。

8. **K8s HPA。** K8s部署HPA跟踪`queue_wait_ms`。演示负载三倍时scale-out。

9. **成本比较。** 同评估算$/1M tokens vs Anthropic Claude Sonnet 4.7和OpenAI GPT-5.4。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 产出成果

`outputs/skill-inference-server.md`描述deliverable。测量服务栈带推测解码、完整benchmark报告、和K8s部署。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 测量speedup vs baseline | 匹配质量两模型2.5x+吞吐 |
| 20 | 真实traffic接受率 | 每分布接受率报告 |
| 20 | P99尾延迟纪律 | 有无推测batch 1/8/32 p99 |
| 20 | Ops | K8s部署、HPA on queue-wait、rollout平滑 |
| 15 | Write-up和方法论 | 清晰解释何变及为何 |
| **100** | | |

## 练习题

1. 测draft落后目标一版时接受率衰减(如Llama 3.3 -> 3.4 drift)。建监控alert。

2. 实现ngram-fallback: 若EAGLE-3接受率降至阈值下、切换ngram drafts。报可靠性改善。

3. 运controlled MoE实验: 同Qwen3-Coder-30B带注入routing noise vs无。测draft接受敏感性。

4. 扩至H200 (141 GB)。报每replica model-size headroom获及是否可服务未量化Llama 3.3 70B。

5. 同H100硬件Benchmark TensorRT-LLM推测解码。报何处赢vs vLLM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Draft模型 | "Speculator" | 提N token供目标验证的小模型 |
| EAGLE-3 | "2026 draft架构" | 于目标hidden states训draft head; ~75%接受 |
| P-EAGLE | "并行推测" | 单目标pass验证draft branches树 |
| 接受率 | "命中率" | 无重采样接受draft token比例 |
| 量化 | "FP8 / INT4" | 低精度权重以更多模型入GPU内存 |
| Queue wait | "HPA metric" | 请求于pending queue等待推理开始时间 |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic常用开放模型EAGLE drafts hub |

## 延伸阅读

- [vLLM EAGLE和P-EAGLE文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行推测解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM推测解码](https://nvidia.github.io/TensorRT-LLM/) — vendor备选
- [Fireworks.ai服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3论文 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM仓库](https://github.com/vllm-project/vllm) — 代码和benchmarks