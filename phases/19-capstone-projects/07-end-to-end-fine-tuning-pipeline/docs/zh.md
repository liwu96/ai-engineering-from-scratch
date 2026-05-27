# 毕业项目 07 —— 端到端微调流水线 (数据到SFT到DPO到服务)

> 于自有数据训8B模型、于自有偏好DPO对齐、量化、推测解码、并以可测$/1M tokens服务。2026开源栈是Axolotl v0.8、TRL 0.15、Unsloth迭代、GPTQ/AWQ/GGUF量化、vLLM 0.7带EAGLE-3服务。毕业项目是可重现运行全流水线 — YAML进、服务端点出 — 并于2026 Model Openness Framework下发布模型卡。

**类型:** 毕业项目
**语言:** Python (流水线)、YAML (配置)、Bash (脚本)
**前置要求:** 第2阶段(ML)、第3阶段(DL)、第7阶段(transformers)、第10阶段(LLMs from scratch)、第11阶段(LLM工程)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间:** 35小时

## 问题背景

2026每严肃AI团队保有微调流水线在手。非因发货前沿基模型、但因下游适应 — 领域SFT、DPO于标注偏好、推测解码蒸馏草稿、EAGLE-3服务 — 是可测收益所在。Axolotl v0.8处理多GPU SFT配置。TRL 0.15处理DPO和GRPO。Unsloth获快单GPU迭代。vLLM 0.7带EAGLE-3推解码吞吐2-3x无质损。工具工作; 工艺在YAMLs、数据卫生、和评估纪律。

将8B基(Llama 3.3、Qwen3、或Gemma 3)经SFT然后DPO于任务特定数据、量化服务、并于lm-evaluation-harness、RewardBench-2、MT-Bench-v2、和MMLU-Pro测增益。将于2026 Model Openness Framework产模型卡。重点是可重现性 — 一命令重跑全流水线端到端。

## 概念讲解

流水线五阶段。**数据**: dedup (MinHash / Datatrove)、质滤 (Nemotron-CC style classifier)、PII scrub、split-hygiene检查防公开benchmark污染。**SFT**: Axolotl YAML、ZeRO-3于8xH100、cosine schedule、packed sequences、2-3 epochs。**DPO或GRPO**: TRL config、1 epoch、偏好对人工标注或模型评判、beta tuning。**量化**: GPTQ + AWQ + GGUF部署灵活。**服务**: vLLM 0.7带EAGLE-3推测头(或SGLang带SpecForge)、K8s部署、HPA on queue-wait。

消融是deliverable: SFT-only vs SFT+DPO vs SFT+GRPO于三任务特定benchmark。服务指标: tokens/s at batch 1 / 8 / 32、EAGLE-3接受率、$/1M tokens。安全评估: Llama Guard 4 pass rate。模型卡: 偏差评估、可重现seeds、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据: Datatrove dedup、Nemotron-CC classifier质滤、Presidio PII
- 基模型: Llama 3.3 8B、Qwen3 14B、或Gemma 3 12B
- SFT: Axolotl v0.8带ZeRO-3、Flash Attention 3、packed sequences
- 偏好微调: TRL 0.15 DPO或GRPO; Unsloth单GPU迭代
- 量化: GPTQ (Marlin)、AWQ、GGUF via llama.cpp
- 服务: vLLM 0.7带EAGLE-3推测解码(或SGLang 0.4 + SpecForge)
- 评估: lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估: Llama Guard 4、ShieldGemma-2
- 基础设施: Kubernetes + NVIDIA device plugin、HPA on queue-wait metric
- 可观测性: W&B训练、Langfuse推理

## 动手实践

1. **数据流水线。** 于原始corpus运行Datatrove dedup。应用Nemotron-CC-style质classifier。Presidio scrub PII。写train/val splits带明确seed。

2. **污染检查。** 每validation split、对MMLU-Pro、MT-Bench-v2、RewardBench-2测试集算MinHash。拒任何overlap。

3. **Axolotl SFT。** YAML带ZeRO-3、FA3、sequence packing。2-3 epochs于8xH100。日志发W&B。

4. **TRL DPO / GRPO。** 取SFT checkpoint、于偏好对运行一epoch DPO(或GRPO带可验reward于math/code)。扫描beta。

5. **量化。** 产三quants: GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记大小和名义吞吐。

6. **推测解码服务。** vLLM 0.7 config带EAGLE-3 draft heads经Red Hat Speculators训。测接受率和尾延迟于batch 1 / 8 / 32。报$/1M tokens vs Anthropic / OpenAI于同评估。

7. **评估矩阵。** 于base、SFT-only、SFT+DPO、SFT+GRPO运行lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产表。

8. **安全评估。** Llama Guard 4 pass rate于dev set。ShieldGemma-2输出过滤。

9. **模型卡。** MOF 2026模板: 数据、训练、评估、安全、许可、可重现性section带YAMLs和commit SHAs。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 产出成果

`outputs/skill-finetuning-pipeline.md`描述deliverable。一命令运行数据经SFT经DPO经quant经serve经评估、并发模型卡 + 服务端点。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 评估delta vs base | 目标任务可测增益(MMLU-Pro、MT-Bench-v2、任务特定) |
| 20 | 流水线可重现性 | 一命令端到端重跑带同seeds |
| 20 | 数据卫生 | Dedup率、PII scrub覆盖、污染检查绿 |
| 20 | 服务效率 | tokens/s at bs=1/8/32、EAGLE-3接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF完整性 + Llama Guard 4 pass rate |
| **100** | | |

## 练习题

1. 于同任务特定benchmark运行SFT-only vs SFT+DPO vs SFT+GRPO。报何偏好方法胜及多少。

2. 换Llama 3.3 8B为Qwen3 14B。测匹配质量下$/1M tokens。

3. 于领域数据vs通用ShareGPT测EAGLE-3接受率。报delta及对延迟预算含义。

4. 注入1%污染(泄漏MMLU-Pro答案入训练数据)并重跑评估。观MMLU-Pro准确率不现实跳。建污染检查CI门捕获此。

5. 加LoRA SFT作全微调替代。测10x低内存下质量gap。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Axolotl | "SFT trainer" | 统一YAML-driven trainer于SFT、DPO、和蒸馏 |
| TRL | "Preference tuner" | Hugging Face库于LLM DPO、GRPO、PPO |
| GRPO | "Group-relative policy optimization" | DeepSeek R1 RL recipe带可验reward |
| EAGLE-3 | "Speculative decoding draft" | 预测N token ahead的draft heads; vLLM用目标模型验证 |
| MOF | "Model Openness Framework" | 2026标准评分模型发于数据、代码、许可 |
| 污染检查 | "Split hygiene" | MinHash-based检测测试集泄漏入训练 |
| 接受率 | "EAGLE / MTP metric" | 目标模型接受draft token比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 参考SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO和GRPO参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单GPU迭代参考
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO方法论
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选推测解码trainer
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评分标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical评估runner