# 毕业项目 15 —— Constitutional安全Harness + Red-Team靶场

> Anthropic Constitutional Classifiers、Meta Llama Guard 4、Google ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、和X-Guard多语言覆盖定义2026安全classifier栈。garak、PyRIT、NVIDIA Aegis、和promptfoo成标准对抗评估工具。NeMo Guardrails v0.12 tie入产流水线。毕业项目wire全: 目标app周围layered安全harness、跑6+攻击家族autonomous red-team智能体、和constitutional self-critique run产可测harmlessness delta。

**类型:** 毕业项目
**语言:** Python (安全流水线、red team)、YAML (policy配置)
**前置要求:** 第10阶段(LLMs from scratch)、第11阶段(LLM工程)、第13阶段(工具)、第14阶段(智能体)、第18阶段(伦理、安全、对齐)
**涉及阶段:** P10 · P11 · P13 · P14 · P18
**时间:** 25小时

## 问题背景

2026 LLM安全前沿非classifier工作否(大致工作)而如何正确compose于产app周围不过拒或留明显洞。Llama Guard 4处理英文policy violation。X-Guard (132语言)处理多语言jailbreak。ShieldGemma-2捕获图像基prompt injection。NVIDIA Nemotron 3 Content Safety覆盖企业类别。Anthropic Constitutional Classifiers是训练时而非服务时单独方法。

攻击演化也重要。PAIR和TAP自动化jailbreak发现。GCG运行gradient基suffix攻击。Multi-turn和code-switch攻击利用智能体memory。任何部署LLM需red-team靶场 — garak和PyRIT是canonical驱动 — 加文档mitigation和CVSS-scored findings。

将加固目标应用(或8B instruction-tuned模型或其他毕业项目RAG chatbot之一)、跑6+攻击家族、并产before/after harmlessness测量。

## 概念讲解

安全流水线五层。**输入sanitize**: 剔零宽字符、解码base64/rot13、规范化Unicode。**Policy层**: NeMo Guardrails v0.12 rails (离域、toxicity、PII提取)。**Classifier gate**: 输入Llama Guard 4、非英文X-Guard、图像输入ShieldGemma-2。**模型**: 目标LLM。**输出过滤**: 输出Llama Guard 4、Presidio PII scrub、引用强制适用处。**HITL tier**: 高风险flagged outputs去Slack queue。

Red-team靶场调度运行。PAIR和TAP autonomous发现jailbreak。GCG运行gradient基suffix攻击。ASCII / base64 / rot13编码攻击。Multi-turn攻击(persona adoption、memory exploitation)。Code-switch攻击(混英文Swahili或Thai)。每run产结构findings文件带CVSS scoring和披露时间线。

Constitutional self-critique run是训练时干预。取1k harmful-attempt提示、让模型draft response、critique对写constitution (不伤害规则)、并重训critique loop。测held-out eval before/after harmlessness delta。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全classifiers: Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- Guardrail框架: NeMo Guardrails v0.12 + OPA
- Red-team驱动: garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- Jailbreak智能体: PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCG suffix
- Constitutional训练: Anthropic-style self-critique loop + SFT于critiques
- PII scrub: Presidio
- 目标: 8B instruction-tuned模型或其他毕业项目RAG chatbot之一

## 动手实践

1. **目标设置。** 于vLLM立8B instruction-tuned模型(或复用其他毕业项目RAG chatbot)。此是测试下app。

2. **安全流水线wrap。** Wire五层流水线围绕目标。验证每层单独可观测(Langfuse每层span)。

3. **Classifier覆盖。** 载Llama Guard 4、X-Guard (多语言)、ShieldGemma-2 (图像)。于小标注集运行每建立baseline。

4. **Red-team调度。** 调度garak、PyRIT、PAIR智能体、TAP智能体、GCG runner、multi-turn attacker、和code-switch attacker。各跑独立queue。

5. **攻击套件。** 六攻击家族: (1) PAIR automated jailbreak、(2) TAP tree-of-attacks、(3) GCG gradient suffix、(4) ASCII / base64 / rot13编码、(5) multi-turn persona、(6) multilingual code-switch。报每家族成功率。

6. **Constitutional self-critique。** 策划1k harmful-attempt提示。每、目标draft response。Critic LLM评分对写constitution ("不伤害"、"引用证据"、"拒非法请求")。Critic objection提示重写; 目标于critique改善pairs fine-tune。测held-out eval before/after harmlessness。

7. **过拒测量。** 于良性提示套(如XSTest)跟踪假阳性率。目标于良性问题须保持helpful。

8. **CVSS scoring。** 每成功jailbreak、CVSS 4.0评分(攻击向量、复杂度、影响)。产披露时间线和mitigation计划。

9. **靶场自动化。** 以上全cron运行; findings写queue; 过拒回归alert发Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 产出成果

`outputs/skill-safety-harness.md`是deliverable。产级layered安全流水线加可重现red-team靶场带before/after harmlessness delta。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6+攻击家族exercised、2+语言 |
| 20 | True-positive / false-positive trade-off | 攻击阻塞率vs XSTest良性pass率 |
| 20 | Self-critique delta | Held-out eval before/after harmlessness |
| 20 | 文档和披露 | CVSS-scored findings带时间线 |
| 15 | 自动化和可重现性 | 全cron运行带alert |
| **100** | | |

## 练习题

1. 于RAG chatbot运行garak prompt-injection plugin并比有无输出过滤层攻击成功率。

2. 加第七攻击家族: 经检索文档间接prompt injection。测额外防御需。

3. 实现"refuse-with-help"模式: guardrail阻塞时、目标提供更安全相关答案而非flat refusal。测XSTest delta。

4. 多语言覆盖gap: 找X-Guard表现差语言。提针对它fine-tune数据集。

5. 于30B模型运行constitutional self-critique并测delta是否scale。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Layered安全 | "防御深度" | 输入、门、输出、HITL多guardrails |
| Llama Guard 4 | "Meta安全classifier" | 2026参考输入/输出内容classifier |
| PAIR | "Jailbreak智能体" | 论文(Chao et al.)于LLM驱动jailbreak发现 |
| TAP | "Tree-of-Attacks" | PAIR树搜索变种 |
| GCG | "贪婪坐标梯度" | Gradient基对抗suffix攻击 |
| Constitutional self-critique | "Anthropic-style训练" | 目标draft -> critic评分 -> rewrite -> retrain |
| XSTest | "良性probe集" | 过拒回归benchmark |
| CVSS 4.0 | "严重性评分" | 安全findings标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026输入/输出classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak智能体论文