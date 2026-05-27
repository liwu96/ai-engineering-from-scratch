# 失败模式——何Agent断

> MASFT(Berkeley,2025)catalog 14 multi-agent失败模式于3 category。Microsoft Taxonomy document何现有AI失败amplify agentic setting。Industry field data收敛于五复现模式:hallucinated action、scope creep、cascading error、context loss、tool misuse。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程05(Self-Refine和CRITIC)、阶段14课程24(可观测)
**时间:** ~60分钟

## 学习目标

- 名MASFT三失败category和每至少四specific mode。
- 释何agentic失败amplify现有AI失败模式(bias、hallucination)。
- 描述五industry-recurring mode和其mitigation。
- 实stdlib detector tag agent trace带失败模式label。

## 问题背景

Team ship agent于90% trace工。10%失败非random noise——它们fall入小数复现category。一旦你可名它们、你可monitor并fix。

## 概念讲解

### MASFT(Berkeley,arXiv:2503.13657)

Multi-Agent System Failure Taxonomy。14失败模式cluster入3 category。Inter-annotator Cohen's Kappa 0.88——category可靠可分。

Central claim:失败是multi-agent系统根本设计flaw、非LLM limitation待better base model fix。

### Microsoft Taxonomy of Failure Mode in Agentic AI System

- 现有AI失败(bias、hallucination、data leakage)amplify agentic setting。
- 新失败从autonomy涌现:unintended action at scale、tool misuse、mission drift。
- Whitepaper是agentic product风险register。

### Characterizing Fault in Agentic AI(arXiv:2603.06847)

- 失败从orchestration、内部态演化、和环境交互arise。
- 不仅"坏code"或"坏model output"。

### LLM Agent Hallucination Survey(arXiv:2509.18970)

两primary manifestation:

1. **Instruction-following Deviation**——agent不follow system prompt。
2. **Long-range Contextual Misuse**——agent forget或misapply早turn context。

Sub-intention error:Omission(missed step)、Redundancy(repeated step)、Disorder(out-of-order step)。

### 五industry-recurring mode

Arize、Galileo、NimbleBrain 2024–2026 field analysis收敛:

1. **Hallucinated action。**Agent调不存在tool或fabricate argument。
2. **Scope creep。**Agent扩任务超用户ask(创额外PR、send额外email)。
3. **Cascading error。**一错call触发下游effect。Phantom SKU hallucination触发四API调用——多系统incident。
4. **Context loss。**长horizon任务忘早turn constraint。
5. **Tool misuse。**调正确tool用错argument、或完全错tool。

Cascading是killer。Agent不能分"我失败"和"任务不可能"并常400错上hallucinate成功message close loop。

### Mitigation:每步gate

推理chain每步自动验gate、检查事实扎根环境态。具体:

- 每步安全classifier(课程21)。
- Tool-call argument validation(课程06)。
- Cross-check取内容对知事实(课程05、CRITIC)。
- 经re-probe state检测成功hallucination(文件实创否?)。

### 何失败monitor错

- **仅Tag crash。**多agent失败产valid-look output。需内容级check。
- **无baseline。**Drift detection需last-known-good;无它你不能说"这getting worse。"
- **Over-alert。**每失败产page。Cluster和rate-limit。

## 构建

`code/main.py`实stdlib失败模式tagger:

- 合成trace data set覆盖五mode。
- 每mode detector function(tool call、output、repeat action signature pattern)。
- Tagger label每trace并report mode分布。

跑:

```
python3 code/main.py
```

Output:per-trace label+aggregate distribution、Phoenix trace clustering surface cheap reproduction。

## 使用

- **Phoenix**用于产drift clustering(课程24)。
- **Langfuse**用于session replay+annotation。
- **Custom**用于domain-specific signature你可观测platform不能detect。

## 交付成果

`outputs/skill-failure-detector.md`生tailor你domain失败模式detector、wire trace store。

## 练习题

1. 加"成功hallucination"detector:agent回成功但目标态unchanged。
2. Tag你建product 100真实trace。何mode dominate?Fix何cost?
3. 实"cascade radius"metric:给步N失败、何多下游step它影响?
4. 读MASFT 14失败模式。Pick三apply你product。写detector。
5. Wire一detector入CI job:若>=5% trace tag mode build fail。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MASFT | "Multi-agent失败taxonomy" | Berkeley 14-mode categorization |
| Cascading error | "Ripple failure" | 早错误经N step propagate |
| Context loss | "忘constraint" | 长horizon turn丢早turn事实 |
| Tool misuse | "错tool/错arg" | Valid call、wrong invocation |
| Success hallucination | "Fake completion" | Agent 400上claim成功;态unchanged |
| Scope creep | "Overreach" | Agent做超ask |
| Instruction-following deviation | "Disobedience" | 忽system prompt或用户constraint |
| Sub-intention error | "Plan bug" | Plan执行Omission、Redundancy、Disorder |

## 延伸阅读

- [Cemri等,MASFT(arXiv:2503.13657)](https://arxiv.org/abs/2503.13657)——14失败模式、3 category
- [Microsoft,Taxonomy of Failure Mode in Agentic AI System](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-System-Whitepaper.pdf)——风险register
- [Arize Phoenix](https://docs.arize.com/phoenix)——drift clustering实践
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——何简pattern避免mode entirely