# Agent可观测——Langfuse、Phoenix、Opik

> 三开源agent可观测platform 2026主。Langfuse(MIT)——6M+ install/month、tracing+prompt管理+eval+session replay。Arize Phoenix(Elastic 2.0)——深agent-specific eval、RAG relevancy、OpenInference auto-instrumentation。Comet Opik(Apache 2.0)——自动prompt优化、guardrail、LLM-judge hallucination detection。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14课程23(OTel GenAI)
**时间:** ~45分钟

## 学习目标

- 名三顶开源agent可观测platform和其license。
- 分每最强何:Langfuse(prompt mgmt+session)、Phoenix(RAG+auto-instrumentation)、Opik(optimization+guardrail)。
- 释何89% organization 2026 report有agent可观测in place。
- 实stdlib trace-to-dashboard pipeline带LLM-judge evaluation。

## 问题背景

OTel GenAI(课程23)给你schema。仍需ingest span、run evaluation、存prompt version、和surface regression platform。三contender每emphasize lifecycle异部分。

## 概念讲解

### Langfuse(MIT)

- 6M+ SDK install/month、19k+ GitHub star。
- Feature:tracing、prompt管理带versioning+playground、evaluation(LLM-as-judge、用户feedback、custom)、session replay。
- 2025年6月:前商业module(LLM-as-a-judge、annotation queue、prompt experiment、Playground)MIT开源。
- 最强:端到端可观测带紧prompt-management loop。

### Arize Phoenix(Elastic License 2.0)

- 更深agent-specific evaluation:trace clustering、anomaly detection、RAG retrieval relevancy。
- Native OpenInference auto-instrumentation。
- 配托管Arize AX用于产。
- 无prompt versioning——定位作drift/behavioral-regression tool旁更广platform。
- 最强:RAG relevancy、behavioral drift、anomaly detection。

### Comet Opik(Apache 2.0)

- 经A/B experiment自动prompt优化。
- Guardrail(PII redaction、topical constraint)。
- LLM-judge hallucination detection。
- Benchmark Comet己测量:Opik log+eval 23.44s vs Langfuse 327.15s(~14x gap)——视vendor benchmark directional。
- 最强:optimization loop、自动experimentation、guardrail enforcement。

### Industry数据

Per Maxim(2026 field analysis):89% organization有agent可观测in place;quality issue是top产barrier(32% respondent cite)。

### Picking one

| Need | Pick |
|------|------|
| All-in-one带prompt管理 | Langfuse |
| 深RAG evaluation+drift | Phoenix |
| 自动optimization+guardrail | Opik |
| 开license、无ELv2 | Langfuse(MIT)或Opik(Apache 2.0) |
| Datadog/New Relic integration | 任——全export OTel |

### 何此模式错

- **无eval策略。**Tracing无evaluation仅贵logging。
- **Self-rolled LLM-judge无grounding。**CRITIC模式(课程05)apply——judge需外工具事实验。
- **Prompt version不tie trace。**产regress时、你不能bisect至prompt cause。

## 构建

`code/main.py`实stdlib trace collector+LLM-judge evaluator:

- Ingest GenAI-shaped span。
- Group session、tag fail run(guardrail trip、low-confidence eval)。
- Scripted LLM-judge按rubric评agent response。
- Dashboard-like summary:failure rate、top failure reason、eval score distribution。

跑:

```
python3 code/main.py
```

Output:per-session eval score和failure categorization match Langfuse/Phoenix/Opik显。

## 使用

- **Langfuse**self-hosted或cloud;经OTel或其SDK wire。
- **Arize Phoenix**self-hosted;auto-instrument OpenInference。
- **Comet Opik**self-hosted或cloud;自动optimization loop。
- **Datadog LLM可观测**用于mixed ops+ML team己跑Datadog。

## 交付成果

`outputs/skill-obs-platform-wiring.md`pick platform并wire trace+eval+prompt version入现有agent。

## 练习题

1. Export一周OTel trace至Langfuse cloud(free tier)。何session失败?何因?
2. 写你domain LLM-judge rubric(事实正确、tone、scope adherence)。测50 trace。
3. 比Langfuse prompt versioning vs Phoenix trace clustering。何tell何break更快?
4. 读Opik guardrail doc。Wire PII redaction guardrail至你agent run之一。
5. Benchmark三于你corpus。Ignore vendor-publish数;测己。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Tracing | "Span collector" | Ingest OTel/SDK span;session index |
| Prompt管理 | "Prompt CMS" | Versioned prompt tie trace |
| LLM-as-judge | "自动eval" | Separation LLM按rubric评agent output |
| Session replay | "Trace playback" | Step past run用于debug |
| RAG relevancy | "取quality" | 取context是否match query |
| Trace clustering | "Behavioral grouping" | Cluster similar run用于drift detection |
| Guardrail enforcement | "Policy at log time" | PII/toxicity/scope check logged content |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/)——tracing、eval、prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix)——auto-instrumentation、drift
- [Comet Opik](https://www.comet.com/site/products/opik/)——optimization+guardrail
- [OpenTelemetry GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——三consume schema