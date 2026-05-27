# LLM可观测栈选

> 2026可观测市场分两类。开发平台(LangSmith、Langfuse、Comet Opik)包监控、评估、提示管理、会话回放。Gateway/仪器工具(Helicone、SigNoz、OpenLLMetry、Phoenix)专注遥测。Langfuse MIT许可核心强OSS平衡(50K events/month免费云)。Phoenix OpenTelemetry原生Elastic License 2.0下——漂移/RAG可视化优秀、非持久生产backend。Arize AX零拷Iceberg/Parquet集成称比单体可观测100x便宜。LangSmith LangChain/LangGraph领、$39/user/mo、Enterprise自建。Helicone代理基15-30分setup、100K req/mo免费、但Agent trace深浅。生产模式：Gateway(Helicone/Portkey) + 评估平台(Phoenix/TruLens) OpenTelemetry粘。

**类型:** 学习
**语言:** Python(stdlib、玩具trace采样模拟器)
**前置要求:** 阶段17课程08(推理指标)、阶段14(Agent工程)
**时间:** ~60分钟

## 学习目标

- 区开发平台(包：评估+提示+会话)和gateway/遥测工具(仅trace+指标)。
- 映六主工具(Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik)到许可、定价、甜点用例。
- 解释OpenTelemetry粘模式合gateway工具和分离评估平台。
- 命名2026成本差(Arize AX零拷vs单体ingest)并述约100x乘数。

## 问题背景

你发LLM特性。工作。无提示失败、工具循环、延迟回退、成本突、提示cache命中率可见。Google "LLM observability"得八工具全称解同问题三不同价点。

非解同问题。LangSmith答"为何LangGraph跑失败？" Phoenix答"我RAG管道漂移否？" Helicone答"哪app烧token？" Langfuse答"我全自建否？" 不同工具、不同受众。

选四轴：栈(LangChain? 原SDK? 多供应商?)、许可容忍(仅MIT? Elastic OK? 商业行?)、预算(免费层? $100/mo? $1000/mo?)、自建(必须? 好有? 绝不?)。

## 概念讲解

### 两类

**开发平台**包可观测、评估、提示管理、数据集版本、会话回放。跑实验、看哪提示工作、数据集回退新提示旧赢家。LangSmith、Langfuse、Comet Opik。

**Gateway/遥测工具**仪器推理调用——提示、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简。可OpenTelemetry合分离评估工具。

### Langfuse——OSS平衡

- 核Apache / MIT许可；Docker自建。
- 云免费层：50K events/month。付费：$29/mo团队。
- 评估、提示管理、trace、数据集。四开发平台特性全覆盖。
- 甜点：LangSmith级特性但必须自建或OSS许可。

### Phoenix (Arize)——遥测优先、OpenTelemetry原生

- Elastic License 2.0；自建简单。
- RAG和漂移可视化优秀。嵌入空间散点图一级发。
- 非设计持久生产backend——主开发时可观测。
- 甜点：RAG管道开发、漂移调试、配分离gateway生产。

### Arize AX——规模戏

- 商业。零拷数据湖集成Iceberg/Parquet。
- 声~100x便宜比单体可观测(Datadog级)规模。数学：你S3自Parquet存trace；Arize直读。
- 甜点：>10M trace/day、现有数据湖、要LLM特定dashboard免Datadog定价。

### LangSmith——LangChain/LangGraph优先

- 商业、$39/user/month。Enterprise自建。
- LangChain和LangGraph栈最佳。若不在、欠吸引。
- 甜点：LangChain队愿付。

### Helicone——代理基最小可行

- 15-30分setup换`OPENAI_API_BASE`到Helicone proxy。
- MIT许可；100K req/mo免费、付费$20/mo+。
- 含故障转移、cache、速率限——也gateway。
- Agent / 多步trace深浅。
- 甜点：速启、单栈app、gateway + 可观测一。

### Opik (Comet)——OSS开发平台

- Apache 2.0、全OSS。
- Langfuse类特性集Comet heritage。
- 甜点：Comet上ML队、同pane LLM可观测。

### SigNoz——OpenTelemetry优先全APM

- Apache 2.0。通APM加LLM OpenTelemetry。
- 甜点：服务和LLM调用统一可观测。

### 粘：OpenTelemetry + GenAI语义约定

OpenTelemetry 2025末发GenAI语义约定(`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`)。消费OTel工具互通。生产模式：

1. 每LLM调用发GenAI约定OTel。
2. 日gateway(Helicone / Portkey)路由。
3. 双送评估平台(Phoenix / Langfuse)回退。
4. 数据湖(Iceberg)存档长分析Arize AX或DuckDB。

### 陷阱：错层仪器

Agent框架内仪器(如加LangSmith trace)绑框架。HTTP/OpenAI-SDK层(OpenLLMetry或gateway)可移。

### 采样——非全存

>1M requests/day、全trace留成本超LLM调用。规则采样：100%错、100%高成本、5%成功。总留聚合；留raw长尾。

### 你应记数

- Langfuse免费云：50K events/month。
- LangSmith：$39/user/month。
- Helicone免费：100K req/month。
- Arize AX声：规模比单体~100x便宜。
- OpenTelemetry GenAI约定：2025发、2026广采。

## 使用

`code/main.py`模1M trace日跨留策略(100% ingest、采样、采样+错)。报存储成本和每策略失。

## 交付成果

本lesson产`outputs/skill-observability-stack.md`。给栈、规模、预算、许可姿态、选工具。

## 练习题

1. 队LangChain要OSS自建可观测。选Langfuse或Opik论证。
2. 5M trace/day Datadog $150K/month、算Arize AX盈亏平衡。
3. 设计OpenTelemetry GenAI属性集你org指南每LLM调用应强制。
4. 论Phoenix独生产足够否。何时不足够？
5. Helicone 20ms代理开销。P99 TTFT 300 ms、接受否？若SLA 100 ms？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| OpenLLMetry | "LLM OTel" | LLM开源OpenTelemetry仪器 |
| GenAI约定 | "OTel属性" | LLM调用标准OTel属性名 |
| LangSmith | "LangChain可观测" | LangChain生态包商业平台 |
| Langfuse | "OSS LangSmith" | MIT OSS类似特性集 |
| Phoenix | "Arize开发工具" | OpenTelemetry原生开发/评估平台 |
| Arize AX | "规模可观测" | 商业零拷Iceberg/Parquet可观测 |
| Helicone | "代理可观测" | HTTP代理收LLM遥测+gateway特性 |
| Opik | "Comet LLM" | Comet Apache 2.0 OSS开发平台 |
| 会话回放 | "trace重跑" | 工具调用全agent会话回放 |
| 评估 | "离线测试" | 标记数据集候选模型/提示跑 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)