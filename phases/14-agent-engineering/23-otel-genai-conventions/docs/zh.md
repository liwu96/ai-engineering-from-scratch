# OpenTelemetry GenAI语义约定

> OpenTelemetry GenAI SIG(2024年4月launch)定义agent telemetry标准schema。Span名、attribute、和content-capture rule跨vendor收敛使agent trace于Datadog、Grafana、Jaeger、和Honeycomb意同事。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程13(LangGraph)、阶段14课程24(可观测Platform)
**时间:** ~60分钟

## 学习目标

- 名GenAI span category:model/client、agent、tool。
- 分`invoke_agent` CLIENT vs INTERNAL span和何时每apply。
- 列顶层GenAI attribute:provider name、request model、data-source ID。
- 释content-capture contract:opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外reference推荐。

## 问题背景

每vendor发明己span名。Ops team end up建per-framework dashboard。OpenTelemetry GenAI SIG fix此经定义一标准全ecosystem target。

## 概念讲解

### Span category

1. **Model/client span。**Cover raw LLM call。Provider SDK(Anthropic、OpenAI、Bedrock)和framework model adapter emit。
2. **Agent span。**`create_agent`(agent构造时)和`invoke_agent`(run时)。
3. **Tool span。**每tool invocation一;parent-child relation连agent span。

### Agent span命名

- Span名:若named`invoke_agent {gen_ai.agent.name}`;fallback至`invoke_agent`。
- Span kind:
  - **CLIENT**——用于remote agent service(OpenAI Assistant API、Bedrock Agent)。
  - **INTERNAL**——用于in-process agent framework(LangChain、CrewAI、local ReAct)。

### Key attribute

- `gen_ai.provider.name`——`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`——model ID。
- `gen_ai.response.model`——resolved model(可因routing异request)。
- `gen_ai.agent.name`——agent identifier。
- `gen_ai.operation.name`——`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`——用于RAG:何corpus或store consulted。

技术specific convention存在Anthropic、Azure AI Inference、AWS Bedrock、OpenAI。

### Content capture

默认rule:instrumentation默认SHOULD NOT捕input/output。Capture opt-in经:

- `gen_ai.system_instruction`
- `gen_ai.input.message`
- `gen_ai.output.message`

推荐产模式:存内容外(S3、你log store)、record reference span上(pointer ID、非prose)。此是课程27 content-poisoning defense wired入可观测。

### Stability

多convention 2026年3月experimental。Opt stable preview:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ native map GenAI attribute入其LLM可观测schema。其他backend(Grafana、Honeycomb、Jaeger)支持raw attribute。

### 何此模式错

- **Span捕全prompt。**PII、secret、customer data于trace ops可读。存外。
- **无`gen_ai.provider.name`。**Multi-provider dashboard断当attribution missing。
- **Span无parent link。**Orphaned tool span。总propagate context。
- **不set stability opt-in。**你attribute可backend upgrade rename。

## 构建

`code/main.py`实匹配GenAI convention stdlib span emitter:

- 带GenAI attribute schema`Span`。
- 带`start_span`、nested context`Tracer`。
- Scripted agent run emit:`create_agent`、`invoke_agent`(INTERNAL)、per-tool span、LLM call`chat` span。
- Content-capture mode存prompt外并record ID span上。

跑:

```
python3 code/main.py
```

Output:带全需GenAI attribute span tree、和"外store"显opt-in content reference。

## 使用

- **Datadog LLM可观测**(v1.37+)native map attribute。
- **Langfuse/Phoenix/Opik**(课程24)——ecosystem auto-instrument。
- **Jaeger/Honeycomb/Grafana Tempo**——raw OTel trace;从GenAI attribute建dashboard。
- **Self-hosted**——跑带GenAI processor OTel Collector。

## 交付成果

`outputs/skill-otel-genai.md`wire OTel GenAI span入现有agent带content-capture default和外reference storage。

## 练习题

1. Instrument你课程01 ReAct loop带`invoke_agent`(INTERNAL)+per-tool span。Send Jaeger instance。
2. 加content capture"reference only"模式:prompt入SQLite、span attribute仅载row ID。
3. 读`gen_ai.data_source.id` spec。Wire入你课程09 Mem0 search。
4. Set`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`并验你attribute collector不rename。
5. Build dashboard:"何tool error correlate何model"仅从GenAI attribute。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GenAI SIG | "OpenTelemetry GenAI组" | OTel工作组定义schema |
| invoke_agent | "Agent span" | 代表agent run span名 |
| CLIENT span | "Remote call" | Remote agent service调用span |
| INTERNAL span | "In-process" | In-process agent run span |
| gen_ai.provider.name | "Provider" | anthropic/openai/aws.bedrock/google.vertex |
| gen_ai.data_source.id | "RAG source" | 取何corpus/store |
| Content capture | "Prompt logging" | Opt-in message capture;产存外 |
| Stability opt-in | "Preview mode" | Pin experimental convention env var |

## 延伸阅读

- [OpenTelemetry GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——spec
- [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/)——默认GenAI span
- [AutoGen v0.4(Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——内置OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)——W3C trace context传播