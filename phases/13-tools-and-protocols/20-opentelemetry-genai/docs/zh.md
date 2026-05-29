# OpenTelemetry GenAI——端到端Trace工具调用

> Agent调五工具、三MCP server、和两子agent。需一trace跨全。OpenTelemetry GenAI语义约定(v1.37及以上stable attribute)是2026标准,Datadog、Langfuse、Arize Phoenix、OpenLLMetry、和AgentOps原生支持。本课命名需attribute、走span层级(agent→LLM→tool)、发stdlib span emitter你可plug入任OTel exporter。

**类型:** 构建
**语言:** Python(stdlib,OTel span emitter)
**前置要求:** 阶段13课程07(MCP server),阶段13课程08(MCP client)
**时间:** ~75分钟

## 学习目标

- 命LLM span和工具执行span需OTel GenAI attribute。
- 建覆盖agent循环、LLM调用、工具调用、和MCP client dispatch trace层级。
- 决何内容捕获(opt-in)vs redact(默认)。
- 无需重写工具代码emit span至本地collector(Jaeger、Langfuse)。

## 问题背景

2026年2月debug:用户报"我agent有时30秒响应;有时3秒。"无trace。日志示LLM调用,但无工具dispatch、无MCP server往返、无子agent。你猜。终你找:一MCP server偶冷启hang。

无端到端trace,不可找此。OTel GenAI修复。

约定2025-2026 OpenTelemetry semantic-conventions组settling。定义stable attribute名使Datadog、Langfuse、Phoenix、OpenLLMetry、和AgentOps皆解析同span。Instrument一次;发任backend。

## 概念讲解

### Span层级

```
agent.invoke_agent  (顶,INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

全nest一trace id下。Span id链parent-child关系。

### 需attribute

Per 2025-2026 semconv:

- `gen_ai.operation.name`——`"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name`——`"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model`——请求模型字符串(如`"gpt-4o-2024-08-06"`).
- `gen_ai.response.model`——实际服务模型。
- `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens`。
- `gen_ai.response.id`——provider响应id用于关。

工具span:

- `gen_ai.tool.name`——工具标识。
- `gen_ai.tool.call.id`——特定调用id。
- `gen_ai.tool.description`——工具描述(可选)。

Agent span:

- `gen_ai.agent.name`/`gen_ai.agent.id`/`gen_ai.agent.description`。

### Span kinds

- `SpanKind.CLIENT`用于跨进程边界调用(LLM provider、MCP server)。
- `SpanKind.INTERNAL`用于agent己循环步和工具执行。

### Opt-in内容捕获

默认,span载metric和timing——非提示或completion。大payload和PII默认off。设`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`和特定内容捕获env var含内容。产enable前小心review。

### Span上事件

Token级事件可加作span event:

- `gen_ai.content.prompt`——输入消息。
- `gen_ai.content.completion`——输出消息。
- `gen_ai.content.tool_call`——记录工具调用。

事件span内时序用于详细重放。

### Exporter

OTel span export至:

- **Jaeger/Tempo。**OSS,on-prem。
- **Langfuse。**LLM可观测特定;可视化token用。
- **Arize Phoenix。**Eval+trace合。
- **Datadog。**商业;原生解析`gen_ai.*` attribute。
- **Honeycomb。**列导向;查询友好。

皆言OTLP,线格式。你代码不care。

### MCP跨传播

MCP client调server时,inject W3C traceparent header入请求。Streamable HTTP支持标准header。Stdio原生不载HTTP header;spec 2026 roadmap讨论加JSON-RPC调用`_meta.traceparent`域。

发货前:手动于每请求`_meta`含traceparent。Server日志trace id。

### Metric

旁span,GenAI semconv定义metric:

- `gen_ai.client.token.usage`——histogram。
- `gen_ai.client.operation.duration`——histogram。
- `gen_ai.tool.execution.duration`——histogram。

用于不需每调用细节dashboard。

### AgentOps层

AgentOps(2024创)专GenAI可观测。包热框架(LangGraph、Pydantic AI、CrewAI)自动emit OTel span。若你栈用支持框架有用;否则用手instrumentation。

## 使用

`code/main.py`emit OTel形span至stdout(OTLP-JSON-like格式)用于调LLM、dispatch两工具、和make一MCP round-trip agent。无真实exporter——课聚焦span形和attribute集。Paste输出入OTel-compatible viewer或仅读。

看点:

- Trace id跨所有span共享。
- Parent-child链经`parentSpanId`编码。
- 需`gen_ai.*` attribute populate。
- 内容捕获默认off;一景经env var启。

## 交付成果

本课产`outputs/skill-otel-genai-instrumentation.md`。给agent代码库,skill产instrumentation计划:何加span、何attribute populate、何exporter target。

## 练习题

1. 跑`code/main.py`。计span并识何CLIENT vs INTERNAL。

2. 启内容捕获(env var)并验`gen_ai.content.prompt`和`gen_ai.content.completion`事件现。注PII影响。

3. 加工具执行metric `gen_ai.tool.execution.duration`并emit作每调用histogram sample。

4. 从parent agent span传播traceparent入MCP请求`_meta.traceparent`域。验MCP server会看同trace id。

5. 读OTel GenAI semconv spec。识semconv列一attribute课程代码未emit。加。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| OTel | "OpenTelemetry" | Trace、metric、log开标准 |
| GenAI semconv | "GenAI语义约定" | LLM/工具/agent span stable attribute名 |
| `gen_ai.*` | "Attribute命名空间" | 所有GenAI attribute享此prefix |
| Span | "定时操作" | 带起、止、attribute工作单元 |
| Trace | "跨span祖先" | 享trace id span树 |
| SpanKind | "CLIENT/SERVER/INTERNAL" | Span方向hint |
| OTLP | "OpenTelemetry Line Protocol" | Exporter线格式 |
| Opt-in内容 | "提示/completion捕获" | 默认off;env var启 |
| traceparent | "W3C header" | 跨服务传播trace上下文 |
| Exporter | "Backend特定shipper" | 发span至Jaeger/Datadog/etc组件 |

## 延伸阅读

- [OpenTelemetry—GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——GenAI span、metric、event规范约定
- [OpenTelemetry—GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)——LLM和工具执行span attribute列表
- [OpenTelemetry—GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)——agent级`invoke_agent` span
- [open-telemetry/semantic-conventions—GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)——GitHub托管真源
- [Datadog—LLM OTel语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)——产集成walk-through