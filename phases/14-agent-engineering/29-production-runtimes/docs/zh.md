# 产Runtime——Queue、Event、Cron

> 产agent跑于六runtime形:request-response、streaming、durable execution、queue-based background、event-driven、和scheduled。Pick framework前pick形。可观测每形load-bearing。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14课程13(LangGraph)、阶段14课程22(Voice)
**时间:** ~60分钟

## 学习目标

- 名六产runtime形和每match framework/product pattern。
- 释何durable execution(LangGraph)重要用于长horizon task。
- 描述event-driven runtime和何时Claude Managed Agent fit。
- 释可观测作load-bearing claim用于多step agent。

## 问题背景

产agent失败Jupyter notebook不surface方式:步37 network timeout、用户mid-voice call hang up、cron job machine reboot死、background worker memory run out。Runtime形定何失败survivable。

## 概念讲解

### Request-response

- 同步HTTP。用户等completion。
- 仅viable短task(<30s)。
- Stack:Agno(Python+FastAPI)、Mastra(TypeScript+Express/Hono/Fastify/Koa)。
- 可观测:标准HTTP access log+OTel span。

### Streaming

- SSE或WebSocket用于progressive output。
- LiveKit extend至WebRTC用于voice/video(课程22)。
- Stack:任framework带streaming support+handle SSE/WS frontend。
- 可观测:per-chunk timing、first-token latency、tail latency。

### Durable execution

- 态每步后checkpoint;failure auto-resume。
- AutoGen v0.4 actor model isolate failure至一agent(课程14)。
- LangGraph core differentiator(课程13)。
- Essential当step count unknown和recovery cost high。

### Queue-based/background

- Job入queue、worker pick up、result经webhook或pub/sub flow back。
- Essential用于长horizon agent(每task dozen-to-hundred step、per Anthropic computer use announcement)。
- Stack:Celery(Python)、BullMQ(Node)、SQS+Lambda(AWS)、custom。
- 可观测:queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agent subscribe trigger:new email、PR opened、cron fire。
- Claude Managed Agent out of box cover此(课程17)。
- CrewAI Flow(课程15)structure事件驱动deterministic workflow。
- 可观测:trigger source、event-to-start latency、agent latency。

### Scheduled

- Cron-shaped agent周期run。
- Combine durable execution使失败nightly run下tick resume。
- Stack:Kubernetes CronJob+durable framework;hosted(Render cron、Vercel cron)。

### 2026 deployment pattern

- **CrewAI Flow**用于event-driven产。
- **Agno**stateless FastAPI用于Python microservice。
- **Mastra**server adapter(Express、Hono、Fastify、Koa)用于embedding。
- **Pipecat Cloud/LiveKit Cloud**用于托管voice(课程22)。
- **Claude Managed Agent**用于托管长运行async。

### 可观测load-bearing

无OpenTelemetry GenAI span(课程23)加Langfuse/Phoenix/Opik backend(课程24)、你不能debug步40失败多step agent。此非optional用于产。是"我们debug快"vs"我们从scratch replay加更多logging"diff。

### 何产runtime失败

- **错形择。**Pick request-response用于5分钟task。用户hang up;worker pile up;retry compound。
- **无DLQ。**Queue worker无dead-letter。失败job vanish。
- **Opaque background work。**Background agent run无trace export。失败invisible直到用户report。
- **跳durable state。**任run>30秒你不能afford restart需durable execution。

## 构建

`code/main.py`是stdlib multi-shape demo:

- Request-response endpoint(plain function)。
- Streaming handler(generator)。
- Queue-based worker带DLQ。
- Event trigger registry。
- Cron-shaped scheduler。

跑:

```bash
python3 code/main.py
```

Output:五trace显每形同task behavior。同agent logic、异outer shell。Durable execution(第六形)intentionally cover课程13 LangGraph checkpointing。

## 使用

- **Request-response**用于chat-style UX。
- **Streaming**用于progressive response。
- **Durable**用于长horizon task。
- **Queue**用于batch/async/长运行。
- **Event**用于agent reactivity。
- **Cron**用于housekeeping(memory consolidation、eval、cost report)。

## 交付成果

`outputs/skill-runtime-shape.md`pick task runtime形并wire可观测需求。

## 练习题

1. 移你课程01 ReAct loop至全六形你stack。何形fit何product surface?
2. 加DLQ queue-based demo。Simulate 10% job failure;surface DLQ size。
3. 写cron-triggered eval agent nightly run你天top 20 trace。
4. 实streaming backpressure:若client慢、pause agent。此何turn budget interact?
5. 读Claude Managed Agent docs。何你移self-hosted长horizon agent至managed?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Request-response | "Synchronous" | 用户等;仅短task |
| Streaming | "SSE/WS" | Progressive output;better UX;latency observable per chunk |
| Durable execution | "Resume from failure" | Checkpointed state;last step restart |
| Queue-based | "Background job" | Producer/worker pool/DLQ |
| Event-driven | "Trigger-based" | Agent react外event |
| DLQ | "Dead-letter queue" | Failed job parking lot |
| Claude Managed Agent | "托管harness" | Anthropic-hosted长运行async带caching+compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——durable execution detail
- [Claude Managed Agent overview](https://platform.claude.com/docs/en/managed-agent/overview)——托管长运行async
- [Anthropic,Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)——"每task dozen-to-hundred step"
- [AutoGen v0.4(Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——actor-model fault isolation