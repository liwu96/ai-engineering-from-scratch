# Agno和Mastra——产Runtime

> Agno(Python)和Mastra(TypeScript)是2026产runtime配。Agno瞄microsecond agent instantiation和stateless FastAPI backend。Mastra ship agent、tool、workflow、unified model routing、和Vercel AI SDK substrate上composite storage。

**类型:** 学习
**语言:** Python、TypeScript
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程13(LangGraph)
**时间:** ~45分钟

## 学习目标

- 识Agno性能目标和何时它们重要。
- 名Mastra三primitive——Agent、Tool、Workflow——和supported server adapter。
- 释何stateless session-scoped FastAPI backend是推荐Agno产path。
- 为给定stack(Python-first vs TypeScript-first)pick Agno vs Mastra。

## 问题背景

LangGraph、AutoGen、CrewAI framework重。欲"仅agent loop、快、于我runtime"team reach Agno(Python)或Mastra(TypeScript)。两trade些framework-owned primitive换raw speed和更紧fit周stack。

## 概念讲解

### Agno

- Python runtime、原Phi-data。
- "无graph、chain、或convoluted pattern——仅pure python。"
- Docs性能目标:~2μs agent instantiation、~3.75 KiB每agent memory、~23 model provider。
- 产path:stateless session-scoped FastAPI backend。每请求fresh agent起;session态存DB。
- Native multimodal(text、image、audio、video、file)和agentic RAG。

速度目标重要当每秒千短life agent(chat fan-in、eval pipeline)。当一agent跑10分钟时不重要。

### Mastra

- TypeScript、建Vercel AI SDK上。
- 三primitive:**Agent**、**Tool**(Zod-typed)、**Workflow**。
- Unified Model Router——94 provider 3,300+ model(2026年3月)。
- Composite storage:memory、workflow、可观测至异backend;ClickHouse推荐scale可观测。
- Apache 2.0带`ee/`目录source-available enterprise license下。
- Server adapter用于Express、Hono、Fastify、Koa;first-class Next.js和Astro integration。
- Ship Mastra Studio(localhost:4111)用于debug。
- 1.0(2026年1月)22k+ GitHub star、300k+周npm download。

### 定位

无一试作LangGraph。它们争于:

- **语言fit。**Agno用于Python-first team;Mastra用于TypeScript-first。
- **Runtime ergonomics。**Agno=近零overhead;Mastra=集成Vercel ecosystem。
- **可观测。**两集成Langfuse/Phoenix/Opik(课程24)但Mastra Studio first-party。

### 何pick每

- **Agno**——Python backend、多短life agent、强perf需、FastAPI shop。
- **Mastra**——TypeScript backend、Next.js/Vercel deploy、unified multi-provider model routing、Zod-typed tool。
- **LangGraph**(课程13)——当durable state和显graph reasoning比raw speed重要。
- **OpenAI/Claude Agent SDK**——当欲provider productized形(课程16–17)。

### 何此模式错

- **Perf-for-perf sake。**Pick Agno因"2μs"看good当workload是每请求一慢agent call。Overhead非瓶颈。
- **Ecosystem lock-in。**Mastra Vercel-flavored integration是Vercel上plus、他处minus。
- **Enterprise license confusion。**Mastra`ee/`目录source-available非Apache 2.0。若计划fork读license。

## 构建

此课主要比较——单代码artifact不justice两framework。见`code/main.py`side-by-side toy:最小"跑agent、stream output、persist session"流实两(一Agno-shaped、一Mastra-shaped)。

跑:

```
python3 code/main.py
```

两结构异但功能等trace。

## 使用

- **Agno**——需speed和FastAPI形Python backend。
- **Mastra**——多provider和workflow primitive TypeScript backend。
- 两ship first-party可观测hook。两集成Langfuse。

## 交付成果

`outputs/skill-runtime-picker.md`按stack、latency预算、和operational形pick Agno、Mastra、LangGraph、或provider SDK。

## 练习题

1. 读Agno docs。移stdlib ReAct loop(课程01)至Agno。何消失?何留?
2. 读Mastra docs。移同loop至Mastra。Tool typing(Zod vs nothing)何变?
3. Benchmark:测你stack agent instantiation latency。Agno 2μs对你workload重要否?
4. Design migration:若Python跑CrewAI、移Agno何断?
5. 读Mastra`ee/`license term。何restriction影响开源fork?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agno | "快Python agent" | Stateless session-scoped agent runtime |
| Mastra | "Vercel AI SDK上TypeScript agent" | Agent+Tool+Workflow+Model Router |
| Unified Model Router | "Multi-provider access" | 94 provider 3,300+ model single client |
| Composite storage | "多backend" | Memory/workflow/可观测各至异store |
| Mastra Studio | "Local debugger" | localhost:4111 UI introspect agent |
| Source-available | "非OSS" | License允source读但限商用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)——性能目标、FastAPI integration
- [Mastra docs](https://mastra.ai/docs)——primitive、server adapter、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/)——Mastra integration cite可观测比