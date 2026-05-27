# AutoGen v0.4——Actor Model和Agent框架

> AutoGen v0.4(Microsoft Research,2025年1月)重设计agent orchestration围绕actor model。异步消息交换、事件驱动agent、fault isolation、自然并发。框架现maintenance mode而Microsoft Agent Framework(public preview 2025年10月)成successor。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程12(Workflow Pattern)
**时间:** ~75分钟

## 学习目标

- 描述actor model:agent作actor、message作唯一IPC、每actor failure isolation。
- 名AutoGen v0.4三API layer——Core、AgentChat、Extension——和每用于何。
- 释何decouple消息delivery和handling给fault isolation和自然并发。
- 实stdlib actor runtime于Python并移两agent code-review flow于它。

## 问题背景

多agent framework同步:一agent产、一agent消、于call stack。失败crash stack。并发bolt on。分布需rewrite。

AutoGen v0.4答:actor model。每agent是actor带私有inbox。消息是唯一交互。Runtime decouple delivery和handling。失败isolate至一actor。并发native。分布仅异transport。

## 概念讲解

### Actor

Actor有:

- 私有态(外从不直触)。
- Inbox(message queue)。
- Handler:`receive(message)->effect`其中effect可"reply"、"send至其他actor"、"spawn new actor"、"update state"、"stop self"。

两actor不共享memory。它们仅可send消息。

### AutoGen v0.4三API layer

1. **Core。**低级actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换、事件驱动。
2. **AgentChat。**任务驱动高级API(v0.2 ConversableAgent replacement)。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extension。**集成——OpenAI、Anthropic、Azure、tool、memory。

### 何decoupling重要

v0.2模型中、调`agent_a.chat(agent_b)`同步阻塞agent_a直到agent_b回。v0.4中、`send(agent_b,msg)`放消息agent_b inbox并return。Runtime后delivery。三后果:

- **Fault isolation。**Agent B crash不crash Agent A——runtime捕B handler failure并决何做(log、retry、dead-letter)。
- **自然并发。**多消息同时flight;actor并发process它们inbox。
- **分布ready。**Inbox+transport是同抽象无论actor in-process或异host。

### Topology

- **RoundRobinGroupChat。**Agent固定rotation轮流。
- **SelectorGroupChat。**Selector agent按对话context pick何go next。
- **Magentic-One。**Reference multi-agent team用于web browsing、code execution、file handling。建AgentChat上。

### 可观测

OpenTelemetry支持built in。每消息emit span;tool call载`gen_ai.*` attribute按2026 OTel GenAI semantic convention(课程23)。

### 状态:maintenance mode

2026年初:AutoGen v0.7.x稳定用于研究和原型。Microsoft移活跃开发至Microsoft Agent Framework(public preview 2025年10月1日;1.0 GA target Q1 2026末)。AutoGen pattern port forward clean——actor model是durable idea。

## 构建

`code/main.py`实stdlib actor runtime:

- `Message`——typed payload带`sender`、`recipient`、`topic`、`body`。
- `Actor`——abstract带`receive(message,runtime)`。
- `Runtime`——event loop带共享queue、delivery、failure isolation。
- 两actor demo:`ReviewerAgent` review代码、`ChecklistAgent`跑checklist;它们交换消息直到consensus。

跑:

```
python3 code/main.py
```

Trace显消息delivery、一actor中simulated failure不crash其他、和共享verdict收敛。

## 使用

- **AutoGen v0.4/v0.7**(maintenance)——稳定用于研究、原型、multi-agent pattern。
- **Microsoft Agent Framework**(public preview)——forward path;同actor-model idea refreshed API。
- **LangGraph swarm topology**(课程13)——经shared-tool handoff类似模式。
- **Custom actor runtime**——当需特定transport(NATS、RabbitMQ、gRPC)。

## 交付成果

`outputs/skill-actor-runtime.md`生最小actor runtime加team template(RoundRobin或Selector)用于给定multi-agent任务。

## 练习题

1. 加dead-letter queue:handler raise时、park失败消息人inspect。Toy中DLQ何频hit?
2. 实`SelectorGroupChat`:selector actor按对话态pick何process下消息。
3. 加分布transport:换in-process queue用JSON-over-HTTP server使actor可跑分离process。
4. Wire每消息OTel span(或no-op stand-in)。Emit`gen_ai.agent.name`、`gen_ai.operation.name`按课程23。
5. 读AutoGen v0.4 architecture post。移toy至真`autogen_core` API。何你skip产中matter?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Actor | "Agent" | 私有态+inbox+handler;无共享memory |
| Message | "Event" | Typed payload;actor交互唯一方式 |
| Inbox | "Mailbox" | 每actor pending消息queue |
| Runtime | "Agent host" | Route消息和isolate failure event loop |
| Topic | "Channel" | Actor间named publish-subscribe route |
| Fault isolation | "Let it crash" | 一actor失败不crash其他 |
| RoundRobinGroupChat | "Fixed-rotation team" | Agent按序轮流 |
| SelectorGroupChat | "Context-routed team" | Selector pick何go next |
| Magentic-One | "Reference team" | Web+code+file multi-agent squad |

## 延伸阅读

- [AutoGen v0.4,Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——重设计post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——graph形alternative
- [OpenTelemetry GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——AutoGen默认emit span