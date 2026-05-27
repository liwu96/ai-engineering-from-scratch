# 代理框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每框架卖同demo(研究代理建报告)和藏同bug(状态schema与编排层斗)。择其抽象匹配你问题形的框架;余是你写两次的胶。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段11课程09(函数调用),阶段11课程16(LangGraph)
**时间:** ~45分钟

## 问题背景

你有需多于一次LLM调用任务。可能是研究工作流(计划、搜索、总结、引用)。可能是代码审管道(解析diff、批评、补、验证)。可能是预订航班、写邮件和报销的多转助手。你择框架。

三天后,你发现框架抽象漏。CrewAI给角色但"研究员"需交结构计划给"作家"时斗。AutoGen给代理间聊天但无一等状态所以你checkpoint是对话日志pickle。LangGraph给状态图但强你命名每过渡于知代理何做前。Agno给你尖叫于试fan out至三并发worker的单代理原语。

修非"择最佳框架"。是匹配框架核心抽象至你问题形。本课画那图。

## 概念讲解

![代理框架矩阵:核心抽象vs问题形](../assets/framework-matrix.svg)

四框架主导2026生态。其核心抽象非同。

| 框架 | 核心抽象 | 最佳适配 | 最差适配 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型状态、节点、条件边、checkpointer。 | 显状态和人机交互interrupt工作流;需time-travel debug产代理。 | 松角色驱动脑暴拓扑未知。 |
| **CrewAI** | `Crew` — 角色(目标、背景)、任务、流程(顺序或层级)。 | 角色扮演或角色驱动带短线性/层级计划工作流。 | crew转历史外任何有状态;复杂分支。 |
| **AutoGen** | `ConversableAgent`对 — 两或更多代理轮聊至exit条件。 | 多代理*对话*(师徒、提议者批评者、执行者审员)思考从聊涌现。 | 已知DAG确定工作流;跨重启需持久状态任何。 |
| **Agno** | `Agent` — 单LLM+工具+记忆,可组合成team。 | 快建单代理和轻量team;强多模态和内置存驱动。 | 深显分支带自定义reducers图。 |

### "抽象"实意

框架核心抽象是你pitch架构时白板画物。

- **LangGraph** → 你画图。节点是步、边是过渡、每点状态对象是类型。心智模型是状态机。
- **CrewAI** → 你画组织图。每角色有职位描述和manager路由任务。心智模型是小专业团队。
- **AutoGen** → 你画Slack DM。两代理互发消息;第三需moderator时加。心智模型是聊天。
- **Agno** → 你画单盒子带工具挂。盒子并列成team。心智模型是"代理带电池"。

### 状态问题

状态是产中多框架选择破处。

- **LangGraph。**类型状态(`TypedDict`或Pydantic模型)、每字段reducers、一等checkpointer(SQLite/Postgres/Redis)。Resume、interrupt和time-travel免费。(见阶段11课程16。)
- **CrewAI。**状态作字符串经`context`字段流于任务间,或结构经`output_pydantic`。无耐用crew存;若crew需跨重启存活你自bolt。
- **AutoGen。**状态是聊天历史和任用户定义`context`。对话转录持久化;任意工作流状态除非你写适配器不持久化。
- **Agno。**内置存驱动(SQLite、Postgres、Mongo、Redis、DynamoDB)经`storage=`附至`Agent` — 对话session和用户记忆自动持久化。非全图checkpointer;session存。

### 分支问题

每非平凡代理分支。谁决分支重要。

- **LangGraph** — 你决,经条件边。路由是带命分支Python函数。分支是编译图一等;checkpointer录何分支取。
- **CrewAI** — 层级模式manager决;顺序模式你build时决。路由隐于任务列表;manager提示词外无一等"if"。
- **AutoGen** — 代理经聊天决。分支从谁下次说涌现。`GroupChatManager`择次说话者;你可手写`speaker_selection_method`但默是LLM驱动。
- **Agno** — 代理决于下次调何工具。Team有coordinator/router/collaborator模式;分支外开发者责。

### 可观察性问题

- **LangGraph** — OpenTelemetry经LangSmith或任OTel exporter。每节点过渡是trace span;checkpoint加倍为可重播trace。LangSmith是一方选项;Langfuse/Phoenix也有适配器。
- **CrewAI** — 2025后一等OpenTelemetry;集成Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** — OpenTelemetry集成经`autogen-core`;AgentOps和Opik有连接器。Trace粒度是每代理消息,非每节点。
- **Agno** — 内置`monitoring=True`标志加OpenTelemetry exporters;紧集成Langfuse为session traces。

### 成本和延迟

四框架都加每调用开销(框架逻辑、验证、序列化)。增加开销大致顺序:Agno ≈ LangGraph < CrewAI ≈ AutoGen。差被框架多LLM路由开销主导。CrewAI层级manager花token决谁下次;AutoGen `GroupChatManager`同样。LangGraph仅在你写`llm.invoke`处花token。Agno单代理路径薄。

当每run成本重要,择显路由(LangGraph边、AutoGen `speaker_selection_method`)而非LLM选路由。

### 互操作

- **LangGraph** ↔ **LangChain**工具、retrievers、LLM。一等MCP适配器(工具作MCP服务器导入)。
- **CrewAI** ↔ 工具继承`BaseTool`;LangChain工具、LlamaIndex工具和MCP工具都适配。Crew-to-crew delegation经`allow_delegation=True`。
- **AutoGen** → `FunctionTool`包任Python callable;MCP适配器可用。紧耦合AG2生态为代理到代理模式。
- **Agno** → `@tool`装饰器或BaseTool子类;MCP适配器;工具可跨代理和team共享。

## 技能

> 你可一句释何给定框架适给定代理问题。

建前检查表:

1. **画形。**这是图(类型状态、命过渡)?角色扮演(专家交接)?聊天(代理聊至完)?单代理带工具?
2. **决谁分支。**开发者决分支→LangGraph。Manager代理决→CrewAI层级。聊天涌现→AutoGen。工具调用决→Agno。
3. **查状态预算。**需resume-from-checkpoint?Time-travel?人run中interrupt?若然,LangGraph默;Agno session覆盖对话范围状态。
4. **查成本预算。**LLM选路由每转花额外token。若代理日跑千次,择显路由。
5. **预算框架开销。**每框架是他依赖。若任务是两LLM调用和一工具,写30行纯Python;无框架比无框架便宜。

拒于可画图、组织图、聊天或代理框前达框架。择其状态模型斗你实需物拒。

## 决策矩阵

| 问题形 | 择框架 | 何 |
|---------------|---------------------|-----|
| 带类型状态、人批准、长跑工作流DAG | LangGraph | 一等状态、checkpointer、interrupts、time-travel。 |
| 带角色研究/写管道 | CrewAI(顺序)或LangGraph子图 | CrewAI角色每任务便宜表达;分支复杂时用LangGraphscale up。 |
| 提议者批评者或师徒对话 | AutoGen | 两代理聊天是其原生形。 |
| 单代理带工具、session、记忆 | Agno | 最薄设置,内置存和记忆。 |
| 带reducers千并行fanout | LangGraph + `Send` | 唯带一等并行dispatch原语。 |
| 快原型,无框架承诺 | 纯Python + 提供方SDK | 无框架是最快框架。 |

## 练习题

1. **易。**取同任务—"研究Anthropic总部,写200字简报,引用源"—LangGraph实(四节点:计划、搜索、写、引用)和CrewAI实(三角色:研究员、作家、编辑)。报每run token成本和代码行。
2. **中。**AutoGen实同任务(研究员↔作家聊,编辑经`GroupChat`加)和Agno实(单代理带`search_tools`和`write_tools`,加session存)。四实现排于(a)每run成本、(b)崩溃后resume能力、(c)写步前注人批准能力。
3. **难。**建决策树脚本`pick_framework.py`取短问题描述(JSON:`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`)返带一句理推荐。于自设六案例验。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编排 | "代理何协调" | 决何节点/角色/代理下次跑层。 |
| 持久状态 | "重启后resume" | 跨进程死亡存活状态,附checkpoint或session存。 |
| LLM选路由 | "让模型决" | 计划LLM每转择次步;灵活但每决策花token。 |
| 显路由 | "开发者决" | Python函数或静态边择次步;便宜且可审计。 |
| Crew | "CrewAI团队" | 角色+任务+流程(顺序或层级)绑入单runnable。 |
| GroupChat | "AutoGen多代理聊天" | N代理间带说话者选择器管对话。 |
| Team(Agno) | "多代理Agno" | Route/coordinate/collaborate模式于代理集。 |
| StateGraph | "LangGraph图" | 类型状态、节点、条件边、checkpointer原语。 |

## 延伸阅读

- [LangGraph文档](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI文档](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno文档](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — 建效代理(2024年12月)](https://www.anthropic.com/research/building-effective-agents) — 模式库(提示词链、路由、并行化、orchestrator-workers、evaluator-optimizer)框架无关。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每框架装饰原语。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI式角色栈基角色扮演基础。
- 阶段11课程16(LangGraph) — 本课benchmarks框架。
- 阶段11课程19(Reflexion) — LangGraph映射干净但CrewAI尴尬模式。
- 阶段11课程22(产可观察性) — 何instrument你择框架。