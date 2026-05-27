# LangGraph——有态Graph和持久执行

> LangGraph是2026低级有态orchestration reference。Agent是态机;节点是函数;边是transition;态immutable并每步后checkpoint。从任失败resume确切停处。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程12(Workflow Pattern)
**时间:** ~75分钟

## 学习目标

- 描述LangGraph core model:态机带immutable态、函数节点、conditional边、和post-step checkpoint。
- 名docs highlight四能力:durable execution、streaming、human-in-the-loop、comprehensive memory。
- 释LangGraph支持三orchestration topology:supervisor、peer-to-peer(swarm)、hierarchical(nested subgraph)。
- 实stdlib态graph带immutable态、conditional边、和checkpoint/resume循环。

## 问题背景

Agent和workflow共享问题:当40步run步38失败时、欲从步38 resume非重新。二等态模型留operator hack retry围绕library assume fresh run。

LangGraph设计答:态是第一类typed object、mutation显式、并checkpoint每节点后persist。Resume是`load_state(session_id)` call。

## 概念讲解

### Graph

Graph定义于:

- **State type。**Typed dict(或Pydantic model)每节点读和mutate。
- **Node。**Pure function`(state)->state_update`。Update return后merge入state。
- **Edge。**Conditional或直节点间transition。
- **Entry和exit。**`START`和`END` sentinel节点标边界。

例:带`classify`、`refund`、`bug`、`sales`、`done`节点agent——routing workflow作graph。

### Durable execution

每节点return后、runtime serialize态并写checkpointer(SQLite、Postgres、Redis、custom)。步N失败时、runtime可`resume(session_id)`并从步N+1 pickup带确切态。

LangGraph docs显highlight产user此matter:Klarna、Uber、J.P. Morgan。Claim非graph形;是graph形加checkpointing使恢复便宜。

### Streaming

每节点可yield partial output。Graph per-node-delta event stream至caller使UI graph跑时update。

### Human-in-the-loop

节点间inspect和改态。实:critical node前pause、surface态人、接受改、resume。Checkpointer使此易因态已serialize。

### Memory

Short-term(run内——state conversation history)和long-term(跨run——经checkpointer加分离long-term store持久)。LangGraph经tool集成外memory系统(Mem0、custom)。

### 三topology

1. **Supervisor。**中央router LLM dispatch至specialist subagent。`create_supervisor()`于`langgraph-supervisor`(虽然LangChain team 2026推荐直经tool call做此获更多context control)。
2. **Swarm/peer-to-peer。**Agent经共享tool面直hand off。无中央router。
3. **Hierarchical。**Supervisor管sub-supervisor、实作nested subgraph。

### 何此模式错

- **Checkpoint太小。**仅checkpoint conversation turn留tool state和memory write不可恢复。全态必须serialize。
- **Non-deterministic节点。**Resume assume节点input产同state update。Random seed、wall-clock、外API必须capture。
- **Over-use conditional边。**每边conditional graph是态机不可reason。Preferred线性chain带偶尔branch。

## 构建

`code/main.py`实stdlib有态graph:

- `State`——typed dict带`message`、`step`、`route`、`output`、`human_approval`。
- `Node`——callable取state回update dict。
- `StateGraph`——node+edge+conditional edge+run+resume。
- `SQLiteCheckpointer`(内存fake)——每节点后serialize态;`load(session_id)` restore。
- Demo graph:classify->branch(refund/bug/sales)->human gate->send。

跑:

```
python3 code/main.py
```

Trace显首run human gate失败、持久、后resume产终output。

## 使用

- **LangGraph**——reference、产ready。用`create_react_agent`、`create_supervisor`、或建己graph。
- **AutoGen v0.4**(课程14)——actor model alternative用于高并发场景。
- **Claude Agent SDK**(课程17)——managed harness带内置session store。
- **Custom**——当需exact控state形或checkpointer backend。

## 交付成果

`outputs/skill-state-graph.md`生于任目标runtime LangGraph形态graph带checkpointing和resume wired。

## 练习题

1. 加conditional edge从`classify`至`end`当classification confidence低于threshold。Human set`route`手动后resume run。
2. 换SQLite-like fake用真实SQLite checkpointer。测per-step serialization overhead。
3. 实parallel edge:两节点并发跑、custom reducer merge。Immutable态此处买何?
4. 读`langgraph-supervisor` reference。移toy至`create_supervisor`。比trace形。
5. 加streaming:每节点跑时yield partial state。印delta当它们arrive。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| State graph | "Agent作态机" | Typed态+node+edge+reducer |
| Checkpointer | "持久backend" | 每节点后serialize态;enable resume |
| Reducer | "态merger" | Function合当前态和节点update |
| Conditional edge | "Branch" | 态function选edge |
| Subgraph | "Nested graph" | 用作另一graph内node graph |
| Durable execution | "失败resume" | 确切态于last successful node restart |
| Supervisor | "Router LLM" | Specialist subagent中央dispatcher |
| Swarm | "P2P agent" | Agent经共享tool hand off;无中央router |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/)——supervisor模式API
- [AutoGen v0.4,Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)——session store和subagent