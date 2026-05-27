# Orchestration模式——Supervisor、Swarm、Hierarchical

> 四orchestration模式2026 framework复现:supervisor-worker、swarm/peer-to-peer、hierarchical、debate。Anthropic guidance:"关键建适你需系统。"从简起;仅当单agent加五workflow pattern不足时加topology。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程12(Workflow Pattern)、阶段14课程25(Multi-Agent Debate)
**时间:** ~60分钟

## 学习目标

- 名四复现orchestration模式和何时每fit。
- 描述2026 LangChain recommendation:tool-call-based supervision vs supervisor library。
- 释Anthropic"建正确系统"rule和何它gate topology choice。
- 实全四于stdlib对common scripted LLM。

## 问题背景

Team reach"multi-agent"于需它前。四pattern framework复现;一旦你可名它们、你可pick正确——或skip topology entirely。

## 概念讲解

### Supervisor-worker

- 中央routing LLM dispatch specialist agent。
- Decide:loop back self、hand off specialist、terminate。
- Specialist不彼此talk;全routing经supervisor。

Framework:LangGraph `create_supervisor`、Anthropic orchestrator-worker、CrewAI Hierarchical Process。

**2026 LangChain recommendation:**经直tool call做supervision而非`create_supervisor`。给更细context engineering控——你决exact每specialist见。

### Swarm/peer-to-peer

- Agent经共享tool面直hand off。
- 无中央router。
- Latency低于supervisor(少hop)。
- 更难reason(无单控点)。

Framework:LangGraph swarm topology、OpenAI Agent SDK handoff(当全agent可hand off全其他)。

### Hierarchical

- Supervisor管sub-supervisor管worker。
- LangGraph nested subgraph实;CrewAI nested crew。
- Scale大agent population以operational complexity cost。

何时需:当单supervisor context budget不能hold全specialist description。

### Debate

- Parallel proposer+iterative cross-critique(课程25)。
- 不真orchestration——更多verification——但framework显作topology choice。

### CrewAI Crew vs Flow

CrewAI formalize两deployment mode:

- **Flow**用于deterministic事件驱动automation(产推荐起点)。
- **Crew**用于autonomous role-based collaboration。

此orthogonal上四pattern但map topology:Flow典型supervisor或hierarchical;Crew典型supervisor带LLM router。

### Anthropic guidance

"LLM space成功非建最sophisticated系统。关键建适你需系统。"

决策序:

1. 单agent+workflow pattern(课程12)——始此。
2. Supervisor-worker——当你有2–4 specialist。
3. Swarm——当latency matter比reasoning clarity多。
4. Hierarchical——仅当supervisor context budget fail。
5. Debate——当accuracy matter比cost多。

### 何此模式错

- **Topology-first thinking。**"需multi-agent"于identify何问题multi-agent solve前。
- **Swarm bouncing handoff。**A->B->A->B。用hop counter。
- **Fake hierarchy。**三层因"enterprise";两实际team。Collapse。

## 构建

`code/main.py`实全四pattern于stdlib对scripted LLM:

- `Supervisor`——中央router。
- `Swarm`——peer-to-peer带直handoff。
- `Hierarchical`——supervisor的supervisor。
- `Debate`——parallel proposer+critique。

每pattern handle同三intent task(refund/bug/sales)。Trace形异。

跑:

```
python3 code/main.py
```

Output:per-pattern trace+op count。Supervisor cleanest;swarm shortest;hierarchical deepest;debate most expensive。

## 使用

- **LangGraph**用于supervisor和hierarchical(nested subgraph)。
- **OpenAI Agent SDK**用于handoff-as-tool(supervisor-shaped)。
- **CrewAI Flow**用于产deterministic。
- **Custom**用于debate或当你欲exact控。

## 交付成果

`outputs/skill-orchestration-picker.md`pick topology并实它。

## 练习题

1. 换supervisor-worker至swarm经remove router。何break?何improve?
2. 加swarm hop counter:3 handoff后refuse。它catch A->B->A bouncing否?
3. 建12-specialist domain两level hierarchical系统。无nesting何context budget fail?
4. Profile四pattern产shaped workload。何metric(latency、cost、accuracy、debuggability)何赢?
5. 读Anthropic"Building Effective Agents"post。Map每产flow四之一。有cleanly不map否?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Supervisor-worker | "Router+specialist" | 中央LLM dispatch specialist;它们不彼此talk |
| Swarm | "Peer-to-peer" | 经共享tool直handoff;无中央router |
| Hierarchical | "Supervisor的supervisor" | Nested subgraph用于大population |
| Debate | "Proposer+critique" | Parallel proposer、cross-critique(课程25) |
| Tool-call-based supervision | "Supervisor无library" | 经直tool call实supervisor用于context control |
| Crew | "Autonomous team" | CrewAI role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI事件驱动产mode |

## 延伸阅读

- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——五pattern+agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction)——Crew vs Flow
- [Du等,Society of Mind(arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)——debate pattern