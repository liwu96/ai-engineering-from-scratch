# Multi-Agent Primitive Model

> 每2026 shipping multi-agent framework — AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework — 是四维设计空间一点。四primitive、无更多:agent、handoff、shared state、orchestrator。此lesson从零build它们、run toy system all four、then map每major framework onto same axis so you can read任new release one paragraph。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14(Agent Engineering)、阶段16课程01(Why Multi-Agent)
**时间:** ~60分钟

## 问题背景

每六month新multi-agent framework ship。AutoGen 2023。CrewAI 2024。LangGraph和OpenAI Swarm 2024。Google ADK April 2025。Microsoft Agent Framework RC February 2026。每press release claim "the right abstraction。"

If try learn them one at time you will burn out。API look different。Doc disagree "agent"何。One framework call shared memory "blackboard、"another call "message pool、"third call "StateGraph。"You start suspect field just churning。

It not。Underneath marketing、four primitive stable。Learn once、read every new framework one paragraph。

## 概念讲解

### 四primitive

1. **Agent** — system prompt plus tool list。Stateless;every run start system prompt和current message history。
2. **Handoff** — structured transfer control one agent another。Mechanically、tool call return new agent or graph edge follow condition。
3. **Shared state** — any data structure more than one agent read(sometimes write)。Message pool、blackboard、key-value store、vector memory。
4. **Orchestrator** — whoever decide who speak next。Option:explicit graph(deterministic)、LLM speaker-selector(soft)、last speaker handoff call(OpenAI Swarm)、或scheduler over queue(swarm architecture)。

That entire design space。Every framework pick default each axis;rest surface syntax。

### 何每2026 framework map it

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool return Agent | caller problem | LLM next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector GroupChat | message pool | selector function(LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task output chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | graph、deterministic |
| Microsoft Agent Framework | agent + orchestration pattern | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifact | host decide |

Surface difference look huge。Underneath:same four knob。

### 何此matter

Once you see primitive、framework comparison become short checklist:

- Does orchestrator trust LLM route(Swarm)or pin routing code(LangGraph)?
- Is shared state full-history(GroupChat)or projected(StateGraph reducer)?
- Can agent modify each other prompt(CrewAI manager)or only hand off(Swarm)?

Those three question answer 80% which framework fit given problem。You stop shopping "the best multi-agent framework" and start designing axis you actually care about。

### Stateless insight

Every primitive except shared state stateless。Agent function(prompt、tool)。Handoff function call。Orchestrator scheduler。**The only stateful thing system shared state。**That where all interesting bug live:memory poisoning(Lesson 15)、message ordering、versioning、write contention。

Framework hide shared state(Swarm)push problem caller。Framework centralize it(LangGraph checkpoint、AutoGen pool)make inspectable but shift coordination cost onto shared-state implementation。

### Single primitive anatomy

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

No memory。No state。Two agent same system prompt和tool interchangeable。Everything look per-agent state actually shared state or handoff protocol。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

Three implementation dominate:

- **Function return** — tool return next agent。This OpenAI Swarm pattern。Agent carry routing tool schema。
- **Graph edge** — LangGraph。Edge declarative。LLM produce value;condition select next node。
- **Speaker selection** — AutoGen GroupChat。Selector function(sometimes itself LLM call)read pool and pick who speak next。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

At minimum、list message。Often more:structured artifact(CrewAI Task output)、typed context(LangGraph reducer)、external memory(MCP、vector DB)。

Two topology:**full pool**(every agent see every message)和**projected**(agent see role-scoped view)。Full pool simple and scale badly。Projected pool scale but require upfront schema design。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Four flavor:

- **Static** — graph fixed build time(LangGraph deterministic、CrewAI Sequential)。
- **LLM-selected** — LLM read pool and pick next speaker(AutoGen、CrewAI Hierarchical)。
- **Handoff-driven** — current agent decide by call handoff tool(Swarm)。
- **Queue-driven** — worker pull shared queue;no explicit next-speaker(swarm architecture、Matrix)。

### 何change between framework

Once primitive fixed、remaining design decision:

- **Memory strategy** — ephemeral vs durable checkpointing(LangGraph checkpointer)。
- **Safety boundary** — who can approve handoff(human-in-the-loop)。
- **Cost accounting** — per-agent token budget。
- **Observability** — tracing handoff、persisting state replay。

All implementable top primitive。None new primitive。

## 构建

`code/main.py` implement four primitive ~150 line stdlib Python。No real LLM — each agent scripted policy so focus stay coordination structure。

File export:

- `Agent` — dataclass name、system prompt、tool、policy function。
- `Handoff` — function return new agent。
- `SharedState` — thread-safe message pool。
- `Orchestrator` — three variant:`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`(simulated)。

Demo run same three-agent pipeline(research → write → review)through all three orchestrator type and print message pool end。You can see output differ only *who pick next*;agent和shared state identical across run。

跑:

```
python3 code/main.py
```

Expected output:three orchestrator run、one per pattern。Each print final message pool。Handoff-driven run reach fewer agent if researcher decide done early — that LLM-routing tradeoff miniature。

## 使用

`outputs/skill-primitive-mapper.md` skill read any multi-agent codebase or framework doc and return four-primitive mapping。Run new framework release get one-paragraph understanding before read doc depth。

## 交付成果

Before adopt new framework、write primitive mapping it。If cannot、doc incomplete or framework inventing fifth primitive(rare — check shared-state flavor not seen)。

Pin mapping architecture doc。When new team member join、send mapping before API doc。When framework version change、diff mapping、非changelog。

## 练习题

1. Run `code/main.py` three time different agent policy。Observe how orchestrator choice change which agent run。
2. Implement fourth orchestrator type:queue-driven one where agent poll shared state work。What deadlock can happen、and how detect?
3. Take LangGraph quickstart(https://docs.langchain.com/oss/python/langgraph/workflows-agents)and rewrite four primitive。Which LangGraph abstraction map 1:1 and which convenience wrapper?
4. Read OpenAI Swarm cookbook(https://developers.openai.com/cookbook/examples/orchestrating_agents)。Identify which four primitive Swarm make most ergonomic、and which one push caller。
5. Find one framework table hide shared state entirely。Explain what break when agent need coordinate across handoff without re-read history。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent | "An LLM with tool" | `(system_prompt, tools, model)` triple。Stateless。 |
| Handoff | "Transfer control" | Structured call name next agent和optional payload。Three implementation:function return、graph edge、speaker selection。 |
| Shared state | "Memory" / "context" | Only stateful part multi-agent system。Message pool or blackboard。 |
| Orchestrator | "Coordinator" | Whoever decide who run next。Static graph、LLM selector、handoff-driven、或queue-driven。 |
| Primitive | "Abstraction" | One four axis every framework parameterize。非framework feature。 |
| Message pool | "Shared chat history" | Full-history shared state。Easy reason、scale badly。 |
| Projected state | "Scoped view" | Role-specific view shared state。Scale、require schema design。 |
| Speaker selection | "Who talk next" | Orchestrator pattern where function(often LLM)pick next agent group。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — clearest articulation handoff-driven orchestration
- [AutoGen stable doc](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection reference LLM-selected orchestration
- [LangGraph workflows and agent](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration和reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agent、Sequential / Hierarchical process
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — live AutoGen v0.2 line after Microsoft move v0.4 maintenance