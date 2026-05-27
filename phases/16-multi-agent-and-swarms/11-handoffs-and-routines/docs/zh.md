# Handoff和Routine — Stateless Orchestration

> OpenAI Swarm(October 2024)distill multi-agent orchestration two primitive:**routine**(instruction + tool system prompt)和**handoff**(tool return another Agent)。No state machine、no branching DSL — LLM route by call right handoff tool。OpenAI Agents SDK(March 2025)production successor。Swarm itself remain cleanest conceptual reference — entire source fit few hundred line。Pattern viral because API surface roughly "agent = prompt + tool;handoff = function return agent。"Limitation:stateless、so memory caller problem。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程04(Primitive Model)
**时间:** ~60分钟

## 问题背景

Every multi-agent framework want you learn DSL:LangGraph node and edge、CrewAI crew and task、AutoGen GroupChat and manager。DSL real abstraction、but make thing feel heavier than need。

Swarm push opposite direction:use tool-calling capability model already have。Handoff become tool call。Orchestrator whichever agent currently hold conversation。State machine implicit agent system prompt。

## 概念讲解

### Two primitive

**Routine。**System prompt define agent role and available tool。Think scoped set instruction:"you triage agent;if user ask refund、hand off refund agent。"

**Handoff。**Tool agent can call return new Agent object。Swarm runtime detect Agent return value and switch active agent next turn。

That entire abstraction。

```
def transfer_to_refunds():
    return refund_agent  # Swarm see Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route user right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

Triage agent system prompt make choose right handoff based user message。LLM tool-calling do routing。

### 何viral

- **Small API。**Two concept learn。
- **Use what model already do。**Tool calling already production-grade across provider。
- **No state-machine burden。**You not describe graph;agent prompt describe hand off。

### Stateless trade

Swarm explicitly stateless between run。Framework keep message history during run、but not persist anything。Memory、continuity、long-running task — all caller problem。

In production(OpenAI Agents SDK、March 2025)this one main thing change:SDK add built-in session management、guardrail、and tracing while keep handoff primitive。

### 何Swarm/handoff fit

- **Triage pattern。**Front-line agent route user specialist。
- **Skill-based handoff。**"If task need code、call coder;if need research、call researcher。"
- **Short、bounded conversation。**Customer support、FAQ-to-ticket、simple workflow。

### 何Swarm struggle

- **Long session shared memory。**Handoff reset conversation state new agent prompt plus history。No persistent state across agent without caller-managed memory。
- **Parallel execution。**Handoff one-at-time — active agent switch。Parallelism require caller orchestrate multiple Swarm run。
- **Audit and replay。**Stateless run hard replay exactly;LLM handoff choice not deterministic。

### OpenAI Agents SDK(March 2025)

Production successor add:

- **Session state。**Persistent thread across run。
- **Guardrail。**Input/output validation hook。
- **Tracing。**Every tool call and handoff log。
- **Handoff filter。**Control what context transfer handoff。

Handoff primitive survive;production ergonomic add around。

### Swarm vs GroupChat

Both use LLM-driven routing、but differ **who pick next**:

- GroupChat:selector(function or LLM)pick next speaker outside。
- Swarm:current agent pick successor by call handoff tool。

Swarm "agent decide what next";GroupChat "manager decide what next。"Swarm decision live active agent tool call;GroupChat live `GroupChatManager`。

## 构建

`code/main.py` implement Swarm scratch:Agent dataclass、handoff mechanism(tool return Agent)、and run loop detect agent switch。

Demo:triage agent route refund、sale、or support specialist。Each specialist own tool。Run loop print each handoff。

跑:

```
python3 code/main.py
```

## 使用

`outputs/skill-handoff-designer.md` design handoff topology given task:which agent exist、which handoff call、what context transfer。

## 交付成果

Checklist:

- **Handoff logging。**Every handoff write trace event from-agent、to-agent、context snapshot。
- **Context transfer rule。**Decide what move handoff:full history(expensive)、last N message、or summary。
- **Guardrail handoff。**Handoff specialist different tool permission must authenticate — otherwise prompt injection can force unwanted handoff。
- **Loop detection。**Two agent hand back and forth common failure;detect simple last-K ring check。
- **Fallback agent。**If handoff target not exist、fall back safe default。

## 练习题

1. Run `code/main.py`、triage refund agent。Confirm second turn active agent refund。
2. Add loop-detection rule:if same two agent hand off 3 time row、force exit。Design fallback。
3. Read OpenAI Agents SDK doc handoff filter。Implement "summarize-on-handoff" version:outgoing agent compress context bullet summary before incoming agent take over。
4. Compare Swarm handoff GroupChatManager selector。Which pattern make prompt injection worse、and why?
5. Read Swarm cookbook(https://developers.openai.com/cookbook/examples/orchestrating_agents)。Identify one explicit design decision Swarm make OpenAI Agents SDK change or keep。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Routine | "Agent prompt" | System prompt + tool list。Define role and available handoff。 |
| Handoff | "Transfer another agent" | Tool active agent call return new Agent。Runtime switch active agent。 |
| Stateless | "No memory between run" | Swarm not persist anything;memory caller responsibility。 |
| Active agent | "Who speaking now" | Agent currently hold conversation。Handoff change this。 |
| Context transfer | "What move handoff" | Policy what history incoming agent see:full、last N、or summarized。 |
| Handoff loop | "Agent ping-pong" | Failure mode two agent keep hand back each other。 |
| OpenAI Agents SDK | "Production Swarm" | March 2025 successor;add session、guardrail、tracing top handoff primitive。 |
| Handoff filter | "Gate transfer" | SDK feature inspect and modify context handoff boundary。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — reference articulation
- [OpenAI Swarm repo](https://github.com/openai/swarm) — original implementation、keep conceptual reference
- [OpenAI Agents SDK doc](https://openai.github.io/openai-agents-python/) — production successor session and tracing
- [Anthropic handoff-in-Claude note](https://docs.anthropic.com/en/docs/claude-code) — how Claude Code subagent use handoff-like pattern via `Task`