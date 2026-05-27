# Group Chat和Speaker Selection

> AutoGen GroupChat和AG2 GroupChat share one conversation N agent;selector function(LLM、round-robin、or custom)pick who speak next。This archetype emergent multi-agent conversation — agent not know role static graph、they just react shared pool。AutoGen v0.2 GroupChat semantic preserve AG2 fork;AutoGen v0.4 rewrite event-driven actor model。Microsoft put AutoGen maintenance mode February 2026 and merge Semantic Kernel Microsoft Agent Framework(RC February 2026)。GroupChat primitive survive both AG2和Microsoft Agent Framework — learn once、use everywhere。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程04(Primitive Model)
**时间:** ~60分钟

## 问题背景

Static graph(LangGraph)great when workflow known。Real conversation not static:sometimes coder ask reviewer、sometimes researcher、sometimes writer。Hardcoding every possible handoff produce edge explosion。You want *agent react shared pool*、with some function decide who talk next。

That exactly AutoGen GroupChat。

## 概念讲解

### Shape

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone read all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Every agent see every message。Selector function invoke each turn pick who speak next。

### Three selector flavor

**Round-robin。**Fixed cycle。Deterministic。Scale linearly N but ignore context — coder get turn even when topic legal review。

**LLM-selected。**Call LLM read recent pool and return best next speaker。Context-aware but slow:every turn add LLM call。AutoGen default。

**Custom。**Python function whatever logic you want。Typical:LLM-selected with fallback rule(e.g. "always give verifier turn after coder")。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agent=[coder, reviewer, tester], message=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` hold selector。When agent complete turn、manager call selector、which return next agent。Loop continue until termination condition。

### Termination

Three common pattern:

- **Max round。**Hard cap total turn。
- **"TERMINATE" token。**Agent emit sentinel message;manager stop when one appear。
- **Goal-reached check。**Lightweight verifier run each turn and stop chat when done。

### AutoGen → AG2 split和Microsoft Agent Framework merge

In early 2025、Microsoft begin major rewrite AutoGen(v0.4)around event-driven actor model。Community fork AutoGen v0.2 GroupChat semantic AG2、preserve API early adopter integrate。

In February 2026、Microsoft announce AutoGen go maintenance mode、with event-driven actor model merge **Microsoft Agent Framework**(RC February 2026、now merge Semantic Kernel)。GroupChat concept survive both track;implementation detail differ。AG2 preferred upstream v0.2-compatible code。

### 何GroupChat fit

- **Emergent conversation。**You not want pre-wire every possible next-speaker。
- **Role-mixing task。**Coder ask researcher、researcher ask archivist、archivist ask coder back。Flow not DAG。
- **Exploratory problem-solving。**Think "brainstorm meeting、"非"assembly line。"

### 何fail

- **Strict determinism。**LLM selector inconsistent。Same prompt、different run、different next speaker。
- **Sycophancy cascade。**Agent defer whoever speak most confidently。Counter-prompt explicitly。
- **Context bloat。**Every agent read every message;after 10 turn context huge。Use projection(Lesson 15)scope view。
- **Hot speaker。**One agent dominate conversation because selector favor specialty。Introduce speaker balance selector feature。

### Group chat vs supervisor

Same primitive、different default:

- Supervisor:one agent plan and other execute。Selector "ask planner what do。"
- Group chat:all agent peer;selector function over shared pool。

Both use four primitive Lesson 04。Group chat default LLM-selected orchestration和full-pool shared state。

## 构建

`code/main.py` implement GroupChat scratch stdlib。Three agent(coder、reviewer、manager)、round-robin和LLM-selected variant、and termination `TERMINATE` token。

Demo print conversation transcript plus selector decision trace both variant。

跑:

```
python3 code/main.py
```

## 使用

`outputs/skill-groupchat-selector.md` configure GroupChat selector given task — round-robin vs LLM-selected vs custom、and what selector input(recent message、agent specialty、turn count)use。

## 交付成果

Checklist:

- **Max round cap。**Always。10-20 typical task。
- **Speaker-balance metric。**Track turn per agent;alert imbalance exceed threshold。
- **Termination token。**`TERMINATE` or dedicated verifier agent。
- **Projection or scoped memory。**After ~10 message、consider give each agent only scoped view prevent context bloat。
- **Selector logging。**For LLM-selected variant、log both selector input and choice。Otherwise debugging impossible。

## 练习题

1. Run `code/main.py`。Compare conversation round-robin vs LLM-selected。Which agent dominate each?
2. Add "max-speaks-per-agent" rule selector。How affect transcript?
3. Implement goal-reached termination:stop when reviewer return "approved。"How often trigger before round cap?
4. Read AutoGen stable doc GroupChat(https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)。Identify default selector use `GroupChatManager`。
5. Read AG2 repo(https://github.com/ag2ai/ag2)and compare v0.2 GroupChat v0.4 event-driven version。何concrete property(throughput、fault-tolerance、composability)v0.4 add?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GroupChat | "Agent one chat room" | Shared message pool + selector function。AutoGen / AG2 primitive。 |
| Speaker selection | "Who talk next" | Function pick next agent。Round-robin、LLM-selected、or custom。 |
| GroupChatManager | "Meeting host" | AutoGen component own selector and loop turn。 |
| ConversableAgent | "Base agent" | AutoGen base class;agent send and receive message。 |
| Termination token | "Stop word" | Sentinel string(usually `TERMINATE`)end chat。 |
| Hot speaker | "One agent dominate" | Failure mode selector keep pick same agent。 |
| Context bloat | "Pool grow unbounded" | Each agent read every prior message;context grow turn。 |
| Projection | "Scoped view" | Role-specific view shared pool prevent context bloat。 |

## 延伸阅读

- [AutoGen group chat doc](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — reference implementation
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework doc](https://microsoft.github.io/agent-framework/) — merge successor、RC February 2026
- [AutoGen v0.4 release note](https://microsoft.github.io/autogen/stable/) — event-driven actor model rewrite detail