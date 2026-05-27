# Shared Memory和Blackboard Pattern

> Two approach coexist 2026 multi-agent system:**message pool**(everyone see everyone message、as AutoGen GroupChat or MetaGPT)和**blackboard with subscription**(agent subscribe relevant event、as Context-Aware MCP or Matrix framework)。Both only stateful part multi-agent system — which mean both where interesting bug live。Reference failure mode **memory poisoning**:one agent hallucinate "fact、"other agent treat verified、and accuracy decay gradually way much harder debug than immediate crash。此lesson build both structure stdlib、inject poisoning attack、and show three mitigation actually work production。

**类型:** 学习+构建
**语言:** Python(stdlib、`threading`)
**前置要求:** 阶段16课程04(Primitive Model)、阶段16课程09(Parallel Swarm Network)
**时间:** ~75分钟

## 问题背景

Multi-agent system need place agent share fact。Literal option "pass everything message" — but that reinvent shared state extra copy。Another "give everyone global log" — but global log grow unbounded and poison easily。Third "project view per agent" — scalable but schema-heavy。

When one agent hallucinate and write hallucination shared state、every downstream agent read that state adopt hallucination fact。By time human notice、reasoning chain five step deep and root cause third message ever write。Debugging multi-agent accuracy decay harder than debugging crash。

This memory poisoning。It second-most-documented failure family MAST taxonomy(Cemri et al.、arXiv:2503.13657)and structural:any shared-memory design without provenance and unwritable verifier exhibit eventually。

## 概念讲解

### Two main topology

**Full message pool。**Every agent read every message。AutoGen GroupChat and MetaGPT use this。Simple、transparent、inspectable、but not scale past ~10 agent because each agent context fill other agent work。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription。**Agent declare interest topic;substrate route only relevant message。CA-MCP(arXiv:2601.11595)和Matrix decentralized framework(arXiv:2511.21686)use this。Scale further、but require upfront schema design make subscription meaningful。

```
                   ┌─ topic: price ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D(subscribe)
                   ├─ topic: order ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E(subscribe)
                   ├─ topic: alert ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F(subscribe)
                   └──────────────────┘
```

### 何each win

- **Full pool** win when agent few(< 10)、heterogeneous、and conversation short-horizon。Reasoning who said what trivial when everyone see everything。
- **Blackboard** win when agent many、homogeneous role but numerous instance(swarm)、and conversation long-running。Routing save token cost and context pollution。

Production system often mix:small full pool top(planning layer)、blackboard below(worker layer)。

### Memory poisoning、one scenario

Three agent work research task。Agent A retrieval agent。Agent B summarizer。Agent C analyst。

1. A fetch page and write message shared state:"The study report 42% accuracy improvement。"
2. Fetched page actually say "4.2% improvement。"A hallucinate decimal。
3. B、reading shared state、write:"Large 42% accuracy gain report(source: A)。"
4. C、reading shared state、write:"Recommend adoption — 42% lift transformative。"
5. Final report cite 42% number never exist。

No agent crash。No test fail。System "work。"Hallucination cross one agent context every downstream agent reasoning via shared state。

### 何this structural

Without shared state、agent A hallucination stay A context。Downstream agent would re-fetch or re-derive and might catch error。With naive shared state、A context become everyone context、and hallucination launder fact。

Problem not shared state per se — it shared state **without provenance and without independent verifier**。Three mitigation address:

1. **Attribute provenance every write。**Every entry shared state record who write、when、under what prompt、and(if applicable)what source agent cite。Downstream agent read skepticism keyed provenance。
2. **Version write;treat them append-only。**Correction new entry supersede old、非in-place update。Audit trail preserve。
3. **Keep at least one agent cannot write shared state。**Read-only verifier agent sample entry、re-fetch source、and flag inconsistency。Because cannot write pool、cannot poison pool。

### Blackboard precedent(Hayes-Roth、1985)

Blackboard pattern predate LLM agent four decade。Hayes-Roth(1985、"A Blackboard Architecture Control")describe specialist Knowledge Source observe global blackboard、contribute partial solution、and trigger other source。2026 blackboard(CA-MCP、Matrix)same pattern LLM agent Knowledge Source and JSON blob partial solution。Old literature document solution write contention、opportunistic control、and consistency modern system rediscover。

### Projection vs full view

Pure blackboard give every subscriber same projection(topic-scoped)。More aggressive design **per-agent projection**:each agent get view customize role。LangGraph state reducer canonical 2026 implementation — reducer function fold global state role-specific slice。

Per-agent projection scale further but need schema。Without one、rebuild ad-hoc projection every agent prompt。

### Write-contention pattern

Multiple agent write simultaneously concurrency problem、非just LLM problem。Three pattern work:

- **Sequential writer(single producer)。**All write go one coordinator agent serialize。Simple、but bottleneck。
- **Optimistic concurrency versioning。**Each entry version;writer fail version mismatch and retry。Classic database technique。
- **Topic partitioning。**Different agent own different topic。No cross-topic contention。Require design partition boundary。

Most 2026 framework default sequential writer because LLM call slow enough contention rare and bottleneck not hurt。

### Unwritable verifier

Most load-bearing mitigation read-only verifier。Implementation rule:

- Verifier share state team(read blackboard or pool)。
- Verifier no write handle shared state — only separate verification channel。
- Verifier independently fetch source cite write。Flag disagreement。
- Verifier own output route human or separate decision agent、never feed back pool。

Without this separation、verifier output become new entry pool、which mean poisoned pool poison verifier、which poison verification。

## 构建

`code/main.py` implement both topology stdlib Python plus toy poisoning attack and three mitigation。

- `MessagePool` — thread-safe append-only log full read-out。
- `Blackboard` — topic-keyed pub/sub per-agent subscription。
- `ProvenanceEntry` — every write record(writer、timestamp、prompt_hash、source_uri)。
- `PoisoningScenario` — run three-agent research task where agent A hallucinate decimal。Print final report。
- `Verifier` — read-only agent re-fetch source and flag inconsistency。Run same scenario verifier present。

跑:

```
python3 code/main.py
```

Expected output:
- Run 1(no verifier):hallucinate 42% propagate final report。
- Run 2(with verifier):verifier flag inconsistency、pool label "flagged"、final report include retraction。

## 使用

`outputs/skill-memory-auditor.md` skill audit any multi-agent system shared-memory design provenance、versioning、and verifier separation。Run new multi-agent architecture before production。

## 交付成果

For any shared-memory design:

- Record provenance every write:`(writer、timestamp、prompt_hash、tool_calls_cited、source_uri)`。
- Make log append-only。Correction new entry reference supersede one。
- Deploy at least one read-only verifier agent independent source access。
- Route verifier output separate channel、非back shared pool。
- Log ratio write supersession — rising ratio early evidence hallucination pattern。

## 练习题

1. Run `code/main.py`。Confirm run 1 propagate hallucination and run 2 catch it。
2. Add second hallucination:agent B invent dataset size。Verifier should catch both without hand-tune either。
3. Switch full pool blackboard topic partition(`price`、`summary`、`analysis`)。Which poisoning scenario topic partitioning make harder pull off、and which not help?
4. Read Hayes-Roth(1985、"A Blackboard Architecture Control")。Identify two control pattern paper not discuss this lesson 2026 system benefit。
5. Read CA-MCP(arXiv:2601.11595)。Map Shared Context Store either MessagePool or Blackboard class `code/main.py`。Which primitive CA-MCP add top?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Message pool | "Shared chat history" | Append-only log every agent read。Full transparency、poor scaling。 |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub。Agent subscribe relevant topic。Scale farther。 |
| Provenance | "Who write what" | Metadata each write:writer、timestamp、prompt、source。 |
| Memory poisoning | "Hallucination spreading" | One agent error enter shared state、downstream agent adopt fact。 |
| Append-only | "No in-place update" | Correction new entry supersede。Preserve audit trail。 |
| Unwritable verifier | "Independent auditor" | Read-only agent re-fetch source and flag inconsistency。 |
| Projection | "Scoped view" | Per-agent view compute global state。LangGraph reducer canonical case。 |
| Knowledge Source | "Specialist agent" | Hayes-Roth 1985 term blackboard participant。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy;memory poisoning coordination-failure sub-family
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store coordinated MCP server
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — message-queue-based blackboard without central orchestrator
- [LangGraph state and reducer](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — per-agent projection pattern production
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — provenance and verification note production deployment