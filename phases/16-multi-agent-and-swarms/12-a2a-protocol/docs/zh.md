# A2A — Agent-to-Agent Protocol

> Google announce A2A April 2025;by April 2026 spec at https://a2a-protocol.org/latest/specification/ and 150+ organization back。A2A horizontal complement MCP(Lesson 13):where MCP vertical(agent ↔ tool)、A2A peer-to-peer(agent ↔ agent)。It define Agent Card(discovery)、task with artifact(text、structured data、video)、opaque task lifecycle、and auth。Production system increasingly pair MCP with A2A。Google Cloud roll A2A support Vertex AI Agent Builder during 2025-2026。

**类型:** 学习+构建
**语言:** Python(stdlib、`http.server`、`json`)
**前置要求:** 阶段16课程04(Primitive Model)
**时间:** ~75分钟

## 问题背景

Your agent need call another agent another system。How?You can expose HTTP endpoint、define bespoke JSON schema、and hope other side speak it。Every pair agent become custom integration。

A2A universal wire protocol that call。Standard discovery、standard task model、standard transport、standard artifact。Like HTTP+REST but agent first-class citizen。

## 概念讲解

### Four element

**Agent Card。**JSON document `/.well-known/agent.json` describe agent:name、skill、endpoint、supported modality、auth requirement。Discovery happen read card。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。**Unit work。Async、stateful object with lifecycle:`submitted → working → completed / failed / canceled`。Client send task、poll or subscribe update。

**Artifact。**Result type produce task。Text、structured JSON、image、video、audio。Artifact typed so different modality first-class。

**Opaque lifecycle。**A2A not prescribe *how* remote agent solve task。Client see state transition and artifact;implementation free use any framework。

### MCP/A2A split

- **MCP**(Lesson 13):agent ↔ tool。Agent read/write via JSON-RPC tool server。Stateless default。
- **A2A**:agent ↔ agent。Peer protocol;both side agent own reasoning。

Production multi-agent system use both。A2A peer call MCP tool side。Split keep two concern clean。

### Discovery flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Or with streaming:SSE subscription `/tasks/{id}/event` push update。

### Auth

A2A support three common pattern:

- **Bearer token** — OAuth2 or opaque。
- **mTLS** — mutual TLS;organization prove identity each other。
- **Signed request** — HMAC over payload。

Auth declare Agent Card;client discover and comply。

### 150+ organization April 2026

Enterprise adoption drive A2A scale。Headline:A2A become way enterprise agent system cross trust boundary。Google Cloud ship Vertex AI Agent Builder A2A support;Microsoft Agent Framework support;most major framework(LangGraph、CrewAI、AutoGen)ship A2A adapter。

### 何A2A win

- **Cross-organization call。**Agent company A call agent company B。Without A2A、every pair bespoke contract。
- **Heterogeneous framework。**LangGraph agent call CrewAI agent call custom Python agent。A2A normalize。
- **Typed artifact。**Video result、structured JSON、audio — all first-class。
- **Long-running task。**Opaque lifecycle + polling make hour-long task straightforward。

### 何A2A struggle

- **Latency-sensitive micro-call。**A2A lifecycle async。Sub-millisecond agent-to-agent not fit;use direct RPC。
- **Tight-coupled in-process agent。**If both agent run same Python process、A2A HTTP round-trip overkill。
- **Small team。**Spec overhead real;internal-only agent may not need formality。

### A2A vs ACP、ANP、NLIP

Several related spec emerge 2024-2026:

- **ACP**(IBM/Linux Foundation) — predecessor A2A、narrower scope。
- **ANP**(Agent Network Protocol) — peer-discovery-heavy、decentralized-first。
- **NLIP**(Ecma Natural Language Interaction Protocol、standardize December 2025) — natural-language content type。

A2A most-adopted peer protocol April 2026。See arXiv:2505.02279(Liu et al.、"A Survey Agent Interoperability Protocol")comparison。

## 构建

`code/main.py` implement A2A-minimal server and client using `http.server` and JSON。Server:

- expose `/.well-known/agent.json`、
- accept `POST /tasks`、
- manage task state、
- return artifact `GET /tasks/{id}`。

Client:

- fetch Agent Card、
- submit task、
- poll until completion、
- read artifact。

跑:

```
python3 code/main.py
```

Script start server background thread、then run client against it。You see complete flow:discovery、submit、poll、artifact。

## 使用

`outputs/skill-a2a-integrator.md` design A2A integration:Agent Card content、task schema、auth choice、streaming vs polling。

## 交付成果

Checklist:

- **Pin spec version。**A2A still evolving;Agent Card should declare protocol version。
- **Idempotent task creation。**Duplicate submission(network retry)should produce one task。
- **Artifact schema。**Declare what shape agent return;consumer should validate。
- **Rate limit + auth。**A2A public-facing;apply standard web security。
- **Dead-letter failed task。**Inspect pattern over time recurring failure type。

## 练习题

1. Run `code/main.py`。Confirm client discover server and receive correct artifact。
2. Add second skill server(e.g. "summarize")。Update Agent Card。Write client pick skill based task type。
3. Implement SSE streaming endpoint:`/tasks/{id}/event` emit state change。何client need differently?
4. Read A2A spec(https://a2a-protocol.org/latest/specification/)。Identify three thing spec mandate this demo not implement。
5. Compare A2A(Agent Card discovery)MCP(server-side capability listing via `listTools`)。何tradeoff self-describing agent and capability-probing?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| A2A | "Agent-to-agent" | Peer protocol agent call other agent across system。Google 2025。 |
| Agent Card | "Agent business card" | JSON `/.well-known/agent.json` describe skill、endpoint、auth。 |
| Task | "Unit work" | Async stateful object lifecycle;artifact produce completion。 |
| Artifact | "Result" | Typed output:text、structured JSON、image、video、audio。First-class media。 |
| Opaque lifecycle | "How solved agent business" | Client see state transition;server free choose framework/tool。 |
| Discovery | "Finding agent" | `GET /.well-known/agent.json` return card。 |
| MCP vs A2A | "Tool vs peer" | MCP:vertical agent ↔ tool。A2A:horizontal agent ↔ agent。 |
| ACP / ANP / NLIP | "Sibling protocol" | Adjacent spec;A2A most-adopted 2026。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — April 2025 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementation and SDK
- [Liu et al. — A Survey Agent Interoperability Protocol](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP comparison