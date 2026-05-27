# Role Specialization — Planner、Critic、Executor、Verifier

> Most common multi-agent decomposition 2026:one agent plan、one execute、one critique or verify。MetaGPT(arXiv:2308.00352)formalize this SOP encode role prompt — Product Manager、Architect、Project Manager、Engineer、QA Engineer — follow `Code = SOP(Team)`。ChatDev(arXiv:2307.07924)chain designer、programmer、reviewer、tester through "chat chain" with "communicative dehallucination"(agent explicitly request missing detail)。Verifier load-bearing:Cemri et al.(MAST、arXiv:2503.13657)show every multi-agent failure can trace missing or broken verification。PwC report 7× accuracy gain(10% → 70%)from structured validation loop CrewAI。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程04(Primitive Model)、阶段16课程05(Supervisor)
**时间:** ~60分钟

## 问题背景

Generic multi-agent system produce generic output。Three coder group chat write three flavor same mediocre code。You can add more agent、add more round、and still not cross quality threshold。

Fix not more agent — it *different* agent。Assign distinct role。Give critic tool planner not have。Give verifier objective test suite。Now system internal disagreement grounded correction、非just parallel guessing。

## 概念讲解

### 四canonical role

**Planner。**Read goal、produce step list or spec。Tool:knowledge retrieval、doc。Output:structured plan。

**Executor。**Read one plan step time、produce artifact。Tool:actual work tool(code compiler、shell、API client)。Output:artifact。

**Critic。**Read executor output against planner intent。Tool:read-only access artifact、static analysis。Output:accept/reject reason。

**Verifier。**Read artifact and run deterministic check。Tool:test runner、type checker、schema validator。Output:pass/fail evidence。

Critic subjective、opinionated、often LLM-based。Verifier objective、deterministic、often code-based。They not same role。

### MetaGPT SOP pattern

MetaGPT(arXiv:2308.00352)encode software engineering SOP role prompt:

- **Product Manager** write PRD。
- **Architect** produce system design。
- **Project Manager** split task。
- **Engineer** implement。
- **QA Engineer** run test。

Each role strict input/output schema。Role prompt say role *is* and what *must produce*。`Code = SOP(Team)` formulation — deterministic SOP turn team LLM predictable pipeline。

### ChatDev communicative dehallucination

ChatDev add key move:when executor need specific detail not plan、it explicitly ask designer before continuing。This prevent classic LLM failure plausibly invent detail。

Implementation:role prompt include "when need specific information not given、ask relevant role name before produce output。"

### 何verifier matter most

Cemri et al.(MAST)trace 1642 multi-agent execution failure。21.3% verification gap — system ship answer no one check。Remaining 79% often trace "there check fail silently or never run。"Verification load-bearing role。

PwC report(CrewAI deployment、2025)add structured validation loop move accuracy 10% 70%。7× gain one role。

### Critic vs verifier

- Critic LLM review artifact quality。Subjective。Can fool plausible prose。
- Verifier deterministic program run artifact。Objective。Give pass/fail evidence。

Use both。Critic catch taste issue verifier cannot articulate。Verifier catch bug critic cannot see because show up only runtime。

### Anti-pattern

Every role system LLM and every role output "look good me。"Classic MAST failure mode。Add at least one verifier pass/fail decide code、非LLM。

### Framework mapping

- **CrewAI** — `Agent(role, goal, backstory)` textbook specialization surface。
- **LangGraph** — node can specialized prompt;edge enforce pipeline。
- **AutoGen** — role-specific ConversableAgent one-word name GroupChat。
- **OpenAI Agents SDK** — handoff tool between role-specialized Agent。

## 构建

`code/main.py` implement 4-role pipeline build simple Python function:

- **Planner** produce spec。
- **Executor** generate code string。
- **Critic**(LLM-simulated)flag obvious issue。
- **Verifier** run generated code sandbox(`exec`)against test case。

Demo run twice:once executor produce correct code(critic + verifier both pass)、once executor produce off-spec code(critic miss bug because look plausible、verifier catch because test fail)。

跑:

```
python3 code/main.py
```

## 使用

`outputs/skill-role-designer.md` take task and produce role roster(3-5 role)、input/output schema per role、和verifier check。Use this before wire agent framework。

## 交付成果

Checklist:

- **At least one deterministic verifier。**Never all-LLM。
- **Explicit I/O schema per role。**Planner return spec、非prose;executor read schema。
- **Communicative dehallucination。**Executor must ask planner when info missing;never invent。
- **Critic/verifier ordering。**Run critic first(cheap、catch design issue)、verifier second(slow、catch bug)。
- **Loop budget。**Max 2 critic-executor revision round before escalate human。

## 练习题

1. Run `code/main.py` and observe verifier catch bug critic miss。Add static-analysis check(count occurrence `return`)additional verifier。何it catch runtime test miss?
2. Add 5th role:"requirements analyst" translate user wish planner-ready spec。何communicative dehallucination request flow up it?
3. Read MetaGPT Section 3("Agent")。List input/output schema each MetaGPT 5 role。
4. Read ChatDev chat-chain diagram(arXiv:2307.07924 Figure 3)。Identify where communicative dehallucination break loop otherwise infinite。
5. PwC 7× accuracy gain come verification loop。Hypothesize three task where add verifier not help — where deterministic check correctness impossible or prohibitively expensive。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Role specialization | "Different agent、different job" | Distinct system prompt tune planner/executor/critic/verifier role。 |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT framing:strict I/O schema per role turn team pipeline。 |
| Communicative dehallucination | "Ask before inventing" | ChatDev pattern:executor ask planner when detail missing rather make up。 |
| Critic | "LLM reviewer" | Subjective、opinionated reviewer。Catch taste issue。Can fool plausible prose。 |
| Verifier | "Deterministic check" | Code-based pass/fail。Test runner、type checker、schema validator。Cannot fool。 |
| Verification gap | "No one check" | 21.3% MAST failure。Answer ship without check would have caught bug。 |
| Revision loop | "Critic send back" | Critic rejection trigger executor re-run feedback。Need budget。 |
| All-LLM anti-pattern | "Look good me" | Every role LLM、no deterministic check。Classic MAST failure。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP-as-role-prompt reference paper
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy;verification gap 21.3% failure
- [CrewAI doc — Agent role](https://docs.crewai.com/en/introduction) — production role specification surface