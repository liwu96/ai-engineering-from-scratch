# Eval驱动Agent开发

> Anthropic guidance:"从简提示起、用全面evaluation优化它们、仅当需时加多步agentic系统。"Evaluation非最后step。是Phase 14每择outer loop drive。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14全。
**时间:** ~60分钟

## 学习目标

- 名三evaluation layer——static benchmark、custom offline、online production——和每用于何。
- 释evaluator-optimizer tight loop。
- 描述2026 best practice:eval live next code、CI run、gate PR。
- Connect每Phase 14课至它generate eval case。

## 问题背景

Agent pass demo。它们产于demo不能predict方式失败。Benchmark答"此模型广capable否?"非"此agent ship我product正确patch否?"答:三层evaluation、连续run、每guardrail和learned rule map eval case。

## 概念讲解

### 三evaluation layer

1. **Static benchmark**——SWE-bench Verified用于code(课程19)、WebArena/OSWorld用于browsing/desktop(课程20)、GAIA用于generalist(课程19)、BFCL V4用于tool use(课程06)。用于cross-model comparison和regression gating。Contamination真实:SWE-bench+发现32.67% solution leakage。常report Verified/+-audited score。

2. **Custom offline eval**——你product shape:
   - LLM-as-judge(Langfuse、Phoenix、Opik——课程24)。
   - Execution-based(run patch、check test)。
   - Trajectory-based(compare action sequence gold;OSWorld-Human显top agent 1.4–2.7x over gold)。

3. **Online eval**——产:
   - Session replay(Langfuse)。
   - Guardrail-triggered alert(课程16、21)。
   - 每step cost/latency tracking(课程23 OTel span)。

### Evaluator-optimizer(Anthropic)

Tight loop:

1. Proposer generate output。
2. Evaluator judge。
3. Refine直到evaluator pass。

此是Self-Refine(课程05)泛化。任你care agent flow可wrap evaluator-optimizer用于reliability。

### 2026 best practice

- Eval live next code。
- CI每PR run。
- Gate merge eval score(如"无regression>5% vs main")。
- 每guardrail map eval case。
- 每learned rule(Reflexion、pro-workflow learn-rule)map failure case。

### Tie Phase 14 together

每Phase 14课generate eval case:

| 课程 | 它generate eval case |
|------|----------------------|
| 01 Agent Loop | Budget-exhausted、infinite-loop guard |
| 02 ReWOO | Planner tool fail时replan正确 |
| 03 Reflexion | Learned reflection retry apply |
| 05 Self-Refine/CRITIC | Judge refined output pass |
| 06 Tool Use | Argument coercion work;unknown tool reject |
| 07–10 Memory | Retrieval citation match source;stale fact invalidate |
| 12 Workflow Pattern | 每pattern产正确output |
| 13 LangGraph | Resume reproduce state exact |
| 14 AutoGen Actor | DLQ catch crash handler |
| 16 OpenAI Agent SDK | Guardrail trip right input |
| 17 Claude Agent SDK | Subagent result return orchestrator |
| 19–20 Benchmark | SWE-bench Verified score、WebArena success rate、OSWorld efficiency |
| 21 Computer Use | 每step安全catch injected DOM |
| 23 OTel | Span emit required attribute |
| 26 Failure Mode | Detector tag known failure |
| 27 Prompt Injection | PVE refuse poisoned retrieval |
| 28 Orchestration | Supervisor route right specialist |
| 29 Runtime Shape | DLQ handle N% failure |

若你eval suite每有case、你Phase 14 cover。

### 何eval驱动开发失败

- **无baseline。**Eval无last-known-good unreadable。存baseline。
- **LLM-judge无grounding。**Judge也hallucinate。CRITIC pattern(课程05)——judge ground外tool。
- **Over-fit eval。**Optimize eval diverge产usefulness。Rotate case。
- **Flaky eval。**Non-deterministic case cause false alarm。Pin seed、snapshot state。

## 构建

`code/main.py`是stdlib eval框架:

- Case registry带category(benchmark、custom、online)。
- Scripted agent under test。
- Evaluator-optimizer loop:propose、judge、refine until pass或max round。
- CI gate:aggregate pass rate+regression baseline。

跑:

```
python3 code/main.py
```

Output:per-case pass/fail、regression flag、CI gate verdict。

## 使用

- 写eval case同repo agent code。
- CI每PR run。
- Regression build fail。
- Track pass rate over time。
- Tie每产失败新case。

## 交付成果

`outputs/skill-eval-suite.md`建agent product三层eval suite带CI gate和regression tracking。

## 练习题

1. Take你一产失败。写eval case reproduce。你agent现pass否?
2. Build你domain LLM-judge rubric三dimension(factual、tone、scope)。Score 50 session。
3. Wire eval suite入CI。>=5% regression build fail。
4. 加trajectory-efficiency metric:agent take何step vs gold trajectory?
5. Map每Phase 14课入你suite eval case。有missing?那是gap close。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "Domain eval" | LLM-as-judge/exec/trajectory于你product shape |
| Online eval | "产eval" | Session replay、guardrail alert、cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | Iterate until judge pass |
| CI gate | "Merge blocker" | Eval regression build fail |
| Baseline | "Last-known-good" | Reference score detect regression |
| Trajectory efficiency | "Step over gold" | Agent step count除人expert minimum |

## 延伸阅读

- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——"从简起、eval优化"
- [OpenAI,SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)——curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)——tool-use benchmark
- [Langfuse docs](https://langfuse.com/)——eval+session replay实践