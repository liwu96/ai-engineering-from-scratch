# Hierarchical Architecture和Its Failure Mode

> Hierarchical supervisor nested。Manager agent over sub-manager over worker。CrewAI `Process.hierarchical` textbook version:`manager_llm` dynamically delegate task and validate output。LangGraph equivalent `create_supervisor(create_supervisor(...))`。It natural pattern when task real org chart。It also pattern most likely collapse managerial looping — manager agent assign work poorly、misinterpret sub-output、or fail reach consensus。Sequential often beat it。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程05(Supervisor Pattern)
**时间:** ~60分钟

## 问题背景

Once supervisor pattern click、natural next step "what if worker themselves supervisor?"Team have sub-team;company have department department。Hierarchical architecture mirror that。

Issue:LLM manager not same human manager。Human manager stable prior what report know。LLM manager re-reason org every turn whatever context。Tiny drift that context、and whole tree misallocate work。

## 概念讲解

### Shape

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Every internal node plan、delegate、and synthesize。Only leaf do work。

### 何shine

- **Clear org mapping。**If real task departmental("legal review doc、finance review doc、engineering review doc、then summarize exec")、hierarchy explicit。
- **Local summarization。**Each sub-manager synthesize team output before top manager see it。Top manager see three sub-manager summary、非fifteen worker output。

### 何break

Three failure mode 2026 post-mortem keep finding:

1. **Task assignment error。**Manager read goal、hallucinate decomposition、and delegate wrong sub-manager。Because sub-manager obediently work what given、error only surface top synthesis — one level remove where human could have caught it。
2. **Output misinterpretation。**Sub-manager return "unable verify claim X。"Top manager summarize "claim X not confirmed。"Meaning drift every level。
3. **Consensus loop。**Two sub-manager disagree;top manager ask reconcile;they re-delegate down;worker re-run;sub-manager return slightly different answer;loop。CrewAI `Process.hierarchical` guard against this step limit、but limit itself now hyperparameter。

### Deciding question

Sequential(linear pipeline)vs hierarchical:does task actually have independent sub-team、or one linear flow pretend tree?If latter、use sequential。If former、use hierarchical but budget explicit reconciliation rule。

### CrewAI implementation

`Process.hierarchical` wire manager LLM over specialist crew。Manager:

- receive top-level task、
- assign subtask crew、
- evaluate crew output、
- decide whether accept、re-delegate、or iterate。

Documentation:https://docs.crewai.com/en/introduction(look "Hierarchical Process" under Core Concept)。

### LangGraph implementation

LangGraph use nested `create_supervisor` call。Inner supervisor own graph;outer supervisor treat inner graph opaque node。This cleaner than CrewAI debugging(you can step through each graph separately)but harder express dynamic reshaping tree。

Reference:https://reference.langchain.com/python/langgraph-supervisor。

## 构建

`code/main.py` run 3-level hierarchy:

- top manager:split task "engineering"和"legal" branch、
- engineering sub-manager:split "frontend"和"backend" worker、
- legal sub-manager:one worker。

Demo contrast happy path(everyone agree)against **perturbed path** where top manager decomposition mislabel "legal" "finance" and watch error cascade — sub-manager obediently do finance work、top synthesizer report finance finding、original legal question go unanswered。

跑:

```
python3 code/main.py
```

Output show both path clear side-by-side "what ask" vs "what deliver。"

## 使用

`outputs/skill-hierarchy-fitness.md` evaluate whether given task should use hierarchical、sequential、or flat supervisor。Input:task description、org structure、reconciliation budget。Output:pattern recommendation specific failure mode guard against。

## 交付成果

If you ship hierarchical:

- **Cap tree depth 2。**Three level already hide most error observability。
- **Explicit reconciliation budget。**Set max round before top manager must commit。Usually 2。
- **Provenance every synthesis。**Each node summary must cite which leaf output produce it。
- **Alert decomposition drift。**Log manager decomposition per step;diff against user query。If decomposition no longer cover query、fire alert。

## 练习题

1. Run `code/main.py` and compare happy vs perturbed。How many level manager hand-off take before top output fully diverge user question?
2. Add third level(top → sub → sub-sub → worker)。Measure how often perturbed path correct itself vs fully diverge depth grow。
3. Implement "canary" worker each sub-manager always ask original user question unchanged。Use canary answer detect decomposition drift。How should manager react when canary disagree synthesized answer?
4. Read CrewAI `Process.hierarchical` doc。Identify one concrete guardrail CrewAI apply(step limit、manager_llm constraint)and describe what failure mode target。
5. Compare nested LangGraph supervisor CrewAI hierarchical。Which make reconciliation loop cheaper detect?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Hierarchical | "Org chart pattern" | Supervisor over supervisor;only leaf do work。 |
| Manager LLM | "The boss" | LLM decompose、assign、and validate internal node。 |
| Decomposition drift | "The boss lost plot" | Top manager split no longer cover original question。 |
| Reconciliation loop | "Endless meeting" | Sub-manager disagree;top re-delegate;worker re-run;loop until budget exhausted。 |
| Depth-2 ceiling | "Don't go deeper 2 level" | Empirical guardrail:3+ level collapse observability。 |
| Canary question | "Ground truth every level" | Worker always ask original query unchanged、detect drift。 |
| Provenance chain | "Who said what" | Trace each synthesis back leaf output produce it。 |

## 延伸阅读

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — textbook hierarchical manager LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — nested supervisor via `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — why Anthropic deliberately choose flat supervisor over hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy;section coordination failure document decomposition drift