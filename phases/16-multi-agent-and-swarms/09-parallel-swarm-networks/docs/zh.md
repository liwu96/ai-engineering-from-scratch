# Parallel / Swarm / Networked Architecture

> Contrast supervisor:no central decider。Agent read shared event bus、pick up work asynchronously、write result back。LangGraph explicitly support "Swarm Architecture" decentralized、dynamic environment。Matrix(arXiv:2511.21686)represent both control和data flow serialized message passed distributed queue eliminate orchestrator bottleneck。Tradeoff explicit:determinism和traceability scalability。Swarm fit task many independent sub-problem;it not fit task need single coherent plan。

**类型:** 学习+构建
**语言:** Python(stdlib、`threading`、`queue`)
**前置要求:** 阶段16课程05(Supervisor Pattern)、阶段16课程04(Primitive Model)
**时间:** ~75分钟

## 问题背景

Supervisor scale few worker。何hundred?Supervisor itself become bottleneck:every decision who what funnel one agent。One slow plan step stall whole system。

Swarm architecture flip design。Instead central planner dispatch work、worker pick work shared queue。"Coordination" bake event bus semantic。No orchestrator;system scale until queue。

## 概念讲解

### Shape

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

No orchestrator。Each worker repeat:pull task、process、write result(and optionally enqueue follow-up)。

### 何swarm fit

- **Many independent task。**Scraping、transforming、classifying。Task not depend each other。
- **Variable-duration work。**If some task 100ms and other 10s、swarm balance load automatically — fast worker pull next job。Supervisor anticipate duration。
- **Throughput over determinism。**You care total completion time、非strict ordering。

### 何swarm fail

- **Ordered workflow。**If step 3 need step 2 output、swarm risk step 3 fire before step 2 done。
- **Global-plan task。**Complex research question benefit planner。Swarm researcher produce independent fact、非coherent report。
- **Debugging。**With no central log and asynchronous work、reproduce bug expensive。

### Matrix(arXiv:2511.21686)

Matrix 2025 paper take swarm natural conclusion:both control flow和data flow serialized message distributed queue。No central coordinator。Fault tolerance come message durability。Scalability message broker problem、非system。

Contribution:programming model multi-agent coordination "what message topic this agent subscribe?"rather "which agent supervisor pick next?"This make system look pub/sub event mesh。

### LangGraph Swarm Architecture

LangGraph 2025 doc explicitly describe "Swarm Architecture" one multi-agent pattern:agent node、but edge form directed graph with cycle and any node can activate pool。Worker pick available work condition、非supervisor assignment。

### Failure mode:starvation和hot-spotting

If all worker pull fastest-available task、long-running task never get pick until only left。Classic queue starvation。

Mitigation:
- Priority queue explicit aging(increase priority wait time)。
- Worker specialization:some worker only take "long" task。
- Back-pressure:limit how many fast task enter queue。

### Content-based routing link

Swarm pair naturally content-based routing(Lesson 22)。Instead generic queue、have one queue per message type。Specialist worker subscribe only type。This basis message-bus architecture scale thousand agent。

## 构建

`code/main.py` implement swarm 4 worker thread pull shared `queue.Queue`。Task variable duration(some fast、some slow)。Demo contrast:

- **Sequential baseline:**one worker process all task serially。
- **Fixed assignment:**each task pre-assign specific worker(supervisor-style)。
- **Swarm:**worker pull shared queue。

Swarm balance load automatically;fixed assignment leave fast worker idle assigned task slow。

跑:

```
python3 code/main.py
```

Output show per-worker task count(swarm distribute unevenly but optimally)and wall-clock time。

## 使用

`outputs/skill-swarm-fit.md` evaluate whether task should use swarm vs supervisor。Input:task independence、duration variance、ordering requirement、debuggability need。

## 交付成果

Checklist:

- **Priority queue aging。**Prevent long-task starvation。
- **Worker idempotency。**Task may pull more once if worker crash mid-run。Worker must idempotent。
- **Durable queue。**Use Kafka、Redis Stream、or database-backed queue production。`queue.Queue` in-memory only。
- **Observability per task。**Every task trace ID;every worker log start/end it。
- **Back-pressure。**If queue grow faster worker drain、slow producer。

## 练习题

1. Run `code/main.py`。How much faster swarm than sequential variable-duration workload?How much faster than fixed assignment?
2. Add priority queue variant(use `queue.PriorityQueue`)。Assign priority task "importance" field。Observe whether low-priority task ever starve continuous load。
3. Implement hot-spot detector:log when any worker process 3× more task than slowest worker。何that indicate task-duration distribution?
4. Read Matrix paper(arXiv:2511.21686)abstract and Section 3。Identify one specific tradeoff Matrix accept(scalability gain)and one give up(traceability、determinism)。
5. Convert swarm demo use `queue.Queue`(task_type、payload)tuple、with worker subscribe only specific type。何routing rule make sense when task heterogeneous?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Swarm architecture | "Decentralized agent" | Worker pull shared queue;no central orchestrator。 |
| Event bus | "Agent subscribe topic" | Message broker route task worker type or content。 |
| Starvation | "Task never run" | Low-priority task never get pick higher-priority work arrive continuously。 |
| Hot-spotting | "One worker drown" | Load imbalance one worker get most task。 |
| Back-pressure | "Slow down producer" | Mechanism signal upstream stop produce when queue fill。 |
| Idempotent worker | "Safe re-run" | Task process twice produce same result。Required because worker may crash mid-run。 |
| Durable queue | "Survive crash" | Queue backed disk or replicated storage;task not lost when worker crash。 |
| Matrix framework | "Full message-passing swarm" | Both data和control flow serialized message distributed queue。 |

## 延伸阅读

- [LangGraph workflow and agent — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — explicit swarm support
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — full message-passing swarm
- [Anthropic engineering — why supervisor not swarm Research](https://www.anthropic.com/engineering/multi-agent-research-system) — why specific production system explicitly choose supervisor over swarm
- [AutoGen v0.4 actor-model doc](https://microsoft.github.io/autogen/stable/) — event-driven actor rewrite、closer swarm than v0.2 GroupChat