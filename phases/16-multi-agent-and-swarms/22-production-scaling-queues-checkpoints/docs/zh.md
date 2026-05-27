# 生产扩展——队列、检查点、持久性

> 扩展多Agent系统到数千并发运行需**持久执行**。LangGraph运行时每超步写检查点`thread_id`键(Postgres默认)；工作者崩溃释放租约另一工作者恢复。Agent可无限睡等人工输入。**MegaAgent**(arXiv:2408.09955)跑每Agent生产者消费者队列三态(Idle / Processing / Response)两层协调(组内聊天+组间管理聊天)。**Fiber/async**赢LLM流线程每任务：线程99%时间等token空、fiber协作yield于I/O。反观点：Ashpreet Bedi"扩展Agent软件"主张**FastAPI + Postgres +无其他**直到负载证明——简单架构比预期走更远。本lesson建持久检查点日志、每Agent工作队列带状态转换、async-vs-thread demo、落地务实"从简开始"规则。

**类型:** 学习+构建
**语言:** Python(stdlib、`asyncio`、`sqlite3`)
**前置要求:** 阶段16课程09(Parallel Swarm Networks)、阶段16课程13(Shared Memory)
**时间:** ~75分钟

## 问题背景

原型多Agent系统在一笔记本三Agent内存事件环工作。移到生产：

- Agent有时跑小时(长研究、人在环等待)。
- 工者进程崩溃。重启丢状态。
- 峰负载是平均10x；需横向扩展。
- 用户付每Agent运行；需恰好一次语义收费。

内存事件环不做任何这些。需持久执行层在下。2026 canonical选项：

1. 带检查点工作流引擎(Temporal、LangGraph运行时)。
2. 带状态存储消息队列(Postgres + SQS/RabbitMQ)。
3. Actor模型框架(MegaAgent每Agent生产者消费者)。
4. 手滚FastAPI + Postgres(Bedi论证)。

本lesson建每迷你。

## 概念讲解

### 持久执行，模式

持久执行引擎每"步"(超步，LangGraph语言)后持久全程序状态。崩溃：

```
工者崩溃中步
  -> 租超时
  -> 另工者拾thread_id
  -> 从最后检查点恢复
  -> 无重复副作用
```

工作要求：

- **可序列化状态。**所有Agent状态必须可持久。带活数据库连接函数闭包不存活。
- **确定性恢复。**给同状态同输入，Agent产同行动(或对LLM调用退外部确定性oracle)。
- **幂等副作用。**外部调用(工具调用、支付)必须幂等或用去重键。

LangGraph每超步写检查点；Temporal每activity写；Restate用事件源日志。三都实现同模式。

### LangGraph运行时

每Agent有`thread_id`；状态是类型dict；每超步写一行到检查点表。恢复时，运行时从最后检查点重放非从头。Agent可`interrupt()`等人工输入；运行时持久释放工者。当输入到达，任工者可恢复。

这是2026年4月参考生产设计。

### MegaAgent每Agent队列

arXiv:2408.09955描述规模实验：一集群数千并发Agent。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- 给Agent i消息
  out_queue  -> 回复+副作用

协调器:
  组内聊天  (同组Agent)
  组间管理聊天  (高层路由)
```

两层协调让组内对话密组间疏——用于保持成本线性于数千Agent模式。

### Async vs线程每任务

LLM调用I/O绑。线程等下token99%时间空。线程每~1MB RAM；10,000并发调用，栈10GB。

Fiber(Python `asyncio`、Go goroutine、Rust `tokio`)协作yield于I/O。同10,000调用舒适在进程。LLM-Agent规模，async非优化——是架构。

例外：CPU绑后处理(嵌入、tokenizer技巧)仍想线程或进程。分离I/O层CPU层。

### Bedi反观点

"扩展Agent软件"(Ashpreet Bedi，2026)论证多数团队在测量负载前过工程。务实默认：

- FastAPI + Postgres。
- 每Agent运行是一行；状态原地更新乐观并发。
- 后台任务通过`pg_notify`或简单Celery工者。
- 重试策略在应用代码。

负载~100并发Agent运行可控任务下，这常全需。测量失败时升级。

规则：当你打简单架构不能解具体问题时采用持久执行框架。过早采用烧时间于不回报仪式。

### 恰好一次语义

付费Agent运行，需"恰好一次有效"(至少一次投递+幂等消费)。工程移动：

- **每运行去重键。**在每副作用调用含。
- **Outbox模式。**副作用先写表，然后单独进程执行。两步幂等。
- **补偿交易。**当副作用成功但其追踪写失败，安排补偿。

这些是数据库工程模式非LLM特定。LLM税只是LLM调用慢；其余标准分布式系统。

### Rainbow部署

Anthropic多Agent研究系统用"rainbow部署":多版Agent运行时并发运行所以长跑Agent不每代码部署杀。新版流量切片canary；旧版Agent完成时退。

这是长跑状态系统标准；2026适配是Agent可活小时，所以部署周期必须容纳。

### Canonical生产清单

- 持久状态(检查点、快照、或outbox+可重放日志)。
- 幂等副作用。
- LLM调用async I/O层。
- 至少一次投递去重。
- Rainbow/canary部署状态工作负载。
- 可观察性：每Agent轨迹、超步审计、重试计数。

## 构建

`code/main.py`实现：

- `CheckpointStore`——SQLite后检查点日志thread-id键。每超步追加行。
- `run_with_checkpoint(agent, thread_id)`——模拟中跑崩溃；第二工者从最后检查点恢复。
- `AgentQueue`——每Agent Idle / Processing / Response状态机带小工作队列。
- `demo_async_vs_threads()`——500并发模拟"LLM调用"通过asyncio和线程；报告墙钟和峰内存(近似)。

跑：

```
python3 code/main.py
```

预期输出：检查点恢复在模拟崩溃后成功；async版处理500并发调用<1s；线程版取几秒每并发单元用数量级更多内存。

## 使用

`outputs/skill-scaling-advisor.md`建议持久执行选择：FastAPI + Postgres、LangGraph运行时、Temporal、或自定义。校准负载、状态保留需、部署频率。

## 交付成果

Canonical生产硬化：

- **从简开始(Bedi规则)。**FastAPI + Postgres直到测量失败。
- **优化前仪表一切。**每运行延迟直方图、每步时间、重试计数、失败分类。
- **副作用Outbox模式。**尤其支付和外部API调用。
- **Rainbow部署。**从不杀部署中飞行Agent运行。
- **当**你打具体问题时采用持久执行引擎(Temporal / LangGraph / Restate)：小时长人在环等待、跨区协调、复杂重试/补偿策略。
- **I/O层Async。**线程仅CPU绑后处理。

## 练习题

1. 跑`code/main.py`。确认检查点恢复工作；测量async vs线程并发差异。
2. 实现**outbox**表：每工具调用先写outbox，然后单独goroutine/任务执行。跑两次验证幂等。
3. 模拟**rainbow部署**：两并发运行时版本；路由新thread_id一半到每；确认旧版飞行线程不打断。
4. 读LangGraph运行时文档(链接下)。识别运行时哪些功能手滚FastAPI + Postgres版本复现最久。那是采用理由还是可推迟？
5. 读MegaAgent(arXiv:2408.09955)Section 3。两层协调(组内+组间管理聊天)显式。草图如何映射到消息队列两队列族。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 持久执行 | "持久程序状态" | 引擎每超步后写状态；崩溃恢复确定性。 |
| 超步 | "事务边界" | 检查点间工作单元。LangGraph术语。 |
| thread_id | "Agent运行标识符" | 绑检查点和恢复逻辑键。 |
| 幂等性 | "安全重试" | 重复副作用产同结果一次尝试。 |
| Outbox模式 | "解耦副作用" | 写意图到表；单独执行器执行并标记完成。 |
| 至少一次投递 | "可能重复" | 消息队列语义；去重键使消费有效一次。 |
| Rainbow部署 | "重叠版本" | 长跑工作负载多运行时版本并发。 |
| Async fiber | "协作yield" | 用户模式并发；I/O绑负载比线程便宜。 |
| 检查点 | "状态快照" | 超步边界序列化状态；恢复键。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph运行时设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 每Agent生产者消费者队列；数千并发Agent两层协调
- [Matrix](https://arxiv.org/abs/2511.21686) — 去中心化框架消息队列作协调基
- [Temporal文档](https://docs.temporal.io/) — 持久执行参考工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产教训包括rainbow部署