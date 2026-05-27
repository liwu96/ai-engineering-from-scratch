# 长时程后台智能体：持久执行

> 生产长时程智能体不在 `while True` 中运行。每次LLM调用成为带检查点、重试和重放的活动。Temporal的OpenAI Agents SDK集成于2026年3月GA。Claude Code Routines (Anthropic)运行计划的Claude Code调用而不需要持久本地进程。会话在人工输入时暂停、在部署中存活、并从按 `thread_id` 键入的最新检查点恢复。在新人体工程学背后是一个旧模式——工作流编排——有一个新输入：LLM调用作为必须在恢复时确定性重放的非确定性活动。

**类型:** 学习
**语言:** Python (stdlib, 最小持久执行状态机)
**前置要求:** 第15阶段 · 10 (权限模式), 第15阶段 · 01 (长时程智能体)
**时间:** ~60分钟

## 问题背景

考虑运行四小时的智能体。它调用三个工具，提示用户两次，进行四十次LLM调用。中途，它运行的主机重启。发生什么？

- 在天真 `while True` 循环中：一切丢失。运行从头重启。三个工具调用（带真实副作用）再次执行。用户再次被提示他们已经批准的事情。四十次LLM调用重新计费。
- 使用持久执行：运行从最近检查点恢复。已完成活动不重新执行；结果从持久日志重放。用户不重新批准他们已经批准的事情。已进行的LLM调用不重新计费。

这是工作流引擎十年来交付的相同模式（Temporal、Cadence、Uber的Cherami）。新的是LLM调用现在是一种活动——非确定性、昂贵、有副作用——它们干净地适合此模式。

本课的运行主题：长时程可靠度衰减（METR观察到"35分钟衰减"——成功率随时间范围大致二次下降）。持久执行使运行比可靠度配置支持的更长，如果设计正确这是一种新的安全失败方式，如果设计错误则不安全。

## 概念讲解

### 活动、工作流和重放

- **工作流**: 确定性编排代码。定义活动序列、分支、等待。必须是确定性的，因此可以从事件日志重放而无令人惊讶的分歧。
- **活动**: 非确定性、可能失败的工作单元。LLM调用、工具调用、文件写入、HTTP请求。每个活动在开始前记录其输入，完成后记录其输出。
- **事件日志**: 持久后端存储。每个活动开始、完成、失败、重试，以及每个工作流决策都被记录。
- **重放**: 恢复时，工作流代码从开始重新运行；每个已完成的活动返回其记录结果而不重新执行。仅未完成的实际运行。

这与React针对虚拟DOM重新渲染，或Git从提交重建工作树的形状相同。编排器中的确定性使持久性便宜。

### 为什么LLM调用适合此模式

LLM调用是：
- 非确定性的（temperature > 0；即使temperature 0也在模型版本间漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果它们调用工具）。

这正是活动配置文件。将每个LLM调用包装为活动给你指数退避重试、跨重启的检查点，以及可重放的调试跟踪。

### 按 `thread_id` 键入的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects和Claude Code Routines都收敛于相同API形状：`thread_id`（或等效）标识会话；每个状态转换持久到后端（PostgreSQL默认、SQLite用于开发、Redis用于缓存）；恢复读取最新检查点。

后端选择很重要：

- **PostgreSQL**: 持久、可查询、跨部署存活。LangGraph默认。
- **SQLite**: 仅本地开发；跨主机丢失数据。
- **Redis**: 快速但短暂，除非配置AOF/快照。
- **Cloudflare Durable Objects**: 透明分布式；按唯一键限定范围；存活数小时到数周。

### 人工输入作为一等状态

提议-然后-提交（第15课）需要持久的"等待人工"状态。工作流暂停，外部队列保持待处理请求，批准从该点精确恢复。没有持久性这是尽力而为；有了它，隔夜批准到达，工作流在早上恢复。

### 35分钟衰减

METR观察到每个测量的智能体类别在~35分钟连续操作后显示可靠度衰减。将任务时长翻倍大致将失败率翻四倍。持久执行不修复此；它让你运行比可靠度配置支持的更长。安全模式是将持久性与需要新HITL的重新进入检查点结合，以及无论挂钟时间如何都限制总计算量的预算紧急停止开关（第13课）。

### 持久执行是错误答案时

- 短于几分钟无人工输入的运行。开销>收益。
- 严格只读信息检索。
- 正确性需要端到端在一个上下文窗口内的任务（一些推理任务；一些一次性生成）。

## 动手实践

`code/main.py` 在stdlib Python中实现最小持久执行引擎。它支持：

- `@activity` 装饰器将输入和输出记录到JSON事件日志。
- 序列活动的工作流函数。
- `run_or_replay(workflow, event_log)` 函数重放已完成活动而不重新执行它们。

驱动程序模拟三活动工作流，中途崩溃，并显示(a)天真重试重新执行一切与(b)重放仅运行缺失活动。

## 产出成果

`outputs/skill-durable-execution-review.md` 审查提议的长运行智能体部署的正确持久执行形状：活动、确定性、检查点后端、人工输入状态和HITL-on-resume策略。

## 练习题

1. 运行 `code/main.py`。观察天真重试与重放之间活动执行计数的差异。更改崩溃点并显示重放计数相应变化。

2. 将玩具引擎转换为显式使用 `thread_id`。模拟共享引擎的两个并发会话并确认它们的事件日志不冲突。

3. 在玩具引擎中取一个活动。引入非确定性（工作流决策内的挂钟时间戳）。演示重放时的分歧。解释真实引擎如何处理此（副作用注册、`Workflow.now()` API）。

4. 阅读LangChain"Runtime behind production deep agents"帖子。列出运行时持久的每个状态并命名每个覆盖的故障模式。

5. 为6小时自主编码任务设计检查点策略。你在哪里检查点？崩溃时恢复是什么样？什么需要新HITL？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 工作流 | "智能体的脚本" | 确定性编排代码；可从事件日志重放 |
| 活动 | "一步" | 非确定性单元（LLM调用、工具调用）；前后记录 |
| 事件日志 | "后端存储" | 每个状态转换的持久记录 |
| 重放 | "恢复" | 重新运行工作流；已完成活动返回记录结果而不重新执行 |
| 检查点 | "保存点" | 按thread_id持久的状态；恢复时最新获胜 |
| thread_id | "会话键" | 限定持久状态的标识符 |
| 35分钟衰减 | "可靠度衰减" | METR：成功率随时间范围大致二次下降 |
| 非确定性 | "重放时漂移" | 挂钟、随机、LLM输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时要求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM调用的活动形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35分钟衰减参考。
