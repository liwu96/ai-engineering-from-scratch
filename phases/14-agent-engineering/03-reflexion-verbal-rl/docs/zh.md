# Reflexion:言语强化学习

> 基于梯度的RL需要数千试验和GPU集群修复失败模式。Reflexion(Shinn等,NeurIPS 2023)用自然语言做:每次失败试验后,智能体写反思、存于情景记忆、下轮试验条件于该记忆。这是Letta睡眠时间计算、Claude Code CLAUDE.md学习、和pro-workflow learn-rule模式背后的模式。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程02(ReWOO)
**时间:** ~60分钟

## 学习目标

- 命名Reflexion三个组件(Actor、Evaluator、Self-Reflector)和情景记忆角色。
- 实现stdlib Reflexion循环配binary evaluator、reflection buffer、和全新重试。
- 为给定任务选择scalar、heuristic、和self-evaluated反馈源。
- 解释为何言语强化捕获基于梯度RL需数千试验修复的错误。

## 问题背景

智能体失败任务。标准RL你会跑数千更多试验、计算梯度、更新权重。昂贵、慢、大多数生产智能体无每个失败训练预算。

Reflexion(Shinn等,arXiv:2303.11366)问不同问题:如果智能体只思考为何失败并用该思考再试?无权重更新。无梯度。只在试验间存储自然语言。

结果:ALFWorld它胜ReAct和其他非fine-tune基线。HotpotQA它改进ReAct。代码生成(HumanEval/MBPP)它设当时SOTA。全无单个梯度步。

## 概念讲解

### 三组件

```
Actor         : 产生轨迹(ReAct式循环)
Evaluator     : 评分轨迹——binary、heuristic、或self-eval
Self-Reflector: 写关于失败的自然语言反思
```

加一数据结构:

```
情景记忆: 前反思列表prepend至下轮试验提示
```

一轮试验运行Actor。Evaluator评分。如果分数低,Self-Reflector产生反思("我选错工具因为我误读问题问X实际问Y")。反思入情景记忆。下轮试验全新开始但看到反思。

### 三evaluator类型

1. **Scalar**——外部binary信号。ALFWorld成功或失败。HumanEval测试通过或失败。最简、最高信号。
2. **Heuristic**——预定义失败签名。"如果智能体连续两次产生相同动作,标记stuck。""如果轨迹超50步,标记inefficient。"
3. **Self-evaluated**——LLM评分自己轨迹。无ground truth可用时需要。较弱信号;配工具支撑验证好(课程05——CRITIC)。

2026默认是混合:scalar可用时、self-eval不可时、heuristic作安全护栏。

### 为何泛化

Reflexion不是新算法多是命名模式。几乎每个生产"自愈"智能体跑某种变体:

- Letta睡眠时间计算(课程08):分智能体反思前对话并写memory block。
- Claude Code `CLAUDE.md`/"save memory"模式:反思作为学习捕获prepend到未来session。
- pro-workflow `/learn-rule`命令:纠正作为显式规则捕获。
- LangGraph reflection节点:评分输出并路由到refine若需要的节点。

全从同洞察派:自然语言是足够富媒介在运行间携带"我从失败学什么"。

### 何时有效何时无效

Reflexion有效当:

- 有清晰失败信号(测试失败、工具错误、错误答案)。
- 任务类可重现(同类型问题可再问)。
- 反思有改进轨迹空间(足够动作预算)。

Reflexion不帮助当:

- 智能体首次已成功。
- 失败是外部(网络down、工具broken)——"网络down"反思不帮助未来运行。
- 反思变迷信——存储关于一次flaky run叙述。

2026陷阱:记忆腐化。反思累积;有些obsolete或错;重运行随情景buffer增长变慢。缓解:周期压缩(课程06)、反思TTL、或分睡眠时间清理智能体(Letta)。

## 动手实践

`code/main.py`在玩具puzzle上实现Reflexion:产生3元素list和为目标。Actor发候选list;Evaluator检查和;Self-Reflector写一行关于何错。反思入情景记忆用于下轮试验。

组件:

- `Actor`——见反思时改进的脚本化策略。
- `Evaluator.binary()`——目标和pass/fail。
- `SelfReflector`——生失败一行诊断。
- `EpisodicMemory`——配TTL语义的bounded list。

运行:

```
python3 code/main.py
```

Trace显示三轮试验。试验1失败、反思存储、试验2见反思改进但仍失败、试验3成功。与基线运行比较(无反思)——它停在试验1答案。

## 实际应用

LangGraph发布reflection作为节点模式。Claude Code `/memory`命令和pro-workflow `/learn-rule`外化情景buffer作为markdown文件。Letta睡眠时间计算在downtime运行Self-Reflector使主智能体保持延迟bound。OpenAI Agents SDK不直接发布Reflexion;你用custom Guardrail按分数reject轨迹和跨运行存活的memory `Session`构建。

## 产出成果

`outputs/skill-reflexion-buffer.md`创建并维护情景buffer配反思捕获、TTL、和去重。给定任务类和失败,它发实际帮助下轮试验的反思(非泛泛"更小心")。

## 练习题

1. 从binary切换到scalar evaluator返回距离metric(离目标多远)。收敛更快否?
2. 给反思加10轮试验TTL。老反思在那点后帮还是伤?
3. 实现heuristic evaluator:如果相同动作重复标记试验stuck。这与Self-Reflector如何交互?
4. 运行Reflexion配对抗Actor忽略反思。强制Actor注意它们的最小反思提示工程是什么?
5. 阅读Reflexion论文Section 4关于AlfWorld。概念重现130%成功率改进:vs vanilla ReAct关键delta是什么?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Reflexion | "自纠正" | Shinn等2023——Actor、Evaluator、Self-Reflector配情景记忆 |
| 言语强化 | "无梯度学习" | 自然语言反思prepend至下轮试验提示 |
| 情景记忆 | "每任务反思" | 一任务类的bounded前反思buffer |
| Scalar evaluator | "Binary成功信号" | Pass/fail或数值分来自ground truth |
| Heuristic evaluator | "Pattern基detector" | 预定义失败签名(如stuck-loop、too-many-steps) |
| Self-evaluator | "LLM-as-judge于己trace" | 无ground truth低信号fallback——配工具支撑验证 |
| 记忆腐化 | "Stale reflection" | 情景buffer满obsolete entry;用压缩/TTL修复 |
| 睡眠时间反思 | "Async自反思" | 热路径外运行Self-Reflector使主智能体快 |

## 延伸阅读

- [Shinn等,Reflexion: Language Agents with Verbal Reinforcement Learning(arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)——标准论文
- [Letta,Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)——生产异步反思
- [Anthropic,Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——管情景buffer作context部分
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——reflection节点模式