# Anthropic Workflow模式——简胜繁

> Schluntz和Zhang(Anthropic,2024年12月)分workflow(预定义path)和agent(动态tool-use)。五workflow模式cover多case。从直API调用起。仅当步不可预测加agent。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)
**时间:** ~60分钟

## 学习目标

- 名Anthropic五workflow模式:prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 释agent-vs-workflow分和每工程成本。
- 识何时选workflow胜agent(反之)。
- 实全五模式于stdlib对scripted LLM。

## 问题背景

Team reach multi-agent framework于欲单function call问题。成本真实:framework加层obscure提示、藏控流、并邀早熟复杂。Schluntz和Zhang 2024年12月文是最引用industry pushback:从简起、仅当复杂性earn cost时加。

## 概念讲解

### Workflow vs agent

- **Workflow。**LLM和工具经预定义代码path orchestrate。工程师own graph。
- **Agent。**LLM动态导己工具并取己步。模型own graph。

两各有地。Workflow更便宜、更快、更易debug。Agent解锁开放问题但使失败模式更难reason。

### Augmented LLM

全五模式基础:一LLM带三能力wired——search(取)、tool(action)、memory(持久)。任API调用可用这些。

### 五模式

1. **Prompt chaining。**Call 1 output是call 2 input。用于任务有clean线性分解时。Step间可选programmatic gate。

2. **Routing。**Classifier LLM pick何下游LLM或工具调。用于categorically异输入需异handling时(tier-1 support vs refund vs bug vs sales)。

3. **Parallelization。**并发跑N LLM call、聚合结果。两形:sectioning(异chunk)和voting(同prompt、N run、majority/synthesis)。

4. **Orchestrator-workers。**Orchestrator LLM动态决何worker(也LLM)跑并synthesize它们output。类agent循环但orchestrator不无限loop。

5. **Evaluator-optimizer。**一LLM提答、另一LLM评估。迭代直到evaluator通过。此是Self-Refine(课程05)泛化。

### 何workflow胜agent

- **可预测任务。**若你可enumerate步、你应该。
- **Cost-bound任务。**Workflow有bounded步数;agent可spiral。
- **Compliance-bound任务。**Auditor欲读graph、非从trajectory推断。

### 何agent胜workflow

- **开放研究。**当下步依赖上步回何。
- **Variable-length任务。**分钟至小时工作步数未知。
- **新domain。**当你尚不知正确workflow——先探索、后codify。

### Context-engineering companion

"Effective context engineering for AI agents"(Anthropic 2025)formalize相邻discipline:200k window是预算非容器。何include、何compact、何let context grow。详于阶段14 context compression课(阶段14早课06于本课程renumber前)。

## 构建

`code/main.py`实全五workflow模式对`ScriptedLLM`:

- `prompt_chain(input,steps)`——sequential。
- `route(input,classifier,handlers)`——classification+dispatch。
- `parallel_vote(prompt,n,aggregator)`——N run、aggregate。
- `orchestrator_workers(task,workers)`——orchestrator pick worker。
- `evaluator_optimizer(task,proposer,evaluator,max_iter)`——loop直到pass。

跑:

```
python3 code/main.py
```

每模式印trace。每模式代码行~10–15;framework cost以千计。

## 使用

- 多任务直API call。
- 仅当模式真需durable state(LangGraph)、actor-model concurrency(AutoGen v0.4)、或role templating(CrewAI)用framework。
- Reach Claude Agent SDK当你欲Claude Code harness形无重建。

## 交付成果

`outputs/skill-workflow-picker.md`为给定任务描述pick正确模式、含决策rationale和workflow不够时refactor path至agent。

## 练习题

1. 实routing带confidence threshold。Threshold下->escalate人。Tier-1 support use case threshold何land?
2. 加timeout于`parallel_vote`。一call hang时何?缺失vote何aggregate?
3. 转`evaluator_optimizer`作bandit:跨iteration留top-2 output使late good result不被late bad overwrite。
4. 合prompt chaining和routing:router pick三chain之一。测token cost vs单big-prompt alternative。
5. Pick你产feature之一。画workflow graph。数步。Agent此处实更好否?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Workflow | "预定义flow" | 工程师own LLM和tool call graph |
| Agent | "Autonomous AI" | 模型own graph;动态tool方向 |
| Augmented LLM | "LLM带工具" | LLM+search+tool+memory;atomic unit |
| Prompt chaining | "Sequential call" | Call N output是call N+1 input |
| Routing | "Classifier dispatch" | Pick何chain/model handle输入 |
| Parallelization | "Fan out" | N并发call;按sectioning或voting aggregate |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM动态pick specialist LLM |
| Evaluator-optimizer | "Proposer+judge" | Iterate直到evaluator pass;Self-Refine泛化 |

## 延伸阅读

- [Anthropic,Building Effective Agents(Dec 2024)](https://www.anthropic.com/research/building-effective-agents)——五workflow模式
- [Anthropic,Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——相伴discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——何时stateful graph earn cost
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——orchestrator-workers模式、产品化