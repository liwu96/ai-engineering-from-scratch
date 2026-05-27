# Self-Refine 和 CRITIC:迭代输出改进

> Self-Refine(Madaan 等,2023)使用一个LLM扮演三个角色——generate、feedback、refine——在循环中运行。7个任务平均提升+20绝对值。CRITIC(Gou 等,2023)通过将验证路由到外部工具来强化feedback步骤。2026年,这个模式在所有框架中发布,称为"evaluator-optimizer"(Anthropic)或guardrail循环(OpenAI Agents SDK)。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程03(Reflexion)
**时间:** ~60分钟

## 学习目标

- 陈述Self-Refine的三个提示词(generate、feedback、refine)并解释为何history对refine提示词至关重要。
- 解释CRITIC的关键洞察:LLM在没有外部支撑的情况下自验证是不可靠的。
- 实现一个带有history和可选外部验证器的stdlib Self-Refine循环。
- 将此模式映射到Anthropic的"evaluator-optimizer"工作流和OpenAI Agents SDK的输出guardrails。

## 问题背景

一个智能体产生了几乎正确的答案。也许一行代码有语法错误。也许摘要太长。也许计划漏掉了边界情况。你需要的是:智能体批判自己的输出,然后修复它。

Self-Refine证明了这在单个模型、无训练数据、无RL的情况下可行。但有一个陷阱:LLM在硬性事实的自验证上表现不佳。CRITIC指明了修复方法——将verify步骤路由到外部工具(搜索、代码解释器、计算器、测试运行器)。

这两篇论文共同定义了2026年迭代改进的默认模式:生成、验证(可用外部时)、改进、验证器通过时停止。

## 概念讲解

### Self-Refine (Madaan 等,NeurIPS 2023)

一个LLM,三个角色:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当feedback说"无问题"或预算耗尽时停止。
```

关键细节:`refine`能看到完整history——所有之前的输出和批判——所以不会重复错误。论文对此进行了消融实验:删除history后质量急剧下降。

要点:7个任务(math、code、acronym、dialog)平均+20绝对提升,包括GPT-4。无需训练、无需外部工具、单模型。

### CRITIC (Gou 等,arXiv:2305.11738,v4 2024年2月)

Self-Refine的弱点:feedback步骤是LLM评判自己。对于事实性声明这是不可靠的(幻觉对产生它的模型来说看起来很有说服力)。CRITIC将`feedback(task, output)`替换为`verify(task, output, tools)`,其中`tools`包括:

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定验证器(单元测试、类型检查器、linter)。

验证器产生基于工具结果的、有据可依的结构化批判。改进器随后基于此批判进行调整。

要点:CRITIC在事实性任务上超越Self-Refine,因为批判是有据可依的。在没有外部验证器的任务上(创意写作、格式化),CRITIC退化为Self-Refine。

### 停止条件

两种常见形式:

1. **验证器通过。**外部测试返回成功。可用时优先选择(单元测试、类型检查器、guardrail断言)。
2. **无feedback发出。**模型说"输出没问题。"更便宜但不可靠;配合max-iteration上限使用。

2026默认:组合它们。"验证器通过时停止,或模型说没问题且iteration>=2,或iteration>=max_iterations。"

### Evaluator-Optimizer (Anthropic,2024)

Anthropic 2024年12月的文章将此命名为五种工作流模式之一。两个角色:

- Evaluator:评分输出并产生批判。
- Optimizer:根据批判修订输出。

循环直到evaluator通过。这就是Anthropic框架下的Self-Refine/CRITIC。Anthropic补充的关键工程细节:evaluator和optimizer的提示词应显著不同,以免模型只是橡皮图章式批准。

### OpenAI Agents SDK 输出guardrails

OpenAI Agents SDK以"output guardrails"的形式发布此模式。Guardrail是在智能体最终输出上运行的验证器。如果guardrail触发(抛出`OutputGuardrailTripwireTriggered`),输出被拒绝,智能体可以重试。Guardrails可以调用工具(CRITIC风格)或作为纯函数(Self-Refine风格)。

### 2026陷阱

- **橡皮图章循环。**同一模型用相同提示风格做生成和批判会收敛于"看起来不错。"使用结构不同的提示词,或用更小更便宜的模型做批判。
- **过度改进。**每次改进pass增加延迟和token。预算1-3次pass;之后升级到人工审查。
- **简单任务上用CRITIC。**如果没有外部验证器,CRITIC退化为Self-Refine;不要为stub验证器付出延迟成本。

## 动手实践

`code/main.py`在玩具任务上实现Self-Refine和CRITIC:给定主题产生简短bullet列表。验证器检查格式(3个bullet、每个60字符以下)。CRITIC添加外部"事实验证器"惩罚已知的幻觉。

组件:

- `generate`——脚本化的生产者。
- `feedback`——LLM风格的自批判。
- `verify_external`——CRITIC风格的有据验证器。
- `refine`——根据history重写输出。
- 停止条件——验证器通过或最多4次迭代。

运行:

```
python3 code/main.py
```

比较Self-Refine与CRITIC的运行。CRITIC捕获Self-Refine遗漏的事实错误,因为外部验证器有自批判缺乏的依据。

## 实际应用

Anthropic的evaluator-optimizer是用Claude友好语言描述的此模式。OpenAI Agents SDK的output guardrails是CRITIC形态(guardrails可调用工具)。LangGraph发布了一个读起来像Self-Refine的reflection节点。Google Gemini 2.5 Computer Use添加了每步安全evaluator,是CRITIC变体:每个action在commit前验证。

## 产出成果

`outputs/skill-refine-loop.md`根据任务形态、验证器可用性和迭代预算配置evaluator-optimizer循环。输出generator、evaluator/verifier和optimizer的提示词,以及停止策略。

## 练习题

1. 用max_iterations=1运行玩具任务。CRITIC仍然有帮助吗?
2. 将外部验证器替换为有噪声的版本(随机30%假阳性)。循环会做什么?这是2026年大多数guardrail栈的现实。
3. 实现"generator-critic不同模型"变体:大模型生成、小模型批判。胜过同模型吗?
4. 阅读CRITIC Section 3(arXiv:2305.11738 v4)。命名三类verification-tool并各举一例。
5. 将OpenAI Agents SDK的`output_guardrails`映射到CRITIC的verifier角色。SDK哪里做错了,哪里做对了?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Self-Refine | "LLM修复自己" | 单模型中的Generate->feedback->refine循环,带有history |
| CRITIC | "工具支撑的验证" | 用外部验证器(search、code、calc、tests)替换feedback |
| Evaluator-Optimizer | "Anthropic工作流模式" | 两个角色——evaluator评分、optimizer修订——循环至收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK验证器,在智能体产生输出后运行 |
| Verify步骤 | "批判阶段" | 承重决策:有据还是自评 |
| Refine history | "模型已尝试什么" | 之前的output+critique prepend到refine提示词;删除后质量崩塌 |
| 橡皮图章循环 | "自我同意失败" | 同提示词critique返回"看起来不错";用结构不同的提示词修复 |
| 停止条件 | "收敛测试" | 验证器通过或无feedback且iteration上限;永不单条件 |

## 延伸阅读

- [Madaan等,Self-Refine(arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)——标准论文
- [Gou等,CRITIC(arXiv:2305.11738)](https://arxiv.org/abs/2305.11738)——工具支撑的验证
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——evaluator-optimizer工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)——output guardrails作为CRITIC形态的verifier