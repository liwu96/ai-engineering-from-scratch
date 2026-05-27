# ReWOO 和 Plan-and-Execute:解耦规划

> ReAct在一个流中交错thought和action。ReWOO分离它们:先做一个大规划,后执行。HotpotQA上5倍少token、+4%准确率,你可以将planner蒸馏到7B模型。Plan-and-Execute泛化它;Plan-and-Act将其扩展到网页导航。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 阶段14课程01(Agent Loop)
**时间:** ~60分钟

## 学习目标

- 解释为何ReWOO的Planner/Worker/Solver分离比ReAct的交错循环节省token并提高鲁棒性。
- 实现plan DAG、依赖顺序executor和合成worker输出的solver——全用stdlib。
- 用2026"五种工作流模式"框架(Anthropic)决定任务应运行plan-then-execute还是交错ReAct。
- 认识何时Plan-and-Act的合成规划数据对长horizon网页或移动任务需要。

## 问题背景

ReAct的交错thought-action-observation循环简单灵活,但每个工具调用必须携带完整先前上下文——包括每个之前的thought。Token使用随深度二次增长。更糟:当工具中途失败时,模型必须从错误观察重新推导整个计划。

ReWOO(Xu等,arXiv:2305.18323,2023年5月)注意到这点并做了一个赌注:先规划整个事物,并行获取证据,最后合成答案。一个LLM调用规划、N个工具调用获取证据(可并行)、一个LLM调用解决。交易是更少灵活性(计划静态)换取更好token效率和更清晰失败模式。

## 概念讲解

### 三角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool调用,可能并行)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner产生DAG。每个节点命名工具、其参数、和依赖哪些更早节点(引用如`#E1`, `#E2`)。Workers按拓扑顺序执行节点。Solver将所有东西缝合在一起。

### 为何5倍少token

ReAct提示长度随步数线性增长。第10步时,提示包含thought 1加action 1加observation 1加thought 2加action 2加observation 2等。每个中间步骤也冗余包含原始提示。

ReWOO付一个planner提示(大)、N个小worker提示(每个只是工具调用,无链)、和一个solver提示。HotpotQA上论文测量约5倍少token同时+4绝对准确率。

### 为何更鲁棒

如果worker 3在ReAct中失败,循环必须在流中途从错误推理出来。在ReWOO中,worker 3返回错误字符串;solver在上下文中看到它与原始计划并可以优雅降级。失败定位是每节点而非每步。

### Planner蒸馏

论文第二个结果:因为planner不看到observation,你可以在175B教师planner输出上fine-tune 7B模型。小模型处理规划;大模型在推理时不需要。这现在标准——许多2026生产智能体用小planner和大executor或反之。

### Plan-and-Execute (LangChain, 2023)

LangChain团队2023年8月文章将ReWOO泛化为模式名:Plan-and-Execute。前planner发出步列表,executor运行每步,可选replanner可以在观察结果后修订。这比ReWOO更接近ReAct(replanner将observation带回规划)但保留token节省。

### Plan-and-Act (Erdogan等,arXiv:2503.09572,ICML 2025)

Plan-and-Act将模式扩展到长horizon网页和移动智能体。关键贡献是合成规划数据:标注轨迹生成器产生训练数据,其中规划显式。用于fine-tune planner模型在WebArena类任务30-50步后仍工作,单个ReAct轨迹失去coherence处。

### 何时选哪个

| 模式 | 何时 |
|------|------|
| ReAct | 短任务、未知环境、需反应式异常处理 |
| ReWOO | 配已知工具的结构任务、token敏感、可并行化证据 |
| Plan-and-Execute | 类ReWOO但配部分执行后重新规划 |
| Plan-and-Act | 长horizon(>30步)、web/mobile/computer-use |
| Tree of Thoughts | 搜索值得付费(课程04) |

Anthropic 2024年12月指导:从最简单开始。如果任务是一个工具调用加摘要,不建ReWOO。如果任务是40步研究任务,不只做ReAct。

## 动手实践

`code/main.py`实现玩具ReWOO:

- `Planner`——脚本化策略从提示发出plan DAG。
- `Worker`——通过注册表分发每个节点的工具调用。
- `Solver`——脚本化组合读证据产生最终答案。
- 依赖解析——引用如`#E1`被先前worker输出替换。

Demo回答"法国首都人口round to millions是多少?"用两步计划:(1)查首都,(2)查人口,后solve。

运行:

```
python3 code/main.py
```

Trace显示完整计划先、后worker结果、后solver组合。比较token数(我们打印粗略字符数)与ReAct式交错运行——ReWOO在这类结构任务胜出。

## 实际应用

LangGraph发布Plan-and-Execute作为recipe(`create_react_agent`用于ReAct、custom graph用于plan-execute)。CrewAI Flow直接编码模式:你前定义任务Flow DAG执行。Plan-and-Act合成数据方法仍多研究;运行时模式(显式plan DAG)通过LangGraph和CrewAI Flow在生产发布。

## 产出成果

`outputs/skill-rewoo-planner.md`从用户请求生成ReWOO plan DAG,给定工具目录。它在交给executor前验证计划(无环、每个引用解析、每个工具存在)。

## 练习题

1. 并行化独立plan节点的worker执行。6节点DAG配2并行组买你什么?
2. 添加replanner节点若任worker返回错误触发。使ReWOO成为Plan-and-Execute的最小改动是什么?
3. 将`Planner`替换为小模型(7B类)并保留`Solver`在前沿模型。比较端到端质量——分离哪里失败?
4. 阅读ReWOO论文Section 4关于planner蒸馏。概念重现175B->7B结果:需要什么训练数据,如何评分plan质量?
5. 将玩具移植到Plan-and-Act轨迹形状:plan是sequence而非DAG。什么权衡改变?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ReWOO | "无observation推理" | 规划、后并行获取证据、后solve——规划提示无observation |
| Plan-and-Execute | "LangChain的plan-execute模式" | ReWOO配可选replanner节点执行后 |
| Plan-and-Act | "缩放plan-execute" | 显式planner/executor分离配合成plan训练数据用于长horizon任务 |
| Evidence reference | "#E1, #E2, ..." | Plan节点placeholder分发时被先前worker输出替换 |
| Planner distillation | "小planner,大executor" | 在大教师planner trace上fine-tune小模型 |
| Token效率 | "少round trip" | HotpotQA论文vs ReAct 5倍少token |
| DAG executor | "拓扑dispatcher" | 依赖顺序运行plan节点;每层并行 |

## 延伸阅读

- [Xu等,ReWOO: Decoupling Reasoning from Observations(arXiv:2305.18323)](https://arxiv.org/abs/2305.18323)——标准论文
- [Erdogan等,Plan-and-Act(arXiv:2503.09572)](https://arxiv.org/abs/2503.09572)——配合成plan的缩放planner-executor
- [LangGraph Plan-and-Execute教程](https://docs.langchain.com/oss/python/langgraph/overview)——框架recipe
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——选最简模式工作