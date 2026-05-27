# 自主编码智能体全景 (2026)

> SWE-bench Verified在不到三年内从4%提升到80.9%。相同的Claude Sonnet 4.5在SWE-agent v1上得分43.2%，在Cline自主上得分59.8%——模型周围的脚手架现在与模型本身同样重要。OpenHands（前身为OpenDevin）是最活跃的MIT许可平台，其CodeAct循环在沙箱中直接执行Python动作而非JSON工具调用。标题数字隐藏了一个方法论问题：500个SWE-bench Verified任务中有161个只需要1-2行更改，而SWE-bench Pro（10+行任务）对相同的前沿模型只有23-59%。

**类型:** 学习
**语言:** Python (stdlib, CodeAct与JSON工具调用比较)
**前置要求:** 第14阶段 · 07 (工具使用), 第15阶段 · 01 (长时程智能体)
**时间:** ~45分钟

## 问题背景

"哪个编码智能体最好"是错误的问题。正确的问题是：在匹配我工作的任务分布上，使用我将在生产中运行的脚手架，我得到什么端到端可靠性？

2022年到2026年之间，领域学到脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承载负载的。Claude Sonnet 4.5在SWE-agent v1上得分43.2% on SWE-bench Verified；在Cline自主脚手架内相同模型得分59.8%。16.6个绝对百分点差异，相同权重。基础模型是组件；循环是产品。

伴随问题是基准饱和隐藏回归。SWE-bench Verified接近饱和，简单任务尾部（500个任务中161个需要≤2行）拉高顶级分数。现实世界质量在SWE-bench Pro（10+行更改）等分布上更好地测量，相同领先者在23-59%区间。

## 概念讲解

### SWE-bench，一段话

SWE-bench（Jimenez等人）获取具有真实补丁和测试套件的GitHub问题，要求智能体生成使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工策划的500个任务子集，移除了模糊和破损任务。SWE-bench Pro是更难的后继——需要10+行更改的任务，当前前沿智能体在23-59%区间。

### 2022 → 2026曲线实际显示什么

- **2022**: 研究模型在原始SWE-bench上约4%。
- **2024**: GPT-4 + Devin风格脚手架约14%；SWE-agent约12%。
- **2025**: Aider和SWE-agent内的Claude 3.5/3.7 Sonnet推进到40-55%区间。
- **2026**: Claude Sonnet 4.5和前沿竞争者在SWE-bench Verified上70-80%+。Epoch AI的排行榜实时跟踪。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、更好的基准（Verified去除噪声）。

### CodeAct与JSON工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为OpenDevin）采取了特定的架构赌注：模型不发出JSON工具调用由主机解码和执行，而是发出Python代码，Jupyter风格内核在沙箱中运行。智能体可以在一个动作中循环文件、链式工具、捕获自己的异常。

权衡：

- **JSON工具调用**: 每个动作一轮；易于审计；组合性有限；默认安全因为每个调用经过显式验证器。
- **CodeAct**: 一个动作可以是整个程序；可组合；需要加固沙箱（OpenHands使用Docker隔离）；故障模式包括沙箱运行时允许的任何东西。

两种架构都在生产中。CodeAct在开放平台中占主导（OpenHands、smolagents）。JSON工具调用在托管服务中保持主导（Anthropic Managed Agents、OpenAI Assistants）其中提供商控制执行器。

### 2026全景中的脚手架

| 脚手架 | 许可 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker中的CodeAct | 最活跃开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 首个端到端SWE-bench脚手架 |
| Aider | Apache-2 | 本地仓库中的差异编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的VS Code智能体 | Sonnet 4.5上最高得分的开放脚手架 |
| Devin (Cognition) | 专有 | 托管VM + 规划器 | 首个"AI软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第10课详细涵盖智能体循环 |

### 为什么脚手架占主导

编码运行是长时程轨迹（第1课）。可靠性跨步骤复合。脚手架购买分数的三个位置：

1. **检索**: 找到正确的文件阅读是静默瓶颈。SWE-agent的ACI、OpenHands的文件索引和Aider的仓库映射都攻击这一点。
2. **验证器循环**: 运行测试、阅读堆栈跟踪、重试是SWE-bench上10+点的增量。
3. **故障遏制**: 错误时回滚的沙箱防止复合损害。有和没有验证器循环的相同模型看起来像两个不同产品。

### 基准饱和与真实分布

OpenHands作者和Epoch AI都标记SWE-bench Verified有简单尾部：500个任务中161个只需要1-2行更改。高分部分由此尾部驱动。SWE-bench Pro限制为10+行更改，对前沿系统返回23-59%分数。你的生产分布几乎肯定更接近Pro而非Verified。

选择智能体的含义：在自己的bug积压上运行Pro类似子集。重要的分数是你运送任务代表性任务上的分数。

## 动手实践

`code/main.py` 在固定迷你任务分布上比较两个玩具智能体脚手架：

1. **JSON工具调用**脚手架，每轮采取一个动作。
2. **CodeAct**脚手架，每动作可发出小段Python代码。

两者使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示CodeAct脚手架以更大每动作爆炸半径为代价在更少轮次解决更多任务。

## 产出成果

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计提议的编码智能体脚手架：检索质量、验证器存在、沙箱隔离、基准到分布契合度。

## 练习题

1. 运行 `code/main.py`。每个脚手架在同一任务集上需要多少轮次？每个的每动作爆炸半径是多少？

2. 阅读OpenHands论文（arXiv:2407.16741）。论文论证CodeAct在复杂任务上击败JSON工具调用。识别论文承认的一个故障模式，并写一句关于该模式何时会在生产中占主导的话。

3. 从你的bug积压中选择一个需要跨两个文件10+行更改的任务。估计(a) JSON工具调用和(b) CodeAct下前沿模型的端到端成功概率。证明差距。

4. SWE-bench Verified有161个单文件、1-2行任务。构造一个排除它们的分数。排行榜如何变化？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并命名策划会遗漏的一类。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准" | 具有真实补丁和测试套件的GitHub问题 |
| SWE-bench Verified | "清理子集" | 500个人工策划任务，存在简单尾部 |
| SWE-bench Pro | "更难子集" | 10+行更改；前沿在23-59% |
| CodeAct | "代码作为动作" | 智能体发出Python；Jupyter风格内核在沙箱中执行 |
| JSON工具调用 | "函数调用" | 每个动作是在执行前验证的结构化JSON载荷 |
| 脚手架 | "智能体框架" | 基础模型周围的检索+规划器+执行器+验证器循环 |
| ACI (智能体-计算机接口) | "SWE-agent的格式" | 为LLM人体工学设计的命令集，非人类shell |
| 验证器循环 | "测试-重试" | 运行测试、阅读输出、修订补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程编码智能体可靠性框架。
