# 投票、自一致性与辩论拓扑

> 最便宜聚合：采样N个独立Agent、多数投票。Wang et al. 2022自一致性用单模型采样N次做这。多Agent扩展为**异构**Agent打破单一化——不同模型、不同提示、不同温度、不同上下文。超越多数投票，辩论拓扑重要：MultiAgentBench(arXiv:2503.01935，ACL 2025)评估星/链/树/图协调，发现**图最适合研究**，~4 Agent以上有"协调税"。AgentVerse(ICLR 2024)记录两种涌现模式——志愿行为和从众行为——从众既是特性（找共识）也是风险（群体思维，Lesson 24）。本lesson映射拓扑空间、构建各变体、测量协调税。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程07(Society of Mind和Debate)、阶段16课程14(共识与BFT)
**时间:** ~75分钟

## 问题背景

辩论可提高精度(Du et al., arXiv:2305.14325)。也可降精度。辩论是否帮助取决于四个结构选择：

1. 谁对谁说话（拓扑）。
2. 多少轮（Du 2023：轮数和Agent数各自独立重要）。
3. Agent是否异构（不同基座模型打破单一化）。
4. 是否有对抗声音（钢铁人论证vs稻草人论证）。

团队把"跑5个Agent投票"附任务常比单Agent退步。失败非随机。追踪拓扑和异构性。本lesson是拓扑地图。

## 概念讲解

### 自一致性，单模型基线

Wang et al. 2022("Self-Consistency Improves Chain of Thought Reasoning")同模型温度>0采样N次，多数投票推理路径答案。GSM8K结果：N=40样本比单贪婪解码显著增益。自一致性是单Agent多Agent投票先驱。

限制：自一致性用一基座模型。错误构造相关。若模型有系统偏，所有N样本共享。

### 多Agent投票，异构扩展

替换N样本为N*不同*Agent。不同基座模型(Claude、GPT、Llama)、不同提示、不同工具访问。好处：错误不相关。代价：不同Agent成本不同；协调它们增加开销。

2026异构辩论canonical名称**A-HMAD**——对抗异构多Agent辩论。非通用采纳，但论文用此术语指"不同模型辩论，减少单一化崩溃相关错误"。

### 四种拓扑

```
星                链               树                图

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (全连接)
```

星：一中心，所有其他只对中心说话。等价于supervisor-worker无后通道。
链：线性，每个Agent见前一个输出。管道式。
树：分层，用于分层Agent系统(Lesson 06)。
图：任意对任意。包括全连接团和任意DAG。

### 协调税(MultiAgentBench)

MultiAgentBench(MARBLE，ACL 2025，arXiv:2503.01935)基准测试星、链、树、图于任务套件含研究、编程、规划。关键测量结果：

- **图**拓扑赢研究任务。信息任意对任意流；Agent可互批评。
- **星**赢快答案事实任务。中心过滤整合。
- **链**赢分步管道（分阶段精炼）。
- **协调税**出现在图拓扑~4 Agent以上。墙钟和token成本增长快于质量。

4 Agent天花板是实证非基本。反映2026 LLM上下文容量：每个Agent上下文填满同伴输出，加Agent N+1边际价值在所有人可见所有人后下降。

### 多Agent辩论策略("Should we be going MAD?")

arXiv:2311.17371是2023 MAD策略调研。关键发现被他人复现：与自一致性*结构相似*MAD变体（独立采样+聚合）同预算时常输自一致性。MAD最帮助当Agent真正异构且辩论有对抗结构（一个Agent反对论证）。

### AgentVerse涌现模式

AgentVerse(ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf)记录多Agent辩论即使无显式设计也涌现的行为：

- **志愿。**Agent未提示主动提供帮助("我可做下一步")。有用：分配工作给最有能力Agent子任务。
- **从众。**Agent调整立场匹配批评者，即使批评者错。这是辩论等价讨好(Lesson 14)。

从众是为何辩论直到一致奖励霸凌。限轮加独立法官缓解。

### 异构性：真正移动精度的旋钮

2024-2026实用文献模式：换你N Agent之一为不同基座模型比增加N 1更大精度提升。直觉是单一化——每个新独立错误源比额外相关样本更值。

极限，异构胜数量。三不同模型比五同模型副本在多数有干净真值任务上好。

### Jury方法

Sibyl框架(Minsky-LLM文献引用)规范化"jury"——小集合专业化Agent每阶段投票精炼答案。不同于朴素多数投票，jury有角色：一个Agent交叉审问、一个提供上下文、一个评分可信度。Jury方法介于朴素投票（便宜、单一化倾向）和全MAD（昂贵、从众倾向）。

### 投票带辩论何时主导

- 问题有真值（事实、数学、代码行为）。投票收敛有意义。
- Agent可访问不同源或工具（异构可用）。
- 轮数有限（典型2-3）且有独立法官或验证者。
- 预算允许3-5 Agent。图拓扑超5-7，协调税主导。

### 投票带辩论何时有害

- 问题意见型。Agent收敛看最自信而非最正确答案。
- 所有Agent共享基座模型。单一化使共识无意义。
- 轮数无界。从众每次赢。
- 任务简单。单Agent自一致性N=5更便宜且同样准确。

## 构建

`code/main.py`实现：

- `run_star(agents, hub, question)`——中心轮询每个工作者，聚合。
- `run_chain(agents, question)`——顺序精炼。
- `run_tree(root, children, question)`——分层深度2聚合。
- `run_graph(agents, question, rounds)`——全对全辩论，限轮。
- 脚本异构旋钮：每个Agent有`error_bias`指示其系统错误。
- 测量框架各拓扑N=3、5、7运行报告(精度、总token、模拟墙钟)。

跑：

```
python3 code/main.py
```

预期输出：拓扑 × N → (精度、token、延迟)表。图在研究型任务N=3-5赢；星在快事实任务赢；图N=7显示协调税（延迟膨胀快于精度）。

## 使用

`outputs/skill-topology-picker.md`是技能读取任务描述推荐拓扑（星/链/树/图）、N（Agent数）、异构profile（基座模型）、轮限。

## 交付成果

任何ensemble：

- 从**自一致性N=5**用一强基座模型开始。便宜基线。
- 若精度重要升级**异构投票N=3**。测量delta。
- 仅当任务有结构（研究、多步）且限轮可行时升级**辩论拓扑**。
- 总记录少数簇。当少数持续正确，你有多样性信号。
- 基准墙钟和token与精度。"更好精度10倍成本"是商业决策。

## 练习题

1. 跑`code/main.py`。画图拓扑协调税曲线：精度vs N，token vs N。曲线在哪个N拐？
2. 实现A-HMAD：三个Agent刻意不同偏。同偏基线与A-HMAD在Lesson 14单一化攻击比如何？
3. 加"法官"角色到图拓扑不投票只评分最终共识。这改变涌现从众行为吗？
4. 读AgentVerse论文(ICLR 2024)。识别你的实现最显著展现哪种涌现行为。能通过提示改变激发相反行为？
5. 读MultiAgentBench(arXiv:2503.01935)Section 4(拓扑实验)。在你的框架上复现"图赢研究"结果于论文一个任务。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 自一致性 | "采样N次投票" | Wang 2022。单模型，N次温度>0采样，推理路径多数投票。 |
| 异构性 | "不同模型" | 不同基座模型或提示族ensemble。打破单一化。 |
| MAD | "多Agent辩论" | Agent跨轮交换批评通用术语。见Du 2023。 |
| A-HMAD | "对抗异构MAD" | MAD变体强调不同模型+对抗结构。 |
| 拓扑 | "谁对谁说话" | 星、链、树、图。决定信息流。 |
| 协调税 | "边际递减" | 图上~4 Agent以上，成本增长快于质量。 |
| 志愿行为 | "未提示帮助" | AgentVerse涌现模式：Agent主动提议做一步。 |
| 从众行为 | "压力下赞同" | AgentVerse涌现模式：Agent对齐批评者。 |
| Jury | "小专业面板" | Sibyl风格ensemble带角色(审问者、上下文、评分者)。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — Agent数和轮数各自独立重要
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 拓扑基准示图最适合研究、链适合管道
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD策略调研；发现MAD同预算常输自一致性
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — 志愿和从众涌现模式
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — 参考基准实现