# MARL——MADDPG、QMIX、MAPPO

> 多Agent协调强化学习遗产，仍指导2026 LLM-Agent系统。**MADDPG**(Lowe et al., NeurIPS 2017, arXiv:1706.02275)引入集中训练分散执行(CTDE)：每个critic训练时见所有Agent状态和行动；测试时只本地actor跑。适合作、竞争、混合设置。**QMIX**(Rashid et al., ICML 2018, arXiv:1803.11485)是单调混合网络值分解；每Agent Q组合联合Q使`argmax`干净分布——StarCraft多Agent挑战(SMAC)主导。**MAPPO**(Yu et al., NeurIPS 2022, arXiv:2103.01955)是集中值函数PPO；"惊人有效"粒子世界、SMAC、Google Research Football、Hanabi最小调。这些支撑必须分散行动Agent团队训练策略。MAPPO是**2026默认合作MARL基线**。本lesson从小网格世界玩具建每个，在肌肉记忆落地三想法，后触LLM-Agent训练。

**类型:** 学习
**语言:** Python(stdlib、小NumPy-free实现)
**前置要求:** 阶段09(强化学习)、阶段16课程09(Parallel Swarm Networks)
**时间:** ~90分钟

## 问题背景

LLM-Agent系统日益训练inter-Agent协调策略：何时退让、何时行动、调哪个peer。告诉你如何训练此策略文献是多Agent强化学习(MARL)，先于LLM波并有小集主导算法。

无模式词汇读MARL论文痛苦。集中训练分散执行(CTDE)、值分解、集中critic非buzzword——它们是特定问题特定答案：

- 独立RL(每Agent单独学)从每Agent视角非平稳。坏。
- 集中RL(一个Agent控所有)不扩展且违反执行约束。
- CTDE得两者最优：全局信息训练、本地策略部署。

## 概念讲解

### 论文用三环境

- **粒子世界(多Agent粒子env)。**简单2D物理合/竞任务。MADDPG原始测试床。
- **StarCraft多Agent挑战(SMAC)。**合作微管理、部分观察。QMIX测试床。离散行动、连续状态。
- **Google Research Football、Hanabi、MPE。**MAPPO基线。

不同env有不同行动/观察类型。算法据此选。

### MADDPG(2017)——CTDE模式

每Agent `i`有actor `mu_i(o_i)`映射自己观察到行动。每Agent也有critic `Q_i(x, a_1, ..., a_n)`训练时见所有观察和所有行动。Actor通过critic评估策略梯度更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为何CTDE：训练时，我们知道每个人行动；用此降每critic方差。部署时，每Agent只见`o_i`并调`mu_i(o_i)`。

失败模式：critic随N Agent增(输入含所有行动)。不扩展过~10 Agent无近似。

### QMIX(2018)——值分解

仅合作。全局奖励是每Agent Q值单调函数之和：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证`argmax_a Q_tot`可由每Agent独立选`argmax_{a_i} Q_i`计算。这正是你需的**分散执行属性**。训练时，混合网络从每Agent Q产`Q_tot`。

为何QMIX在SMAC赢：合作StarCraft微管理有同质Agent、本地obs、全局奖励——值分解完美匹配。

失败模式：单调约束受限；某些任务奖励结构非单调可分解(一个Agent牺牲团队)。扩展(QTRAN、QPLEX)放松此。

### MAPPO(2022)——被忽视默认

多Agent PPO：PPO加集中值函数。每Agent有自己的策略；所有Agent共享(或每Agent有)见全状态的值函数。Yu et al. 2022五基准测MAPPO对MADDPG、QMIX及其扩展发现：

- MAPPO匹或赢粒子世界、SMAC、Google Research Football、Hanabi、MPE上离策略MARL方法。
- 最小超参数调需。
- 训练稳定；跨seed可复现。

社区低估on-policy MARL直到此论文。2026，MAPPO是合作MARL默认基线；任何新方法必须赢它。

### LLM-Agent工程师为何应关心

三直接用：

1. **路由训练。**元Agent选哪个子Agent处理任务。这是N分散子Agent和一集中路由MARL问题。MAPPO适合。
2. **角色涌现。**生成式Agent模拟，训练Agent随时间采用互补角色是伪装MARL问题。QMIX风格值分解构造强制互补。
3. **多Agent工具使用。**当Agent共享工具和竞争预算，CTDE训练产生尊重资源约束可部署本地策略。

实践警告：2026，多数生产LLM-Agent系统提示其策略而非训练。MARL进当你有 大量交互数据、 清晰奖励信号、 和愿意投资训练基础设施。

### CTDE作设计模式超RL

即使无训练，CTDE是有用架构模式：

- *设计*时，假设全队可见。
- *运行*时，强制分散执行：每Agent只见`o_i`。

模式强制你保持每Agent状态显式并预先思考部分可观察性。许多生产多Agent系统静默假设共享状态无处不在——CTDE纪律阻止。

### 非平稳问题

当多Agent同时学习，每Agent环境(含其他策略)非平稳。经典单Agent RL证明破。本lesson MARL算法都解决：

- MADDPG：全局critic见所有行动，所以值估计平稳。
- QMIX：值分解移学习到联合Q空间最优性良定义。
- MAPPO：集中值函数平抑他人策略变方差。

LLM-Agent系统，非平稳显现"我Agent上月工作，现在上游那个Agent改，我的失灵"。MARL CTDE训练是原则修复；提示级修复更快但不持久。

### 本lesson不覆盖什么

训练实际网络是阶段09话题。本lesson建脚本策略版演示CTDE、值分解、集中值模式无梯度更新。目标是内化模式后拾完整MARL库(PyMARl、MARLlib、RLlib multi-agent)。

## 构建

`code/main.py`实现三模式演示，都在微型2 Agent合作网格世界：

- 环境：4x4网格2 Agent、一奖励颗粒。奖励=1若任Agent达颗粒；任务完成。
- `IndependentAgents`——每Agent视其他为环境。基线。
- `MADDPGStyle`——集中critic计算联合值；actor策略从中更新。脚本策略改进。
- `QMIXStyle`——单调混合器值分解。
- `MAPPOStyle`——集中值函数；策略对共享基线更新。

四种跑相同episode报告平均步达目标。CTDE变体收敛比独立基线更短路径。

跑：

```
python3 code/main.py
```

预期输出：独立Agent平均~6步；CTDE变体向~3.5步收敛(4x4网格最优是3)。模式差异在脚本策略也显现。

## 使用

`outputs/skill-marl-picker.md`是技能为给定多Agent任务选MARL算法：合作vs竞争、同质vs异质、行动空间类型、规模、奖励信号。

## 交付成果

MARL生产稀有。当你用时：

- **从MAPPO开始。**2022论文建此基线；先复现省周追更花哨方法。
- **记录每Agent观察和行动流。**无每Agent轨迹调试MARL无望。
- **分离训练代码执行代码。**CTDE是纪律；让执行路径真只见`o_i`。
- **奖励塑警告。**MARL极度敏感奖励设计。塑中一个协调bug Agent学会利用。跑对抗测试。
- **对LLM Agent**，先考虑提示级策略。仅当交互数据+奖励信号+基础设施都在时投资MARL训练。

## 练习题

1. 跑`code/main.py`。测量独立vs MAPPO风格Agent步达目标差距。差距在6x6网格增还是缩？
2. 实现竞争变体：两Agent、一颗粒、只有第一个达者得奖励。哪个模式干净处理竞争？历史MADDPG。
3. 读MADDPG(arXiv:1706.02275)Section 3。用你自己词符号伪代码实现确切critic更新规则。
4. 读MAPPO(arXiv:2103.01955)。作者为何论证集中值+PPO赢其基准离策略MARL？列三最强声称。
5. 应用CTDE作设计模式于假设LLM-Agent系统(如研究Agent+总结者+编码者)。设计时有但运行时无的联合信息是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MARL | "多Agent RL" | 多Agent系统强化学习。 |
| CTDE | "集中训练分散执行" | 全局信息训练；本地策略部署。 |
| MADDPG | "多Agent DDPG" | CTDE每Agent critic见所有观察+行动。 |
| QMIX | "值分解" | 每Agent Q单调混合。合作。 |
| MAPPO | "多Agent PPO" | 集中值函数PPO。2026默认基线。 |
| 值分解 | "个体Q之和" | 联合Q表为每Agent Q单调函数。 |
| 非平稳 | "移动目标" | 每Agent环境随其他学习变。核心MARL问题。 |
| On-policy / off-policy | "从当前/回放学习" | PPO是on-policy(MAPPO)；DDPG和Q-learning是off-policy。 |
| SMAC | "StarCraft多Agent挑战" | 合作微管理基准；QMIX主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — MAPPO结果可读框架
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft多Agent挑战