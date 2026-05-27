# 多智能体强化学习

> 单智能体强化学习假设环境是静态的。把两个学习型智能体放在同一个世界中,这个假设就失效了:每个智能体都是另一个智能体环境的一部分,而且两者都在变化。多智能体强化学习就是让学习在马尔可夫假设不再成立时收敛的一系列技巧。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程04(Q-learning)、阶段9课程06(REINFORCE)、阶段9课程07(Actor-Critic)
**时间:** ~45分钟

## 问题背景

一个机器人学习在房间中导航是单智能体强化学习问题。足球团队不是。AlphaStar对战StarCraft对手不是。竞价智能体的市场不是。两辆车协商四向停车不是。许多现实世界问题不是。

在每个多智能体场景中,从任意一个智能体的视角看,其他智能体*是*环境的一部分。当它们学习和改变行为时,环境变得非平稳。马尔可夫属性——"下一个状态只依赖当前状态和我的动作"——被违反,因为下一个状态还依赖*其他*智能体的选择,而它们的策略是移动目标。

这破坏了表格收敛证明(Q-learning的保证假设静态环境)。也破坏了朴素深度强化学习:智能体互相追逐循环,永不收敛到稳定策略。你需要多智能体特定技术:集中训练/分散执行、反事实基线、联赛训练、自我博弈。

2026应用:机器人群、交通路由、自动驾驶车队、市场模拟器、多智能体LLM系统(阶段16),以及任何有多个智能玩家的游戏。

## 概念讲解

![四种MARL regime:独立、集中critic、self-play、league](../assets/marl.svg)

**形式化:马尔可夫博弈。**MDP的推广:状态`S`,联合动作`a = (a_1, …, a_n)`,转移`P(s' | s, a)`,以及每个智能体的奖励`R_i(s, a, s')`。每个智能体`i`在自己的策略`π_i`下最大化自己的回报。如果奖励相同,是**完全合作**。如果是零和,是**对抗性**。如果混合,是**一般博弈**。

**核心挑战:**

- **非平稳性。**从智能体`i`视角看`P(s' | s, a_i)`依赖`π_{-i}`,后者在变化。
- **贡献分配。**共享奖励下,哪个智能体导致了它?
- **探索协调。**智能体必须探索互补策略,而非冗余探索同一状态。
- **可扩展性。**联合动作空间随`n`指数增长。
- **部分可观测性。**每个智能体只看到自己的观测;全局状态隐藏。

**四种主导范式:**

**1. 独立Q-learning / 独立PPO (IQL, IPPO)。**每个智能体学习自己的Q或策略,将其他智能体视为环境的一部分。简单,有时可行(尤其是经验回放作为平滑智能体建模技巧时)。理论收敛:无。实践中:松耦合任务还行,紧耦合任务糟糕。

**2. 集中训练,分散执行 (CTDE)。**最常见的现代范式。每个智能体有自己的*策略*`π_i`,条件于局部观测`o_i`——部署时标准分散执行。在*训练*期间,集中critic `Q(s, a_1, …, a_n)`条件于完整全局状态和联合动作。例子:
- **MADDPG**(Lowe等 2017):DDPG配每个智能体一个集中critic。
- **COMA**(Foerster等 2017):反事实基线——问"如果我采取动作`a'`而非当前动作,奖励会是什么?"——分离我的贡献。
- **MAPPO**/**IPPO**配共享critic(Yu等 2022):PPO配集中价值函数。2026年合作MARL的主导方法。
- **QMIX**(Rashid等 2018):价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`配单调混合。

**3. 自我博弈。**同一智能体的两个副本互相对抗。对手的策略*是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和博弈;训练信号对称。

**4. 联赛训练。**自我博弈向一般博弈/对抗环境的扩展:保持过去和当前策略的种群,从联赛中采样对手,训练对抗它们。添加exploiter(专攻击败当前最佳)和主exploiter(专攻击败exploiter)。AlphaStar(StarCraft II)。当游戏存在"石头剪刀布"策略循环时需要。

**通信。**允许智能体互相发送学习的消息`m_i`。合作场景有效。Foerster等(2016)证明了可微智能体间通信可以端到端训练。今天的LLM多智能体系统(阶段16)本质上是自然语言通信。

## 动手实践

本课使用6×6 GridWorld配两个合作智能体。它们从对角开始,必须到达共享目标。共享奖励:任一智能体还在移动时每步`-1`,两者都到达时`+10`。见`code/main.py`。

### Step 1: 多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是`|A|² = 16`。全局状态是两个位置。

### Step 2: 独立Q-learning

每个智能体运行自己的Q表,以联合状态为键。每步:两者选ε-贪婪动作,收集联合转移,各自用共享奖励更新自己的Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上可行,因为奖励密集且对齐。紧耦合任务失败(如一个智能体必须*等待*另一个)。

### Step 3: 集中Q配分解价值更新

用一个Q覆盖联合动作`Q(s, a_1, a_2)`。从共享奖励更新。执行时分散化通过边缘化:`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级联合动作空间换取*正确*的全局视角。

### Step 4: 简单自我博弈(对抗性2智能体)

同一智能体,两个角色。训练智能体A对抗智能体B;`K`轮后,将A的权重复制到B。对称训练,一致进展。AlphaZero配方缩影。

## 陷阱

- **非平稳回放。**独立智能体的经验回放比单智能体更糟,因为旧转移由现已过时的对手生成。修复:重标注或按近期加权。
- **贡献分配模糊。**长轮次后共享奖励;无法清晰说明哪个智能体贡献。修复:反事实基线(COMA),或每智能体奖励塑形。
- **策略漂移/追逐。**每个智能体的最佳响应随对方更新而变。修复:集中critic、慢学习率,或逐个冻结。
- **通过协调的奖励破解。**智能体发现设计者未预期的协调漏洞。竞价智能体收敛到出价零。修复:仔细奖励设计、行为约束。
- **探索冗余。**两个智能体探索相同状态-动作对。修复:每智能体熵奖励,或角色条件化。
- **联赛循环。**纯自我博弈可能陷入支配循环。修复:配多样对手的联赛训练。
- **样本爆炸。**`n`智能体 × 状态空间 × 联合动作。用函数近似;分解动作空间(每智能体一个策略输出头)。

## 实际应用

2026 MARL应用图谱:

| 领域 | 方法 | 备注 |
|------|------|------|
| 合作导航/操控 | MAPPO / QMIX | CTDE;共享critic + 分散actors。 |
| 双人游戏(棋类、Go、扑克) | Self-play配MCTS(AlphaZero) | 零和;对称训练。 |
| 复杂多人(Dota、StarCraft) | League训练 + 模仿预训练 | OpenAI Five, AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / PPO配attention | 部分观测;可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | `n` → ∞时均值场RL。 |
| LLM多智能体系统(阶段16) | 自然语言通信 + 角色条件化 | 智能体规划层的RL循环。 |

2026年,MARL最大增长领域是LLM基础:语言模型智能体群体协商、辩论、构建软件。RL表现为*轨迹级*输出的偏好优化,而非词元级(阶段16课程03)。

## 产出成果

保存为`outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: 为给定任务选择正确的多智能体RL范式(IPPO, CTDE, self-play, league)。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定`n`智能体的任务,输出:

1. 范式分类。合作/对抗/一般博弈。论证。
2. 算法。IPPO / MAPPO / QMIX / self-play / league。理由关联耦合紧度和奖励结构。
3. 信息访问。集中训练(什么全局信息给critic)?分散执行?
4. 贡献分配。反事实基线、价值分解,或奖励塑形。
5. 探索计划。每智能体熵、种群训练,或联赛。

拒绝紧耦合合作任务上用独立Q-learning。拒绝一般博弈配循环风险时推荐self-play。标记任何无固定对手评估的MARL流水线(cherry-picked self-play数字常见)。
```

## 练习题

1. **简单。**在2智能体合作GridWorld上训练独立Q-learning。多少轮次后平均回报>0?绘制联合学习曲线。
2. **中等。**添加"协调"任务:目标仅在两个智能体同一轮踏上时到达。独立Q仍收敛吗?什么破坏了?
3. **困难。**实现MAPPO风格训练的集中critic,比较协调任务上与独立PPO的收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Markov game | "多智能体MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`;每个智能体有自己的奖励。 |
| CTDE | "集中训练,分散执行" | 训练时联合critic;每个智能体的策略只用局部观测。 |
| IPPO | "独立PPO" | 每个智能体单独运行PPO。简单基线;常被低估。 |
| MAPPO | "多智能体PPO" | PPO配条件于全局状态的集中价值函数。 |
| QMIX | "单调价值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)`允许分散argmax。 |
| COMA | "反事实多智能体" | Advantage = 我的Q减去边缘化我动作的期望Q。 |
| Self-play | "智能体对抗过去自己" | 单智能体,两个角色;零和博弈标准。 |
| League play | "种群训练" | 缓存过去策略,从池中采样对手;处理策略循环。 |

## 延伸阅读

- [Lowe等(2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275)——配集中critic的CTDE。
- [Foerster等(2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926)——贡献分配的反事实基线。
- [Rashid等(2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485)——配单调性的价值分解。
- [Yu等(2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955)——PPO对MARL惊人有效。
- [Vinyals等(2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)——大规模联赛训练。
- [Silver等(2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)——零和博弈纯自我博弈。
- [Sutton & Barto(2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf)——教材多智能体场景和CTDE解决的非平稳问题短述。
- [Zhang, Yang & Başar(2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635)——覆盖合作、竞争、混合MARL配收敛结果的综述。