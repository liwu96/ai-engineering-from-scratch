# 蒙特卡洛方法——从完整情节学习

> 动态规划需要模型。蒙特卡洛只需要情节。运行策略,观察回报,平均它们。强化学习最简单想法——解锁所有后续。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程01(MDP)、阶段9课程02(动态规划)
**时间:** ~75分钟

## 问题背景

动态规划优雅,但假设可对每状态动作查询`P(s' | s, a)`。真实世界几乎不这样。机器人无法解析计算关节扭矩后相机像素分布。定价算法无法积分每可能客户反应。大语言模型无法枚举词元后所有可能延续。

需要仅能从环境*采样*的方法。运行策略。得轨迹`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它估计价值。这是蒙特卡洛。

DP到MC转换哲学重要:从*已知模型+精确备份*到*采样展开+平均回报*。方差跳,适用性爆。本课后每个强化学习算法——TD、Q-learning、REINFORCE、PPO、GRPO——核心是蒙特卡洛估计器,有时顶上加自举。

## 概念讲解

![蒙特卡洛:展开、计算回报、平均;首次访问vs每次访问](../assets/monte-carlo.svg)

**核心想法一行:**`V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`,其中`G^{(i)}(s)`是策略`π`下访问`s`后观察回报。

**首次访问vs每次访问MC。**给定多次访问状态`s`的情节,首次访问MC只从首次访问计数回报;每次访问MC计所有访问。极限两者无偏。首次访问更易分析(iid样本)。每次访问每情节用更多数据,实践通常更快收敛。

**增量平均。**不存所有回报,更新运行平均:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重组:`V_new = V_old + α · (target - V_old)`配`α = 1/n`。换`1/n`为常数步长`α ∈ (0, 1)`得跟踪`π`变化非平稳MC估计器。这移动是从MC到TD到每个现代强化学习算法的整个跳跃。

**探索现是问题。**DP枚举触及每状态。MC只看策略访问状态。如`π`确定,整个状态空间区域永不采样,价值估计永为零。三修复,历史顺序:

1. **探索起始。**每情节从随机对某状态开始。保证覆盖;实践不现实(不能"重置"机器人到任意状态)。
2. **ε-贪婪。**对当前Q贪婪行动,但概率`ε`选随机动作。所有状态-动作对渐近采样。
3. **异策略MC。**行为策略`μ`下收集数据,重要性采样学习目标策略`π`。高方差,但到DQN等回放缓冲方法的桥梁。

**蒙特卡洛控制。**评估→改进→评估,恰如策略迭代,但评估基于采样:

1. 运行`π`,得情节。
2. 从观察回报更新`Q(s, a)`。
3. 使`π`对`Q`ε-贪婪。
4. 重复。

温和条件下概率1收敛到`Q*`和`π*`(每对无限频繁访问,`α`满足Robbins-Monro)。

## 动手实践

### Step 1:展开→对某状态列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

无模型,只有`env.reset()`和`env.step(s, a)`。相同接口如gym环境但简化。

### Step 2:计算回报(反向扫描)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一遍,`O(T)`。反向递推`G_t = r_{t+1} + γ G_{t+1}`避免重求和。

### Step 3:首次访问MC评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行做工作:首次访问标记状态已见、增量计数、更新运行平均。

### Step 4:ε-贪婪MC控制(同策略)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5:与DP金标准比较

MC对`V^π`估计应情节数→∞时与课程02DP结果一致。实践:4×4 GridWorld上50,000情节达DP答案`~0.1`内。

## 陷阱

- **无限情节。**MC需情节*终止*。如策略可永远循环,封`max_steps`并视封为隐失败。随机策略GridWorld常超时——正常,确保正确计数。
- **方差。**MC用完整回报。长情节方差巨大——末尾一个不幸奖励等量移`V(s_0)`。TD方法(课程04)用自举削减。
- **状态覆盖。**新Q上贪婪MC平局只试一个动作。*必须*探索(ε-贪婪、探索起始、UCB)。
- **非平稳策略。**如`π`变(MC控制),旧回报来自不同策略。常数αMC处理;样本平均MC不。
- **异策略重要性采样。**权重`π(a|s)/μ(a|s)`跨轨迹乘。方差随视界爆。用每决策加权IS封或切换TD。

## 实际应用

2026年蒙特卡洛方法角色:

| 用例 |为何MC |
|------|------|
| 短视界游戏( Blackjack、 poker) | 情节自然终止;回报干净。 |
| 记录策略离线评估 | 存轨迹上平均折现回报。 |
| 蒙特卡洛树搜索(AlphaZero) | 树叶MC展开引导选择。 |
| 大语言模型强化学习评估 | 给定策略采样完成上计算平均奖励。 |
| PPO基线估计 | 优势目标`A_t = G_t - V(s_t)`用MC `G_t`。 |
| 教强化学习 | 实际工作最简算法——剥离自举看核心。 |

现代深度强化学习算法(PPO、SAC)通过`n`步回报或GAE插值纯MC(完整回报)和纯TD(一步自举)。两端点都是同估计器实例。

## 产出成果

存`outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛展开评估策略,产收敛报告,若有DP比较。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定环境(情节式,配reset+step API)和策略,输出:

1. 方法。首次访问vs每次访问MC。理由。
2. 情节预算。目标数、方差诊断、期望标准误。
3. 探索计划。ε调度(若需)或探索起始。
4. 金标准比较。表格则DP最优V*;否则Q-learning/PPO基线界。
5. 终止检查。最大步封、超时、非终止轨迹处理。

拒绝无有限视界封非情节任务上运行MC。拒绝表格任务每状态少于100情节报告V^π估计。标记零方差动作策略为探索风险。
```

## 练习题

1. **简单。**4×4 GridWorld上均匀随机策略实现首次访问MC评估。跑10,000情节。绘`V(0,0)`vs情节数,对照DP答案。
2. **中等。**`ε ∈ {0.01, 0.1, 0.3}`实现ε-贪婪MC控制。20,000情节后比较平均回报。曲线什么样?偏差-方差权衡在哪?
3. **困难。**重要性采样实现*异策略*MC:均匀随机策略`μ`下收集数据,确定性最优策略`π`估计`V^π`。比较朴素ISvs每决策ISvs加权IS。哪个方差最低?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 蒙特卡洛 | "随机采样" | 分布上iid样本平均估计期望。 |
| 回报`G_t` | "未来奖励" | 步`t`到情节末折现奖励和:`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问MC | "每状态计一次" | 情节中仅首次访问贡献价值估计。 |
| 每次访问MC | "用所有访问" | 每次访问贡献;略偏但更样本效率。 |
| ε-贪婪 | "探索噪声" | 概率`1-ε`选贪婪动作;概率`ε`选随机动作。 |
| 重要性采样 | "从错误分布采样校正" | `π(a|s)/μ(a|s)`产品重权回报,从`μ`数据估计`V^π`。 |
| 同策略 | "从自己数据学习" | 目标策略=行为策略。朴素MC、PPO、SARSA。 |
| 异策略 | "从别人数据学习" | 目标策略≠行为策略。重要性采样MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf)——规范处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726)——首次访问vs每次访问分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf)——异策略MC和方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362)——现代低方差IS估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343)——MC/TD自玩收敛到超人玩的首次大规模实证演示;本阶段后半每课程概念先驱。