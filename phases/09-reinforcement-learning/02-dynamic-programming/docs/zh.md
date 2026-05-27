# 动态规划——策略迭代与价值迭代

> 动态规划是强化学习"作弊"版本。你已知转移和奖励函数;迭代贝尔曼方程直到`V`或`π`停止移动。它是每个基于采样方法试图接近的基准。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程01(MDP)
**时间:** ~75分钟

## 问题背景

你有一个已知模型的MDP:可对任意状态-动作对查询`P(s' | s, a)`和`R(s, a, s')`。库存管理员知道需求分布。棋盘游戏有确定性转移。GridWorld四行Python。你有*模型*。

无模型强化学习(Q-learning、PPO、REINFORCE)发明于无模型情况——只能从环境采样。但当有模型时,有更快更好的方法:动态规划。Bellman1957年设计。仍定义正确性:人们说"此MDP最优策略",指DP返回的策略。

2026年需要它们三个原因。第一,强化学习研究中每个表格环境(GridWorld、FrozenLake、CliffWalking)用DP求解产生金标准策略。第二,精确值让你*调试*采样方法:Q-learning对`V*(s_0)`估计与DP答案差30%,Q-learning有bug。第三,现代离线强化学习和规划方法(MCTS、AlphaZero搜索、阶段9课程10基于模型强化学习)都在学习或给定模型上迭代贝尔曼备份。

## 概念讲解

![策略迭代和价值迭代并排](../assets/dp.svg)

**两种算法,都是贝尔曼上固定点迭代。**

**策略迭代。**交替两步到策略停止变化。

1. *评估:*给定策略`π`,重复应用`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`计算`V^π`直到收敛。
2. *改进:*给定`V^π`,使`π`对`V^π`贪婪:`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛保证因 改进步要么保持`π`要么对某状态严格增加`V^π`, 确定性策略空间有限。通常即使大状态空间也~5–20次外迭代收敛。

**价值迭代。**把评估和改进坍成一扫。应用贝尔曼*最优性*方程:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复到`max_s |V_{new}(s) - V(s)| < ε`。末尾取贪婪动作提取策略。每次迭代严格更快——无内评估循环——但通常需更多迭代收敛。

**广义策略迭代(GPI)。**统一框架。价值函数和策略锁在双向改进循环;任何驱动两者到相互一致的方法(异步价值迭代、修改策略迭代、Q-learning、actor-critic、PPO)是GPI实例。

**为何`γ < 1`重要。**贝尔曼算子是sup范数`γ`-收缩:`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。收缩暗示唯一固定点和几何收敛。放弃`γ < 1`失保证——需有限视界或吸收终止状态。

## 动手实践

### Step 1:构建GridWorld MDP模型

用课程01相同4×4 GridWorld。加随机变体:概率`0.1`智能体滑到随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`返回`(s', r, p)`列表。这是整个模型。

### Step 2:策略评估

给定策略`π(s) = {action: prob}`,迭代贝尔曼方程到`V`停止移动:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3:策略改进

用对`V`贪婪策略替换`π`。如`π`未变,返回——已达最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4:组合

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意开始
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4典型收敛:4–6次外迭代。输出`V*(0,0) ≈ -6`和严格减少步数策略。

### Step 5:价值迭代(单循环版本)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同固定点,更少代码行。

## 陷阱

- **忘记处理终止。**对吸收状态应用贝尔曼仍拾取"最佳动作"什么都不改。用`if s == terminal: V[s] = 0`保护。
- **sup范数vs L2收敛。**用`max |V_new - V|`,非平均。理论保证在sup范数。
- **原地vs同步更新。**原地更新`V[s]`(Gauss-Seidel)比单独`V_new`字典(Jacobi)收敛更快。生产代码用原地。
- **策略平局。**如两动作Q值相等,`argmax`可能每迭代不同打破平局,导致"策略稳定"检查振荡。用稳定打破(固定顺序首动作)。
- **状态空间爆炸。**DP每扫`O(|S| · |A|)`。工作到~10⁷状态。超过需函数近似(阶段9课程05起)。

## 实际应用

2026年,DP是正确性基准和规划器内循环:

| 用例 | 方法 |
|------|------|
| 精确求解小表格MDP | 价值迭代(更简)或策略迭代(更少外步) |
| 验证Q-learning/PPO实现 | 玩具环境上与DP最优V*比较 |
| 基于模型强化学习(阶段9课程10) | 学习转移模型上贝尔曼备份 |
| AlphaZero/MuZero规划 | 蒙特卡洛树搜索=异步贝尔曼备份 |
| 离线强化学习(CQL、IQL) | 保守Q-迭代——OOD动作惩罚DP |

每次有人说"最优价值函数",指"DP固定点"。论文见`V*`或`Q*`,想这循环。

## 产出成果

存`outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小表格MDP。报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定已知模型MDP,输出:

1. 选择。策略迭代vs价值迭代。理由关|S|、|A|、γ。
2. 初始化。V_0、起始策略。收敛敏感度。
3. 停止。sup范数容忍ε。期望扫描数。
4. 验证。精确计算V*(s_0)。提取贪婪策略。
5. 用途。如何用此基准调试/评估基于采样方法。

拒绝状态空间>10⁷上运行DP。拒绝无sup范数检查声称收敛。标记无限视界任务上γ≥1为保证违规。
```

## 练习题

1. **简单。**4×4 GridWorld上`γ ∈ {0.9, 0.99}`跑价值迭代。多少扫到`max |ΔV| < 1e-6`?打印`V*`为4×4网格。
2. **中等。***随机*GridWorld(滑概率`0.1`)上比较策略迭代vs价值迭代。计数:扫描、墙钟时间、终`V*(0,0)`。哪个迭代收敛更快?墙钟?
3. **困难。**构建修改策略迭代:评估步只跑`k`扫而非到收敛。`k ∈ {1, 2, 5, 10, 50}`绘`V*(0,0)`误差vs `k`。曲线告诉你评估/改进权衡什么?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 策略迭代 | "DP算法" | 交替评估(`V^π`)和改进(对`V^π`贪婪`π`)到策略停止变化。 |
| 价值迭代 | "更快DP" | 一扫应用贝尔曼最优性备份;几何收敛到`V*`。 |
| 贝尔曼算子 | "递归" | `(T V)(s) = max_a Σ P (r + γ V(s'))`;sup范数γ-收缩。 |
| 收缩 | "为何DP收敛" | 任何算子`T`满足`||T x - T y|| ≤ γ ||x - y||`有唯一固定点。 |
| GPI | "一切都是DP" | 广义策略迭代:任何驱动`V`和`π`到相互一致的方法。 |
| 同步更新 | "Jacobi式" | 整扫用旧`V`;干净可分析但更慢。 |
| 原地更新 | "Gauss-Seidel式" | 更新时用`V`;实践收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf)——策略迭代和价值迭代规范呈现。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html)——收缩映射论证严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)——修改策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/)——原始策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html)——DP到近似-DP/深度强化学习桥梁,每后续课程使用。