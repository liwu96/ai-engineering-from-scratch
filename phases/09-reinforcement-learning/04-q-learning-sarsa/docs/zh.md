# 时序差分——Q-learning与SARSA

> 蒙特卡洛等待情节结束。TD每步后通过自举下一价值估计更新。Q-learning异策略且乐观;SARSA同策略且谨慎。两者一行代码。两者支撑本阶段每个深度强化学习方法。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程01(MDP)、阶段9课程02(动态规划)、阶段9课程03(蒙特卡洛)
**时间:** ~75分钟

## 问题背景

蒙特卡洛有效但有两个昂贵需求。需终止情节,且只在最终回报入后更新。如情节1000步,MC等1000步才更新任何东西。高方差、低偏、实践慢。

动态规划相反——零方差自举备份——但需已知模型。

时序差分(TD)学习折中。从单转移`(s, a, r, s')`,形一步目标`r + γ V(s')`并微调`V(s)`向它。无模型。无完整情节。RHS用近似`V`有偏,但方差比MC剧烈降低且第一步起在线更新。

这是现代强化学习——DQN、A2C、PPO、SAC——转折点。阶段9其余是在本课写的一步TD更新上叠函数近似和技巧层。

## 概念讲解

![Q-learning vs SARSA: 异策略max vs 同策略Q(s', a')](../assets/td.svg)

**V的TD(0)更新:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号量是TD误差`δ = r + γ V(s') - V(s)`。MC中`G_t - V(s_t)`在线类比。收敛需`α`满足Robbins-Monro(`Σ α = ∞`, `Σ α² < ∞`)和所有状态无限频繁访问。

**Q-learning。**控制异策略TD方法:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`假设从`s'`起将遵循*贪婪*策略,无论智能体实际采取什么动作。解耦使Q-learning学习`Q*`同时智能体ε-贪婪探索。Mnih等(2015)转此为Atari深度Q-learning(课程05)。

**SARSA。**同策略TD方法:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名是元组`(s, a, r, s', a')`。SARSA用智能体*实际*下一步采取动作`a'`,非贪婪`argmax`。收敛到任何ε-贪婪`π`运行下`Q^π`,极限`ε → 0`成`Q*`。

**悬崖行走差异。**经典悬崖行走任务(掉悬崖=奖励-100),Q-learning沿悬崖边学最优路径但探索时偶尔受罚。SARSA离悬崖一步学更安全路径因它把探索噪声纳入Q值。训练时,两者`ε → 0`达最优。实践重要:部署时探索实际发生,SARSA行为更保守。

**期望SARSA。**用`π`下期望值替换`Q(s', a')`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比SARSA低方差(无`a'`采样),相同同策略目标。常现代教材默认。

**n步TD和TD(λ)。**自举前等`n`步插值TD(0)和MC。`n=1`是TD,`n=∞`是MC。TD(λ)用几何权重`(1-λ)λ^{n-1}`平均所有`n`。大多深度强化学习用`n`在3到20间。

## 动手实践

### Step 1:ε-贪婪策略上SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。Q-learning*唯一*差别是目标行。

### Step 2:Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`解耦目标与行为。那一符号是同策略与异策略差异。

### Step 3:学习曲线

每100情节跟踪平均回报。简单确定GridWorld上Q-learning收敛更快;悬崖行走SARSA更保守。`code/main.py`中4×4 GridWorld上,两者`α=0.1, ε=0.1`~2000情节后近最优。

### Step 4:与DP真值比较

跑价值迭代(课程02)得`Q*`。查`max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康表格TD智能体10,000情节后4×4 GridWorld落`~0.5`内。

## 陷阱

- **初始Q值重要。**乐观初始化(负奖励任务`Q = 0`)鼓励探索。悲观初始化可永远困贪婪策略。
- **α调度。**常数`α`对非平稳问题好。衰减`α_n = 1/n`理论上收敛但实践太慢——钉`α`于`[0.05, 0.3]`并监控学习曲线。
- **ε调度。**高起(`ε=1.0`),衰减到`ε=0.05`。"GLIE"(极限无穷探索贪婪)是收敛条件。
- **Q-learning最大偏。**`Q`噪声时`max`算子向上偏。导致过估——Hasselt双重Q-learning(课程05DDQN用)用两Q表修复。
- **非终止情节。**TD可无终止学习,但需封步或正确处理封处自举。标准:视封非终止,继续自举。
- **状态哈希。**如状态是元组/张量,用可哈希键(元组,非列表;舍入浮点元组,非原始)。

## 实际应用

2026年TD格局:

| 任务 | 方法 | 原因 |
|------|------|------|
| 小表格环境 | Q-learning | 直接学最优策略。 |
| 同策略安全关键 | SARSA/期望SARSA | 探索时保守。 |
| 高维状态 | DQN(阶段9课程05) | 配回放和目标网神经网Q函数。 |
| 连续动作 | SAC/TD3(阶段9课程07) | Q网TD更新;策略网发动作。 |
| 大语言模型强化学习(奖励模型基) | PPO/GRPO(阶段9课程08、12) | GAE TD式优势actor-critic。 |
| 离线强化学习 | CQL/IQL(阶段9课程08) | 保守正则化Q-learning。 |

2026论文中"强化学习"90%是Q-learning或SARSA某细化。深入前手指理解表格更新。

## 产出成果

存`outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: 表格或小特征强化学习任务选Q-learning、SARSA、期望SARSA。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定表格或小特征环境,输出:

1. 算法。Q-learning/SARSA/期望SARSA/n步变体。一句理由关同策略vs异策略和方差。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0值(乐观vs零)和论证。
4. 收敛诊断。目标学习曲线,若DP可能`|Q - Q*|`检查。
5. 部署警示。推理时探索如何行为?需SARSA保守吗?

拒绝状态空间>10⁶上应用表格TD。拒绝无最大偏警示发货Q-learning智能体。标记全程ε=1.0训练智能体(无利用阶段)。
```

## 练习题

1. **简单。**4×4 GridWorld实现Q-learning和SARSA。绘2000情节学习曲线(每100情节平均回报)。谁收敛更快?
2. **中等。**构建悬崖行走环境(4×12,末行悬崖奖励-100重置开始)。比较Q-learning和SARSA终策略。截每走路径。哪个近悬崖?
3. **困难。**实现双重Q-learning。噪声奖励GridWorld(每步奖励加高斯噪声σ=5),展示Q-learning有意义过估`V*(0,0)`而双重Q-learning不。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| TD误差 | "更新信号" | `δ = r + γ V(s') - V(s)`,自举残差。 |
| TD(0) | "一步TD" | 仅用下一状态估计每转移后更新。 |
| Q-learning | "异策略强化学习入门" | 下状态动作`max`TD更新;无论行为策略学`Q*`。 |
| SARSA | "同策略Q-learning" | 用实际下一动作TD更新;当前ε-贪婪π学`Q^π`。 |
| 期望SARSA | "低方差SARSA" | π下期望替换采样`a'`。 |
| GLIE | "正确探索调度" | 极限无穷探索贪婪;Q-learning收敛需要。 |
| 自举 | "目标用当前估计" | 区别TD与MC。偏源但巨大方差减。 |
| 最大偏 | "Q-learning过估" | 噪声估计`max`向上偏;双重Q-learning修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698)——原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf)——TD(0)、SARSA、Q-learning、期望SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)——最大偏修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542)——期望SARSA动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems)——命名SARSA论文(当时"修改连接ist Q-learning")。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf)——TD(0)到TD(n)泛化,Q-learning到资格痕迹及PPO中GAE路径。