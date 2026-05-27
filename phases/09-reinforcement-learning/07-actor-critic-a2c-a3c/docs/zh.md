# Actor-Critic——A2C与A3C

> REINFORCE噪声大。加一个学习`V̂(s)`的critic，从回报减去，得相同期望但更低方差的优势。这就是actor-critic。A2C同步运行；A3C跨线程运行。两者是每个现代深度强化学习方法的心智模型。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程04(TD学习)、阶段9课程06(REINFORCE)
**时间:** ~75分钟

## 问题背景

朴素REINFORCE有效但方差极差。蒙特卡洛回报`G_t`情节间可摆动10倍。将该噪声乘`∇ log π`再平均产梯度估计器需数千情节才能将策略移同距离DQN更新少得多能移。

方差来自用原始回报。如减基线`b(s_t)`——任何状态函数，包括学习价值——期望不变方差降。最佳可行基线是`V̂(s_t)`。现乘`∇ log π`量是*优势*：

`A(s, a) = G - V̂(s)`

动作产高于平均回报则好；低于则坏。配学习critic的REINFORCE是*actor-critic*。Critic给actor低方差教师。这是2015后每个深度策略方法(A2C、A3C、PPO、SAC、IMPALA)。

## 概念讲解

![Actor-critic:策略网加价值网,TD残差作优势](../assets/actor-critic.svg)

**两网络,一共享损失:**

- **Actor** `π_θ(a | s)`:策略。采样行动。策略梯度训练。
- **Critic** `V_φ(s)`:估计状态期望回报。最小化`(V_φ(s) - target)²`训练。

**优势。**两标准形式:

- *MC优势:*`A_t = G_t - V_φ(s_t)`。无偏,较高方差。
- *TD优势:*`A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏(用`V_φ`),远低方差。也称*TD残差*`δ_t`。

**n步优势。**插值两者:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`纯TD。`n = ∞`MC。大多实现Atari用`n = 5`,MuJoCo上PPO用`n = 2048`。

**广义优势估计(GAE)。**Schulman等(2016)提所有n步优势指数加权平均:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

配`λ ∈ [0, 1]`。`λ = 0`TD(低方差,高偏)。`λ = 1`MC(高方差,无偏)。`λ = 0.95`是2026默认——调至偏/方差旋钮达所愿。

**A2C:同步优势actor-critic。**跨`N`并行环境收集`T`步。每步算优势。合批次更新actor和critic。重复。A3C更简、更可扩同胞。

**A3C:异步优势actor-critic。**Mnih等(2016)。产`N`工作线程,每跑环境。每工作线程自己展开算梯度,异步应用到共享参数服务器。无需回放缓冲——工作线程跑不同轨迹解相关。A3C证明可CPU规模训。2026,基于GPU的A2C(批次并行环境)主导因GPU要大批次。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项:策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`, `c_e ~ 0.01`是典型起点。

## 动手实践

### Step 1:Critic

线性critic `V_φ(s) = w · features(s)`配MSE更新:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

表格环境critic数百情节收敛。Atari,替换线性critic为共享CNN骨干+价值头。

### Step 2:n步优势

给长度`T`展开和自举终`V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`是critic目标。`advantages`乘`∇ log π`。

### Step 3:组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

同策略,一展开一更新,actor和critic分离学习率。

### Step 4:并行化(A3C vs A2C)

- **A3C:**产`N`线程。每跑自己环境和自己前向传。周期推梯度更新到共享主。主无锁——竞争可,仅加噪声。
- **A2C:**单进程跑`N`环境实例,叠观察成`[N, obs_dim]`批次,批前向传,批反向传。更高GPU利用,确定,更易推理。2026默认。

玩具代码单线程清晰;重写批次A2C是三行numpy。

## 陷阱

- **Critic偏先actor梯度。**如critic随机,基线无信息,训于纯噪声。先暖critic数百步才开策略梯度,或用慢actor学习率。
- **优势归一化。**归一化优势每批次零均值/单位标准差。大幅稳定训练近零成本。
- **共享骨干。**图像输入actor和critic用共享特征提取器。分离头。共享特征两损失免费搭。
- **同策略合约。**A2C数据仅一更新。更则梯度有偏(重要性采样校正PPO加)。
- **熵坍塌。**无`c_e > 0`,策略数百更新近确定停止探索。
- **奖励尺度。**优势幅度依赖奖励尺度。归一化奖励(如运行标准差除)跨任务一致梯度幅度。

## 实际应用

A2C/A3C 2026少最终选择但是后一切精化架构:

| 方法 | 与A2C关系 |
|------|----------|
| PPO | A2C +裁剪重要性比率配多轮更新 |
| IMPALA | A3C +V-trace异策略校正 |
| SAC(阶段9课程07) | 异策略A2C配软价值critic(下课) |
| GRPO(阶段9课程12) | A2C无critic——组相对优势 |
| DPO | A2C坍成偏好排名损失,无采样 |
| AlphaStar / OpenAI Five | A2C配联赛训+模仿预训 |

如见2026论文"优势",思actor-critic。

## 产出成果

存`outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: 给定环境产A2C / A3C / GAE配置,配优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算,输出:

1. 并行化。A2C(GPU批次)vs A3C(CPU异步)和工作线程数。
2. 展开长度T。每环境每更新步数。
3. 优势估计器。n步或GAE(λ);指定λ。
4. 损失权重。`c_v`(价值), `c_e`(熵),梯度裁剪。
5. 学习率。Actor和critic(若用分离)。

拒视界>1000环境单工作线程A2C(太同策略,太慢)。拒无优势归一化发货。标记`c_e = 0`且观察熵<0.1运行为熵坍塌。
```

## 练习题

1. **简单。**4×4 GridWorld上MC优势(`G_t - V(s_t)`训actor-critic。比阶段9课程06REINFORCE配运行均值基线样本效率。
2. **中等。**切到TD残差优势(`r + γ V(s') - V(s)`)。测优势批次方差。降多少?
3. **困难。**实现GAE(λ)。扫`λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘终回报vs样本效率。此任务偏/方差甜点在哪?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Actor | "策略网" | `π_θ(a|s)`,策略梯度更新。 |
| Critic | "价值网" | `V_φ(s)`,MSE回归回报/TD目标更新。 |
| 优势 | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)`或其估计器。`∇ log π`乘数。 |
| TD残差 | "δ" | `δ_t = r + γ V(s') - V(s)`;一步优势估计。 |
| GAE | "插值旋钮" | n步优势指数加权总和,参数化`λ`。 |
| A2C | "同步actor-critic" | 跨环境批次;一展开一梯度步。 |
| A3C | "异步actor-critic" | 工作线程推梯度到共享参数服务器。原始论文;2026较少。 |
| Bootstrap | "视界处用V" | 截展开,加`γ^n V(s_{t+n})`闭合总和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)——A3C,原始异步actor-critic论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)——GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf)——基础;配Ch. 9函数近似读critic神经网络时。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561)——可扩分布actor-critic配V-trace异策略校正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/)——生产A2C/PPO实现值得读。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms)——双时间尺度actor-critic分解基础收敛结果。