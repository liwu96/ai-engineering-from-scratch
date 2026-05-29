# 策略梯度 — 从零实现REINFORCE

> 停止估计价值函数。直接参数化策略，计算期望回报的梯度，沿梯度上升。Williams(1992)用一个定理写下了它。这就是PPO、GRPO以及所有大语言模型强化学习循环存在的原因。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3课程03(反向传播)、阶段9课程03(蒙特卡洛)、阶段9课程04(TD学习)
**时间:** ~75分钟

## 问题背景

Q-learning和DQN参数化的是*价值*函数，通过`argmax Q`选择动作。对离散动作和离散状态没问题。但当动作连续时就失效了(如何对10维力矩向量做`argmax`?)，或者当你需要随机策略时也不行(`argmax`本质上是确定的)。

策略梯度方法直接参数化*策略*。`π_θ(a | s)`是一个输出动作分布的神经网络。从中采样来行动。对期望回报关于`θ`求梯度，沿梯度上升。无需`argmax`，无需贝尔曼递推，只需对`J(θ) = E_{π_θ}[G]`做梯度上升。

REINFORCE定理(Williams 1992)告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个情节，计算回报，在每步乘以`∇ log π_θ(a | s)`，取平均，梯度上升，完成。

2026年每个大语言模型强化学习算法——PPO、DPO、GRPO——都是REINFORCE的改进。深入理解它是本阶段其余课程以及阶段10课程07(RLHF实现)和课程08(DPO)的前提。

## 概念讲解

![策略梯度：softmax策略，log-π梯度，回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。**对任意由`θ`参数化的策略`π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中`G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`是从步`t`起的折现回报，期望取遍从`π_θ`采样的完整轨迹`τ`。

**证明很短。**对期望中的`J(θ) = Σ_τ P(τ; θ) G(τ)`求微分。用`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`(对数导数技巧)。展开`log P(τ; θ) = Σ log π_θ(a_t | s_t) + 与θ无关的环境项`。环境项消失，两行代数给出定理。

**方差削减技巧。**朴素REINFORCE方差极高——回报有噪声，`∇ log π`有噪声，两者乘积噪声很大。两个标准修复：

1. **基线减法。**将`G_t`替换为`G_t - b(s_t)`，其中基线`b(s_t)`不依赖于`a_t`。无偏，因为`E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：评论家学到的`b(s_t) = V̂(s_t)` → actor-critic(课程07)。
2. **未来奖励。**将`Σ_t G_t · ∇ log π_θ(a_t | s_t)`替换为`Σ_t G_t^{从t} · ∇ log π_θ(a_t | s_t)`。对给定动作只有未来回报才重要——过去奖励只贡献零均值噪声。

��合后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的REINFORCE——A2C(课程07)和PPO(课程08)的直接祖先。

**Softmax策略参数化。**对离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中`f_θ`是任何输出每动作分数的神经网络。梯度有简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。**`π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)`有闭式形式。这就是阶段9课程07 SAC所需要的全部。

## 动手实践

### Step 1: softmax策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

表格环境用线性策略(每动作一个权重向量)。Atari上换成CNN，保留softmax头。

### Step 2: 采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: 记录对数概率的展开

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度`∇ log π(a|s) = e_a - π(·|s)`(动作`a`的独热编码减去概率)是softmax策略梯度的核心。务必深刻理解。

### Step 5: 基线

对最近情节的`G`取运行均值作为基线，足以完成4×4 GridWorld的方差削减；~500情节后收敛。将基线升级为学到的`V̂(s)`就得到actor-critic。

## 陷阱

- **梯度爆炸。**回报可能很大。在乘以`∇ log π`之前，始终将批次的`G`归一化到`~N(0, 1)`。
- **熵坍缩。**策略过早收敛到近确定性动作，停止探索，陷入局部最优。修复：在目标中加入熵奖励`β · H(π(·|s))`。
- **高方差。**朴素REINFORCE需要数千情节。评论家基线(课程07)或TRPO/PPO的信任区域(课程08)是标准修复。
- **样本低效。**同策略意味每次更新后丢弃所有转移。通过重要性采样的异策略校正可召回数据，代价是方差增大(PPO的比率是截断的IS权重)。
- **梯度非平稳。**100个情节前的同一梯度使用旧`π`。同策略方法为此每几次展开就更新。
- **信用分配。**不用未来奖励，过去奖励贡献噪声。始终使用未来奖励。

## 实际应用

2026年REINFORCE很少直接运行，但其梯度公式无处不在：

| 用例 | 派生方法 |
|------|----------|
| 连续控制 | 配高斯策略的PPO / SAC |
| 大语言模型RLHF | 配KL惩罚的PPO，在词元级策略上运行 |
| 大语言模型推理(DeepSeek) | GRPO——带组相对基线的REINFORCE，无评论家 |
| 多智能体 | 中心化评论家REINFORCE(MADDPG, COMA) |
| 离散动作机器人 | A2C, A3C, PPO |
| 仅偏好设置 | DPO——将REINFORCE改写为偏好似然损失，无需采样 |

当你在2026年训练脚本中看到`loss = -advantage * log_prob`，那就是带基线的REINFORCE。整篇论文(DPO, GRPO, RLOO)都是在这一行之上的方差削减技巧。

## 产出成果

存`outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: 为给定任务产生REINFORCE / actor-critic / PPO训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定环境(离散/连续动作、视界、奖励统计),输出:

1. 策略头。Softmax(离散)或高斯(连续),含参数量。
2. 基线。无(朴素)、运行均值、学到的`V̂(s)`或A2C评论家。
3. 方差控制。默认开启未来奖励、回报归一化、梯度裁剪值。
4. 熵奖励。系数β及衰减调度。
5. 批次大小。每次更新的情节数；同策略数据新鲜度约定。

拒绝视界>500步的无基线REINFORCE。拒绝连续动作控制使用softmax头。标记任何`β=0`且观测到策略熵<0.1的运行为熵坍缩。
```

## 练习题

1. **简单。**4×4 GridWorld上用线性softmax策略实现REINFORCE。不用基线训练1,000情节。绘学习曲线；测方差(回报标准差)。
2. **中等。**加运行均值基线。再次训练。比较与朴素运行的样本效率和方差。基线将收敛步数减少了多少?
3. **困难。**加熵奖励`β · H(π)`。遍历`β ∈ {0, 0.01, 0.1, 1.0}`。绘终态回报和策略熵。此任务上甜点在哪里?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；由对数导数技巧推导。 |
| REINFORCE | "原始PG算法" | Williams(1992)；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧 | "分数函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可处理。 |
| 基线 | "方差削减" | 任何从`G`中减去的`b(s)`；无偏因为`E[b · ∇ log π] = 0`。 |
| 未来奖励 | "只有未来回报才重要" | 用`G_t^{从t}`代替完整`G_0`；正确且方差更低。 |
| 熵奖励 | "鼓励探索" | `+β · H(π(·|s))`项防止策略坍缩。 |
| 同策略 | "从刚看到的数据训练" | 梯度期望关于当前策略——不能直接复用旧数据。 |
| 优势 | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线REINFORCE乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)——原始REINFORCE论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)——带函数近似的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf)——教材阐述。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html)——配PyTorch代码的清晰讲解。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf)——方差削减和自然梯度视角，连接REINFORCE与信任区域族(TRPO, PPO)。
