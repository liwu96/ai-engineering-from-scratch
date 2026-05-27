# 近端策略优化 (PPO)

> A2C每次更新后丢弃每轮轨迹。PPO将策略梯度包装在裁剪重要性比率中,使你能在同一数据上做10+轮而不让策略爆炸。Schulman等(2017)。仍是2026年默认策略梯度算法。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程06(REINFORCE)、阶段9课程07(Actor-Critic)
**时间:** ~75分钟

## 问题背景

A2C(课程07)是同策略:梯度`E_{π_θ}[A · ∇ log π_θ]`要求从*当前*`π_θ`采样数据。一次更新后`π_θ`改变;你用的数据现是异策略。重用它你的梯度有偏。

轨迹昂贵。在Atari上,8环境×128步一轮轨迹 = 1024转移和环境时间十几秒。一个梯度步后丢弃浪费。

信任域策略优化(TRPO, Schulman 2015)是首个修复:约束每次更新使新旧策略KL散度低于`δ`。理论干净,但每次更新需共轭梯度求解。2026年无人运行TRPO。

PPO(Schulman等2017)用简单裁剪目标替换硬信任域约束。额外一行代码。每轮十轮。无共轭梯度。足够好的理论保证。九年后仍是MuJoCo到RLHF一切默认策略梯度算法。

## 概念讲解

![PPO裁剪代理目标:比率裁剪在1 ± ε](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略vs收集数据策略的似然比率。`r_t = 1`意味无变化。`r_t = 2`意味新策略比旧策略两倍可能采取`a_t`。

**裁剪代理。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项:

- 如果优势`A_t > 0`且比率试图增长超过`1 + ε`,裁剪压平梯度——不要把好动作推到超过旧概率`+ε`。
- 如果优势`A_t < 0`且比率试图增长超过`1 - ε`(意味我们会使坏动作比其裁剪减少更可能),裁剪限制梯度——不要把坏动作推到低于`-ε`。

`min`处理另一方向:如果比率已移动到*有益*方向,你仍获得梯度(不会伤害你的那侧无裁剪)。

典型`ε = 0.2`。绘制目标为`r_t`函数:"好侧"平顶"坏侧"平底的分段线性函数。

**完整PPO损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与A2C相同actor-critic结构。三个系数,通常`c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`。

**训练循环。**

1. 在`N`并行环境各`T`步收集`N × T`转移。
2. 计算优势(GAE),冻结为常数。
3. 将`π_{θ_old}`冻结为当前`π_θ`快照。
4. 对`K`轮,对每个`(s, a, A, V_target, log π_old(a|s))`小批次:
   - 计算`r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用`L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃轨迹。返回步骤1。

`K = 10`和64小批次是标准超参数集。PPO鲁棒:±50%内确切数字很少重要。

**KL惩罚变体。**原文提出配自适应KL惩罚的替代:`L = L^{PG} - β · KL(π_θ || π_old)`配基于观测KL调整的`β`。裁剪版主导;KL变体在RLHF存活(那里KL到参考策略是你要的独立约束)。

## 动手实践

### Step 1: 轨迹时捕获`log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在轨迹时取一次。更新轮间不变。

### Step 2: 计算GAE优势(课程07)

与A2C相同。批次间归一化。

### Step 3: 裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播-surrogate,加价值损失,减熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪→零梯度"模式是PPO核心。如果新策略在有益方向漂移太远,更新停止。

### Step 4: 价值和熵

向critic目标加标准MSE和actor熵奖励,与A2C相同。

### Step 5: 诊断

每次更新看三件事:

- **平均KL** `E[log π_old - log π_θ]`。应在`[0, 0.02]`。如果突破`0.1`,降低`K_EPOCHS`或`LR`。
- **裁剪比例**——比率在`[1-ε, 1+ε]`外的样本比例。应`~0.1-0.3`。如果`~0`,裁剪从不触发→提高`LR`或`K_EPOCHS`。如果`~0.5+`,你过拟合轨迹→降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic质量指标。应随critic学习爬向1。

## 陷阱

- **裁剪系数误调。**`ε = 0.2`是事实标准。到`0.1`使更新太胆怯;`0.3+`邀请不稳定。
- **太多轮。**`K > 20`常规 destabilize因为策略漂离`π_old`远。限制轮数,尤其大网络。
- **无奖励归一化。**大奖励尺度侵蚀裁剪范围。计算优势前归一化奖励(运行std)。
- **忘记优势归一化。**每批次零均值/单位std归一化是标准。跳过破坏大多数基准PPO。
- **学习率未衰减。**PPO受益于线性LR衰减到零。常数LR常更差。
- **重要性比率数学错误。**总是`exp(log_new - log_old)`数值稳定,而非`new / old`。
- **错误梯度符号。**最大化代理 = *最小化*`-L^{CLIP}`。翻转符号是最常见PPO bug。

## 实际应用

PPO是2026年惊人数量领域默认RL算法:

| 用例 | PPO变体 |
|------|---------|
| MuJoCo / 机器人控制 | PPO配高斯策略、GAE(0.95) |
| Atari / 离散游戏 | PPO配分类策略、滚动128步轨迹 |
| LLM RLHF | PPO配KL惩罚到参考模型、响应末RM奖励 |
| 大规模游戏智能体 | IMPALA + PPO(AlphaStar, OpenAI Five) |
| 推理LLM | GRPO(课程12)——无critic PPO变体 |
| 偏好专属数据 | DPO——PPO+KL闭合形式解、无在线采样 |

PPO*损失形状*——裁剪代理 + 价值 + 熵——是DPO、GRPO和几乎每个RLHF流水线脚手架。

## 产出成果

保存为`outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: 为给定环境产生PPO训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定环境和训练预算,输出:

1. 轨迹大小。`N`环境 × `T`步。
2. 更新计划。`K`轮、小批次大小、LR计划。
3. 代理参数。`ε`(裁剪)、`c_v`、`c_e`、优势归一化开。
4. 优势。GAE(`λ`)配显式`γ`和`λ`。
5. 诊断计划。KL、裁剪比例、解释方差阈值配警报。

拒绝`K > 30`或`ε > 0.3`(不安全信任域)。拒绝无优势归一化或KL/裁剪监控任何PPO运行。标记裁剪比例持续高于0.4为漂移。
```

## 练习题

1. **简单。**在4×4 GridWorld上用`ε=0.2, K=4`运行PPO。在匹配环境步比较样本效率与A2C(每轮一轮)。
2. **中等。**扫描`K ∈ {1, 4, 10, 30}`。绘制回报vs环境步并追踪每次更新平均KL。这个任务上什么`K`时KL爆炸?
3. **困难。**用自适应KL惩罚替换裁剪代理(如果`KL > 2·target`则`β`加倍,如果`KL < target/2`则减半)。比较最终回报、稳定性和无裁剪性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 重要性比率 | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`;偏离收集数据策略。 |
| 裁剪代理 | "PPO主技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`;有益侧裁剪外平梯度。 |
| 信任域 | "TRPO / PPO意图" | 限制每次更新KL以保证单调改进。 |
| KL惩罚 | "软信任域" | 替代PPO:`L - β · KL(π_θ || π_old)`。自适应`β`。 |
| 裁剪比例 | "裁剪触发频率" | 诊断——应0.1-0.3;外意味误调。 |
| 多轮训练 | "数据重用" | 每轮K轮;方差成本换取样本效率。 |
| 近同策略 | "大致同策略" | PPO名义同策略但K>1轮安全用略异策略数据。 |
| PPO-KL | "另一个PPO" | KL惩罚变体;在KL到参考已是约束的RLHF用。 |

## 延伸阅读

- [Schulman等(2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)——论文。
- [Schulman等(2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)——TRPO,PPO前身。
- [Andrychowicz等(2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)——每个PPO超参数消融。
- [Ouyang等(2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)——InstructGPT;PPO在RLHF配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)——配PyTorch清晰现代阐述。
- [CleanRL PPO实现](https://github.com/vwxyzjn/cleanrl)——许多论文用的参考单文件PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)——语言模型PPO生产配方;与课程09(RLHF)并读。
- [Engstrom等(2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)——"37代码级优化"论文;哪些PPO技巧承载哪些是传说。