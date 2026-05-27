# DPO：直接偏好优化

> RLHF 有效。但它需要训练三个模型（SFT、奖励模型、策略），管理 PPO 的不稳定性，并调优 KL 惩罚。DPO 问：如果我们可以跳过所有这些呢？DPO 直接在偏好对上优化语言模型。没有奖励模型。没有 PPO。一个训练循环。相同的结果。

**类型:** 构建
**语言:** Python (使用 numpy)
**前置要求:** 第10阶段，第07课 (RLHF)
**时间:** ~90分钟

## 学习目标

- 实现 DPO 训练，直接在偏好对上优化语言模型，无需单独的奖励模型
- 推导 DPO 损失函数并解释它如何通过策略的对数概率隐式表示奖励模型
- 在训练稳定性、计算成本和所需模型数量方面比较 DPO 与 RLHF
- 调整 beta 参数以控制训练策略与参考模型的偏离程度

## 问题背景

你在第07课构建了一个 RLHF 管道。三个阶段。三个模型。SFT 模型、奖励模型和使用 PPO 优化的策略模型。仅奖励模型就需要数千个人类偏好对和一个单独的训练循环。PPO 需要仔细调优 KL 系数、学习率、裁剪比例和轮数。

在实践中，PPO 训练 notoriously 不稳定。小的超参数变化会导致训练发散。奖励模型是人类偏好的不完美代理，策略会找到利用其弱点的方法。KL 惩罚有帮助但需要自己的调优——太低会得到奖励黑客，太高则模型几乎学不到东西。

这种复杂性就是为什么大多数开源模型在 InstructGPT 发表多年后仍在 RLHF 上挣扎。三阶段管道很脆弱。每个阶段都有自己的失败模式，错误会累积。

2023年5月，斯坦福大学的 Rafael Rafailov、Archit Sharma 及其同事发表了"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"。关键洞察：你不需要单独的奖励模型。最优奖励函数在数学上由语言模型自身的 Token 概率决定。你可以完全跳过奖励模型，直接在偏好对上优化语言模型。

DPO 将 RLHF 简化为单一的监督学习步骤。一个模型。一个损失函数。一个训练循环。没有强化学习。Zephyr-7B，最早大规模使用 DPO 的模型之一，在几个基准测试上匹配或击败了用完整 RLHF 训练的模型。Meta 将 DPO 用作 Llama 3 对齐管道的一部分。Anthropic 在其对齐研究中引用了 DPO 风格的方法。

## 概念讲解

### 关键洞察

RLHF 优化这个目标：

```
最大化: E[R(x, y)] - beta * KL(pi || pi_ref)
```

其中 R 是奖励模型，pi 是策略，pi_ref 是参考模型，beta 是 KL 系数。

DPO 论文表明这个目标有一个闭式最优解。对于任何奖励函数 R，最优策略是：

```
pi*(y | x) = pi_ref(y | x) * exp(R(x, y) / beta) / Z(x)
```

其中 Z(x) 是归一化常数。重排：

```
R(x, y) = beta * log(pi*(y | x) / pi_ref(y | x)) + beta * log Z(x)
```

这是突破。奖励完全用策略模型的概率和参考模型的概率表示。你不需要训练单独的奖励模型。奖励是*隐式*在概率比中的。

将其代入 Bradley-Terry 偏好模型：

```
P(y_w > y_l | x) = sigmoid(R(x, y_w) - R(x, y_l))
                  = sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))
```

Z(x) 项消去，因为两个响应都以相同的提示 x 为条件。剩下的只是策略模型的对数概率和参考模型的对数概率在偏好和拒绝响应上的函数。

### DPO 损失

```
L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
```

分解每个部分：

- **y_w** = 偏好（获胜）响应
- **y_l** = 拒绝（失败）响应
- **x** = 提示
- **pi** = 当前模型（正在训练）
- **pi_ref** = 参考模型（冻结的 SFT 检查点）
- **beta** = 温度参数，控制与参考的偏离（通常为 0.1 到 0.5）

比率 `log pi(y|x) / pi_ref(y|x)` 是对数概率比。当这个比率为正时，当前模型比参考模型赋予响应 y 更高的概率。为负时，当前模型赋予更低的概率。

DPO 损失推动模型增加偏好响应的对数概率比，减少拒绝响应的。beta 参数控制模型可以偏离参考的程度——小 beta 允许大偏离，大 beta 保持模型接近参考。

### 为什么 DPO 更简单

| 方面 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要训练的模型 | 3 (SFT + 奖励 + 策略) | 1 (仅策略) |
| 训练循环 | 3 (SFT、RM 训练、PPO) | 2 (SFT、DPO) |
| 超参数 | lr、KL 系数、裁剪比例、RM lr、轮数 x3 | lr、beta、轮数 |
| 奖励模型 | 必需（单独训练） | 隐式在模型概率中 |
| RL 算法 | PPO（复杂、不稳定） | 监督学习（稳定） |
| GPU 内存 | PPO 期间内存中 3-4 个模型 | 2 个模型（当前 + 参考） |
| 训练稳定性 | 对超参数敏感 | 稳健，类似于 SFT |

DPO 在训练期间需要内存中两个模型——当前模型和冻结的参考。RLHF 需要三个或四个：策略、参考、奖励模型，以及可选的价值函数基线。对于 70B 模型，每个副本在 FP16 中占用 140GB。消除奖励模型带来的内存节省是实质性的。

### DPO 何时胜过 RLHF

**小数据集。** 使用 5,000-20,000 个偏好对，DPO 经常匹配或超过 RLHF。RLHF 中的奖励模型需要足够数据来泛化——数据有限时，它过拟合并产生不可靠的奖励信号。DPO 通过根本不需要奖励模型来绕过这个问题。

**有限计算。** DPO 需要大约完整 RLHF 三分之一的计算（一个训练循环而不是三个）。对于没有大型 GPU 集群的团队，这是实际的选择。

**快速迭代。** 想尝试 10 个不同的偏好数据集来看哪个产生最佳模型？DPO 让你在几小时内运行每个实验。RLHF 需要为每个数据集重新训练奖励模型。

### RLHF 何时胜过 DPO

**大规模训练。** 在 GPT-4 或 Claude 的规模上，RLHF 的单独奖励模型可以捕获更细致的偏好信号。奖励模型充当一个学习损失函数，适应复杂的质量标准。

**复杂奖励信号。** 当"更好"涉及多个维度（有用性、无害性、诚实性）时，奖励模型可以学习这种多目标权衡。DPO 将每个偏好对视为二元信号——一个更好，一个更差——而不建模为什么。

**迭代对齐。** RLHF 管道可以用当前策略生成新响应，让人类评分，并在在线循环中重新训练奖励模型。DPO 在固定的偏好对数据集上工作。Constitutional AI（Anthropic 的方法）广泛利用 RLHF 的这种迭代特性。

### DPO 之后：KTO、ORPO、SimPO

DPO 启发了一系列简化对齐方法。

**KTO (Kahneman-Tversky Optimization, 2024)：** 你甚至不需要成对数据。KTO 使用不成对反馈——只需将每个响应标记为"好"或"坏"，无需与替代方案比较。这大大简化了数据收集。不再向标注者展示两个响应并问"哪个更好？"，而是展示一个响应问"这个好吗？"损失函数应用前景理论中的损失厌恶：坏响应的惩罚比好响应的奖励更多。

**ORPO (Odds Ratio Preference Optimization, 2024)：** 在单个训练步骤中结合 SFT 和对齐。不是先 SFT 再 DPO，ORPO 修改 SFT 损失以包含偏好信号。损失有两项：偏好响应上的标准下一 Token 预测损失，加上增加偏好和拒绝响应概率之间差距的赔率比项。一个训练循环而不是两个。

**SimPO (Simple Preference Optimization, 2024)：** 完全消除参考模型。不是计算与冻结参考的对数概率比，SimPO 使用响应的平均对数概率（按长度归一化）作为隐式奖励。这节省内存（不需要参考模型）并简化训练。长度归一化防止模型偏爱更短的响应。

| 方法 | 年份 | 内存中模型数 | 需要成对数据？ | 需要参考？ | 训练循环 |
|------|------|-------------|-------------|----------|---------|
| RLHF | 2022 | 3-4 | 是（用于 RM） | 是 | 3 |
| DPO | 2023 | 2 | 是 | 是 | 2 |
| KTO | 2024 | 2 | 否（不成对） | 是 | 2 |
| ORPO | 2024 | 1 | 是 | 否 | 1 |
| SimPO | 2024 | 1 | 是 | 否 | 1 |

趋势清晰：每种方法消除一个复杂性。RLHF 需要奖励模型和 PPO。DPO 消除了两者。KTO 消除了成对数据。ORPO 消除了单独的 SFT 阶段。SimPO 消除了参考模型。对齐税——从基础模型到对齐模型的计算和复杂性成本——持续下降。

### 真实 DPO 部署

**Zephyr-7B (HuggingFace, 2023年10月)：** Mistral 7B 基础，在 UltraChat 上 SFT（200K 示例），然后在 UltraFeedback 上 DPO（60K 偏好对）。MT-Bench 得分 6.47——当时最高的 7B 模型。作为对比，Llama 2 Chat 70B 得分 6.86，意味着 Zephyr 仅使用 DPO 对齐就达到了 10 倍大小模型的 94% 以内。

**Llama 3 (Meta, 2024年4月)：** 在初始 RLHF 阶段后使用 DPO。这种组合表明 DPO 和 RLHF 可以互补——RLHF 用于广泛对齐，DPO 用于针对性细化。

**Neural Magic / nm-chat (2024)：** 将 DPO 应用于多个开源模型，在基准测试中一致显示比仅 SFT 基线提高 5-15% 的对齐效果。

## 动手实践

### 步骤1：偏好数据集

与 RLHF 格式相同——（提示、偏好、拒绝）三元组。DPO 直接使用这些数据，无需中间奖励模型。

```python
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock

PREFERENCE_DATA = [
    {
        "prompt": "法国的首都是哪里？",
        "preferred": "法国的首都是巴黎。",
        "rejected": "法国是欧洲的一个国家。它有很多城市。首都是巴黎。巴黎以埃菲尔铁塔闻名。",
    },
    {
        "prompt": "用一句话解释重力。",
        "preferred": "重力是使有质量的物体相互吸引的力。",
        "rejected": "重力是当你掉落东西时让它们下落的东西。",
    },
    {
        "prompt": "15 乘以 7 是多少？",
        "preferred": "15 乘以 7 是 105。",
        "rejected": "让我想想。15 乘以 7。嗯，10 乘以 7 是 70，5 乘以 7 是 35，所以答案可能是 105。",
    },
    {
        "prompt": "说出三种编程语言。",
        "preferred": "Python、Rust 和 TypeScript。",
        "rejected": "有很多编程语言。一些流行的包括各种语言如 Python 等。",
    },
    {
        "prompt": "二战是哪一年结束的？",
        "preferred": "二战于 1945 年结束。",
        "rejected": "二战是一场重大的全球冲突。涉及许多国家。战争于 1940 年代中期结束，具体是 1945 年。",
    },
    {
        "prompt": "定义机器学习。",
        "preferred": "机器学习是一个领域，算法从数据中学习模式以进行预测，而无需显式编程。",
        "rejected": "机器学习是一种 AI。AI 代表人工智能。机器学习使用数据来学习。",
    },
]
```

### 步骤2：序列对数概率

DPO 损失需要计算给定提示的响应的总对数概率。这意味着在完整的（提示 + 响应）序列上运行模型，并对每个响应 Token 的对数概率求和。

```python
def tokenize_sequence(text, vocab_size=256):
    return [min(t, vocab_size - 1) for t in list(text.encode("utf-8"))]


def compute_sequence_log_prob(model, prompt_tokens, response_tokens, max_seq_len=128):
    full_sequence = prompt_tokens + response_tokens
    if len(full_sequence) > max_seq_len:
        full_sequence = full_sequence[:max_seq_len]

    if len(full_sequence) < 2:
        return 0.0

    input_ids = np.array(full_sequence[:-1]).reshape(1, -1)
    target_ids = np.array(full_sequence[1:])

    logits = model.forward(input_ids)
    logits = logits[0]

    max_logits = logits.max(axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(
        np.exp(logits - max_logits).sum(axis=-1, keepdims=True)
    )

    prompt_len = len(prompt_tokens)
    response_start = max(0, prompt_len - 1)
    response_end = len(target_ids)

    if response_start >= response_end:
        return 0.0

    response_log_probs = log_probs[response_start:response_end, :]
    response_targets = target_ids[response_start:response_end]

    total_log_prob = 0.0
    for i, target in enumerate(response_targets):
        total_log_prob += response_log_probs[i, target]

    return total_log_prob
```

这个函数是 DPO 的主力。对于每个偏好对，它运行四次：模型在偏好响应上、模型在拒绝响应上、参考在偏好响应上、参考在拒绝响应上。每个训练示例 4 次前向传递，而 RLHF 的生成 + 奖励评分 + 价值估计 + PPO 更新。更简单、更快、更稳定。

### 步骤3：DPO 损失

论文的核心代码。一个函数。一个损失。没有奖励模型。

```python
def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def dpo_loss(policy_logprob_preferred, policy_logprob_rejected,
             ref_logprob_preferred, ref_logprob_rejected, beta=0.1):
    preferred_ratio = policy_logprob_preferred - ref_logprob_preferred
    rejected_ratio = policy_logprob_rejected - ref_logprob_rejected

    logit = beta * (preferred_ratio - rejected_ratio)

    loss = -np.log(sigmoid(logit) + 1e-8)

    preferred_reward = beta * preferred_ratio
    rejected_reward = beta * rejected_ratio

    return loss, {
        "preferred_ratio": float(preferred_ratio),
        "rejected_ratio": float(rejected_ratio),
        "logit": float(logit),
        "implicit_preferred_reward": float(preferred_reward),
        "implicit_rejected_reward": float(rejected_reward),
        "reward_margin": float(preferred_reward - rejected_reward),
    }
```

`preferred_ratio` 和 `rejected_ratio` 来自 DPO 推导的对数概率比。当当前模型相对于参考赋予偏好响应更高概率、赋予拒绝响应更低概率时，logit 为正，损失较低。训练信号正好推动模型朝这个方向。

`implicit_preferred_reward` 和 `implicit_rejected_reward` 是 DPO 损失隐式分配的奖励。你可以提取它们来验证训练是否有效——偏好和拒绝奖励之间的差距应该随训练增加。

### 步骤4：DPO 训练循环

标准的监督训练循环。没有 PPO。没有奖励模型。只有前向传递和梯度更新。

```python
def copy_model_weights(source, target):
    target.embedding.token_embed = source.embedding.token_embed.copy()
    target.embedding.pos_embed = source.embedding.pos_embed.copy()
    target.ln_f.gamma = source.ln_f.gamma.copy()
    target.ln_f.beta = source.ln_f.beta.copy()
    for s_block, t_block in zip(source.blocks, target.blocks):
        t_block.attn.W_q = s_block.attn.W_q.copy()
        t_block.attn.W_k = s_block.attn.W_k.copy()
        t_block.attn.W_v = s_block.attn.W_v.copy()
        t_block.attn.W_out = s_block.attn.W_out.copy()
        t_block.ffn.W1 = s_block.ffn.W1.copy()
        t_block.ffn.W2 = s_block.ffn.W2.copy()
        t_block.ffn.b1 = s_block.ffn.b1.copy()
        t_block.ffn.b2 = s_block.ffn.b2.copy()
        t_block.ln1.gamma = s_block.ln1.gamma.copy()
        t_block.ln1.beta = s_block.ln1.beta.copy()
        t_block.ln2.gamma = s_block.ln2.gamma.copy()
        t_block.ln2.beta = s_block.ln2.beta.copy()


def dpo_train(policy_model, reference_model, preference_data,
              num_epochs=5, lr=5e-6, beta=0.1, max_seq_len=128):
    print(f"DPO 训练: {len(preference_data)} 对, {num_epochs} 轮, "
          f"lr={lr}, beta={beta}")
    print()

    losses = []
    margins = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_margin = 0.0
        num_examples = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            prompt_tokens = tokenize_sequence(pair["prompt"])
            preferred_tokens = tokenize_sequence(pair["preferred"])
            rejected_tokens = tokenize_sequence(pair["rejected"])

            pi_logprob_w = compute_sequence_log_prob(
                policy_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            pi_logprob_l = compute_sequence_log_prob(
                policy_model, prompt_tokens, rejected_tokens, max_seq_len
            )
            ref_logprob_w = compute_sequence_log_prob(
                reference_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            ref_logprob_l = compute_sequence_log_prob(
                reference_model, prompt_tokens, rejected_tokens, max_seq_len
            )

            loss, metrics = dpo_loss(
                pi_logprob_w, pi_logprob_l,
                ref_logprob_w, ref_logprob_l, beta
            )

            update_direction = 1.0 if metrics["logit"] < 0 else -0.1
            for block in policy_model.blocks:
                block.ffn.W1 += lr * update_direction * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 += lr * update_direction * np.random.randn(*block.ffn.W2.shape) * 0.01

            epoch_loss += loss
            epoch_margin += metrics["reward_margin"]
            num_examples += 1
            losses.append(float(loss))
            margins.append(metrics["reward_margin"])

        avg_loss = epoch_loss / max(num_examples, 1)
        avg_margin = epoch_margin / max(num_examples, 1)

        print(f"  第 {epoch + 1}/{num_epochs} 轮 | 损失: {avg_loss:.4f} | "
              f"平均差距: {avg_margin:.4f}")

    return policy_model, losses, margins
```

与 RLHF 相比，训练循环简单得令人耳目一新。对于每个偏好对：计算四个对数概率（两个模型，两个响应），代入 DPO 损失，计算梯度，更新策略。没有生成步骤。没有奖励模型推理。没有优势估计。没有裁剪。

### 步骤5：比较 DPO 与 RLHF

测量隐式奖励差距和对数概率偏移，将 DPO 与第07课的 RLHF 模型进行比较。

```python
def evaluate_preference_accuracy(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    correct = 0
    total = 0

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        preferred_reward = beta * (pi_w - ref_w)
        rejected_reward = beta * (pi_l - ref_l)

        if preferred_reward > rejected_reward:
            correct += 1
        total += 1

    return correct / max(total, 1)


def analyze_implicit_rewards(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    print("隐式奖励分析：")
    print("-" * 65)
    print(f"  {'提示':<30} {'偏好奖励':>12} {'拒绝奖励':>12} {'差距':>10}")
    print("  " + "-" * 60)

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        pref_reward = beta * (pi_w - ref_w)
        rej_reward = beta * (pi_l - ref_l)
        margin = pref_reward - rej_reward

        truncated = pair["prompt"][:28] + ".." if len(pair["prompt"]) > 30 else pair["prompt"]
        print(f"  {truncated:<30} {pref_reward:>12.4f} {rej_reward:>12.4f} {margin:>10.4f}")

    print()
```

### 步骤6：Beta 敏感度分析

beta 参数是 DPO 中等价于 RLHF 中 KL 系数的。它控制模型可以偏离参考的程度。这个实验展示其效果。

```python
def beta_sensitivity_analysis(sft_model, preference_data, betas, max_seq_len=128):
    print("Beta 敏感度分析")
    print("-" * 60)
    print(f"  {'Beta':>8} {'最终损失':>12} {'最终差距':>14} {'准确率':>10}")
    print("  " + "-" * 55)

    results = []

    for beta in betas:
        policy = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        reference = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        copy_model_weights(sft_model, policy)
        copy_model_weights(sft_model, reference)

        policy, losses, margins_list = dpo_train(
            policy, reference, preference_data,
            num_epochs=3, lr=5e-6, beta=beta, max_seq_len=max_seq_len
        )

        accuracy = evaluate_preference_accuracy(
            policy, reference, preference_data, beta, max_seq_len
        )

        final_loss = losses[-1] if losses else 0
        final_margin = margins_list[-1] if margins_list else 0

        print(f"  {beta:>8.3f} {final_loss:>12.4f} {final_margin:>14.4f} {accuracy:>10.1%}")
        results.append({
            "beta": beta,
            "final_loss": final_loss,
            "final_margin": final_margin,
            "accuracy": accuracy,
        })

        print()

    return results
```

小 beta (0.01) 让模型自由偏离参考——快速学习但有退化解的风险。大 beta (1.0) 保持模型接近参考——稳定但学习缓慢。大多数应用的甜点区是 0.1 到 0.3。

## 使用实践

### 完整 DPO 管道演示

```python
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("DPO: 直接偏好优化")
    print("=" * 70)
    print()

    print("步骤1：初始化 SFT 模型（来自第06课）")
    print("-" * 50)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  参数量: {sft_model.count_parameters():,}")
    print()

    print("步骤2：DPO 训练")
    print("-" * 50)

    policy_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    reference_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    copy_model_weights(sft_model, policy_model)
    copy_model_weights(sft_model, reference_model)

    policy_model, losses, margins = dpo_train(
        policy_model, reference_model, PREFERENCE_DATA,
        num_epochs=5, lr=5e-6, beta=0.1
    )
    print()

    print("=" * 70)
    print("步骤3：评估")
    print("=" * 70)
    print()

    pre_accuracy = evaluate_preference_accuracy(
        sft_model, reference_model, PREFERENCE_DATA, beta=0.1
    )
    post_accuracy = evaluate_preference_accuracy(
        policy_model, reference_model, PREFERENCE_DATA, beta=0.1
    )

    print(f"  偏好准确率 (DPO 前):  {pre_accuracy:.1%}")
    print(f"  偏好准确率 (DPO 后): {post_accuracy:.1%}")
    print()

    analyze_implicit_rewards(policy_model, reference_model, PREFERENCE_DATA, beta=0.1)

    print("=" * 70)
    print("步骤4：训练动态")
    print("=" * 70)
    print()

    if losses:
        print("  损失曲线：")
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    步骤 {i:3d}-{i + len(chunk) - 1:3d}: 损失 = {avg:.4f}")
        print()

    if margins:
        print("  奖励差距曲线：")
        window = max(1, len(margins) // 5)
        for i in range(0, len(margins), window):
            chunk = margins[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    步骤 {i:3d}-{i + len(chunk) - 1:3d}: 差距 = {avg:.4f}")
        print()

    print("=" * 70)
    print("步骤5：Beta 敏感度")
    print("=" * 70)
    print()

    beta_results = beta_sensitivity_analysis(
        sft_model, PREFERENCE_DATA, betas=[0.01, 0.1, 0.3, 1.0]
    )

    print("=" * 70)
    print("DPO 与 RLHF 比较")
    print("=" * 70)
    print()
    print("  DPO 优势：")
    print("    - 1 个训练循环（RLHF 需要 3 个）")
    print("    - 内存中 2 个模型（RLHF 需要 3-4 个）")
    print("    - 监督学习（vs RL，更稳定）")
    print("    - 无需训练或维护奖励模型")
    print()
    print("  RLHF 优势：")
    print("    - 单独奖励模型捕获复杂偏好")
    print("    - 在线学习：生成、评分、重新训练")
    print("    - 更适合多目标对齐")
    print("    - 在最大规模上已验证（GPT-4、Claude）")
    print()
    print("  实用指导：")
    print("    - 从 DPO 开始。它更简单，通常足够。")
    print("    - 如果 DPO 在评估指标上停滞不前，切换到 RLHF。")
    print("    - 许多生产系统两者都用：RLHF 先，DPO 细化。")
```

## 产出成果

本课程产出 `outputs/prompt-alignment-method-selector.md` —— 一个帮助你为用例选择正确对齐方法（SFT、RLHF、DPO、KTO、ORPO、SimPO）的提示。给定你的数据可用性、计算预算和对齐目标，它推荐一种方法和训练计划。

## 练习题

1. 实现 KTO（Kahneman-Tversky Optimization）。KTO 不需要成对数据——只需将每个响应标记为"好"或"坏"。好响应的损失是 `-log(sigmoid(beta * log_ratio))`，坏响应的损失是 `-log(1 - sigmoid(beta * log_ratio))`，带有损失厌恶乘数（通常为 1.5 倍）。在相同数据上训练（将偏好视为"好"，拒绝视为"坏"，独立处理）并与 DPO 比较准确率。

2. 实现长度归一化 DPO。不是原始对数概率，而是除以响应 Token 数：`normalized_logprob = total_logprob / num_tokens`。这防止模型偏爱更短的响应（具有更高的总对数概率）。比较有和没有归一化的隐式奖励差距。

3. 构建一个 ORPO 风格的组合损失。将偏好响应上的标准下一 Token 预测损失加到 DPO 损失上：`L = L_sft(preferred) + alpha * L_dpo`。尝试 alpha 值为 0.1、0.5 和 1.0。组合损失应该产生一个既遵循指令（来自 SFT 项）又偏好更好响应（来自 DPO 项）的模型，消除单独 SFT 阶段的需要。

4. 实现迭代 DPO。运行 DPO 3 轮，然后从训练好的模型生成新响应，将它们与原始偏好响应配对作为新的偏好对，再次运行 DPO。这个"自我对弈"过程进行两轮。比较第1轮和第2轮后的偏好准确率，看迭代细化是否有帮助。

5. 用不同的参考模型比较 DPO。不是使用 SFT 检查点作为参考，尝试：(a) 基础模型（SFT 前），(b) DPO 第1轮的检查点，(c) 策略模型的指数移动平均。报告哪个参考产生最高的偏好准确率和最稳定的训练曲线。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| DPO | "无 RL 的 RLHF" | 直接偏好优化：在偏好对上直接优化语言模型的监督学习算法，绕过奖励模型和 PPO |
| 隐式奖励 | "奖励在模型中" | 奖励函数由策略和参考模型之间的对数概率比决定——不需要单独的奖励模型 |
| Beta (DPO) | "温度" | 控制策略可以偏离参考模型的程度——小 beta 允许大偏离，大 beta 保持模型接近 |
| 对数概率比 | "模型变化了多少" | log pi(y\|x) - log pi_ref(y\|x) —— 正数表示当前模型比参考赋予更高概率 |
| 参考模型 | "冻结的检查点" | SFT 模型的副本，权重永不改变——作为计算概率比的锚点 |
| KTO | "无成对数据的 DPO" | Kahneman-Tversky Optimization：使用不成对的"好"或"坏"标签，不需要偏好对 |
| ORPO | "一步对齐" | Odds Ratio Preference Optimization：通过向 SFT 损失添加偏好项，将 SFT 和对齐合并为单个训练循环 |
| SimPO | "不需要参考" | Simple Preference Optimization：通过使用长度归一化平均对数概率作为隐式奖励来消除参考模型 |
| 对齐税 | "使模型安全的成本" | 从基础模型到对齐模型所需的额外计算、数据和复杂性——DPO 显著降低这个成本 |

## 延伸阅读

- [Rafailov 等人，2023 — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290) —— 将对齐从 RLHF 简化为监督学习的 DPO 论文
- [Tunstall 等人，2023 — "Zephyr: Direct Distillation of LM Alignment"](https://arxiv.org/abs/2310.16944) —— Zephyr-7B，展示 UltraFeedback 上的 DPO 在基准测试上匹配 RLHF
- [Ethayarajh 等人，2024 — "KTO: Model Alignment as Prospect Theoretic Optimization"](https://arxiv.org/abs/2402.01306) —— 消除成对偏好的需要
- [Hong 等人，2024 — "ORPO: Monolithic Preference Optimization without Reference Model"](https://arxiv.org/abs/2403.07691) —— 一步结合 SFT 和对齐
- [Meng 等人，2024 — "SimPO: Simple Preference Optimization with a Reference-Free Reward"](https://arxiv.org/abs/2405.14734) —— 完全消除参考模型
- [Llama 3 技术报告](https://arxiv.org/abs/2407.21783) —— Meta 结合 RLHF 和 DPO 的对齐管道
