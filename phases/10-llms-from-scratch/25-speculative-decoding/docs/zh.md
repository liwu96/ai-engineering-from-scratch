# 推测解码与 EAGLE

> 前沿 LLM 生成一个 Token 需要数十亿参数上的完整前向传递。该前向传递严重过度配置：大多数时候，小得多的模型可以正确猜测接下来的 3-5 个 Token，而大模型只需要*验证*猜测。猜测正确时，你以一次的价格获得 5 个 Token。推测解码（Leviathan 等人 2023）使其精确，EAGLE-3（2025）将接受率推至约每验证 4.5 个 Token——在匹配输出分布的情况下实现 4-5 倍加速。

**类型:** 构建
**语言:** Python (使用 numpy)
**前置要求:** 第10阶段第12课（推理优化），第10阶段第04课（预训练微型 GPT）
**时间:** ~75分钟

## 问题背景

H100 上 70B 类模型的解码吞吐量通常为 40-80 Token/秒。每个 Token 需要读取 HBM 中所有模型权重的完整前向传递。你不能在不改变输出的情况下让模型变小。你不能超过内存增加批处理大小。你被困住了——除非你能让模型每次前向传递输出多个 Token。

自回归生成看起来本质上是顺序的：`x_{t+1} = sample(p(· | x_{1:t}))`。但有一个并发机会。如果你有一个便宜的预测器说"接下来的 4 个 Token 很可能是 [a, b, c, d]"，你可以在**大模型的单次前向传递**中验证所有 5 个位置并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过巧妙的接受/拒绝规则使其精确，保留目标模型的采样分布。相同的输出分布，快 2-4 倍。

## 概念讲解

### 双模型设置

- **目标模型** `M_p`：你想要样本的大、慢、高质量模型。分布：`p(x)`。
- **草稿模型** `M_q`：小、快、低质量模型。分布：`q(x)`。5-30 倍小。

每步：

1. 草稿模型自回归提议 `K` 个 Token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型单次前向传递并行运行在所有 `K+1` 个位置，为每个提议 Token 产生 `p(x_k)`。
3. 通过下面的改进拒绝采样规则从左到右接受/拒绝每个 Token。接受最长的匹配前缀。
4. 如果任何 Token 被拒绝，从校正分布采样替换并停止。否则从 `p(· | x_1...x_K)` 采样一个奖励 Token。

如果草稿与目标完美匹配，你每次目标前向获得 K+1 个 Token。如果草稿在第 1 个位置错误，你只获得 1 个 Token。

### 精确性规则

推测解码是**可证明在分布上等价于从 p 采样**。拒绝规则：

```
对于每个草稿 Token x_t：
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        从残差采样替换：(p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差异的正部分。当草稿和目标一致（`p ≈ q`）时，接受接近 1。当它们不一致时，构造残差分布使得整体样本仍然恰好是 `p`。

**贪婪情况。** 对于 temperature=0 采样，只需检查 `argmax(p) == x_t`。如果是，接受；否则输出 `argmax(p)` 并停止。

### 预期加速

如果草稿模型的每 Token 接受率是 `α`，每次目标前向传递产生的预期 Token 数为：

```
E[Token] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度，α ∈ [0, 1]
```

在 `α = 0.8, K = 4`：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 每前向 Token。单次目标前向成本大致为 `cost_q * K + cost_p`（K 个草稿步骤加一次目标验证）。如果 `cost_p >> cost_q * K`，加速比为 `3.36× / 1 = 3.36×` 吞吐量。

唯一真实的参数是 `α`，它完全取决于草稿-目标对齐。好的草稿是一切。

### 训练草稿：蒸馏

随机小模型做草稿很差。标准配方是从目标蒸馏：

1. 选择小架构（70B 目标约 1B，7B 目标约 500M）。
2. 在大型文本语料库上运行目标模型；存储其下一 Token 分布。
3. 用 KL 散度针对目标分布（而非针对真实 Token）训练草稿。

结果：`α` 通常在代码上为 0.6-0.8，自然语言聊天为 0.7-0.85。生产环境中加速 2-3 倍。

### EAGLE：树形草稿 + 特征重用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准推测解码的两个低效：

1. 草稿做 K 个串行步骤，每个都是完整堆栈。但草稿可以重用目标最近验证的特征（隐藏状态）——目标已经计算出草稿从头开始重新推导的丰富表示。
2. 草稿输出线性链。如果草稿能输出*树形*候选（每个节点多个猜测），目标的单次前向传递可以通过树注意力掩码并行验证多个候选路径，并选择最长的接受分支。

EAGLE-1 改变：
- 草稿输入 = 位置 t 的目标最终隐藏状态，而非原始 Token。
- 草稿架构 = 1 个 Transformer 解码器层（非单独小模型）。
- 输出 = 每深度 K = 4-8 个候选，深度 4-6 的树。

EAGLE-2（2024）添加动态树拓扑：树在草稿不确定的地方变宽，在自信的地方保持窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3（Li 等人 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除固定顶层特征依赖，用新的"测试时模拟"损失训练草稿——草稿在与目标测试时分布匹配而非强制教师训练分布的输出上训练。接受率从 0.75（EAGLE-2）上升到 0.82（EAGLE-3），每验证平均 Token 从 3.0 到 4.5。

### 树注意力验证

当草稿输出树时，目标模型在单次前向传递中使用**树注意力掩码**验证它——编码树拓扑而非纯线的因果掩码。每个节点只关注其在树中的祖先。验证传递仍是一次前向、一次矩阵乘法；拓扑掩码仅增加少量 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的首 Token 候选，`c, d, e, f` 是次 Token 候选，全部六个位置在单次前向传递中验证。输出是任何接受路径上的最长前缀。

### 何时有效，何时无效

**有效：**
- 聊天/完成文本可预测（代码、常见英语、结构化输出）。`α` 高。
- GPU 计算在解码期间未充分利用的设置（内存受限阶段）。树形草稿使用可用 FLOPs。

**无效/无收益：**
- 高度随机输出（高温创意写作）。`α` 降至 `1/|vocab|`。
- 并发非常高的批处理服务——批处理已经填满 FLOPs，树验证几乎没有空间。
- 非常小的目标模型，草稿没有小多少。

生产环境通常报告聊天加速 2-3 倍，代码生成 3-5 倍，创意写作接近零。

## 动手实践

`code/main.py`：

- 实现精确拒绝规则的 `speculative_decode(target, draft, prompt, K, temperature)` 并验证它保留目标分布（经验 KL < 0.01 vs 朴素目标采样）。
- EAGLE 风格的树形草稿器，构建带 top-p 分支的深度 K 树。
- 为验证器构建正确因果模式的树注意力掩码构建器。
- 接受率测试工具，在微型 LM（从 GPT-2-medium 目标蒸馏 GPT-2-small 草稿）上运行两者。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮推测解码。返回接受的 Token 列表。"""
    # 1. 草稿 K 个 Token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个草稿位置 + 1 个额外位置计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 全部 K 接受 → 从目标采样奖励 Token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用实践

- **vLLM** 和 **SGLang** 提供一流推测解码。标志：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 草稿）。
- **Medusa 头**（Cai 等人 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads")：替代草稿模型，在目标本身添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 产出成果

本课程产出 `outputs/skill-speculative-tuning.md` —— 分析目标模型工作负载并选择的技能：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到朴素解码。

## 练习题

1. 实现精确拒绝规则并经验验证它。通过 `speculative_decode` 和朴素目标采样运行 10K 样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，绘制每目标前向预期 Token。找到 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练微型草稿。在 100M Token 上用 KL 损失将 30M GPT-2 草稿从 124M GPT-2 目标蒸馏。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树形草稿。不是链，让草稿在每深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长的正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行推测解码。显示 α 崩溃，算法因草稿开销比朴素解码慢。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 目标模型 | "大模型" | 你想要样本的慢、高质量模型（p 分布） |
| 草稿模型 | "推测者" | 小、快预测器（q 分布）；5-30 倍小 |
| K / 草稿长度 | "前瞻" | 每次验证传递推测的 Token 数 |
| α / 接受率 | "命中率" | 草稿提议被接受的每 Token 概率 |
| 精确拒绝规则 | "接受测试" | r < p/q 比较保留目标分布 |
| 残差分布 | "校正 p-q" | (p - q)+ / ||(p - q)+||_1，拒绝时从中采样的分布 |
| 树形草稿 | "分支推测" | 草稿输出候选树，单次传递中用树结构注意力掩码验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 目标本身上的 K 个额外预测头；无单独草稿模型 |
| EAGLE 特征重用 | "隐藏状态草稿" | 草稿输入是目标最后隐藏状态而非原始 Token，缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在与目标测试时分布匹配的输出上训练草稿，而非强制教师 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) —— 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving 等人, 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) —— DeepMind 的并发推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) —— 草稿模型替代方案：并行头
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) —— 特征重用和树形草稿
- [Li 等人, 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) —— 动态树拓扑
- [Li 等人, 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) —— 训练时测试时匹配
- [Fu, Haotian, Peng 等人, 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) —— Jacobi/前瞻解码，无推测者的替代方案
