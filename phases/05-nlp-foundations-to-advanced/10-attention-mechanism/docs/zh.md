# 注意力机制 — 突破

> 解码器不再眯眼盯着压缩摘要，开始查看整个源。此后的一切都是注意力加工程。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 09（序列到序列模型）
**时间：** 约45分钟

## 问题背景

第09课以可测量的失效结束。在玩具复制任务上训练的GRU编码器-解码器，长度5时89%准确性到长度80时接近随机。原因是结构性的，不是训练错误：编码器收集的每个信息位必须适合一个固定大小的隐藏状态，解码器从未看到其他任何东西。

Bahdanau、Cho和Bengio在2014年发表了一个三行修复。不是只给解码器最终编码器状态，而是保持每个编码器状态。在每个解码器步，计算编码器状态的加权平均，其中权重表示"解码器现在需要多大程度上查看编码器位置 `i`？"那个加权平均是上下文，每个解码器步都变化。

这就是全部想法。Transformer扩展了它。自注意将其应用于单个序列。多头注意并行运行它。但2014版本已经打破了瓶颈，一旦你有了它，转向Transformer就是工程，不是概念。

## 概念讲解

![Bahdanau注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码器步 `t`：

1. 使用先前解码器隐藏状态 `s_{t-1}` 作为**查询**。
2. 针对每个编码器隐藏状态 `h_1, ..., h_T` 打分。每个编码器位置一个标量。
3. Softmax分数得到注意力权重 `α_{t,1}, ..., α_{t,T}`，和为1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器取 `c_t` 加上先前输出词元，产生下一个词元。

加权平均就是重点。当解码器需要将"Je"翻译成"I"时，它对"Je"上的编码器状态权重高，其他低。当它需要"not"时，对"pas"权重高。上下文向量每步重塑。

## 形状（每个人都会在这里出错的地方）

这是每个注意力实现第一次出错的地方。慢慢读。

| 东西 | 形状 | 注释 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | Softmax后在所有 `i` 上 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态相同形状 |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状 `(d_s,)`，`h_i` 形状 `(d_h,)`。
- `W_a` 形状 `(d_attn, d_s)`。`U_a` 形状 `(d_attn, d_h)`。
- tanh内的和形状 `(d_attn,)`。
- `v_α` 形状 `(d_attn,)`。与 `v_α` 的内积坍缩为标量。**这就是 `v_α` 的作用。** 不是魔法。是将注意力维度向量变成标量分数的投影。

**Luong（乘性）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果编码器双向则跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状 `(d_s, d_h)`。移除等维约束。
- `concat`：本质上是Bahdanau形式。自Luong的前两种更便宜以来很少使用。

**一个值得命名的Bahdanau / Luong陷阱。** Bahdanau使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong使用 `s_t`（生成*之后*的状态）。混合它们产生微妙错误的梯度，极难调试。选一篇论文并坚持其约定。

## 动手实践

### 步骤1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面表格检查形状。`encoder_states` 形状 `(T_enc, d_h)`。`projected_enc` 形状 `(T_enc, d_attn)`。`projected_dec` 形状 `(d_attn,)` 并广播。`combined` 形状 `(T_enc, d_attn)`。`scores` 形状 `(T_enc,)`。`weights` 形状 `(T_enc,)`。`context` 形状 `(d_h,)`。发货。

### 步骤2：Luong点积和通用

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行。这就是Luong论文落地的原因。大多数任务上相同准确性，代码少得多。

### 步骤3：一个计算数值示例

给定三个编码器状态（大致"cat"、"sat"、"mat"）和一个与第一个对齐的解码器状态，注意力分布集中在位置0。如果解码器状态转移以与最后一个对齐，注意力移动到位置2。上下文向量跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。然后将解码器状态移近第三个编码器状态，观察权重转移。就是这样。注意力是显式对齐。

### 步骤4：为什么这是通往Transformer的桥梁

将上面的语言翻译成Q/K/V：

- **查询** = 解码器状态 `s_{t-1}`
- **键** = 编码器状态（我们打分的东西）
- **值** = 编码器状态（我们加权求和的东西）

在经典注意力中，键和值是相同的东西。自注意分离它们：你可以用不同学习投影在单个序列上查询自身。多头注意并行运行不同学习投影。Transformer堆叠整个阶段多次并放弃RNN。

数学相同。形状相同。从Bahdanau注意力到缩放点积注意力的教学跳跃主要是符号。

## 实际应用

PyTorch和TensorFlow直接提供注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

那就是Transformer注意力层。查询批次5个位置，键/值批次10个位置，每个128维，8个头。`output` 是新的上下文增强查询。`weights` 是可可视化的5x10对齐矩阵。

### 经典注意力仍然重要的地方

- 教学。单头、单层、基于RNN的版本使每个概念可见。
- 设备序列任务，Transformer装不下。
- 任何2014-2017年的论文。不知道Bahdanau的约定你会误读它。
- 机器翻译中的细粒度对齐分析。原始注意力权重是可解释性工具，即使在Transformer模型上，阅读它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可解释。它们是和为1的权重；你可以绘制它们；高意味着"看了这个"。审稿人喜欢它们。

它们不像看起来那么可解释。Jain和Wallace（2019）表明，对于某些任务，注意力分布可以置换并用任意替代替换而不改变模型预测。永远不要将注意力权重作为推理证据报告，除非有消融或反事实检查。

## 产出成果

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: 调试注意力实现中的形状错误。
phase: 5
lesson: 10
---

给定损坏的注意力实现，你识别形状不匹配。输出：

1. 哪个矩阵形状错误。命名张量。
2. 它应该是什么形状，从（d_s、d_h、d_attn、T_enc、T_dec、batch_size）推导。
3. 一行修复。转置、重塑或投影。
4. 捕获回归的测试。通常：断言 `output.shape == (batch, T_dec, d_h)` 和 `weights.shape == (batch, T_dec, T_enc)` 和 `weights.sum(dim=-1) close to 1`。

拒绝推荐静默广播的修复。广播隐藏的错误稍后表现为静默准确性下降，这是最糟糕的注意力错误。

对于Bahdanau混淆，坚持解码器输入是 `s_{t-1}`（步前状态）。对于Luong，`s_t`（步后状态）。对于点积，标记查询和键之间的维度不匹配为最常见的首次错误。
```

## 练习题

1. **简单。** 实现 `softmax` 掩码，使编码器中的填充词元获得注意力权重零。在可变长度序列批次上测试。
2. **中等。** 向Luong `general` 形式添加多头注意力。将 `d_h` 拆分为 `n_heads` 组，每头运行注意力，拼接。验证单头情况与先前实现匹配。
3. **困难。** 用Bahdanau注意力在第09课的玩具复制任务上训练GRU编码器-解码器。绘制准确性vs序列长度。与无注意力基线对比。你应该看到随着长度增加差距扩大，确认注意力打破瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Attention | 看东西 | 值序列的加权平均，权重从查询-键相似度计算。 |
| Query, Key, Value | QKV | 三个投影：Q提问，K匹配，V返回。 |
| Additive attention | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong点积/通用 | 分数是 `q^T k` 或 `q^T W k`。更便宜，大多数任务上相同准确性。 |
| Alignment matrix | 漂亮的图片 | 注意力权重作为 `(T_dec, T_enc)` 网格。阅读它看模型注意什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的PyTorch演练。
