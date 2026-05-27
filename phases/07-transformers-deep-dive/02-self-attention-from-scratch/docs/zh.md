# 自注意力机制从零实现

> 注意力是查找表,每个词问"谁对我重要?"——并学习答案。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3(深度学习核心)、阶段5课程10(序列到序列)
**时间:** ~90分钟

## 学习目标

- 仅用NumPy从零实现scaled dot-product自注意力,包括query/key/value投影和softmax加权求和
- 构建多头注意力层分裂head、并行计算注意力、拼接结果
- 追踪注意力矩阵如何捕获词元关系并解释为何sqrt(d_k)缩放防止softmax饱和
- 应用因果掩码将双向注意力转为自回归(decoder式)注意力

## 问题背景

RNN逐个词元处理序列。到词元50时,词元1信息已压缩过50步。长程依赖被压成固定大小隐藏状态——LSTM gating永不完全解决的瓶颈。

2014 Bahdanau注意力论文展示修复:让解码器回看每个编码器位置并决定哪些对当前步重要。但仍bolt在RNN上。2017"Attention Is All You Need"论文问更尖锐问题:如果注意力是*唯一*机制?无循环。无卷积。仅注意力。

自注意力让序列中每个位置在单并行步attend到每个其他位置。这就是让transformer快、可扩展、主导的原因。

## 概念讲解

### 数据库查找类比

想注意力为软数据库查找:

```
传统数据库:
  Query: "法国首都"  -->  精确匹配  -->  "巴黎"

注意力:
  Query: "法国首都"  -->  与所有keys相似度  -->  所有values加权混合
```

每个词元生成三向量:
- **Query (Q)**:"我在找什么?"
- **Key (K)**:"我含什么?"
- **Value (V)**:"如果被选中我提供什么信息?"

Query与所有keys点积产生注意力分数。高分意味"此key匹配我query"。这些分数权重values。输出是values加权求和。

### Q, K, V计算

每个词元嵌入通过三学习权重矩阵投影:

```
输入嵌入(n词元序列,每d维):

  X = [x1, x2, x3, ..., xn]       形状: (n, d)

三权重矩阵:

  Wq  形状: (d, dk)
  Wk  形状: (d, dk)
  Wv  形状: (d, dv)

投影:

  Q = X @ Wq    形状: (n, dk)      每词元query
  K = X @ Wk    形状: (n, dk)      每词元key
  V = X @ Wv    形状: (n, dv)      每词元value
```

视觉上,对一词元:

```
             Wq
  x_i ------[*]------> q_i    "我在找什么?"
       |
       |     Wk
       +----[*]------> k_i    "我含什么?"
       |
       |     Wv
       +----[*]------> v_i    "我提供什么?"
```

### 注意力矩阵

一旦有所有词元Q, K, V,注意力分数形成矩阵:

```
Scores = Q @ K^T    形状: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 attend每个key多少
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

每行:一词元在整个序列上注意力
```

### 为何缩放?

点积随维度dk增长。如果dk=64,点积可在数十范围,推softmax到梯度消失区域。修复:除sqrt(dk)。

```
缩放分数 = (Q @ K^T) / sqrt(dk)
```

这保持值在softmax产生有用梯度范围。

### Softmax转分数为权重

Softmax将原始分数转为每行概率分布:

```
q1原始分数:   [2.1, 0.3, 0.1, 0.8, 0.2]
                        |
                     softmax
                        |
注意力权重:   [0.52, 0.09, 0.07, 0.14, 0.08]   (和约1.0)
```

现在每词元有权重集说明attend每个其他词元多少。

### Values加权求和

每词元最终输出是所有value向量加权求和:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

对词元1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整管道

```
                    +-------+
  X (输入)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (输入)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (输入)  ----->|  @ Wv  |-----> V ---------->| 加权     |----> 输出
                    +-------+          ^          | 求和     |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实践

### Step 1: 从零Softmax

Softmax转原始logits为概率。减max数值稳定。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Step 2: Scaled dot-product注意力

核心函数。取Q, K, V矩阵返注意力输出加权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Step 3: 配学习投影自注意力类

配Wq, Wk, Wv权重矩阵Xavier式scaling初始化的完整自注意力模块。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Step 4: 在句子上运行

为句子造假嵌入并观注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("注意力权重(每行:那词元看哪):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Step 5: ASCII热图可视化注意力

映射注意力权重到字符快速视觉。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 实际应用

PyTorch的`nn.MultiheadAttention`做我们建的加多头分裂输出投影:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"输入形状:            {X_torch.shape}")
print(f"输出形状:           {output.shape}")
print(f"注意力权重形状: {attn_weights.shape}")
print(f"\n注意力权重(head平均):")
print(attn_weights[0].detach().numpy().round(3))
```

关键差异:多头注意力并行跑多注意力函数,每配自己dk = d_model / n_heads大小Q, K, V投影,后拼接结果。这让模型同时attend不同关系类型。

## 产出成果

本课程产:
- `outputs/prompt-attention-explainer.md`——用数据库查找类比解释注意力提示

## 练习题

1. 修改`scaled_dot_product_attention`接受可选掩码矩阵在softmax前设某些位置负无穷(这是因果/解码器掩码工作方式)
2. 从零实现多头注意力:将Q, K, V裂为`n_heads`块、每块跑注意力、拼接、通过最终权重矩阵Wo投影
3. 取同长两不同句子、喂同SelfAttention实例、比注意力模式。何变?何同?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Query (Q) | "问题向量" | 输入学习投影代表此词元找什么信息 |
| Key (K) | "标签向量" | 学习投影代表此词元含什么信息,与queries匹配 |
| Value (V) | "内容向量" | 学习投影载基于注意力分数聚合的实际信息 |
| Scaled dot-product注意力 | "注意力公式" | softmax(QK^T / sqrt(dk)) @ V——缩放防高维softmax饱和 |
| 自注意力 | "词元看自己和其他" | Q, K, V同序列来注意力,让每位置attend每其他位置 |
| 注意力权重 | "多少关注" | 位置上概率分布,softmax在缩放点积上产 |
| 多头注意力 | "并行注意力" | 跑配不同投影多注意力函数后拼接结果获更富表示 |

## 延伸阅读

- [Attention Is All You Need(Vaswani等,2017)](https://arxiv.org/abs/1706.03762)——原始transformer论文
- [The Illustrated Transformer(Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)——最佳完整架构视觉walkthrough
- [The Annotated Transformer(Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)——配解释逐行PyTorch实现