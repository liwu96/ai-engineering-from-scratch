# 多头注意力机制

> 一个注意力头一次学一种关系。八个头学八个。头免费。多拿些。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力机制从零实现)
**时间:** ~75分钟

## 问题背景

单自注意力头计算一个注意力矩阵。那矩阵捕获一种关系——通常是训练信号最小化loss的那种。如果你的数据有主谓一致、共指、长程话语、和句法分块全缠在一起,单头把它们糊成单softmax分布丢半信号。

2017 Vaswani论文修复:并行跑多注意力函数,每配自己Q, K, V投影,拼接输出。每头在`d_model / n_heads`维更小子空间操作。总参数不变。表达能力上。

多头注意力是2026每个transformer默认配的。唯一争论是*多少*头和key/value是否共享投影(Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention)。

## 概念讲解

![多头注意力分裂、attend、拼接](../assets/multi-head-attention.svg)

**分裂。**取形`(N, d_model)`的`X`。投影到Q, K, V各形`(N, d_model)`。reshape到`(N, n_heads, d_head)`其中`d_head = d_model / n_heads`。transpose到`(n_heads, N, d_head)`。

**并行attend。**每头内跑scaled dot-product注意力。每头产`(N, d_head)`。头在嵌入不同子空间操作注意力计算本身间从不交谈。

**拼接和投影。**头栈回`(N, d_model)`乘学习输出矩阵`W_o`形`(d_model, d_model)`。`W_o`是头混合处。

**为何工作。**每头可专化不与其他竞争表示预算。2019–2024探针研究示不同头角色:位置头、attend前词元头、复制头、命名实体头、归纳头(支撑上下文学习)。

**2026变体谱系:**

| 变体 | Q头 | K/V头 | 使用者 |
|---------|---------|-----------|---------|
| 多头(MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query(MQA) | N | 1 | PaLM, Falcon |
| Grouped-query(GQA) | N | G(如N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent(MLA) | N | 压缩到低秩 | DeepSeek-V2, V3 |

GQA是现代默认因为它切KV-cache内存`N/G`因子同时保持近完整质量。MLA更远通过压缩K/V到latent空间后计算时投影回——费FLOPs、省更多内存。

## 动手实践

### Step 1: 从已有单头注意力分裂头

取课程02的`SelfAttention`包配split/concat对。见`code/main.py`numpy实现;逻辑:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一reshape一transpose。无循环。这正是PyTorch`nn.MultiheadAttention`下做。

### Step 2: 每头跑scaled-dot-product注意力

每头得自己Q, K, V切片。注意力成批量matmul:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

真实硬件上`Qh @ Kh.transpose(...)`是一`bmm`。GPU见单批量matmul形`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。加头免费。

### Step 3: Grouped-Query Attention变体

仅key和value投影变。Q得`n_heads`组;K和V得`n_kv_heads < n_heads`组并重复匹配:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这省内存因为仅`n_kv_heads`副本在KV cache活,非`n_heads`。Llama 3 70B用64 query头配8 KV头——8× cache缩减。

### Step 4: 探针每头学什么

配4头在短句跑MHA。每头,打印`(N, N)`注意力矩阵。你会见不同头挑出不同结构即使随机初始化——部分信号、部分子空间旋转对称。

## 实际应用

PyTorch一行版:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention CUDA上auto-dispatch Flash Attention。
# GQA,传Q形(B, n_heads, N, d_head)和K,V形
# (B, n_kv_heads, N, d_head)。PyTorch处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少头?**2026生产模型经验:

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小(~125M) | 768 | 12 | 64 |
| 基(~350M) | 1024 | 16 | 64 |
| 大(~1B) | 2048 | 16 | 128 |
| 前沿(~70B) | 8192 | 64 | 128 |

`d_head`几乎总在64或128。是一个头能"看"多少的单位。降到32下头开始与缩放因子`sqrt(d_head)`战;超256失"多小专家"益。

## 产出成果

见`outputs/skill-mha-configurator.md`。技能为新transformer给定参数预算、序列长度、和部署目标推荐头数、kv头数、和投影策略。

## 练习题

1. **简单。**取`code/main.py`MHA改`n_heads`从1到16配`d_model=64`固定。绘合成copy任务小单层模型loss。多头帮、plateau、或害?
2. **中等。**实现MQA(一KV头共享所有query头)。测参数数比全MHA降多少。算N=2048推理KV-cache大小缩多少。
3. **困难。**实现微小Multi-head Latent Attention:压缩K,V到秩`r` latent、存latent在KV cache、注意力时解压。什么`r`时cache内存跨全MHA 1/8下同时质量保持验证ppl 1位内?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Head | "单注意力电路" | 维`d_head = d_model / n_heads`一Q/K/V投影配自己注意力矩阵。 |
| d_head | "头维度" | 每头隐藏宽;生产几乎总64或128。 |
| Split / combine | "reshape技巧" | `(N, d_model) ↔ (n_heads, N, d_head)`注意力周围reshape+transpose。 |
| W_o | "输出投影" | 拼接头后应用`(d_model, d_model)`矩阵;头混合处。 |
| MQA | "一KV头" | Multi-Query Attention:单共享K/V投影。最小KV cache,些质量损。 |
| GQA | "Llama 2后默认" | Grouped-Query Attention配`n_kv_heads < n_heads`;重复匹配Q。 |
| MLA | "DeepSeek技巧" | Multi-head Latent Attention:K,V压缩到低秩latent、attend时解压。 |
| 归纳头 | "上下文学习后电路" | 检测前出现并复制后续的头对。 |

## 延伸阅读

- [Vaswani等(2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762)——原始多头spec。
- [Shazeer(2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)——MQA论文。
- [Ainslie等(2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)——训后如何转MHA到GQA。
- [DeepSeek-AI(2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)——MLA和为何它cache内存上胜MHA/GQA。
- [Olsson等(2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)——头实际做什么机制看。