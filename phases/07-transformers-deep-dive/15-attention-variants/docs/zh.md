# 注意力变体——滑动窗口、稀疏、差分

> 全注意力是圆。每词元见每词元,内存付代价。四变体弯圆形状并恢复半成本。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力机制)、阶段7课程03(多头)、阶段7课程12(KV Cache/Flash注意力机制)
**时间:** ~60分钟

## 问题背景

全注意力序列长`O(N²)`内存和`O(N²)`计算。对128K上下文Llama 3 70B每层16亿注意力入口,乘80层。Flash注意力机制(课程12)隐藏了`O(N²)`激活内存但不改变算术成本——每个词元仍然关注每个其他词元。

三类变体改注意力矩阵拓扑本身:

1. **滑动窗口注意力(SWA)。**每个词元仅关注固定窗口内的邻居，而非整个前缀。内存和计算降到`O(N · W)`,W是窗口。Gemma 2/3、Mistral 7B首层、Phi-3-Long。
2. **稀疏/块注意力。**仅选对`(i, j)`评分;余强制零权重。Longformer、BigBird、OpenAI稀疏transformer。
3. **差分注意力。**配分离Q/K投影算两注意力图,减一从另一。杀"注意力汇"权重渗入首几词元。Microsoft DIFF Transformer(2024)。

这些共存。2026前沿模型常混:多数层SWA-1024,每五层全局全注意力,少数差分头清检索。Gemma 3 5:1 SWA比全局比例是当前教科书默认。

## 概念讲解

### 滑动窗口注意力(SWA)

每个位置`i`的query仅关注`[i - W, i]`范围内的位置(因果SWA)或`[i - W/2, i + W/2]`(双向)。窗口外词元得分数矩阵`-inf`。

```
全因果:                  滑动窗口(W=4):
位置0-7                 位置0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对`N = 8192`和`W = 1024`,分数矩阵期望1024 × 8192非零行——8×减。

**KV cache配SWA缩。**每层仅保最后`W`词元K和V。对Gemma-3形配置(1024窗口,128K上下文),KV cache降128×。

**质量成本。**SWA-only transformer难长程检索。修复:交错SWA层与全注意力层。Gemma 3用5:1 SWA:全局。Mistral 7B用因果-SWA栈信息"向前流"过重叠窗口——每层扩有效感受野W,L层后模型可attend `L × W`词元回。

### 稀疏/块注意力

提前定`N × N`稀疏模式。三规范形状:

- **局部+跨步(OpenAI稀疏transformer)。**Attend最后W词元加之前每`stride`词元。捕获局部和长程`O(N · sqrt(N))`计算。
- **Longformer/BigBird。**局部窗口+少量全局词元(如`[CLS]`)attend所有人并被所有人attend+随机稀疏链接。匹质量2×上下文。
- **原生稀疏注意力(DeepSeek, 2025)。**学哪些`(Q, K)`块重要;kernel级跳零块。FlashAttention兼容。

稀疏注意力是kernel工程故事。数学简单(掩分数矩阵);赢来自从不载零入口进SRAM。FlashAttention-3和2026 FlexAttention API让自定义稀疏模式PyTorch一等公民。

### 差分注意力(DIFF Transformer, 2024)

常规注意力有"注意力汇"问题:softmax强制每行和1,故不想attend任何特定词元倾权重于首词元(或首几)。此偷本应去真实内容容量。

差分注意力通过计算**两个**注意力图并相减来修复这一问题:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

`λ`是学习标量(典型0.5–0.8)。A1捕获真实内容权重;A2捕获汇。减消汇,重分配权重给相关词元。

报告结果(Microsoft 2024):5–10%低困惑度,1.5–2×更长有效上下文同训长度,更尖锐针堆检索。

### 变体对比

| 变体 | 计算 | KV cache | 质量vs全 | 生产使用 |
|------|------|----------|----------|----------|
| 全注意力 | O(N²) | 每层O(N) | 基线 | 每模型默认层 |
| SWA(窗口1024) | O(N·W) | 每层O(W) | -0.1困惑,配全局层好 | Gemma 2/3, Phi-3-Long |
| 局部+跨步稀疏 | O(N·√N) | 混合 | 类SWA | OpenAI稀疏transformer, Longformer |
| BigBird(局部+全局+随机) | 近O(N) | 混合 | 2×上下文匹全 | 早长上下文BERT |
| 原生稀疏(DeepSeek-V3.2) | O(N · 活化分数) | O(N) | 0.05困惑内 | DeepSeek-V3.2, 2025 |
| 差分 | O(2·N²) | O(2N) | -5到-10%困惑 | DIFF Transformer, 早2026模型 |

## 动手实践

见`code/main.py`。实现因果掩码比较器，在玩具序列上并排展示全注意力、SWA、局部+跨步、差分注意力。

### Step 1: 全因果掩码(基线)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

课程07基线。下三角;对角上零权重。

### Step 2: 滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一参数——`window`。对`window >= n`,恢复全因果注意力。对`window = 1`,每词元仅attend自己。

### Step 3: 局部+跨步稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密局部窗口加每`stride`词元回序列始。感受野随额外层log步长。

### Step 4: 差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，通过学习的混合系数相减。代码中比较单注意力与差分注意力的汇热图，观察注意力汇的消散。

### Step 5: KV cache大小

打印每变体`N = 131072`每层cache大小。SWA和稀疏变体降10–100×。差分加倍。清醒付内存账。

## 实际应用

2026生产模式:

```python
from transformers import AutoModelForCausalLM
# Gemma 3混SWA(窗口=1024)和全局层5:1。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ FlexAttention接受掩函数:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

此编译到自定义Triton kernel。常见模式FlashAttention-3速度10%内,且掩函数是Python callable。

**何时选每:**

- **纯粹全注意力**——每层到~16K上下文，或检索质量至关重要时。
- **SWA + 全局混**——长上下文(>32K),训练和推理内存受限。2026默认32K以上。
- **稀疏块注意力**——自定义kernel,自定义模式。留专用工作负载(检索、音频)。
- **差分注意力**——注意力汇污染伤任何工作负载(长上下文RAG、针堆)。

## 产出成果

见`outputs/skill-attention-variant-picker.md`。技能给定目标上下文长度、检索需求和训练/推理计算开销为新模型选注意力拓扑。

## 练习题

1. **简单。**运行`code/main.py`。验证`window=4` SWA每行零化最后4词元外所有。验证`window=n`重现全因果注意力bit-identical。
2. **中等。**课程07毕业项目上实现因果SWA配`window=1024`。tinyshakespeare训1,000步。验证损失vs全注意力回退多少?峰值内存降多少?
3. **困难。**毕业项目模型实现Gemma-3形5:1层混(5 SWA, 1全局)。比损失、内存和生成质量vs纯SWA和纯全局基线匹参数。
4. **困难。**实现配每头学习`λ`差分注意力。合成检索任务训(一针,2,000干扰)。匹参数测检索准确率vs单注意力基线。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 滑动窗口注意力(SWA) | "局部注意力" | 每个query关注其最后W个词元；KV cache缩小到O(W)。 |
| 有效感受野 | "模型看多远回" | L层W窗口SWA栈,到L × W词元。 |
| Longformer/BigBird | "局部+全局+随机" | 配少数总attend全局词元稀疏模式;早长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek kernel技巧" | 学块级稀疏;kernel级跳零块同时保质量。 |
| 差分注意力 | "两图,一减" | DIFF Transformer:减学习λ乘第二注意力图从第一消注意力汇。 |
| 注意力汇 | "权重渗到词元0" | Softmax归一化强制行和1;无信息query倾权重于位置0。 |
| FlexAttention | "掩作Python" | PyTorch 2.5+ API编译任意掩函数进FlashAttention形kernel。 |
| 层类型混 | "5:1 SWA比全局" | 栈中交错稀疏和全注意力层保质量降内存。 |

## 延伸阅读

- [Beltagy, Peters, Cohan(2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)——规范滑动窗口+全局词元论文。
- [Zaheer等(2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)——局部+全局+随机。
- [Child等(2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)——OpenAI局部+跨步模式。
- [Gemma Team(2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)——1:1 SWA:全局混。
- [Gemma Team(2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786)——5:1混配窗口=1024现教科书默认。
- [Ye等(2024). Differential Transformer](https://arxiv.org/abs/2410.05258)——DIFF Transformer论文。
- [Yuan等(2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089)——DeepSeek-V3.2学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/)——实际应用中掩作callable模式API参考。