# 完整Transformer — 编码器+解码器

> 注意力机制是主角。其余一切——残差连接、归一化、前馈网络、交叉注意力——都是让你能深层堆叠它的脚手架。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力机制)、阶段7课程03(多头注意力机制)、阶段7课程04(位置编码)
**时间:** ~75分钟

## 问题背景

单个注意力层是特征提取器,不是模型。每层一次矩阵乘法对语言来说容量不足。你需要深度——而深度若无正确管道会崩溃。

2017年Vaswani论文封装了六个设计决策,把单个注意力层变成可堆叠块。此后每个transformer——仅编码器(BERT)、仅解码器(GPT)、编码器-解码器(T5)——继承相同骨架。2026年块已被改进(RMSNorm、SwiGLU、pre-norm、RoPE),但骨架相同。

本课程是骨架。后续课程专化它——课程06讲编码器、07讲解码器、08讲编码器-解码器。

## 概念讲解

![编码器和解码器块内部结构,已连接](../assets/full-transformer.svg)

### 六个组件

1. **嵌入+位置信号。**词元→向量。位置通过RoPE(现代)或sinusoidal(经典)注入。
2. **自注意力。**每个位置关注其他每个位置。解码器中需要掩码。
3. **前馈网络(FFN)。**位置级两层MLP:`W_2 · activation(W_1 · x)`。默认扩展比4×。
4. **残差连接。**`x + sublayer(x)`。无此,约6层后梯度消失。
5. **层归一化。**`LayerNorm`或`RMSNorm`(现代)。稳定残差流。
6. **交叉注意力(仅解码器)。**查询来自解码器,键和值来自编码器输出。

### 编码器块(用于BERT、T5编码器)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。无掩码。所有位置看到所有位置。

### 解码器块(用于GPT、T5解码器)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯仅解码器架构(GPT)中,交叉注意力被省略,只有掩码自注意力+FFN。

### Pre-norm vs post-norm

原始论文:`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm约2019年失宠——无仔细预热深层训练更难。Pre-norm(`LN`在子层*之前*)是2026默认:Llama、Qwen、GPT-3+、Mistral全用它。

### 2026现代化块

Vaswani 2017发布LayerNorm + ReLU。现代栈替换两者。生产块实际样貌:

| 组件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN激活 | ReLU | SwiGLU |
| FFN扩展 | 4× | 2.6×(SwiGLU用三矩阵,总参数匹配) |
| 位置 | Sinusoidal绝对 | RoPE |
| 注意力 | 全MHA | GQA(或MLA) |
| 偏置项 | 有 | 无 |

RMSNorm去掉LayerNorm的均值中心化(少一次减法),节省计算且经验上至少同样稳定。SwiGLU(`Swish(W1 x) ⊙ W3 x`)在Llama、PaLM和Qwen论文中持续比ReLU/GELU FFN优约0.5点困惑度。

### 参数计数

对`d_model = d`且FFN扩展`r`的一个块:

- MHA:`4 · d²`(Q, K, V, O投影)
- FFN(SwiGLU):`3 · d · (r · d)` ≈ `3rd²`
- 归一化:可忽略

在`d = 4096, r = 2.6, layers = 32`(约Llama 3 8B),总计:`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~每层1.5B参数 × 32 ≈ 7B`(加嵌入和头)。匹配公布计数。

## 动手实践

### Step 1: 构建块

用课程03的微小`Matrix`类(为独立性复制到本文件):

- `layer_norm(x, eps=1e-5)`——减均值,除标准差。
- `rms_norm(x, eps=1e-6)`——除RMS。无均值减法。
- `gelu(x)`和`silu(x) * W3 x`(SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)`和`decoder_block(x, enc_out, params)`。

见`code/main.py`完整连线。

### Step 2: 连接2层编码器和2层解码器

堆叠它们。把编码器输出传给每个解码器交叉注意力。在输出投影前加最终LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: 在玩具示例上跑前向

喂入6词元源和5词元目标。验证输出形状是`(5, vocab)`。无训练——本课程讲架构,不讲损失。

### Step 4: 换入RMSNorm + SwiGLU

用RMSNorm和SwiGLU替换LayerNorm和ReLU-FFN。确认形状仍匹配。这是2026现代化,一次函数替换。

## 实际应用

PyTorch/TF参考实现:`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但多数2026生产代码自己写块因为:

- Flash注意力机制在注意力内部调用,非通过`nn.MultiheadAttention`。
- GQA/MLA不在标准库参考中。
- RoPE、RMSNorm、SwiGLU不是PyTorch默认。

HF `transformers`有干净参考块你应该读:`modeling_llama.py`是典型2026仅解码器块。约500行,值得走一遍。

**编码器vs解码器vs编码器-解码器——何时选:**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入→结构化输出(翻译、摘要) | 编码器-解码器 | T5、BART、Whisper |

仅解码器赢得语言因为它扩展最干净且同时处理理解和生成。编码器-解码器在输入有清晰"源序列"身份时仍最佳(翻译、语音识别、结构化任务)。

## 产出成果

见`outputs/skill-transformer-block-reviewer.md`。技能对照2026默认审查新transformer块实现,标记缺失部分(pre-norm、RoPE、RMSNorm、GQA、FFN扩展比)。

## 练习题

1. **简单。**在`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`算encoder_block参数。通过实现块并`sum(p.numel() for p in block.parameters())`验证。
2. **中等。**从post-norm换到pre-norm。初始化两者并测12层堆叠后随机输入的激活范数。Post-norm激活应爆炸;pre-norm应保持有界。
3. **困难。**在玩具复制任务(复制`x`反转)实现4层编码器-解码器。训100步。报告损失。换入RMSNorm + SwiGLU + RoPE——损失降否?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Block | "一个transformer层" | 残差连接包裹的norm + attention + norm + FFN栈。 |
| Residual | "跳跃连接" | `x + f(x)`输出;使梯度流过深栈。 |
| Pre-norm | "先归一化,后计算" | 现代:`x + sublayer(LN(x))`。深层训练无需预热体操。 |
| RMSNorm | "不带均值的LayerNorm" | 除RMS;少一次运算,相同经验稳定性。 |
| SwiGLU | "大家都换的FFN" | `Swish(W1 x) ⊙ W3 x → W2`。语言模型困惑度胜ReLU/GELU。 |
| Cross-attention | "解码器看编码器的方式" | Q来自解码器,K/V来自编码器输出的MHA。 |
| FFN expansion | "中间MLP多宽" | 隐藏大小对d_model比,通常4(LayerNorm)或2.6(SwiGLU)。 |
| Bias-free | "去掉+b项" | 现代栈在linear层省偏置;微小困惑度改进,更小模型。 |

## 延伸阅读

- [Vaswani等(2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)——原始块spec。
- [Xiong等(2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)——为何深层pre-norm胜post-norm。
- [Zhang, Sennrich(2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)——RMSNorm。
- [Shazeer(2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)——SwiGLU论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)——典型2026仅解码器块。