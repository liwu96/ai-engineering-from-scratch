# 位置编码 — Sinusoidal、RoPE、ALiBi

> 注意力是排列不变的。"The cat sat on the mat"和"mat the on sat cat the"无位置信号产同样输出。三种算法修复——每配"位置"意味不同赌注。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力)、阶段7课程03(多头注意力)
**时间:** ~45分钟

## 问题背景

Scaled dot-product注意力顺序盲。注意力矩阵`softmax(Q K^T / √d) V`从两两相似度计算。乱`X`行,输出行同样方式乱。注意力内无东西关心位置。

这在词袋模型非bug。对语言、代码、音频、视频——顺序承载意义的任何东西——致命。

修复是某种方式将位置注入嵌入。三时代答案:

1. **绝对sinusoidal**(Vaswani 2017)。将位置`sin/cos`加到嵌入。简单、无学习、训长外推差。
2. **RoPE——旋转位置嵌入**(Su 2021)。按角度比例位置旋转Q和K向量。点积中直接编码*相对*位置。2026主导。
3. **ALiBi——线性偏置注意力**(Press 2022)。跳嵌入完全;基于距离向注意力分数加每头线性惩罚。出色长度外推。

到2026,基本每个前沿开源模型用RoPE:Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型用ALiBi或现代变体。绝对sinusoidal历史。

## 概念讲解

![Sinusoidal绝对vs RoPE旋转vs ALiBi距离偏置](../assets/positional-encoding.svg)

### 绝对sinusoidal

预计算固定矩阵`PE`形`(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

后`X' = X + PE[:N]`注意力前。每维是不同频率正弦。模型从相位pattern学读位置。`max_len`外失败:模型只看位置0-2047时没告诉位置2048发生什么。

### RoPE

旋转Q和K向量(非嵌入)。对维度对`(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 默认
```

用位置`pos_k`同样旋转keys。点积`q'_m · k'_n`变成仅`(m - n)`函数。即:**注意力分数只依赖相对距离**,尽管旋转基于绝对位置。漂亮技巧。

扩展RoPE:`base`可缩放(NTK-aware, YaRN, LongRoPE)外推更长上下文无重训。Llama 3这样从8K扩到128K上下文。

### ALiBi

跳嵌入技巧。直接偏置注意力分数:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

`m_h`是头特定斜率(如`1 / 2^(8·h/H)`)。近词元boost;远词元penalize。无训时成本。论文示长度外推胜sinusoidal匹配RoPE原训长。

### 2026选何

| 变体 | 外推 | 训成本 | 使用者 |
|---------|---------------|---------------|---------|
| 绝对sinusoidal | 差 | 免费 | 原transformer、早BERT |
| 学习绝对 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 配缩放好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | fine-tune阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 免费 | BLOOM、MPT、Baichuan |

RoPE胜出，因为它可以无缝嵌入注意力机制而不改变架构、编码相对位置，且`base`超参为长上下文微调提供了简洁的调节旋钮。

## 动手实践

### Step 1: sinusoidal编码

见`code/main.py`。4行计算:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

首注意力层前将此加嵌入矩阵。

### Step 2: RoPE应用于Q, K

RoPE原地操作Q和K。对每维对:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键:位置`m`Q和位置`n`K应用同样函数。点积每坐标对拾`cos((m-n)·θ_i)`因子。注意力免费学相对位置。

### Step 3: ALiBi斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # softmax前加注意力分数
```

softmax前将`bias[h]`加头`h`的`(seq_len, seq_len)`注意力分数矩阵。

### Step 4: 验证RoPE相对距离属性

取两随机向量`a, b`。按`(pos_a, pos_b)`旋转。后按`(pos_a + k, pos_b + k)`。两点积须浮点误差内匹配。该属性是RoPE要点——它不变绝对偏移、仅相对gap重要。

## 实际应用

PyTorch 2.5+在`torch.nn.functional`发RoPE utilities。多数生产代码用`flash_attn`或`xformers`其中RoPE在注意力kernel内应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026长上下文技巧:**

- **NTK-aware插值。**4K扩16K+时rescale `base`到`base * (scale_factor)^(d/(d-2))`。
- **YaRN。**长上下文保注意力熵更智插值。Llama 3.1 128K用。
- **LongRoPE。**微软2024方法用进化搜索选每维缩放因子。Phi-3-Long用。
- **位置插值+微调。**仅缩位置扩展因子并微调1-5B词元。惊人有效。

## 产出成果

见`outputs/skill-positional-encoding-picker.md`。技能为新模型给定目标上下文长度、外推需、训预算选编码策略。

## 练习题

1. **简单。**绘`max_len=512, d=128`sinusoidal `PE`矩阵热图。确认"维度索引增条纹变宽"pattern。
2. **中等。**实现NTK-aware RoPE缩放。长度256序列训微小LM、后配不配缩放测长度1024。测困惑度。
3. **困难。**同注意力模块实现ALiBi和RoPE。长度512copy任务训4层transformer。测时外推2048。比退化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 位置编码 | "告注意力顺序" | 加到嵌入或注意力编码位置任何信号。 |
| Sinusoidal | "原始那个" | 几何频率`sin/cos`加嵌入;不外推。 |
| RoPE | "旋转嵌入" | 位置依赖角度旋转Q, K;点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 注意力分数加`-m·|i-j|`;无嵌入需、外推好。 |
| base | "RoPE旋钮" | RoPE频率缩放器;推理增扩上下文。 |
| NTK-aware | "RoPE缩放技巧" | 上下文扩时rescale `base`使高频维不被挤。 |
| YaRN | "那个花哨" | 保注意力熵每维插值+外推。 |
| 外推 | "训长外工作" | 位置方案能否服训见`max_len`外正确输出? |

## 延伸阅读

- [Vaswani等(2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)——原始sinusoidal。
- [Su等(2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)——RoPE论文。
- [Press, Smith, Lewis(2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)——ALiBi。
- [Peng等(2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)——SOTA RoPE缩放。
- [Chen等(2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)——Meta Llama 2长上下文论文。
- [Ding等(2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)——微软方法Phi-3-Long用、Use It节引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)——每RoPE缩放方案(default、linear、dynamic、YaRN、LongRoPE、Llama-3)生产级实现。