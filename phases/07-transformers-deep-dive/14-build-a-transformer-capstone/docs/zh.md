# 从零构建Transformer——毕业项目

> 十三课程。一模型。无捷径。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程01到13。勿跳。
**时间:** ~120分钟

## 问题背景

你读每论文。实现注意力、多头拆分、位置编码、编码器和解码器块、BERT和GPT损失、MoE、KV cache。现让它们在真实任务协同工作。

毕业项目:端到端训小仅解码器transformer于字符级语言建模任务。读莎士比亚。生成新莎士比亚。足够小可在笔记本10分钟内训。足够正确换更大数据集和更长训练得真实LM。

此课程"nanoGPT"。非原创——Karpathy 2023 nanoGPT教程是每个学生至少写一次参考实现。我们借形状并围绕已覆盖内容重整。

## 概念讲解

![Transformer-from-scratch框图](../assets/capstone.svg)

架构,标注:

```
输入词元 (B, N)
   │
   ▼
词元嵌入 + 位置嵌入  ◀── 课程04(RoPE选项)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 课程05
│  MultiHeadAttention(因果)         │  ◀── 课程03 + 07(因果掩码)
│  残差                             │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 课程05
│  残差                             │
└────────────────────────────────── ┘
   │
   ▼
最终RMSNorm
   │
   ▼
lm_head(与词元嵌入绑定)
   │
   ▼
logits (B, N, V)
   │
   ▼
移一交叉熵                           ◀── 课程07
```

### 我们产出

- `GPTConfig`——一处配置所有超参。
- `MultiHeadAttention`——因果、批、可选Flash风格路径(PyTorch `scaled_dot_product_attention`)。
- `SwiGLUFFN`——现代FFN。
- `Block`——预归一化、残差包装注意力+FFN。
- `GPT`——嵌入、堆块、LM头、generate()。
- AdamW、余弦LR、梯度裁剪训练循环。
- Shakespeare文本字符级分词器。

### 我们不产出

- RoPE——课程04概念实现。此处用学习位置嵌入简化。练习要求换入RoPE。
- KV cache生成——每生成步重算全前缀注意力。更慢但更简单。练习要求加KV cache。
- Flash Attention——PyTorch 2.0+若输入匹配自动分发;我们用`F.scaled_dot_product_attention`。
- MoE——每块单FFN。课程11见MoE。

### 目标度量

Mac M2笔记本,4层、4头、d_model=128 GPT训2,000步于`tinyshakespeare.txt`:

- 训练损失约6分钟从~4.2(随机)收敛到~1.5。
- 采样输出看莎士比亚形:古词、换行、正确名如"ROMEO:"现。
- 验证损失(留出文本最后10%)紧随训练损失;此规模/预算不过拟合。

## 动手实践

此课用PyTorch。装`torch`(CPU版够)。见`code/main.py`。脚本处理:

- 缺失下载`tinyshakespeare.txt`(或读本地)。
- 字节级字符分词器。
- 训/验拆分90/10。
- 支持硬件bf16 autocast训练循环。
- 训练完采样。

### Step 1: 数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65唯一字符。小词表。适4字节vocab_size。无BPE,无分词器折腾。

### Step 2: 模型

见`code/main.py`。块是课程05教科书——预归一化、RMSNorm、SwiGLU、因果MHA。4/4/128参数数:~800K。

### Step 3: 训练循环

取随机批长256词元窗口。前向。移一交叉熵。反向。AdamW步。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Step 4: 采样

给定提示词,重复前向、从top-p logits采样、追加、继续。500词元后停。

### Step 5: 读输出

2,000步后:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

非莎士比亚。但莎士比亚形。~800K参数和笔记本6分钟明显胜。

## 实际应用

此毕业项目是参考架构。三扩展产真实:

1. **换分词器。**用BPE(如`tiktoken.get_encoding("cl100k_base")`)。词表大小从65跳到~50,000。模型容量需缩放补。
2. **更大语料训。**用`OpenWebText`或`fineweb-edu`(HuggingFace)。单A100 10B词元~24小时训125M参数GPT。
3. **加RoPE + KV cache + Flash Attention。**下练习带你每步。

最终成125M参数GPT生成流利英语。非前沿模型。但同代码路径——仅更大——是Karpathy、EleutherAI和Allen Institute 2026训研究检查点所用。

## 产出成果

见`outputs/skill-transformer-review.md`。技能审查从零transformer实现跨所有13前课正确性。

## 练习题

1. **简单。**运行`code/main.py`。验证训模型最终步验证损失低于2.0。改`max_steps`从2,000到5,000——验证损失继续改进否?
2. **中等。**换学习位置嵌入为RoPE。`MultiHeadAttention`内Q和K应用旋转。训并验证验证损失至少一样低。
3. **中等。**采样循环实现KV cache。有无cache生成500词元。笔记本墙钟应改进5–20×。
4. **困难。**加第二头预测下一+一词元(MTP——DeepSeek-V3多词元预测)。联合训。有助否?
5. **困难。**换每块单FFN为4专家MoE。路由器+top-2路由。匹配激活参数看验证损失变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| nanoGPT | "Karpathy教程仓库" | 最小仅解码器transformer训练代码,~300 LOC;规范参考。 |
| tinyshakespeare | "标准玩具语料" | ~1.1 MB文本;2015来每字符-LM教程用它。 |
| 绑定嵌入 | "共享输入/输出矩阵" | LM头权重=词元嵌入矩阵转置;省参数,改进质量。 |
| bf16 autocast | "训练精度技巧" | bf16运行前向/反向,fp32保优化器状态;2021来标准。 |
| 梯度裁剪 | "止尖峰" | 全局梯度范数限1.0;防训爆炸。 |
| 余弦LR调度 | "2020+默认" | LR线性升温(预热)后余弦形衰减到峰值10%。 |
| MFU | "模型FLOP利用率" | 实现FLOPs / 理论峰值;2026 40%稠密,30% MoE强。 |
| 验证损失 | "留出损失" | 模型从未见数据交叉熵;过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)——经典标注实现。