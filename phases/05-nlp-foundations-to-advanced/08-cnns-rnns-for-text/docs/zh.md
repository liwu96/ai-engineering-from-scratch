# 用于文本的CNN和RNN

> 卷积学习n-gram。循环网络记忆。两者都被注意力取代。两者在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**前置要求：** 第3阶段 · 11（PyTorch简介），第5阶段 · 03（词嵌入），第4阶段 · 02（从零实现卷积）
**时间：** 约75分钟

## 问题背景

TF-IDF和Word2Vec产生忽略词序的平面向量。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时携带信号。

两个架构族在Transformer到来之前填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用1D卷积。宽度为3的过滤器是可学习的二元词组检测器：跨越三个词并输出分数。堆叠不同宽度（2、3、4、5）检测多尺度模式。最大池化为固定大小表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个词元，维护向前携带信息的隐藏状态。顺序、记忆承载、灵活输入长度。从2014年到2017年主导序列建模，然后注意力发生。

本课程构建两者，然后指出促使注意力出现的失效。

## 概念讲解

**TextCNN**（Kim，2014）。词元被嵌入。宽度`k`的1D卷积在连续的`k`-gram嵌入上滑动过滤器，产生特征图。全局最大池化在图上挑选最强激活。拼接来自多个滤波器宽度的最大池化输出。送入分类头。

为什么有效。过滤器是可学习的n-gram。最大池化位置不变，所以"not good"在评论开头或中间触发相同特征。三个滤波器宽度，每个100个滤波器，给你300个学习的n-gram检测器。训练是并行的；没有顺序依赖。

**RNN。** 每个时间步`t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享 `W`、`U`、`b`。时间`T`的隐藏状态是整个前缀的摘要。对于分类，在 `h_1 ... h_T` 上池化（最大、平均或最后）。

简单RNN遭受梯度消失。**LSTM**添加门决定遗忘什么、存储什么、输出什么，稳定长序列梯度。**GRU**简化LSTM到两个门；参数更少，性能相似。

**双向RNN**正向和反向各运行一个RNN，拼接隐藏状态。每个词元的表示看到左右上下文。对标注任务至关重要。

## 动手实践

### 步骤1：PyTorch中的TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化输出无论输入长度都是固定大小。

### 步骤2：LSTM分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

在序列上最大池化，不是最后状态池化。对于分类，最大池化通常优于取最后隐藏状态，因为长序列末尾的信息倾向于主导最后状态。

### 步骤3：梯度消失演示（直觉）

没有门的简单RNN无法学习长程依赖。考虑玩具任务：预测词元`A`是否出现在序列中的任何位置。如果`A`在位置1且序列是100个词元长，从损失回来的梯度必须通过循环权重的99次乘法反向传播。如果权重小于1，梯度消失。如果大于1，它爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 在weight=0.9超过100步时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第100步到第1步的梯度实际上为零。
```

LSTM用**细胞状态**修复，通过网络只进行加法交互（遗忘门乘法缩放，但梯度仍沿"高速公路"流动）。GRU用更少参数做类似事情。两者给你稳定训练通过100+步序列。

### 步骤4：为什么这仍然不够

三个问题即使在LSTM后仍然存在。

1. **顺序瓶颈。** 在1000步序列上训练RNN需要1000步串行前向/后向。无法跨时间并行。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器只看到编码器的最后隐藏状态，压缩在整个输入上。长输入丢失细节。第09课直接涵盖这个。
3. **远程依赖准确性天花板。** LSTM胜过简单RNN，但仍难以在200+步上传播特定信息。

注意力解决所有三个。Transformer完全放弃循环。第10课是转折点。

## 实际应用

PyTorch的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face提供预训练嵌入作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

使用-当-它适合约束清单。

- **边缘/设备推理。** 带GloVe嵌入的TextCNN比Transformer小10-100倍。如果你的部署目标是手机，这是你的栈。
- **流式/在线分类。** RNN一次处理一个词元；Transformer需要完整序列。对于实时传入文本，LSTM仍然获胜。
- **基线小模型。** 在新任务上快速迭代。在CPU上5分钟训练TextCNN。
- **有限数据的序列标注。** BiLSTM-CRF（第06课）对于1k-10k标注句子仍是生产级NER架构。

其他一切都交给Transformer。

## 产出成果

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 为给定约束集选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、Transformer微调，或"使用预训练Transformer作为冻结编码器+小头"。
2. 嵌入输入：随机初始化、GloVe/fastText冻结，或上下文化Transformer嵌入。
3. 5行训练配方：优化器、学习率、批量大小、轮数、正则化。
4. 一个监控信号。对于RNN/CNN模型：注意力机制缺失意味着它们错过远程依赖；检查每长度准确性。对于Transformer：如果学习率太高，微调崩溃；检查训练损失。

拒绝在数据少于约500标注示例时推荐微调Transformer，除非展示TextCNN/BiLSTM基线已停滞。标记边缘部署为需要架构优先于一切。
```

## 练习题

1. **简单。** 在3类玩具数据集上训练TextCNN（你自己发明数据）。验证滤波器宽度（2、3、4）在平均F1上优于单一宽度（3）。
2. **中等。** 为LSTM分类器实现最大池化、平均池化和最后状态池化。在小数据集上比较；记录哪个池化获胜并假设原因。
3. **困难。** 构建BiLSTM-CRF NER标注器（结合第06课和本课）。在CoNLL-2003上训练。与第06课仅CRF基线和BERT微调对比。报告训练时间、内存和F1。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| TextCNN | 文本CNN | 词嵌入上的1D卷积栈，带全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控RNN | 添加输入/遗忘/输出门+细胞状态。稳定训练通过长序列。 |
| GRU | 简单LSTM | 两个门代替三个。相似准确性，更少参数。 |
| Bidirectional | 双向 | 前向+后向RNN拼接。每个词元看到上下文两侧。 |
| Vanishing gradient | 训练信号消失 | 简单RNN中重复乘以<1权重使早期步梯度实际上为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN论文。八页。可读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM论文。意外地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让LSTM对每个人可及的图表。
