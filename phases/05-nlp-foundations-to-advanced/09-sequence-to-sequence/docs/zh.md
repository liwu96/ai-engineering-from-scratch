# 序列到序列模型

> 两个RNN假装成翻译器。它们遇到的瓶颈是注意力存在的原因。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 08（用于文本的CNN + RNN），第3阶段 · 11（PyTorch简介）
**时间：** 约75分钟

## 问题背景

分类将可变长度序列映射到单个标签。翻译将可变长度序列映射到另一个可变长度序列。输入和输出使用不同词汇表，可能不同语言，不保证长度相同。

seq2seq架构（Sutskever、Vinyals、Le，2014）用一个故意简单的配方破解了这个。两个RNN。一个读取源句子并产生固定大小上下文向量。另一个读取该向量并逐词元生成目标句子。第08课中你为分类写的相同代码，以不同方式粘合在一起。

这值得研究有两个原因。首先，上下文向量瓶颈是NLP中教学上最有用的失效。它促使注意力出现和Transformer擅长的一切。其次，训练配方（教师强制、计划采样、推理时束搜索）仍适用于每个现代生成系统，包括LLM。

## 概念讲解

**编码器。** 读取源句子的RNN。其最终隐藏状态是**上下文向量** — 整个输入的固定大小摘要。据说没有丢失，除了源。

**解码器。** 另一个从上下文向量初始化的RNN。每一步取先前生成的词元作为输入，在目标词汇表上产生分布。采样或argmax挑选下一个词元。反馈回去。重复直到产生`<EOS>`词元或达到最大长度。

**训练：** 每个解码器步的交叉熵损失，在序列上求和。通过两个网络的标准时间反向传播。

**教师强制。** 训练期间，解码器在时间`t`的输入是位置`t-1`的*真实*词元，不是解码器自己的先前预测。这稳定训练；没有它，早期错误级联，模型永远学不到。推理时，你必须使用模型自己的预测，所以总是存在训练/推理分布差距。那个差距叫**暴露偏见**。

**瓶颈。** 编码器学到的关于源的所有内容必须挤压到那个上下文向量中。长句子丢失细节。罕见词模糊。重新排序（chat noir vs black cat）必须记忆，不能计算。

注意力（第10课）通过让解码器查看*每个*编码器隐藏状态，不只是最后一个，直接修复这个。这就是全部。

## 动手实践

### 步骤1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 形状 `[batch, seq_len, hidden_dim]` — 每个输入位置一个隐藏状态。`hidden` 形状 `[1, batch, hidden_dim]` — 最后一步。第08课说"为分类池化输出"。这里我们保持最后隐藏状态作为上下文向量，忽略每步输出。

### 步骤2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次一步调用。输入：一批单个词元和当前隐藏状态。输出：下一个词元的词汇表logits和更新后的隐藏状态。

### 步骤3：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得命名的参数。`ignore_index=0` 跳过填充词元的损失。`teacher_forcing_ratio` 是在每个步使用真实词元与模型预测的概率。从1.0（完全教师强制）开始，在训练过程中退火到~0.5以缩小暴露偏见差距。

### 步骤4：推理循环（贪婪）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪婪解码在每个步挑选最高概率词元。一旦承诺一个词元，就无法收回。**束搜索**保持前`k`个部分序列活着，并在最后挑选得分最高的完整序列。束宽3-5是标准。

### 步骤5：瓶颈演示

在玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确性。

```
seq_len=5   复制准确性：98%
seq_len=10  复制准确性：91%
seq_len=20  复制准确性：62%
seq_len=40  复制准确性：23%
```

单个GRU隐藏状态无法无损记忆40词元输入。信息在每个编码器步都存在，但解码器只看到最后状态。注意力直接修复这个。

## 实际应用

PyTorch有 `nn.Transformer` 和基于 `nn.LSTM` 的seq2seq模板。Hugging Face的 `transformers` 库提供完整编码器-解码器模型（BART、T5、mBART、NLLB），在数十亿词元上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器放弃RNN用于Transformer。高层形状（编码器、解码器、逐词元生成）与2014年seq2seq论文相同。每个块内的机制不同。

### 何时仍选择基于RNN的seq2seq

几乎从不，对于新项目。特定例外：

- 流式翻译，你一次消费一个输入词元，有界内存。
- 设备文本生成，Transformer内存成本过高。
- 教学。理解编码器-解码器瓶颈是理解为什么Transformer获胜的最快路径。

### 暴露偏见及其缓解

- **计划采样。** 在训练过程中退火教师强制比率，使模型学习从自己的错误中恢复。
- **最小风险训练。** 在句子级BLEU分数上训练，而不是词级交叉熵。更接近你实际想要的。
- **强化学习微调。** 用指标奖励序列生成器。现代LLM RLHF使用。

三者仍适用于基于Transformer的生成。

## 产出成果

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计序列到序列流水线。
phase: 5
lesson: 09
---

给定任务（翻译、摘要、改写、问题重写），输出：

1. 架构。预训练Transformer编码器-解码器（BART、T5、mBART、NLLB）是默认。仅对特定约束使用基于RNN的seq2seq。
2. 起始检查点。命名它（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。将检查点与任务和语言覆盖匹配。
3. 解码策略。确定性输出用贪婪，质量用束搜索（宽4-5），多样性用温度采样。一句话理由。
4. 发货前要验证的一个失效模式。暴露偏见表现为较长输出生成漂移；在90百分位长度采样20个输出并目视检查。

拒绝在少于一百万并行示例的情况下推荐从头训练seq2seq。标记任何对用户内容使用贪婪解码的流水线为脆弱（贪婪重复和循环）。
```

## 练习题

1. **简单。** 实现玩具复制任务。在输入-输出对（目标等于源）上训练GRU seq2seq。测量长度5、10、20的准确性。重现瓶颈。
2. **中等。** 添加束宽3的束搜索解码。在小并行语料库上与贪婪对比测量BLEU。记录束搜索获胜的地方（通常是最后词元）和没有区别的地方。
3. **困难。** 在10k对改写数据集上微调 `facebook/bart-base`。在留出输入上比较微调模型的束-4输出与基础模型。报告BLEU并挑选10个定性示例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Encoder | 输入RNN | 读取源。产生每步隐藏状态和最终上下文向量。 |
| Decoder | 输出RNN | 从上下文向量初始化。逐词元生成目标。 |
| Context vector | 摘要 | 最终编码器隐藏状态。固定大小。注意力解决的瓶颈。 |
| Teacher forcing | 使用真实词元 | 训练时提供真实前一个词元。稳定学习。 |
| Exposure bias | 训练/测试差距 | 在真实词元上训练的模型从未练习从自己的错误中恢复。 |
| Beam search | 更好的解码 | 每步保持前k个部分序列活着，而不是贪婪承诺。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始seq2seq论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入GRU和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的seq2seq + 注意力代码。
