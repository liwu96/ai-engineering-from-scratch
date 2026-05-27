# 命名实体识别

> 提取名字。处理模糊边界、嵌套实体和领域术语之前，这听起来很容易。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 02（BoW + TF-IDF），第5阶段 · 03（词嵌入）
**时间：** 约75分钟

## 问题背景

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（可能是）、US（GPE）。好的NER系统用正确类型提取所有这些。坏的系统遗漏iPhone，将水果Apple与公司Apple混淆，将"US"标记为PERSON。

NER是每个结构化提取流水线下的主力军。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的基础、法律合同提取。你从未完全看到它；你总是依赖它。

本课程从经典路径（基于规则、HMM、CRF）走到现代路径（BiLSTM-CRF，然后是Transformer）。每一步解决前一步的特定限制。模式就是教训。

## 概念讲解

**BIO标注**（或BILOU）将实体提取变成序列标注问题。用 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（任何实体外部）标注每个词元。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词元实体链：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解BIO的模型可以提取任意跨度。

架构演进：

- **基于规则。** 正则 + 地名表查找。已知实体上高精度，新实体零覆盖。
- **HMM。** 隐马尔可夫模型。给定标签的词发射概率，标签到标签的转移概率。Viterbi解码。在标注数据上训练。
- **CRF。** 条件随机场。类似HMM但是判别式，所以可以混合任意特征（词形、大写、相邻词）。2026年低资源部署的经典生产主力。
- **BiLSTM-CRF。** 神经特征而非手工制作。LSTM双向读取句子，CRF层强制执行一致的标签序列。
- **基于Transformer。** 用词元分类头微调BERT。最佳准确性。最多计算。

## 动手实践

### 步骤1：BIO标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤2：手工特征

对于经典（非神经）NER，特征是游戏。有用的：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词高信号。

### 步骤3：简单规则+字典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产地名表有从Wikipedia和DBpedia抓取的数百万条目。覆盖良好。消歧（公司Apple vs 水果Apple）糟糕。这就是统计模型获胜的原因。

### 步骤4：CRF步骤（草图，非完整实现）

50行内完整CRF没有概率理论基础并不启发人。使用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是L1和L2正则化。`all_possible_transitions=True` 让模型学习到非法序列（例如 `I-ORG` 在 `O` 后）不太可能，这就是CRF如何强制执行BIO一致性而不需要你写约束。

### 步骤5：BiLSTM-CRF添加什么

特征变成学习的。输入：词嵌入（GloVe或fastText）。LSTM从左到右和从右到左读取。拼接的隐藏状态通过CRF输出层。CRF仍然强制执行标签序列一致性；LSTM用手工特征替换为学习的特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于CRF层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工CRF的增益可测量但比你预期的要小，除非你有数万标注句子。

## 实际应用

spaCy提供开箱即用的生产级NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标记为 `ORG` 而不是 `PRODUCT` — spaCy的小模型对产品实体覆盖弱。大模型（`en_core_web_lg`）做得更好。Transformer模型（`en_core_web_trf`）做得更好。

Hugging Face用于基于BERT的NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的B-X、I-X词元合并成跨度。没有它，你得到词级标签并必须自己合并。

### 基于LLM的NER（2026年选项）

零样本和少样本LLM NER现在在许多领域与微调模型竞争，且在标注数据稀缺时显著更好。

- **零样本提示。** 给LLM实体类型列表和示例模式。要求JSON输出。开箱即用；在新领域上准确性中等。
- **ZeroTuneBio风格提示。** 将任务分解为候选提取→含义解释→判断→重新检查。多阶段提示（非一次性）在生物医学NER上显著提升准确性。同样模式适用于法律、金融和科学领域。
- **带RAG的动态提示。** 为每次推理调用从小的标注种子集中检索最相似的标注示例；动态构建少样本提示。在2026年基准测试中，这将GPT-4生物医学NER F1比静态提示提升11-12%。
- **每实体类型分解。** 对于长文档，一次提取所有实体类型的单个调用随着长度增长而失去召回率。每实体类型运行一次提取。更高推理成本，显著提升准确性。这是临床笔记和法律合同的标准模式。

2026年生产推荐：在收集训练数据前从零样本LLM基线开始。通常F1足够好，你不需要微调。

### 经典NER仍然获胜的地方

即使有LLM可用，经典NER在以下情况获胜：

- 延迟预算低于50毫秒。
- 你有数千标注示例且需要98%+ F1。
- 领域有稳定本体，预训练CRF或BiLSTM很好地转移。
- 监管约束要求本地、非生成式模型。

### 它崩溃的地方

- **领域偏移。** CoNLL训练的NER在法律合同上表现比地名表还差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是ORG和FACILITY。标准BIO无法表示重叠跨度。你需要嵌套NER（多遍或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." 词级模型有时分割这个。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学NER标签如DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型不知道。Scispacy和BioBERT是那里的起点。

## 产出成果

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: 为给定提取任务选择正确的NER方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（领域、标签集、语言、延迟、数据量），输出：

1. 方法。基于规则+地名表、CRF、BiLSTM-CRF或Transformer微调。
2. 起始模型。命名它（spaCy模型ID、Hugging Face检查点ID或"custom, trained from scratch"）。
3. 标注策略。BIO、BILOU或基于跨度。一句话理由。
4. 评估。使用 `seqeval`。始终报告实体级F1（不是词元级）。

拒绝在少于500标注示例的情况下推荐微调Transformer，除非用户已有预训练领域模型。标记嵌套实体为需要基于跨度或多遍模型。如果用户提到"生产规模"且标签与CoNLL-2003相同，要求地名表审计。
```

## 练习题

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆）并在10个句子上验证往返一致性。
2. **中等。** 在CoNLL-2003英语NER数据集上训练上面的sklearn-crfsuite CRF。使用 `seqeval` 报告每实体F1。典型结果：~84 F1。
3. **困难。** 在领域特定NER数据集（医学、法律或金融）上微调 `distilbert-base-cased`。与spaCy小模型对比。记录数据泄漏检查并写下什么让你惊讶。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| NER | 提取名字 | 用类型（PERSON、ORG、GPE、DATE等）标注词元跨度。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 外部。 |
| BILOU | 更好的BIO | 添加 `L-X`（最后）、`U-X`（单元）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 建模标签间转移，不只是发射。强制执行有效序列。 |
| Nested NER | 重叠实体 | 一个跨度与另一个的子跨度是不同实体。BIO无法表达这个。 |
| Entity-level F1 | 正确NER指标 | 预测跨度必须与真实跨度完全匹配。词级F1高估准确性。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF论文。经典。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入成为标准的词元分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。始终使用它。
