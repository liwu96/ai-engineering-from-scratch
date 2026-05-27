# 自然语言推理——文本蕴含

> "t蕴含h"意味人读t会结论h真。自然语言推理是预测蕴含/矛盾/中性任务。表面无聊,生产承重。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程05(情感分析)、阶段5课程13(问答)
**时间:** ~60分钟

## 问题背景

你构建摘要器。它产摘要。你怎么知摘要不含幻觉?

你构建聊天机器人。它答"是。"你怎么知答案被检索段落支持?

你需按主题分类10,000新闻文章。无训练标签。能复用模型吗?

三问题都归约到自然语言推理。自然语言推理问:给定前提`t`和假设`h`,`h`被`t`蕴含、矛盾或中性(无关)?

- **幻觉检查:**`t`=源文档,`h`=摘要主张。非蕴含=幻觉。
- **锚定问答:**`t`=检索段落,`h`=生成答案。非蕴含=捏造。
- **零样本分类:**`t`=文档,`h`=言化标签("这是体育")。蕴含=预测标签。

一任务,三生产用。这是为什么每个RAG评估框架发货自然语言推理模型在底层。

## 概念讲解

![自然语言推理:三路分类,前提vs假设](../assets/nli.svg)

**三标签。**

- **蕴含。**`t`→`h`。"猫在垫上"蕴含"有猫"。
- **矛盾。**`t`→¬`h`。"猫在垫上"矛盾"无猫"。
- **中性。** 无推理。"猫在垫上"对"猫饿"中性。

**非逻辑蕴含。** 自然语言推理是*自然*语言推理——典型人读者推断什么,非严逻辑。"John遛狗"自然语言推理蕴含"John有狗",但严一阶逻辑只在你公理化所有权时承认。

**数据集。**

- **SNLI**(2015)。570k人注对,图像标题作前提。窄域。
- **MultiNLI**(2017)。433k跨10类对。2026标准训练语料。
- **ANLI**(2019)。对抗自然语言推理。人写例专门破现有模型。更难。
- **DocNLI, ConTRoL**(2020–21)。文档长前提。测多跳和远推理。

**架构。** Transformer编码器(BERT、RoBERTa、DeBERTa)读`[CLS]前提[SEP]假设[SEP]`。`[CLS]`表示喂3路softmax。MNLI训,保留基准评估,分布内对90%+准确率。

**通过自然语言推理零样本。** 给定文档和候选标签,转每标签为假设("本文是体育")。算每蕴含概率。选最大。这是Hugging Face`zero-shot-classification`管道背后机制。

## 动手实践

### Step 1:跑预训练自然语言推理模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产自然语言推理,`facebook/bart-large-mnli`和`microsoft/deberta-v3-large-mnli`是开源默认。DeBERTa-v3顶排行榜。

### Step 2:零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

模板默认"This example is about {label}."。用`hypothesis_template`定制。无训练数据需。无微调。开箱工作。

### Step 3:RAG忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是RAGAS忠实度核心。分裂生成答案为原子主张。每主张对检索上下文查。报告蕴含分数。

### Step 4:手卷自然语言推理分类器(概念)

见`code/main.py`仅stdlib玩具:前提和假设通过词汇重叠+否定检测比。不竞争Transformer模型——但显任务形状:两文本入,3路标签出,损失=`{蕴含,矛盾,中性}`上交叉熵。

## 陷阱

- **仅假设捷径。** SNLI上模型可仅从假设预测标签~60%因"不"、"无"、"永不"关联矛盾。检测标签泄漏强基线。
- **词汇重叠启发。** 子序列启发("每子序列蕴含")过SNLI但失败HANS/ANLI。用对抗基准。
- **文档长退化。** 单句自然语言推理模型在文档长前提降20+F1。长上下文用DocNLI训模型。
- **零样本模板敏感。** "This example is about {label}" vs "{label}" vs "The topic is {label}"可摆准确率10+点。调模板。
- **领域不匹配。** MNLI训通用英文。法律、医疗和科学文本需领域特定自然语言推理模型(如SciNLI、MedNLI)。

## 实际应用

2026栈:

| 用例 | 模型 |
|------|------|
| 通用自然语言推理 | `microsoft/deberta-v3-large-mnli` |
| 快/边缘 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类(轻量) | `facebook/bart-large-mnli` |
| 文档级自然语言推理 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG幻觉检测 | RAGAS/DeepEval内自然语言推理层 |

2026元模式:自然语言推理是文本理解万能胶。每当需"A支持B吗?"或"A矛盾B吗?"——先自然语言推理再另一大语言模型调用。

## 产出成果

存`outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: 为分类/忠实度/零样本任务选自然语言推理模型、标签模板和评估设置。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定用例(忠实度检查、零样本分类、文档级推理),输出:

1. 模型。命名自然语言推理检查点。理由绑领域、长度、语言。
2. 模板(如零样本)。言化模式。例。
3. 阈值。蕴含截止决策规则。基于校准理由。
4. 评估。保留标签集准确率、仅假设基线、对抗子集。

拒绝发货零样本分类无100例标签健全检查。拒绝文档长前提用句子级自然语言推理模型。标记任何自然语言推理解幻觉声明——它减少;不消除。
```

## 练习题

1. **简单。** 20手编(前提、假设、标签)覆盖三类三元组跑`facebook/bart-large-mnli`。测准确率。加对抗"子序列启发"陷阱("我没吃蛋糕" vs "我吃了蛋糕")看是否破。
2. **中等。** 比100 AG News标题零样本模板`"This text is about {label}"`vs`"The topic is {label}"`vs`"{label}"`。报准确率摆。
3. **困难。** 构RAG忠实度检查器:原子主张分解+每主张自然语言推理。50金上下文RAG生成答案评估。测假正和假负率vs手标签。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 自然语言推理 | Natural Language Inference | 前提-假设关系3路分类。 |
| RTE | Recognizing Textual Entailment | 自然语言推理旧名;同任务。 |
| 蕴含 | "t暗示h" | 典型读者给定t会结论h真。 |
| 矛盾 | "t排除h" | 典型读者给定t会结论h假。 |
| 中性 | "未定" | t到h无推理。 |
| 零样本分类 | 自然语言推理作分类器 | 言化标签为假设,选最大蕴含。 |
| 忠实度 | 答案支持吗? | (检索上下文,生成答案)上自然语言推理。 |

## 延伸阅读

- [Bowman等(2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326)——SNLI。
- [Williams, Nangia, Bowman(2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426)——MultiNLI。
- [Nie等(2019). Adversarial NLI](https://arxiv.org/abs/1910.14599)——ANLI基准。
- [Yin, Hay, Roth(2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161)——自然语言推理作分类器。
- [He等(2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)——2026自然语言推理主力。