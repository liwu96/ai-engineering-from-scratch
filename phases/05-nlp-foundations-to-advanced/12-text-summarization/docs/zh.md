# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者的意思。不同任务，不同陷阱。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 02（BoW + TF-IDF），第5阶段 · 11（机器翻译）
**时间：** 约75分钟

## 问题背景

一篇2000字的新闻文章进入你的feed。你需要120字来捕捉它。你可以从文章中挑选三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都叫摘要。它们是完全不同的问题。

抽取式摘要是排序问题。给每句话打分，返回前`k`个。输出总是语法正确的，因为是原文提取的。风险是错过分布在文章中的内容。

生成式摘要是生成问题。Transformer以输入为条件产生新文本。输出流畅且压缩，但可能幻觉源中没有的事实。风险是自信的编造。

本课程构建两者，以及每个拥有的失效模式。

## 概念讲解

![抽取式TextRank vs 生成式Transformer](../assets/summarization.svg)

**抽取式。** 将文章视为图，节点是句子，边是相似性。在图上运行PageRank（或类似），通过与其他一切连接的程度给句子打分。得分最高的句子就是摘要。规范实现是**TextRank**（Mihalcea和Tarau，2004）。

**生成式。** 在文档-摘要对上微调Transformer编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档，通过交叉注意力逐词元生成摘要。Pegasus特别使用间隙句子预训练目标，使其无需太多微调就擅长摘要。

用**ROUGE**（面向召回的摘要评估替补）评估。ROUGE-1和ROUGE-2评分unigram和bigram重叠。ROUGE-评分最长公共子序列。越高越好，但40 ROUGE-L是"好"，50是"卓越"。每篇论文报告三个。使用 `rouge-score` 包。

## 动手实践

### 步骤1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

值得命名的两件事。相似度函数使用对数归一化词重叠，这是原始TextRank变体。TF-IDF向量的余弦也有效。阻尼因子0.85和迭代次数是PageRank默认值。

### 步骤2：用BART生成式

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(长新闻文章文本)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN在CNN/DailyMail语料库上微调。开箱即用产生新闻风格摘要。对于其他领域（科学论文、对话、法律），使用对应的Pegasus检查点或在目标数据上微调。

### 步骤3：ROUGE评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。没有它，"running" 和 "run" 算作不同词，ROUGE低估。

### 超越ROUGE（2026年摘要评估）

ROUGE二十年来主导摘要指标，2026年单独使用已不足。大规模NLG论文元分析显示：

- **BERTScore**（上下文化嵌入相似性）通过2023年获得发展，现在大多数摘要论文与ROUGE一起报告。
- **BARTScore** 将评估视为生成：通过预训练BART给定源赋予摘要的可能性来评分。
- **MoverScore**（上下文化嵌入上的地球移动距离）在2025年摘要基准中达到顶峰，因为它比ROUGE更好地捕获语义重叠。
- **FactCC** 和 **基于QA的忠实性** 在2021-2023年常见，现在常被 **G-Eval** 取代（带思维链推理的GPT-4提示链，评分连贯性、一致性、流畅性、相关性）。
- **G-Eval** 和类似LLM裁判方法在设计良好的评分标准时与人类判断匹配~80%。

生产推荐：为遗留比较报告ROUGE-L，为语义重叠报告BERTScore，为连贯性和事实性报告G-Eval。在50-100人工标注摘要上校准。

### 步骤4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要幻觉风险低得多，因为输出是源中逐字提取的，尽管如果源句子被断章取义、过时或顺序错误引用，仍可能误导。这是生产系统对合规相关内容仍偏好抽取式的主要原因。

要命名的幻觉类型：

- **实体替换。** 源说 "John Smith." 摘要说 "John Brown."
- **数字漂移。** 源说 "25,000." 摘要说 "25 million."
- **极性翻转。** 源说 "rejected the offer." 摘要说 "accepted the offer."
- **事实发明。** 源未提及CEO。摘要说CEO批准。

有效的评估方法：

- **FactCC。** 在源句和摘要句之间训练的二元分类器。预测事实/非事实。
- **基于QA的忠实性。** 问QA模型答案在源中的问题。如果摘要支持不同答案，标记。
- **实体级F1。** 比较源与摘要中的命名实体。仅在摘要中出现的实体可疑。

对于任何用户可见且事实性重要的地方（新闻、医疗、法律、金融），抽取式是更安全的默认。生成式需要在循环中进行事实性检查。

## 实际应用

2026年栈：

| 用例 | 推荐 |
|------|------|
| 新闻，3-5句摘要，英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的T5 |
| 多文档，长篇 | 任何32k+上下文的LLM，提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，低幻觉风险 | TextRank 或 `sumy` 的LSA / LexRank |

当计算不受限时，截至2026年长上下文LLM常优于专用模型。权衡是成本和可复现性；专用模型给出更一致的输出。

## 产出成果

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，命名库，事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。抽取式或生成式。一句话解释原因。
2. 起始模型/库。命名它。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或LLM提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干的rouge-score）。生成式加事实性检查。
4. 要探测的一个失效模式。实体替换是生成式新闻摘要中最常见的；标记源实体不出现在摘要中的样本。

拒绝医疗、法律、金融或受监管内容的生成式摘要，除非有事实性门。标记超过模型上下文窗口的输入为需要分块map-reduce摘要（不只是截断）。
```

## 练习题

1. **简单。** 在5篇新闻文章上运行TextRank。将top-3句子与参考摘要对比。测量ROUGE-L。你应该在CNN/DailyMail风格文章上看到30-45 ROUGE-L。
2. **中等。** 实现实体级事实性：从源和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率和摘要实体对源的精确率。高精确率和低召回率意味着安全但简洁；低精确率意味着幻觉实体。
3. **困难。** 在50篇CNN/DailyMail文章上对比BART-large-CNN与LLM（Claude或GPT-4）。报告ROUGE-L、事实性（按实体F1）和每次摘要成本。记录每个获胜的地方。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Extractive | 挑选句子 | 从源返回逐字句子。从不幻觉。 |
| Abstractive | 改写 | 以源为条件生成新文本。可能幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的n-gram / LCS重叠。 |
| TextRank | 基于图的抽取式 | 句子相似图上的PageRank。 |
| Factuality | 是否正确 | 摘要声明是否受源支持。 |
| Hallucination | 编造内容 | 摘要中源不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus和间隙句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性全景论文。
