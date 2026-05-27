# 问答系统

> 三种系统塑造了现代问答。抽取式找答案片段。检索增强式将其锚定在文档中。生成式产出答案。每个现代AI助手都是这三者的混合。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程11(机器翻译)、阶段5课程10(注意力机制)
**时间:** ~75分钟

## 问题背景

用户输入"第一代iPhone什么时候发布?"并期待"2007年6月29日"。不是"苹果历史悠久,变化多端。"不是孤立的"2007"没有句子。直接、有据、正确的答案。

过去十年三种架构主导问答:

- **抽取式问答。** 给定问题和已知包含答案的段落,找段落中答案片段的起止索引。SQuAD是标准基准。
- **开放域问答。** 段落未给定。先检索相关段落,再抽取或生成答案。这是当今每个RAG管道的基础。
- **生成式/闭卷问答。** 大语言模型从参数记忆回答。无检索。推理最快,事实可靠性最低。

2026趋势是混合:检索最佳几段,然后提示生成模型基于那些段落回答。这就是RAG,课程14深入覆盖检索部分。本课构建问答部分。

## 概念讲解

![问答架构:抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 用Transformer(BERT族)一起编码问题和段落。训练两个头预测答案的起始和结束词元索引。损失是有效位置的交叉熵。输出是段落中的片段。构造上永不幻觉,构造上永不处理段落无法回答的问题。

**检索增强式(RAG)。** 两阶段。首先,检索器从语料库找top-`k`段落。其次,阅读器(抽取式或生成式)用那些段落产出答案。检索器-阅读器分离让各自独立训练和评估。现代RAG常在两者间加重排器。

**生成式。** 解码器仅大语言模型(GPT、Claude、Llama)从学习权重回答。无检索步骤。常识优秀,罕见或近期事实灾难性。幻觉率与预训练数据中事实频率反相关。

## 动手实践

### Step 1:预训练模型做抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`在SQuAD 2.0上训练,包含不可答问题。默认`question-answering`管道返回最高分片段即使模型的空分数胜出——它不*自动*返回空答案。要获得显式"无答案"行为,传`handle_impossible_answer=True`给管道调用:管道只在空分数超过每个片段分数时返回空答案。无论哪种方式总检查`score`字段。

### Step 2:检索增强管道(草图)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器(Sentence-BERT)通过语义相似度找相关段落。抽取式阅读器(RoBERTa-SQuAD)从合并的top段落拉出答案片段。小语料库上工作。百万文档语料库,用FAISS或向量数据库。

### Step 3:生成式配RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式重要。显式告诉模型锚定上下文并在上下文不足时返回"我不知道",比朴素提示减少幻觉率40-60%。更精细模式加引用、置信分数和结构化抽取。

### Step 4:反映真实世界的评估

SQuAD用**精确匹配(EM)**和**词元级F1**。EM是归一化后(小写、剥标点、去冠词)严格匹配——要么预测精确匹配要么得0分。F1在预测和参考词元重叠上计算并给部分分。两者低估转述:"June 29, 2007" vs "June 29th, 2007"通常EM得0(序数破归一化)但仍从重叠词元获可观F1。

生产问答:

- **答案准确率**(大语言模型评判或人评判,因为指标不捕获语义等价)。
- **引用准确率。** 引用段落实际支持答案吗?生成引用和检索段落间字符串匹配自动检查平凡。
- **拒绝校准。** 答案不在检索段落时,系统正确说"我不知道"吗?测假置信率。
- **检索召回。** 评估阅读器前,测检索器是否把正确段落放入top-`k`。阅读器不能修复缺失段落。

### RAGAS:2026生产评估框架

`RAGAS`专为RAG系统构建,是2026发货默认。它无需金标准就评分四维:

- **忠实度。** 答案中每个主张来自检索上下文吗?通过自然语言推理基蕴涵测量。主要幻觉指标。
- **答案相关性。** 答案针对问题吗?通过从答案生成假设问题并与真问题比较测量。
- **上下文精确率。** 检索片段中多少实际相关?低精确率=提示噪声。
- **上下文召回。** 检索集包含所有需信息吗?低召回=阅读器不能成功。

无参考评分让你在活生产流量上评估无需 curated 金答案。对精确匹配指标无用开放问题顶层叠大语言模型评判。

`pip install ragas`。插你的检索器+阅读器。每查询得四个标量。退步警报。

## 实际应用

2026栈。

| 用例 | 推荐 |
|------|------|
| 给定段落,找答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库上,闭卷不可接受 | RAG:密集检索器+大语言模型阅读器 |
| 文档存储上实时 | RAG配混合(BM25+密集)检索器+重排器(课程14) |
| 对话式问答(追问) | 大语言模型配对话历史+每轮RAG |
| 高事实性、监管领域 | 权威语料库上抽取式;永不单独生成式 |

抽取式问答2026不时髦因为配大语言模型的RAG处理更多情况。仍发货于需逐字引用语境:法律研究、合规监管、审计工具。

## 产出成果

存`outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: 选择问答架构、检索策略和评估计划。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

给定需求(语料库大小、问题类型、事实性约束、延迟预算),输出:

1. 架构。抽取式、RAG配抽取阅读器、RAG配生成阅读器或闭卷大语言模型。一句话理由。
2. 检索器。无、BM25、密集(命名编码器)或混合。
3. 阅读器。SQuAD调谐模型、命名大语言模型或"领域微调DistilBERT"。
4. 评估。抽取基准EM+F1;生产答案准确率+引用准确率+拒绝校准。命名测什么和怎么测。

拒绝监管或合规敏感问题的闭卷大语言模型答案。拒绝无检索召回基线的问答系统(检索器未浮出正确段落无法评估阅读器)。标记需多跳推理问题需专门多跳检索器如HotpotQA训练系统。
```

## 练习题

1. **简单。** 在10篇Wikipedia段落上设上述SQuAD抽取管道。手编10问题。测答案正确率。段落和问题干净应见7-9正确。
2. **中等。** 加拒绝分类器。当top检索分数低于阈值(如0.3余弦),返回"我不知道"而非调用阅读器。保留集上调阈值。
3. **困难。** 在你选的10,000文档语料库上构建RAG管道。实现混合检索(BM25+密集)配RRF融合(见课程14)。测有无混合步答案准确率。文档哪种问题类型获益最多。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 抽取式问答 | 找答案片段 | 预测给定段落内答案的起止索引。 |
| 开放域问答 | 语料库上问答 | 无给定段落;必须检索再回答。 |
| RAG | 检索再生成 | 检索增强生成。检索器+阅读器管道。 |
| SQuAD | 标准基准 | Stanford问答数据集。EM+F1指标。 |
| 幻觉 | 编造答案 | 阅读器输出不受检索上下文支持。 |
| 拒绝校准 | 知何时闭嘴 | 无法回答时系统正确说"我不知道"。 |

## 延伸阅读

- [Rajpurkar等(2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)——基准论文。
- [Karpukhin等(2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)——DPR,问答标准密集检索器。
- [Lewis等(2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)——命名RAG论文。
- [Gao等(2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)——综合RAG综述。