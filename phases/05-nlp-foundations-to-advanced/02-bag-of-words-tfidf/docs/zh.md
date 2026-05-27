# 词袋模型、TF-IDF与文本表示

> 先计数，后思考。TF-IDF在2026年仍有定义明确的任务上击败嵌入模型。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 01（文本处理），第2阶段 · 02（从零实现线性回归）
**时间：** 约75分钟

## 问题背景

模型需要数字。你有字符串。

每个NLP流水线都必须回答同一个问题。如何将可变长度的词元流转换为分类器可以消费的固定大小向量？该领域的第一个答案是奏效的最简单方案：统计单词。构建向量。

这个向量支撑的生产NLP比任何嵌入模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25之前）、第一波情感分析、前十年学术NLP基准测试。2026年的从业者仍然在狭窄的分类任务上首先选择它。它快速、可解释，并且在词存在就是关键的任务上与4亿参数嵌入模型通常难以区分。

本课程从零开始构建词袋模型，然后是TF-IDF。然后展示scikit-learn用三行代码做同样的事。然后指出让你转向嵌入的失效模式。

## 概念讲解

**词袋模型（BoW）**丢弃顺序。对每个文档，统计每个词汇单词出现的次数。向量长度是词汇表大小。位置 `i` 是单词 `i` 的计数。

**TF-IDF**重新加权BoW。出现在每个文档中的单词信息量低，因此缩小权重。在整个语料库中罕见但在单个文档中频繁的单词是信号，因此放大权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（有多少文档包含该词），`N` 是总文档数。`log` 使随处可见的词权重有界。

关键属性：两者都产生具有可解释轴的稀疏向量。你可以查看训练分类器的权重，了解哪些词将文档推向每个类别。用768维BERT嵌入无法做到这一点。

## 动手实践

### 步骤1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分档文档列表（任何词级分词器都可以；本课程的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引0是第一个文档中看到的第一个词。约定各不相同；scikit-learn按字母顺序排序。

### 步骤2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现多少次"。文档1有两次 `cat` 因为它确实如此。文档0有零次 `ran` 因为它没有。

### 步骤3：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 确保每个文档都有的词IDF为1（不是0），与scikit-learn的默认值匹配。其他实现使用原始 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`, `cat`, `sat`, `dog`, `ran`）。`the` 出现在所有三个中，所以它的IDF很低。`dog` 出现在一个中，所以它的IDF很高。向量是稀疏的（大多数条目很小），有区分性的词突出来。

### 步骤5：L2归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，较长的文档获得较大的向量并在相似性分数中占主导。L2归一化将每个文档放在单位超球面上。行之间的余弦相似度现在只是点积。

## 实际应用

scikit-learn提供生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一键完成分词、词汇表和BoW。`TfidfVectorizer` 添加IDF加权和L2归一化。两者都返回稀疏矩阵。对于10万个文档，密集版本无法装入内存；保持稀疏直到分类器需要密集。

改变一切的参数：

| 参数 | 效果 |
|------|------|
| `ngram_range=(1, 2)` | 包含二元词组。通常提升分类效果。 |
| `min_df=2` | 删除出现在少于2个文档中的词。在嘈杂数据上修剪词汇表。 |
| `max_df=0.95` | 删除出现在超过95%文档中的词。无需硬编码列表近似停用词移除。 |
| `stop_words="english"` | scikit-learn的内置停用词列表。任务相关 — 情感分析不应该*删除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而不是原始 `tf`。当一个词在单个文档中重复多次时帮助。 |

### TF-IDF仍然获胜的情况（截至2026年）

- 垃圾邮件检测、主题标注、日志异常标记。词存在就是关键；语义细微差别不重要。
- 低数据场景（数百个标注示例）。TF-IDF加逻辑回归没有预训练成本。
- 任何延迟重要的地方。TF-IDF加线性模型在微秒内响应。将文档嵌入到Transformer需要10-100毫秒。
- 必须解释其预测的系统。检查分类器的系数。最正面的词就是原因。

### TF-IDF失效的情况

语义失明的失效。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是负面评论。一个是正面评论。它们的TF-IDF重叠正好是 `{the, movie, was}`。词袋分类器必须记住 `not` 靠近 `good` 会翻转标签。它可以在足够数据上学习这个，但永远不如理解句法的模型优雅。

另一个失效：推理时的词汇外词。在IMDb评论上训练的BoW模型对 `Zoomer-approved` 一无所知，如果该词元从未在训练中出现。子词嵌入（第04课）处理这个。TF-IDF无法做到。

### 混合：TF-IDF加权嵌入

2026年中等数据分类的实用默认：使用TF-IDF权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入获得语义能力，从TF-IDF获得罕见词强调。分类器在汇聚向量上训练。这在大约5万个标注示例以下的情感、主题和意图分类上优于单独使用任何一种。

## 产出成果

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐BoW、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述，输出：

1. 表示方法（BoW、TF-IDF、Transformer嵌入或混合）。用一句话解释原因。
2. 特定向量化器配置。命名库。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 一个发货前要测试的失效模式。

拒绝在用户有少于500个标注示例时推荐嵌入，除非他们展示TF-IDF基线上语义失效的证据。拒绝为情感分析移除停用词（否定词携带信号）。标记类别不平衡为需要不仅仅是向量化器改变。

示例输入："将30k客户支持工单分类到12个类别。大多数工单是2-3句话。仅英语。需要为审计日志提供可解释性。"

示例输出：

- 表示方法：TF-IDF。3万个示例不算小；可解释性要求排除了密集嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时就是停用词（"not working" vs "working"）。
- 要测试的失效：验证 `min_df=3` 不会删除罕见的类别关键词。运行 `get_feature_names_out` 按类别过滤并目视检查。
```

## 练习题

1. **简单。** 在L2归一化TF-IDF输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为1.0，词汇不重叠的文档得分为0.0。
2. **中等。** 向 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n`-gram的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上为 `["the cat", "cat sat"]` 产生二元词组计数。
3. **困难。** 使用GloVe 100d向量（下载一次，缓存）构建上述TF-IDF加权嵌入混合。在20 Newsgroups数据集上比较与纯TF-IDF和纯平均池化嵌入的分类准确性。报告哪种在何处获胜。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BoW | 词频向量 | 文档中词汇单词的计数。丢弃顺序。 |
| TF | 词频 | 文档中词的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 包含该词至少一次的文档计数。 |
| IDF | 逆文档频率 | `log(N / df)` 平滑。降低出现在所有地方的词的权重。 |
| Sparse vector | 大多数为零 | 词汇表通常为10k-100k词；大多数在任何给定文档中都不存在。 |
| Cosine similarity | 向量角度 | L2归一化向量的点积。1是相同，0是正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 规范API参考，加上每个参数的注释。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使TF-IDF成为十年默认值的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026年关于旧方法何时获胜及为何的见解。
