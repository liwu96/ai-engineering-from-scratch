# 信息检索与搜索

> BM25精确但脆弱。密集检索撒大网但漏关键词。混合是2026默认。其余都是调参。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程02(BoW+TF-IDF)、阶段5课程04(GloVe、FastText、子词)
**时间:** ~75分钟

## 问题背景

用户输入"如果某人撒谎骗钱会发生什么"并期待找到实际覆盖的法律条款:"IPC第420条"。关键词搜索完全错过(无共享词汇)。语义搜索如果嵌入向量未在法律文本上训练也会错过。真实搜索必须处理两者。

信息检索是每个RAG系统、每个搜索栏、每个文档网站模糊查找下的管道。2026生产工作的架构不是单一方法。它是互补方法的链,每层捕获前一层的失败。

本课构建每个部分并命名各自捕获哪些失败。

## 概念讲解

![混合检索:BM25+密集+RRF+交叉编码器重排](../assets/retrieval.svg)

四层。选需要的。

1. **稀疏检索(BM25)。** 快,精确匹配准确,语义差。倒排索引上跑。百万文档每查询<10毫秒。正确找法规引用、产品代码、错误消息、命名实体。
2. **密集检索。** 编码查询和文档为向量。最近邻搜索。捕获转述和语义相似度。漏差一字符的精确关键词匹配。FAISS或向量数据库每查询50-200毫秒。
3. **融合。** 合并稀疏和密集排序列表。倒数排名融合(RRF)是简单默认因为它忽略原始分数(不同尺度)只用排名位置。加权融合是你知道某信号在领域主导时的选项。
4. **交叉编码器重排。** 取融合top-30。跑交叉编码器(查询+文档一起,评分每对)。保留top-5。交叉编码器每对比双编码器慢但准确得多。只跑top-30摊销。

三路检索(BM25+密集+学习稀疏如SPLADE)2026基准胜两路但需学习稀疏索引基础设施。多数团队,两路加交叉编码器重排是甜点。

## 动手实践

### Step 1:BM25从零实现

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个参数值得知道。`k1=1.5`控制词频饱和;更高意味词重复权重更多。`b=0.75`控制长度归一化;0忽略文档长度,1完全归一化。默认是Robertson原始论文推荐,很少需调。

### Step 2:双编码器密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2归一化嵌入向量使点积等于余弦。`all-MiniLM-L6-v2`是384维,快,足够强用于多数英文检索。多语言工作用`paraphrase-multilingual-MiniLM-L12-v2`。顶级准确率用`bge-large-en-v1.5`或`e5-large-v2`。

### Step 3:倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60`常数来自原始RRF论文。更高`k`平化排名差异贡献;更低`k`让top排名主导。60是发布默认,很少需调。

### Step 4:混合搜索+重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段组合。BM25找词汇匹配。密集找语义匹配。RRF合并两排名无需分数校准。交叉编码器重排top-30用查询-文档对一起,捕获双编码器漏的细粒度相关性。保留top-5。

### Step 5:评估

| 指标 | 含义 |
|------|------|
| Recall@k | 正确文档存在的查询中,多少次在top-k? |
| MRR(平均倒数排名) | 首相关文档1/rank平均。 |
| nDCG@k | 考虑相关性梯度,非仅二元相关/不。 |

RAG特指,检索器的**Recall@k**最重要数字。正确段落不在检索集时阅读器无法回答。

调试提示:失败查询,对比稀疏和密集排名。一个找正确文档另一个不,你有词汇不匹配(修复:加缺失半)或语义歧义(修复:更好嵌入或重排器)。

## 实际应用

2026栈:

| 规模 | 栈 |
|------|------|
| 1k-100k文档 | 内存BM25+`all-MiniLM-L6-v2`嵌入+RRF。无单独DB。 |
| 100k-10M文档 | FAISS或pgvector做密集+Elasticsearch/OpenSearch做BM25。并行跑。 |
| 10M+文档 | Qdrant/Weaviate/Vespa/Milvus配混合支持。交叉编码器重排top-30。 |
| 最佳质量前沿 | 三路(BM25+密集+SPLADE)+ColBERT晚交互重排 |

无论选什么,预算评估。基准端到端RAG准确率前基准检索召回。阅读器不能修复检索器漏的。

### 2026生产RAG惨痛教训

- **80%RAG失败追溯到摄入和分块,非模型。** 团队花周换大语言模型调提示词同时检索安静地每三查询返错上下文。先修复分块。
- **分块策略比分块大小重要。** 固定大小分裂破表、代码和嵌套头。句子感知是默认;语义或大语言模型基分块对技术文档和产品手册有回报。
- **父文档模式。** 检索小"子"分块求精确。同父节多个子块出现时,换入父块保上下文。这稳定提升答案质量无需重训练。
- **k_rerank=3通常最优。** 每额外分块加词元成本和生成延迟不提升答案质量。如k=8仍比k=3好,重排器欠表现。
- **HyDE/查询扩展。** 从查询生成假设答案,嵌入它,检索。桥接短问题和长文档间措辞缺口。无训练免费精度提升。
- **上下文预算低于8K词元。** 稳定触及该限意味重排器阈值太松。
- **版本一切。** 提示词、分块规则、嵌入模型、重排器。任何漂移静默破答案质量。忠实度、上下文精确率和未答问题率CI门在用户见前阻塞退步。
- **三路检索(BM25+密集+学习稀疏如SPLADE)基准胜两路**,尤对混合专有名词和语义查询。基础设施支持SPLADE索引时发货。

据2026行业测量,正确检索设计减少幻觉70-90%。多数RAG性能增益来自更好检索,非模型微调。

## 产出成果

存`outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: 为给定语料库和查询模式选检索栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求(语料库大小、查询模式、延迟预算、质量线、基础设施约束),输出:

1. 栈。BM25仅、密集仅、混合(BM25+密集+RRF)、混合+交叉编码器重排或三路(BM25+密集+学习稀疏)。
2. 密集编码器。命名特定模型。匹配语言、领域和上下文长度。
3. 重排器。命名特定交叉编码器模型如用。标记重排在top-30上加30-100毫秒延迟。
4. 评估计划。Recall@10是主要检索器指标。多答案用MRR。先基线,测量增量改进对它。

拒绝为有命名实体、错误代码或产品SKU的语料库推荐密集仅除非用户有证据密集处理精确匹配。拒绝为高风险检索(法律、医疗)跳重排,最终top-5决定用户答案。
```

## 练习题

1. **简单。** 在500文档语料库实现上述`hybrid_search`。测20查询。比BM25仅、密集仅和混合recall@5。
2. **中等。** 加MRR计算。每测试查询配已知正确文档,找BM25、密集和混合排名中正确文档排名。报每MRR。
3. **困难。** 用MultipleNegativesRankingLoss(Sentence Transformers)在领域微调密集编码器。从500查询-文档对构建训练集。比前后微调recall。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| BM25 | 关键词搜索 | Okapi BM25。按词频、IDF和长度评分文档。 |
| 密集检索 | 向量搜索 | 编码查询+文档为向量,找最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时快。 |
| 交叉编码器 | 重排模型 | 一起编码查询+文档。慢但准确。 |
| RRF | 排名融合 | 用`1/(k+rank)`求和合并两排名。 |
| Recall@k | 检索指标 | 相关文档在top-k的查询分数。 |

## 延伸阅读

- [Robertson和Zaragoza(2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)——权威BM25处理。
- [Karpukhin等(2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)——DPR,标准双编码器。
- [Formal等(2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)——学习稀疏检索器闭合与密集缺口。
- [Cormack, Clarke, Büttcher(2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)——RRF论文。
- [Khattab和Zaharia(2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)——晚交互检索。