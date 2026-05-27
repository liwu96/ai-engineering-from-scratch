# RAG分块策略

> 分块配置影响检索质量如嵌入模型选择(Vectara NAACL 2025)。分块错则无重排能救。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程14(信息检索)、阶段5课程22(嵌入模型)
**时间:** ~60分钟

## 问题背景

你放50页合同进RAG系统。用户问:"终止条款是什么?"检索器返封面页。为何?因模型512词元块训,终止条款20页后,跨页断,无本地关键词绑查询。

修复非"买更好嵌入模型"。修复是分块。多大?重叠?在哪分?配周围上下文?

2026年2月基准显意外结果:

- Vectara 2026研究:递归512词元分块胜语义分块69%→54%准确率。
- SPLADE + Mistral-8B在Natural Questions:重叠零可测收益。
- 上下文悬崖:响应质量在约2,500词元上下文急剧降。

"显"(语义分块、20%重叠、1000词元)常错。本课构建六策略直觉并告何时用哪个。

## 概念讲解

![六分块策略在一段可视化](../assets/chunking.svg)

**固定分块。** 每N字符或词元分。最简基线。中句断。好压缩,差连贯。

**递归。** LangChain的`RecursiveCharacterTextSplitter`。先`\n\n`分,再`\n`,再`.`,再空格。干净回退。2026默认。

**语义。** 每句子嵌入。算相邻句子余弦相似度。相似度低阈值时分。保主题连贯。慢;有时产40词元碎片伤检索。

**句子。** 句边界分。每块一句或N句窗。~5k词元内语义分块匹配成本分数。

**父文档。** 存小子块检索*和*大父块上下文。子检索;返父。优雅降:坏子块仍返合理父。

**迟分块(2024)。** 先词元级嵌入整文档,再池词元嵌入成块嵌入。保跨块上下文。长上下文嵌入器(BGE-M3, Jina v3)工作。高算力。

**上下文检索(Anthropic, 2024)。** 每块前加大语言模型生摘要其文档位置("此块是终止条款第3.2节...")。Anthropic自基准35-50%检索改进。索引贵。

### 胜每默认规则

匹配块大小到查询类型:

| 查询类型 | 块大小 |
|----------|--------|
| 事实("CEO名是什么?") | 256-512词元 |
| 分析/多跳 | 512-1024词元 |
| 全节理解 | 1024-2048词元 |

NVIDIA 2026基准。块应大到含答案加本地上下文,小到检索器top-K聚焦答案而非上下文噪声。

## 动手实践

### Step 1:固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Step 2:语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

领域调`threshold`。太高→碎片。太低→一巨块。

### Step 3:父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察:去重父。多子可映同父;返全浪费上下文。

### Step 4:上下文检索(Anthropic模式)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化块。查询时,检索益于额外周围信号。

### Step 5:评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

总基准。"最佳"策略可能不匹配任何博客。

## 陷阱

- **仅事实查询评估分块。** 多跳查询显不同胜者。用查询类型分层评估集。
- **语义分块无最小大小。** 产40词元碎片伤检索。总强制`min_tokens`。
- **重叠cargo cult。** 2026研究发现重叠常零收益双倍索引成本。测,不假设。
- **无最小/最大强制。** 5词元或5000词元块都破检索。钳。
- **跨文档分块。** 永不让块跨两文档。总每文档分块,再合并。

## 实际应用

2026栈:

| 情况 | 策略 |
|------|------|
| 首建,未知语料 | 递归,512词元,无重叠 |
| 事实问答 | 递归,256-512词元 |
| 分析/多跳 | 递归,512-1024词元+父文档 |
| 重交叉引用(合同、论文) | 迟分块或上下文检索 |
| 对话/对话语料 | 轮级块+说话人元数据 |
| 短话语(推文、评论) | 一文档=一块 |

从递归512开始。50查询评估集测recall@5。从那调。

## 产出成果

存`outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: 为给定语料和查询分布选分块策略、大小和重叠。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定语料(文档类型、平均长度、领域)和查询分布(事实/分析/多跳),输出:

1. 策略。递归/句子/语义/父文档/迟/上下文。理由。
2. 块大小。词元计数。理由绑查询类型。
3. 重叠。默认0;如>0理由。
4. 最小/最大强制。`min_tokens`、`max_tokens`卫。
5. 评估计划。50查询分层评估集(事实、分析、多跳)recall@5。

拒绝无最小/最大块大小强制分块策略。拒绝无消融显有帮助重叠>20%。标记语义分块推荐无最小词元底线。
```

## 练习题

1. **简单。** 20页文档用固定(512,0)、递归(512,0)和递归(512,100)分块。比块计数和边界质量。
2. **中等。** 5文档构30查询评估集。测递归、语义和父文档recall@5。哪个胜?匹配博客吗?
3. **困难。** 实现上下文检索。测基线递归MRR改进。报索引成本(大语言模型调用)vs准确率增益。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 块 | 文档片 | 嵌入、索引和检索子文档单元。 |
| 重叠 | 安全边 | 相邻块共享N词元;2026基准常无用。 |
| 语义分块 | 智分块 | 相邻句子嵌入相似度降时分。 |
| 父文档 | 两级检索 | 小子检索,大父返。 |
| 迟分块 | 嵌入后分块 | 词元级嵌入整文档,池成块向量。 |
| 上下文检索 | Anthropic技巧 | 索引前每块加大语言模型生摘要。 |
| 上下文悬崖 | 2500词元墙 | RAG中观察到约2.5k上下文词元质量降(2026年1月)。 |

## 延伸阅读

- [Yepes等/LangChain—Recursive Character Splitting文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/)——生产默认。
- [Vectara(2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070)——分块如嵌入选择重要。
- [Jina AI—Late Chunking in Long-Context Embedding Models(2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)——迟分块论文。
- [Anthropic—Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)——大语言模型生上下文前缀35-50%检索改进。
- [NVIDIA 2026块大小基准—Premai总结](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)——按查询类型块大小。