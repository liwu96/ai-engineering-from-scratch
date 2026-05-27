# GloVe、FastText与子词嵌入

> Word2Vec为每个词训练一个嵌入。GloVe分解共现矩阵。FastText嵌入片段。BPE搭建通往Transformer的桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 03（从零实现Word2Vec）
**时间：** 约45分钟

## 问题背景

Word2Vec留下了两个开放问题。

首先，有一条并行研究线直接分解共现矩阵（LSA、HAL）而不是在线进行skip-gram更新。Word2Vec的迭代方法本质上更好，还是差异源于两种方法处理计数的方式？**GloVe**回答了：用深思熟虑选择的损失进行矩阵分解匹配或击败Word2Vec，且训练成本更低。

其次，两种方法对从未见过的词都没有处理方案。`Zoomer-approved`、`dogecoin`、上周创造的任何专有名词、罕见词根的每个变位形式。**FastText**通过嵌入字符n-gram修复：一个词是其部分的和，包括词素，所以即使是词汇外词也能获得合理的向量。

第三，一旦Transformer到来，问题再次转变。词级词汇表上限约一百万条目；真实语言比那更开放。**字节对编码（BPE）**及其变体通过频繁子词单元的词汇表解决，涵盖一切。每个现代LLM的每个现代分词器都是子词分词器。

本课程涵盖所有三个，然后解释何时选择哪个。

## 概念讲解

**GloVe（全局向量）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 出现在词 `i` 上下文中的频率。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。加权损失使频繁词对不占主导。完成。

**FastText。** 一个词是其字符n-gram的和加上词本身。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成向量的和。像Word2Vec一样训练。好处：未见词（`whereupon`）从已知的n-gram组合。

**BPE（字节对编码）。** 从字符级词汇表（或字节）开始。统计语料库中每对相邻字符。合并最频繁的成对为新词元。重复 `k` 次。结果：`k + 256` 个词元的词汇表，其中频繁序列（`ing`、`tion`、`the`）是单个词元，罕见词分解成熟悉的片段。每个句子都能分词成某些东西。

## 动手实践

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

两个值得命名的移动部分。加权函数 `f(x) = (x/x_max)^alpha` 降低非常频繁的词对（如 `(the, and)`）的权重，使它们不占损失主导。最终嵌入是 `W`（中心）和 `W_tilde`（上下文）表的和。将两者相加是已发表的技巧，往往优于仅使用一个。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其n-gram集合（通常是3到6个字符）表示。词嵌入是其n-gram嵌入的和。对于skip-gram训练，在Word2Vec使用单个向量的地方插入这个。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见词，只要其某些n-gram已知，你仍能获得向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以两者彼此靠近。

### BPE：学习的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最频繁的相邻对。经过足够多迭代，频繁的子串（`low`、`est`、`tion`）成为单个词元，罕见词干净地分裂。

真实的GPT/BERT/T5分词器学习30k-100k次合并。结果：任何文本都能分词成有界长度的已知ID序列，永远没有OOV。

## 实际应用

实践中，你很少自己训练这些。你加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer时代的BPE风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2约定）。每个现代分词器都是BPE变体、WordPiece（BERT）或SentencePiece（T5、LLaMA）。

### 何时选择哪个

| 情况 | 选择 |
|------|------|
| 预训练通用词向量，不需要OOV容忍 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误/新词/形态丰富语言 | FastText |
| 任何进入Transformer的（训练或推理） | 模型自带的分词器。绝不更换。 |
| 从零训练自己的语言模型 | 首先在语料库上训练BPE或SentencePiece分词器 |
| 生产文本分类与线性模型 | 仍是TF-IDF。第02课。 |

## 产出成果

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新语言模型或文本流水线选择分词方法。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你输出：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级）。一句话原因。
2. 词汇表大小目标（例如，英语专用LM为32k，多语言为64k-100k）。
3. 确切训练命令的库调用。命名库。引用参数。
4. 一个可复现性陷阱。分词器-模型不匹配是最常见的静默生产错误；指出哪一对必须一起使用。

拒绝在用户微调预训练LLM时推荐训练自定义分词器。拒绝为任何面向生产推理的模型推荐词级分词。标记非英语/多脚本语料库为需要带字节回退的SentencePiece。
```

## 练习题

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个n-gram集合的Jaccard重叠。你应该看到大量共享片段（`pla`、`lay`、`play`），这就是FastText在形态变体上转移良好的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词汇表增长。绘制每语料库字符的词元数作为合并次数的函数。你应该看到开始时快速压缩，渐近接近每词元~2-3个字符。
3. **困难。** 在莎士比亚全集上训练1k合并的BPE。比较常见词与罕见专有名词的分词。测量前后的平均每词词元数。写下什么让你惊讶。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Co-occurrence matrix | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 上下文中的次数。 |
| Subword | 词片段 | 字符n-gram（FastText）或学习的词元（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻对直到达到目标词汇表大小。 |
| OOV | 词汇外 | 模型从未见过的词。Word2Vec/GloVe失败。FastText和BPE处理它。 |
| Byte-level BPE | 原始字节上的BPE | GPT-2方案。词汇表从256字节开始，所以永远没有OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe论文，七页，仍然是最好的损失推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将BPE引入现代NLP的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece和SentencePiece在实践中如何不同的简明参考。
