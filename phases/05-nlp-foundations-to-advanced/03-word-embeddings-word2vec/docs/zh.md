# 词嵌入 — 从零实现Word2Vec

> 一个词由其所伴随的同伴定义。在这个理念上训练一个浅层网络，几何结构就会显现。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 02（BoW + TF-IDF），第3阶段 · 03（从零实现反向传播）
**时间：** 约75分钟

## 问题背景

TF-IDF知道 `dog` 和 `puppy` 是不同的词。它不知道它们意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来弥补，但这在罕见词、领域术语以及你未预料到的每种语言上都失败了。

你想要一种表示方式，其中 `dog` 和 `puppy` 在空间中彼此靠近。其中 `king - man + woman` 靠近 `queen`。其中在 `dog` 上训练的模型可以免费向 `puppy` 转移一些信号。

Word2Vec给了我们那个空间。两层神经网络，万亿词元训练运行，2013年发表。架构几乎令人尴尬地简单。结果重塑了NLP十年。

## 概念讲解

**分布假设**（Firth，1957）："一个词由其所伴随的同伴定义。"如果两个词出现在相似的上下文中，它们可能意思相似。

Word2Vec有两种变体，都利用了这个理念。

- **Skip-gram。** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为2。
- **CBOW（连续词袋）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram训练较慢但更好地处理罕见词。它成为了默认选择。

网络有一个没有非线性的隐藏层。输入是词汇表上的独热向量。输出是词汇表上的softmax。训练后，你扔掉输出层。隐藏层权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          这就是嵌入
```

技巧：100k词上的softmax计算量过大。Word2Vec使用**负采样**将其变成二元分类任务。预测"这个上下文词是否出现在这个中心词附近？是或否"。每对训练词采样少量负（非共现）词，而不是计算整个词汇表上的softmax。

## 动手实践

### 步骤1：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个（中心，上下文）对是一个正训练样本。

### 步骤2：嵌入表

两个矩阵。`W` 是中心词嵌入表（你保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇表大小10k和维度100是现实的；对于教学，50词汇 × 16维度足以看到几何结构。

### 步骤3：负采样目标

对每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使点积 `W[center] · W'[context]` 对正样本高，对负样本低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正样本对上的逻辑损失（希望sigmoid接近1）加上负样本对上的逻辑损失（希望sigmoid接近0）。梯度流向两个表。完整推导在原始论文中；如果你想牢记，用铅笔和纸走一遍。

### 步骤4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上经过足够多轮训练后，共享上下文的词具有相似的中心嵌入。在玩具语料库上，你会微弱地看到这种效果。在数十亿词元上，你会戏剧性地看到它。

### 步骤5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的300维Google News向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室。而是因为向量 `(king - man)` 捕获了类似"皇室"的东西，将它加到 `woman` 上靠近皇室女性区域。

## 实际应用

从零编写Word2Vec是教学。生产NLP使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己训练Word2Vec。你下载预训练向量。

- **GloVe** — Stanford的共现矩阵分解方法。50d、100d、200d、300d检查点。良好的通用覆盖。第04课专门介绍GloVe。
- **fastText** — Facebook的Word2Vec扩展，嵌入字符n-gram。通过组合子词处理词汇外词。第04课。
- **Google News上的预训练Word2Vec** — 300d，3M词词汇，2013年发表。至今每天仍在下载。

### 2026年Word2Vec仍然获胜的情况

- 轻量级领域特定检索。在医疗摘要上训练一小时，获得专业向量，这是任何通用模型无法捕获的。
- 类比风格的特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它以获得性别中性的轴。仍在公平性研究中使用。
- 可解释性。100维足够小，可以通过PCA或t-SNE绘制并实际看到聚类形成。
- 任何必须在无GPU设备上推理的地方。Word2Vec查找就是单行获取。

### Word2Vec失效的地方

多义词墙。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs 家具）共享它。下游分类器无法从向量中区分语义。

上下文嵌入（ELMo、BERT、之后的每个Transformer）通过基于周围上下文为每次出现的词产生不同的向量来解决这个问题。这就是从Word2Vec到BERT的飞跃：从静态到上下文。第7阶段涵盖Transformer部分。

词汇外问题是另一个失效。Word2Vec从未见过 `Zoomer-approved` 如果它不在训练数据中。没有回退。fastText通过子词组合修复（第04课）。

## 产出成果

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查训练好的word2vec模型。运行类比，找到邻居，诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探测训练好的词嵌入以验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三个标准类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告top-1结果及其余弦值。
2. 用户提供的五个领域特定词的最近邻测试。打印top-5邻居及其余弦值。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内。
4. 一个退化检查。如果任何嵌入的范数低于0.01或高于100，模型有训练错误。标记它。

拒绝仅基于类比准确性声明模型良好。类比基准是可被操纵的，并不能转移到下游任务。推荐内在评估 + 下游评估一起。
```

## 练习题

1. **简单。** 在微小语料库（20个关于猫和狗的句子）上运行训练循环。200轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在top 3中返回 `dog`。如果没有，增加轮数或词汇表。
2. **中等。** 添加高频词的子采样。频率高于 `10^-5` 的词以与其频率成比例的概率从训练对中丢弃。测量对罕见词相似性的影响。
3. **困难。** 在20 Newsgroups语料库上训练模型。计算两个偏见轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业有最大的偏见差距。这是公平性研究者使用的那种探测。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Word embedding | 词向量 | 从上下文学习的密集、低维（通常100-300）表示。 |
| Skip-gram | Word2Vec技巧 | 从中心词预测上下文词。比CBOW慢，对罕见词更好。 |
| Negative sampling | 训练捷径 | 用与 `k` 个随机词的二元分类替换完整词汇表上的softmax。 |
| Static embedding | 每词一个向量 | 无论上下文如何都相同的向量。在多义词上失效。 |
| Contextual embedding | 上下文敏感向量 | 基于周围词每次出现产生不同向量。Transformer产生的。 |
| OOV | 词汇外 | 训练未见过的词。Word2Vec无法为这些产生向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果原始论文的数学感觉密集。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实际有效的生产训练设置。
