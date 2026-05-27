# 嵌入模型深入解析 — 2026年深度探讨

> Word2Vec给你每个词一个向量。现代嵌入模型给你每个段落一个向量,跨语言,配稀疏、稠密和多向量视图,大小适配你的索引。选错你的RAG就检索错误的东西。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程03(Word2Vec)、阶段5课程14(信息检索)
**时间:** ~60分钟

## 问题背景

你的RAG系统40%的时间检索错误段落。罪魁祸首很少是向量数据库或提示词。是嵌入模型。

2026年选择嵌入意味着在五个轴上选择:

1. **稠密vs稀疏vs多向量。**每个段落一个向量,还是每个词元一个,还是稀疏加权词袋。
2. **语言覆盖。**单语言英语模型在英语专属任务上仍胜出。多语言模型在语料混合时胜出。
3. **上下文长度。**512词元vs 8,192 vs 32,768——实际有效容量常是宣称上限的60-70%。
4. **维度预算。**全精度3,072浮点数 = 每向量12 KB。100M向量时,存储$1,300/月。Matryoshka截断削减4×。
5. **开源vs托管。**开源权重意味你控制栈和数据。托管意味用控制换永远最新。

本课程命名这些权衡,让你基于证据选择,而非上个季度流行什么。

## 概念讲解

![稠密、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入。**每个段落一个向量(通常384-3,072维)。余弦相似度按语义接近度排名段落。OpenAI `text-embedding-3-large`, BGE-M3稠密模式, Voyage-3。默认选择。

**稀疏嵌入。**SPLADE风格。Transformer为每个词表词元预测权重,然后将大多数归零。结果是大小|词表|的稀疏向量。捕获词汇匹配(像BM25)但配学习的词元权重。关键词重查询强。

**多向量(迟交互)。**ColBERTv2, Jina-ColBERT。每个词元一个向量。用MaxSim评分:对每个查询词元,找最相似文档词元,求和分数。存储和评分更贵,但在长查询和领域特定语料胜出。

**BGE-M3:三者合一。**单模型同时输出稠密、稀疏和多向量表示。每个可独立查询;分数通过加权求和融合。2026默认当你想从一个检查点获得灵活性。

**Matryoshka表示学习。**训练使向量前N维形成有用的独立嵌入。将1,536维向量截断到256维,约1%准确率换取6×存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+支持。

### MTEB排行榜讲述部分故事

Massive Text Embedding Benchmark——发布时(2022)8任务类型56任务,MTEB v2扩展到100+任务。2026年初,Gemini Embedding 2登顶检索(67.71 MTEB-R)。Cohere embed-v4领先通用(65.2 MTEB)。BGE-M3领先开源多语言(63.0)。排行榜必要但不充分——总在你的领域基准测试。

### 三层模式

| 用例 | 模式 |
|------|------|
| 快速首遍 | 稠密双编码器(BGE-M3, text-3-small) |
| 召回提升 | 稀疏(SPLADE, BGE-M3 sparse) + RRF融合 |
| Top-50精确率 | 多向量(ColBERTv2)或交叉编码器重排器 |

大多数生产栈三者都用。

## 动手实践

### Step 1: 基线——配Sentence-BERT的稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`使点积等于余弦相似度。总是设置它。

### Step 2: Matryoshka截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3和Voyage-4训练使前几级无损。非Matryoshka模型(原Sentence-BERT)截断时急剧退化。

### Step 3: BGE-M3多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引,一次推理调用。分数融合:

```python
dense_score = ... # dense_vecs余弦
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域调权重。

### Step 4: 自定义任务MTEB评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行候选模型。不要只信任排行榜排名——你的领域重要。

### Step 5: 从头手写余弦

见`code/main.py`。平均Hashing Trick嵌入(只用stdlib)。不与Transformer嵌入竞争,但展示结构:分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档用同一模型。**有些模型(Voyage, Jina-ColBERT)用不对称编码——查询和文档通过不同路径。总是检查模型卡。
- **缺失前缀。**`bge-*`模型需要查询前加`"Represent this sentence for searching relevant passages: "`。忘记则3-5点召回差距。
- **过度截断Matryoshka。**1,536 → 256通常安全。1,536 → 64不安全。在你的评估集验证。
- **上下文截断。**大多数模型静默截断超过最大长度的输入。长文档需要分块(见课程23)。
- **忽略延迟尾部。**MTEB分数隐藏p99延迟。600M模型可能比335M模型高2点但每查询成本3×。

## 实际应用

2026栈:

| 情况 | 选择 |
|------|------|
| 英语专属、快、API | `text-embedding-3-large`或`voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3`或`Qwen3-Embedding-8B` |
| 长上下文(32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU专属部署 | Nomic Embed v2(137M参数, MoE) |
| 存储受限 | Matryoshka截断 + int8量化 |
| 关键词重查询 | 加SPLADE稀疏, RRF与稠密融合 |

2026模式:从BGE-M3或text-3-large开始,用MTEB在你的领域评估,领域特定模型胜出超3点则替换。

## 产出成果

保存为`outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: 为给定语料和部署选择嵌入模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料(大小、语言、领域、平均长度)、部署目标(云/边缘/本地)、延迟预算和存储预算,输出:

1. 模型。命名检查点或API。一句话理由。
2. 维度。全/Matryoshka截断/int8量化。理由关联存储预算。
3. 模式。稠密/稀疏/多向量/混合。理由。
4. 模型卡要求的查询前缀/模板。
5. 评估计划。领域相关MTEB任务 + 配nDCG@10的留出领域评估。

拒绝无领域验证截断Matryoshka到<64维。拒绝10k段落以下语料用ColBERTv2(开销不合理)。标记配512词元窗口模型路由的>8k词元长文档语料。
```

## 练习题

1. **简单。**用`bge-small-en-v1.5`编码100句子,全维(384),然后Matryoshka 128。测量10查询MRR下降。
2. **中等。**在你的领域500段落上比较BGE-M3稠密、稀疏和colbert。recall@10哪个胜出?RRF融合是否胜过最佳单模式?
3. **困难。**在你的top-2领域任务上对三个候选模型运行MTEB。报告MTEB分数、100查询批次p99延迟、$/1M查询。选帕累托最优。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 稠密嵌入 | 向量 | 每文本一个固定大小向量。余弦相似度排名。 |
| 稀疏嵌入 | 学习的BM25 | 每词表词元一个权重;大多零;端到端训练。 |
| 多向量 | ColBERT风格 | 每词元一个向量;MaxSim评分;更大索引,更好召回。 |
| Matryoshka | 俄罗斯娃娃技巧 | 前N维自身是有效更小嵌入。 |
| MTEB | 基准 | Massive Text Embedding Benchmark——发布时56任务,v2 100+。 |
| BEIR | 检索基准 | 18零样本检索任务;常引用跨领域鲁棒性。 |
| 不对称编码 | 查询≠文档路径 | 模型对查询和文档用不同投影。 |

## 延伸阅读

- [Reimers, Gurevych(2019). Sentence-BERT](https://arxiv.org/abs/1908.10084)——双编码器论文。
- [Muennighoff等(2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)——排行榜论文。
- [Chen等(2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216)——统一三模式模型。
- [Kusupati等(2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)——维度阶梯训练目标。
- [Santhanam等(2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)——生产迟交互。
- [Hugging Face MTEB排行榜](https://huggingface.co/spaces/mteb/leaderboard)——实时排名。