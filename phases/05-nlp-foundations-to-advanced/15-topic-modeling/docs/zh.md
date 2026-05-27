# 主题建模——LDA和BERTopic

> LDA:文档是主题混合,主题是词分布。BERTopic:文档在嵌入空间聚类,聚类是主题。同目标,不同原语。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程02(BoW+TF-IDF)、阶段5课程03(Word2Vec)
**时间:** ~45分钟

## 问题背景

你有10,000客服工单、50,000新闻文章或200,000推文。需知道集合讲什么不读它。无标签类别。甚至不知多少类别存在。

主题建模无监督回答。给语料库,返小集合连贯主题,每文档返这些主题上的分布。

两算法族主导。LDA(2003)视每文档为潜在主题混合,每主题为词分布。推理是贝叶斯。仍发货于需混合成员主题分配和可解释词级概率分布的生产。

BERTopic(2020)用BERT编码文档,UMAP降维,HDBSCAN聚类,基于类TF-IDF提取主题词。短文本、社交媒体和语义相似度比词重叠重要处胜出。每文档一个主题,长内容是限制。

本课构建两者直觉并命名给定语料库选哪个。

## 概念讲解

![LDA混合模型vs BERTopic聚类](../assets/topic-modeling.svg)

**LDA生成故事。** 每主题是词分布。每文档是主题混合。生成文档中词,从文档混合采样主题,再从该主题分布采样词。推理反转:给观测词,推断每文档主题分布和每主题词分布。Collapsed Gibbs采样或变分贝叶斯做数学。

关键LDA输出:

- `doc_topic`:矩阵`(n_docs, n_topics)`,每行和为1(文档主题混合)。
- `topic_word`:矩阵`(n_topics, vocab_size)`,每行和为1(主题词分布)。

**BERTopic管道。**

1. 用句子Transformer(如`all-MiniLM-L6-v2`)编码每文档。384维向量。
2. UMAP降维到~5维。BERT嵌入太高维无法聚类。
3. HDBSCAN聚类。密度基,产可变大小聚类和"离群"标签。
4. 每聚类,在聚类文档上算基于类TF-IDF提取top词。

输出每文档一个主题(加-1离群标签)。可选,HDBSCAN概率向量软成员。

## 动手实践

### Step 1:scikit-learn做LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意:停词移除,min_df和max_df滤稀有和无处不在词,CountVectorizer(非TfidfVectorizer)因LDA期待原始计数。

### Step 2:BERTopic(生产)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1`滤掉BERTopic离群桶(HDBSCAN无法聚类文档)。`min_topic_size`控HDBSCAN最小聚类大小;BERTopic库默认10。此例课程规模显设15。超10,000文档语料库,增到50或100。

### Step 3:评估

两法输出主题词。问题是那些词是否连贯。

- **主题连贯性(c_v)。** 组合滑窗上下文top词对NPMI(归一化逐点互信息),聚合分数为主题向量,通过余弦相似度比较那些向量。更高更好。用`gensim.models.CoherenceModel`配`coherence="c_v"`。
- **主题多样性。** 所有主题top词中唯一词分数。更高更好(主题不重叠)。
- **定性检查。** 读每主题top词。命名真实事物吗?人判断仍是最后防线。

## 何时选哪个

| 情况 | 选 |
|------|------|
| 短文本(推文、评论、标题) | BERTopic |
| 长文档配主题混合 | LDA |
| 无GPU/有限算力 | LDA或NMF |
| 需文档级多主题分布 | LDA |
| 大语言模型集成做主题标签 | BERTopic(直接支持) |
| 资源受限边缘部署 | LDA |
| 最大语义连贯 | BERTopic |

最大实际考量是文档长度。BERT嵌入截断;LDA计数在任何长度工作。文档长于嵌入模型上下文,要么分块+聚合要么用LDA。

## 实际应用

2026栈:

- **BERTopic。** 短文本和语义重要处默认。
- **`gensim.models.LdaModel`。** 经典LDA生产用,成熟,实战测试。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验简易LDA。
- **NMF。** 非负矩阵分解。LDA快替代,短文本可比质量。
- **Top2Vec。** 类BERTopic设计。社区小但某些基准好。
- **FASTopic。** 更新,大语料库比BERTopic快。
- **大语言模型基标签。** 跑任意聚类,再提示模型命名每聚类。

## 产出成果

存`outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: 为语料库选LDA或BERTopic。指定库、旋钮、评估。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料库描述(文档数、平均长度、领域、语言、算力预算),输出:

1. 算法。LDA/NMF/BERTopic/Top2Vec/FASTopic。一句话理由。
2. 配置。主题数:`recommended = max(5, round(sqrt(n_docs)))`,40,000文档以下语料库钳到200;仅语料库真大(>40k)允许>200并注增加算力成本。神经方法`min_df`/`max_df`滤和嵌入模型也属此。
3. 评估。主题连贯性(c_v)通过`gensim.models.CoherenceModel`,主题多样性,和20样本人读。
4. 探测失败模式。LDA,"垃圾主题"吸收停词和频词。BERTopic,-1离群聚类吞噬歧义文档。

拒绝无分块策略在嵌入模型上下文窗口以上文档上BERTopic。拒绝极短文本(推文、<10词元评论)上LDA因连贯崩塌。标记任何n_topics选择<5可能错;标记40k以下语料库>200可能过分裂。
```

## 练习题

1. **简单。** 在20 Newsgroups数据集上5主题LDA。打印每主题top 10词。手标每主题。算法找真类别了吗?
2. **中等。** 在同20 Newsgroups子集上BERTopic。比发现主题数、top词和定性连贯与LDA。哪个更干净显露真类别?
3. **困难。** 算你语料库上LDA和BERTopic c_v连贯。5、10、20、50主题跑各。绘连贯vs主题数。报哪种方法跨主题数更稳。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 主题 | 语料库讲的事 | 词概率分布(LDA)或相似文档聚类(BERTopic)。 |
| 混合成员 | 文档是多主题 | LDA分配每文档所有主题上分布。 |
| UMAP | 降维 | 保局部结构流形学习;BERTopic用。 |
| HDBSCAN | 密度聚类 | 找可变大小聚类;产离群"噪声"标签(-1)。 |
| c_v连贯 | 主题质量指标 | 滑窗内top主题词平均逐点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan(2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)——LDA论文。
- [Grootendorst(2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)——BERTopic论文。
- [Röder, Both, Hinneburg(2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)——引入c_v等论文。
- [BERTopic文档](https://maartengr.github.io/BERTopic/)——生产参考。优秀示例。