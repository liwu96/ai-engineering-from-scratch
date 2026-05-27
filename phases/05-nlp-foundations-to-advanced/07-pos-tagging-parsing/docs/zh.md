# 词性标注与句法分析

> 语法曾有一段时间不流行。然后每个LLM流水线都需要验证结构化提取，它又回来了。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 01（文本处理），第2阶段 · 14（朴素贝叶斯）
**时间：** 约45分钟

## 问题背景

第01课承诺词形还原需要词性标注。不知道 `running` 是动词，词形还原器无法将其还原为 `run`。不知道 `better` 是形容词，它无法还原为 `good`。

那个承诺隐藏了整个子领域。词性标注分配语法类别。句法分析恢复句子的树结构：哪个词修饰哪个，哪个动词管辖哪些论元。经典NLP花了二十年完善两者。然后深度学习将它们坍缩成预训练Transformer上的词元分类任务，研究界继续前进。

不是应用界。每个结构化提取流水线仍在底层使用词性和依存树。LLM生成的JSON根据语法约束验证。问答系统使用依存分析分解查询。机器翻译质量评估器检查分析树的对齐。

值得了解。本课程介绍标注集、基线和停止从头实现并调用spaCy的点。

## 概念讲解

**词性标注**用语法类别标注每个词元。**Penn Treebank（PTB）**标注集是英语默认。36个标注， casual读者觉得挑剔的区分：`NN` 单数名词，`NNS` 复数名词，`NNP` 单数专有名词，`VBD` 动词过去时，`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies（UD）**标注集更粗（17个标注）且语言无关；它成为跨语言工作的默认。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**产生树。两种主要风格：

- **成分分析。** 名词短语、动词短语、介词短语彼此嵌套。输出是非终结符类别（NP、VP、PP）的树，词作为叶子。
- **依存分析。** 每个词有一个它依赖的单一中心词，用语法关系标注。输出是每条边为（中心，依赖，关系）三元组的树。

依存分析在2010年代获胜，因为它干净地跨语言泛化，特别是自由词序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手实践

### 步骤1：最频繁词基线

奏效的最简单词性标注器。对每个词，预测它在训练中最常有的标注。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在Brown语料库上，这个基线达到~85%准确性。不好，但任何严肃模型都不应低于的底线。

### 步骤2：二元词组HMM标注器

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标注的标注），发射概率（给定标注的词）。用拉普拉斯平滑从计数估计两者。用Viterbi解码（标注格上的动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown上的二元词组HMM达到~93%准确性。从85%到93%的跳跃主要是转移概率 — 模型学习 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 步骤3：为什么现代标注器击败这个

转移+发射概率是局部的。它们无法捕获 `saw` 在 "I bought a saw" 中是名词但在 "I saw the movie" 中是动词。带任意特征的CRF（后缀、词形、前后词、词本身）达到~97%。BiLSTM-CRF或Transformer达到~98%+。

这个任务的天花板由标注者分歧设定。人类标注者在Penn Treebank上约97%时间一致。超过98%的模型可能过拟合测试集。

### 步骤4：依存分析草图

完整依存分析从零开始超出范围；经典教科书处理在Jurafsky和Martin中。两个经典家族：

- **基于转移**的分析器（arc-eager、arc-standard）像移进-归约分析器：读取词元，移到栈上，应用创建弧的归约动作。贪婪解码快速。经典实现是MaltParser。现代神经版本：Chen和Manning的基于转移分析器。
- **基于图**的分析器（Eisner算法、Dozat-Manning双仿射）为每个可能的中心-依赖边打分并挑选最大生成树。更慢但更准确。

大多数应用工作，调用spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下到上阅读 `dep` 列，句子的语法结构显现。

## 实际应用

每个生产NLP库将词性和依存分析器作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确、与分词+NER+词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford的CoreNLP继承者。60+语言的最先进。
- **trankit**。基于Transformer，良好的UD准确性。
- **NLTK**。`pos_tag`。可用、慢、旧。教学足够好。

### 2026年这仍然重要的地方

- **词形还原。** 第01课需要词性来正确词形还原。始终。
- **LLM输出的结构化提取。** 验证生成的句子是否尊重语法约束（例如，主谓一致、必需修饰语）。
- **基于方面的情感。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析分解为结构化约束。
- **跨语言转移。** UD标注和依存关系是语言无关的，实现新语言的零样本结构化分析。
- **低计算流水线。** 如果你无法部署Transformer，词性+依存分析+地名表让你走得很远。

## 产出成果

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游NLP任务设计经典词性+依存流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定下游任务（信息提取、重写验证、查询分解、词形还原），你输出：

1. 标注集。仅英语的Penn Treebank用于传统流水线，多语言或跨语言的Universal Dependencies。
2. 库。大多数生产用spaCy，学术级多语言用stanza，最高UD准确性用trankit。命名具体模型ID。
3. 集成模式。展示调用库和消费所需属性（`.pos_`、`.dep_`、`.head`）的3-5行代码。
4. 要测试的失效模式。名词-动词歧义（`saw`、`book`、`can`）和PP附着歧义是经典陷阱。采样20个输出并目视检查。

拒绝推荐从头构建自己的分析器。从头构建分析器是研究项目，不是应用任务。标记任何消费词性标注而不处理小写/大写变体的流水线为脆弱。
```

## 练习题

1. **简单。** 在小标注语料库（例如NLTK的Brown子集）上使用最频繁词标注基线，在留出句子上测量准确性。验证~85%结果。
2. **中等。** 训练上面的二元词组HMM并报告每标注精确率/召回率。HMM最混淆哪些标注？
3. **困难。** 使用spaCy的依存分析从1000句样本中提取主-谓-宾三元组。在50个手动标注的三元组上评估。记录提取失败的地方（通常是被动、并列和省略主语）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| POS tag | 词的类型 | 语法类别。PTB有36个；UD有17个。 |
| Penn Treebank | 标准标注集 | 英语特定。细粒度动词时态和名词数。 |
| Universal Dependencies | 多语言标注集 | 比PTB粗；语言中性；跨语言工作默认。 |
| Dependency parse | 句子树 | 每个词有一个中心，每条边有语法关系。 |
| Viterbi | 动态规划 | 给定发射和转移，找到最高概率标注序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 词性和分析的经典教科书处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标注集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上暴露的每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器带入主流的论文。
