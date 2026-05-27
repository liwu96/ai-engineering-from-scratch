# 情感分析

> 典型的NLP任务。关于经典文本分类需要知道的一切都在这里体现。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 02（BoW + TF-IDF），第2阶段 · 14（朴素贝叶斯）
**时间：** 约75分钟

## 问题背景

"The food was not great." 正面还是负面？

情感分析听起来简单。评论者说他们喜欢或不喜欢某物。给句子打标签。它成为典型NLP任务的原因是，每个看起来简单的案例都隐藏着一个困难案例。否定翻转含义。反讽反转它。"Not bad at all" 是正面尽管有两个负面编码词。表情符号携带比周围文本更多的信号。领域词汇重要（音乐评论中的 `tight` 与时尚评论中的 `tight`）。

情感分析是经典NLP的工作实验室。如果你理解为什么每个朴素基线都有特定的失效模式，你就理解为什么发明了每个更丰富的模型。本课程从零开始构建朴素贝叶斯基线，添加逻辑回归，并指出使生产情感分析成为合规级问题的陷阱。

## 概念讲解

经典情感分析是一个两步配方。

1. **表示。** 将文本转换为特征向量。BoW、TF-IDF或n-gram。
2. **分类。** 在标注示例上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是奏效的简单模型。假设每个特征在给定标签下独立。从计数估计 `P(word | positive)` 和 `P(word | negative)`。推理时，相乘概率。"朴素"独立性假设可笑地错误，但结果惊人地强。原因是：稀疏文本特征和中等数据下，分类器更关心每个词偏向哪一侧，而不是多少。

逻辑回归修复独立性假设。它为每个特征学习权重，包括负权重。`not good` 作为二元词组特征获得负权重。朴素贝叶斯无法为它从未标注的二元词组做到这一点。

## 动手实践

### 步骤1：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意小。实际工作使用数万示例（IMDb、SST-2、Yelp极性）。数学相同。

### 步骤2：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）是拉普拉斯平滑。没有它，未见于某类别的词概率为零，log爆炸。实践中 `alpha=0.01` 常见。`alpha=1.0` 是教学默认值。

### 步骤3：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2正则化在这里重要。文本特征稀疏；没有L2模型会记忆训练示例。从 `0.01` 开始并调整。

### 步骤4：处理否定（失效模式）

考虑 "not good" 和 "not bad"。BoW分类器看到 `{not, good}` 和 `{not, bad}` 并从训练中出现更多的那个学习。二元词组分类器看到 `not_good` 和 `not_bad` 并将它们学习为不同特征。这通常足够。

当你没有二元词组时奏效的一个更粗糙修复：**否定作用范围**。将否定词后的词元前缀 `NOT_` 直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们相反的权重。三行预处理，情感基准上可测量的准确性提升。

### 步骤5：重要的评估指标

仅准确性在类别不平衡时误导。真实情感语料库通常70-80%正面或70-80%负面；恒定多数分类器获得80%准确性且毫无价值。报告以下每一项：

- **每类精确率和召回率。** 每类一对。宏观平均它们以获得尊重类别平衡的单个数字。
- **Macro-F1（不平衡数据的主要指标）。** 每类F1分数的平均值，等权重。类别不平衡时用此代替准确性。
- **Weighted-F1（替代）。** 与macro相同但按类别频率加权。当不平衡本身有业务含义时与macro-F1一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标前始终检查；它揭示模型混淆哪两类。
- **每类错误样本。** 抽取每类5个错误预测。阅读它们。没有什么能替代阅读实际错误。

对于严重不平衡数据（> 95-5比例），报告 **AUROC** 和 **AUPRC** 代替准确性。AUPRC对少数类更敏感，这通常是你关心的（垃圾邮件、欺诈、罕见情感）。

**要避免的常见错误。** 在不平衡数据上报告micro-F1而不是macro-F1给出一个看起来很高的数字，因为它被多数类主导。Macro-F1迫使你看到少数类性能。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 实际应用

scikit-learn用六行代码正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元词组使 `not_good` 成为特征。`sublinear_tf=True` 减弱重复词。这三个标志是75%准确性基线与85%准确性基线在SST-2上的区别。

### 何时转向Transformer

- 反讽检测。经典模型在此失败。句号。
- 长评论中情感在中途转变。
- 基于方面的情感。"Camera was great but battery was terrible." 你需要将情感归因于方面。只有Transformer或结构化输出模型能做到。
- 非英语、低资源语言。多语言BERT给你零样本基线。

如果你需要上述任何一项，跳到第7阶段（Transformer深入讲解）。否则，TF-IDF上的朴素贝叶斯或逻辑回归加二元词组加否定处理是2026年生产基线。

### 可复现性陷阱（再次）

重训练情感模型是常规的。重新评估它们不是。论文报告的准确性数字使用特定分割、特定预处理、特定分词器。如果你不比较使用相同流水线的新模型与基线，你会得到误导性的差异。始终在流水线上重新生成基线，而不是论文的数字。

## 产出成果

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），你输出：

1. 特征提取配方。指定分词器、n-gram范围、停用词策略（通常保留）、否定处理（作用范围前缀或二元词组）。
2. 分类器。基线用朴素贝叶斯，生产用逻辑回归，仅当领域需要反讽/方面/跨语言时才用Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类错误样本（不只是标量）。
4. 一个部署后要监控的失效模式。领域漂移和反讽是前两个。

拒绝为情感任务推荐删除停用词。拒绝在类别不平衡（例如90%正面）时仅报告准确性作为唯一指标。标记子词丰富语言为需要FastText或Transformer嵌入而非词级TF-IDF。
```

## 练习题

1. **简单。** 在scikit-learn流水线中将 `apply_negation` 作为预处理步骤添加，并在小情感数据集上测量F1变化。
2. **中等。** 实现类别加权逻辑回归（传递 `class_weight="balanced"` 给scikit-learn，或自己推导梯度）。测量在合成90-10类别不平衡上的效果。
3. **困难。** 通过在情感模型残差上训练第二个分类器来构建反讽检测器。记录你的实验设置。当准确性低于偶然性时警告读者（2类反讽的偶然水平~50%，大多数第一次尝试落在那里）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Polarity | 正面或负面 | 二元标签；有时扩展到中性或细粒度（5星）。 |
| Aspect-based sentiment | 每方面极性 | 将情感归因于文本中提到的特定实体或属性。 |
| Negation scoping | 反转附近词元 | 在"not"后将词元前缀 `NOT_` 直到标点符号。 |
| Laplace smoothing | 计数加1 | 防止朴素贝叶斯中零概率特征。 |
| L2 regularization | 收缩权重 | 向损失添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础综述。很长，但前四节涵盖经典所有内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 展示二元词组 + 朴素贝叶斯在短文本上难以击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和你要调整的每个参数的参考。
