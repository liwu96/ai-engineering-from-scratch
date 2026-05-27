# 朴素贝叶斯

> "朴素"假设是错的，但它仍然工作。这就是它的美。

**类型:** 构建
**语言:** Python
**前置要求:** 第2阶段, 课程01-07 (分类, 贝叶斯定理)
**时间:** ~75分钟

## 学习目标

- 从零实现带Laplace平滑的多项朴素贝叶斯文本分类
- 解释为何朴素独立假设数学错但实践产正确类排序
- 比较多项、伯努利和高斯朴素贝叶斯变体并为给定特征类型选对的
- 在高维稀疏数据评估朴素贝叶斯vs逻辑回归并解释偏差方差权衡

## 问题背景

你需分类文本。邮件成垃圾邮件或非垃圾邮件。顾客评论成正或负。支持票据成类别。你有千特征(每词一)和有限训练数据。

多数分类器这里窒息。逻辑回归需够样本可靠估计千权重。决策树一次分裂一词狂过拟合。10,000维KNN无意义因每点与每其他点等远。

朴素贝叶斯处理这。它做数学错假设(给定类每特征独立其他每特征)，它仍文本分类胜"聪明"模型，尤其小训练集。它单遍数据训练。它扩到百万特征。它产概率估计(虽常因独立假设校准差)。

理解为何错假设导好预测教你ML根本东西: 最佳模型非最正确那个，它是对你数据偏差方差权衡最好那个。

## 概念讲解

### 贝叶斯定理(快回顾)

贝叶斯定理翻转条件概率:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们要`P(class | features)` -- 给定文档词其属类概率。我们可从:
- `P(features | class)` -- 这类文档见这些词似然
- `P(class)` -- 类先验概率(垃圾邮件总体多常见？)
- `P(features)` -- 证据，所有类相同，比较时可忽略

最高`P(class | features)`类胜。

### 朴素独立假设

精确计算`P(features | class)`需估计所有特征一起联合概率。10,000词词汇，需估计2^10,000可能组合分布。不可能。

朴素假设: 给定类每特征条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

非一不可能联合分布，你估计n简单每特征分布。每只需计数。

这假设明显错。词"机器"和"学习"在任何文档不独立。但分类器不需正确概率估计。它需正确排序 -- 哪类最高概率。独立假设引入系统误差，但那些误差对所有类相似影响，所以排序保持正确。

### 为何仍工作

三原因:

1. **排序胜校准。** 分类只需最高排名类正确。即使P(spam) = 0.99999当真实概率是0.7，分类器仍正确选垃圾邮件。我们不需正确概率。我们需要正确胜者。

2. **高偏差，低方差。** 独立假设是强先验。它重约束模型，防过拟合。有限训练数据，微错但稳定模型胜理论上正确但狂不稳定模型。这是偏差方差权衡在行动。

3. **特征冗余抵消。** 相关特征供冗余证据。分类器双算这证据，但它也为正确类双算。如果"机器"和"学习"总一起出现，两者都供"技术"类证据。NB计两次，但为正确类计两次。

第四，实用原因: 朴素贝叶斯极快。训练是单遍数据计频率。预测是矩阵乘。你可秒训练百万文档。这速度意味你比慢模型更快迭代、试更多特征集、跑更多实验。

### 数学逐步

让我们追踪具体例。假设两类: 垃圾和非垃圾。词汇三词: "免费", "钱", "会议"。

训练数据:
- 垃圾邮件提"免费"80次，"钱"60次，"会议"10次(共150词)
- 非垃圾邮件提"免费"5次，"钱"10次，"会议"100次(共115词)
- 40%邮件垃圾，60%非垃圾

带Laplace平滑(alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件含: "免费"(2次)，"钱"(1次)，"会议"(0次)。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件大差距胜。词"免费"出现两次是垃圾邮件强证据。注意"会议"不出现对两log求和贡献零(0 * log(P)) -- 多项NB，缺失词无效果。是伯努利NB显式建模词缺失。

### 三变体

朴素贝叶斯三味。各不同建模`P(feature | class)`。

#### 多项朴素贝叶斯

建模每特征作计数。文本数据最佳，特征是词频或TF-IDF值。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha`是Laplace平滑(下解释)。这变体是文本分类主力。

#### 高斯朴素贝叶斯

建模每特征作正态分布。连续特征最佳。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

各类得自己每特征均值和方差。这工作好当特征在各类内真随钟曲线。

#### 伯努利朴素贝叶斯

建模每特征作二元(存在或缺失)。短文本或二元特征向量最佳。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

异于多项，伯努利显式惩罚词缺失。如果"免费"典型在垃圾邮件出现但此邮件缺失，伯努利计这作反对垃圾邮件证据。

### 何时用每变体

| 变体 | 特征类型 | 最佳用 | 例 |
|---------|-------------|----------|---------|
| 多项 | 计数或频率 | 文本分类，词袋 | 邮件垃圾，主题分类 |
| 高斯 | 连续值 | 带正态特征表格数据 | Iris分类，传感器数据 |
| 伯努利 | 二元(0/1) | 短文本，二元特征向量 | SMS垃圾，存在/缺失特征 |

### Laplace平滑

当词出现在测试数据但从未在特定类训练数据出现会发生什么？

无平滑: `P(word | class) = 0/N = 0`。一零乘整产品使`P(class | features) = 0`，不管所有其他证据。单未见词毁整预测，不管多少其他证据支持。

Laplace平滑给每特征加小计数`alpha`(通常1):

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

alpha=1，每词得至少微小概率。词"困惑"在测试邮件出现不再杀垃圾概率。平滑有贝叶斯解释: 它等价放均匀Dirichlet先验在词分布。

更高alpha意味更强平滑(更均匀分布)。更低alpha意味模型更信数据。Alpha是调超参。

alpha效果:

| Alpha | 效果 | 何时用 |
|-------|--------|-------------|
| 0.001 | 几无平滑，信数据 | 极大训练集，无预期未见特征 |
| 0.1 | 轻平滑 | 大训练集 |
| 1.0 | 标准Laplace平滑 | 默认起点 |
| 10.0 | 重平滑，平分布 | 极小训练集，多预期未见特征 |

### Log空间计算

乘数百概率(各小于1)导浮点下溢。产品成浮点零即使真值极小正数。

解: 工作在log空间。非乘概率，加它们log:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这把预测转点积:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘。这就是为何朴素贝叶斯预测如此快 -- 它与单层线性模型同操作。

### 朴素贝叶斯vs逻辑回归

两者文本线性分类器。差在它们建模什么。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式(建模P(X\|Y)) | 判别式(建模P(Y\|X)) |
| 训练 | 计频率 | 优化损失函数 |
| 小数据 | 更好(强先验帮助) | 更差(不够估计权重) |
| 大数据 | 更差(错假设伤) | 更好(灵活边界) |
| 特征 | 假设独立 | 处理相关 |
| 速度 | 单遍，极快 | 迭代优化 |
| 校准 | 差概率 | 更好概率 |

经验规则: 从朴素贝叶斯开始。如果你有够数据且NB平台，切换逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[建词汇]
    C --> D[计词频]
    D --> E[应用平滑]
    E --> F[算log概率]
    F --> G[预测: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践，我们工作在log空间避浮点下溢。非乘多小概率，我们加它们log:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 构建

`code/naive_bayes.py`代码从零实现多项NB和高斯NB。

### 多项NB

从零实现:

1. **fit(X, y)**: 对每类，计每特征频率。加Laplace平滑。算log概率。存类先验(类频率log)。

2. **predict_log_proba(X)**: 对每样本，算log P(class) + sum log P(feature_i | class)对所有类。这是矩阵乘: X @ log_probs.T + log_priors。

3. **predict(X)**: 返回最高log概率类。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞: 拟合后，预测只是矩阵乘加偏置。这就是为何朴素贝叶斯如此快。

### 高斯NB

连续特征，我们估计各类各特征均值和方差:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测用每特征高斯PDF，跨特征乘(log空间加)。

### 演示: 文本分类

代码生成合成词袋数据模拟两类(技术文章vs体育文章)。各类有不同词频分布。多项NB用词计数分类它们。

合成数据这样工作: 我们创建200"词"(特征列)。词0-39在技术文章高频体育低频。词80-119在体育高频技术低频。词40-79两者中频。这创建现实场景: 些词强类指示器，其他噪声。

### 演示: 连续特征

代码生成Iris类数据(3类，4特征，高斯簇)。高斯NB用各类均值和方差分类。各类有不同中心(均值向量)和不同散布(方差)，模拟真实世界测量在类别间系统差异。

代码也演示:
- **平滑比较:** 训练多项NB用不同alpha值示平滑强度对精度效果。
- **训练大小实验:** NB精度如何随训练数据从20到1600样本增改善。NB极少样本达合理精度 -- 这是主要优势。
- **混淆矩阵:** 各类精度、召回和F1分数示NB在哪犯错。

### 预测速度

朴素贝叶斯预测是矩阵乘。对n样本d特征k类:
- 多项NB: 一矩阵乘(n x d) @ (d x k) = O(n * d * k)
- 高斯NB: n * k高斯PDF评估，各跨d特征 = O(n * d * k)

两者每维线性。比KNN(需对所有训练点距离计算)或带RBF核SVM(需对所有支持向量核评估)。NB预测时快数级。

## 使用

用sklearn，两变体一行:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

文本分类用sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py`代码比较从零实现与sklearn在同数据验证正确性。

### TF-IDF与朴素贝叶斯

原始词计数给每词每出现等权重。但常见词如"the"和"is"在每类频繁出现 -- 它们不带信息。TF-IDF(词频逆文档频率)降权常见词升权稀有区分词。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF值非负，所以与多项NB工作。TF-IDF + 多项NB组合是文本分类最强基线之一。它在少于10,000训练样本数据集常胜更复杂模型。

### 伯努利NB短文本

短文本(推文, SMS, 聊天消息)，伯努利NB可胜多项NB。短文本词计数低，所以多项NB依赖频率信息噪声。伯努利NB只关心存在或缺失，短文本更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer中`binary=True`标志转所有计数到0/1。无它，伯努利NB仍工作但见它非设计计数。

### 校准NB概率

NB概率校准差。当NB说P(spam) = 0.95，真实概率可能是0.7。如果你需可靠概率估计(例如，设阈值或与其他模型组合)，用sklearn CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这拟合逻辑回归在NB原始分数上用交叉验证。结果概率更近真实类频率。

### 常见陷阱

1. **负特征值。** 多项NB需非负特征。如果你有负值(如TF-IDF特定设置或标准化特征)，用高斯NB替代，或移特征为正。

2. **零方差特征。** 高斯NB除方差。如果特征对类零方差(所有值相同)，概率计算断。代码给所有方差加小平滑项(1e-9)防这。

3. **类不平衡。** 如果99%邮件非垃圾，先验P(not-spam) = 0.99太强它压倒似然证据。你可手动设类先验或用sklearn class_prior参数。

4. **特征缩放。** 多项NB不需缩放(它工作在计数)。高斯NB也不需缩放(它估计每特征统计)。这是逻辑回归和SVM优势，它们对特征尺度敏感。

## 交付成果

本课程产生:
- `outputs/skill-naive-bayes-chooser.md` -- 选对NB变体决策技能
- `code/naive_bayes.py` -- 从零多项NB和高斯NB，带sklearn比较

### 朴素贝叶斯何时失败

NB失败当独立假设导错误排序(非仅错误概率)。这发生在:

1. **强特征交互。** 如果类依赖两特征组合但不依赖单独(XOR类模式)，NB完全错过。每特征单独无证据，NB不能非线性组合它们。

2. **高度相关特征反对证据。** 如果特征A说"垃圾邮件"特征B说"非垃圾邮件"，但A和B完美相关(它们现实总同意)，NB将见冲突证据何处无冲突。

3. **极大训练集。** 有够数据，判别模型如逻辑回归学真决策边界胜NB。小数据帮助的独立假设现在拖模型后腿。

实践，这些失败模式文本分类罕见。文本特征多，个别弱，独立假设误差倾向抵消。对少强相关特征表格数据，先考虑逻辑回归或树基模型。

## 练习题

1. **平滑实验。** 文本数据用alpha值0.01, 0.1, 1.0, 10.0, 和100.0训练多项NB。绘精度vs alpha。性能峰值在哪？为何极高alpha伤？

2. **特征独立测试。** 取真实文本数据集。挑两明显相关词("机器"和"学习")。算P(word1 | class) * P(word2 | class)并比较P(word1 AND word2 | class)。独立假设多错？它影响分类精度吗？

3. **伯努利实现。** 扩展代码带伯努利NB类。转词袋到二元(存在/缺失)并比较精度vs多项NB在文本数据。伯努利何时胜？

4. **NB vs逻辑回归。** 文本数据训练两者。开始100训练样本增到10,000。绘两者精度vs训练集大小。逻辑回归何点超越朴素贝叶斯？

5. **垃圾过滤器。** 构建完整垃圾分类器: 分词原始邮件文本，建词汇，创建词袋特征，训练多项NB，用精度和召回评估(非仅精度 -- 为什么？)。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | "简单概率分类器" | 应用贝叶斯定理带特征给定类条件独立假设分类器 |
| 条件独立 | "特征不影响彼此" | P(A, B \| C) = P(A \| C) * P(B \| C) -- 知B告你新无关于A一旦你知道C |
| Laplace平滑 | "加一平滑" | 给每特征加小计数防零概率主导预测 |
| 先验 | "见数据前你信什么" | P(class) -- 观察任何特征前各类概率 |
| 似然 | "数据拟合多好" | P(features \| class) -- 如果类已知观察这些特征概率 |
| 后验 | "见数据后你信什么" | P(class \| features) -- 观察特征后更新类概率 |
| 生成式模型 | "建模数据如何生成" | 学P(X \| Y)和P(Y)然后用贝叶斯定理得P(Y \| X)模型 |
| 判别式模型 | "建模决策边界" | 直接学P(Y \| X)不建模X如何生成模型 |
| log概率 | "避下溢" | 工作在log P替代P防多小数乘积成浮点零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三变体带数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项vs伯努利文本经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 文本NB改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明NB比LR少数据更快收敛