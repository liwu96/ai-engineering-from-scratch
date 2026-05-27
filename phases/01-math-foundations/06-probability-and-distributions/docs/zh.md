# 概率与分布

> 概率是AI用来表达不确定性的语言。

**类型:** 学习
**语言:** Python
**前置要求:** 第1阶段, 课程01-04
**时间:** ~75分钟

## 学习目标

- 从零实现伯努利、类别分布、泊松分布、均匀分布和正态分布的PMF和PDF
- 计算期望值、方差，并使用中心极限定理解释为什么高斯分布无处不在
- 构建softmax和log-softmax函数，使用数值稳定性技巧（减去最大logit值）
- 从logits计算交叉熵损失，并将其与负对数似然联系起来

## 问题背景

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从50,000个候选词中选择下一个词。一个扩散模型通过从学习到的分布中采样来生成图像。这些都是概率的实际应用。

模型做出的每个预测都是一个概率分布。每个损失函数衡量预测分布与真实分布的距离。每个训练步骤调整参数，使一个分布更像另一个分布。没有概率，你无法阅读任何ML论文，无法调试任何模型，也无法理解为什么你的训练损失是NaN。

## 概念讲解

### 事件、样本空间与概率

样本空间S是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到0到1之间的数值。

```
抛硬币:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷骰子:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了所有概率:
1. P(A) >= 0 对任何事件A
2. P(S) = 1 (总会发生某件事)
3. P(A或B) = P(A) + P(B) 当A和B不能同时发生时

其他所有内容（贝叶斯定理、期望、分布）都源于这三条规则。

### 条件概率与独立性

P(A|B)是在B发生的条件下A发生的概率。

```
P(A|B) = P(A且B) / P(B)

例子: 扑克牌
  P(国王 | 人头牌) = P(国王且人头牌) / P(人头牌)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件独立意味着知道其中一个对另一个没有任何信息:

```
独立:   P(A|B) = P(A)
等价于: P(A且B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌不是独立的。

### 概率质量函数与概率密度函数

离散随机变量有概率质量函数(PMF)。每个结果都有可以直接读取的特定概率。

```
PMF: P(X = k)

公平骰子:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数(PDF)。单点的密度值不是概率。概率来自对密度函数在区间上的积分。

```
PDF: f(x)

P(a <= X <= b) = f(x)从a到b的积分

f(x)可以大于1 (密度, 不是概率)
f(x) dx从-inf到+inf的积分 = 1
```

这个区别在ML中很重要。分类输出是PMF（离散选择）。VAE潜在空间使用PDF（连续）。

### 常见分布

**伯努利分布:** 一次试验，两个结果。建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**类别分布:** 一次试验，k个结果。建模多分类（softmax输出）。

```
P(X = i) = p_i,  其中 p_i之和 = 1
例子: P(猫) = 0.7,  P(狗) = 0.2,  P(鸟) = 0.1
```

**均匀分布:** 所有结果等可能。用于随机初始化。

```
离散: P(X = k) = 1/n 对于k在{1, ..., n}
连续: f(x) = 1/(b-a) 对于x在[a, b]
```

**正态分布(高斯):** 铃形曲线。由均值(mu)和方差(sigma^2)参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态: mu = 0, sigma = 1
  68%的数据在1个sigma内
  95%在2个sigma内
  99.7%在3个sigma内
```

**泊松分布:** 固定区间内稀有事件的计数。建模事件率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```
离散:   E[X] = x_i * P(X = x_i)之和
连续: E[X] = x * f(x) dx的积分
```

方差衡量围绕均值的分散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在ML中，期望值以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型稳定性。梯度的高方差意味着噪声训练。

### 联合分布与边缘分布

联合分布P(X, Y)描述两个随机变量的组合。

联合PMF例子(X = 天气, Y = 雨伞):

| | Y=0 (无伞) | Y=1 (有伞) | 边缘P(X) |
|---|---|---|---|
| X=0 (晴天) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (雨天) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过求和消去另一个变量:

```
P(X = x) = 对所有y求和 P(X = x, Y = y)
```

上表中的行和列总计就是边缘分布。

### 为什么正态分布无处不在

中心极限定理: 许多独立随机变量的和（或平均）收敛到正态分布，无论原始分布是什么。

```
掷1个骰子:  均匀分布 (平坦)
2个骰子的平均:  三角形 (峰值)
30个骰子的平均: 近乎完美的铃形曲线

这对任何起始分布都成立。
```

这就是为什么:
- 测量误差近似正态（许多小的独立来源）
- 神经网络的权重初始化使用正态分布
- SGD中的梯度噪声近似正态（许多样本梯度的和）
- 正态分布是给定均值和方差的最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘会很快下溢到零。

```
P(句子) = P(词1) * P(词2) * ... * P(词_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (约30个词后下溢)
```

对数概率解决这个问题。乘法变成加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> 有限数值 (不下溢)
```

规则:
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0 (因为 0 < P <= 1)
- 更负 = 更不可能
- 交叉熵损失是正确类别的负对数概率

### Softmax作为概率分布

神经网络输出原始分数(logits)。Softmax将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / 对所有j求和exp(z_j)

性质:
  - 所有输出在(0, 1)
  - 所有输出之和为1
  - 保持输入的相对顺序
  - exp()放大logits之间的差异
```

Softmax技巧: 在指数化之前减去最大logit值以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (安全)

相同结果, 无溢出。
```

Log-softmax结合softmax和log以实现数值稳定性。PyTorch内部使用这个来计算交叉熵损失。

### 采样

采样指从分布中抽取随机值。在ML中:
- Dropout随机采样要归零的神经元
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要逆变换采样、拒绝采样或重参数化技巧（用于VAE）等技术。

## 动手实践

### 步骤1: 概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(国王 | 人头牌) = {p_king_given_face:.4f}")
```

### 步骤2: 从零实现PMF和PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤3: 期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"骰子: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤4: 从分布采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤5: Softmax和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤6: 中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤7: 可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化在 `code/probability.py`。

## 实际应用

使用NumPy和SciPy，以上所有内容都是一行代码:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零构建了这些。现在你知道库函数在做什么了。

## 练习题

1. 为指数分布实现逆变换采样。通过采样10,000个值并比较直方图与真实PDF来验证。

2. 为两个不均匀的骰子构建联合分布表。计算边缘分布并检查骰子是否独立。

3. 计算输出logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 的5分类器的交叉熵损失，正确类别是索引3。然后用PyTorch的 `nn.CrossEntropyLoss` 验证答案。

4. 编写一个函数，接受对数概率列表并返回最可能的序列、总对数概率和等效的原始概率。用每个词概率为0.01的50词句子测试。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 样本空间 | "所有可能性" | 实验所有可能结果的集合S |
| PMF | "概率函数" | 给出每个离散结果的精确概率的函数，总和为1 |
| PDF | "概率曲线" | 连续变量的密度函数。对其在区间上积分得到概率 |
| 条件概率 | "给定某条件的概率" | P(A|B) = P(A且B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | "它们互不影响" | P(A且B) = P(A) * P(B)。知道一个事件对另一个没有任何信息 |
| 期望值 | "平均值" | 所有结果的概率加权总和。损失函数就是期望值 |
| 方差 | "分散程度" | 期望的偏离均值平方偏差。高方差 = 噪声大、不稳定的估计 |
| 正态分布 | "铃形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。因CLT而无处不在 |
| 中心极限定理 | "平均趋向正态" | 许多独立样本的平均收敛到正态分布，无论来源是什么 |
| 联合分布 | "两个变量一起" | P(X, Y)描述X和Y结果每种组合的概率 |
| 边缘分布 | "消去另一个变量" | P(X) = sum_y P(X, Y)。从联合分布恢复单个变量的分布 |
| 对数概率 | "概率的对数" | log P(x)。将乘积转换为求和，防止长序列的数值下溢 |
| Softmax | "将分数转为概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值logits映射到有效概率分布 |
| 交叉熵 | "损失函数" | -sum(p_true * log(p_predicted))。衡量两个分布的差异。越小越好 |
| Logits | "原始模型输出" | softmax之前的未归一化分数。以逻辑函数命名 |
| 采样 | "抽取随机值" | 根据概率分布生成值。模型如何生成输出 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均趋向正态的可视化证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖此处所有内容及更多的简洁参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性重要及如何实现