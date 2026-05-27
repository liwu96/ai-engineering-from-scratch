# Transformer前的文本生成——N-gram语言模型

> 若词惊人,模型差。困惑度让惊讶成数。平滑保它有限。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程01(文本处理)、阶段2课程14(朴素贝叶斯)
**时间:** ~45分钟

## 问题背景

Transformer前,RNN前,词嵌入前,语言模型通过计数词跟在前`n-1`词后频率预测下词。计数"the cat"→"sat"47次,"the cat"→"jumped"12次,"the cat"→"refrigerator"0次。归一化得概率分布。

那是n-gram语言模型。1980到2015跑在每个语音识别器、每个拼写检查器和每个短语基机器翻译系统。当你需便宜设备上语言建模仍跑。

有趣问题是对未见n-gram怎么办。原始计数基模型对未见任何事分配零概率,灾难性因为句子长且几乎每个长句子含至少一个未见序列。50年平滑研究修复。Kneser-Ney平滑是结果,现代深度学习继承其经验传统。

## 概念讲解

![N-gram模型:计数、平滑、生成](../assets/ngram.svg)

**N-gram概率:**`P(w_i | w_{i-n+1}, ..., w_{i-1})`。定`n`(trigram典型3,4-gram典型4)。从计数算:

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未见任何n-gram得零概率。2007Brown语料库研究发现即使4-gram模型在训练中30%保留4-gram未见。无平滑无法在任何真实文本评估。

**平滑方法,按精巧顺序:**

1. **Laplace(加一)。** 每计数加1。简单,稀有事件上糟。
2. **Good-Turing。** 基频率-频率从高频事件重分配概率质量到未见。
3. **插值。** 用可调权重组合n-gram、(n-1)-gram等估计。
4. **回退。** 若n-gram计数零,回退到(n-1)-gram。Katz回退归一化此。
5. **绝对折扣。** 从所有计数减固定折扣`D`,重分配到未见。
6. **Kneser-Ney。** 绝对折扣加低阶模型巧妙选择:用*续概率*(词出现在多少上下文中)而非原始频率。

Kneser-Ney洞察深刻。"San Francisco"是常见bigram。Unigram"Francisco"大多在"San"后出现。朴素绝对折扣给"Francisco"高unigram概率(因计数高)。Kneser-Ney注意"Francisco"只在一个上下文出现并相应降其续概率。结果:结尾"Francisco"的新bigram得适当低概率。

**评估:困惑度。** 保留测试集每词平均负对数似然指数。低更好。困惑度100意味模型像在100词中均匀选择那样困惑。

```text
困惑度 = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实践

### Step 1:trigram计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是词元化句子列表。输出是n-gram计数和上下文计数。`<s>`和`</s>`是句子边界。

### Step 2:Laplace平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每计数加1。平滑但过分配质量到未见事件,也伤稀有已知事件。

### Step 3:Kneser-Ney(bigram,插值)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三活动部分。`continuation_prob`捕获"词出现在多少不同上下文?"(Kneser-Ney创新)。`lambda_prev`是折扣释放质量,用于加权回退。最终概率是折扣主项加加权续项。

### Step 4:采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每种子总给不同输出。类束搜索输出,每步选argmax(贪婪)加小随机旋钮(温度)。

### Step 5:困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

低更好。Brown语料库,良好调谐4-gram KN模型困惑度约140。Transformer LM同测试集15-30。差约10倍。该差是领域移走原因。

## 实际应用

- **经典自然语言处理教学。** 平滑、MLE和困惑度最清晰接触。
- **KenLM。** 生产n-gram库。用作语音和MT系统中低延迟重评分器。
- **设备上自动完成。** 键盘trigram模型。仍是。
- **基线。** 声明神经LM好前总算n-gram LM困惑度。若Transformer未宽胜KN,有错。

## 产出成果

存`outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: 训练神经LM前构建可复现n-gram语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途(下词预测、重评分、困惑度基线),输出:

1. N-gram阶。通用英文trigram,大语料库4-gram,语音重评分5-gram。
2. 平滑。Modified Kneser-Ney默认;Laplace仅教学。
3. 库。`kenlm`生产,`nltk.lm`教学,自建仅学习。
4. 评估。保留困惑度配训练和测试集间一致词元化。

拒绝报告比较系统间不同词元化算困惑度——困惑度数仅相同词元化下可比。标记测试集OOV率;KN除非训练时保留特殊<UNK>词元否则处理OOV差。
```

## 练习题

1. **简单。** 在1,000句莎士比亚语料库训trigram LM。生成20句。局部合理全局不连贯。这是标准演示。
2. **中等。** 在保留莎士比亚分割上为你KN模型实现困惑度。与Laplace比。应见KN困惑度降30-50%。
3. **困难。** 构建trigram拼写校正器:给定错拼词及其上下文,生成校正并按LM下上下文概率排。在Birkbeck拼写语料库(公开)上评估。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| N-gram | 词序列 | `n`连续词元序列。 |
| 平滑 | 避零 | 重分配概率质量使未见事件得非零概率。 |
| 困惑度 | LM质量指标 | 保留数据上`exp(-平均对数概率)`。低更好。 |
| 回退 | 回退到短上下文 | 若trigram计数零,用bigram。Katz回退形式化此。 |
| Kneser-Ney | n-gram最佳平滑 | 绝对折扣+低阶模型续概率。 |
| 续概率 | KN特有 | `P(w)`按`w`出现上下文数加权,非原始计数。 |

## 延伸阅读

- [Jurafsky和Martin—Speech and Language Processing, Chapter 3(2026草案)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)——n-gram LM和平滑权威处理。
- [Chen和Goodman(1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739)——定Kneser-Ney为最佳n-gram平滑器论文。
- [Kneser和Ney(1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394)——原始KN论文。
- [KenLM](https://kheafield.com/code/kenlm/)——快生产n-gram LM,2026仍用于延迟敏感应用。