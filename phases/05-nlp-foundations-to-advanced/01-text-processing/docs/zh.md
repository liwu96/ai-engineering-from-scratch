# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的，模型是离散的。预处理是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** 第2阶段 · 14（朴素贝叶斯）
**时间：** 约45分钟

## 问题背景

模型无法直接读取 "The cats were running."。它读取的是整数。

每个NLP系统都要面对相同的三个问题：单词从哪里开始？单词的词根是什么？如何处理 "run"、"running"、"ran" —— 何时将它们视为相同，何时视为不同。

分词出错，模型就会从垃圾数据中学习。如果分词器将 `don't` 处理为一个词元但 `do n't` 处理为两个，训练分布就会分裂。如果词干提取器将 `organization` 和 `organ` 压缩为相同的词干，主题建模就会失效。如果词形还原器需要词性标注（POS）上下文但你未提供，动词会被当作名词处理。

本课程从零开始构建三种预处理原语，然后展示NLTK和spaCy如何实现相同的工作，让你了解其中的权衡。

## 概念讲解

三种操作。每种都有其作用和失效模式。

**分词**将字符串分割成词元。"词元"故意模糊，因为正确的粒度取决于任务。经典NLP使用词级别，Transformer使用子词级别，没有空格的语言使用字符级别。

**词干提取**根据规则砍掉后缀。快速、激进、简单。`running -> run`。`organization -> organ`。第二个就是失效模式。

**词形还原**使用语法知识将单词还原为其词典形式。更慢、更准确、需要查找表或形态分析器。`ran -> run`（需要知道"ran"是"run"的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且可以容忍噪音时使用词干提取（搜索索引、粗略分类）。当语义重要时使用词形还原（问答、语义搜索、任何用户会阅读的内容）。

## 动手实践

### 步骤1：基于正则表达式的分词器

最简单实用的分词器将非字母数字字符作为分隔符，同时将标点符号作为独立词元保留。不完美，不是最终方案，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序：带可选内部撇号的单词（`don't`、`it's`）、纯数字、任何非空白非字母数字字符作为独立词元（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失效模式。`3pm`分割为 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务来说足够好。URL、邮箱、话题标签都会断裂。生产环境需要在通用模式前添加特定模式。

### 步骤2：Porter词干提取器（仅步骤1a）

完整的Porter算法有五阶段规则。仅步骤1a就覆盖了最常见的英语后缀并展示了模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则使得 `ponies -> poni` 而不是 `pony`。真正的Porter有步骤1b可以修复这个问题。规则竞争，前面的规则胜出。顺序比任何单个规则都重要。

### 步骤3：基于查找表的词形还原器

真正的词形还原需要形态学知识。一个可行的教学版本使用小型词元表和回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，我们的回退只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、复数形式变化（`children -> child`）。这就是为什么生产系统使用WordNet、spaCy的形态分析器或完整的形态学分析器。

### 步骤4：将它们组合成流水线

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器。第5阶段 · 07（词性标注）会构建一个。目前，将所有内容默认为 `NOUN` 并承认这个限制。

## 实际应用

NLTK和spaCy提供生产级版本。每种只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`处理缩写、Unicode、边缘情况，这些都是你的正则表达式会遗漏的。`PorterStemmer`运行全部五个阶段。`WordNetLemmatizer`需要将词性标签从NLTK的Penn Treebank方案转换为WordNet的缩写集合。上面的转换线连接是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy将整个流水线隐藏在 `nlp(text)` 后面。分词、词性标注和词形还原同时运行。规模上比NLTK更快。开箱即用更准确。代价是你不能轻易替换单个组件。

### 如何选择

| 情况 | 选择 |
|------|------|
| 教学、研究、替换组件 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer流水线（你会使用模型的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失效模式

大多数教程教授算法就停止了。两件事会困扰真正的预处理流水线，而且它们几乎从未被覆盖。

**可复现性漂移。** NLTK和spaCy在不同版本之间改变分词和词形还原器的行为。在spaCy 2.x中产生 `['do', "n't"]` 的内容在3.x中可能产生 `["don't"]`。你的模型是在一个分布上训练的。推理现在在另一个分布上运行。准确性悄无声息地下降，没人知道为什么。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结20个样本句子的预期分词。每次升级时运行它。

**训练/推理不匹配。** 在训练时使用激进的预处理（小写、停用词移除、词干提取），在原始用户输入上部署，看着性能暴跌。这是生产NLP中最常见的单一故障。如果你在训练时预处理，必须在推理时运行完全相同的函数。将预处理作为函数放在模型包内发送，而不是作为服务团队会重写的notebook单元格。

## 产出成果

一个可复用的提示，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为NLP任务推荐分词、词干提取和词形还原设置。
phase: 5
lesson: 01
---

你为经典NLP预处理提供建议。给定任务描述，你输出：

1. 分词选择（正则表达式、NLTK word_tokenize、spaCy或Transformer分词器）。解释原因。
2. 是否词干提取、词形还原、两者都要或都不要。解释原因。
3. 特定库调用。命名函数。如果涉及NLTK，引用词性标签转换。
4. 用户应该测试的一个失效模式。

拒绝为用户可见文本推荐词干提取。拒绝在没有词性标签的情况下推荐词形还原。标记非英语输入为需要不同流水线。
```

## 练习题

1. **简单。** 扩展 `tokenize` 以将URL保留为单个词元。测试：`tokenize("Visit https://example.com today.")` 应产生一个URL词元。
2. **中等。** 实现Porter步骤1b。如果单词包含元音且以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则（`hopping -> hop`，不是 `hopp`）。
3. **困难。** 构建一个使用WordNet作为查找表但在WordNet没有条目时回退到你的Porter词干提取器的词形还原器。在标记好的语料库上测量准确性，与纯WordNet和纯Porter对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Token | 一个词 | 模型消费的任何单位。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则的后缀剥离结果。不总是真正的词。 |
| Lemma | 词典形式 | 你会查找的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性标注 | 类别如名词、动词、形容词。词形还原准确需要它。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖于它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线如何连接。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
