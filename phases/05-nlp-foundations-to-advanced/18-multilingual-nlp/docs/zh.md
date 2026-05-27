# 多语言自然语言处理

> 一个模型,100+语言,大多数零训练数据。跨语言迁移是2020年代实际奇迹。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程04(GloVe、FastText、子词)、阶段5课程11(机器翻译)
**时间:** ~45分钟

## 问题背景

英文有数十亿标签例。乌尔都语有数千。迈蒂利语几乎无。任何服务全球受众的实用自然语言处理系统必须工作在无任务特定训练数据的语言长尾。

多语言模型通过同时多语言训练一模型解决此。共享表示让模型从高资源语言学技能迁移到低资源语言。英文情感分析微调模型,乌尔都语开箱出奇好情感预测。这是零样本跨语言迁移,重塑自然语言处理如何发货世界。

本课命名权衡、标准模型和一决策绊倒多语言工作新团队:选迁移源语言。

## 概念讲解

![通过共享多语言嵌入空间跨语言迁移](../assets/multilingual.svg)

**共享词汇。** 多语言模型用所有目标语言文本训SentencePiece或WordPiece分词器。词汇共享:同子词单元代表相关语言同词素。英文和意大利语`anti-`得同词元。

**共享表示。** 多语言掩语言建模预训Transformer学不同语言语义相似句子产相似隐藏状态。mBERT、XLM-R和NLLB都显此。英文"cat"嵌入近法语"chat"和西班牙语"gato",全句嵌入也。

**零样本迁移。** 一种语言(通常英文)标签数据微调模型。推理时,模型支持任何语言跑。无需目标语言标签。类型相关语言结果强,远语言弱。

**少样本微调。** 目标语言加100-500标签例。分类任务准确率跳到英文基线95-98%。这是多语言自然语言处理单一最划算杠杆。

## 模型

| 模型 | 年 | 覆盖 | 注 |
|------|------|------|------|
| mBERT | 2018 | 104语言 | Wikipedia训。首实用多语言LM。低资源弱。 |
| XLM-R | 2019 | 100语言 | CommonCrawl训(比Wikipedia大很多)。设跨语言基线。Base 270M, Large 550M。 |
| XLM-V | 2023 | 100语言 | XLM-R配1M词元词汇(vs 250k)。低资源好。 |
| mT5 | 2020 | 101语言 | T5架构多语言生成。 |
| NLLB-200 | 2022 | 200语言 | Meta翻译模型;含55低资源语言。 |
| BLOOM | 2022 | 46语言+13编程 | 开176B大语言模型多语言训。 |
| Aya-23 | 2024 | 23语言 | Cohere多语言大语言模型。阿拉伯语、印地语、斯瓦希里语强。 |

按用例选。分类用XLM-R-base作理智默认好。生成任务mT5或NLLB视翻译vs开放生成。大语言模型风格工作配Aya-23或Claude显式多语言提示。

## 源语言决策(2026研究)

多数团队默认英文作微调源。近期研究(2026)显这常错。

语言相似性比原始语料库大小更好预测迁移质量。斯拉夫目标,德语或俄语常胜英文。印度目标,印地语常胜英文。**qWALS**相似度指标(2026,基于World Atlas of Language Structures特征)量化此。**LANGRANK**(Lin等, ACL 2019)是分开更早方法从语言相似性、语料库大小和遗传相关组合排名候选源语言。

实践规则:若目标语言有类型近高资源亲缘,先试那微调,再比英文微调。

## 动手实践

### Step 1:零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一模型,三语言,同API。自然语言推理数据训XLM-R通过蕴涵技巧迁移分类好。

### Step 2:多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间近。不同英文句子远。这是跨语言检索、聚类和相似度工作原因。

### Step 3:少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

100-500目标语言例,`num_train_epochs=5`和`learning_rate=2e-5`是安全默认。更高学习率让多语言对齐崩塌你得英文仅模型。

## 实际工作的评估

- **每语言保留集准确率。** 不聚合。聚合藏长尾。
- **基准单语言基线。** 有足够数据语言,从头训单语言模型有时胜多语言。测。
- **实体级测试。** 目标语言命名实体。多语言模型对远拉丁文字脚本分词常弱。
- **跨语言一致性。** 两语言同义应产同预测。测缺口。

## 实际应用

2026栈:

| 任务 | 推荐 |
|------|------|
| 分类,100语言 | XLM-R-base(~270M)微调 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译,200语言 | `facebook/nllb-200-distilled-600M`(见课程11) |
| 生成多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言自然语言处理 | XLM-V或相关高资源语言领域特定微调 |

若性能重要总预算目标语言微调。零样本是起点,非最终答案。

### 分词税(低资源语言什么错)

多语言模型所有语言共享一分词器。词汇在英文、法语、西班牙语、中文、德语主导语料库训。主导集外任何语言,三税静默复合:

- **繁殖税。** 低资源语言文本每词分词成远多词元比英文。印地语句子需等价英文句子3-5×词元。那3-5×吃你上下文窗口、训练效率和延迟。
- **变体恢复税。** 每错拼、变音变体、Unicode归一化不匹配或大小写变在嵌入空间成冷启动无关序列。模型不能学原生说话者认为明显正字对应。
- **容量溢出税。** 税1和2耗上下文位置、层深和嵌入维。推理剩余系统性小比同模型高资源语言得。

实际症状:模型印地语正常训,损失曲线看对,评估困惑度看合理,生产输出微妙错。形态中间崩。罕见屈曲不可恢复。**你不能数据缩放方式出破分词器。**

缓解:选目标语言覆盖好分词器(XLM-V的1M词元词汇是直接修复);训前验证保留目标文本分词繁殖;真长尾脚本用字节级回退(SentencePiece `byte_fallback=True`, GPT-2风格字节级BPE)永无OOV。

## 产出成果

存`outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: 为多语言自然语言处理任务选源语言、目标模型和评估计划。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求(目标语言、任务类型、每语言可用标签数据),输出:

1. 微调源语言。默认英文;若目标语言有类型近高资源语言查LANGRANK或qWALS。
2. 基模型。XLM-R(分类)、mT5(生成)、NLLB(翻译)、Aya-23(生成大语言模型)。
3. 少样本预算。如可用从100-500目标语言例开始。仅标签不可行零样本。
4. 评估计划。每语言准确率(不聚合)、跨语言一致性、非拉丁脚本实体级F1。

拒绝发货无每语言评估多语言模型——聚合指标藏长尾失败。标记低分词覆盖脚本(阿姆哈拉语、提格雷语、多非洲语言)需字节回退模型(SentencePiece byte_fallback=True,或GPT-2风格字节级分词器)。
```

## 练习题

1. **简单。** 在英文、法语、印地语和阿拉伯语每语言10句跑零样本分类管道。报每准确率。应见法强、印地语还行、阿拉伯语变。
2. **中等。** 用`paraphrase-multilingual-MiniLM-L12-v2`构小混合语言语料库跨语言检索器。英文查询,检索任何语言文档。测recall@5。
3. **困难。** 比英文源和印地语源印地语分类任务微调。两 regime下500目标语言例少样本微调。报哪源产更好印地语准确率及多少。这是LANGRANK论题缩影。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 多语言模型 | 一模型多语言 | 跨语言共享词汇和参数。 |
| 跨语言迁移 | 一语言训另一跑 | 源微调,目标评估无目标语言标签。 |
| 零样本 | 无目标语言标签 | 目标语言无微调迁移。 |
| 少样本 | 小目标标签 | 100-500目标语言例用于微调。 |
| mBERT | 首多语言LM | Wikipedia预训104语言BERT。 |
| XLM-R | 标准跨语言基线 | CommonCrawl预训100语言RoBERTa。 |
| NLLB | Meta200语言MT | No Language Left Behind。含55低资源语言。 |

## 延伸阅读

- [Conneau等(2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)——XLM-R论文。
- [Pires, Schlinger, Garrette(2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502)——启动跨语言迁移研究线分析论文。
- [Costa-jussà等(2022). No Language Left Behind](https://arxiv.org/abs/2207.04672)——NLLB-200论文。
- [Üstün等(2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827)——Aya, Cohere多语言大语言模型。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance(2026)](https://www.mdpi.com/2504-4990/8/3/65)——qWALS/LANGRANK源语言论文。