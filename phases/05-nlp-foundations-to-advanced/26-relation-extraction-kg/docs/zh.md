# 关系抽取与知识图谱构建

> 命名实体识别找到了实体。实体链接锚定了它们。关系抽取找到它们之间的边。知识图谱是节点、边及其溯源的总和。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程06(命名实体识别)、阶段5课程25(实体链接)
**时间:** ~60分钟

## 问题背景

分析师读:"Tim Cook于2011年成为Apple的CEO。"四个事实:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取(RE)将自由文本转为结构化三元组`(subject, relation, object)`。聚合整个语料库就得到知识图谱。聚合并查询就得到RAG/检索增强生成、分析或合规审计的推理基。

2026问题:大语言模型热情地抽取关系。太热情。它们幻觉源文本不支持的三元组。无溯源无法区分真实三元组和貌似合理的虚构。2026答案是AEVS风格锚定-验证管道。

## 概念讲解

![文本→三元组→知识图谱](../assets/relation-extraction.svg)

**三元组形式。**`(subject_entity, relation_type, object_entity)`。关系来自封闭本体(Wikidata属性、FIBO、UMLS)或开放集合(OpenIE风格,什么都行)。

**三种抽取方法。**

1. **规则/模式基。**Hearst模式:"X such as Y"→`(Y, isA, X)`。加手工正则。脆弱、精确、可解释。
2. **监督分类器。**给定句中两实体提及,从固定集预测关系。TACRED、ACE、KBP训。2015–2022标准。
3. **生成式大语言模型。**提示模型输出三元组。开箱工作。需溯源,否则幻觉貌似合理的垃圾。

**AEVS(锚定-抽取-验证-补充, 2026)。**当前幻觉缓解框架:

- **锚定。**识别每实体跨度及关系短语跨度配精确位置。
- **抽取。**生成链接锚定跨度的三元组。
- **验证。**匹配每三元组元素回源文本;拒绝无支持者。
- **补充。**覆盖遍确保无锚定跨度被丢弃。

幻觉急剧降。需更多算力但可审计。

**开放vs封闭权衡。**

- **封闭本体。**固定属性列表(如Wikidata 11,000+属性)。可预测。可查询。难发明。
- **开放IE。**任意动词短语成关系。高召回。低精确。查询乱。

生产知识图谱常混合:开放IE发现,再规范化关系到封闭本体后合并进主图。

## 动手实践

### Step 1:模式基抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

见`code/main.py`完整玩具抽取器。Hearst模式仍发货于领域特定管道因其可调试。

### Step 2:监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL是seq2seq关系抽取器:文本入,三元组出,已是Wikidata属性id。远监督数据微调。标准开源权重基线。

### Step 3:大语言模型提示抽取配锚定

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每返回跨度对源。拒绝`text[start:end] != triple_entity`者。这是AEVS"验证"步最小形式。

### Step 4:规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化常占工程工作60-80%。预算它。

### Step 5:构小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每RAG/检索增强生成-知识图谱系统的原子。用RDF三元存储(Blazegraph、Virtuoso)、属性图(Neo4j)或向量增强图存储扩展。

## 陷阱

- **关系抽取前共指。**"He founded Apple"——关系抽取需知"he"是谁。先跑共指(课程24)。
- **实体规范化。**"Apple Inc"和"Apple"须解析到同节点。先实体链接(课程25)。
- **幻觉三元组。**大语言模型输出文本不支持的三元组。强制跨度验证。
- **关系规范化漂移。**开放IE关系不一致("was born in," "came from," "is a native of")。坍缩到规范id否则图不可查。
- **时序错误。**"Tim Cook is CEO of Apple"——现在真,2005假。多关系时间有界。用限定符(Wikidata `P580`开始时间、`P582`结束时间)。
- **领域不匹配。**REBEL Wikipedia训。法律、医疗和科学文本常需领域微调关系抽取模型。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 快生产,通用领域 | REBEL或LlamaPred配Wikidata规范化 |
| 领域特定(生物医学、法律) | SciREX风格领域微调+定制本体 |
| 大语言模型提示,审计输出 | AEVS管道:锚定→抽取→验证→补充 |
| 高量新闻信息抽取 | 模式基+监督混合 |
| 从零建知识图谱 | 开放IE+手工规范化遍 |
| 时序知识图谱 | 抽取配限定符(开始/结束时间、时间点) |

集成模式:命名实体识别→共指→实体链接→关系抽取→本体映射→图加载。每阶段是潜在质量门。

## 产出成果

存`outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: 设计配溯源和规范化的关系抽取管道。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定语料(领域、语言、量)和下游用(知识图谱RAG、分析、合规),输出:

1. 抽取器。模式基/监督/大语言模型/AEVS混合。理由绑精确vs召回目标。
2. 本体。封闭属性列表(Wikidata/领域)或开放IE配规范化遍。
3. 溯源。每三元组带源字符跨度+文档id。审计不可协商。
4. 合并策略。规范实体id+关系id+时序限定符;去重策略。
5. 评估。200手标三元组精确/召回+大语言模型抽取样本幻觉率。

拒绝无跨度验证(源溯源)大语言模型基关系抽取管道。拒绝无规范化开放IE输出流入生产图。标记无时序限定符时间有界关系(雇主、配偶、职位)管道。
```

## 练习题

1. **简单。** 5新闻文章句跑`code/main.py`模式抽取器。手查精确。
2. **中等。** 同句用REBEL(或小大语言模型)。比三元组。哪个抽取器精确高?召回高?
3. **困难。** 构AEVS管道:大语言模型抽取+验证跨度对源。测50 Wikipedia风格句验证步前后幻觉率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 三元组 | 主体-关系-客体 | `(s, r, o)`元组,知识图谱原子单位。 |
| 开放IE | 抽任何 | 开放词汇关系短语;高召回,低精确。 |
| 封闭本体 | 固定模式 | 有界关系类型集(Wikidata、UMLS、FIBO)。 |
| 规范化 | 归一一切 | 映射表面名/关系到规范id。 |
| AEVS | 锚定抽取 | 锚定-抽取-验证-补充管道(2026)。 |
| 溯源 | 真值源链接 | 每三元组带文档id+字符跨度到源。 |
| 远监督 | 廉价标签 | 对齐文本配现有知识图谱创训练数据。 |

## 延伸阅读

- [Mintz等(2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf)——远监督论文。
- [Huguet Cabot, Navigli(2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf)——seq2seq关系抽取主力。
- [Wadden等(2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546)——联合信息抽取。
- [AEVS——Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178)——2026幻觉缓解设计。
- [Wikidata SPARQL教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial)——规范图查询。