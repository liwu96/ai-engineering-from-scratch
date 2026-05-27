# 实体链接与消歧

> 命名实体识别找"Paris"。实体链接决定:巴黎法国?Paris Hilton?巴黎德州?Paris(特洛伊王子)?无链接,知识图谱保持歧义。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程06(命名实体识别)、阶段5课程24(共指消解)
**时间:** ~60分钟

## 问题背景

句子读:"Jordan beat the press."命名实体识别标"Jordan"为PERSON。好。但哪个Jordan?

- Michael Jordan(篮球)?
- Michael B. Jordan(演员)?
- Michael I. Jordan(Berkeley ML教授——ML论文中这混淆真)?
- Jordan(国家)?
- Jordan(希伯来名)?

实体链接(EL)解每提及到知识基唯一条目:Wikidata、Wikipedia、DBpedia或你领域KB。两子任务:

1. **候选生成。** 给"Jordan",哪些KB条目合理?
2. **消歧。** 给上下文,哪候选正确?

两步可学习。两步基准。组合管道十年稳——变的是消歧器质量。

## 概念讲解

![实体链接管道:提及→候选→消歧实体](../assets/entity-linking.svg)

**候选生成。** 给提及表面形("Jordan"),别名索引查候选。Wikipedia别名字典覆大多命名实体:"JFK"→John F. Kennedy、Jacqueline Kennedy、JFK机场、JFK(电影)。典型索引每提及返10-30候选。

**消歧:三方法。**

1. **先验+上下文(Milne & Witten, 2008)。** `P(entity | mention) × context-similarity(entity, text)`。工作好,快,无训练。
2. **嵌入基(ESS/REL/Blink)。** 编提及+上下文。编每候选描述。选最大余弦。2020-2024默认。
3. **生成式(GENRE, 2021; 大语言模型基, 2023+)。** 词元逐词元解码实体规范名。约束到有效实体名trie使输出保证有效KB id。

**端到端vs管道。** 现代模型(ELQ、BLINK、ExtEnD、GENRE)一跑命名实体识别+候选生成+消歧。管道系统仍主导生产因可换组件。

### 两测量

- **提及召回(候选生成)。** 金提及中正确KB条目出现在候选列表分数。全管道底。
- **消歧准确率/F1。** 给正确候选,top-1对多少。

总报两。80%候选召回上99%消歧系统是80%管道。

## 动手实践

### Step 1:从Wikipedia重定向构别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia别名数据:~18M(别名,实体)对。Wikidata dumps下载。存倒索引。

### Step 2:上下文基消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard重叠玩具。替换嵌入向量余弦相似度(见`code/main.py` step-2 transformer版)。

### Step 3:嵌入基(BLINK风格)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时,每KB实体嵌入一次。查询时,提及+上下文嵌入一次,候选池点积,选最大。

### Step 4:生成实体链接(概念)

GENRE字符逐字符解码实体Wikipedia标题。约束解码(见课程20)保仅有效标题可输出。KB后备trie紧密集成。现代后继REL-GEN和大语言模型提示EL配结构输出。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{\"title\": \"...\"}}"""
```

配白名单(Outlines `choice`),这是2026最简EL管道发货。

### Step 5:AIDA-CoNLL评估

AIDA-CoNLL标准EL基准:1,393 Reuters文章,34k提及,Wikipedia实体。报库内准确率(`P@1`)和库外NIL检测率。

## 陷阱

- **NIL处理。** 有些提及不在KB(新兴实体、不明人)。系统须预测NIL而非猜错实体。分开测。
- **提及边界错。** 上游命名实体识别漏部分跨度("Bank of America"仅标"Bank")。EL召回降。
- **流行偏。** 训系统过预测频实体。ML论文上"Michael I. Jordan"提及常链篮球Jordan。
- **跨语言EL。** 中文文本提及映射英文Wikipedia实体。需多语言编码器或翻译步。
- **KB陈旧。** 新公司、事件、人不在去年Wikipedia dump。生产管道需刷新循环。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 通用英文+Wikipedia | BLINK或REL |
| 跨语言,KB=Wikipedia | mGENRE |
| 大语言模型友好,少提及/天 | 提示Claude/GPT-4配候选列表+约束JSON |
| 领域特定KB(医疗、法律) | 定制BERT配KB感知检索+领域AIDA风格集微调 |
| 极低延迟 | 仅精确匹配先验(Milne-Witten基线) |
| 研究SOTA | GENRE/ExtEnD/生成大语言模型EL |

2026发货生产模式:命名实体识别→共指→每提及EL→折叠聚类到每聚类一规范实体。输出:文档每实体一KB id,非每提及一。

## 产出成果

存`outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: 设计实体链接管道——KB、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定用例(领域KB、语言、量、延迟预算),输出:

1. 知识基。Wikidata/Wikipedia/定制KB。版本日期。刷新周期。
2. 候选生成器。别名索引、嵌入或混合。目标提及召回@K。
3. 消歧器。先验+上下文、嵌入基、生成式或大语言模型提示。
4. NIL策略。顶分数阈值、分类器或显NIL候选。
5. 评估。提及召回@30、top-1准确率、保留集NIL检测F1。

拒绝无提及召回基线EL管道(候选生成浮出正确实体才能评估消歧器)。拒绝无约束输出有效KB id大语言模型提示EL管道。标记流行偏影响少数实体(如名冲突)无领域微调系统。
```

## 练习题

1. **简单。** `code/main.py`10歧义提及(Paris、Jordan、Apple)实现先验+上下文消歧器。手标正确实体。测准确率。
2. **中等。** 50歧义提及句子Transformer编码。每候选描述嵌入。比嵌入基消歧Jaccard上下文重叠。
3. **困难。** 构1k实体领域KB(如公司员工+产品)。实现命名实体识别+EL端到端。测100保留句精确召回。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 实体链接(EL) | 链到Wikipedia | 映射提及到唯一KB条目。 |
| 候选生成 | 可能是谁? | 返提及可能KB条目短列表。 |
| 消歧 | 选对的 | 用上下文评分候选,选胜者。 |
| 别名索引 | 查表 | 从表面形→候选实体映射。 |
| NIL | 不在KB | 显预测无KB条目匹配。 |
| KB | 知识基 | Wikidata、Wikipedia、DBpedia或你领域KB。 |
| AIDA-CoNLL | 基准 | 1,393 Reuters文章配金实体链接。 |

## 延伸阅读

- [Milne, Witten(2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf)——基础先验+上下文方法。
- [Wu等(2020). Zero-shot Entity Linking with Dense Entity Retrieval(BLINK)](https://arxiv.org/abs/1911.03814)——嵌入基主力。
- [De Cao等(2021). Autoregressive Entity Retrieval(GENRE)](https://arxiv.org/abs/2010.00904)——配约束解码生成EL。
- [Hoffart等(2011). Robust Disambiguation of Named Entities in Text(AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf)——基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants(2020)](https://arxiv.org/abs/2006.01969)——开源生产栈。