# 共指消解

> "她叫了他。他没有接听。医生正在吃午饭。"三次提到两个人，却没有名字。共指消解弄清谁是谁。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程06(命名实体识别)、阶段5课程07(词性标注与解析)
**时间:** ~60分钟

## 问题背景

从300词文章提取Apple Inc.每提及。文章说"Apple"易。说"公司"、"他们"、"库比蒂诺科技巨头"或"乔布斯公司"难。无消解这些提及到同实体,命名实体识别管道漏60-80%提及。

共指消解链接每指代同真实世界实体表达式成一聚类。是表面级自然语言处理(命名实体识别、解析)与下游语义(IE、问答、摘要、KG)间胶。

2026为何重要:

- 摘要:"CEO宣布..."vs"Tim Cook宣布..."——摘要应命名CEO。
- 问答:"她叫谁?"需消解"她"。
- 信息抽取:知识图谱"PER1创立Apple"和"Jobs创立Apple"作分开条目错。
- 多文档IE:合并跨文章同事件提及是跨文档共指。

## 概念讲解

![共指聚类:提及→实体](../assets/coref.svg)

**任务。** 输入:文档。输出:提及(跨度)聚类,每聚类指一实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性。** "CEO"、"公司"
- **代词性。** "他"、"她"、"他们"、"它"
- **同位语。** "Tim Cook, Apple的CEO,"

**架构。**

1. **规则基(Hobbs, 1978)。** 语法树基代词消解用语法规则。好基线。代词上意外难胜。
2. **提及对分类器。** 每对提及(m_i, m_j)预测是否共指。传递闭包聚类。2016前标准。
3. **提及排名。** 每提及排候选先行词(含"无先行词")。选顶。
4. **跨度基端到端(Lee等, 2017)。** Transformer编码器。枚举所有候选跨度到长度帽。预测提及分数。每跨度预测先行词概率。贪婪聚类。现代默认。
5. **生成式(2024+)。** 提示大语言模型:"列出本文每代词及其先行词。"易例工作好,长文档和稀有指代挣扎。

**评估指标。** 五标准指标(MUC、B³、CEAF、BLANC、LEA)因无单指标捕获聚类质量。报前三平均作CoNLL F1。2026 CoNLL-2012 SOTA:~83 F1。

**已知难例。**

- 定描述指代数页前引入实体。
- 桥接照应("轮子"→前提车)。
- 中文日语零照应。
- 前照应(代词在指代前):"当**她**走进,玛丽微笑。"

## 动手实践

### Step 1:预训练神经共指(AllenNLP/spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

长文档,得如:
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### Step 2:规则基代词消解器(教学)

见`code/main.py`仅stdlib实现:

1. 提提及:命名实体(大写跨度)、代词(字典查)、定描述("X")。
2. 每代词,看前K提及评分:
   - 性/数一致(启发)
   - 近性(近赢)
   - 语法角色(主语偏好)
3. 链最高分先行词。

不竞争神经模型。但显搜索空间和端到端模型须决策。

### Step 3:用大语言模型共指

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两失败模式盯。首先,大语言模型过合并("他"和"她"指两不同人)。其次,大语言模型长文档静默丢提及。总验证跨度偏检查。

### Step 4:评估

标准conll-2012脚本算MUC、B³、CEAF-φ4并报平均。内评估,从标注测试集跨度级精确率召回开始,再加提及链接F1。

## 陷阱

- **单例爆炸。** 有些系统报每提及作自己聚类。B³宽容。MUC惩罚。总检查三指标。
- **长上下文代词。** 超2,000词元文档性能降~15 F1。小心分块。
- **性别假设。** 硬编码性别规则非二元指代、组织、动物破。用学习模型或中性评分。
- **大语言模型长文档漂移。** 单API调用不能可靠跨50+段聚类提及。用滑窗+合并。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 英文,单文档 | `en_coreference_web_trf`(spaCy-experimental)或AllenNLP神经共指 |
| 多语言 | SpanBERT/XLM-R在OntoNotes或多语言CoNLL训 |
| 跨文档事件共指 | 专门端到端模型(2025–26 SOTA) |
| 快大语言模型基线 | GPT-4o/Claude配结构化输出共指提示词 |
| 生产对话系统 | 规则基回退+神经主+关键槽人工审 |

2026发货集成模式:先命名实体识别,跑共指,合共指聚类进命名实体识别实体。下游任务见每聚类一实体,非每提及一实体。

## 产出成果

存`outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: 选共指方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例(单文档/多文档、领域、语言),输出:

1. 方法。规则基/神经跨度基/大语言模型提示/混合。一句话理由。
2. 模型。神经时命名检查点。
3. 集成。操作顺序:分词→命名实体识别→共指→下游任务。
4. 评估。保留集CoNLL F1(MUC+B³+CEAF-φ4平均)+20文档手工聚类审。

拒绝无滑窗合并超2,000词元文档大语言模型仅共指。拒绝无提及级精确召回报告跑共指管道。标记部署于人口多样文本性别启发系统。
```

## 练习题

1. **简单。** 5手编段跑`code/main.py`规则基消解器。测对真值提及链准确率。
2. **中等。** 新闻文章用预训练神经共指模型。比你手工标注聚类。哪失败?
3. **困难。** 构共指增强命名实体识别管道:先命名实体识别,再经共指聚类合并。测100文章实体覆盖改进vs仅命名实体识别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 提及 | 一个引用 | 指实体文本跨度(名、代词、名词短语)。 |
| 先行词 | "它"指什么 | 后提及共指的前提及。 |
| 聚类 | 实体提及 | 所有指同真实世界实体提及集。 |
| 照应 | 后引用 | 后提及指前("他"→"John")。 |
| 前照应 | 前引用 | 前提及指后("当他到达,John...")。 |
| 桥接 | 隐引用 | "我买车。轮子坏。"(那车的轮子。) |
| CoNLL F1 | 排行榜数 | MUC、B³、CEAF-φ4 F1分数平均。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26—Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf)——权威教科书章。
- [Lee等(2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045)——跨度基端到端。
- [Joshi等(2020). SpanBERT](https://arxiv.org/abs/1907.10529)——改进共指预训练。
- [Pradhan等(2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/)——基准。
- [Hobbs(1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064)——规则基经典。