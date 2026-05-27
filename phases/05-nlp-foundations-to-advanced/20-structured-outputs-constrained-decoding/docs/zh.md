# 结构化输出与约束解码

> 问大语言模型要JSON。大多数时间得JSON。生产中,"大多数"是问题。约束解码通过采样前编辑logits把"大多数"变"总是"。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程17(聊天机器人)、阶段5课程19(子词词元化)
**时间:** ~60分钟

## 问题背景

分类器提示大语言模型:"返{positive, negative, neutral}之一。"模型返"情感是positive——此评论明显有利因客户显式说他们..."。解析器崩。分类器F1是0.0。

自由生成非契约。是建议。生产系统需契约。

2026存在三层:

1. **提示词。** 好好问。"仅返JSON对象。"前沿模型工作~80%,小模型少。
2. **原生结构化输出API。** OpenAI `response_format`、Anthropic工具用、Gemini JSON模式。支持模式可靠。供应商锁。
3. **约束解码。** 每生成步修改logits使模型*不能*发无效词元。构造100%有效。任何本地模型工作。

本课构建三者直觉并命名何时用哪个。

## 概念讲解

![约束解码每步遮蔽无效词元](../assets/constrained-decoding.svg)

**约束解码如何工作。** 每生成步,大语言模型产全词汇(~100k词元)上logit向量。*logit处理器*坐模型和采样器间。它算目标语法中当前位置哪些词元有效——JSON Schema、正则、上下文无关语法——并设所有无效词元logits负无穷。剩余logits上softmax只放概率质量在有效延续。

2026实现:

- **Outlines。** 编JSON Schema或正则成有限状态机。每词元得O(1)有效下词元查找。FSM基,故递归模式需扁平。
- **XGrammar/llguidance。** 上下文无关语法引擎。处理递归JSON Schema。近零解码开销。OpenAI2025结构化输出实现中承认llguidance。
- **vLLM引导解码。** 内置`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`通过Outlines、XGrammar或lm-format-enforcer后端。
- **Instructor。** 任何大语言模型Pydantic基包装器。验证失败重试。跨供应商,但不修改logits——依赖重试+结构化输出感知提示词。

### 反直觉结果

约束解码常比无约束生成*更快*。两原因。首先,它缩下词元搜索空间。其次,聪明实现为强制词元(脚手架如`{"name": "`——每字节定)完全跳词元生成。

### 成本陷阱

字段顺序重要。`answer`放`reasoning`前,模型思考前答。JSON有效。答案错。无验证捕获。

```json
// 坏
{"answer": "yes", "reasoning": "because ..."}

// 好
{"reasoning": "... therefore ...", "answer": "yes"}
```

模式字段顺序是逻辑,非格式。

## 动手实践

### Step 1:从零正则约束生成

见`code/main.py`独立FSM实现。核心思想30行:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM跟踪语法目前为止满意部分。`valid_tokens(state, tokenizer)`算哪些词汇词元可推进FSM不离接受路径。

### Step 2:Outlines做JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错。永。FSM使无效输出不可达。

### Step 3:Instructor做供应商无关Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同机制。Instructor不触logits。它格式模式进提示词、解析输出、验证失败重试(默认3次)。任何供应商工作。重试加延迟和成本。跨供应商移植性是卖点。

### Step 4:原生供应商API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。支持模式与Outlines可靠性平。无本地模型管理。锁供应商。

## 陷阱

- **递归模式。** Outlines扁平递归到固定深。树结构输出(嵌套评论、AST)需XGrammar或llguidance(CFG基)。
- **巨大枚举。** 10,000选枚举编译慢或超时。切换检索器:先预测top-k候选,约束那些。
- **语法太严。** 强`date: "YYYY-MM-DD"`正则模型不能输`"unknown"`缺日期。模型补偿编日期。允`null`或哨兵。
- **早承诺。** 见上字段顺序陷阱。总先放推理。
- **供应商JSON模式无模式。** 纯JSON模式只保证有效JSON,非*你用例*有效。总提供全模式。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| OpenAI/Anthropic/Google模型,简单模式 | 原生供应商结构化输出 |
| 任何供应商,Pydantic工作流,可容忍重试 | Instructor |
| 本地模型,需100%有效,扁模式 | Outlines(FSM) |
| 本地模型,递归模式 | XGrammar或llguidance |
| 自托管推理服务器 | vLLM引导解码 |
| 批处理重试可接受 | Instructor+最便宜模型 |

## 产出成果

存`outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: 选结构化输出方法、模式设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定用例(供应商、延迟预算、模式复杂度、失败容忍),输出:

1. 机制。原生供应商结构化输出、Instructor重试、Outlines FSM或XGrammar CFG。一句话理由。
2. 模式设计。字段顺序(推理先,答案后)、"未知"可空字段、枚举vs正则、必填字段。
3. 失败策略。最大重试、回退模型、优雅`null`处理、分布外拒绝。
4. 验证计划。模式合规率(目标100%)、语义有效性(大语言模型评判)、字段覆盖率、延迟p50/p99。

拒绝任何`answer`或`decision`放推理字段前设计。拒绝无模式用裸JSON模式。标记FSM仅库后递归模式。
```

## 练习题

1. **简单。** 无约束解码提示小开源模型(如Llama-3.2-3B)求`Review(sentiment, confidence, evidence_span)`。测100评论解析为有效JSON分数。
2. **中等。** 同语料库Outlines JSON模式。比合规率、延迟和语义准确度。
3. **困难。** 从零实现电话号码(`\d{3}-\d{3}-\d{4}`)正则约束解码器。验证1000样本0无效输出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 约束解码 | 强有效输出 | 每生成步遮蔽无效词元logits。 |
| Logit处理器 | 约束的东西 | 函数:`(logits, state)→masked_logits`。 |
| FSM | 有限状态机 | 编译语法表示;O(1)有效下词元查找。 |
| CFG | 上下文无关语法 | 处理递归语法;比FSM慢但更表达。 |
| 模式字段顺序 | 重要吗? | 是——首字段承诺;总推理前答案。 |
| 引导解码 | vLLM名 | 同概念,集成推理服务器。 |
| JSON模式 | OpenAI早版 | 保证JSON语法;不保证模式匹配。 |

## 延伸阅读

- [Willard, Louf(2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702)——Outlines论文。
- [XGrammar论文(2024)](https://arxiv.org/abs/2411.15100)——快CFG基约束解码。
- [vLLM—Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)——推理服务器集成。
- [OpenAI—Structured Outputs指南](https://platform.openai.com/docs/guides/structured-outputs)——API参考+陷阱。
- [Instructor库](https://python.useinstructor.com/)——Pydantic+跨供应商重试。
- [JSONSchemaBench(2025)](https://arxiv.org/abs/2501.10868)——基准6约束解码框架。