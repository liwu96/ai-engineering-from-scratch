# 结构输出——JSON Schema、Pydantic、Zod、约束解码

> "请模型好回JSON"即使前沿模型失败率为5%至15%。结构输出用约束解码闭该差距:模型字面禁发违schema的token。OpenAI严格模式、Anthropic schema类型工具用、Gemini`responseSchema`、Pydantic AI`output_type`、Zod`.parse`是同念五面形。本课建schema验器和严格模式契约学习者将用于每产提取管道。

**类型:** 构建
**语言:** Python(stdlib,JSON Schema 2020-12子集)
**前置要求:** 阶段13课程02(函数调用深究)
**时间:** ~75分钟

## 学习目标

- 写JSON Schema 2020-12用于提取目标用正约束(enum、min/max、required、pattern)。
- 释为何严格模式和约束解码给不同于"生成后验"的保证。
- 分三失模式:解析错、schema违、模型拒。
- 发带类型修和类型拒处的提取管道。

## 问题背景

Agent读采购订单email需将自由文本转`{customer, line_items, total_usd}`。三法。

**法一:提示JSON。**"JSON回复带域customer、line_items、total_usd。"前沿模型工85至95%。六式失:缺括号、尾逗号、错类型、幻觉域、token限截断、"这是你JSON:"等散文漏。

**法二:生成后验。**自由生成、解析、验schema、失败重试。可靠但贵——每重试付费,截断bug每次一轮。

**法三:约束解码。**提供者解码时执schema。无效token从采样分布mask出。输出保解析保验。失塌至一模:拒(模型决定输入不合schema)。

2026每前沿提供者发某种法三。

- **OpenAI。**`response_format: {type: "json_schema", strict: true}`加响应`refusal`若模型拒。
- **Anthropic。**`tool_use`输入schema执;`stop_reason: "refusal"`非事,但`end_turn`无工具调用是信号。
- **Gemini。**请求级`responseSchema`;2026 Gemini发选定类型token级语法约束。
- **Pydantic AI。**`output_type=InvoiceModel`发结构`RunResult`类型化`InvoiceModel`。
- **Zod(TypeScript)。**运行时解析器验提供者输出对Zod schema;配OpenAI`beta.chat.completions.parse`。

共线:声明schema一次,端到端执。

## 概念讲解

### JSON Schema 2020-12——通用语

每提供者接JSON Schema 2020-12。最常用构造:

- `type`:一于`object`、`array`、`string`、`number`、`integer`、`boolean`、`null`。
- `properties`:域名至子schema映。
- `required`:须现域名列表。
- `enum`:允许值闭集。
- `minimum`/`maximum`(数)、`minLength`/`maxLength`/`pattern`(字符串)。
- `items`:每数组元素子schema。
- `additionalProperties`: `false`禁额外域(默认因模式异)。

OpenAI严格模式加三求:每属性须列`required`、`additionalProperties: false`处处、无未解`$ref`。若破此,API请求时回400。

### Pydantic,Python绑定

Pydantic v2经`model_json_schema()`从dataclass形模型生成JSON Schema。Pydantic AI包此你写:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent框架将schema译入OpenAI严格模式、Anthropic`input_schema`或Gemini`responseSchema`于边。模型输出回类型化`Invoice`实例。验错raise带类型错路径`ValidationError`。

### Zod,TypeScript绑定

Zod(`z.object({customer: z.string(), ...})`)是TS等价。OpenAI Node SDK露`zodResponseFormat(Invoice)`译入API JSON Schema payload。

### Refusal

严格模式不可强模型答。若输入不合schema("email是诗非订单"),模型发`refusal`域含因。你代码须处此为一等结果而非失。拒亦有用作安全信号:模型问从保护内容email提取信用卡号回带安全因拒。

### 开源约束解码

开权重实现用三技。

1. **语法基解码**(`outlines`、`guidance`、`lm-format-enforcer`):从schema建确定性有限自动机;每步mask会违FSM的token logits。
2. **Logit mask带JSON解析器**:与模型锁步跑流JSON解析器;每步算有效下一token集。
3. **推测解码带验器**:便宜draft模型提token,验器执schema。

商提供者幕后择一。2026态是短结构输出比纯生成快,长结构约同速。

### 三失模式

1. **解析错。**输出非有效JSON。严格模式不可能。非严格提供者仍可。
2. **Schema违。**输出解析但违schema。严格模式不可能。外常见。
3. **拒。**模型拒。须类型化结果处。

### 重试策略

外严格模式(Anthropic工具用、非严格OpenAI、旧Gemini),恢复模式:

```
生成 -> 解析 -> 验 -> 若败,注入错并重试,最多3x
```

一重试通常够。三重试捕弱模型flake。超三是坏schema信号:模型某些输入无法满足,提示或schema需修。

### 小模型支持

约束解码工于小模型。3B参数开模型带语法执胜70B参数模型带原始提示于结构任务。这是产结构输出主因:它解耦可靠性和模型大小。

## 使用

`code/main.py`发stdlib最小JSON Schema 2020-12验器(类型、required、enum、min/max、pattern、items、additionalProperties)。它包`Invoice`schema并跑假LLM输出经验器,示解析错、schema违、拒路径。换假输出为任提供者真实响应于产。

看点:

- 验器回类型化`[ValidationError]`列表带路径和消息。那是你想露至重试提示的形。
- 拒分支不重试。它记录并回类型化拒。阶段14课程09用拒作安全信号。
- `additionalProperties: false`查于对抗测试输入发,示为何严格模式闭门于幻觉域。

## 交付成果

本课产`outputs/skill-structured-output-designer.md`。给自由文本提取目标(订单、支持票、简历等),skill产JSON Schema 2020-12兼容严格模式和Pydantic模型镜像之,带类型化拒和重试stub处。

## 练习题

1. 跑`code/main.py`。加第四测试案其`total_usd`是负数。确验器用`minimum`约束路径拒。

2. 扩验器支持带discriminator的`oneOf`。常见案:`line_item`是产品或服务,标签`kind`。严格模式此有微妙规;查OpenAI结构输出指南。

3. 写同Invoice schema为Pydantic BaseModel并比`model_json_schema()`输出与你手roll schema。识Pydantic默认设一域手roll版漏。

4. 测拒率。构十不应可提取输入(歌词、数学证明、空email)并经带严格模式真实提供者跑。计拒vs幻觉输出。这是你拒知重试基线真。

5. 读OpenAI结构输出指南从头至尾。识其显禁严格模式一构JSON Schema允许。后设计用该禁构非本质schema并重构为严格兼容。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| JSON Schema 2020-12 | "Schema spec" | 每现代提供者说的IETF-draft schema方言 |
| 严格模式 | "保schema" | OpenAI旗经约束解码执schema |
| 约束解码 | "Logit mask" | 解码时执mask无效下一token |
| 拒 | "模型拒" | 输入不合schema时类型化结果 |
| 解析错 | "无效JSON" | 输出未解析为JSON;严格不可能 |
| Schema违 | "错形" | 解析但违类型/required/enum/范围 |
| `additionalProperties: false` | "无额外允许" | 禁未知域;OpenAI严格需 |
| Pydantic BaseModel | "类型输出" | 发和验JSON Schema的Python类 |
| Zod schema | "TypeScript输出类型" | TS运行时schema用于提供者输出验 |
| 语法执 | "开权重约束解码" | FSM基logit mask,如outlines/guidance |

## 延伸阅读

- [OpenAI—Structured outputs](https://platform.openai.com/docs/guides/structured-outputs)——严格模式、拒和schema需
- [OpenAI—Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)——2024年8月发帖解释解码保证
- [Pydantic AI—Output](https://ai.pydantic.dev/output/)——类型化output_type绑定序列化至每提供者
- [JSON Schema—2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes)——规范spec
- [Microsoft—Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)——企业部署笔记和严格模式警