# 函数调用深究——OpenAI、Anthropic、Gemini

> 三前沿提供者2024收敛于同工具调用循环后于余皆异。OpenAI用`tools`和`tool_calls`。Anthropic用`tool_use`和`tool_result`块。Gemini用`functionDeclarations`和unique-id关。本课并排diff三使代码发于一提供者移植他时不破。

**类型:** 构建
**语言:** Python(stdlib,schema翻译器)
**前置要求:** 阶段13课程01(工具接口)
**时间:** ~75分钟

## 学习目标

- 述OpenAI、Anthropic、Gemini函数调用payload三形差(声明、调用、结果)。
- 翻译一工具声明跨三提供者格式并预严格模式约束何异。
- 用每提供者`tool_choice`强、禁或自择工具调用。
- 知每提供者硬限(工具数、schema深度、参数长度)及违限时发错签名。

## 问题背景

函数调用请求形因提供者异。2026产栈三例:

**OpenAI Chat Completions/Responses API。**传`tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应含`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`其中`arguments`是JSON字符串须解析。严格模式(`strict: true`)经约束解码执schema合规。

**Anthropic Messages API。**传`tools: [{name, description, input_schema}]`。响应回`content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input`已解析(对象,非字符串)。回新`user`消息含`{type: "tool_result", tool_use_id, content}`块。

**Google Gemini API。**传`tools: [{functionDeclarations: [{name, description, parameters}]}]`(嵌`functionDeclarations`)。响应到`candidates[0].content.parts: [{functionCall: {name, args, id}}]`其中`id`Gemini 3起unique用于并行调用关。回`{functionResponse: {name, id, response}}`。

同循环。异域名、异嵌套、异string-vs-object约定、异关机制。写天气agent于OpenAI的团队付两天移植Anthropic又一天Gemini仅管道。

本课建翻译器统三格式入一规范工具声明并于边路由。阶段13课程17泛化同模式入LLM gateway。

## 概念讲解

### 共结构

每提供者需五物:

1. **工具列表。**每工具名、描述、输入schema。
2. **工具择。**强特定工具、禁工具或让模型决定。
3. **调用发射。**结构输出命名工具和参数。
4. **调用id。**关响应至正调用(并行重要)。
5. **结果注入。**消息或块将结果绑回调用。

### 域域形差

| 面 | OpenAI | Anthropic | Gemini |
|----|--------|-----------|--------|
| 声明包 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema域 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant消息上`tool_calls[]` | 类型`tool_use`的`content[]` | 类型`functionCall`的`parts[]` |
| 参数类型 | JSON字符串化 | 解析对象 | 解析对象 |
| Id格式 | `call_...`(OpenAI生成) | `toolu_...`(Anthropic) | UUID(Gemini 3+) |
| 结果块 | 角色`tool`,`tool_call_id` | `user`带`tool_result`,`tool_use_id` | 带匹配`id`的`functionResponse` |
| 强工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格schema | `strict: true` | schema即schema(总执) | 请求级`responseSchema` |

### 你会实撞限

- **OpenAI。**每请求128工具。Schema深度5。参数字符串<=8192字节。严格模式需无`$ref`、无重叠`oneOf`/`anyOf`/`allOf`、每属性列`required`。
- **Anthropic。**每请求64工具。Schema深度效无界但实限10。无严格模式旗;schema是契约模型倾向合规。
- **Gemini。**每请求64函数。Schema类型是OpenAPI 3.0子集(JSON Schema 2020-12略异)。Gemini 3起并行调用unique-id。

### `tool_choice`行为

三模每支持,异名。

- **Auto。**模型择工具或文本。默认。
- **Required/Any。**模型须调至少一工具。
- **None。**模型勿调工具。

加每提供者独一模:

- **OpenAI。**按名强特定工具。
- **Anthropic。**按名强特定工具;`disable_parallel_tool_use`旗分单vs多。
- **Gemini。**`mode: "VALIDATED"`路由每响应经schema验器无论模型意图。

### 并行调用

OpenAI`parallel_tool_calls: true`(默认)一发多调用于assistant消息。你全跑并回批工具角色消息含每`tool_call_id`一条。Anthropic历史单调;`disable_parallel_tool_use: false`(Claude 3.5默认)启多。Gemini 2允并行调用但不给稳定id;Gemini 3加UUID使乱序响应干净关。

### 流

三皆支持流工具调用。线格式异:

- **OpenAI。**`tool_calls[i].function.arguments`delta块增量到。你累积至`finish_reason: "tool_calls"`。
- **Anthropic。**Block-start/block-delta/block-stop事件。`input_json_delta`块载部分参数。
- **Gemini。**`streamFunctionCallArguments`(Gemini 3新)发带`functionCallId`块使多并行调用可交错。

阶段13课程03深究并行+流重组。本课聚焦声明和单调用形。

### 错和修

无效参数错看亦异。

- **OpenAI(非严格)。**模型回`arguments: "{bad json}"`,你JSON解析失败,你注入错消息并重调。
- **OpenAI(严格)。**验于解码间;无效JSON不可能但`refusal`可现。
- **Anthropic。**`input`可含意外域;schema是建议。验服务端。
- **Gemini。**OpenAPI 3.0怪:对象域上`enum`静默忽略;自验。

### 翻译器模式

你代码中规范工具声明看此(你择形):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三小函数翻它至三提供者形。`code/main.py`harness正做此,后round-trip假工具调用经每提供者响应形。无网络需——本课教形,非HTTP。

产团队包此翻译器入`AbstractToolset`(Pydantic AI)、`UniversalToolNode`(LangGraph)或`BaseTool`(LlamaIndex)。阶段13课程17发gateway露OpenAI形API前三任一前。

## 使用

`code/main.py`定义规范`Tool`dataclass和三翻译器发OpenAI、Anthropic、Gemini声明JSON。后解析手制每形提供者响应入同规范调用对象,示语义皮下一致。跑它并排diff三声明。

看点:

- 三声明块仅包和域名异。
- 三响应块异于调用居处(顶级`tool_calls`、`content[]`块、`parts[]`条)。
- 一`canonical_call()`函数从三响应形提取`{id, name, args}`。

## 交付成果

本课产`outputs/skill-provider-portability-audit.md`。给一提供者函数调用集成,skill产移植审计:赖何提供者限、何域需改名、移植每他提供者时何破。

## 练习题

1. 跑`code/main.py`验三提供者声明JSON全序列化同底层`Tool`对象。改规范工具加enum参数确仅Gemini翻译器需处OpenAPI怪。

2. 加每提供者`ListToolsResponse`解析器提取模型`list_tools`或发现调用后返工具列表。OpenAI无原生;记此不对称。

3. 实`tool_choice`转换:将规范`ToolChoice(mode="force", tool_name="x")`映入三提供者形。后映`mode="any"`和`mode="none"`。查课diff表。

4. 择三提供者之一读其函数调用指南从头至尾。找其schema规一域他二不支持。候选:OpenAI`strict`、Anthropic`disable_parallel_tool_use`、Gemini`function_calling_config.allowed_function_names`。

5. 写测试向量:工具调用其参数违声明schema。经每提供者验器(课程01stdlib版可作代理)并录何错发。文档产你会用何提供者严格。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 函数调用 | "工具用" | 结构工具调用发射的提供者级API |
| 工具声明 | "工具spec" | 名+描述+JSON Schema输入payload |
| `tool_choice` | "强/禁" | Auto/required/none/特定名模式 |
| 严格模式 | "Schema执" | OpenAI旗约束解码匹配schema |
| `tool_use`块 | "Anthropic调用形" | 内联内容块带id、name、input |
| `functionCall`部分 | "Gemini调用形" | 含name、args、id的`parts[]`条 |
| 参数-as-string | "JSON字符串化" | OpenAI回args为JSON字符串,非对象 |
| 并行工具调用 | "一轮fan-out" | 一assistant消息多工具调用 |
| Refusal | "模型拒" | 严格模式独有拒绝块替调用 |
| OpenAPI 3.0子集 | "Gemini schema怪" | Gemini用JSON-Schema类方言带小差 |

## 延伸阅读

- [OpenAI—Function calling guide](https://platform.openai.com/docs/guides/function-calling)——含严格模式和并行调用的规范参考
- [Anthropic—Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)——`tool_use`和`tool_result`块语义
- [Google—Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)——并行调用、unique id和OpenAPI子集
- [Vertex AI—Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)——Gemini企业面
- [OpenAI—Structured outputs](https://platform.openai.com/docs/guides/structured-outputs)——严格模式schema执细节