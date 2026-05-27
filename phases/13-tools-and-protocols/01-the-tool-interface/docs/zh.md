# 工具接口——为何Agent需要结构化I/O

> 语言模型产生token。程序执行动作。两者间的差距是工具接口:让模型请求动作、宿主执行的契约。2026每个栈——OpenAI、Anthropic、Gemini的函数调用;MCP的`tools/call`;A2A的任务部分——都是同一四步循环的不同编码。本课命名该循环并示运行所需最小机制。

**类型:** 学习
**语言:** Python(stdlib,无LLM)
**前置要求:** 阶段11(LLM completion API)
**时间:** ~45分钟

## 学习目标

- 释为何仅生成文本的LLM本身无法对真实世界执行动作。
- 画四步工具调用循环(describe→decide→execute→observe)并命名每步所有者。
- 写三部分工具描述:名、JSON Schema输入、确定性执行器函数。
- 分纯工具和副作用工具并述分际安全意义。

## 问题背景

LLM输出下一token的概率分布。这是整个输出面。若问聊天模型"Bengaluru现天气如何",它可写合理句子,但无法调天气API。句子偶合正确或过时三天。

填补该差距是工具接口目的。宿主程序——你的agent runtime、Claude Desktop、ChatGPT、Cursor或自定义脚本——向模型广告可调工具列表。模型当决定需动作时,发出结构化payload命名工具及其参数。宿主解析payload,真运行工具,反馈结果。循环持续至模型决定无需更多调用。

该契约首版2023年6月发于OpenAI"functions"参数。Anthropic随Claude 2.1`tool_use`块。Gemini数月后加`functionDeclarations`。每提供者现露同形:JSON-Schema类型工具列表入,JSON payload工具调用出。Model Context Protocol(2024年11月)泛化契约使一工具注册服每模型。A2A(2026年4月,v1.0)为agent间委托叠同原语。

四步循环是其下不变量。阶段13余课是其展开。

## 概念讲解

### 步一:describe

宿主用三域声明每工具。

- **名。**稳定机读标识符。`get_weather`,非"天气物"。
- **描述。**一段自然语言简述。"当用户问特定城市现条件时用。勿用于历史数据。"
- **输入schema。**JSON Schema对象(draft 2020-12)述工具参数。

模型收列表。现代提供者用提供者特定模板将这些声明序列化入系统提示,故你作为调用者仅处理结构化形式。

### 步二:decide

给用户消息和可用工具,模型择三行为之一。

1. **直接文本回答。**无工具调用。
2. **调一或多工具。**发出结构化调用对象。`parallel_tool_calls: true`下(OpenAI和Gemini默认,Anthropic可选入)模型可一发多调用。
3. **拒绝。**严格模式结构输出可产类型化`refusal`块替调用。

工具调用payload有三稳定域:调用`id`、工具`name`、JSON`arguments`对象。id存在使宿主可将后结果与特定调用关联,这于并行调用乱序返回时重要。

### 步三:execute

宿主收调用,对声明schema验参数,运行执行器。无效参数意味模型幻觉域或用错类型——弱模型常见失模式。产宿主于无效参数做三事之一:快失败并向模型露错、用约束解析器修JSON、或含验错入提示重试模型。

执行器本身是普通代码。Python、TypeScript、shell命令、数据库查询。它产结果,通常是字符串但可是任JSON值或结构内容块(文本、图像或MCP资源引用)。结果须可序列化。

### 步四:observe

宿主将工具结果附至对话(为带匹配`id`的`tool`角色消息)并重调模型。模型现于上下文有工具输出可产终答案或请求更多调用。持续至模型停发调用或宿主达迭代计安全限。

### 信任分际

工具分两味安全有异。

- **纯。**只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。安全推测性调用。
- **后果性。**改态、花钱、触用户数据。`send_email`、`delete_file`、`execute_trade`。须门控。

Meta 2026 agent安全"二元律"说单轮可合最多二:不可信输入、敏感数据、后果动作。工具接口是你执法处——通过拒调用、需用户确认或升scope。见阶段13课程15全安全章和阶段14课程09 agent级权限策略。

### 循环居处

| 上下文 | 谁describe | decide | execute |
|--------|------------|---------|---------|
| 单轮函数调用(OpenAI/Anthropic/Gemini) | App开发者 | LLM | App开发者 |
| MCP | MCP server | LLM经MCP client | MCP server |
| A2A | Agent Card发布者 | 调agent | 被调agent |
| Web浏览器(函数调用agent) | 浏览器扩展/WebMCP | LLM | 浏览器runtime |

处处,同四步。列名变;结构不变。

### 何不只提示模型emit JSON?

"请模型JSON回复"是函数调用前模式。前沿模型失~5至15%,小模型更。失模式含缺括号、尾逗号、幻觉域、错类型。后需JSON修pass、重试或约束解码器。

原生函数调用优三因。首,提供者于精确调用形端到端训模型,故有效JSON率于严格模式升至98至99%。次,调用payload居己协议槽,非自由文本内——故工具调用永不漏入用户可见回复。三,提供者用约束解码执schema合规(OpenAI严格模式、Anthropic`tool_use`、Gemini`responseSchema`)。输出保验。

阶段13课程02走三提供者API并排。阶段13课程04深究结构输出。

### 熔断器

循环于模型停发调用或宿主达最大轮数终止。产宿主设此于5至20轮。超此,你几乎确定于模型无法退出循环。Claude Code默认20;OpenAI Assistants 10;Cursor agent模式25。

替代——无界循环——每六月显为"agent一夜花$400 API调用"事后析。勿无界发货。

阶段14课程12深覆盖错恢复和自愈;阶段17覆盖产速率限。

### 阶段13何往

- 课程02至05精提供者级工具调用面。
- 课程06至14泛化循环入MCP。
- 课程15至18御循环对恶意server、对抗用户、未认证远程auth面。
- 课程19至22扩模式至agent间协作、可观测、路由、打包。
- 课程23发用每原语完整生态。

余课都是四步循环展开。记住为不变量。

## 使用

`code/main.py`无LLM跑四步循环。假"decider"函数模拟模型通过匹配用户消息模式;执行器、schema验器、observe步 harness是真。跑它看完整请求/响应编排带可打印中间态,后换假decider为任真实提供者。

看点:

- 工具注册每工具持三域:名、描述、schema、执行器引用。
- 验器是最小JSON Schema子集(类型、required、enum、min/max)仅stdlib写。阶段13课程04发更全版。
- 循环于五限迭代计。产agent需正此熔断器。

## 交付成果

本课产`outputs/skill-tool-interface-reviewer.md`。给草稿工具定义(名+描述+schema+执行器轮廓),skill审其循环适配:名是否机稳定、描述是否完整用法简述、schema是否正确用JSON Schema 2020-12、纯vs后果分类是否显。

## 练习题

1. 加第四工具`get_stock_price(ticker)`入`code/main.py`。写描述"当用户问 ticker 现股价时用。勿用于历史价或市场总结。"跑harness确假decider路由提ticker查询至新工具。

2. 破schema验器。传调用其`arguments`对象缺required域,确宿主执前拒。后传调用带额外未知域。决定:宿主应拒或忽略?用安全论证理择。

3. 分类harness每工具为纯或后果性。加`consequential: true`旗至需它注册条目,改循环打"需用户确认"线每当后果工具择。这是每产宿主需确认门形状。

4. 纸上画四步循环填上表提供者列为你爱客户端(Claude Desktop、Cursor、ChatGPT或自定义栈)。交验阶段13课程06 MCP特版。

5. 读OpenAI函数调用指南从头至尾。识一域居请求而非四步循环所呈。释加何便而非必需。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 工具 | "模型可调物" | 名+JSON-Schema类型输入+执行器函数三元组 |
| 函数调用 | "原生工具用" | 提供者级API支持发出结构工具调用非散文 |
| 工具调用 | "模型动作请求" | 模型发出带`id`、`name`、`arguments`的JSON payload |
| 工具结果 | "工具返回" | 执行器输出,包入带匹配id的`tool`角色消息 |
| 并行工具调用 | "同时多调用" | 一模型轮多调用对象,独立且按id可排序 |
| 严格模式 | "保JSON" | 约束解码强模型输出验对声明schema |
| 纯工具 | "只读工具" | 无副作用;安全重跑 |
| 后果工具 | "动作工具" | 改外态;需门控、审计或用户确认 |
| 四步循环 | "工具调用周期" | describe→decide→execute→observe |
| 宿主 | "Agent runtime" | 持工具注册、调模型、运行执行器的程序 |

## 延伸阅读

- [OpenAI—Function calling guide](https://platform.openai.com/docs/guides/function-calling)——OpenAI式工具声明和调用形规范参考
- [Anthropic—Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)——Claude`tool_use`/`tool_result`块格式
- [Google—Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)——Gemini`functionDeclarations`和并行调用语义
- [Model Context Protocol—Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)——工具接口提供者无关泛化
- [JSON Schema—2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes)——每现代工具API说的schema方言