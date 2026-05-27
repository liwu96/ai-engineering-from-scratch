# MCP基础——原语、生命周期、JSON-RPC基

> MCP前每集成是特例。Model Context Protocol,Anthropic2024年11月初发现Linux Foundation Agentic AI Foundation托管,标准发现和调用使任客户端可对话任server。2025-11-25 spec命名六原语(三server、三client)、三phase生命周期和JSON-RPC 2.0线格式。学那些本phase MCP章余课成读。

**类型:** 学习
**语言:** Python(stdlib,JSON-RPC解析器)
**前置要求:** 阶段13课程01至05(工具接口和函数调用)
**时间:** ~45分钟

## 学习目标

- 命所有六MCP原语(server上tools、resources、prompts;client上roots、sampling、elicitation)并每给一用例。
- 走三phase生命周期(initialize、operation、shutdown)并述每phase谁发何消息。
- 解析和发JSON-RPC 2.0请求、响应、通知包。
- 释`initialize`能力谈判何及无何破。

## 问题背景

MCP前,每工具用agent有己协议。Cursor有MCP形但不兼容工具系统。Claude Desktop发不同一。VS Code Copilot扩展有第三。建"Postgres query"工具团队写同工具三次,每至异宿主API。复用需复制代码。

结果是特例集成爆炸和生态velocity上限。

MCP通过标准线格式修复此。单MCP server工于每MCP client:Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf、2026年4月300+客户端。110M月SDK下载。10,000+公开server。Linux Foundation 2025年12月于新Agentic AI Foundation下托管。

本phase用spec修订是**2025-11-25**。它加async Tasks(SEP-1686)、URL-mode elicitation(SEP-1036)、sampling with tools(SEP-1577)、incremental scope consent(SEP-835)、OAuth 2.1 resource-indicator语义。阶段13课程09至16覆盖那些扩展。本课止于基础。

## 概念讲解

### 三server原语

1. **Tools。**可调动作。阶段13课程01同四步循环。
2. **Resources。**露数据。只读内容URI可址:`file:///path`、`db://query/...`、自定义scheme。
3. **Prompts。**复用模板。宿主UI slash-command;server供模板,client填参数。

### 三client原语

4. **Roots。**server可触URI集。Client声明;server尊重。
5. **Sampling。**Server请求client模型做completion。使server托管agent循环无需server侧API key。
6. **Elicitation。**Server中途问client用户结构输入。表或URL(SEP-1036)。

MCP每能恰属此六之一。阶段13课程10至14每深覆盖。

### 线格式:JSON-RPC 2.0

每消息是JSON对象带这些域:

- 请求:`{jsonrpc: "2.0", id, method, params}`。
- 响应:`{jsonrpc: "2.0", id, result | error}`。
- 通知:`{jsonrpc: "2.0", method, params}`——无`id`,无响应期望。

基础spec约15方法,按原语分组。重要:

- `initialize`/`initialized`(握手)
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`(server-to-client)
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三phase生命周期

**Phase 1:initialize。**

Client发`initialize`带其`capabilities`和`clientInfo`。Server回己`capabilities`、`serverInfo`、及讲spec版本。Client消化响应后发`notifications/initialized`。此后,任方可按谈判能力发请求。

**Phase 2:operation。**

双向。Client调`tools/list`发现,后`tools/call`调。Server可发`sampling/createMessage`若声明该能。Server可发`notifications/tools/list_changed`当工具集变。Client可发`notifications/roots/list_changed`当用户改root scope。

**Phase 3:shutdown。**

任方闭transport。MCP无结构shutdown方法;transport(stdio或Streamable HTTP,阶段13课程09)载连接终信号。

### 能力谈判

`initialize`握手`capabilities`是契约。Server例:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Server声明可发`tools/list_changed`通知并支持`resources/subscribe`。Client同意通过声明己:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

若Client未声明`sampling`,Server勿调`sampling/createMessage`。对称:若Server未声明`resources.subscribe`,Client勿试订阅。

这防生态漂。不支持sampling client仍是有效MCP client;不调`sampling`server仍是有效MCP server。仅不合用该特性。

### 结构内容和错形

`tools/call`回类型块`content`数组:`text`、`image`、`resource`。阶段13课程14加MCP Apps(`ui://`交互UI)至列表。

错用JSON-RPC错码。Spec定义加:`-32002` "Resource not found"、`-32603` "Internal error",加MCP特错数据作`error.data`。

### Client能力vs工具调用细节

常见混淆:`capabilities.tools`是client是否支持tool-list-changed通知。Client是否会调特定工具是runtime择由其模型驱动,非能力旗。能力旗是spec级契约。模型择正交。

### 何JSON-RPC而非REST?

JSON-RPC 2.0(2010)是轻量双向协议。REST是client发起。MCP需server发起消息(sampling、通知),故带对称请求/响应形的JSON-RPC是自然fit。JSON-RPC亦于stdio和WebSocket/Streamable HTTP干净组无需重发明HTTP请求形。

## 使用

`code/main.py`发最小JSON-RPC 2.0解析器和发射器,后手走`initialize`→`tools/list`→`tools/call`→`shutdown`序列,打印每消息。无真transport;仅消息形。比Further Reading链spec验每包。

看点:

- `initialize`双向声明能力;响应有`serverInfo`和`protocolVersion: "2025-11-25"`。
- `tools/list`回`tools`数组;每条有`name`、`description`、`inputSchema`。
- `tools/call`用`params.name`和`params.arguments`。
- 响应`content`是`{type, text}`块数组。

## 交付成果

本课产`outputs/skill-mcp-handshake-tracer.md`。给MCP client-server交互pcap式转录,skill注释每消息何原语、何生命周期phase、何能力赖。

## 练习题

1. 跑`code/main.py`。识能力谈判发生行并述若Server未声明`tools.listChanged`何变。

2. 扩解析器处`notifications/progress`。消息形:`{method: "notifications/progress", params: {progressToken, progress, total}}`。长跑`tools/call`中发并确client handler会示进度条。

3. 读MCP 2025-11-25 spec从头至尾——全文档约80页。识大多server不需的一能力旗。提示:涉resource订阅。

4. 纸上画假设"cron job"特性属何原语。(提示:server欲client定时调。六原语今无一fit。)MCP 2026 roadmap有draft SEP。

5. 解析GitHub上开MCP server一session log。计请求vs响应vs通知消息。算生命周期vs操作流量分。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MCP | "Model Context Protocol" | 模型到工具发现和调用开协议 |
| Server原语 | "Server露何" | tools(动作)、resources(数据)、prompts(模板) |
| Client原语 | "Client让server用何" | roots(scope)、sampling(LLM回调)、elicitation(用户输入) |
| JSON-RPC 2.0 | "线格式" | 对称请求/响应/通知包 |
| `initialize`握手 | "能力谈判" | 首消息对;server和client声明支持特性 |
| `tools/list` | "发现" | Client问server当前工具集 |
| `tools/call` | "调用" | Client问server带参数执行工具 |
| `notifications/*_changed` | "突变事件" | Server告诉client原语列表已变 |
| 内容块 | "类型结果" | 工具结果中`{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | "Spec演进提案" | 命名draft提案(如SEP-1686用于async Tasks) |

## 延伸阅读

- [Model Context Protocol—Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)——规范spec文档
- [Model Context Protocol—Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture)——六原语心智模型
- [Anthropic—Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)——2024年11月发帖
- [MCP blog—First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)——一周年回顾和2025-11-25 spec改
- [WorkOS—MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update)——SEP-1686、1036、1577、835、1724总结