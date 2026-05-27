# MCP资源和提示——工具外上下文露

> 工具得90%MCP注意。其他两server原语解异问题。资源露数据读;提示露复用模板作slash-command。大多server应资源而非包读入工具,提示而非client提示中硬编码workflow。本课命名决策规并走`resources/*`和`prompts/*`消息。

**类型:** 构建
**语言:** Python(stdlib,resource+提示handler)
**前置要求:** 阶段13课程07(MCP server)
**时间:** ~45分钟

## 学习目标

- 给定域择露能作工具、资源或提示。
- 实`resources/list`、`resources/read`、`resources/subscribe`并处`notifications/resources/updated`。
- 实`prompts/list`和`prompts/get`带参数模板。
- 识宿主何时露提示作slash-command vs自注入上下文。

## 问题背景

天真笔记app MCP server露一切作工具:`notes_read`、`notes_list`、`notes_search`。这包每数据访问入模型驱动工具调用。后果:

- 模型须决每可能受益上下文查询是否调`notes_read`。
- 只读内容不可订阅或流至宿主侧面板。
- Client UI(Claude Desktop resource附面板、Cursor"Include file"picker)不可露数据。

正分:露数据作resource、露突变或计算动作作工具、露复用多步workflow作提示。每原语有其UX affordance和其访问模式。

## 概念讲解

### 工具vs资源vs提示——决策规

| 能 | 原语 |
|----|------|
| 用户欲搜索、滤、或转换数据 | 工具 |
| 用户欲宿主含此数据作上下文 | 资源 |
| 用户欲模板化workflow可重跑 | 提示 |

指南:若模型会于每相关查询受益调用,是工具。若用户会受益附至对话,是资源。若整体多步workflow是用户欲重用单元,是提示。

### 资源

`resources/list`回`{resources: [{uri, name, mimeType, description?}]}`。`resources/read`取`{uri}`回`{contents: [{uri, mimeType, text | blob}]}`。

URI可是任可址:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`(自定义scheme)
- `memory://session-2026-04-22/recent`(server特定)

`contents[]`支持文本和二进制。二进制用`blob`作base64编码字符串加`mimeType`。

### 资源订阅

声明`{resources: {subscribe: true}}`于能力。Client调`resources/subscribe {uri}`。Server于resource变时发`notifications/resources/updated {uri}`。Client重读。

用例:笔记server其资源是盘上文件;文件watcher触发更新通知;Claude Desktop于宿主外编辑时重pull文件入上下文。

### 资源模板(2025-11-25加)

`resourceTemplates`让你露参数化URI模式:`notes://{id}`带`id`作completion目标。Client可于resource picker自动补全id。

### 提示

`prompts/list`回`{prompts: [{name, description, arguments?}]}`。`prompts/get`取`{name, arguments}`回`{description, messages: [{role, content}]}`。

提示是模板填至宿主喂其模型消息列表。例,`code_review`提示取`file_path`参数并回三消息序列:系统消息、带文件体用户消息、带推理模板assistant kickoff。

### 宿主和提示

Claude Desktop、VS Code、Cursor露提示作chat UI slash-command。用户打`/code_review`并从表择参数。Server提示是"用户shortcut"和"全提示发模型"间契约。

非每client支持提示——查能力谈判。Server声明提示能但client无提示支持简单不会看slash command。

### "list changed"通知

资源和提示皆于集变时emit`notifications/list_changed`。刚导入20新笔记笔记server发`notifications/resources/list_changed`;client重调`resources/list`取新增。

### 内容类型约定

文本:`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制:`image/png`、`application/pdf`、加`blob`域。
MCP Apps(课程14):`ui://` URI中`text/html;profile=mcp-app`。

### 动态资源

资源URI不必对应静态文件。`notes://recent`可每次读回最新五笔记。`db://query/users/active`可执行参数化查询。Server自由动态计算内容。

规:若client可按URI缓存,URI须稳定。若计算是一次性,URI应含时间戳或nonce使client缓存不陈。

### 订阅vs polling

订阅能client经`notifications/resources/updated`得server push。订阅前client或不支持它宿主通过重读poll。两者spec兼容。Server能力声明告诉client支持何。

订阅成本:server上每session态(订阅何)。保订阅集bounded;断连client应timeout。

### 提示vs系统提示

MCP提示非系统提示。宿主系统提示(己操作指令)和MCP提示(server供模板用户调)并存。行为良好client不让server提示覆己系统提示;它叠。

## 使用

`code/main.py`扩课程07笔记server带:

- 每笔记资源(`notes://note-1`等)带`resources/subscribe`支持。
- `review_note`提示渲染至三消息模板。
- 文件watcher模拟笔记修改时发`notifications/resources/updated`。
- `notes://recent`动态资源总是回最新五笔记。

跑demo看全流。

## 交付成果

本课产`outputs/skill-primitive-splitter.md`。给提议MCP server,skill分类每能作工具/资源/提示带理。

## 练习题

1. 跑`code/main.py`。观初resource列表,后触发笔记编辑并验`notifications/resources/updated`事件发。

2. 加`resources/list_changed`发射器:当新笔记创建,发通知使client重发现。

3. 设计GitHub MCP server三提示:`summarize_pr`、`triage_issue`、`release_notes`。每带参数schema。提示体应无需进一步编辑可跑。

4. 取课程07 server现工具并分类应留工具或拆资源加工具对。一句理。

5. 读spec`server/resources`和`server/prompts`节。识`resources/read`中一域少填但spec支持。提示:看resource content上`_meta`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 资源 | "露数据" | URI可址宿主可读内容 |
| 资源URI | "数据指针" | Scheme前缀标识符(`file://`、`notes://`等) |
| `resources/subscribe` | "观变" | 特URI client-opt-in server-push更新 |
| `notifications/resources/updated` | "资源变" | 告client订阅资源有新内容信号 |
| 资源模板 | "参数化URI" | 带宿主picker补全提示URI模式 |
| 提示 | "Slash-command模板" | 带参数槽命名多消息模板 |
| 提示参数 | "模板输入" | 宿主渲染前收集类型参数 |
| `prompts/get` | "渲染模板" | Server回填消息列表 |
| 内容块 | "类型块" | `{type: text | image | resource | ui_resource}` |
| Slash-command UX | "用户shortcut" | 宿主露提示作`/`开头命令 |

## 延伸阅读

- [MCP—Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources)——资源URI、订阅、模板
- [MCP—Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts)——提示模板和slash-command集成
- [MCP—Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)——全`resources/*`消息参考
- [MCP—Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts)——全`prompts/*`消息参考
- [MCP—Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/)——扩展官方文档社区指南