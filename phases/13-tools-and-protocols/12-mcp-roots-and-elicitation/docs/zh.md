# Roots和Elicitation——Scope和途中用户输入

> 硬编码路径用户开不同项目即破。预填工具参数用户欠指定即破。Roots scope server至用户控URI集;elicitation中途停工具调用问用户结构输入经表或URL。两client原语,两常见MCP失模式修。SEP-1036(URL-mode elicitation,2025-11-25)2026 H1实验——依赖前查SDK版本。

**类型:** 构建
**语言:** Python(stdlib,roots+elicitation demo)
**前置要求:** 阶段13课程07(MCP server)
**时间:** ~45分钟

## 学习目标

- 声`roots`并响应`notifications/roots/list_changed`。
- 限server文件操作于声明root集内URI。
- 用`elicitation/create`问用户确认或结构输入途中工具调用。
- 择form-mode和URL-mode elicitation(后实验;漂风险注)。

## 问题背景

笔记MCP server产撞两具体失败。

**破路径假设。**Server写对`~/notes`。异机用户笔记在`~/Documents/Notes`得工具调用静默失败(文件未找)或更糟,写错处。

**缺参数用户会知。**用户问"删旧TPS report笔记"。模型调`notes_delete(title: "TPS report")`但有三匹配笔记2023、2024、2025。工具不可猜。"歧义"失败烦;跑三 catastrophic。

Roots修首:client于`initialize`声明server可触URI集。Elicitation修次:server停工具调用并发`elicitation/create`问用户择。

## 概念讲解

### Roots

Client于`initialize`声明root列表:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Server后可调`roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Server MUST视roots作边界:root集外任文件读或写拒。此非client执(server仍是用户信任代码),但spec兼容server honor。

用户加或删root时,client发`notifications/roots/list_changed`。Server重调`roots/list`并更新边界。

### 何roots是client原语

Roots由client声明因它们代表用户同意模型。用户告诉Claude Desktop"给此笔记server这两目录访问"。Server不可扩scope。

### Elicitation:form-mode默认

`elicitation/create`取表schema加自然语言提示:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删'TPS report'?多笔记匹配;择一。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Client渲染表,收用户答,回:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三可能动作:`accept`(用户填)、`decline`(用户闭)、`cancel`(用户abort整工具调用)。

表schema flat——v1不支持嵌对象。SDK典型拒比单层更复杂。

### Elicitation:URL mode(SEP-1036,实验)

2025-11-25新。非schema,server发URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

Client于浏览器开URL,等完成,用户回时回。用于OAuth流、支付授权、签文档表不够处。

漂风险注:SEP-1036响应形仍在settling;某些SDK回callback URL,其他回completion token。用于产URL mode前读你SDK发注。

### 何elicitation是正工具

- 破性动作前用户确认(破性hint+elicitation)。
- 歧义解(择N匹配一)。
- 首运行setup(API key、目录、偏好)。
- OAuth式流(URL mode)。

### 何elicitation错

- 填工具required参数模型可散文问。用正常重提示,非elicitation dialog。
- 高频调用。Elicitation打断对话;勿循环内发。
- 任server可后验。验、回错、让模型文本问用户。

### Human-in-the-loop桥

Elicitation加sampling共启MCP"human-in-the-loop"模型。Server agent循环可暂停用户输入(elicitation)或模型推理(sampling)。阶段13课程11覆盖sampling;本课覆盖elicitation。合用于全途中控。

## 使用

`code/main.py`扩笔记server带:

- `roots/list`响应server于root-list-changed通知后重查。
- `notes_delete`工具用`elicitation/create`歧义解当多笔记匹配。
- `notes_setup`工具用URL-mode elicitation开首运行配置页(模拟)。
- 边界查拒声明roots外URI操作。

Demo跑三景:快乐路(一匹配)、歧义解(三匹配,elicitation发)、root外写(拒)。

## 交付成果

本课产`outputs/skill-elicitation-form-designer.md`。给可能需用户确认或歧义解工具,skill设计elicitation表schema和消息模板。

## 练习题

1. 跑`code/main.py`。触发歧义解路;确模拟用户答路由回工具。

2. 加新工具`notes_archive`每次需elicitation确认(破性hint)。查UX:比模型文本重问何比?

3. 实URL-mode elicitation用于首运行OAuth流。注漂风险并加SDK版本守。

4. 扩`roots/list`处:通知到时,server应原子重读并重扫可能出scope开文件handle。

5. 读GitHub SEP-1036 issue讨论thread。识影响server应何处URL-mode callback一开问。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Root | "同意边界" | Client允许server触URI |
| `roots/list` | "Server问scope" | Client回当前root集 |
| `notifications/roots/list_changed` | "用户改scope" | Client信号root集已变 |
| Elicitation | "途中问用户" | Server发起结构用户输入请求 |
| `elicitation/create` | "方法" | Elicitation请求JSON-RPC方法 |
| Form mode | "Schema驱动表" | Flat JSON Schema渲染client UI表 |
| URL mode | "浏览器redirect" | SEP-1036实验;开URL等 |
| `accept`/`decline`/`cancel` | "用户响应结果" | Server处三分 |
| 歧义解 | "择一" | 工具有N候选常见elicitation用例 |
| Flat表 | "仅顶级属性" | Elicitation schema不可嵌 |

## 延伸阅读

- [MCP—Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots)——规范roots参考
- [MCP—Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation)——规范elicitation参考
- [Cisco—What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements)——2025-11-25加walk-through
- [MCP—GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol)——URL-mode elicitation proposal(实验,漂风险)
- [The New Stack—How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/)——UX walk-through