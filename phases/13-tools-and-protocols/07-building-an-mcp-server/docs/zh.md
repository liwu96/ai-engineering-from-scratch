# 建MCP Server——Python+TypeScript SDKs

> 大多MCP教程仅示stdio hello-world。真server露工具加资源加提示、处能力谈判、发结构错、跨SDK同工。本课端到端建笔记server:stdlib stdio transport、JSON-RPC dispatch、三server原语、和纯函数式适Python SDK FastMCP或TypeScript SDK毕业。

**类型:** 构建
**语言:** Python(stdlib,stdio MCP server)
**前置要求:** 阶段13课程06(MCP基础)
**时间:** ~75分钟

## 学习目标

- 实`initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、`prompts/get`方法。
- 写dispatch循环从stdin读JSON-RPC消息并写响应至stdout。
- 发JSON-RPC 2.0 spec和MCP加码结构错响应。
- 毕stdlib实现至FastMCP(Python SDK)或TypeScript SDK无需重写工具逻辑。

## 问题背景

用远程transport(阶段13课程09)或auth层(阶段13课程16)前,需干净本地server。本地意stdio:server作为子进程被client spawning,消息经stdin/stdout newline分隔流。

2025-11-25 spec定stdio消息编码为带显`\n`分隔JSON对象。此无SSE;SSE是旧远程模式2026年中移(Atlassian Rovo MCP server 2026年6月30日弃;Keboola 2026年4月1日)。stdio,一JSON对象每行是全线格式。

笔记server是好形因它练所有三server原语。Tools做突变(`notes_create`)。Resources露数据(`notes://{id}`)。Prompts发模板(`review_note`)。本课形泛至任域。

## 概念讲解

### Dispatch循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三规:

- 勿打印非JSON-RPC包至stdout。Debug日志去stderr。
- 每请求须匹配带同`id`响应。
- 通知勿响应。

### 实`initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

仅声明所支持。Client赖能力集门特性。

### 实`tools/list`和`tools/call`

`tools/list`回`{tools: [...]}`每条有`name`、`description`、`inputSchema`。`tools/call`取`{name, arguments}`回`{content: [blocks], isError: bool}`。

内容块类型化。最常见:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错来两形。协议级错(未知方法、坏参数)是JSON-RPC错。工具级错(有效调用但工具失败)回作`{content: [...], isError: true}`。让模型看上下文中失败。

### 实resources

Resources设计只读。`resources/list`回manifest;`resources/read`回内容。URI可`file://...`、`http://...`或自定义scheme如`notes://`。

当你露数据作resource而非工具:

- 模型不"调"它;client可用户请求注入上下文。
- 订阅让server于resource变时push更新(阶段13课程10)。
- 阶段13课程14扩此带`ui://`交互resources。

### 实prompts

Prompts是带命名参数模板。宿主露作slash-command。`review_note`提示可能取`note_id`参数并产client喂其模型多消息提示模板。

### Stdio transport微妙

- Newline分隔JSON。无长度前缀帧。
- 勿缓冲。每写后`sys.stdout.flush()`。
- Client控生命周期。stdin闭(EOF),干净退出。
- 勿静默处SIGPIPE;记录并退出。

### 注解

每工具可载`annotations`述安全属性:

- `readOnlyHint: true`——纯读,安全重试。
- `destructiveHint: true`——不可逆副作用;client应确认。
- `idempotentHint: true`——同输入产同输出。
- `openWorldHint: true`——交互外系统。

Client用这些决UX(确认dialog、状态指示器)和路由(阶段13课程17)。

### 毕业路

`code/main.py`中stdlib server约180行。FastMCP(Python)塌同逻辑至decorator式:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK有等价形。毕业路是备好时drop-in;概念(能力、dispatch、内容块)同。

## 使用

`code/main.py`是完整stdio笔记MCP server,仅stdlib。它处`initialize`、`tools/list`、`tools/call`三工具(`notes_list`、`notes_search`、`notes_create`)、每笔记`resources/list`和`resources/read`、和`review_note`提示。可通过管道JSON-RPC消息驱动:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

看点:

- Dispatcher是`dict[str, Callable]`按方法名key。
- 每工具执行器回内容块列表,非裸字符串。
- 执行器raise时设`isError: true`。

## 交付成果

本课产`outputs/skill-mcp-server-scaffolder.md`。给域(笔记、票、文件、数据库),skill scaffold MCP server带正工具/资源/提示分和SDK毕业路。

## 练习题

1. 跑`code/main.py`并用手建JSON-RPC消息驱动。练`notes_create`,后`resources/read`取新笔记。

2. 加`notes_delete`工具带`annotations: {destructiveHint: true}`。验client会露确认dialog(需真宿主;Claude Desktop工)。

3. 实`resources/subscribe`使server每当笔记修改push`notifications/resources/updated`。加keepalive任务。

4. 移server至FastMCP。Python文件应缩至少于80行。线行为须同;用同JSON-RPC测试框架验。

5. 读spec`server/tools`节并识本课server未实一工具定义域。(提示:有数;择一加。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MCP server | "露工具物" | 经stdio或HTTP讲MCP JSON-RPC进程 |
| stdio transport | "子进程模型" | Server被client spawning;经stdin/stdout通信 |
| Dispatcher | "方法路由" | JSON-RPC方法名至handler函数映 |
| 内容块 | "工具结果块" | 工具响应`content`数组类型元素 |
| `isError` | "工具级失败" | 信号工具失败;区JSON-RPC错 |
| 注解 | "安全提示" | readOnly/destructive/idempotent/openWorld旗 |
| FastMCP | "Python SDK" | MCP协议上decorator式高层框架 |
| Resource URI | "可址数据" | `file://`、`db://`、或自定义scheme识resource |
| 提示模板 | "Slash-command简" | 带参数槽宿主UI server供模板 |
| 能力声明 | "特性toggle" | `initialize`中按原语旗声明 |

## 延伸阅读

- [Model Context Protocol—Python SDK](https://github.com/modelcontextprotocol/python-sdk)——参考Python实现
- [Model Context Protocol—TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)——并行TS实现
- [FastMCP—server framework](https://gofastmcp.com/)——MCP server decorator式Python API
- [MCP—Quickstart server guide](https://modelcontextprotocol.io/quickstart/server)——用任SDK端到端教程
- [MCP—Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)——tools/*消息完整参考