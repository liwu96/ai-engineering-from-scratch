# MCP Apps——经`ui://`交互UI资源

> 仅文本工具输出cap agent可示。MCP Apps(SEP-1724,2026年1月26日官方)让工具回沙箱交互HTML内嵌渲染于Claude Desktop、ChatGPT、Cursor、Goose、VS Code。仪表盘、表、地图、3D场景,全经一扩展。本课走`ui://`资源scheme、`text/html;profile=mcp-app` MIME、iframe沙箱postMessage协议、和安全面来自让server渲染HTML。

**类型:** 构建
**语言:** Python(stdlib,UI resource emitter),HTML(sample app)
**前置要求:** 阶段13课程07(MCP server),阶段13课程10(resources)
**时间:** ~75分钟

## 学习目标

- 从工具调用回`ui://`资源并设正MIME和metadata。
- 声工具关联UI带`_meta.ui.resourceUri`、`_meta.ui.csp`、`_meta.ui.permissions`。
- 实iframe沙箱postMessage JSON-RPC用于UI-to-host通信。
- 用CSP和permissions-policy默认防UI-origin攻击。

## 问题背景

2025-era `visualize_timeline`工具可回"这是14笔记时间组织:...".那是段落。用户实欲交互timeline。MCP Apps前,择:client特定widget API(Claude artifacts、OpenAI Custom GPT HTML),或无UI。

MCP Apps(SEP-1724,2026年1月26日发)标准契约。工具结果含`resource`其URI是`ui://...`其MIME是`text/html;profile=mcp-app`。Host渲染于沙箱iframe带限CSP和除非显grant无网络访问。iframe内UI经微小postMessage JSON-RPC方言发消息给host。

每兼容client(Claude Desktop、ChatGPT、Goose、VS Code)渲染同`ui://`资源同方式。一server、一HTML bundle、通用UI。

## 概念讲解

### `ui://`资源scheme

工具回:

```json
{
  "content": [
    {"type": "text", "text": "这是你笔记timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

Host后于`ui://notes/timeline` URI调`resources/read`得:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe沙箱

Host渲染HTML于沙箱`<iframe>`带:

- `sandbox="allow-scripts allow-same-origin"`(或更严server声明)
- Server声明CSP经响应头apply。
- Host origin无cookie、无localStorage。
- 网络访问限CSP `connectSrc`。

### postMessage协议

iframe经`window.postMessage`通信host。微小JSON-RPC 2.0方言:

总pin`targetOrigin`至peer精确origin,接收侧验`event.origin`对allowlist后处任payload。勿用`"*"`用于此channel任侧——body载工具调用和资源读。

```js
// iframe to host(pin至host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe(pin至iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI可调host侧方法:

- `host.callTool(name, arguments)`——调server工具。
- `host.readResource(uri)`——读MCP resource。
- `host.getPrompt(name, arguments)`——取提示模板。
- `host.close()`——dismiss UI。

每调用仍经MCP协议并继server权限。

### Permissions

`_meta.ui.permissions`列表请求额外能:

- `camera`——访问用户camera(用于扫描文档UI)。
- `microphone`——语音输入。
- `geolocation`——位置。
- `network:*`——比`connectSrc`单允更广网络访问。

每权限是UI渲染前用户见提示。

### 安全风险

iframe中HTML仍是HTML。新攻击面:

- **经UI提示注入。**恶意server UI可示文本看像系统消息并骗用户。Host渲染应显区server UI和host UI。
- **经`connectSrc`渗漏。**若CSP允`connect-src: *`,UI可送数据任处。默认应严。
- **Clickjacking。**UI叠host chrome。Host须防z-index操纵并执opacity规。
- **偷焦点。**UI取键盘焦点并捕下消息。Host须拦截。

阶段13课程15深覆盖此作MCP安全部分;本课引。

### `ui/initialize`握手

iframe载后,经postMessage发`ui/initialize`:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

Host回能力和session token。UI用session token于每后续host调用。

### AppRenderer/AppFrame SDK原语

ext-apps SDK露两便利原语:

- `AppRenderer`(server侧)——包React/Vue/Solid组件并发带正MIME和metadata`ui://`资源。
- `AppFrame`(client侧)——收资源,挂iframe,并中介postMessage。

可用这些或手roll HTML和JSON-RPC。

### 生态态

MCP Apps 2026年1月26日发。2026年4月client支持:

- **Claude Desktop。**2026年1月全支持。
- **ChatGPT。**经Apps SDK全支持(同底层MCP Apps协议)。
- **Cursor。**Beta;经settings启。
- **VS Code。**仅Insider build。
- **Goose。**全支持。
- **Zed、Windsurf。**Roadmap。

产server:仪表盘、地图可视化、数据表、图builder、沙箱IDE preview。

## 使用

`code/main.py`扩笔记server带`visualize_timeline`工具回`ui://notes/timeline` resource,加`resources/read` handler于该URI回小但完整HTML bundle带SVG timeline。HTML是stdlib模板——无build系统。postMessage于JS注释画因stdlib不可驱动浏览器。

看点:

- 工具响应`_meta.ui`载resourceUri、CSP、permissions。
- HTML渲染无网络访问;所有数据inline。
- JS调`host.callTool`经`window.parent.postMessage`(文档但此stdlib demo inert)。

## 交付成果

本课产`outputs/skill-mcp-apps-spec.md`。给受益交互UI工具,skill产完整MCP Apps契约:`ui://` URI、CSP、permissions、postMessage入口、安全checklist。

## 练习题

1. 跑`code/main.py`并检发HTML。直接浏览器开HTML;验SVG渲染。后画UI会用`host.callTool("notes_update", ...)`postMessage契约。

2. 紧CSP:移`'unsafe-inline'`并用nonce基script policy。HTML生成代码何变?

3. 加第二UI资源`ui://notes/editor`带就地编辑笔记表。用户提交时,iframe调`host.callTool("notes_update", ...)`。

4. 审UI攻击面。恶意server何处可注入内容?iframe沙箱防何和不防何?

5. 读SEP-1724 spec并识此玩具实现未用MCP Apps SDK一能。(提示:组件级态sync。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MCP Apps | "交互UI资源" | SEP-1724扩展2026-01-26发 |
| `ui://` | "App URI scheme" | UI bundle资源scheme |
| `text/html;profile=mcp-app` | "MIME" | MCP App HTML内容类型 |
| Iframe沙箱 | "渲染容器" | 带CSP和permissions浏览器sandboxing UI |
| postMessage JSON-RPC | "UI-to-host线" | host调用微小JSON-RPC-over-postMessage方言 |
| `_meta.ui` | "工具-UI绑" | 链工具结果至UI resource metadata |
| CSP | "Content-Security-Policy" | 声允许script、network、style源 |
| AppRenderer | "Server SDK原语" | 转框架组件入`ui://`资源 |
| AppFrame | "Client SDK原语" | Iframe挂helper中介postMessage |
| `ui/initialize` | "握手" | UI至host首postMessage |

## 延伸阅读

- [MCP ext-apps—GitHub](https://github.com/modelcontextprotocol/ext-apps)——参考实现和SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)——形式spec文档
- [MCP—Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview)——高层文档
- [MCP blog—MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)——2026年1月发帖
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/)——JSDoc式SDK参考