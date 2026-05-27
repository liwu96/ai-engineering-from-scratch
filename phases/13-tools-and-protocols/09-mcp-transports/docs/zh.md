# MCP Transports——stdio vs Streamable HTTP vs SSE迁移

> stdio本地工别处不。Streamable HTTP(2025-03-26)是远程标准。旧HTTP+SSE transport弃并于2026年中移。择错transport付迁移;择对买远程可host MCP server带session持续和DNS-rebinding护。

**类型:** 学习
**语言:** Python(stdlib,Streamable HTTP端点骨架)
**前置要求:** 阶段13课程07、08(MCP server和client)
**时间:** ~45分钟

## 学习目标

- 基部署形(本地vs远程、单进程vs fleet)择stdio和Streamable HTTP。
- 实Streamable HTTP单端点模式:POST请求、GET session流。
- 执`Origin`验和session-id语义御DNS-rebinding。
- 迁移遗留HTTP+SSE server至Streamable HTTP于2026年中移deadline前。

## 问题背景

首MCP远程transport(2024-11)是HTTP+SSE:两端点,一client POST、一Server-Sent-Events channel server-to-client流。工。亦笨:每session两端点、某CDN前缓存破、和长活SSE连接硬依赖某些WAF激进终止。

2025-03-26 spec换Streamable HTTP:一端点,POST client请求,GET建session流,两者享`Mcp-Session-Id`头。建或迁移后每server用Streamable HTTP。旧SSE模式弃——Atlassian Rovo 2026年6月30日移;Keboola 2026年4月1日;大多余企业server 2026年底。

Stdio仍重要本地server。Claude Desktop、VS Code、每IDE形client经stdio spawn server。正心智模型:stdio用于"本机",Streamable HTTP用于"网路"。无交叉。

## 概念讲解

### stdio

- 子进程transport。Client spawn server,经stdin/stdout通信。
- 每行一JSON对象。Newline分隔。
- 无session id;进程身份是session。
- 无auth需(子进程继父信任边界)。
- 勿用于远程server——需SSH或socat tunnel,此时用Streamable HTTP。

### Streamable HTTP

单端点`/mcp`(或任路径)。支持三HTTP方法:

- **POST /mcp。**Client发JSON-RPC消息。Server回单JSON响应或SSE流一或多响应(用于批响应和该请求相关通知有用)。
- **GET /mcp。**Client开长活SSE channel。Server用于server-to-client请求(sampling、通知、elicitation)。
- **DELETE /mcp。**Client显终止session。

Session由首响应`Mcp-Session-Id`头server设和每后续请求client echo识。Session id MUST密码随机(128+位);client选id拒安全。

### 单端点vs两

旧spec两端点模式2026仍可调——spec声明"legacy compatible"。但所有新server应单端点。官方SDK发单端点;仅当对话未迁移远程用遗留模式。

### `Origin`验和DNS-rebinding

浏览器非MCP client(今),但攻击者可造网页说服浏览器POST至`localhost:1234/mcp`——用户本地MCP server听处。若server不查`Origin`,浏览器同源策略不救因`Origin: http://evil.com`是有效跨源。

2025-11-25 spec要server拒`Origin`不在allowlist请求。Allowlist典型含MCP client宿(`https://claude.ai`、`vscode-webview://*`)和本地UI localhost变体。

### Session id生命周期

1. Client发首请求无`Mcp-Session-Id`。
2. Server配随机id,于响应头设`Mcp-Session-Id`。
3. Client echo该头于所有后续请求和`GET /mcp`流。
4. Session可被server撤销;client后续请求见404并须重初始化。
5. Client可显DELETE session干净shutdown。

### Keepalive和重连

SSE连接掉。Client通过带同`Mcp-Session-Id`重GET重建。Server MUST queue中断错过事件(至合理窗口)并经client echo`last-event-id`头重播。

阶段13课程13覆盖Tasks,使长跑工作存甚至全session重连。

### 向后兼容probe

Client欲支持新旧server:

1. POST至`/mcp`。
2. 若响应`200 OK`带JSON或SSE,这是Streamable HTTP。
3. 若响应`200 OK`带`Content-Type: text/event-stream`且`Location`头指第二端点,这是遗留HTTP+SSE;跟`Location`。

### Cloudflare、ngrok、hosting

2026产远程MCP server跑于Cloudflare Workers(带其MCP Agents SDK)、Vercel Functions、或containerized Node/Python。键:你hosting须支持长活HTTP连接用于SSE GET。Vercel免费tier盖10秒不适合。Cloudflare Workers支持无限流。

### Gateway组

当你用gateway(阶段13课程17)前多MCP server,gateway是单Streamable HTTP端点重写session id和复用上游。工具于gateway层合并;client见单逻辑server。

### Transport失模式

- **stdio SIGPIPE。**子进程死中写raise SIGPIPE;server应干净退出。Client应检测EOF并标记session死。
- **HTTP 502/504。**Cloudflare、nginx、其他proxy上游失败发这些。Streamable HTTP client应短backoff后重试一次。
- **SSE连接掉。**TCP RST、proxy timeout、client网络变闭流。Client带`Mcp-Session-Id`和可选`last-event-id`重连恢复。
- **Session撤销。**Server无效session id;client下请求见404。Client须重握手。
- **时钟偏。**Client Resource-TTL计算偏server。Client应视server时间戳权威。

### 何绕Streamable HTTP

某些企业内部网络后MCP server部署gRPC或消息队列transport。这是非标准——MCP spec不正式定义这些。Gateway可露Streamable HTTP面于MCP client同时内部用gRPC。保外surface spec兼容;gateway执翻译。

## 使用

`code/main.py`用`http.server`(stdlib)实最小Streamable HTTP端点。它处`/mcp`POST、GET、DELETE,首响应设`Mcp-Session-Id`,验`Origin`,拒非allowlist origin请求。Handler复课程07笔记server dispatch逻辑。

看点:

- POST handler读JSON-RPC body,dispatch,写JSON响应(单响应变体;SSE变体结构类似)。
- `Origin`查拒默认`http://evil.example`探但接受`http://localhost`。
- Session id是随机128位hex字符串;server内存中持每session态。

## 交付成果

本课产`outputs/skill-mcp-transport-migrator.md`。给HTTP+SSE(遗留)MCP server,skill产迁移计划至Streamable HTTP带session-id持续、Origin查、向后兼容probe支持。

## 练习题

1. 跑`code/main.py`。从`curl`POST `initialize`并观`Mcp-Session-Id`响应头。Echo头POST第二请求并验session持续。

2. 加GET handler开SSE流。每五秒发一`notifications/progress`事件。带同session id重GET重连并验server接受。

3. 实`last-event-id`重播逻辑。重连时,重播该id后生成任何事件。

4. 扩`Origin`验支持通配模式(`https://*.example.com`)并验接受`https://app.example.com`但拒`https://evil.example.com.attacker.net`。

5. 取官方注册遗留HTTP+SSE server(有数)并画迁移:端点处理、session id生成、头语义何变。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| stdio transport | "本地子进程" | stdin/stdout上JSON-RPC,newline分隔 |
| Streamable HTTP | "远程transport" | 单端点POST+GET+可选SSE,2025-03-26 spec |
| HTTP+SSE | "遗留" | 2026年中移两端点模型 |
| `Mcp-Session-Id` | "Session头" | Server配随机id echo于每后续请求 |
| `Origin` allowlist | "DNS-rebinding防" | 拒Origin未批准请求 |
| 单端点 | "一URL" | `/mcp`处POST/GET/DELETE所有session操作 |
| `last-event-id` | "SSE重播" | 用于恢复掉流不miss事件头 |
| 向后兼容probe | "新旧检测" | Client响应形查自择transport |
| 活HTTP | "SSE流" | Server于一TCP连接分时小时push事件 |
| Session撤销 | "强重初始化" | Server无效session id;client须重握手 |

## 延伸阅读

- [MCP—Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)——stdio和Streamable HTTP规范参考
- [MCP—Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)——引入Streamable HTTP修订
- [Cloudflare—MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/)——Workers托管Streamable HTTP模式
- [AWS—MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http)——跨部署形比
- [Atlassian—HTTP+SSE弃通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484)——具体迁移deadline例