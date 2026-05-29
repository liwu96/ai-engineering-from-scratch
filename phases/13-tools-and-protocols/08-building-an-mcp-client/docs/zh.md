# 建MCP Client——发现、调用、session管理

> 大多MCP内容发server教程并挥手client。Client代码是硬编排居处:进程spawning、能力谈判、跨多server工具列表合并、sampling回调、重连、命名空间冲突解。本课建多server client提三异MCP server入一flat工具命名空间模型。

**类型:** 构建
**语言:** Python(stdlib,多server MCP client)
**前置要求:** 阶段13课程07(建MCP server)
**时间:** ~75分钟

## 学习目标

- Spawn MCP server作子进程,完`initialize`,并发`notifications/initialized`。
- 维每server session态(能力、工具列表、最后见通知id)。
- 跨多server合并工具列表入一带冲突处命名空间。
- 路工具调用至拥它server并重组响应。

## 问题背景

真agent宿主(Claude Desktop、Cursor、Goose、Gemini CLI)同时载多MCP server。用户可能filesystem server、Postgres server、GitHub server同时跑。Client工作:

1. Spawn每server。
2. 独立握手每。
3. 每调`tools/list`并flatten结果。
4. 当模型emit`notes_search`,于合并命名空间查并路由至正server。
5. 处任server通知(`tools/list_changed`)不block。
6. Transport失败重连。

手roll所有分离"toy"和"serviceable"。官方SDK包此,但心智模型须是你。

## 概念讲解

### 子进程spawning

`subprocess.Popen`带`stdin=PIPE, stdout=PIPE, stderr=PIPE`。设`bufsize=1`用text mode逐行读。每server一进程;client每server持一`Popen` handle。

### 每server session态

每server一`Session`对象持:

- `process`——Popen handle。
- `capabilities`——`initialize`时server声明。
- `tools`——最后`tools/list`结果。
- `pending`——请求id至等响应promise/future映。

请求本性async;server A`tools/call`发同时server B中调用不可block。用线程带queue或asyncio。

### 合并命名空间

Client见聚合工具列表时,名可撞。两server可能都露`search`。Client有三择:

1. **按server名prefix。**`notes/search`、`files/search`。清但丑。
2. **静默先来。**后server`search`覆前。风险;藏碰撞。
3. **碰撞拒。**拒载第二server;通知用户。安全敏感宿主最安。

Claude Desktop用prefix-by-server。Cursor用碰撞拒带清错。VS Code MCP亦用prefix-by-server。

### 路由

合并后,dispatch表映`tool_name -> session`。模型按名emit调用;client找session并写`tools/call`消息至该server stdin,后等响应。

### Sampling回调

若server于`initialize`声明`sampling`能,可发`sampling/createMessage`问client跑其LLM。Client须:

1. Block进一步请求至该server直sample解,或pipeline若实现支持并发。
2. 调其LLM提供者。
3. 发响应回server。

课程11端到端覆盖sampling。本课stub完完整。

### 通知处

`notifications/tools/list_changed`意重调`tools/list`。`notifications/resources/updated`意若使用重读resource。通知不可产响应——勿试ack。

常见client bug:block读循环于`tools/call`当通知坐流中。用后台读线程push每消息入queue;主线程dequeue和dispatch。

### 重连

Transport可失败:server崩溃、OS杀进程、stdio pipe破。Client于stdout检测EOF并视为session死。择:

- 静默重启server并重握手。只读server OK。
- 露失败给用户。有用户可见session状态server OK。

阶段13课程09覆盖Streamable HTTP重连语义;stdio更简。

### Keepalive和session id

Streamable HTTP用`Mcp-Session-Id`头。Stdio无session id——进程身份IS session。Keepalive ping可选;stdio pipe不活跃破。

## 使用

`code/main.py` spawn三模拟MCP server作子进程,握手每,合并工具列表,并路工具调用至正者。"server"实是其他Python进程跑toy responder(无真LLM)。跑它看:

- 三初始化,每带己能力集。
- 三`tools/list`结果合并入7工具命名空间。
- 基工具名路由决策。
- 命名空间prefixing防碰撞。

看点:

- `Session` dataclass干净持每server态。
- 后台读线程dequeue stdout每线不block主线程。
- Dispatch表是简`dict[str, Session]`。
- 碰撞处显:当两server声明同名,后者带prefix重命名。

## 交付成果

本课产`outputs/skill-mcp-client-harness.md`。给MCP server声明列表(名、command、args),skill生成启动框架、合并工具列表、并提供带碰撞解路由函数。

## 练习题

1. 跑`code/main.py`并观server spawn log。SIGTERM杀一模拟server进程并观client如何检测EOF并标记session死。

2. 实命名空间prefixing。当两server露`search`,重命名第二作`<server>/search`。更新dispatch表并验工具调用路由正。

3. 加连接池式server重启backoff:连续失败指数backoff,盖30秒,三失败后发通知给用户。

4. 纸上画支持100并发MCP server client。何数据结构替简dispatch dict?(提示:trie用于prefix命名空间,加每server工具计数metric。)

5. 移client至官方MCP Python SDK。SDK包`stdio_client`和`ClientSession`。代码应缩从~200行至~40行保多server路由。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MCP client | "Agent宿主" | Spawning server和编排工具调用进程 |
| Session | "每server态" | 能力、工具列表、pending请求簿记 |
| 合并命名空间 | "一工具列表" | Flat跨所有活跃server工具名集 |
| 命名空间碰撞 | "两server同工具" | Client须prefix、拒或先来duplicate |
| 路由 | "谁得此调用?" | 从工具名至拥server dispatch |
| 后台读 | "非block stdout" | 线程或任务将server stdout drain入queue |
| Sampling回调 | "LLM-as-a-service" | Client handler用于server`sampling/createMessage` |
| `notifications/*_changed` | "原语突变" | Client须重发现或重读信号 |
| 重连策略 | "当server死" | Transport失败时重启语义 |
| Stdio session | "进程=session" | 无session id;子进程生命周期是session |

## 延伸阅读

- [Model Context Protocol—Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client)——规范client行为
- [MCP—Quickstart client guide](https://modelcontextprotocol.io/quickstart/client)——Python SDK hello-world client教程
- [MCP Python SDK—client module](https://github.com/modelcontextprotocol/python-sdk)——参考`ClientSession`和`stdio_client`
- [MCP TypeScript SDK—Client](https://github.com/modelcontextprotocol/typescript-sdk)——TS并行
- [VS Code—MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp)——VS Code如何于单编辑器宿主复用多MCP server