# Async Tasks(SEP-1686)——Call-Now、Fetch-Later用于长跑工作

> 真agent工作需分钟至小时:CI run、深研究综合、批export。同步工具调用掉连接、超时、或block UI。SEP-1686,2025-11-25合并,加Tasks原语:任请求可augment成task,结果可后fetch或经态通知流。漂风险注:Tasks 2026 H1实验;SDK surface仍在spec设计。

**类型:** 构建
**语言:** Python(stdlib,async task态机)
**前置要求:** 阶段13课程07(MCP server),阶段13课程09(transports)
**时间:** ~75分钟

## 学习目标

- 识何时工具从同步升至task-augmented(>30秒server侧工作)。
- task生命周期:`working`→`input_required`→`completed`/`failed`/`cancelled`。
- 持task态使崩溃不失进行工作。
- 正poll`tasks/status`并fetch`tasks/result`。

## 问题背景

`generate_report`工具跑多分钟提取管道。同步模型下择:

1. 持连接开三分钟。远程transport掉;client超时;UI冻。
2. 立回placeholder;需client poll自定义端点。破MCP统一。
3. Fire-and-forget;无结果。

皆不好。SEP-1686加第四:task augmentation。任请求(典型`tools/call`)可tag作task。Server立回task id。Client poll`tasks/status`并完成时fetch`tasks/result`。Server侧态存重启。

## 概念讲解

### Task augmentation

请求成task通过设`params._meta.task.required: true`(或`optional: true`,server决)。Server立回:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl`是server持态承诺;ttl后task结果弃。

### 每工具opt-in

工具注解可声明task支持:

- `taskSupport: "forbidden"`——此工具总同步跑。快工具安全。
- `taskSupport: "optional"`——client可请求task-augmentation。
- `taskSupport: "required"`——client MUST用task augmentation。

`generate_report`工具会是`required`。`notes_search`工具会是`forbidden`。

### 状态

```
working  -> input_required -> working  (经elicitation循环)
working  -> completed
working  -> failed
working  -> cancelled
```

态机append-only:一旦`completed`、`failed`、`cancelled`,task终态。

### 方法

- `tasks/status {taskId}`——回当前态和进度提示。
- `tasks/result {taskId}`——block或未完回404。
- `tasks/cancel {taskId}`——幂等;终态忽略。
- `tasks/list`——可选;枚举活跃和最近完task。

### 流态变

Server支持时,client可订阅态通知:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

流而非poll的client得好UX。Poll总支持作最小surface。

### 持久态

Spec要声明task支持server持态。崩溃不应失ttl内完结果。Store范围SQLite至Redis至文件系统。课程13 harness用文件系统。

### 取消语义

`tasks/cancel`幂等。若task执中,server试停(查executor协取消)。若已终态,请求no-op。

### 崩溃恢复

Server进程重启时:

1. 载所有持task态。
2. 标任何process死`working` task作`failed`带错`CRASH_RECOVERY`。
3. 保`completed`/`failed`/`cancelled`于其ttl。

### Async task加sampling

Task本身可调`sampling/createMessage`。这是长跑研究task工方式:server task线程需时sample client模型,同时client UI示task作`working`带周期进度更新。

### 何此实验

SEP-1686 2025-11-25发但更广roadmap呼三开问题:持久订阅原语、子task(parent-child task关系)、result-TTL标准化。Expect 2026 spec演。产代码应视Task稳定仅常见case并防未来SDK变用于子task。

## 使用

`code/main.py`实持久task store(文件系统back)和`generate_report`工具于后台线程跑。Client调工具,立得task id,poll`tasks/status`当worker更新进度,完时fetch`tasks/result`。取消工;崩溃恢复模拟杀worker线程并重载态。

看点:

- Task态JSON持于`/tmp/lesson-13-tasks/<id>.json`。
- Worker线程更新`progress`域;poll示它推进。
- Client侧取消设event;worker查并早退。
- "崩溃"态reload标进行task作`failed`带`CRASH_RECOVERY`。

## 交付成果

本课产`outputs/skill-task-store-designer.md`。给长跑工具(研究、build、export),skill设计task store(态形、ttl、持久)、择正taskSupport旗、画进度通知。

## 练习题

1. 跑`code/main.py`。Kick off `generate_report` task,poll status,后fetch result。

2. 中run加`tasks/cancel`调用。验worker honor它并态成`cancelled`。

3. 模崩溃恢复:杀worker线程,重启loader,并观`CRASH_RECOVERY`失败模式。

4. 扩store至SQLite。持久赢同;查询开(列session X所有task)。

5. 读MCP 2026 roadmap post。识最可能影响SDK API设计一年内Tasks相关开问题一。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Task | "长跑工具调用" | 带`_meta.task`用于async执行请求augment |
| SEP-1686 | "Tasks spec" | 2025-11-25加Tasks Spec Evolution Proposal |
| `_meta.task` | "Task包" | 含id、state、ttl请求级metadata |
| taskSupport | "工具旗" | 每工具`forbidden`/`optional`/`required` |
| `tasks/status` | "Poll方法" | 取当前态和可选进度提示 |
| `tasks/result` | "Fetch结果" | 回完payload或未完404 |
| `tasks/cancel` | "停它" | 幂等取消请求 |
| ttl | "保留budget" | Server承诺持task态毫秒 |
| `notifications/tasks/updated` | "态push" | Server发起态变事件 |
| 持久store | "崩溃安态" | 文件系统/SQLite/Redis持久层 |

## 延伸阅读

- [MCP—GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686)——起源提案和全讨论
- [WorkOS—MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows)——设计walk-through带理
- [DeepWiki—MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations)——机制和态机
- [FastMCP—Tasks](https://gofastmcp.com/servers/tasks)——SDK级task实现模式
- [MCP blog—2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)——开问题和2026优先含子task