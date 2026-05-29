# A2A——Agent-to-Agent协议

> MCP是agent-to-tool。A2A(Agent2Agent)是agent-to-agent——让异框架opaque agent协作开协议。Google 2025年4月发,2025年6月捐Linux Foundation,2026年4月达v1.0带150+支持者含AWS、Cisco、Microsoft、Salesforce、SAP、ServiceNow。它吸收IBM ACP并加AP2支付扩展。本课走Agent Card、Task生命周期、和两transport绑定。

**类型:** 构建
**语言:** Python(stdlib,Agent Card+Task框架)
**前置要求:** 阶段13课程06(MCP基础),阶段13课程08(MCP client)
**时间:** ~75分钟

## 学习目标

- 分agent-to-tool(MCP)和agent-to-agent(A2A)用例。
- 于`/.well-known/agent.json`发带技能和端点metadata Agent Card。
- 走Task生命周期(submitted→working→input-required→completed/failed/canceled/rejected)。
- 用带Part(text、file、data)Message和Artifact作输出。

## 问题背景

客服agent需委托报告写至专写agent。pre-A2A择:

- 自定义REST API。工但每pairing是特例。
- 共代码库。需两agent跑同框架。
- MCP。不fit:MCP用于调工具,非两agent协作时保每agent opaque内推理。

A2A填gap。它建模交互作一agent发送Task至另一,带生命周期、消息、和artifact。被调agent内态opaque——调用者仅看task态转换和终输出。

A2A是"让跨框架agent互话"协议。它不替MCP;两互补。

## 概念讲解

### Agent Card

每A2A兼容agent于`/.well-known/agent.json`发card:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是URL基:fetch card,学A2A端点URL,枚技能。

### 签名Agent Card(AP2)

AP2扩展(2025年9月)加Agent Card加密签名。发布者用JWT签己card;消费者验。防冒。

### Task生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (经message循环)
```

Client经`tasks/send`发起。被调agent转态;client经SSE订阅态更新或poll。

### Message和Part

消息载一或多Part:

- `text`——纯内容。
- `file`——base64 blob带mimeType。
- `data`——类型化JSON payload(被调agent结构输入)。

例:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifact

输出是Artifact,非裸字符串。Artifact是命名、类型输出:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifact可作chunk流。调用者累积。

### 两transport绑定

1. **JSON-RPC over HTTP。**`/a2a`端点,POST用于请求,可选SSE用于流。默认绑定。
2. **gRPC。**用于gRPC原生企业环境。

两绑定载同逻辑消息形。

### Opacity保

关键设计原则:被调agent内态opaque。调用者看task态和artifact。被调agent chain-of-thought、其工具调用、其子agent委托——全不可见。不同于MCP工具调用透明。

理:A2A使竞争者协作不露内部。A2A可是"调此客服agent"不调用者学该agent何实服务。

### Timeline

- **2025-04-09。**Google宣布A2A。
- **2025-06-23。**捐Linux Foundation。
- **2025-08。**吸收IBM ACP。
- **2025-09。**AP2扩展(Agent Payments)发。
- **2026-04。**v1.0发带150+支持组织。

### 与MCP关系

| 维 | MCP | A2A |
|----|-----|-----|
| 用例 | Agent-to-tool | Agent-to-agent |
| Opacity | 透明工具调用 | Opaque内推理 |
| 典型调用者 | Agent runtime | 另一agent |
| 态 | 工具调用结果 | Task带生命周期 |
| 授权 | OAuth 2.1(阶段13课程16) | JWT签名Agent Card(AP2) |
| Transport | Stdio/Streamable HTTP | JSON-RPC over HTTP/gRPC |

欲调特定工具用MCP。欲委托整task至另一agent用A2A。多产系统用两:agent用MCP作其工具层和A2A作其协作层。

## 使用

`code/main.py`实最小A2A框架:研究agent发其card,写agent收`tasks/send`带part含PDF和文本指令,转态working→input_required→working→completed,并回文本artifact。全stdlib;用内存transport聚焦消息形。

看点:

- Agent Card JSON形。
- Task id分配和态转换。
- 带混合类型part Message。
- Task中途input-required分支。
- Completion上Artifact回。

## 交付成果

本课产`outputs/skill-a2a-agent-spec.md`。给应可被其他agent调新agent,skill产Agent Card JSON、技能schema、和端点蓝图。

## 练习题

1. 跑`code/main.py`。跟踪完整Task生命周期,含被调agent求澄清input-required pause。

2. 加签名Agent Card。用card规范JSON上HMAC签。写验器并验mutated card失败。

3. 实task流:写agent经SSE发三增量artifact chunk并调用者累积它们。

4. 设计包MCP server A2A agent。映每MCP工具至A2A技能。注trade-off——失何opacity?

5. 读A2A v1.0 announcement并识2026年4月任框架未实一特性。(提示:涉多hop task委托。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| A2A | "Agent-to-Agent协议" | Opaque agent协作开协议 |
| Agent Card | "`.well-known/agent.json`" | 发描述agent技能和端点metadata |
| Skill | "可调用单元" | Agent支持命名操作(类比MCP工具) |
| Task | "委托单元" | 带生命周期和终artifact工作项 |
| Message | "Task输入" | 载Part(text、file、data) |
| Part | "类型块" | Message`text`/`file`/`data`元素 |
| Artifact | "Task输出" | Completion上回命名、类型输出 |
| AP2 | "Agent Payments Protocol" | Agent Card信任和支付签名扩展 |
| Opacity | "黑箱协作" | 被调agent内部对调用者藏 |
| Input-required | "Task暂停" | Agent需更多信息时生命周期态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/)——规范A2A specification
- [a2aproject/A2A—GitHub](https://github.com/a2aproject/A2A)——参考实现和SDK
- [Linux Foundation—A2A发press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)——2025年6月治理转移
- [Google Cloud—A2A协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)——roadmap和partner momentum
- [Google Dev—A2A 1.0里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258)——v1.0发注和backward-compat指导