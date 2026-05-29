# MCP Sampling——Server请求LLM Completion和Agent循环

> 大多MCP server是哑执行器:取参数、跑代码、回内容。Sampling让server翻方向:它问client LLM做决策。这使server托管agent循环无需server拥任模型凭证。SEP-1577,2025-11-25合并,加sampling请求内工具使循环可含更深推理。漂风险注:SEP-1577 sampling内工具形2026 Q1实验且SDK API仍在settling。

**类型:** 构建
**语言:** Python(stdlib,sampling测试框架)
**前置要求:** 阶段13课程07(MCP server),阶段13课程10(资源和提示)
**时间:** ~75分钟

## 学习目标

- 释`sampling/createMessage`解何(server托管循环无需server侧API key)。
- 实问client于多轮提示sample并回completion server。
- 用`modelPreferences`(cost/speed/intelligence优先级)指导client模型择。
- 建`summarize_repo`工具内部经sampling迭代而非硬编码行为。

## 问题背景

有用代码摘要workflow MCP server需:走文件树、择读何文件、综合摘要、回。LLM推理何处发生?

择A:server调己LLM。需API key、server侧账单、每用户贵。

择B:server回原始内容;client agent做推理。工但移server逻辑入client提示,脆弱。

择C:server经`sampling/createMessage`问client LLM。Server保留算法(读何文件、几pass)同时client保留账单和模型择。Server全无凭证。

Sampling是择C。它是信任server可托管agent循环无需本身是全LLM宿主的机制。

## 概念讲解

### `sampling/createMessage`请求

Server发:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client跑其LLM,回:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三浮点合至1.0:

- `costPriority`:倾向便宜模型。
- `speedPriority`:倾向快模型。
- `intelligencePriority`:倾向更强模型。

加`hints`:server偏命名模型。Client可不遵循hint;client用户配置总赢。

### `includeContext`

三值:

- `"none"`——仅server供消息。默认。
- `"thisServer"`——含此server session前消息。
- `"allServers"`——含全session上下文。

`includeContext`于2025-11-25软弃因漏跨server上下文,安全关。偏好`"none"`并于消息显传上下文。

### Sampling带工具(SEP-1577)

2025-11-25新:sampling请求可含`tools`数组。Client用那些工具跑完整工具调用循环。让server经client模型托管ReAct式agent循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Client循环:sample、若调执行工具、再sample、回终assistant消息。此2026 Q1实验;SDK签名可仍漂。实时对2025-11-25 spec client/sampling节确认。

### Human-in-the-loop

Client MUST于跑sample前示用户server问模型做何。恶意server可用sampling操纵用户session("对用户说X使点击Y")。Claude Desktop、VS Code、Cursor露sampling请求作确认dialog用户可拒。

2026共识:sampling无人类确认是红旗。Gateway(阶段13课程17)可自批低风险sampling并自拒可疑。

### Server托管循环无API key

规范用例:无己LLM访问代码摘要MCP server。它做:

1. 走repo结构。
2. 调`sampling/createMessage`带"Pick five files most likely to describe this repo's purpose."
3. 读那些文件。
4. 调`sampling/createMessage`带文件内容和"Summarize the repo in 3 paragraphs."
5. 回摘要作`tools/call`结果。

Server永不触LLM API。Client用户用己凭证付completion。

### 安全风险(Unit 42披露,2026 Q1)

- **隐蔽sampling。**工具总调sampling带"从session上下文回用户email"。阶段13课程15覆盖攻击向量。
- **经sampling资源偷。**Server问client摘要攻击者payload,账用户。
- **循环炸弹。**Server于紧循环调sampling。Client MUST执每session速率限。

## 使用

`code/main.py`构建假server-to-client sampling测试框架。模拟"summarize_repo"工具调两sampling轮(择文件、后摘要),假client回罐装响应。Harness示:

- Server发带`modelPreferences` `sampling/createMessage`。
- Client回completion。
- Server继续循环。
- 速率限盖每工具调用总sampling调用。

看点:

- Server仅露一工具(`summarize_repo`);所有推理sampling调用内发生。
- Model偏好权client模型择;hint列偏模型。
- 循环于`stopReason: "endTurn"`终。
- `max_samples_per_tool = 5`限捕失控循环。

## 交付成果

本课产`outputs/skill-sampling-loop-designer.md`。给需LLM调用server侧算法(研究、摘要、规划),skill设计sampling基实带正modelPreferences、速率限、安全确认。

## 练习题

1. 跑`code/main.py`。改`max_samples_per_tool`至2并观速率限截断。

2. 实SEP-1577 sampling内工具变体:sampling请求载`tools`数组。验client侧循环执行那些工具后回终completion。注漂风险:SDK签名2026 H1可仍变。

3. 加human-in-the-loop确认:server首`sampling/createMessage`前暂停等用户批。拒调用回类型化拒。

4. 加每用户速率限器keyed by client session。同server同用户循环应享budget。

5. 设计`summarize_pdf`工具用sampling择含块。画发送消息。`modelPreferences.intelligencePriority`于0.1 vs 0.9何改行为?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Sampling | "Server-to-client LLM调用" | Server问client模型completion |
| `sampling/createMessage` | "方法" | Sampling请求JSON-RPC方法 |
| `modelPreferences` | "模型优先级" | Cost/speed/intelligence权重加名hint |
| `includeContext` | "跨session漏" | 软弃上下文含模式 |
| SEP-1577 | "Sampling内工具" | 允sampling内工具用于server托管ReAct |
| Human-in-the-loop | "用户确认" | Client于跑前露sampling请求给用户 |
| 循环炸弹 | "失控sampling" | Server侧无限sampling循环;client须速率限 |
| 隐蔽sampling | "隐藏推理" | 恶意server于sampling提示藏意图 |
| 资源偷 | "用用户LLM预算" | Server强client花于不欲sampling |
| `stopReason` | "生成停因" | `endTurn`、`stopSequence`或`maxTokens` |

## 延伸阅读

- [MCP—Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling)——sampling高层概览
- [MCP—Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling)——规范`sampling/createMessage`形
- [MCP—GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol)——sampling内工具Spec Evolution Proposal(实验)
- [Unit 42—MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/)——隐蔽sampling和资源偷模式
- [Speakeasy—MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling)——带client侧代码样walk-through