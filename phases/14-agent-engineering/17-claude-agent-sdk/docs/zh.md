# Claude Agent SDK——Subagent和Session Store

> Claude Agent SDK是Claude Code 框架 library形。Built-in tool、subagent用于context isolation、hook、W3C trace传播、session store parity。Claude Managed Agent是托管替代用于长运行异步工作。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程10(Skill Library)
**时间:** ~75分钟

## 学习目标

- 释Anthropic Client SDK(raw API)和Claude Agent SDK(框架形)间diff。
- 描述subagent——parallelization和context isolation——和何时reach它们。
- 名Python SDK session store面(`append`、`load`、`list_session`、`delete`、`list_subkey`)和`--session-mirror`角色。
- 实stdlib 框架带built-in tool、isolated context subagent spawn、lifecycle hook、和session store。

## 问题背景

Raw LLM API给你一round-trip。产agent需tool执行、MCP server、lifecycle hook、subagent spawn、session持久、trace传播。Claude Agent SDK ship此形作library——同框架 Claude Code用、exposed用于custom agent。

## 概念讲解

### Client SDK vs Agent SDK

- **Client SDK(`anthropic`)。**Raw Message API。你own loop、tool、state。
- **Agent SDK(`claude-agent-sdk`)。**Built-in tool执行、MCP连接、hook、subagent spawn、session store。Claude Code loop作library。

### Built-in tool

SDK ship 10+ tool out of box:file read/write、shell、grep、glob、web fetch、更多。Custom tool经标准tool-schema interface注册。

### Subagent

Anthropic文档两目的:

1. **Parallelization。**并发跑独立工作。"找这20 module每测试file"是20 parallel subagent task。
2. **Context isolation。**Subagent用己context window;仅结果回orchestrator。Orchestrator budget preserved。

Python SDK近加:`list_subagent()`、`get_subagent_message()`用于读subagent transcript。

### Session store

TypeScript protocol parity:

- `append(session_id,message)`——加turn。
- `load(session_id)`——restore对话。
- `list_session()`——enumerate。
- `delete(session_id)`——cascade至subagent session。
- `list_subkey(session_id)`——列subagent key。

`--session-mirror`(CLI flag)mirror transcript至外文件当它stream、用于debug。

### Hook

可注册lifecycle hook:

- `PreToolUse`、`PostToolUse`——gate或audit tool call。
- `SessionStart`、`SessionEnd`——setup和teardown。
- `UserPromptSubmit`——act于用户input模型见前。
- `PreCompact`——context compaction前run。
- `Stop`——agent exit cleanup。
- `Notification`——side-channel alert。

Hook是pro-workflow(阶段14课程reference)和类似系统加cross-cutting behavior方式。

### W3C trace context

Caller上active OTel span经W3C trace context header propagate入CLI subprocess。全多process trace作一trace现于你backend。

### Claude Managed Agent

托管替代(beta header`managed-agent-2026-04-01`)。长运行异步工作、built-in prompt caching、built-in compaction。Trade控用于托管infrastructure。

### 何此模式错

- **Subagent over-spawn。**100 tiny task spawn 100 subagent。Overhead dominate。Batch instead。
- **Hook creep。**每team加hook;startup time balloon。Quarterly review hook。
- **Session bloat。**Session累积;size增。用`list_session`+expiry policy。

## 构建

`code/main.py`实SDK形于stdlib:

- `Tool`、`ToolRegistry`带built-in`read_file`、`write_file`、`list_dir`。
- `Subagent`——私有context、isolated run、结果回。
- `SessionStore`——append、load、list、delete、list_subkey。
- `Hook`——`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- Demo:main agent spawn 3 subagent parallel(每isolated)、聚合结果、persist session。

跑:

```
python3 code/main.py
```

Trace显subagent context isolation(orchestrator context size bounded)、hook执行、和session持久。

## 使用

- **Claude Agent SDK**用于Claude-first product欲Claude Code 框架形。
- **Claude Managed Agent**用于托管长运行异步工作。
- **OpenAI Agents SDK**(课程16)用于OpenAI-first counterpart。
- **LangGraph+custom tool**若你欲graph形state machine。

## 交付成果

`outputs/skill-claude-agent-scaffold.md`scaffold Claude Agent SDK app带subagent、hook、session store、MCP server attachment、和W3C trace传播。

## 练习题

1. 加subagent spawner batch 20 task入5 parallel subagent组。测orchestrator context size vs one-per-task。
2. 实`PreToolUse`hook rate-limit`write_file` call(每session每分5)。Trace行为。
3. Wire`list_subkey`render subagent tree。Deep nesting何看?
4. 移toy至真`claude-agent-sdk` Python package。Tool registration何变?
5. 读Claude Managed Agent docs。何你从self-hosted switch至managed?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent SDK | "Claude Code作library" | Harness形:tool、MCP、hook、subagent、session store |
| Subagent | "Child agent" | 分离context、己budget;结果bubble up |
| Session store | "对话DB" | Persist、load、list、delete turn带subagent cascade |
| Hook | "Lifecycle callback" | Pre/post tool、session、prompt submit、compact、stop |
| W3C trace context | "跨process trace" | Parent span propagate入CLI subprocess |
| Managed Agent | "托管框架" | Anthropic-hosted长运行异步工作 |
| `--session-mirror` | "Transcript mirror" | Session turn stream时写至外文件 |
| MCP server | "Tool surface" | 外tool/resource source attach agent |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Code library形
- [Anthropic,Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——产模式
- [Claude Managed Agent overview](https://platform.claude.com/docs/en/managed-agent/overview)——托管替代
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——counterpart