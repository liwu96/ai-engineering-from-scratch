# Capstone——建完整工具生态

> 阶段13教每件。此capstone线它们入一产形系统:带工具+资源+提示+task+UI MCP server、边OAuth 2.1、RBAC gateway、多server client、A2A子agent调用、OTel trace入collector、CI工具毒检测、和AGENTS.md+SKILL.md bundle。终你可捍每架构择。

**类型:** 构建
**语言:** Python(stdlib,端到端生态框架)
**前置要求:** 阶段13课程01至21
**时间:** ~120分钟

## 学习目标

- 合露工具、资源、提示、和带`ui://` app task MCP server。
- Server前OAuth 2.1 gateway执RBAC和pin hash。
- 写多server client用OTel GenAI attribute端到端trace。
- 委部分workload至A2A子agent;验opacity保。
- 包全栈带AGENTS.md+SKILL.md使其他agent可drive。

## 问题背景

发货"研究和报告"系统:

- 用户问:"总结2026 agent protocol最引用三arXiv paper。"
- 系统:经MCP搜arXiv;经A2A委托paper摘要至专写agent;聚合结果;渲染交互报告作MCP Apps`ui://` resource;每步日志OTel。

阶段13每原语现。这非toy——2026 Anthropic(Claude Research product)、OpenAI(GPTs with Apps SDK)、和第三方发货产研究助手系统有正此形。

## 概念讲解

### 架构

```
[user] -> [client] -> [gateway(OAuth 2.1+RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search(pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report(长)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent(tasks/send)
                                                      |
                                                      +- OTel GenAI span
```

### Trace层级

```
agent.invoke_agent
 ├── llm.chat(kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transition(opaque internal)
 ├── mcp.call -> tools/call generate_report(task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result(completed,回ui:// resource)
 └── llm.chat(终synthesis)
```

一trace id。每span正`gen_ai.*` attribute。

### 安全姿态

- OAuth 2.1+PKCE带资源指示器pin audience至gateway。
- Gateway持上游凭证;用户永不见。
- RBAC:`alice`有`research:read`、`research:write`,可调全工具。`bob`有`research:read`,不可调`generate_report`。
- Pin描述manifest:工具hash变任何server drop。
- 二元律audit:无工具合不可信输入、敏感数据、和后果动作。

### 渲染

终`generate_report` task回内容块加`ui://report/current` resource。Client host(Claude Desktop等)渲染沙箱iframe交互仪表盘。仪表盘含排序paper列表、引用计数、和按钮调`host.callTool('summarize_paper', {arxiv_id})`用于用户点击任paper。

### 打包

全发货作:

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶级workflow
  servers/
    research-mcp/               # MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # A2A agent
  gateway/
    config.yaml                 # RBAC+pin manifest
```

用户`docker compose up` deploy。Claude Code、Cursor、Codex、opencode用户可调`run-research` skill drive系统。

### 每阶段13课程贡献何

| 课程 | Capstone用何 |
|------|-------------|
| 01-05 | 工具接口、provider-portability、并行调用、schema、lint |
| 06-10 | MCP原语、server、client、transport、资源+提示 |
| 11-14 | Sampling、roots+elicitation、async task、`ui://` app |
| 15-17 | 工具毒、OAuth 2.1、gateway+注册 |
| 18 | A2A子agent委托 |
| 19 | OTel GenAI trace |
| 20 | LLM层路由gateway |
| 21 | SKILL.md+AGENTS.md打包 |

## 使用

`code/main.py`缝前课模式入一可跑demo。全stdlib,全in-process使你可端到端读。跑研究和报告景全流:gateway握手、OAuth 2.1模拟、tools/list合并、generate_report作task、A2A调writer、ui:// resource回、OTel span emit。

看点:

- 每hop共享一trace id。
- Gateway策略block第二用户写。
- Task lifecycle转working→completed并回文本加ui://内容。
- A2A调用内态对orchestrator opaque。
- AGENTS.md和SKILL.md是另一agent重现workflow需唯一文件。

## 交付成果

本课产`outputs/skill-ecosystem-blueprint.md`。给产需(研究、摘要、自动化),skill产完整架构:何MCP原语、何gateway控、何A2A调用、何telemetry、何打包。

## 练习题

1. 跑`code/main.py`。注单trace id和span何nest。计demo触阶段13多少原语。

2. 扩demo:加第二后端MCP server(如`bibliography`)并验gateway合并其工具入同命名空间。

3. 换假A2A写agent真agent跑于子进程。用课程19 框架。

4. 加routing gateway中orchestrator和LLM间PII redaction步。验用户查询中email刷。

5. 为会维此系统teammate写AGENTS.md。应少于五分钟读并给他们于Cursor或Codex drive capstone所需全。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Capstone | "Phase-13集成demo" | 用每原语端到端系统 |
| 研究和报告 | "景" | 搜、摘要、渲染模式 |
| 生态 | "全件合" | Server+client+gateway+子agent+telemetry+包 |
| Trace层级 | "单trace id" | 每hop span享trace;parent-child经span id |
| Gateway发token | "转auth" | Client仅见gateway token;gateway持上游cred |
| 合命名空间 | "全工具一flat列表" | 多server于gateway合并,collision prefix |
| Opacity边界 | "A2A调用藏内部" | 子agent推理对orchestrator不可见 |
| 三层栈 | "AGENTS.md+SKILL.md+MCP" | 项目上下文+workflow+工具 |
| Defense-in-depth | "多安全层" | Pin hash、OAuth、RBAC、二元律、audit log |
| Spec合规矩阵 | "何发spec需" | Deliverable对2025-11-25需checklist映 |

## 延伸阅读

- [MCP—Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)——合并参考
- [MCP blog—2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)——协议何往
- [a2a-protocol.org](https://a2a-protocol.org/latest/)——A2A v1.0参考
- [OpenTelemetry—GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——规范trace约定
- [Anthropic—Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)——产agent runtime模式