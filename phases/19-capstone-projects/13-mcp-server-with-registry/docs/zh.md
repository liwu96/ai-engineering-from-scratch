# 毕业项目 13 —— MCP服务带注册表和治理

> Model Context Protocol停作未来成2026默认工具使用spec。Anthropic、OpenAI、Google、和每主要IDE皆ship MCP clients。Pinterest发布内部MCP servers生态。AAIF Registry于`.well-known`形式化能力metadata。AWS ECS发布参考无状态部署。Block goose-agent将同协议放托管助手内。2026产形态是: StreamableHTTP传输、OAuth 2.1 scopes、OPA policy gating、和让平台团队发现、验证、启用servers的注册表。端到端建此。

**类型:** 毕业项目
**语言:** Python (服务、via FastMCP)或TypeScript (@modelcontextprotocol/sdk)、Go (registry服务)
**前置要求:** 第11阶段(LLM工程)、第13阶段(工具和MCP)、第14阶段(智能体)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P11 · P13 · P14 · P17 · P18
**时间:** 25小时

## 问题背景

MCP成工具使用lingua franca。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI、和每托管智能体现皆消费MCP servers。产挑战非authoring servers (FastMCP使易)而规模部署带企业要求: 每租户OAuth scopes、破坏性工具OPA policy、StreamableHTTP无状态scaling、发现注册表、每工具调用audit logs。Pinterest内部MCP生态和AAIF Registry spec设2026 bar。

将建MCP server暴露10内部工具(Postgres只读、S3 listing、Jira、Linear、Datadog等)、平台发现注册表UI、和破坏性工具人工审批门。负载测试演示StreamableHTTP横向scaling。审计trail满足企业安全审查。

## 概念讲解

MCP 2026修订强制StreamableHTTP作默认传输。与早先stdio-and-SSE形态不同、StreamableHTTP默认无状态: 单HTTP端点接受JSON-RPC请求、流响应、并支持长连接notifications。无状态意味负载均衡后横向可scale。

授权是OAuth 2.1带每工具scope。令牌携带scope如`jira:read`、`s3:list`、`postgres:query:readonly`。MCP server于工具调用时检查scopes、非仅session开始。高风险工具、server拒任何scope未于最近N分钟elevated到`approved:by:human`的调用 — elevation来自Slack review card。

注册表是独立服务。每MCP server暴露`.well-known/mcp-capabilities`文档带工具manifest、传输URL、auth要求。注册表poll、验证、索引。平台团队用注册表UI看何工具可用、何scopes需、何团队own。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务框架: FastMCP (Python)或`@modelcontextprotocol/sdk` (TypeScript)
- 传输: StreamableHTTP over HTTPS (无状态)
- Auth: OAuth 2.1带workload identity via SPIFFE / SPIRE
- Policy: OPA / Rego规则每工具; policy decision服务每请求
- 注册表: 自托管、消费`.well-known/mcp-capabilities` manifests
- 人工审批: Slack交互消息破坏性工具
- 部署: AWS ECS Fargate或Fly.io、每租户一服务或共享带租户scoping
- Audit: 结构JSONL每租户bucket带每调用lineage

## 动手实践

1. **工具面。** 暴露10内部工具: Postgres只读query、S3 list objects、Jira search/fetch、Linear search/fetch、Datadog metric query、PagerDuty on-call lookup、GitHub只读、Notion搜索、Slack搜索、Salesforce读。每工具带类型schema和scope标签。

2. **FastMCP服务。** 挂载工具。配StreamableHTTP传输。加OAuth token introspection和scope enforcement中间件。

3. **OPA policy。** Rego policy每工具: 何scopes允许调用、何PII redaction应用、何payload-size caps应用。每工具调用调decision服务。

4. **注册表服务。** 独立Go或TS服务poll注册servers `.well-known/mcp-capabilities`、JSON Schema验证、并暴露list / search / validate / enable-disable UI。

5. **能力manifest。** 每server暴露`.well-known/mcp-capabilities`带: 工具列表、auth要求、传输URL、owner团队、SLO。

6. **破坏性工具分离。** 变态态工具(Jira create、Linear create、Postgres write)于第二MCP server带更严auth流: tokens须有`approved:by:human` scope经Slack card于15分钟内elevated。

7. **审计日志。** 每租户仅追加JSONL: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写前Presidio PII redaction。

8. **负载测试。** 100并发clients于StreamableHTTP。演示横向scaling加第二replica; 显load balancer重分布无session stickiness。

9. **一致性测试。** 运行官方MCP一致性套件于两servers。通过所有强制section。

## 使用它

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 产出成果

`outputs/skill-mcp-server.md`描述deliverable。产级MCP服务 + 注册表 + 审计层于内部工具带OAuth 2.1 scopes和OPA gating。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | Spec一致性 | StreamableHTTP + 能力manifest通过MCP一致性测试 |
| 20 | 安全 | Scope enforcement、OPA覆盖每工具、secret卫生 |
| 20 | 可观测性 | 每工具调用审计日志带PII redaction |
| 20 | 规模 | 100-client负载测试横向scale演示 |
| 15 | 注册表UX | 发现 / 验证 / enable-disable workflow |
| **100** | | |

## 练习题

1. 加新工具(Confluence搜索)。经注册表验证流ship不触core服务。

2. 写OPA policy redact含`email`、`ssn`、或`phone`名列的Postgres query结果。用probe query exercise。

3. Benchmark StreamableHTTP vs stdio本地延迟。报每调用p50/p95。

4. 实现每租户quota: 每工具每租户每分钟最多N调用。经第二OPA rule强制。

5. 从[mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance)运行MCP一致性套件并修复每失败。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| StreamableHTTP | "2026 MCP传输" | 无状态HTTP + streaming; 网络servers取代SSE + stdio |
| 能力manifest | "Well-known文档" | `.well-known/mcp-capabilities`带工具列表、auth、传输URL |
| OPA / Rego | "Policy引擎" | Open Policy Agent授权工具调用对外部规则 |
| Scope elevation | "Approved-by-human" | 经Slack审批授予短期scope、破坏性工具需 |
| 注册表 | "工具发现" | 从能力manifest索引MCP servers的服务 |
| Workload identity | "SPIFFE / SPIRE" | OAuth token issuance加密服务identity |
| 一致性套件 | "Spec测试" | 官方MCP测试battery于StreamableHTTP + 工具manifest正确性 |

## 延伸阅读

- [Model Context Protocol 2026路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力metadata、注册表
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026注册表spec
- [AWS ECS参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考产部署
- [Pinterest内部MCP生态](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP usage](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python服务框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity参考