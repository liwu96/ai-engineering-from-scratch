# MCP Gateway和注册——企业控面板

> 企业不可让每dev装随机MCP server。Gateway集中auth、RBAC、审计、速率限、缓存、工具毒检测,后露合并工具面作单MCP端点。官方MCP注册(Anthropic+GitHub+PulseMCP+Microsoft,命名空间验)是规范上游。本课命名gateway fit、走最小实、并survey 2026 vendor landscape。

**类型:** 学习
**语言:** Python(stdlib,最小gateway)
**前置要求:** 阶段13课程15(工具毒),阶段13课程16(OAuth 2.1)
**时间:** ~45分钟

## 学习目标

- 释MCP gateway居处(MCP client和多后端MCP server间)。
- 实五gateway责任:auth、RBAC、audit、速率限、策略。
- Gateway层执pin-tool-hash manifest。
- 分官方MCP注册和元注册(Glama、MCPMarket、MCP.so、Smithery、LobeHub)。

## 问题背景

Fortune 500有30批MCP server、5000开发者、合规和审计需、和安全团队欲集中策略。让每开发者IDE装任意server非starter。

Gateway模式:

1. Gateway跑作单Streamable HTTP端点开发者连。
2. Gateway持每后端MCP server凭证。
3. 每开发者请求经gateway己OAuth认证和scope。
4. Gateway路由调用至后端server,apply策略。
5. 全调用日志审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——皆2025-2026发gateway或gateway特性。

同时,官方MCP注册作规范上游:策、命名空间验、reverse-DNS命名server gateway可pull。元注册(Glama、MCPMarket、MCP.so、Smithery、LobeHub)跨多源聚合server。

## 概念讲解

### 五gateway责任

1. **Auth。**OAuth 2.1识开发者;映至用户角色。
2. **RBAC。**每用户策略:何server、何工具、何scope。
3. **Audit。**每调用日志何人、何物、何时、结果。
4. **Rate limit。**每用户/每工具/每server cap防滥用。
5. **Policy。**拒毒描述、执二元律、redact PII。

### Gateway作单端点

对开发者,gateway看像一MCP server。内部路由至N后端。Session id(阶段13课程09)于边界重写。

### 凭证vaulting

开发者永不见后端token。Gateway持(或代理至identity provider)。`notes:read` gateway开发者可转访问笔记MCP server带gateway己后端凭证——但仅策略绑转访问。

### Gateway工具hash pinning

Gateway持批工具描述manifest(SHA256 hash)。发现时,fetch每后端`tools/list`,比hash manifest,并删任何描述突变工具。这是阶段13课程15 rug-pull防御集中apply。

### Policy-as-code

进gateway用OPA/Rego、Kyverno或Styra表达策略。规则"用户`alice`仅可调org `acme` repo上`github.open_pr`"声明编码。简gateway用手码Python。两形valid。

### Session-aware路由

用户session混server时,gateway复用:开发者单MCP session持N后端session,每server一。任后端通知经gateway路由至开发者session。

### 命名空间合并

Gateway从所有后端合并工具命名空间,典型collision prefix。`github.open_pr`、`notes.search`。使路由无歧。

### 注册

- **官方MCP注册(`registry.modelcontextprotocol.io`)。**Anthropic、GitHub、PulseMCP、Microsoft托管下发。命名空间验(reverse-DNS:`io.github.user/server`)。基质量pre-filter。
- **Glama。**搜索中心元注册聚合多源。
- **MCPMarket。**商业倾向目录带vendor列。
- **MCP.so。**社区目录;开提交。
- **Smithery。**包管理式安装流。
- **LobeHub。**LobeChat app内UI集成注册。

企业gateway默认从官方注册pull,允admin策元注册加,拒任unpin。

### Reverse-DNS命名

官方注册强公开server reverse-DNS名:`io.github.alice/notes`。命名空间防squatting并使信任委托更清。

### Vendor survey,2026年4月

| Vendor | 强 |
|--------|-----|
| Cloudflare MCP Portals | Edge托管;OAuth集成;免费tier |
| Kong AI Gateway | K8s原生;细策略;日志OpenTelemetry |
| IBM ContextForge | 企业IAM;合规;审计export |
| TrueFoundry | DevOps倾向;metric首 |
| MintMCP | 开发者平台导向 |
| Envoy AI Gateway | 开源;可定制filter |

阶段17(产infra)gateway ops更深。

## 使用

`code/main.py`发约150行最小gateway:假Bearer token认证用户,持每用户RBAC策略,路由请求至两后端MCP server,每调用写审计日志,执速率限,并拒任何后端工具描述hash不匹配pin manifest。

看点:

- `RBAC` dict keyed by `user_id`带允`server_tool`条。
- `AUDIT_LOG`是append-only事件列表。
- 速率限用每用户token bucket。
- Pin manifest是`server::tool -> hash` dict。

## 交付成果

本课产`outputs/skill-gateway-bootstrap.md`。给企业MCP计划(用户、后端、合规),skill产gateway配置spec。

## 练习题

1. 跑`code/main.py`。允用户调;后不许用户调;后速率限超burst。验三流。

2. 加策略从结果redact PII回client前。用简单regex pass对SSN形字符串;注gap(email、电话)。

3. 扩审计日志emit OpenTelemetry GenAI span。阶段13课程20覆盖精确属性。

4. 设计50开发者5后端(笔记、github、postgres、jira、slack)RBAC策略。谁每只读?谁写?

5. 读Cloudflare企业MCP post从头至尾。识Cloudflare发一特性此stdlib gateway无。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Gateway | "MCP proxy" | Client和后端间集中server |
| 凭证vaulting | "后端token留server侧" | 开发者永不见上游token |
| Session-aware路由 | "多后端session" | Gateway每开发者session复N后端session |
| 工具hash pinning | "批manifest" | 每批工具描述SHA256;集中block rug-pull |
| RBAC | "每用户策略" | 工具和server角色基访问控 |
| Policy-as-code | "声明规则" | OPA/Rego、Kyverno、Styra策略gateway执 |
| 审计日志 | "何人何物何时" | 合规append-only事件日志 |
| 速率限 | "每用户token bucket" | 每分钟cap防滥用 |
| 官方MCP注册 | "规范上游" | `registry.modelcontextprotocol.io`,命名空间验 |
| Reverse-DNS命名 | "注册命名空间" | `io.github.user/server`约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/)——规范上游,命名空间验
- [Cloudflare—Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/)——带OAuth和策略gateway模式
- [agentic-community—MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry)——开源参考gateway
- [TrueFoundry—What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway)——特性比文章
- [IBM—MCP context forge](https://github.com/IBM/mcp-context-forge)——IBM企业gateway