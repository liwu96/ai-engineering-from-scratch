# MCP安全II——OAuth 2.1、资源指示器、增量Scope

> 远程MCP server需授权,非仅认证。2025-11-25 spec对齐OAuth 2.1+PKCE+资源指示器(RFC 8707)+protected-resource metadata(RFC 9728)。SEP-835加增量scope同意403 WWW-Authenticate步升授权。本课实步升流作态机使你可看每跳。

**类型:** 构建
**语言:** Python(stdlib,OAuth态机模拟器)
**前置要求:** 阶段13课程09(transports),阶段13课程15(安全I)
**时间:** ~75分钟

## 学习目标

- 分资源server和授权server责任。
- 走PKCE保护OAuth 2.1授权码流。
- 用`resource`(RFC 8707)和protected-resource metadata(RFC 9728)防confused-deputy攻击。
- 实步升授权:server回403带WWW-Authenticate求更高scope;client重提示用户同意并重试。

## 问题背景

早MCP(2025前)发远程server带临时API key甚至无auth。2025-11-25 spec闭该gap带完整OAuth 2.1 profile。

三实世需:

- **常远程server。**用户装访问其Notion/GitHub/Gmail远程MCP server。OAuth 2.1带PKCE是正形。
- **Scope升。**笔记server授`notes:read`后可需`notes:write`用于特定动作。非重跑全流,步升(SEP-835)求额外scope。
- **Confused deputy防。**Client持audience-scope Server A token。Server A恶意并试现token给Server B。资源指示器(RFC 8707)pin token至其意audience。

OAuth 2.1非新。新是MCP profile:特定需流(仅授权码+PKCE;无implicit、无默认client credential)、每token请求资源指示器mandatory、和protected-resource metadata发使client知何往。

## 概念讲解

### 角色

- **Client。**MCP client(Claude Desktop、Cursor等)。
- **资源server。**MCP server(笔记、GitHub、Postgres等)。
- **授权server。**发token。可同服务作资源server或分离IdP(Auth0、Keycloak、Cognito)。

MCP profile,资源和授权server CAN同host但SHOULD URL分。

### 授权码+PKCE

流:

1. Client生`code_verifier`(随机)和`code_challenge`(SHA256)。
2. Client redirect用户至`/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权server redirect至`redirect_uri?code=...`。
4. Client POST至`/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权server验verifier hash对存challenge并发access token。
6. Client用token:每请求资源server`Authorization: Bearer ...`。

PKCE防授权码拦截攻击。资源指示器防token别处有效。

### Protected-resource metadata(RFC 9728)

资源server发`.well-known/oauth-protected-resource`文档:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client从资源server发现授权server。减配置——client仅需资源URL。

### 资源指示器(RFC 8707)

Token请求`resource`参数pin token意audience。发token含`aud: "https://notes.example.com"`。另一MCP server收此token查`aud`拒。

### Scope模型

Scope是空格分隔字符串。常见MCP约定:

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*`用于admin能(慎用)
- `profile:read`用于身份

Scope择应最小特权:需时求,需更多步升。

### 步升授权(SEP-835)

用户授`notes:read`。后问agent删笔记。Server回:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client见insufficient_scope错,提示用户同意dialog用于额外scope,执行迷你OAuth流,带新token重试请求。

### Token audience验

每请求:server查`token.aud == self.resource_url`。不匹配=401。停跨server token重用。

### 短活token和rotation

Access token SHOULD短活(1小时默认)。Refresh token每次refresh rotate。Client后台静默refresh。

### 无token passthrough

Sampling server(阶段13课程11)MUST NOT传client token至其他服务。Sampling请求是边界。

### Confused deputy防

Token绑`aud`。Client绑`client_id`。每请求验两。Spec显禁pre-MCP远程工具生态常见"传token"模式。

### Client ID发现

每MCP client于固定URL发其metadata。授权server可fetch client metadata文档发现redirect URI和联系信息。移手动client注册。

### Gateway和OAuth

阶段13课程17示企业gateway何OAuth:gateway持上游server凭证,给client token是gateway发,上游token永不离gateway。翻信任模型——用户一次gateway认证;gateway处N server授权。

## 使用

`code/main.py`模拟全OAuth 2.1步升流作态机。它实:

- PKCE code-verifier/challenge生成。
- 带资源指示器授权码流。
- Protected-resource metadata端点。
- 带audience查token验。
- `insufficient_scope`步升。

本课无HTTP server;态机内存跑使你可trace每跳。阶段13课程17 gateway lesson线实际transport。

## 交付成果

本课产`outputs/skill-oauth-scope-planner.md`。给带工具远程MCP server,skill设计scope集、pinning规、步升策略。

## 练习题

1. 跑`code/main.py`。跟踪两次scope步升流。注步升何跳重复。

2. 加refresh-token rotation:每次refresh发新refresh token并无效旧。模拟偷refresh token于rotation后用并验失败。

3. 实protected-resource metadata端点作真实HTTP响应用stdlib http.server。镜像课程09`/mcp`端点。

4. 设计GitHub MCP server scope层级:读repo、写PR、批PR、合PR、admin。每层间用步升。

5. 读RFC 8707和RFC 9728。识9728中一域MCP用异RFC例。(提示:涉`scopes_supported`。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| OAuth 2.1 | "现代OAuth" | 合RFC强制PKCE并禁implicit流 |
| PKCE | "Proof-of-possession" | 代码验器+挑战击败授权码拦截 |
| 资源指示器 | "Token audience" | RFC 8707`resource`参数pin token至一server |
| Protected-resource metadata | "发现文档" | RFC 9728`.well-known/oauth-protected-resource` |
| 步升授权 | "增量同意" | SEP-835按需加scope流 |
| `insufficient_scope` | "403带WWW-Authenticate" | Server信号重同意更大scope |
| Confused deputy | "跨服务token重用" | 信任持有者不当转发token攻击 |
| 短活token | "Access token TTL" | 快过期bearer;refresh token续 |
| Scope层级 | "最小特权栈" | 渐进scope集层间步升 |
| Client ID metadata | "Client发现文档" | Client发己OAuth metadata URL |

## 延伸阅读

- [MCP—Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization)——规范MCP OAuth profile
- [den.dev—MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/)——2025-11-25改walk-through
- [RFC 8707—Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707)——audience-pinning RFC
- [RFC 9728—OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728)——发现文档RFC
- [Aembit—MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/)——实步升流walk-through