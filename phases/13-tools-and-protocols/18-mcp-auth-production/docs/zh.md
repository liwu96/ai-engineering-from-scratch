# MCP产认证——DCR、JWKS Rotation、iii原语Audience-Pinned Token

> 课程16立内存OAuth 2.1态机。2026,每MCP server发货真实org坐后产认证:动态client注册(RFC 7591)、授权server metadata发现(RFC 8414)、3 am token验不断JWKS rotation、和拒confused-deputy重用audience-pinned token。本课线所有经iii原语——`iii.registerTrigger`用于HTTP和cron、`iii.registerFunction`用于auth逻辑、`state::set/get`用于缓存key——使auth面可观测、可重启、可重放如engine每其他workload。

**类型:** 构建
**语言:** Python(stdlib,iii原语mock于课环境)
**前置要求:** 阶段13课程16(OAuth 2.1态机),阶段13课程17(gateways)
**时间:** ~90分钟

## 学习目标

- 经RFC 8414 metadata发现授权server并验契约。
- 实RFC 7591动态client注册使MCP client无admin干预enroll。
- 用cron触发缓存和rotate JWKS key使签名验存key roll-over。
- Pin token至单MCP资源用RFC 8707资源指示器并拒confused-deputy重用。
- 线每端点和后台job作iii原语——HTTP trigger、cron trigger、命名函数、`state::*`读——使单重启重建auth面。
- 读IdP能矩阵并拒部署当IdP不可满足MCP auth profile。

## 问题背景

课程16模拟器内存跑OAuth 2.1。产有内存仅模拟器不见三操gap。

首gap是enrollment。真实org跑数百MCP server和数千MCP client。Operator不手注册每Cursor用户作OAuth client。RFC 7591动态client注册让client对授权server`POST /register`并当场收`client_id`(可选`client_secret`)。Server于RFC 8414 metadata发`registration_endpoint`;client无out-of-band配置发现。

次gap是key rotation。JWT验赖授权server签名key,发作JSON Web Key Set(JWKS)。授权server按schedule rotate(常小时,有时incident response更快)。MCP server启动时一次fetch JWKS验好至rotation窗口——后每请求失败至重启。产线JWKS作缓存值带refresh job于前key过期前overwrite cache,加cache miss时fall-back fetch用于cache新key token到案。

三gap是audience绑定。课程16引RFC 8707资源指示器。产,该指示器成每请求硬claim查。MCP server比`token.aud`对己规范资源URL并HTTP 401拒不匹配。这是唯一防上游MCP server(或恶意client持一server token)于同信任mesh另一server replay token。

本课视每gap作iii原语。Metadata文档是HTTP trigger回函数输出。JWKS rotation是cron触发调`auth::rotate-jwks`,写`state::set("auth/jwks/<issuer>", ...)`. JWT验是函数他人经`iii.trigger("auth::validate-jwt", token)`调。MCP server本身只是另一HTTP触发验后dispatch。重启engine:触发注册重建;态存;auth面可操无需手动reconciliation。

## 概念讲解

### RFC 8414——OAuth Authorization Server Metadata

`/.well-known/oauth-authorization-server`文档述client需全:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Client给MCP资源URL链发现:RFC 9728 `oauth-protected-resource`(资源server文档)命issuer,后`oauth-authorization-server`(此RFC)命每端点。Client永不硬编码授权URL。

信任IdP用于MCP前验契约:

- `code_challenge_methods_supported`含`S256`(PKCE per RFC 7636)。
- `grant_types_supported`含`authorization_code`并拒`password`和`implicit`。
- `registration_endpoint`现(RFC 7591支持)。
- `response_types_supported`恰`["code"]`用于OAuth 2.1。

缺任, MCP server拒部署此IdP。部署manifest错,非代码。

### RFC 9728(回顾)——Protected Resource Metadata

课程16覆盖RFC 9728。产delta:此文档是client找*此*MCP server信任授权server唯一处。单MCP server可接多IdP token(一staff、一partner)。RFC 9728声明集;RFC 8414文档每IdP支持何。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591——Dynamic Client Registration

无DCR,每MCP client(Cursor、Claude Desktop、自定义agent)需IdP admin out-of-band交换。有DCR,client post:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server回`client_id`和`registration_access_token`用于后更新:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`是跑用户设备MCP client正确默认。它们仅得`client_id`——无`client_secret`exfiltrate。PKCE供public client需proof-of-possession。

三产pitfall:

- 注册端点须按源IP速率限。无,敌actor脚本百万假注册并耗`client_id`命名空间。iii使这易:注册HTTP触发调`auth::rate-limit`函数后dispatch至registrar。
- `software_statement`(签名JWT担保client)某些企业IdP需。课程mock跳;产线验步拒除localhost redirect URI外未签名注册。
- `registration_access_token`须存hash而非plaintext。此token偷意味攻击者可重写client redirect URI。

### RFC 8707(回顾)——资源指示器

课程16立形。产规:每token请求含`resource=<canonical-mcp-url>`,MCP server于每调用验`token.aud`匹配己资源URL。若MCP server可达`https://notes.example.com/mcp`,规范URL是`https://notes.example.com`——路径组件除外使单server host一audience下多路径。

### RFC 7636(回顾)——PKCE

PKCE是OAuth 2.1强制。课程授权码流总载`code_challenge`和`code_verifier`。Server拒无verifier或verifier不hash对存challenge token请求。

### MCP Spec 2025-11-25 Auth Profile

MCP spec(2025-11-25)精于MCP server授权层须做:

- 发`/.well-known/oauth-protected-resource`(RFC 9728)。
- 仅经`Authorization: Bearer ...`接token。
- 验`aud`、`iss`、`exp`和每请求需scope。
- 回`WWW-Authenticate`载`Bearer error=...`用于每401和403,含适用处`scope=`和`resource=`参数。
- 拒`aud`不匹配规范资源token。
- 拒`iss`不在protected-resource metadata`authorization_servers`列表token。

OAuth 2.1 draft是substrate;RFC 8414/7591/8707/9728+RFC 7636是surface;MCP spec是profile。

### IdP能矩阵

非每IdP支持完整MCP profile。下矩阵文档2025-11-25 spec事实能声明。是*部署gate*,非推荐。

| IdP类 | RFC 8414 metadata | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 注 |
|---|---|---|---|---|---|
| Self-hosted(Keycloak) | yes | yes | yes(24.x起) | yes | 课程MCP profile参考IdP;支持每RFC端到端 |
| Enterprise SSO(Microsoft Entra ID) | yes | yes(premium tier) | yes | yes | DCR可用异tenant tier;部署前验target tenant |
| Enterprise SSO(Okta) | yes | yes(Okta CIC/Auth0) | yes | yes | DCR Auth0(现Okta CIC)可用;经典Okta org需admin预注册 |
| Social login IdP(通用) | varies | rarely | rarely | yes | 大多social IdP视client作静态partner;勿赖DCR。仅作identity源,叠己MCP-aware授权server |
| Custom/homegrown | depends | depends | depends | depends | 若发货己,发货完整profile。跳上四RFC任一破MCP auth契约 |

部署manifest拒规:若选IdP不回`registration_endpoint`且不列`S256`于`code_challenge_methods_supported`,MCP server拒start。无降级模式。

### JWKS rotation模式带iii

产失模式是陈JWKS cache。用cron触发和`state::*`cache解:

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时,cron触发调`auth::rotate-jwks`,fetch `<issuer>/.well-known/jwks.json`并写`state::set("auth/jwks/<issuer>", {keys, fetched_at})`. 验器读`state::get`。Token`kid`cache缺触发同步`auth::rotate-jwks`调用作fall-back。这同时处两案:scheduled rotation(cron)和key-overlap window(同步fall-back)。

态形:

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时两key是稳态。授权server通过引入下key(`k_2026_04`)后退前(`k_2026_03`)rotate,故旧key发token保有效至过期。Cache持union;验器按`kid`pick。

### iii原语线(本课实部分)

五原语合auth面:

```python
# 1. RFC 8414 metadata文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591动态client注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT验作可调用函数(资源server触发它)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 增量scope步升(SEP-835自L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron驱动JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP server本身不直接调验。它做:

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

此间接是iii bet。明天你swap验器用于fanout并行咨询两IdP,或加span emitter,或缓存正验。MCP server不改。

### Confused-deputy walk-through带audience绑定

Server A(`notes.example.com`)和Server B(`tasks.example.com`)皆注册于同授权server。Server A妥协。攻击者取用户笔记token并replay对Server B。

Server B验器:

1. 解JWT,按`kid`fetch JWKS,验签名。
2. 查`iss`对protected-resource metadata`authorization_servers`。(过——同IdP.)
3. 查`aud == "https://tasks.example.com"`。(败——token`aud`是`https://notes.example.com`。)
4. 回401带`WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

Audience claim是协议层唯一防此攻击。性能跳它是产最常见错;验器须每请求跑,非仅session start。

### 失模式

- **陈JWKS。**验器key rotation后拒有效token。修是上cron+fall-back模式。勿无refresh job缓存JWKS。
- **缺`aud` claim。**某些IdP默认omit `aud`除非token请求现`resource`。验器须拒缺`aud` token,非视缺作wildcard。
- **Scope升race。**同用户两并发步升流皆可成功并产不同scope两access token。验器须用请求上现token,非查"用户当前scope"——那造TOCTOU窗口。
- **注册token偷。**泄`registration_access_token`让攻击者重写redirect URI。Hash这些at rest;需client每次更新现cleartext;怀疑时rotate。
- **`iss`未pin。**接受任`iss`验器让攻击者立己授权server,为target audience注册client,并发token。Protected-resource metadata`authorization_servers`列表是allow-list;执它。

## 使用

`code/main.py`走完整产流用stdlib Python和小`iii_mock`注册模拟`iii.registerFunction`、`iii.registerTrigger`、`iii.trigger`、和`state::set/get`。流:

1. 授权server于`/.well-known/oauth-authorization-server`发RFC 8414 metadata。
2. MCP client调metadata端点,发现注册端点。
3. MCP client post至`/register`(RFC 7591)并收`client_id`。
4. MCP client跑PKCE保护授权码流(RFC 7636)带`resource`指示器(RFC 8707)。
5. MCP client带`Authorization: Bearer ...`调MCP server工具。
6. MCP server触发`auth::validate-jwt`,读JWKS从`state::get`。
7. Cron触发发`auth::rotate-jwks`,替state中JWKS。
8. 下调用对新key验无需重启。
9. 对异MCP资源confused-deputy尝试得401带audience mismatch。

Mock JWT此用HS256带共享secret(使课程stdlib跑)。产用RS256或EdDSA带上JWKS模式;验逻辑否则一致。

## 交付成果

本课产`outputs/skill-mcp-auth-iii.md`。给MCP server配置和IdP能集,skill emit iii原语注册、JWKS rotation schedule、scope映射、和IdP不支持完整RFC profile时拒规。

## 练习题

1. 跑`code/main.py`。跟踪9步流。注`auth::rotate-jwks` overwrite前`state::get`立回陈数据处,及下请求现对新key验。

2. 加新IdP至protected-resource metadata`authorization_servers`列表。发新IdP签名token并验验器接受。发未列IdP签名token并验验器拒带`WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 实`auth::rate-limit`作iii函数并从注册HTTP触发内调它registrar跑前。用每源IP token bucket持于`state::set("auth/ratelimit/<ip>", ...)`。

4. 读RFC 7591并识课程`/register` handler未验两域。加验。(提示:`software_statement`和`redirect_uris` URI scheme。)

5. 读MCP spec 2025-11-25 authorization节。找`WWW-Authenticate` header上一条normative requirement课程验器未现emit。加。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ASM | "OAuth metadata文档" | RFC 8414`/.well-known/oauth-authorization-server` JSON |
| DCR | "自助client注册" | RFC 7591`POST /register`流 |
| JWKS | "JWT验公钥" | JSON Web Key Set,从`jwks_uri`fetch,按`kid`索引 |
| 资源指示器 | "Audience参数" | RFC 8707`resource`参数pin token至一server |
| `aud` claim | "Audience" | JWT claim验器比规范资源URL |
| Confused deputy | "Token replay" | Server A发token现给Server B攻击 |
| `iss` allow-list | "信任授权server" | Protected-resource metadata`authorization_servers`命名集 |
| Key rotation | "Rolling JWKS" | 带overlap窗口周期替签名key |
| Public client | "Native或浏览器client" | 无`client_secret`OAuth client;PKCE补偿 |
| `WWW-Authenticate` | "401/403响应头" | 载`Bearer error=...`指令驱动client恢复 |

## 延伸阅读

- [MCP—Authorization spec(2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization)——本课实MCP auth profile
- [RFC 8414—OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414)——发现契约
- [RFC 7591—OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591)——DCR
- [RFC 7636—Proof Key for Code Exchange(PKCE)](https://datatracker.ietf.org/doc/html/rfc7636)——public-client proof-of-possession
- [RFC 8707—Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707)——audience pinning
- [RFC 9728—OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728)——资源server发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1)——合并OAuth substrate