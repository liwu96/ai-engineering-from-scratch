# 安全——Secret、API Key Rotation、Audit Log、Guardrail

> 消secret sprawl集中vault (HashiCorp Vault、AWS Secrets Manager、Azure Key Vault)。永不存凭证config文件、VCS env文件、spreadsheets。IAM role胜静态key；CI/CD OIDC。AI-gateway模式2026解：app → gateway → model provider、gateway runtime vault拉凭证。Vault旋转、全app分钟取——无重部署、无Slack "新key谁有"消息。Rotation政策≤90天；每commit TruffleHog / GitGuardian / Gitleaks扫描。零信任：MFA、SSO、RBAC/ABAC、短命token、设备姿态。PII scrubbing实体识别mask PHI/PII转发前；一致tokenization (Mesh approach)敏值映射稳placeholder LLM保代码/关系语义。网络egress：LLM服务专用VPC/VNet subnet whitelist仅`api.openai.com`、`api.anthropic.com`等；block所有其他出站。2026事件驱动：Vercel供应链攻击妥协CI/CD凭证exfiltrate env var跨千客户部署。

**类型:** 学习
**语言:** Python(stdlib、玩具PII scrubber + audit log writer)
**前置要求:** 阶段17课程19(AI Gateway)、阶段17课程13(可观测)
**时间:** ~60分钟

## 学习目标

- 列举四secret管理反模式(VCS config文件、硬编码env、spreadsheets、静态key)并命名替代。
- 解释AI-gateway-pulls-from-vault模式2026生产标准。
- 实现PII scrubber一致tokenization (同值→同placeholder)语义存。
- 命名2026 Vercel供应链事件和CI/CD凭证卫生教训。

## 问题背景

实习生提交含API key `.env`。速删。key已git历史——GitGuardian扫描捕、rotation进程"Slack队、更新40 config文件、重部署全服务"。8小时后、半服务活半等部署窗。

另、用户提示含"My SSN is 123-45-6789"。提示OpenAI。有BAA但内政策PII mask转发前。未。

另、EKS集群LLM pod可达任Internet host。某人DNS lookup攻击控域exfil数据。无阻。

LLM服务安全需址三向量。Vault-backed凭证。PII scrubbing。网络egress过滤。Audit log。

## 概念讲解

### 集中vault + IAM-role拉

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。一真相源。

**IAM role**：app/gateway IAM identity认证、非静态key。Vault返token生命周期secret。

**AI-gateway模式**：gateway请求时vault拉`OPENAI_API_KEY`。Vault旋转；下请求新key。无重部署。

### Rotation政策≤ 90天

全API key、vault root token、CI/CD凭证。可能自动旋转。手动旋转log和track。

### Secret扫描

- **TruffleHog**——commit regex + entropy。
- **GitGuardian**——商业、高精度。
- **Gitleaks**——OSS、CI跑。

每commit跑。PR新secret检测阻。

### 零信任姿态

- 全账户MFA必需。
- SSO SAML/OIDC。
- RBAC (role基)或ABAC (attribute基)细粒度访问。
- 短命token (小时、非日)。
- 设备姿态——仅corp设备盘加密。

### PII / PHI scrubbing

提示离infra前：

1. 实体识别(spaCy NER、Presidio、商业)。
2. Mask匹配实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致tokenization (Mesh approach)：同值同placeholder LLM保关系。
4. 可选LLM响应反向映射。

静态regex过滤捕基本模式；NER捕更多。双用。

### 输入 + 输出guardrail

输入：阻已知jailbreak、禁话题；每用户速率限。

输出：regex scrub漏secret (API key模式、refusal context email模式)、policy违分类器。

### 网络egress whitelist

LLM服务专用subnet：
- Whitelist：`api.openai.com`、`api.anthropic.com`、向量DB端点、vault端点。
- 其他：drop。
- DNS allowlist-only resolver (避DNS-tunneling exfil)。

### Audit log

每LLM调用不可变log：
- Timestamp。
- 用户/租户。
- 提示hash (隐私非raw提示)。
- 模型+版本。
- Token数。
- 成本。
- 响应hash。
- 任guardrail触。

监管要求留(SOC 2 1年、HIPAA 6年)。

### 2026 Vercel事件

供应链攻击：妥协CI/CD凭证exfiltrate env var跨千客户部署。教训：CI/CD凭证prod等效。Vault存。窄scope。激进旋转。

### 你应记数

- Rotation政策：≤ 90天。
- 每commit扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD凭证妥协 → 千客户env var漏。
- Audit log留：SOC 2 = 1年、HIPAA = 6年。

## 使用

`code/main.py`实现玩具PII scrubber一致tokenization和append-only audit log。

## 交付成果

本lesson产`outputs/skill-llm-security-plan.md`。给监管范围和当前状态、计划vault迁移、scrubber、egress、audit log。

## 练习题

1. 跑`code/main.py`。发两提示引用同SSN。确认双得同placeholder。
2. 设计vLLM-on-EKS部署调OpenAI + Anthropic + Weaviate网络egress政策。
3. git历史发现key (2年旧)。正确响应何——旋转key、scrub历史、或双？论证。
4. Audit log涨10 GB/天。设计留层(hot 30d、warm 12mo、cold 6yr)。
5. 论反向tokenization (LLM响应代真值)复杂性是否比placeholder可见。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Vault | "secret存" | 集中凭证管理服务 |
| IAM role | "identity基认证" | Role app认；返短命凭证 |
| OIDC CI/CD | "云发token" | CI无静态key——OIDC identity |
| TruffleHog / GitGuardian / Gitleaks | "secret扫描器" | Commit时secret检测 |
| RBAC / ABAC | "访问控制" | Role基vs attribute基 |
| PII scrubbing | "数据mask" | 移或tokenize敏实体 |
| 一致tokenization | "稳placeholder" | 同值→同token每时间 |
| Mesh approach | "Mesh tokenization" | 语义保tokenization模式 |
| Egress whitelist | "出站allowlist" | 仅允域可达 |
| Audit log | "不可变历史" | 合规append-only记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII检测和匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)