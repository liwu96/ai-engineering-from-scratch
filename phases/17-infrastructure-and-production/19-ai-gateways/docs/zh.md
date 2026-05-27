# AI Gateways——LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway居app和模型provider间。核心特性provider路由、fallback、重试、速率限、secret引用、可观测、guardrail。2026市场分：**LiteLLM** MIT OSS 100+ provider、OpenAI兼容、但~2000 RPS断(8 GB内存、发布基准级联失败)；最佳Python、<500 RPS、dev/原型。**Portkey**控制平面位(guardrail、PII redaction、jailbreak检测、audit trail)、2026年3月Apache 2.0开源、20-40 ms延迟开销、$49/mo生产层。**Kong AI Gateway**建Kong Gateway——Kong自基准同12 CPUs：Portkey 228%快、LiteLLM 859%快；$100/model/month定价(Plus层最多5)；enterprise-fit若已在Kong。**Bifrost** (Maxim AI)——自动重试可配backoff、OpenAI 429 fallback Anthropic。**Cloudflare / Vercel AI Gateways**——托管、零ops、基本重试。数据居住驱动自建决策；Portkey和Kong中OSS + 可选托管。

**类型:** 学习
**语言:** Python(stdlib、玩具gateway路由模拟器)
**前置要求:** 阶段17课程01(托管LLM平台)、阶段17课程16(模型路由)
**时间:** ~60分钟

## 学习目标

- 列举六核心gateway特性(路由、fallback、重试、速率限、secret、可观测、guardrail)。
- 映四2026 gateway (LiteLLM、Portkey、Kong AI、Bifrost)到规模天花板和用例。
- 引Kong基准(Portkey 228%、LiteLLM 859%)并解释>500 RPS为何重要。
- 给数据居住和ops预算选自建vs托管。

## 问题背景

产品调OpenAI、Anthropic、自建Llama。每provider不同SDK、错模型、速率限、auth方案。要failover (OpenAI 429试Anthropic)、单credential存、统一可观测、每租户速率限。

app层重构耦合每服务每provider。Gateway层合到一进程一API (典型OpenAI兼容)扇出到provider。

## 概念讲解

### 六核心特性

1. **Provider路由**——OpenAI、Anthropic、Gemini、自建等一API后。
2. **Fallback**——429、5xx、或质量失败、重试别处。
3. **重试**——指数backoff、界限尝试。
4. **速率限**——每租户、每key、每模型。
5. **Secret引用**——runtime vault拉凭证(永不app)。
6. **可观测**——OTel + GenAI属性(阶段17课程13) + 成本归属。
7. **Guardrail**——PII redaction、jailbreak检测、允许话题过滤。

### LiteLLM——MIT OSS、Python

- 100+ provider、OpenAI兼容、router配置、fallback、基本可观测。
- Kong基准约2000 RPS断；8 GB内存footprint、持续负载级联失败。
- 最佳：Python app、<500 RPS、dev/staging gateway、实验路由。
- 成本：OSS $0；云免费层存。

### Portkey——控制平面定位

- 2026年3月Apache 2.0 OSS。Guardrail、PII redaction、jailbreak检测、audit trail。
- 每请求延迟开销20-40 ms。
- 生产层$49/mo留 + SLA。
- 最佳：监管行业需guardrail + 可观测包。

### Kong AI Gateway——规模戏

- 建Kong Gateway(成熟API gateway产品、lua+OpenResty)。
- Kong自基准12-CPU等效：Portkey 228%快、LiteLLM 859%快。
- 定价：$100/model/month、Plus层最多5。
- 最佳：已在Kong；>1000 RPS；愿许可。

### Bifrost (Maxim AI)

- 自动重试可配backoff。
- OpenAI 429 fallback Anthropic是规范食谱。
- 新进；商业。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零ops。基本重试和可观测。
- 最佳：Edge-serving JavaScript app Cloudflare/Vercel。
- Guardrail和速率限比Kong/Portkey有限。

### 自建vs托管

数据居住是驱动。医疗金融默认自建(LiteLLM或Portkey OSS或Kong)。消费产品默认托管(Cloudflare AI Gateway)或中层(Portkey托管)。混：监管租户自建、其他托管。

### 延迟预算

- LiteLLM：5-15 ms开销典型。
- Portkey：20-40 ms开销。
- Kong：3-8 ms开销。
- Cloudflare/Vercel：1-3 ms开销(edge优势)。

Gateway延迟直加TTFT。TTFT P99 < 100 ms SLA、Kong或Cloudflare。P99 < 500 ms、任。

### 速率限语义重要

简单token-bucket中规模行。多租户需sliding-window + burst allowance + 每租户tiering。LiteLLM token-bucket；Kong sliding-window；Portkey tiered。

### Gateway + 可观测 + 路由复合

阶段17课程13(可观测) + 16(模型路由) + 19(gateway)生产同层。选一工具覆三或线：多2026部署合Helicone(可观测)或Portkey(guardrail)Kong(规模)分角色。

### 你应记数

- LiteLLM：~2000 RPS断、8 GB内存。
- Portkey：20-40 ms开销；2026年3月Apache 2.0。
- Kong：Portkey 228%快、LiteLLM 859%快。
- Kong定价：$100/model/month、Plus层最多5。
- Cloudflare/Vercel：edge 1-3 ms开销。

## 使用

`code/main.py`模gateway路由3 provider fallback注入429/5xx。报延迟、重试率、fallback命中率。

## 交付成果

本lesson产`outputs/skill-gateway-picker.md`。给规模、ops姿态、合规、延迟预算、选gateway。

## 练习题

1. 跑`code/main.py`。配OpenAI→Anthropic→自建fallback。5% provider错误率期望命中率何？
2. SLA TTFT P99 < 200 ms基线300 ms。哪gateway预算内？
3. 医疗客户需自建 + PII redaction + audit。选Portkey OSS或Kong。
4. 比LiteLLM vs Kong：何RPS天花板队应迁？
5. 设计多租户SaaS速率限政策：免费层、试用层、付费层。Token-bucket或sliding-window？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Gateway | "API broker" | 居app和provider进程 |
| LiteLLM | "MIT那个" | Python OSS、100+ provider、2K RPS断 |
| Portkey | "guardrail gateway" | 控制平面 + 可观测、Apache 2.0 |
| Kong AI Gateway | "规模那个" | 建Kong Gateway、基准领 |
| Bifrost | "Maxim gateway" | 重试 + Anthropic fallback食谱 |
| Cloudflare AI Gateway | "edge托管" | Edge部署托管gateway、零ops |
| PII redaction | "数据洗" | 模型发前regex + NER mask |
| Jailbreak检测 | "提示注入guard" | 用户输入分类器 |
| Audit trail | "监管log" | 每LLM调用不可变记录 |
| Token-bucket | "简单速率限" | Refill基速率限器 |
| Sliding-window | "精速率限" | 时间窗速率限器；更好公平 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)