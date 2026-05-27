# LLM路由层——LiteLLM、OpenRouter、Portkey

> 提供者lock-in贵。异工具调用workload适异模型。路由gateway给一API面、retry、failover、成本track、和guardrail。三archetype主2026:LiteLLM(开源self-hosted)、OpenRouter(managed SaaS)、Portkey(产级,2026年3月开源)。本课命名决策准则并走stdlib路由gateway。

**类型:** 学习
**语言:** Python(stdlib,routing+failover+cost tracker)
**前置要求:** 阶段13课程02(函数调用),阶段13课程17(gateways)
**时间:** ~45分钟

## 学习目标

- 分self-hosted、managed、和产级路由择。
- 实定义优先序provider失败fallback chain。
- 跟跨provider每请求成本和token用。
- 为给定产约束择LiteLLM、OpenRouter、Portkey。

## 问题背景

Provider路由重要景:

1. **成本。**Claude Sonnet成本Haiku 3倍。triage任务Haiku够;synthesis任务Sonnet值。每请求路由。

2. **Failover。**OpenAI有坏小时。每请求失败。欲无重部署自动fallback至Anthropic。

3. **延迟。**活chat UI需快time-to-first-token。批summarizer不。按延迟SLA路由。

4. **合规。**EU用户须留EU region。按region路由。

5. **实验。**同workload A/B两模型。按test bucket路由。

每集成手码全是重复。路由gateway给一OpenAI兼容API并处余。

## 概念讲解

### OpenAI兼容proxy形

每人言OpenAI形。路由gateway露`/v1/chat/completions`,接受OpenAI schema,内代理至Anthropic/Gemini/Cohere/Ollama/任。Client不care。

### 模型alias

非`claude-3-5-sonnet-20251022`,你代码言`our_smart_model`。Gateway映alias至真模型。Anthropic发Claude 4时,你改alias server侧;你代码不触物。

### Fallback chain

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway于config定义此。Retry计对budget使fallback cascade不爆成本。

### 语义缓存

相同或近相同提示hit cache而非provider。重复agent loop省可达30至60%。Key是embedding基;近相同提示享cache slot。

### Guardrail

Gateway级:

- **PII redaction。**发提示前regex或ML基pass。
- **Policy violation。**拒禁内容提示。
- **Output filter。**刷completion防漏。

Portkey和Kong皆发opinionated guardrail。LiteLLM留optional。

### 每key速率限

一API key=一team。每key budget防一team消费共享quota。大多gateway支持此。

### Self-hosted vs managed trade-off

| 因 | LiteLLM(self-hosted) | OpenRouter(managed) | Portkey(产) |
|----|----------------------|---------------------|-------------|
| Code | 开源,Python | Managed SaaS | 开源(2026年3月)+managed |
| Setup | Deploy proxy | Sign up | Either |
| Provider | 100+ | 300+ | 100+ |
| Billing | 你己key | OpenRouter credit | 你己key |
| 可观测 | OpenTelemetry | Dashboard | 全OTel+PII redaction |
| 最适合 | 欲全控team | 快原型 | 产合规 |

你有SRE team并欲数据主权时LiteLLM赢。欲单订阅无infra时OpenRouter赢。需guardrail和合规out of box时Portkey赢。

### 成本track

每请求载`provider`、`model`、`input_tokens`、`output_tokens`。乘每模型每token价(gateway持pricing sheet拉)。每用户/每team/每项目聚合。

### MCP加路由

Gateway可路由LLM调用和MCP sampling请求。Sampling请求modelPreferences偏特定模型时,gateway译至正backend。这是阶段13课程17(MCP gateway)和本课路由gateway有时merge一服务处。

### 路由策略

- **静态优先。**列表首;错fallback。
- **Load balancing。**Round-robin或weighted。
- **成本aware。**择最便宜模型满足延迟/质量。
- **延迟aware。**择过去N分钟最快模型。
- **任务aware。**提示classifier路由coding至一模型,summarization至另一。

## 使用

`code/main.py`约150行实路由gateway:接受OpenAI形请求,译至每provider stub,跑优先fallback chain,跟每请求成本,并于输入apply PII redaction pass。三景跑:正请求、primary-provider outage触发fallback、PII漏redaction捕。

看点:

- `ROUTES` dict:alias->优先序concrete provider列表。
- Fallback loop于5xx retry。
- Cost tracker乘token用每模型rate。
- PII redactor发前刷SSN形pattern。

## 交付成果

本课产`outputs/skill-routing-config-designer.md`。给workload profile(延迟、成本、合规),skill择LiteLLM/OpenRouter/Portkey并产路由config。

## 练习题

1. 跑`code/main.py`。触发outage景;验fallback落第二provider并成本正归。

2. 加语义缓存:提示SHA256是lookup key;cache hit立回。测重复调用成本省。

3. 加提示classifier路由"code..."提示至偏intelligence alias和"summarize..."提示至偏speed alias。

4. 设计每team budget:每team有月开销cap;cap hit后gateway拒请求。择执粒度(每请求或windowed)。

5. 读LiteLLM、OpenRouter、Portkey文档并排。命每发其他二无一特性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 路由gateway | "LLM proxy" | 多provider前一API面层 |
| OpenAI兼容 | "说OpenAI schema" | 接`/v1/chat/completions`形,译至任backend |
| 模型alias | "our_smart_model" | 你代码中名gateway映至concrete模型 |
| Fallback chain | "Retry列表" | 失败时试provider有序列表 |
| 语义缓存 | "提示embedding cache" | Key是提示embedding;近duplicate享cache hit |
| Guardrail | "输入/输出filter" | Redact PII、拒policy violation |
| 每key速率限 | "Team budget" | API key scope quota |
| 成本track | "每请求花" | 聚合token用x每模型价 |
| LiteLLM | "开proxy" | Self-hostable OSS路由gateway |
| OpenRouter | "托管SaaS" | 带credit基billing托管gateway |
| Portkey | "产选项" | 开源+managed带内置guardrail |

## 延伸阅读

- [LiteLLM—docs](https://docs.litellm.ai/)——self-hosted路由gateway
- [OpenRouter—quickstart](https://openrouter.ai/docs/quickstart)——managed路由SaaS
- [Portkey—docs](https://portkey.ai/docs)——带guardrail产路由
- [TrueFoundry—LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter)——决策指南
- [Relayplane—LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026)——vendor survey