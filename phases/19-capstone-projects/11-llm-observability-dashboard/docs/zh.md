# 毕业项目 11 —— LLM可观测性 & 评估仪表板

> Langfuse开源了。Arize Phoenix发布了2026 GenAI semconv映射。Helicone和Braintrust皆押注每用户成本归因。Traceloop的OpenLLMetry成事实SDK instrumentation。产形态是ClickHouse存traces、Postgres存metadata、Next.js存UI、和小规模eval jobs (DeepEval、RAGAS、LLM-judge)运行于采样traces。自托管建一、从至少四SDK家族摄入、并演示五分钟内捕获注入回归。

**类型:** 毕业项目
**语言:** TypeScript (UI)、Python / TypeScript (摄入 + evals)、SQL (ClickHouse)
**前置要求:** 第11阶段(LLM工程)、第13阶段(工具)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P11 · P13 · P17 · P18
**时间:** 25小时

## 问题背景

2026运行产traffic的每AI团队保有可观测性平面与模型并行。成本归因。幻觉检测。漂移监控。Jailbreak信号。SLO仪表板。PII泄漏alert。开源参考 — Langfuse、Phoenix、OpenLLMetry — 收敛于OpenTelemetry GenAI语义约定作摄入schema。现可用一SDK instrument OpenAI、Anthropic、Google、LangChain、LlamaIndex、和vLLM并ship兼容spans。

将建自托管仪表板从至少四SDK家族摄入、于采样traces运行小规模eval jobs、检测漂移、并alert。测量bar: 给故意注入回归(开始产PII的prompt)、仪表板捕获并五分钟内fired alert。

## 概念讲解

摄入是OTLP HTTP。SDK产GenAI-semconv spans: `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans落ClickHouse作列式分析; metadata (users、sessions、apps)落Postgres。

Evals作batch jobs运行于采样traces。DeepEval评分faithfulness、toxicity、和answer relevance。RAGAS评分检索指标当trace携带检索context。自定义LLM-judges运行领域特定检查(PII泄漏、离policy response)。Eval runs写回同ClickHouse作eval spans链接父trace。

漂移检测watch embedding-space分布随时间(PSI或KL divergence于prompt embeddings)加eval-score趋势。Alerts发Prometheus Alertmanager然后Slack / PagerDuty。UI是Next.js 15带Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄入: OpenTelemetry SDKs + GenAI语义约定; OTLP HTTP传输
- Collector: OpenTelemetry Collector带tail-sampling processor (成本控制)
- 存储: ClickHouse spans、Postgres metadata、S3 raw event archive
- Evals: DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、自定义LLM-judge
- 漂移: PSI / KL于池化prompt embeddings (sentence-transformers)周
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDKs支持: OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 动手实践

1. **Collector配置。** OpenTelemetry Collector带OTLP HTTP receiver、tail-sampler保持100% error traces和10% successes、exporters到ClickHouse和S3。

2. **ClickHouse schema。** 表`spans`列镜像GenAI semconv: `gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`、加JSON bag存长payload。按user_id和app_id加二级索引。

3. **SDK覆盖测试。** 写小client app用每SDK (OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM)配OpenLLMetry auto-instrument。验证各产canonical GenAI spans落ClickHouse。

4. **Eval jobs。** 定时job读最近15分钟采样traces并运行DeepEval faithfulness、toxicity、和answer relevance。输出是eval spans链接父trace。

5. **自定义LLM-judge。** PII泄漏judge: 给response、调guard LLM评分PII泄漏可能性。高评分response落triage queue。

6. **漂移检测。** 周job算本周池化prompt embeddings与4周trailing baseline PSI。若PSI超阈值、alert。

7. **仪表板。** Next.js 15带pages: overview (spans/sec、cost/user、p95 latency)、traces (搜索 + waterfall)、evals (faithfulness趋势、toxicity)、漂移(PSI随时间)、alerts。

8. **Alerting链。** Prometheus exporter读eval score聚合和latency percentiles; Alertmanager路由Slack警告和PagerDuty关键breach。

9. **回归probe。** 注入bug: evaluated chatbot开始1%时间泄漏假SSN。测MTTR: 从bug部署到Slack alert。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 产出成果

`outputs/skill-llm-observability.md`是deliverable。给LLM应用、仪表板摄入traces、运行evals、alert漂移、并于Next.js surfacing cost/user分解。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | Trace-schema覆盖 | 产canonical GenAI spans SDK家族数(目标: 6+) |
| 20 | Eval正确性 | DeepEval / RAGAS评分vs手标注集 |
| 20 | 仪表板UX | 注入回归MTTR(低于5分钟目标) |
| 20 | 成本 / 规模 | 1k spans/sec持续摄入无backlog |
| 15 | Alerting + 漂移检测 | Prometheus/Alertmanager链端到端exercised |
| **100** | | |

## 练习题

1. 为Haystack框架加自定义instrumentation。验证canonical spans落ClickHouse带忠实`gen_ai.*`属性。

2. 于同traces换DeepEval为Phoenix evaluators。测两eval引擎间score drift。

3. 细化漂移检测器: 按app-id而非全局算PSI。显示per-app漂移trail。

4. 加"user impact"页: cost-per-user和failure-rate-per-user带sparklines。

5. 建tail-sampling policy保持100% toxicity > 0.5 traces加10%其余分层样本。测采样bias引入。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GenAI semconv | "OTel LLM属性" | 2025 OpenTelemetry LLM span属性spec (system、model、tokens) |
| Tail sampling | "后trace采样" | Collector于trace完成后决定保持或丢弃(可peek errors) |
| PSI | "人口稳定性指数" | 漂移指标比两分布; > 0.2通常信号有意义漂移 |
| LLM-judge | "Eval作模型" | LLM评分另LLM输出于rubric (faithfulness、toxicity、PII) |
| Tail-sampling policy | "Keep-rule" | 决定何trace持久vs丢弃规则; errored + sample-rate |
| Eval span | "链接eval trace" | 携带eval score链接原LLM调用span的子span |
| Cost per user | "单位经济" | 窗口内user_id归因美元成本; 关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开源可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 备选参考带强漂移支持
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK家族
- [OpenTelemetry GenAI语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入schema
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选eval-first平台
- [ClickHouse文档](https://clickhouse.com/docs) — 列式span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator库