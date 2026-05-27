# Shadow Traffic、Canary Rollout、和LLM渐进部署

> LLM rollout合软件部署最难部分：无单元测试、弥散失败模式、延迟信号。序列(1) shadow模式——复制prod请求候选模型、log、比零用户影响；捕显分布问题非质量保证；(2) canary rollout——渐进流量移10% → 25% → 50% → 75% → 100%每步gate；追踪延迟百分位、cost/request、error/refusal率、输出长度分布、用户反馈率；(3) A/B testing稳定性确认后异替代。非确定性不可减——同输入GPU FP非associativity加batch-size方差高达15%精度变异。成本是变量非常——20%好模型可3x贵每调用。Rollback速决定：若rollback需重部署、太慢。政策活config/flag；模型活registry pinned digest；rollback = flip政策 + revert阈值 + pin旧模型秒。

**类型:** 学习
**语言:** Python(stdlib、玩具canary进展模拟器)
**前置要求:** 阶段17课程13(可观测)、阶段17课程21(A/B Testing)
**时间:** ~60分钟

## 学习目标

- 区shadow模式(零影响比)、canary(活流量渐进)、A/B(稳定性确认比)。
- 列举五LLM特定canary指标(延迟、cost/request、error/refusal、输出长度分布、用户反馈)。
- 解释LLM非确定性(高达15%)改rollout"稳定"何义。
- 设计rollback路径秒(政策flip)非小时(重部署)。

## 问题背景

发新模型。离线eval示3%精度增益。生产翻开。24小时内、成本涨40%、用户拇指下涨8%、三客户ticket报"怪答案"。你rollback。重部署3小时。周末毁。

每片可避。Shadow模式会捕40%成本突任何用户见前。Canary会拇指下移10%停。Policy-flag rollback会30秒。纪律填"离线eval好看"和"真用户乐"间gap。

## 概念讲解

### Shadow模式

候选接收同请求生产；输出log、非返用户。零用户影响。Log：

- 输出内容(diff生产)。
- Token数(成本delta)。
- 延迟。
- Refusal和error。

捕：成本突、长度回退、显refusal变、硬error。非捕：用户感知质量delta。Shadow烟测、非质量测。

### Canary rollout

渐进流量移gate。典型进展：1% → 10% → 25% → 50% → 75% → 100%。每步5指标gate：

1. **延迟百分位**——P50、P95、P99。破：canary P99 > 1.5x基线。
2. **Cost per request**——混合$。破：>20%基线上。
3. **Error / refusal率**——5xx加显refusal。破：2x基线。
4. **输出长度分布**——均 + P99。破：分布移。
5. **用户反馈率**——拇指下 / ticket filing。破：1.5x基线。

### 非确定性新方差

同输入产非同输出。原因：

- GPU FP非associativity(浮点归约顺序batch异)。
- Batch-size方差(同提示batch 128 vs batch 16)。
- Sampling (temperature > 0)。

测：同eval集run-to-run高达15%精度变异。Rollout"稳定"意指标期望方差内、非同基线。设gate噪声底上。

### 成本是变量

20%好模型可3x贵每调用。Cost/request五gate之一。"好"模型破单元经济是rollback case。

### Rollback武器

- Policy flag (feature flag系统)：config翻百分比；秒。
- Model pinning (registry digest)：pinned模型不自升级。
- Rollback = revert flag + set pinned digest前。秒非小时。

若栈需重部署rollback、rolling前修。

### 工具

**Argo Rollouts** / **Flagger**——Kubernetes渐进交付控制器。Istio/Linkerd权重路由集成。

**Istio权重路由**——service-mesh级流量分。

**KServe / Seldon Core**——模型serving内置canary。

**Feature flags**——LaunchDarkly、Flagsmith、Unleash。政策级flip、无重部署。

### 指标节奏

Canary gate每5-15分查依赖流量量。1%流量10 req/min给每窗50-150数据点——延迟够但用户反馈噪。10%给~10x多。进展应每步停够久积够样本。

### A/B步可选

若新模型显异(异行为、异成本曲线、异调)、canary过后50% A/B测。若仅改进版、canary gate过跳100%。

### 你应记数

- Canary进展：1% → 10% → 25% → 50% → 75% → 100%。
- 非确定性天花板：同输入高达15% run-to-run方差。
- 五canary指标：延迟、成本、error/refusal、输出长度、用户反馈。
- 成本gate：>20%基线上破。
- Rollback：秒非小时。

## 使用

`code/main.py`模canary rollout注入回退。报rollout停何阶段何gate触发。

## 交付成果

本lesson产`outputs/skill-rollout-runbook.md`。给候选模型、基线、风险容忍、设计shadow→canary→100%计划。

## 练习题

1. 跑`code/main.py`。注入25%成本回退。Canary停何阶段？
2. 新模型离线3%精度增益但cost/request +18%。发否？依赖政策——写双边。
3. 设计60秒端到端rollback。列需基础设施。
4. 非确定性eval示±7%。设canary gate不假警。用何乘数？
5. Shadow模式canary前捕40%成本突。写shadow触发警规则。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Shadow模式 | "复制到新" | 零影响发候选log |
| Canary | "渐进流量" | gate渐进用户暴露rollout |
| Gate | "rollout检查" | 阻进展指标阈值 |
| 非确定性 | "LLM方差" | 不可减run-to-run异 |
| Policy flag | "flag翻rollback" | Config级rollback、秒非小时 |
| Model pin | "registry digest" | 模型版不可变引用 |
| Argo Rollouts | "K8s渐进" | Kubernetes原生canary/rollback控制器 |
| KServe | "推理K8s" | 模型serving canary原语 |
| Istio权重 | "mesh分" | Service-mesh流量分器 |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)