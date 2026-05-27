# LLM FinOps——单元经济和多租户归属

> 传统FinOps LLM支出破。成本token-transaction、非resource-uptime。Tag不映——API调用transaction、非asset。工程决策(提示设计、上下文窗、输出长度)财务决策。2026 playbook三归属维度首日仪器：每用户(`user_id`) seat pricing和扩张、每任务(`task_id` + `route`)产品表面成本和优先级、每租户(`tenant_id`)单元经济和续约。四token层——prompt、tool、memory、response——桶藏支。多租户产品强制阶梯：每租户速率限(2-3x期望峰、清429 + retry-after)；日支cap (1.5-3x合同上限；触速率紧+警)；支z-score > 4 kill switch (auto-pause + page on-call)。归属模式：tag-and-aggregate、telemetry-joiner (trace-ID → billing；最高精度)、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。单元指标：cost per resolved query、cost per generated artifact——非$/M tokens。追溯tag总漏；请求创时仪器。

**类型:** 学习
**语言:** Python(stdlib、玩具成本归属模拟器带kill switch)
**前置要求:** 阶段17课程13(可观测)、阶段17课程14(cache)
**时间:** ~60分钟

## 学习目标

- 解释为何传统FinOps (tag + tier) LLM支出破并命名三新归属维度。
- 列举四token层(prompt、tool、memory、response)并为何单桶billing藏成本。
- 设计多租户产品强制阶梯(速率→支cap→kill switch)。
- 选单元指标(cost per resolved query / artifact)而非$/M tokens。

## 问题背景

账$40,000。不知：
- 何租户花。
- 何产品特性驱。
- 何用户滥。
- 提示膨胀、工具调用、内存放大何根。

Provider侧tag-and-aggregate云资源(EC2、S3)标签传播行item。LLM API调用不自tag——调用点stamp user/task/tenant并传。追溯归属总漏边界case。

## 概念讲解

### 三归属维度

**每用户**(`user_id`)：谁何成本。驱seat pricing、扩张对话、识power user。

**每任务**(`task_id` + `route`)：何产品表面何成本。驱特性优先级、杀贵特性决策。

**每租户**(`tenant_id`)：何客户盈利。驱单元经济、续约定价、层阈值。

首日调用点全三仪器。追溯总差。

### 四token层

| 层 | 例 | 典型总% |
|-------|---------|---------------------|
| Prompt | 系统+用户输入 | 40-60% |
| Tool | 工具调用结果回喂 | 20-40% (agent负载) |
| Memory | 前对话/检索文档 | 10-30% |
| Response | 模型输出 | 10-30% |

四桶一起优化盲。归属schema分。

### 强制阶梯

1. **速率限**每租户。2-3x期望峰。返429带`Retry-After`。租户摩擦见；无惊账。

2. **日支cap**每租户。1.5-3x合同上限。触：紧速率限 + 警customer-success。

3. **Kill switch**支z-score > 4租户基线相对。Auto-pause租户；page on-call；升ops + CS。

### 归属模式

- **Tag-and-aggregate**：stamp metadata header；后聚合。简；粗。
- **Telemetry joiner**：trace-ID join trace billing。最高精度。成熟队做。
- **Sampling + extrapolation**：5-10%采样、乘。粗支成本效；漏尾。
- **Model-based allocation**：回归推成本驱。无tag遗留数据。
- **Event-sourced**：成本事件流(Kafka / Kinesis)。实时。
- **Real-time streaming**：dashboard亚秒更新。

### Cost per X单元指标

$/M tokens vendor speak。产品指标：

- Cost per resolved support ticket。
- Cost per generated article。
- Cost per successful agent task。
- Cost per user-session-minute。

成本绑产品结果。否则优化无锚。

### 成本归属trace形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每调用发。数据湖存。维度聚合。阶段17课程13可观测栈活。

### 复合省栈

栈：cache + batch + route + gateway。四全：
- Cache L2 (阶段17课程14)：~10x便宜输入。
- Batch (阶段17课程15)：50%折扣。
- Route便宜模型(阶段17课程16)：60%成本减。
- Gateway效率(阶段17课程19)：冗余+重试。

最佳栈：~5-10%朴素基线。多队2-3杠杆启；少栈四。

### 你应记数

- 归属维度：每用户、每任务、每租户。
- 四token层：prompt、tool、memory、response。
- Kill switch：支z-score > 4。
- 单元指标：cost per resolved query、非$/M tokens。
- 栈优化：~5-10%基线可能。

## 使用

`code/main.py`模多租户LLM服务三层强制阶梯。注入滥租户示kill switch火。

## 交付成果

本lesson产`outputs/skill-finops-plan.md`。给产品和规模、设计归属schema和强制阶梯。

## 练习题

1. 跑`code/main.py`。何z-score kill switch火？何选阈值？
2. 设计每租户、每任务成本dashboard。首5视图何？
3. 最大租户单元经济负。按客户影响序提三干预。
4. 算支持产品cost per resolved ticket：3M tokens/ticket、~800 tickets/day、GPT-5缓存率。
5. 论追溯tag可能工作否。何时接受？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Per-user归属 | "用户级成本" | 每调用`user_id` stamp |
| Per-task归属 | "特性成本" | `task_id` + `route`识产品表面 |
| Per-tenant归属 | "客户成本" | `tenant_id`；驱单元经济 |
| 四token层 | "成本层" | prompt + tool + memory + response |
| 速率限 | "429 guard" | 每租户上限gateway强制 |
| 日支cap | "日上限" | 租户scope预算带警 |
| Kill switch | "auto-pause" | 支z-score > 4触自动停 |
| Cost per resolved | "产品单元指标" | 成本绑产品结果、非token |
| Telemetry joiner | "trace-to-billing" | 最高精度归属模式 |
| 栈优化 | "cache+batch+route+gateway" | 复合省到~5-10%基线 |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)