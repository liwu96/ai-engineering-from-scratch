# Batch APIs——50%折扣行业标准

> 每主provider发异步batch API 50%折扣和~24小时周转。OpenAI、Anthropic、Google、和大多数推理平台(Fireworks batch tier、Together batch)实现同模式。栈batch提示cache夜管道到同步未cache成本~10%。规则残暴简单：若非交互、属batch。内容生成管道、文档分类、数据提取、报告生成、批量标记、目录标记——容忍24小时延迟者是桌上钱直到移batch。2026生产模式triage每新LLM负载三道：交互(同步cache)、半交互(异步队列fallback)、batch(夜、缓存输入栈)。假装交互但容忍分钟延迟负载浪费最多。

**类型:** 学习
**语言:** Python(stdlib、玩具batch-vs-sync成本模拟器)
**前置要求:** 阶段17课程14(提示和语义cache)
**时间:** ~45分钟

## 学习目标

- 命名三provider batch APIs (OpenAI、Anthropic、Google)和共50%折扣 + 24h周转保证。
- 算batch + 缓存输入栈夜分类负载成本、比同步未cache基线。
- Triage负载交互/半交互/batch并论证道。
- 命名两陷阱：部分交互性(用户期望快于24h)和输出schema漂(batch文件格式provider异)。

## 问题背景

队发夜报告生成管道。50,000文档、每总结、聚类总结、草执行简。同步跑4小时$2,000/夜。听batch APIs。

Batch得50%折扣。也系统提示提示cache(50k调用共享)。栈、账降$180/夜——基线~9%。同管道、三配置改。

Batch是LLM成本工具箱最便宜杠杆无人拉。原多组织：队想"实时"SLA实际"早晨到"。本lesson是桌上90%账。

## 概念讲解

### 三batch APIs

**OpenAI Batch API**：JSONL文件上传请求列表。保证24小时周转(实践~2-8小时)。50%折扣输入输出token。`/v1/batches`端点。Cache-eligible输入也缓存输入定价顶。

**Anthropic Message Batches**：JSONL上传。24小时周转。50%折扣。支持`cache_control`——cache写显、batch内读自动。

**Google Vertex AI Batch Prediction**：BigQuery或GCS输入。Gemini类似50%折扣。Vertex管道集成。

### 语义：异步、非慢

Batch是"我保证24小时内返"——非"这要24小时"。典型P50 2-6小时。Provider调度batch峰外窗GPU库存低用时。

### 栈cache

50k文档总结同4K-token系统提示：

- 同步未cache：50000 × ($input × 4000 + $output × 200)全率。
- 同步cache：系统提示首写后cache；余49999得10x便宜输入。
- Batch cache：上述加读和写50%折扣。

栈：batch + cache = ~10%同步未cache账。任何夜跑共享系统提示负载应此用。

### 负载triage

**交互**——用户等响应。TTFT重要。同步调用提示cache。不能batch。

**半交互**——用户提交任务、分钟回查。异步队列batch fallback同步。中量RAG索引想。

**Batch**——用户期望结果"早晨到"或"下小时"。内容管道、规模分类、离线分析。总batch、总栈cache。

常见错：分类一切交互因管道生产。生产非延迟规——SLA是。

### 部分交互性陷阱

些特性似交互但容忍5-10分钟。例：夜客户健康报告"刷新"按钮。用户点刷新；等10分钟行。队同步发。50并发刷新成本10x batch并发邮件递成本。

问："24小时对此用户何义？"若答"他们不注意"，batch。

### 输出schema陷阱

Batch文件格式provider异：

- OpenAI：JSONL、每行一请求。
- Anthropic：JSONL、每行一消息；响应格式嵌入。
- Vertex：BigQuery表或GCS前缀TFRecord。

写"跨provider batch客户端"意每provider适配码。Gateway声多provider batch (Portkey、LiteLLM些层)仍薄包原始格式。

### 你应记数

- Provider batch折扣：50%平输入+输出。
- 周转SLA：24小时保证、2-6小时典型P50。
- 栈batch + 缓存输入：~10%同步未cache成本。
- 负载triage规则：若24h延迟接受、总batch。

## 使用

`code/main.py`算同步、同步+cache、batch、batch+cache成本50k文档负载。报$和百分比省。

## 交付成果

本lesson产`outputs/skill-batch-triager.md`。给负载特性、triage交互/半/batch并估省。

## 练习题

1. 跑`code/main.py`。100k文档管道3K-token系统提示500-token输出、算栈(batch + cache) vs 同步基线省。
2. 选真实产品三特性。每triage交互/半/batch。
3. 用户抱怨报告3小时。batch误triage或合理交互？写决策准则。
4. Batch API返SLA 24h但P99 20小时。如何传达用户——边界下系统行为何？
5. 算盈亏平衡：哪共享前缀长batch + cache比自留GPU夜跑便宜？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Batch API | "异步折扣" | 50%折扣24h周转 |
| JSONL | "batch格式" | 每行一JSON请求；OpenAI/Anthropic标准 |
| Message Batches | "Anthropic batch" | Anthropic batch API产品名 |
| Batch prediction | "Vertex batch" | Vertex AI batch API产品 |
| 周转SLA | "24h承诺" | 保证、非典型；典型2-6h |
| 负载triage | "交互性决策" | 交互/半/batch路由决策 |
| 输出schema | "响应格式" | 每provider JSONL布局；非可移 |
| 栈折扣 | "batch + cache" | 同用时~10%未cache同步账 |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL格式和`/v1/batches`语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch格式和`cache_control`交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)