# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大云服务商，三种不同策略。AWS Bedrock 是模型市场 —— Claude、Llama、Titan、Stability、Cohere 统一在一个 API 后面。Azure OpenAI 是 OpenAI 独家合作伙伴关系加上预配置吞吐量单元 (PTU) 用于专用容量。Vertex AI 是 Gemini 优先，拥有最佳长上下文和多模态故事。2026 年 Artificial Analysis 测量 Azure OpenAI 在 Llama 3.1 405B 等效模型上中位数约为 50 毫秒，Bedrock 约为 75 毫秒 —— PTU 解释了差距，因为专用容量胜过共享按需。决策规则不是"哪个最快"而是"哪个模型目录和 FinOps 界面匹配我的产品"。本课程教你带着写下来的权衡做选择，而不是凭感觉。

**类型:** 学习
**语言:** Python (标准库，玩具成本与延迟比较器)
**前置要求:** 第 11 阶段 (LLM 工程)，第 13 阶段 (工具与协议)
**时间:** ~60 分钟

## 学习目标

- 说出三种平台策略 (市场 vs 独家 vs Gemini 优先) 并将每种匹配到产品用例。
- 解释 Azure OpenAI 中预配置吞吐量单元 (PTU) 能给你带来什么，以及为什么在 405B 规模上按需 Bedrock 通常慢约 25 毫秒。
- 为每个平台绘制 FinOps 归因界面 (Bedrock 应用推理配置文件 vs Vertex 每团队项目 vs Azure 范围 + PTU 预留)。
- 写下"至少双供应商"政策并解释为什么单供应商锁定是 2026 年的昂贵错误。

## 问题背景

你为产品选择了 Claude 3.7 Sonnet。现在你需要提供它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费和跨供应商的速率限制。

更深层的问题是目录。如果你需要在同一产品中使用 Claude、Llama 和 Gemini，你无法从同一个地方全部购买，除非那个地方同时是 Bedrock 加 Vertex 加 Azure OpenAI。超大规模云服务商不是可互换的 —— 它们各自对谁拥有模型层做出了不同的押注。

本课程绘制这三种押注、延迟差距、FinOps 差距和锁定风险。

## 概念讲解

### 三种策略

**AWS Bedrock** —— 市场。Claude (Anthropic)、Llama (Meta)、Titan (AWS 第一方)、Stability (图像)、Cohere (嵌入)、Mistral，加上图像和嵌入子目录。一个 API、一个 IAM 界面、一个 CloudWatch 导出。Bedrock 的押注是客户想要可选择性胜过单一模型。

**Azure OpenAI** —— 独家合作伙伴关系。你可以在 Azure 数据中心获得 GPT-4/4o/5/o 系列、DALL·E、Whisper 和 OpenAI 模型的微调。"Azure OpenAI 服务"目录中没有非 OpenAI 模型 —— 那些去 Azure AI Foundry (单独产品)。Azure 的押注是 OpenAI 保持前沿地位，客户想要对该特定关系的企业控制。

**Vertex AI** —— Gemini 优先，其他次之。Gemini 1.5/2.0/2.5 Flash 和 Pro，加上模型花园 (第三方)。Vertex 的押注是多模态长上下文 —— 100 万 token Gemini 上下文是差异化优势。

### 规模上的延迟差距

Artificial Analysis 运行持续基准测试。在等效 Llama 3.1 405B 部署上 (共享按需)，Azure OpenAI 中位数首 token 延迟约为 50 毫秒；Bedrock 约为 75 毫秒。差距不是 AWS 的失败 —— 是容量模型差异。Azure 销售 PTU (预配置吞吐量单元)，为你的租户预留 GPU 容量。Bedrock 的等效产品 (预配置吞吐量) 存在但每个单元起价约 21 美元/小时，大多数客户停留在共享按需上。

按需共享容量与所有其他客户的流量竞争。专用容量不竞争。如果你的产品 SLA 是 P99 下 TTFT < 100 毫秒，你要么在 Azure 上购买 PTU，要么购买 Bedrock 预配置吞吐量，要么接受默认方差。

### 预配置吞吐量经济学

Azure PTU：预留的推理计算块。相比按需最高节省约 70%，适用于可预测工作负载。无论流量如何每小时固定成本 —— 空闲时也要为预留付费。盈亏平衡点通常在持续利用率的 40-60% 左右。

Bedrock 预配置吞吐量：根据模型和地区每小时 21-50 美元。数学类似 —— 盈亏平衡点在峰值利用率的一半左右。需要月度承诺。

Vertex 预配置容量按 Gemini SKU 销售；价格因模型和地区而异，公开宣传较少。

### FinOps 界面 —— 真正的差异化因素

**Bedrock 应用推理配置文件** 是市场中最干净的归因。用 `team`、`product`、`feature` 标记配置文件；通过它路由所有模型调用；CloudWatch 无需后处理即可按配置文件分解成本。2025 年添加，仍然是最细粒度的超大规模云原生。

**Vertex** 归因是每团队项目加随处标签。你将每个团队建模为 GCP 项目，在每个资源上放置标签，使用 BigQuery 计费导出 + DataStudio 进行汇总。更多工作，但 BigQuery 让你可以用任意 SQL 分析成本数据。

**Azure** 依赖订阅/资源组范围加标签，PTU 预留作为一等成本对象。标签从资源组继承，不从请求继承，因此每请求归因需要 Application Insights 自定义指标或一个盖章头部的网关。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 除非你插桩否则最不透明。

### 锁定是 2026 年的风险

当单一模型主导时，单超大规模云承诺是可以的。在 2026 年，前沿每季度移动 —— Claude 3.7 一个季度，Gemini 2.5 下一个，GPT-5 再下一个。锁定到一个平台将你锁定在三分之二的前沿之外。

有效团队采用的模式：任何产品关键 LLM 调用至少双供应商。Bedrock 加 Azure OpenAI 是常见的组合 —— Claude 来自一个，GPT 来自另一个，它们之间故障转移，同一个网关。成本提升可以忽略，因为网关路由最优；可用性提升在停机期间 (如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 停机) 是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数地区的 BAA；VPC 端点；护栏。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；受监管企业默认选择。
Vertex：HIPAA、GDPR、每个地区的数据驻留；Google Cloud 的合规堆栈。

三者都满足基本复选框。差异在于数据保留政策、日志处理方式，以及滥用监控是否读取你的流量 (大多数默认选择加入；企业可选择退出)。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上中位数 TTFT (带 PTU)：~50 毫秒
- Bedrock 按需中位数 TTFT：~75 毫秒
- Bedrock 预配置吞吐量：每单元每小时 21-50 美元
- Azure PTU 盈亏平衡：~40-60% 持续利用率
- 高利用率下 PTU 相比按需节省：最高 70%

## 使用它

`code/main.py` 在合成工作负载上比较三个平台 —— 它建模按需与 PTU 经济学、TTFT 方差和成本归因保真度。运行它以查看 PTU 在何处划算，以及市场的模型广度何时胜过 TTFT 差距。

## 产出成果

本课程产生 `outputs/skill-managed-platform-picker.md`。给定工作负载配置文件 (需要的模型、TTFT SLA、日量、合规要求)，它推荐主平台、备用方案和 FinOps 插桩计划。

## 练习题

1. 运行 `code/main.py`。对于 70B 级模型，在什么持续利用率下 Azure PTU 胜过按需？计算盈亏平衡并与宣传的 40-60% 范围比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计双供应商部署 —— 哪个去哪个超大规模云，什么网关在前面，故障转移政策是什么？
3. 受监管的医疗客户需要 BAA、美国东部数据驻留和 P99 下亚 100 毫秒 TTFT。选择平台并用三个具体功能证明。
4. 你发现本月 Bedrock 账单增长了 4 倍而流量没有变化。没有应用推理配置文件，你如何找到罪魁祸首？有配置文件需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 —— 直接 Anthropic API、Bedrock 按需还是 Bedrock 预配置吞吐量？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|----------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中带有企业控制的 OpenAI 独家模型 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，模型花园用于第三方模型 |
| PTU | "专用容量" | 预配置吞吐量单元 —— 预留推理 GPU，按小时定价 |
| 应用推理配置文件 | "Bedrock 标签" | 带标签的每产品成本/使用配置文件，CloudWatch 原生 |
| 模型花园 | "Vertex 目录" | Vertex AI 的第三方模型部分，与 Gemini 分开 |
| 至少双供应商 | "LLM 冗余" | 在每个关键 LLM 路径上跨 ≥2 超大规模云运行的政策 |
| BAA | "HIPAA 文书" | 商业伙伴协议；PHI 需要；三者都提供 |
| 滥用监控 | "日志监视器" | 供应商端对提示/输出的安全扫描；企业可选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) —— 权威费率表和预配置吞吐量定价。
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) —— PTU 经济学和费率表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) —— Gemini 层级和模型花园附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) —— 跨供应商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) —— 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) —— 归因机制并排比较。
