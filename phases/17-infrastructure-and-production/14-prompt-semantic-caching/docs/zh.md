# 提示cache和语义cache经济

> **定价快照2026-04。** 下数声反映vendor rate卡本lesson发布时；引用下游前核对链接文档。

> cache两层。L2(provider级)提示/前缀cache复重复前缀attention KV——Anthropic提示cache文档声高达90%成本减和85%延迟减长提示；Claude 3.5 Sonnet cache读$0.30/M vs $3.00/M新TTL 5分钟和1小时TTL选项2x写溢价(docs.anthropic.com, 2026-04)。OpenAI提示cache自动用提示≥1024 token并缓存输入价约90%折扣比新(platform.openai.com, 2026-04)；确每模型缓存率依赖活rate卡。L1(app级)语义cache嵌入相似命中跳LLM。Vendor "95%精度"指匹配正确性、非命中率——报告生产命中率10%(开放聊天)到70%(结构FAQ)；无provider发官方基线、视这些社区遥测非保证。生产坑：并行化杀cache(N并行请求首cache写前发可数倍涨花费)、前缀内动态内容全阻cache命中。ProjectDiscovery报告7%到74%命中率(2025-11)移动态文本出cacheable前缀。

**类型:** 学习
**语言:** Python(stdlib、玩具两层cache模拟器)
**前置要求:** 阶段17课程04(vLLM Serving)、阶段17课程06(SGLang RadixAttention)
**时间:** ~60分钟

## 学习目标

- 区L2提示/前缀cache(provider KV复)和L1语义cache(相似提示跳LLM)。
- 解释Anthropic `cache_control`显标记和两TTL选项(5分vs 1小时)及其价乘数。
- 给命中率、提示/响应混、token价算期望月省。
- 命名并行化反模式涨账5-10x和动态内容反模式塌命中率。

## 问题背景

你RAG服务加提示cache。账平。测命中率；7%。提示似静态但非——系统提示含当前日期精确分、请求ID、随机例重排多样性。每请求写新cache条、读零。

另、Agent每用户问跑十并行工具调用。十provider首cache写完成前到。十写、零读。账5-10x "有cache"应成本。

cache是协议、非flag。两层、两不同失败模式。

## 概念讲解

### L2——provider提示/前缀cache

Provider存cacheable前缀attention KV、复下请求匹配前缀。付写成本一、读近乎免费。

**Anthropic (Claude 3.5 / 3.7 / 4 series)**：请求显`cache_control`标记。你标哪块cacheable。TTL：5分钟(写成本1.25x基)或1小时(写成本2x基)。Cache读：$0.30/M Claude 3.5 Sonnet vs $3.00/M新——10x便宜(docs.anthropic.com, 2026-04)。率每模型异(Opus/Haiku分发)；总核对活定价页。

**OpenAI**：自动cache提示≥1024 token(platform.openai.com, 2026-04)。无显flag。缓存输入约10x便宜比当前gpt-4o/gpt-5 rate卡新。文档和发布笔记无官方命中率基线；社区报告30–60%精细提示设计。监`usage.cached_tokens`测自。

**Google (Gemini)**：显API上下文cache；1M-token上下文cache更值。

**自建(vLLM、SGLang)**：阶段17课程06覆RadixAttention——同模式自计算。

### L1——app级语义cache

调用LLM前、哈希提示、嵌入、找类似cache请求(余弦相似阈值上、典型0.95+)。命中、返cache响应。失、调用LLM并cache结果。

开源：Redis Vector Similarity、GPTCache、Qdrant。商业：Portkey Cache、Helicone Cache。

Vendor精度声指返cache响应语义适当频率——非命中频率。生产命中率：

- 开放聊天：10-15%。
- 结构FAQ / 支持：40-70%。
- 代码问：20-30%(小变种杀命中)。
- 语音Agent重复提示：50-80%(语音归一化固定集)。

### 并行化反模式

Agent 10工具调用并行。全10同4K-token系统提示。Anthropic cache写请求级；首cache写provider见提示后约300 ms完成。请求2-10同毫秒窗到各见cache失。付10写溢价、0读折扣。

修：批sequential-first——请求1独跑、2-10等1 cache填后发。首工具调用加300 ms；账省5-10x。

### 动态内容反模式

系统提示似：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每请求唯一。每请求写。零命中。

修：全静态移cacheable前缀；动态内容cache边界后append：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 7%到74%命中率此法解剖发。

### 栈批+cache夜负载

Batch APIs (阶段17课程15) 50%折扣24小时周转。缓存输入顶加~10x。夜分类、标记、报告生成负载可栈到同步未cache成本~10%。

### 你应记数

定价点2026-04链接vendor文档捕获、数月漂——依赖前核对。

- Anthropic缓存读：Claude 3.5 Sonnet $0.30/M、约10x便宜比新输入(docs.anthropic.com)。
- Anthropic cache写溢价：1.25x (5分TTL)或2x (1小时TTL)。
- OpenAI自动cache：提示≥1024 token用；缓存输入价约当前rate卡新输入10%(platform.openai.com)。
- 语义cache命中率(社区报告)：开放聊天~10%；结构FAQ达~70%。非vendor文档基线。
- ProjectDiscovery：7% → 74%命中率移动态出前缀(project blog, 2025-11)。
- 并行化反模式：典型报告N并行请求首cache写失账5–10x涨。

## 使用

`code/main.py`模L1 + L2 cache混负载。报命中率、账、示并行化惩罚。

## 交付成果

本lesson产`outputs/skill-cache-auditor.md`。给提示模板和流量、审计cacheability并荐重构。

## 练习题

1. 跑`code/main.py`。切并行化flag。账变多少？
2. 系统提示有日期。移出。示前后命中率数学。
3. 算1小时TTL (2x写) vs 5分钟TTL (1.25x写)盈亏平衡给请求到达率。
4. 语义cache 0.95阈值命中率20%。0.85命中率50%但错cache响应见。选正确阈值论证。
5. 用户问批10并行子查询。重写cache友好无加端到端延迟。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| L2提示cache | "前缀cache" | Provider存KV重复前缀 |
| `cache_control` | "Anthropic cache标记" | 标cacheable块显属性 |
| Cache写溢价 | "写税" | 首失写cache额外成本(1.25x或2x) |
| L1语义cache | "嵌入cache" | App级哈希嵌入LLM调用前 |
| GPTCache | "LLM cache库" | 热OSS L1 cache库 |
| Cache命中率 | "命中/总" | cache服务请求分数 |
| 并行化反模式 | "N写陷阱" | N并行请求失cache N次 |
| 动态内容陷阱 | "提示时间陷阱" | 前缀动态字节杀命中率 |
| RadixAttention | "副本内cache" | SGLang前缀cache实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官`cache_control`语义和TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动cache行为和资格。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)