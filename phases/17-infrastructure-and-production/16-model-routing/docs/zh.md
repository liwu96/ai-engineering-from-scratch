# 模型路由作为成本减原语

> 动态broker评每请求(任务类型、token长度、嵌入相似、信心)并发简单查询便宜模型、复杂升前沿模型。也称模型级联。生产案例示US/UK/EU部署iso-quality 20-60%成本减；高量SaaS 30%路由效率改转六位年省。2026背景LLM推理价年降~10x——GPT-4级token 2022末$20/M到2026 ~$0.40/M。降多服务栈(阶段17课程04-09)而非硬件。路由是你转价降成margin无产品回退。失败模式便宜模型漂：路由推40%到弱模型、推理任务质量降3-5%、季度无人注意。Gate路由线质量指标而非离线评估集。

**类型:** 学习
**语言:** Python(stdlib、玩具级联路由模拟器)
**前置要求:** 阶段17课程01(托管LLM平台)、阶段17课程19(AI Gateways)
**时间:** ~60分钟

## 学习目标

- 解释模型级联：便宜优先信心检查、低信心升。
- 列举四路由信号(任务分类、提示长度、嵌入相似已知硬集、首遍自信心)。
- 算目标路由分和质量失容忍期望混合成本。
- 命名漂监控指标(线质量gate)捕便宜模型爬。

## 问题背景

服务GPT-5 $80k/month。分析示70%查询简单："巴黎几点？" "重写这句。" Haiku级模型完美处理3%成本。30%需GPT-5推理——编码、数学、多步规划。

若路由70%便宜30%昂贵、账降~65%同产品质量。这是路由。诀是建broker无质量回退。

## 概念讲解

### 四路由信号

1. **任务分类**：简单/复杂/codegen/math/chat。规则分类器、小LLM (Haiku级$0.25/M)、或嵌入相似标记桶。输出：路由 = cheap / balanced / frontier。

2. **提示长度**：提示>4K token常需前沿连贯。提示<500 token通常不需。

3. **嵌入相似已知硬集**：若查询近(余弦> 0.88)已知硬桶、升前沿直。

4. **首遍自信心**：发便宜；若模型log-probs低信心或拒或输出避险语言、前沿重试。~10%流量加P95延迟但余90%省50%+。

### 三模式

**Pre-route** (首分类器)：加~5-10ms延迟；总最快。

**Cascade** (便宜优先、低信心升)：中位延迟~1.2x (便宜跑加验)、升~2x。最佳质量底。

**Ensemble route** (并行跑便宜和前沿样本、reward-model选)：最高质量、最高成本；仅关键A/B用。

### 实现

AI gateways (阶段17课程19)露路由。LiteLLM `router`配置fallback和成本路由。Portkey有guard + 路由。Kong AI Gateway插件基路由。OpenRouter模型市露推荐API。

开源：RouteLLM (LMSYS)、Not Diamond (商业)、Prompt Mule。

### 2026价曲线

| 模型级 | 2022末 | 2026 | 变 |
|-------------|-----------|------|--------|
| GPT-4级质量 | ~$20/M | ~$0.40/M | 50x便宜 |
| 前沿 (GPT-5, Claude 4) | — | ~$3-10/M | 新级 |

改多服务效率——阶段17课程04-09核心课转provider侧成本降。路由让你app层捕这些益而非等全用户迁便宜级。

### 漂是真风险

路由发40%便宜模型。六个月、任务分布移(用户更精、问更长)。路由不觉因分类器Q1数据训。质量默降。无人抱怨够大声。竞基准失才知。

Gate路由线质量指标：

- 每路由用户拇指上/下。
- 每路由留样(5%)自动LLM-judge。
- 升率：若级联踢升>30%、便宜模型过路由。
- 每路由拒率。

### 你应记数

- 2026 iso-quality路由省：20-60%案例。
- LLM价降2022-2026：年~10x聚合。
- GPT-4级2022 vs 2026：~$20/M → ~$0.40/M。
- 级联延迟影响：中位~1.2x、升~2x (~10%流量)。

## 使用

`code/main.py`模混合负载pre-route、级联、ensemble。报混合成本、质量失、升率。

## 交付成果

本lesson产`outputs/skill-router-plan.md`。给负载和质量预算、选路由模式和信号。

## 练习题

1. 跑`code/main.py`。何精度底级联胜pre-route？
2. 用户基30%企业(复杂查询)、70%免费层(简单)。设计路由分。何线指标gate？
3. 路路由降质量2%但省40%。发否？依赖产品——论证双边。
4. 实现信心检查用OpenAI / Anthropic API logprobs。何阈值始？
5. 六个月、升率从8%爬到22%。诊断三因和每修。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 模型路由 | "成本broker" | 每请求动态模型选 |
| 模型级联 | "便宜优先升" | 跑便宜、低信心落前沿 |
| Pre-route | "首分类" | 首分类器；无重跑 |
| Ensemble route | "并行选" | 跑多、reward-model选最佳 |
| 升率 | "升路由%" | 级联请求升分数 |
| RouteLLM | "LMSYS路由" | OSS路由库 |
| Not Diamond | "商业路由" | SaaS模型路由产品 |
| 漂 | "便宜爬" | 分布移路由不觉 |
| 线质量gate | "活检查" | 自动LLM-judge采样活流量 |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — 多模型gateway路由原语。