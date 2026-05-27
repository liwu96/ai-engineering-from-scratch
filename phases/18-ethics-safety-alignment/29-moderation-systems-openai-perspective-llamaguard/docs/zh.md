# Moderation系统——OpenAI、Perspective、Llama Guard

> 产moderation系统操作课程12-16定义安全政策。OpenAI Moderation API：`omni-moderation-latest` (2024)建于GPT-4o文+图一call分类；比前版multilingual test set好42%；response schema返13 category booleans — harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；大多开发者free。Layered patterns：Input moderation(pre-generation)、Output moderation(post-generation)、Custom moderation(域规则)。Async parallel calls hide latency；placeholder responses on flag。Llama Guard 3/4(课程16)：14 MLCommons hazards、Code Interpreter Abuse、8语言(v3)、multi-image(v4)。Perspective API (Google Jigsaw)：毒性评分predating LLM-as-moderator wave；主要单维度毒性带severe-toxicity/insult/profanity变种；content-moderation研baseline。Deprecations：Azure Content Moderator deprecated 2024年2月、retired 2027年2月、换Azure AI Content Safety。

**类型:** 构建
**语言:** Python(stdlib、三层moderation harness)
**前置要求:** 阶段18课程16(Llama Guard / Garak / PyRIT)
**时间:** ~60分钟

## 学习目标

- 描述OpenAI Moderation API category taxonomy和何异Llama Guard 3 MLCommons集。
- 描述三层moderation pattern(input、output、custom)和名每一失败模式。
- 描述Perspective API位置作pre-LLM-era baseline和何研续用。
- 陈Azure deprecation时间线。

## 问题背景

课程12-16描述攻和防御工具。课程29覆部署moderation系统操作防御于用户触产品surface。三层pattern是2026默认配置。

## 概念讲解

### OpenAI Moderation API

`omni-moderation-latest` (2024)。建于GPT-4o。文+图一call分类。大多开发者free。

Categories(13 booleans于response schema):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

多模态支持施于`violence`、`self-harm`、和`sexual`但不`sexual/minors`；其余text-only。

`code/main.py` code harness我们缩`/threatening`、`/intent`、`/instructions`、和`/graphic`子类于顶层parents教简。产代码应全13-category schema。

比前代moderation endpoint multilingual test set好42%。Per-category scores；应用设thresholds。

### Llama Guard 3/4

课程16覆。14 MLCommons hazard categories(组织异OpenAI 13 response-schema booleans)。支持8语言(v3)。Llama Guard 4 (2025年4月)原生多模态、12B。

OpenAI和Llama Guard taxonomies重叠但分。OpenAI有"illicit"作广category；Llama Guard有"violent crimes"和"non-violent crimes"分。部署选基于其policy-taxonomy fit。

### Perspective API (Google Jigsaw)

毒性评分系统predating LLM-as-moderator wave(pre-2020)。Categories：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主score(TOXICITY)带子维度变种。

广用content-moderation研baseline因API稳、文档、和数年calibration数据。现代LLM邻用例、Llama Guard或OpenAI Moderation典型更fit。

### 三层pattern

1. **Input moderation。** 生前分类用户提示。若flag拒。Latency：一分类器call。
2. **Output moderation。** 交付前分类模型输出。若flag换拒。Latency：生后一分类器call。
3. **Custom moderation。** 域特定规则(regex、allowlists、business policy)。Run于input或output。

三层sequential设计：input moderation须生前完、output moderation生后run。Parallelism施于层内 — 同文跑多分类器(如OpenAI Moderation + Llama Guard + Perspective)并发hide每分类器latency。可选优化、placeholder响("一moment、checking...")示input moderation完时并token-1 streaming deferred。Flag行为可配置：拒、sanitize、escalate人review。

### 失败模式

- **仅Input。** 不捕输出幻觉(课程12-14编码攻绕input classifiers)。
- **仅Output。** 允任何input达模型；增成本；暴露内推理于攻者。
- **仅Custom。** 不跨categories鲁棒；regexes脆。

Layered是默认。Belt-and-suspenders。

### Azure deprecation

Azure Content Moderator：deprecated 2024年2月、retired 2027年2月。换Azure AI Content Safety、LLM基并与Azure OpenAI集成。迁移是2024-2027 Azure部署field级项目。

### Phase 18何处

课程16覆moderation工具于红队context。课程29覆部署moderation。课程30闭于当前双用能力证据。

## 使用

`code/main.py`三层moderation harness：input moderator(keyword + category score)、output moderator(同分类器于输出)、custom moderator(域规则)。可跑inputs并观何层捕何。

## 交付成果

本lesson产`outputs/skill-moderation-stack.md`。给部署、荐moderation stack配置：何分类器于input、何于output、何custom规则、和何法官于边缘例。

## 练习题

1. 跑`code/main.py`。跑良性、边缘、和有害input三层。报何层每发。
2. 扩harness带Perspective-API风格特定类别毒性评分。比threshold行为category score。
3. 读OpenAI Moderation API docs和Llama Guard 3 category list。绘每OpenAI category近Llama Guard categories。识三categories不干净绘。
4. 设计code-assistant部署(如GitHub Copilot)moderation stack。识最和最相关categories并提custom规则。
5. Azure Content Moderator retire 2027年2月。Plan迁移Azure AI Content Safety。识迁移最高风险元素。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o基13-category(文)分类器带部多模态支持 |
| Perspective API | "Google Jigsaw毒性" | Pre-LLM-era毒性评分baseline |
| Llama Guard | "MLCommons 14-category" | Meta hazard分类器(v3: 8B文、8语言；v4: 12B多模态) |
| Input moderation | "pre-generation filter" | 模型call前用户提示分类器 |
| Output moderation | "post-generation filter" | 交付前模型输出分类器 |
| Custom moderation | "域规则" | 部署特定规则(regex、allowlist、policy) |
| Layered moderation | "全三层" | 标准产部署pattern |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure换