# 提示词缓存与上下文缓存

> 你系统提示词是4,000 tokens。你RAG上下文是20,000 tokens。你随每请求发两者。你也为两者付—每时间。提示词缓存让提供方侧保那前缀热并于复用时你付正率10%。正确用，它减推理成本50–90%和首token延迟40–85%。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段11课程01(提示词工程)，阶段11课程05(上下文工程)，阶段11课程11(缓存和成本)
**时间:** ~60分钟

## 问题背景

编码代理于对话每轮发同15,000-token系统提示词至Claude。20轮于$3/M输入token仅输入成本$0.90—于任用户实消息前。乘10,000日对话账单于永不改文达$9,000/日。

你不可缩提示词无伤质量。你不可避发它—模型每轮需它。唯移动是停付提供方已见前缀全价。

那移动是提示词缓存。Anthropic于2024年8月发它(带2025年1小时延TTL变种)、OpenAI后那年自动化它、Google于Gemini 1.5旁发显上下文缓存、三现于其前沿模型供它为一等特性。

## 概念讲解

![提示词缓存:写一次、读便宜](../assets/prompt-caching.svg)

**机制。**当请求前缀匹近请求前缀，提供方从前跑服KV-cache而非重编码token。你首时付小写溢价后每时付大读折扣。

**2026三提供方味。**

| 提供方 | API风格 | 击折扣 | 写溢价 | 默TTL | 最小可缓存 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上显`cache_control`标记 | 90%输入减 | 25%附加 | 5分(延至1小时) | 1,024 tokens(Sonnet/Opus), 2,048(Haiku) |
| OpenAI | 自动前缀检测 | 50%输入减 | 无 | 至1小时(尽力) | 1,024 tokens |
| Google(Gemini) | 显`CachedContent` API | 存储计费;读约正25% | 每token·小时存储费 | 用户设(默1小时) | 4,096 tokens(Flash), 32,768(Pro) |

**不变。**三仅缓存前缀。若任token于请求间异，首异token后一切失。把*稳*部于顶、*变*部于底。

### 缓存友好布局

```
[系统提示词]          <-- 缓此
[工具定义]           <-- 缓此
[少样本例]           <-- 缓此
[检索文档]           <-- 若复用则缓，否则不
[对话历史]           <-- 缓至最后轮
[当前用户消息]        <-- 永不缓存(每次异)
```

违序—用户消息于系统提示词上、动态检索间穿插少样本—缓存永不击。

### 盈亏计算

Anthropic 25%写溢价意缓存块须至少读两次才净省钱。1写+1读平每请求0.675x成本(省32%);1写+10读平0.205x(省80%)。规则:缓存你期于TTL内复用至少3次任物。

## 构建

### 步骤1:Anthropic带显标记提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "你是高级Python审员。精确随评分标准。\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control`标记告Anthropic存块5分钟。那窗内复用击;窗后过期再写。

**响应用量字段:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 于1.25x付
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 于0.1x付
```

于CI查两字段—若`cache_read_input_tokens`跨请求持零，你缓存键漂移。

### 步骤2:一小时延TTL

于长跑批作业，5分默作业间过期。设`ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1小时TTL成本2x写溢价(基线50%而非25%)但于批复用前缀超5次快付回。

### 步骤3:OpenAI自动缓存

OpenAI给你无配。任超1,024 token前缀匹近请求得50%折扣自动。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 折扣部分
```

同缓存友好布局规则应用。两物杀OpenAI缓存不杀Anthropic:改`user`字段(用作缓存键组件)和重排工具。

### 步骤4:Gemini显上下文缓存

Gemini视缓存为创和命的一等对象:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["审这代码:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini缓存存活时按token·小时收费存储，并于约正输入率25%读。这是当你跨多会话天复用同巨大提示词时正确形。

### 步骤5:产中测击率

见`code/main.py`模拟三提供方会计师追写/读/失计数并算每1K请求混合成本。于目击率发门—多Anthropic产设置应暖后见>80%读比例。

## 2026仍发陷阱

- **顶动态时间戳。**`"当前时间: 2026-04-22 15:30:02"`于系统提示词顶。每请求失。移时间戳至缓存断点下。
- **工具重排。**于稳序序列化工具—部署间dict reshuffle破每击。
- **自由文近重复。**"You are helpful." vs "You are a helpful assistant."—一字节差=全失。
- **太小块。**Anthropic强1,024-token底(Haiku 2,048)。小块静不缓存。
- **盲成本仪表板。**裂"输入token"为缓存vs未缓存。否则流量降看如缓存胜。

## 使用

2026缓存栈:

| 情况 | 择 |
|-----------|------|
| 稳10k+系统提示词代理、多轮 | Anthropic `cache_control`带5分TTL |
| 批作业复用前缀30+分钟 | Anthropic带`ttl: "1h"` |
| GPT-5上无服务器端点、无自定义基础设施 | OpenAI自动(仅使你前缀稳且长) |
| 多日复用巨大代码/文档语料库 | Gemini显`CachedContent` |
| 跨提供方回退 | 保持缓存友好前缀布局跨提供方同使任击工作 |

配语义缓存(阶段11课程11)于用户消息层:提示词缓存理*token同*复用、语义缓存理*义同*复用。

## 交付成果

存`outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: 设计缓存友好提示词布局并择正确提供方缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定提示词(系统+工具+少样本+检索+历史+用户)和用量配置(每小时请求、需TTL、提供方)，输出:

1. 布局。重排序节带单缓存断点标记;释何节稳、何节变。
2. 提供方模式。Anthropic cache_control、OpenAI自动或Gemini CachedContent。从TTL和复用模式释。
3. 盈亏。TTL内期每写读数;无缓存vs净成本带数学。
4. 验计划。CI断言第二同请求cache_read_input_tokens > 0;仪表板按缓存vs未缓存token裂。
5. 失败模式。列此设置缓存将失三最可能原因(动态时间戳、工具重排、近重复文)和你何防每。

拒发于断点上置动态字段缓存计划。拒启1h TTL无使2x写溢价付回复用计数。
```

## 练习题

1. **易。**取10轮对话带5,000-token系统提示词对Claude。无`cache_control`跑后带跑。报每输入token账单。
2. **中。**写测架，给提示词模板和请求日志，算期击率和每提供方美元省(Anthropic 5m、Anthropic 1h、OpenAI自动、Gemini显)。
3. **难。**建布局优化器:给提示词和标`stable=True/False`字段列表，重写提示词于最大缓存友好位置置单缓存断点无失信息。于真Anthropic端验。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存 | "使长提示词便宜" | 匹前缀复用提供方侧KV-cache;重复输入token50-90%折扣。 |
| `cache_control` | "Anthropic标记" | 内容块属性告"至此一切可缓存";`{"type": "ephemeral"}`。 |
| 缓存写 | "付溢价" | 首请求填充缓存;Anthropic输入率约1.25x计、OpenAI免费。 |
| 缓存读 | "折扣" | 后匹前缀请求;Anthropic 10%、OpenAI 50%、Gemini约25%计。 |
| TTL | "存活多久" | 缓存保暖秒;Anthropic 5分默(延1h)、OpenAI尽力至1h、Gemini用户设。 |
| 延TTL | "1小时Anthropic缓存" | `{"type": "ephemeral", "ttl": "1h"}`;2x写溢价但批复用值。 |
| 前缀匹 | "何我缓存失" | 缓存仅当从起始至断点每token字节同时击。 |
| 上下文缓存(Gemini) | "显式" | Google命名、存储计费缓存对象;最佳于大语料库多日复用。 |

## 延伸阅读

- [Anthropic — 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、盈亏表。
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API和存储定价。
- [Anthropic工程 — 长上下文工作负载提示词缓存](https://www.anthropic.com/news/prompt-caching) — 原发帖带延迟数。
- 阶段11课程05(上下文工程) — 何切片提示词使缓存可落。
- 阶段11课程11(缓存和成本) — 配用户消息上语义缓存提示词缓存。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存暴露用户KV缓存内存模型;释何缓存前缀重读比重算约10×便宜。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill是提示词缓存短截阶段;这论文释何TTFT于缓存击剧降而TPOT不受。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与推测解码、Flash Attention和MQA/GQA并肩为弯推理成本曲线杠杆;读此为他三。