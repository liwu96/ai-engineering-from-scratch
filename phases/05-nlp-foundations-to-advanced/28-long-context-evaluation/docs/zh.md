# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro宣称10M词元上下文。1M词元时，8针MRCR降到26.3%。宣称≠可用。长上下文评估告诉你你正在使用的模型的真实容量。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程13(问答)、阶段5课程23(分块策略)
**时间:** ~60分钟

## 问题背景

你有200页合同。模型宣称1M词元上下文。你粘贴合同问:"终止条款是什么?"模型答——但从封面页答,因终止条款在120k词元深处,超过模型实际注意力范围。

这是2026上下文容量差距。规格单说1M或10M。现实说60-70%可用,"可用"取决于任务。

- **检索(干草堆单针):** 前沿模型宣称最大值附近近乎完美。
- **多跳/聚合:** 大多数模型超~128k急剧退化。
- **分散事实推理:** 第一个失败的任务。

长上下文评估测这些轴。本课命名基准、每个实际测什么、如何为你领域构定制针测试。

## 概念讲解

![NIAH基线,RULER多任务,LongBench整体](../assets/long-context-eval.svg)

**Needle-in-a-Haystack(NIAH, 2023)。** 控制深度放事实("魔法词是pineapple")在长上下文。让模型检索。扫描深度×长度。原始长上下文基准。前沿模型现饱和;必要但不充分基线。

**RULER(Nvidia, 2024)。** 4类13任务类型:检索(单/多键/多值)、多跳追踪(变量追踪)、聚合(常见词频)、问答。可配置上下文长(4k到128k+)。揭示饱和NIAH但多跳失败的模型。2024版中,17个宣称32k+上下文模型仅半在32k保持质量。

**LongBench v2(2024)。** 503选择题,8k-2M词上下文,六任务类:单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构数据。真实世界长上下文行为生产基准。

**MRCR(多轮共指消解)。** 大规模多轮共指。8针、24针、100针变体。暴露模型注意力退化前能处理多少事实。

**NoLiMa。** "非词汇针。"针和查询无字面重叠;检索需一步语义推理。比NIAH难。

**HELMET。** 拼接多文档,从任意一个问问题。测选择性注意。

**BABILong。** 嵌bAbI推理链进无关干草堆。测干草堆推理,非仅检索。

### 实际报什么

- **宣称上下文窗口。** 规格单数。
- **有效检索长度。** NIAH某阈值(如90%)通过。
- **有效推理长度。** 多跳或聚合该阈值通过。
- **退化曲线。** 准确率vs上下文长度,每任务类型画。

规格单两个数:检索有效和推理有效。推理有效常是宣称窗口25-50%。

## 动手实践

### Step 1:为你领域定制NIAH

见`code/main.py`。骨架:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫`depth_ratio`∈{0, 0.25, 0.5, 0.75, 1.0}×`total_tokens`∈{1k, 4k, 16k, 64k}。画热力图。这是你目标模型NIAH卡。

### Step 2:多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

问"三个魔法词是什么?"需检索全部三个。单针成功不预测多针成功。

### Step 3:多跳变量追踪(RULER风格)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需链三赋值。128k前沿模型常降至50-70%准确率。

### Step 4:你栈上LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报每类准确率。聚合分数藏大任务级差异。

## 陷阱

- **仅NIAH评估。** 1M词元通过NIAH对多跳无意义。总跑RULER或定制多跳测试。
- **均匀深度采样。** 多实现仅测depth=0.5。测depth=0, 0.25, 0.5, 0.75, 1.0——"中间丢失"效应真实。
- **词汇重叠填充。** 针与填充共享关键词则检索变简单。用NoLiMa风格非重叠针。
- **忽略延迟。** 1M词元提示花30-120秒预填充。测首词元时间配准确率。
- **供应商自报数。** OpenAI、Google、Anthropic都发自己分数。总在你用例独立重跑。

## 实际应用

2026栈:

| 情况 | 基准 |
|------|------|
| 快健全检查 | 定制NIAH 3深度×3长度 |
| 生产模型选 | RULER(13任务)在你目标长度 |
| 真实世界问答质量 | LongBench v2单文档问答子集 |
| 多跳推理 | BABILong或定制变量追踪 |
| 对话/对话 | MRCR 8针在你目标长度 |
| 模型升级回归 | 固定内部NIAH+RULER harness,每新模型跑 |

生产经验法则:不做NIAH+1推理任务在你意图长度前不信上下文窗口。

## 产出成果

存`outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: 为给定模型和用例设计长上下文评估电池。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和用例,输出:

1. 测试。NIAH深度×长度网格;RULER多跳;定制领域任务。
2. 采样。每长度深度0, 0.25, 0.5, 0.75, 1.0。
3. 指标。检索通过率;推理通过率;首词元时间;每查询成本。
4. 截止。有效检索长度(90%通过)和有效推理长度(70%通过)。报两。
5. 回归。固定harness,每模型升级重跑,显差异。

拒绝仅从模型卡信上下文窗口。拒绝任何多跳负载仅NIAH评估。拒绝供应商自报长上下文分数作独立证据。
```

## 练习题

1. **简单。** 构NIAH配3深度(0.25, 0.5, 0.75)×3长度(1k, 4k, 16k)。跑任意模型。画通过率作3×3热力图。
2. **中等。** 加3针变体。测每长度检索全部3。比同长度单针通过率。
3. **困难。** 构变量追踪任务(X1→X2→X3,3跳)嵌64k填充。测3前沿模型准确率。报每模型有效推理长度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| NIAH | 干草堆找针 | 填充放事实,让模型检索。 |
| RULER | NIAH加强版 | 检索/多跳/聚合/问答13任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍超阈值长度。 |
| 中间丢失 | 深度偏 | 模型对长输入中间内容注意力不足。 |
| 多针 | 同时多事实 | 多放置;测注意力处理,非仅检索。 |
| MRCR | 多轮共指 | 8、24或100针共指;暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询无字面词元重叠;需推理。 |

## 延伸阅读

- [Kamradt(2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)——原始NIAH仓库。
- [Hsieh等(2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654)——多任务基准。
- [Bai等(2024). LongBench v2](https://arxiv.org/abs/2412.15204)——真实世界长上下文评估。
- [Modarressi等(2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666)——更难针。
- [Kuratov等(2024). BABILong](https://arxiv.org/abs/2406.10149)——干草堆推理。
- [Liu等(2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)——深度偏论文。