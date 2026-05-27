# 大语言模型评估 — RAGAS, DeepEval, G-Eval

> 精确匹配和F1错过语义等价。人工审查不可扩展。LLM-as-judge是生产答案——配足够校准以信任数值。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程13(问答)、阶段5课程14(信息检索)
**时间:** ~75分钟

## 问题背景

你的RAG系统回答:"June 29th, 2007."
黄金参考是:"June 29, 2007."
精确匹配评分0。F1评分约75%。人工会评分100%。

乘以10,000个测试案例。再乘以检索器、分块、提示词或模型的每次变更。你需要一个理解意义、大规模廉价运行、不隐瞒回归、暴露正确失败模式的评估器。

2026年有三个框架占据这个问题。

- **RAGAS。**检索增强生成评估。四个RAG指标(忠实度、答案相关性、上下文精度、上下文召回)配NLI + LLM评判后端。研究支持,轻量级。
- **DeepEval。**LLM的Pytest。G-Eval、任务完成、幻觉、偏见指标。CI/CD原生。
- **G-Eval。**一个方法(也是DeepEval指标):LLM-as-judge配思维链、自定义准则、0-1评分。

三者都依赖LLM-as-judge。本课程构建对该方法的直觉及其周围的信任层。

## 概念讲解

![四个评估维度,LLM-as-judge架构](../assets/llm-evaluation.svg)

**LLM-as-judge。**用LLM按评分标准对输出评分替换静态指标。给定`(query, context, answer)`,提示评判LLM:"按忠实度评分0-1。"返回分数。

为何有效:LLM以极小成本近似人工判断。GPT-4o-mini约$0.003每评分案例,使1000样本回归评估运行低于$5。

为何静默失败:

1. **评判偏见。**评判者偏好更长答案、自己模型家族的答案、匹配提示风格的答案。
2. **JSON解析失败。**坏JSON → NaN分数 → 静默从聚合中排除。RAGAS用户知道这个痛苦。用try/except + 显式失败模式把关。
3. **模型版本漂移。**升级评判者改变每个指标。冻结评判模型 + 版本。

**RAG四个指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| 忠实度 | 答案中每个声明是否来自检索上下文? | NLI-based蕴含 |
| 答案相关性 | 答案是否回答了问题? | 从答案生成假设问题;与真实问题比较 |
| 上下文精度 | 检索块中,多少比例相关? | LLM评判 |
| 上下文召回 | 检索是否返回所有所需? | LLM评判对照黄金答案 |

**G-Eval。**定义自定义准则:"答案是否引用了正确来源?"框架自动展开为思维链评估步骤,然后评分0-1。适用于RAGAS未覆盖的领域特定质量维度。

**校准。**在与人工标签有相关性前绝不信任原始评判分数。运行100个手标注示例。绘制评判vs人工。计算Spearman rho。如果rho < 0.7,你的评判评分标准需要改进。

## 动手实践

### Step 1: 用NLI的忠实度(RAGAS风格)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm`是任意可调用:提示词str -> 生成str。
# 示例:llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""将此答案分解为简单事实声明(每行一个):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明。NLI检查每个声明对照检索上下文。忠实度 = 支撑比例。

### Step 2: 答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder:任何实现.encode(texts, normalize_embeddings=True) -> ndarray的模型
# 例如,encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"写出{n}个此答案可能是答案的问题:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问不同,相关性下降。

### Step 3: G-Eval自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="答案应该事实准确并匹配预期输出。",
    evaluation_steps=[
        "读取预期输出。",
        "读取实际输出。",
        "列出实际输出中的事实声明。",
        "对每个声明,标记被预期输出支撑或未支撑。",
        "返回分数 = 支撑比例。",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="第一代iPhone何时发布?",
                   actual_output="2007年6月29日。",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤是评分标准。显式步骤比隐式"评分0-1"提示词更稳定。

### Step 4: CI门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"忠实度回归于{case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"相关性回归于{case.id}"
```

作为pytest文件发布。每个PR运行。回归时阻止合并。

### Step 5: 从头玩具评估

见`code/main.py`。只用stdlib的忠实度(答案声明与上下文重叠)和相关性(答案词元与问题词元重叠)近似。非生产。展示结构。

## 陷阱

- **无校准。**0.3与人工标签相关性的评判是噪声。发布前要求校准运行。
- **自评估。**用同一LLM生成和评判使分数膨胀10-20%。用不同模型家族做评判。
- **成对评判的位置偏见。**评判者偏好第一个展示的选项。总是随机顺序并运行两个方向。
- **原始聚合隐藏失败。**平均分数0.85常隐藏5%灾难性失败。总是检查底部分位数。
- **黄金数据集腐化。**未版本化的评估集随时间漂移破坏纵向比较。每次变更标记数据集。
- **LLM成本。**大规模,评判调用主导成本。用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 实际应用

2026栈:

| 用例 | 框架 |
|------|------|
| RAG质量监控 | RAGAS(4指标) |
| CI/CD回归门控 | DeepEval + pytest |
| 自定义领域准则 | G-Eval within DeepEval |
| 在线流量实时监控 | RAGAS reference-free模式 |
| 人在环抽查 | LangSmith或Phoenix配标注UI |
| 红队/安全评估 | Promptfoo + DeepEval |

典型栈:RAGAS监控、DeepEval CI、G-Eval新维度。三者都运行;它们有用分歧。

## 产出成果

保存为`outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: 设计配校准评判和CI门控的LLM评估计划。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定用例(RAG / agent / 生成任务),输出:

1. 指标。忠实度/相关性/上下文精度/上下文召回 + 任何配准则的自定义G-Eval指标。
2. 评判模型。命名模型 + 版本,成本vs准确率理由。
3. 校准。手标注集大小,目标Spearman rho vs人工 > 0.7。
4. 数据集版本化。标签策略、变更日志、分层。
5. CI门控。每指标阈值、回归窗口逻辑、底分位数警报。

拒绝依赖未测试≥50人工标注示例的评判。拒绝自评估(同一模型生成 + 评判)。拒绝无底部10%暴露的聚合报告。标记任何评判升级无并行基线评估的流水线。
```

## 练习题

1. **简单。**在10个已知幻觉的RAG示例上用RAGAS。验证忠实度指标捕获每个。
2. **中等。**手标注50问答答案0-1正确性。用G-Eval评分。测量评判与人工间Spearman rho。
3. **困难。**用DeepEval构建pytest CI门控。故意回归检索器。验证门控失败。添加底部分位数警报通过最低10%阈值检查。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| LLM-as-judge | 用LLM评分 | 提示评判模型按评分标准对输出评分0-1。 |
| RAGAS | RAG指标库 | 开源评估框架配4个无参考RAG指标。 |
| 忠实度 | 答案有据吗? | 答案声明被检索上下文蕴含的比例。 |
| 上下文精度 | 检索块相关吗? | Top-K块中实际重要的比例。 |
| 上下文召回 | 检索找到所有吗? | 黄金答案声明被检索块支撑的比例。 |
| G-Eval | 自定义LLM评判 | 评分标准 + 思维链评估步骤 + 0-1分数。 |
| 校准 | 信任但要验证 | 评判分数与人工分数间Spearman相关。 |

## 延伸阅读

- [Es等(2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)——RAGAS论文。
- [Liu等(2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)——G-Eval论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction)——开源生产栈。
- [Zheng等(2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)——偏见、校准、限制。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers)——集成RAGAS、DeepEval、Phoenix的统一框架。