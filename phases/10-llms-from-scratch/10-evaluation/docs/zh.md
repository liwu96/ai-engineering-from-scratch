# 评估：基准、评测、LM Harness

> 古德哈特定律：当一项措施成为目标时，它就不再是一项好措施。每个前沿实验室都在游戏基准。MMLU 分数上升，而模型仍然无法可靠地数出"strawberry"中有多少个 R。唯一重要的评估是*你的*评估——在*你的*任务上，用*你的*数据。

**类型:** 构建
**语言:** Python
**前置要求:** 第10阶段，第01-05课（从头开始的 LLM）
**时间:** ~90分钟

## 学习目标

- 构建自定义评估工具，针对语言模型运行多选和开放式基准测试
- 解释为什么标准基准（MMLU、HumanEval）饱和并无法区分前沿模型
- 使用适当的指标实现任务特定评估：精确匹配、F1、BLEU 和 LLM 作为评判者评分
- 设计针对你特定用例的自定义评估套件，而非仅依赖公共排行榜

## 问题背景

MMLU 于 2020 年发布，包含 57 个学科的 15,908 个问题。三年内，前沿模型将其饱和。GPT-4 得分 86.4%。Claude 3 Opus 得分 86.8%。Llama 3 405B 得分 88.6%。排行榜压缩到一个 3 分范围内，差异是统计噪声，而非真实能力差距。

与此同时，这些相同模型在 10 岁儿童不假思索就能处理的任务上失败。Claude 3.5 Sonnet，在 MMLU 上得分 88.7%，最初无法数出"strawberry"中的字母——一个需要零世界知识和零推理的任务，只是字符级迭代。HumanEval 用 164 个问题测试代码生成。模型得分 90%+，同时仍在产生在任何初级开发人员都会发现的边缘情况上崩溃的代码。

基准性能与现实世界可靠性之间的差距是 LLM 评估的核心问题。基准告诉你模型在基准上的表现如何。它们几乎不告诉你该模型将如何在*你的*特定任务、*你的*特定数据、*你的*特定失败模式下表现。如果你正在构建客户支持机器人，MMLU 无关紧要。如果你正在构建代码助手，HumanEval 只覆盖函数级生成——它不说关于跨文件调试、重构或解释代码的任何内容。

你需要自定义评估。不是因为基准无用——它们对粗略模型选择有用——而是因为最终评估必须与部署条件完全匹配。

## 概念讲解

### 评估格局

评估有三类，每类有不同的成本和信号质量。

**基准**是标准化测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你在基准上运行模型并获得分数。优势：每个人都使用相同的测试，所以你可以比较模型。劣势：模型和训练数据越来越多地污染这些基准。实验室在包含基准问题的数据上训练。分数上升。能力可能没有。

**自定义评估**是你为特定用例构建的测试套件。你定义输入、预期输出和评分函数。法律文档摘要器在法律文档上评估。SQL 生成器在你的数据库模式上评估。这些创建起来很昂贵，但它们是唯一预测生产性能的评估。

**人类评估**使用付费标注员在有用性、正确性、流畅性和安全性等标准上评判模型输出。开放式任务的金标准，自动化评分失败。Chatbot Arena 已收集超过 200 万条跨 100+ 模型的人类偏好投票。劣势：成本（每次评判 $0.10-$2.00）和速度（数小时到数天）。

### 为什么基准会失效

三个机制导致基准分数停止反映真实能力。

**数据污染。** 训练语料库抓取互联网。基准问题在互联网上。模型在训练期间看到答案。这不是传统意义上的作弊——实验室不故意包含基准数据。但网络规模抓取几乎不可能排除。

**应试教学。** 实验室优化训练混合以获得基准性能。如果训练混合的 5% 是 MMLU 风格的多选，模型学习格式和答案分布。MMLU 是 4 选多选。模型学习答案分布在 A/B/C/D 上大致均匀，即使模型不知道答案也有帮助。

**饱和。** 当每个前沿模型在基准上得分 85-90% 时，基准停止区分。剩余的 10-15% 问题可能模糊、标签错误或需要晦涩的领域知识。在 MMLU 上从 87% 提升到 89% 可能意味着模型记住了两个更多的晦涩问题，而非变得更聪明。

### 困惑度：快速健康检查

困惑度衡量模型对一系列 Token 的惊讶程度。形式上，它是指数化的平均负对数似然：

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

困惑度为 10 意味着模型平均像在每个 Token 位置从 10 个选项中均匀选择一样不确定。越低越好。GPT-2 在 WikiText-103 上获得约 30 的困惑度。GPT-3 获得约 20。Llama 3 8B 获得约 7。

困惑度对在同一测试集上比较模型有用，但它有盲点。模型可以通过擅长预测常见模式而在困惑度上表现好，同时在罕见但重要的模式上表现差。它也说不出关于指令遵循、推理或事实准确性的任何内容。用它作为理智检查，而非最终判决。

### LLM 作为评判者

使用强模型来评估弱模型的输出。想法很简单：让 GPT-4o 或 Claude Sonnet 在 1-5 分的尺度上对正确性、有用性和安全性进行评分。每次评判花费约 $0.01（使用 GPT-4o-mini），与人工评判的相关性出奇地好——大多数任务约 80% 一致。

评分提示比模型更重要。模糊的提示（"评价这个响应"）产生嘈杂的分数。带评分标准的结构化提示（"如果答案事实正确并引用来源则打 5 分，正确但未引用来源打 4 分，部分正确打 3 分..."）产生一致、可重复的分数。

失败模式：评判模型表现出位置偏见（在成对比较中偏好第一个响应）、冗长偏见（偏好更长的响应）和自我偏好（GPT-4 给 GPT-4 输出打分高于等效 Claude 输出）。缓解措施：随机化顺序、按长度归一化、使用与正在评估的模型不同的评判者。

### 成对比较的 ELO 评分

Chatbot Arena 的方法。向人类（或 LLM 评判者）展示来自不同模型的相同提示的两个响应。挑选更好的一个。从数千次这样的比较中，计算每个模型的 ELO 评分——与国际象棋中使用的相同系统。

ELO 优势：相对排名比绝对评分更可靠，优雅地处理平局，比独立评分每个输出用更少的比较收敛。截至 2026 年初，Chatbot Arena 排名显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在顶部分隔在 20 个 ELO 点内。

### 评估框架

**lm-evaluation-harness** (EleutherAI)：标准的开源评估框架。支持 200+ 基准。用一个命令在 MMLU、HellaSwag、ARC 等上运行任何 Hugging Face 模型。用于开放 LLM 排行榜。

**RAGAS**：专门用于 RAG 管道的评估框架。测量忠实性（答案是否与检索到的上下文匹配？）、相关性（检索到的上下文是否与问题相关？）和答案正确性。

**promptfoo**：用于提示工程的配置驱动评估。在 YAML 中定义测试用例，针对多个模型运行，获得通过/失败报告。对提示进行回归测试有用——确保提示更改不会破坏现有测试用例。

### 构建自定义评估

唯一对生产重要的评估。过程：

1. **定义任务。** 模型到底应该做什么？要精确。"回答问题"太模糊。"给定客户投诉邮件，提取产品名称、问题类别和情感"是一个可以评估的任务。

2. **创建测试用例。** 原型评估最少 50 个，生产 200+。每个测试用例是一个（输入，预期输出）对。包括边缘情况：空输入、对抗性输入、模糊输入、其他语言的输入。

3. **定义评分。** 结构化输出精确匹配。文本相似性 BLEU/ROUGE。开放式质量 LLM 作为评判者。提取任务 F1。用权重组合多个指标。

4. **自动化。** 每次评估用一个命令运行。没有手动步骤。以支持随时间比较的方式存储结果。

5. **跟踪随时间变化。** 孤立地看评估分数毫无意义。你需要趋势线。上次提示更改后分数提高了吗？切换模型后下降了吗？版本化你的评估与你的提示。

| 评估类型 | 每次评判成本 | 与人类一致性 | 最适合 |
|-----------|-------------|-----------|--------|
| 精确匹配 | ~$0 | 100%（适用时） | 结构化输出、分类 |
| BLEU/ROUGE | ~$0 | ~60% | 翻译、摘要 |
| LLM 作为评判者 | ~$0.01 | ~80% | 开放式生成 |
| 人类评估 | $0.10-$2.00 | N/A（是真实值） | 模糊的、高风险的 |

## 动手实践

### 步骤1：最小评估框架

定义核心抽象。一个评估用例有输入、预期输出和可选的元数据字典。一个评分器接受预测和参考并返回 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 步骤2：评分函数

构建精确匹配、Token F1 和模拟的 LLM 作为评判者评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 步骤3：ELO 评分系统

实现成对比较与 ELO 更新。这正是 Chatbot Arena 用于排名模型的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 步骤4：困惑度计算

使用 Token 概率计算困惑度。在实践中你会从模型的 logits 中获取这些。这里我们用概率分布模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 步骤5：汇总结果

计算评估运行的汇总统计：均值、中位数、阈值通过率以及按指标细分。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} 汇总")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    均值:      {stats['mean']:.3f}")
        print(f"    中位数:    {stats['median']:.3f}")
        print(f"    标准差:    {stats['std']:.3f}")
        print(f"    范围:      [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    通过率:    {stats['pass_rate']:.1%} (阈值 >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### 步骤6：运行完整管道

将所有内容连接起来。定义任务，创建测试用例，模拟两个模型，运行评估，从成对比较计算 ELO，打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "法国的首都是哪里？": "巴黎",
        "2 + 2 是多少？": "4",
        "谁写了哈姆雷特？": "威廉·莎士比亚",
        "PyTorch 是用什么语言写的？": "Python 和 C++",
        "水的沸点是多少？": "100 摄氏度",
    }
    return responses.get(prompt, "我不知道")

def demo_model_bad(prompt):
    responses = {
        "法国的首都是哪里？": "巴黎是法国的首都",
        "2 + 2 是多少？": "答案是四",
        "谁写了哈姆雷特？": "莎士比亚",
        "PyTorch 是用什么语言写的？": "Python",
        "水的沸点是多少？": "212 华氏度",
    }
    return responses.get(prompt, "未知")

cases = [
    EvalCase("法国的首都是哪里？", "巴黎"),
    EvalCase("2 + 2 是多少？", "4"),
    EvalCase("谁写了哈姆雷特？", "威廉·莎士比亚"),
    EvalCase("PyTorch 是用什么语言写的？", "Python 和 C++"),
    EvalCase("水的沸点是多少？", "100 摄氏度"),
]

suite = EvalSuite(
    name="常识",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "模型 A (简洁)")
print_summary(summarize_results(results_bad), "模型 B (冗长)")
```

"好"模型给出精确答案。"坏"模型给出冗长的释义。精确匹配严厉惩罚冗长模型。Token F1 和 LLM 作为评判者更宽容。这说明了为什么指标选择很重要：同一模型根据你如何评分看起来伟大或糟糕。

### 步骤7：ELO 锦标赛

跨多轮在模型之间运行成对比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO 排行榜:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### 步骤8：困惑度比较

比较不同质量"模型"之间的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "强模型"), (0.7, "中等模型"), (0.4, "弱模型")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): 困惑度 = {ppl:.2f}")
```

## 使用实践

### lm-evaluation-harness (EleutherAI)

在任何模型上运行基准的标准工具。

```python
# pip install lm-eval
# 命令行:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

用于提示工程的配置驱动评估。在 YAML 中定义测试并针对多个提供者运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "用一个词回答：{{question}}"

tests:
  - vars:
      question: "法国的首都是哪里？"
    assert:
      - type: contains
        value: "巴黎"
  - vars:
      question: "2 + 2 是多少？"
    assert:
      - type: equals
        value: "4"
```

### RAGAS 用于 RAG 评估

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 测量通用评估遗漏的内容：模型的答案是否基于检索到的上下文，而不仅仅是答案在抽象意义上是否"正确"。

## 产出成果

本课程产出 `outputs/prompt-eval-designer.md` —— 一个可重用提示，为任何任务设计自定义评估套件。给它任务描述，它生成测试用例、评分函数和通过/失败阈值建议。

它还产出 `outputs/skill-llm-evaluation.md` —— 一个决策框架，根据你的任务类型、预算和延迟要求选择正确的评估策略。

## 练习题

1. 添加"一致性"评分器，将相同输入通过模型运行 5 次并测量输出匹配频率。确定性输入上的不一致答案揭示脆弱的提示或高温设置。

2. 扩展 ELO 跟踪器以支持多个评判函数（精确匹配、F1、LLM 作为评判者）并加权。比较当你重度加权精确匹配与重度加权 F1 时排行榜如何变化。

3. 为特定任务构建评估套件：5 个类别的电子邮件分类。创建 100 个测试用例，包括边缘情况的多样示例（可能属于多个类别的电子邮件、空邮件、其他语言的邮件）。测量不同"模型"（基于规则、关键词匹配、模拟 LLM）的表现。

4. 实现污染检测：给定一组评估问题和训练语料库，检查评估问题的百分比（或接近的释义）出现在训练数据中。这是研究人员审计基准有效性的方式。

5. 构建"模型差异"工具。给定两个模型版本的评估结果，突出显示哪些具体测试用例改进、哪些退步、哪些保持不变。这是评估等价于代码差异的东西——对了解更改是帮助还是伤害至关重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MMLU | "基准" | 大规模多任务语言理解——57 个学科的 15,908 道多选题，到 2025 年超过 88% 饱和 |
| HumanEval | "代码评估" | OpenAI 的 164 个 Python 函数完成问题，仅测试孤立函数生成 |
| SWE-bench | "真实编码评估" | 来自 12 个 Python 仓库的 2,294 个 GitHub 问题，测量端到端错误修复包括测试生成 |
| 困惑度 | "模型有多困惑" | exp(-avg(log P(token_i given context))) —— 越低表示模型赋予实际 Token 更高概率 |
| ELO 评分 | "模型的国际象棋排名" | 从成对胜负记录计算的相对技能评分，Chatbot Arena 用于排名 100+ 模型 |
| LLM 作为评判者 | "用 AI 给 AI 打分" | 强模型按照评分标准给弱模型输出打分，大多数任务与人工评判约 80% 一致，约 $0.01/次 |
| 数据污染 | "模型看到了测试" | 训练数据包含基准问题，在没有改善真实能力的情况下虚增分数 |
| 评估套件 | "一堆测试" | 度量特定能力的（输入、预期输出、评分器）三元组版本化集合 |
| 通过率 | "正确百分比" | 评估用例分数超过阈值的分数——比平均分数更具可操作性，因为它衡量可靠性 |
| Chatbot Arena | "模型排名网站" | LMSYS 平台，200 万+ 人类偏好投票，通过 ELO 评分产生最值得信赖的真实世界 LLM 排行榜 |

## 延伸阅读

- [Hendrycks 等人，2021 — "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) —— MMLU 论文，尽管饱和仍是最常被引用的 LLM 基准
- [Chen 等人，2021 — "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) —— OpenAI 的 HumanEval 论文，建立了代码生成评估方法论
- [Zheng 等人，2023 — "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) —— 系统分析使用 LLM 评估 LLM，包括位置偏见和冗长偏见发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) —— 众包模型比较平台，200 万+ 投票，最值得信赖的真实世界 LLM 排名
