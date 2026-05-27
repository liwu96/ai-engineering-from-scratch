# 宪法 AI 与自我改进

> RLHF 需要人类在循环中。宪法 AI 用模型自身替换其中大部分。写一份原则列表，让模型根据这些原则批判自己的输出，并在批判上训练。DeepSeek-R1 在 2025 年进一步推进：让模型生成数百万推理轨迹，用规则评分，并在结果上运行 GRPO。2026 年前沿模型中的大部分"对齐工作"是模型自己对齐自己。本课程构建这两个循环。

**类型:** 构建
**语言:** Python (stdlib + numpy)
**前置要求:** 第10阶段，第06-08课 (SFT, RLHF, DPO)
**时间:** ~45分钟

## 学习目标

- 实现宪法 AI 两阶段循环：自我批判加自我修正，然后在修正对上偏好训练
- 推导 GRPO 目标（DeepSeek-R1 的组相对策略优化）并将其与 PPO 的价值函数基线对比
- 用基于规则的结果奖励生成可验证的推理轨迹并评分，无需单独的奖励模型
- 决定自我改进何时胜过人类偏好数据，何时它会崩溃成模式搜索

## 问题背景

你在第07课和第08课构建了 RLHF 和 DPO。两者都依赖相同的昂贵输入：人类偏好对。Anthropic 的 InstructGPT 时代管道使用了大约 33,000 次比较。Llama 2 Chat 使用了超过 150 万。Claude 3 使用了更多。这些数据缓慢、昂贵，并且偏向标注者当天恰好相信的东西。

2022 年宪法 AI 论文问了一个简单问题。如果模型自己生成偏好标签呢？给它一份书面原则列表——"宪法"——让它批判自己的响应。批判成为训练信号。

2024 年，DeepSeek 进一步推进了这个想法。他们表明，对于任何有可验证结果的任务（数学有已知答案、代码要么通过测试要么失败、游戏要么赢要么输），你可以完全跳过批判者。生成许多候选解决方案。用确定性规则给每个评分。在奖励上运行策略梯度算法。DeepSeek-R1 几乎没用人类偏好数据就这样训练，并匹配了 o1 级别的推理性能。

这两个循环——宪法 AI 用于主观行为，基于规则的 RL 用于可验证行为——是 2026 年的主导对齐配方。过去用于 RLHF 的人类偏好预算现在支付一个更小的步骤：选择宪法和选择奖励规则。

## 概念讲解

### 宪法 AI 循环

Bai 等人（2022）将管道结构化为两个阶段。

**阶段1：来自 AI 反馈的监督学习（SL-CAI）。** 从一个有帮助但可能有害的 SFT 模型开始。用潜在有害请求提示它。对于每个响应，让*同一模型*根据宪法原则批判其响应，然后修正。在修正后的响应上微调。数据集是（提示，修正响应）对。

**阶段2：来自 AI 反馈的强化学习（RLAIF）。** 采样响应成对。让模型选择哪个更符合宪法。成对偏好训练奖励模型。然后在模型上运行 PPO 或 DPO 使用该奖励。与 RLHF 的关键区别：偏好来自模型，而非人类。

宪法是杠杆。Anthropic 的原始版本有 16 条原则（后来扩展）。一条原则读起来像"请选择对来自各种文化背景的任何人都不太可能令人反感的响应。"你为每个步骤选择原则，有时随机，有时基于提示类别。

### 宪法实际做什么

宪法将对齐合同从*数据*转移到*文本*。在 RLHF 下改变行为意味着重新标注数千对。在 CAI 下改变行为意味着编辑一段文字。这是主要的实际胜利。

它有代价。模型的自我判断只与其起始校准一样好。如果 SFT 模型有盲点——例如，它无法识别操纵性措辞——批判步骤继承这些盲点。CAI 压缩对齐循环但无法将信号放大到基础模型的天花板之上。这就是每个生产 CAI 管道仍然使用一些人类偏好数据的原因，通常是纯 RLHF 的 5-10%。

### GRPO：组相对策略优化

DeepSeek 在 DeepSeekMath 论文（2024）中引入了 GRPO，并将其用作 DeepSeek-R1（2025）的支柱。GRPO 是 PPO 的一种变体，去除了价值函数。

回顾 PPO 的目标（来自第07课）：

```
L_PPO = E[min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)]
```

其中 `A` 是优势，通常使用学习价值网络 `V(s)` 用 GAE 估计。价值网络是与策略相同大小的第二个模型。它使内存翻倍并引入自己的训练循环。

GRPO 扔掉价值函数。对于每个提示，它采样一组 G 个响应（通常为 G=16 或 64）。计算每个响应的奖励，然后在组内归一化：

```
A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

优势是响应奖励相对于其兄弟姐妹的 z 分数。没有价值函数。该组充当自己的基线。

```
L_GRPO = E[min(r(theta) * A_group, clip(r(theta), 1-eps, 1+eps) * A_group)] - beta * KL(pi || pi_ref)
```

与参考模型的 KL 惩罚仍然存在，与 PPO 相同。裁剪比例仍然存在。消失的是单独的批判者。

### 为什么 GRPO 对推理重要

对于推理任务，奖励通常是稀疏且二元的：最终答案是对还是错。在稀疏二元奖励上训练的价值函数是浪费——它无法学习有用的中间估计，因为几乎每个状态在最终步骤之前都有相同的预期回报。GRPO 的组归一化给你一个立即的相对信号：在同一数学问题的 16 次尝试中，哪些尝试高于平均水平？

这正是你从基于规则的奖励中得到的信号形状：

- **数学：** sympy 或符号检查器决定是否匹配最终答案。
- **代码：** 测试套件决定通过/失败。
- **格式：** 正则表达式决定答案是否在必需的 XML 标签中。
- **多步证明：** 证明助手（Lean、Coq）决定有效性。

DeepSeek-R1-Zero 仅使用两个奖励进行训练：数学基准的准确性和格式合规性（答案在 `<answer>` 标签内）。没有人类偏好。没有批判模型。DeepSeek 论文描述的"顿悟时刻"——模型自发学习自我检查和回溯——仅从 GRPO 对稀疏规则奖励的训练中涌现。

### 过程奖励模型 vs 结果奖励模型

你仍然有一个设计选择：奖励最终答案（结果奖励模型，ORM）或奖励每个中间步骤（过程奖励模型，PRM）。

| 轴 | ORM | PRM |
|------|-----|-----|
| 每轨迹信号 | 1 个数字 | N 个数字（每步一个） |
| 监督来源 | 最终答案检查 | 步骤级标签或自我评判 |
| 训练成本 | 便宜 | 昂贵 |
| 信用分配 | 稀疏、嘈杂 | 密集、针对性 |
| 奖励黑客风险 | 较低 | 较高（模型优化 PRM 伪影） |
| 被谁使用 | DeepSeek-R1、R1-Zero | OpenAI o1（据称）、Math-Shepherd |

2024-2025 年的共识是 ORM 加 GRPO 比 PRM 更好地扩展。PRM 每 Token 样本效率更高，但需要昂贵的步骤标记数据，并倾向于崩溃成捷径行为（编写对 PRM 看起来好但不推进证明的步骤）。对于大多数团队，ORM + GRPO 是首先要尝试的。

### 自我改进：反馈倍增器

一旦你有了双循环模式（批判/修正和组相对 RL 与规则奖励），你可以将它们链接。

1. 从 SFT 模型开始。
2. 每个提示生成多个候选响应。
3. 用基于规则的奖励（用于可验证任务）或宪法批判者（用于主观任务）给它们评分。
4. 将顶级候选保留为新的 SFT 数据或偏好对。
5. 微调。用改进的模型返回步骤2。

DeepSeek 在 R1-Zero 之后应用时称之为"拒绝采样微调"。Anthropic 称之为"宪法 AI 蒸馏"的更早版本。模式是：每次迭代放大模型中已有的信号。它不添加新信号。如果模型根本无法解决问题类别 X，再多的自我改进也无法创造该能力。

危险是模式崩溃。自我生成的数据总是比训练语料库更窄的分布。经过 3-5 轮自我蒸馏后，模型通常在创造性任务上失去多样性，变得过度自信，并表现出特征性的"AI 腔"（重复措辞、公式化结构）。生产管道混合自我生成数据和一小部分新鲜人类数据以保持分布诚实。

### 何时使用什么

- **纯 CAI：** 主观行为（语气、安全性、拒绝风格）。你有明确定义的宪法。你没有干净的可验证结果。
- **GRPO + ORM：** 可验证任务（数学、代码、结构化提取）。你可以廉价检查正确性。奖励是稀疏且二元的。
- **DPO 在自我生成对上：** 混合。使用宪法产生偏好对，然后用 DPO（第08课）而非 PPO/GRPO 训练。
- **完整 RLHF：** 当你需要多目标权衡，而规则或短宪法无法表达时，仍然适用。

大多数 2026 年前沿管道运行全部四个。CAI 用于安全层。GRPO 用于推理后训练。DPO 用于偏好打磨。小型 RLHF 用于抵抗其他方法的残余行为。

## 动手实践

代码在纯 Python + numpy 中实现三个东西。一个宪法 AI 自我批判循环。基于规则的奖励检查器用于简单算术。一个最小 GRPO 训练器，在第04课的微型语言模型上运行。

### 步骤1：宪法

原则列表。在生产中，每行会更丰富并按类别标记。对于本课程，保持简短。

```python
CONSTITUTION = [
    "响应必须直接回答问题，不回避。",
    "响应不能包含不必要的填充或冗余。",
    "如果问题有单一数字答案，请直接陈述该数字。",
    "响应不能拒绝合理、良性的请求。",
]
```

### 步骤2：自我批判与修正

在真实系统中，模型自己批判。在本课程中，我们用手写评分标准模拟批判者，以便管道在没有 LLM 调用的情况下运行。

```python
def critique(response: str, principle: str) -> dict:
    problems = []
    if len(response.split()) > 40 and "plainly" in principle:
        problems.append("答案埋没在额外文字中")
    if response.strip().lower().startswith(("我不能", "我无法", "作为 AI")):
        problems.append("无正当理由的拒绝")
    if response.count(",") > 4:
        problems.append("过多回避")
    return {"principle": principle, "problems": problems}

def revise(response: str, critique_result: dict) -> str:
    if "答案埋没" in " ".join(critique_result["problems"]):
        return response.split(".")[-2].strip() + "."
    if "无正当理由的拒绝" in " ".join(critique_result["problems"]):
        return "这是答案：" + response.split(":")[-1].strip()
    return response
```

修正函数是一个占位符。使用真实 LLM 时，它将是第二个提示："给定批判，重写响应。"

### 步骤3：基于规则的奖励

对于可验证任务，完全替换批判者。这个检查器评分算术答案。

```python
import re

def reward_math(prompt: str, response: str) -> float:
    try:
        expected = eval(prompt.replace("What is ", "").replace("?", "").strip())
    except Exception:
        return 0.0
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    return 1.0 if int(numbers[-1]) == expected else 0.0

def reward_format(response: str) -> float:
    return 1.0 if re.search(r"<answer>.*</answer>", response) else 0.0
```

两个确定性规则。没有训练数据。没有人类标签。组合奖励是 `reward_math + 0.1 * reward_format`，惩罚缺失格式但不淹没正确性。

### 步骤4：组相对优势

给定同一提示的一组响应的奖励列表，计算 z 分数：

```python
import numpy as np

def group_relative_advantage(rewards: list[float]) -> np.ndarray:
    r = np.array(rewards, dtype=float)
    if r.std() < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std() + 1e-8)
```

如果组中每个样本都有相同的奖励，优势为零，没有梯度信号流动。这是一个特性。它告诉你提示要么被当前策略轻易解决，要么不可能解决，该步骤应该跳过。

### 步骤5：GRPO 更新

一步，符号梯度。在生产中这将是 torch autograd 传递。这里我们直接展示更新规则。

```python
def grpo_step(policy_logprobs: np.ndarray, ref_logprobs: np.ndarray,
              advantages: np.ndarray, beta: float = 0.01, clip_eps: float = 0.2) -> dict:
    ratios = np.exp(policy_logprobs - ref_logprobs)
    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -np.minimum(unclipped, clipped).mean()
    kl = (ref_logprobs - policy_logprobs).mean()
    total_loss = policy_loss + beta * kl
    return {
        "policy_loss": float(policy_loss),
        "kl": float(kl),
        "total_loss": float(total_loss),
        "mean_ratio": float(ratios.mean()),
    }
```

这是 PPO 的裁剪代理，有一个变化：优势来自组相对 z 分数，而非价值函数。没有 V(s) 要训练。没有 GAE。该组是基线。

### 步骤6：自我改进轮次

将各部分联系在一起。采样一组，用规则给每个响应评分，计算优势，报告你将输入真实优化器的指标。

```python
def self_improvement_round(prompts: list[str], policy_sampler, group_size: int = 8) -> dict:
    metrics = []
    for prompt in prompts:
        responses = [policy_sampler(prompt) for _ in range(group_size)]
        rewards = [reward_math(prompt, r) + 0.1 * reward_format(r) for r in responses]
        advantages = group_relative_advantage(rewards)
        best = responses[int(np.argmax(rewards))]
        metrics.append({
            "prompt": prompt,
            "mean_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_response": best,
            "advantages": advantages.tolist(),
        })
    return {"per_prompt": metrics,
            "overall_mean": float(np.mean([m["mean_reward"] for m in metrics]))}
```

## 使用实践

运行 `code/main.py` 从头到尾运行两个循环。CAI 循环产生一组（初始、修正）对，你可以在其上微调。GRPO 循环为算术问题产生每提示奖励统计，展示组相对优势如何让弱采样器在没有价值函数或人类标签的情况下改进。

数字不是重点。在真实运行中，使用训练好的模型，奖励均值应该跨轮次攀升，奖励标准差应该保持为正（如果它崩溃到零，策略已经模式崩溃，你应该停止），与参考的 KL 应该缓慢增长。这三条曲线——奖励均值上升、标准差稳定、KL 有界——是 GRPO 或 CAI 管道的生产健康检查。

## 产出成果

本课程产出 `outputs/skill-self-improvement-auditor.md`。向它提供提议的自我改进管道，它强制执行不可协商的门控：实际可验证的奖励规则、相对于参考的 KL 预算、多样性下限和人类数据配额。它拒绝批准声称"纯自我改进"而没有任何外部基础的循环。

## 练习题

1. 用 LLM 调用替换步骤2中的手写批判者。使用任何本地聊天模型。测量批判和修正实际改进响应的频率与保持不变的情况。

2. 添加关于事实性的第三条宪法原则。在需要事实声明的提示（首都、日期）上运行管道，测量有多少修正移除事实错误与引入新错误。

3. 在 CAI 阶段2产生的偏好对上实现 DPO。取 20 个提示，每个生成两个响应，让批判者每对选一个赢家，然后在第08课的 DPO 损失上运行。与相同数据上的 GRPO 路径比较。

4. 向 GRPO 目标添加熵正则化。项 `-alpha * entropy(policy)` 且 alpha=0.01 鼓励多样化采样。测量它是否延迟跨 5 轮自我改进的模式崩溃。

5. 为两步算术问题构建过程奖励评分器。给定"(3+4)*5 是多少？"，模型必须展示中间步骤 3+4=7。将中间步骤与最终答案分开评分，并比较 PRM 加权 GRPO 与纯 ORM 加权 GRPO 在 10 轮上的表现。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 宪法 AI | "模型自己对齐自己" | 两阶段管道（自我批判 + RLAIF），用模型自我评判替代大部分人类偏好标签 |
| RLAIF | "无人类的 RLHF" | 来自 AI 反馈的强化学习——在模型自身生成的偏好上运行 PPO 或 DPO |
| GRPO | "无价值函数的 PPO" | 组相对策略优化——每提示采样 G 个响应，使用 z 分数组奖励作为优势 |
| ORM | "奖励答案" | 结果奖励模型——仅在最终答案上的单一标量奖励 |
| PRM | "奖励每一步" | 过程奖励模型——在每个中间推理步骤上的奖励，通常从步骤标记数据训练 |
| 基于规则的奖励 | "确定性评分器" | 验证器（正则表达式、sympy、测试套件），无需学习模型返回二元或数字分数 |
| 拒绝采样 FT | "保留赢家，重新训练" | 采样许多响应，筛选到最高奖励的，添加到 SFT 数据，重新训练 |
| 模式崩溃 | "模型停止多样化" | 后训练策略集中在响应空间的狭窄区域；在一组上测量为下降的奖励标准差 |
| KL 预算 | "你能漂移多远" | 优化器在训练停止前允许从参考模型累积的总 KL 散度 |
| R1 时刻 | "模型学会回溯" | DeepSeek 报告的行为，其中仅对结果奖励训练的策略在其思维链中自发发展自我检查和回溯 |

## 延伸阅读

- [Bai 等人，2022 — "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) —— Anthropic 的原始 CAI 论文，包含 SL-CAI + RLAIF 两阶段管道
- [Shao 等人，2024 — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300) —— 引入 GRPO
- [DeepSeek-AI，2025 — "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"](https://arxiv.org/abs/2501.12948) —— R1 和 R1-Zero，大规模 GRPO + 规则奖励
- [Lightman 等人，2023 — "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050) —— OpenAI 的 PRM800K 和过程奖励模型案例
- [Wang 等人，2024 — "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"](https://arxiv.org/abs/2312.08935) —— 通过蒙特卡洛 rollout 自动标记 PRM
- [Huang 等人，2024 — "Large Language Models Cannot Self-Correct Reasoning Yet"](https://arxiv.org/abs/2310.01798) —— 关于没有外部基础的自我改进的怀疑论观点
