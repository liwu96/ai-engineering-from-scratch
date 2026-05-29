# 奖励建模与RLHF

> 人类无法为"好助手回复"写奖励函数,但可比两回复选更好者。拟合奖励模型于比较,强化学习语言模型于它。Christiano 2017。InstructGPT 2022。转GPT-3成ChatGPT配方。2026大多被DPO替——但心智模型留。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程05(情感)、阶段9课程08(PPO)
**时间:** ~45分钟

## 问题背景

你训了语言模型于下一词元预测目标。它写语法英语。也撒谎、漫谈、拒拒绝。你不能用更多预训练修——网络文本是问题,非解。

你要*标量奖励*说"回复A比回复B好于指令X"。手写该奖励函数不可能。"有用"非词元闭式表达。但人类可比两输出标偏好。大规模收集便宜。

RLHF(Christiano等 2017;Ouyang等 2022)转偏好成奖励模型,然后PPO优化大语言模型于该奖励。三步:SFT → RM → PPO。是Ship ChatGPT、Claude、Gemini和2023–2025每个对齐大语言模型配方。

2026 PPO步大多被DPO(阶段10课程08)替因更便宜且对齐调近乎好。但*奖励模型*部分仍存于每个Best-of-N采样器、每个可验证奖励强化学习管道、每个用过程奖励模型的推理模型。理解RLHF理解全对齐栈。

## 概念讲解

![三阶段RLHF:SFT、成对偏好RM训、配KL惩罚PPO](../assets/rlhf.svg)

**阶段1:监督微调(SFT)。**从预训基模型起。目标行为人类写演示上微调(指令跟回复、助回复等)。结果:模型`π_SFT`*偏向好行为*但仍有无界动作空间。

**阶段2:奖励模型训。**

- 收集提示词`x`回复对`(y_+, y_-)`,人类标"y_+优于y_-。
- 训奖励模型`R_φ(x, y)`给`y_+`更高分。
- 损失:**Bradley-Terry成对逻辑**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ是sigmoid。奖励差异暗示偏好对数几率。BT自1952(Bradley-Terry)标准,现代RLHF主导选择。

- `R_φ`常从SFT模型初始化,顶上标量头。同Transformer骨干;单线性层输出奖励。

**阶段3:配KL惩罚PPO对RM。**

- 从`π_SFT`初始化可训策略`π_θ`。持冻结*参考*`π_ref = π_SFT`。
- 回复`y`末奖励:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL惩罚防`π_θ`任意漂移离`π_SFT`——是*正则化器*,非硬信任域。`β`典型`0.01`-`0.05`。
- 用此奖励跑PPO(课程08)。优势词元级轨迹算,但RM仅评完整回复。

**为何KL?**无它,PPO乐找奖励骇策略——RM仅训于分布内完成。分布外回复可比分任何人类写高。KL持`π_θ`近RM训于曲面。是RLHF单最重要旋钮。

**2026状态:**

- **DPO**(Rafailov 2023):闭式代数坍阶段2+3成偏好数据单监督损失。无RM,无PPO。对齐基准同质量分数计算。阶段10课程08覆盖。
- **GRPO**(DeepSeek 2024–2025):PPO配组相对基线替critic,奖励来自*验证器*(代码跑/数学答匹配)替人类训RM。推理模型主导。阶段9课程12覆盖。
- **过程奖励模型(PRM):**评部分解(每推理步),用于RLHF和GRPO推理变体。
- **Constitutional AI / RLAIF:**用对齐大语言模型替人类产偏好。扩偏好预算。

## 动手实践

本课用小合成"提示词"和"回复"表字符串。RM是词袋表示线性评分器。无真大语言模型——管道*形*重要,非规模。见`code/main.py`。

### Step 1:合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

真RLHF此替人类标注器。形——`(提示词,优回复,拒回复)`——同。

### Step 2:Bradley-Terry奖励模型

线性分:`R(x, y) = w · bag(y)`。最小化BT成对对数损失训:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

数百更新后,`w`给好词词元正权,坏词负。

### Step 3:RM顶PPO式策略

玩具策略从词汇产单词元。RM下评词元,算`log π_θ(token | prompt)`,加KL到参考惩罚,用裁剪PPO代理。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # theta上ppo式更新,视奖励为回报
    ...
```

### Step 4:监控KL

每更新跟踪均值`KL(π_θ || π_ref)`。如爬过`~5-10`,策略漂移远`π_SFT`——降`β`升或奖励骇起。此真RLHF顶诊断。

### Step 5:配TRL生产配方

一旦理解玩具管道,此真库用户写同循环。Hugging Face[TRL](https://huggingface.co/docs/trl)是参考实现——`RewardTrainer`阶段2和`PPOTrainer`(内置KL到参考)阶段3。

```python
# 阶段2:成对偏好奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行:{"prompt", "chosen", "rejected"}——Bradley-Terry格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 阶段3:配KL惩罚到SFT参考PPO对RM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats含:mean_kl, clip_frac, value_loss——三PPO诊断
```

库为你做三事。`adap_kl_ctrl=True`实现自适应β调度:如观察KL超`target_kl`,β加倍;低于半,β减半。参考模型约定冻结——须不意外与`policy`共享参数。价值头与策略同骨干(`AutoModelForCausalLMWithValueHead`附标量MLP头),故TRL报`policy/kl`和`value/loss`分离。

## 陷阱

- **过优化/奖励骇。**RM不完美;`π_θ`找对抗完成分高但坏。症状:奖励无限爬而人类评分平或降。修:早停,升`β`,扩RM训数据。
- **长度骇。**助回复RM常隐式奖励长度。策略学填回复。缓:长度归一化奖励,或RLAIF配长度感知RM。
- **太小RM。**RM须至少如策略大。小RM不能忠评分策略输出。
- **KL调。**太低β→漂移和奖励骇。太高β→策略几乎不变。标准技是*自适应*β定步KL。
- **偏好数据噪声。**~30%人类标签噪声或模糊。训RM于协议过滤数据或BT用温度校准。
- **异策略问题。**PPO数据首轮后略异策略。如课程08监控裁剪分。

## 实际应用

2026 RLHF分层:

| 层 | 目标 | 方法 |
|------|------|------|
| 指令跟、有用、无害 | 对齐 | DPO(阶段10课程08)优于RLHF-PPO。 |
| 推理正确(数学、代码) | 能力 | GRPO配验证器奖励(阶段9课程12)。 |
| 视界多步任务 | 智能体 | PPO / GRPO配步上过程奖励模型。 |
| 安全/拒行为 | 安全 | RLHF-PPO配分离安全RM,或Constitutional AI。 |
| 推理时Best-of-N | 快对齐 | 解码时用RM;无策略训需。 |
| 奖励蒸馏 | 推理计算 | 冻结大语言模型顶训小"奖励头"。 |

RLHF 2022–2024是*那*方法。2026,生产对齐管道DPO首,PPO仅RM密集或安全关键步。

## 产出成果

存`outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: 设计大语言模型RLHF / DPO / GRPO对齐管道,含RM、KL和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定基大语言模型、目标行为(对齐/推理/拒/智能体)和偏好或验证器预算,输出:

1. 阶段。SFT? RM? DPO? GRPO?配论证。
2. 偏好或验证器源。人类、AI反馈、规则基、单元测过、或奖励蒸馏。
3. KL策略。固定β、自适应β、或DPO(隐式KL)。
4. 诊断。均值KL、奖励稳定、过优化护(保留人类评)。
5. 安全门。红队集、拒率、安全RM分离于有用RM。

拒无KL监控发货RLHF-PPO。拒用小于目标策略RM。拒仅长度奖励。标记无盲人评保留集管道为缺过优化护。
```

## 练习题

1. **简单。**`code/main.py`中Bradley-Terry奖励模型训于500合成偏好对。保留100对测成对准确率。应超90%。
2. **中等。**玩具PPO-RLHF循环跑`β ∈ {0.0, 0.1, 1.0}`。每,绘RM分vs KL到参考更新上。哪个奖励骇?
3. **困难。**同偏好数据实现DPO(闭式偏好似然损失)并比RLHF-PPO管道用计算和达终RM分。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RLHF | "对齐强化学习" | 三阶段SFT + RM + PPO管道(Christiano 2017, Ouyang 2022)。 |
| 奖励模型(RM) | "评分网" | Bradley-Terry成对偏好拟合学习标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`;标准RM目标。 |
| KL惩罚 | "留近参考" | 奖励中`β · KL(π_θ || π_ref)`;反奖励骇正则化器。 |
| 奖励骇 | "Goodhart定律" | 策略利用RM缺陷;症状:奖励升,人类评平。 |
| RLAIF | "AI标偏好" | RLHF标签来自另大语言模型替人类。 |
| PRM | "过程奖励模型" | 评部分推理步;推理管道用。 |
| Constitutional AI | "Anthropic方法" | 显式规则导AI产偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)——RLHF开论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)——ChatGPT后配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325)——更早RLHF摘要。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290)——DPO;2026后RLHF默认。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)——RLAIF和自评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862)——HH论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl)——生产`RewardTrainer`和`PPOTrainer`。读trainer源自适应KL和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla——配图三阶段管道规范走查。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)——库;`examples/`有Llama、Mistral和Qwen端到端RLHF脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf)——奖励假说观;奖励骇思必备前置。