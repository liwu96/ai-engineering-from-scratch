# 直接偏好优化家族

> Rafailov等(2023)表明RLHF最优解有偏好数据的闭式解,你可以跳过显式奖励模型直接优化策略。那个洞察产生一个家族——IPO、KTO、SimPO、ORPO、BPO——每个修复DPO的一个失败模式。2026年,直接对齐算法比PPO发布更多前沿后训练运行。但课程2的过优化曲线仍适用:DAA不逃避Goodhart,它们只是移动咬的地方。

**类型:** 学习
**语言:** Python (stdlib、六变偏好loss比器)
**前置要求:** 阶段18课程01(InstructGPT)、阶段18课程02(奖励黑客)、阶段10课程08(DPO基础)
**时间:** ~75分钟

## 学习目标

- 从RLHF-with-KL最优导出DPO闭式。
- 陈述IPO、KTO、SimPO、ORPO、BPO每个修复DPO失败模式。
- 区分"隐奖励gap"和"偏好强度"并解释IPO的identity mapping重要。
- 解释为何Rafailov等(NeurIPS 2024)证明DAA尽管无显式RM仍过优化。

## 问题背景

RLHF目标(课程1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有已知最优:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

所以奖励隐式由最优策略比参考定义:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

代入Bradley-Terry偏好似然和partition函数`Z(x)`取消因为它只依赖`x`。剩余的是仅策略参数loss——无奖励模型需。这是DPO。

折衷:推导假设最优可达、偏好数据分布内、参考策略真模态锚。这些无确切持有。每个家族成员修复不同违假设。

## 概念讲解

### DPO (Rafailov等,2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi_y_l | x) / pi_ref(y_l | x))
)
```

何错:

- 隐奖励gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。小偏好可产生任意大gap。
- Loss驱chosen和rejected log-prob反方向。可以chosen绝对log-prob下降只要rejected落更快。这是Degraded Chosen Response现象。
- 分布外偏好(rare rare pair vs rare rare pair)产生任意隐奖励。

### IPO (Azar等,2024)

Identity Preference Optimization将log-sigmoid替换为偏好概率的identity mapping。Loss变成有界目标的平方误差:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Margin由`1/(2 beta)`有界。偏好强度和隐奖励gap比例。无blow-up。

### KTO (Ethayarajh等,2024)

Kahneman-Tversky Optimization完全放弃成对结构。给定单标注输出和二元"desirable"或"undesirable"信号,它映射到前景理论效用:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

配得失不同权重(损失厌恶)。益:可用非配数据,远更多。

### SimPO (Meng等,2024)

Simple Preference Optimization使训练信号与生成对齐。完全移除参考策略并用长度归一化log-likelihood:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

配margin `gamma`稳定。长度归一化移除利用DPO长度偏失败模式的激励(更长`y_w`结构上给更大log-prob gap)。

### ORPO (Hong等,2024)

Odds-Ratio Preference Optimization向标准SFT负对数似然添加偏好项:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无参考策略——SFT项是正则器。从基模型到对齐模型单阶段训练。无分离SFT checkpoint。

### BPO (ICLR 2026提交, OpenReview id=b97EwMUWu7)

识别Degraded Chosen Responses问题:DPO保排名`y_w > y_l`但`y_w`绝对log-prob可下降。BPO添加单行校正惩罚chosen响应向下移动。报告Llama-3.1-8B-Instruct数学推理DPO上+10.1%准确率。

### 通用结果:DAA仍过优化

Rafailov等"直对齐算法奖励模型过优化Scaling Laws"(NeurIPS 2024)在多数据集KL预算上用DPO、IPO、SLiC训策略。gold奖励vs KL曲线有相同Gao等峰塌形状。隐奖励在训练时查询分布外样本;KL正则不稳此。

DAA不逃避Goodhart。它们改变咬的面从"奖励模型过优化"到"参考策略比过优化"。通用修复——好数据、集成、早停——双适用。

### 选择(2026)

- 若有大量配偏好数据:DPO配保守beta,SimPO若长度偏显。
- 若有非配二元反馈:KTO。
- 若要从基模型单阶段管道:ORPO。
- 若见DPO日志chosen降:BPO。
- 若偏好强度广变DPO饱和:IPO。

每实验室在全五上跑battery每任务选赢家。无理由数学推理和安全同优。

## 使用

`code/main.py`在玩具偏好数据集上比六loss(DPO、IPO、KTO、SimPO、ORPO、BPO)真实偏好强度变化。每loss配同500对样本小softmax策略优。绘终win rate、chosen-log-prob drift、隐奖励spread每方法。

## 产出成果

本课程产`outputs/skill-preference-loss-selector.md`。给定数据集统计(配vs非配、变vs均匀偏好强度、长度分布)和目标(单阶段或SFT-then-preference),荐偏好loss并报告失败模式保护。

## 练习题

1. 跑`code/main.py`。报告DPO和BPO终chosen-log-prob落。BPO应保chosen绝对概率高——验证。
2. 修改偏好数据全对等强度。六方法最鲁棒?何退化?解释IPO优势。
3. 让Rejected响应平均chosen 2x长。不改他、数值示DPO长度exploitation和SimPO修。
4. Rafailov等(NeurIPS 2024)声称DAA过优化。重现单点版本:绘chosen-minus-rejected KL散度观DPO大beta过优化。
5. 读BPO论文摘要(OpenReview b97EwMUWu7)。写下BPO加DPO单行校正。确认`code/main.py`实现。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| DPO | "无奖励模型RLHF" | RLHF最优闭式导loss;仅策略参数 |
| 隐奖励 | "log比" | `beta * log(pi(y|x) / pi_ref(y|x))`——DPO隐含奖励 |
| IPO | "有界DPO" | log-sigmoid换identity;隐奖励gap `1/(2 beta)`限 |
| KTO | "非配DPO" | 单标注前景理论效用配损失厌恶 |
| SimPO | "无参考DPO" | 长度归一log-likelihood + margin;无参考策略 |
| ORPO | "单阶段DPO" | NLL + odds-ratio偏好项;基模型一通过训 |
| BPO | "chosen保DPO" | DPO加chosen响应绝对log-prob降罚 |
| Degraded Chosen | "chosen降" | DPO降chosen log-prob只要rejected落更快 |
| DAA | "直对齐算法" | 任跳显RM偏好loss方法 |

## 延伸阅读

- [Rafailov等 — 直偏好优化(NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar等 — 从人类偏好学习通用理论范式(AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036)——IPO
- [Ethayarajh等 — KTO:模型对齐作为前景理论优化(arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO(NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO(EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — 行为保优化(ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov等 — DAA奖励模型过优化Scaling Laws(NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)