# 缩放定律

> 2020年Kaplan论文说:更大模型,更低损失。2022年Hoffmann论文说:你训练不足。计算分两个桶——参数和词元——分配比例并不显而易见。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段7课程07(GPT)
**时间:** ~45分钟

## 问题背景

当你有C FLOPs训练算力并想要最佳模型,面临两个旋钮:

1. **多少参数(N)?** 更大模型,更高容量。
2. **多少训练词元(D)?** 更多数据,更好利用容量。

FLOPs约按`6 × N × D`缩放。可推N升D降,或D升N降。哪个更好?

2022前,答案是"猛推N"。GPT-3(2020)175B参数训于~300B词元。约1.7词元每参数比例。Kaplan缩放定律支持此。

Hoffmann等(2022),训小模型家族叫Chinchilla,发现不同:最优比例近**20词元每参数**。GPT-3训练不足10×。Chinchilla(70B参数,1.4T词元)每基准胜GPT-3(175B,300B词元)同时推理成本低2.5×。

2026是Chinchilla世界——带一重要转折。Llama 3 8B训于15万亿词元,比例1,875词元每参数。超出Chinchilla最优94倍。推理成本比训练成本对大规模使用模型更重要,故过训(超Chinchilla)换取更小部署足迹是2026默认。

## 概念讲解

![Chinchilla曲线:不同N/D比例下损失vs计算](../assets/scaling-laws.svg)

### Hoffmann定律

从Chinchilla论文,损失遵循:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数(非嵌入)。
- `D` = 训练词元。
- `α ≈ 0.34`, `β ≈ 0.28`(大致对称)。
- `E ≈ 1.69`,不可约损失上限。
- `A ≈ 406`, `B ≈ 411`。

两项缩放时相互权衡。固定计算(C = 6ND)下对N求导并解:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优:20词元每参数。

### 为何仍要过训

Chinchilla最优最小化每训练FLOP的训练损失。但训练成本付一次;推理成本永久。

对每月服务万亿词元聊天机器人,推理主导总成本。Llama方法:训更小,更长。8B配15T词元深度推理优化:

- 适消费级GPU。
- 延迟是70B Chinchilla最优的零头。
- 质量对多数任务足够近。

DeepMind 2024论文("过训是新最优")形式化此。对推理主导工作负载,正确比例近100–500词元每参数,取决于服务量。

### 涌现vs平滑

声称:某些能力(算术、多步推理、思维链跟随)在某规模"涌现"突现。

Schaeffer等(2023)论证此是测量假象:涌现度量用不连续评分(精确匹配、阈值准确率)隐藏底层logits平滑改进。连续度量(交叉熵)示平滑曲线。

2026共识:通过连续损失预测可靠。基准跳常是评分器假象。按连续度量规划预算。

### 2026图景

缩放定律仍工作,但:

| 因素 | 如何变化 |
|------|----------|
| 数据质量 | 筛选"好"词元(Phi风格)曲线移>2×有效计算 |
| MoE | 总参数与激活FLOPs解耦;按激活FLOPs缩放定律 |
| 后训练 | 某些能力(指令跟随、代码)SFT+RLHF移比预训更多 |
| 多模态 | 图像+文本词元一起缩放;每模态分离曲线 |
| 合成数据 | 模型生成训练数据;有效计算可复利 |

Muon优化器(Kimi Moonlight, 2024)示配匹数据AdamW~2×有效计算增益。部分2026训练运行默认使用Muon。改变的是缩放定律中的绝对常数，而非其形状。

## 动手实践

见`code/main.py`。实现Chinchilla损失方程并在数计算预算求解计算最优`(N, D)`。

### Step 1: Chinchilla损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

绘`L`作`(N, D)`等高线固定`C = 6ND`。找最小。

### Step 2: 计算最优前沿

对计算预算从`1e17`到`1e25` FLOPs,找最小化损失`(N, D)`受`6ND = C`。验证比例`D/N ≈ 20`。

### Step 3: 过训成本

计算为训更小10×模型付额外损失(最优N的1/10,最优D的10×)。报告换推理FLOP节省(正比于N)。

### Step 4: 比真实模型

代入GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3(激活参数)已知`(N, D)`对,比预测vs报告损失。

## 实际应用

你不太可能自己训前沿模型。但缩放定律告诉你:

1. **微调有足够数据否。** 若任务特定数据低于基模型每参数20词元,期望饱和于某损失地板。
2. **选更大基模型否。** 若花全部预算于推理,选更小、更长训模型。
3. **回报何处递减。** 超1000× Chinchilla最优,log-损失变噪声。

**2026研究轨迹:**

- **数据受限区。** 网有有限高质量词元(~5–10万亿英语过滤后)。前沿预训逼近此天花板。合成数据、多语言、多模态、RLHF缩放微调是下一杠杆。
- **计算乘数技巧。** Muon优化器、MoE、更好数据策管——每项都改变绝对常数，而非渐近线。
- **RL缩放定律。** 开问题。早证据示RL样本幂律但指数与预训很不同。

## 产出成果

见`outputs/skill-training-budget-estimator.md`。技能给定计算预算、部署约束和目标损失为新训练运行选`(N, D, hours, GPU)`。

## 练习题

1. **简单。**运行`code/main.py`。打印计算预算`1e20`, `1e22`, `1e24`的Chinchilla最优`(N, D)`。比真实模型表。
2. **中等。**实现Hoffmann损失作为计算函数曲线。绘计算最优前沿损失vs `log10(C)`。识别定律预测何时需`>10^28` FLOPs下0.1交叉熵降。
3. **困难。**在同数据集训5小模型(100K到10M参数)拟合自己缩放定律。估`α`和`E`。你指数与发表匹配如何?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 参数(N) | "模型大小" | 非嵌入权重计数;决定容量。 |
| 词元(D) | "训练数据" | 训练期间见到的词元数；决定参数的利用程度。 |
| 计算(C) | "花费FLOPs" | 标准transformer约`6 × N × D`。 |
| Chinchilla最优 | "D/N ≈ 20" | 每预训FLOP最小化损失比例。 |
| 过训 | "超Chinchilla" | 花额外训练FLOPs省推理FLOPs;D/N >> 20。 |
| 不可约损失 | "地板" | 缩放定律`E`项;数据自身熵。 |
| 涌现能力 | "规模突跳" | 常是评分器假象;连续损失平滑。 |
| 有效计算 | "训练效率乘数" | 更好数据/优化器/架构乘FLOP效力。 |

## 延伸阅读

- [Kaplan等(2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)——首缩放定律论文;训练不足。
- [Hoffmann等(2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)——Chinchilla。
- [Schaeffer等(2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)——涌现作为测量假象。
- [Sardana, Frankle(2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448)——为何Llama过训对其工作负载正确。
- [Jordan等(2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)——2×计算乘数。