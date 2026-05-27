# LLM差隐私

> DP-SGD仍是标准 — 噪注入梯度更新供正式(epsilon, delta)保障。算、存、和效用开销实；参数效DP微调(LoRA + DP-SGD)是常见2025配置(ACM 2025)。两证据体张力：canary基成员推理(Duan等人, 2024)报语言模型限成功；训数据提取(Carlini等人, 2021; Nasr等人, 2025)恢复实质逐字记忆。解(arXiv:2503.06808, 2025年3月)：gap在测何 — 插入canary vs "最可提取"数据。新canary设计启loss基MIA无shadow模型并产首非LLM于实数据实DP保障DP审计。替代：PMixED (arXiv:2403.15638) — 推时私预测经下token分布专家mix；DP合成数据生成(Google Research 2024)。涌攻：差隐私逆转经LLM反馈 — 置信分漏。

**类型:** 构建
**语言:** Python(stdlib、DP-SGD噪注入和ε-δ accountant示范)
**前置要求:** 阶段01课程09(信息论)、阶段10课程01(大模型训)
**时间:** ~60分钟

## 学习目标

- 定义(epsilon, delta)差隐私和陈DP-SGD recipe。
- 解释2024-2025张力：canary MIA vs训数据提取给不同图画。
- 描述PMixED和何推时私预测是DP训替代。
- 描述差隐私逆转经LLM反馈攻。

## 问题背景

LLM记忆。Carlini等人2021示产语言模型需时复训文逐字。DP是正式防御：训使输出证对任单训例不敏感。2024-2025证据示DP-SGD需但发ε值可不配威胁模型。

## 概念讲解

### (ε, δ)差隐私

随算法M是(ε, δ)-DP若任两数据集异一例和任事件S:
P(M(D) in S) <= e^ε * P(M(D') in S) + δ。

解释：输出分布够近(ε参数化)使任单个体贡献不可可靠推断、除概率δ。

### DP-SGD

Abadi等人 2016。标准recipe:
1. 样mini-batch。
2. 算每例梯度。
3. 每例梯度剪阈值C。
4. 和剪梯度加Gaussian噪std σ * C。
5. 用噪和更新参数。

隐私成本accountant追(Moments Accountant、Rényi DP accountant)。LLM文献报ε值威胁模型、数据敏感、和效用目标异大；无全"安"默认ε。发例跨约ε ≈ 1–10于些LLM训设、但这些示 — 非荐默认。低ε一般需更噪并可增效用损。

### LoRA + DP-SGD

前沿模型全DP-SGD prohibitive。LoRA (Hu等人 2022)限梯度更新小adapter、减每例梯度存。LoRA + DP-SGD是常见2025配置。DP保障施于adapter；基模型持固。

### 2024-2025张力

两证据线:

- **Canary MIA (Duan等人 2024)。** 训数据插独canary、测成员推理攻者能否识。报语言模型限成功。示MIA难。
- **训数据提取(Carlini 2021, Nasr等人 2025)。** 提模型前缀；测是否复训逐字文。报实质记忆。示MIA易于相关义。

2025年3月解(arXiv:2503.06808)：两测不同物。MIA问"例e在D否？"于插入canary。提取问"何恢D？" "最可提取"例是隐私关；canary低估此因其非优化可提取。

新canary设计。Loss基MIA无shadow模型。首非LLM于实数据实DP保障DP审计。

### DP训替代

- **PMixED (arXiv:2403.15638)。** 推时私预测。下token分布专家mix；每专家见训数据shard；聚合加噪DP。全避DP训。
- **DP合成数据生成(Google Research 2024)。** LoRA微调带DP-SGD、样合成数据、下游分类器训于合成数据。

双侧步全DP训效用成本于不同威胁模型成本。

### 差隐私逆转经LLM反馈

涌2025攻。用DP训模型置信分oracle重识个体。即使输出不漏、置信分布可。

防御：不露置信、或露前截/量化。此是(ε, δ)-DP训外加要求。

### Phase 18何处

课程20-21是偏/公平。课程22是隐私。课程23是水印溯源。课程27覆监管数据溯源层。

## 使用

`code/main.py`玩具二元分类数据DP-SGD模拟。可扫噪乘σ和剪范C并追(ε, δ)预算和精度成本。"Canary攻"插独训例并测log-loss测能否检DP前后。

## 交付成果

本lesson产`outputs/skill-dp-audit.md`。给语言模型部署DP声明、审计：(ε, δ)值、用accountant、MIA评估协议、和是否置信露向量已评估。

## 练习题

1. 跑`code/main.py`。扫σ于{0.5, 1.0, 2.0}并报(ε, δ)-精度trade-off。识效用崩点。
2. 实canary插入和log-loss测。测DP-SGD σ = 1.0前后检率。
3. 读Nasr等人2025训数据提取。何提取成功不中度ε崩？此何意于MIA作评估？
4. 设计用PMixED (arXiv:2403.15638)部署全操于推时。PMixED址何威胁模型DP-SGD不？
5. 草DP逆转经LLM反馈攻。设计限置信漏countermeasure并估其部署成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| DP | "(ε, δ)差隐私" | 正式隐私：输出分布邻数据集改下近 |
| DP-SGD | "噪注入SGD" | 梯度剪 + Gaussian噪加；标准DP训 |
| LoRA + DP-SGD | "效私微调" | 低秩adapter DP-SGD；标准2025配置 |
| MIA | "成员推理" | 确例是否训数据攻 |
| Canary | "插入水印例" | 测DP漏独训例 |
| PMixED | "推私mix" | 推时DP经下token分布专家mix |
| DP逆转 | "置信漏攻" | 用模型置信oracle重识攻 |

## 延伸阅读

- [Abadi等人 — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — 标准DP训算法
- [Carlini等人 — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — 规范提取论文
- [Duan等人 — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — 限成功MIA
- [Kowalczyk等人 — Auditing DP for LLMs (arXiv:2503.06808, 2025年3月)](https://arxiv.org/abs/2503.06808) — 张力解
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — 推时私预测