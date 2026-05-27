# 可Scale监督和弱到强泛化

> Burns等人(OpenAI Superalignment, "Weak-to-Strong Generalization", 2023)提出超对齐问题代理：用弱模型产标签微调强模型。若强模型从不完美弱监督正确泛化、当前人尺对齐方法可延到超人系统。可Scale监督和W2SG互补。可Scale监督(debate、递归奖励建模、任务分解)增监督者有效能力以跟上监督下模型。W2SG确强模型从不完美监督正确泛化。Debate Helps W2SG(arXiv:2501.13124, 2025年1月)合二者。

**类型:** 学习
**语言:** Python(stdlib、W2SG gap模拟器)
**前置要求:** 阶段18课程01(instruction-following)、阶段18课程10(AI Control)、阶段09(RL基础)
**时间:** ~60分钟

## 学习目标

- 定义可Scale监督和弱到强泛化并解释何互补。
- 描述Burns等人2023实验设：用GPT-2标签微调GPT-4。
- 解释性能差恢复(PGR)度量和何测。
- 陈述三主可Scale监督机制(debate、递归奖励建模、任务分解)和每优势。

## 问题背景

阶段18每对齐技术假设监督者可评估模型行为。当模型超人、监督者是弱环。超对齐问：弱监督者能否可靠产强、对齐模型？

Burns等人化为操作实证设：强用弱监督、测强模型能力少弱监督下存多少。此非超对齐解——是代理让测进步。

## 概念讲解

### W2SG: Burns等人设

- 弱模型: GPT-2类。
- 强模型: GPT-4类。
- 目标: 任务上强GPT-4天花板。

流程:
1. 获弱模型任务零样本预测。
2. 用弱标签数据微调强模型。
3. 测强微调模型精度。

基线: 弱模型自精度、强模型金标签监督天花板。

差度量: 性能差恢复(PGR) = (微调 - 弱) / (天花板 - 弱)。PGR 1.0意弱监督全闭差；PGR 0意弱监督无助。

### Burns等人实证发现

跨NLP、chess-puzzle、奖励建模任务、PGR持续正：强模型泛化超弱监督错。任务特定PGR从~20%到~80%。强模型从不完美标签"理解"意任务、用预训prior填。

Burns等人注限:
- 此弱强差是能力、非对齐。弱模型对齐；真超人模型可被人对齐但能力差大多。
- 强模型泛化可是任务prior而非恢复真意图。
- Elicit强模型latent知识是实问题；PGR测特定操作化。

### 可Scale监督: 三机制

- **Debate (Irving等人 2018)。** 两U实例对辩；弱法官决。假设：找说服真答比找说服假答易。近2024-2025工(Khan等人、Michael等人)示debate时有助时不助、助否依赖任务结构。
- **递归奖励建模 (Leike等人 2018)。** U帮人训U+1奖励模型。监督者有效能力随模型长。
- **任务分解 (Christiano, Shlegeris, Amodei 2018)。** 解硬任务为子任务人可查、递归。假设可分解性。

每机制假设任务结构或中间组件对齐某事。

### 为何可Scale监督和W2SG互补

可Scale监督增监督者有效信号质量。
W2SG闭监督者能供不完美信号差。

Lang等人 — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)合: debate协议供好弱标签、强模型训于那标签。报NLP任务PGR增益。

### 组织戏剧

OpenAI Superalignment队2024年5月Jan Leike离Anthropic后解散。议程(可Scale监督、W2SG、自对齐研)续于Anthropic和学术实验室 — MATS(课程28)、Redwood(课程10)、Apollo(课程8)、METR(课程28)。组织结构变；研问不变。

### Phase 18何处

课程6-10描述威胁和U不可信假设下防御范式。课程11是攻范式：使监督者强验U对齐。课程12-16转对抗评估实用工具。

## 使用

`code/main.py`合成任务W2SG微调模拟。弱标签器70%精度有结构错；强模型金标签95%天花板。用弱标签微调强模型、测PGR、比强金和弱独。

## 交付成果

本lesson产`outputs/skill-w2sg-pgr.md`。给监督设描述、识弱监督者、强模型、监督质量、算(或求)PGR。标声明是"弱可监督强"或"弱+监督机制可监督强"。

## 练习题

1. 跑`code/main.py`。报弱精度0.60、0.70、0.80下PGR。解释PGR曲线形状。
2. 修改弱标签器有结构错(如特定输入类总错)。PGR增、减、或持？解释。
3. 读Burns等人2023第4.3节(NLP任务)。复现"置信辅助loss"直觉：强模型比弱标签更置信时、谁胜？
4. 设计合debate和任务分解软件工程任务可Scale监督协议。名每组件一失败模式并解释合址或不址每。
5. 述何会证伪"弱到强泛化是超对齐可行路"声明。具体需何实证签名。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 可Scale监督 | "使监督者更强" | 增监督者评更力模型能力的机制 |
| W2SG | "弱监督强" | 弱标签微调强模型并测能力恢复 |
| PGR | "性能差恢复" | (微调 - 弱) / (天花板 - 弱)；1.0 = 全闭、0 = 无助 |
| Debate | "两U实例辩" | 可Scale监督机制：弱法官选两U捍卫者 |
| RRM | "递归奖励建模" | U帮训U+1奖励模型；监督能力跟U |
| 任务分解 | "子任务人查" | 解硬任务为人可验子任务、递归 |
| Superalignment | "对齐超人AI" | 关心人不能直评模型对齐研议程 |

## 延伸阅读

- [Burns等人 — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG论文
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — debate机制
- [Leike等人 — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 递归奖励建模
- [Khan等人 — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024强辩者debate实证研
- [Lang等人 — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 debate + W2SG合