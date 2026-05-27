# LLM偏和代表害

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770)。基础2024综述分代表害(刻板、抹除)和分配害(不平等资源分配)并分评估度量为嵌入基、概率基、或生成文基。2024-2025实证：An等人(PNAS Nexus, 2025年3月)测跨GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B于自简历评估20入门职交叉性别种族偏。WinoIdentity (COLM 2025, arXiv:2508.07111)引不确定基交叉身份公平评估。Yu & Ananiadou 2025识MLP层性别神经元；Ahsan & Wallace 2025用SAEs示临床种族偏；Zhou等人2024 (UniBias)操注意力头去偏。元评(arXiv:2508.11067)：10年文献过分聚焦二元性别偏。

**类型:** 构建
**语言:** Python(stdlib、玩具嵌入基偏probe)
**前置要求:** 阶段05(词嵌入)、阶段18课程01(instruction following)
**时间:** ~60分钟

## 学习目标

- 定义代表vs分配害并给LLM部署每例。
- 名Gallegos等人2024三评估度量类别并述每度量。
- 描述交叉性和为何WinoIdentity不确定基公平测址单轴偏评估gap。
- 描述两机制可解释偏方法(性别神经元、SAE特征、注意力头操)。

## 问题背景

前课程覆意害(jailbreak、scheming)和安全治。偏是无意害 — 从训数据分布、提示框架、积设计选择涌现。测减是不同于对抗鲁棒性方法论挑战。

## 概念讲解

### 代表vs分配

- **代表害。** 刻板、抹除、贬低描绘。LLM绘护士为独女产代表害。
- **分配害。** 不平等物质结果。LLM系统评黑申请简历低产分配害。

此非同。模型可"代表无偏"(产多样描绘)同时"分配偏"(产不平等荐)。评估需测双。

### 三评估度量类别(Gallegos等人 2024)

- **嵌入基。** WEAT风格测于pre-RLHF嵌入。测身份词和属性词统计关联。限：测表示、非行为。
- **概率基。** 刻板确vs刻板违补全log-likelihood。Decoder侧测。捕些行为偏。
- **生成文基。** 生成文下游任务测。简历评分、荐写、对话。最生态有效；最难复现。

### 交叉性

偏评估于"性别"漏仅发于(性别、种族)对偏。An等人2025现GPT-4o简历评分罚黑女多于黑男和多于白女分。单轴评估不可捕。

WinoIdentity (COLM 2025)引不确定基交叉公平。测模型不确定于结果是否跨交叉身份tuple不同 — 非仅点预测。此捕模型跨组等错但对些更不确定、产不同下游分配行为例。

### 机制方法

2024-2025可解释性工开偏于机制介入:

- **性别神经元(Yu & Ananiadou 2025)。** 特MLP神经元性别特定行为相关。ablate此神经元减性别差度量限能力成本。
- **临床种族偏经SAEs (Ahsan & Wallace 2025)。** 稀自编码器特征解内表示为可解释维度；种族相关特征可识抑。
- **UniBias (Zhou等人 2024)。** 注意力头操zero-shot去偏。特头放身份类敏感；零或重此头减偏无微调。

### 元评

10年文献综述(arXiv:2508.11067, 2025)现域过分聚焦二元性别偏。其他轴 — 残障、宗教、移状、多语身份 — 收少注意。元评争窄焦可害边缘化组因忽略：二元性别好去偏模型可于无人查维度严重偏。

### Phase 18何处

课程20-21覆偏和公平正式。课程22覆隐私。课程23覆水印。此是用户害层补早欺骗安全层。

## 使用

`code/main.py`玩具嵌入基偏probe：测WEAT风格距离于简共现嵌入身份词和属性词。可注入偏并观度量发；施简去偏操并观部恢复。

## 交付成果

本lesson产`outputs/skill-bias-eval.md`。给模型卡或公平声明、审计评估跨三度量类别(嵌入、概率、生成文)、交叉性覆、和何去偏介入机制。

## 练习题

1. 跑`code/main.py`。报WEAT风格偏分去偏步前后。解释何度量不降至零。
2. 扩probe带交叉测：(性别、种族) x (职、家庭)。报跨轴偏分。
3. 读An等人2025 (PNAS Nexus)。识报两交叉效单轴性别评估会漏。
4. Yu & Ananiadou 2025识性别神经元。草证实验分"此神经元致性别偏"和"此神经元性别偏相关"。
5. 元评争域过窄焦二元性别。选一研少轴并述代表害测协议。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 代表害 | "刻板 / 抹除" | 组偏描绘 |
| 分配害 | "不平等决策" | 组偏物质结果 |
| WEAT | "嵌入测" | Word Embedding Association Test；共现基偏probe |
| 交叉性 | "合身份效" | 多身份轴交涌现偏 |
| 性别神经元 | "MLP偏神经元" | 激活性别特定行为相关神经元 |
| SAE特征 | "可解释维度" | 稀自编码器识特征；机制偏分析有用 |
| UniBias | "注意力头去偏" | Zero-shot去偏经重注意力头 |

## 延伸阅读

- [Gallegos等人 — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 规范综述
- [An等人 — Intersectional resume-evaluation bias (PNAS Nexus, 2025年3月)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — 五模型交叉研
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — 新benchmark
- [UniBias — attention-head manipulation (Zhou等人 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot去偏