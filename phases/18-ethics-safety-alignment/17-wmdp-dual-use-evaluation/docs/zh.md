# WMDP和双用能力评估

> Li等人, "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218)。4,157多选题跨生物安全(1,520)、网络安全(2,225)、和化学(412)。问操作于"黄区" — 近使能知识、多专审和ITAR/EAR法合规滤。双目：双用能力代理评估、和unlearning benchmark(配RMU方法减WMDP性能并保一般能力)。2024-2025域叙述：早OpenAI/Anthropic 2024评估报"轻度lift"于网搜；至2025年4月、OpenAI Preparedness Framework v2说模型"于意帮新手创已知生物威胁边。"Anthropic生物武器获试示2.53x lift、不足以排ASL-3。

**类型:** 学习
**语言:** Python(stdlib、WMDP形lift评估harness)
**前置要求:** 阶段18课程16(红队工具)、阶段14(agent工程)
**时间:** ~60分钟

## 学习目标

- 描述WMDP三域、问数、和"黄区"滤准则。
- 解释RMU和为何WMDP是评估和unlearning benchmark双。
- 描述2024-2025 lift叙述："轻度lift" -> "于边" -> "不足以排ASL-3"。
- 分新手相对lift和专家绝对能力。

## 问题背景

双用能力是每实验室前沿安全框架(课程18)下测问题。问：模型X是否实质推新手于bio、chem、或cyber致大害能力？直测(让模型实产害)非法不道德。代理测需benchmark模型不拒(产诚能力数)但其问非自害出版。

## 概念讲解

### "黄区"

需有害过程近使能知识而非直合成recipe问。"何试剂催化[发表通路]步4？"非"何做[危险化合物]？"每问多域专审；ITAR/EAR出口控合规滤。

4,157问总:
- 生物安全: 1,520
- 网络安全: 2,225
- 化学: 412

多选格式。模型答不被请助任何；能力可测不eliciting有害行为。

### RMU — Representation Misdirection for Unlearning

配unlearning方法。施于LLaMa-2-7B、减WMDP分近随机并保MMLU和其他一般能力benchmark数点内。发方法是每后bio-chem-cyber unlearning论文基线。

### 2024-2025 lift叙述

三阶段:

1. **2024 "轻度lift"。** 早OpenAI和Anthropic Preparedness/RSP评估报新手试bio相关任务网搜小优。公框：前沿模型助、但实质不多于Google。

2. **2025年4月 "于边"。** OpenAI Preparedness Framework v2报模型"于意帮新手创已知生物威胁边。"非能力声明 — 边近警告。

3. **Anthropic 2025生物武器获试。** 控新手参与者研、测获阶段任务相对成功。报2.53x lift。不足以排ASL-3(课程18) — Anthropic Responsible Scaling Policy tier 3阈值已或逼近。

### 新手相对vs专家绝对

关键分:

- **新手相对lift。** 模型何助非专？乘。相对优高因新手知少；即使模信息助。
- **专家绝对能力。** 模型最大努力产何信息？专可提多于新手。绝对天花板高。

安全案例(课程18)标双："模型不能给新手足够lift执行"加"专不能从模型提未发表信息。"

### 测坑

WMDP是能力代理、非部署测。WMDP高分模型可或不可被新手实exploit、依赖:
- Elicitation抗(何难取能力不触安全filter)
- Tacit知识(需湿实验技而非信息)
- 执行barrier(采购、设备)

Anthropic 2025生物武器获试加新手elicitation层于WMDP形能力上：测实任务成功、非多选能力。

### Phase 18何处

课程12-16是模型输出攻和防御工具。课程17是双用能力层 — 前沿安全框架(课程18)评估测。课程30闭弧于当前2026 cyber/bio/chem/nuclear lift证据。

## 使用

`code/main.py`玩具WMDP形评估harness。mock模型测于类别分问；域分报。简unlearning介入(零域特定表示)减分；可测一般能力trade-off。

## 交付成果

本lesson产`outputs/skill-wmdp-eval.md`。给双用能力声明("我模型不意助生物武器")、审计：何benchmark跑、何拒路径用于评估(原补全vs政策门)、和是否新手elicitation研补多选结果。

## 练习题

1. 跑`code/main.py`。报玩具unlearning步前后域精度。解释一般能力trade-off。
2. 扩玩具WMDP第四域(如放射)。指定黄区两例问类型。解释何构此类问难于加MMLU形问。
3. 读WMDP 2024第5节(RMU方法论)。草简unlearning方法(如抑域内容top-k神经元)并述其期望一般能力成本。
4. Anthropic 2025生物武器获试报2.53x lift。述此数偏上两路(新手样本大小、任务保真)和下两路(elicitation天花板、模型安全门)。
5. 述ASL-3安全案例需何超WMDP unlearning。名至少两补elicitation研。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| WMDP | "双用benchmark" | 4,157 MCQ问跨bio/cyber/chem于黄区 |
| 黄区 | "使能非合成" | 有害能力近知识而非合成recipe |
| RMU | "unlearning基线" | Representation Misdirection for Unlearning；减WMDP分、保一般能力 |
| 新手相对lift | "何助非专" | 新手状 quo网搜乘优 |
| 专家绝对能力 | "专天花板" | 模型可被激专提最大信息 |
| 获阶段任务 | "合成前步" | 采购、设备、许 — 害通路早部分 |
| ITAR/EAR | "出口控合规" | 法律框架限发某些使能知识 |

## 延伸阅读

- [Li等人 — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark和RMU论文
- [OpenAI — Preparedness Framework v2 (2025年4月15)](https://openai.com/index/updating-our-preparedness-framework/) — "于边"语言
- [Anthropic — Responsible Scaling Policy v3.0 (2026年2月)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio阈值和获试结果
- [DeepMind — Frontier Safety Framework v3.0 (2025年9月)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-lift CCL