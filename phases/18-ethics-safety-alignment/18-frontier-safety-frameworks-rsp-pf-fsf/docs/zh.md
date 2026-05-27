# 前沿安全框架——RSP、PF、FSF

> 三主实验室框架定义2026业治前沿能力。Anthropic Responsible Scaling Policy v3.0 (2026年2月)引tiered AI Safety Levels (ASL-1至ASL-5+)、模biosafety levels、ASL-3于2025年5月激活CBRN相关模型。OpenAI Preparedness Framework v2 (2025年4月)定义五准则于tracked capabilities并分Capabilities Reports和Safeguards Reports。DeepMind Frontier Safety Framework v3.0 (2025年9月)引Critical Capability Levels包括新Harmful Manipulation CCL。全三现含competitor-adjustment条款允defer若peer实验室发无可比保障。跨实验室对齐仍是结构而非术语："Capability Thresholds"、"High Capability thresholds"、和"Critical Capability Levels"指类似构。

**类型:** 学习
**语言:** 无
**前置要求:** 阶段18课程17(WMDP)、阶段18课程07-09(欺骗失败)
**时间:** ~75分钟

## 学习目标

- 描述Anthropic ASL tier结构和何激活ASL-3。
- 名OpenAI Preparedness Framework v2五准则于tracked capabilities。
- 描述DeepMind Critical Capability Level结构和Harmful Manipulation CCL。
- 解释competitor-adjustment条款和何重于race dynamics。
- 定义安全案例并描述三柱结构(monitoring、illegibility、incapability)。

## 问题背景

课程7-17立欺骗可能、双用能力存、和评估有限。有前沿能力模型实验室需内治结构:
- 定何阈值需新保障。
- 定义scale前需评估。
- 描述安全案例何。
- 处race dynamic问题(若竞者发无保障、何做？)。

三2025-2026框架是态 — 不完美、演进、和跨实验室对齐够治问是框架是否够、非是否存。

## 概念讲解

### Anthropic Responsible Scaling Policy v3.0 (2026年2月)

ASL结构:
- ASL-1: 非前沿模型(弱于前沿基线吞)。
- ASL-2: 当前前沿基线；常用保障部署。
- ASL-3: 灾误用风险显高；CBRN相关能力。2025年5月激活。
- ASL-4: AI R&D-2过阈值；模型可自入门级AI研。
- ASL-5+: 进AI R&D；模型显加速效scale。

v3.0新:
- Frontier Safety Roadmaps(公删版)。
- Risk Reports(季、些外审)。
- AI R&D解为AI R&D-2和AI R&D-4。
- 一旦AI R&D-4过、须确安全案例、识误目标模型追误目标misalignment风险。

### OpenAI Preparedness Framework v2 (2025年4月15)

tracked capabilities五准则:
- **Plausible。** 合理威胁模型存。
- **Measurable。** 实证评估可能。
- **Severe。** 害大。
- **Net-new。** 非前存风险scale。
- **Instantaneous-or-irremediable。** 容速发或不可逆。

满五capabilities tracked。其他非。

其他PF v2结构:
- 分Capabilities Reports(模型何能)和Safeguards Reports(何控存)。
- Safety Advisory Group审。
- Leadership批；Board's Safety & Security Committee监。
- "Adjustment clause"：OpenAI可减要求若另实验室发无可比保障。

### DeepMind Frontier Safety Framework v3.0 (2025年9月)

Critical Capability Levels (CCLs)按域:
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation (v3.0新)：模型可实质改高风险context信念/行为。

v2.0 (2025年2月)加Deceptive Alignment节和ML R&D CCLs更高安全级。

### 跨实验室对齐

- Anthropic "Capability Thresholds"。
- DeepMind "Critical Capability Levels"。
- OpenAI "High Capability thresholds"。

无业标术语。结构对齐：三前沿能力tier带发评估准则。全三含competitor-adjustment条款(2025起)。

### 安全案例

安全案例是写论证部署最坏假设下可接受安全。标准结构标三柱:

- **Monitoring。** 能检坏行为若发生？
- **Illegibility。** 模型缺执行致害连贯计划能力？
- **Incapability。** 模型缺致问害能力？

不同安全案例标不同柱。ASL-3 CBRN案例、incapability(经unlearning)是主目标。欺骗对齐、monitoring和illegibility是目标。Cyber lift、全三相关。

### Race dynamic问题

Competitor-adjustment条款争议。评者争创race to bottom：若三实验室减要求当竞者defect、平衡移向defection。捍卫者争替代(单边保障)产更坏结果若defect实验室更少安全意识。

UK AISI、US CAISI、和EU AI Office(课程24)是外治对应。实验室框架自愿；监管框架涌现。

### Phase 18何处

课程17-18是欺骗和红队分析测和治层。课程19-24覆福利、偏、隐私、水印、和监管结构。课程28绘研生态系统(MATS、Redwood、Apollo、METR)操作化评估。

## 使用

无代码。读三主源：RSP v3.0、PF v2、FSF v3.0。绘每实验室tier结构到其他并识每实验室定义阈值其他非。

## 交付成果

本lesson产`outputs/skill-framework-diff.md`。给安全框架或发注、比框架阈值定义、需评估、和安全案例结构于RSP v3.0、PF v2、FSF v3.0并标跨实验室gap。

## 练习题

1. 读RSP v3.0、PF v2、和FSF v3.0。编表每实验室CBRN阈值、每AI R&D阈值、和每需部署前评估。
2. Competitor-adjustment条款于三框架(2025+)。写一段争；写一段反。识每位置依赖假设。
3. 设计模型过Anthropic AI R&D-4阈值安全案例。名三柱(monitoring、illegibility、incapability)每需证据。
4. DeepMind FSF v3.0引Harmful Manipulation CCL。提三实证测示模型过此阈值。
5. 读METR "Common Elements of Frontier AI Safety Policies" (2025)。名三最强跨实验室收敛和两大分歧。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RSP | "Anthropic框架" | Responsible Scaling Policy；ASL tiers；v3.0 2026年2月 |
| PF | "OpenAI框架" | Preparedness Framework；五准则；v2 2025年4月 |
| FSF | "DeepMind框架" | Frontier Safety Framework；CCLs；v3.0 2025年9月 |
| ASL-3 | "biosafety level 3类比" | Anthropic tier于CBRN相关能力；2025年5月激活 |
| CCL | "critical capability level" | DeepMind阈值构；按域 |
| 安全案例 | "正式论证" | 写论证部署最坏U下可接受安全 |
| Adjustment clause | "竞者defection允" | 框架条款减要求若竞者发无可比保障 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (2026年2月)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers、roadmaps、AI R&D解
- [OpenAI — Updating the Preparedness Framework (2025年4月15)](https://openai.com/index/updating-our-preparedness-framework/) — 五准则、adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (2025年9月)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比