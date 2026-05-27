# 对齐研生态系统——MATS、Redwood、Apollo、METR

> 五组织定义2026非实验室对齐研层。MATS (ML Alignment & Theory Scholars)：527+研者2021年底起、180+论文、10K+引用、h-index 47；2024夏cohort并为501(c)(3)带~90学者和40导师；80% pre-2025校友工于安全/security带200+于Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：应用对齐实验室Buck Shlegeris建；引AI Control(课程10)；与UK AISI合控安全案例。Apollo Research：前沿实验室pre-deployment scheming评估；著In-Context Scheming(课程8)和Towards Safety Cases for AI Scheming。METR (Model Evaluation and Threat Research)：task基能力评估、autonomous-task时间horizon研；"Common Elements of Frontier AI Safety Policies"比实验室框架。Eleos AI Research：模型福利pre-deployment评估(课程19)；行Claude Opus 4福利评估。

**类型:** 学习
**语言:** 无
**前置要求:** 阶段18课程01-27(前Phase 18课程)
**时间:** ~45分钟

## 学习目标

- 识非实验室对齐研生态系统五组织和核心产出。
- 描述MATS scale(学者、论文、h-index)和人才管道角色。
- 描述Redwood AI Control议程和UK AISI合。
- 描述METR task基评估方法论。

## 问题背景

前沿实验室(课程18)内产安全评估并发选结果。实验室外生态系统是评估验、novel失败模式首发现、和人才训处。理解生态系统助解何研发现何人信。

## 概念讲解

### MATS (ML Alignment & Theory Scholars)

2021年底起。研mentorship program；学者花10-12周与senior研者于特定对齐问题。

Scale(2026):
- 527+研者自inception。
- 180+论文发。
- 10K+引用。
- h-index 47。
- 2024夏：90学者 + 40导师；并为501(c)(3)。

职业结果：~80% pre-2025校友工于安全/security。200+于Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。Buck Shlegeris建。引AI Control议程(课程10)。与UK AISI合控安全案例。 advising DeepMind和Anthropic于评估设计。

规范论文：Greenblatt, Shlegeris等人, "AI Control" (arXiv:2312.06942, ICML 2024)；Alignment Faking (Greenblatt, Denison, Wright等人, arXiv:2412.14093, 与Anthropic合)。

风格：特定威胁模型、最坏敌、具体协议可stress-test。

### Apollo Research

前沿实验室pre-deployment scheming评估。著In-Context Scheming(课程8, arXiv:2412.04984)。2025 OpenAI anti-scheming训合partner。产Towards Safety Cases for AI Scheming (2024)。

风格：agent setting评估欺骗涌现；三柱分解(misalignment、goal-directedness、situational awareness)。

### METR (Model Evaluation and Threat Research)

Task基能力评估。Autonomous-task完成时间horizon研。"Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025)比实验室框架。

与Apollo合AI Scheming安全案例sketch co-author。

风格：长horizon任务评估、实证能力测、框架综。

### Eleos AI Research

模型福利pre-deployment评估。行Claude Opus 4福利评估system card第5.3节文档。供课程19福利相关声明外方法论查。

### 流

MATS训研者。毕业生去Anthropic、DeepMind、OpenAI(实验室安全队)或Redwood、Apollo、METR、Eleos(外评估)。外评估者与实验室和UK AISI / CAISI合。出版物喂生态系统回MATS下cohort。

### 何此层重

单源评估不可靠：实验室评估自模型有结构利益冲突。外评估者可raise和验实验室可underreport失败模式。2024 Sleeper Agents论文(课程7)是Anthropic + Redwood；Alignment Faking是Anthropic + Redwood；In-Context Scheming是Apollo；Anti-Scheming是Apollo + OpenAI。多组织结构是质量控。

### Phase 18何处

课程7-11引用Redwood和Apollo工；课程18引用METR框架比；课程19引用Eleos。课程28是Phase依赖生态系统显组织图。

## 使用

无代码。读METR "Common Elements of Frontier AI Safety Policies"作外综如何增实验室内政策工值例。

## 交付成果

本lesson产`outputs/skill-ecosystem-map.md`。给对齐声明或评估、识组织、出版物venue、和方法论风格、并cross-check对知counterpart组织。

## 练习题

1. 选课程7-15一论文并识组织involved。Cross-check作者对MATS校友和当前生态系统affiliation。
2. 读METR "Common Elements of Frontier AI Safety Policies。"识其强调三跨实验室收敛和两大分歧。
3. MATS职业结果是~80%安全/security。争此selection pressure是否adaptive(训域)或biased(滤异端位置)。
4. Redwood和Apollo都做控/scheming工但风格异。选失败模式并述每何investigate。
5. Eleos AI是唯一纯模型福利组织。设计hypothetical第二组织焦于不同福利邻问(认知自由、机器人embodiment、等)并述其方法论。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MATS | "mentorship program" | ML Alignment & Theory Scholars；527+研者2021起 |
| Redwood Research | "控实验室" | 应用对齐；AI Control作者；UK AISI partner |
| Apollo Research | "scheming evals" | 前沿实验室pre-deployment scheming评估 |
| METR | "task-horizon evals" | Task基能力评估；框架综 |
| Eleos AI | "福利实验室" | 模型福利pre-deployment评估 |
| 人才管道 | "MATS -> 实验室" | MATS毕业生流至Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外评估 | "非实验室查" | 非模型生产者做评估；增可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control论文
- [Apollo Research](https://www.apolloresearch.ai/) — scheming评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论