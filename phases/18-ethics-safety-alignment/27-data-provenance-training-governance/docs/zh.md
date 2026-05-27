# 数据溯源和训数据治

> EU AI Act需GPAI机器可读opt-out标准2025年8月(经EU Copyright Directive TDM exception)。California AB 2013 (签2024) — Generative AI训数据透明需开发者发数据集summary带12mandated字段。2025 DPA对齐于合法利益：Irish DPC (2025年5月21)接受Meta LLM训于first-party public EU/EEA成人内容带safeguard经EDPB opinion；Cologne Higher Regional Court (2025年5月23)驳回injunction；Hamburg DPA撤urgency；UK ICO (2025年9月23)发LinkedIn AI-training safeguard(透明、简opt-out、延objection windows)正面监管响并续监控 — 非正式clearance。Brazilian ANPD (2024年7月2)暂停Meta处理于不足信息透明；预防措于2024年8月30撤经Meta提合规plan。关键不可逆问题：cookie-consent框架设计实时、可逆tracking；一旦数据在模型权重、手术erasure不可能 — 无实GDPR right-to-erasure于训神经网络。合规窗口于采集时。Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee等人, "Consent in Crisis", 2024年7月)：大audit示AI数据commons速降因出版者加robots.txt限。

**类型:** 学习
**语言:** Python(stdlib、12-field California AB 2013 scaffold生成器)
**前置要求:** 阶段18课程24(监管)、阶段18课程26(cards)
**时间:** ~60分钟

## 学习目标

- 描述California AB 2013 12 mandated字段于Generative AI训数据透明。
- 陈2025 DPA位置于合法利益LLM训(Irish DPC、UK ICO、Hamburg、Cologne)。
- 描述不可逆问题：何GDPR right-to-erasure无训神经网络实等价。
- 陈Data Provenance Initiative "Consent in Crisis"发现。

## 问题背景

训数据治是每model card(课程26)和监管义务(课程24)上游。2024-2025监管景于三原则consolidate：opt-out基础设施、per-dataset披露、和公开数据合法利益accommodation。采集时不合规提供者不可下游remediate。

## 概念讲解

### California AB 2013

签2024。文档须于2026年1月1日或前发于2022年1月1日或后发系统。Section 3111(a)需开发者发训用数据集高层summary带12法定项:
1. 数据集源或owner。
2. 数据集何助AI系统意目的描述。
3. 数据集数据点数(一般范围可；动态数据集估)。
4. 数据点类型描述(标签数据集标签类型；无标签一般特征)。
5. 数据集是否含copyright、trademark、或patent保数据、或全public domain。
6. 数据集是否购或licensed。
7. 数据集是否含个人信息(per Cal. Civ. Code §1798.140(v))。
8. 数据集是否含aggregate consumer信息(per Cal. Civ. Code §1798.140(b))。
9. 开发者cleaning、processing、或其他modification、带意目的。
10. 数据采集期、带notice若采集ongoing。
11. 数据集首次开发用日期。
12. 系统是否用或续用合成数据生成。

Item 12(合成数据)对Gebru等人 2018 datasheets新。Item 7(个人信息)触发Privacy Rights Act (CPRA)义务。法豁安全/integrity、aircraft-operation、和federal-only national-security系统(Section 3111(b))。

### EU AI Act(课程24)和TDM opt-out

EU Copyright Directive text-and-data-mining exception允训于公开内容除非rightholder opts out。EU AI Act GPAI Code of Practice Copyright章需GPAI提供者尊机器可读opt-out信号(robots.txt、C2PA "No AI Training" claim、等)。

### 2025 DPA收敛于合法利益

Irish DPC (2025年5月21)：Meta训于first-party public EU/EEA成人用户内容plan接受带safeguard经EDPB opinion。Cologne Higher Regional Court (2025年5月23)驳回对Meta injunction：opt-out够。Hamburg DPA撤urgency procedure于EU-wide consistency。UK ICO (2025年9月23)发正面监管响 — 非正式clearance — 于LinkedIn AI训resume带类似safeguard和ongoing监控。

收敛原则：合法利益可justify训于公开first-party内容带opt-out。Consent不需。

### Brazilian ANPD (2024年6月)

暂停Meta处理巴西用户数据于AI训于不足信息透明。异于EU DPAs结果 — ANPD优先透明于合法利益admissibility。

### 不可逆问题

Cookie-consent设计实时、可逆tracking。训数据异：一旦数据入模型权重、手术erasure不可能。从头重训是唯一全remediation、且贵 prohibitive。

部remediation:
- **Unlearning。** 近移；MIA测(课程22)。
- **Influence function基定位。** 识数据最影响权重；选择性更新。
- **Fine-tune-suppression。** 训模型拒数据派输出。

无全解问题。合规窗口于采集时。

### Data Provenance Initiative

dataprovenance.org。Longpre, Mahara, Lee等人 "Consent in Crisis" (2024年7月)：AI训数据commons大audit。发现：出版者加速度加robots.txt限。公开可训commons速收缩。2023 -> 2024约25%顶训源加些限。意：未来训数据可用依赖新获取范式(licensing、合成生成、激励参与)。

### Phase 18何处

课程26是模型级文档。课程27是数据集级治。合定义透明层。课程28绘研此问生态系统。

## 使用

`code/main.py`玩具数据集生California AB 2013合规12-field数据集summary scaffold。可填字段并观何触发隐私或copyright后续义务。

## 交付成果

本lesson产`outputs/skill-provenance-check.md`。给训用数据集、查AB 2013 12-field覆、opt-out基础设施合规、DPA对齐、和不可逆风险评估。

## 练习题

1. 跑`code/main.py`。玩具数据集产12-field summary并识何字段under-specified。
2. EU Copyright Directive TDM opt-out机器可读。提opt-out信号标准格式并比robots.txt和C2PA "No AI Training。"
3. 读Data Provenance Initiative "Consent in Crisis" (2024年7月)。述三最快限内容类别并争一经济后果。
4. 2025 DPA对齐接受公开内容训合法利益。构合法利益不够场景并识提供者需何法律basis。
5. 草训数据溯源manifest组AB 2013字段和每数据集C2PA签溯源链。识一技和一法barrier。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| AB 2013 | "California法" | Generative AI训数据透明；12 mandated字段 |
| TDM exception | "text-and-data-mining" | EU Copyright Directive训数据exception带opt-out |
| 合法利益 | "EU basis" | GDPR Article 6 basis可justify公开内容训 |
| Opt-out信号 | "机器可读no-train" | robots.txt、C2PA "No AI Training"、TDM.Reservation |
| 不可逆 | "不能un-train" | 数据在模型权重非手术可移 |
| Unlearning | "近似移" | 后训介入减模型特定数据依赖 |
| Consent in Crisis | "DPI audit" | 2024年7月发现robots.txt限加速 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Generative AI训数据透明法
- [EU AI Act + GPAI Code of Practice (课程24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright章
- [Longpre, Mahari, Lee等人 — Consent in Crisis (dataprovenance.org, 2024年7月)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — 监管context