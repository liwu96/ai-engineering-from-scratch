# Model、System、和Dataset Cards

> 三文档格式构AI透明。Model Cards (Mitchell等人 2019) — 模型营养标签：训数据、量分分析、道德考虑、caveats；仅0.3% Hugging Face model cards文档道德考虑(Oreamuno等人 2023)。Datasheets for Datasets (Gebru等人 2018, CACM) — motivation、composition、collection process、labeling、distribution、maintenance；电子datasheet类比。Data Cards (Pushkarna等人, Google 2022) — 模块分层细节(telescopic、periscopic、microscopic)作不同读者边界对象。2024-2025发展：经LLM自生成(CardGen, Liu等人 2024)；model-card细节关高达29%下载增于HF (Liang等人 2024)；可验attestation(Laminator, Duddu等人 2024)；可持续报告加碳/水(Jouneaux等人 2025年7月)；EU/ISO监管卡涌现。System Cards (Sidhpurwala 2024; Meta系统级透明；"Blueprints of Trust" arXiv:2509.20394) — 端到端AI系统文档覆安全能力、提示注入保、数据exfiltration检、与人价值对齐。

**类型:** 构建
**语言:** Python(stdlib、model-card + datasheet + system-card生成器)
**前置要求:** 阶段18课程18(安全框架)、阶段18课程24(监管)
**时间:** ~60分钟

## 学习目标

- 描述原Mitchell等人 2019 model card和Gebru等人 2018 datasheet。
- 描述Data Cards telescopic/periscopic/microscopic分层。
- 描述System Cards及其端到端覆。
- 陈三2024-2025发展(自生成、可验attestation、可持续报告)。

## 问题背景

监管框架(课程24)和实验室安全政策(课程18)需文档。文档格式从模型特定(model cards)到数据集特定(datasheets)到系统特定(system cards)演进。每址不同透明scope。2024-2025自化和可验attestation工址长期采纳问题。

## 概念讲解

### Model Cards (Mitchell等人 2019)

节:
- Model details。
- Intended use。
- Factors(评估相关人口或环境因素)。
- Metrics。
- Evaluation data。
- Training data。
- Quantitative analyses(按因素分)。
- Ethical considerations。
- Caveats和recommendations。

采纳问题：Oreamuno等人 2023审计Hugging Face model cards现仅0.3%文档道德考虑。

### Datasheets for Datasets (Gebru等人 2018)

电子datasheet类比。节:
- Motivation(何创数据集)。
- Composition(何在其中)。
- Collection process(何组装)。
- Labeling(若适用)。
- Uses(意、禁、风险)。
- Distribution。
- Maintenance。

CACM 2021发。Datasheet是上游文档；model card依赖datasheet准确。

### Data Cards (Pushkarna等人, Google 2022)

模块分层细节。三zoom级:
- **Telescopic。** 非专高概。
- **Periscopic。** ML practitioner中概。
- **Microscopic。** 审计者详特征级文档。

边界对象框：不同读者从同文档提不同信息。

### System Cards

Scope：端到端AI系统含model + safety stack + 部署context。节典型含:
- 安全能力。
- 提示注入保。
- 数据exfiltration检。
- 与声明人价值对齐。
- Incident response。

Sidhpurwala 2024和Meta系统级透明工。"Blueprints of Trust" (arXiv:2509.20394)形式化System Card作部署层补Model Cards。

### 2024-2025发展

- **CardGen (Liu等人 2024)。** 经LLM自model-card生成；报标准化Mitchell 2019字段比多人类写card更高客观。
- **下载相关(Liang等人 2024)。** 详model cards关高达29%高下载率于HF — 采纳压现市场驱动、非仅合规驱动。
- **Laminator (Duddu等人 2024)。** 经硬件TEE / 加签可验attestation — 允model card载proof-of-claim、非仅claim。
- **可持续(Jouneaux等人 2025年7月)。** 碳、水、和算能footprint加；涌ISO标准。
- **监管卡。** EU AI Act(课程24) GPAI Code of Practice Transparency章需model cards作合规artifact。

### Phase 18何处

课程24-25是监管和CVE层。课程26是文档层。课程27是训数据治、是datasheet上游。课程28是研生态系统产cards引用评估。

## 使用

`code/main.py`玩具部署生最小model card、datasheet、和system card。每跟规范节结构。可查格式并比三scope。

## 交付成果

本lesson产`outputs/skill-card-audit.md`。给model card、datasheet、或system card、审计节覆、量分、和是否可验attestation存。

## 练习题

1. 跑`code/main.py`。查生cards。识弱(placeholder-only)节并述何证据强。
2. 延model card带两人口组量分分析(课程20)。
3. 读Oreamuno等人2023于0.3%采纳率。提一model card规范结构改会增道德考虑采纳。
4. Laminator (Duddu等人 2024)用TEEs可验attestation。设计载评估结果加密attestation model-card字段并述验者角色。
5. 写System Card(非Model Card)于你过项目或假部署。识对第三审计者最高值节。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Model Card | "Mitchell card" | Mitchell等人 2019 ML模型标准文档 |
| Datasheet | "Gebru datasheet" | Gebru等人 2018 数据集标准文档 |
| Data Card | "Pushkarna card" | Google 2022 模块分层数据文档 |
| System Card | "部署card" | 端到端AI系统文档含安全stack |
| 边界对象 | "不同读者、一doc" | Data Cards框：同文档服务异audience |
| 可验attestation | "Laminator attestation" | 加密或TEE proof附于文档claim |
| 可持续字段 | "碳 / 水footprint" | 涌2025加环境accounting |

## 延伸阅读

- [Mitchell等人 — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 规范model card
- [Gebru等人 — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet论文
- [Pushkarna等人 — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala等人 — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card形式化