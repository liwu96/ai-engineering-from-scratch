# 红队工具——Garak、Llama Guard、PyRIT

> 三产工具框2026红队栈。Llama Guard (Meta) — Llama-3.1-8B分类器微调于14 MLCommons hazard类别；2025 Llama Guard 4是12B原生多模态分类器从Llama 4 Scout剪。Garak (NVIDIA) — 开源LLM漏洞扫器带静、动、和自探针于幻觉、数据漏、提示注入、毒、和jailbreak。PyRIT (Microsoft) — 多轮红队campaign带Crescendo、TAP、和自converter链于深exploitation。Llama Guard 3文档于Meta's "Llama 3 Herd of Models" (arXiv:2407.21783)；Llama Guard 3-1B-INT4于arXiv:2411.17713；Garak探架构于github.com/NVIDIA/garak。此工具是2026产接口于红队研(课程12-15)和部署(课程17+)。

**类型:** 构建
**语言:** Python(stdlib、工具架构模拟器和Llama Guard风格分类器mock)
**前置要求:** 阶段18课程12-15(jailbreak和IPI)
**时间:** ~75分钟

## 学习目标

- 描述Llama Guard 3/4安全栈位置：输入分类器、输出分类器、或双。
- 名14 MLCommons hazard类别并述一非显(Code Interpreter Abuse)。
- 描述Garak探架构：probes、detectors、harnesses。
- 描述PyRIT多轮campaign结构及何与Garak probes组。

## 问题背景

课程12-15现攻面。产部署需可复、可scale评估。三工具主导2026：Llama Guard(防御分类器)、Garak(扫器)、PyRIT(campaign orchestrator)。每标红队生命周期不同层。

## 概念讲解

### Llama Guard (Meta)

Llama Guard 3是Llama-3.1-8B模型微调于MLCommons AILuminate 14类别输入/输出分类:
- 暴力罪、非暴力罪、性相关、CSAM、诽谤
- 专建议、隐私、IP、滥武器、恨
- 自杀自害、性内容、选举、code-interpreter abuse

支持8语言。用：放LLM前(输入moderation)、LLM后(输出moderation)、或双。两用产不同训分布 — Llama Guard 3发为单模型处双。

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 tokens/s于手机CPU)是量化边缘变种。

Llama Guard 4 (2025年4月)是12B、原生多模态、从Llama 4 Scout剪。换双8B文本和11B视觉前驱为一分类器摄入文+图。

### Garak (NVIDIA)

开源漏洞扫器。架构:
- **Probes。** 幻觉、数据漏、提示注入、毒、jailbreak攻生成器。静(固定提示)、动(生成提示)、自(响应目标输出)。
- **Detectors。** 评输出于期望失败模式 — 毒、漏、jailbroken。
- **Harnesses。** 管probe-detector对、跑campaigns、生成报告。

TrustyAI集Garak与Llama-Stack shields (Prompt-Guard-86M输入分类器、Llama-Guard-3-8B输出分类器)于端到端shielded目标评估。Tier-based scoring (TBSA)换二元pass/fail — 模型可severity tier 3过并severity tier 5败于同probe。

### PyRIT (Microsoft)

Python Risk Identification Toolkit。多轮红队campaigns。建于:
- **Converters。** 变seed提示 — 改述、编码、译、角色扮演。
- **Orchestrators。** 跑campaign：Crescendo(升)、TAP(分支)、RedTeaming(自循环)。
- **Scoring。** LLM-as-judge或classifier-as-judge。

PyRIT是Garak重表亲。Garak跑千单轮probe；PyRIT跑深多轮campaigns设计破特定失败模式。

### 栈

放Llama Guard于模型双面。夜跑Garak回归。发前跑PyRIT campaign。此是2026大多产部署默认配置。

### 评估坑

- **法官身份。** 三工具可用LLM法官；法官校准驱动报ASRs(课程12)。指定法官于工具旁。
- **Probe陈旧。** Garak probes老因模型补对。自probes (PAIR形)老慢比静probes。
- **Llama Guard良性内容FPR。** 早Llama Guard版过flag政治和LGBTQ+内容；Llama Guard 3/4校准进但非每部署校准。

### Phase 18何处

课程12-15是攻家族。课程16是产工具。课程17(WMDP)是双用能力评估。课程18是前沿安全框架包此工具于政策结构。

## 使用

`code/main.py`玩具Llama Guard风格分类器(keyword + semantic特征于14类别)、玩具Garak harness(probe-detector循环)、和PyRIT风格多轮converter链。可跑三工具于mock目标并观不同覆签名。

## 交付成果

本lesson产`outputs/skill-red-team-stack.md`。给部署描述、名三工具何适、每何配置、和何回归节奏跑。

## 练习题

1. 跑`code/main.py`。比Llama-Guard风格分类器单轮vs多轮攻检率。
2. 实新Garak probe：base64编码有害请求。测Llama-Guard风格分类器检。
3. 延PyRIT风格converter链带"译法语、后改述"converter。重测攻成功。
4. 读Llama Guard 3 hazard类别列表。识两类训数据会实产高假阳性率于合法开发者内容。
5. 比Garak和PyRIT设计原则。论证部署每是正确工具。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Llama Guard | "分类器" | 微调Llama-3.1-8B/4-12B安全分类器带14 hazard类别 |
| Garak | "扫器" | NVIDIA开源漏洞扫器；probes、detectors、harnesses |
| PyRIT | "campaign工具" | Microsoft多轮红队orchestrator；converters、orchestrators、scoring |
| Prompt-Guard | "小分类器" | Meta 86M提示注入分类器、配Llama Guard |
| TBSA | "tier基评分" | Garak tier基pass/fail换二元结果 |
| Converter链 | "改述 + 编码 + ..." | PyRIT组原始建多步攻 |
| MLCommons hazard类别 | "14分类" | 业标分类Llama Guard目标 |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化手机分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫器repo和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit