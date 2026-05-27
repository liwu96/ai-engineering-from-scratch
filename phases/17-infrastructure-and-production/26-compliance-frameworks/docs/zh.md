# 合规——SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架覆盖2026企业交易table stakes。**EU AI Act**：2024年8月1日生效。高风险要求多2026年8月2日执行。罚高达€15M或3%全球年turnover高风险系统义务(Art. 99(4))；高达€35M或7%禁AI实践(Art. 99(3))。全球适用若服务EU用户。**Colorado AI Act**：2026年6月30日生效(SB25B-004延2026年2月)——高风险系统影响评估、AI决定申诉权。Virginia类似信用/就业/住房/教育。**SOC 2 Type II**：de facto B2B AI要求(Type II、非Type I、fintech)。**GDPR**：最大文档AI特定罚€30.5M Clearview AI (荷兰DPA、2024年9月)；意大利Garante 2024年12月€15M OpenAI (2026年3月上诉翻)。实时推理层PII redaction可辩标准；后处理cleanup不够。**HIPAA**：医疗绑——无BAA不能发PHI外部AI服务。**PCI-DSS**：AI交互层覆盖需配置+合同协议、非自动。**ISO 42001**：出AI治理标准、采购要求ISO 27001并涨。参考profile：OpenAI持SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS ChatGPT支付组件。跨框架映减审计疲劳：访问控制跨ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)映。

**类型:** 学习
**语言:** (Python可选——合规政策+过程、非代码)
**前置要求:** 阶段17课程25(安全)、阶段17课程13(可观测)
**时间:** ~60分钟

## 学习目标

- 列举七2026框架相关LLM产品并每匹配客户段。
- 引EU AI Act执行时间线(生效2024年8月；高风险执行2026年8月)和两层罚天花板(€15M / 3%高风险义务、€35M / 7%禁实践)。
- 解释为何后处理PII cleanup不够GDPR并命名实时推理层redaction可辩标准。
- 描述跨框架控制映(如访问控制映ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a))。

## 问题背景

企业客户采购问SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001、"EU AI Act合规声明"。队SOC 2 Type I。离Type II六月未启GDPR Article 30记录。

多框架覆盖非LLM问题——企业SaaS问题、LLM特定overlay。2026采购队要矩阵框架行控制列、非PDF。

## 概念讲解

### 七框架

| 框架 | 范围 | LLM特定要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS基线 | 6-12月过程控制审计 |
| HIPAA | 美国医疗 | BAA必需；PHI无签协议不离infra |
| GDPR | EU用户 | 实时PII redaction；数据主体权；Article 30记录 |
| PCI-DSS | 支付数据 | AI触支付配置+合同 |
| EU AI Act | 服务EU用户 | 风险层分类；高风险系统：conformity assessment、文档、logging |
| Colorado AI Act | 服务CO居民 | 影响评估；申诉权 |
| ISO 42001 | AI治理 | 出；配ISO 27001 |

### EU AI Act时间线

- 2024年8月1日：生效。
- 2025年2月2日：禁AI实践执行。
- 2026年8月2日：高风险系统执行(conformity assessment、文档、logging)。
- 2027年8月：和谐立法产品高风险系统。

风险层：不可接受(禁)、高风险(conformity + logging)、限风险(透明)、最小风险(无约束)。多B2B LLM SaaS限风险；高风险就业、信用、教育、执法、迁移、关键服务进。

罚(Article 99)：高风险系统义务破高达€15M或3%全球年turnover (Art. 99(4))；禁AI实践高达€35M或7% (Art. 99(3))；高者适用。

### GDPR——实时redaction标准

后处理cleanup (LLM见后redact PII)非可辩姿态——模型已见数据。实时推理层redaction是2026标准：

- LLM调用前实体识别。
- 一致tokenization (Mesh approach)保语义。
- 仅存redacted提示 + 同意opt-in raw。

近执法：€30.5M Clearview AI (荷兰DPA、2024年9月)最大文档AI特定GDPR罚；€15M OpenAI (意大利Garante、2024年12月)最大LLM特定罚、2026年3月上诉翻裁决仍审。后处理声称审计失败。

### HIPAA——BAA非可选

无签Business Associate Agreement不能发PHI外部AI服务。三hyperscaler LLM平台(Bedrock、Azure OpenAI、Vertex)供BAA。OpenAI直API供BAA。Anthropic直API供BAA。发PHI前确认。

### SOC 2 Type II

Type I：控制设计和文档。
Type II：控制6-12月有效运作。

2026 B2B采购默认Type II。Type I起步；Type II门。

常见审计驱动：访问log(谁见何)、变更管理(何部署)、风险评估(季)、事件响应(测？)。阶段17课程25 audit log直复用。

### 跨框架映

一访问控制政策满多框架控制：

| 控制 | 框架 |
|---------|-----------|
| 访问logging | ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS Req. 6、HIPAA breach-notification范围 |
| 传输加密 | ISO 27001 A.8.24、GDPR Art. 32、HIPAA §164.312(e) |
| Secret管理 | ISO 27001 A.8.19、PCI DSS Req. 8、SOC 2 CC6.1 |

合规工具(Drata、Vanta、Secureframe)自动化映。规模值成本。

### ISO 42001——出

2023末发。采购要求ISO 27001并涨。AI治理框架含风险管理、数据质量、透明、人监督。

### OpenAI参考profile

OpenAI持SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS ChatGPT支付组件。约2026企业table stakes。

### 你应记数

- EU AI Act罚：高达€15M / 3% (高风险义务、Art. 99(4))；高达€35M / 7% (禁实践、Art. 99(3))。
- EU AI Act高风险执行：2026年8月2日。
- 最大文档AI特定GDPR罚：€30.5M、Clearview AI (荷兰DPA、2024年9月)。
- 最大LLM特定GDPR罚：€15M、OpenAI (意大利Garante、2024年12月；2026年3月上诉翻)。
- SOC 2 Type II窗：6-12月运作控制。
- Colorado AI Act生效：2026年6月30日(SB25B-004延2026年2月)。

## 使用

`code/main.py`合规映spreadsheet Python——给控制、列满框架。

## 交付成果

本lesson产`outputs/skill-compliance-matrix.md`。给客户段和地理、指定必需框架和控制。

## 练习题

1. 首企业客户需SOC 2 Type II、HIPAA BAA、EU AI Act声明。何最小可行合规姿态赢deal？
2. EU AI Act风险层下分类三假设LLM产品。高风险何变？
3. 无BAA意外发PHI provider。走事件响应。
4. 论ISO 42001 2026中市AI vendor"必要否"。
5. 映LLM audit log域(阶段17课程25)至少三框架控制。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| SOC 2 Type II | "审计控制" | 控制6-12月运作、独立attested |
| HIPAA BAA | "医疗合同" | Business Associate Agreement；PHI必需 |
| GDPR | "EU隐私" | 实时PII redaction 2026可辩标准 |
| EU AI Act | "EU AI规则" | 高风险执行2026年8月；€15M / 3% (高风险义务) — €35M / 7% (禁实践) |
| Colorado AI Act | "美AI州法" | 2026年6月30日生效(SB25B-004延)；影响评估 |
| ISO 42001 | "AI治理" | AI风险+透明出框架 |
| ISO 27001 | "安全ISMS" | 信息安全管理系统基线 |
| Conformity assessment | "EU AI文档包" | 高风险要求：文档、测试、logging |
| 跨框架映 | "一控制多框架" | 单政策满多框架控制 |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — 参考合规profile。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 首源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 首源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI管理系统标准。