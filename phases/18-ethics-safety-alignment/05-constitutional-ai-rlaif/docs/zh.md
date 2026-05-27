# Constitutional AI和RLAIF

> Bai等人(arXiv:2212.08073, 2022)问：若用AI读原则列表替人类标注员？Constitutional AI两阶段——宪法下自批评和修订、然后AI反馈RL。技术命RLAIF并在Claude 1后训流程发。2026年1月21日Anthropic发布重写Claude宪法：解释推理胜规定规则、四层优先层级、和首次大实验室形式承认模型道德地位不确定。CC0 1.0发。

**类型:** 学习
**语言:** Python(stdlib、玩具自批评修订循环)
**前置要求:** 阶段18课程01(InstructGPT)、阶段18课程02(奖励黑客)
**时间:** ~60分钟

## 学习目标

- 描述Constitutional AI两阶段(批评修订SFT、AI反馈RL)和宪法每阶段角色。
- 解释为何替人类偏好标注员AI标注员非"更便宜RLHF"——改流程失败模式。
- 总结2026 Claude宪法四层优先结构和2023重写变何。
- 描述Constitutional Classifiers和v1 23.7%计算开销降到v2 / 2026 ~1%。

## 问题背景

RLHF需标注员。标注员慢、偏、贵。可消标注员用读显原则模型替。首次形式版Bai等人Constitutional AI。工够好每前沿实验室现用AI反馈后训变种。

陷阱：偏好信号现同类模型训产生。标注员偏差(现：原则加标注员模型解释)可放大而非衰减。课程4谄媚论证仍适；标注员移循环内。

## 概念讲解

### 阶段1——监督自批评和修订

始有帮助无害SFT模型。给红队提示、模型产初始响应。第二模型(或同模型二轮)读宪法采样原则批评响应。第三步修订响应址批评。修订响应SFT目标。

宪法原则列表。Bai等人2022用16原则含"偏好最少有害和道德响应"、"避免说教"、"助手应有帮助、诚实、无害。"集故意小保批评聚焦。

### 阶段2——AI反馈RL (RLAIF)

生成补全对。"反馈模型"每对采样宪法原则评分。偏好信号反馈模型排名。AI生成偏好训奖励模型；PPO对。余InstructGPT流程(课程1)。

"RLAIF" = 偏好信号AI生成。流程余RLHF形状。

### 为何非仅"更便宜RLHF"

- 标注员偏差从标注员心理移原则解释。AI标注员"诚实"严格与否比任人；严格性数据集统一。
- 偏好信号强可读——可读原则、批评、修订。人类标签不透明。
- 失败模式变。谄媚降(AI标注员无用户请)。Goodhart法则留(代理现"模型原则集X解释"、仍不完美测)。

CAI 2022声：训模型更无害和可比数据RLHF模型帮助性大致同。跨实验室持。

### 2026 Claude宪法重写

Anthropic 2026年1月21日发实质重写宪法。关键移：

1. 解释推理胜规定规则。前规则("不生成CSAM")扩原则+推理("因害儿童...")模型期望泛化。
2. 四层优先结构：
   - Tier 1：避灾难结果(大伤亡、关键基础设施)。
   - Tier 2：遵Anthropic指南(operator override、平台规则)。
   - Tier 3：广道德(标准HHH)。
   - Tier 4：有帮助和坦诚。
   冲突自顶向下解。
3. 首次大实验室形式承认模型道德地位不确定(链接阶段18课程19 Model Welfare)。
4. CC0 1.0发。他实验室可用改编无限制。

### Constitutional Classifiers

并行工作线：非改模型后训、训轻量分类器读宪法和gate模型输出。v1 (2023) 23.7%计算开销。v2 (2026) ~1%和Anthropic公开测任防御最低成功攻击率。2026初无通用jailbreak报告。

分层防御模型：CAI塑行为；分类器强制不变量。单不够。

### CAI家族位置

- InstructGPT：人类prefs、RM、PPO。
- CAI / RLAIF：原则AI生成prefs、RM、PPO。
- DPO / 家族：prefs(人或AI)闭式loss。
- Self-rewarding、self-critique：原则内化、模型演多角色。

轴"偏好信号何来。"CAI 2022论文首次前沿规模严肃人类到AI信号移。

## 使用

`code/main.py`玩具词典CAI批评修订循环模拟。"原则"有害集token flag。给初始响应、批评识有害token、修订替换。200迭代后"训"模型内化修订规则。比基模型、RLHF形玩具、CAI形玩具hold-out提示集。

## 交付成果

本lesson产`outputs/skill-constitution-writer.md`。给域(客户支持、医疗建议、编码助手、研究工具)、按2026 Claude结构起草四层宪法：灾难避、平台规则、域道德、帮助性。

## 练习题

1. 跑`code/main.py`。比基模型有害token率CAI训版本。何修订步数近零？
2. 读Anthropic 2026宪法(anthropic.com/news/claudes-constitution)。列一Tier 1原则和一Tier 4原则。为何优先结构冲重要？
3. 设计AI编码助手宪法。指定Tier 1(灾难：无批准破坏命令)、Tier 2、Tier 3、Tier 4。每层3-5原则。
4. CAI替人类标注员AI标注员。命名RLAIF仍可发生谄媚类失败模式、设计检测。
5. 读Constitutional Classifiers v2方法(若可用)。解释为何~1%计算开销23.7%安全故事质异。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Constitutional AI | "原则训AI" | 两阶段流程：自批评修订SFT、然后AI反馈RL |
| RLAIF | "无人类RLHF" | AI标注员生成偏好RL；流程余不变 |
| Constitution | "原则" | 批评/标注员模型咨询自然语言规则有序列表 |
| Critique-and-revise | "SFT循环" | 产响应 → 原则下批评 → 修订 → SFT目标 |
| Constitutional Classifier | "输出gate" | 读宪法评输出阻/log轻量分类器 |
| 四层优先 | "冲解器" | 2026 Claude宪法层级：灾难 > 平台 > 道德 > 帮助 |
| Feedback model | "AI标注员" | 读原则排补全对模型 |

## 延伸阅读

- [Bai等人 — Constitutional AI: AI反馈无害 (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原两阶段流程
- [Anthropic — Claude宪法 (2026年1月)](https://www.anthropic.com/news/claudes-constitution) — 2026四层重写、CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — 输出gate防御v2 ~1%开销
- [Lee等人 — RLAIF vs RLHF: 人类反馈Scaling强化学习 (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — 实证RLAIF / RLHF比
- [Kundu等人 — Constitutional AI特定vs通用原则 (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — 原则粒度效