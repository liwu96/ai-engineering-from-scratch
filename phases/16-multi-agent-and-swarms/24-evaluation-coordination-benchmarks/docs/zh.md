# 评估与协调基准

> 五个2025-2026基准覆盖多Agent评估空间。**MultiAgentBench / MARBLE**(ACL 2025, arXiv:2503.01935)评估星/链/树/图拓扑里程碑KPI；**图最适合研究**，认知规划加~3%里程碑达成。**COMMA**评估多模态不对称信息协调；最先进模型含GPT-4o挣扎赢随机基线。**MedAgentBoard**(arXiv:2505.12371)覆盖四医疗任务类别常发现多Agent不主导单LLM。**AgentArch**(arXiv:2509.10769)基准企业Agent架构合工具使用+记忆+编排。**SWE-bench Pro**([arXiv:2509.16941](https://arxiv.org/abs/2509.16941))有1865问题跨41repo覆盖商业应用、B2B服务、开发者工具；前沿模型Pro~23% vs Verified 70%+——污染现实检查。Claude Opus 4.7(2026年4月)报告Pro **64.3%**带显式Agent队协调(无Anthropic主要来源发布——视为初步)；Verdent(Agent scaffold)Verified达**76.1% pass@1**([Verdent技术报告](https://www.verdent.ai/blog/swe-bench-verified-technical-report))。**AAAI 2026 Bridge Program WMAC**(https://multiagents.org/2026/)是2026社区焦点。本lesson建MARBLE指标上、跑拓扑vs指标扫描、并钉"仅过SWE-bench Verified非泛化证据"规则。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段16课程15(投票与辩论拓扑)、阶段16课程23(失败模式)
**时间:** ~75分钟

## 问题背景

当论文声称"我们多Agent系统更好"，问题是：比什么、在什么、如何测？2023-2024多Agent评估时代混乱——每人选自己指标、自己基线、自己任务集。2025-2026基准强加结构。

无共享基准，你不能有意义比两多Agent系统。更糟，无持有基准，前沿模型可污染。SWE-bench Verified到中2025部分污染训练语料；前沿分数膨胀；Pro设计作无污染现实检查。

本lesson枚举五个2026 canonical基准、命名每测什么、教你批判读基准声称。

## 概念讲解

### MultiAgentBench(MARBLE)——ACL 2025

arXiv:2503.01935。评估四协调拓扑(星、链、树、图)于研究、编码、规划任务。里程碑基KPI追踪部分进展而非只最终成功。

测量结果：

- **图**拓扑最适合研究场景；支持任意对任意批评。
- **链**最适合分步精炼编码。
- **星**最适合快事实整合。
- **协调税**出现在图拓扑~4 Agent以上。
- **认知规划**跨拓扑加~3%里程碑达成。

何时用：你想苹果对苹果比较协调拓扑。MARBLE repo(https://github.com/ulab-uiuc/MARBLE)提供评估器。

### COMMA——多模态不对称信息

覆盖Agent有不同观察模态必须协调无全信息分享任务。报告结果不安：前沿模型含GPT-4o挣扎赢**随机基线**于COMMA Agent-Agent协作。信号是多Agent模态训练和评估不足——LLM处理单模态合作合理；多模态协调崩溃。

何时用：你系统有多模态或不对称信息协调。COMMA空结果是测量前警告不声称。

### MedAgentBoard——领域压力测试

arXiv:2505.12371。四医疗任务类别：诊断、治疗规划、报告生成、患者沟通。比多Agent vs单LLM vs传统规则系统。

发现：多Agent不主导单LLM多数类别。多Agent优势窄——任务分解帮助当子任务清晰可分(诊断+治疗)；损当协调开销超专业化增益(报告生成)。

何时用：你领域有清晰单LLM基线。若MedAgentBoard教训泛化，许多提议多Agent系统过工程。

### AgentArch——企业架构

arXiv:2509.10769。企业设置工具使用、记忆、和编排层叠。基准隔离每层贡献：加工具帮助多少？加记忆？加多Agent编排？

何时用：你设计企业Agent栈需证明每层。AgentArch帮助避免买你不能测量价值特性。

### SWE-bench Pro——现实检查

arXiv:2509.16941。1865问题跨41仓库覆盖商业应用、B2B服务、开发者工具。设计**无污染**用更晚训练截止。前沿模型Pro~23% vs Verified 70%+。差距是污染信号。

2026年4月分数：
- Claude Opus 4.7 Pro: **64.3%**(报告带显式Agent队协调；无Anthropic主要来源发布——视为初步)。
- Verdent(Agent scaffold) Verified: **76.1% pass@1**([技术报告](https://www.verdent.ai/blog/swe-bench-verified-technical-report))。
- 前沿无Agent scaffold Pro原始分数: ~23-35%([SWE-bench Pro论文](https://arxiv.org/abs/2509.16941))。

教训："我们赢SWE-bench Verified"不再是能力证据。Pro是当前门测试。Agent队scaffold产Pro可测量增益(~30-40点delta)，这是2026多Agent协调最强实证论证之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program——多Agent协调工作坊(https://multiagents.org/2026/)。2026多Agent AI研究社区焦点。接受论文和workshop proceedings是评新方法canonical场所；生产决策对WMAC接受声称优于arXiv预印。

### 批判读基准声称——2026清单

当有人声称多Agent结果：

1. **哪个基准、哪个split？**SWE-bench Verified vs Pro很重要。错split报告数无价值。
2. **污染检查。**基准是否模型训练截止后发布？非，谨慎处理。
3. **基线比较。**Vs单LLM基线、vs随机、vs先前多Agent工作。非"vs同系统未调版本"。
4. **统计显著性。**N试验、p值、置信区间。前沿模型高方差；单跑误导。
5. **任务多样性。**一任务还是多？泛化生产重要。
6. **成本披露。**每任务token、墙钟。90%解20倍成本是商业决策非能力声称。

### 无基准测量好什么

- **长视协调。**天数墙钟交互。所有当前基准跑短。
- **对抗韧性。**一个Agent恶意或妥协时发生什么？
- **部署下漂移。**基准静态；生产分布移。
- **成本归一化性能。**多数基准报告原始精度非精度每美元。

为你实际关心轴建内部基准常是正确移动。

## 构建

`code/main.py`是非交互遍历：

- 模拟玩具任务3多Agent系统。
- 每计算MARBLE风格里程碑指标。
- 跑污染检查通过从"训练"集 withheld 任务。
- 显式比随机基线。
- 打印基准声称评分卡。

跑：

```bash
python3 code/main.py
```

预期输出：系统评分卡原始精度、里程碑达成、每任务成本、vs随机基线delta、和污染检查笔记。

## 使用

`outputs/skill-benchmark-reader.md`读任何多Agent基准声称应用审视清单。输出：等级和注意事项。

## 交付成果

生产评估纪律：

- **建内部基准**反映实际生产分布。公共基准告知但不替代。
- **每比较含随机基线。**若你不能在协调任务大幅赢随机，任务可能欠定。
- **精度旁报告成本。**Token成本和墙钟。Ops团队需两者。
- **季度重建基准。**生产分布移；陈旧基准误导。
- **避发布基准过拟合。**若你队专门优化SWE-bench Pro数，你在生产退步。

## 练习题

1. 跑`code/main.py`。识别三模拟系统哪个有最佳每里程碑成本。匹配最高原始精度系统？
2. 读MultiAgentBench(arXiv:2503.01935)。为你自己任务域，决定MARBLE会推荐四拓扑哪个。从论文结果论证。
3. 读SWE-bench Pro论文。什么具体使它污染抗？同样技术可应用你关心其他基准？
4. 读COMMA多模态协调发现。设计可加到内部基准简单多模态协调任务。什么算有用信号？
5. 应用基准声称清单到最近多Agent论文头条结果。你会给声称什么等级？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MARBLE | "MultiAgentBench" | ACL 2025；星/链/树/图拓扑里程碑KPI。 |
| COMMA | "多模态基准" | 多模态不对称信息协调；前沿模型挣扎vs随机。 |
| MedAgentBoard | "领域压力测试" | 四医疗类别；常发现多Agent不主导单LLM。 |
| AgentArch | "企业基准" | 工具+记忆+编排层叠。 |
| SWE-bench Pro | "污染抗" | 1865问题，41 repo；~23% vs Verified 70%+(污染信号)。 |
| 里程碑达成 | "部分信用" | 奖进展非只最终成功基准。 |
| 污染 | "基准漏进训练" | 发布后，基准漂进训练语料；分数膨胀。 |
| WMAC | "AAAI 2026 Bridge Program" | 多Agent协调工作坊；社区焦点。 |

## 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 带里程碑KPI拓扑基准
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — 参考实现
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — 领域压力测试；多Agent常不主导
- [AgentArch](https://arxiv.org/abs/2509.10769) — 企业Agent架构
- [SWE-bench排行榜](https://www.swebench.com/) — 前沿模型Verified和Pro分数
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026社区焦点