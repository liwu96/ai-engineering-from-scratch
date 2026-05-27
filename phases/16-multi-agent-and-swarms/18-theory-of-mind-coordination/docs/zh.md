# 心智理论与涌现协调

> Li et al.(arXiv:2310.10701)示LLM Agent在合作文本游戏展现**涌现高阶心智理论**(ToM)——推理另一Agent对第三Agent信念的信念——但因上下文管理和幻觉长视规划失败。Riedl(arXiv:2510.05174)测量人口跨高阶协同发现**仅**ToM提示条件产身份链接区分和目标导向互补；低容量LLM只示虚假涌现。即，协调涌现是提示条件性和模型依赖，非免费。本lesson实现最小ToM-aware Agent、跑合作任务带和不带ToM提示、测量协调delta对Riedl 2025协议。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程07(Society of Mind和Debate)、阶段16课程17(Generative Agents)
**时间:** ~75分钟

## 问题背景

多Agent协调常看起来神奇：Agent分工、预判彼此、避冗余。通常这"涌现"是提示工程伪影——有人告诉Agent"协调"。删提示，删协调。

Riedl 2025发现更严：控条件下，协调仅当Agent提示推理**其他Agent心智**(ToM)时涌现。无ToM提示，即使强模型示不存活统计控协调模式。这对生产重要：团队部署"多Agent协调"特性是提示依赖和脆弱。

本lesson视ToM为特定能力（推理关于信念的信念）、建最小ToM-aware Agent、测量真正协调vs提示装扮。

## 概念讲解

### ToM意味什么

发展心理学：3岁孩想任何人内心世界匹配自己。5岁孩理解他人有不同信念。7岁孩推理关于信念的信念("她想我想球在杯下")。这些是零阶、一阶、二阶ToM。

LLM Agent，ToM阶映射：

- **零阶：**无他人模型。Agent只基于自己观察行动。
- **一阶：**Agent有每其他Agent信念模型。"Alice相信X。"
- **二阶：**Agent建模递归信念。"Alice相信Bob相信X。"

Li et al. 2023发现一阶和二阶ToM在合作游戏LLM Agent涌现但随长视和不可靠通信降级。

### Sally-Anne测试，简要

1985假信念测试：Sally放弹珠篮A，离开。Anne移到篮B。Sally回时她会找哪？一阶ToM孩说篮A(Sally信念异现实)。无ToM孩说篮B。

GPT-4-era LLM平铺Sally-Anne风格测试通过。当叙述长、场景变多次、问题间接措辞时失败。这是2026生产LLM ToM实践状态。

### Riedl协调测量

Riedl(arXiv:2510.05174)建人口规模测试：N Agent、合作目标、变提示条件。测量：

1. **身份链接区分。**Agent随时间发展稳定角色区分吗？
2. **目标导向互补。**Agent行动互补彼此（不同子任务）而非重复？
3. **高阶协同。**统计测量群组达成任何子集不能达成的。

结果：仅ToM提示条件所有三指标产高于基线信号。无ToM提示，中等容量模型指标徘徊接近偶然。大模型无显式ToM提示示些协调但效果小于显式提示。

### 协调幻觉

无统计控，demo中"涌现协调"常反映：

- 提示工程内置协调（系统提示说"一起工作")。
- 观察者偏（我们见预期模式）。
- 成功跑后选择。

声称"涌现协调"无可测量信号的生产系统应视为营销。声称前测量。

### 最小ToM-aware Agent

结构：

```
agent state:
  own_beliefs:    {Agent相信的事实}
  other_models:   {other_agent_id -> {Agent归因于他们的信念}}
  actions_last_N: [他人行动历史]

observation update:
  - 从直接观察更新own_beliefs
  - 从他们的行动+先前信念更新other_models[agent_id]

action selection:
  - 列举候选行动
  - 每个预测在建模信念下每其他Agent下步做什么
  - 选在这些预测下最大化联合结果行动
```

`other_models`属性是ToM状态。一阶ToM只保持一层。二阶加`other_models[i][other_models_of_j]`——我想Agent i想Agent j相信什么。

### 长视为何有害

Li et al. 文档：上下文限导致Agent忘哪个信念属于谁。幻觉给其他Agent模型加假信念。两者产"我以为他想X"错误随时间复合。

论文和2024-2026后续文档缓解：

- **显式ToM状态在提示。**结构格式：`{agent_id: belief_list}`。强制检索保身份-信念绑定。
- **更短推理链。**每轮更少ToM更新减少复合幻觉。
- **外部ToM存储。**LLM上下文外维护模型；每轮只注入相关部分。

### ToM在生产何处失败

- **对抗设置。**好ToM Agent更易操纵（你可建模他们对你建模，然后利用）。
- **异构团队。**模型不同时，对一对手工作的ToM模型不泛化。
- **真值依赖任务。**ToM关于信念；若正确性依赖事实，ToM可分心。

### 你可实际测量的协调

三实践信号团队协调是真实而非提示装扮：

1. **随时间互补。**跨多轮任务，Agent行动覆盖不相交子任务吗？
2. **预判。**Agent A轮T+1行动依赖T+2对B行动预测且预测正确吗？
3. **纠正。**当A轮T误读B信念，A到T+2纠正了吗？

这些在日志多Agent系统可测量。它们是"协调"叙述实质版本。

## 构建

`code/main.py`实现：

- `ToMAgent`——追踪自己信念和每其他Agent信念模型。
- 合作任务：三Agent必须从三箱收三token；每箱可持一token。Agent不能通信；从彼此行动推断意图。
- 两配置：`zeroth_order`(无ToM)和`first_order`(ToM一层信念模型)。
- 200随机试验测量：完成率、重复率(两Agent瞄准同箱)、平均完成轮数。

跑：

```
python3 code/main.py
```

预期输出：零阶Agent~35%率重复努力、~60%试验10轮完成。一阶ToM Agent~5%重复、~95%完成。Delta是可测量协调效果。

## 使用

`outputs/skill-tom-auditor.md`是技能审计多Agent系统声称"涌现协调"。检查提示装扮、控统计显著性、测量互补。

## 交付成果

协调声称清单：

- **控条件。**系统无协调提示版本。测量两者。
- **统计测试。**系统与控差异在`p < 0.05`显著吗？
- **互补测量。**随时间行动不相交度，非只最终成功。
- **失败案例日志。**Agent误协调时，ToM状态像什么？
- **模型容量披露。**若效果在更小模型消失，说明。

## 练习题

1. 跑`code/main.py`。确认一阶ToM减重复率约7x。扩展到5 Agent和5箱差距持续？
2. 实现二阶ToM(Agent A建模B对C想法)。比一阶改进？在什么任务？
3. 注入**幻觉**到ToM状态：每轮随机翻一个信念。这降一阶性能多少？
4. 读Li et al.(arXiv:2310.10701)。复现"长视降级"发现：轮数从10到30增长，一阶ToM性能如何变？
5. 读Riedl 2025(arXiv:2510.05174)。在模拟日志实现高阶协同统计。无ToM提示条件效果存在？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 心智理论 | "理解他人心智" | 建模另一Agent信念能力。按阶分级(0, 1, 2+)。 |
| Sally-Anne测试 | "假信念测试" | 1985发展心理学；LLM平铺版通过，复杂版失败。 |
| 一阶ToM | "A相信X" | 建模一他人对事实信念。 |
| 二阶ToM | "A相信B相信X" | 递归建模更深一层。 |
| 身份链接区分 | "随时间稳定角色" | Riedl指标：角色持久非随机。 |
| 目标导向互补 | "不相交行动" | Agent瞄不同子任务非同一个。 |
| 高阶协同 | "群组超任何子集" | Riedl真正协调统计测量。 |
| 协调幻觉 | "看起来协调" | 无可测量信号提示装扮协调外观。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 合作游戏涌现ToM；长视失败模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 人口规模测量；ToM提示是承载条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — ToM概念1978起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne论文(1985)