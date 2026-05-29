# 谈判与讨价还价

> Agent谈判资源、价格、任务分配和条款。2026基准集清晰：NegotiationArena(arXiv:2402.05863)示LLM可通过人设操纵("绝望")提升回报~20%；"Measuring Bargaining Abilities"(arXiv:2402.15813)示买方比卖方难且规模不帮助——其**OG-Narrator**(确定性报价生成器+LLM叙述者)推成交率从26.67%到88.88%；大规模自主谈判竞赛(arXiv:2503.06416)跑~180k谈判发现**隐藏思维链**Agent通过对对方隐藏推理获胜；Bhattacharya et al. 2025在哈佛谈判项目指标排名Llama-3最有效、Claude-3激进、GPT-4最公平。本lesson实现Contract Net Protocol(FIPA前身，Lesson 02)、接入LLM风格买方/卖方、运行OG-Narrator风格分解、测量成交率随每结构选择变化。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程02(FIPA-ACL Heritage)、阶段16课程09(Parallel Swarm Networks)
**时间:** ~75分钟

## 问题背景

两个Agent需对价格达成一致。留给纯语言提示，2024-2026 LLM成交率惊人低(arXiv:2402.15813紧密参数化讨价~27%)。规模不修复：GPT-4讨价结构不比GPT-3.5好；它在讨价*语言*上更好。

根源是LLM混淆两工作——决定报价和叙述报价。OG-Narrator分离：确定性报价生成器计算数值移动；LLM只叙述。成交率跳~89%。

这镜像经典多Agent发现：解耦机制与通信层赢。Contract Net Protocol(FIPA，1996；Smith，1980)是参考任务市场机制。插入LLM到叙述位得现代LLM驱动任务市场。

## 概念讲解

### Contract Net，一段话

Smith 1980 Contract Net Protocol：**管理者**广播**提案请求(cfp)**；**竞标者**响应**propose**消息含报价；管理者选赢家发送**accept-proposal**给赢家和**reject-proposal**给输家。赢家执行工作。可选消息：**refuse**(竞标者拒提)。FIPA编码为`fipa-contract-net`交互协议。

### OG-Narrator为何赢

"Measuring Bargaining Abilities of Language Models"(arXiv:2402.15813)观察到：

- LLM常破讨价规则（无意义价格报价、忽略对方ZOPA）。
- 锚定差（接受坏首报价；象征而非策略量反报价）。
- 规模不修复。更大模型用类似策略错误做更合理语言。

OG-Narrator分解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ 报价生成器       │ price → │  LLM叙述者       │ → message
           │  (确定性)        │        │  (写人类风格     │
           │                  │        │   陪同词)        │
           └──────────────────┘        └──────────────────┘
```

报价生成器是经典讨价策略：Rubinstein讨价模型、Zeuthen策略、或简单价格上tit-for-tat。LLM叙述。消息含确定性价格和自然语言框架。

成交率跳因为：
- 价格停在讨价区。
- 锚定策略而非情绪。
- LLM做擅长事：写作。

### NegotiationArena发现

arXiv:2402.05863提供canonical基准。头条发现：

- LLM可用人设("我周五前绝望卖出")改进回报~20%——人设操纵是真战术。
- 公平/合作Agent被对抗者利用；防御需显式反姿态。
- 对称配对~40%基准场景收敛到不公平结果。

这不是"LLM是差讨价者"。是"LLM讨价太像人类，包括可利用部分"。

### 思维链隐藏

大规模自主谈判竞赛(arXiv:2503.06416)跨多LLM策略跑~180k谈判。赢家隐藏推理对对方：

- 若Agent打印"我只到$75；保留价$70"到公开可见scratchpad，对手读到。
- 赢家私下计算策略；输出通道只含报价和最小必需叙述。

这是2026回声经典博弈论(Aumann 1976理性与信息)：透露私人估值损回报。LLM不直觉此并愉快在推理痕迹里打保留价，对对方可见。

工程教训：分离私人scratchpad上下文与公开消息上下文。非可选。

### Bhattacharya et al. 2025——模型排名

哈佛谈判项目指标(原则谈判、BATNA尊重、利益互惠)：

- **Llama-3**最有效达成讨价(成交率+回报)。
- **Claude-3**最激进谈判者(高锚、晚让步)。
- **GPT-4**最公平(跨配对回报方差最小)。

这是2025快照。要点非哪个模型2026年4月赢——是不同基座模型有持久谈判风格。异构ensemble(Lesson 15)包含此作为多样性源。

### 通过Contract Net + LLM任务分配

Contract Net现代LLM多Agent复用：

1. 管理者Agent分解任务为单元。
2. 广播`cfp`带任务描述给工作者Agent。
3. 每工作者返回报价：`(price, eta, confidence)`其中price可是token、计算单元或美元。
4. 管理者选赢家（单或多，依任务）并授予。
5. 拒工作者可竞标其他任务。

这扩展超100工作者因为协调是广播-响应，非同步聊天。生产使用：Microsoft Agent Framework编排模式、某些LangGraph实现。

### LLM-Stakeholders交互谈判

NeurIPS 2024(https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf)引入多方可评分游戏带**秘密分数**和**最低接受阈值**。每stakeholder有私人效用；LLM必须从消息推断。这是两方讨价推广到N方联盟形成。相关生产任务市场异构工作者能力。

### 叙述vs机制规则

跨所有2024-2026谈判基准，一致工程规则是：

> 让LLM叙述。不让LLM计算报价。

若报价需是数值（价格、ETA、数量），从谈判状态确定性生成，让LLM产框架。若报价需是提案结构（任务分解、角色分配），让LLM起草，但发送前验证schema和约束检查。

## 构建

`code/main.py`实现：

- `ContractNetManager`、`ContractNetTask`、`Bid`——管理者+竞标者、广播cfp、收集提案、授予。
- `og_narrator_bargain(state, rng)`——OG-Narrator买方：确定性Zeuthen风格让步向中点。
- `seller_response(state, rng)`——确定性卖方反报价策略（两风格结构真值）。
- `naive_llm_bargain(state, rng)`——模拟全LLM讨价者：高方差选价格，常ZOPA外。
- 测量：1000试验成交率，每试验采样新保留价。

跑：

```
python3 code/main.py
```

预期输出：naive-LLM成交率~65-75%；OG-Narrator成交率~85-95%；15-25点差距是分解报价生成与叙述结构优势。加Contract Net任务市场分配例三竞标者一任务。

## 使用

`outputs/skill-bargainer-designer.md`设计讨价协议：谁生成报价（确定性或LLM）、谁叙述、私人scratchpad如何分离公开消息、成交率如何监控。

## 交付成果

生产讨价清单：

- **分离scratchpad。**私人状态永不到达对方上下文。不可协商。
- **确定性报价生成。**价格、数量、ETA：计算，不提示。
- **验证所有入报价**对schema。在协议边界拒ZOPA外报价。
- **限轮。**最多3-5轮；死锁升级调解者。
- **持续测量成交率和回报方差**。成交率下降是症状——常提示漂移或对方侧攻击。
- **记录所有拒提案**带确定性理由。Contract Net管理者，输竞标者需理解为何。

## 练习题

1. 跑`code/main.py`。确认OG-Narrator在成交率胜naive-LLM。多少？
2. 实现**人设基础回报改进**(arXiv:2402.05863)——买方在叙述采用"本周绝望买"人设，报价生成器不变。成交率或回报变？
3. 实现思维链**隐藏**：维护不传对方的私人scratchpad字符串。若意外漏（模拟换通道）发生什么？
4. 扩展Contract Net到N竞标者拍卖带保留价。当所有报价超保留，管理者如何选最低价格vs最高质量？选哪个授予规则为何？
5. 读Bhattacharya et al. 2025哈佛谈判项目指标。实现两种不同风格讨价者（激进vs公平）。测量对称和非对称配对回报方差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Contract Net | "任务市场" | Smith 1980，FIPA 1996。cfp + propose + accept/reject。Canonical任务市场。 |
| ZOPA | "可能协议区" | 买方最大和卖方最小重叠。报价外不可成交。 |
| BATNA | "谈判失败最佳替代" | 若此交易失败的退路。设保留价。 |
| OG-Narrator | "报价生成器+叙述者" | 分解：确定性报价，LLM叙述。 |
| Zeuthen策略 | "风险最小让步" | 基风险限让步经典报价生成器。 |
| Rubinstein讨价 | "交替报价均衡" | 无限范围折扣博弈论模型。 |
| CoT隐藏 | "藏推理" | arXiv:2503.06416赢家保持私人scratchpad；公开通道只示报价。 |
| 人设操纵 | "情绪姿态" | arXiv:2402.05863：绝望/紧迫人设~20%回报增益。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 基准；人设操纵和利用发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator和买方比卖方难结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k谈判；思维链隐藏赢
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 多方可评分游戏带秘密效用
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers