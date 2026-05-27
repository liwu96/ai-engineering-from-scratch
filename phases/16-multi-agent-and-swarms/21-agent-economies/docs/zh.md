# Agent经济体、Token激励、声誉

> 视自主Agent(METR 1小时到8小时工作曲线)需经济能动性。涌现**5层栈**是：**DePIN**(物理计算)→**Identity**(W3C DIDs+声誉资本)→**Cognition**(RAG+MCP)→**Settlement**(账户抽象)→**Governance**(Agentic DAOs)。生产Agent激励网络包括**Bittensor**(TAO subnet奖励任务特定模型)、**Fetch.ai / ASI Alliance**(ASI-1 Mini LLM+FET token)、和**Gonka**(transformer PoW重分配计算到生产性AI任务)。学术工作：AAMAS 2025去中心化LaMAS用**Shapley值信用归属**公平奖励贡献Agent；Google Research"大型语言模型机制设计"提出**token拍卖**单调聚合下二价支付。本lesson建最小Agent市场、应用Shapley值信用归属于多Agent管道、跑二价token拍卖让博弈论机制落地具体。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段16课程16(谈判与讨价还价)、阶段16课程09(Parallel Swarm Networks)
**时间:** ~75分钟

## 问题背景

多Agent系统复杂当Agent联合产价值但需单独奖励。经典机制——平分、最后贡献者全拿——不公平或可博弈。联盟Shapley值奖励构造公平但计算贵。2025-2026文献推有用近似：Shapley采样、单调聚合拍卖、链上声誉从确认贡献累积。

超信用归属，领域转向真实经济Agent：Bittensor TAO奖励挖矿计算调subnet特定模型、Fetch.ai/ASI奖励ASI-1 Mini LLM使用用FET token、Gonka重分配transformer工作量证明向生产性AI任务。自主交易Agent存在今天；问题是如何对齐激励。

本lesson视Agent经济体为特定问题族——信用归属、机制设计、声誉——并用最小数学建每使想法粘。

## 概念讲解

### 5层Agent经济体栈

1. **DePIN(物理计算)。**去中心化基础设施租GPU、存储、带宽。Bittensor subnet、Render Network、Akash。非Agent特定；Agent用。
2. **Identity。**W3C去中心化标识符(DIDs)给每Agent持久ID独立平台。声誉累积到DID。Agent Network Protocol(ANP)用DID作发现层。
3. **Cognition。**Agent推理环：LLM+RAG+MCP。这是其他阶段建。
4. **Settlement。**账户抽象(ERC-4337)让Agent从自己余额付gas无需持ETH。Agent可付服务、彼此、或计算。
5. **Governance。**Agentic DAOs：治理结构人和Agent投票协议变，投票权绑声誉。

非每生产系统用全部五。Bittensor用1、2、部分3、部分4、无5。OpenAI Agent只用3除无。栈是参考地图非要求。

### Bittensor、Fetch.ai、Gonka——什么在跑

**Bittensor(TAO)。**Subnet是专门任务(语言建模、图像生成、预测)。矿工提交模型输出。验证者排名；质押加权评分分布TAO奖励。每subnet有自己的评估。经济教训：付任务特定输出质量非用计算。

**Fetch.ai / ASI Alliance。**ASI-1 Mini LLM在Fetch.ai网络跑；用户付FET token推理。Agent作为peer叙述更强：Fetch上Agent可调另一任务并付FET。

**Gonka。**Transformer工作量证明："工作"是transformer前向通过。矿工通过跑有已知正确输出推理任务赚(从训练数据)。资源生产性PoW而非哈希基PoW。

三都生产级2026年4月。回报分布不同。Bittensor奖励质量相对subnet验证者；Fetch奖励效用由付费用户测；Gonka奖励可验证推理工作。

### Shapley值信用归属

三Agent合作任务。输出评分0.8。谁贡献什么？

Shapley值：唯一信用分配满足四公理(效率、对称、线性、空)。对Agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中`S_i_O`是排序`O`中`i`前Agent集。实践：枚举所有排列、记录每排列每Agent边际贡献、平均。

N=3 Agent，有6排列。N=10，3.6M——所以实践采样排序而非枚举。

### 聚合二价拍卖

Google Research("大型语言模型机制设计")提出聚合LLM输出二价token拍卖。设置：N Agent各提完成；各有被选中私人价值。拍卖者选最高价值提案付*第二高*值。单调聚合下(值依赖选哪个提案非多少投标)，这诚实——Agent投真实值。

为何这对LLM系统重要：你可外包完成任务给多Agent不同定价；拍卖选最优+公平付，Agent无激励误报。

### 声誉资本

DID绑声誉分数从确认贡献累积。简单更新规则：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

衰减因子`alpha`近1。声誉：

- 读便宜路由决策("发难任务给高声誉Agent")。
- Forge贵(随时间累积、绑DID)。
- 可削减：失败验证贡献减。

### AAMAS 2025去中心化LaMAS

LaMAS提案(AAMAS 2025)结合：DID身份、Shapley值信用归属、简单拍卖机制。关键声称：去中心化信用归属步使系统可审计免疫单点操纵。

### 经济何处崩溃

- **价格oracle操纵。**若信用函数可博弈，Agent会博弈。每机制需对抗测试。
- **Sybil攻击。**一操作者启动N假Agent膨胀自己贡献。DIDs慢但不停；声誉forge成本是缓解。
- **验证成本。**信用归属公平性等于验证者。若验证便宜(小LLM)，可博弈；若贵(人panel)，系统不扩展。
- **监管悬垂。**Agent经济体与金融监管交叉。Bittensor、Fetch、Gonka都在某些司法区2026灰区运营。

### Agent经济体何时有意义

- **异质操作者开放网络。**无单队控所有Agent。
- **可验证输出。**无验证，信用归属是猜。
- **长视工作流。**一次任务不受益声誉累积。
- **Token化支付在司法区法律可行**。

在闭企业系统，经济让步更简单分配(管理者分配工作、指标内部)。经济文献主要适用开放网络。

## 构建

`code/main.py`实现：

- `shapley(value_fn, agents)`——小N精确Shapley枚举计算。
- `second_price_auction(bids)`——诚实机制；赢家付第二高。
- `Reputation`——DID绑声誉带指数衰减和削减。
- Demo 1：三Agent合作，精确Shapley归属信用。
- Demo 2：五Agent竞任务槽；二价拍卖选赢家+支付。
- Demo 3：100轮任务分配给异质声誉Agent；声誉加权路由赢随机。

跑：

```
python3 code/main.py
```

预期输出：每Agent Shapley值；拍卖结果示诚实投均衡；声誉加权路由示预热后随机10-20%质量增益。

## 使用

`outputs/skill-economy-designer.md`设计最小Agent经济体：身份层选择、信用归属机制、支付机制、声誉规则。

## 交付成果

2026运行Agent经济体：

- **从声誉开始非token。**声誉便宜实现单独有价值；token加法律经济复杂。
- **奖励前验证。**无独立验证步不分布信用。自报告质量累积sybil游戏。
- **Shapley采样非Shapley精确。**采样100-1000排序；精确枚举不扩展。
- **限衰减因子和最低声誉。**无界衰减擦合法贡献者；太慢衰减奖励陈旧高声誉Agent。
- **对抗审计机制。**开网络前跑红队场景。每机制有博弈论；你想找洞非攻击者。

## 练习题

1. 跑`code/main.py`。确认Shapley值和等于总值(效率公理)。改值函数；Shapley分配改预期方向？
2. 实现Shapley*采样*(Monte Carlo K排序)。K如何影响近似精度？N=4比较精确。
3. 实现拍卖前联盟形成步：Agent可合并队并作为单元投标。哪些联盟形成？结果Pareto优于单独投标？
4. 读Google Research机制设计帖。识别一个假设若违反破坏诚实。LLM设置失败模式像什么？
5. 读AAMAS 2025去中心化LaMAS论文。在合成任务实现其Shapley步10 Agent。精确计算多久？100次采样接近多少？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| DePIN | "去中心化物理基础设施" | Token激励计算/存储/带宽。Bittensor、Akash、Render。 |
| DID | "去中心化标识符" | W3C便携ID规范。Agent声誉绑DID非平台。 |
| ERC-4337 | "账户抽象" | 合约账户可sponsor gas，使Agent支付。 |
| Shapley值 | "公平信用归属" | 满足效率、对称、线性、空的唯一分配。 |
| 二价拍卖 | "Vickrey拍卖" | 诚实机制：赢家付第二高投标。单调聚合兼容。 |
| 声誉资本 | "累积质量分数" | DID绑确认贡献分数；随时间衰减。 |
| Agentic DAO | "Agent+人治理" | Agent投票者一等公民DAO，投票权绑声誉。 |
| TAO / FET / GPU credits | "Token denomination" | Bittensor TAO、Fetch.ai FET、各种DePIN token。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 5层Agent经济体栈调研
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 单调聚合token拍卖
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley值信用归属
- [Bittensor TAO文档](https://docs.bittensor.com/) — subnet结构和奖励分布
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM和FET token
- [W3C Decentralized Identifiers (DIDs)规范](https://www.w3.org/TR/did-core/) — 身份基础