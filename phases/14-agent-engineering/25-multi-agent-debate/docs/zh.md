# Multi-Agent辩论和协作

> Du等(ICML 2024,"Society of Mind")跑N model instance独立提答、后R轮迭代critique彼此收敛。Improves factuality、rule-following、reasoning。Sparse topology胜full mesh于token cost。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程12(Workflow Pattern)、阶段14课程05(Self-Refine和CRITIC)
**时间:** ~60分钟

## 学习目标

- 释辩论protocol:N proposer、R round、收敛共享答。
- 描述何辩论improve factuality、rule-following、和reasoning。
- 释sparse topology:非每debater需见每其他。
- 实stdlib辩论于scripted LLM带full-mesh和sparse variant;测token cost vs accuracy。

## 问题背景

Self-Refine(课程05)是一模型critique己——risk groupthink。CRITIC(课程05)扎根critique外tool——不总available。辩论引入第三mode:多instance、cross-critique、disagreement收敛。

## 概念讲解

### Society of Mind(Du等,ICML 2024)

- N model instance独立提答同问题。
- R轮、每模型读他人proposal并critique。
- 模型按critique update答。
- R轮后、回convergent答。

原experiment用N=3、R=2因cost。Hard problem(MMLU、GSM8K、Chess Move Validity、biography generation)准确率更多agent和更多round improve。

Cross-model组合胜single-model辩论:ChatGPT+Bard together>任alone。

### Sparse topology

"Improving Multi-Agent Debate with Sparse Communication Topology"(arXiv:2406.11776,2024–2025)显full-mesh辩论不总optimal。Sparse topology(star、ring、hub-and-spoke)可匹配准确率更低token cost。每debater仅见peer subset。

Implication:

- Full mesh N=5、R=3=5×3=15 proposal、每读4 peer=60 critique op。
- Star N=5、R=3(一hub+4 spoke)=15 proposal、spoke仅读hub=12 critique op。

### 何辩论帮

- **Factuality。**N独立proposal、cross-check减hallucination。
- **Rule-following。**Chess move validity——一模型漏rule、他人catch。
- **开放reasoning。**多framing narrow正确答。

### 何辩论伤

- **Latency-sensitive UX。**N×R serial round latency你可无。
- **Cost-sensitive scale。**每question N×R token。
- **简事实lookup。**一lookup比五辩论便宜。

### 2026实际实例

- **Anthropic orchestrator-worker**(课程12)——辩论一variant带synthesis step。
- **LangGraph supervisor**(课程13)——中央router+specialist agent可实辩论作node。
- **OpenAI Agent SDK**(课程16)——agent back and forth handoff用于iterative critique。
- **Multi-agent eval**——pair辩论+evaluator-optimizer用于eval signal。

### 何此模式错

- **收敛崩溃。**全agent收敛首错答。Mitigate用required disagreement round。
- **Hub失败。**Star topology、坏hub corrupt everyone。Rotate或用多hub。
- **Prompt homogenization。**全agent用同prompt;它们产同答。用异prompt和/或model。

## 构建

`code/main.py`实stdlib辩论:

- `Debater` class(scripted LLM带per-debater opinion drift)。
- `FullMeshDebate`和`SparseDebate` runner。
- 三question:一factual、一rule-based、一reasoning。
- Metric:convergent答、收敛round、总critique op。

跑:

```
python3 code/main.py
```

Output:per-protocol accuracy和cost;sparse match full mesh 2/3 question更低cost。

## 使用

- **Anthropic orchestrator-worker**用于简2–3-worker辩论。
- **LangGraph**用于stateful multi-round辩论带checkpointing。
- **Custom**用于研究或specialized correctness guarantee。

## 交付成果

`outputs/skill-debate.md`scaffold multi-agent辩论带configurable topology、N、R、和收敛rule。

## 练习题

1. 实"forced disagreement"rule:round 1、每debater须产异proposal。测收敛速度效果。
2. 加confidence-weighted aggregation:debater回(answer,confidence);aggregator weight confidence。帮否?
3. 换一"agent"用异scripted LLM异opinion。异质improve accuracy否?
4. 测full mesh vs sparse你3 question token cost。Plot cost vs accuracy。
5. 读Society of Mind论文。移你toy至N=5、R=3。何break?何get better?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Debate | "Multi-agent critique" | N proposer、R round cross-critique、收敛 |
| Full mesh | "Everyone read everyone" | 每debater每轮读每peer |
| Sparse topology | "Limited peer view" | Debater读peer subset |
| Hub-and-spoke | "Star topology" | 一central debater、N-1 spoke仅读hub |
| Convergence | "Agreement" | Debater收敛共享答 |
| Society of Mind | "Du等辩论论文" | ICML 2024 multi-agent辩论method |

## 延伸阅读

- [Du等,Society of Mind(arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)——canonical multi-agent辩论
- [Sparse Communication Topology(arXiv:2406.11776)](https://arxiv.org/abs/2406.11776)——sparse topology结果
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——orchestrator-worker辩论variant
- [Madaan等,Self-Refine(arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)——single-model自critique counterpart