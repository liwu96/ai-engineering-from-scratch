# AI SRE——多Agent事件响应、Runbook、预测检测

> AI SRE用LLM基础设施数据(log、runbook、服务拓扑)RAG grounding自动化调查、文档、协调阶段。2026架构模式多agent orchestration——专agent (log、metric、runbook) supervisor协调；AI提议假设和查询、人批准判断。Datadog Bits AI和Azure SRE Agent托管产品发。Runbook演化：NeuBird Hawkeye对抗评估(两模型分析同事件；同意=信心、不同意=不确定)；操作记忆跨队变持久。自动修复持谨慎：AI建议、人批准。全自主行动窄(重启pod、rollback特定部署)紧guardrail——任何"设忘"卖过。出前沿：预事件预测。MIT研究历史log + GPU温度 + API错误模式训LLM测集89%停运10-15分早预测。投：2026末95%企业LLM自动failover。

**类型:** 学习
**语言:** Python(stdlib、玩具多agent事件triage模拟器)
**前置要求:** 阶段17课程13(可观测)、阶段17课程24(Chaos Engineering)
**时间:** ~60分钟

## 学习目标

- 图多agent AI SRE架构：supervisor + 专agent (log、metric、runbook) + 人批准gate。
- 解释为何自动修复窄(重启pod、revert deploy)而非广(重架服务)。
- 命名对抗评估模式(NeuBird Hawkeye)：两模型同意=信心；不同意=升。
- 引MIT 89%早检测结果和操作约束：预测无actuation仅dashboard。

## 问题背景

On-call工程师3 a.m.被页。"Checkout高错误率。"查Datadog、Loki、三runbook、部署log。30分后识根因vLLM KV cache spike OOM。重启pod；错误清。

2026调查前20分可自动。服务grouping log、关联部署、匹配runbook——全RAG + tool-use。监督agent首遍triage假设人开Datadog前呈现。

全自主修复异问题。重启pod：安全。Scale GPU池：政策允安全。重架服务：绝对不。纪律画窄线。

## 概念讲解

### 多agent架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor事件分子查询。专agent工具访问(log search、PromQL、文档检索)。Supervisor综合、假设+证据呈现人。人批准或重定向。

### 自动修复范围

**安全(窄)**：重启pod、revert特定部署、预批准界限scale池、启预批准feature flag。

**非安全(广)**：改服务拓扑、修改资源限、部署新代码、改IAM、改数据库。

任何"设忘"卖过。安全集AI SRE成熟涨、但边界实。

### 对抗评估(NeuBird Hawkeye)

两模型独立分析同事件。若根因同意、信心高。若不同意、升人两假设可见。简模式、对幻觉根因有效过滤。

### 操作记忆

队流传统SRE默杀——部落知识离。AI SRE向量DB存runbook + post-mortem；agent每新事件检索。新工程师加入、AI全历史。

### 预事件预测

MIT 2025研究：历史log、GPU温度、API错误模式训LLM测集89%停运10-15分前预测。

现实检查：预测无actuation dashboard。操作问题"预测时何做？"预emptive drain？Pager？Auto-scale？答政策特定。

### 2026产品

- **Datadog Bits AI**——Datadog内托管SRE copilot。
- **Azure SRE Agent**——Azure原生。
- **NeuBird Hawkeye**——对抗eval + 操作记忆。
- **PagerDuty AIOps**——triage + deduplication。
- **Incident.io Autopilot**——事件指挥 + 协调。

### Runbook代码化

Runbook Confluence页演进版本markdown结构节(symptom、hypothesis、verify、act)。结构runbookRAG检索好。AI-SRE rollout始非结构runbook转结构。

### 你应记数

- MIT早检测：89%停运、10-15分lead时间。
- 多agent triage：supervisor + (log、metric、runbook) + 人。
- 安全自动修复集：重启pod、revert deploy、界限内scale。
- 对抗eval：两模型独立；同意=信心。

## 使用

`code/main.py`模多agent triage：log agent找错、metric agent找CPU spike、runbook agent匹配已知问题。Supervisor排假设。

## 交付成果

本lesson产`outputs/skill-ai-sre-plan.md`。给当前on-call、事件量、队成熟、设计AI SRE rollout。

## 练习题

1. 跑`code/main.py`。若log和metric agent不同意？Supervisor何解？
2. 定义服务三"安全"自动修复行动。论证每。
3. 写结构runbook模板：节、必需域、验证命令。
4. 预测检测12分lead火。政策何——pager、预drain、或双？
5. 论3人队2026应AI SRE否或等。虑成熟、量、风险。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| AI SRE | "on-call agent" | LLM-backed事件调查 + 协调 |
| Supervisor agent | "orchestrator" | 顶层agent事件分子查询 |
| 专agent | "域agent" | 工具访问子agent (log、metric、runbook) |
| 自动修复 | "AI修" | 窄预批准行动；非广重架 |
| 操作记忆 | "向量runbook" | 向量DB post-mortem + runbook RAG |
| 对抗eval | "两模型检查" | 独立分析；同意=信心 |
| NeuBird Hawkeye | "对抗那个" | 对抗eval + 记忆模式产品 |
| Bits AI | "Datadog SRE agent" | Datadog托管AI SRE |
| 预事件预测 | "早检测" | 停运10-15分lead预测 |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)