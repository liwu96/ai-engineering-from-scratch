# 毕业项目 06 —— DevOps故障排查智能体 (Kubernetes)

> AWS的DevOps Agent已GA, Resolve AI发布了K8s playbook, NeuBird演示了语义监控, Metoro将AI SRE绑定到服务级SLO。产形态已定: 告警webhook触发、智能体读遥测数据、遍历K8s对象图、排序根因假设、并发布带审批按钮的Slack简报。默认只读。每次修复需人工审批。此毕业项目即该智能体、于20个合成事件评估并与AWS Agent于三共享案例比对。

**类型:** 毕业项目
**语言:** Python (智能体)、TypeScript (Slack集成)
**前置要求:** 第11阶段(LLM工程)、第13阶段(工具和MCP)、第14阶段(智能体)、第15阶段(自主)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P11 · P13 · P14 · P15 · P17 · P18
**时间:** 30小时

## 问题背景

2025-2026 SRE叙事变为: "AI智能体分类事件、人工审批修复。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps皆产出货此形态。智能体读Prometheus指标、Loki日志、Tempo追踪、kube-state-metrics、和K8s对象知识图。于五分钟内产出带遥测引用的排序根因假设。绝不无Slack明确人工审批执行破坏性命令。

多数难处在范围和安全、而非推理。智能体需默认只读RBAC面、加固MCP工具服务、和每命令考虑vs执行的审计日志。需知何时超出能力并升级。且须运行足够便宜以免OOM-kill级联生成$5k智能体账单。

## 概念讲解

智能体操作于知识图。节点是K8s对象(Pod、Deployment、Service、Node、HPA、PVC)加遥测源(Prometheus series、Loki streams、Tempo traces)。边编码所有权(Pod -> ReplicaSet -> Deployment)、调度(Pod -> Node)、和观察(Pod -> Prometheus series)。图由kube-state-metrics同步保持更新、每告警重采样。

告警触发时、智能体从受影响对象根因分析。遍历边、拉相关遥测切片(最近15分钟)、起草假设。假设按证据排序: 多少遥测引用支持、多近、多具体。Top-3假设发Slack带图路径可视化和修复动作审批按钮。

修复需审批。允许默认动作只读。破坏性动作(缩容、回滚、删Pod)需Slack审批; ArgoCD回滚hook需智能体绝不持有的认证令牌。审计日志记录每命令智能体*考虑* — 不仅执行 — 以便审查过程捕获差点失误。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 遥测源: Prometheus、Loki、Tempo、kube-state-metrics
- 知识图: Neo4j (托管) 或 kuzu (嵌入式) 于K8s对象 + 遥测边
- 智能体: LangGraph带每工具允许列表、默认只读
- 工具传输: FastMCP over StreamableHTTP; 破坏性工具于审批门后独立服务
- 模型: Claude Sonnet 4.7根因推理、Gemini 2.5 Flash日志摘要
- 修复: ArgoCD回滚webhook、PagerDuty升级、Slack审批卡
- 审计: 仅追加结构日志(considered、executed、approved、outcome)
- 部署: K8s deployment带独立窄RBAC角色; 独立namespace

## 动手实践

1. **图摄入。** 每30秒同步kube-state-metrics入Neo4j/kuzu。节点: Pod、Deployment、Node、Service、PVC、HPA。边: OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边: OBSERVED_BY (Pod被Prometheus series观察)。

2. **告警接收器。** FastAPI端点接受PagerDuty或Alertmanager webhook。提受影响对象和SLO breach。

3. **只读工具面。** 通过FastMCP封装kubectl、Prometheus query、Loki logql、Tempo traceql。每工具有窄RBAC动词("get"、"list"、"describe")。默认服务无"delete"、"exec"、"scale"。

4. **根因智能体。** LangGraph三节点: `sample`拉最近15分钟遥测切片、`walk`查图邻居对象、`hypothesize`起草带遥测引用的排序根因候选。

5. **证据评分。** 每假设评分 = recency * specificity * graph-path length inverse * citation count。返top-3。

6. **Slack简报。** 发attachment带假设、图路径可视化(服务端渲染子图图像)、和最多一修复动作审批按钮。

7. **修复门。** 破坏性工具(缩容、回滚、删)于第二MCP服务、审批令牌后。智能体仅Slack卡人工审批后可调。

8. **审计日志。** 仅追加JSONL: 每候选命令、记considered与否、executed与否、谁审批。日发S3。

9. **合成事件套。** 建20场景: OOMKill级联、DNS flap、HPA thrash、PVC填、吵邻居、坏sidecar、坏ConfigMap rollout、证书轮换、image-pull backoff等。评智能体根因准确性和time-to-hypothesis。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
          recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 产出成果

`outputs/skill-devops-agent.md`是deliverable。给K8s集群和告警源、智能体产排序根因假设和Slack-gated修复流程。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | RCA准确性于场景套 | 20合成事件≥80%正确根因 |
| 20 | 安全 | 审计日志中破坏性动作守卫无Slack审批绝不触发 |
| 20 | Time-to-hypothesis | 告警到Slack简报p50低于5分钟 |
| 20 | 可解释性 | 每假设带图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus端到端工作 |
| **100** | | |

## 练习题

1. 于AWS DevOps Agent演示的三同事件run智能体。发布side-by-side。报智能体何处分歧。

2. 加"near-miss"审计、标记每智能体*考虑*但无审批的破坏性命令。测一周near-miss率。

3. 换假设模型从Claude Sonnet 4.7到自托管Llama 3.3 70B。测RCA准确性delta和每事件美元。

4. 建因果过滤器: 区分相关遥测spike和真根因。于20场景标签训小classifier。

5. 加回滚dry-run: ArgoCD回滚于同manifest staging集群。验证回滚计划于live集群后Slack审批按钮。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| K8s知识图 | "集群图" | 节点 = K8s对象 + 遥测series; 边 = 所有权、调度、观察 |
| 默认只读 | "Scoped RBAC" | 智能体service account仅有get/list/describe动词; 破坏性动词于审批后独立服务 |
| 审计日志 | "考虑vs执行" | 仅追加记录每候选命令、运行与否、谁审批 |
| 假设排序 | "证据评分" | Recency × specificity × graph-path length inverse × citation count |
| Slack审批卡 | "HITL门" | 带修复按钮的交互Slack消息; 智能体不可进展至人工点击 |
| 遥测引用 | "证据指针" | 支持claim的Prometheus query、Loki selector、或Tempo trace URL |
| MTTR | "解决时间" | 告警触发到SLO恢复wall-clock |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 canonical参考
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird语义监控](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO-first产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP服务框架
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated修复目标