# Kubernetes GPU自动扩展——Karpenter、KAI Scheduler、Gang Scheduling

> 三层非一。Karpenter动态供节点(一分钟内、比Cluster Autoscaler快40%)。KAI Scheduler处理gang调度、拓扑感知、和分层队列——它防7-of-8部分分配陷阱七节点等烧钱缺一GPU。应用级自动扩展器(NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler)推理特定信号扩展——队列深度、KV cache利用——非CPU/DCGM占空比。经典HPA陷阱是`DCGM_FI_DEV_GPU_UTIL`是占空比测量：100%可是10请求或100。vLLM预分配KV cache内存，所以内存永不触发缩。本lesson教你组三层避默认Karpenter `WhenEmptyOrUnderutilized`政策推理中终止运行GPU任务。

**类型:** 学习
**语言:** Python(stdlib、玩具队列深度自动扩展模拟器)
**前置要求:** 阶段17课程02(推理平台经济)、阶段17课程04(vLLM Serving Internals)
**时间:** ~75分钟

## 学习目标

- 图三层自动扩展(节点供、gang调度、应用级)并命名每层用工具。
- 解释为何`DCGM_FI_DEV_GPU_UTIL`是vLLM错HPA信号并命名两替换(队列深度、KV cache利用)。
- 描述gang调度和KAI Scheduler防部分分配失败模式(7 of 8 GPU空)。
- 命名Karpenter合并政策(`WhenEmptyOrUnderutilized`)终止运行GPU任务并陈述2026安全替代。

## 问题背景

队Kubernetes部署LLM服务。设HPA`DCGM_FI_DEV_GPU_UTIL`信号。服务工作时间钉100%利用。HPA永扩展——已认满。手加副本；TTFT降。HPA仍不扩。信号骗你。

另，用Cluster Autoscaler节点。1M token提示凌晨2点到；集群3分钟供节点、请求超时。

再另，部署70B模型需8 GPU跨2节点。集群7 GPU自由1散三节点。Cluster Autoscaler为缺1 GPU供节点。七节点等4分钟烧钱Kubernetes起最后GPU。

三层三不同失败模式。2026 GPU感知自动扩展非"开HPA"。是组节点供、gang调度、和应用信号自动扩展。

## 概念讲解

### 层1——节点供(Karpenter)

Karpenter观察待pod一分钟内供节点(~45-60s vs Cluster Autoscaler GPU节点~90-120s)。它动态按`NodePool`约束选实例类型——若pod需8 H100集群无匹配节点，Karpenter直接供一而非扩展现有组。

**合并陷阱**：Karpenter默认`consolidationPolicy: WhenEmptyOrUnderutilized`对GPU池危险。它会终止运行GPU节点迁移pod到更便宜合适实例。推理负载意味着驱逐运行请求并在新节点重加载70B模型。损是分钟容量加请求失败。

GPU池安全设：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让Karpenter一小时后合并真空节点但永不驱逐运行任务。

### 层2——gang调度(KAI Scheduler)

KAI Scheduler(项目"Karp"后改名)处理默认kube-scheduler不：

**Gang调度**——全或无调度。需8 GPU分布式推理pod要么8全起要么都不。无此，得部分分配陷阱：7 of 8 pod起、无限等、烧钱。

**拓扑感知**——知哪GPU共享NVLink、哪同机架、哪间InfiniBand。据此放pod。DeepSeek-V3 67B张量并行负载必须在一NVLink域；KAI Scheduler尊此。

**分层队列**——多队争同GPU池带优先级配额。队A生产紧缺被队B训练任务抢占仅若优先级规则允。

KAI部署旁kube-scheduler作第二调度器；注解工作负载用它。Ray和vLLM production-stack都集成。

### 层3——应用级信号

**HPA陷阱**：`DCGM_FI_DEV_GPU_UTIL`是占空比指标——测GPU每采样间隔是否工作。100%利用可意10并发请求或100；GPU任忙。占空比扩展是盲目扩展。

更糟，vLLM类似引擎预分配KV cache内存(达`--gpu-memory-utilization`)。内存用90%甚至一请求。基于内存HPA永不缩。

**2026替换信号**：

- 队列深度(等prefill请求数)。
- KV cache利用(活跃序列分配块比例)。
- 每副本P99 TTFT(SLA信号)。
- Goodput(每秒满足所有SLO请求数)。

NVIDIA Dynamo Planner和llm-d Workload Variant Autoscaler消费这些信号扩展副本。它们完全替LLM服务HPA。

### 何时用什么

| 扩决策 | 工具 |
|--------|------|
| 加/删节点 | Karpenter |
| 调多GPU任务 | KAI Scheduler |
| 加/删副本 | Dynamo Planner / llm-d WVA(或队列深度自定义HPA) |
| 选GPU类型 | Karpenter NodePool |
| 抢低优先级 | KAI Scheduler队列 |

### 解耦prefill/decode复杂一切

若你跑解耦prefill/decode(阶段17课程17)，你有两pod类不同扩展触发：prefill pod队列深度扩展、decode pod KV cache压力扩展。llm-d暴露这些为分`Services`每角色HPA。勿试单HPA前两者。

### 冷启动此处重要

冷启动缓解(阶段17课程10)是节点供时间变用户可见处。Karpenter 45-60秒预热加20GB模型加载加引擎初始化意从零请求2-5分钟。SLO关键路径保持热池(`min_workers=1`)、或应用层Modal风格检查点。

### 你应记住数

- Karpenter节点供：~45-60s vs Cluster Autoscaler ~90-120s(GPU节点)。
- KAI Scheduler防部分分配浪费——7-of-8陷阱。
- `DCGM_FI_DEV_GPU_UTIL`作HPA信号：断；用队列深度或KV利用。
- Karpenter `WhenEmptyOrUnderutilized`：终止运行GPU任务。推理用`WhenEmpty + consolidateAfter: 1h`。

## 使用

`code/main.py`模拟三层自动扩展突发GPU负载。比朴素HPA(占空比)、队列深度HPA、和KAI-gang调度扩展。报告未满足请求、空GPU分钟、和综合分。

## 交付成果

本lesson产`outputs/skill-gpu-autoscaler-plan.md`。给集群拓扑、负载形状、SLO、设计三层自动扩展计划。

## 练习题

1. 跑`code/main.py`。突发负载下朴素占空比HPA丢队列深度HPA捕获多少请求？差异从哪来？
2. 设计集群服Llama 3.3 70B FP8 H100 SXM5 Karpenter NodePool。指定`capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`、和污点保持非GPU工作负载离这些节点。
3. 队报告部署Pending卡因"GPU可用但pod不调度"。诊断——Karpenter、kube-scheduler、或KAI Scheduler？哪些指标确认？
4. 选信号自动扩展解耦prefill pod和解码pod不同信号。论证两者。
5. 计算`WhenEmptyOrUnderutilized`合并陷阱成本24x7生产服务平均60请求丢弃事件/天P99 TTFT > 10s。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Karpenter | "节点供器" | Kubernetes节点自动扩展器；亚分钟供 |
| Cluster Autoscaler | "老扩展器" | Kubernetes节点自动扩展器前身；更慢、组基 |
| KAI Scheduler | "GPU调度器" | 二级调度器gang+拓扑+队列 |
| Gang调度 | "全或无" | 原子调度N pod或全推迟 |
| 拓扑感知 | "机架感知" | 基NVLink/IB/机架放pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU利用" | 占空比指标；非LLM扩展信号 |
| 队列深度 | "等请求" | 正确prefill绑扩展HPA信号 |
| KV cache利用 | "内存压力" | 正确decode绑扩展HPA信号 |
| 合并 | "Karpenter合并" | 终止节点到更便宜实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐运行GPU任务政策 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置例。
- [Karpenter中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并政策语义和GPU安全默认。
- [NVIDIA — Kubernetes解耦LLM推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner扩展信号。
- [Ray文档 — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray集成模式。
- [AWS EKS计算和自动扩展最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管Kubernetes特定指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler设计