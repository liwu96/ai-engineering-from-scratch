# 无服务器LLM冷启动缓解

> 20 GB模型镜像5-10分钟(7B)到20+分钟(70B)从冷到服务。真无服务器世界、非预热——是停运。缓解五层操作：预种子节点镜像(Bottlerocket AWS、双卷架构)、模型流(NVIDIA Run:ai Model Streamer、vLLM原生)、GPU内存快照(Modal checkpoint、10x快重启)、热池(`min_workers=1`)、分层加载(ServerlessLLM NVMe→DRAM→HBM管道、10-200x延迟减)、和活迁移移输入token(KB)而非KV cache(GB)。Modal发布2-4s冷启动底线；Baseten 5-10s默认、预热亚秒。本lesson教你测、预算、栈五层。

**类型:** 学习
**语言:** Python(stdlib、玩具冷启动路径模拟器)
**前置要求:** 阶段17课程02(推理平台经济)、阶段17课程03(GPU自动扩展)
**时间:** ~60分钟

## 学习目标

- 列举冷启动缓解五层并命名每层工具或模式。
- 算70B模型总冷启动时间(节点供)+(权重下载)+(权重载HBM)+(引擎初始化)和。
- 解释为何活迁移传输入token(KB)非KV cache(GB)及代价何(重计算)。
- 命名热池权衡(付空闲GPU或接受冷启动尾)和`min_workers > 0`强制SLA阈值。

## 问题背景

无服务器LLM端点夜缩零。晨8点流量突。首请求等：

1. Karpenter供GPU节点：45-60s。
2. 容器拉30 GB镜像权重：120-300s。
3. 引擎载权重入HBM：45-120s依赖模型大小和存储速。
4. vLLM或TRT-LLM初始化CUDA graph、KV cache池、tokenizer：10-30s。

总：220-510s(约3-8分钟)首token回。SLA是2s。你发热池(`min_workers=1`)问题似消失——但现付一空闲GPU 24x7。若你服务5产品每一热副本、5 × 24 × 30 = 3,600 GPU小时/月无论无单用户调。

冷启动缓解是如何保无服务器经济同时接近常开延迟。

## 概念讲解

### 层1——预种子节点镜像(Bottlerocket)

AWS、Bottlerocket双卷架构分OS数据。快照数据卷容器镜像预拉；引用快照ID `EC2NodeClass`。新节点权重已本地NVMe启——步2和步3部分消。Karpenter原生工。典型省：大模型每冷启动2-4分钟。

GCP等效：自定义VM镜像预baked容器层。Azure：托管磁盘快照同模式。

### 层2——模型流(Run:ai Model Streamer)

非答首请求前载全文件、权重层层流入GPU内存首transformer块驻即开始处理。NVIDIA Run:ai Model Streamer vLLM 2026原生发。工S3、GCS、本地NVMe。大模型权重载时间约半重叠I/O计算设置。

### 层3——GPU内存快照(Modal)

Modal首载后取GPU状态快照(权重、CUDA graph、KV cache区)。后续重启直反序列入HBM——10x快重初始化。这是最近"2秒启热GPU"。权衡：快照每GPU拓扑特定、若Karpenter迁你不同SKU、重快照。

### 层4——热池(min_workers=1)

最简缓解：保持一副本总就绪。成本是一GPU每小时率24x7。算术小模型残酷(付$0.85-$1.50/hr避30s冷启动)大模型友善(付$4/hr避5分钟冷启动)。热池强制SLA阈值：典型70B+模型TTFT P99 < 60s。

### 层5——分层加载(ServerlessLLM)

ServerlessLLM视存储层次：NVMe(快大)、DRAM(中等分层)、HBM(小即)。权重预载DRAM；按需入HBM。论文报告冷载朴素磁盘到HBM 10-200x延迟减。生产采用早但vLLM集成存。

### 层6——活迁移(bonus模式)

节点变不可(spot驱逐、node drain)、传统模式冷启另副本排空请求队列。活迁移移输入token(千字节)到有模型加载目的地并目的地重算KV cache。重算比网络传GB KV cache便宜。适用于解耦部署。

### 热池数学

P99 TTFT SLA 2s服务、问题非"热池是否"而是"多少热副本、哪路径得"。

- 高值交互路径(实时聊天、语音Agent)：`min_workers=1-2`。
- 后台批路径(夜分类)：缩零接受、5-10分钟冷启动容忍。
- 高级层：每租户`min_workers`专用容量。

### 优化前测量

70B模型新节点冷启动解剖(示)：

| Phase | 时间 | 缓解 |
|-------|------|-----------|
| 节点供 | 50s | Bottlerocket + 预种子镜像、热池 |
| 镜像拉 | 180s | 预种子数据卷(消) |
| 权重到HBM | 75s | 模型流(半)；GPU快照(消) |
| 引擎初始化 | 20s | 持CUDA graph cache |
| 首前向 | 3s | 最小固有延迟 |
| **总冷** | **328s** | |
| **缓解后总** | **~15s** | 22x减 |

### 你应记数

- Modal冷启动：2-4s(GPU快照)。
- Baseten默认冷启动：5-10s；预热亚秒。
- 原始70B冷启动：3-8分钟。
- Run:ai Model Streamer：~2x权重载加速。
- ServerlessLLM分层加载：10-200x延迟减(论文数)。

## 使用

`code/main.py`建模冷启动路径带不带每缓解。报总冷启动时间、热池成本、和热池自付盈亏请求率。

## 交付成果

本lesson产`outputs/skill-cold-start-planner.md`。给SLA、模型大小、流量形状、选缓解栈。

## 练习题

1. 跑`code/main.py`。算热副本比付冷启动税额外请求丢SLO便宜盈亏平衡请求率。
2. 部署13B模型P99 TTFT SLA 3s。选达最小缓解栈(最少层)。
3. Bottlerocket预种子消镜像拉但权重仍快照到HBM载。7 GB/s快照后NVMe读70B模型算墙钟。
4. 无服务器提供商GPU快照(Modal)队拒因"快照漏PII。"论证双边——真实风险何、缓解何(临时快照、加密、命名空间隔离)？
5. 设计分层热池政策：付费用户、试用用户、批负载多少热副本？示数学。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 冷启动 | "大暂停" | 新副本请求到首token时间 |
| 热池 | "常开最小" | `min_workers >= 1`保至少一副本就绪 |
| 预种子镜像 | "baked AMI" | 容器权重预驻节点镜像 |
| Bottlerocket | "AWS节点OS" | AWS容器优化OS双卷快照支持 |
| 模型流 | "流载" | 重叠权重I/O计算设置 |
| GPU快照 | "checkpoint到HBM" | 序列化后载GPU状态；重启反序列 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储层层次；按需载 |
| 活迁移 | "移token" | 传输入(KB)、目的地重算KV |
| `min_workers` | "热副本" | 无服务器最小保活数 |
| 缩零 | "全无服务器" | 空无成本；接受全冷启动税 |

## 延伸阅读

- [Modal — 冷启动性能](https://modal.com/docs/guide/cold-start) — Modal发布基准和checkpoint架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预种子数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 重叠权重载计算设置。
- [Baseten — 冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预热playbook。
- [ServerlessLLM论文(USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Kubernetes解耦LLM推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 解耦部署活迁移