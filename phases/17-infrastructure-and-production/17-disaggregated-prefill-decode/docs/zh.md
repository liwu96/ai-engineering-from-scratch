# 解耦Prefill/Decode——NVIDIA Dynamo和llm-d

> Prefill计算绑；解码内存绑。同GPU跑双浪费一资源。解耦分到分离池KV cache间传NIXL (RDMA/InfiniBand或TCP fallback)。NVIDIA Dynamo (GTC 2025公告、1.0 GA)居vLLM/SGLang/TRT-LLM上——Planner Profiler + SLA Planner自动速率匹配prefill:decode比达SLO。NVIDIA发布吞吐增益 ballpark——developer.nvidia.com (2025-06)示DeepSeek-R1 MoE GB200 NVL72 + Dynamo中等延迟范围 ~6x改进、Dynamo产品页(developer.nvidia.com、未注日期)宣传GB300 NVL72 + Dynamo比Hopper MoE吞吐高达50x。"30x"数是全栈Blackwell + Dynamo + DeepSeek-R1报告社区聚合；我们未找到单首源声确30x、视作方向性主张。llm-d (Red Hat + AWS) Kubernetes原生：prefill / decode / router独立Service每角色HPA。llm-d 0.5加层级KV卸载、cache感知LoRA路由、UCCL网络、缩零。经济：多客户披露内部rollup建议$2M级推理支出30–40%省(即$600-800K/年) 同位serving换解耦Dynamo恒定SLA；确$2M→$600-800K数是内复合、非单发案例——用作量级锚、非引用。短提示(<512 tokens、短输出)不值传成本。

**类型:** 学习
**语言:** Python(stdlib、玩具解耦vs同位模拟器)
**前置要求:** 阶段17课程04(vLLM Serving)、阶段17课程08(推理指标)
**时间:** ~75分钟

## 学习目标

- 解释为何prefill和解码不同优GPU分配并量化同位浪费。
- 图解耦架构：prefill池、解码池、NIXL KV传、router。
- 命名解耦不划算条件(短提示、短输出)。
- 区NVIDIA Dynamo (栈上)和llm-d (Kubernetes原生)并匹配每运维上下文。

## 问题背景

跑Llama 3.3 70B 8 H100s。混负载(长提示+短输出)、GPU解码闲因多算prefill。不同负载(短提示+长输出)、反。同位prefill + decode意双过供。

预算影响：20-40% GPU时间浪费错资源。买H100算跑内存绑解码、或买H100 HBM带宽跑计算绑prefill。双贵浪费。

解耦分prefill和解码到分离池每瓶颈尺寸。KV cache prefill池到解码池传高带宽互连。

## 概念讲解

### 为何瓶颈异

**Prefill**——全输入提示一forward跑transformer。矩阵乘主导；计算绑。H100 FP8给~2000 TFLOPS有用吞吐。批效率好——一forward处理多token。

**Decode**——一token一迭代、每迭代读全权重。内存带宽绑。HBM3给~3 TB/s。批效率高并发才好——权重读batch amortize。

同位：买双优GPU。H100双好但同价。规模、prefill池H100 / 计算重；解码池H200 / 内存重、或激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL是NVIDIA跨节点传。RDMA/InfiniBand可用、否则TCP fallback。传延迟真——典型20-80 ms 4K-token提示70B FP8 KV cache。这是为何短提示不值解耦：传税超省。

### Dynamo vs llm-d

**NVIDIA Dynamo** (GTC 2025公告、1.0 GA)：
- 居vLLM、SGLang、TRT-LLM上orchestrator。
- Planner Profiler测负载、SLA Planner自动配prefill:decode比。
- Rust核心、Python扩展。
- 吞吐增益：NVIDIA报告DeepSeek-R1 MoE GB200 NVL72 + Dynamo中等延迟范围 6x(developer.nvidia.com, 2025-06)；社区"高达30x"全Blackwell + Dynamo + DeepSeek-R1栈无单首源、视方向性。
- GB300 NVL72 + Dynamo：Dynamo产品页MoE吞吐比Hopper高达50x(developer.nvidia.com、未注日期)。

**llm-d** (Red Hat + AWS、Kubernetes原生)：
- Prefill / decode / router独立Kubernetes Service。
- 每角色HPA队列深度(prefill) / KV利用(decode)信号。
- `topologyConstraint packDomain: rack`打包prefill+decode clique同rack高带宽KV传。
- llm-d 0.5 (2026)：层级KV卸载、cache感知LoRA路由、UCCL网络、缩零。

Dynamo若要托管栈上orchestrator。llm-d若要Kubernetes原生原语并CNCF生态。

### 经济

内复合(非单发案例——量级锚)：

- $2M/年推理支出同位serving。
- 换解耦Dynamo。
- 同请求量、同P99延迟SLA。
- 报告省：$600K–$800K/年(30–40%减)。
- 无新硬件。

我们多客户披露合而非单可引用案例；近发数据点Baseten 2x快TTFT / 61%高吞吐Dynamo KV路由(baseten.co, 2025-10)、VAST + CoreWeave 40–60% KV命中率60–130%更多token/$投(vastdata.com, 2025-12)。省来池尺寸正；prefill重负载(RAG 8K+前缀)益比平衡多。

### 何时不解耦

- 提示< 512 tokens和输出< 200 tokens：传税主导增益。
- 小集群(< 4 GPUs)：池多样性不足。
- 队不能操两GPU池每角色扩展：Dynamo帮非简单。
- 无RDMA fabric：TCP传税重。

### Router集成阶段17课程11

解耦router KV cache感知(阶段17课程11)。请求到解码池持有前缀——若无匹配、流prefill → decode。命中率和解耦复合——cache感知router定是否需新prefill。

### MoE Blackwell真数所在

GB300 NVL72 + Dynamo示MoE吞吐比Hopper基线50x。MoE expert路由prefill计算重但解码内存重(expert cache)、解耦双赢。2026前沿模型服务MoE主导(DeepSeek-V3、未来GPT-5变种)。

### 你应记数

基准数漂——NVIDIA和推理栈每季发更新结果。引用前核对。

- DeepSeek-R1 GB200 NVL72 + Dynamo：中等延迟范围吞吐~6x比基线(developer.nvidia.com, 2025-06)；社区"高达30x"全Blackwell + Dynamo栈主张是方向聚合无单首源。
- GB300 NVL72 + Dynamo：MoE吞吐比Hopper高达50x(developer.nvidia.com、未注日期)。
- 省锚(内复合、非单案例)：$600-800K/年$2M年支出恒定SLA。
- 解耦阈值：提示>512 tokens + 输出>200 tokens。
- KV NIXL传：70B FP8 4K提示KV 20-80 ms。

## 使用

`code/main.py`模同位vs解耦serving。报吞吐、每请求成本、提示长度交叉。

## 交付成果

本lesson产`outputs/skill-disaggregation-decider.md`。给负载和集群、决定解耦否。

## 练习题

1. 跑`code/main.py`。何提示长度解耦胜同位？
2. 设计RAG服务P99前缀长度8K、输出300的prefill池和解码池。
3. Dynamo vs llm-d：纯Kubernetes店无Python runtime偏好选一。
4. 算KV传成本：4K prefill 70B FP8 = ~500 MB KV。RDMA 100 GB/s传= 5 ms。TCP 10 GB/s = 50 ms。何SLA关键？
5. MoE expert路由改KV访问模式。解耦不同expert每token激活MoE行为何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 解耦serving | "分prefill/decode" | 每阶段分离GPU池 |
| NIXL | "NVIDIA传" | Dynamo跨节点KV传(RDMA/TCP) |
| NVIDIA Dynamo | "orchestrator" | vLLM/SGLang/TRT-LLM栈上协调 |
| llm-d | "Kubernetes原生" | Red Hat + AWS K8s解耦栈 |
| Planner Profiler | "Dynamo自动配" | 测负载、配池比 |
| SLA Planner | "Dynamo政策" | 自动速率匹配prefill:decode达SLO |
| `packDomain: rack` | "llm-d拓扑" | 打包prefill+decode同rack快KV |
| UCCL | "统一集体" | llm-d 0.5网络层缩零 |
| MoE expert路由 | "每token expert" | DeepSeek-V3模式；解耦益 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)