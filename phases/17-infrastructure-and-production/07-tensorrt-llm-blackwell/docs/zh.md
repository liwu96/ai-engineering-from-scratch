# Blackwell上TensorRT-LLM与FP8和NVFP4

> TensorRT-LLM仅NVIDIA但Blackwell赢。GB200 NVL72带Dynamo编排、SemiAnalysis InferenceX测2026 Q1-Q2 120B模型$0.012每百万token、对H100+vLLM $0.09/M——7x经济gap。栈是三浮点 regime复合：FP8仍KV cache和attention kernel关键因动态范围需；NVFP4(4-bit microscaling)处理权重和激活；多token预测(MTP)和解耦prefill/decode加另2-3x顶。Day-0模型支持直载FP4权重无后训练转换。2026工程队捕获：TRT-LLM是闭NVIDIA栈、采用换可移植吞吐。混模型硬件跑数学前commit。

**类型:** 学习
**语言:** Python(stdlib、玩具FP8/NVFP4内存成本计算器)
**前置要求:** 阶段17课程04(vLLM Serving Internals)、阶段10课程13(量化)
**时间:** ~75分钟

## 学习目标

- 解释为何FP8仍KV cache和attention关键即使权重NVFP4。
- 算前沿模型BF16、FP8、NVFP4下HBM footprint推理省从哪。
- 命名Blackwell特定特性TRT-LLM用(day-0 FP4、MTP、解耦服务、all-to-all原语)。
- 决定何时TRT-LLM NVIDIA锁值7x成本gap vs Hopper vLLM。

## 问题背景

2026推理经济学前沿是"每美元多少token"。答案依赖四栈选：硬件代(Hopper H100/H200 vs Blackwell B200/GB200)、精度(BF16 → FP8 → NVFP4)、服务引擎(vLLM vs SGLang vs TRT-LLM)、编排(朴素vs解耦vs Dynamo)。

Hopper vLLM、120B MoE跑~$0.09每百万token。Blackwell TRT-LLM + Dynamo、同模型跑~$0.012——7x便宜。些gap硬件(Blackwell每GPU LLM吞吐11-15x Hopper)。些栈：FP4权重、MTP draft、解耦prefill/decode、和NVLink 5 MoE专家通信all-to-all。

不能在NVIDIA栈外复现。这是权衡——可移植换经济。理解哪栈选给哪gap份额是本lesson要点。

## 概念讲解

### 为何FP8仍是KV cache基线

2026常见错：假设NVFP4适用处处。不。KV cache需FP8(8-bit浮点)因存attention key和value跨宽动态范围。KV量化FP4致灾难精度损——分布尾截attention分崩溃。FP8指数位给KV cache需范围。

NVFP4(2025-2026)适用权重和激活。微缩放：每块权重有自己scale因子小块跨不同动态范围无损每tensor scale。激活、FP4撑因激活层内小范围。

典型Blackwell配：

- 权重：NVFP4(4-bit microscaling)。
- 激活：NVFP4。
- KV cache：FP8。
- Attention accumulator：FP32(softmax稳)。

### TRT-LLM用Blackwell特定原语

- **Day-0 FP4权重**：模型提供商直发FP4权重；TRT-LLM载无后训练转换。无AWQ / GPTQ步FP4。
- **多token预测(MTP)**：同EAGLE想法(阶段17课程05)但集成TRT-LLM build。
- **解耦服务**：prefill和decode分GPU池、KV cache NVLink或InfiniBand传。同Dynamo想法(阶段17课程20)。
- **All-to-all通信原语**：NVLink 5 MoE专家通信延迟切3x vs Hopper。TRT-LLM MoE kernel为此调。
- **NVFP4 + MXFP8微缩放**：Blackwell Tensor Core硬件加速scale因子处理。

### 你应记数

- HGX B200 GPT-OSS-120B via TRT-LLM $0.02/M token。
- GB200 NVL72 via Dynamo(编排TRT-LLM) $0.012/M token。
- H100 + vLLM ≈ $0.09/M token可比负载。
- TRT-LLM更新三月2.8x吞吐增益(2026)。
- Blackwell vs Hopper每GPU LLM吞吐11-15x。
- MLPerf Inference v6.0(2026年4月)：Blackwell主每提交任务。

### FP4质量实成本

NVFP4激进。推理重负载(思维链、数学、长上下文代码生成)、FP4权重显降。每块校准缓但不消。队发推理模型常FP8权重+FP4激活妥协、或H200 FP8全。

规则：always在评估集验证任务质量前commit NVFP4权重。

### 为何这是NVIDIA锁决策

TRT-LLM是C++ + CUDA +闭源kernel。模型需特定GPU SKU编译。无AMD、无Intel、无ARM。若你infra策略多供应商、TRT-LLM TRT-LLM服层非启——你仍可混硬件vLLM服。若NVIDIA独、7x gap付锁。

### 2026实用食谱

$100M+年推理账、Hopper + vLLM留7-10x桌上。迁成本主导负载Blackwell + TRT-LLM + Dynamo。保持实验层H100 + vLLM模型迭代速。每NVFP4转换模型生产前验证质量。

### 解耦bonus

TRT-LLM解耦服务(分prefill和decode池)阶段17课程20深覆。Blackwell、乘法器栈：FP4权重 × MTP加速 × 解耦放 × cache感知路由。7x数假设全栈。

## 使用

`code/main.py`算HBM footprint、解码吞吐(内存绑regime)、和$/M-token模型跨三栈：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。跑见复合效应和每改贡献gap份额。

## 交付成果

本lesson产`outputs/skill-trtllm-blackwell-advisor.md`。给负载、模型大小、年token量、决定Blackwell + TRT-LLM栈值NVIDIA锁。

## 练习题

1. 跑`code/main.py`。120B MoE 30%活跃参数、算H100 BF16、H100 FP8、B200 NVFP4/FP8内存带宽限解码吞吐。最大跳从哪来？
2. 客户$2M/年H100 + vLLM。7x经济gap12月摊Blackwell + TRT-LLM迁移需多少Blackwell GPU？
3. NVFP4权重转换MATH精度降3点。命名两恢复路径：一质量优先(保FP8权重)、一成本优先(域数据校准)。
4. 读MLPerf v6.0推理结果。哪任务Blackwell-Hopper gap最小为何？
5. 算405B模型NVFP4权重+FP8 KV cache 128k上下文HBM需。单GB200 NVL72节点装？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| FP8 | "8-bit浮点" | 8-bit浮点；KV cache和attention用因动态范围 |
| NVFP4 | "4-bit微缩" | NVIDIA 4-bit微缩FP格式；Blackwell权重和激活 |
| MXFP8 | "MX八" | 微缩FP8变体；Blackwell Tensor Core硬件加速 |
| Day-0 FP4 | "发FP4权重" | 模型提供商权重FP4发；无后训练转换步 |
| MTP | "多token预测" | TRT-LLM集成投机解码draft(阶段17课程05) |
| 解耦服务 | "分prefill/decode" | Prefill和decode分GPU池；KV NVLink/IB传 |
| All-to-all | "MoE专家通信" | Token路由专家GPU通信模式；NVLink 5切3x |
| InferenceX | "SemiAnalysis推理bench" | 2026行业接受每token成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026年4月MLPerf结果。
- [NVIDIA — MoE推理Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all和MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM上解耦编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布Blackwell数基准套