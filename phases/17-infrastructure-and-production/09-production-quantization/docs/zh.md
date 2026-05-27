# 生产量化——AWQ、GPTQ、GGUF K-quant、FP8、MXFP4/NVFP4

> 量化格式非通用选——硬件、服务引擎、负载函数。GGUF Q4_K_M或Q5_K_M CPU和edge、llama.cpp和Ollama发。GPTQ vLLM内赢需同基多LoRA。AWQ Marlin-AWQ kernel 7B类模型~741 tok/s INT4最佳Pass@1——2026数据中心生产默认。FP8 Hopper、Ada、Blackwell中层——近无损广支持。NVFP4和MXFP4(Blackwell微缩)激进需每块验证。两陷阱咬队：校准数据集必须匹配部署域、KV cache权重量化分离——AWQ lesson"我模型4 GB今"忘生产批大小10-30 GB KV cache。

**类型:** 学习
**语言:** Python(stdlib、玩具格式内存吞吐比)
**前置要求:** 阶段10课程13(量化基础)、阶段17课程04(vLLM Serving Internals)
**时间:** ~75分钟

## 学习目标

- 命名六生产量化格式和2026甜点。
- 格式选给定硬件(CPU vs GPU、Hopper vs Blackwell)、引擎(vLLM、TRT-LLM、llama.cpp)、负载(常规聊天、推理、多LoRA)。
- 算选格式权重内存省和KV cache未触。
- 命名校准数据集坑域流量量化模型降。

## 问题背景

量化减内存和HBM带宽、解码需。FP16 70B模型140 GB权重。权重量化INT4(AWQ或GPTQ)模型35 GB——一H100装KV cache空间、因128并发序列2k上下文、KV cache单20-30 GB。

但量化非免费。激进量化降质、尤其推理重任务。不同格式不同引擎工。不同硬件原生支持不同精度。2026格式动物园实、你不能抄他人选——你必须基于栈挑。

## 概念讲解

### 六格式

| 格式 | Bits | 甜点 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、edge、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM多LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心GPU生产 | vLLM(Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell多用户 | TRT-LLM |

### GGUF——CPU/edge默认

GGUF文件格式非量化方案本身——它包K-quant变体(Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0)一容器。Q4_K_M和Q5_K_M生产默认——近BF16质量4-5 bits。CPU或edge服务最佳因llama.cpp最快CPU推理引擎。

vLLM吞吐损：7B ~93 tok/s——格式非GPU kernel优化。部署目标CPU/edge用GGUF。非GPU数据中心。

### GPTQ——vLLM多LoRA

GPTQ后训练量化算法带校准通过。Marlin kernel GPU快(2.6x速非Marlin GPTQ)。7B ~712 tok/s。

独赢：GPTQ-Int4 vLLM支持LoRA adapter。若你服基座模型加10-50微调变体(每作LoRA)、GPTQ是路。NVFP4 2026初不支持LoRA。

### AWQ——数据中心GPU默认

激活感知权量化。量化时保护~1%最显著权重。Marlin-AWQ kernel：朴素10.9x速。7B ~741 tok/s、INT4格式最佳Pass@1。

新GPU服务选AWQ除非需多LoRA(GPTQ)或激进Blackwell FP4(NVFP4)。

### FP8——可靠中层

8-bit浮点。近无损。广支持。Hopper Tensor Core原生加速FP8。Blackwell继承。FP8是2026安全默认当质量不可商量(推理、医疗、代码生成)。内存省INT4半但质量风险远低。

### MXFP4 / NVFP4——Blackwell激进

微缩FP4。每块权重有自己scale因子。激进但Blackwell Tensor Core硬件加速。FP8每token字节减半——阶段17课程07经济赢。

警告：
- 2026初LoRA支持未。
- 推理重负载质量降可见。
- 每模型评估集验证。

### 校准陷阱

AWQ和GPTQ需校准数据集——典型C4或WikiText。域模型(代码、医疗、法律)、通用网络文本校准让算法错保护权重决策。HumanEval Pass@1可降数点。

修复：域数据校准。数百域样本常够。发前评估集测。

### KV cache陷阱

AWQ缩权重4 bits。KV cache分离FP16/FP8留。70B模型AWQ：

- 权重：~35 GB(INT4从140 GB)。
- KV cache 128并发×2k上下文：~20 GB。
- 激活：~5 GB。
- 总：~60 GB——H100 80GB装。

朴素"我量化模型4 GB"忘另30-50 GB。HBM预算整体。

分离、KV cache量化(FP8 KV或INT8 KV)是不同选择自有权衡——它直接影响attention精度非免费赢。

### AWQ INT4推理危险

思维链、数学、长上下文代码生成——这些激进量化显损。AWQ INT4 MATH降~3-5点。推理重负载、发FP8或BF16；接受内存成本。

### 2026选指南

- CPU/edge服务：GGUF Q4_K_M。完。
- GPU服务、常规聊天、无LoRA：AWQ。
- GPU服务、多LoRA：GPTQ Marlin。
- 推理负载：FP8。
- Blackwell数据中心、验证质量：NVFP4 + FP8 KV。
- 模糊：每候选格式跑1,000样本评估。

## 使用

`code/main.py`算模型大小范围六格式内存footprint(权重+KV+激活)和相对吞吐。示KV cache主导处、权重压缩付处、和FP8安全选处。

## 交付成果

本lesson产`outputs/skill-quantization-picker.md`。给硬件、模型大小、负载类型、质量容忍、选格式产校准/验证计划。

## 练习题

1. 跑`code/main.py`。70B模型128并发2k上下文、算每格式总HBM。哪格式一H100 80GB装？
2. 有7B编码模型。选格式论证。若质量容忍错、恢复路径何？
3. 算医疗域模型AWQ校准数据集需大小。为何更多数据非总更好？
4. 读Marlin-AWQ kernel论文或发布笔记。三句解释为何AWQ 7B 741 tok/s而原始GPTQ ~712。
5. 何何时AWQ权重配FP8 KV cache合理vs KV留BF16？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GGUF | "llama.cpp格式" | 文件格式包K-quant变体；CPU/edge默认 |
| Q4_K_M | "Q4 K M" | 4-bit K-quant中等；生产GGUF默认 |
| GPTQ | "gee pee tee q" | 后训练INT4校准；vLLM支持LoRA |
| AWQ | "a w q" | 激活感知INT4；Marlin kernel；INT4最佳Pass@1 |
| Marlin kernel | "快INT4 kernel" | Hopper INT4定制CUDA kernel；10x速 |
| FP8 | "8-bit浮点" | Hopper/Ada/Blackwell安全精度默认 |
| MXFP4 / NVFP4 | "微缩四" | Blackwell 4-bit FP每块scale因子 |
| 校准数据集 | "校数据" | 选量化参数输入文本；必须匹配域 |
| KV cache量化 | "KV INT8" | 权重分离选择；影响attention精度 |

## 延伸阅读

- [VRLA Tech — LLM量化2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 比较基准。
- [Jarvis Labs — vLLM量化完全指南](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 格式吞吐数。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 格式逐格式选。
- [vLLM文档 — 量化](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持格式和标志。
- [AWQ论文(arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原AWQ公式。
- [GPTQ论文(arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原GPTQ公式