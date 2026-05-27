# 推理指标——TTFT、TPOT、ITL、Goodput、P99

> 四指标决定推理部署是否工作。TTFT是prefill加队列加网络。TPOT(等ITL)是内存绑解码每token成本。端到端延迟TTFT加TPOT乘输出长度。吞吐是跨舰队聚合每秒token。但产品重要是goodput——每请求同时满足每SLO分数。高吞吐低goodput意你处理token永不及时达用户。2026 Llama-3.1-8B-Instruct TRT-LLM参考数：平均TTFT 162 ms、平均TPOT 7.33 ms、平均E2E 1,093 ms。总报P50、P90、P99——永不仅平均。且看测量陷阱：GenAI-Perf排除TTFT ITL计算、LLMPerf含；两工具同跑TPOT不同意。

**类型:** 学习
**语言:** Python(stdlib、玩具百分位计算器和goodput报告器)
**前置要求:** 阶段17课程04(vLLM Serving Internals)
**时间:** ~60分钟

## 学习目标

- 精确定义TTFT、TPOT、ITL、E2E、吞吐、goodput并命名每测组件。
- 解释为何平均是LLM服务错统计和如何读P50/P90/P99。
- 构SLO多约束(如TTFT<500 ms AND TPOT<15 ms AND E2E<2 s)并算goodput对。
- 命名两基准工具同跑TPOT不同意并解释为何。

## 问题背景

"我们吞吐15,000 token每秒。" 所以？若40%请求爆2秒端到端、用户弃会。吞吐单不告诉你产品是否工作。

推理有多延迟轴每失败不同。Prefill计算绑提示长度扩展。解码内存绑批大小扩展。队列延迟是运维问题。网络是物理距离问题。需每不同指标、需百分位、需单复合说"用户得期望否"——即goodput。

## 概念讲解

### TTFT——首token时间

`TTFT = queue_time + network_request + prefill_time`

Prefill提示长主导。Llama-3.3-70B FP8 H100、32k提示~800 ms纯prefill。队列时间调度器负载行为。网络请求含TLS线时间。TTFT用户看任何流回前延迟。

### TPOT / ITL——间token延迟

一量多名。`TPOT`(每输出token时间)、`ITL`(间token延迟)、`每token解码延迟`——全同。是首后连续流token间时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

同Llama-3.3-70B H100栈分块prefill、TPOT平均~7 ms。无分块prefill、长prefill邻序列、TPOT可突50 ms。看P99非平均。

### E2E延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

长输出(>500 token)、E2E TPOT主导。短输出长提示、E2E TTFT主导。报输出长度条件E2E。

### 吞吐

`throughput = total_output_tokens / elapsed_time`

聚合指标。告舰队效率。不告单请求健康。

### Goodput——你实关心指标

`goodput = 每请求满足(TTFT <= a) AND (TPOT <= b) AND (E2E <= c)分数`

SLO是多约束。请求"好"仅每约束持。Goodput是份额。高吞吐60% goodput是失败。低吞吐99% goodput是目标。

2026、goodput MLPerf Inference v6.0提交和AI平台提供商内部SLA跟踪用指标。

### 为何平均是错统计

LLM延迟分布右偏。一长prefill邻解码批可发500 token TPOT ~7 ms和20 token TPOT ~60 ms。平均TPOT 9 ms。P99 TPOT 65 ms。用户常打P99——这是为何离开。

总报三元(P50、P90、P99)。用户体验、P99是你优化。

### 参考数——Llama-3.1-8B-Instruct TRT-LLM、2026

- 平均TTFT：162 ms
- 平均TPOT：7.33 ms
- 平均E2E：1,093 ms
- P99 TPOT：变10-25 ms依赖分块prefill配置。

这些发布NVIDIA参考点。模型大小变(70B会3-5x)、硬件(H100 vs B200 ~3x)、负载变。

### 测量陷阱

两最用2026基准工具同跑TPOT不同意：

- **NVIDIA GenAI-Perf**：排除TTFT ITL计算。ITL从token 2始。
- **LLMPerf**：含TTFT。ITL从token 1始。

TTFT 500 ms和100输出token 700 ms总解码请求、GenAI-Perf报`ITL = 700/99 = 7.07 ms`、LLMPerf报`ITL = 1200/100 = 12.00 ms`。工具选择改数。

总声明工具。总发布定义。

### 构SLO

合理面向消费者70B聊天模型2026 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s <300 token输出。
- Goodput目标 >= 99%。

企业SLO紧TTFT(200-400 ms)松E2E。要点写下、测三者、追踪goodput单复合。

### 如何测量

- 跑真实流量或现实合成(LLMPerf `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`)。
- 峰并发2x基准跑目标。
- 跑30-50迭代、取合并样本百分位。
- 发布工具名、工具版、模型、硬件、并发、提示分布。

## 使用

`code/main.py`玩具goodput计算器。生成合成延迟分布、应用SLO、算goodput。示GenAI-Perf vs LLMPerf同trace TPOT差异。

## 交付成果

本lesson产`outputs/skill-slo-goodput-gate.md`。给负载和SLO、产CI/CD就基准配方部署gate goodput而非吞吐。

## 练习题

1. 跑`code/main.py`。生成1%尾突分布。P99 TPOT从30 ms紧15 ms时goodput如何变？
2. 供应商引"Llama 3.3 70B H100 15,000 tok/s"。命名信前三问。
3. 为何分块prefill保P99 TPOT不保平均TPOT？
4. 构语音助手消费者SLO(首token听非读)。哪指标最用户可见？
5. 读LLMPerf README和GenAI-Perf文档。识别三其他指标工具不同意。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| TTFT | "首token时间" | 队列+网络+prefill；长提示prefill主导 |
| TPOT | "每输出token时间" | 首后每token内存绑解码成本 |
| ITL | "间token延迟" | 多工具同TPOT(非全部——见GenAI-Perf) |
| E2E | "端到端" | TTFT + TPOT * output_len；响应侧网络加 |
| 吞吐 | "tok/s" | 舰队效率；无延迟百分位无用 |
| Goodput | "SLO满足率" | 每请求同时满足每SLO约束分数 |
| P99 | "尾" | 100中1最差延迟；用户体验指标 |
| SLO多约束 | "联合" | 三延迟界AND；任一违反请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具不同意ITL含TTFT否 |

## 延伸阅读

- [NVIDIA NIM — LLM基准指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT canonical定义。
- [Anyscale — LLM服务基准指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替定义测量食谱。
- [BentoML — LLM推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真部署应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray基开源基准。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA基准工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业接受goodput基基准