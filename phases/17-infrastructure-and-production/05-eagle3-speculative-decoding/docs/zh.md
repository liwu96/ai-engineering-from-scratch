# 生产EAGLE-3投机解码

> 投机解码配快draft模型和目标模型。Draft提K token；目标单前向验证；接受token免费。2026，EAGLE-3是生产级变体——它训练draft head目标模型隐状态而非原始token，推接受率alpha入通用聊天0.6-0.8带。正确问题非"draft多快"而是"我的流量alpha什么？"若alpha落到~0.55下，投机解码高并发净负因每拒draft花第二目标前向通过。本lesson教你先测alpha后翻标志。

**类型:** 学习
**语言:** Python(stdlib、玩具接受率模拟器)
**前置要求:** 阶段17课程04(vLLM Serving Internals)、阶段10课程18(多Token预测)
**时间:** ~60分钟

## 学习目标

- 命名投机解码三代并解释EAGLE-3从EAGLE-2和经典draft模型变什么。
- 定义接受率alpha、从alpha和K(draft长度)算预期加速、并识别目标并发平衡alpha。
- 解释为何投机解码vLLM 2026选入(非默认)为何无测alpha翻它是生产反模式。
- 写测量计划：哪个基准、哪个提示分布、哪个并发点、哪个指标门槛。

## 问题背景

解码内存绑。H100跑Llama 3.3 70B FP8，每解码token读~140 GB/s权重发一token。GPU计算解码近空闲——瓶颈HBM带宽非matmul吞吐。

投机解码利用gap。用便宜draft模型生成K候选token，然后目标模型单前向验证全K。每验证token有效免费(摊入目标本得批-K前向)。

经典draft模型方法用同族更小模型(Llama 3.2 1B为Llama 3.3 70B drafting)。工作但接受率平庸——更小模型分布离目标。EAGLE、然后EAGLE-2、然后EAGLE-3直接目标模型内部状态上轻draft head训练，所以draft分布紧追踪目标。这为何alpha从draft模型0.4到EAGLE-3 0.6-0.8。

捕获：EAGLE-3 vLLM 2026选入。`speculative_config`必须显设。无标志无加速。无测真实流量alpha队翻常尾延迟变差非更好。

## 概念讲解

### 投机解码实际买什么

无投机解码，每token成本是一目标前向。投机解码draft长度K和接受alpha，每目标前向预期token`1 + K * alpha`。加速`(1 + K * alpha) / (1 + epsilon)`其中epsilon是draft+验证开销。K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界数聚2-3x因alpha少那高生产流量和epsilon高批大小增。

### 为何alpha是唯一重要指标

拒token不消失——它们强第二目标前向第一个拒token。alpha落到0.4负载，付draft开销+验证+重滚。高并发(如256并发)、解码批已够大内存带宽gap"目标独"和"目标带验证"缩。最2026硬件alpha 0.55下、投机解码净负。

alpha因负载变。ShareGPT风格通用聊天、ShareGPT训练EAGLE-3打0.6-0.8。域特定流量(代码、医疗、法律)、通用数据训练draft head落到0.4-0.6。域特定draft head训练恢复alpha——它是轻快训练工作比目标微调。

### EAGLE代一览

- **经典draft模型**：同族小模型。Alpha 0.3-0.5。基础设施简单——两模型加载、draft每目标前向K前向跑。
- **EAGLE-1(2024)**：目标隐状态(最后层)训练单draft head。Alpha ~0.5-0.6。目标顶小参数开销。
- **EAGLE-2(2025)**：自适应draft长度和树基draft(单目标通过验证多分支)。Alpha ~0.6-0.7。更复杂draft调度器。
- **EAGLE-3(2025-2026)**：多目标层(不只最后)训练draft head、更好对齐。Alpha ~0.6-0.8通用聊天。

### 2026生产食谱

1. 发目标模型朴素。测基线TTFT、ITL、吞吐目标并发。
2. vLLM `speculative_config`启EAGLE-3 draft。重跑基准。
3. 记接受率alpha。vLLM V1报`spec_decode_metrics.accepted_tokens_per_request`。除请求draft长度得alpha。
4. 若生产流量分布alpha < 0.55、禁投机解码或训练域特定EAGLE-3 draft。
5. 生产并发重跑。确认P99 ITL未变差。

### 生产陷阱：P99尾

平均ITL投机解码降。P99可变差若不调。拒draft触发两通过序列(draft + verify-fail + reroll)。满批下、两通过序列化。看P99 ITL非P50。

### EAGLE-3已部署处

Google 2025 AI Overviews部署投机解码(同质量更快响应)。vLLM V1运`speculative_config`作文档接口；V1 N-gram GPU投机解码是分块prefill兼容变体。SGLang支持EAGLE-3作前缀重负载推荐draft路径。

### 平衡数学一行

预期加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设`S = 1`解alpha：`alpha_breakeven = verify_overhead / K`。典型verify_overhead ~0.15和K=5：`alpha_breakeven = 0.03`。但那是裸解码数学。高并发验证开销升解码批已跨序列摊内存读，所以有效alpha_breakeven爬到~0.45-0.55实践。

### 何时不用投机解码

- 延迟不敏批1离线生成。用朴素目标。
- 非常短输出(<50 token)。Draft开销和验证成本主导。
- 无域训练draft head专门域。Alpha太低。
- vLLM v0.18.0+ draft模型投机解码+`--enable-chunked-prefill`。此组合不编译。文档例外是V1 N-gram GPU投机解码。

## 使用

`code/main.py`模拟解码环带和不带投机解码跨alpha值范围和draft长度K。打印平衡alpha、测量加速、和尾行为。跑多(alpha、K)组合见投机解码停付处。

## 交付成果

本lesson产`outputs/skill-eagle3-rollout.md`。给目标模型、流量分布描述、和并发目标、产分EAGLE-3推出计划——基准基线、启配置、测alpha、门槛alpha >= 0.55、看P99 ITL。

## 练习题

1. 跑`code/main.py`。K=5，2x加速需何alpha？3x加速？对verify_overhead多敏？
2. 设生产流量分70%通用聊天、30%代码。通用聊天ShareGPT训练EAGLE-3 alpha 0.7；代码alpha 0.4。混合alpha何和投机解码净正？
3. 读vLLM `speculative_config`文档。命名三模式(draft模型、EAGLE、N-gram)和哪个分块prefill兼容。
4. 启EAGLE-3后平均ITL降25%但P99 ITL升15%。诊断提缓解。
5. 算Llama 3.3 70B EAGLE-3 draft head内存成本。与跑Llama 3.2 1B经典draft比如何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 投机解码 | "draft+验证" | 便宜模型提K token、单目标前向验证全K |
| 接受率alpha | "投机接受率" | Draft token目标接受分数；唯一重要指标 |
| Draft长度K | "投机k" | Draft每目标前向提多少token；典型4-8 |
| 验证开销epsilon | "投机开销" | 验证+重滚比朴素目标前向额外成本；批增 |
| EAGLE-3 | "最新EAGLE" | 2025-2026变体；多目标层训练draft head；通用聊天alpha 0.6-0.8 |
| `speculative_config` | "vLLM投机配置" | vLLM V1显式选入；无默认无加速 |
| N-gram投机解码 | "N-gram draft" | GPU侧N-gram查找提示draft；分块prefill兼容 |
| 平衡alpha | "无操作alpha" | 投机解码零加速alpha；生产并发看此 |
| 拒draft两通过 | "重滚成本" | Draft拒两目标前向；驱P99尾 |

## 延伸阅读

- [vLLM — 投机解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config`和V1分块prefill兼容权威源。
- [vLLM投机配置API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确字段集。
- [EAGLE论文(arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原EAGLE draft head公式。
- [EAGLE-2论文(arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应draft和树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 投机解码高效LLM系统。
- [BentoML — 投机解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产推出清单