# LLM API负载测试——为何k6和Locust欺骗

> 传统负载测试器非为流响应、变输出长度、token级指标、GPU饱和设计。两陷阱咬多队。GIL陷阱：Locust token级测Python GIL下tokenization跑、高并发请求生成竞争；tokenization backlog涨报间token延迟——客户端瓶颈、非服务器。提示均匀陷阱：循环同提示测token分布一点；真实流量变长度和异前缀匹配。LLMPerf `--mean-input-tokens` + `--stddev-input-tokens`修。2026工具映：LLM专用(GenAI-Perf、LLMPerf、LLM-Locust、guidellm) token级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025年9月)**——流感知、Kubernetes原生分布式TestRun/PrivateLoadZone CRD、最佳CI/CD gate；Vegeta Go恒速率饱和；Locust 2.43.3仅LLM-Locust扩展流。负载模式：稳态、爬、突(autoscaling测)、浸(内存漏)。

**类型:** 构建
**语言:** Python(stdlib、玩具真实提示生成器 + 延迟收集器)
**前置要求:** 阶段17课程08(推理指标)、阶段17课程03(GPU自动扩展)
**时间:** ~75分钟

## 学习目标

- 解释两反模式(GIL陷阱、提示均匀陷阱)使通用负载测试器LLM API欺骗。
- 给目的选工具：LLMPerf (benchmark跑)、k6 + 流扩展(CI gate)、guidellm (大规模合成)、GenAI-Perf (NVIDIA参考)。
- 设计四负载模式(稳、爬、突、浸)并命名每捕失败模式。
- 用mean + stddev输入token而非定长构真实提示分布。

## 问题背景

k6测LLM端点500并发用户。通过。发。生产200实际用户服务倒——P99 TTFT爆、GPU钉。

两事发。一、k6发500同提示——请求合并和前缀cache似500并发解码实际一。二、k6不流响应间token延迟眼体验；见一HTTP连接、非500 token异间隔到。

LLM负载测试是自有纪律。

## 概念讲解

### GIL陷阱(Locust)

Locust Python用、客户端GIL下tokenization跑。高并发tokenization队列请求生成后。报间token延迟含客户端tokenization backlog。觉服务器慢；测试器。

修：LLM-Locust扩展移tokenization分离进程、或用编译语言Harness (k6、LLMPerf tokenizers.rs)。

### 提示均匀陷阱

全知负载测试器可配一提示。循环10,000迭代每发同提示。服务器每见同前缀——前缀cache命中率近100%、吞吐看好。

修：提示分布采样。LLMPerf用`--mean-input-tokens 500 --stddev-input-tokens 150`——异长度、异内容。

### 四负载模式

1. **稳态**——恒RPS 30-60分。捕：基线性能回退。
2. **爬**——线性0到目标RPS 15分。捕：容量断点、暖异常。
3. **突**——突3-10x RPS 2分然后回。捕：autoscaling延迟、队列饱和、冷启影响。
4. **浸**——稳态4-8小时。捕：内存漏、连接池漂、可观测溢。

### 2026工具映

**LLMPerf** (Anyscale)——Python但Rust-backed tokenization。Mean/stddev提示。流感知。最佳默认性能跑。

**NVIDIA GenAI-Perf**——NVIDIA参考。Triton client；全面指标覆盖。注ITL排除TTFT；LLMPerf含。两工具同服务器产异TPOT。

**LLM-Locust** (TrueFoundry)——Locust扩展修GIL陷阱。熟Locust DSL + 流指标。

**guidellm**——大规模合成benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025年9月)**：
- k6自身(Go、编译、无GIL)加流感知指标。
- k6 Operator TestRun / PrivateLoadZone CRD Kubernetes原生分布式测。
- 最佳CI/CD gate和SLA测。

**Vegeta**——Go、比k6简。恒速率HTTP饱和。非LLM感知但gateway / 速率限测好。

**Locust 2.43.3 stock**——LLM有GIL陷阱。仅LLM-Locust扩展。

### CI SLA gate

PR跑k6：

- 基线RPS每30-50迭代。
- Gate：P50/P95 TTFT、5xx < 5%、TPOT阈值下。
- 破断build。

### 真实提示分布

真实流量样本构(若有)或发分布(如ShareGPT提示聊天、HumanEval代码)。Mean + stddev喂LLMPerf。避免循环一提示绝。

### 你应记数

- k6 Operator 1.0 GA：2025年9月。
- k6 v2026.1.0：流感知指标。
- 典型LLMPerf跑：并发X 100-1000请求。
- 典型CI gate：每PR 30-50迭代。
- 四模式：稳、爬、突、浸。

## 使用

`code/main.py`模负载测试真实提示分布、测有效TPOT、示均匀提示陷阱。

## 交付成果

本lesson产`outputs/skill-load-test-plan.md`。给负载和SLA、选工具设计四负载模式。

## 练习题

1. 跑`code/main.py`。比均匀vs真实分布——gap何处？
2. 写CI gate k6脚本：TTFT P95 < 800 ms 100并发、runtime 5分钟。
3. 浸测试示内存50 MB/小时涨。命名三因和选间仪器。
4. 突测试10 RPS到100 RPS。若Karpenter + vLLM production-stack到位期望恢复时间何(阶段17课程03 + 18)？
5. GenAI-Perf报TPOT=6ms；LLMPerf报TPOT=11ms同服务器。解释。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| LLMPerf | "LLM harness" | Anyscale benchmark工具、流感知 |
| GenAI-Perf | "NVIDIA工具" | NVIDIA参考harness |
| LLM-Locust | "LLM Locust" | Locust扩展修GIL陷阱 |
| guidellm | "合成benchmark" | 大规模合成工具 |
| k6 Operator | "K8s k6" | CRD基分布式k6 |
| GIL陷阱 | "Python客户端开销" | Tokenization backlog涨报延迟 |
| 提示均匀陷阱 | "单提示骗" | 循环同提示cache命中、吞吐涨 |
| 稳态 | "恒负载" | N分平RPS |
| 爬 | "线性升" | 0到目标时间 |
| 突 | "burst测" | 突乘数然后回 |
| 浸 | "长测" | 小时漏检 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)