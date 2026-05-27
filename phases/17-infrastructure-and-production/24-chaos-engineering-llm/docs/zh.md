# LLM生产Chaos Engineering

> 2026 LLM chaos engineering自有纪律。生产实验前prerequisite：定义SLI/SLO、trace+metric+log可观测、自动rollback、runbook、on-call。架构四平面：control (实验调度器)、target (服务、infra、数据存)、safety (guard + abort + 流量过滤)、observability (metric + trace + log)、feedback (入SLO调)。Guardrail强制：burn-rate警日错误预算烧> 2x期望暂停实验；抑制窗 + trace-ID correlation dedupe警噪。节奏：周小canary + SLO review；月game day + postmortem；季跨队韧性审计 + 依赖映。LLM特定实验：内存过载、网络失败、provider停运、畸形提示、KV cache驱逐风暴。工具：Harness Chaos Engineering (LLM推导推荐、blast-radius downscaling、MCP工具集成)；LitmusChaos (CNCF)；Chaos Mesh (CNCF Kubernetes原生)。

**类型:** 学习
**语言:** Python(stdlib、玩具chaos实验runner)
**前置要求:** 阶段17课程23(AI SRE)、阶段17课程13(可观测)
**时间:** ~60分钟

## 学习目标

- 命名五chaos engineering prerequisite (SLI/SLO、可观测、rollback、runbook、on-call)并解释跳任破实践。
- 图四平面(control、target、safety、observability)和feedback loop入SLO。
- 列举五LLM特定实验(内存过载、网络失败、provider停运、畸形提示、KV驱逐风暴)。
- 给栈选工具——Harness、LitmusChaos、Chaos Mesh。

## 问题背景

传统栈chaos测试确立。LLM栈新失败模式加。4K-token提示毒字符tokenizer stall 12秒。上游provider 429；gateway重试；服务OOM重试放大并发。KV cache驱逐风暴突发负载重prefill cascade算饱和。

无单元测试现。Chaos engineering用户发现前发现。

## 概念讲解

### Prerequisite

生产chaos无：

1. **SLI/SLO**——定义服务级指标和目标。
2. **可观测**——trace、metric、log、dashboard wired。
3. **自动rollback**——阶段17课程20 policy-flag rollback。
4. **Runbook**——结构、阶段17课程23。
5. **On-call**——人响应。

缺任chaos变真事件。

### 四平面 + feedback

**Control plane**——实验调度器(Litmus workflow、Chaos Mesh schedule、Harness UI)。

**Target plane**——服务、pod、node、负载均衡器、数据存。

**Safety plane**——kill switch、抑制窗、blast-radius限、错误预算gate。

**Observability plane**——正常metric + trace-ID correlation chaos-induced和自然失败区分。

**Feedback loop**——发现回SLO调、runbook更新、代码修。

### Guardrail强制

- **Burn-rate警**：日错误预算烧超2x期望暂停实验。
- **抑制窗**：blast radius内非实验警静实验期间。
- **Trace-ID correlation**：实验诱发错误全标签on-call dedupe。

### 五LLM特定实验

1. **内存过载**——高并发长上下文请求KV cache抢占风暴强制。观察：服务优雅shed或崩溃？

2. **网络失败**——推理gateway和provider间连通切。观察：fallback SLA内kick？(阶段17课程19)

3. **Provider停运模拟**——OpenAI 100% 429。观察：路由Anthropic failover？(阶段17课程16、19)

4. **畸形提示**——注入tokenizer stall payload (如深嵌unicode、大UTF-8 codepoint)。观察：单请求锁worker？

5. **KV驱逐风暴**——饱和vLLM block budget强制驱逐。观察：LMCache恢复或服务降？

### 节奏

- **周**——staging小canary实验、可能5% prod。
- **月**——特定场景game day；跨队出席；postmortem。
- **季**——跨队韧性审计；依赖映更新。

### 工具

- **Harness Chaos Engineering**——商业；AI推导实验推荐；blast-radius downscaling；MCP工具集成。
- **LitmusChaos**——CNCF graduated；Kubernetes workflow基。
- **Chaos Mesh**——CNCF sandbox；Kubernetes原生CRD风格。
- **Gremlin**——商业；广支持。
- **AWS FIS** / **Azure Chaos Studio**——托管云供。

### 小始

首实验：稳流量下pod-kill一decode replica。观察rerouting和恢复。安全、毕业网络chaos。

首LLM特定实验：注入provider 429 5分。观察fallback。多队发现fallback未全测。

### 你应记数

- 四平面：control、target、safety、observability。
- Burn-rate暂停：2x期望日预算烧。
- 节奏：周canary、月game day、季审计。
- 五LLM实验：内存、网络、provider、畸形提示、KV风暴。

## 使用

`code/main.py`模三chaos实验safety plane gate。报何实验会触burn-rate abort。

## 交付成果

本lesson产`outputs/skill-chaos-plan.md`。给栈和成熟、选首三实验和工具。

## 练习题

1. 跑`code/main.py`。何实验触burn-rate gate为何？
2. 设计vLLM基RAG服务首五chaos实验。含成功准则。
3. Burn-rate警暂停实验。何定根因——chaos或自然？
4. 论chaos应生产跑或仅staging。何时生产正确？
5. 命名三LLM特定失败模式通用网络chaos不能复现。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| SLI / SLO | "服务目标" | 指标+目标；必需prerequisite |
| Blast radius | "范围" | 实验影响服务/用户集 |
| Burn-rate警 | "预算gate" | 错误预算烧率> 2x期望火 |
| Game day | "月演练" | 计划跨队chaos练习 |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos工具 |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes原生chaos |
| Harness CE | "商业AI助" | Harness chaos AI推荐 |
| 畸形提示 | "tokenizer炸弹" | Stall tokenization输入 |
| KV驱逐风暴 | "抢占cascade" | 驱逐触发重prefill |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)