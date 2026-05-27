# vLLM Production Stack与LMCache KV卸载

> vLLM production-stack是参考Kubernetes部署——router、engine、可观测线一起。LMCache是KV卸载层抽KV cache出GPU内存跨查询和engine复用(CPU DRAM、然后disk/Ceph)。vLLM 0.11.0 KV Offloading Connector (2026年1月)异步可插Connector API (v0.9.0+)。卸载延迟非用户面。LMCache无共享前缀仍值——GPU KV slot尽、抢占请求CPU恢复而非重算prefill。发布基准16x H100 (80GB HBM)跨4 a3-highgpu-4g：KV cache超HBM、原生CPU卸载和LMCache大吞吐改；低KV footprint、全配置基线匹配小开销。

**类型:** 学习
**语言:** Python(stdlib、玩具KV spill模拟器)
**前置要求:** 阶段17课程04(vLLM Serving)、阶段17课程06(SGLang/RadixAttention)
**时间:** ~60分钟

## 学习目标

- 图vLLM production-stack层：router、engine、KV卸载、可观测。
- 解释KV Offloading Connector API (v0.9.0+)和0.11.0异步路径隐卸载延迟。
- 量化LMCache CPU-DRAM何时助(KV > HBM) vs加开销(KV小装HBM)。
- 给部署约束选原生vLLM CPU卸载和LMCache connector。

## 问题背景

vLLM serving GPU 100% HBM并发升抢占事件。请求被逐、重队、同一2K-token提示一分钟重prefill四次。GPU算花冗prefill；goodput远低raw吞吐。

加GPU线贵。加HBM不可能。但CPU DRAM便宜——一socket 512 GB+延迟比HBM差量级但"暂暖"KV cache行。

LMCache抽KV cache到CPU DRAM抢占请求快恢复、跨engine重复前缀共享cache无每engine重prefill。

## 概念讲解

### vLLM production-stack

`github.com/vllm-project/production-stack`是参考Kubernetes部署：

- **Router**——cache感知(阶段17课程11)。消费KV事件。
- **Engines**——vLLM worker。每GPU或每TP/PP组一。
- **KV cache卸载**——LMCache部署或原生connector。
- **可观测**——Prometheus scrape、Grafana dashboard、OTel trace。
- **Control plane**——服务发现、配置、滚动更新。

Helm chart + operator发。

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 Connector API可插KV cache backend发。Engine卸载块到connector；connector存(RAM、disk、对象存储、LMCache)。请求需块、connector载回。

vLLM 0.11.0 (2026年1月)加异步卸载路径——卸载后台发engine常见情况不阻塞。端到端延迟吞吐仍依赖负载形状、KV cache命中率、系统压力；vLLM自笔记指自定义kernel卸载低命中率可降吞吐、异步调度投机解码有已知交互问题。

### 原生CPU卸载vs LMCache

**原生vLLM CPU卸载**：engine本地。存KV块host RAM。快实现、零网络跳。不跨engine。

**LMCache connector**：集群规模。块共享LMCache server (CPU DRAM + Ceph/S3 tier)。块任engine可访。16x H100基准发。

单engine HBM压力选原生。多engine共享前缀选LMCache (RAG公共系统提示、多租户共享模板)。

### 基准行为

16x H100 (80 GB HBM) 4 a3-highgpu-4g测：

- 低KV footprint(短提示、低并发)：全配置基线匹配、LMCache加~3-5%开销。
- 中footprint：LMCache跨engine前缀重用始助。
- KV超HBM：原生CPU卸载和LMCache吞吐大改；LMCache更益因跨engine共享。

### LMCache何时决定

- 多租户serving系统提示跨租户共享。
- RAG文档块查询重复。
- 微调变种(LoRA)同基座模型KV重用减冗工作。
- 抢占重负载：CPU恢复比重prefill便宜。

### 何时不启用

- 小HBM压力——付开销无益。
- 短上下文(<1K tokens)——传时间 > 重prefill。
- 单租户单提示负载——无重用捕。

### 解耦serving集成

阶段17课程17解耦serving + LMCache复合：prefill池到解码池KV传若不用落LMCache；后续查询LMCache拉。阶段17课程11 cache感知router可路由本地OR LMCache共享cache匹配engine。

### 你应记数

- vLLM 0.9.0：Connector API发。
- vLLM 0.11.0 (2026年1月)：异步卸载路径；端到端延迟影响依赖负载、KV命中率、系统压力(非绝对保证)。
- 16x H100基准：LMCache KV footprint超HBM时助。
- 小HBM压力：3-5%开销无益。

## 使用

`code/main.py`模抢占重负载带不带LMCache。报避免重prefill、吞吐增益、盈亏平衡HBM利用。

## 交付成果

本lesson产`outputs/skill-vllm-stack-decider.md`。给负载形状和vLLM部署、决定原生vs LMCache vs无。

## 练习题

1. 跑`code/main.py`。何HBM利用LMCache始付？
2. 租户6K-token系统提示200查询/小时共享。算每租户期望LMCache省。
3. LMCache server单点失效。设计HA策略(副本、fallback原生)。
4. LMCache存Ceph盘。4K-token KV 70B FP8 (500 MB)、读时间vs重prefill何？
5. 论vLLM 0.11.0异步路径"免费否"——开销隐何处？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Production-stack | "参考部署" | vLLM Kubernetes Helm chart + operator |
| Connector API | "KV backend接口" | vLLM 0.9.0+可插KV存接口 |
| 原生CPU卸载 | "engine本地spill" | 同engine host RAM存KV |
| LMCache | "集群KV cache" | 跨engine KV cache server CPU DRAM + disk |
| 0.11.0 async | "非阻塞卸载" | 卸载隐engine流后 |
| 抢占 | "逐让位" | HBM满KV cache shuffle |
| 前缀重用 | "同系统提示" | 多查询共享开头；cache命中 |
| Ceph tier | "disk tier" | cache层次DRAM下持久存 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径细节。