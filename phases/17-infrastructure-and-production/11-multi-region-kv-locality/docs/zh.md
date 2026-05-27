# 多区域LLM服务和KV cache局部性

> 轮询负载平衡对缓存LLM推理主动有害。不落持有前缀节点请求付全prefill成本——长提示P50约800 ms vs cache命中~80 ms。2026生产模式是cache感知路由器(vLLM Router Rust、llm-d router)消费KV cache事件前缀哈希匹配路由。最近研究(GORGO)跨区网络延迟显式路由目标项。商业"跨区推理"服务(Bedrock跨区推理、GKE多集群gateway)视推理opaque——处理可用性非TTFT。JPMorgan和Mayo Clinic 2024年11月跑us-east-1故障转移~22分钟。DR现实：32% LLM DR失败因队备份权重忘tokenizer文件或量化配置。

**类型:** 学习
**语言:** Python(stdlib、玩具前缀cache感知路由模拟器)
**前置要求:** 阶段17课程04(vLLM Serving)、阶段17课程06(SGLang RadixAttention)
**时间:** ~60分钟

## 学习目标

- 解释为何轮询负载平衡断缓存推理并量化TTFT惩罚。
- 图cache感知路由器：输入(KV cache事件)、算法(前缀哈希匹配)、决胜(GPU利用)。
- 命名32% DR失败驱动LLM(缺失tokenizer文件/量化配置)并陈述三文件DR清单。
- 区商业跨区服务(Bedrock CRI、GKE Multi-Cluster Gateway)和KV感知路由。

## 问题背景

服务跑us-east-1、us-west-2、eu-west-1。放ALB前轮询。生产前缀cache命中率降到8%。TTFT P50三倍。vLLM日志每请求付全prefill成本。

轮询无状态服务最优。LLM推理设计状态化——KV cache编码模型见一切。盲路由是错cache路由。

另，队DR计划。备份模型权重S3跨区。区域停运；尝试故障转移；副本拒启。忘tokenizer.json、量化配置、和RoPE缩放配置在分离桶未同步。

多区域LLM服务是cache问题、路由问题、和DR卫生问题——非负载平衡问题。

## 概念讲解

### Cache感知路由

请求带提示达。路由器哈希前缀(如首512 token)；问每副本"你有此前缀缓存？"。副本KV cache事件pub/sub通道分配驱逐块发。路由器选匹配副本、无匹配落GPU利用决胜。

**vLLM Router**(Rust、2026 production-stack)：订阅`kv.cache.block_added`事件、维护前缀哈希→副本索引、O(1)查路由。无匹配落最队列深度。

**llm-d router**：同模式、Kubernetes原生。ControlPlane API发事件。

**SGLang RadixAttention**(阶段17课程06)是内副本等效。跨副本路由严格上游。

### 数

TTFT P50 2K token提示、Llama 3.3 70B FP8、H100：
- Cache命中(同副本、前缀驻)：~80 ms。
- Cache失(冷prefill)：~800 ms。

10x gap。若路由器跨副本打60-80%前缀cache、你N副本容量近似单副本性能。若打10%、你近似朴素扩展。

### 跨区有新约束——网络延迟

区间RTT：
- us-east-1 ↔ us-west-2：~65 ms。
- us-east-1 ↔ eu-west-1：~75 ms。
- us-east-1 ↔ ap-southeast-1：~220 ms。

若路由从us-east-1请求到ap-southeast-1热前缀、省prefill(800 → 80 ms)被440 ms往返淹没。GORGO(2026研究)显式——联合最小`prefill_time + network_latency`非仅prefill。常答案区域路由除大多MB前缀prefill主导。

### 商业"跨区推理"此处无助

AWS Bedrock跨区推理容量压力时自动路由请求其他区。优化可用性非TTFT、视推理opaque。GKE Multi-Cluster Gateway同——服务级故障转移、无KV cache感知。

你仍需应用层cache感知路由器即使用这些。它们处理"us-east-1着火"情况。Cache感知路由处理TTFT情况。

### DR卫生——32%缺失文件问题

广引用2026统计：32% LLM DR失败因队备份权重忘：

- `tokenizer.json`或`tokenizer.model`
- 量化配置(`quantize_config.json`、AWQ scale、GPTQ zero-point)
- 模型特定配置(RoPE缩放、attention mask、chat template)
- 引擎配置(`vllm_config.yaml`、采样默认、LoRA adapter manifest)

修复是三文件最小DR manifest：

1. HF模型repo下所有文件(权重+配置+tokenizer)。
2. 引擎特定服务配置。
3. 部署manifest(K8s YAML、Dockerfile、依赖锁)。

加：季度DR演练。JPMorgan us-east-1演练2024年11月22分钟恢复仅因playbook排练。

### 数据居住正交

EU客户PHI不能离EU。若你cache感知路由器发Paris起源请求us-east-1前缀匹配、你GDPR违规无论TTFT增益。居住边界路由分区前优化cache。

### 你应记数

- Cache命中vs失TTFT gap：~10x(2K提示80 ms vs 800 ms)。
- US-EU区间RTT：~75 ms。
- DR失败：32%缺tokenizer/量化配置。
- JPMorgan us-east-1故障转移2024年11月：22分钟(30分钟SLA)。

## 使用

`code/main.py`模拟三路由策略(轮询、cache感知区域、cache感知全球)多区域负载。报cache命中率、TTFT P50/P99、跨区账。

## 交付成果

本lesson产`outputs/skill-multi-region-router.md`。给区域、居住约束、SLA、设计路由计划。

## 练习题

1. 跑`code/main.py`。75 ms RTT、何提示长度跨区路由赢本地仅路由？
2. Cache命中率从70%降到12%。诊断三可能原因和每确认可观察。
3. 设计vLLM 5 LoRA adapter服务70B AWQ量化模型DR manifest。列每文件和配置。
4. 论证Bedrock跨区推理对严格TTFT SLO fintech"够"否。引用具体行为。
5. Paris起源请求匹配us-east-1前缀。你路由否？写政策。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Cache感知路由 | "智能LB" | 前缀哈希匹配KV cache持有副本路由 |
| KV cache事件 | "cache pub-sub" | 副本发块加/驱逐；路由器索引 |
| 前缀哈希 | "cache键" | 首N token哈希用作路由器查 |
| GORGO | "跨区路由研究" | arXiv 2602.11688；网络延迟显式项 |
| 跨区推理 | "Bedrock CRI" | AWS产品；可用性故障转移非TTFT感知 |
| DR manifest | "备份列表" | 恢复需每文件——非仅权重 |
| 数据居住 | "GDPR边界" | 哪区看用户数据法律约束 |
| RTT | "往返时间" | 网络延迟；US-EU 75 ms、US-APAC 220 ms |
| LLM感知LB | "cache命中LB" | Cache感知路由器作产品类 |

## 延伸阅读

- [BentoML — 多云和跨区推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO(2602.11688)](https://arxiv.org/html/2602.11688v1) — 跨区KV cache复用带网络延迟项。
- [TianPan — 多区域LLM服务Cache局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock跨区推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache感知路由器源