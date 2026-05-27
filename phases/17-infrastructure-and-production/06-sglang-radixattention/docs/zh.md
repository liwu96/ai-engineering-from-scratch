# 前缀重负载SGLang和RadixAttention

> SGLang视KV cache为一等可复资源存radix树。vLLM FCFS(先到先服)调度请求，SGLang cache感知调度器优先更长共享前缀请求——有效深度优先radix遍历使热分支驻HBM。Llama 3.1 8B ShareGPT类1K提示，SGLang~16,200 tok/s到vLLM~12,500、~29%优。前缀重RAG负载优势达6.4x。语音克隆形负载cache命中率清86%。2026部署400,000+ GPU跨xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。Gotcha是6.4x数前缀序不一致时蒸发——序是工程师杠杆。

**类型:** 学习
**语言:** Python(stdlib、玩具radix树cache+cache感知调度器)
**前置要求:** 阶段17课程04(vLLM Serving Internals)、阶段14(Agentic RAG)
**时间:** ~75分钟

## 学习目标

- 图RadixAttention：前缀如何存radix树和KV块如何跨同根分支序列共享。
- 解释cache感知调度和为何FCFS前缀重流量错。
- 算预期加速给负载前缀cache命中率和提示长度分布。
- 命名提示序纪律使6.4x数真实vs失上行。

## 问题背景

经典服务视每请求提示 opaque。即使5,000 RAG请求全开始同2,000 token系统提示+同检索前言、vLLM prefill那2,000 token前缀5,000次。GPU反复做同工作。

观察：Agentic和RAG负载提示几乎总共享长前缀。系统提示、工具schema、少样本例、检索header、对话历史——全跨请求重复。若存前缀KV cache一次复用、不重prefill。

RadixAttention做这。Token索引radix树；每节点拥有路径从根token序列KV块。新请求走树：任节点token匹配复用节点KV块。Prefill成本比例"新"后缀非全提示。

挑战调度。若两请求共享2,000 token前缀和第三共享同前缀200 token、想两长共享请求一起服使长前缀留HBM。FCFS反——先到先服、潜在在下长前缀请求打前驱逐热分支。

## 概念讲解

### Radix树作KV索引

Radix树(紧trie)存token序列。每节点拥有token范围和该范围计算KV块。子扩展序列一或更多token。

```
root
 |- "You are a helpful assistant..."  (2,000 token, 124 KV block)
      |- "Context: <doc A>..."        (500 token, 31 block)
           |- "Question: Alice..."    (80 token, 5 block)
           |- "Question: Bob..."      (95 token, 6 block)
      |- "Context: <doc B>..."        (520 token, 33 block)
```

新请求系统提示+"Context: <doc A>"+"Question: Carol"入。调度器走：系统前缀匹配(124 block复用)、doc-A分支匹配(31 block复用)、然后仅"Question: Carol"分配新block(4 block)。Prefill成本：4 block新token。无树：160 block。~40x省prefill。

### Cache感知调度

Radix树背复用无意义若cache churn。两关键政策：

1. **深度优先dispatch**。从队列选下请求时、优先根同分支当前运行集请求。这使热分支钉。
2. **分支级LRU非块级**。驱逐全分支(从最短用叶开始)而非单块、所以cache形状匹配radix形状。

FCFS违两者。共享2,000 token请求坐在共享50请求后、然后2,000 token分支被驱逐纳50 token一个。

### 你应记基准数

- Llama 3.1 8B、H100、ShareGPT 1K提示：SGLang ~16,200 tok/s vs vLLM ~12,500(~29%优)。
- 前缀重RAG(同系统+同文档、变问题)：SGLang上达6.4x。
- 语音克隆负载：86.4%前缀cache命中率。
- SGLang客户生产命中率：50-99%取决提示纪律。
- 2026部署400,000+ GPU。

### 序gotcha

6.4x数依赖一致提示模板序。若你客户端构提示某些请求`[system, tools, context, history, question]`其他`[system, context, tools, history, question]`、树不能找共享前缀。人看共享前缀是radix树两不同序列。

工程师杠杆：提示模板是cache键。定序。放一切不变(system、tools、schemas)最前。放检索context次。放用户问题最后。勿交织动态内容前缀。

研究真实例：移动态内容出可cache前缀一部署一次改从7%到74% cache命中率。

### RadixAttention赢输何处

赢：
- RAG(同检索前言、变问题)。
- Agent(同工具schema、变查询)。
- 长系统提示聊天。
- 语音/视觉负载重复前言。

输(回vLLM级吞吐)：
- 单次生成独特提示(代码完成、无系统提示开放聊天)。
- 动态提示每请求交织独特内容前缀。

### 为何这是调度器问题非仅kernel问题

可KV复用kernel技巧实现。SGLang洞察复用仅在调度器保持热分支驻时付。朴素"若可用复用"政策混合负载下churn cache。Radix树索引调度器是kernel技巧转29%生产优。

### 与vLLM交互

两系统非严格竞。2026 vLLM加前缀缓存(`--enable-prefix-caching`)和cache感知路由器(vLLM Router Rust)。差距闭但未全消——SGLang整栈radix优先；vLLM graft上。前缀复用主导负载SGLang仍默认。无强前缀模式通用服务、vLLM仍等或更好。

## 使用

`code/main.py`实现玩具radix树KV cache加调度器两政策：FCFS和cache感知。同负载跑两者、报前缀cache命中率和吞吐delta。然后跑"乱序"负载示6.4x崩溃。

## 交付成果

本lesson产`outputs/skill-radix-scheduler-advisor.md`。给负载描述(提示模板形状、检索模式、并发租户数)、产提示序处方和SGLang采用go/no-go。

## 练习题

1. 跑`code/main.py`。比FCFS和cache感知同负载。delta从哪来——prefill省、decode省、或队列延迟？
2. 改负载随机混`[system, tools, context]`。重跑。命中率发生什么？为何？
3. 算Llama 3.1 8B保持2,000 token系统提示作一radix分支HBM成本。与无前缀复用16序列批成本比。
4. 读SGLang RadixAttention论文。三句解释为何树形LRU驱逐赢前缀重负载块形LRU。
5. 客户报仅8% cache命中率。命名三可能原因和每跑诊断。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RadixAttention | "SGLang thing" | KV cache索引radix树共享前缀复块 |
| Radix树 | "紧trie" | 每节点拥有token范围及其KV块树 |
| Cache感知调度器 | "热分支优先" | 优先共享驻分支请求调度器 |
| 前缀cache命中率 | "提示多少免费" | 提示token分数从复用KV块服 |
| FCFS | "先到先服" | 默认调度破前缀局部 |
| 分支级LRU | "驱逐叶" | 匹配radix形状驱逐政策 |
| 提示模板序 | "cache键" | 提示组件序决定树可共享什么 |
| 系统提示钉 | "驻前缀" | 保持不变系统部分钉避驱逐thrash |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源和文档。
- [SGLang文档](https://sgl-project.github.io/) — RadixAttention和调度细节。
- [SGLang论文 — Efficiently Programming Large Language Models(arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS博客 — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数和调度器理由。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM自己的radix类似实现比较