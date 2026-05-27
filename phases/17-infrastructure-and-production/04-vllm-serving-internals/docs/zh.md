# vLLM Serving Internals：PagedAttention、连续批处理、分块Prefill

> vLLM 2026统治建三复合默认非单技巧。PagedAttention总开。连续批处理解码迭代间注入新请求到活跃批。分块prefill切长提示使解码token永不饥饿。全开一H100 SXM5 Llama 3.3 70B FP8 128并发推2,200-2,400 tok/s——约vLLM自己默认上25%和朴素PyTorch环3-4x。本lesson读调度器和attention kernel到你可图级别，`code/main.py`玩具连续批器vLLM方式调度prefill和decode。

**类型:** 学习
**语言:** Python(stdlib、玩具连续批调度器)
**前置要求:** 阶段17课程01(模型服务)、阶段11(LLM工程)
**时间:** ~75分钟

## 学习目标

- 解释PagedAttention作KV cache分配器：块、块表、为何生产负载碎片<4%。
- 图迭代级连续批处理：完成序列离批新入无排空。
- 一句描述分块prefill并命名保哪延迟指标(提示：是TTFT尾非平均吞吐)。
- 命名2026 vLLM v0.18.0 gotcha咬队一次开每优化。

## 问题背景

朴素PyTorch服务环一次一请求跑：tokenize、prefill、decode到EOS、返回。一用户工作。一百，是耐心人队列。显修——静态批——垫每请求窗口最长提示、垫每decode最长预期输出、并停整批最慢序列。付不用垫、快请求等慢。

vLLM一次解三问题。PagedAttention阻止KV cache碎片食60-80% GPU内存经典连续分配方式。连续批处理让请求加入离批每解码迭代间，所以批总满真工作。分块prefill破32k token提示~512 token切片解码交错，所以长提示不冻结GPU每解码token。

2026生产默认三全开。需理解每做什么因失败模式全在调度器非模型。

## 概念讲解

### PagedAttention作虚拟内存系统

KV cache是`num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`每序列。Llama 3.3 70B 8192 token BF16约1.25 GB每序列。若每请求预保留8192槽但平均请求只用1500 token，浪费~82%保留HBM。经典批付此浪费。

PagedAttention借OS虚拟内存想法。KV cache非每序列连续。它固定大小块分配(默认16 token)。每序列有块表映射逻辑token位置到物理块ID。序列长过分配块时，加一块。完成时，其块回池。

碎片从60-80%(经典)降到<4%(PagedAttention)。无标志启PagedAttention——它是vLLM唯一分配器。旋钮是`--gpu-memory-utilization`(默认0.9)，告vLLM加载权重和激活后为KV块保留多少HBM。

### 迭代级连续批处理

老"动态批"等窗口(如10 ms)填批，然后跑prefill + decode + decode + decode到每序列完成。快序列早离坐空GPU完慢。

连续批处理每解码步操作。调运行序列集`RUNNING`列表。每迭代：

1. `RUNNING`中任序列刚EOS或max_tokens移除。
2. 调度器看等待队列。若有空闲KV块，它入新序列(prefill或恢复)。
3. 前向跑`RUNNING`现在内容，每序列发一新token。

批大小永不垫固定数。不同输出位置序列共享一融合前向。2026 vLLM这叫`V1 scheduler`。关键不变：调度器每解码迭代跑一次非每请求一次。

### 分块prefill保TTFT尾

Prefill计算绑。32k token提示Llama 3.3 70B一H100纯prefill~800 ms。prefill跑时，批中每其他序列解码token等。服务环中，一长提示首token延迟(TTFT)成几十其他用户间token延迟(ITL)突。

分块prefill切prefill固定大小块(默认512 token)并每块调度为单位。块间调度器可推解码序列一token。付小绝对prefill延迟损(每块几ms)换低解码抖动。发布基准混合负载P99 ITL从~50 ms降到~15 ms。

### 三默认交互

三特性互假设。PagedAttention给调度器细粒KV资源交易。连续批处理需细粒资源入新序列不强全局重洗。分块prefill是调度器同`RUNNING`列表决策——它是多一调度器政策非分离系统。

不需知每标志。需知调度器优化：KV块预算下goodput、受分块prefill切片。

### 2026 v0.18.0 gotcha

vLLM v0.18.0不能`--enable-chunked-prefill`与draft模型投机解码(`--speculative-model`)合。文档例外是V1 scheduler N-gram GPU投机解码。队无读发布笔记翻每标志得启动运行错误非软回退。若投机增益值启分块prefill、重访选择——2026正确答案常是EAGLE-3无分块prefill、非draft模型加分块prefill不编译。

### 你应记住数

- Llama 3.3 70B FP8、H100 SXM5、128并发、三全开：2,200-2,400 tok/s。
- 同模型、默认vLLM(无分块prefill)：~1,800 tok/s。
- 同模型、朴素PyTorch前向环：~600 tok/s。
- 生产负载下PagedAttention KV碎片浪费：<4%。
- 混合负载P99 ITL：分块prefill~15 ms、无~50 ms。

### 调度器像什么

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 调度prefill块+解码一批
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 token
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一融合GPU调用
```

`code/main.py`是这环stdlib Python假token数假前向延迟。跑示分块prefill长prefill间保解码序列活。

## 使用

`code/main.py`模拟vLLM风格调度器可切特性。跑见：

- `NAIVE`模式：一次一请求、无批。
- `STATIC`模式：垫等、经典批。
- `CONTINUOUS`模式：迭代级入释放。
- `CONTINUOUS + CHUNKED`模式：prefill切片解码交错。

输出示总吞吐(虚拟秒token)、TTFT平均、和P99 ITL。`CONTINUOUS + CHUNKED`行应混合流量主导。

## 交付成果

本lesson产`outputs/skill-vllm-scheduler-reader.md`。给服务配置(批大小、KV内存利用、分块prefill大小、投机配置)，产调度器诊断命名三默认哪个瓶颈调什么。

## 练习题

1. 跑`code/main.py`。比`STATIC`到`CONTINUOUS`混合短长请求负载。吞吐差距从哪来——prefill效率、decode效率、或尾延迟？
2. 改玩具调度器加`--max-num-batched-tokens`。H100跑Llama 3.3 70B FP8正确值是什么？(提示：它是KV块大小和空闲块数函数非裸HBM。)
3. 重读vLLM v0.18.0发布笔记。哪些标志组合互斥？列它们。
4. 计算KV cache碎片浪费1000请求trace平均1500输出token、std 600 token、下 连续每请求分配8192最大、 PagedAttention 16 token块。
5. 一段解释为何分块prefill帮P99 ITL但孤立不帮吞吐。吞吐赢实践中从哪来？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| PagedAttention | "KV trick" | KV cache固定大小块分配器；碎片<4% |
| 块表 | "page table" | 每序列逻辑token位置到物理KV块映射 |
| 连续批处理 | "动态批、但对" | 每解码迭代入/释放决策 |
| 分块prefill | "prefill切" | 破长prefill 512 token切片解码交错 |
| TTFT | "首token时间" | Prefill+队列+网络；长提示prefill主导 |
| ITL | "间token延迟" | 连续解码token间时间；批大小主导 |
| Goodput | "吞吐满足SLO" | 每秒token每请求仍打TTFT和ITL目标 |
| V1 scheduler | "新调度器" | vLLM 2026调度器；N-gram spec decode是分块prefill兼容路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活后KV块保留HBM分数 |

## 延伸阅读

- [vLLM文档 — 投机解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 分块prefill和投机解码兼容官方源。
- [vLLM发布笔记(NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026发布节奏和版本特定行为。
- [vLLM博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 原写仍定义如何思考分配器。
- [PagedAttention论文(arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 详细V1调度器遍历带火焰图