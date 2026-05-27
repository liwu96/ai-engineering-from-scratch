# DualPipe 并行

> DeepSeek-V3 在2048个 H800 GPU 上训练，MoE 专家分散在多个节点上。跨节点专家 all-to-all 通信成本为每1个 GPU 小时计算需要1个 GPU 小时通信。GPU 一半时间空闲。DualPipe（DeepSeek，2024年12月）是一种双向流水线，将前向和后向计算与它们触发的 all-to-all 通信重叠。气泡减少，吞吐量上升，而两个模型参数副本的保持（给出名称的"dual"）一旦专家并行性已经在各个 rank 之间分散专家，成本就很便宜。本课程是关于 DualPipe 实际做什么以及为什么 Sea AI Lab 的 DualPipeV 改进以更紧的气泡为代价消除了2倍参数成本的学习型讲解。

**类型:** 学习
**语言:** Python (stdlib, 调度模拟器)
**前置要求:** 第10阶段·05（分布式训练，FSDP，DeepSpeed），第10阶段·14（开源模型架构和 MoE）
**时间:** ~60分钟

## 学习目标

- 说出 DualPipe 前向-后向块的四个组成部分以及为什么每个都有自己的重叠窗口。
- 解释大规模时的流水线气泡问题，以及"无气泡"在实践中的含义与营销中的含义。
- 手动追踪8个 PP rank 和16个微批次的 DualPipe 调度，并确认前向和反向流填满彼此的空闲槽。
- 说明 DualPipeV（Sea AI Lab，2025）做出的权衡：以更大气泡为代价消除2倍参数复制，当专家并行性不活跃时。

## 问题背景

在2k H800 GPU 上训练6710亿 MoE 模型遇到三个复合瓶颈：

1. **内存压力。** 每个 GPU 持有模型的一个分片。序列8k、61层、128头跨度的激活内存是巨大的。
2. **流水线气泡。** 传统流水线并行（GPipe，1F1B）在等待阶段输入或梯度时让 GPU 空闲。在8个阶段，即使使用1F1B调度，大约12%的 GPU 时间可能是气泡。
3. **跨节点 all-to-all。** 专家并行性的 MoE 将专家分散在节点上。每次前向传递触发一次 all-to-all 将 Token 分派到它们的专家，以及一次组合。在2k GPU 上，这很容易变成1:1的计算与通信比。

每个都有单独的解决方案：梯度检查点用于内存，Zero Bubble（Sea AI Lab，2023）用于流水线气泡，专家并行通信内核用于 all-to-all。DualPipe 做的是让它们协同工作。调度在单个前向-后向块内重叠计算和通信，从流水线两端同时注入微批次，并使用结果调度在计算窗口内隐藏 all-to-all。

报告结果：几乎消除流水线气泡，在 DeepSeek-V3 的14.8T Token 训练运行中 GPU 利用率超过95%。

## 概念讲解

### 流水线并行复习

将 N 层模型拆分到 P 个设备。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。微批次流经设备0到P-1，然后从P-1到0向后。每个设备只有在前一个设备发送其输出时才能开始其前向阶段，只有当下游设备发送上游梯度时才能开始向后。

GPipe（Huang et al.，2019）一次调度一个微批次，浪费大部分 GPU 时间。1F1B（Narayanan et al.，2021）为多个微批次交错前向和后向传递。Zero Bubble（Qi et al.，2023）将后向传递分成两部分——输入后向（B）和权重后向（W）——并调度它们填满气泡。Zero Bubble 之后，流水线几乎是紧的。

DualPipe 是下一步。它在其上添加两个想法：

### 想法1：块分解

每个前向块被分成四个组件：

- **注意力。** Q/K/V 投影、注意力、输出投影。
- **All-to-all 分派。** 将 Token 发送到其专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all 组合。** 将专家输出带回的跨节点通信。

后向块添加每个的梯度版本。DualPipe 调度它们，使 all-to-all 分派与下一个块的注意力计算并行发生，all-to-all 组合与后续块的 MLP 计算并行发生。

### 想法2：双向调度

大多数流水线调度从阶段0注入微批次并流向阶段P-1。DualPipe 从两端注入微批次。阶段0看到起源于那里的前向微批次；阶段P-1也看到起源于那里的前向微批次。两个流在中间相遇。

为此，设备 `i` 必须持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 的"dual"部分：每个设备保持两份它需要的模型层副本（每个方向一份）。在 DeepSeek-V3 的规模上，这是2倍的参数复制成本。这是可以承受的，因为专家并行性已经将 MoE 专家分散得很薄，复制非专家层两次是小菜一碟。

关键的是，一个方向的向前流和另一个方向的向后流正好在单向调度中会有气泡的地方重叠。气泡消失了。

### 手绘调度

考虑 P = 4个 rank，8个微批次，分成4个向前/4个向后。时间从左到右移动；行是设备 rank。

```
           时间 →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读取"F4/F5R"符号：rank 1 正在运行微批次4的前向（在流水线中从左到右）和微批次5的前向（从右到左）在同一时间段。这就是"双向"在操作上的含义。

在 rank 2，交叉流更早重叠；在 rank 0 和 P-1，它们最晚重叠。在调度的稳定中间阶段，每个 rank 运行X方向的向前与Y方向的后向重叠。计算繁忙。前向传递的 all-to-all 分派隐藏在后向计算内。All-to-all 组合隐藏在前向计算内。气泡被挤出。

### 气泡核算

标准1F1B流水线气泡（每 rank 浪费时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低，但不是零。DualPipe，在稳定阶段，如果微批次计数可被2倍的流水线深度整除，则气泡为零。在稳定阶段之外（预热和冷却），有一些气泡，但它不随微批次数量增长——这是论文强调的关键特性。

在营销术语中："无气泡"。在技术术语中：气泡不随微批次计数增长。Sea AI Lab 的后续分析（DualPipeV / 减半）显示，只有当专家并行性不是瓶颈时才有完全无气泡；有 EP 驱动的 all-to-all，总是存在一些调度折衷。

### DualPipeV — 改进

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠成在单个参数副本上运行的"V形"调度。气泡比 DualPipe 略大，但内存节省是实质性的。DeepSeek 在他们开源的 DualPipe 实现中将 DualPipeV 作为 EP 关闭模式采用。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|------|---------|-----------|------|------------|
| 每设备参数副本 | 2 | 1 | 1 | 1 |
| 与微批次的气泡 | 恒定 | 小增长 | 增长 | 增长 |
| 计算-通信重叠 | 完整 | 部分 | 最小 | 部分 |
| 何时使用 | EP 重的 MoE | 稠密或 EP 轻 | 基线 | 任何流水线 |

### 14.8T Token 运行的含义

DeepSeek-V3 的预训练在2.8M GPU 小时内消耗2048个 H800 GPU 上的14.8T Token。使用朴素1F1B，他们会损失12-15%的流水线气泡——340-420K GPU 小时，足够训练一个完整的700亿模型。DualPipe 恢复了大部分。没有内部日志直接量化贡献是困难的，但论文中的声明是训练平均 GPU 利用率超过95%。

对于较小的运行（1k GPU 以下），DualPipe 是过度的——流水线气泡相对于总成本较小，稠密模型训练很少遇到 all-to-all 瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 堆栈中的位置

- 与 **FSDP**（第10阶段·05）互补。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。它们结合。
- 与 **ZeRO-3** 梯度分片兼容。两个副本复制的簿记需要与 ZeRO 的分片梯度合作。
- 需要为特定集群拓扑调整的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 实际应用

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印1F1B、Zero Bubble、DualPipe 和 DualPipeV 每个的稳定阶段利用率。它是一个教学工具——数字匹配论文的定性声明，不是关于生产测量加速的声明。

模拟器的价值：用不同的 P 和微批次计数运行它，观察气泡分数如何为1F1B增长但不为 DualPipe。

真实训练运行的集成考虑：

- 选择一个能干净整除微批次计数的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考。
- 预计第一次要在调度本身上烧掉一周的调试时间。簿记很繁琐。
- 监控每 rank 的 GPU 利用率，而不仅仅是聚合。DualPipe 的收益来自收紧落后者。

## 产出成果

本课程产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规范（GPU 计数、拓扑、互连、模型形状），它推荐流水线并行策略、要使用的调度算法以及目标规模下的预期气泡分数。

## 练习题

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异并用每百万 Token 训练恢复的 GPU 小时表示。

2. 手绘 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时间段。识别气泡消失的最早时间段。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图5。识别 DualPipe 前向块内 all-to-all 分派的重叠窗口。解释计算调度如何隐藏它。

4. 计算 P=8 个流水线阶段和70B稠密模型以及 P=16 个流水线阶段和671B MoE 模型的 DualPipe 2倍参数开销。显示为什么 MoE 情况的开销比例更小（大多数参数是专家，分散在大型 EP 组中）。

5. 将 DualPipe 与 Chimera（2021年来自竞争的调度器）比较。使用论文的第3.4节作为参考，识别 DualPipe 添加的 Chimera 没有的两个特定特性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 流水线气泡 | "每 rank 空闲时间" | 因为流水线阶段等待其输入或梯度而浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一个前向/一个后向交错调度；DualPipe 击败的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将后向分成B（输入梯度）和W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次计数增长 |
| DualPipeV | "减半" | V形改进，以更大气泡为代价消除2倍参数复制 |
| 块 | "流水线工作单元" | 一个微批次通过一个流水线阶段的前向或后向传递 |
| All-to-all 分派 | "将 Token 发送到专家" | 将 Token 路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all 组合 | "将专家输出带回" | MLP 后收集专家输出的跨节点通信 |
| 专家并行性 (EP) | "跨 GPU 的专家" | 跨 rank 分片 MoE 专家，使不同 GPU 持有不同专家 |
| 流水线并行性 (PP) | "跨 GPU 的层" | 跨 rank 分片模型层；DualPipe 调度的维度 |
| 气泡分数 | "浪费的 GPU 时间" | （气泡时间/总时间）；DualPipe 将其推向零的分数 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV（减半）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 为 DeepSeek 的 EP 关闭模式提供信息的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 比较的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文和气泡问题
