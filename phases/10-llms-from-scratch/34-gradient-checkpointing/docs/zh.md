# 梯度检查点与激活重计算

> 反向传播保留每个中间激活。在 70B 参数和 128K 上下文中，这是每秩 3 TB 激活。检查点用 FLOPs 换内存：重计算而非保存。问题是丢弃哪些段，答案不是"全部"。

**类型:** 构建
**语言:** Python（使用 numpy，可选 torch）
**前置要求:** 第10阶段第04课（预训练微型 GPT），第10阶段第05课（扩展与分布式）
**时间:** ~70分钟

## 问题背景

训练 Transformer 为每层存储每个需要反向传播微分的操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于隐藏大小 `d`、序列长度 `L`、批处理 `B` 的层，这约为每层 `12 * B * L * d` 浮点数。

对于 `d=8192, L=8192, B=1`，BF16 中每层 800 MB。64 层模型是 51 GB 激活——这还没乘以微批处理大小、还没加注意力-softmax 中间值（每头 `L^2`）、还没考虑张量并行部分副本。

双向账单：BF16 权重加优化器状态可能适配 80GB，但激活将其推过。梯度检查点（又称激活重计算）是标准修复。丢弃大多数激活；在反向传播期间重做前向以取回它们。成本：额外 FLOPs。收益：内存按检查点段数与总层数的比率下降。

朴素地做，检查点每步增加约 33% 的前向传递 FLOPs。做得好——按 Korthikanti 等人的"智能选择"——以低于 5% 的 FLOP 开销节省 5 倍内存。有了 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE，这真的很重要：你既负担不起内存，也负担不起浪费的计算。

## 概念讲解

### 反向传播实际需要什么

`output = layer(input)`。反向传播想要 `grad_input` 和 `grad_params`。要计算它们，它需要：

- `input`（计算 `grad_params = input.T @ grad_output` 用于线性层）
- 一些激活导数中间值（ReLU/GELU/softmax 的导数取决于激活值）

前向传递自动将它们存储在 autograd 图中。每个 `tensor.retain_grad()` 和每个需要输入的操作都保留引用。

### 朴素完全检查点

将网络分成 `N` 段。前向期间，仅存储每段的*输入*。当反向传播需要中间值时，重新运行该段的前向以物化它们，然后微分。

示例：32 层 Transformer 分成 32 段，每段 1 层。

- 内存：32 层输入（小）vs 32 *（每层激活体积）（巨大）。
- 额外计算：每段额外一次前向，即总前向 FLOPs 约 33%（因为反向是前向的 2 倍，完整步骤变成 1 + 1 + 2 = 4 单位而非 1 + 2 = 3）。

这是原始 Chen 等人 2016 配方：每 `sqrt(L)` 层一个检查点以平衡内存和计算。对于 L=64，即 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活成本相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留便宜存储的激活（线性投影、残差），仅重计算昂贵的（注意力）。你为最小 FLOPs 付费但节省 O(L^2) 内存。

Megatron-Core 将其作为"选择性"激活重计算实现。用于大多数 2024+ 前沿训练运行。

### 卸载

重计算的替代方案：在前向和反向之间将激活发送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重物化成本时有益。混合策略很常见：某些层检查点，其他层卸载。

FSDP2 将卸载作为一等选项。当 GPU 内存受限但 CPU-GPU 传输有空间时，卸载表现出色。

### 重计算成本模型

每步 FLOPs，朴素检查点每 `k` 层（共 `L` 层）：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段中每层额外一次前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点仅重计算注意力内核，非整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

完全检查点（段大小 1）：仅存储 `L * input_volume`（标准 Transformer 约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存储 `L/k * A` 加上活动段内 `k-1` 层。

在 `k = sqrt(L)` 时，内存和重计算成本都与 `sqrt(L)` 缩放——均匀成本层的最优权衡。

### 何时不检查点

- 流水线阶段内层已经在运行中。它们反正必须完成。
- 如果它们主导阶段计算，第一层和最后一层（在 Transformer 中罕见）。
- 已经使用 FlashAttention 的注意力内核——Flash 已经快速重计算 softmax，所以层级检查点增加很少。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装段。PyTorch 仅存储 `input`，反向时重计算其他一切。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些段包装。

3. **手动显式重计算：** 自己写反向传播，调用自定义 `recompute_forward` 与存储输入重复前向。

三者功能结果相同。包装器是标准习语。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入必须在重计算时收集或重新分散；处理通信成本。
- **流水线并行：** 典型模式是检查点每个流水线阶段的前向，使反向顺序微批处理可以重用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向匹配，否则 FP8 尺度漂移。大多数框架快照尺度。

## 动手实践

### 步骤1：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤2：需要所有激活的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤3：每 k 层检查点内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用实践

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的规范包装器。包装函数；仅存储输入，反向时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练标准。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy` 将分片卸载到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：CPU 卸载用于优化器状态和激活，补充检查点。

## 产出成果

本课程产出 `outputs/prompt-activation-recompute-policy.md` —— 一个提示，接受你的模型配置（层、隐藏、序列、批处理）和可用 GPU 内存，并发出每层重计算策略（无/选择性/完全/卸载）。

## 练习题

1. 验证正确性。运行 `model_forward` + `model_backward`（完全激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（段）。参数梯度必须机器精度相同。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOP 开销和内存。找到曲线拐点。

3. 实现选择性检查点：存储注意力模块输入但不存储其内部。在 seq=8192 的 32 层模型上测量与完整层检查点相比的 FLOP 开销。

4. 添加卸载。将段输入保存到模拟"CPU 缓冲区"（单独列表）。测量"PCIe 带宽"为字节/时间，找到卸载与重计算的盈亏平衡点。

5. 对真实 PyTorch Transformer 进行基准测试，有和没有 `torch.cuda.checkpoint`。测量内存（通过 `torch.cuda.max_memory_allocated`）和步进时间。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 梯度检查点 | "通过重做前向节省内存" | 仅存储段输入；反向期间重计算中间值以获取梯度支持张量 |
| 激活重计算 | "与检查点相同" | 相同技术的 HPC 风格名称 |
| 段大小 (k) | "每检查点多少层" | 其内部被丢弃和物化的层数 |
| 选择性检查点 | "Korthikanti 的技巧" | 仅重计算昂贵存储的激活（注意力 softmax）；保留便宜的 |
| 完全检查点 | "朴素版本" | 每段中重计算每层的中间值 |
| 块检查点 | "粗粒度" | 检查点整个 Transformer 块；最大粒度 |
| FLOP 开销 | "计算税" | 每步额外 FLOPs =（重计算 FLOPs）/（前向 + 反向 FLOPs）；朴素 33%，选择性 5% |
| 激活卸载 | "发送到 CPU" | 前向->反向间将激活移到 CPU RAM；重计算替代方案 |
| sqrt-L 规则 | "经典最优" | 对于均匀成本层，最优检查点间距是 sqrt(L) 层 |
| 注意力-softmax 体积 | "O(L^2) 问题" | L^2 * 头数 * 批处理浮点数；在长上下文中主导激活内存 |

## 延伸阅读

- [Chen 等人，2016 — "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 形式化梯度检查点的原始论文
- [Korthikanti 等人，2022 — "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算和正式成本分析
- [Pudipeddi 等人，2020 — "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 通过反向模式重物化的替代恒定内存方法
- [Ren 等人，2021 — "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 规模化的激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、完全和块模式
