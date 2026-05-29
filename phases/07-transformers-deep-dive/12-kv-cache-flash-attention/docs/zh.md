# KV Cache、Flash注意力机制与推理优化

> 训练并行且FLOP受限。推理串行且内存受限。不同瓶颈,不同技巧。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力机制)、阶段7课程05(完整Transformer)、阶段7课程07(GPT)
**时间:** ~75分钟

## 问题背景

朴素自回归解码器生成`N`词元做`O(N²)`工作:每步重算整个前缀注意力。4K词元响应是16M注意力操作,多数冗余。前缀词元每隐藏状态一旦算好确定——你仅需新词元query对缓存前一切key和value。

此外,注意力本身移大量数据。标准注意力显N×N分数矩阵、N×d softmax输出、N×d最终输出——HBM读写太多。N≥2K,注意力成内存受限先FLOP受限。经典注意力kernel低用现代GPU 4-10×。

两优化,均来自Dao等,推前沿推理从"慢"到"快":

1. **KV cache。**存每前缀词元K和V向量。每新词元注意力是对缓存key一次query。推理从每生成步`O(N²)`降到`O(N)`。
2. **Flash注意力机制。**tile注意力计算使完整N×N矩阵永不触HBM。softmax+matmul全在SRAM。A100上2-4× wall-clock加速;H100 FP8上5-10×。

到2026两者通用。每个生产推理栈(vLLM、TensorRT-LLM、SGLang、llama.cpp)假设它们。每个前沿模型都默认启用Flash注意力机制。

## 概念讲解

![KV cache增长和Flash注意力机制tile](../assets/kv-cache-flash-attn.svg)

### KV cache数学

每解码器层,每词元,每头:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K和V
```

对7B模型配32层、32头、d_head=128、fp16:

```
每词元每层 = 2 * 128 * 2 = 512字节
每词元(32层) = 16 KB
32K上下文 = 512 MB
```

对Llama 3 70B(80层、d_head=128、GQA配8 KV头):

```
每词元每层 = 2 * 8 * 128 * 2 = 4096字节(4 KB)
32K上下文 = 10.4 GB
```

这10GB就是为何Llama 3 70B在128K上下文、批大小为1时，40GB A100的大部分显存都被KV cache占用。

**GQA是KV cache胜利。**MHA配64头会是32GB。MLA压缩更远。

### Flash注意力机制——tile技巧

标准注意力:

```
S = Q @ K^T          (HBM读, N×N, HBM写)
P = softmax(S)       (HBM读, HBM写)
O = P @ V            (HBM读, HBM写)
```

三HBM往返。H100上,HBM带宽3 TB/s;SRAM 30 TB/s。每次HBM访问都比片上操作慢约10倍。

Flash注意力机制:

```
对每Q块(tile大小约128×128):
    加Q_tile进SRAM
    对每K, V块:
        加K_tile, V_tile进SRAM
        算S_tile = Q_tile @ K_tile^T     (SRAM)
        运softmax聚合                      (SRAM)
        积进O_tile                         (SRAM)
    写O_tile到HBM
```

每个tile只有一次HBM往返。总内存占用从`O(N²)`降到`O(N)`。反向传播从正向重算某些值而非存储它们——又一次内存胜利。

**数值技巧。**运行softmax跨tile维护`(max, sum)`使最终归一化精确。非近似——Flash注意力机制算与标准注意力bit-identical输出(除fp16非结合性)。

**版本演进:**

| 版本 | 年份 | 关键变化 | 参考硬件加速 |
|------|------|----------|--------------|
| Flash 1 | 2022 | Tile SRAM kernel | A100上2× |
| Flash 2 | 2023 | 更好并行,因果优先顺序 | A100上3× |
| Flash 3 | 2024 | Hopper异步,FP8 | H100上1.5-2×(~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5级管道,软件exp2 | 推理优先(初始仅正向) |

Flash 4发时仅正向。训练仍用Flash 3。Flash 4 GQA和varlen支持待(2026中)。

### 投机解码——其他延迟赢

便宜模型提议N词元。大模型并行验证全部N。如果验证接受k词元,你为k生成付1大模型前向。典型代码和散文k=3-5。

2026默认:
- **EAGLE 2 / Medusa。**集成草稿头共享验证器隐藏状态。2-3×加速无质量损。
- **配草稿模型投机解码。**消费硬件上2-4×加速。
- **Lookahead解码。**Jacobi迭代;无草稿模型需。小众但免费。

### 连续批

经典批推理:等最慢序列完,后起新批。短响应早完浪费GPU。

连续批(首先由Orca提出，现已集成在vLLM、TensorRT-LLM、SGLang中):旧序列完成后立刻换入新请求。典型聊天工作负载5-10×吞吐增益。

### PagedAttention——KV cache作虚拟内存

vLLM头条功能。KV cache在16词元块分配;页表映射逻辑位置到物理块。让你跨并行样本共享KV(beam search、并行采样)、提示词缓存热换前缀、和去碎片内存。比朴素连续分配4×吞吐改进。

## 动手实践

见`code/main.py`。我们实现:

1. 朴素`O(N²)`增量解码器。
2. `O(N)` KV缓存解码器。
3. tile softmax模拟Flash注意力机制运行max算法。

### Step 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

原理简单：在每层每个头的列表中存储不断追加的K、V向量。

### Step 2: tile softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash注意力机制式softmax(qK^T)V配运行max/sum。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次`softmax(qK) V`bit-identical,但任何时工作集是`tile × d_head`块,非完整`N × d_head`。

### Step 3: 100词元生成比朴素vs缓存解码

算注意力操作。朴素:`O(N²)`=5050。缓存:`O(N)`=100。代码打印两者。

## 实际应用

```python
# HuggingFace transformers仅解码器generate()自动启用KV cache。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper用FA3
    torch_dtype="bfloat16",
)
# generate()自动用KV cache
```

vLLM生产:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求前缀缓存是2026大赢——同系统提示词、few-shot示例或长上下文文档跨调用复用KV。对智能体工作负载配重复工具提示词,前缀缓存常规5×吞吐增益。

## 产出成果

见`outputs/skill-inference-optimizer.md`。技能为新推理部署选注意力实现、KV cache策略、量化和投机解码。

## 练习题

1. **简单。**运行`code/main.py`。确认朴素和缓存解码器产同输出;注意op数差。
2. **中等。**实现前缀缓存:给定提示词P和几补全,对P一次前向填KV cache,后每补全分支。测比每重编码P加速。
3. **困难。**实现玩具PagedAttention:KV cache在固定16词元块配free-list。序列完时回块池。模拟1000配变长聊天补全。比连续分配内存碎片。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| KV cache | "让解码快技巧" | 存每前缀词元K和V；新词元的query关注已缓存的K/V而无需重新计算。 |
| HBM | "GPU主内存" | 高带宽内存;H100上80GB,B200上192GB。~3TB/s带宽。 |
| SRAM | "芯片内存" | 每SM快内存,H100上约256KB每SM。~30TB/s带宽。 |
| Flash注意力机制 | "Tile注意力kernel" | 算注意力不显N×N在HBM。 |
| 连续批 | "无等待批" | 老序列出立刻换新序列进,不排批。 |
| PagedAttention | "vLLM头条" | KV cache在固定块配页表分配;消碎片。 |
| 前缀缓存 | "复用长提示词" | 跨请求缓存共享前缀KV;智能体大成本砍。 |
| 投机解码 | "草稿+验证" | 便宜草稿模型提议词元;大模型一次pass验证k。 |

## 延伸阅读

- [Dao等(2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)——Flash 1。
- [Dao(2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)——Flash 2。
- [Shah等(2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)——Flash 3。
- [FlashAttention-4发注(Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention)——Blackwell 5级管道和软件exp2技巧;读repo README课程提仅正向发注意事项。
- [Kwon等(2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)——vLLM论文。
- [Leviathan等(2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)——投机解码。
- [Li等(2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)——课程引集成草稿法EAGLE-1/2论文。
- [Cai等(2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)——与EAGLE并引Medusa法。
- [vLLM docs—PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)——16词元块和页表设计典型深挖。