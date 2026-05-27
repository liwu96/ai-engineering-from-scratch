# 混合专家模型 (MoE)

> 稠密70B transforme每词元激活每参数。671B MoE每词元仅激活37B并在每个基准胜它。稀疏性是这十年最重要缩放想法。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段7课程07(GPT)
**时间:** ~45分钟

## 问题背景

稠密transformer推理FLOPs等于参数数(前向×2)。放大稠密模型每词元付全账。到2024前沿撞计算墙:要有意义更聪明,需指数更多每词元FLOPs。

混合专家破此链接。每FFN换`E`独立专家+路由器每词元选`k`专家。总参数=`E × FFN_size`。每词元激活参数=`k × FFN_size`。典型2026配置:`E=256`,`k=8`。存储随`E`缩放,计算随`k`缩放。

2026前沿几乎全MoE:DeepSeek-V3(671B总/37B激活)、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。Artificial Analysis独立排行榜,前10开源模型全MoE。

## 概念讲解

![MoE层:路由器每词元选E中k专家](../assets/moe.svg)

### FFN换

稠密transformer块:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE块:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每词元选E中k
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每专家是独立FFN(典型SwiGLU)。路由器是单linear层。每词元选自己`k`专家得门控混合输出。

### 负载平衡问题

如果路由器把90%词元过专家3,其他专家饿死。三修复被试:

1. **辅助负载平衡损失**(Switch Transformer、Mixtral)。加专家使用方差比例惩罚。工作,但添超参和第二梯度信号。
2. **专家容量+词元丢弃**(早期Switch)。每专家最多处理`C × N/E`词元;溢词元跳层。伤质量。
3. **无辅助损失平衡**(DeepSeek-V3)。加学习每专家偏置移路由器top-k选择。偏置在训练损失外更新。无对主目标惩罚。2024大解锁。

DeepSeek-V3方法:每训练步后,对每专家,查使用高于或低于目标。偏置`±γ`推。选择用`scores + bias`。门控用专家概率是原始`scores`不变。解耦路由与表达。

### 共享专家

DeepSeek-V2/V3还把专家分*共享*和*路由*。每词元过所有共享专家。路由专家通过top-k选。共享专家捕获公共知识;路由专家专化。V3跑1共享专家加256路由中top-8。

### 细粒度专家

经典MoE(GShard、Switch):每专家宽如完整FFN。`E`小(8-64),`k`小(1-2)。

现代细粒度MoE(DeepSeek-V3、Qwen-MoE):每专家更窄(1/8 FFN大小)。`E`大(256+),`k`更大(8+)。同总参数,但组合缩放快得多。`C(256, 8) = 400万亿`每词元可能"专家"。质量升,延迟平。

### 成本profile

每词元,每层:

| 配置 | 每词元激活参数 | 总参数 |
|------|----------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B(稠密) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2(MoE) | ~32B | 1T |

DeepSeek-V3几乎每个基准胜Llama 3 70B(稠密)同时**每词元更少激活FLOPs**。更多参数=更多知识。更多激活FLOPs=每词元更多计算。MoE解耦它们。

### 代价:内存

所有专家在GPU活无论谁触发。671B模型需约1.3TB fp16权重VRAM。前沿MoE部署需专家并行——跨GPU shard专家、跨网络路由词元。延迟主导于all-to-all通信,非matmul。

## 动手实践

见`code/main.py`。纯stdlib紧凑MoE层配:

- `n_experts=8` SwiGLU-ish专家(每一个linear,示教)
- top-k=2路由
- softmax归一化门权重
- 每专家偏置无辅助损失平衡

### Step 1: 路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over所选专家原始分数
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择,非门重。这是DeepSeek-V3技巧——偏置纠负载不平衡不导模型预测。

### Step 2: 100词元过路由器

追踪哪些专家常触发。无偏置,用偏斜。配偏置更新循环(过用`-γ`,欠用`+γ`),几次迭代收敛均匀分布。

### Step 3: 参数数比

打印MoE配置"稠密等价"。DeepSeek-V3形:256路由+1共享,8激活,d_model=7168。总参数数惊人。激活数是稠密Llama 3 70B七分之一。

## 实际应用

HuggingFace加载:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026生产推理:vLLM原生支持MoE路由。SGLang有最快专家并行路径。两者自动处理top-k选择和专家并行。

**何时选MoE:**
- 要前沿质量更低每词元推理成本。
- 有VRAM/专家并行基础设施。
- 工作词元重(聊天、代码)非上下文重(长文档)。

**何时不选MoE:**
- 边缘部署——付全存储任何激活FLOP。
- 延迟关键单用户服务——专家路由添开销。
- 小模型(<7B)——MoE质量优势只在计算阈值(~6B激活参数)上现。

## 产出成果

见`outputs/skill-moe-configurator.md`。技能给定参数预算、训练词元和部署目标为新MoE选E、k和共享专家布局。

## 练习题

1. **简单。**运行`code/main.py`。观无辅助损失偏置更新50迭代如何均匀专家使用。
2. **中等。**用hash基路由器(确定,无学习)换学习路由器。比质量和平衡。为何学习路由器更好?
3. **困难。**实现GRPO式"rollout匹配路由"(DeepSeek-V3.2技巧):推理时log哪些专家触发,梯度计算时强制同路由。测玩具策略梯度设置效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 专家 | "FFN之一" | 独立前馈网络;参数专化FFN计算稀疏切片。 |
| 路由器 | "门" | 小linear层每词元对每专家评分;top-k选择。 |
| Top-k路由 | "每词元k激活专家" | 每词元FFN计算精确过k专家,门控加权。 |
| 辅助损失 | "负载平衡惩罚" | 惩罚偏斜专家使用额外损失项。 |
| 无辅助损失 | "DeepSeek-V3技巧" | 仅路由器选择上每专家偏置平衡;无额外梯度。 |
| 共享专家 | "总开" | 每词元过额外专家;捕获公共知识。 |
| 专家并行 | "按专家shard" | 不同专家分布不同GPU;跨网络路由词元。 |
| 稀疏性 | "激活参数<总参数" | 比`k × expert_size / (E × expert_size)`;DeepSeek-V3 37/671≈5.5%。 |

## 延伸阅读

- [Shazeer等(2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)——想法。
- [Fedus, Zoph, Shazeer(2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)——Switch,经典MoE。
- [Jiang等(2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088)——Mixtral 8×7B。
- [DeepSeek-AI(2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)——MLA +无辅助损失MoE + MTP。
- [Wang等(2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)——偏置基平衡论文。
- [Dai等(2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)——课程路由器用细粒度+共享专家拆。
- [Kim等(2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596)——原始共享专家论文。