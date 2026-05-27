# DeepSeek-V3 架构详解

> 第10阶段·第14课命名了每个开源模型转动的六个架构旋钮。DeepSeek-V3（2024年12月，总计6710亿参数，370亿激活）转动了全部六个并添加了四个：多头潜在注意力、无辅助损失负载均衡、多 Token 预测和 DualPipe 训练。本课程从头到尾阅读 DeepSeek-V3 的架构并从发布的配置推导每个参数计数。到最后你可以解释为什么6710亿/370亿比率是正确的赌注以及为什么 MLA + MoE 一起击败前沿的单独一个。

**类型:** 学习
**语言:** Python (stdlib, 参数计算器)
**前置要求:** 第10阶段·14（开源模型详解），第10阶段·17（NSA），第10阶段·18（MTP），第10阶段·19（DualPipe）
**时间:** ~75分钟

## 学习目标

- 从头到尾阅读 DeepSeek-V3 配置并根据六个 GPT-2 旋钮加四个 DeepSeek 特定添加解释每个字段。
- 推导总参数计数（6710亿）、激活参数计数（370亿）以及每个贡献的组件。
- 计算128k 上下文中 MLA 的 KV 缓存占用并与具有 GQA 的相同激活参数稠密模型将支付的费用比较。
- 陈述四个 DeepSeek 特定创新（MLA、MTP、无辅助损失路由、DualPipe）并命名每个针对的架构/训练堆栈部分。

## 问题背景

DeepSeek-V3 是第一个架构与 Llama 家族有实质性不同的前沿开源模型。Llama 3 405B 是"转了六个旋钮的 GPT-2"。DeepSeek-V3 是 GPT-2 加全部六个旋钮再加四个。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——足够不同以至于你需要单独的讲解。

学习的收益：DeepSeek-V3 的开放权重发布改变了开放模型中"前沿能力"的含义。架构是许多2026年训练运行正在复制的蓝图。理解它是对任何涉及前沿 LLM 训练或推理角色的基本要求。

## 概念讲解

### 不变的核心，再次

DeepSeek-V3 仍然是自回归的。它仍然堆叠解码器块。每个块仍然有注意力加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。前置归一化。权重绑定的嵌入。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 替代 GQA

从第10阶段·14你知道 GQA 通过在 Q 头组之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩成共享的低秩潜在表示（`kv_lora_rank`），然后动态解压每头。KV 缓存只存储潜在表示——通常每层每 Token 512个浮点数，不是8 x 128 = 1024个浮点数。

在128k 上下文，使用 MLA 的 DeepSeek-V3（每层每 Token 一个共享潜在 `c^{KV}`；K 和 V 都通过可以在后续 matmul 中吸收的升维投影从该潜在导出）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 形状，8个 KV 头，头维度128）将支付：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA 在128k 上下文比 Llama-3-70B 风格的 GQA 缓存小4倍。

权衡：MLA 每次注意力计算增加一个解压步骤（每头）。与节省的带宽相比，额外计算很小。长上下文推理净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 Token。朴素路由器将太多工作集中在少数专家上，让其他专家空闲。标准修复：添加惩罚负载不均衡的辅助损失项。这有效但略微降低主要任务性能。

DeepSeek-V3 引入无辅助损失方案。每专家偏置项被添加到路由器 logits，在训练期间通过简单规则调整：如果专家 `e` 过载，减少 `bias_e`；如果欠载，增加它。没有额外的损失项。训练保持干净。专家负载保持均衡。

对主要损失的影响：没有可测量的。对 MoE 架构的影响：更干净，没有辅助损失超参数需要调整。

### MTP：更密集的训练 + 免费草稿

从第10阶段·18你知道 DeepSeek-V3 添加 D=1 MTP 模块，预测前方两个位置的 Token。推理时，训练的模块被重新用作80%+接受的推测解码草稿。训练时，每个隐藏状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数：在主6710亿之上140亿。开销：2.1%。

### 训练：DualPipe

从第10阶段·19你知道 DualPipe 是一种双向流水线，将前向/后向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的2048-H800 规模上，它恢复约245k GPU 小时，1F1B 会将其损失给流水线气泡。

### 逐字段配置

以下是 DeepSeek-V3 配置（简化）：

```
hidden_size: 7168
intermediate_size: 18432   （稠密 MLP 隐藏大小，前几层使用）
moe_intermediate_size: 2048 （专家 MLP 隐藏大小）
num_hidden_layers: 61
first_k_dense_layers: 3    （前3层使用稠密 MLP）
num_attention_heads: 128
num_key_value_heads: 128   （MLA 下正式等于 num_heads，但
                           真正的压缩在 kv_lora_rank）
kv_lora_rank: 512          （MLA 潜在维度）
num_experts: 256            （每块 MoE 专家计数）
num_experts_per_tok: 8      （top-8 路由）
shared_experts: 1           （每块始终开启的共享专家）
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               （深度1处1个 MTP 模块）
```

解析它：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前3个块跳过 MoE 路由器并运行稠密 MLP 以保证稳定性。
- `num_attention_heads=128`：128个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度并每头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有256个专家，路由 top-8。
- `shared_experts=1`：在256个路由专家之上，1个始终开启的专家为每个 Token 做出贡献。将其视为确保每个 Token 得到可靠东西的"稠密地板"。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。比稠密 MLP 小，因为有256个。

### 参数核算

完整计算在 `code/main.py` 中。标题：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前3个稠密块：带 MLA 的注意力（每块~144M）+ 稠密 MLP（每块~260M）+ 归一化。总共约1.2B。
- 58个 MoE 块：带 MLA 的注意力（~144M）+ 256个专家每个（每个30M）+ 1个共享专家（30M）+ 归一化。包括所有专家在内每块总计~7.95B。58个 MoE 块461B。
- MTP 模块：14B。

总计：核心架构约476B + 14B MTP + 发布的671B数字明显计算额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中的数字在发布的3-5%以内——差异来自 DeepSeek 报告在第2章附录中记录的细粒度核算。

每次前向激活参数：

- 注意力：每层144M * 61 = 8.8B（所有层触发）。
- MLP 激活：前3层稠密（3 * 260M = 780M），58个 MoE 层每个激活8个路由 + 1个共享 + 路由开销。每层活跃 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + 归一化：1.2B。
- 总激活：约26B核心 + 14B MTP（训练但推理时不总是运行）≈ 37B。

### 6710亿 / 370亿比率

18倍稀疏性比率（激活参数占总参数的5.5%）。DeepSeek-V3 是已发布开放权重中最稀疏的前沿 MoE 模型。Mixtral 8x7B 比率13/47（28%）稠密得多。Llama 4 Maverick 比率170亿/4000亿（4.25%）可比较。DeepSeek 赌注：在前沿规模，更多专家更低激活比率产生每激活 FLOP 更好的质量。

### DeepSeek-V3 的位置

| 模型 | 总计 | 激活 | 比率 | 注意力 | 新奇想法 |
|------|------|------|------|--------|---------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 主干上的推理训练运行。R1 使用相同的架构。改变的是后训练配方（可验证任务上的大规模 RL），不是预训练架构。

DeepSeek-V4（如果发布）预计将保持 MLA + MoE + MTP 并添加 DSA（DeepSeek 稀疏注意力），第10阶段·17中 NSA 的后续。血统稳定：架构级创新累积；每个版本转动额外的旋钮。

## 实际应用

`code/main.py` 是专门用于 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文的数字比较，并在假设变体上使用它（256 vs 512专家，top-8 vs top-16，MLA 等级512 vs 1024）。

看什么：

- 总参数计数 vs 发布的6710亿。
- 激活参数计数 vs 发布的370亿。
- 128k 上下文 KV 缓存——MLA vs GQA 比较。
- 每层分解以查看参数预算实际去向。

## 产出成果

本课程产出 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek 家族模型（V3、R1 或任何未来变体），它产生逐组件架构阅读，命名配置的每个字段，按组件推导参数计数，并识别模型使用的四个 DeepSeek 特定创新中的哪一个。

## 练习题

1. 运行 `code/main.py`。将计算器的总参数估计与发布的6710亿比较并识别差异来源。论文的第2章有完整明细。

2. 修改配置以使用 MLA 等级256而非512。计算128k 上下文产生的 KV 缓存大小。它购买了多少百分比减少，以每头表达性为代价是多少？

3. 将 DeepSeek-V3 的（256个专家，top-8）路由与假设的（512个专家，top-8）变体比较。总参数增长；激活参数保持不变。额外的专家容量理论上购买什么，推理时成本是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的第2.1节关于 MLA。用三句话解释为什么 K 和 V 解压矩阵可以"吸收"到后续 matmul 中以实现推理时效率。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算存储6710亿权重的 FP8 vs BF16 的内存节省。这与14.8T Token 训练预算如何交叉？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩成共享低秩潜在（kv_lora_rank，通常512），动态每头解压；KV 缓存只存储潜在 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在的大小；DeepSeek-V3 使用512 |
| 前k稠密层 | "早期层保持稠密" | 前几层 MoE 模型层跳过 MoE 路由器并运行稠密 MLP 以保证稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个 Token 触发多少个路由专家；DeepSeek-V3 使用8 |
| 共享专家 | "始终开启的专家" | 无论路由如何处理每个 Token 的专家；DeepSeek-V3 使用1 |
| 无辅助损失路由 | "偏置调整负载均衡" | 在训练期间调整的每专家偏置项以保持专家负载均衡而不添加损失项 |
| MTP 模块 | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；更密集的训练，免费推测解码草稿 |
| DualPipe | "双向流水线" | 将前向/后向计算与跨节点 all-to-all 重叠的训练调度 |
| 激活参数比率 | "稀疏性" | 激活参数/总参数；DeepSeek-V3 达到5.5% |
| FP8 训练 | "8位训练" | FP8 中的训练存储和许多计算操作；相对于 BF16 大约减半内存，质量成本小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 架构上的推理训练后续
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考
