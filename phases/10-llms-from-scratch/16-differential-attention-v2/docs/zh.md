# 差分注意力 (V2)

> Softmax 注意力将少量概率分散到每个不匹配的 Token 上。在10万个 Token 上，这些噪声累积起来淹没了信号。差分 Transformer (Ye et al., ICLR 2025) 通过计算两个 softmax 的差值来修复这个问题，减去共享的噪声基底。DIFF V2 (Microsoft, 2026年1月) 是生产堆栈的重写：匹配基线 Transformer 的解码延迟，无需自定义内核，兼容 FlashAttention。本课程是 V1 到 V2 的端到端讲解，包含一个可在 stdlib Python 中运行的差分操作工作玩具实现。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 第7阶段·02（自注意力），第7阶段·15（注意力变体），第10阶段·14（架构详解）
**时间:** ~60分钟

## 学习目标

- 精确说明为什么 softmax 注意力有噪声基底，以及为什么它随上下文长度增长。
- 推导差分注意力公式并解释为什么减法能抵消共享的噪声成分同时保留信号。
- 回顾 V1 到 V2 的差异：什么变快了，什么变简单了，什么变得更稳定了，以及为什么每个改变对于生产预训练都是必要的。
- 用纯 Python 从头实现差分注意力，并在合成信号加噪声查询上实证验证噪声抵消特性。

## 问题背景

标准 softmax 注意力有一个数学特性，在大规模时会变成操作上的麻烦。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不可能产生精确的零——每个不匹配的 Token 都得到一些正质量。该残余质量是噪声，并且随上下文长度缩放。在12.8万个 Token 时，即使每个不匹配的 Token 只得到0.001%的概率，12.7999万个 Token 合起来贡献约12%的总量。模型必须学会绕过一个随上下文增长的噪声基底。

这在经验上表现为注意力头干扰：长上下文 RAG 中的幻觉引用，10万 Token 检索任务上的"迷失在 middle"失败，以及32k 以上的针在干草堆基准上的微妙准确性下降。差分 Transformer 论文 (arXiv:2410.05258, ICLR 2025) 测量了差距：DIFF Transformer 在相同大小的基线模型上达到了更低的困惑度、更高的长上下文准确性和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。它的值缓存必须在每个解码步骤加载两次，它需要破坏 FlashAttention 兼容性的自定义 CUDA 内核，它的每头 RMSNorm 在70B+ 规模的长期训练中不稳定。DIFF V2 (Microsoft unilm 博客, 2026年1月20日) 修复了所有三个问题。本课程讲解两个版本，构建差分操作符，并在玩具查询上基准测试噪声抵消。

## 概念讲解

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不是0——它在方差 `||q||^2 / d` 下围绕零波动。Softmax 归一化后，每个无关 Token 仍然贡献 `O(1/N)` 到加权和。无关 Token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小量。

模型想要的是类似硬 top-k 的东西：在匹配 Token 上高权重，在其他地方接近零权重。Softmax 太平滑，无法直接做到这一点。

### 差分思想

将每个头的 Q 和 K 投影分成两个：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了两个图共享的任何噪声分布。如果两个图在12.7万个无关 Token 上都有大致均匀的权重（在随机初始化时它们会这样），这些就会抵消。信号——在少数真正相关 Token 上的峰值权重——只有在它以相同幅度出现在两个图中时才会抵消，而一旦模型训练，这不会发生。

`lambda` 是每个头的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小的正数如0.8。

### 为什么这类似于带头噪声抵消

想象两个嘈杂的麦克风录制同一个声音。两个都拾取说话者加上相关的背景噪声。将一个从另一个中减去，共享的噪声就消失了。声音幸存下来，因为两个信号在相位或幅度上相差足够大以防止完全抵消。每头的 `lambda` 正是学习这种平衡。

### V1 vs V2：差异

V1 保持参数量与基线 Transformer 相等。为了每头得到两个查询，它将头维度减半。这牺牲了头的表达能力——更痛苦的是——每头的值缓存减半。解码必须每步加载值缓存两次（每个 softmax 分支一次）。结果：尽管参数量匹配，解码比基线慢。

V2 将查询头数量加倍，保持 KV 头相同（从升维投影借用参数）。头维度保持与基线相同。减法后，额外维度被投影回以匹配基线 Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配基线（KV 缓存只加载一次）。
2. FlashAttention 无需改动即可运行（无需自定义内核）。
3. 解码时的算术强度上升（每字节从 HBM 加载有更多计算）。

V2 还移除了 V1 用于稳定减法的每头 RMSNorm。在70B级预训练规模上，该 RMSNorm 在训练后期不稳定。V2 用更简单的初始化方案替代它，无需额外模块即可保持训练稳定。

### 何时使用它

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG (64k+) | 更清晰的注意力图，更少的幻觉引用 |
| 针在干草堆基准 | 32k 以上显著准确性提升 |
| 多文档 QA | 更少的跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得架构改变 |
| 短聊天 (< 4k) | 与基线基本无法区分 |

价值随上下文长度增长。在4k Token 时，噪声基底足够小，标准注意力就可以。在128k 时，它正在伤害你。

### 如何与其他2026年旋钮叠加

| 特性 | 与 DIFF V2 兼容？ |
|------|------------------|
| GQA | 是 (V2 增加 Q 头，不是 KV 头) |
| MLA (DeepSeek) | 原则上可以，没有已发表的论文将它们结合 |
| MoE | 是 (注意力独立于 MLP 块) |
| RoPE | 是 (不变) |
| YaRN / 长上下文缩放 | 是 (正是 DIFF 最帮助的地方) |
| FlashAttention | 是 (V2 可以，V1 不行) |
| 推测解码 | 是 (注意力改变对推测解码循环不可见) |

## 动手实践

`code/main.py` 用纯 Python 实现差分注意力。一个具有已知信号加噪声结构的玩具查询让你直接测量噪声抵消比率。

### 步骤1：标准 softmax 注意力

Stdlib 矩阵操作：列表的列表，手动 matmul，用减去最大值实现数值稳定的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤2：将 Q, K 分成两半

V1 风格：将头维度减半。V2 风格：保持头维度并将头数量加倍。玩具实现使用 V1 以 pedagogical 清晰——数学相同，只有簿记不同。

### 步骤3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负数。这没问题——值缓存仍然处理有符号贡献。随后的 V 投影吸收符号。

### 步骤4：噪声抵消测量

构建一个长度1024的合成序列。将信号 Token 放在已知位置，其余用噪声填充。计算 (a) 标准 softmax 注意力在信号位置的权重和 (b) 差分注意力权重。测量每个的信噪比。根据两个分支训练差异的程度，DIFF 注意力可靠地产生高出3倍-10倍的信噪比。

### 步骤5：V1 vs V2 参数核算

给定配置 (hidden=4096, heads=32, d_head=128)，打印：

- 基线 Transformer：Q, K, V 每个大小为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q, K 每个大小为 `hidden * hidden`，V 大小为 `hidden * hidden`（不变），内部头维度减半。添加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。额外维度在 O_W 之前投影回。添加相同的 `lambda` 参数。

玩具测量 V2 的额外参数成本（每个注意力块大约 `hidden * hidden` 额外）并打印它。

## 实际应用

截至2026年4月，DIFF V2 尚未在每个生产推理服务器中发布，但正在 vLLM 和 SGLang 中集成。同时，它出现在：

- Microsoft 内部长上下文生产模型。
- 几个针对256k+ 上下文的开源模型训练运行的研究复现。
- 将 DIFF 注意力与交替层滑动窗口注意力结合的混合架构。

2026年何时使用它：

- 从头开始训练针对64k+ 有效上下文的新模型。从一开始添加差分注意力；稍后重新训练很昂贵。
- 对"迷失在 middle"失败主导评估的长上下文模型进行微调。Q 投影上的 LoRA 可以近似 DIFF 结构。

何时不使用：

- 你正在服务具有稳定长上下文性能的预训练稠密模型。重新训练成本很少在现有权重上回本。
- 你的上下文总是小于16k。噪声基底可忽略。

## 产出成果

本课程产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉配置文件和训练预算，它产生将差分注意力添加到新预训练运行或 LoRA 微调的集成计划。

## 练习题

1. 运行 `code/main.py`。验证差分注意力报告的信噪比高于合成查询上的标准 softmax 注意力。改变噪声幅度并显示标准注意力变得无法使用的交叉点。

2. 计算从基线到 DIFF V1 和从基线到 DIFF V2 的7B级模型 (hidden=4096, heads=32, d_head=128, 32层) 的参数计数增量。显示哪些组件获得参数，哪些保持不变。

3. 阅读 DIFF V1 论文的第3节 (arXiv:2410.05258) 和 DIFF V2 Hugging Face 博客的第2节。用两句话解释为什么需要 V1 每头 RMSNorm 以及为什么 V2 可以在不导致训练发散的情况下移除它。

4. 实现一个消融：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上，测量信噪比如何随扫掠变化。识别最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择8个 KV 头和32个 Q 头。显示 KV 缓存大小与具有相同 (8, 32) 配置的基线 GQA 模型匹配。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 差分注意力 | "两个 softmax 相减" | 将 Q, K 分成两半，计算两个 softmax 图，将第二个（按 lambda 缩放）从第一个中减去，然后乘以 V |
| 噪声基底 | "softmax 的非零尾部" | Softmax 放在每个无关 Token 上的 O(1/N) 权重，在长上下文中求和为 O(1) |
| lambda | "减法尺度" | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；将头维度减半以保持参数量，需要自定义内核，解码更慢 |
| DIFF V2 | "2026年1月修复" | 将 Q 头加倍保持 KV 头；匹配基线解码速度并与 FlashAttention 兼容 |
| 每头 RMSNorm | "V1 稳定器" | V1 在差分后应用的额外归一化；V2 移除它以防止训练后期不稳定 |
| 信噪比 | "多少注意力被浪费" | 真实信号位置的权重与无关位置平均权重的比率 |
| 迷失在 middle | "长上下文失败模式" | 检索准确性在长上下文中间文档处下降的经验现象——差分注意力减少这一点 |
| 算术强度 | "每加载字节的 FLOP" | V2 通过每 KV 加载加倍查询提高的解码比率；对内存受限解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 带有噪声抵消理论和长上下文消融的原始论文
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产堆栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 为什么减法恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — 差分注意力针对的长上下文基准
