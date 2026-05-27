# GPT — 因果语言建模

> BERT看两边。GPT只看过去。三角掩码是现代AI后果最严重的单行代码。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意力机制)、阶段7课程05(完整Transformer)、阶段7课程06(BERT)
**时间:** ~75分钟

## 问题背景

语言模型回答一个问题:给定前`t-1`词元,词元`t`的概率分布是什么?在那个信号上训练——下一词元预测——你得到可逐词元生成任意文本的模型。

要在整个序列上端到端并行训练,你需要每位置预测仅依赖更早位置。否则模型通过看答案 trivially作弊。

因果掩码做这。它是softmax前加到注意力分数的单个上三角`-inf`矩阵。softmax后,那些位置变0。每位置仅可attend到自己和更早位置。且因你对整个序列一次应用,一次前向你得N并行下一词元预测。

GPT-1(2018)、GPT-2(2019)、GPT-3(2020)、GPT-4(2023)、GPT-5(2024)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们全是仅解码器因果transformer配相同核心循环。只是更大、更好数据、更好RLHF。

## 概念讲解

![因果掩码创造三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长`N`序列,建`N × N`矩阵:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

softmax前把`M`加到原始注意力分数。`exp(-inf) = 0`,故掩码位置贡献零权重。注意力矩阵每行是仅过去位置概率分布。

实现成本:一次`torch.tril()`调用。计算时间:纳秒。领域影响:一切。

### 并行训练,串行推理

训练:一次前向整个`(N, d_model)`序列,算N个交叉熵损失(每位置一个),求和,反向。沿序列并行。这是为何GPT训练扩展——一次GPU pass处理批中1M词元。

推理:你逐词元生成。喂`[t1, t2, t3]`,得`t4`。喂`[t1, t2, t3, t4]`,得`t5`。喂`[t1, t2, t3, t4, t5]`,得`t6`。KV cache(课程12)存`t1…tn`隐藏状态故你每步不重算。但推理串行深度=输出长度。那是自回归税,为何解码是每个大语言模型延迟瓶颈。

### 损失——shift-by-one

给定词元`[t1, t2, t3, t4]`:

- 输入:`[t1, t2, t3]`
- 目标:`[t2, t3, t4]`

对每位置`i`,算`-log P(target_i | inputs[:i+1])`。求和。这是整个序列交叉熵。

你听过的每个transformer LM在这个损失上训练。预训练、微调、SFT——同损失,不同数据。

### 解码策略

训练后,采样选择比人们想更重要。

| 方法 | 做什么 | 何时用 |
|------|--------|--------|
| Greedy | 每步argmax | 确定任务、代码补全 |
| Temperature | logits除T,采样 | 创意任务,高T=更多多样性 |
| Top-k | 仅从top-k词元采样 | 杀低概率尾 |
| Top-p(nucleus) | 从累积prob≥p最小集采样 | 2020+默认;适应分布形状 |
| Min-p | 保`p > min_p * max_p`词元 | 2024+;拒长尾比top-p更好 |
| Speculative decoding | 草稿模型提议N词元,大模型验证 | 同质量2-3×延迟降 |

2026年,min-p + temperature 0.7是开源权重模型合理默认。投机解码是任何生产推理栈基本配置。

### "GPT配方"工作原因

1. **仅解码器。**无编码器开销。每层一次attention + FFN pass。
2. **缩放。**124M → 1.5B → 175B → 万亿。Chinchilla缩放定律(课程13)告诉你如何花计算。
3. **上下文学习。**约6B-13B出现。模型可无需微调follow few-shot示例。
4. **RLHF。**人类偏好后训练把原始预训练文本转聊天助手。
5. **Pre-norm + RoPE + SwiGLU。**规模稳定训练。

核心架构自GPT-2未大变。所有有趣发生在数据、规模和后训练。

## 动手实践

### Step 1: 因果掩码

见`code/main.py`。一行:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

softmax前加到注意力分数。这是整个机制。

### Step 2: 2层GPT式模型

堆两解码器块(掩码自注意力+FFN,无交叉注意力)。加词元嵌入、位置编码和unembedding(绑到词元嵌入矩阵——GPT-2以来标准技巧)。

### Step 3: 下一词元预测,端到端

在20词元玩具词表上,每位置产logits。对shift-by-one目标算交叉熵损失。无梯度——这是前向健全检查。

### Step 4: 采样

实现greedy、temperature、top-k、top-p、min-p。在固定提示词上各跑比输出。采样函数10行。

## 实际应用

PyTorch,2026写法:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

底层,`generate()`跑前向、拉最终位置logits、采样下词元、append、重复。每个生产大语言模型推理栈(vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX)配重度优化实现同循环——批prefill、连续批、KV cache paging、投机解码。

**GPT vs BERT,各一行:**GPT预测`P(x_t | x_{<t})`。BERT预测`P(x_masked | x_unmasked)`。损失决定模型能否生成。

## 产出成果

见`outputs/skill-sampling-tuner.md`。技能为新生成任务选采样参数并标记何时需确定解码。

## 练习题

1. **简单。**运行`code/main.py`并验证因果注意力矩阵softmax后是下三角。抽查:行3权重仅在列0-3。
2. **中等。**实现宽度4beam search。比10短提示词beam-4 vs greedy困惑度。Beam总胜否?(提示:通常翻译,非开放聊天。)
3. **困难。**实现投机解码:用微小2层模型作草稿和6层模型作验证器。测100个长64补全wall-clock加速。确认输出匹配验证器greedy。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Causal mask | "三角" | softmax前加到注意力分数的上三角`-inf`矩阵使位置`i`仅看位置`≤ i`。 |
| 下一词元预测 | "损失" | 模型分布对每位置真下词元交叉熵。 |
| 自回归 | "一次生成一个" | 输出反馈作输入;仅训练时并行,生成时非。 |
| Logits | "softmax前分数" | LM头softmax前原始输出;采样发生在这些上。 |
| Temperature | "创意旋钮" | logits除T;T→0=greedy,T→∞=均匀。 |
| Top-p | "核采样" | 截分布到累积≥p最小集;从剩余采样。 |
| Min-p | "比top-p更好" | 保`p ≥ min_p × max_p`词元;截断适应分布锐度。 |
| Speculative decoding | "草稿+验证" | 便宜模型提议N词元;大模型并行验证。 |
| Teacher forcing | "训练技巧" | 训练时,喂真前词元给解码器,非模型预测。每个seq2seq LM标准。 |

## 延伸阅读

- [Radford等(2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)——GPT-1。
- [Radford等(2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)——GPT-2。
- [Brown等(2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)——GPT-3和上下文学习。
- [Leviathan, Kalman, Matias(2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)——投机解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)——典型因果LM参考代码。