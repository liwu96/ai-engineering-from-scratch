# Transformer为何出现 — RNN的问题

> RNN逐个处理词元。Transformer一次性处理所有词元。那个单一的架构赌注改变了2017年后深度学习的每个scaling曲线。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段3(深度学习核心)、阶段5课程09(序列到序列)、阶段5课程10(注意力机制)
**时间:** ~45分钟

## 问题背景

2017年前,地球上每个state-of-the-art序列模型——语言、翻译、语音——都是循环神经网络。LSTM和GRU在ImageNet等效翻译基准上赢了半个十年。它们是任何人唯一有的工具。

它们有三个致命弱点。顺序计算意味你不能沿时间轴并行化:词元`t+1`需要词元`t`的隐藏状态。1,024词元序列意味GPU上1,024串行步骤,而GPU每周期可做1,000,000浮点运算。训练wall-clock时间在为并行设计的硬件上随序列长度线性scaling。

梯度消失意味50词元前的信息已经压缩过50个非线性。门控循环单元(LSTM, GRU)软化挤压但从未消除。长程依赖——"去年夏天我去京都的飞机上读的那本书是……"——常规失败。

固定宽度隐藏状态意味编码器在解码器看到任何东西前将整个源序列挤压到单个向量。源是5词元还是500词元不重要;瓶颈是同样形状。

2017论文"Attention Is All You Need"提议激进的事:完全放弃循环。让每个位置并行地attend到每个其他位置。用一个大矩阵乘法训练而非1,024个顺序的。

到2026年结果主导每个模态。语言(GPT-5, Claude 4, Llama 4)、视觉(ViT, DINOv2, SAM 3)、音频(Whisper)、生物学(AlphaFold 3)、机器人(RT-2)。同样block,不同输入。

## 概念讲解

![RNN顺序计算vs Transformer并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。**RNN计算`h_t = f(h_{t-1}, x_t)`。每步依赖前一步。你不能在`h_4`前计算`h_5`。在现代配10,000+并行核心GPU上,这浪费长序列99%硅片。

**注意力作为广播。**自注意力为每对`(i, j)`同时计算`output_i = sum_j(a_ij * v_j)`。整个N×N注意力矩阵在一个批量matmul填充。没有步骤依赖另一个。GPU爱它。

**加速不是常数。**是`O(N)`串行深度和`O(1)`串行深度的差异。实践中,在匹配硬件N=512时transformer每轮训练5-10×更快,差距随序列长度扩大直到你撞到注意力的`O(N²)`内存墙(Flash注意力后来修复——见课程12)。

**Transformer代价。**注意力内存scaling为`O(N²)`。2K上下文还好。128K上下文,你需要滑动窗口、RoPE extrapolation、Flash注意力tiling、或线性注意力变体。循环时间和内存都是`O(N)`;transformer交易时间换内存然后通过并行赢回时间。

**归纳偏移转变。**RNN假设局部和近性。Transformer不假设——每对都是候选注意力。这就是为何transformer需要更多数据训练好但一旦有数据scaling更远。Chinchilla(2022)形式化:给定足够词元,等参数计数transformer总是胜RNN。

## 动手实践

这里没有神经网络——我们数值模拟核心瓶颈让你在笔记本感受差距。

### Step 1: 测量串行深度

见`code/main.py`。我们建两函数。一个编码序列为加法链(串行,像RNN)。一个编码为并行reduction(广播,像注意力)。同样数学,不同依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 不可并行:h依赖前h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 每x独立
```

我们在长达100,000元素序列上定时两者。RNN版O(N)单CPU pipeline。即使在纯Python,attention-style reduction在长度≥1,000胜出因为Python`sum()`在C实现无每步解释器开销迭代。

### Step 2: 计数理论操作

两算法做N加。差异是*依赖深度*:下个开始前多少操作必须顺序发生。RNN深度=N。Attention深度=树reduction时log(N),或并行scan时1。深度而非操作计数决定GPU时间。

### Step 3: 长序列实证scaling

我们打印让O(N)差距可见的时间表。2026 Mac笔记本上,1,000元素下序列太快测量。100,000序列显示清晰线性扫描。缩放到12层LSTM等效的16,384词元transformer你看到为何2016训练wall-clock是阻塞器。

## 实际应用

2026何时仍选RNN:

| 情况 | 选择 |
|------|------|
| 流推理、一次一词元、常数内存 | RNN或状态空间模型(Mamba, RWKV) |
| 极长序列(>1M词元)注意力内存爆炸处 | 线性注意力、Mamba 2、Hyena |
| 无matmul加速器边缘设备 | Depthwise-separable RNN仍FLOPs/watt胜 |
| 其他任何(训练、批推理、128K内上下文) | Transformer |

状态空间模型(SSM)如Mamba本质是RNN配结构参数化给它们两者最佳:`O(N)` scan内存、通过选择性scan并行训练。它们恢复90%transformer质量配更好长上下文scaling。2026大多数前沿实验室训练混合SSM+transformer模型(如Jamba, Samba)——循环不死,它是组件。

## 产出成果

见`outputs/skill-architecture-picker.md`。技能为新序列问题给定长度、吞吐、和训练预算约束选架构。它应总拒绝推荐纯RNN用于>1B词元训练运行而不陈述权衡。

## 练习题

1. **简单。**取`code/main.py`的`rnn_style`并将标量隐藏状态替换为长度64隐藏状态向量。重测。串行开销随隐藏状态维度增长多少?
2. **中等。**在纯Python实现并行prefix-sum(Hillis-Steele scan)。验证它产生与长度1024串行scan同样数值输出。计数深度。
3. **困难。**将attention-style reduction移植到GPU PyTorch。扫序列长度从64到65,536定时两者。绘制并解释曲线形状。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 循环 | "RNN顺序" | 计算中步骤`t`依赖步骤`t-1`,强制沿时间轴串行执行。 |
| 串行深度 | "图何深" | 依赖ops最长链;即使在无限硬件上也bounds wall-clock。 |
| 注意力 | "让词元互看" | 加权求和`sum_j a_ij v_j`其中`a_ij`来自位置i和j相似度分数。 |
| 上下文窗口 | "模型见多少" | 注意力层可作为输入的位置数;二次内存成本在此scaling。 |
| 归纳偏移 | "架构中假设" | 关于数据看起来像什么prior;CNN假设平移不变,RNN假设近性。 |
| 状态空间模型 | "有代数的RNN" | 为通过结构状态空间矩阵并行训练参数化的循环。 |
| 二次瓶颈 | "为何上下文贵" | 注意力内存=序列长度`O(N²)`;Flash注意力隐藏常数而非scaling。 |

## 延伸阅读

- [Vaswani等(2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)——杀死主流NLP循环的论文。
- [Bahdanau, Cho, Bengio(2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)——注意力出生处,螺栓在RNN上。
- [Hochreiter, Schmidhuber(1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)——原始LSTM论文,作为记录。
- [Gu, Dao(2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)——现代循环对transformer的回答。