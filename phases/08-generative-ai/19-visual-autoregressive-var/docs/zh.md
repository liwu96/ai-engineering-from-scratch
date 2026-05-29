# 视觉自回归建模(VAR):下一尺度预测

> 扩散模型时间迭代采样(去噪步)。VAR尺度迭代采样——它预测1x1词元,后2x2,后4x4,到终分辨率,每尺度条件于前。2024论文示VAR图像生成匹GPT式缩放定律并在同计算预算胜DiT。此课建核心机制。

**类型:** 构建
**语言:** Python(配PyTorch)
**前置要求:** 阶段7课程03(多头注意力机制)、阶段8课程06(DDPM)
**时间:** ~90分钟

## 问题背景

自回归生成主导语言建模因缩放可预测:更多计算、更多参数、更低困惑度、更好输出。图像生成2024前有两主AR尝试:PixelRNN/PixelCNN(像素逐像素)和DALL-E 1 / Parti / MuseGAN(VQ-VAE码词元逐词元)。

两者都受生成顺序问题所限。像素和词元排列在2D网格上，但AR模型需要以1D光栅顺序访问。早期角落的像素对图像最终形态毫无概念。生成质量的缩放比GPT文本建模差，且在匹配计算量下从未达到扩散模型质量。

VAR换生成内容修复生成顺序问题。非空间逐预测图像词元,VAR增分辨率预测全图像。步1:预测1x1词元(全图像"摘要")。步2:预测2x2词元网格(粗特征)。步3:预测4x4网格。步K:预测终(H/8)x(W/8)网格。

每尺度attend所有前尺度("尺度顺序"因果)并在自己尺度内并行。顺序问题消失:尺度k全图像一transformer pass产。

## 概念讲解

### VQ-VAE多尺度分词器

VAR需**多尺度离散分词器**。图像x,产渐高分辨率词元网格序列:

```
x -> encoder -> latent f
f -> tokenize at 1x1: 词元网格 z_1 形(1, 1)
f -> tokenize at 2x2: 词元网格 z_2 形(2, 2)
...
f -> tokenize at (H/p)x(W/p): 词元网格 z_K 形(H/p, W/p)
```

每z_k用同码书(典型大小4096-16384)。每尺度tokenization非独立——训使每尺度残差和重构f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

此**残差VQ**变体。尺度k捕获尺度1..k-1漏。解码器取所有尺度嵌入和产图像。

多尺度VQ分词器一次训(像VQGAN)后冻。所有生成工作由顶自回归模型做。

### 下一尺度预测

生成模型是transformer见所有前尺度词元并预测下尺度词元。

输入序列结构:
```
[START, z_1词元, z_2词元, z_3词元, ..., z_K词元]
```

位置嵌入编码尺度指数和尺度内空间位置。注意力尺度顺序因果:尺度k位置(i, j)词元可attend尺度1..k所有词元和尺度k本身内更早词元(VAR用固定位置注意力无尺度内因果——尺度内所有位置并行预测)。

训练损失:每尺度k,给定所有前尺度词元预测词元z_k。离散VQ码交叉熵损失。结构同GPT除"序列"现尺度结构。

### 生成

推理:
```
生成 z_1 = 从 p(z_1)采样                    # 1词元
生成 z_2 = 从 p(z_2 | z_1)采样              # 4词元并行
生成 z_3 = 从 p(z_3 | z_1, z_2)采样         # 16词元并行
...
解码: f = embed-and-upsample尺度1..K和
图像 = VAE_decoder(f)
```

K = 10尺度,生成是10 transformer前向pass。每pass产全尺度并行——无尺度内每词元自回归。256x256图像这约10 pass vs DiT 28-50。

### 为何下一尺度预测优于下一词元预测

三结构胜:
1. **粗到细对齐自然图像统计。**人类视觉感知和图像数据集均示尺度依赖规律:低频结构稳定可预测;高频细节条件于低频内容。下一尺度预测利用此。
2. **尺度内并行生成。**不像GPT式词元AR,VAR一步产尺度所有词元。有效生成长度log尺度而非线性。
3. **无生成顺序偏。**尺度k词元见尺度k-1全;无迫早词元晚上下文可用前承诺"左"或"上"偏。

### 缩放定律

Tian等示VAR ImageNet FID遵循幂律缩放曲线——恰如GPT困惑度。加倍参数或计算可靠减半错误。这是首图像生成模型示此缩放行为清晰如语言模型。结果是VAR尺度预测从计算可预测,非每架构经验猜测。

### 与扩散关系

VAR和扩散分享同数据压缩故事:两者把生成问题拆成易子问题序列。

- 扩散:逐步加噪声,学撤一步。
- VAR:逐步加分辨率,学预测下尺度。

它们是问题的不同分解维度。两者都能产生可处理的条件分布。实验上VAR推理更快（更少前向pass，尺度内完全并行），在类条件ImageNet任务上匹配或超过DiT。文本条件VAR(VARclip, HART)是活跃研究方向。

## 动手实践

`code/main.py`中你:
1. 合成"图像"数据(2D Gaussian环)上建微小**多尺度VQ分词器**。
2. 训**VAR式transformer**下一尺度预测词元。
3. 调transformer4次(4尺度)并解码采样。
4. 验证尺度顺序训练使生成尺度内并行。

此玩具实现。点是看尺度结构注意力掩和尺度内并行生成实际工作。

## 产出成果

此课产`outputs/skill-var-tokenizer-designer.md`——技能设计多尺度分词器:尺度数、尺度比例、码书大小、残差共享、解码器架构。

## 练习题

1. **尺度数消融。**配4、6、8、10尺度训VAR。测重构质量vs自回归pass数。更多尺度=更细残差=更好质量但更多pass。

2. **码书大小。**配码书大小512、4096、16384训分词器。更大码书给更好重构但预测更难。找膝。

3. **尺度内并行检查。**训VAR,显式测注意力模式。尺度k内,模型attend跨尺度位置但不尺度内?验证掩实现。

4. **VAR vs DiT缩放。**同ImageNet类条件任务,匹配参数预算(如33M、130M、458M)训VAR和DiT。绘FID vs计算。VAR应每大小胜DiT——小尺度重现论文结果。

5. **文本条件。**扩展VAR取文本嵌入(CLIP pooled)作额外条件输入via adaLN。此HART配方。文本对齐采样FID改进多少?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| VAR | "Visual AutoRegressive" | VQ词元金字塔上下一尺度预测图像生成 |
| 下一尺度预测 | "先粗后细预测" | 模型在增分辨率尺度预测词元,条件所有前尺度 |
| 多尺度VQ分词器 | "残差VQ" | VQ-VAE产K增分辨率词元网格,解码器全尺度求和 |
| 尺度k | "金字塔层k" | K分辨率层之一,从k=1时1x1到k=K时(H/p)x(W/p) |
| 尺度内并行 | "每尺度一前向" | 尺度k所有词元一transformer pass预测,不自回归 |
| 跨尺度因果 | "尺度顺序注意力" | 尺度k词元可attend尺度1..k全但非尺度k+1..K |
| 残差VQ | "加性tokenization" | 每尺度词元编码低尺度留残差;解码器全尺度嵌入求和 |
| VAR缩放定律 | "图像GPT缩放" | FID遵循计算幂律,如语言模型困惑度 |
| HART | "混合VAR+文本" | 文本条件VAR变体结合MaskGIT式迭代解码和VAR尺度结构 |
| 尺度位置嵌入 | "(尺度,行,列)三元" | 位置编码携尺度指数和尺度内空间坐标 |

## 延伸阅读

- [Tian等, 2024—"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)——VAR论文,规范参考
- [Peebles and Xie, 2022—"Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748)——DiT,扩散对比基线
- [Esser等, 2021—"Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841)——VQGAN,VAR多尺度分词器扩展的分词器家族
- [van den Oord等, 2017—"Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937)——VQ-VAE,离散图像tokenization基础
- [Tang等, 2024—"HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812)——文本条件VAR