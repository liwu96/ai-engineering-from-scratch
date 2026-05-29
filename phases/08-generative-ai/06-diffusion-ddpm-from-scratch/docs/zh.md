# 扩散模型——从零DDPM

> Ho, Jain, Abbeel(2020)给这个领域提供了一个令其欲罢不能的配方。一千小步用噪声破坏数据。训一个神经网络预测噪声。推理时反向过程。今天每个主流图像、视频、3D和音乐模型跑在此循环上,可能配流匹配或一致性技巧顶。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3课程02(反向传播)、阶段8课程02(VAE)
**时间:** ~75分钟

## 问题背景

你想要`p_data(x)`采样器。GAN玩minimax博弈常发散。VAE从Gaussian解码器产模糊样本。你真想要是训练目标是(a)单稳定损失(无鞍点,无minimax), (b) `log p(x)`下界(所以有似然),和(c)匹SOTA质量样本。

Sohl-Dickstein等(2015)有理论答案:定义Markov链`q(x_t | x_{t-1})`逐步加Gaussian噪声,训反向链`p_θ(x_{t-1} | x_t)`去噪。Ho, Jain, Abbeel(2020)示损失可简化到一行——预测噪声——并清数学。2020是好奇。2021产SOTA样本。2022成Stable Diffusion。2026是基底。

## 概念讲解

![DDPM:前向噪声,反向去噪](../assets/ddpm.svg)

**前向过程`q`。**`T`小步加Gaussian噪声。闭式——数学可处理原因——是累计步也Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

`α̅_t = ∏_{s=1..t} (1 - β_s)`对`β_t`调度。T=1000步线性从1e-4到0.02选`β_t`,`x_T`约`N(0, I)`。

**反向过程`p_θ`。**学神经网络`ε_θ(x_t, t)`预测所加噪声。给定`x_t`,去噪:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

`σ_t`或`sqrt(β_t)`或学习方差。表达式丑但仅代数——给定后验`q(x_{t-1} | x_t, x_0)`解`x_{t-1}`并代`x_0`用噪声预测估计。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据采样`x_0`,随机`t`,采样`ε ~ N(0, I)`,闭式一次算噪`x_t`,回归噪声。一损失,无minimax,无KL,无重参技巧。

**采样。**起`x_T ~ N(0, I)`。从`t = T`到`1`迭代反向步。完成。

## 为何有效

三直觉:

1. **去噪易;生成难。**`t=T`,数据纯噪声——网需解平凡问题。`t=0`,网仅清几像素。中间`t`,问题难但网从每噪声级有流同权多梯度。
2. **隐分数匹配。**Vincent(2011)证预测噪声等价估`∇_x log q(x_t | x_0)`,*分数*。反向SDE用此分数走密度梯度上——向高概率区引导随机走。
3. **ELBO简到简单MSE。**全变分下界每时间步有KL项。DDPM参化下KL项简到噪声预测MSE配特定系数;Ho丢系数(称"简单"损失)质量*改进*。

## 动手实践

`code/main.py`实现1-D DDPM。数据是两模式混合。"网"是微小MLP取`(x_t, t)`输出预测噪声。训练是一行损失。采样迭代反向链。

### Step 1: 前向调度(闭式)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: 一次采样`x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: 一训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: 反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

1-D问题配40时间步和24单元MLP,~200轮学两模式混合。

## 时间条件

网需知去噪哪时间步。两标准选项:

- **Sinusoidal嵌入。**像Transformer位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。过MLP,广播进网。
- **FiLM/组归一化条件。**投影嵌入到每通道scale/bias(FiLM)每块。

玩具码用sinusoidal → concat。生产U-Net用FiLM。

## 陷阱

- **调度重要。**线性`β`是DDPM默认但余弦调度(Nichol & Dhariwal, 2021)同算给更好FID。质量停滞时换调度。
- **时间步嵌入脆弱。**原始`t`作浮点玩具1-D工作但图像失败;总用恰当嵌入。
- **V预测vs ε预测。**窄区(很小或很大t),`ε`信号噪声比差。V预测(`v = α·ε - σ·x`)更稳定;SDXL、SD3和Flux用它。
- **无分类器引导。**推理,算条件和无条件`ε`,后`ε_cfg = (1 + w) · ε_cond - w · ε_uncond`配`w ≈ 3-7`。课程08覆盖。
- **1000步多。**生产用DDIM(20-50步)、DPM-Solver(10-20步)或蒸馏(1-4步)。见课程12。

## 实际应用

| 角色 | 2026典型栈 |
|------|------------|
| 图像像素空间扩散(小,玩具) | DDPM + U-Net |
| 图像潜空间扩散 | VAE编码器 + U-Net或DiT(课程07) |
| 视频潜空间扩散 | 时空DiT(Sora, Veo, WAN) |
| 音频潜空间扩散 | Encodec + 扩散transformer |
| 科学(分子、蛋白、物理) | 等变扩散(EDM, RFdiffusion, AlphaFold3) |

扩散是通用生成骨干。流匹配(课程13)是2024-2026竞者通常同质量推理速度胜。

## 产出成果

存`outputs/skill-diffusion-trainer.md`。技能取数据集+计算预算输出:调度(线性/余弦/sigmoid)、预测目标(ε/v/x)、步数、引导scale、采样器家族、和评估协议。

## 练习题

1. **简单。**`code/main.py`中改T从40到10。样本质量(输出可视直方图)如何退化?何T两模式结构坍?
2. **中等。**ε预测换V预测。重推导反向步。比终样本质量。
3. **困难。**加无分类器引导。类标签`c ∈ {0, 1}`条件,训时丢10%,采样时用`ε = (1+w)·ε_cond - w·ε_uncond`。测`w = 0, 1, 3, 7`条件模式命中率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 前向过程 | "加噪声" | 固定Markov链`q(x_t | x_{t-1})`破坏数据。 |
| 反向过程 | "去噪" | 学习链`p_θ(x_{t-1} | x_t)`重构数据。 |
| β调度 | "噪声阶梯" | 每步方差;线性、余弦或sigmoid。 |
| α̅ | "Alpha bar" | 累积积`∏(1 - β)`;从`x_0`闭式`x_t`。 |
| 简单损失 | "噪声MSE" | `||ε - ε_θ(x_t, t)||²`;全变分推导坍到此。 |
| ε预测 | "预测噪声" | 输出是所加噪声;标准DDPM。 |
| V预测 | "预测速度" | 输出是`α·ε - σ·x`;跨t更好条件。 |
| DDPM | "那论文" | Ho等2020;线性β,1000步,U-Net。 |
| DDIM | "确定性采样器" | 非Markov采样器,20-50步,同训练目标。 |
| 无分类器引导 | "CFG" | 混条件和无条件噪声预测放大条件。 |

## 生产注:扩散推理是步数问题

DDPM论文使用T=1000个反向步。没有人会在生产中那样做。每真实推理栈选三策略之一——每清晰映射生产框定"延迟从何来":

1. **更快采样器,同模型。**DDIM(20-50步)、DPM-Solver++(10-20)、UniPC(8-16)。反向循环直换;训`ε_θ`权不动。延迟砍20-50×。
2. **蒸馏。**训学生更少步匹教师:渐进蒸馏(2 → 1)、一致性模型(任意 → 1-4)、LCM、SDXL-Turbo、SD3-Turbo。延迟另砍5-10×,需重训。
3. **缓存和编译。**`torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM扩散后端、`xformers`/SDPA注意力、bf16权重。每步延迟砍~2×。配(1)和(2)堆。

对生产扩散服务器预算对话同生产文献LLM:延迟`num_steps × step_cost + VAE_decode`,吞吐`batch_size × (num_steps × step_cost)^-1`。TTFT小(一步);TPOT等价是全响应时间因图像生成从用户视角"all-at-once"。

## 延伸阅读

- [Sohl-Dickstein等(2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)——扩散论文,超前时代。
- [Ho, Jain, Abbeel(2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)——DDPM。
- [Song, Meng, Ermon(2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)——DDIM,更少步。
- [Nichol & Dhariwal(2021). Improved DDPM](https://arxiv.org/abs/2102.09672)——余弦调度,学习方差。
- [Dhariwal & Nichol(2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)——分类器引导。
- [Ho & Salimans(2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)——CFG。
- [Karras等(2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)——统一记号,最清配方。