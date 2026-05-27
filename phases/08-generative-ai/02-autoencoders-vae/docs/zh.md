# 自编码器与变分自编码器 (VAE)

> 普通自编码器压缩后重建。它记忆。它不生成。加一个技巧——强制编码看起来高斯——你得到采样器。那个单一技巧,`z = μ + σ·ε`重参数化,是为何2026你用的每个潜扩散和流匹配图像模型输入都有VAE。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3课程02(反向传播)、阶段3课程07(CNN)、阶段8课程01(分类体系)
**时间:** ~75分钟

## 问题背景

把784像素MNIST数字压缩到16数编码,后重建。普通自编码器重建MSE优异但编码空间是团块混乱。在编码空间随机取点,解码,你得噪声。它无采样器。它是穿着外套的压缩模型。

你实际要的是:(a)编码空间是干净、平滑可采样分布——比如各向同性高斯`N(0, I)`,(b)解码任何样本产合理数字,和(c)编码器和解码器仍压缩良好。三目标,一架构,一损失。

Kingma 2013 VAE通过训练编码器输出*分布*`q(z|x) = N(μ(x), σ(x)²)`、通过KL惩罚把该分布拉向先验`N(0, I)`、后从`q(z|x)`采样`z`再解码来解此。推理时,丢编码器,采样`z ~ N(0, I)`,解码。KL惩罚是强制编码空间结构化的东西。

2026年VAE很少单独部署——扩散在原始图像质量上已超越——但它们是每个潜扩散模型(SD 1/2/XL/3、Flux、AudioCraft)首选编码器。学VAE你学你用每个图像管道不可见第一层。

## 概念讲解

![自编码器vs VAE:重参数化技巧](../assets/vae.svg)

**自编码器。**`z = encoder(x)`,`x̂ = decoder(z)`,损失=`||x - x̂||²`。编码空间无结构。

**VAE编码器。**输出两向量:`μ(x)`和`log σ²(x)`。这些定义`q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。**从`q(z|x)`采样不可微。把采样重写为`z = μ + σ·ε`其中`ε ~ N(0, I)`。现在`z`是`(μ, σ)`确定函数加无参数噪声——梯度流过`μ`和`σ`。

**损失。**证据下界(ELBO),两项:

```
损失 = 重建 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推`x̂`向`x`。KL推`q(z|x)`向先验。它们权衡。小β(<1)=更锐样本、编码空间更不高斯。大β(>1)=更干净编码空间、更模糊样本。β-VAE (Higgins 2017)使此旋钮著名并开启解耦研究。

**采样。**推理时:采`z ~ N(0, I)`,前向过解码器。一次前向——无扩散式迭代采样。

## 动手实践

`code/main.py`实现无numpy或torch的微小VAE。输入是从8-D中2分量高斯混合抽取的8维合成数据。编码器和解码器是单隐藏层MLP。我们实现tanh激活、前向、损失和手写反向。非生产——教学。

### Step 1: 编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用`log σ²`而非`σ`使网络输出无约束(σ的softplus是陷阱——σ≈0时梯度死)。

### Step 2: 重参数化和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Step 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确闭式KL因两分布皆高斯。不要数值积分。2026年人们仍发布配蒙特卡洛KL估计的代码——无故慢3倍。

### Step 4: 生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行。

## 陷阱

- **后验塌缩。**KL项推`q(z|x) → N(0, I)`如此激进`z`不载`x`信息。修复:β退火(起β=0,渐到1)、free bits或跳过不活跃维KL。
- **模糊样本。**高斯解码器似然暗示MSE重建,L2贝叶斯最优(均值)——一组合理数字均值是模糊数字。修复:离散解码器(VQ-VAE、NVAE),或仅用VAE作编码器并在潜空间上叠扩散(Stable Diffusion这么做)。
- **β太大太早。**见后验塌缩。起β≈0.01渐升。
- **潜维太小。**MNIST用16-D,ImageNet 256²用256-D,ImageNet 1024²用2048-D。Stable Diffusion VAE压缩512×512×3 → 64×64×4(空间面积32倍下采样,通道32倍)。

## 实际应用

2026 VAE栈:

| 情况 | 选择 |
|------|------|
| 扩散图像潜编码器 | Stable Diffusion VAE(`sd-vae-ft-ema`)或Flux VAE |
| 音频潜编码器 | Encodec(Meta)、SoundStream或DAC(Descript) |
| 视频潜空间 | Sora时空patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜空间(Transformer建模) | VQ-VAE、RVQ(ResidualVQ) |
| 生成用连续潜空间 | 普通VAE,后在该潜空间条件流/扩散模型 |

潜扩散模型是编码器和解码器间住扩散模型的VAE。VAE做粗压缩,扩散做重活。视频(VAE + 视频扩散DiT)和音频(Encodec + MusicGen transformer)同模式。

## 产出成果

存`outputs/skill-vae-trainer.md`。

技能取:数据集profile + 潜维目标 + 下游用途(重建、采样或潜扩散输入)输出:架构选择(普通/β/VQ/RVQ)、β调度、潜维、解码器似然(高斯vs类别)和评估计划(重建MSE、每维KL、`q(z|x)`和`N(0, I)`间Fréchet距离)。

## 练习题

1. **简单。**在`code/main.py`改`β`为`0.01`、`0.1`、`1.0`、`5.0`。记录最终重建MSE和KL。哪个β对合成数据帕累托最优?
2. **中等。**用伯努利似然(交叉熵损失)换高斯解码器似然。在同合成数据二值化版比样本质量。
3. **困难。**扩`code/main.py`为迷你VQ-VAE:用K=32条目码本最近邻查找换连续`z`。比重建MSE并报告多少码本条目被用(码本塌缩真实)。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`,学MSE。不生成。 |
| VAE | 配采样器的AE | 编码器输出分布,KL惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`;`q = p(z|x)`时紧。 |
| 重参数化 | `z = μ + σ·ε` | 把随机节点重写为确定+纯噪声。使采样可反向传播。 |
| 先验 | `p(z)` | 潜空间目标分布,通常`N(0, I)`。 |
| 后验塌缩 | "KL项赢" | 编码器忽略`x`,输出先验;解码器必须幻觉。 |
| β-VAE | 可调KL权重 | `损失 = recon + β·KL`。高β=更解耦但更模糊。 |
| VQ-VAE | 离散潜空间 | 用最近码本向量换连续`z`;使Transformer建模可能。 |

## 生产注:VAE是扩散服务器最热路径

在Stable Diffusion/Flux/SD3管道,VAE每请求调两次——一次编码(如果做img2img/inpainting)一次解码。1024²时解码pass常是整个管道单最大激活内存峰因为它把`128×128×16`潜空间上采样回`1024×1024×3`。两实际后果:

- **切片或tile解码。**`diffusers`暴露`pipe.vae.enable_slicing()`和`pipe.vae.enable_tiling()`。Tiling用小接缝artifact换`O(tile²)`内存而非`O(H·W)`。1024²+消费GPU必需。
- **bf16解码器,fp32数值做最终resize。**SD 1.x VAE以fp32发布且在1024²+转fp16时*静默产生NaN*。SDXL发布`madebyollin/sdxl-vae-fp16-fix`——总优先fp16-fix变体或用bf16。

## 延伸阅读

- [Kingma & Welling(2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——VAE论文。
- [Higgins等(2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)——解耦β-VAE。
- [van den Oord等(2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)——VQ-VAE。
- [Vahdat & Kautz(2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898)——最佳图像VAE。
- [Rombach等(2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)——Stable Diffusion;VAE作编码器。
- [Défossez等(2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)——Encodec,音频VAE标准。