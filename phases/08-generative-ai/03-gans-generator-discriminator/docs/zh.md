# GAN — 生成器vs判别器

> Goodfellow在2014的技巧是完全跳过密度。两个网络。一个造假。一个抓它们。它们战斗直到假货与真实不可区分。不该工作。经常不工作。当它工作时,样本仍是狭窄领域文献中最锐利的。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3课程02(反向传播)、阶段3课程08(优化器)、阶段8课程02(VAE)
**时间:** ~75分钟

## 问题背景

VAE产生模糊样本因为其MSE解码器loss对*均值*图像是Bayes最优——许多合理数字均值是模糊数字。你要奖励*合理性*而非像素接近任何单一目标的loss。合理性无闭式。你必须学它。

Goodfellow想法:训练分类器`D(x)`区分真实图像和假货。训练生成器`G(z)`愚弄`D`。`G`loss信号是`D`当前认为什么东西看起来真实。此信号随`G`改进更新,追逐移动目标。若两网络收敛,`G`学了数据分布而从未写下`log p(x)`。

这是对抗训练。数学是minimax博弈:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026年GAN不再是SOTA生成器(扩散和流匹配吃了那个皇冠)。但StyleGAN 2/3仍是历来发布的最锐利人脸模型,GAN判别器在扩散训练中用作*感知损失*,对抗训练驱动快速1步蒸馏(SDXL-Turbo, SD3-Turbo, LCM)让你发布实时扩散。

## 概念讲解

![GAN训练:生成器和判别器minimax](../assets/gan.svg)

**生成器`G(z)`。**映射噪声向量`z ~ N(0, I)`到样本`x̂`。解码器形网络(稠密或转置卷积)。

**判别器`D(x)`。**映射样本到标量概率(或分数)。真实→1,假货→0。

**Loss。**两个交替更新:

- **训`D`:**`loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。二元交叉熵真实=1,假货=0。
- **训`G`:**`loss_G = -log D(G(z))`。这是Goodfellow用的*非饱和*形式(原始`log(1 - D(G(z)))`当`D`确信时饱和杀死梯度)。

**训练循环。**一步`D`,一步`G`。重复。

**为何工作。**如果`G`完美匹配`p_data`,则`D`不能比机会更好并在到处输出0.5;`G`无更多梯度。均衡。

**为何崩。**模式塌缩(`G`找`D`不能分类的一个模式并永远铸造它)、梯度消失(`D`学太快`log D`饱和)、训练不稳定(学习率、批大小、任何东西)。

## 让GAN工作的变体

| 年 | 创新 | 修复 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、批归一化、LeakyReLU——首个稳定架构。 |
| 2017 | WGAN, WGAN-GP | BCE换Wasserstein距离+梯度惩罚。修复梯度消失。 |
| 2017 | Spectral normalization | Lipschitz-bound判别器。2026判别器仍用。 |
| 2018 | Progressive GAN | 先训低分辨率,加层。首个megapixel结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络+自适应实例归一化。固定领域照片真实state of the art。 |
| 2021 | StyleGAN3 | Alias-free、平移等变——2026仍是人脸黄金标准。 |
| 2022 | StyleGAN-XL | 条件、类感知、更大规模。 |
| 2024 | R3GAN | 配更强正则重品牌;1024²无tricks工作。 |

## 动手实践

`code/main.py`在1-D数据上训微小GAN:两高斯混合。生成器和判别器是单隐藏层MLP。手实现前向、反向、和minimax循环。目标是看到两关键失败模式(模式塌缩+梯度消失)发生。

### Step 1: 非饱和loss

原始Goodfellow loss `log(1 - D(G(z)))`当D高置信把G假货分类为假货时走向0。那时G梯度基本零——G不能改进。非饱和形式`-log D(G(z))`有相反渐近:当D确信时爆炸,给G强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2: 每生成器步一判别器步

```python
for step in range(steps):
    # 训D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新假货
    update_G(fake_batch)
```

G用新假货,否则梯度陈旧。

### Step 3: 监视模式塌缩

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: 一个模式被饿死")
```

经典症状:两个真实模式之一停止被生成。判别器停止纠正它因为从未见它作为假货。

## 陷阱

- **判别器太强。**降D学习率2-5x,或加实例/层噪声。如果D达>95%准确率,G死了。
- **生成器记忆一个模式。**对D输入加噪声、用minibatch-discriminator层、或切WGAN-GP。
- **批归一化泄漏统计。**真实批+假货批流过同BN层混合统计。用实例归一化或谱归一化替代。
- **Inception-score博弈。**FID和IS低样本计数噪。eval用≥10k样本。
- **单次采样是条件任务谎言。**仍需CFG scale、截断tricks、和重采样得可用输出。

## 实际应用

2026 GAN栈:

| 情况 | 选择 |
|------|------|
| 照片真实人脸、固定姿势 | StyleGAN3(最锐利、最小) |
| 动漫/风格化人脸 | StyleGAN-XL或Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN(阶段8课程04)或ControlNet(阶段8课程08) |
| 快1步文本到图像 | 扩散对抗蒸馏(SDXL-Turbo, SD3-Turbo) |
| 扩散训练器内感知损失 | 图像crop上小GAN判别器 |
| 任何多模态、开放式 | 别——用扩散或流匹配 |

GAN锐利但狭窄。一旦领域开放——照片、任意文本提示、视频——切扩散。对抗技巧作为组件存活(感知损失、蒸馏),而非独立生成器。

## 产出成果

存`outputs/skill-gan-debugger.md`。技能取失败GAN运行(loss曲线、样本grid、数据集大小)并输出可能原因排列表、一行修复、和重跑协议。

## 练习题

1. **简单。**配默认设置跑`code/main.py`。后设`D_LR = 5 * G_LR`重跑。G loss崩到常数多快?
2. **中等。**Goodfellow BCE loss换WGAN loss:`loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`,并裁剪D权重到`[-0.01, 0.01]`。训练更稳定否?比较wall-clock收敛。
3. **困难。**扩1-D例到2-D数据(环上8高斯混合)。追踪生成器在步1k、5k、10k捕获多少8模式。实现minibatch判别重测。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 生成器 | "G" | 噪声到样本网络,`G: z → x̂`。 |
| 判别器 | "D" | 分类器`D: x → [0, 1]`,真实vs假货。 |
| Minimax | "博弈" | 联合目标`min_G max_D`。 |
| 非饱和loss | "修复" | G用`-log D(G(z))`而非`log(1 - D(G(z)))`。 |
| 模式塌缩 | "G记忆一个东西" | 生成器尽管多样数据产少不同输出。 |
| WGAN | "Wasserstein" | BCE换Earth-Mover距离+梯度惩罚;更平滑梯度。 |
| Spectral norm | "Lipschitz技巧" | 约束D权重范数bound斜率;稳定训练。 |
| StyleGAN | "那个工作的" | 映射网络+AdaIN;人脸最佳类,2026仍。 |

## 生产注:单次推理是GAN持久优势

GAN不再在开放域生成样本质量胜,但它们仍推理成本胜。在生产推理文献词汇GAN有:

- **无prefill、无decode阶段。**单`G(z)`前向pass。TTFT≈总延迟。
- **无KV-cache压力。**唯一状态是权重。批大小bounded by激活内存,非cache。
- **平凡连续批。**因每请求取相同固定FLOPs,服务器目标occupancy静态批常最优。无in-flight调度器需。

这是为何GAN蒸馏(SDXL-Turbo, SD3-Turbo, ADD, LCM)是2026快速文本到图像主导技术:它塌缩20-50步扩散管道到1-4 GAN式前向pass同时保持扩散基分布。对抗loss存活作为训练时旋钮把慢生成器变快。

## 延伸阅读

- [Goodfellow等(2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)——原始GAN论文。
- [Radford等(2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434)——首个稳定架构。
- [Arjovsky, Chintala, Bottou(2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875)——WGAN。
- [Miyato等(2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)——SN。
- [Karras等(2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)——StyleGAN2。
- [Karras等(2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423)——StyleGAN3。
- [Sauer等(2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)——SDXL-Turbo。