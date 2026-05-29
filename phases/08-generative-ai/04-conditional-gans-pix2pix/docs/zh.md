# 条件GAN和Pix2Pix

> 2014-2017首个大解锁是控制GAN产生什么。附标签、或图像、或句子。Pix2Pix做图像版仍在狭窄图像到图像任务上胜过每个通用文本到图像模型。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程03(GANs)、阶段4课程06(U-Net)、阶段3课程07(CNNs)
**时间:** ~75分钟

## 问题背景

无条件GAN采样任意人脸。demo有用,生产无用。你要:*草图映射照片*、*地图映射航拍照片*、*白天场景映射夜晚*、*灰度图彩色化*。所有这些,你给定输入图像`x`必须输出`y`配某种语义对应。每`x`有多个合理`y`。均方误差把它们压成糊。对抗loss不,因为"看起来真实"是锐利的。

条件GAN(Mirza & Osindero, 2014)向`G`和`D`输入加条件`c`。Pix2Pix(Isola等,2017)特化此:条件是完整输入图像、生成器是U-Net、判别器是*patch-based*分类器(PatchGAN)、loss是对抗+L1。那个配方甚至在2026狭窄图像到图像领域胜过从头文本到图像模型因为它在*配对数据*训练——你有精确所需信号。

## 概念讲解

![Pix2Pix:U-Net生成器,PatchGAN判别器](../assets/pix2pix.svg)

**条件G。**`G(x, z) → y`。Pix2Pix中,`z`是G内dropout(无输入噪声——Isola发现显式噪声被忽略)。

**条件D。**`D(x, y) → [0, 1]`。输入是*对*(条件,输出)。这是关键差异:D必须判`y`是否与`x`一致,而非仅`y`是否看起来真实。

**U-Net生成器。**配skip连接跨瓶颈的编码器-解码器。对输入和输出共享低层结构任务(边缘、轮廓)关键。无skip,高频细节消失。

**PatchGAN判别器。**非输出单real/fake分数,D输出`N×N`grid每cell判~70×70像素感受野。平均。这是马尔可夫随机场假设:真实感局部。更快训练、更少参数、更锐输出。

**Loss。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1项稳定训练推G向已知目标。L1比L2给更锐边缘(中位数而非均值)。`λ = 100`是Pix2Pix默认。

## CycleGAN——无配对时

Pix2Pix需配对`(x, y)`数据。CycleGAN(Zhu等,2017)以额外loss代价弃此要求:*cycle consistency* loss。两生成器`G: X → Y`和`F: Y → X`。训练它们使`F(G(x)) ≈ x`和`G(F(y)) ≈ y`。这让你无配对示例翻译马到斑马、夏到冬。

2026年,非配对图像到图像主要通过扩散(ControlNet, IP-Adapter)而非CycleGAN,但cycle-consistency想法存活于几乎每个非配对域适应论文。

## 动手实践

`code/main.py`在1-D数据实现微小条件GAN。条件`c`是类标签(0或1)。任务:为给定类从条件分布产生样本。

### Step 1: 向G和D的输入添加条件

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot编码是最简方式。更大模型用学习嵌入、FiLM调制、或交叉注意力。

### Step 2: 条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*真实分布,而非边缘。

### Step 3: 验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略。**G学边缘化,D从不惩罚因为条件信号弱。修复:更激进条件D(早层而非仅晚)、用projection判别器(Miyato & Koyama 2018)。
- **L1权重太低。**G漂向任意真实看输出,不忠实。Pix2Pix式任务起λ≈100。
- **L1权重太高。**G产模糊输出因为L1仍是L_p范数。训练稳定后退火下。
- **D中ground truth泄漏。**拼接`(x, y)`作D输入,非仅`y`。无此D不能检查一致。
- **每类模式塌缩。**每类可独立塌缩。跑类条件多样性检查。

## 实际应用

2026图像到图像任务状态:

| 任务 | 最佳方法 |
|------|----------|
| 草图→照片、同域、配对数据 | Pix2Pix / Pix2PixHD(仍快、仍锐) |
| 草图→照片、非配对 | ControlNet配Scribble conditioning模型 |
| 语义seg→照片 | SPADE / GauGAN2或SD + ControlNet-Seg |
| 风格迁移 | 扩散配IP-Adapter或LoRA;GAN方法legacy |
| 深度→照片 | ControlNet-Depth over Stable Diffusion |
| 超分辨率 | Real-ESRGAN(GAN)、ESRGAN-Plus、或SD-Upscale(扩散) |
| 彩色化 | ColTran、扩散彩色化器、或Pix2Pix-color |
| 白天→夜晚、季节、天气 | CycleGAN或ControlNet-based |

Pix2Pix当有数千配对示例、任务窄且可重复、需快推理时仍是正确工具。通用开放域任务,扩散胜。

## 产出成果

存`outputs/skill-img2img-chooser.md`。技能取任务描述、数据可用性(配对vs非配对、N样本)、延迟/质量预算,输出:方法(Pix2Pix、CycleGAN、ControlNet变体、SDXL + IP-Adapter)、训练数据要求、推理成本、和评估协议(LPIPS、FID、任务特定)。

## 练习题

1. **简单。**修改`code/main.py`加第三类。确认G仍映射每类噪声正确模式。
2. **中等。**1-D设置用感知式loss替换L1(如小冻结D作特征提取器)。改变条件分布锐度否?
3. **困难。**1-D设置勾CycleGAN:两分布、两生成器、cycle loss。示它无配对数据学映射它们间。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 条件GAN | "GAN配标签" | G(z, c), D(x, c)。两网络见条件。 |
| Pix2Pix | "图像到图像GAN" | 配对cGAN配U-Net G和PatchGAN D + L1 loss。 |
| U-Net | "配skip编码器解码器" | 对称conv网络;skip保高频。 |
| PatchGAN | "局部真实分类器" | D输出per-patch分数而非全局分数。 |
| CycleGAN | "非配对图像翻译" | 两G + cycle-consistency loss;无配对数据。 |
| SPADE | "GauGAN" | 用语义map归一化中间激活;分割到图像。 |
| FiLM | "特征线性调制" | 从条件的per-feature affine变换;便宜条件。 |

## 生产注:Pix2Pix作延迟bound基线

当有配对数据和窄任务(草图→渲染、语义map→照片、天→夜),Pix2Pix单次推理在延迟上胜扩散一个数量级。生产比较常:

| 路 | 步 | 单L4上512²典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix(U-Net forward) | 1 | ~30 ms |
| SD-Inpaint或SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix在静态批吞吐胜(每请求相同FLOPs)。扩散在质量和泛化胜。现代做法通常为窄任务发布Pix2Pix式蒸馏模型，并以扩散作为尾部输入的后备方案。

## 延伸阅读

- [Mirza & Osindero(2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)——cGAN论文。
- [Isola等(2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)——Pix2Pix。
- [Zhu等(2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)——CycleGAN。
- [Wang等(2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585)——Pix2PixHD。
- [Park等(2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)——SPADE / GauGAN。
- [Miyato & Koyama(2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)——projection D。