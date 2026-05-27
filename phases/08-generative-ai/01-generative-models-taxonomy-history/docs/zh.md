# 生成模型 — 分类体系与历史

> 每个图像模型、文本模型、视频模型和3D模型都归入五个类别之一。选错类别你会和数学斗争数周。选对类别,领域过去十二年的进展会在你脑中干净堆叠。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段2(机器学习基础)、阶段3(深度学习核心)、阶段7课程14(Transformers)
**时间:** ~45分钟

## 问题背景

生成模型做一件事:给定从某未知分布`p_data(x)`抽取的训练样本,输出看起来来自同一分布的新样本。人脸、句子、MIDI文件、蛋白质结构——如果眯眼看,都是同一问题。

困难在于`p_data`生活在百万维空间(512×512 RGB图像约786k维),样本坐在该空间内薄流形上,你可能只有约10M示例。暴力密度 hopeless。每个生成模型是把一个困难问题换成稍不那么困难问题的妥协。

过去十二年有五个家族存活。知道每个家族做何种妥协告诉你为何它在某些任务胜出而在其他崩溃。

## 概念讲解

![生成模型五家族——按建模对象分类](../assets/taxonomy.svg)

**1. 显式密度,可计算。**把`log p(x)`写成实际可求值的和。自回归模型(PixelCNN、WaveNet、GPT)分解`p(x) = ∏ p(x_i | x_<i)`。归一化流(RealNVP、Glow)把`p(x)`构建为简单基的可逆变换。优点:精确似然,干净训练损失。缺点:自回归推理顺序(长序列慢),流需可逆架构(架构受限)。

**2. 显式密度,近似。**从下界`log p(x)`(ELBO)并优化界。VAE(Kingma 2013)用配变分后验的编码器-解码器。扩散模型(DDPM, Ho 2020)训练隐式优化加权ELBO的去噪器。扩散是2026图像、视频和3D主导骨干。

**3. 隐式密度。**完全跳密度;学习生成器`G(z)`产样本和判别器`D(x)`分辨真假。GAN(Goodfellow 2014)。推理快(一次前向)但训练臭名昭著不稳定。StyleGAN 1/2/3即使在2026仍是固定域照片真实(人脸、卧室)最佳。

**4. 基分数/连续时间。**直接学习log密度梯度`∇_x log p(x)`(分数)。Song & Ermon (2019)展示分数匹配把扩散推广到SDE。流匹配(Lipman 2023)是2024-2026热点:无模拟训练、更直路径、比DDPM快4-10倍采样。Stable Diffusion 3、Flux、AudioCraft 2全用流匹配。

**5. 基离散词元的自回归。**用VQ-VAE或残差量化器压缩高维数据为离散词元短序列,后用Transformer建模词元序列。Parti、MuseNet、AudioLM、VALL-E、Sora的patch tokenizer全用这。这是类别1加学习tokenizer。

## 简史

| 年份 | 模型 | 为何重要 |
|------|------|----------|
| 2013 | VAE (Kingma) | 首个配有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度,无似然——惊人锐利样本。 |
| 2015 | DRAW, PixelCNN | 顺序图像生成。 |
| 2017 | Glow, RealNVP | 可逆流;深度配精确似然。 |
| 2017 | Progressive GAN | 首个百万像素人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 该域照片真实人脸仍难超越。 |
| 2020 | DDPM (Ho) | 扩散变实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文本到图像入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 潜扩散+文本条件=商品化。 |
| 2022 | ControlNet, LoRA | 预训练扩散精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模+更好训练动态。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散;流匹配胜。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 扩散骨干一步采样。 |

## 五问分诊

当新生成模型论文发布,在方法部分前回答这五问。

1. **建模什么?**像素、潜空间、离散词元、3D高斯、网格、波形?
2. **密度显式还是隐式?**他们写`log p(x)`否?
3. **采样:一次还是迭代?**迭代意味推理更慢;一次通常意味对抗或蒸馏。
4. **条件:无条件、类、文本、图像、姿态?**这决定损失和架构脚手架。
5. **评估:FID、CLIP分数、IS、人类偏好、任务准确率?**各有已知失败模式(见课程14)。

你会为本阶段每课重答这五问。到末尾,它们会是反射。

## 动手实践

本课程代码是轻量可视化:用三种玩具方法(核密度、离散直方图、最近样本"GAN式"生成器)从样本拟合1-D高斯混合,这样你可在打印一屏的问题上看显式vs隐式密度区别。

运行`code/main.py`。它从双模高斯混合抽2000样本,后打印:

```
显式密度(直方图): p(x in [-0.5, 0.5]) ≈ 0.38
近似密度(KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
隐式(最近样本生成): 20新样本打印,无p(x)
```

注意:前两者让你问"这点多可能?"第三者不能。这是每后续课程关键的*显式vs隐式*区别。

## 实际应用

2026年,哪个家族,哪个任务?

| 任务 | 最佳家族 | 原因 |
|------|----------|------|
| 照片真实人脸、窄域 | StyleGAN 2/3 | 仍最锐利、推理最快。 |
| 通用文本到图像 | 潜扩散+流匹配 | SD3、Flux.1、DALL-E 3。 |
| 快文本到图像 | Rectified flow+蒸馏 | SDXL-Turbo、SD3-Turbo、LCM。 |
| 文本到视频 | 扩散Transformer+流匹配 | Sora、Veo 2、Kling。 |
| 语音+音乐 | 基词元AR(AudioLM、VALL-E、MusicGen)或流匹配(AudioCraft 2) | 离散词元扩展便宜。 |
| 3D场景 | 高斯Splatting拟合、扩散先验 | 3D-GS重建、扩散新视角。 |
| 密度估计(无采样) | 流 | 唯有精确`log p(x)`家族。 |
| 模拟/物理 | 流匹配、分数SDE | 直线路径、平滑向量场。 |

## 产出成果

存为`outputs/skill-model-chooser.md`。

技能取任务描述输出:(1)用哪个家族,(2)三个开源和三个托管选项排序列,(3)应关注的可能失败模式,(4)计算/时间预算。

## 练习题

1. **简单。**对这五个产品,识别家族和骨干:ChatGPT图像、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **中等。**你明天要读的论文声称比扩散快100倍采样。写三问检查速度在条件和高清下是否存活。
3. **困难。**取你关心的一个域(如蛋白质结构、CAD、分子、轨迹)。对该域当前SOTA模型答五问分诊,并草绘更好模型会改什么。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 生成模型 | "它造新东西" | 学`p_data(x)`采样器,可选暴露`log p(x)`。 |
| 显式密度 | "你能算它" | 模型提供闭式或可计算`log p(x)`。 |
| 隐式密度 | "GAN式" | 仅采样器——无法算给定点的`p(x)`。 |
| ELBO | "证据下界" | `log p(x)`可计算下界;VAE和扩散优化它。 |
| Score | "log密度梯度" | `∇_x log p(x)`;扩散和SDE模型学此场。 |
| 流形假设 | "数据住在表面" | 高维数据集中在低维流形;为何降维有效。 |
| 自回归 | "预测下一块" | 分解联合为条件乘积。 |
| 潜空间 | "压缩码" | 解码器可从中重建输入的低维表示。 |

## 生产注:五家族,五推理形状

每家族映射不同推理服务器成本曲线。生产推理文献把大语言模型推理框架为prefill + decode;同样分解适用于此:

- **自回归(类别1和5)。**顺序decode主导延迟;KV-cache、连续批和投机解码全直接适用。
- **VAE/扩散/流匹配(类别2和4)。**大语言模型意义下无decode。成本=`num_steps × step_cost`,`step_cost`是全潜分辨率transformer或U-Net前向。生产旋钮是步数(DDIM/DPM-Solver/蒸馏)、批大小和精度(bf16/fp8/int4)。
- **GAN(类别3)。**一次前向。无调度、无KV-cache。TTFT≈总延迟。这是为何StyleGAN窄域UX仍胜。

当你见论文摘要"比扩散快",翻译为"更少步×相同步成本"或"相同步×更便宜步成本"。其余是营销。

## 延伸阅读

- [Goodfellow等(2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)——GAN论文。
- [Kingma & Welling(2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)——VAE论文。
- [Ho, Jain, Abbeel(2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)——DDPM论文。
- [Song等(2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)——扩散作SDE。
- [Lipman等(2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)——流匹配论文。
- [Esser等(2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)——Stable Diffusion 3。