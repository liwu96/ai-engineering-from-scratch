# 流匹配与校正流

> 扩散模型走曲路从噪声到数据故需20-50采样步。流匹配(Lipman等, 2023)和校正流(Liu等, 2022)训直路。直路意味更少步意味更快推理。Stable Diffusion 3、Flux.1、AudioCraft 2全2024换流匹配。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程06(DDPM)、阶段1(微积分)
**时间:** ~45分钟

## 问题背景

DDPM反向过程是从`N(0, I)`回数据分布的1000步随机游走。DDIM坍到20-50确定性步。想要更少步——理想一。阻是解反向过程ODE刚性;路曲。

如能训模型噪声到数据路是*直线*,从`t=1`到`t=0`单Euler步将工作。流匹配直接建此:定义从`x_1 ∼ N(0, I)`到`x_0 ∼ 数据`直线插值,训向量场`v_θ(x, t)`匹其时间导数,推理积分。

校正流(Liu 2022)更远:迭代校正路配reflow过程产渐近线性ODE。两reflow迭代后,2步采样器匹50步DDPM质量。

## 概念讲解

![流匹配:噪声和数据间直线插值](../assets/flow-matching.svg)

### 直线流

定义:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中`x_0 ~ 数据`和`x_1 ~ N(0, I)`。沿此直线时间导数常:

```
dx_t / dt = x_1 - x_0
```

定义神经向量场`v_θ(x_t, t)`训其匹此导数:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

此**条件流匹配**损失(Lipman 2023)。训练免模拟:永不展开ODE。仅采`(x_0, x_1, t)`回归。

### 采样

推理时,反向时间积分学向量场:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

始于`x_1 ~ N(0, I)`,Euler步下到`t=0`。

### 校正流(Liu 2022)

直线流工作但学路*非真直*——曲因多`x_0`可映射同`x_1`。校正流reflow步:

1. 随机配训流模型v_1。
2. 从`x_1`积分v_1到其着陆`x_0`采样N配`(x_1, x_0)`。
3. 那些配样本训v_2。因配现"ODE匹",其间直线插值真更平。
4. 重复。

实践2 reflow迭代到近线,启2-4步推理。SDXL-Turbo、SD3-Turbo、LCM全从流匹配蒸馏模型。

### 何此2024赢图像

三原因:

1. **免模拟训练**——训练无ODE展开,实现简单。
2. **更好损失几何**——直路有一致信噪比,而DDPM ε损失在调度边SNR差。
3. **更快推理**——SDXL-Turbo质量4-8步;一致性蒸馏1步。

## 流匹配vs DDPM——精确联系

Gaussian条件路流匹配是扩散*配特定噪声调度*。选`x_t = α(t) x_0 + σ(t) x_1`调度流匹配恢复Stratonovich重式扩散配`v = α'·x_0 - σ'·x_1`。Gaussian路两者代数等价。

流匹配添:目标*清晰度*(简单速度)、更干净损失、和非Gaussian插值实验许可。

## 动手实践

`code/main.py`实现双模Gaussian混合上1-D流匹配。向量场`v_θ(x, t)`是训配直线目标微小MLP。推理,积分1、2、4、20 Euler步比样本质量。

### Step 1: 训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播+更新
```

### Step 2: 多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: 比步数

期望4步采样器已匹20步质量——延迟大事。

## 陷阱

- **时间参数化。**流匹配用`t ∈ [0, 1]`配`t=0`数据、`t=1`噪声。DDPM用`t ∈ [0, T]`配`t=0`数据、`t=T`噪声。同方向,不同尺度。论文常错。
- **调度选择。**校正流直线是"那个"流匹配调度,但可用cosine或logit-normal t采样(SD3做此)更好尺度覆盖。
- **Reflow成本。**生成reflow配数据集是每样本全推理pass。仅真需1-2步推理时reflow。
- **无分类器引导仍适用。**线性组合仅换ε为v:`v_cfg = (1+w) v_cond - w v_uncond`。

## 实际应用

| 用例 | 2026栈 |
|------|--------|
| 文本到图像,最佳质量 | 流匹配:SD3、Flux.1-dev |
| 文本到图像,1-4步 | 蒸馏流匹配:Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 流匹配基一致性蒸馏(LCM、PCM) |
| 音频生成 | 流匹配:Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配混扩散(Sora、Veo、Stable Video) |
| 科学/物理(粒子轨迹、分子) | 流匹配+等变向量场 |

2025-2026论文说"比扩散快"几乎总是流匹配+蒸馏。

## 产出成果

存`outputs/skill-fm-tuner.md`。技能取扩散式模型规格转流匹配训练配置:调度选择、时间采样分布(uniform/logit-normal)、优化器、reflow计划、目标步数、评估协议。

## 练习题

1. **简单。**跑`code/main.py`比1步vs 20步MSE vs真数据分布。
2. **中等。**换uniform `t`采样到logit-normal(集中采样mid-t)。模型质量改进否?
3. **困难。**实现一reflow迭代:积分第一模型生成配(x_0, x_1),对配训第二模型,比1步样本质量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 流匹配 | "直线扩散" | 训`v_θ(x, t)`匹插值上`x_1 - x_0`。 |
| 校正流 | "Reflow" | 迭代过程校正学流。 |
| 向量场 | "v_θ" | 模型输出——移`x_t`方向。 |
| 直线插值 | "路径" | `x_t = (1-t)·x_0 + t·x_1`;平凡目标导数。 |
| Euler采样器 | "1阶ODE解器" | 最简单积分器;路直时工作好。 |
| Logit-normal t | "SD3采样" | 集中`t`采样到中值梯度最强处。 |
| 一致性蒸馏 | "1步采样器" | 训学生直接从任意`x_t`映射到`x_0`。 |
| 速度CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`;同技巧,新变量。 |

## 生产注:Flux.1-schnell是流匹配最快

流匹配生产赢是Flux.1-schnell——流匹配DiT蒸馏到1-4推理步同时保Flux-dev级质量。Niels"8GB机跑Flux"笔记本是参考部署配方:T5+CLIP编码、量化MMDiT去噪(schnell 4步vs dev 50步)、VAE解码。成本核算:

| 变体 | 步 | L4上1024²延迟 | 总FLOPs(相对) |
|------|-----|---------------|---------------|
| Flux.1-dev(原始) | 50 | ~15秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2秒 | 0.08×(12×快) |
| SDXL-base | 30 | ~4秒 | 0.25× |
| SDXL-Lightning 2步 | 2 | ~0.3秒 | 0.03× |

生产规则:**流匹配基+蒸馏=2026快文本到图像默认。**每主厂发此组合:SD3-Turbo(SD3+流+蒸馏)、Flux-schnell(Flux-dev+校正流校正)、CogView-4-Flash。纯扩散基仅留旧检查点。

## 延伸阅读

- [Liu, Gong, Liu(2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)——校正流。
- [Lipman等(2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)——流匹配。
- [Esser等(2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)——SD3,大规模校正流。
- [Albergo, Vanden-Eijnden(2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797)——盖FM+扩散通用框架。
- [Song等(2023). Consistency Models](https://arxiv.org/abs/2303.01469)——扩散/流1步蒸馏。
- [Sauer等(2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042)——turbo变体。
- [Black Forest Labs(2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/)——生产流匹配。