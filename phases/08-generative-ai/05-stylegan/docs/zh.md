# StyleGAN

> 多数生成器同时把`z`搅进每层。StyleGAN拆开它:先map `z`到中间`w`,后通过AdaIN在每分辨率层*注入*`w`。那单一改变解缠潜空间并使照片真实人脸七年来成为已解问题。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程03(GAN)、阶段4课程08(归一化)、阶段3课程07(CNN)
**时间:** ~45分钟

## 问题背景

DCGAN通过转置卷积栈把`z`映射到图像。问题:`z`控制一切——姿态、光照、身份、背景——缠在一起。沿`z`一轴移,四全变。你不能问模型"同人,不同姿态"因表示不那样分解。

Karras等(2019, NVIDIA)提出:停止把`z`直接喂进conv层。喂常数`4×4×512`张量作网络输入。学8层MLP映射`z ∈ Z → w ∈ W`。通过*自适应实例归一化*(AdaIN)在每分辨率注入`w`:归一化每conv特征图,后用`w` affine投影缩放和偏移。加每层噪声做随机细节(皮肤毛孔、发丝)。

结果:`W`有大致正交轴"高层风格"(姿态、身份)vs"精细风格"(光照、颜色)。你可两图像间swap风格用图像A的`w`在低分辨率层和图像B的`w`在高。这解锁编辑、跨域风格化和整个"StyleGAN-inversion"研究线。

## 概念讲解

![StyleGAN: mapping网络+AdaIN+每层噪声](../assets/stylegan.svg)

**Mapping网络。**`f: Z → W`,8层MLP。`Z = N(0, I)^512`。`W`不强高斯——它学数据适应形状。

**Synthesis网络。**从学习常数`4×4×512`起。每分辨率块:`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率倍增:4, 8, 16, 32, 64, 128, 256, 512, 1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中`y_scale`和`y_bias`来自`w` affine投影。每特征图归一化,后restyle。"风格"此处是特征图一二阶统计。

**每层噪声。**单通道高斯噪声加到每特征图,由可学习每通道因子缩放。控制随机细节不影响全局结构。

**截断技巧。**推理时,采样`z`,算`w = mapping(z)`,后`w' = ŵ + ψ·(w - ŵ)`其中`ŵ`是多样本平均`w`。`ψ < 1`权衡多样性和质量。几乎每个StyleGAN demo用`ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|------|------|------|
| StyleGAN | 2019 | Mapping网络+AdaIN+噪声+progressive growing。 |
| StyleGAN2 | 2020 | Weight demodulation换AdaIN(修复droplet artifact);skip/residual架构;path-length正则。 |
| StyleGAN3 | 2021 | Alias-free卷积+等变kernel;消纹理粘像素网格。 |
| StyleGAN-XL | 2022 | 类条件,1024²,ImageNet。 |
| R3GAN | 2024 | 更强正则重品牌;FFHQ-1024用20×更少参数闭扩散差距。 |

2026年StyleGAN3仍是默认用于窄域高FPS照片真实、few-shot域适应(在新数据集100图像训练,冻结mapping)、基于inversion编辑(找重建真实照片的`w`,后编辑该`w`)。开域文本到图像,它非工具——扩散是。

## 动手实践

`code/main.py`在1-D实现玩具"style-GAN lite":mapping MLP、synthesis函数取学习常数向量并用`w`派生scale/bias调制、每层噪声。它展示通过affine-modulation注入`w`匹配或胜把`z`拼接到生成器输入。

### Step 1: mapping网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: 自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每特征图scale和bias来自`w`通过线性投影。

### Step 3: 每层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道sigma可学习。

## 陷阱

- **水滴伪影。**StyleGAN 1在特征图产生水滴状伪影，因为AdaIN使均值归零。StyleGAN 2 weight demodulation通过缩放卷积权重而非激活修复。
- **纹理粘。**StyleGAN 1和2纹理跟像素坐标,非对象坐标(插值时可见)。StyleGAN 3 alias-free卷积配windowed sinc滤波修复。
- **模式覆盖。**截断`ψ < 0.7`看干净但从窄锥采样;如需多样性用`ψ = 1.0`。
- **Inversion有损。**把真实照片invert进`W`通常通过优化或编码器(e4e, ReStyle, HyperStyle)做。结果多次迭代漂移。

## 实际应用

| 用例 | 方法 |
|------|------|
| 照片真实人脸(动漫、产品、窄域) | StyleGAN3 FFHQ / 自定义微调 |
| 从照片人脸编辑 | e4e inversion + StyleSpace / InterFaceGAN方向 |
| Face swap / reenactment | StyleGAN + 编码器 + 混合 |
| Avatar管道 | StyleGAN3配ADA低数据微调 |
| 从少图像域适应 | 冻结mapping网络,微调synthesis |
| 多模态或文本条件生成 | 别——用扩散 |

产品级demo答案是"人脸照片"，StyleGAN在推理成本（单次前向传播，4090上<10ms）和相同质量下的锐度上胜过扩散。

## 产出成果

存`outputs/skill-stylegan-inversion.md`。技能取真实照片输出:inversion方法(e4e / ReStyle / HyperStyle)、预期潜损失、编辑预算(在`W`可移多远才artifact)、已知好编辑方向列表(年龄、表情、姿态)。

## 练习题

1. **简单。**运行`code/main.py`配`adain_on=True`和`adain_on=False`。比固定潜空间vs扰动潜空间输出分布。
2. **中等。**实现mixing正则:训练批,算`w_a`、`w_b`,对synthesis前半用`w_a`后半用`w_b`。解码器学解耦风格否?
3. **困难。**取预训练StyleGAN3 FFHQ模型(ffhq-1024.pkl)。通过在标号样本上训SVM找控制"微笑"的`w`方向;报告可推多远才身份漂移。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Mapping网络 | "那个MLP" | `f: Z → W`,8层,解耦潜几何和数据统计。 |
| W空间 | "风格空间" | Mapping网络输出;大致解耦。 |
| AdaIN | "自适应实例norm" | 归一化特征图,后用`w`投影scale+shift。 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`,ψ<1权衡多样性和质量。 |
| Path-length正则 | "PL reg" | 惩罚图像每单位`w`变化大变;使`W`更平滑。 |
| Weight demodulation | "StyleGAN2修复" | 归一化conv权重而非激活;杀droplet artifact。 |
| Alias-free | "StyleGAN3技巧" | Windowed sinc滤波;消纹理粘像素网格。 |
| Inversion | "找真实图像的w" | 优化或编码`x → w`使`G(w) ≈ x`。 |

## 生产注:为何StyleGAN 2026仍部署

4090上StyleGAN3在不到10ms生成1024² FFHQ人脸——`num_steps = 1`,无VAE解码,无交叉注意力pass。生产术语这是任何图像生成器地板延迟。同分辨率50步SDXL + VAE解码管道约3秒。这是**300倍**的差距，对于窄域产品（avatar服务、ID文档管道、批量人脸生成），总体运营成本更低。

两操作后果:

- **无调度器，无批处理器。**目标GPU利用率下的静态批最优。连续批（LLM和扩散必需）无任何收益，因为每次请求消耗相同FLOPs。
- **截断`ψ`是安全旋钮。**`ψ < 0.7`从mapping网络范围窄锥采样。这是服务层对样本方差唯一杠杆。高峰负载降`ψ`,高级用户升。

## 延伸阅读

- [Karras等(2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948)——StyleGAN。
- [Karras等(2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)——StyleGAN2。
- [Karras等(2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423)——StyleGAN3。
- [Tov等(2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766)——e4e inversion。
- [Sauer等(2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273)——StyleGAN-XL。
- [Huang等(2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441)——现代最小GAN配方。