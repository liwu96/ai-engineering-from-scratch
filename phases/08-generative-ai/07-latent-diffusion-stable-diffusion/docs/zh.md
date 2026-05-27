# 潜扩散与Stable Diffusion

> 512×512图像像素空间扩散是计算战争罪。Rombach等(2022)注意你不需要全部786k维生成图像——你需要足够捕获语义结构,和分离解码器余。VAE潜空间内跑扩散。那一想法是Stable Diffusion。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程02(VAE)、阶段8课程06(DDPM)、阶段7课程09(ViT)
**时间:** ~75分钟

## 问题背景

512²像素空间扩散意味U-Net跑形`[B, 3, 512, 512]`张量。每采样步500M参数U-Net约~100 GFLOPS。50步每图像5 TFLOPS。十亿图像训计算账荒谬。

大多FLOPs推感知不重要细节过网——有损VAE可压缩掉的高频纹理。Rombach想法:一次训VAE(*第一阶段*),冻,完全4通道64×64潜空间(*第二阶段*)跑扩散。同U-Net。像素1/16。可比质量~64x少FLOPs。

此Stable Diffusion配方。SD 1.x/2.x用860M U-Net过`64×64×4`潜,SDXL用2.6B U-Net过`128×128×4`,SD3换U-Net为配流匹配扩散Transformer(DiT)。Flux.1-dev(Black Forest Labs, 2024)发12B参数DiT-MMDiT。全跑同两阶段基底。

## 概念讲解

![潜扩散:VAE压缩+潜空间扩散](../assets/latent-diffusion.svg)

**两阶段,分离训。**

1. **阶段1——VAE。**编码器`E(x) → z`,解码器`D(z) → x`。目标压缩:每空间轴8×下采样+调通道使总潜大小约像素计数1/16。损失=重构(L1 + LPIPS感知)+ KL(小权重故`z`不迫太Gaussian,因不需从`z`精确采样)。常配对抗损失训故解码图像锐。
2. **阶段2——`z`上扩散。**把`z = E(x_real)`作数据。训U-Net(或DiT)去噪`z_t`。推理:扩散采样`z_0`,后`x = D(z_0)`。

**文本条件。**两额外组件。冻结文本编码器(SD 1.x CLIP-L, SD 2/XL CLIP-L+OpenCLIP-G, SD3和Flux T5-XXL)。交叉注意力注入:每U-Net块取`[Q = 图像特征, K = V = 文本词元]`混入。词元是文本影响图像唯一方式。

**损失函数同课程06。**同DDPM/流匹配噪声MSE。仅换数据域。

## 架构变体

| 模型 | 年份 | 骨干 | 潜形状 | 文本编码器 | 参数 |
|------|------|------|--------|------------|------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L(77词元) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏 | 128×128×4 | 同 | 1-4步采样 |
| SD3 | 2024 | MMDiT(多模态DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT蒸馏 | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4步 |

趋势:U-Net换DiT(潜patch transformer),缩放文本编码器(T5提示词跟随胜CLIP),增潜通道(4 → 16给更多细节头空间)。

## 动手实践

`code/main.py`课程06 DDPM上堆玩具1-D"VAE"(身份编码器+解码器,演示;真实VAE是conv网)并加无分类器引导类条件。示同扩散损失无论跑原始1-D值或编码值——关键洞察。

### Step 1: 编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小scale
def decode(z):    return z * 2.0
```

真实VAE有训练权。教学上,此线性映射足示扩散在`z`上操作不关心原始数据空间。

### Step 2: `z`空间扩散

同课程06 DDPM。网见数据是`z = E(x)`。采样`z_0`后,`D(z_0)`解码。

### Step 3: 无分类器引导

训时,丢类标签10%(换空词元)。推理,算`ε_cond`和`ε_uncond`,后:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导(全多样), `w = 3` = 默认, `w = 7+` = 饱和/过锐。

### Step 4: 文本条件(概念,非代码)

类标签换冻结文本编码器输出。交叉注意力喂文本嵌入U-Net:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

此类条件扩散模型和Stable Diffusion唯一实质差异。

## 陷阱

- **VAE-scale不匹。**SD 1.x VAE编码后施缩放常数(`scaling_factor ≈ 0.18215`)。忘此使U-Net训潜方差严重错。每检查点发一个。
- **文本编码器静默错。**SD3需T5-XXL配>=128词元,CLIP-only fallback有损。总查`use_t5=True`否则提示词保真坍。
- **混潜空间。**SDXL、SD3、Flux全用不同VAE。SDXL潜训LoRA在SD3不工作。Hugging Face diffusers 0.30+拒载不匹检查点。
- **CFG太高。**`w > 10`产饱和、油图像并过拟合提示词代价多样。甜点是`w = 3-7`。
- **负提示词泄漏。**空负提示词变空词元;填充负提示词变`ε_uncond`。这些非同;些管道静默默认空。

## 实际应用

2026生产栈:

| 目标 | 推荐骨干 |
|------|----------|
| 窄域,配对数据,从零训模型 | SDXL微调(LoRA/全)——最快发 |
| 开域文本到图像,开权重 | Flux.1-dev(12B, Apache/非商)或SD3.5-Large |
| 最快推理,开权重 | Flux.1-schnell(1-4步, Apache)或SDXL-Lightning |
| 最佳提示词跟随,托管 | GPT-Image / DALL-E 3(仍), Midjourney v7, Imagen 4 |
| 编辑工作流 | Flux.1-Kontext(2024 12月)——原生接受图像+文本 |
| 研究,基线 | SD 1.5——古老但研究透 |

## 产出成果

存`outputs/skill-sd-prompter.md`。技能取文本提示词+目标风格输出:模型+检查点、CFG scale、采样器、负提示词、分辨率、可选ControlNet/IP-Adapter组合、和每步问答检查表。

## 练习题

1. **简单。**配引导`w ∈ {0, 1, 3, 7, 15}`跑`code/main.py`。按类记录均值样本。何`w`类均值超真实数据均值分?
2. **中等。**玩具线性编码器换tanh-MLP编码器/解码器对配重构损失。新潜上重训扩散。样本质量变否?
3. **困难。**配diffusers设真实Stable Diffusion推理:载`sdxl-base`, CFG=7跑30 Euler步,计时。现换`sdxl-turbo`配4步CFG=0。同主题,不同质量——描述何变为何。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 第一阶段 | "VAE" | 训练编码器/解码器对;压缩512²到64²。 |
| 第二阶段 | "U-Net" | 潜空间上扩散模型。 |
| CFG | "引导scale" | `(1+w)·ε_cond - w·ε_uncond`;调条件强度。 |
| 空词元 | "空提示词嵌入" | `ε_uncond`用无条件嵌入。 |
| 交叉注意力 | "文本如何入" | 每U-Net块attend文本词元作K和V。 |
| DiT | "扩散Transformer" | U-Net换潜patch transformer;缩放更好。 |
| MMDiT | "多模态DiT" | SD3架构:文本和图像流配联合注意力。 |
| VAE缩放因子 | "魔数" | 潜除~5.4故扩散单位方差空间操作。 |

## 生产注:8GB消费GPU跑Flux-12B

参考Flux集成是规范"我消费GPU,可发此?"配方。技巧同生产推理文献列扩散DiT用三旋钮配方:

1. **交错加载。**Flux有三网络不需VRAM共存:T5-XXL文本编码器(~10 GB fp32)、CLIP-L(小)、12B MMDiT、和VAE。先编提示词,*删*编码器,载DiT,去噪,*删*DiT,载VAE,解码。消费8GB GPU仅每阶段一。
2. **bitsandbytes 4位量化。**T5编码器和DiT`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存砍8×,质量降对文本到图像不可感知(Aritra基准链接notebook)。
3. **CPU offload。**`pipe.enable_model_cpu_offload()`每前向pass进展自动换模块CPU和GPU间。加10-20%延迟但管道能跑。

内存核算:`10 GB T5 / 8 = 1.25 GB`量化, `12 B参数 × 0.5 字节 = ~6 GB`量化DiT,加激活。stas00术语这是TP=1推理极端端——无模型并行,最大量化。生产H100上跑TP=2或TP=4;单开发者笔记本,此是配方。

## 延伸阅读

- [Rombach等(2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)——Stable Diffusion。
- [Podell等(2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)——SDXL。
- [Peebles & Xie(2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)——DiT。
- [Esser等(2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)——SD3, MMDiT。
- [Ho & Salimans(2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)——CFG。
- [Labs(2024). Flux.1 — Black Forest Labs公告](https://blackforestlabs.ai/announcing-black-forest-labs/)——Flux.1家族。
- [Hugging Face Diffusers文档](https://huggingface.co/docs/diffusers/index)——上每检查点参考实现。