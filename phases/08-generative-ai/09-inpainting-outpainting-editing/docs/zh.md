# Inpainting、Outpainting与图像编辑

> 文本到图像造新物。Inpainting修旧物。生产,70%计费图像工作是编辑——换背景、去logo、扩画布、再生手。Inpainting是扩散赚钱处。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程07(潜扩散)、阶段8课程08(ControlNet & LoRA)
**时间:** ~75分钟

## 问题背景

客户发来完美产品照，但背景中有干扰视线的标牌。你想擦掉标牌，同时让其余像素保持精确一致。不能从零跑文本到图像——结果会有不同颜色、不同光照、不同产品角度。你想仅掩区*再生*,并再生尊周围上下文。

那是inpainting。变体:

- **Inpainting。**掩内再生,保外像素。
- **Outpainting。**掩外(或画布外)再生,保内。
- **图像编辑。**全图像再生但保原始语义或结构忠实(SDEdit, InstructPix2Pix)。

2026每个扩散管道都提供inpainting模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们同原理工作。

## 概念讲解

![Inpainting:掩感知去噪配上下文保重注入](../assets/inpainting.svg)

### 朴素方法(何错)

配掩跑标准文本到图像。每采样步,噪潜未掩区换前向扩散干净图像。工作...差。边界假象渗透因模型无掩区何信息。

### 正确inpainting模型

训修改U-Net取9输入通道而非4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道是VAE编码源图像副本加单通道掩。训时,随机掩图像区并训模型仅去噪掩区同时未掩区作干净条件信号。推理,模型能"见"掩区周围并产连贯补。

SD-Inpaint、SDXL-Inpaint、Flux-Fill全用此9通道(或类比)输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit(Meng等, 2022)——免费编辑

源图像加噪声到某中间`t`,后配新提示词从`t`反向链到0。无重训。起始`t`选择权衡忠实和创意自由:

- `t/T = 0.3` → 近同源,小风格变
- `t/T = 0.6` → 中等编辑,保粗结构
- `t/T = 0.9` → 从近噪生成,最小源保

### InstructPix2Pix(Brooks等, 2023)

`(input_image, instruction, output_image)`三元组微调扩散模型。推理,条件于输入图像和文本指令("让它日落"、"加龙")。两CFG scale:图像scale和文本scale。

### RePaint(Lugmayr等, 2022)

保持标准无条件扩散模型。每反向步,重采样——偶尔跳回更噪态重生成。避边界假象。无训inpainting模型时用。

## 动手实践

`code/main.py`5维数据玩具1-D inpainting方案实现。5维混合数据训DDPM,每样本5浮点来自两簇之一。推理,"掩"5维中2,每步注入未掩三噪前向版本,仅掩维重生成。

### Step 1: 5-D DDPM数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: 全5维训去噪器

标准DDPM。网输出5维噪声预测对5维噪输入。

### Step 3: 掩感知反向推理

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 未掩维换干净源新鲜噪版
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...后x_t上跑正常反向步
```

此朴素方法玩具1-D数据工作。真实图像inpainting用9通道输入因纹理连贯更重要。

### Step 4: Outpainting

Outpainting是掩反转inpainting:掩新(前不存在)画布,余填原始。同训练目标。

## 陷阱

- **边界。**朴素方法留可见边界因梯度信息不流过掩。修复:掩扩张8-16像素,或用正确inpainting模型。
- **掩泄漏。**条件图像未掩区低质或噪,它污染掩内生成。去噪或微模糊。
- **CFG与掩大小交互。**小掩上高CFG = 饱和patch。小编辑降CFG。
- **SDEdit忠实崖。**从`t/T = 0.5`到`t/T = 0.6`可失主体身份。扫和检查点。
- **提示词不匹。**提示词应描述*全*图像,非仅新内容。"猫坐椅上"非"猫"。

## 实际应用

| 任务 | 管道 |
|------|------|
| 去对象,小掩 | SD-Inpaint或Flux-Fill,标准提示词 |
| 换天 | SD-Inpaint + "日落蓝天" |
| 扩画布 | SDXL outpaint模式(8px羽)或Flux-Fill配outpaint掩 |
| 重生手/脸 | SD-Inpaint配提示词重述主体+ControlNet-Openpose |
| 换一区风格 | 掩区SDEdit配`t/T=0.5` |
| "让它日落" | InstructPix2Pix或Flux-Kontext |
| 背景替换 | SAM掩 → SD-Inpaint |
| 超高保真 | Flux-Fill或GPT-Image(托管)最难情况 |

SAM(Meta Segment Anything, 2023) + 扩散inpaint是2026背景移管道。SAM 2(2024)视频工作。

## 产出成果

存`outputs/skill-editing-pipeline.md`。技能取原始图像+编辑描述+可选掩(或SAM提示词)输出:掩生成方法、基模型、CFG scale(图像+文本)、SDEdit-t或inpainting模式、和问答检查表。

## 练习题

1. **简单。**`code/main.py`中掩维分数从0.2到0.8变。何分数inpaint质量(掩维残)等无条件生成?
2. **中等。**实现RePaint:每10反向步,跳回5步(加噪)重去噪。测是否减掩边残。
3. **困难。**Hugging Face diffusers比:SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill20脸重生任务。姿态跟随和身份保分评分。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Inpainting | "填洞" | 掩内再生;保外像素。 |
| Outpainting | "扩画布" | 画布外再生;保内。 |
| 9通道U-Net | "正确inpainting模型" | `noisy | encoded-source | mask`作输入U-Net。 |
| SDEdit | "配噪级Img2img" | 噪到时间`t`,配新提示词去噪。 |
| InstructPix2Pix | "仅文本编辑" | (图像,指令,输出)三元组微调扩散。 |
| RePaint | "无重训" | 反向时定期重噪减边界。 |
| SAM | "Segment Anything" | 点击或框掩生成器;配inpaint。 |
| Flux-Kontext | "配上下文编辑" | Flux变体接受参考图像+指令编辑。 |

## 生产注:编辑管道延迟敏感

用户编辑图像期望亚5秒轮转。1024²30步SDXL-Inpaint L4上3-4秒,加SAM掩生成(~200 ms)和VAE编/解码(~500 ms合)。生产框定,这是TTFT界而非吞吐界——批1,低并发,最小每阶段:

- **SAM-H慢那个。**1024² SAM-H ~200 ms;SAM-ViT-B ~40 ms质量损失小。SAM 2(视频)加时间开销;单图像编辑不用。
- **可能时跳编码。**`pipe.image_processor.preprocess(img)`编码到潜。如你前生成有潜(迭代编辑UI典型),直通过`latents=...`跳一VAE编码。
- **掩扩张吞吐也重要。**小掩意味U-Net前向pass大多浪费(未掩像素clamp)。`diffusers` `StableDiffusionInpaintPipeline`跑全U-Net;仅9通道正确inpaint变体利用掩计算。
- **Flux-Kontext是2025答案。**(source_image, instruction)单前向pass——无分离掩,无SDEdit噪扫。H100上发编辑~1.5秒。架构教训:坍阶段。

## 延伸阅读

- [Lugmayr等(2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865)——无训inpainting。
- [Meng等(2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)——SDEdit。
- [Brooks, Holynski, Efros(2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800)——文本指令编辑。
- [Kirillov等(2023). Segment Anything](https://arxiv.org/abs/2304.02643)——SAM,掩源。
- [Ravi等(2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)——视频SAM。
- [Hertz等(2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626)——注意力级编辑。
- [Black Forest Labs(2024). Flux.1-Fill和Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/)——2024工具。