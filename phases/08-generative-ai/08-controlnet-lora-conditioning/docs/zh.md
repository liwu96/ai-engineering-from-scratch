# ControlNet、LoRA与条件控制

> 仅文本是笨拙控制信号。ControlNet让你克隆预训扩散模型并用深度图、姿态骨架、涂鸦或边缘图像导向。LoRA让你训10M参数微调2B参数模型。一起它们把Stable Diffusion从玩具变成2026每机构发的图像管道。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程07(潜扩散)、阶段10(LLMs from Scratch——LoRA基础)
**时间:** ~75分钟

## 问题背景

"穿红裙女人在忙街遛狗"提示词不给模型*狗在何处*、*女人何姿态*、或*街何视角*信息。文本钉约需指定图像10%。余视觉且不能词高效描述。

每信号(姿态、深度、canny、分割)从零训新条件模型禁止。你想保持2.6B参数SDXL骨干冻结,附读条件小侧网络,并轻推骨干中间特征。那是ControlNet。

你还想教模型新概念(你脸、你产品、你风格)不重训全模型。你要100x更小delta。那是LoRA——插入现有注意力权重低秩适配器。

ControlNet + LoRA + 文本 = 2026实践者工具箱。大多生产图像管道叠2-5 LoRAs、1-3 ControlNets、和IP-Adapter在SDXL/SD3/Flux基顶。

## 概念讲解

![ControlNet克隆编码器;LoRA加低秩delta](../assets/controlnet-lora.svg)

### ControlNet(Zhang等, 2023)

取预训SD。*克隆*U-Net编码器半。冻原始。训克隆接受额外条件输入(边缘、深度、姿态)。配*零卷积*跳连接(1×1卷积零初始化——起no-op,学delta)克隆连回原始解码器半。

```
SD U-Net解码器:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味ControlNet起身份——训前无害。1M(提示词, 条件, 图像)三元组配标准扩散损失训。

每模态ControlNet作小侧模型发(~360M SDXL, ~70M SD 1.5)。推理时可组:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA(Hu等, 2021)

模型任何线性层`W ∈ R^{d×d}`,冻`W`加低秩delta:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

`r << d`。注意力秩4-16标准,重度微调秩64-128。新参数数:`2 · d · r`而非`d²`。SDXL注意力`d=640`, `r=16`:每适配器20k参数而非410k——20x减。全模型:LoRA常20-200MB vs基5GB。

推理可缩LoRA:`W' = W + α · B @ A`。`α = 0.5-1.5`正常。多LoRAs可加性叠(带常警告它们非线性交互)。

### IP-Adapter(Ye等, 2023)

小适配器接受*图像*作条件(文本旁)。用CLIP图像编码器产图像词元,交叉注意力旁注入文本词元。每基模型~20MB。让你做"此参考风格生成图像"无需LoRA。

## 可组合性矩阵

| 工具 | 控何 | 大小 | 何时用 |
|------|------|------|--------|
| ControlNet | 空间结构(姿态、深度、边缘) | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个人化、风格 |
| IP-Adapter | 参考图像风格或主体 | 20MB | 无文本可描述外观 |
| 文本反转 | 单概念作新词元 | 10KB | 旧,多被LoRA换 |
| DreamBooth | 主体上全微调 | 2-5GB | 强身份,高算 |
| T2I-Adapter | 更轻ControlNet替代 | 70MB | 边缘设备,推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。都用。

## 动手实践

`code/main.py`1-D模拟两机制:

1. **LoRA。**预训线性层`W`。冻。训低秩`B @ A`使`W + BA`匹目标线性层。示`r = 1`足完美学秩1修正。
2. **ControlNet-lite。**"冻结基"预测器和"侧网络"读额外信号。侧网络输出可学习标量零初始化门控(零卷积版)。训并观门升。

### Step 1: LoRA数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W冻结; A, B是可训低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: 零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate零初始化
h = base(x) + gated
```

步0输出同基。早训练慢更新`gate`——无灾难漂移。

## 陷阱

- **过缩LoRAs。**`α = 2`或`α = 3`是常见"让它更强"黑客产过风格化/破坏输出。保持`α ≤ 1.5`。
- **ControlNet权重冲突。**Pose ControlNet权重1.0和Depth ControlNet权重1.0通常过射。权重和≈ 1.0安全默认。
- **LoRA错基。**SDXL LoRAs静默SD 1.5不操作因注意力维度不匹。Diffusers 0.30+警告。
- **文本反转漂移。**一检查点训词元另一漂移严重。LoRA更可移植。
- **LoRA权重合并和存储。**LoRA可烘焙进基模型权更快推理(无运行时加),但失运行时缩`α`能力。保两版本。

## 实际应用

| 目标 | 2026管道 |
|------|----------|
| 复制品牌艺术风格 | ~30精选图像秩32训LoRA |
| 我脸放进生成图像 | DreamBooth或LoRA + IP-Adapter-FaceID |
| 特定姿态+提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考+提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble或ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting(课程09) |
| 快1步风格 | SDXL-Turbo上LCM-LoRA |

## 产出成果

存`outputs/skill-sd-toolkit-composer.md`。技能取任务(输入资产:提示词、可选参考图像、可选姿态、可选深度、可选涂鸦)输出工具栈、权重、和可复制种子协议。

## 练习题

1. **简单。**`code/main.py`中LoRA秩`r`从1到4变。何秩LoRA精确匹秩2目标delta?
2. **中等。**两目标变换分训两LoRA。一起载并示加性交互。何时交互破线性?
3. **困难。**用diffusers叠:SDXL-base + Canny-ControlNet(权重0.8) + 风格LoRA(α 0.8) + IP-Adapter(权重0.6)。栈权重变时测FID-vs-提示词跟随权衡。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ControlNet | "空间控制" | 克隆编码器+零卷积跳;读条件图像。 |
| 零卷积 | "起身份" | 1×1卷积零初始化;ControlNet起no-op。 |
| LoRA | "低秩适配器" | `W + B @ A`, `r << d`;全微调100x少参数。 |
| 秩r | "旋钮" | LoRA压缩;4-16典型,64+重度个人化。 |
| α | "LoRA强度" | LoRA delta运行时缩放。 |
| IP-Adapter | "参考图像" | CLIP-图像词元小图像条件适配器。 |
| DreamBooth | "全主体微调" | ~30主体图像训全模型。 |
| 文本反转 | "新词元" | 仅学新词嵌入;旧,多被换。 |

## 生产注:LoRA换、ControlNet道、多租户服务

真实文本到图像SaaS同基检查点服务数百LoRAs和几十个ControlNets。服务问题很似LLM多租户(生产文献覆盖LLM连续批和LoRAX/S-LoRA):

- **热换LoRAs,不合并。**`W' = W + α·B·A`进基给~3-5%更快每步推理但冻`α`和基。保LoRAs VRAM热作秩r deltas;diffusers暴露`pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`每请求激活。换成本是`2 · d · r · num_layers`权——MB级,亚秒。
- **ControlNet作第二注意力道。**克隆编码器配基并跑。两ControlNets权重各1.0 = 每步两额外前向pass,非一合并pass。批大小可用空间成倍缩减。为每个活跃ControlNet预留约1.5×步成本。
- **量化LoRAs也。**如量化基(见课程07, 8GB Flux),LoRA delta也干净量化到8位或4位。QLoRA式加载让你4位Flux基上叠5-10 LoRAs不炸内存。

Flux特定:Niels Flux-on-8GB notebook量化基到4位;量化基上叠风格LoRA(`pipe.load_lora_weights("user/style-lora")`)配`weight_name="pytorch_lora_weights.safetensors"`仍工作。这是2026大多SaaS机构发配方。

## 延伸阅读

- [Zhang, Rao, Agrawala(2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)——ControlNet。
- [Hu等(2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)——LoRA(原LLMs;移植扩散)。
- [Ye等(2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721)——IP-Adapter。
- [Mou等(2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453)——ControlNet更轻替代。
- [Ruiz等(2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)——DreamBooth。
- [HuggingFace Diffusers—ControlNet/LoRA/IP-Adapter文档](https://huggingface.co/docs/diffusers/training/controlnet)——参考管道。