# 视频生成

> 图像是2D张量。视频是3D张量。理论相同;计算难10-100x。OpenAI Sora(2024 2月)证可能。到2026 Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0和WAN 2.2发文本1080p生产视频——开权重栈(CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2)落后12月。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程07(潜扩散)、阶段7课程09(ViT)、阶段8课程06(DDPM)
**时间:** ~45分钟

## 问题背景

10秒1080p视频24fps是240帧1920×1080×3像素。~1.5 GB原始数据每片段。像素空间扩散不可行。需要:

1. **时空压缩。**VAE编码视频而非帧成时空patch序列。
2. **时间连贯。**帧需秒间共享内容、光照和对象身份。网需建模运动。
3. **计算预算。**视频训练同模型大小比图像贵10-100x。
4. **条件。**文本、图像(首帧)、音频、或另一视频。大多生产模型接受全部四。

解决架构是**扩散Transformer(DiT)**时空patch上应用,在大(提示词, caption, 视频)数据集训练。同课程06扩散损失。

## 概念讲解

![视频扩散:patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

3D VAE编码视频(学习时空压缩)。潜形状`[T_latent, H_latent, W_latent, C_latent]`。分成大小`[t_p, h_p, w_p]`patch。Sora形模型,`t_p = 1`(每帧patch)或`t_p = 2`(每两帧)。10秒1080p视频压缩到~20,000-100,000 patch。

### 时空DiT

Transformer处理平patch序列。每patch有3D位置嵌入(时间 + y + x)。注意力常分解:

- **空间注意力**每帧patch内。
- **时间注意力**同空间位置跨帧。
- **全3D注意力**贵16-100x;仅低分辨率或研究用。

### 文本条件

大文本编码器交叉注意力(Sora T5-XXL, CogVideoX-5B用T5-XXL)。长提示词很重要——Sora训练集使用GPT为每个片段生成密集重标注描述，平均每片段200个词元。

### 训练

时空潜上标准扩散损失(ε或v预测)。数据:网视频 + ~100M策管片段 + 合成文本caption。算:小研究跑10,000+ GPU小时;Sora规模100,000+。

## 2026生产格局

| 模型 | 日期 | 最大时长 | 最大分辨率 | 开权重? | 特点 |
|------|------|----------|------------|--------|------|
| Sora(OpenAI) | 2024-02 | 60s | 1080p | 否 | 首模型大尺度示世界模拟器属性 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产Sora推理5x更快 |
| Veo 2(Google) | 2024-12 | 8s | 4K | 否 | 2025最高质量+物理 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频和更强相机控制 |
| Kling 1.5/2.1(快手) | 2024-2025 | 10s | 1080p | 否 | 2025 Q1最佳人类运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 顶专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 最强角色一致性 |
| CogVideoX(THUDM) | 2024 | 10s | 720p | 是(2B, 5B) | 首开5B规模视频 |
| HunyuanVideo(腾讯) | 2024-12 | 5s | 720p | 是(13B) | 2024末开SOTA |
| Mochi-1(Genmo) | 2024-10 | 5.4s | 480p | 是(10B) | 最宽容许可 |
| WAN 2.2(阿里) | 2025-07 | 5s | 720p | 是 | 2025中最强开模型 |

开权重比图像空间更快缩差距:HunyuanVideo + WAN 2.2 LoRAs 2026中已驱动大多开源工作流。

## 动手实践

`code/main.py`模拟核心时空DiT想法:patchify小合成视频,加每patch位置嵌入,并配patch上transformer风格注意力去噪全序列。无numpy;纯Python。示相邻帧patch共享去噪器和位置嵌入时1-D也现时间连贯。

### Step 1: patchify合成1-D"视频"

```python
def make_video(T_frames=8, rng=None):
    # "视频"是平滑轨迹1-D值序列
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: 每帧位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: 去噪器见全序列

非独立去噪每帧,微小网拼接所有帧值+其位置嵌入并联合预测所有帧噪声。

### Step 4: 时间连贯测试

训练后采样视频。测帧到帧delta。如模型学时间结构,delta保持比独立采样每帧更小。

## 陷阱

- **独立每帧采样=闪烁。**如分别每帧跑图像扩散,输出闪烁因每帧噪声独立。视频扩散通过注意力或共享噪声耦合帧修复。
- **朴素3D注意力=OOM。**10秒1080p潜上全3D注意力是数百亿操作。分解成空间+时间。
- **数据captioning比大小更重要。**Sora主升级前工作是训~10x更详caption(GPT-4重标片段)。OpenAI技术报告明确此。
- **首帧条件。**大多生产模型也接受图像作首帧。这是"图像到视频"模式;训练含此变体。
- **物理漂移。**长片段(>10s)累积细微不一致。滑窗生成+关键帧锚定助。

## 实际应用

| 用例 | 2026选 |
|------|--------|
| 最高质量文本到视频,托管 | Veo 3或Sora |
| 相机控制电影 | Runway Gen-3配运动刷 |
| 跨片段角色一致 | Pika 2.0或Kling 2.1 |
| 开权重,快微调 | WAN 2.2 + LoRA |
| 图像到视频 | WAN 2.2-I2V, Kling 2.1 I2V, 或Runway |
| 音频到视频lip sync | Veo 3(原生音频)或专用lip-sync模型 |
| 视频编辑 | Runway Act-Two, Kling Motion Brush, Flux-Kontext(静止帧) |

质量平视频每秒成本2024到2026降20x。

## 产出成果

存`outputs/skill-video-brief.md`。技能取视频简(时长、aspect ratio、风格、相机计划、主体一致、音频)输出:模型+托管、提示词框架(相机语言、主体描述、运动描述符)、种子+可复制协议、和帧级问答检查表。

## 练习题

1. **简单。**`code/main.py`中比(a)独立每帧采样, (b)联合序列采样帧到帧delta。报告delta均值和方差。
2. **中等。**加首帧条件:钉帧0到给定值并采样余。测钉值如何传播。
3. **困难。**HuggingFace diffusers本地GPU跑CogVideoX-2B。720p 6秒片段计时20推理步。Profile时空注意力识别瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Video VAE | "3D VAE" | 编码器压缩`(T, H, W, C)` → 时空潜。 |
| Patch | "词元" | 潜定大小3D块;DiT输入。 |
| 分解注意力 | "空间+时间" | 先空间后时间跑注意力;跳全3D注意力。 |
| 图像到视频(I2V) | "动画此照片" | 模型取图像+文本,输出从它起视频。 |
| 关键帧条件 | "锚帧" | 钉特定帧控视频弧。 |
| 运动刷 | "方向提示" | 用户图像上画运动向量UI输入。 |
| Recaptioning | "密集重标注" | LLM重标训练片段配详提示词。 |
| 闪烁 | "时间假象" | 帧到帧不一致;耦合去噪修复。 |

## 生产注:视频潜是内存带宽问题

10秒1080p片段24fps是240帧 × 1920 × 1080 × 3 ≈ 1.5 GB原始像素。4×视频VAE压缩(`2 × 空间 × 2 × 时间`)后潜每请求~100 MB。时空DiT 30步批1过,每步移~3 GB过HBM——内存带宽,非FLOPs,瓶颈。

三生产旋钮,全直从生产推理文献推理章:

- **DiT跨TP。**文本到视频模型常≥10B参数。TP=4跨4 H100s标准;405B级模型PP=2 × TP=2。每步延迟TP到all-reduce墙大致线性降。
- **帧批=连续批。**生成时,视频概念上是注意力链接帧批。连续批(in-flight调度)适用:如模型架构允许滑窗生成,帧`t-1`返时开始渲染帧`t+1`。
- **片段级prefill缓存。**图像到视频,首帧条件类比LLM提示词prefill:算一次,跨时间解码pass复用。此是视频有效KV-cache。

## 延伸阅读

- [Brooks等(2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)——Sora技术报告。
- [Yang等(2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)——CogVideoX。
- [Kong等(2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603)——HunyuanVideo。
- [Genmo(2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi)——Mochi-1。
- [阿里巴巴(2025). WAN 2.2](https://wanvideo.io/)——2025中开SOTA。
- [Ho, Salimans, Gritsenko等(2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458)——开创视频扩散论文。
- [Blattmann等(2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818)——Stable Video Diffusion祖先。