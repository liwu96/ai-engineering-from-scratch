# Vision Transformers和Patch-Token原语

> 多模态前,图像必须成transformer可吃token序列。2020 ViT论文答此用16x16像素patch、线性投影和位置embedding。五年后每个2026前沿模型(Claude Opus 4.7于2576px原生、Gemini 3.1 Pro、Qwen3.5-Omni)仍这样开始—编码器从ViT改至DINOv2改至SigLIP 2,加register token,位置方案成2D-RoPE,但原语存。本课读patch-token管道端到端并用stdlib Python建,使阶段12余课有"视觉token"具心智模型。

**类型:** 学习
**语言:** Python(stdlib,patch tokenizer + geometry calculator)
**前置要求:** 阶段7(Transformers),阶段4(计算机视觉)
**时间:** ~120分钟

## 学习目标

- 将HxWx3图像转成带正确位置编码patch token序列。
- 计给定(patch size, resolution, hidden dim, depth)ViT的序列长度、参数量和FLOPs。
- 命ViT从2020研究至2026产三升级:自监督预训练(DINO/MAE)、register token和原生分辨率pack。
- 为下游任务择CLS pooling、mean pooling或register token。

## 问题背景

Transformer操作向量序列。文已是序列(字节或token)。图像是三色通道2D像素网格—非序列。若flatten每像素,224x224 RGB图像成150,528 token,于那长度self-attention不可开(序列长度二次方)。

2020前方法CNN特征提取器bolt至前:ResNet产7x7 2048维向量特征图,feed那49 token至transformer。这工但继承CNN偏(平移等变性、局部感受野)和失transformer规模胃口。

Dosovitskiy et al. (2020)问直问题:若跳CNN何如?分图像至固定大小patch(如16x16像素),线性投每patch至向量,加位置embedding,feed序列至vanilla transformer。时此异端—无卷积视觉。于够数据(JFT-300M,后LAION)于ImageNet超ResNet并续改进。

于2026 ViT原语无疑问基础。每个开权VLM视觉塔是某后代(DINOv2、SigLIP 2、CLIP、EVA、InternViT)。问题非"应使patch?"而是"何patch size、何分辨率调度、何预训练目标、何位置编码。"

## 概念讲解

### Patch作token

给图像`x`形`(H, W, 3)`和patch size `P`,你刻图像至`(H/P) x (W/P)`非重叠patch网格。每patch是`P x P x 3`像素立方。flatten每立方至`3 P^2`向量。用共享线性投影`W_E`形`(3 P^2, D)`映每patch至模型隐藏维`D`。

ViT-B/16典配:
- 分辨率224,patch size 16 → 网格14x14 → 196 patch token。
- 每patch是`16 x 16 x 3 = 768`像素值,投至`D = 768`。
- 加可学`[CLS]` token → 序列长197。

patch投影数学上等同于核大小`P`、stride `P`、`D`输出通道2D卷积。这是产代码实法—`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"框架是概念;核框架是效。

### 位置embedding

Patch无固有顺序—transformer视它们为bag。早ViT加可学1D位置embedding(每位置一768维向量,197个)。工,但绑模型至训练分辨率:推理时若改网格需插值位置表。

现代视觉backbone用2D-RoPE(Qwen2-VL M-RoPE、SigLIP 2默)或分解2D位置。2D-RoPE基于patch(row, column)索引旋query和key向量,使模型从旋角推断相对2D位置。无位置表。模型推理时处理任意网格大小。

### CLS token、pool输出和register token

何是图像级表示?三择共存:

1. `[CLS]` token。prepend可学向量至patch序列。经所有transformer块后,CLS token隐藏态是图像表示。继承自BERT。原ViT、CLIP用。
2. Mean pool。平均patch token输出隐藏态。SigLIP、DINOv2、多现代VLM用。
3. Register token。Darcet et al. (2023)观察无显sink token训练ViT发育劫持self-attention高范"artifacts" patch。加4–16可学register token吸收此载并改密预测质量(分割、深度)。DINOv2和SigLIP 2都带registers发。

择对下游任务重要。CLS适分类。对feed patch token至LLM VLM,你跳pool完全—每patch成LLM输入token。Registers于handoff前丢弃(它们是脚手架非内容)。

### 预训练:监督、对比、掩码、自蒸馏

2020 ViT于JFT-300M监督分类预训练。快被:

- CLIP (2021):于400M对比图文。阶段12课程02。
- MAE (2021, He et al.):掩75% patch,重构像素。自监督,纯图工。
- DINO (2021) / DINOv2 (2023):师生自蒸馏,无标签无caption。2023 DINOv2 ViT-g/14是最强纯视觉backbone和"密特征"用例默。
- SigLIP / SigLIP 2 (2023, 2025):带sigmoid loss和NaFlex原生纵横比CLIP。2026开VLM(Qwen、Idefics2、LLaVA-OneVision)主导视觉塔。

你预训练择定backbone适何:CLIP/SigLIP适语义文配,DINOv2适密视觉特征,MAE适下游微调起点。

### Scaling law

ViT scaling (Zhai et al. 2022)建ViT质量于模型大小、数据大小和算力服从可预测律。于定算力:
- 大模型+多数据 → 好质量。
- Patch size是序列长度vs保真杠杆。Patch 14(DINOv2/SigLIP SO400m典型)每图像产多token于patch 16;OCR和密任务好,速度差。
- 分辨率是他大杠杆。224至384至512几总助,于FLOPs二次方成本。

ViT-g/14 (1B参数,patch 14,分辨率224 → 256 token)和SigLIP SO400m/14 (400M参数,patch 14)是2026开VLM两个工马编码器。

### ViT参数量

全计算在`code/main.py`。ViT-B/16于224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

于载checkpoint前每ViTball-park此。Backbone大小定你下游VLM VRAM基线。

### 2026产配

2026多开VLM发编码器是SigLIP 2 SO400m/14于原生分辨率(NaFlex)。它有:
- 400M参数。
- Patch size 14,默分辨率384 → 每图像729 patch token。
- 图像级任务mean pool;VQA全729 patch流入LLM。
- 4 register token,LLM handoff前丢弃。
- 原生纵横比图像级scaling 2D-RoPE。

那配每决追溯至你可读论文。

## 使用

`code/main.py`是patch tokenizer和geometry calculator。它取(image H, W, patch P, hidden D, depth L)并报:

- Patch后网格形和序列长。
- 合成8x8像素toy图像token序列(走flatten + project路径)。
- 按patch embed、position embed、transformer块和头分解参数量。
- 目标分辨率每forward pass FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384比较表。

跑它。参数量匹配发数。玩patch size和分辨率感token数成本。

## 交付成果

本课产`outputs/skill-patch-geometry-reader.md`。给ViT配(patch size, resolution, hidden dim, depth),它产token数、参数量和VRAM估计带理。于你为VLM择视觉backbone时使此技能—防"token炸和LLM context填"惊。

## 练习题

1. 计Qwen2.5-VL于原生1280x720输入patch size 14的patch-token序列长。何与CLS-only表示比?

2. 1080p帧(1920x1080)于patch 14产何token?于30 FPS过5分钟视频,总视觉token何?何成本最省:pooling、帧采样或token合并?

3. 纯Python实patch token mean pooling。验DINOv2输出196 token mean-pool匹模型`forward`返pool embedding。

4. 读"Vision Transformers Need Registers"(arXiv:2309.16588)节3。两句述何artifacts registers吸收和何为下游密预测重要。

5. 改`code/main.py`支持patch-n'-pack:给不同分辨率图像列表,产单pack序列和block-diagonal attention mask。达阶段12课程06时验。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | "16x16像素方" | 输入图像固定大小非重叠区域;成一token |
| Patch embedding | "线性投影" | 映flatten patch像素至D维向量共享学矩阵(或stride=P Conv2d) |
| CLS token | "类token" | prepend可学向量其终隐藏态代表全图;2026可选 |
| Register token | "Sink token" | 吸收ViTs预训练中发育高范attention artifacts额外可学token |
| Position embedding | "位置信息" | 使序列order-aware每位置向量或旋;2D-RoPE现代默 |
| Grid | "Patch网格" | 给定分辨率和patch size(H/P) x (W/P) 2D patch数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2特性:单模型无重训服多纵横比和分辨率 |
| Backbone | "视觉塔" | VLM中其patch-token输出feed LLM预训练图像编码器 |
| Pooling | "图像级总结" | 策略转patch token成一向量:CLS、mean、attention pool或register基 |
| Patch 14 vs 16 | "细vs粗网格" | Patch 14每图像产多token,OCR保真好,慢;Patch 16典默 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE,自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 规模自蒸馏,无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — Register token和artifact分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026默视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验scaling law。