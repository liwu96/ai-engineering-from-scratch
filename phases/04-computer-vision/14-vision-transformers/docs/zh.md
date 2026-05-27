# 视觉Transformer (ViT)

> 切图像为patch，每patch作词，跑标准transformer。别回头。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程02(自注意)，阶段4课程04(图像分类)
**时间:** ~45分钟

## 学习目标

- 从零实现patch嵌入、学习位置嵌入、类token和transformer编码器块构建微ViT
- 解释为何ViT被认为需大量预训数据直到DeiT和MAE证非
- 比ViT、Swin和ConvNeXt于架构先验(无、局部窗注意、卷骨干)
- 用`timm`微调预训ViT于小数据集用标准线性探测 / 微调配方

## 问题背景

十年，卷积 synonymous 计算机视觉。CNN有强归纳偏 — 局部性、平移等变 — 无人想你可替。然后Dosovitskiy等(2020)示纯transformer应于平图像patch，无卷积机制，可规模匹或胜最佳CNN。

捕获"规模"。ViT于ImageNet-1k输ResNet。ViT于ImageNet-21k或JFT-300M预训后ImageNet-1k微调胜它。结论是transformer缺有用先验但可从足数据学。后继工作(DeiT、MAE、DINO)示用正训配方 — 强增强、自监督预训、蒸馏 — ViT小数据也训好。

2026，纯CNN仍边缘设备竞争(ConvNeXt最强)，但transformer主一切其他：分割(Mask2Former、SegFormer)、检测(DETR、RT-DETR)、多模态(CLIP、SigLIP)、视频(VideoMAE、VJEPA)。ViT块结构是需知。

## 概念讲解

### 管道

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["平为<br/>(196, 768) token"]
    FLAT --> CAT["prepend<br/>[CLS] token"]
    CAT --> POS["加学习<br/>位置嵌入"]
    POS --> ENC["N transformer<br/>编码器块"]
    ENC --> CLS["取[CLS]<br/>token输出"]
    CLS --> HEAD["MLP分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七步。Patch -> token -> 注意 -> 分类器。每变种(DeiT、Swin、ConvNeXt、MAE预训)改一两七余不动。

### Patch嵌入

首conv是秘。核大小16，步幅16，故224x224图像变14x14格16x16 patch，每投射到768维嵌入。那单conv既patch化又线性投。

```
输入:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, 无填充):
输出: (768, 14, 14)
平空间: (196, 768)
```

196 patch = 196 token。每token特征维768 (ViT-B)、1024 (ViT-L)或1280 (ViT-H)。

### 类token

单学习向量prepend序列：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   形(197, 768)
```

N transformer块后，`[CLS]`输出是全局图像表示。分类头仅读此一向量。

### 位置嵌入

Transformer无内置空间位置概念。加学习向量到每token：

```
tokens = tokens + learned_pos_embedding   (也形(197, 768))
```

嵌入是模型参数；梯度训练适配到2D图像结构。正弦2D替代存但实罕用。

### Transformer编码器块

标准。多头自注意、MLP、残连、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP是两层带GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16叠12块，每12注意头，总86M参数。

### 为何pre-LN

早transformer用post-LN (`x = LN(x + sublayer(x))`)并难训过6-8层无warmup。Pre-LN (`x = x + sublayer(LN(x))`)稳训深网无warmup。每ViT和每现代LLM用pre-LN。

### Patch大小权衡

- 16x16 patch -> 196 token，标准。
- 32x32 patch -> 49 token，快但低分辨率。
- 8x8 patch -> 784 token，细但O(n^2)注意成本缩坏。

大patch = 少token = 快但少空间细节。SwinV2用4x4 patch于层次窗。

### DeiT的ImageNet-1k训ViT配方

原ViT需JFT-300M胜CNN。DeiT (Touvron等, 2020)单ImageNet-1k训ViT-B达81.8% top-1四改：

1. 重增强：RandAugment、Mixup、CutMix、随机擦除。
2. 随机深度(训练时随机丢整块)。
3. 重增强(同图像每批采3次)。
4. CNN教师蒸馏(可选，升精度)。

每现代ViT训配方降DeiT。

### Swin vs ConvNeXt

- **Swin** (Liu等, 2021) — 窗基注意。每块在局部窗注意；交替块移窗混跨窗信息。带回CNN似局部先验保注意算子。
- **ConvNeXt** (Liu等, 2022) — 重设计CNN配Swin架构选择(深卷、LayerNorm、GELU、倒瓶颈)。示差距非"注意vs卷积"而是"现代训配方 + 架构"。

2026，ConvNeXt-V2和Swin-V2皆生产级；正选依赖推理栈(ConvNeXt边缘编译更好)和预训语料。

### MAE预训

掩自编码器(He等, 2022)：随机掩75% patch，训编码器仅处理可见25%，训小解码器从编码器输出重建掩patch。预训后，丢解码器微调编码器。

MAE使ViT可单ImageNet-1k训，达SOTA，是现默认自监督配方。

## 构建

### 步骤1: Patch嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一conv，一平，一转置。那是全图像到token步。

### 步骤2: Transformer块

Pre-LN，多头自注意，MLP带GELU，残连。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention`处理分头、缩点积和输出投。`batch_first=True`故形`(N, seq, dim)`。

### 步骤3: ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"输出: {vit(x).shape}")
print(f"参数: {sum(p.numel() for p in vit.parameters()):,}")
```

约280万参数 — 微ViT CPU可训。真ViT-B是86M；同类定义`dim=768, depth=12, num_heads=12`。

### 步骤4: 健全检查 — 单图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"概率:  {logits.softmax(-1)}")
```

应无错跑。概率和1。

## 使用

`timm`船每ViT变种带ImageNet预训权重。一行：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm`是2026视觉transformer生产默认。支持ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer和数十其他同API。

多模态工作(图像 + 文)，`transformers`船CLIP、SigLIP、BLIP-2、LLaVA。那些中图像编码器皆是ViT变种。

## 交付成果

本课程产：

- `outputs/prompt-vit-vs-cnn-picker.md` — 基数据集大小、算和推理栈选ViT、ConvNeXt或Swin提示词
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — 验证ViT patch嵌入和位置嵌入形配模型期序列长技能，捕最常移植bug

## 练习题

1. **(易)** 练上微ViT前向过每中间张量形。验：输入`(N, 3, 64, 64)` -> patch `(N, 16, 192)` -> 带CLS `(N, 17, 192)` -> 分类器输入`(N, 192)` -> 输出`(N, num_classes)`。

2. **(中)** 微调预训`timm` ViT-S/16于课4合成CIFAR数据集。比同数据ResNet-18微调。报训时间和终精度。

3. **(难)** 为微ViT实现MAE预训：掩75% patch，训编码器 + 小解码器重建掩patch。评合成数据预训前后线性探测精度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Patch嵌入 | "首conv" | 核大小 = 步幅 = patch大小conv；转图像为token嵌入格 |
| 类token | "[CLS]" | prepend到token序列学习向量；其终输出是全局图像表示 |
| 位置嵌入 | "学习pos" | 加到每token学习向量使transformer知每patch何来 |
| Pre-LN | "子层前LayerNorm" | 稳transformer变种：`x + sublayer(LN(x))`而非`LN(x + sublayer(x))` |
| 多头注意 | "并行注意" | 标准transformer注意分num_heads独立子空间，后拼接 |
| ViT-B/16 | "基，patch 16" | 标准大小：dim=768, depth=12, heads=12, patch_size=16, image=224；约86M参数 |
| DeiT | "数据高效ViT" | 单ImageNet-1k强增强训ViT；证大预训数据集非严格需 |
| MAE | "掩自编码器" | 自监督预训：掩75% patch，重建；主导ViT预训配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy等, 2020)](https://arxiv.org/abs/2010.11929) — ViT论文
- [DeiT: Data-efficient Image Transformers (Touvron等, 2020)](https://arxiv.org/abs/2012.12877) — 如何单ImageNet-1k训ViT
- [Masked Autoencoders are Scalable Vision Learners (He等, 2022)](https://arxiv.org/abs/2111.06377) — MAE预训
- [timm文档](https://huggingface.co/docs/timm) — 生产中用每视觉transformer参考