# Vision Transformer (ViT)

> 图像patch网格。句子词元网格。同一transformer吃两者。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段4课程03(CNN)、阶段4课程14(Vision Transformer介绍)
**时间:** ~45分钟

## 问题背景

2020前,计算机视觉意味卷积。ImageNet、COCO和检测基准每个SOTA用CNN骨干。Transformer为语言。

Dosovitskiy等(2020)——"An Image is Worth 16x16 Words"——展示你可完全弃卷积。把图像切成定大小patch,每个patch线性投影到嵌入,喂序列给普通transformer编码器。在足够规模(ImageNet-21k预训练或更大),ViT匹或胜ResNet基模型。

ViT是2026更广模式起点:一架构,多模态。Whisper tokenize音频。ViT tokenize图像。机器人action tokens。视频pixel tokens。Transformer不在乎——喂序列它学。

到2026,ViT及其后代(DeiT、Swin、DINOv2、ViT-22B、SAM 3)拥有多数视觉。CNN仍在边缘设备和延迟敏感任务胜。其余栈某处有ViT。

## 概念讲解

![图像→patches→词元→transformer](../assets/vit.svg)

### Step 1——patchify

把`H × W × C`图像切成`N × (P·P·C)`平patch序列。典型设置:`224 × 224`图像,`16 × 16` patches → 196个768值patch。

```
图像(224, 224, 3) → 14 × 14网格16x16x3 patches → 196个长768向量
```

Patch大小是杠杆。更小patch=更多词元、更好分辨率、二次注意力成本。更大patch=更粗、更便宜。

### Step 2——线性嵌入

单学习矩阵把每平patch投影到`d_model`。等价于kernel大小`P`stride `P`卷积。PyTorch这是字面`nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——2行实现。

### Step 3——前置`[CLS]`词元、加位置嵌入

- 前置可学习`[CLS]`词元。其最终隐藏状态是分类用图像表示。
- 加可学习位置嵌入(ViT原始)或sinusoidal 2D(后变体)。
- 2024+ RoPE扩展到2D位置,有时无显式嵌入。

### Step 4——标准transformer编码器

堆L块`LayerNorm → Self-Attention → + → LayerNorm → MLP → +`。同BERT。无视觉特定层。这是论文教学要点。

### Step 5——头

分类:取`[CLS]`隐藏状态→linear→softmax。DINOv2或SAM,弃`[CLS]`,直接用patch嵌入。

### 重要变体

| 模型 | 年份 | 变化 |
|------|------|------|
| ViT | 2020 | 原始。固定patch大小,全全局注意力。 |
| DeiT | 2021 | 蒸馏;仅ImageNet-1k可训练。 |
| Swin | 2021 | 配shifted windows分层。固定亚二次成本。 |
| DINOv2 | 2023 | 自监督(无标签)。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B参数;缩放定律适用。 |
| SigLIP | 2023 | ViT + 语言pair,sigmoid对比损失。 |
| SAM 3 | 2025 | Segment anything;ViT-Large + 可提示掩码解码器。 |

### 为何花了一段时间

ViT需*很多*数据匹CNN因它无CNN归纳偏置(平移不变性、局部性)。无>100M标注图像或强自监督预训练,CNN仍在匹配计算下胜。DeiT 2021用蒸馏技巧修复;DINOv2 2023用自监督永久修复。

## 动手实践

见`code/main.py`。纯stdlib patchify + 线性嵌入 + 健全检查。无训练——任何现实规模ViT需PyTorch和数小时GPU时间。

### Step 1: 伪造图像

24 × 24 RGB图像作(R, G, B) tuple行列表。用6×6 patches → 16 patches,每108维嵌入向量。

### Step 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster顺序:跨网格行主序。每个ViT用此顺序。

### Step 3: 线性嵌入

每平patch乘随机`(patch_flat_size, d_model)`矩阵。验证前置`[CLS]`后输出形状`(N_patches + 1, d_model)`。

### Step 4: 为现实ViT算参数

打印ViT-Base参数数:12层,12头,d=768,patch=16。比ResNet-50(~25M)。ViT-Base落~86M。ViT-Large ~307M。ViT-Huge ~632M。

## 实际应用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2嵌入是2026图像特征默认。**冻结骨干,训微小头。分类、检索、检测、captioning皆可。Meta DINOv2检查点在每个非文本视觉任务胜CLIP。

**Patch大小选择。**小模型用16×16(ViT-B/16)。稠密预测(分割)用8×8或14×14(SAM、DINOv2)。极大模型用14×14。

## 产出成果

见`outputs/skill-vit-configurator.md`。技能给定数据集大小、分辨率和计算预算为新视觉任务选ViT变体和patch大小。

## 练习题

1. **简单。**运行`code/main.py`。验证patch数等于`(H/P) * (W/P)`且平patch维度等于`P*P*C`。
2. **中等。**实现2D sinusoidal位置嵌入——每patch`row`和`col`两独立sinusoidal码,拼接。喂进微小PyTorch ViT并在CIFAR-10比可学习位置嵌入准确率。
3. **困难。**建3层ViT(PyTorch),在1000 MNIST图像配4×4 patches训。测测试准确率。现加DINOv2预训练在同1000图像(简化:仅训编码器从掩patch预测patch嵌入)。准确率改进否?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Patch | "视觉-transformer词元" | 图像`P × P × C`区域像素值平向量。 |
| Patchify | "切+平" | 把图像切非重叠patch,每patch平成向量。 |
| `[CLS]`词元 | "图像摘要" | 前置可学习词元;其最终嵌入是图像表示。 |
| 归纳偏置 | "模型假设" | ViT比CNN少先验;需更多数据补差距。 |
| DINOv2 | "自监督ViT" | 无标签训练用图像增强+动量教师。2026最佳通用图像特征。 |
| SigLIP | "CLIP后继" | ViT + 文本编码器配sigmoid对比损失训;匹配计算下比CLIP更好。 |
| Swin | "窗口ViT" | 配局部注意力+shifted windows分层ViT;亚二次。 |
| Register tokens | "2023技巧" | 几额外可学习token吸收attention sinks;改进DINOv2特征。 |

## 延伸阅读

- [Dosovitskiy等(2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)——ViT论文。
- [Touvron等(2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)——DeiT。
- [Liu等(2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)——Swin。
- [Oquab等(2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)——DINOv2。
- [Darcet等(2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)——DINOv2 register-token修复。