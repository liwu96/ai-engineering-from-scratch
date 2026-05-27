# Any-Resolution Vision: Patch-n'-Pack和NaFlex

> 真图像非224x224方。收据是9:16,图表是16:9,医学扫可能是4096x4096,手机截图是9:19.5。2024前VLM答—resize一切至固定方—丢OCR、文档理解和高清场景解析信号。NaViT (Google, 2023)示你可pack变分辨率patch入单transformer batch带block-diagonal masking。Qwen2-VL M-RoPE (2024)完全丢绝对位置表。LLaVA-NeXT AnyRes tile高分辨率图像为base + sub-images。SigLIP 2 NaFlex变种(2025)今是开VLM默编码器欲单checkpoint服每纵横比。本课端到端实patch-n'-pack。

**类型:** 构建
**语言:** Python(stdlib,patch packer + block-diagonal mask)
**前置要求:** 阶段12课程01(ViT patch),阶段12课程05(LLaVA)
**时间:** ~120分钟

## 学习目标

- Pack变分辨率图像batch patch入一序列并建block-diagonal attention mask。
- 给定任务择AnyRes tiling(LLaVA-NeXT)、NaFlex(SigLIP 2)和M-RoPE(Qwen2-VL)。
- 无resizing算OCR、图表和摄影token预算。
- 命square-resize三失模式:squished文、cropped内容、padding浪费token。

## 问题背景

Transformer期序列。Batch是同长序列stack。若你图像224x224,你每得196 patch token,padding不需,工完。训于224,推理于224,永不思分辨率。

世界不合。文档portrait(8.5x11英寸,2:3-ish)。图表截图landscape(16:9)。收据高瘦(1:3)。医学影像发于2048x2048或更大。移动设备截图1170x2532(0.46:1)。

2024前三选项和何每失:

1. Resize至固定方(224x224或336x336)。Squish歪文和脸。Downscale毁图表标签和OCR内容。LLaVA-1.5前标准实践。
2. Crop至固定纵横比。你丢图像大部,择crop位置是己视觉问题。
3. Pad至最长边。Fix歪但portrait图像50%+ token浪费于padding。所有那些pad token二次方attention成本。

2024-2025答:让transformer于图像原生分辨率吃patch,并图如何pack异质batch入一序列无浪费算。

## 概念讲解

### NaViT和patch-n'-pack

NaViT (Dehghani et al., 2023)是示此规模工论文。念是机械:

1. 每batch图像,于择patch size(如14)算其原生patch网格。
2. Flatten每图像patch入其己变长序列。
3. Concatenate所有图像patch入batch一长序列。
4. 建block-diagonal attention mask使图像A patch仅attend图像A内。
5. 携每patch位置信息(2D RoPE或fractional position embedding)。

三图像batch 336x336(576 token)、224x224(256 token)和448x336(768 token)成一1600-token序列带1600x1600 block-diagonal mask。无padding。无浪费算。Transformer理任纵横比。

NaViT也引fractional patch dropping训间—于batch随机丢50% patch—既regularize又加速训。SigLIP 2继承此。

### AnyRes(LLaVA-NeXT)

LLaVA-NeXT AnyRes是务实替。给高分辨率图像和固定编码器(CLIP或SigLIP于336),tile图像:

1. 从预定义集择grid布局—(1x1),(1x2),(2x1),(1x3),(3x1),(2x2)等—最适图像纵横比。
2. Tile全图像入grid;每tile成336x336 crop。
3. 也产thumbnail:全图像resize至336x336作global-context token。
4. 经冻336-encoder编码每tile。Concatenate tile token + thumbnail token。

672x672图像于2x2 grid加thumbnail:4 * 576 + 576 = 2880视觉token。贵但效—LLM见本细节和全局context。

AnyRes是你编码器冻且仅支持一分辨率择路。它炸大图像token数(1344x1344图像于4x4 grid是9216 + 576 ≈ 9800 token,填8k LLM context大部)。

### M-RoPE(Qwen2-VL)

Qwen2-VL引Multimodal Rotary Position Embedding。代NaViT fractional position或AnyRes tile-and-thumbnail,每patch携3D位置(temporal, height, width)。Query/key旋理任H、W和时长。

M-RoPE发原生动态分辨率无重训。推理时你feed任HxW图像,patch embedder产H/14 x W/14 token,每token得其(t=0, r=row, c=col)位置,RoPE旋attention正频,完。Qwen2.5-VL和Qwen3-VL续此。InternVL3 V2PE是同念带每模态变编码。

Unlike AnyRes,M-RoPE是O(H x W / P^2) token于原生分辨率—无乘tile开销。Unlike NaViT,它仍期每forward单图像。跨分辨率batch仍需上patch-n'-pack。

### NaFlex(SigLIP 2)

NaFlex是SigLIP 2 checkpoint原生flex模式。单模型推理服多序列长(256,729,1024 token)。内训时用NaViT类patch-n'-pack和每patch绝对fractional position。卖点:一checkpoint,推理时按任务择token预算。

语义任务(分类、检索),256 token。OCR或图表理解,1024 token。无重训。

### Packing mask

Block-diagonal mask是多实绊处。长`N_total`覆图像`i=0..B-1`带长`n_i` pack序列,形`(N_total, N_total)` mask `M`是1若两索引落同图像block,else 0。你可从cumulative length list建:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

这是PyTorch `torch.block_diag`或显gather一行。FlashAttention variable-length path(`cu_seqlens`)完全跳mask并用cumulative-length tensor直attend内序列—典型batch比dense mask快~10x。

### Token预算

按任务择策略:

- OCR/文档:1024-4096 token。SigLIP 2 NaFlex于1024,或AnyRes 3x3 + thumbnail。
- 图表和UI:384-448原生729-1024 token。Qwen2.5-VL动态分辨率带max pixels cap。
- 自然照:256-576 token够。下游LLM见够。内容密高处付token。
- 视频:空间pool后每帧64-128 token,2-8 FPS。课程12.17覆盖此。

2026产规则:择每任务max-pixels cap,于原生纵横比编码至那cap,pack batch,跳padding。Qwen2.5-VL暴露`min_pixels`和`max_pixels`正此knob。

## 使用

`code/main.py`为异质图像batch带整数像素坐标实patch-n'-pack。它:

- 取(H,W)图像大小列表。
- 于patch size 14算每图像patch序列长。
- Pack它们入总长`sum(n_i)`一序列。
- block-diagonal attention mask(密,清晰)。
- 比packed成本vs square-resize和AnyRes tiling。
- 打mixed batch(收据、图表、截图、照片)token预算表。

跑它。出数是何每2026开VLM用patch-n'-pack因。

## 交付成果

本课产`outputs/skill-resolution-budget-planner.md`。给mixed纵横比工作负载(OCR、图表、照片、视频帧)和总token预算,它择正策略(NaFlex、AnyRes、M-RoPE或fixed-square)并发每请求配。于你为产sizing VLM时用此技能—防静10x token炸杀延迟预算。

## 练习题

1. 收据600x1500(1:2.5)。于patch size 14,原生分辨率何token?square-resize至336后何?实践中何失多OCR精度?

2. 建四图像长256、576、729、1024 batch block-diagonal mask。验attention矩阵2585x2585有正`256^2 + 576^2 + 729^2 + 1024^2`非零entry。

3. 1792x896图像patch 14,比:(a)square-resize至336后encode,(b)AnyRes 2x1 + thumbnail,(c)M-RoPE原生。何用最少token?何保最多细节?

4. 实fractional patch dropping:给packed序列,uniformly random丢50% token,更新block-diagonal mask。测mask稀疏变。

5. 读Qwen2-VL论文(arXiv:2409.12191)节3.2。两句述`min_pixels`和`max_pixels`控何和何两bound重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch-n'-pack | "NaViT类packing" | Concatenate异图像变长patch序列入一batch维 |
| Block-diagonal mask | "Packing mask" | Attention mask限每图像patch仅attend自己非pack邻居 |
| AnyRes | "LLaVA-NeXT tiling" | 分高分辨率图像为固定大小tile grid加全局thumbnail;用固定编码器编码每tile |
| NaFlex | "SigLIP 2原生flex" | 单SigLIP 2 checkpoint推理服256/729/1024 token预算无重训 |
| M-RoPE | "多模RoPE" | 3D旋位置编码(时、行、列)理任H、W、T无位置表 |
| cu_seqlens | "FlashAttention packing" | FlashAttention varlen path用cumulative-length tensor替密block-diagonal mask |
| min_pixels / max_pixels | "分辨率bound" | Qwen2.5-VL每请求knob capping token数于极小或极大输入 |
| 视觉token预算 | "每图像何token" | 每图像发patch token粗计数;定LLM提示预算和attention成本 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)