# LLaVA和视觉指令微调

> LLaVA (2023年4月)是星球上被复制最多多模态架构。它替BLIP-2 Q-Former为2-layer MLP,替Flamingo gated cross-attention为朴素token拼接,并训于158k GPT-4从纯文caption生成视觉指令转。2023至2026间建VLM任从业者建某LLaVA变种。LLaVA-1.5加AnyRes。LLaVA-NeXT bump分辨率。LLaVA-OneVision统图、多图和视频于一配方。本课读配方、实projector并释"简者胜"何。

**类型:** 构建
**语言:** Python(stdlib,projector + instruction-template builder)
**前置要求:** 阶段12课程02(CLIP),阶段11(LLM工程—指令微调)
**时间:** ~180分钟

## 学习目标

- 建映ViT patch embedding(维1024)至LLM embedding维(维4096)2-layer MLP projector。
- LLaVA两阶段配方:(1)558k caption对projector对齐,(2)158k GPT-4生成转视觉指令微调。
- 构LLaVA格式提示带图像token placeholder、系统提示和user/assistant转。
- 释社区何从Q-Former移至MLP尽管Q-Former token预算胜。

## 问题背景

BLIP-2 Q-Former(课程12.03)压缩图像至32 token。干净、高效、benchmark好。但有两问题。

第一,Q-Former可训但其loss非终任务。阶段1训ITC+ITM+ITG。阶段2训LM loss。Query学某中间表示LLM后需解码。信息瓶颈中失。

第二,Q-Former占188M参数,于LLaVA 2023 scale你需与目标LLM co-design。换LLM,重训Q-Former。换视觉编码器,重训。每组合是独R&D项目。

LLaVA答 embarrassingly简:取ViT 576 patch token,经每2-layer MLP(`1024 → 4096 → 4096`),dump全576入LLM输入序列。无瓶颈。无阶段1于怪目标预训。仅直接LM loss训MLP。

数据何来?LLaVA第二洞察:用GPT-4(纯文)生成instruction数据。Feed GPT-4图像COCO caption和bounding-box数据,让它产对话、描述和复杂推理问题。158k instruction-response转免费。无人工标注。

结果:VLM于8 A100跑一天,于MMMU超Flamingo,并发开checkpoint社区可扩。2023末它spawn 50+ fork。

## 概念讲解

### 架构

LLaVA-1.5于13B:
- 视觉编码器:CLIP ViT-L/14 @ 336(阶段1冻,阶段2可选unfreeze)。
- Projector:带GELU激活2-layer MLP,`1024 → 4096 → 4096`。
- LLM:Vicuna-13B(后Llama-3.1-8B)。

图像+文提示forward pass:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占LLM context 576 token。于2048 context,留1472 token文。于32k context,是舍入误差。

### 阶段1:Projector对齐

冻ViT。冻LLM。仅训2-layer MLP。数据集:558k图文对(LAION-CC-SBU)。Loss:caption上语言建模,条件于投图像token。

单epoch batch 128几小时完。Projector学映ViT空间至LLM空间。无任务特定监督。

### 阶段2:视觉指令微调

Unfreeze projector(仍可训)。Unfreeze LLM(通常全,有时LoRA)。于158k视觉指令转训。

Instruction数据是技。Liu et al.生成它经:
1. 取COCO图像。
2. 提文描述(5人工caption + bounding-box列表)。
3. 发GPT-4带三提示模板:
   - Conversation:"生成此图用户和助手间来回对话。"
   - Detailed description:"给图像富、详描述。"
   - Complex reasoning:"问需图像推理问题,后答它。"
4. 解GPT-4输出为(instruction, response)对。

此都不直触图像—仅文描述。GPT-4 plausible图像内容hallucinate。些噪,但工:158k转够解锁对话。

### 何社区复制此

- 无阶段1特定loss调。全LM loss。
- Projector几小时训,非天。
- LLM可换(LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3)仅重训projector。
- 视觉instruction数据管道用GPT-4并为新域便宜重生成。

### LLaVA-1.5和LLaVA-NeXT

LLaVA-1.5 (2023年10月)加:
- Academic任务数据(VQA、OKVQA、RefCOCO)混入instruction tuning。
- 更好系统提示。
- 2048 → 32k context。

LLaVA-NeXT (2024年1月)加:
- AnyRes:分高分辨率图像为2x2或1x3 336x336裁剪网格,加一全局低分辨率缩略图。每裁剪成576 token;每图约2880视觉token。OCR和图表任务跳。
- 更好instruction数据混合带ShareGPT4V(高质量GPT-4V caption)。
- 更强base LLM(Mistral-7B、Yi-34B)。

### LLaVA-OneVision

课程12.08深覆盖OneVision。短版:同projector,但训带覆盖单图、多图和视频于一模型共享视觉token预算curriculum。

### 与Q-Former比

| | Q-Former(BLIP-2) | MLP(LLaVA) |
|---|---|---|
| 每图视觉token | 32 | 576(基)或2880(AnyRes) |
| 可训参数 | 188M + LM | 40M + LM |
| 阶段1 loss | ITC+ITM+ITG | 仅LM |
| LLM drop-in | 需重训 | 最小重训换 |
| 多图 | awkward | 自然(concat) |
| 视频 | awkward | 自然(每帧concat) |
| Token预算 | 小 | 大 |

MLP于简和token灵活胜。Q-Former于token预算胜。2023末token预算不再binding约束(LLM context长至32k-128k+)和简主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>`是placeholder token。Tokenization前,它被576视觉token(或AnyRes 2880)替。Tokenizer见略长于训序列,但LLM理novel输入因阶段1教它。

### 参数经济

LLaVA-1.5-7B分解:
- CLIP ViT-L/14 @ 336:303M(阶段1冻,阶段2常unfreeze)。
- Projector(2x linear):~22M可训。
- Llama-7B:7B。
- 总:7.3B参数。阶段2可训:全7B + 22M projector。

阶段2训成本:~20小时于8xA100。这是关键数—一天、一节点、可复现。这是何LLaVA传开。

## 使用

`code/main.py`实:

1. 2-layer MLP projector(维16 → 32 → 32 toy scale)纯Python。
2. 提示构建管道:系统提示+`<image>`被N投token替+用户转+assistant生成placeholder。
3. 何576-token视觉块于LLM context看可视器(2k/32k/128k context消费百分比)。

## 交付成果

本课产`outputs/skill-llava-vibes-eval.md`。给LLaVA族checkpoint,它跑10提示vibes-eval套(3 captioning、3 VQA、2 reasoning、2 refusal)并报人读scorecard。非benchmark;确projector和LLM接好烟测。

## 练习题

1. 计`1024 → 4096 → 4096` 2-layer MLP projector可训参数量。带GELU和bias,它占LLaVA-13B何分?

2. 构"refusal"例LLaVA提示—图像含私人个体。写期望assistant响应。何LLaVA零shot应拒此和何训数据需强拒?

3. 读LLaVA-NeXT blog AnyRes节。计1344x672图像于AnyRes视觉token数。与336x336基576 token比。

4. LLaVA阶段1 projector于caption LM loss训。若跳阶段1直去阶段2(视觉指令微调)何?引Prismatic VLMs ablation(arXiv:2402.07865)为答。

5. LLaVA-Instruct-150k用GPT-4与COCO caption生成instruction。为新域(医学X光、卫星图像),述四步数据管道生成域instruction。每步何可出错?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Projector | "MLP桥" | 映ViT维至LLM维带GELU 2-layer MLP |
| 图像token | "<image> placeholder" | 推理前被N投视觉token替提示标记 |
| 视觉指令微调 | "LLaVA阶段2" | 于GPT-4生成(图,instruction,response) triplet训 |
| 阶段1对齐 | "Projector预训" | 冻ViT和LLM,于caption LM loss训projector |
| AnyRes | "多裁剪tile" | 分高分辨率图像为tile网格并拼接每tile视觉token |
| LLaVA-Instruct | "GPT-4生成" | 从COCO caption + GPT-4合成158k instruction-response对 |
| 视觉编码器冻 | "Backbone锁" | CLIP权重阶段1不更新,有时阶段2也不 |
| ShareGPT4V | "更好caption" | GPT-4V生成1M密caption,为高质量对齐用 |
| VQA | "视觉问答" | 关于图像答自由形式问题任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 ablation系统测projector和数据择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密caption数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间ablation。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统单图、多图、视频。