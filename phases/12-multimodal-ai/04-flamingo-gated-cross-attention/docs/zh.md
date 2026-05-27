# Flamingo和Few-Shot VLMs Gated Cross-Attention

> DeepMind Flamingo (2022)于任何人前做两事。示单模型可理任意interleaved图像、视频和文序列。示VLM可in-context学—给few-shot提示带三例(图,caption)对和模型caption新图无任何梯度步。机制:gated cross-attention层,插入冻LLM现有层间,带学tanh gate初零使LLM文能初始化保。本课走Flamingo Perceiver resampler和gated cross-attention架构—Gemini interleaved输入和Idefics2视觉token祖先。

**类型:** 学习
**语言:** Python(stdlib,gated cross-attention + Perceiver resampler demo)
**前置要求:** 阶段12课程03(BLIP-2 Q-Former)
**时间:** ~120分钟

## 学习目标

- 释何gated cross-attention经tanh(gate) = 0初始化保冻LLM文能。
- 走Perceiver resampler:N图像patch → K固定"latent" query经cross-attention。
- 述Flamingo何理interleaved图文序列带重图像位置因果mask。
- 复现few-shot多模态提示结构(3图caption例后query图)。

## 问题背景

BLIP-2 feed 32视觉token入冻LLM输入层。适每提示一图。但若你想feed*多*图interleaved文,如"此图A,caption它;此图B,caption它;今此图C,caption它"?LLM self-attention需理单流图像token和文token,何位置可attend何图问题得棘。

Flamingo答:不改LLM输入流完全。于现有LLM块间插入额外cross-attention层。文token仍流经LLM因果self-attention如常。每几LLM块间,文token也经新gated层cross-attend图像特征。Gate(初零)意步零新层是no-op—模型行为完全像预训练LLM。随训进展gate开和视觉信息开始流。

Flamingo答第二问题:何理每提示变图像数(0,1,或多)?Perceiver resampler—小cross-attention模块取你何数patch产固定数视觉latent token。LLM cross-attention层见同形无论提示何数图。

## 概念讲解

### 冻LLM

Flamingo从冻Chinchilla 70B LLM开始。全70B权重未触。现文self-attention和FFN正常操作。

### Perceiver resampler

每提示图像,ViT产N patch token。Perceiver resampler有K固定可学latent(Flamingo用K=64)。每resampler块是两子步:

1. Cross-attention:K latent attend N patch token(Q从latent,K/V从patch)。
2. Self-attention + FFN于latent内。

6 resampler块后,输出是K=64维1024视觉token,无论ViT产何数patch。224x224图像(196 patch)和480x480图像(900 patch)都出64 resampler token。

视频,resampler时用:每帧patch产64 latent,时位置编码让模型分t=0和t=N。全视频成T * 64视觉token。

### Gated cross-attention

冻LLM每M层间(Flamingo用M=4),插新gated cross-attention块:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha`是可学标量初零。
- `tanh(0) = 0`,故初始化gated支贡零。
- 随`alpha`离零,cross-attention贡平滑长。
- Residual连接意即使全开gate不overwrite LLM文表示;仅上加视觉信息。

这是Flamingo单最重要设计择:视觉条件化是加性、gated、初始化零。Flamingo步0是完美Chinchilla 70B于纯文输入。

### Interleaved输入Masked cross-attention

提示如"<图A> caption A <图B> caption B <图C> ?",每文token应仅见序列前图像。Cross-attention mask强制:位置`t`文token仅attend图像resampler token其图像索引`i < i_t`其中`i_t`是位置`t`前最近图像。"仅见最后前图像"或"见所有前图像"都有效择;Flamingo择前。

### In-context few-shot学习

Flamingo提示看如:

```
<图1> A photo of a cat. <图2> A photo of a dog. <图3> A photo of a
```

模型见完成模式并出"bird"(或图3示何)。无梯度步。冻LLM in-context学习能经gated cross-attention承载—这是论文点睛和何重要。

### 训数据

Flamingo于三数据集训:

1. MultiModal MassiveWeb (M3W):43M interleaved图文web页,重构读序。
2. Image-Text Pairs (ALIGN + LTIP):4.4B对。
3. Video-Text Pairs (VTP):27M短视频clip。

OBELICS (2023)是interleaved web corpus开复现,Idefics、Idefics2和多数"Flamingo类"开模型训于。

### OpenFlamingo和Otter

OpenFlamingo (2023)是开复现。架构同(Perceiver resampler + 冻LLaMA或MPT上gated cross-attention)。Checkpoint于3B、4B、9B。质量因小base LLM和少数据后Flamingo。

Otter (2023)建OpenFlamingo上加MIMIC-IT(多模态instruction数据集)instruction tuning,示gated cross-attention instruction following也工。

### 后代

- Idefics / Idefics2 / Idefics3: Hugging Face gated cross-attention lineage,逐简(Idefics2丢resampler为直接patch token带adaptive pooling)。
- Flamingo-to-Chameleon过渡:2024多队移至early-fusion(课程12.11);Flamingo类gated cross-attention留存于需backbone冻结产。
- Gemini interleaved输入:概念继承Flamingo interleaved格式灵活,虽确切机制proprietary。

### 与BLIP-2比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥 | Q-Former一次于输入 | 每M层Gated cross-attention |
| 视觉token | 每图32 | 每图每cross-attn层64 |
| 冻LLM | 是 | 是 |
| Few-shot in-context | 弱 | 强—论文centerpiece |
| Interleaved输入 | 无原生支持 | 是,设计目标 |
| 训数据 | 130M对 | 1.3B对 + 43M interleaved页 |
| 参数量 | 188M训 | ~10B训(cross-attn层) |
| 算力 | 8 A100上天 | 千TPUv4上周 |

预算上单图VQA择BLIP-2。Interleaved、few-shot或多图推理择Flamingo/Idefics2。

## 使用

`code/main.py`演示:

1. 36假patch token上8可学latent Perceiver resampler(纯Python cross-attention)。
2. Gated cross-attention步`alpha = 0` → 输出等于输入(LLM未改),后`alpha = 2.0` → 视觉贡混入。
3. "(图1)(文1)(图2)(文2)"序列2D attention mask interleaved-mask builder。

## 交付成果

本课产`outputs/skill-gated-bridge-diagnostic.md`。给开VLM配(resampler Y/N, cross-attn频率, gate方案),它识Flamingo lineage元素并释冻结策略。适debug何微调降文能(答:gate得太快太宽)。

## 练习题

1. 计Flamingo-9B视觉参数量:9B LLM + 1.4B gated cross-attention层 + 64M resampler。总参数何分是训?

2. PyTorch实gated residual `y = tanh(alpha) * cross + x`。示实验`alpha=0`,`y==x`初始化完全。

3. 读OpenFlamingo节3.2(arXiv:2308.01390)何他们理batch多图当每提示异图像数。述padding策略。

4. 何Flamingo cross-attention mask让文token attend*仅最近*前图像而非所有前图像?读Flamingo论文节2.4并释权衡。

5. In-context few-shot:构4例"图像 → 主对象颜色"prompt为新Flamingo变种。述期望精度模式当你变例数0至8。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Perceiver resampler | "固定latent cross-attention" | 从变数输入patch产K固定token模块 |
| Gated cross-attention | "Tanh-gated桥" | Residual层`y = tanh(alpha)*cross + x`,可学alpha,初0 |
| Interleaved输入 | "混序列" | 图文按读序自由混提示格式 |
| 冻LLM | "无LLM梯度" | 文LLM权重不更新;仅resampler + cross-attn层训 |
| Few-shot | "In-context例" | 提示中给几(图,答)对;模型无微调泛化 |
| OBELICS | "Interleaved web corpus" | 141M web页图文按读序开数据集 |
| Chinchilla | "70B冻base" | Flamingo冻文LLM,从DeepMind Chinchilla论文 |
| Gate schedule | "alpha何动" | 训期间cross-attention gate开速率 |
| Cross-attn频率 | "每M层" | Gated cross-attention块插入频率;Flamingo用M=4 |
| OpenFlamingo | "开复现" | MosaicML/LAION 3-9B开checkpoint;架构同Flamingo |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — Interleaved web corpus。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用Perceiver架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — Instruction-tuned Flamingo后代。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo方法现代简化。