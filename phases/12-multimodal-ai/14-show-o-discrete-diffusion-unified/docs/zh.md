# Show-o和离散扩散统模型

> Transfusion混连续和离散表示。Show-o (Xie et al., 2024年8月)走另路:文token用causal next-token prediction,图像token用MaskGIT类masked离散扩散。两者坐一transformer内带hybrid attention mask。结果一backbone统VQA、文至图、inpainting和混模态生成,每模态一tokenizer,一loss formulation(next-token扩至masked prediction)。本课走Show-o设计—何masked离散扩散是并行、少步图像生成器—并与Transfusion和Emu3比。

**类型:** 学习
**语言:** Python(stdlib,masked-discrete-diffusion sampler)
**前置要求:** 阶段12课程13(Transfusion)
**时间:** ~120分钟

## 学习目标

- 释masked离散扩散:均匀mask token后问transformer恢复它们调度。
- 比并行图像解码(Show-o、MaskGIT)与自回归图像解码(Chameleon、Emu3)于速度和质量。
- 命Show-o一checkpoint理三任务:T2I、VQA、图像inpainting。
- 择mask调度(cosine、linear、truncated)并推理其对样本质量效。

## 问题背景

Transfusion两loss训工但有更棘动态—连续扩散loss于离散NTP loss不同数值scale。Balancing loss权重是hyperparameter search。架构效但复杂。

Show-o答:保持两模态离散(如Chameleon),但经masked离散扩散而非sequential生成图像。训目标成单masked-token-prediction自然泛化next-token-prediction。

## 概念讲解

### Masked离散扩散(MaskGIT)

原Chang et al. (2022) MaskGIT技优雅。从全masked图像(每token是特殊`<MASK>` id)开始。每步,并行预测所有masked token,后保top-K最置信预测并re-mask余。~8-16迭代后,全token填。每步unmask何token调度调—cosine调度工好。

训简:从[0,1]均匀采样mask比例,应用于图像VQ token,训transformer恢复masked。Exactly BERT为文做,scale至图像生成。

### Show-o:一transformer,hybrid mask

Show-o把MaskGIT放入causal语言模型transformer。Attention mask是:

- 文token:causal(标准LLM)。
- 图像token:图像块内全bidirectional(使masked token预测时可每他图像token见)。
- 文至图:文attend前图像,图像attend前文。

训交替:
1. 文序列标准NTP。
2. T2I样本:文→图像带masked图像token,masked-token-prediction loss。
3. VQA样本:图像→文带masked文token(实仅NTP)。

统loss是`<MASK>` token上交叉熵,覆盖文NTP(仅最后token"masked")和图像masked扩散(随机subset masked)。

### 并行采样

Show-o于~16步生成图像而非~1000(每token自回归)或~20(扩散)。每步,并行预测所有masked token;commit top-K置信;重复。

比:
- Chameleon / Emu3(token上自回归):N_tokens forward pass,典型每图像1024-4096。
- Transfusion(连续扩散):~20步,每步全transformer pass。
- Show-o(masked离散扩散):~16步,每步全transformer pass。

Show-o同scale模型快于Chameleon,大致匹配Transfusion步数带低每步成本(离散vocab logits vs连续MSE loss)。

### 一checkpoint任务

Show-o推理支持四任务,提示格式择:

- 文生成:标准自回归文输出。
- VQA:图像入,文出。
- T2I:文入,图像出经masked离散扩散。
- Inpainting:图像带些token masked,填。

Inpainting能从masked prediction训免费来。Mask VQ-token grid区,feed余加文提示,预测masked token。

### Mask调度

每步unmask何token调度形质量。Show-o荐cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

步0,全token masked(比例1.0)。步T,无masked。Cosine集中mass于中range比例预测最informative处。Linear调度也工但plateau快。

### Show-o2

Show-o2 (2025 follow-up, arXiv 2506.15564) scale Show-o:更大LLM基,好tokenizer,改进mask调度。同架构模式。

### Show-o坐处

2026 taxonomy:

- 离散token + NTP:Chameleon、Emu3。简但推理慢。
- 离散token + masked扩散:Show-o、MaskGIT、LlamaGen、Muse。并行采样,仍tokenizer有损。
- 连续+扩散:Transfusion、MMDiT、DiT。最高质量,更复杂训。
- VLM内连续+flow matching:JanusFlow、InternVL-U。最新。

按任务择:Show-o当你欲一开模型T2I+inpainting+VQA带合理速度;Transfusion当质量paramount你可负担两loss plumbing。

## 使用

`code/main.py`模拟Show-o采样:

- 16 VQ token toy grid。
- Mock"transformer"基于提示和当前unmasked token预测logits。
- 8步cosine调度并行masked采样。
- 打中间态(mask pattern evolution)和终token。

跑它,观mask步步dissolve。

## 交付成果

本课产`outputs/skill-unified-gen-model-picker.md`。给需理解(VQA、captioning)和生成(T2I、inpainting)带开权重约束产,择Show-o族、Transfusion/MMDiT族和Emu3/Chameleon族带具体trade-off。

## 练习题

1. Masked离散扩散~16步采样。何不1?步0若你unmask一切何破?

2. Inpainting masked扩散免费。提产用例(真或假)Show-o inpainting超专家模型。

3. Cosine调度vs linear调度:trace T=8每步unmasked token数。何更balance?

4. 512x512 Show-o图像1024 token。Vocab K=16384,模型emit 1024*log2(16384)=14,336 bits(~1.75 KiB)数据。Stable Diffusion输出512*512*24 bits = 6,291,456 bits(~768 KiB)原始像素。压缩比何和质量何买?

5. 读LlamaGen(arXiv:2406.06525)。LlamaGen类条件自回归图像模型何与Show-o masked方法异?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Masked离散扩散 | "MaskGIT类" | 训预测masked token;推理,迭代unmask最置信预测 |
| Cosine调度 | "Unmask调度" | 推理步mask比例decay;集中置信长于中range |
| 并行解码 | "全token同时" | 每步一forward pass预测全masked token序列,后commit top-K |
| Hybrid attention | "Causal + bidirectional" | 文token上causal和图像块内bidirectional mask |
| Inpainting | "填生成" | 条件于带些token masked图像,预测缺失;训目标免费来 |
| Commitment rate | "每步Top-K" | 每迭代声"done"何token;控推理vs质量trade-off |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)