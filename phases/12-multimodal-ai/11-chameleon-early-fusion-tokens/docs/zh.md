# Chameleon和早融合Token-Only多模态模型

> 我们至今见每VLM保持图像和文分离。视觉token从视觉编码器来,流入projector,后LLM内遇文。视觉和文词汇永不overlap。Chameleon (Meta, 2024年5月)问:若它们何如?训VQ-VAE转图像成共享词汇离散token序列。每多模态文档今是一序列—文token和图像token interleaved,单自回归loss。副作用:模型可生混模态输出—单推理调用交替文和图像token。本课读早融合论点并端到端建toy版。

**类型:** 构建
**语言:** Python(stdlib,VQ-VAE tokenizer + interleaved decoder)
**前置要求:** 阶段12课程05,阶段8(生成AI)
**时间:** ~180分钟

## 学习目标

- 释何共享词汇+单loss改模型可做。
- 述VQ-VAE何tokenize图像成与transformer next-token目标兼容离散序列。
- 命Chameleon训稳定性技:QK-Norm、dropout placement、LayerNorm ordering。
- 比Chameleon vs BLIP-2 Q-Former方法并述何每是正择。

## 问题背景

适配器基VLM(LLaVA、BLIP-2、Qwen-VL)视文和图像为两异物。文token经`embed(text_token)`;图像经`visual_encoder(image) → projector → ... pseudo_tokens`。模型有两输入路径中途merge。

三后果:

1. LLM仅可consume图像,非emit它们。输出仅文。
2. 混模态文档(文章交替段落和图像)awkward—你于模型外解析多模态输入或chain生成。
3. 分布错配。视觉token和文token居隐藏空间异区,造微妙对齐问题。

Chameleon拒前提:图像仅是共享词汇离散token序列。于interleaved文档训模型,一loss,一自回归解码器,并你免费解锁混模态生成。

## 概念讲解

### VQ-VAE作图像tokenizer

Tokenizer是vector-quantized variational autoencoder。架构:

- 编码器:CNN + ViT映图像至空间特征图,如32x32维256特征。
- Codebook:学词汇K向量(Chameleon用8192),维256。
- 量化:每空间特征,L2距离查最近codebook entry。替连续特征为整数索引。
- 解码器:CNN取量化特征回像素。

训:VAE重构loss + commitment loss + codebook loss。Codebook索引成图像离散alphabet。

Chameleon:一图像成32*32 = 1024 token取自词汇8192。与文token(LLM BPE词汇,如32000)concatenate。终词汇:40192。Transformer见一序列,一loss。

### 共享词汇

Chameleon词汇合文token、图像token和模态separator。每token有单ID。输入embedding layer映每ID至D维隐藏向量。输出投影映隐藏回vocab logits。Softmax择下token,无论模态。

Separator重要:`<image>`和`</image>`标签bracket图像token序列。生成时,若模型emit `<image>`,下游软件知下1024 token是VQ索引送解码器像素渲染。

### 混模态生成

推理是共享词汇next-token prediction。例提示:"Draw a cat and describe it。"Chameleon emit:

```
<image> 4821 1029 2891 ... (1024图像token) </image>
The cat is orange, sitting on a windowsill...
```

模型autonomous择序—它可产图像后文、文后图像或interleave。同解码器,同loss。

比适配器VLM生成仅文。Chameleon reopen模型输出模态问题。

### 训稳定性—QK-Norm、dropout、LayerNorm ordering

早融合训于规模不稳定。Chameleon论文档三技:

- QK-Norm。attention内query和key投影apply LayerNorm,于dot product前。防深度logit magnitude炸。多2024后大模型用。
- Dropout placement。每residual-add后dropout,非仅attention和MLP后。图像token梯度可主导需多regularization。
- LayerNorm ordering。Residual分支上Pre-LN(标准),加最后块skip connection额外LN。稳终层梯度流。

无这些技,34B参数Chameleon训多checkpoint diverge。有它们,它收敛。训配方贡献多于架构。

### Tokenizer重构天花板

VQ-VAE是有损。8192 codebook entry和每512x512图像1024 token,重构PSNR顶约26-28 dB。这够可识别图像生成但显坏于连续空间扩散(Stable Diffusion 3达32+ dB)。

Tokenizer是瓶颈。好tokenizer(MAGVIT-v2、IBQ、SBER-MoVQGAN)升天花板。Emu3(课程12.12)仅经更好tokenizer达SDXL质量生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon(早融合,共享vocab):
- 一loss,一解码器。
- 生混模态输出。
- Tokenizer是质量天花板。
- 贵:推理路径每生成图像VQ-VAE解码器。

BLIP-2 / LLaVA(晚融合,分离塔):
- 视觉入,文出仅。
- 重用预训LLM。
- 理解无tokenizer瓶颈。
- 便宜:单forward pass。

按任务择。若需图像生成,Chameleon族。若仅需理解,适配器-VLM简且重用多预训算。

### Fuyu和AnyGPT

Fuyu (Adept, 2023)是相关方法:完全跳分离视觉编码器,feed原始图像patch经LLM输入投影如它们是token,无tokenizer。简于Chameleon,失共享vocab输出生成。

AnyGPT (Zhan et al., 2024)扩Chameleon至四模态:文、图像、speech、music。每同VQ-VAE技,共享transformer。任到任生成。课程12.16多覆盖。

## 使用

`code/main.py`建toy端到端早融合模型:

- Tiny VQ-VAE类quantizer映8x8 patch至codebook索引(K=16)。
- 共享词汇(文id 0..31)+(图像id 32..47)+(separator 48,49)。
- Toy自回归解码器(bigram表)于合caption +图像token序列训。
- 采样loop交替emit文+图像token给提示。

代码意保transformer tiny(bigram)使你可端到端trace信号流。

## 交付成果

本课产`outputs/skill-tokenizer-vs-adapter-picker.md`。给产规(仅理解vs理解+生成、需图像质量、成本预算),它择Chameleon族(早融合)vs LLaVA族(晚融合)并定量规则理。

## 练习题

1. Chameleon用K=8192 codebook entry和每512x512图像1024 token。估vs 24-bit RGB图像压缩比。有损?何有损?

2. 4K图像(3840x2160)同VQ-VAE密度产何图像token?Chameleon类模型可单推理调用生成4K图像?何先破—context、tokenizer质量或KV cache?

3. 纯Python实QK-Norm。给64维query和key,示LayerNorm前后dot product。何深度magnitude控重要?

4. 读Chameleon节2.3训稳定性。述论文于34B无QK-Norm观察确切失模式。"norm炸"signature何?

5. 扩toy解码器emit混模态响应给纯文提示。测训数据分布60%文先/40%图像先模型何频择图像先vs文先。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 早融合 | "统token" | 图像从步一转成共享transformer词汇离散token |
| VQ-VAE | "图像tokenizer" | CNN + ViT + codebook映图像至transformer可预测整数索引 |
| 共享词汇 | "一字典" | 覆文+图像+模态separator单token ID空间 |
| QK-Norm | "Attention稳" | query和keydot product前apply LayerNorm,防norm炸 |
| 混模态生成 | "文+图像输出" | 单pass推理autonomous产interleaved文和图像token |
| Codebook大小 | "K entry" | VQ-VAE可量化离散向量数;trade压缩为保真 |
| Tokenizer天花板 | "重构限" | 解码VQ token可达最佳PSNR;bound模型图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)