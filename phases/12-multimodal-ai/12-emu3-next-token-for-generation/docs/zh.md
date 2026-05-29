# Emu3:图像和视频生成Next-Token Prediction

> BAAI Emu3 (Wang et al., 2024年9月)是2024应结束扩散vs自回归辩论结果。单Llama类decoder-only transformer,仅next-token-prediction目标训,跨文+VQ图像token+3D VQ视频token统词汇,图像生成超SDXL和感知超LLaVA-1.6。无CLIP loss。无扩散调度。推理用classifier-free guidance为质量,但核心训目标是teacher forcing next-token prediction。发于Nature。本课读Emu3论点—为何更好tokenizer加scale是所需全—并与扩散方法比。

**类型:** 学习
**语言:** Python(stdlib,3D视频tokenizer数学 + autoregressive sampler骨架)
**前置要求:** 阶段12课程11(Chameleon)
**时间:** ~120分钟

## 学习目标

- 释何Emu3单loss next-token目标工尽管长持假设图像质量需扩散。
- 述3D视频tokenizer:spatiotemporal VQ codebook看何,何patch跨时间。
- 比Emu3 vs Stable Diffusion XL于(训算、推理成本、质量天花板)。
- 命同Emu3模型三角:Emu3-Gen(图像生成)、Emu3-Chat(感知)、Emu3-Stage2(视频生成)。

## 问题背景

2024通识:图像生成需扩散。论点:离散图像token失多信息重构细节,自回归采样千token累误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney都用某种扩散。Chameleon(课程12.11)小规模部分证伪了这一假设但质量未匹SDXL。

Emu3直攻论点。Claim:好视觉tokenizer+够scale+next-token loss=同模型扩散超图像生成亦做感知。

赌发时争议。两年后,开源统生成族(Emu3、Show-o、Janus-Pro、Transfusion)是研究默路径;产前沿模型现用某变种。

## 概念讲解

### Emu3 tokenizer

关键是视觉tokenizer。Emu3训定制IBQ类tokenizer(Inverse Bottleneck Quantizer, SBER-MoVQGAN族)于每token 8x8分辨率减。512x512图像成64x64 = 4096 token codebook大小32768。

此大於Chameleon每512x512 1024 token K=8192但每token便宜(小codebook lookup,简codec)。关键指标:重构PSNR于30.5 dB,与Stable Diffusion连续latent空间32 dB竞争。

视频:3D VQ tokenizer编码spatiotemporal patch(4x4x4像素)至一整数。4s clip 8 FPS有32帧;256x256带4x空间和4x时减,token数(256/4)*(256/4)*(32/4)=64*64*8=32,768 token。

Tokenizer质量是天花板。Emu3贡献部分是"我们训了非常好tokenizer。"

### 单loss训练

Emu3用一目标:跨文token、2D图像token和3D视频token共享词汇next-token prediction。权重训时乘模态特定因子balance贡献,但loss函数完全同。

训混:
- 图像生成:`<文caption> <image> 图像token </image>`
- 图像感知:`<image> 图像token </image> <question> 文token`
- 视频生成:`<文caption> <video> 视频token </video>`
- 视频感知:类似。
- 纯文:标准NTP。

模型从数据分布学何时emit图像token vs文token。生成从模型`<image>`标签后预测图像token涌现。

### Classifier-free guidance和temperature

自回归图像生成推理时classifier-free guidance(CFG)好很多。Emu3用它:生成两次,一次全caption,一次空caption,混logits带guidance weight(典型3.0-7.0)。这与扩散用CFG技同,借至自回归设置。

Temperature重要:太高,artifacts;太低,mode collapse。Emu3荐temperature感知1.0,图像生成0.8。

### 三角,一模型

Emu3发为三功能异API但一底层权重集:

- Emu3-Gen。图像生成。输入文,输出图像token。
- Emu3-Chat。VQA和captioning。输入图像(token),输出文。
- Emu3-Stage2。视频生成和视频VQA。输入文或视频,输出文或视频。

无任务特定头。仅异提示模板。同checkpoint。

### Benchmark

Emu3论文(2024年9月):

- 图像生成:MJHQ-30K FID超SDXL(5.4 vs 5.6),GenEval overall(0.54 vs 0.55—统计tie),Deep-Eval composite on-par。
- 图像感知:VQAv2超LLaVA-1.6(75.1 vs 72.4),MMMU大致匹。
- 视频生成:4秒clip质量FVD与Sora时代公开benchmark模型竞争。

数非总胜—Emu3 trade这里点那里点—但"next-token prediction是所需全"claim跨模态可捍卫。

### 算力成本

Emu3于约300 billion多模token训7B参数模型。GPU-hour大致可比Llama-2-7B预训(A100类硅2k-4k GPU-year)。扩散模型如Stable Diffusion 3于类似预算训但需分离文编码器和更复杂管道。

推理,Emu3每图像慢于SDXL:4096图像token 30 tok/s是每512x512图像~2分钟,vs SDXL 2-5秒。Speculative decoding和KV cache optimization缩缝但不关。自回归图像生成算重;这是长期存在的权衡。

### 何重要

Emu3深贡献是概念。若next-token prediction scale至匹扩散图像生成,统模型路径(一loss,一backbone,任模态)可行。未来模型不需分离文编码器、分离扩散scheduler、分离VAE。一transformer,每模态一tokenizer,scale。

Show-o、Janus-Pro和InternVL-U都建上或挑战此论。中国实验室(BAAI、DeepSeek)于2025通过比美国实验室更激进于这方向发。

## 使用

`code/main.py`建两toy piece:

- 2D vs 3D VQ tokenizer count calculator:给(resolution, patch, clip_length, FPS),算图像vs视频token数。
- Autoregressive图像token sampler带classifier-free guidance于temperature。

CFG实匹配Emu3配方—混条件和无条件logits带guidance weight。

## 交付成果

本课产`outputs/skill-token-gen-cost-analyzer.md`。给生成产规(图像或视频,目标分辨率,质量tier,延迟预算),它算token数、推理成本并择Emu3族vs扩散。

## 练习题

1. Emu3每512x512图像于8x8减产4096 token。计1024x1024和2048x2048等价。推理延迟何?

2. 读Emu3节3.3视频tokenizer。述3D VQ patch形和何是4x4x4非8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0:何视觉效?Trace `code/main.py`数学。

4. 计Emu3-7B于300B token训FLOPs并与Stable Diffusion 3比。何更贵训?

5. Emu3于FID超SDXL但VQAv2未超专VLM。释何统loss方法于异benchmark示异强度vs专家。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Next-token prediction | "NTP" | 标准自回归loss:给token[0..i]预测token[i+1];tokenized时适每模态 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | VQ-VAE类大codebook(32768+)和好于Chameleon重构 |
| 3D VQ | "Spatiotemporal quantizer" | (时、行、列)索引codebook;一token覆4x4x4像素立方 |
| Classifier-free guidance | "CFG" | 条件和无条件logits混带权重gamma;推理boost图像质量 |
| 统词汇 | "共享token" | 文+图像+视频都从同一整数空间取;模型预测何模态下次来 |
| MJHQ-30K | "图像生成benchmark" | Midjourney质量30k提示benchmark;Emu3于此报FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)