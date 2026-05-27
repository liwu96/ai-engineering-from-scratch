# 开权重VLM配方:何实重要

> 2024-2026开权重VLM文献是ablation表森林。Apple MM1试13组合图像编码器、connector和数据混。Allen AI Molmo证详人工caption胜GPT-4V蒸馏。Cambrian-1跑20+编码器比。Idefics2形式化五轴设计空间。Prismatic VLMs比27训配方于控benchmark。出所有噪音,小集结果跨论持:图像编码器重要多于connector架构,数据混重要多于任,详人工caption胜蒸馏合数据。本课读那些表使你不须。

**类型:** 学习+实验
**语言:** Python(stdlib,ablation table parser + recipe picker)
**前置要求:** 阶段12课程05(LLaVA基线)
**时间:** ~180分钟

## 学习目标

- 命五轴VLM设计空间:图像编码器、connector、LLM、数据混、分辨率调度。
- 读MM1/Idefics2/Cambrian-1 ablation表并预何knob移给定benchmark。
- 给定算力预算和任务混为新VLM择配方(编码器、connector、数据、分辨率)。
- 释何详人工caption于同token数胜GPT-4V蒸馏。

## 问题背景

百开权重VLM存。多"好"和"state-of-the-art"间缝非架构。是数据、分辨率调度和编码器择。知何knob先转当你模型underperform省你5M GPU-hour错。

2023波(LLaVA-1.5、InstructBLIP、MiniGPT-4)于caption pair预训+LLaVA-Instruct-150k跑。好基线。顶MMMU 35%。

2024波(MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs)跑 exhaustive ablation。结果惊且实。

## 概念讲解

### 五轴设计空间

Idefics2 (Laurençon et al., 2024)命轴:

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器异patch size、分辨率和预训目标。
2. Connector。MLP(2-4层)、Q-Former(32 query + cross-attn)、Perceiver Resampler(64 query)、C-Abstractor(convolutional + bilinear pooling)。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM大小是主导参数成本。
4. 训数据。Caption pair(CC3M, LAION)、interleaved(OBELICS, MMC4)、instruction(LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron)。
5. 分辨率调度。固定224/336/448、AnyRes、原生动态。训ramp或常。

每产VLM每轴择。多MMMU分方差由轴1、4、5释—非何connector你择。

### 轴1:编码器> connector

MM1节3.2示:从CLIP ViT-L/14换SigLIP SO400m/14加3+ MMMU点。Connector从MLP换Perceiver Resampler加少于1点。Idefics2复现:SigLIP > CLIP,同token数Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1"Cambrian Vision Encoders Match-Up"(Tong et al., 2024)于vision-centric benchmark(CV-Bench)跑20+编码器。榜顶是DINOv2和SigLIP混;CLIP中pack;ImageBind和ViT-MAE低。CLIP ViT-L至DINOv2 ViT-g/14缝CV-Bench~5-7点。

2026开VLM默编码器是SigLIP 2 SO400m/14为语义+密特征,有时与DINOv2 ViT-g/14特征concatenate(Cambrian"Spatial Vision Aggregator"做此)。

### 轴2:Connector设计wash

MM1、Idefics2、Prismatic和MM-Interleaved都达同结论:于固定视觉token数,connector架构barely重要。Mean-pooled patch上2-layer MLP于同token预算32-query Q-Former内1点工。

何重要是token数。多视觉token = 多LLM算 = 好性能至点,后diminishing return。64 token每图像OCR太少。576-1024 token是多开VLM甜点。2048+仅帮文档和图表。

Q-Former vs MLP是成本问题非质量问题:Q-Former不管图像分辨率capped 32-64 token;MLP发全patch token。高清输入,Q-Former省LLM context;低分辨率,差是噪。

### 轴3:LLM大小定天花板

LLM从7B至13B加倍MMMU可靠每VLM论文加2-4点。70B你saturation多benchmark。VLM多模推理天花板是LLM文推理天花板—视觉编码器仅feed它,非替它推理。

这是何Qwen2.5-VL-72B和Claude Opus 4.7 crush MMMU-Pro和ScreenSpot-Pro:语言brain巨大。7B VLM不可通过clever connector设计替70B VLM。

### 轴4:数据—详人工caption胜蒸馏

Molmo + PixMo (Deitke et al., 2024)是2024每应读结果。Allen AI使人工标注者1-3分钟密speech-to-text pass述图像,得712K密caption图像。训数据无处GPT-4V蒸馏。

Molmo-72B于11/11 benchmark超Llama-3.2-90B-Vision。缝非架构—是caption质量。详人工caption每图像含5-10x多信息短web caption于GPT-4V蒸馏hallucinate处事实接地。

ShareGPT4V (Chen et al., 2023)和Cauldron(Idefics2)同人+GPT-4V caption同playbook。趋势清:2026前沿,caption密 > caption量 > 蒸馏便利。

### 轡5:分辨率及其调度

Idefics2 ablation:384 → 448加1-2点。448 → 980带图像splitting(AnyRes)OCR benchmark另加3-5。Flat分辨率训中精度plateau;分辨率ramping(224开,448或原生完)训快完高。

Cambrian-1跑分辨率vs token权衡:于固定算力,你可低分辨率多token或高分辨率少token。OCR高分辨率胜;一般场景理解低分辨率多token胜。

2026产配方:阶段1于384固定训,阶段2带动态分辨率至1280 OCR重任务。

### Prismatic控比

Prismatic VLMs (Karamcheti et al., 2024)是控全轴论文。同13B LLM、同instruction数据、同评估—一次仅一轴异。结果:

- 每图像视觉token数释~60%方差。
- 编码器择释~20%。
- Connector架构释~5%。
- 余(数据混、scheduler、LR)剩~15%。

这是粗分解,是文献"何应先ablate"最干净答。

### 2026 picker

给证据,2026新项目默开VLM配方:

- 编码器:SigLIP 2 SO400m/14原生分辨率NaFlex,若需分割/grounding与DINOv2 ViT-g/14 concatenate。
- Connector:Patch token上2-layer MLP。跳Q-Former除非你token constrained。
- LLM:Qwen2.5 / Llama-3.1 / Gemma 2,7B为成本,70B为质量,按目标延迟择。
- 数据:PixMo + ShareGPT4V + Cauldron, topped up任务特定instruction数据。
- 分辨率:动态(min 256, max 1280像素每长边)。
- 调度:阶段1对齐(projector-only),阶段2全微调,阶段3任务特定微调。

每那些默追溯至本课末引论文测ablation。

## 使用

`code/main.py`是ablation table parser和recipe picker。它编码MM1和Idefics2 ablation table(condensed)并让你query:

- "给定预算X和任务Y,何配方胜?"
- "若我7B Llama上SigLIP换CLIP,期望MMMU缝何?"
- "何轴应先ablate为80%置信答?"

输出是排配方列表带期望benchmark缝和"先ablate"荐。

## 交付成果

本课产`outputs/skill-vlm-recipe-picker.md`。给目标任务混、算力预算和延迟目标,它发全配方(编码器、connector、LLM、数据混、分辨率调度)带ablation引用每择。停工程师每新VLM项目始reinventing Idefics2 ablation table。

## 练习题

1. 读MM1节3.2。于固定2B LLM预算50M图像,何编码器胜?13B LLM会答flip?何?

2. Cambrian-1发现DINOv2 + SigLIP concatenate于vision-centric benchmark单超任但MMMU无信号。预何benchmark增益何平。

3. 你目标是2B LLM上移动UI代理。择编码器、connector、分辨率和数据混。用具体ablation table每择理。

4. Molmo发4B和72B模型。4B与闭7B VLM竞争;72B于11/11 benchmark超Llama-3.2-90B-Vision。何告诉你LLM-size plateau假设?

5. 设计ablation table于7B VLM隔离数据混质量vs编码器质量。最少何训run?提四轴设。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Ablation | "转一knob" | 多run训异于精一设计空间轴,持余常 |
| Connector | "桥"/"projector" | 映视觉编码器输出入LLM token空间可训模块(MLP、Q-Former、Perceiver) |
| 详人工caption | "密caption" | 多句人工写描述(典型80-300 token)富于web alt text |
| 蒸馏 | "GPT-4V caption" | 强proprietary VLM生成训数据;便利但易继承hallucination |
| AnyRes / 动态res | "高清路径" | 经tiling或M-RoPE feed大于编码器原生分辨率图像策略 |
| Resolution ramp | "课程" | 低分辨率开增训调度,加速对齐学习 |
| Vision-centric bench | "CV-Bench / BLINK" | 强密视觉感知非语言重推理评估 |
| PixMo | "Molmo数据" | Allen AI 712K密caption图像数据集;人speech转录密caption |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)