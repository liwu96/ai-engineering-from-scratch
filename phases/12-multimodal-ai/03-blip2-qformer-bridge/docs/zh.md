# 从CLIP至BLIP-2 — Q-Former作模态桥

> CLIP对齐图和文但不能生成caption、答问题或持对话。BLIP-2 (Salesforce, 2023)解此带小可训桥:32可学query向量经cross-attention于冻ViT特征上attend,后slot直入冻LLM输入流。188M参数桥连11B LLM至ViT-g/14。每2026 adapter基VLM—MiniGPT-4、InstructBLIP、LLaVA cousins—是后代。本课读Q-Former架构、释两阶段训并建toy版feed视觉token入冻文解码器。

**类型:** 构建
**语言:** Python(stdlib,cross-attention + learnable-query demo)
**前置要求:** 阶段12课程02(CLIP),阶段7(Transformers)
**时间:** ~180分钟

## 学习目标

- 释何冻视觉编码器和冻LLM间可训瓶颈于成本和稳定胜端到端微调。
- 实固定可学query集attend外图像特征cross-attention块。
- 走BLIP-2两阶段预训:表示(ITC + ITM + ITG)后生成(冻解码器LM loss)。
- 比Q-Former至LLaVA用更简MLP projector并议何择胜。

## 问题背景

你有冻ViT产每图256 patch token维1408。你有冻7B LLM期token embedding维4096。显桥—1408至4096线性层—工,但feed全256 patch token入LLM context每图耗256额外token。32图batch视觉模态独耗8192 token。

BLIP-2问题:可你压缩256-token图像表示至远少token(如32)保够信息使LLM caption、答问题和图像推理?并可你训此桥无触冻backbones,保持训成本仅桥参数?

答:Q-Former。32可学"query"向量cross-attend至ViT patch token,产LLM消费32-token视觉总结。188M参数总。于触LLM前用对比、匹配和生成目标训。

## 概念讲解

### 可学query

Q-Former核心技:非让LLM文token attend图像patch,引入新32可学query向量`Q`并让*它们*attend图像patch。Query是模型参数—训时学和每图像用同32 query。

Cross-attention后,每query持图像压缩总结—"述主对象"、"述背景"、"数对象"等。Query非字面语义标签专;它们学何编码使下游loss降。

### 架构

Q-Former是小transformer(12层,~100M参数)带两路径:

1. Query路径:32 query向量流经self-attention(彼此间),后cross-attention于冻ViT patch token,后FFN。
2. 文路径:BERT类文编码器与query路径共享self-attention和FFN权重。文路径cross-attention禁。

训时两路径跑。Query和文经共享self-attention交互,使query可文条件于需它任务(ITM, ITG)。推理时VLM handoff,仅query流经,产32视觉token。

### 两阶段训

BLIP-2两阶段预训:

阶段1:表示学习(无LLM)。三loss:
- ITC(图文对比):pool query token和文CLS token间CLIP类对比。
- ITM(图文匹配):二元分类器—此图文配对match?Hard-negative mined。
- ITG(图接地文生成):文上因果LM head,条件于query。强query编码文可生成内容。

仅Q-Former训。ViT冻。无LLM涉及。

阶段2:生成学习。附冻LLM(OPT-2.7B或Flan-T5-XL等)。经小线性层投32 query输出至LLM embedding维。Prepend它们至文提示。于拼接提示+图+caption序列LM loss训仅线性投影和Q-Former。

阶段2后,Q-Former+投影是全视觉适配器。推理:图像→ViT→Q-Former→线性proj→prepend至文→冻LLM出输出。

### 参数经济

BLIP-2 ViT-g/14 (1.1B冻) + OPT-6.7B (6.7B冻) + Q-Former (188M训) = 8B总,188M训。Q-Former独是全栈参数~2.4%。训成本反映此: handful A100上天vs端到端周。

质量:BLIP-2零VQA匹或超Flamingo-80B于50x小。桥工。

### InstructBLIP和instruction-aware Q-Former

InstructBLIP (2023)扩Q-Former带额外输入:instruction文己。Cross-attention时,query今可图像patch和instruction。Query可每instruction专("数车","述氛围")而非学单固定总结。Benchmark增益于held-out任务。

### MiniGPT-4和projector-only方法

MiniGPT-4保Q-Former但仅训输出线性投影冻余。便宜,但成本是质量—query是BLIP-2非你。快迭代好,非最佳架构。

### 何LLaVA择更简

LLaVA (2023,课程12.05)替Q-Former为plain 2-layer MLP投每ViT patch token入LLM空间—24x24网格每图576 token,全feed LLM。更坏压缩但让LLM attend raw patch。时此争议;2023末主导因视觉instruction数据(LLaVA-Instruct-150k)证MLP可训保够信号。权衡:LLaVA context填快,但自然scale至多图和视频。

于2026域分:Q-Former存于token预算重要(长视频,多图);MLP projector主导于每token质量优先。

### Gated cross-attention:Flamingo祖先

Flamingo(课程12.04)早BLIP-2用同cross-attention念但于每冻LLM层,非单桥。BLIP-2示你可仅压缩至输入层仍工。Gemini和Idefics合两者:interleaved输入token加可选gated cross-attention为in-context few-shot。

### 2026后代

- Q-Former:BLIP-2、InstructBLIP、MiniGPT-4和为token预算多视频语言模型。
- Perceiver resampler:Flamingo变种(课程12.04);Idefics族、Eagle、OmniMAE。
- MLP projector:LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool:VILA、PaliGemma。

四都有效。决问题是受限token预算或每token质量。

## 使用

`code/main.py`建stdlib Q-Former类cross-attention:

1. 模拟256图像patch token(维128)。
2. 例化32可学query(维128)。
3. 跑scaled-dot-product cross-attention(Q从query,K/V从patch)。
4. 经线性层投至LLM维(512)。
5. 出32 LLM-ready视觉token。

纯Python全数学(向量嵌循环)。Toy但正形。Attention-weight矩阵印使你可何patch每query拉。

## 交付成果

本课产`outputs/skill-modality-bridge-picker.md`。给目标VLM配(视觉编码器token数、LLM context预算、部署约束、质量目标),它荐Q-Former vs MLP vs Perceiver resampler带短理和每桥参数量估计。

## 练习题

1. PyTorch实cross-attention块。验32 query和256 key/value,attention-weight矩阵32 x 256和每行softmax后sum至1。

2. BLIP-2阶段1 Q-Former跑三loss同时:ITC、ITM、ITG。写每forward签名伪代码。何需文编码器路径活跃?

3. 比参数量:Q-Former(12层,768 hidden)vs 2-layer MLP projector(1408 → 4096,两层)。何LLM scale 188M Q-Former成本训效率付回?

4. 读BLIP-2论文(arXiv:2301.12597)节3.2何Q-Former初始化。释何从BERT-base(非随机)初始化加速收敛。

5. 10分钟视频1 FPS采样至60帧,计每帧token成本(Q-Former→32 token/帧)vs(MLP projector→576 token/帧)。何入128k-token LLM context窗口?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Q-Former | "Querying transformer" | 32可学query向量cross-attend冻ViT特征小transformer |
| 可学query | "视觉软提示" | cross-attention query侧固定参数集;每模型学,所有输入共享 |
| Cross-attention | "Q从此,K/V从他" | Query、key、value来自异源attention;何query从ViT patch拉 |
| ITC | "图文对比" | CLIP类loss于Q-Former pooled query vs 文CLS |
| ITM | "图文匹配" | Hard-negative mined对上二元分类器;强query判细粒度错配 |
| ITG | "图接地文生成" | 文生成条件于query因果LM loss;强query编码文可解码内容 |
| 两阶段预训 | "表示后生成" | 阶段1训Q-Former独(ITC/ITM/ITG);阶段2附冻LLM训仅投影+Q-Former |
| 冻backbone | "不微调" | 视觉编码器和LLM权重固定;仅桥训 |
| 投影头 | "线性至LLM维" | 映Q-Former输出至LLM embedding维终线性层 |
| Perceiver resampler | "Flamingo版" | 类似可学query cross-attention,Flamingo用于每层而非单桥 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — ITC/ITM/ITG trio前驱。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — 阶段1训概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — instruction-aware Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — projector-only方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学query cross-attention通用架构。