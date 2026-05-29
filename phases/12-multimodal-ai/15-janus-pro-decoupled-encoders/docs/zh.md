# Janus-Pro:统多模态模型解耦编码器

> 统多模态模型有不可避免张力。理解欲语义特征—SigLIP或DINOv2输出向量富概念级信息。生成欲重构友好code—VQ token compose回锐像素。两目标于单编码器不兼容。Janus (DeepSeek, 2024年10月)和Janus-Pro(DeepSeek, 2025年1月)argue修是停试:解耦两编码器。任务间共享transformer body但路由理解经SigLIP和生成经VQ tokenizer。7B,Janus-Pro GenEval超DALL-E 3而MMMU匹LLaVA。本课读何两编码器工于一失处。

**类型:** 构建
**语言:** Python(stdlib,dual-encoder routing + shared-body signal)
**前置要求:** 阶段12课程13(Transfusion),阶段12课程14(Show-o)
**时间:** ~120分钟

## 学习目标

- 释何单共享编码器妥协理解或生成质量。
- 述Janus-Pro路由:理解输入侧SigLIP特征,生成输入和输出侧VQ token。
- Trace使Janus-Pro成功而Janus不数据混scaling。
- 比解耦(Janus-Pro)、耦合连续(Transfusion)和耦合离散(Show-o)架构。

## 问题背景

统模型跨理解和生成共享transformer body。前尝试(Chameleon、Show-o、Transfusion)都用一视觉tokenizer两方向。Tokenizer是妥协:

- 优化重构(生成):VQ-VAE捕获细像素细节但产弱语义连贯token。
- 优化语义(理解):SigLIP embedding group"猫"图像近"猫"token但不许好重构。

Show-o和Transfusion为此付于一方向显质量税。Janus-Pro问:何需一tokenizer当任务有异需?

## 概念讲解

### 解耦视觉编码

Janus-Pro架构分离两编码器:

- 理解路径。输入图像→SigLIP-SO400m→2-layer MLP→transformer body。
- 生成路径。输入图像(若conditioning于现图像)→VQ tokenizer→token IDs→transformer body。
- 输出生成。Transformer预测图像token→VQ解码器→像素。

Transformer body共享。Body上游和下游任务特定。

输入由提示格式disambiguate:`<understand>`标签路由经SigLIP;`<generate>`路由经VQ。或路由从任务隐式。

### 何此工

理解loss得SigLIP特征,CLIP类预训已调语义相似。模型感知benchmark于Show-o/Transfusion改进因输入特征任务更好。

生成loss得VQ token,tokenizer已调重构。图像质量于Show-o改进因VQ code compose回像素干净。

共享transformer body见两输入分布(SigLIP和VQ)并学与两者工。Claim:够数据+够参数,body吸收切换。

### 数据scaling—Janus vs Janus-Pro

Janus(原,arXiv 2410.13848)引解耦但小scale(1.3B参数,限数据)。Janus-Pro(arXiv 2501.17811)scale:

- 7B参数(vs 1.3B)。
- 阶段1(对齐)90M图文对从72M上。
- 阶段2(统)72M从26M上。
- 阶段3加200k图像生成instruction样本。

Upshot:Janus-Pro-7B MMMU匹LLaVA(60.3 vs ~58)和GenEval超DALL-E 3(0.80 vs 0.67)。一开模型,统谱两边竞争。

### JanusFlow—rectified flow变种

JanusFlow(arXiv 2411.07975)换VQ生成路径为rectified-flow生成路径(连续)。Split成SigLIP为理解+rectified-flow为生成。质量天花板升更。架构保持解耦编码器共享body。

### 共享body工作

Transformer body处理统序列但带两输入分布。它工作是:

- 理解:consume SigLIP特征+文token→自回归emit文。
- 生成:consume文token+(可选图像VQ token)→自回归emit图像VQ token。

Body每块无模态特定权重。它是你期望Qwen或Llama内找文类transformer,加两输入适配器。

Interesting,这意Janus-Pro body可从预训LLM初始化。Janus-Pro确从DeepSeek-MoE-7B初始化。择重要:LLM贡献纯from scratch统模型难达推理能。

### 与InternVL-U比

InternVL-U(课程12.10)是2026 follow-up。它合:

- 原生多模预训(InternVL3 backbone)。
- 解耦编码器路由(SigLIP入,VQ +扩散头出)。
- 统理解+生成+编辑。

InternVL-U subsume Janus-Pro架构择入更大框架。解耦编码器念今规模统模型默。

### 限制

解耦编码器加架构复杂。两tokenizer训,两输入路径维,两失模式集。对不需生成产,Janus-Pro over-engineered—择LLaVA族理解模型。

对不需理解产,Janus-Pro overqualified—择Stable Diffusion 3 / Flux模型。

对需两者产,Janus-Pro今是参考开架构。

## 使用

`code/main.py`模拟Janus-Pro路由:

- 两mock编码器:SigLIP类(产256维语义向量)和VQ类(产整数code)。
- 基任务标签择编码器提示路由。
- 共享body(stand-in)理token序列无论何编码器产。
- 从阶段1(对齐)至阶段3(instruction tune)weighted-sample schedule switch。

打3例路由路径:图像QA、T2I、图像编辑。

## 交付成果

本课产`outputs/skill-decoupled-encoder-picker.md`。给欲统生成+理解于前沿ish质量产,它择Janus-Pro、JanusFlow或InternVL-U带具体数据scale荐。

## 练习题

1. Janus-Pro-7B GenEval超DALL-E 3。释何7B开模型可生成匹前沿闭源模型但理解不。

2. 实路由函数:给提示文,分类为`understand`或`generate`。何处理歧义提示如"describe and then sketch"?

3. JanusFlow替VQ路径为rectified flow。Transformer body今输出何,loss何变?

4. 提Janus-Pro架构可处理多一解耦编码器第四任务。例:图像分割(DINO类),深度(MiDaS类)。

5. 读Janus-Pro节4.2数据scaling。何数据阶段对T2I质量增益vs Janus贡献最?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 解耦编码 | "两视觉编码器" | 每方向分离tokenizer或编码器:理解语义,生成重构 |
| 共享body | "一transformer" | 单transformer理任编码器输出;无模态特定权重 |
| SigLIP为理解 | "语义特征" | CLIP族视觉塔提供富概念特征但重构差 |
| VQ为生成 | "重构code" | Vector-quantized token解码干净回像素 |
| JanusFlow | "Rectified-flow变种" | Janus-Pro带连续flow-matching生成头替VQ |
| 路由标签 | "任务标签" | 择输入编码器提示marker(`<understand>`/`<generate>`)

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)