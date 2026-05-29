# InternVL3:原生多模态预训练

> InternVL3前每开VLM随同三步配方:取文LLM万亿文token训,附接视觉编码器,后微调缝。这工但有alignment debt—文LLM全预训预算于纯文和不原生理视觉token。当你后加视觉,LLM需重学何关视觉输入与文推理无忘文。InternVL3 (Zhu et al., 2025年4月)拒后加方法:一预训run,文和多模态interleaved从一步开始。结果78B参数匹Gemini 2.5 Pro于MMMU-Pro开。本课读原生预训案和何改当你做它。

**类型:** 学习
**语言:** Python(stdlib,training-corpus mixer)
**前置要求:** 阶段12课程05,阶段12课程07(配方)
**时间:** ~120分钟

## 学习目标

- 释何后加VLM训accumulates alignment debt,引用三可测症状(catastrophic forgetting、answer drift、视觉文不一致)。
- 述InternVL3原生预训语料混和何文:interleaved:caption比重要。
- 比V2PE(变视觉位置编码)与Qwen2-VL M-RoPE。
- 命Visual Resolution Router(ViR)和Decoupled Vision-Language(DvD)部署优化。

## 问题背景

后加VLM训是默。LLaVA、BLIP-2、Qwen-VL、Idefics—all取已预训LLM(Llama、Vicuna、Qwen、Mistral)并加视觉。训阶段典型看:

1. 冻LLM + 冻视觉编码器 + 可训projector,于caption对训align embedding。
2. Unfreeze LLM,于instruction数据训(LLaVA-Instruct、ShareGPT4V)。
3. 可选任务特定微调。

Alignment debt三症状现:

- Catastrophic forgetting。后加VLM忘纯文技能。GSM8K分降5-10点。Hellaswag分降。纯文代理regress。
- Answer drift。同视觉问题小措辞异答。视觉编码器连接LLM绑定弱于LLM己token。
- 视觉文不一致。VLM可正确述图像后答问题矛盾己述。视觉token不与LLM内一致性检查同文参与方式。

这些症状有充分的文献记录。MM1.5节4量化。LLaVA-OneVision ablation暗示。原生预训是答。

## 概念讲解

### 原生多模态预训练

InternVL3从scratch于原生多模态从一步语料训。混是:

- 40%纯文数据(FineWeb、Proof-Pile-2等)。
- 35% interleaved图文数据(OBELICS、MMC4类)。
- 20%配图文caption数据。
- 5%视频文数据。

视觉token、文token和跨模交互都从首梯度步参与同loss。无对齐预训、无projector冻结阶段、无catastrophic forgetting recover。

训是base模型单阶段。Instruction tuning随,但base模型已理视觉token为一等公民。

### V2PE(变视觉位置编码)

Qwen2-VL用固定轴分配M-RoPE。InternVL3引V2PE:位置编码每模态类(文、图像、视频)异带可学scaling。实践:

- 文token得1D位置(文索引)。
- 图像patch得2D位置(行、列)。
- 视频帧得3D位置(时、行、列)。

三共享同RoPE频率基,但每band隐藏维分配是学参数而非固定split。预训期间trade off时vs空间频分辨率自由。

V2PE ablation claim:同算力视频benchmark M-RoPE上1-2点。非革命,但clean。

### Visual Resolution Router(ViR)

部署优化。非所有图像需全分辨率编码。一对象低细节照片于1280px原生编码浪费token。ViR是小分类器预编码前预测答问题需最小分辨率。

路由有三tier:低分辨率(256 token)、中(576)、高(2048+)。产流量60%查询低或中够。净效:等质量2-3x吞吐。

### Decoupled Vision-Language deployment(DvD)

当你服大VLM,视觉编码器每图像跑一次但LLM每输出token自回归跑。两组件有不同瓶颈(视觉=GPU memory bandwidth conv + attention;LLM=KV cache)。DvD分它们至带streaming间分离GPU。

8B + 400M编码器模型,DvD大致double每节点吞吐vs同位置。

### 单阶段vs多阶段质量

InternVL3主benchmark claim:78B参数,匹Gemini 2.5 Pro MMMU-Pro。38B,匹GPT-4o。8B,领开-8B leaderboard。全于单阶段预训+instruction-tune配方。

Alignment-debt假设可测:InternVL3-8B失少文benchmark点(MMLU、GSM8K)于Qwen2.5-VL-7B每单位视觉benchmark增益。模型更是generalist因训是一块非两。

### InternVL3.5和InternVL-U

InternVL3.5(2025年8月)scale配方。同原生预训方法,多数据,多参数。MMMU改进incremental。

InternVL-U(2026)加统生成—图像输出经同backbone上MMDiT heads。"U"代表"Understanding + generation"(理解+生成),追Transfusion类统模型(课程12.13)。同原生预训backbone支持理解和生成heads。

### 原生预训练权衡

原生预训练非免费:

- 算力。从scratch训新VLM成本同训文LLM—百万GPU-hour。后加适应重用现LLM权重,省大多成本。
- 数据。规模interleaved图文语料稀。OBELICS是141M文档;MMC4是571M。文alone发于15T token。多模预训数据稀缺是硬约束。
- Base-LLM重用。原生预训练放弃后drop新LLM选项。后加让你换Llama-3.1为Llama-4仅重训adapter。

InternVL3赌:alignment debt坏于重用失。Benchmark backs claim。Cost-to-produce bars未来实验室便宜复现。后加VLM将存因它们对多项目仍便宜。

## 使用

`code/main.py`是training-corpus mixer和ViR router模拟器。它:

- 取目标语料混(%text,%interleaved,%caption,%video)并算每模态期望步。
- 于batch查询模拟ViR路由(分布:50%低细节、30%中、20%高细节)并报平均token数。
- 给编码器vs LLM FLOPs报DvD吞吐估计。
- 打参数、算力、数据和期望alignment-debt症状后加vs原生预训并排。

## 交付成果

本课产`outputs/skill-native-vs-posthoc-auditor.md`。给拟VLM训计划,它audit是否去原生或后加,flag alignment-debt风险,并荐语料混。于你sizing新开VLM项目并择训策略时用它。

## 练习题

1. 估InternVL3-8B(原生预训)和LLaVA-OneVision-7B(后加)算力缝。GPU-hour比例大约?何释缝?

2. InternVL3报40%文/35% interleaved/20% caption/5%视频。若你目标任务是视频重,提新比并argue何base模型仍需实质文和caption数据。

3. 读MM1.5节4 forgetting。命后加训示最大regression确benchmark。Regression成本何?

4. ViR路由60%流量至低分辨率编码。何类查询它misroute(当高分辨率需时发至低分辨率)?提三router失模式。

5. DvD分视觉和LLM至分离GPU。何流量模式DvD hurt吞吐而非help?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 原生多模预训 | "从scratch一起" | 文+图像+视频token从步1参与loss,非后bolt |
| Alignment debt | "后加惩罚" | Bolt vision上冻LLM带来可测文技能和答一致性regression |
| V2PE | "变视觉位置编码" | 每模态可学位置编码分配;InternVL3 M-RoPE successor |
| ViR | "分辨率路由" | 编码前每query择最小需分辨率小分类器,省推理token |
| DvD | "解耦部署" | 视觉编码器一GPU,LLM另一,带流handoff;大VLM吞吐double |
| InternVL-U | "统理解+生成" | 2026 follow-up加图像生成head至原生预训backbone |
| Interleaved语料 | "OBELICS / MMC4" | 自然读序图文文档;原生预训原料 |

## 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)