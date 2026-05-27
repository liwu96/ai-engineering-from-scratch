# LLaVA-OneVision:单图、多图、视频于一模型

> LLaVA-OneVision前(Li et al., 2024年8月)开VLM界有分离lineage:LLaVA-1.5单图、Mantis和VILA多图模型、Video-LLaVA和Video-LLaMA视频模型。每赢己benchmark败于他。LLaVA-OneVision argue单课程可训一模型主导三场景,且emergent task-transfer效(单图技能export至视频,多图推理export至单图)超专家总和。配方deceptively简:跨场景持常visual-token预算,加显课程从单图移至OneVision(多图)至视频。本课读预算、课程和emergent行为。

**类型:** 构建
**语言:** Python(stdlib,token budget solver + curriculum planner)
**前置要求:** 阶段12课程05(LLaVA),阶段12课程06(any-resolution)
**时间:** ~180分钟

## 学习目标

- 设计跨单图、多图和视频输入持常visual-token预算。
- 序训课程从单图迁至视频无catastrophic forgetting。
- 释何单模型课程正时同参数数胜专家。
- 命LLaVA-OneVision报告三emergent能:多摄推理、set-of-mark提示、iPhone截图代理。

## 问题背景

图像、多图和视频各不同stress模型。

单图欲高分辨率token(AnyRes,~2880视觉token)catch OCR和细细节。每sample预算:一图像,2880 token。

多图欲几中分辨率图像(~576 token每)使跨图像推理fit context。每sample预算:4-8图像,576每,2300-4600 token。

视频欲多帧低分辨率(~196 token每帧pool后)capture时动态。每sample预算:8-32帧,196每,1600-6200 token。

若你训分离模型,你择一预算。若你训一模型,你需预算跨场景scale sensibly无炸context。

OneVision前,默答"训一场景,忽略他。"Video-LLaVA retrofit视频上图像模型带额外训阶段。LLaVA-NeXT加多图支持带tiling。无干净理三者。

## 概念讲解

### OneVision token预算

LLaVA-OneVision择统visual-token预算约3000-4000 token每sample,不同分配每场景:

- 单图:AnyRes-9(3x3 tile + thumbnail),每tile 384 729 patch,激进bilinear pooling 2x2 → 182每tile。总:9 * 182 + 182 = 1820 token。或AnyRes-4于729-per-tile = 2916 + 729。
- 多图:每图像中分辨率(384,无tiling),729 token无pooling。预算6图像 → 4374 token。
- 视频:32帧384分辨率带激进3x3 bilinear pool → 81 token每帧。总:32 * 81 = 2592 token。

分配保大常总token。LLM永不见batch炸context。编码器产不同geometry每场景,但LLM消费同预算。

### 三阶段课程

LLaVA-OneVision三阶段训:

1. 单图SFT(阶段SI)。全数据单图加文。高分辨率AnyRes输入训。此教感知、OCR和细粒理解。用LLaVA-NeXT数据加OneVision特定单图数据。
2. OneVision SFT(阶段OV)。混单图+多图+视频(uniformly sampled frame)。统token预算训。此教模型理异质batch形。无权重reset—从阶段SI续。
3. 任务迁(阶段TT)。续目标任务混,典型多图或视频重depending on产。可选deploy微调。

关键:课程序重要。视频先或多图先训产坏图像性能于单图先,即使同数据。论文显ablate此。

### 何课程工

单图训建感知基。Patch token携细粒视觉特征;LLM学集成它们与文。多图和视频引结构挑战(何图像是何,何发生先)强感知基难学。

若你从scratch一起训所有场景,模型underfit感知(每batch限单图数据)和overfit结构(多多图/视频数据)。结果:模型随跨图像推理模式但视觉浅。

课程序给你阶段SI感知强,后阶段OV组合/时推理,无失任。

### Emergent跨场景技能

LLaVA-OneVision论文报告三emergent能:

1. 多摄推理。多图+视频分开训;推理时,问理多摄驾驶场景。模型正确整合视图尽管训从未见确格式。
2. Set-of-mark提示。用户于图像注带编号mark对象;模型理"mark 3何做relative mark 7。"训从未见mark或注;从空间grounding +多图引用组合学。
3. iPhone截图代理。用户供iPhone屏截图并问plan下次click。训于UI截图、用户工作流视频和多图前后对。泛化至代理用例。

这些非训任务;从课程组合结构涌现。

### 视觉token pooling

Token预算需pooling。OneVision用patch grid上bilinear interpolation:24x24 = 576 patch成12x12 = 144(2x因子)或8x8 = 64(3x因子)。Pooling于patch-grid空间做,非token空间,保局部。

每场景pooling因子择是己hyperparameter。少pooling = 多token = 富表示。多pooling = 少token = 多帧/图像fit。

### LLaVA-OneVision-1.5

2025 follow-up(LLaVA-OneVision-1.5, arXiv 2509.23661)是"全开"训数据、模型权重和代码。某benchmark匹proprietary缝和配方民主化。同课程,多数据,好base LLM。无架构改。

### 与Qwen2.5-VL比

Qwen2.5-VL(课程12.09)做不同择。它用M-RoPE和动态FPS替固定pooling。其预算scale输入—1分钟视频用多token于5秒视频。LLaVA-OneVision定预算scale pooling。两工;它们trade configurability为predictability。

## 使用

`code/main.py`是OneVision类VLM课程和预算planner。给每sample token预算和目标场景混(如40%单图、30%多图、30%视频),它:

- 每场景分分辨率、pooling因子和帧。
- 查每场景fit共享预算内。
- 报期望token数、LLM FLOPs和何场景under-tokenized。
- 打阶段阶段训调度。

用于plan OneVision微调或sanity-check VLM deploy每请求成本。

## 交付成果

本课产`outputs/skill-onevision-budget-planner.md`。给目标任务分布和每sample预算,它发AnyRes因子、每帧pooling、视频帧数和课程阶段权重。于你训或微调统场景VLM时用此。

## 练习题

1. 你产支持80%单图、10%多图(2-4图像)、10%视频(8-16帧)。设计token预算。何你放从不做重多图省额外预算?

2. 读LLaVA-OneVision节4.3(emergent能)。提第四emergent技能课程likely unlock但论文未报。

3. Swap课程序—先训多图,后单图,后视频。预何benchmark degrade何。

4. 论文报仅8帧每sample训视频benchmark。推理时30秒视频泛化?何先破—token预算或时推理?

5. 24x24 patch至12x12 bilinear pooling是每维4x减。stdlib Python实pooling并验每2x2 block mean匹bilinear输出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| OneVision场景 | "单图、多图或视频" | 统VLM理三输入形之一;预算跨持常 |
| Token预算 | "每sample何token" | 每训/推理sample LLM见总视觉token,典型3000-4000 |
| 课程 | "训序" | 阶段序(单图→多图→视频)为emergent迁择 |
| Bilinear pooling | "Token缩" | 于patch grid(2D)apply bilinear interpolation减token数保局部 |
| Emergent技能 | "非训仍工" | 出现推理无匹配训数据能力,因课程组合 |
| AnyRes-k | "k-tile设" | k固定分辨率sub-tile加一thumbnail,典型k ∈ {4, 9} |
| 任务迁 | "跨场景泛化" | 单图学技能用于视频(反之)经共享backbone |

## 延伸阅读

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)