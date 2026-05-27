# 视频语言模型:时间Token和Grounding

> 视图非照片堆。5秒clip有因果序、行动动词、事件时序图像模型无法表示。Video-LLaMA (Zhang等，2023年6月)交付首开源视频LLM带音频视觉grounding。VideoChat和Video-LLaVA缩模式。2025 Qwen2.5-VL的TMRoPE闭前沿私模差距。每系统解时间token不同——每clip Q-former、每帧concat-pool、每token TMRoPE。这课读模式，建统一vs动态帧采样器，评于时间grounding任务。

**类型:** 构建
**语言:** Python (stdlib，帧采样器+时间grounding评估器)
**前置要求:** 第12阶段·08(LLaVA-OneVision)
**时间:** ~180分钟

## 学习目标

- 解释何时间位置编码改视频VLM性能独立于视觉编码器。
- 比统一、动态FPS、事件驱动帧采样于tokens-per-second vs grounding准确度。
- 描述每clip Q-former (Video-LLaMA) vs pooled-per-frame (Video-LLaVA) vs M-RoPE-per-token (Qwen2.5-VL)设计。
- 命四个视频基准:VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题背景

1分钟视频30 FPS是1800帧。每帧196视觉token (ViT-B 224)，那是352k token——超任何2024代LLM上下文。

三减策略存:

1. 子采样帧(1-8 FPS依内容)。
2. 每帧patch token激进池(3x3或4x4双线性池)。
3. 经Q-former压缩取16帧clip出64 token。

每权衡不同。子采样失时间细节。池失空间细节。Q-former失两者一点但省token。

时间位置编码是另一轴:模型何知帧5在帧6前？选包括简1D时间RoPE (Video-LLaMA)、学时间嵌入(Video-LLaVA)、TMRoPE (Qwen2.5-VL，全3D)。

## 概念讲解

### Video-LLaMA:每clip Q-former +音频分支

Video-LLaMA (2023)是首开源视频LLM。架构:

- 16帧clip于2 FPS(故8秒)。
- 每帧ViT特征→Video Q-former交叉注意全16帧→32学查询→LLM。
- 并行音频分支:波形→ImageBind音频编码器→Audio Q-former→32查询→LLM。

强:音频视觉联合推理。弱:固clip长，无任意时间grounding。

### VideoChat和Video-LLaVA

VideoChat保Video-LLaMA思想但弃音频简化。Video-LLaVA (Lin等，2023)训单一视觉编码器于图像和视频帧("投影前对齐")，给统一表示。两者皆冻CLIP编码器+MLP+LLM。

两者不长视频处理。皆是8-16帧系统。

### Qwen2.5-VL和TMRoPE

Qwen2.5-VL引TMRoPE——时间-模态旋位置嵌入。每patch token带(t, h, w)位置其中t是实际时间戳(非帧索引)。

与简时间嵌入关键差:

- 绝对时间，非索引。模型见"4.2秒"非"帧15"。
- 每token旋，非每clip。每视觉token独立旋于时间戳。
- 兼动态FPS。若此处2 FPS采样彼处4 FPS，TMRoPE原生处理不均间距。

TMRoPE使"猫何秒跳？"查询。模型可出"4.2秒。"Video-LLaMA仅能说"clip早期。"

### 帧采样策略

统一:均匀于持续采N帧。简，失运动峰。

动态FPS:依运动强度自适应采样。光流或帧差选高运动段密集采样。Qwen2.5-VL训于此。

事件驱动:运轻量检测器，多采动作处。VideoAgent用。

关键帧+上下文:于镜头边界采+几邻帧。用于电影内容。

### 每帧池

1 FPS和576 token每帧，5分钟clip是172,800 token。Qwen2.5-VL-72B 128k上下文可做但贵。

3x3双线性池减每帧64 token→5分钟19,200 token。甜点大多任务。

更激进池(6x6→每帧16 token)用于agent工作流空间细节更少。

### 四视频基准

- VideoMME:综合视频理解，短+中+长。
- TempCompass:细粒时间推理，"前"/"后"问题。
- EgoSchema:长视第一人称视频。
- Video-MMMU:多模态多科视频问题。

全视频VLM评估击四。它们压不同轴——TempCompass全序，EgoSchema是3+分钟推理，VideoMME跨持续。

### Grounding输出格式

时间grounding输出格式:

- 自由文本:"猫约4秒跳。"易解析但不精确。
- 结构JSON:`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL训此。
- Token基:特殊`<time>4.1</time>` token交答案。Qwen2.5-VL内部格式。

Token基下游用最准确。Qwen2.5-VL JSON输出格式直解析。

### 2026最佳实践

2026视频VLM:

- 编码器:SigLIP 2带M-RoPE或TMRoPE (Qwen2.5-VL)。
- 帧采样:动态FPS(1-4依运动)带最大帧帽。
- 每帧池:3x3双线性。
- 输出:结构JSON带时间+事件字段。
- 基准:VideoMME + TempCompass通用；EgoSchema长视。

## 使用它

`code/main.py`包括:

- 统一和动态FPS帧采样器。
- 玩具时间grounding评估器:给"真相"事件于时间T和模型输出，容忍度评分准确。
- 比Video-LLaMA (16帧，Q-former)、Video-LLaVA (8帧，MLP)、Qwen2.5-VL (动态FPS + TMRoPE)。

## 发货它

这课产`outputs/skill-video-vlm-frame-planner.md`。给视频任务(监控、动作识别、时间grounding、总结)，选帧采样器、池因子、输出格式、预期准确度级。

## 练习题

1. 3分钟烹饪demo，选统一vs动态FPS。用token计数据。

2. TMRoPE加何简时间嵌入表不能？

3. 写VLM可学发的JSON schema时间grounding。含错例。

4. 读Video-LLaVA第3节"投影前对齐"。何优于训分开图像视频编码器？

5. 给VideoMME排行榜，2026顶开源模型和顶私模差距何？差距多少归于时间编码vs基础LLM规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 时间grounding | "时间定位答" | VLM输出特定时间戳范围事件何时发生 |
| TMRoPE | "时间-多模态RoPE" | 3D旋位置带绝对时间戳，Qwen2.5-VL用 |
| 动态FPS | "运动感知采样" | 高运动段多采帧，静态少采 |
| 帧池 | "每帧空间压缩" | 双线性插值减每帧patch于LLM前 |
| Video Q-former | "Clip压缩器" | 交叉注意瓶颈N帧映K学查询 |
| VideoMME | "视频bench" | 综合短/中/长视频基准，2500+样 |

## 延伸阅读

- [Zhang等—Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li等—VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin等—Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team—Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin等—VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)