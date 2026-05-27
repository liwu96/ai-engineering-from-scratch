# 百万Token上下文长视频理解

> 1小时4K视频24 FPS，patch和嵌入，产约60百万token。2小时播客转录30,000 token。全蓝光电影，即使激进池压缩，数百千token。Google Gemini 1.5 (2024年3月)开此时代带10百万token上下文，小时视频可靠needle-in-a-haystack召回。LWM (Liu等，2024年2月)示ring attention缩路径。LongVILA和Video-XL缩摄取更。VideoAgent换原始上下文agent检索。每方法是算、召回、工程复杂度不同权衡。这课并读。

**类型:** 构建
**语言:** Python (stdlib，needle-in-haystack模拟器+agent检索路由)
**前置要求:** 第12阶段·17(视频时间token)
**时间:** ~180分钟

## 学习目标

- 算长视频总视觉token数于不同FPS和池。
- 解释三缩路径:蛮上下文(Gemini 1.5)、ring attention (LWM)、token压缩(LongVILA / Video-XL)。
- 比原始上下文视频VLM vs agent检索视频VLM (VideoAgent)于准确度和延迟。
- 设计needle-in-a-haystack测试30分钟视频测特定分钟召回。

## 问题背景

Qwen2.5-VL大小patch单帧384原生分辨率是~729 token。3x3池每帧81 token。30分钟clip 1 FPS = 1800帧 = 145,800 token。2025开源VLM可做，紧。2 FPS，291,600 token——仅最大上下文适。

2小时电影1 FPS是583k token。超大多2026开源模型；需Gemini 2.5 Pro或更激进池。

三缩路径现。

## 概念讲解

### 路径1:蛮上下文(Gemini 1.5, Claude Opus)

抛硬件问题。缩上下文百万token，一切一次forward pass处理。

Gemini 1.5 Pro启1M token；Gemini 1.5 Ultra至10M；Gemini 2.5 Pro 2026可靠小时视频。论文(arXiv:2403.05530)记录needle-in-a-haystack召回99.7%至~9.5M token。

工程:定制注意力实现带内存层次(局+全+疏)加MoE专家路由长上下文效率。未全公开。不开源。

### 路径2:Ring attention (LWM, LongVILA)

Ring attention分布长序跨设备"ring"每设备持块。全序注意力通过每设备发块至下一ring模式算部分注意聚合。

LWM (Liu等，2024)训1M token上下文模型。训练算缩上下文线，非二次——二次注意击跨ring设备摊。

LongVILA (arXiv:2408.10188)适模式VLM。1400帧视频192 token每帧=268k上下文，ring attention跨8路并行训。

### 路径3:Token压缩(Video-XL, LongVA)

比蛮上下文便宜:LLM见序前激进压缩。

Video-XL (arXiv:2409.14485)用视觉总结token:N帧clip产单"总结"token注意于N。推理时，LLM见一总结token每clip，剧缩上下文。

LongVA延LLM上下文从200k到2M带"长上下文转移"技术。训长上下文文本，转移长上下文视频经共享表示。

Token压缩权衡特定时间戳召回可缩性。模型知大概何发生但有时失精确帧。

### 路径4:Agent检索(VideoAgent)

不喂全视频LLM。代，视视频数据库用LLM查询。

VideoAgent (arXiv:2403.10517):

1. LLM读问题。
2. LLM问检索工具相关clip("示猫片段")。
3. 工具返匹配clip时间戳。
4. LLM经VLM读那些clip。
5. LLM组答或问后续查询。

这是LLM-as-agent模式适长视频。推理便宜(仅相关clip编码)，工程难(检索质量变瓶颈)。

### Needle-in-a-haystack基准

标准长上下文测试:插入独一视觉或文本标记于视频随机点，然后问需召回查询。

指标:Recall@k跨视频长和标记位置。

Gemini 2.5 Pro评分>99%召回至90分钟视频。开源72B模型(Qwen2.5-VL-72B, InternVL3-78B)30分钟评分~85-90%，60分钟后退。

VideoAgent可匹配或击原始上下文模型2+小时因检索击needle若工具好。

### 何路选

15分钟clip前沿准确度:开源72B +原生上下文通常工。选Qwen2.5-VL-72B。

30分钟到1小时内容:LongVILA或Video-XL开源；Gemini 2.5 Pro闭。质量杠重要——前沿闭。

2+小时内容:VideoAgent或类似检索模式。代，总结更小chunk喂层次总结。

### 2026生产模式

实践，生产长视频流水线混:

1. 全视频运动态FPS采样+激进池(得100k-token全局表示)。
2. 传72B VLM全局总结。
3. 若用户问详问，用总结作索引运agent检索。

这合蛮上下文全局理解和检索局部细节。

## 使用它

`code/main.py`:

- 算token预算视频从1分钟到3小时不同FPS +池。
- 模拟needle-in-a-haystack运行:注入标记随机时间戳，问问题，评分召回。
- 含agent检索路由模拟器选特定clip喂下游VLM。

运预算表感缩差距。

## 发货它

这课产`outputs/skill-long-video-strategy-planner.md`。给视频持续和查询复杂度，选蛮上下文、压缩、agent检索，算延迟+质量预期。

## 练习题

1. 45分钟讲座1 FPS，81 token每帧。总token何？适哪些模型上下文？

2. 设计needle-in-a-haystack测试:何分钟注入标记，何是精确查询格式？

3. 比蛮上下文Qwen2.5-VL-72B (80k上下文)和VideoAgent (Claude 3.5 +检索)于1小时视频。何赢召回？何赢延迟？

4. Ring attention内存成本缩序长线缩设备数线。解释何若弃ring旋阶段何失败。

5. 读Gemini 1.5第5节needle-in-a-haystack。论文发现何于1M vs 10M token边界召回？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 蛮上下文 | "仅更多token" | 缩LLM上下文百万token；一次pass处理一切 |
| Ring attention | "LWM式并行" | 分布注意力模式每设备持块旋 |
| Token压缩 | "总结token" | 经学压缩器每clip减token于LLM前 |
| Needle-in-haystack | "NIH测试" | 插独一标记随机点，测时问模型召回 |
| Agent检索 | "LLM作查询规划器" | LLM问检索工具相关clip，经VLM读，组答 |
| VideoAgent | "视频检索模式" | 典型agent检索设计:问→工具→clip→答 |

## 延伸阅读

- [Gemini Team—Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu等—LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue等—LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu等—Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang等—VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)