# 音频语言模型:Whisper到Audio Flamingo 3弧

> Whisper (Radford等，2022年12月)解语音识别——680k小时弱监督多语语音、简编码器解码器transformer、基准使每后续ASR发布引它。但识别非推理。问"录音何乐器"或"说话人何情绪"或"3分钟何发生"需音频理解，非转录。Qwen-Audio、SALMONN、LTU、NVIDIA Audio Flamingo 3 (AF3，2025年7月)渐进建栈:保Whisper类编码器、栓Q-former、训音频文本指令数据、加思维链推理。这课走弧。

**类型:** 构建
**语言:** Python (stdlib，log-Mel频谱图+音频Q-former骨架)
**前置要求:** 第6阶段(语音和音频)，第12阶段·03(Q-Former)
**时间:** ~180分钟

## 学习目标

- 从波形算log-Mel频谱图:窗、FFT、滤波bank、log变换。
- 比编码器选:Whisper编码器、BEATs、AF-Whisper混。每何时赢。
- 建音频Q-former:N学查询交叉注意频谱图patch。
- 解释级联(Whisper-then-LLM) vs端到端音频LLM训练:何端到端推理更缩。

## 问题背景

语音识别Whisper解。OCR-of-audio商品。但"商品"停转录。若模型不能推理所听——时间、说话人、情绪、音乐结构、环境声——转录不能驱产品特征。

三显路:

1. 级联:Whisper转录，LLM推理转录。纯语音场景工。音乐、环境音频、多说话人重叠、情绪失败。

2. 端到端音频LLM:音频编码器直喂音频token入LLM，跳转录。留声信息(情绪、说话人、环境)。需新训练数据。

3. 混:音频编码器+文本解码器可转录推理。Qwen-Audio和Audio Flamingo选此路。

## 概念讲解

### Log-Mel频谱图:输入特征

每音频编码器始同特征:log-Mel频谱图。

1. 重采样16 kHz。
2. 短时傅里叶变换25ms窗，10ms跳。
3. 取FFT结果幅。
4. 施Mel滤波bank(典型80滤波log距0-8000 Hz)扭感知频。
5. Log压缩(log(1 + x))动态范围。

结果:2D数组形(T, 80)其中T时间帧数。30秒clip 100 Hz帧率:(3000, 80)。

### Whisper编码器

Whisper编码器是12层ViT式transformer处理log-Mel频谱图作时间帧序。输出:每时间帧一隐藏态向量。

ASR，Whisper解码器是交叉注意transformer生成文本token条件编码器输出。标准编码器解码器。

ALM(音频LLM)，你想编码器输出作不同LLM输入。模式:Whisper编码器冻，Q-former可训，LLM冻或调。

### BEATs和音频特定编码器

Whisper训于语音主导数据。音乐和环境音频较弱。

BEATs (Chen等，2022)是AudioSet训自监督transformer。同参数数比Whisper捕音乐和环境声更好。

AF-Whisper (Audio Flamingo 3混):concat Whisper + BEATs特征作音频输入。Whisper带语言信号，BEATs带声信号。

### 音频Q-former

同模式BLIP-2视觉Q-former。固数学习查询(常32或64)交叉注意音频编码器输出帧。查询变音频token消费LLM。

训练对齐阶段:仅Q-former，对比+标注损失音频文本对(AudioCaps, Clotho)。指令阶段:端到端，解冻LLM，训指令数据。

### 弧——SALMONN, Qwen-Audio, AF3

SALMONN (Tang等，2023):Whisper + BEATs + Q-former + LLaMA。首开源音频LLM带严推理能力。MMAU基准示~0.55复合。

Qwen-Audio (Chu等，2023):类似架构，训富数据集，调多轮对话。MMAU ~0.60。

LTU —听，思，解(Gong等，2023):显推理数据，聚焦音频clip思维链。更小但更聚焦。

Audio Flamingo 3 (Goel等，2025年7月):当前开源SOTA。8B LLM背(Qwen2 7B)，Whisper-large编码器concat BEATs，64查询Q-former，训1M+音频文本指令对。MMAU 0.72，匹配私前沿些子任务。

AF3也引按需思维链音频:模型可选发思维token("让我先识乐器:...")前最终答。复杂推理任务启思维时准确提3-5点。

### 级联vs端到端

级联流水线:

1. Whisper转录音频→文本。
2. LLM推理文本。

完美工于"总结此播客。"失败于:
- "此歌何情绪？"——情绪在声，非词。
- "何人说话，Alice或Bob？"——需说话人识别。
- "何秒爆炸发生？"——时间grounding失于文本。
- "此真或生成音频？"——deepfake检测需声特征。

端到端留声信号。Qwen-Audio和AF3原生处理音乐、环境、情绪。

### 2026生产配方

新音频理解产品:

- 级联若:转录目标，无音乐，无情绪推理。
- AF3 / Qwen-Audio族若:音乐、情绪、多说话人、复杂音频推理。

级联便宜简。端到端更强。

### MMAU——音频推理基准

MMAU (Massive Multimodal Audio Understanding)是2024-2025音频推理基准:

- 10,000音频文本QA对跨语音、音乐、环境声。
- 覆分类、时间推理、因果推理、开QA。
- 测级联流水线系统失。

开源SOTA (AF3) 0.72；私前沿~0.78 (Gemini 2.5 Pro, Claude Opus 4.7)。差距小于VideoMME开源闭差距，示音频LLM成熟。

## 使用它

`code/main.py`:

- 实现log-Mel频谱图算stdlib:窗、朴素DFT、Mel滤波bank。
- 音频Q-former骨架:给编码器输出帧，算Q, K, V, attention，发N token。
- 级联vs端到端比玩具任务。

## 发货它

这课产`outputs/skill-audio-llm-pipeline-picker.md`。给音频任务(转录、音乐标注、情绪推理、多说话人diarization、环境分类)，选级联、端到端AF3、或混。

## 练习题

1. 算log-Mel频谱图维30秒clip 16kHz，25ms窗，10ms跳，80 Mel bin。48kHz何变？

2. 何Whisper音乐表现差？BEATs捕何音频特征Whisper不？

3. 音频Q-former 64查询vs 32:何任务复杂度64付？32省算何？

4. 读AF3第4节按需思维。提三音频任务思维链帮最多。

5. 实现简diarization流水线用AF3输出。何信号说话人变？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Log-Mel频谱图 | "Mel特征" | Mel滤波bank后2D(时间，频率)log幅值数组 |
| 音频Q-former | "音频Perceiver" | 音频编码器输出到固长查询馈LLM交叉注意瓶颈 |
| 级联 | "ASR-then-LLM" | Whisper转录文本LLM推理流水线；失声信息 |
| 端到端 | "音频LLM" | 音频特征经Q-former直入LLM；留声信号 |
| BEATs | "音频AudioSet编码器" | AudioSet训SSL transformer；音乐+环境声强 |
| MMAU | "音频推理bench" | 10k QA对跨语音、音乐、环境；2024评估标准 |
| 按需思维 | "音频CoT" | 模型可选发推理token前最终答，提准确3-5点 |

## 延伸阅读

- [Radford等—Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu等—Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel等—Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang等—SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong等—LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)