# 音频生成

> 音频是16-48 kHz的一维信号。五秒片段是80-240k采样。无Transformer能直接attend该序列。2026年每生产音频模型解决方案相同:神经编解码器(Encodec、SoundStream、DAC)压缩音频到50-75 Hz离散词元,Transformer或扩散模型生成词元。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(音频特征)、阶段6课程04(ASR)、阶段8课程06(DDPM)
**时间:** ~45分钟

## 问题背景

三音频生成任务:

1. **文本到语音(TTS)。**给定文本,产语音。干净语音窄带且有强语音结构——Transformer-over-词元解好。VALL-E(Microsoft)、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。**给定提示词(文本、旋律、和弦进程、流派),产音乐。分布更广。MusicGen(Meta)、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音频效果/声音设计。**给定提示词,产环境音或Foley。AudioGen、AudioLDM 2、Stable Audio Open。

全三跑同基底:神经音频编解码器+词元-AR或扩散生成器。

## 概念讲解

![音频生成:编解码器词元+Transformer或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec(Meta, 2022)、SoundStream(Google, 2021)、Descript Audio Codec(DAC, 2023)。卷积编码器压缩波形到每时间步向量;残差向量量化(RVQ)转换每向量到K码书索引级联。解码器反转。24 kHz音频2 kbps用8 RVQ码书75 Hz = 600词元/秒。

```
波形 (16000 samples/sec)
    └─ 编码器 conv ─┐
                     ├─ RVQ层1 → 75 Hz索引
                     ├─ RVQ层2 → 75 Hz索引
                     ├─ ...
                     └─ RVQ层8
```

### 两生成范式叠加

**词元自回归。**平RVQ词元成序列,跑decoder-only Transformer。MusicGen用"延迟并行"配每流偏移并行发K码书流。VALL-E从文本提示词+3秒语音样本产语音词元。

**潜扩散。**包编解码器词元作连续潜或配类别扩散建模。Stable Audio 2.5用连续音频潜上流匹配。AudioLDM 2用文本到mel到音频扩散。

2024-2026趋势:流匹配赢音乐(更快推理、更干净样本)而词元-AR仍主语音因其自然因果且流好。

## 生产格局

| 系统 | 任务 | 骨干 | 延迟 |
|------|------|------|------|
| ElevenLabs V3 | TTS | 词元-AR + 神经声码器 | ~300ms首个词元 |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态AR | ~200ms |
| NaturalSpeech 3 | TTS | 潜流匹配 | 非流 |
| Stable Audio 2.5 | 音乐/SFX | DiT + 音频潜上流匹配 | ~10s1分钟片段 |
| Suno v4 | 全歌 | 未公开;疑词元-AR | ~30s每歌 |
| Udio v1.5 | 全歌 | 未公开 | ~30s每歌 |
| MusicGen 3.3B | 音乐 | Encodec 32kHz上词元-AR | 实时 |
| AudioCraft 2 | 音乐+SFX | 流匹配 | ~5s5s片段 |
| Riffusion v2 | 音乐 | 频谱图扩散 | ~10s |

## 动手实践

`code/main.py`模拟核心想法:训微小下词元Transformer于两不同"风格"产合成"音频词元"序列(风格A交替低高词元,风格B单调斜坡)。条件于风格采样。

### Step 1: 合成音频词元

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "语音式":交替
        return [i % vocab_size for i in range(length)]
    # "音乐式":斜坡
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: 训微小词元预测器

条件于风格的bigram式预测器。要点模式:编解码器词元→交叉熵训练→自回归采样。

### Step 3: 条件采样

给定风格词元和起始词元,从预测分布采下词元。续20-40词元。

## 陷阱

- **编解码器质量封输出质量。**如编解码器不能忠实表示声音,生成器质量无帮助。DAC当前开最佳。
- **RVQ误差累积。**每RVQ层建模前层残差。层1误差传播。高层温度0采样助。
- **音乐结构。**30秒词元是75 Hz下20k+词元。Transformer难。MusicGen用滑窗+提示词续;Stable Audio用短片段+交叉淡化。
- **边界假象。**生成片段间交叉淡化需细心重叠-加。
- **干净数据需求。**音乐生成器需数万小时授权音乐。Suno/Udio RIAA诉讼(2024)浮出此。
- **语音克隆伦理。**3秒样本+文本提示词足让VALL-E/XTTS/ElevenLabs克隆语音。每生产模型需滥用检测+退出列表。

## 实际应用

| 任务 | 2026栈 |
|------|--------|
| 商业TTS | ElevenLabs、OpenAI TTS或Azure Neural |
| 语音克隆(同意验证) | XTTS v2(开)或ElevenLabs Pro |
| 背景音乐,快 | Stable Audio 2.5 API、Suno或Udio |
| 配歌词音乐 | Suno v4或Udio v1.5 |
| 声音效果/Foley | AudioCraft 2、ElevenLabs SFX或Stable Audio Open |
| 实时语音智能体 | GPT-4o realtime或Gemini Live |
| 开权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音/翻译 | HeyGen、ElevenLabs Dubbing |

## 产出成果

存`outputs/skill-audio-brief.md`。技能取音频简(任务、时长、风格、语音、许可)输出:模型+托管、提示词格式(流派标签、风格描述符、结构标记)、编解码器+生成器+声码器链、种子协议、和评估计划(MOS/CLAP分数/TTS CER/用户A/B)。

## 练习题

1. **简单。**跑`code/main.py`显设风格。验证生成序列匹风格模式。
2. **中等。**加延迟并行解码:模拟2词元流须偏1步。训联合预测器。
3. **困难。**用HuggingFace transformers本地跑MusicGen-small。三不同提示词生成10秒片段;A/B风格跟随。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 编解码器 | "神经压缩" | 音频编码器/解码器;典型输出50-75 Hz词元。 |
| RVQ | "残差VQ" | K量化器级联;每建模前残差。 |
| 词元 | "编解码器符号" | 码书离散索引;1024或2048典型。 |
| 延迟并行 | "偏码书" | 配交错偏移发K词元流减序列长。 |
| 流匹配 | "2024音频胜利" | 扩散直路替代;更快采样。 |
| 语音提示词 | "3秒样本" | 说话者嵌入或词元前缀引导克隆语音。 |
| Mel频谱图 | "可视化" | 对数幅度感知频谱图;多TTS系统用。 |
| 声码器 | "Mel到波" | Mel频谱图回音频神经组件。 |

## 生产注:音频是流问题

音频是用户期望*生成时*到达而非全一次的唯一输出模态。生产术语TPOT重要(Time Per Output Token)因用户听速是目标吞吐——非读速。16kHz音频~75词元/秒(Encodec)编码,服务器须每用户生≥75词元/秒保播放平滑。

两架构后果:

- **流匹配音频模型不能简单流。**Stable Audio 2.5和AudioCraft 2一次pass渲染固定片段长。流需分片段重叠边界——想滑窗扩散——加100-300ms延迟开销vs编解码器AR模型。

如产品是"活语音聊"或"实时音乐续",选编解码器AR路。如是"提交渲染30秒片段",流匹配赢质量和总延迟。

## 延伸阅读

- [Défossez等(2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)——编解码器标准。
- [Zeghidour等(2021). SoundStream](https://arxiv.org/abs/2107.03312)——首广用神经音频编解码器。
- [Kumar等(2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)——DAC。
- [Wang等(2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111)——VALL-E。
- [Copet等(2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284)——MusicGen。
- [Liu等(2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734)——AudioLDM 2。
- [Stability AI(2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5)——2025流匹配文本到音乐。