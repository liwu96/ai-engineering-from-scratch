# 文本转语音(TTS)——从Tacotron到F5和Kokoro

> ASR将语音反转成文本;TTS将文本反转成语音。2026栈三部分:文本→词元、词元→mel、mel→波形。每部分有笔记本跑默认模型。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图与Mel)、阶段5课程09(Seq2Seq)、阶段7课程05(完整Transformer)
**时间:** ~75分钟

## 问题背景

有字符串:"Please remind me to water the plants at 6 pm."需3秒自然音频片段,正确韵律(停顿、重音),"plants"发对元音,CPU上300 ms内跑用于实时语音助手。还要换声、处理代码切换输入("remind me at 6 pm, daijoubu?")、名字不丢脸。

现代TTS管道:

1. **文本前端。**归一化文本(日期、数、邮箱),转音素或子词词元,预测韵律特征。
2. **声学模型。**文本→mel频谱图。Tacotron 2(2017)、FastSpeech 2(2020)、VITS(2021)、F5-TTS(2024)、Kokoro(2024)。
3. **声码器。**Mel→波形。WaveNet(2016)、WaveRNN、HiFi-GAN(2020)、BigVGAN(2022)、2024+神经编解码声码器。

2026声学+声码器分割随端到端扩散和流匹配模型模糊。但三部分心智模型调试仍持。

## 概念讲解

![Tacotron、FastSpeech、VITS、F5/Kokoro并排](../assets/tts.svg)

**Tacotron 2(2017)。**Seq2seq:字符嵌入→BiLSTM编码器→位置敏感注意力→自回归LSTM解码器发mel帧。慢(AR),长文本不稳。仍引为基线。

**FastSpeech 2(2020)。**非自回归。时长预测器输出每音素得多少mel帧。1-pass,比Tacotron10×快。失些自然性(单调对齐)但到处发货。

**VITS(2021)。**联合训编码器+基于流时长+HiFi-GAN声码器端到端配变分推断。高质量,单模型。2022–2024开源TTS主导。变体:YourTTS(多说话人零样本)、XTTS v2(2024, Coqui)。

**F5-TTS(2024)。**流匹配上扩散Transformer。自然韵律,5秒参考音频零样本声音克隆。2026开源TTS排行榜顶。335M参数。

**Kokoro(2024)。**小(82M),CPU可跑,实时用最佳英语TTS。闭词汇仅英语,apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。**商业最先进。ElevenLabs v2.5情绪标签("[whispered]", "[laughing]")和角色声音主导2026有声书生产。

### 声码器演进

| 年代 | 声码器 | 延迟 | 质量 |
|------|---------|------|------|
| 2016 | WaveNet | 仅离线 | 发布SOTA |
| 2018 | WaveRNN | ~实时 | 好 |
| 2020 | HiFi-GAN | 100×实时 | 近人 |
| 2022 | BigVGAN | 50×实时 | 跨说话人/语言泛化 |
| 2024 | SNAC, DAC(神经编解码) | AR模型集成 | 离散词元,位效 |

2026大多"TTS"模型文本到波形端到端;mel频谱图是内部表示。

### 评估

- **MOS(平均意见分)。**1–5分,众包。仍金标准;痛慢。
- **CMOS(比较MOS)。**A-vs-B偏好。每标注更窄置信区间。
- **UTMOS、DNSMOS。**无参考神经MOS预测器。排行榜用。
- **CER(字符错误率)经ASR。**Whisper跑TTS输出,算输入文本CER。可懂性代理。
- **SECS(说话人嵌入余弦相似度)。**声音克隆质量。

2026 LibriTTS test-clean数:

| 模型 | UTMOS | CER(经Whisper) | 大小 |
|------|-------|-----------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手实践

### Step 1:音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥。避免喂原始文本给VITS级下质量任何东西。

### Step 2:Kokoro运行(2026 CPU默认)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" =美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32张量, sr=24000
```

离线跑,单文件,82M参数。

### Step 3:F5-TTS配声音克隆运行

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传5秒参考片段+其转录;F5克隆韵律和音色。

### Step 4:HiFi-GAN声码器从头

太大不适教程脚本,但形:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4上采样块,总共256x从mel率到音频率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # ->波形
```

训:对抗(短窗判别器) + mel频谱图重建损失 +特征匹配损失。商品化——用`hifi-gan`库或nvidia-NeMo预训检查点。

### Step 5:完整管道(伪代码)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 实时英语语音助手 | Kokoro(CPU)或XTTS v2(GPU) |
| 5秒参考声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书叙述 | ElevenLabs v2.5或XTTS v2 +微调 |
| 低资源语言 | 目标语言5–20小时数据训VITS |
| 表达/情绪标签 | ElevenLabs v2.5或StyleTTS 2微调 |

2026开源领袖:**F5-TTS质量,Kokoro效率**。除非历史学家不碰Tacotron。

## 陷阱

- **无文本归一化器。**"Dr. Smith"读"Doctor"还是"Drive"? "2026"读"twenty twenty six"还是"two zero two six"? 音素化器前归一化。
- **OOV专名词。**"Ghumare" → "ghyu-mair"? 未知词元发货备选字形到音素模型。
- **裁剪。**声码器输出少裁剪,但推理时mel缩放不匹配可超±1.0。总`np.clip(wav, -1, 1)`。
- **采样率不匹配。**Kokoro输出24 kHz;下游管道期望16 kHz →重采样或得混叠。

## 产出成果

存`outputs/skill-tts-designer.md`。设计给定声音、延迟和语言目标TTS管道。

## 练习题

1. **简单。**跑`code/main.py`。从玩具词汇构音素字典,估每音素时长,打印假"mel"调度。
2. **中等。**装Kokoro,声音`af_bella`和`am_adam`合成同句子。比音频时长和主观质量。
3. **困难。**录自己5秒参考片段。用F5-TTS克隆。报参考和克隆输出间SECS。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 音素 | 声单元 | 抽象声类;英语39(ARPABet)。 |
| 时长预测器 | 每音素多久 | 非自回归模型输出;每音素整数帧。 |
| 声码器 | Mel→波形 | mel频谱图到原始样本神经网络映射。 |
| HiFi-GAN | 标准声码器 | 基GAN;2020–2024主导。 |
| MOS | 主观质量 | 人类评分者1–5平均意见分。 |
| SECS | 声克隆指标 | 目标和输出说话人嵌入间余弦相似度。 |
| F5-TTS | 2024开源SOTA | 流匹配扩散;零样本克隆。 |
| Kokoro | CPU英语领袖 | 82M参数模型,Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884)——seq2seq基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103)——端到端基于流。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885)——当前开源SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646)——2026仍发货声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)——2024 CPU友好英语TTS。