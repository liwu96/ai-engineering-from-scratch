# 神经音频编解码器——EnCodec、SNAC、Mimi、DAC和语义-声学分离

> 2026音频生成几乎全基于词元。EnCodec、SNAC、Mimi和DAC把连续波形转成离散序列让transformer预测。语义vs声学词元分离——第一码本作语义,余作声学——是自Transformer以来音频最重要架构转变。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段6课程02(频谱图)、阶段10课程11(量化)、阶段5课程19(子词词元化)
**时间:** ~60分钟

## 问题背景

语言模型工作于离散词元。音频连续。想语音/音乐用大语言模型风格模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——首先需**神经音频编解码器**:学习编码器离散音频到小词汇词元,及匹配解码器重建波形。

两族涌现:

1. **重建优先编解码器**——EnCodec、DAC。优化感知音频质量。词元"声学"——捕获一切包括说话人身份、音色、背景噪声。
2. **语义优先编解码器**——Mimi(Kyutai)、SpeechTokenizer。强制第一码本编码语言/音素内容(常通过从WavLM蒸馏)。后续码本是声学细节。

2024-2026洞察:**纯重建编解码器文本生成语音时给你模糊语音。**词元上大语言模型得在同码本学语言结构AND声学结构,不缩放。分离它们——语义码本0、声学码本1-N——是Moshi和Sesame CSM工作原因。

## 概念讲解

![四种编解码格局:EnCodec、DAC、SNAC(多尺度)、Mimi(语义+声学)](../assets/codec-comparison.svg)

### 核心技巧:残差向量量化(RVQ)

非一大码本(好质量需百万码),所有现代音频编解码用**RVQ**:小码本级联。第一码本量化编码器输出;第二量化残差;等等。每码本1024码。8码本=有效词汇1024^8=10^24。

推理时,解码器每帧求所有选中码重建。

### 2026四种重要编解码

**EnCodec(Meta, 2022)。** 基线。波形上编码器-解码器,RVQ瓶颈。24 kHz,可能32码本,默认4码本@1.5 kbps。用`1D卷积+transformer+1D卷积`架构。MusicGen用。

**DAC(Descript, 2023)。** RVQ配L2归一化码本、周期激活函数、改进损失。任何开源编解码最高重建保真——12码本时有时与原始语音不可分。44.1 kHz全带。

**SNAC(Hubert Siuzdak, 2024)。** 多尺度RVQ——粗码本比细码本低帧率跑。有效分层建模音频:~12 Hz粗"草图"+50 Hz细节。Orpheus-3B用因为层结构映射LM基生成好。

**Mimi(Kyutai, 2024)。** 2026颠覆者。12.5 Hz帧率(极低),8码本@4.4 kbps。码本0**从WavLM蒸馏**——训练预测WavLM语音内容特征。码本1-7声学残差。此分离支撑Moshi(课程15)和Sesame CSM。

### 帧率对语言建模重要

低帧率=短序列=快LM。

| 编解码 | 帧率 | 1秒=N帧 | 适用 |
|------|------|----------|------|
| EnCodec-24k | 75 Hz | 75 | 音乐、一般音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k(粗) | ~12 Hz | 12 | AR-LM高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

12.5 Hz下,10秒话语仅125编解码帧——transformer易预测。

### 语义vs声学词元

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义词元(Mimi码本0)。** 编码说的什么——音素、词、内容。从WavLM经辅助预测损失蒸馏。
- **声学词元(码本1-7)。** 编码音色、说话人身份、韵律、背景噪声、细细节。

AR LM先预测语义词元(以文本为条件),再预测声学词元(以语义+说话人参考为条件)。此分解是现代TTS可零样本克隆声音原因:语义模型处理内容;声学模型处理音色。

### 2026重建质量(比特每秒,低比特率好)

| 编解码 | 比特率 | PESQ | ViSQOL |
|------|------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统编解码如Opus每比特感知质量仍胜。神经编解码胜于**离散词元**(Opus不产)和**生成模型质量**(LM用这些词元能做什么)。

## 动手实践

### Step 1:EnCodec编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps时`n_codebooks=8`。每码0-1023(10位)。

### Step 2:解码并测重建

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3:语义-声学分离(Mimi风格)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本0 WavLM对齐。可训文本到语义transformer——词汇比直接到音频小。然后单独声学到波形解码器以说话人参考为条件。

### Step 4:为何AR LM词元上工作

10秒语音片段在Mimi 12.5 Hz × 8码本:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000词元对transformer是微不足道上下文。256M参数transformer可毫秒内现代GPU生成10秒语音。

## 实际应用

映射问题→编解码:

| 任务 | 编解码 |
|------|------|
| 一般音乐生成 | EnCodec-24k |
| 最高保真重建 | DAC-44.1k |
| AR LM语音(TTS) | SNAC或Mimi |
| 流式全双工语音 | Mimi(12.5 Hz) |
| 文本音效库 | EnCodec + T5条件 |
| 细粒度音频编辑 | DAC +修补 |

经验法则:**构建生成模型,起于Mimi或SNAC。构建压缩管道,用Opus。**

## 陷阱

- **太多码本。** 加码本线性增保真但LM序列长度也线性增。停在8-12。
- **帧率不匹配。** 12.5 Hz Mimi上训LM然后50 Hz EnCodec上微调静默失败。
- **假设所有码本等。** Mimi码本0载内容;丢它毁可懂性。丢码本7几乎无感。
- **仅用重建质量作唯一指标。** 编解码可重建好但语义结构坏则LM基生成无用。

## 产出成果

存`outputs/skill-codec-picker.md`。为给定生成或压缩任务选编解码。

## 练习题

1. **简单。** 跑`code/main.py`。实现玩具标量+残差量化器并测加码本重建误差。
2. **中等。** 装`encodec`并在保留语音片段比1、4、8、32码本。绘PESQ或MSE vs比特率。
3. **困难。** 载Mimi。编片段。用随机整数替码本0;解码。然后同样替码本7。比两腐蚀——码本0腐蚀应毁可懂性;码本7腐蚀应几乎无变。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RVQ | 残差量化 | 小码本级联;每量化前残差。 |
| 帧率 | 编解码速度 | 每秒多少词元帧。低=快LM。 |
| 语义码本 | 码本0(Mimi) | 从SSL特征蒸馏码本;编码内容。 |
| 声学码本 | 其余一切 | 音色、韵律、噪声、细细节。 |
| PESQ/ViSQOL | 感知质量 | 与MOS相关客观指标。 |
| EnCodec | Meta编解码 | RVQ基线;MusicGen用。 |
| Mimi | Kyutai编解码 | 12.5 Hz帧率;语义-声学分离;撑Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438)——RVQ基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546)——最高保真开源。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411)——多尺度RVQ。
- [Kyutai (2024). Mimi编解码](https://kyutai.org/codec-explainer)——语义-声学分离,WavLM蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143)——两阶段语义/声学范式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312)——原始可流RVQ编解码。