# 音频基础——波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。Mel特征是机器学习友好的形式。每个现代ASR和TTS管道都沿这条路径前进,第一步是理解采样和傅里叶变换。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段1课程06(向量与矩阵)、阶段1课程14(概率分布)
**时间:** ~45分钟

## 问题背景

麦克风产生压力-时间信号。神经网络消费张量。两者之间是一堆约定,违反时会产生静默bug:模型训练正常但词错误率翻倍,或TTS发出嘶嘶声,或语音克隆系统记住麦克风而非说话者。

语音系统的每个bug追溯到三个问题之一:

1. 数据录制时的采样率是多少,模型期望什么?
2. 信号是否混叠?
3. 你操作的是原始样本还是频率表示?

正确处理这些,阶段6其余部分可控。弄错它们,即使Whisper-Large-v4也产生垃圾。

## 概念讲解

![波形、采样、DFT和频率bin可视化](../assets/audio-fundamentals.svg)

**波形。**一维浮点数组,范围`[-1.0, 1.0]`。按样本编号索引。转换为秒:除以采样率`t = n / sr`。16 kHz下10秒片段是160,000个浮点数数组。

**采样率(sr)。**每秒多少样本。2026年常见采样率:

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、遗留VOIP。Nyquist在4 kHz杀死辅音。ASR避免使用。 |
| 16 kHz | ASR标准。Whisper、Parakeet、SeamlessM4T v2都消费16 kHz。 |
| 22.05 kHz | 旧模型TTS声码器训练。 |
| 24 kHz | 现代TTS(Kokoro、F5-TTS、xTTS v2)。 |
| 44.1 kHz | CD音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真TTS(VALL-E 2、NaturalSpeech 3)。 |

**Nyquist-Shannon定理。**采样率`sr`可无歧义表示高达`sr/2`的频率。`sr/2`边界是*Nyquist频率*。高于Nyquist的能量被*混叠*——折叠到更低频率——并破坏信号。降采样前始终低通滤波。

**位深度。**16-bit PCM(有符号int16,范围±32,767)是通用交换格式。音乐用24-bit,内部DSP用32-bit浮点。`soundfile`等库读取int16但暴露`[-1, 1]`内float32数组。

**傅里叶变换。**任何有限信号是不同频率正弦波之和。离散傅里叶变换(DFT)对`N`样本计算`N`个复系数——每频率bin一个。`bin k`映射到频率`k · sr / N` Hz。幅度是该频率振幅,角度是相位。

**FFT。**快速傅里叶变换:`N`是2的幂时DFT的`O(N log N)`算法。每个音频库底层用FFT。16 kHz下1024样本FFT给出512个可用频率bin,覆盖0–8 kHz,分辨率15.6 Hz。

**分帧+窗。**不对整个片段FFT。切成重叠*帧*(典型25 ms,10 ms跳步),每帧乘窗函数(Hann、Hamming)消除边缘不连续,然后FFT每帧。这是短时傅里叶变换(STFT)。课程02从这里继续。

## 动手实践

### Step 1:读取片段并绘制波形

`code/main.py`仅用stdlib `wave`模块保持演示无依赖。生产用`soundfile`或`torchaudio.load`(都返回`(waveform, sr)`元组):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # 形(T,), sr=int
```

### Step 2:从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz下1秒440 Hz正弦(音乐A)是16,000个浮点。用`wave.open(..., "wb")`以16-bit PCM编码写入。

### Step 3:手工计算DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——`N=256`验证正确性可行,真实音频无用。真实代码调用`numpy.fft.rfft`或`torch.fft.rfft`。

### Step 4:找到主频

幅度峰索引`k_star`映射到频率`k_star * sr / N`。440 Hz正弦上运行应在bin `440 * N / sr`返回峰。

### Step 5:演示混叠

10 kHz采样7 kHz正弦(Nyquist = 5 kHz)。7 kHz音高于Nyquist折叠到`10 − 7 = 3 kHz`。FFT峰出现在3 kHz。这是经典混叠演示,每个DAC/ADC都配砖墙低通滤波器的原因。

## 实际应用

2026年实际部署栈:

| 任务 | 库 | 原因 |
|------|-----|------|
| 读/写WAV/FLAC/OGG | `soundfile`(libsndfile封装) | 最快、稳定、返回float32。 |
| 重采样 | `torchaudio.transforms.Resample`或`librosa.resample` | 内置正确抗混叠。 |
| STFT / Mel | `torchaudio`或`librosa` | GPU友好;PyTorch生态。 |
| 实时流 | `sounddevice`或`pyaudio` | 跨平台PortAudio绑定。 |
| 检查文件 | `ffprobe`或`soxi` | CLI、快、报告sr/通道/编解码。 |

决策规则:**匹配采样率优先于匹配其他任何东西**。Whisper期望16 kHz单声道float32。传入44.1 kHz立体声会得垃圾,看起来像模型bug。

## 产出成果

存`outputs/skill-audio-loader.md`。技能帮你检查音频输入匹配下游模型期望,不匹配时正确重采样。

## 练习题

1. **简单。**16 kHz下合成1秒220 Hz + 440 Hz + 880 Hz混合。运行DFT。确认预期bin处三峰。
2. **中等。**48 kHz录制3秒语音WAV。用`torchaudio.transforms.Resample`(抗混叠)降采样到16 kHz,再用朴素降采样(每第三样本)到16 kHz。FFT两者。混叠出现在哪?
3. **困难。**仅用`math`和Step 3 DFT从头构建STFT。帧大小400,跳步160,Hann窗。用`matplotlib.pyplot.imshow`绘制幅度。这是课程02的频谱图。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 采样率 | 每秒多少样本 | ADC测量信号的频率(Hz)。 |
| Nyquist | 能表示的最大频率 | `sr/2`;高于它的能量混叠回来。 |
| 位深度 | 每样本分辨率 | `int16` = 65,536级别;`float32` = `[-1, 1]`内24-bit精度。 |
| DFT | 序列的傅里叶变换 | `N`样本→`N`复频率系数。 |
| FFT | 快速DFT | `N` = 2的幂时`O(N log N)`算法。 |
| Bin | 频率列 | `k · sr / N` Hz;分辨率 = `sr / N`。 |
| STFT | 频谱图底层 | 时间上分帧+加窗FFT。 |
| 混叠 | 奇怪频率幽灵 | Nyquist以上能量镜像到更低bin。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)——采样定理论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm)——免费、权威DSP教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html)——带代码实用 walkthrough。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434)——真实世界音频非干净正弦波的原因参考。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/)——10分钟清理频率bin直觉。