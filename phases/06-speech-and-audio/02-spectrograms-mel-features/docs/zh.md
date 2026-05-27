# 频谱图、Mel尺度与音频特征

> 神经网络不善消费原始波形。它们消费频谱图。mel频谱图更好。2026年每个ASR、TTS、音频分类器成败于这预处理选择。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程01(音频基础)
**时间:** ~45分钟

## 问题背景

取10秒16 kHz片段。160,000浮点,全在`[-1, 1]`,与标签"狗叫"或"词cat"几乎完美无关。原始波形有信息但模型难以提取形式。相隔100 ms两相同音素完全不同原始样本。

频谱图修复此。坍人类感知忽略时细节(微秒抖)并保感知关注结构(哪些频率能量,~10–25 ms时间窗)。

Mel频谱图推更远。人类音高感知对数:100 Hz vs 200 Hz听"同距离"如1000 Hz vs 2000 Hz。Mel尺度扭曲频率轴匹配。mel尺度频谱图是2010到2026语音机器学习最重要特征。

## 概念讲解

![波形到STFT到mel频谱图到MFCC阶梯](../assets/mel-features.svg)

**STFT(短时傅里叶变换)。**切片波形成重叠帧(典型:25 ms窗,10 ms跳步=16 kHz下400样本/160样本)。每帧乘窗函数(Hann默认;Hamming略不同权衡)。每帧FFT。叠幅度谱成`(n_frames, n_freq_bins)`形矩阵。那是频谱图。

**Log幅度。**原始幅度跨5-6量级。取`log(|X| + 1e-6)`或`20 * log10(|X|)`压缩动态范围。每个生产管道用log幅度,非原始幅度。

**Mel尺度。**Hz频率`f`映射mel `m`:`m = 2595 * log10(1 + f / 700)`。映射1 kHz下约线性,上约对数。覆盖0–8 kHz80 mel bin是标准ASR输入。

**Mel滤波器组。**mel尺度等间距三角滤波器集。每滤波器是相邻FFT bin加权求和。STFT幅度乘滤波器组矩阵一次matmul给mel频谱图。

**Log-mel频谱图。**`log(mel_spec + 1e-10)`。Whisper输入。Parakeet输入。SeamlessM4T输入。通用2026音频前端。

**MFCCs。**取log-mel频谱图,用DCT(类型II),留前13系数。解相关特征并进一步压缩。~2015 CNN/Transformer原始log-mel赶上前主导特征。说话人识别(x-vector、ECAPA)仍用。

**分辨率权衡。**更大FFT=更好频率分辨率但更差时间分辨率。25 ms / 10 ms音频机器学习默认;音乐50 ms / 12.5 ms;瞬态检测(鼓击、爆破音)5 ms / 2 ms。

## 动手实践

### Step 1:帧化波形

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160`下10秒16 kHz片段产998帧。

### Step 2:Hann窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT前逐元素乘。消除非零端截断致谱泄漏。

### Step 3:STFT幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产用`torch.stft`或`librosa.stft`(FFT后台,向量化)。此处循环教学;`code/main.py`短片段上跑。

### Step 4:mel滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400`覆盖0–8 kHz80 mels给`(80, 201)`矩阵。`(n_frames, 201)` STFT幅度乘转置得`(n_frames, 80)` mel频谱图。

### Step 5:log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代:`librosa.power_to_db`(参考归一化dB)、`10 * log10(power + eps)`。Whisper用更复杂裁剪+归一化流程(见Whisper `log_mel_spectrogram`)。

### Step 6:MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

每log-mel帧用DCT,留前13系数。那是MFCC矩阵。首系数常弃(编总能量)。

## 实际应用

2026栈:

| 任务 | 特征 |
|------|------|
| ASR(Whisper、Parakeet、SeamlessM4T) | 80 log-mels,10 ms跳步,25 ms窗 |
| TTS声学模型(VITS、F5-TTS、Kokoro) | 80 mels,5–12 ms跳步细时控 |
| 音频分类(AST、PANNs、BEATs) | 128 log-mels,10 ms跳步 |
| 说话人嵌入(ECAPA-TDNN、WavLM) | 80 log-mels或原始波形自监督学习 |
| 音乐(MusicGen、Stable Audio 2) | EnCodec离散词元(非mel) |
| 关键词检测 | 微设备40 MFCC |

经验:**非音乐,80 log-mels起。**任何偏差举证责任在你。

## 2026仍发货陷阱

- **Mel数不匹配。**训80 mels,推理128 mels。静默失败。两端记特征形。
- **上游采样率不匹配。**22.05 kHz算mel与16 kHz看不同。特征化前*先*修SR。
- **dB vs log。**Whisper期望log-mel,非dB-mel。些HF管道自动检测;你自定义代码不。
- **归一化漂移。**训每话语归一化,推理全局归一化。生产bug双词错误率。
- **填充泄漏。**片段末零填充产生末帧平谱。对称填充或复制。

## 产出成果

存`outputs/skill-feature-extractor.md`。技能为给定模型目标选特征类型、mel数、帧/跳步、归一化。

## 练习题

1. **简单。**跑`code/main.py`。合chirp(频率扫200→4000 Hz)并每帧打印argmax mel bin。绘(可选)并确认配扫。
2. **中等。**`n_mels`在`{40, 80, 128}`和`frame_len`在`{200, 400, 800}`重跑。时间轴测锐峰带宽。哪组合解chirp最好?
3. **困难。**实现`power_to_db`并比较AudioMNIST上小CNN分类器ASR准确率用 log-mel、dB-mel配`ref=max`、MFCC-13+delta+delta-delta。报告top-1准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 帧 | 切片 | 喂一FFT的25 ms波形块。 |
| 跳步 | 步长 | 连续帧间样本;ASR默认10 ms。 |
| 窗 | Hann/Hamming东西 | 端 taper到零逐点乘。 |
| STFT | 频谱图生成器 | 分帧+加窗FFT;产时间×频率矩阵。 |
| Mel | 扭曲频率 | 对数感知尺度;`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 矩阵 | STFT投mel bin三角滤波器。 |
| Log-mel | Whisper输入 | `log(mel_spec + eps)`;2026标准化。 |
| MFCC | 老式特征 | log-mel DCT;13系数,解相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420)——MFCC论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/)——原始mel尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py)——读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html)——`mfcc`、`melspectrogram`、跳步/窗参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers)——Parakeet+Canary模型生产规模管道。