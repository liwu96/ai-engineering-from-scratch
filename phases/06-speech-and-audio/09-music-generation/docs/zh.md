# 音乐生成——MusicGen、Stable Audio、Suno和版权震动

> 2026音乐生成:Suno v5和Udio v4主导商业;MusicGen、Stable Audio Open和ACE-Step领跑开源。技术问题基本解决。法律问题(Warner Music $500M和解、UMG和解)在2025-2026重塑领域。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图)、阶段4课程10(扩散模型)
**时间:** ~75分钟

## 问题背景

文本 → 30秒到4分钟音乐片段,含歌词、人声和结构。三个子问题:

1. **器乐生成。** 文本如"lo-fi hip-hop鼓配暖键盘" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成(含人声+歌词)。** "关于德州雨夜的乡村歌" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控。** 延伸现有片段、重生成桥段、换风格、分轨分离或修补。Udio的修补+分轨分离是2026要匹配的功能。

## 概念讲解

![音乐生成:词元语言模型vs扩散模型,2026模型图谱](../assets/music-generation.svg)

### 神经编解码词元上词元语言模型

Meta的**MusicGen**(2023, MIT)和许多衍生:以文本/旋律嵌入为条件,自回归预测EnCodec词元(32 kHz, 4码本),用EnCodec解码。300M-3.3B参数。强基线;超过30秒困难。

**ACE-Step**(开源, 4B XL 2026年4月发布)扩展此用于完整歌曲歌词条件生成。开源社区最接近Suno的东西。

### Mel或潜空间上扩散模型

**Stable Audio**(2023)和**Stable Audio Open**(2024):压缩音频上潜扩散。擅长循环、声音设计、氛围纹理。结构化完整歌曲不强。

**AudioLDM/AudioLDM2**: T2I风格潜扩散文本转音频,泛化到音乐、音效、语音。

### 混合(生产)——Suno、Udio、Lyria

闭源权重。可能是AR编解码语言模型+扩散声码器配专用人声/鼓/旋律头。Suno v5(2026)是ELO 1293质量领袖。Udio v4加修补+分轨分离(贝斯、鼓、人声分开下载)。

### 评估

- **FAD(Fréchet音频距离)。** 用VGGish或PANNs特征的生成vs真实音频分布嵌入级距离。低好。MusicGen small: MusicCaps上4.5 FAD;SOTA ~3.0。
- **音乐性(主观)。** 人类偏好。Suno v5 ELO 1293领先。
- **文本-音频对齐。** 提示词和输出间CLAP分数。
- **音乐性瑕疵。** 离拍过渡、人声短语漂移、30秒后结构丢失。

## 2026模型图谱

| 模型 | 参数 | 长度 | 人声 | 许可 |
|------|------|------|------|------|
| MusicGen-large | 3.3B | 30 s | 无 | MIT |
| Stable Audio Open | 1.2B | 47 s | 无 | Stability非商业 |
| ACE-Step XL(2026年4月) | 4B | > 2 min | 有 | Apache-2.0 |
| YuE | 7B | > 2 min | 有,多语 | Apache-2.0 |
| Suno v5(闭源) | ? | 4 min | 有,ELO 1293 | 商业 |
| Udio v4(闭源) | ? | 4 min | 有+分轨 | 商业 |
| Google Lyria 3(闭源) | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 min | 有 | 商业API |

## 法律格局(2025-2026)

- **Warner Music vs Suno和解。** $500M。WMG现对Suno的AI相似度、音乐权利和用户生成曲目有监督权。Udio有类似UMG和解。
- **欧盟AI法案**+**加州SB 942**: AI生成音乐必须披露。
- **Riffusion/MusicGen**MIT下无合规负担但也无商业人声。

安全发货模式:

1. 仅生成器乐(MusicGen、Stable Audio Open、MIT/CC0输出)。
2. 用商业API(Suno、Udio、ElevenLabs Music)配每次生成许可。
3. 在自有或授权目录上训练(大多数企业最终到这里)。
4. 用水印+元数据标记生成。

## 动手实践

### Step 1:MusicGen生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种大小:`small`(300M,快)、`medium`(1.5B)、`large`(3.3B)。Small够用于"想法是否落地"。

### Step 2:旋律条件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody取色度图并在换音色时保留曲调。有用用于"给我这个旋律作弦乐四重奏版"。

### Step 3:FAD评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算VGGish-嵌入距离。用于风格级回归测试;非人类听众替代。

### Step 4:加入大语言模型-音乐工作流

结合课程7-8想法:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 实际应用

| 目标 | 栈 |
|------|------|
| 器乐声音设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime(闭源) |
| 完整歌曲配人声(商业) | Suno v5或Udio v4配明确许可 |
| 完整歌曲配人声(开源) | ACE-Step XL或YuE |
| 短广告铃声 | MusicGen旋律条件化哼唱参考 |
| 音乐视频背景 | MusicGen + Stable Video扩散模型 |

## 2026仍发货陷阱

- **版权清洗提示词。** "Taylor Swift风格歌"——商业Suno/Udio现过滤这些,开源模型不。加自己过滤列表。
- **超过30秒重复/漂移。** AR模型循环。交叉淡化多个生成,或用ACE-Step保结构连贯。
- **节奏漂移。** 模型偏离BPM。提示词中用BPM标签并用librosa的`beat_track`后过滤。
- **人声可懂性。** Suno优秀;开源模型词常糊。歌词重要用商业API或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用适当立体声重建升级(ezst、Cartesia立体声扩散模型)。

## 产出成果

存`outputs/skill-music-designer.md`。选模型、许可策略、长度/结构计划和披露元数据用于音乐生成部署。

## 练习题

1. **简单。** 跑`code/main.py`。产"生成式"和弦进程+鼓模式作ASCII符号——音乐生成卡通。想回放可用任何MIDI渲染器。
2. **中等。** 装`audiocraft`,用MusicGen-small跨4风格提示词生成10秒片段,测对参考风格集FAD。
3. **困难。** 用ACE-Step(或MusicGen-melody),对同曲调不同音色提示词生成三变体。算CLAP与提示词相似度验证对齐。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| FAD | 音频FID | 真实vs生成嵌入分布间Fréchet距离。 |
| 色度图 | 音高旋律 | 12维每帧向量;旋律条件化输入。 |
| 分轨 | 器材轨道 | 分离贝斯/鼓/人声/旋律作WAV。 |
| 修补 | 重生成一段 | 遮罩时间窗;模型仅重生成那部分。 |
| CLAP | 文本-音频CLIP | 对比音频-文本嵌入;评估文本-音频对齐。 |
| EnCodec | 音乐编解码 | Meta神经编解码MusicGen用;32 kHz, 4码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284)——开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358)——声音设计默认。
- [ACE-Step](https://github.com/ace-step/ACE-Step)——开源4B完整歌曲生成器,2026年4月。
- [Suno v5 platform docs](https://suno.com)——商业质量领袖。
- [AudioLDM2](https://arxiv.org/abs/2308.05734)——音乐+音效潜扩散模型。
- [WMG-Suno和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/)——2025年11月先例。