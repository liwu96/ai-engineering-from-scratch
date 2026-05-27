# 声音克隆与转换

> 声音克隆用别人的声音读你的文本。声音转换把你的声音重写成别人的同时保留你说的内容。两者都依赖同一个基础:把说话人身份和内容分离。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程06(说话人识别)、阶段6课程07(TTS)
**时间:** ~75分钟

## 问题背景

2026年,5秒音频片段就足以在消费级GPU上产生任何人声音的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox都提供零样本或少样本克隆。这项技术既是福音(无障碍TTS、配音、辅助声音)也是武器(诈骗电话、政治深度伪造、知识产权盗窃)。

两个密切相关任务:

- **声音克隆(TTS侧):** 文本 + 5秒参考声音 → 该声音的音频。
- **声音转换(语音侧):** 源音频(A说X) + B的参考声音 → B说X的音频。

两者都把波形分解为(内容、说话人、韵律)并从一个源取内容与另一个源取说话人重新组合。

2026年关键约束:**水印和同意门槛在欧盟(AI法案,2026年8月生效)和加州(AB 2905,2025年生效)法律强制要求**。你的管道必须发出不可听水印并拒绝未经同意的克隆。

## 概念讲解

![声音克隆vs转换:分解、交换说话人、重组](../assets/voice-cloning.svg)

**零样本克隆。** 把5秒片段传给在数千说话人上训练的模型。说话人编码器把片段映射到说话人嵌入;TTS解码器基于该嵌入加文本生成条件。

使用:F5-TTS(2024)、YourTTS(2022)、XTTS v2(2024)、OpenVoice v2(2024)。

**少样本微调。** 录制目标声音5-30分钟。LoRA微调基础模型一小时。质量从"还行"跃升到"难以区分"。Coqui和ElevenLabs都支持此模式;社区用F5-TTS实现。

**声音转换(VC)。** 两族:

- **识别-合成。** 运行ASR类模型提取内容表示(如软音素后验、PPG),然后用目标说话人嵌入重合成。对语言和口音鲁棒。使用KNN-VC(2023)、Diff-HierVC(2023)。
- **解耦。** 训练自编码器在瓶颈潜空间分离内容、说话人和韵律。推理时交换说话人嵌入。质量较低但更快。使用AutoVC(2019)、VITS-VC变体。

**神经编解码器克隆(2024+)。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——把音频当作来自SoundStream/EnCodec的离散词元,在编解码器词元上训练大型自回归或流匹配模型。短提示词上质量媲美ElevenLabs。

### 伦理要点,不是附加项

**水印。** PerTh(Perth)和SilentCipher(2024)在音频中嵌入~16-32位ID不可感知。经受重编码、流式和常见编辑。生产就绪开源。

**同意门槛。** 每个克隆输出必须配可验证同意记录。"我,Rohit,于2026-04-22,授权此声音用于X目的。"存储防篡改日志。

**检测。** AASIST、RawNet2和Wav2Vec2-AASIST作为检测器发货。ASVspoof 2025挑战发布ElevenLabs、VALL-E 2和Bark输出的最新检测器EER 0.8–2.3%。

### 2026数据

| 模型 | 零样本? | SECS(目标相似度) | WER(可懂性) | 参数 |
|------|---------|-----------------|-------------|------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70对大多数听众与目标难以区分。

## 动手实践

### Step 1:识别-合成分解(main.py代码演示)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单;实现复杂性在`tts_model`和说话人编码器。

### Step 2:F5-TTS零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频完全匹配;不匹配破坏对齐。

### Step 3:KNN-VC声音转换

```python
import torch
from knnvc import KNNVC  # 2023模型, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC运行WavLM提取源和目标池每帧嵌入,然后用池中最近邻替换每源帧。非参数化,一分钟目标语音即可工作。

### Step 4:嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回payload字节
```

~32位payload,MP3重编码和轻微噪声后可检测。

### Step 5:同意门槛

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 实际应用

2026栈:

| 情况 | 选择 |
|------|------|
| 5秒零样本克隆,开源 | F5-TTS或OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换(重写) | KNN-VC或Diff-HierVC |
| 多说话人微调 | StyleTTS 2 +说话人适配器 |
| 跨语言克隆 | XTTS v2或VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转录未对齐。** F5-TTS等要求参考文本与参考音频完全匹配,标点也要。
- **混响参考。** 回声破坏克隆。录制干声、近麦。
- **情绪不匹配。** "愉快"参考训练产生一切愉快克隆。匹配参考情绪到目标用途。
- **语言泄漏。** 克隆英语说话人然后让模型说法语常带口音;用跨语言模型(XTTS、VALL-E X)。
- **无水印。** 2026年8月起欧盟法律不可发货。

## 产出成果

存`outputs/skill-voice-cloner.md`。设计克隆或转换管道配同意门槛+水印+质量目标。

## 练习题

1. **简单。** 跑`code/main.py`。演示说话人嵌入交换,计算交换前后两"说话人"间余弦。
2. **中等。** 用OpenVoice v2克隆自己的声音。测量参考和克隆间SECS。通过Whisper测CER。
3. **困难。** 对20个克隆应用SilentCipher水印,跑128 kbps MP3编码+解码,检测payload。报比特准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 零样本克隆 | 5秒够用 | 预训练模型+说话人嵌入;无训练。 |
| PPG | 音素后验图 | 每帧ASR后验用作语言无关内容表示。 |
| KNN-VC | 最近邻转换 | 用最近目标池帧替换每源帧。 |
| 神经编解码TTS | VALL-E风格 | 在EnCodec/SoundStream词元上AR模型。 |
| 水印 | 不可听签名 | 音频中嵌入比特,经受重编码。 |
| SECS | 克隆保真度 | 目标和克隆说话人嵌入间余弦。 |
| AASIST | 深度伪造检测 | 反欺骗模型;检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885)——开源SOTA零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111)和[VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370)——神经编解码TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879)——解耦基声音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975)——检索基VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher)——生产就绪32位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/)——检测器vs合成器军备竞赛,2026更新。