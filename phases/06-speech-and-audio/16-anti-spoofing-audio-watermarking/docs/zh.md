# 语音反欺骗与音频水印——ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆发货快于防御。2026生产语音系统需两物:检测器(AASIST、RawNet2)分类真vs假语音,和水印(AudioSeal)经受压缩和编辑。发货两者否则不发货声音克隆。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程06(说话人识别)、阶段6课程08(声音克隆)
**时间:** ~75分钟

## 问题背景

三相关防御:

1. **反欺骗/深度伪造检测。** 给音频片段,是合成还是真?ASVspoof基准(ASVspoof 2019→2021→5)是金标准。
2. **音频水印。** 在生成音频中嵌入检测器可后取不可感知信号。AudioSeal(Meta)和WavMark是开源选项。
3. **认证来源。** 音频文件+元数据加密签名。C2PA/内容真实性倡议。

检测处理不合作对手。水印处理合规——AI生成音频应可识别为AI生成。2026两者均需。

## 概念讲解

![反欺骗vs水印vs来源——三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5——2024-2025基准

最大变化:

- **众包数据**(非工作室干净)——现实条件。
- **~2000说话人**(vs前~100)。
- **32攻击算法。** TTS+声音转换+对抗扰动。
- **两赛道。** 反措施(CM)独立检测;欺骗鲁棒ASV(SASV)用于生物识别系统。

ASVspoof 5上最新:~7.23% EER。旧ASVspoof 2019 LA:0.42% EER。真实世界部署:期望野外片段5-10% EER。

### AASIST和RawNet2——检测模型族

**AASIST**(2021, 更至2026)。频谱特征上图注意力。ASVspoof 5反措施任务当前SOTA。

**RawNet2。** 原始波形上卷积前端+TDNN骨干。简单基线;微调后仍竞争。

**NeXt-TDNN + SSL特征。** 2025变体:ECAPA风格+WavLM特征+焦点损失。ASVspoof 2019 LA达0.42% EER。

### AudioSeal——2024水印默认

Meta的**AudioSeal**(2024年1月, v0.2 2024年12月)。关键设计:

- **定位。** 16 kHz样本分辨率(1/16000秒)每帧检测水印。
- **生成器+检测器联合训练。** 生成器学嵌入不可听信号;检测器学通过增强找它。
- **鲁棒。** 经MP3/AAC压缩、EQ、速度偏±10%、噪声混+10 dB SNR。
- **快。** 检测器485×实时跑;比WavMark快1000×。
- **容量。** 16位payload(可编码模型ID、生成时间戳、用户ID)嵌入每话语。

### WavMark

AudioSeal前开源基线。可逆神经网络,32位/秒。问题:

- 同步暴力慢。
- 高斯噪声或MP3压缩可移。
- 不实时友好。

### WaveVerify(2025年7月)

解决AudioSeal弱点——特时间操作(反转、速度)。用FiLM基生成器+专家混合检测器。标准攻击与AudioSeal竞争;处理时间编辑。

### 对手利用缺口

AudioMarkBench:"音高偏移下,所有水印显示比特恢复准确率低于0.6,接近完全移除。"**音高偏移是通用攻击。**无2026水印对激进音高修改完全鲁棒。这是需要检测(AASIST)配水印原因。

### C2PA/内容真实性倡议

非ML技术——清单格式。音频文件载创建工具、作者、日期加密签名元数据。Audobox/Seamless用它。来源好;坏行为者重编码并剥元数据则无用。

## 动手实践

### Step 1:简单频谱特征检测器(玩具)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音常异常平高频能量。生产检测器用AASIST,非此。但直觉持。

### Step 2:AudioSeal嵌入+检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1]中浮点——水印存在概率
# decoded_payload: 16位;与嵌入payload匹配
```

### Step 3:评估——EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4:生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每生成发货:(1)水印,(2)签名清单,(3)保留策略合规审计日志。

## 实际应用

| 用例 | 防御 |
|------|------|
| 发货TTS/声音克隆 | 每输出AudioSeal嵌入(不可协商) |
| 生物识别语音解锁 | AASIST + ECAPA集成;活体挑战 |
| 呼叫中心欺诈检测 | 入呼20%样上AASIST |
| 播客真实性 | 上传时C2PA签名,AI生成则AudioSeal |
| 研究/训练检测器 | ASVspoof 5训/开发/评估集 |

## 陷阱

- **水印但从不跑检测器。** 无意义。CI中发货检测器。
- **检测无校准。** AASIST在ASVspoof LA过拟合;真实世界精度降。领域上校准。
- **音高偏移缺口。** 激进音高偏移移大多水印。有检测回退。
- **元数据剥-重托管。** C2PA易绕过重编码。总加密+感知(水印)防御一起加。
- **活体作检测。** 让用户说随机短语。防重放攻击但非实时克隆。

## 产出成果

存`outputs/skill-spoof-defender.md`。为语音生成部署选检测模型、水印、来源清单和操作剧本。

## 练习题

1. **简单。** 跑`code/main.py`。玩具检测器+合成音频上玩具水印嵌入/检测。
2. **中等。** 装`audioseal`,在TTS输出中嵌入16位payload,重解码。用噪声破坏音频并测比特恢复准确率。
3. **困难。** ASVspoof 2019 LA上微调RawNet2或AASIST。测EER。在F5-TTS生成片段保留集上测——看OOD检测如何退化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ASVspoof | 基准 | 双年挑战;2024=ASVspoof 5。 |
| CM(反措施) | 检测器 | 分类器:真语音vs合成/转换。 |
| SASV | 说话人验证+CM | 集成生物识别+欺骗检测。 |
| AudioSeal | Meta水印 | 定位、16位payload、比WavMark快485×。 |
| 比特恢复准确率 | 水印存活 | 攻击后payload比特恢复分数。 |
| C2PA | 来源清单 | 创建/作者加密元数据。 |
| AASIST | 检测器族 | 图注意力基反欺骗SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264)——水印默认。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150)——时间攻击MoE检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200)——SOTA检测骨干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf)——鲁棒性评估。
- [C2PA规格](https://c2pa.org/specifications/specifications/)——来源清单格式。