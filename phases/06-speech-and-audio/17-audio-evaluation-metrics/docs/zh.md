# 音频评估——WER、MOS、UTMOS、MMAU、FAD和开放排行榜

> 你不能发货你不能测量的。本课命名2026每个音频任务指标:ASR(WER、CER、RTFx)、TTS(MOS、UTMOS、SECS、WER-on-ASR-round-trip)、音频语言(MMAU、LongAudioBench)、音乐(FAD、CLAP)、说话人(EER)。加你比较的排行榜。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段6课程04、06、07、09、10;阶段2课程09(模型评估)
**时间:** ~60分钟

## 问题背景

每个音频任务有多指标,各测不同轴。用错指标是你发货仪表盘上看棒但生产中看糟的模型原因。2026规范列表:

| 任务 | 主要 | 辅助 |
|------|------|------|
| ASR | WER | CER、RTFx、首词元延迟 |
| TTS | MOS/UTMOS | SECS、WER-on-ASR-round-trip、CER、TTFA |
| 声音克隆 | SECS(ECAPA余弦) | MOS、CER |
| 说话人验证 | EER | minDCF、工作点FAR/FRR |
| 分离 | DER | JER、说话人混淆 |
| 音频分类 | top-1、mAP | macro F1、每类召回 |
| 音乐生成 | FAD | CLAP、听板MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench、AudioCaps FENSE |
| 流式S2S | 延迟P50/P95 | WER、MOS |

## 概念讲解

![音频评估矩阵——指标vs任务vs 2026排行榜](../assets/eval-landscape.svg)

### ASR指标

**WER(词错误率)。** `(S + D + I) / N`。评分前小写、剥标点、归一化数字。用`jiwer`或OpenAI `whisper_normalizer`。<5%=阅读语音人级。

**CER(字符错误率)。** 同公式,字符级。用于音调语言(普通话、粤语)词分割模糊。

**RTFx(逆实时因子)。** 墙钟秒处理音频秒。高好。Parakeet-TDT达3380×。Whisper-large-v3 ~30×。

**首词元延迟。** 音频输入到首转录词元墙钟。流式关键。Deepgram Nova-3:~150毫秒。

### TTS指标

**MOS(平均意见分)。** 1-5人评。金标准但慢。每模型20+听众/样本,100+样本。

**UTMOS(2022-2026)。** 学习MOS预测器。标准基准与人类MOS相关~0.9。F5-TTS: UTMOS 3.95;真实: 4.08。

**SECS(说话人编码器余弦相似度)。** 声音克隆用。参考和克隆输出ECAPA嵌入余弦。>0.75=可认克隆。

**WER-on-ASR-round-trip。** TTS输出上跑Whisper,算输入文本WER。捕获可懂性回归。2026 SOTA:<2% CER。

**TTFA(首音频时间)。** 墙钟延迟。Kokoro-82M:~100毫秒;F5-TTS:~1秒。

### 声音克隆专用

**SECS + MOS + CER**三重。克隆高分SECS低MOS意味音色对但不自然;反意味自然声但错说话人。

### 说话人验证

**EER(等错误率)。** 假接受率等于假拒绝率阈值。VoxCeleb1-O上ECAPA:0.87%。

**minDCF(最小检测成本)。** 选工作点加权成本(常FAR=0.01)。比EER更生产相关。

### 分离

**DER(分离错误率)。** `(FA + Miss + Confusion) / total_speaker_time`。漏语音+假警语音+说话人混淆,各作分数。AMI会议:DER ~10-20%现实。pyannote 3.1 +精确率-2商业:好录音<10% DER。

**JER(Jaccard错误率)。** DER替代,短段偏鲁棒。

### 音频分类

多标签:**mAP(平均精确率)**全类。AudioSet: BEATs-iter3 0.548 mAP。

多类独占:**top-1、top-5准确率**。Speech Commands v2: 99.0% top-1(Audio-MAE)。

不平衡:**macro F1**+**每类召回**。报每类——聚合准确率隐藏哪些类失败。

### 音乐生成

**FAD(Fréchet音频距离)。** 真实vs生成音频VGGish-嵌入分布距离。MusicCaps上MusicGen-small: 4.5。MusicLM: 4.0。低好。

**CLAP分数。** CLAP嵌入文本-音频对齐分数。>0.3=合理对齐。

**听板MOS。** 消费级音乐仍最终裁决。Suno v5 TTS Arena ELO 1293(从配人偏好)。

### 音频语言基准

**MMAU(大规模多音频理解)。** 10k音频问答对。

**MMAU-Pro。** 1800难项,四类:语音/声/音乐/多音频。4路随机机会25%。Gemini 2.5 Pro总体~60%;所有模型多音频~22%。

**LongAudioBench。** 多分钟片段配语义查询。Audio Flamingo Next超Gemini 2.5 Pro。

**AudioCaps/Clotho。** 字幕基准。SPICE、CIDEr、FENSE指标。

### 流式语音到语音

**延迟P50/P95/P99。** 用户语音结束到首可听响应墙钟。Moshi: 200毫秒;GPT-4o实时: 300毫秒。

**输出WER/MOS。**

**打断响应性。** 用户打断到助手静音时间。目标<150毫秒。

### 2026排行榜

| 排行榜 | 赛道 | URL |
|------|------|------|
| Open ASR Leaderboard(HF) | 英语+多语+长形 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena(HF) | 英语TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT,配票ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM推理 | `mmaubenchmark.github.io` |
| SpeakerBench/VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU音乐子集 | 音乐LALM | (MMAU内) |
| HEAR基准 | 自监督音频 | `hearbenchmark.com` |

## 动手实践

### Step 1:WER配归一化

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2:TTS往返WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3:声音克隆SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4:音乐生成FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5:说话人验证EER(课程6同码)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 实际应用

配每部署固定评估Harness,每模型更新跑。三基本规则:

1. **评分前归一化。** 小写、标点剥、数字扩。报归一化规则。
2. **报分布,非平均。** 延迟P50/P95/P99。分类每类召回。MMAU每类。
3. **跑一个规范公共基准。** 即使生产数据不同,在Open ASR/TTS Arena/MMAU上报告让评审苹果对苹果比。

## 陷阱

- **UTMOS外推。** VCTK风格干净语音训;噪声/克隆/情绪音频评分差。
- **MOS板偏。** 20 Amazon Mechanical Turk工 ≠ 20目标用户。高风险付领域板。
- **FAD依赖参考集。** 跨模型比同一参考分布。
- **聚合WER。** 总体5% WER可隐藏口音语音30% WER。按人口切片报。
- **公共基准饱和。** 大多数前沿模型标准基准近天花板。建反映流量内部保留集。

## 产出成果

存`outputs/skill-audio-evaluator.md`。为任意音频模型发布选指标、基准和报告格式。

## 练习题

1. **简单。** 跑`code/main.py`。玩具输入上算WER/CER/EER/SECS/FAD-ish/MMAU-ish。
2. **中等。** 构TTS往返WER harness。Kokoro或F5-TTS输出跑Whisper。50提示词上算WER。标WER>10%提示词。
3. **困难。** 课程10 LALM选择在MMAU-Pro语音+多音频子集(各50项)评分。报每类准确率并与发布数比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| WER | ASR分数 | 归一化后词级`(S+D+I)/N`。 |
| CER | 字符WER | 音调语言或字符级系统用。 |
| MOS | 人类意见 | 1-5评分;20+听众×100样本。 |
| UTMOS | ML MOS预测器 | 学习模型;与人MOS相关~0.9。 |
| SECS | 声音克隆相似度 | 参考和克隆ECAPA余弦。 |
| EER | 说话人验证分数 | FAR=FRR阈值。 |
| DER | 分离分数 | (FA+Miss+Confusion)/总。 |
| FAD | 音乐生成质量 | VGGish嵌入上Fréchet距离。 |
| RTFx | 吞吐量 | 墙钟秒每音频秒。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer)——WER/CER库配归一化工具。
- [UTMOS(Saeki等 2022)](https://arxiv.org/abs/2204.02152)——学习MOS预测器。
- [Fréchet音频距离(Kilgour等 2019)](https://arxiv.org/abs/1812.08466)——音乐生成标准。
- [Open ASR排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)——2026实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)——人票TTS排行榜。
- [MMAU-Pro基准](https://mmaubenchmark.github.io/)——LALM推理排行榜。
- [HEAR基准](https://hearbenchmark.com/)——音频SSL基准。