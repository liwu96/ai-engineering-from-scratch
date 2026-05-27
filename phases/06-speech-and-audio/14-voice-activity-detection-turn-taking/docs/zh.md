# 语音活动检测与轮次交接——Silero、Cobra和刷新技巧

> 每个语音智能体生死于两决策:用户此刻在说话吗,他们说完了吗?VAD回答第一。轮次检测(VAD+静音悬停+语义终点模型)回答第二。任一错则助手要么裁用户要么永不停。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程11(实时音频)、阶段6课程12(语音助手)
**时间:** ~45分钟

## 问题背景

语音智能体每20毫秒块做三不同决策:

1. **这帧是语音吗?** ——VAD。二元,每帧。
2. **用户已开始新话语吗?** ——起拍检测。
3. **用户已结束吗?** ——终点检测(轮次结束)。

朴素答(能量阈值)任何噪声失败——交通、键盘、人群嘈杂。2026答:Silero VAD(开源、深学习)+轮次检测模型(语义终点)+VAD校准静音悬停。

## 概念讲解

![VAD级联:能量→Silero→轮次检测器→刷新技巧](../assets/vad-turn-taking.svg)

### 三层VAD级联

**层1:能量门。** 最便宜。-40 dBFS阈值RMS。滤明显静音但阈值上任何噪声触发。

**层2:Silero VAD**(2020-2026, MIT)。1M参数。6000+语言训。单CPU线程每30毫秒块~1毫秒跑。5% FPR下87.7% TPR。开源默认。

**层3:语义轮次检测器。** LiveKit轮次检测模型(2024-2026)或自己小分类器。分"句中停"vs"说完"。用语言上下文(语调+近词),不止静音。

### 关键参数及默认

- **阈值。** Silero输出概率;语音分类>0.5(默认)或>0.3(敏感)。低阈值=少首词裁剪,多假正。
- **最小语音时长。** 拒短于250毫秒语音——通常是咳嗽或椅子噪声。
- **静音悬停(终点检测)。** VAD返0后,等500-800毫秒再声轮次结束。太短→打断用户。太长→感迟钝。
- **预滚动缓冲。** VAD触发前保300-500毫秒音频。防"嘿"裁剪。

### 刷新技巧(Kyutai 2025)

流式STT模型有前瞻延迟(Kyutai STT-1B 500毫秒,STT-2.6B 2.5秒)。正常语音结束后等那么长才转录。刷新技巧:VAD触发音结束时,**发刷新信号给STT**强立即输出。STT~4×实时处理,500毫秒缓冲~125毫秒完。

端到端:125毫秒VAD+刷新STT=对话延迟。

### 2026 VAD比较

| VAD | TPR @ 5% FPR | 延迟 | 许可 |
|------|--------------|------|------|
| WebRTC VAD(Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD(2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD(Picovoice) | 98.9% | ~1 ms | 商业 |
| pyannote分割 | 95% | ~10 ms | MIT-ish |

Silero是正确默认。Cobra是合规/精度升级。能量仅VAD无位置于2026生产。

## 动手实践

### Step 1:能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2:Python中Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3:轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4:刷新技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT(Kyutai、Deepgram、AssemblyAI)须支持刷新才工作。Whisper流式不支持——基于块总等块。

## 实际应用

| 情况 | VAD选择 |
|------|----------|
| 开源、快、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设端(手机) | Silero VAD ONNX |
| 研究/分离 | pyannote分割 |
| 零依赖回退 | WebRTC VAD(遗留) |
| 需轮次结束质量 | Silero + LiveKit轮次检测器层叠 |

经验法则:永不发货能量仅VAD除非真无其他选择。

## 陷阱

- **固定阈值。** 安静工作,噪声失败。设备校准或换Silero。
- **太短静音悬停。** 智能体句中打断。500-800毫秒是对话语音甜点。
- **太长悬停。** 感迟钝。目标用户A/B测试。
- **无预滚动缓冲。** 用户音频首200-300毫秒丢。总保滚动预滚动。
- **忽略语义终点。** "嗯,让我想..."含长停。用户恨中途被打。用LiveKit轮次检测器或类似。

## 产出成果

存`outputs/skill-vad-tuner.md`。为工作负载选VAD模型、阈值、悬停、预滚动和轮次检测策略。

## 练习题

1. **简单。** 跑`code/main.py`。模拟语音+静音+语音+咳嗽序列并测三层VAD。
2. **中等。** 装`silero-vad`,处理5分钟录音,调阈值最小化首词裁剪和假触发。报精确率/召回率。
3. **困难。** 构迷你轮次检测器:Silero VAD + 最近10词嵌入上3层MLP(用sentence-transformers)。在手标轮次结束数据集训。比Silero仅超10% F1。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| VAD | 语音检测器 | 每帧二元:这是语音吗? |
| 轮次检测 | 终点检测 | VAD+静音悬停+语义终点。 |
| 静音悬停 | 语音后等 | 声轮次结束前等时间;500-800毫秒。 |
| 预滚动 | 预语音缓冲 | VAD触发前保300-500毫秒音频。 |
| 刷新技巧 | Kyutai技巧 | VAD→刷新STT→125毫秒而非500毫秒延迟。 |
| 语义终点 | "他们有意停吗?" | 看词而非仅静音ML分类器。 |
| TPR @ FPR 5% | ROC点 | 标准VAD基准;Silero 87.7%, WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad)——参考开源VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/)——商业精度领袖。
- [Kyutai——Unmute+刷新技巧](https://kyutai.org/stt)——亚200毫秒工程技巧。
- [LiveKit——轮次检测](https://docs.livekit.io/agents/logic/turns/)——生产语义终点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/)——遗留基线。
- [pyannote分割](https://github.com/pyannote/pyannote-audio)——分离级分割。