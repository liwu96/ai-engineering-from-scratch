# 实时音频处理

> 批次管道处理文件。实时管道在下20毫秒到达前处理下20毫秒。每个对话AI、广播工作室和电话机器人生死由这个延迟预算决定。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图)、阶段6课程04(ASR)、阶段6课程07(TTS)
**时间:** ~75分钟

## 问题背景

你想要感觉活着的声音助手。人类对话轮次延迟约230毫秒(静音到响应)。超500毫秒感觉机械;超1500毫秒感觉坏了。2026完整**听→理解→响应→说**循环预算:

| 阶段 | 预算 |
|------|------|
| 麦→缓冲 | 20 ms |
| VAD | 10 ms |
| ASR(流式) | 150 ms |
| 大语言模型(首个词元) | 100 ms |
| TTS(首块) | 100 ms |
| 渲染→扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi(Kyutai, 2024)测200毫秒全双工。GPT-4o-realtime(2024)测~320毫秒。2022级联管道发货2500毫秒。10倍改进来自三技术:(1)到处流式,(2)部分结果异步管道,(3)可中断生成。

## 概念讲解

![流式音频管道配环形缓冲、VAD门、中断](../assets/real-time.svg)

**帧/块/窗。** 实时音频流为固定大小块。常见选:20毫秒(16 kHz下320样本)。下游一切必须跟上这个节奏。

**环形缓冲。** 固定大小循环缓冲。生产者线程写新帧,消费者线程读。防止热路径分配。大小≈最大延迟×采样率;2秒16 kHz环=32,000样本。

**VAD(语音活动检测)。** 无人说话时门控下游工作。Silero VAD 4.0(2024)在CPU上每30毫秒帧<1毫秒跑。87.7% TPR @ 5% FPR。`webrtcvad`是较老替代。

**流式ASR。** 音频到达时发部分转录模型。Parakeet-CTC-0.6B流式模式(NeMo, 2024)320毫秒延迟下2–5% WER。Whisper-Streaming(Macháček等, 2023)分块Whisper用于近流式~2秒延迟。

**中断。** 用户在助手说话时说话,必须检测打断、停TTS、丢弃剩余大语言模型输出。全在100毫秒内,否则用户感觉聋助手。

**WebRTC Opus传输。** 20毫秒帧,48 kHz,自适应比特率8–128 kbps。浏览器和移动标准。LiveKit、Daily.co、Pion是2026构语音应用栈。

**抖动缓冲。** 网络包乱序/迟到。抖动缓冲重排平滑;太小→可听间隙,太大→延迟。典型60–80毫秒。

### 常见坑

- **线程竞争。** Python GIL+重模型可饿死音频线程。用C回调音频库(sounddevice、PortAudio)并让Python离热路径。
- **采样率转换延迟。** 管道内重采样加5–20毫秒。要么预重采样要么用零延迟重采样器(PolyPhase、`soxr_hq`)。
- **TTS预热。** 即使快TTS如Kokoro首请求有100–200毫秒预热。缓存模型+首次真实轮次前用假跑预热。
- **回声消除。** 无AEC,TTS输出重入麦并触发ASR于机器人自己声音。WebRTC AEC3是开源默认。

## 动手实践

### Step 1:环形缓冲

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz下32,000样本=2秒。

### Step 2:VAD门

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产用Silero VAD替代:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3:流式ASR

```python
# Parakeet-CTC-0.6B流式经NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4:中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 打断
        # 然后喂给流式ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖异步I/O和可取消TTS流式。WebRTC peerconnection.stop()于音频轨是规范方式。

## 实际应用

2026栈:

| 层 | 选择 |
|------|------|
| 传输 | LiveKit(WebRTC)或Pion(Go) |
| VAD | Silero VAD 4.0 |
| 流式ASR | Parakeet-CTC-0.6B或Whisper-Streaming |
| 大语言模型首词元 | Groq、Cerebras、vLLM流式 |
| 流式TTS | Kokoro或ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API或Moshi |

## 陷阱

- **缓冲500毫秒求安全。** 缓冲*是*延迟底线。缩小它。
- **不钉线程。** 音频回调在低于UI优先级线程=负载下卡顿。
- **TTS块太小。** 子200毫秒块使声码器瑕疵可听。320毫秒块是甜点。
- **无抖动缓冲。** 真网络抖动;无平滑得爆音。
- **单次错误处理。** 音频管道必须崩溃proof。一异常杀会话。

## 产出成果

存`outputs/skill-realtime-designer.md`。设计实时音频管道配每阶段具体延迟预算。

## 练习题

1. **简单。** 跑`code/main.py`。模拟环形缓冲+能量VAD;打印假10秒流每阶段延迟。
2. **中等。** 用`sounddevice`,构穿通循环20毫秒帧处理麦并每帧打印VAD状态。
3. **困难。** 用`aiortc`构全双工回声测试:浏览器→WebRTC→Python→WebRTC→浏览器。用1 kHz脉冲测玻璃到玻璃延迟。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 环形缓冲 | 循环队列 | 固定大小、无锁(或SPSC锁)FIFO用于音频帧。 |
| VAD | 静音门 | 模型或启发式标记语音vs非语音。 |
| 流式ASR | 实时STT | 音频到达时发部分文本;有界前瞻。 |
| 抖动缓冲 | 网络平滑器 | 队列重排乱序包;典型60–80毫秒。 |
| AEC | 回声消除 | 减扬声器到麦反馈路径。 |
| 打断 | 用户中断 | 系统检测TTS中途用户语音;必须取消播放。 |
| 全双工 | 同时双向 | 用户和机器人可同时说话;Moshi是全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)——分块近流式Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf)——全双工200毫秒延迟。
- [LiveKit Agents框架(2024)](https://docs.livekit.io/agents/)——生产音频智能体编排。
- [Silero VAD仓库](https://github.com/snakers4/silero-vad)——亚1毫秒VAD, Apache 2.0。
- [WebRTC AEC3论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/)——开源下回声消除。