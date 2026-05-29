# 构建语音助手管道——阶段6顶点

> 将课程01-11的全部内容拼接起来。构建一个能听、能推理、能回话的语音助手。2026年这是一个已解决的工程问题，而不是研究问题——但集成细节决定能否上线。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程04、05、06、07、11;阶段11课程09(函数调用);阶段14课程01(智能体循环)
**时间:** ~120分钟

## 问题背景

构建端到端助手:

1. 捕麦输入(16 kHz单声道)。
2. 检测用户语音开始/结束。
3. 流式转录。
4. 传转录给可调用工具的大语言模型(定时器、天气、日历)。
5. 流式大语言模型文本到TTS。
6. 播音频回用户。
7. 用户中途打断响应时停。

延迟目标:笔记本CPU上用户说完话800毫秒内首个TTS音频字节。质量目标:无漏词、无静音幻影字幕、无声音克隆泄漏、无提示词注入成功。

## 概念讲解

![语音助手管道:麦→VAD→STT→大语言模型+工具→TTS→扬声器](../assets/voice-assistant.svg)

### 七组件

1. **音频捕获。** 麦→16 kHz单声道→20毫秒块。通常Python用`sounddevice`或生产用原生AudioUnit/ALSA/WASAPI。
2. **VAD(课程11)。** Silero VAD @阈值0.5,最小语音250毫秒,静音悬停500毫秒。信号"开始"和"结束"。
3. **流式STT(课程4-5)。** Whisper-streaming、Parakeet-TDT或Deepgram Nova-3(API)。部分+最终转录。
4. **带工具调用大语言模型。** GPT-4o/Claude 3.5/Gemini 2.5 Flash。工具JSON schema。流式词元。
5. **流式TTS(课程7)。** Kokoro-82M(最快开源)或Cartesia Sonic(商业)。20个大语言模型词元后启TTS。
6. **播放。** 扬声器出;opus编码用于低带宽网络。
7. **中断处理器。** TTS播放时VAD触发则停播放、取消大语言模型、重启STT。

### 三失败模式你会遇

1. **首词裁剪。** VAD起拍晚。"嘿"丢失。启阈值0.3,非0.5。
2. **中途响应打断混乱。** 用户打断后大语言模型继续生成;助手盖用户说话。连VAD→取消-大语言模型。
3. **静音幻觉。** Whisper静预热帧输出"Thanks for watching"。总VAD门。

### 2026生产参考栈

| 栈 | 延迟 | 许可 | 备注 |
|------|------|------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业API | 2026行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 多开源 | DIY友好 |
| Moshi(全双工) | 200-300 ms | CC-BY 4.0 | 单模型;不同架构,课程15 |
| Vapi / Retell(托管) | 300-500 ms | 商业 | 最快上线;定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘 |

## 动手实践

### Step 1:麦捕获配分块(伪代码)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2:VAD门控轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3:流式STT→大语言模型→TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4:大语言模型循环内工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5:中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 实际应用

见`code/main.py`跑模拟用stub模型连七组件,无硬件也能看管道形。真实实现换stub:

- `silero-vad`(`pip install silero-vad`)
- `deepgram-sdk`或`openai-whisper`
- `openai`(`gpt-4o`)或`anthropic`
- `kokoro`或`cartesia`
- `sounddevice`用于I/O

## 陷阱

- **日志PII永久。** 全轮音频大多管辖PII。30天保留,静态加密。
- **无打断。** 用户会打断。助手必须停说话。
- **阻塞TTS。** 同步TTS阻塞事件循环。用async或单独线程。
- **无工具调用错误处理。** 工具失败。大语言模型必须得回错误+重试一次,然后优雅降级。
- **过度幻觉过滤。** 过滤则助手重复"我帮不了这个"。欠过滤则说任何东西。保留集校准。
- **无唤醒词选项。** 常听是隐私风险。加唤醒词门(Porcupine或openWakeWord)。

## 产出成果

存`outputs/skill-voice-assistant-architect.md`。给定预算+规模+语言+合规约束,产完整栈规格。

## 练习题

1. **简单。** 跑`code/main.py`。用stub模块模拟一个完整轮端到端并打印每阶段延迟。
2. **中等。** 换STT stub为预录`.wav`上真实Whisper模型。测WER和端到端延迟。
3. **困难。** 加工具调用:实现`get_weather`(任意API)和`set_timer`。路由大语言模型通过工具并验证用户说"设5分钟定时器"时正确函数触发且口述回复确认。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 轮次 | 用户+助手往返 | 一个VAD边界用户语音+一个大语言模型-TTS响应。 |
| 打断 | 中途打断 | 助手说话时用户说话;助手停。 |
| 唤醒词 | "嘿助手" | 短关键词检测器;Porcupine、Snowboy、openWakeWord。 |
| 终点检测 | 轮次结束 | VAD+最小静音决定用户说完。 |
| 预滚动 | 预语音缓冲 | VAD触发前保200-400毫秒音频免首词裁剪。 |
| 工具调用 | 函数调用 | 大语言模型发JSON;运行时分发;结果循环反馈。 |

## 延伸阅读

- [LiveKit——语音智能体快速入门](https://docs.livekit.io/agents/)——生产级参考。
- [Pipecat——语音智能体示例](https://github.com/pipecat-ai/pipecat)——DIY友好框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)——托管语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)——全双工参考(课程15)。
- [Porcupine唤醒词](https://picovoice.ai/products/porcupine/)——唤醒词门控。
- [Anthropic——工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)——大语言模型函数调用。