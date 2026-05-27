# Voice Agent——Pipecat和LiveKit

> Voice agent是2026第一类产category。Pipecat给你Python frame-based pipeline(VAD→STT→LLM→TTS→transport)。LiveKit Agent桥AI模型至用户WebRTC。产latency target premium stack端到端450–600ms。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程12(Workflow Pattern)
**时间:** ~60分钟

## 学习目标

- 描述Pipecat frame-based pipeline:DOWNSTREAM(source→sink)和UPSTREAM(control)。
- 名canonical voice pipeline stage和Pipecat支持transport。
- 释LiveKit Agent两voice agent class(MultimodalAgent、VoicePipelineAgent)和何时每fit。
- Summarize 2026产latency预期和何驱动架构择。

## 问题背景

Voice agent非TTS bolt on text loop。Latency budget brutal(~600ms)、partial audio默认、turn detection是model、transport从telephony SIP至WebRTC。或你建frame-based pipeline(Pipecat)或你lean platform(LiveKit)。

## 概念讲解

### Pipecat(pipecat-ai/pipecat)

- Python frame-based pipeline framework。
- `Frame`→`FrameProcessor` chain。
- 两flow方向:
  - **DOWNSTREAM**——source→sink(audio入、TTS出)。
  - **UPSTREAM**——feedback和control(cancel、metric、barge-in)。
- `PipelineTask`管lifecycle带event(`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`)和observer用于metric/tracing/RTVI。

典型pipeline:

```
VAD(Silero)→STT→LLM(context alternate user/assistant)→TTS→transport
```

Transport:Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flow加结构对话(state machine)。Pipecat Cloud是托管runtime。

### LiveKit Agent(livekit/agent)

- 桥AI模型至用户WebRTC。
- Key concept:`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两voice agent class:
  - **MultimodalAgent**——经OpenAI Realtime或等效直audio。
  - **VoicePipelineAgent**——STT→LLM→TTS cascade;给text-level control。
- 经transformer model semantic turn detection。
- Native MCP integration。
- 经SIP telephony。
- 50+ model无API key经LiveKit Inference;200+更多经plugin。

### 商业platform

Vapi(~450–600ms optimized premium stack)和Retell(~600ms end-to-end跨180 test call)建于这些上。当你欲托管voice stack无WebRTC team时pick platform。

### 何此模式错

- **无barge-in handling。**用户interrupt;agent续talk。需Pipecat UPSTREAM cancel frame、LiveKit等效。
- **STT confidence ignored。**Low-confidence transcript feed LLM作gospel。Gate confidence或request confirmation。
- **TTS mid-sentence cutoff。**Pipeline cancel mid-utterance时、TTS需知或cut audio。
- **Latency budget ignored。**每component加50–200ms。Ship前sum你chain。

### 典型2026 latency

- VAD:20–60ms
- STT partial:100–250ms
- LLM first token:150–400ms
- TTS first audio:100–200ms
- Transport RTT:30–80ms

端到端450–600ms premium。800–1200ms common。>1500ms任何感觉broken。

## 构建

`code/main.py`是frame-based toy pipeline:

- `Frame`类型(audio、transcript、text、tts_audio、control)。
- `Processor` interface带`process(frame)`。
- 五stage pipeline(VAD→STT→LLM→TTS→transport)作scripted processor。
- UPSTREAM cancel frame demo barge-in。

跑:

```
python3 code/main.py
```

Trace显正常flow和barge-in cancel止TTS mid-utterance。

## 使用

- **Pipecat**用于全控——custom processor、Python-first、pluggable provider。
- **LiveKit Agent**用于WebRTC-first deployment和telephony。
- **Vapi/Retell**用于托管voice agent无WebRTC team。
- **OpenAI Realtime/Gemini Live**用于直audio-in/audio-out(MultimodalAgent)。

## 交付成果

`outputs/skill-voice-pipeline.md`scaffold Pipecat形voice pipeline带VAD+STT+LLM+TTS+transport加barge-in handling。

## 练习题

1. 加metric observer你toy pipeline:每stage每秒数frame。Latency何accumulated?
2. 实confidence-gated STT:threshold下、request"你能重复?"
3. 加semantic turn detection:简rule——若transcript end "?"、turn end。
4. 读Pipecat transport doc。换stdlib transport用SmallWebRTCTransport config(stub)。
5. Measure OpenAI Realtime vs STT+LLM+TTS cascade同query。Text-level control何latency成本?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Frame | "Event" | Pipeline typed data unit(audio、transcript、text、control) |
| Processor | "Pipeline stage" | 带`process(frame)`handler |
| DOWNSTREAM | "Forward flow" | Source至sink:audio入、speech出 |
| UPSTREAM | "Feedback flow" | Control:cancel、metric、barge-in |
| VAD | "Voice activity detection" | 测何用户speaking |
| Semantic turn detection | "Smart end-of-turn" | Model-based决用户done |
| MultimodalAgent | "直audio agent" | Audio入、audio出;中间无text |
| VoicePipelineAgent | "Cascade agent" | STT+LLM+TTS;text-level control |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction)——frame-based pipeline、processor、transport
- [LiveKit Agent docs](https://docs.livekit.io/agent/)——WebRTC+voice primitive
- [Vapi](https://vapi.ai/)——托管voice platform
- [Retell AI](https://www.retellai.com/)——托管voice、latency-benchmarked