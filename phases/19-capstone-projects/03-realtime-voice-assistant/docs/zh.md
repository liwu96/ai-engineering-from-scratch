# 毕业项目 03 —— 实时语音助手 (ASR到LLM到TTS)

> 感觉对的语音agent端到端延迟800ms下、知你何时停说话、处barge-in、可call tool不stall。Retell、Vapi、LiveKit Agents、和Pipecat全2026达此bar。此同形态：流ASR、turn-detector、流LLM、和流TTS、全经WebRTC wired带每hop激进延迟预算。建一个、测WER和MOS和false-cutoff rate、并于packet loss下run。

**类型:** 毕业项目
**语言:** Python (agent + pipeline)、TypeScript (web client)
**前置要求:** 第6阶段(语音和音频)、第7阶段(transformers)、第11阶段(LLM工程)、第13阶段(工具)、第14阶段(agent)、第17阶段(基础设施)
**涉及阶段:** P6 · P7 · P11 · P13 · P14 · P17
**时间:** 30小时

## 问题背景

语音是2025-2026最快移动AI UX类别。技天花板每季度降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0、和Pipecat 0.0.70全sub-800ms first-audio-out reachable。Bar非仅latency。是交互感：不切用户off、不被切off、mid-sentence interruption恢复、mid-conversation call tool不stall音频、survive jittery mobile networks。

你不能经stitch三个REST calls达。架构是pipelined streaming端到端。建它并失败模式可见：VAD tuned for phone audio于background TV firing、turn-detector等punctuation never comes、TTS buffers 400ms before emitting。毕业项目是load下逐一fix并发表latency-and-quality报告。

## 概念讲解

Pipeline五流stages：**audio in** (WebRTC from browser或PSTN)、**ASR** (Deepgram Nova-3或faster-whisper流partial transcripts)、**turn detection** (VAD加小turn-detector model读partial transcripts completion cues)、**LLM** (turn judge complete时流tokens)、**TTS** (first LLM token ~200ms内流audio out)。

三cross-cutting concerns。**Barge-in**：用户agent说话时开始说话、TTS cancel和ASR immediately pickup。**Tool use**：mid-conversation function calls (weather、calendar)须side channel run不stall音频；agent pre-fill acknowledgement token ("一moment...")若latency exceed 300ms。**Backpressure**：packet loss下、partial transcripts held、VAD raise speech-gate threshold、agent avoid speaking over unacknowledged message。

测量bar定量。WER under 8%于Hamming VAD benchmark at 15 dB SNR。First-audio-out p50 under 800ms于100测calls。False-cutoff rate under 3%。MOS above 4.2于TTS。50 concurrent calls于单g5.xlarge。此数是deliverable。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (或 Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- Transport：LiveKit Agents 1.0 (WebRTC)加Twilio PSTN gateway；Pipecat 0.0.70作alternate framework
- ASR：Deepgram Nova-3 (streaming、sub-300ms first partial)或faster-whisper Whisper-v3-turbo自hosted
- VAD：Silero VAD v5加LiveKit turn-detector (读partial transcripts小transformer)
- LLM：OpenAI GPT-4o-realtime于tight integration、Gemini 2.5 Flash Live、或cascaded Claude Haiku 4.5 (streaming completions、separate audio path)
- TTS：Cartesia Sonic-2 (lowest first-byte)、ElevenLabs Flash v3、或open-source Orpheus于自host
- Tools：FastMCP side-channel于weather/calendar/booking；agent pre-emits filler若tool take >300ms
- Observability：OpenTelemetry voice spans、Langfuse voice traces带audio replay
- Deployment：单g5.xlarge (24GB VRAM)于自hosted Whisper + Orpheus；hosted APIs于lowest latency

## 动手实践

1. **WebRTC session。** 立LiveKit room和web client流麦克风音频。Server、attach agent worker joins room。

2. **ASR streaming。** Feed 20ms PCM frames于Deepgram Nova-3 (或GPU faster-whisper)。Subscribe partial和final transcripts。Log per-partial latency。

3. **VAD和turn detector。** Run Silero VAD v5于frame stream。speech-end event、fire LiveKit turn-detector对latest partial transcript。仅commit "turn complete"当VAD说silence for 500ms并turn-detector scores completion > 0.6。

4. **LLM stream。** turn complete时、start LLM call带running conversation加final transcript。Stream tokens out。First token、hand off于TTS。

5. **TTS stream。** Cartesia Sonic-2 streams audio chunks back。First chunk须first LLM token 200ms内server leave。Emit chunks于LiveKit room；client经WebRTC jitter buffer play。

6. **Barge-in。** VAD detects新user speech当TTS playing时、immediately cancel TTS stream、drop remaining LLM output、并re-arm ASR。Publish `tts_canceled` span。

7. **Tool side channel。** Register weather和calendar作function-calling tools。invoked时、concurrently fire call；若300ms内不resolve、LLM emit "一moment、让我查" filler；tool return后resume。

8. **Eval harness。** Record 100 calls。Compute WER (对held-out transcript)、false-cutoff rate (TTS cancelled当user mid-sentence)、first-audio-out p50、TTS MOS (human或NISQA)、和jitter-loss test (drop 3% packets)。

9. **Load test。** Drive 50 concurrent calls于单g5.xlarge带synthetic caller。Measure sustained first-audio-out p95。

## 使用它

```
caller: "东京明天天气何"
[asr  ] partial @280ms: "东京明天"
[asr  ] partial @540ms: "东京明天天气"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "东京明天部分多云..."
turn latency: 1040ms user-stop -> audio-out
```

## 产出成果

`outputs/skill-voice-agent.md`是deliverable。给domain (customer support、scheduling、或kiosk)、立LiveKit agent带ASR/VAD/LLM/TTS pipeline tuned于测量bar。评分标准：

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | End-to-end latency | 100记录calls p50 first-audio-out under 800ms |
| 20 | Turn-taking quality | Hamming VAD benchmark false-cutoff rate under 3% |
| 20 | Tool-use correctness | Mid-conversation tool calls return right data不stall audio |
| 20 | Reliability under packet loss | 3% packet drop注入WER和turn-taking stability |
| 15 | Eval harness completeness | 公config reproducible measurements |
| **100** | | |

## 练习题

1. 换Deepgram Nova-3为g5.xlarge faster-whisper v3 turbo。测latency和WER gap。识CPU-vs-GPU决策何重。
2. 加interruption-arbitration policy：用户tool call期间barge-in时agent何？比三policy (hard cancel、finish-tool-then-stop、queue next turn)。
3. Run adversarial turn-detector test：给用户mid-sentence长pause。Tune VAD silence threshold和turn-detector score threshold于lowest false-cutoff不吹past 900ms。
4. 同agent经Twilio deploy于PSTN。比PSTN first-audio-out vs WebRTC。解释jitter-buffer和codec差异。
5. 加非英语语言语音activity detection (日语、西班牙语)。测Silero VAD v5 false-trigger rate vs language-specific fine-tunes。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Turn detection | "End of utterance" | Classifier给VAD silence和partial transcript、决用户done speaking |
| Barge-in | "Interruption handling" | VAD detects新user speech时TTS mid-playback cancel |
| First-audio-out | "Latency" | 用户停说话到first audio packet server leave时间 |
| VAD | "Speech gate" | Model classifying audio frames speech vs silence；Silero VAD v5是2026 default |
| Jitter buffer | "Audio smoothing" | Client-side buffer briefly hold packets absorb network variance |
| Filler | "Acknowledgment token" | Agent emit短phrase避免silence当tool slow |
| MOS | "Mean opinion score" | Perceptual speech quality rating；NISQA是automated proxy |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — alternate Python-first streaming agent framework
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — integrated speech models参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD参考model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS参考
- [Retell AI architecture](https://docs.retellai.com) — 产voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternate产参考