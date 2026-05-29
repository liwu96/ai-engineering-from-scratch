# 流式语音到语音——Moshi、Hibiki和全双工对话

> 2024-2026年重新定义了语音AI。Moshi发布了一个以200毫秒延迟同时听和说的单一模型。Hibiki逐块进行语音到语音翻译。两者都抛弃了ASR→大语言模型→TTS管道，转而采用基于Mimi编解码词元的统一全双工架构。这是新的参考设计。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段6课程13(神经音频编解码器)、阶段6课程11(实时音频)、阶段7课程05(完整Transformer)
**时间:** ~75分钟

## 问题背景

课程11+12构建的每个语音智能体有根本延迟底约300-500毫秒:VAD触发、STT处理、大语言模型推理、TTS生成。每阶段有自己最小延迟。可调优并行化,但管道形封顶你。

Moshi(Kyutai, 2024-2026)问不同问题:若无管道会怎样?若单一模型直接音频入音频出,连续,文本作中间"内心独白"而非必需阶段?

答案是**全双工语音到语音**。理论延迟160毫秒(80毫秒Mimi帧+80毫秒声学延迟)。单L4 GPU实际延迟200毫秒。那是最佳级联管道语音智能体一半。

## 概念讲解

![Moshi架构:两并行Mimi流+内心独白文本](../assets/moshi-hibiki.svg)

### Moshi架构

**输入。** 两Mimi编解码流,都在12.5 Hz × 8码本:

- 流1:用户音频(Mimi编码,持续到达)
- 流2:Moshi自己音频(Moshi生成)

**Transformer。** 7B参数时间Transformer处理两流和文本"内心独白"流。每80毫秒步:

1. 消最新用户Mimi词元(8码本)。
2. 消最近Moshi Mimi词元(8码本,已产)。
3. 生成下Moshi文本词元(内心独白)。
4. 生成下Moshi Mimi词元(经小深度Transformer, 8码本)。

全三流——用户音频、Moshi音频、Moshi文本——并行跑。Moshi可说话时听用户;用户打断时可自打断;可插话("嗯")不破主话语。

**深度Transformer。** 帧内,8码本不并行预测——有码本间依赖。小2层"深度Transformer"80毫秒内顺序预测它们。这是AR编解码LM标准分解(VALL-E、VibeVoice也用)。

### 何内心独白文本有用

无显式文本,模型得隐式在声学流建模语言。Moshi洞察:强制它同时音频发文本词元。文本流本质Moshi说转录。此改语义连贯,易换语言模型头,并免费转录。

### Hibiki:流式语音到语音翻译

同架构,翻译对训。源音频入,目标语言音频出,连续。Hibiki-Zero(2026年2月)消词级对齐训练数据需——用句级数据+GRPO强化学习延迟优化。

初始支持四语言对;可用≈1000小时适配新语言。

### 更大Kyutai栈(2026)

- **Moshi**——全双工对话(法语优先,英语良好支持)
- **Hibiki/Hibiki-Zero**——同时语音翻译
- **Kyutai STT**——流式ASR(500毫秒或2.5秒前瞻)
- **Kyutai Pocket TTS**——100M参数TTS CPU跑(2026年1月)
- **Unmute**——公服务器组合这些全管道

L40S GPU吞吐:64并发会话@3×实时。

### Sesame CSM——表亲

Sesame CSM(2025)用类似想法——Llama-3骨干配Mimi编解码头。但CSM单向(取上下文+文本,产语音)而非全双工。它是市场上最好"语音在场"TTS;不完全是Moshi全双工能力。

### 2026性能数字

| 模型 | 延迟 | 用例 | 许可 |
|------|------|------|------|
| Moshi | 200 ms(L4) | 全双工英语/法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz帧率 | 法语↔英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同 | 5语言对,无对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件TTS | Apache-2.0 |
| GPT-4o实时 | ~300 ms | 闭源,OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源,Google API | 商业 |

## 动手实践

### Step 1:接口

Moshi暴露WebSocket服务器取80毫秒Mimi编码音频块并返80毫秒Mimi编码音频块。双向。持续。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Step 2:全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两向同时跑。Python asyncio或Rust futures是标准传输。

### Step 3:训练目标(概念)

每80毫秒帧`t`:

- 输入:`user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 预测:`moshi_text[t]`,然后`moshi_mimi[t, codebook_0..7]`

文本音频前预测(内心独白);音频深度Transformer内码本顺序预测。

### Step 4:Moshi赢和不赢处

Moshi赢:

- 便宜硬件上亚250毫秒端到端。
- 自然插话和打断。
- 无管道粘代码。

Moshi不赢:

- 工具调用(未训;需单独大语言模型路径)。
- 长推理(Moshi是8B级对话模型,非Claude/GPT-4)。
- 细领域事实准确。
- 大多数生产企业用例(2026仍用管道)。

## 实际应用

| 情况 | 选择 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi, CSM |
| 企业工具智能体 | 管道(课程12),非Moshi |
| 上下文自定义语音TTS | Sesame CSM |
| 语音到语音,任意语言 | GPT-4o实时或Gemini 2.5 Live(商业) |

## 陷阱

- **有限工具调用。** Moshi是对话模型,非智能体框架。工具需配管道。
- **特定声音条件。** Moshi用单训练人设;克隆是单独训练跑。
- **语言覆盖。** 法语+英语优秀;其他有限。Hibiki-Zero帮助,但仍需训练数据。
- **资源成本。** 全Moshi会话占GPU槽;非便宜共享租户部署模式。

## 产出成果

存`outputs/skill-duplex-pipeline.md`。为语音智能体工作负载选管道vs全双工架构,配理由。

## 练习题

1. **简单。** 跑`code/main.py`。符号模拟两流+内心独白架构。
2. **中等。** 从HuggingFace拉Moshi,跑服务器,测一次对话。测用户语音结束到Moshi响应开始墙上时钟延迟。
3. **困难。** 取你课程12管道智能体并比20匹配测试话语P50延迟vs Moshi。写何时管道架构仍赢。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 全双工 | 同时听和说 | 同模型上两音频流同时活跃。 |
| 内心独白 | 模型文本流 | Moshi音频输出同时发文本词元。 |
| 深度Transformer | 码本间预测器 | 80毫秒帧内预测8码本小transformer。 |
| Mimi | Kyutai编解码 | 12.5 Hz × 8码本;语义+声学;撑Moshi。 |
| 流式S2S | 音频→音频实况 | 块块翻译/对话,无管道阶段。 |
| 插话 | "嗯"反应 | Moshi可不破自己轮次发小确认。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi——语音-文本基础模型](https://arxiv.org/html/2410.00037v2)——论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345)——无对齐数据流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)——CSM规格。
- [Kyutai——Moshi仓库](https://github.com/kyutai-labs/moshi)——安装+服务器。
- [OpenAI——实时API](https://platform.openai.com/docs/guides/realtime)——闭源商业同行。
- [Kyutai——Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)——底层STT/TTS框架。