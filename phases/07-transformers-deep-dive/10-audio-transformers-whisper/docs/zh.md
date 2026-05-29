# 音频Transformer — Whisper架构

> 音频是时间上频率图像。Whisper是吃mel频谱图并说话的ViT。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段7课程08(编码器-解码器)、阶段7课程09(ViT)
**时间:** ~45分钟

## 问题背景

Whisper前(OpenAI, Radford等2022),最先进自动语音识别(ASR)意味wav2vec 2.0和HuBERT——自监督特征提取器加微调头。高质量、昂贵数据管道、域脆弱。多语言语音识别需每语族单独模型。

Whisper下了三个赌注：

1. **在一切上训练。**从互联网刮680,000小时97语言弱标注音频。无干净学术语料。无音素标签。
2. **单模型多任务。**一解码器联合训转录、翻译、语音活动检测、语言ID和时间戳,通过任务词元。
3. **标准编码器-解码器transformer。**编码器消费log-mel频谱图。解码器自回归产文本词元。无vocoder、无CTC、无HMM。

结果:Whisper large-v3在口音、噪声和零干净标注数据语言间鲁棒。它是2026每个开源语音助手和多数商业默认语音前端。

## 概念讲解

![Whisper管道:音频→mel→编码器→解码器→文本](../assets/whisper.svg)

### Step 1——重采样+窗口

16kHz音频。剪/补到30秒。算log-mel频谱图:80 mel bins,10ms stride→约3,000帧×80特征。这是Whisper见"输入图像"。

### Step 2——卷积stem

两Conv1D层kernel 3 stride 2把3,000帧减到1,500。序列长减半不添多参数。

### Step 3——编码器

24层(large)transformer编码器过1,500时间步。Sinusoidal位置编码、自注意力、GELU FFN。产1,500 × 1,280隐藏状态。

### Step 4——解码器

24层transformer解码器。它从BPE词表自回归产词元,词表是GPT-2超集配几个音频特殊词元。

### Step 5——任务词元

解码器提示词以控制词元起告诉模型做什么:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在此约定上训。你通过前缀控任务。2026指令调等价,但应用于语音。

### Step 6——输出

Beam search(宽5)配log-prob阈值。无`<|notimestamps|>`词元时每0.02秒音频预测时间戳。

### Whisper大小

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM(fp16) |
|------|------|------|---------|------|------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB(4层解码器) |

Large-v3-turbo(2024)把解码器从32层砍到4。8×更快解码,<1 WER点退。解码速度解锁是为何Whisper-turbo是2026实时语音代理默认。

### Whisper不做的事

- 无说话人分离(谁在说)。配pyannote做那。
- 无原生实时流——30秒窗口固定。现代wrapper(`faster-whisper`、`WhisperX`)通过VAD+overlap bolt流。
- 无外部分块30秒外长形上下文。实际好用因人类语音转录很少需长程上下文。

### 2026格局

| 任务 | 模型 | 注 |
|------|------|-----|
| 英语ASR | Whisper-turbo、Moonshine | Moonshine边缘4×更快 |
| 多语言ASR | Whisper-large-v3 | 97语言 |
| 流ASR | faster-whisper + VAD | 150ms延迟目标可达成 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器pattern,但Whisper形 |
| 音频+语言 | AudioLM、SeamlessM4T | 文本词元+音频词元在单transformer |

## 动手实践

见`code/main.py`。我们不训Whisper——我们建log-mel频谱图管道+任务词元提示词格式器。这些是生产你实际碰部分。

### Step 1: 合成音频

生成16kHz采样440Hz1秒正弦波。16,000样本。

### Step 2: log-mel频谱图(简化)

完整mel频谱图需FFT。我们做简化分帧+每帧能量版展示管道无需`librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧=25ms,hop=10ms。匹配Whisper窗口。每帧能量替mel bins教学。

### Step 3: 补到30s

Whisper总处理30秒块。补(或剪)频谱图到3,000帧。

### Step 4: 建提示词词元

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这是完整任务控面。4词元前缀。

## 实际应用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快,OpenAI兼容:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026何时选Whisper:**

- 单模型多语言ASR。
- 嘈杂多样音频鲁棒转录。
- 研究/原型ASR——最快起点。

**何时选别的:**

- 边缘超低延迟流——Moonshine同质量比Whisper快。
- 需<200ms实时对话AI——专用流ASR。
- 说话人分离——Whisper不支持；可附加pyannote实现。

## 产出成果

见`outputs/skill-asr-configurator.md`。技能为新语音应用选ASR模型、解码参数和预处理管道。

## 练习题

1. **简单。**运行`code/main.py`。确认16kHz1秒信号10ms hop约100帧。30秒:约3,000帧。
2. **中等。**用`numpy.fft`建完整log-mel频谱图。验证80 mel bins在数值误差内匹配`librosa.feature.melspectrogram(n_mels=80)`。
3. **困难。**实现流推理:把音频切成10秒窗口配2秒overlap、每块跑Whisper、合转录。测5分钟podcast样本单pass vs word-error rate。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Mel频谱图 | "音频图像" | 2D表示:一轴频bin,另一时间帧;每cell log缩放能量。 |
| Log-mel | "Whisper所见" | Mel频谱图过log;近似人类响度感知。 |
| 帧 | "一时间切片" | 25ms样本窗口;10ms stride重叠。 |
| 任务词元 | "语音提示词前缀" | 解码器提示词中`<|transcribe|>` / `<|translate|>`等特殊词元。 |
| 语音活动检测(VAD) | "找语音" | ASR前去静gate;大幅砍成本。 |
| CTC | "Connectionist Temporal Classification" | 无对齐训练经典ASR损失;Whisper不用。 |
| Whisper-turbo | "小解码器,全编码器" | large-v3编码器+4层解码器;8×更快解码。 |
| Faster-whisper | "生产wrapper" | CTranslate2重实现;int8量化;比OpenAI参考快4×。 |

## 延伸阅读

- [Radford等(2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)——Whisper论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper)——参考代码+模型权重。读`whisper/model.py`看Conv1D stem +编码器+解码器顶到底约400行。
- [OpenAI Whisper—`whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py)——课程Step 5-6描述的beam-search+任务词元逻辑在此;500行,完全可读。
- [Baevski等(2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)——先驱;某些设置仍是SOTA特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)——生产wrapper,比参考快4×。
- [Jia等(2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)——2024边缘友好ASR,Whisper形但更小。
- [HuggingFace blog—"Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper)——典型微调配方含mel频谱图预处理器和词元-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)——完整实现(编码器、解码器、交叉注意力、生成)镜像课程架构图。