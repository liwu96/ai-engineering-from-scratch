# Whisper——架构与微调

> Whisper是30秒窗口的Transformer编码器-解码器，在68万小时多语言弱监督音频-文本对上训练。单一架构，多种任务，跨99种语言具有鲁棒性。2026年的参考ASR。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程04(ASR)、阶段5课程10(注意力机制)、阶段7课程05(完整Transformer)
**时间:** ~75分钟

## 问题背景

Whisper,OpenAI 2022年9月发,是首个作为商品发货的ASR:贴音频,得文本,99语言,噪声鲁棒,笔记本跑。2024 OpenAI已发Large-v3和Turbo变体;2026,Whisper是从播客转录到语音助手到YouTube字幕一切默认基线。

但Whisper非可永远视为黑盒管道。域移杀——技术术语、说话者口音、专名词、短片段、静音。需知:

1. 内部实际是什么。
2. 如何正确给分块、流式或长形音频。
3.何时微调及如何。

## 概念讲解

![Whisper编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。**标准Transformer编码器-解码器。

- 输入:30秒log-mel频谱图,80 mels,10 ms跳步→3000帧。短片段零填,长分块。
- 编码器:卷积下采样(步长2) + `N` Transformer块。Large-v3:32层,1280维,20头。
- 解码器:`N` Transformer块配因果自注意力 +跨注意力编码器输出。同编码器大小。
- 输出:51,865词元词汇字节对编码词元。

Large-v3 1.55B参数。Turbo用4层解码器(从32),延迟降8×配<1% WER损。

**提示词格式。**Whisper是解码器提示词特殊词元导的多任务模型:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world. <|endoftranscript|>
```

- `<|en|>`——语言标签;强翻译vs转录行为。
- `<|transcribe|>`或`<|translate|>`——任意语言输入翻译成英语输出,或逐字。
- `<|notimestamps|>`——跳词级时间戳(快)。

提示词让一模型做多任务。改`<|en|>`成`<|fr|>`转录法语。

**30秒窗。**一切定30秒。长片段需分块;短填。窗非原生流——这是WhisperX、Whisper-Streaming和faster-whisper存在原因。

**Log-mel归一化。**`(log_mel - mean) / std`统计来自Whisper训语料。*必须*用Whisper预处理(`whisper.audio.log_mel_spectrogram`),非`librosa.feature.melspectrogram`。

### 2026变体

| 变体 | 参数 | 延迟(A100) | WER(LibriSpeech-clean) |
|------|------|------------|------------------------|
| Tiny | 39M | 1×实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming(2024) | 1.55B | 流式 | 2.0% |

### 微调

2026规范流程:

1. 收10–100小时目标域音频配对齐转录。
2. 跑`transformers.Seq2SeqTrainer`配`generate_with_loss`回调。
3. 参数高效:注意力层`q_proj`, `k_proj`, `v_proj`上LoRA降GPU内存4×配<0.3 WER代价。
4. <10小时冻结编码器。仅调解码器。
5. 用Whisper自己词元器和提示词格式;永换词元器。

社区结果:Medium于20小时医听写微调WER从12%降到医词汇4.5%。Turbo于4小时冰岛语微调WER从18%降到6%。

## 动手实践

### Step 1:开箱Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防失控重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

总覆关键默认:`temperature=0.0`(采样默认0.0 → 0.2 → 0.4…回退链),`condition_on_previous_text=False`(防级联幻觉问题),和`no_speech_threshold=0.6`(静音检测)。

### Step 2:分块长形

```python
# whisperx是2026配词级时间戳长形参考
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX加(1)Silero VAD门,(2)wav2vec 2.0词级对齐,(3)`pyannote.audio`说话者分离。2026生产转录工作马。

### Step 3:LoRA微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M可训 / 809M总
```

然后标准Trainer循环。每1000步检查点。保留WER评估。

### Step 4:查每层学什么

```python
# 解码时抓跨注意力权重看解码器注意什么。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions:层 × 头 × 步 × 源长
```

热图可视化——见解码器步扫描编码帧对角对齐。那对角是Whisper词时间戳概念。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 通用英语,离线 | Large-v3-turbo经`whisperx` |
| 移动/边缘 | Whisper-Tiny量化(int8)或Moonshine |
| 多语长形 | Large-v3经`whisperx` +说话者分离 |
| 低资源语言 | LoRA微调Medium或Turbo |
| 流式(2 s延迟) | Whisper-Streaming或Parakeet-TDT |
| 词级时间戳 | WhisperX(wav2vec 2.0强对齐) |

`faster-whisper`(CTranslate2后端)是2026最快CPU+GPU推理运行时——比原生快4×输出同。

## 2026仍发货陷阱

- **静音幻觉文本。**Whisper训于字幕含"Thanks for watching!", "Subscribe!",歌词。总VAD门前调。
- **`condition_on_previous_text`级联。**一幻觉污染后续窗。除非需块间流畅设`False`。
- **短片段填。**2秒片段填30秒可尾静音幻觉。用`pad=False`或VAD门。
- **错mel统计。**用librosa mels替Whisper产近随机输出。用`whisper.audio.log_mel_spectrogram`。

## 产出成果

存`outputs/skill-whisper-tuner.md`。给定域设计Whisper微调或推理管道。

## 练习题

1. **简单。**跑`code/main.py`。词元化Whisper式提示词,算解码形预算,打印10分钟片段分块调度。
2. **中等。**装`faster-whisper`,转录10分钟播客,比WER于人转录。试`language="auto"`vs强`language="en"`。
3. **困难。**用HF `datasets`,选Whisper难语言(如乌尔都语),Medium配LoRA 2轮2小时微调,报WER差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 30秒窗 | Whisper限 | 硬输入上限;更长音频分块。 |
| SOT | 转录始 | `<|startoftranscript|>`启解码器提示词。 |
| 时间戳词元 | 时对齐 | 每0.02 s偏移是51k词汇特殊词元。 |
| Turbo | 快变体 | 4解码器层,8×快,<1% WER退。 |
| WhisperX | 长形包装 | VAD + Whisper + wav2vec对齐 +说话者分离。 |
| LoRA微调 | 效调 | 注意力加低秩适配;训~0.3%参数。 |
| 幻觉 | 静默失败 | Whisper从噪声/静音产流畅英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356)——原始架构和训配方。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363)——4层解码器,8×提速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747)——长形,词对齐,说话者分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper)——CTranslate2后端,4×快。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)——规范LoRA /全微调走查。