# 语音识别(ASR)——CTC、RNN-T、注意力机制

> 语音识别是每时间步音频分类,由知英语和静音的序列模型粘合。CTC、RNN-T和注意力机制是三种方式。选一理解为何。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图与Mel)、阶段5课程08(文本CNN与RNN)、阶段5课程10(注意力机制)
**时间:** ~45分钟

## 问题背景

有10秒16 kHz片段。要字符串:"turn on the kitchen lights"。挑战结构:音频帧不一一对应字符。词"okay"可占200 ms或1200 ms。静音断话语。些音素比另长。输出词元数不先知。

三种表述解:

1. **CTC(连接时序分类)。**每帧发词元概率含特殊*空白*。解码时坍重复和空白。非自回归,快。wav2vec 2.0、MMS用。
2. **RNN-T(循环神经网络转录器)。**联合网络配编码帧和前词元预测下一词元。可流。Google设备端ASR、NVIDIA Parakeet用。
3. **注意力编码器-解码器。**编码器压音频成隐藏状态,解码器跨注意力自回归产词元。Whisper、SeamlessM4T用。

2026,LibriSpeech test-clean SOTA WER 1.4%(Parakeet-TDT-1.1B, NVIDIA)和1.58%(Whisper-Large-v3-turbo)。差异小;部署差异大。

## 概念讲解

![三种ASR表述:CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC直觉。**让编码器输出`T`帧级分布于`V+1`词元(V字符+空白)。长度`U < T`目标字符串`y`,坍成`y`的任何帧对齐计。CTC损失对所有对齐求和。推理:每帧argmax,坍重复,删空白。

优点:非自回归,可流,零前瞻。缺点:*条件独立假设*——每帧预测独立于另,故无内部语言模型。用外部LM束搜索或浅融合修。

**RNN-T直觉。**加*预测器*网络嵌词元历史和*接合器*合预测器状态与编码帧成`V+1`联合分布(+1是空/不发)。显式模CTC忽略的条件依赖。可流因每步仅条件于前帧和前词元。

优点:可流+内部语言模型。缺点:训更复杂和内存饥(3D损失格);RNN-T损失核自成库类。

**注意力编码器-解码器。**编码器(6-32 Transformer层)于log-mel帧。解码器(6-32 Transformer层)跨注意力编码器输出自回归产词元。无对齐约束——注意力可看音频任意处。非流除非限注意力(分块Whisper-Streaming, 2024)。

优点:离线ASR最高质,标准seq2seq工具易训。缺点:自回归延迟比例于输出长;无工程不能流。

### WER:唯一数字

**词错误率** = `(S + D + I) / N`,S=替,D=删,I=插,N=参考词数。词级匹配Levenshtein编辑距离。低更好。WER高于20%一般不可用;低于5%读语音人类平。2026标准基准数:

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

全编码器-解码器或RNN-T基。纯CTC系统(wav2vec 2.0)test-clean约1.8–2.1%。

## 动手实践

### Step 1:贪婪CTC解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits:每帧概率向量列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两规则:坍连续重复,删空白。例:`a a _ _ a b b _ c` → `a a b c`。

### Step 2:束搜索CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (词元, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产用前缀树束搜索配语言模型融合;此是概念骨架。

### Step 3:WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4:Whisper推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026最强通用ASR一行。24 GB GPU~20×实时跑。

### Step 5:Parakeet或wav2vec 2.0流式

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式ASR需分块编码器注意和延续状态;用支持库(NeMo Parakeet, `transformers`管道配`chunk_length_s`)。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 英语,离线,最质 | Whisper-large-v3-turbo |
| 多语,鲁棒 | SeamlessM4T v2 |
| 流式,低延迟 | Parakeet-TDT-1.1B或Riva |
| 边缘,移动,<500 ms延迟 | Whisper-Tiny量化或Moonshine(2024) |
| 长形 | Whisper配VAD分块(WhisperX) |
| 域特定(医、法) | 微调wav2vec 2.0 +域语言模型融合 |

## 2026仍发货陷阱

- **无VAD。**Whisper于静音产幻觉("Thanks for watching!")。总VAD门。
- **字符vs词vs子词WER。**归一化后(小写,标点删)报词级WER。
- **语言识别漂移。**Whisper自动语言识别噪声片段错路由日语或威尔士语;知时强`language="en"`。
- **无分块长片段。**Whisper30秒窗。更长用`chunk_length_s=30, stride=5`。

## 产出成果

存`outputs/skill-asr-picker.md`。给定部署目标选模型、解码策略、分块和语言模型融合。

## 练习题

1. **简单。**跑`code/main.py`。贪婪解码手工CTC输出并算WER于参考。
2. **中等。**正确实现Step 2前缀树束搜索(计空白合并规则)。10例合成数据集与贪婪比。
3. **困难。**用`whisper-large-v3-turbo`于[LibriSpeech test-clean](https://www.openslr.org/12)。算前100话语WER。与发表数比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| CTC | 空白词元损失 | 所有帧到词元对齐边缘;非自回归。 |
| RNN-T | 流式损失 | CTC +下一词元预测器;理词序。 |
| 注意力编码器-解码器 | Whisper式 | 编码器 +跨注意解码器;最佳离线质。 |
| WER | 你报的数字 | 词级`(S+D+I)/N`。 |
| 空白 | 空东西 | CTC中特殊词元示"此帧无发"。 |
| 语言模型融合 | 外部语言模型 | 束搜索时加权重语言模型log概率。 |
| VAD | 静音门 | 语音活动检测器;裁非语音。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)——CTC论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711)——RNN-T论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)——2022规范论文;v3-turbo 2024扩。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b)——2026开放ASR排行榜领。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)——25+模型活基准。