# 音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o音频

> 2026音频语言模型推理语音+环境声+音乐。Qwen2.5-Omni-7B在MMAU-Pro上匹敌GPT-4o音频。Audio Flamingo Next在LongAudioBench上超Gemini 2.5 Pro。开源vs闭源差距本质闭合——除了多音频任务,所有人接近随机。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段6课程04(ASR)、阶段12课程03(视觉语言模型)、阶段7课程10(音频Transformer)
**时间:** ~45分钟

## 问题背景

你有5秒音频:狗叫、有人喊"stop!",然后静音。有用问题跨多轴:

- **转录。** "说了什么?"——ASR领域。
- **语义推理。** "人有危险吗?"——需联合理解叫+喊+静音。
- **音乐推理。** "什么乐器演奏旋律?"
- **长音频检索。** "这90分钟讲座中讲师在哪解释梯度下降?"

单一模型用一个提示词回答所有这些是**音频语言模型**(LALM/ALM)。与纯ASR分离:LALMs产自由形式自然语言答案,不止转录。

## 概念讲解

![音频语言模型:音频编码器+投影器+大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

每个2026 LALM同骨架:

1. **音频编码器。** Whisper编码器、BEATs、CLAP、WavLM或模型自定义编码器。
2. **投影器。** 线性或MLP桥接音频编码器特征到大语言模型词元嵌入空间。
3. **大语言模型。** Llama/Qwen/Gemma基解码器。取交错文本+音频词元;生成文本。

训练:

- **阶段1。** 冻编码器+大语言模型;仅训投影器于ASR/字幕数据。
- **阶段2。** 全/LoRA微调于指令遵循音频任务(问答、推理、音乐理解)。
- **阶段3(可选)。** 语音入/语音出加语音解码器。Qwen2.5-Omni和AF3-Chat做此。

### 2026模型图谱

| 模型 | 骨干 | 音频编码器 | 输出模态 | 访问 |
|------|------|------------|----------|------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义+Whisper | 文本+语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本+语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA非商业 |
| SALMONN | Vicuna | Whisper+BEATs | 文本 | Apache-2.0 |
| LTU/LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST+Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro(闭源) | Gemini | 专有 | 文本+语音 | API |
| GPT-4o音频(闭源) | GPT-4o | 专有 | 文本+语音 | API |

### 基准现实检验(2026)

**MMAU-Pro。** 1800问答对覆盖语音/声/音乐/混合。含多音频子集。

| 模型 | 总体 | 语音 | 声 | 音乐 | 多音频 |
|------|------|------|-----|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o音频 | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人都是判决。** 4选项多选随机机会=25%;大多数模型在那附近。LALMs仍难比两片段。

### LALMs 2026有用之处

- **呼叫中心录音合规审计。** "智能体提到必需披露了吗?"
- **无障碍。** 向聋用户描述声事件(不止转录)。
- **内容审核。** 检测暴力语言+威胁语气+背景上下文。
- **播客/会议章节化。** 语义总结,不止说话人轮次。
- **音乐目录分析。** "找所有B段有转调的曲目。"

### 何处无用(暂)

- 细粒度音乐理论(和弦级以下)。
- 长对话说话人归属推理(超过10分钟退化)。
- 多音频比较(22-26%勉强高于随机)。
- 实时流式推理(大多数是离线批推理)。

## 动手实践

### Step 1:查询Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2:投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这么简单。投影器通常1-3线性层。在ASR对(音频→转录)上训练是阶段1 pretext任务。

### Step 3:MMAU/LongAudioBench基准

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报每类(语音/声/音乐/多音频)。聚合数字隐藏模型失败处。

## 实际应用

| 任务 | 2026选择 |
|------|----------|
| 自由形式音频问答(开源) | Qwen2.5-Omni-7B |
| 开源长音频最佳 | Audio Flamingo Next |
| 闭源最佳 | Gemini 2.5 Pro |
| 语音入/语音出智能体 | Qwen2.5-Omni或GPT-4o音频 |
| 音乐推理 | Audio Flamingo 3或2(音乐专用AF-CLAP) |
| 呼叫中心审计 | Gemini 2.5 Pro API,配策略文档RAG/检索增强生成 |

## 陷阱

- **过度信任多音频。** 任务需"哪个片段有X",随机级性能真实。
- **长音频退化。** 超10分钟,大多数模型说话人归属崩溃。先分离(课程6),再总结。
- **静音幻觉。** LALMs用Whisper编码器继承同Whisper风格问题。VAD门。
- **基准樱桃采摘。** 供应商博客突出最佳类。自己跑MMAU-Pro多音频子集。

## 产出成果

存`outputs/skill-alm-picker.md`。为给定音频理解任务选LALM+基准子集+输出模态(文本vs语音)。

## 练习题

1. **简单。** 跑`code/main.py`看玩具投影器模式+假LALM路由(音频嵌入,文本词元)→输出词元。
2. **中等。** 在100 MMAU-Pro语音项上评分Qwen2.5-Omni-7B。与论文报数比。
3. **困难。** 建最小音频字幕基线:BEATs编码器+2层投影器+冻结Llama-3.2-1B。仅微调投影器于AudioCaps。与SALMONN在Clotho-AQA比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| LALM | 音频ChatGPT | 音频编码器+投影器+大语言模型解码器。 |
| 投影器 | 适配器 | 小MLP映射音频特征到大语言模型嵌入空间。 |
| MMAU | 基准 | 10k音频问答对跨语音、声、音乐。 |
| MMAU-Pro | 更难MMAU | 1800多音频/推理重问题。 |
| LongAudioBench | 长形评估 | 多分钟片段配语义查询。 |
| 语音入/语音出 | 语音原生 | 模型摄入语音并发出语音不经文本绕道。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759)——参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)——语音入-语音出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128)——开源长音频领袖。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905)——LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289)——双编码器先驱。
- [MMAU-Pro排行榜](https://mmaubenchmark.github.io/)——2026实时排名。