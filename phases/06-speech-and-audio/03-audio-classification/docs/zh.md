# 音频分类——从MFCC上k-NN到AST和BEATs

> 从"狗叫vs警笛"到"这是什么语言"，一切都是音频分类。特征是mel频谱。架构每十年都在演变。评估指标仍是AUC、F1和每类召回率。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图与Mel)、阶段3课程06(CNN)、阶段5课程08(文本CNN与RNN)
**时间:** ~75分钟

## 问题背景

得10秒片段。想知道:"是什么?"城市声(警笛、钻、狗)、语音命令(是/否/停)、语言识别(en/es/ar)、说话人情感(愤怒/中性)、或环境声(室内/室外、嘈杂)。全*音频分类*,2026基线架构成熟:log-mel→CNN或Transformer→softmax。

核心难不是网络。是数据。音频数据集残酷类不平衡、强域移(干净vs噪声)、标签噪声(谁定"城市嘈杂"vs"餐厅噪声"?)。问题80%是整理、增强、评估,非换CNN为Transformer。

## 概念讲解

![音频分类阶梯:MFCC上k-NN到AST到BEATs](../assets/audio-classification.svg)

**MFCC上k-NN(1990s基线)。**每片段MFCC展平,算与标记库余弦相似,返前K多数票。干净小数据集(Speech Commands、ESC-50)惊人强。无GPU跑。

**log-mel上2D CNN(2015-2019)。**视`(T, n_mels)` log-mel为图像。用ResNet-18或VGG式。时间轴全局平均池。类softmax。仍是大多2026 Kaggle竞赛基线。

**音频频谱图Transformer,AST(2021-2024)。**Patch化log-mel(如16×16 patch),加位置嵌入,喂ViT。监督学习AudioSet(mAP 0.485)最优。

**BEATs和WavLM-base(2024-2026)。**百万小时自监督预训。用需监督数据1-10%微调任务。2026这是非语音音频默认起点。BEATs-iter3用1/4计算AudioSet比AST高1-2 mAP。

**Whisper编码器冻结骨干(2024)。**取Whisper编码器,弃解码器,附线性分类器。语言识别和简单事件分类零音频增强近最优。"免费午餐"基线。

### 类不平衡是真挑战

ESC-50:50类,每类40片段——平衡,易。UrbanSound8K:10类,10:1不平衡。AudioSet:632类100,000:1长尾。有效技术:

- 训时平衡采样(评估不)。
- Mixup:线性插两片段(及标签)作增强。
- SpecAugment:mask随机时间和频率带。简;关键。

### 评估

- 多类排他(Speech Commands):top-1准确率、top-5准确率。
- 多类多标签(AudioSet、UrbanSound式):平均精度(mAP)。
- 重不平衡:每类召回+宏F1。

2026应知数字:

| 基准 | 基线 | SOTA 2026 | 来源 |
|------|------|-----------|------|
| ESC-50 | 82%(AST) | 97.0%(BEATs-iter3) | BEATs论文(2024) |
| AudioSet mAP | 0.485(AST) | 0.548(BEATs-iter3) | HEAR排行榜2026 |
| Speech Commands v2 | 98%(CNN) | 99.0%(Audio-MAE) | HEAR v2结果 |

## 动手实践

### Step 1:特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2:固定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简但强:时间轴mean+variance给13系数MFCC 26维固定嵌入。瞬间跑。ESC-50上近2017败最优神经网络基线。

### Step 3:k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4:升级log-mel上CNN

PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M参数。RTX 4090上ESC-50约10分钟训。80%+准确率。

### Step 5:2026默认——微调BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs,用`beats`库`microsoft/BEATs-base`;transformers API同形。

## 实际应用

2026栈:

| 情况 | 起始 |
|------|------|
| 微数据集(<1000片段) | MFCC均值k-NN(你基线)+音频增强 |
| 中数据集(1K–100K) | BEATs或AST微调 |
| 大数据集(>100K) | 从头训或微调Whisper编码器 |
| 实时、边缘 | 40-MFCC CNN,int8量化(KWS式) |
| 多标签(AudioSet) | BCE损失+mixup+SpecAugment BEATs-iter3 |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107基线 |

决策:**冻结骨干起,非新模型**。微调BEATs头小时得SOTA 95%,非周。

## 产出成果

存`outputs/skill-classifier-designer.md`。为给定音频分类任务选架构、增强、类平衡策略、评估指标。

## 练习题

1. **简单。**跑`code/main.py`。4类合成数据集(不同音高纯音)训k-NN MFCC基线。报告混淆矩阵。
2. **中等。**用[mean, var, skew, kurtosis]换`summarize`。4矩池化在相同合成数据集败mean+var吗?
3. **困难。**用`torchaudio`,ESC-50 fold 1上训2D CNN。报告5折交叉验证准确率。加SpecAugment(时间mask=20,频率mask=10)并报告增量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| AudioSet | 音频ImageNet | Google 2M片段、632类弱标记YouTube数据集。 |
| ESC-50 | 小分类基准 | 50类×40片段环境声。 |
| AST | 音频频谱图Transformer | log-mel patch上ViT;2021最优。 |
| BEATs | 自监督音频 | Microsoft模型,iter3 2026领AudioSet。 |
| Mixup | 配增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基mask增强 | 频谱图随机时间和频率带置零。 |
| mAP | 主多标签指标 | 类和阈间平均精度。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)——2021–2024记录架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)——2024+默认。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779)——主导音频增强。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50)——50类基准存活。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/)——632类YouTube分类;仍是金标准。