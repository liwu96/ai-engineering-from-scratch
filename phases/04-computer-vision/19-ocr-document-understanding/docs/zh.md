# OCR与文档理解

> OCR是三阶段管道 — 检测文本框、识别字符、然后排版布局。每个现代OCR系统都会重新排序这些阶段或合并它们。

**类型:** 学习 + 使用
**语言:** Python
**前置要求:** 阶段4课程06(检测)，阶段7课程02(自注意)
**时间:** ~45分钟

## 学习目标

- 追经典OCR管道(检 -> 识 -> 布)和现代端到端替代(Donut、Qwen-VL-OCR)
- 实现CTC(连接时分类)损为序列到序列OCR训
- 用PaddleOCR或EasyOCR为生产文档解析无训
- 区OCR、布局解析和文档理解 — 并每任务择正工具

## 问题背景

满文图像处皆在：收据、发票、证件、扫书、表格、白板、标志、截屏。从中提取结构数据 — 非仅字符，而是"这是总额" — 是最高价值应用视觉问题。

域分三技能层：

1. **OCR本身**: 转像素为文。
2. **布局解析**: 组OCR输出为区(标题、正文、表、头)。
3. **文档理解**: 从布局提取结构域("invoice_total = $42.50")。

每层有经典和现代方法，"我要图像中文"和"我需这收据总额"差距大过多队意识。

## 概念讲解

### 经典管道

```mermaid
flowchart LR
    IMG["图像"] --> DET["文检测<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["字/行<br/>边界框"]
    BOX --> CROP["裁每区"]
    CROP --> REC["识<br/>(CRNN + CTC)"]
    REC --> TXT["文串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["读序文"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文检测**产每行或每字四边形。
- **识**裁每区到固定高，跑CNN + BiLSTM + CTC产字符序列。
- **布局**重建读序(上到下、左到右拉丁；阿拉伯、日语不同)。

### CTC一段

OCR识从固定长特征图产变长序列。CTC (Graves等, 2006)让你无字符级对齐训。模型每时步输(vocab + blank)分布；CTC损边化所有对齐减为目标文后合重复去blank。

```
原输出: "h h h _ _ e e l l _ l l o _ _"
合重复去blank后: "hello"
```

CTC是CRNN 2015工作因仍是2026多生产OCR模型训方式。

### 现代端到端模型

- **Donut** (Kim等, 2022) — ViT编码器 + 文解码器；读图像直发JSON。无文检测、无布局模块。
- **TrOCR** — ViT + transformer解码器为行级OCR。
- **Qwen-VL-OCR / InternVL** — 全视觉语言模型微调OCR任务；2026复杂文档最佳精度。
- **PaddleOCR** — 成熟生产包中经典DB + CRNN管道；仍开源主力。

端到端模型需更多数据和算但跳多阶段管道误累积。

### 布局解析

结构文档，跑布局检测器(LayoutLMv3、DocLayNet)标每区：标题、段落、图、表、脚注。读序变"按布局序迭区、拼接"。

表格，用**键值提取**模型(富视文档Donut、平扫LayoutLMv3)。它们收图像 + 检测文 + 位置预结构键值对。

### 评估指标

- **字符错率(CER)** — Levenshtein距 / 参考长。低佳。生产目标：清扫< 2%。
- **词错率(WER)** — 词级同。
- **结构域F1** — 键值任务；测`{invoice_total: 42.50}`是否正确现。
- **JSON编距** — 端到端文档解析；Donut论文引归一化树编距。

## 构建

### 步骤1: CTC损 + 贪解码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax vocab含blank索引0
    targets:        (N, S) int目标(无blank)
    input_lengths:  (N,) 每样本用时步
    target_lengths: (N,) 每样本目标长
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: 索引序列列表(blank去，重复合)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`可用时用高效CuDNN实现。贪解码比束搜简通常CER差1%内。

### 步骤2: 微CRNN识器

最小CNN + BiLSTM为行OCR。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高输入(CNN最大池高到1)。宽是CTC时维。

### 步骤3: 合成OCR

生成白底黑数字串为端到端冒烟测。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"图像: {imgs.shape}   目标: {targets.shape}   长: {lengths.tolist()}")
```

真OCR数据集加字体、噪、旋转、模糊和色。上管道同。

### 步骤4: 训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

损应这简合成数据200步从~3降到~0.2。

## 使用

三生产路：

- **PaddleOCR** — 成熟、快、多语言。一行用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — Python原生、多语言、PyTorch骨干。
- **Tesseract** — 经典；模型挣扎老扫文档仍有用。

端到端文档解析，用Donut或VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

收据、发票和可复结构表格，微调Donut。任意文档或带推理OCR，VLM如Qwen-VL-OCR是现默认。

## 交付成果

本课程产：

- `outputs/prompt-ocr-stack-picker.md` — 给文档类型、语言和结构选Tesseract / PaddleOCR / Donut / VLM-OCR提示词
- `outputs/skill-ctc-decoder.md` — 从零写贪和束搜CTC解码器含长归一化技能

## 练习题

1. **(易)** 训TinyCRNN于5位随机数字串500步。报留出集CER。

2. **(中)** 替贪解码为束搜(beam_width=5)。报CER差。束搜何输入胜？

3. **(难)** 用PaddleOCR于20收据集，提取行项，算手标真值{item_name, price}对F1。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| OCR | "像素到文" | 转图像区为字符序列 |
| CTC | "无对齐损" | 无每时步标签训序列模型损；边化对齐 |
| CRNN | "经典OCR模型" | Conv特征提取 + BiLSTM + CTC；2015基线仍生产用 |
| Donut | "端到端OCR" | ViT编码器 + 文解码器；图像直发JSON |
| 布局解析 | "找区" | 检测和标文档中标题/表/图/段落区 |
| 读序 | "文序列" | 识区排序为句；拉丁平、混布局非平 |
| CER / WER | "错率" | 字或词粒度Levenshtein距 / 参考长 |
| VLM-OCR | "读的LLM" | 训或提示OCR任务视觉语言模型；复杂文档现SOTA |

## 延伸阅读

- [CRNN (Shi等, 2015)](https://arxiv.org/abs/1507.05717) — 原CNN+RNN+CTC架构
- [CTC (Graves等, 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原CTC论文；算法思想密集
- [Donut (Kim等, 2022)](https://arxiv.org/abs/2111.15664) — OCR无文档理解transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源生产OCR栈