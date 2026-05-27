# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。把它们放回一起你得到为输入→输出任务构建的模型:翻译、摘要、重写、转录。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段7课程06(BERT)、阶段7课程07(GPT)
**时间:** ~45分钟

## 问题背景

仅解码器GPT和仅编码器BERT各为不同目标拆解2017架构。但许多任务天然输入输出:

- 翻译:英语→法语。
- 摘要:5000词元文章→200词元摘要。
- 语音识别:音频词元→文本词元。
- 结构抽取:散文→JSON。

对这些,编码器-解码器是最干净适配。编码器产源稠密表示。解码器生成输出,每步交叉attend到该表示。训练是输出侧shift-by-one。同GPT损失,仅条件在编码器输出。

两论文定义现代剧本:

1. **T5**(Raffel等2019)。"Text-to-Text Transfer Transformer。"每个自然语言处理任务重框架为文本入文本出。单架构、单词表、单损失。在掩码span预测上预训练(输入corrupt spans,输出decode它们)。
2. **BART**(Lewis等2019)。"Bidirectional and Auto-Regressive Transformer。"去噪自编码器:多种方式corrupt输入(shuffle、mask、delete、rotate),要求解码器重建原始。

2026年编码器-解码器格式在输入结构重要处存活:

- Whisper(语音→文本)。
- Google翻译栈。
- 一些有分明context-and-edit结构的代码补全/修复模型。
- Flan-T5及变体做结构推理任务。

仅解码器赢聚光灯,但编码器-解码器从未离开。

## 概念讲解

![配交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源词元 ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                           │
目标词元 ─▶ 解码器块                       │
            ├─▶ 掩码自注意力               │
            ├─▶ 交叉注意力 ◀───────────────┘
            └─▶ FFN
           ↓
         下词元logits
```

关键,编码器每输入跑一次。解码器自回归跑但每步交叉attend到*相同*编码器输出。缓存编码器输出对长输入是免费加速。

### T5预训练——span corruption

随机选输入span(平均长3词元,共15%)。每span换唯一sentinel:`<extra_id_0>`、`<extra_id_1>`等。解码器仅输出corrupt spans配sentinel前缀:

```
源: The quick <extra_id_0> fox jumps <extra_id_1> dog
目标: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜信号。T5论文消融中与MLM(BERT)和prefix-LM(UniLM)竞争。

### BART预训练——多噪声去噪

BART试五种噪声函数:

1. 词元掩码。
2. 词元删除。
3. 文本填充(掩span,解码器插正确长度)。
4. 句子置换。
5. 文档旋转。

文本填充+句子置换组合产最佳下游数。解码器总重建原始。BART输出是全序列,非仅corrupt spans——故预训练计算比T5高。

### 推理

同GPT自回归生成。Greedy/beam/top-p采样适用。Beam search(宽4-5)是翻译和摘要标准因输出分布比聊天窄。

### 2026何时选各变体

| 任务 | 编码器-解码器? | 原因 |
|------|----------------|------|
| 翻译 | 通常yes | 清源序列;固定输出分布;beam search有效 |
| 语音到文本 | Yes(Whisper) | 输入模态异于输出;编码器塑音频特征 |
| 聊天/推理 | No,仅解码器 | 无持久"输入"——对话是序列 |
| 代码补全 | 通常no | 配长上下文仅解码器胜;代码模型如Qwen 2.5 Coder仅解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS胜早期仅解码器基线;现代仅解码器LLM匹它们 |
| 结构抽取 | 两者皆可 | T5干净因"文本→文本"吸收任何输出格式 |

约2022趋势:仅解码器接管编码器-解码器曾拥有的任务因(a)指令调仅解码器LLM通过提示泛化到任何,(b)一架构比两更易扩展,(c)RLHF假设解码器。编码器-解码器在输入模态不同(语音、图像)或beam search质量重要处坚持。

## 动手实践

见`code/main.py`。我们对玩具语料库实现T5式span corruption——本课最有用单块因它出现在此后每个编码器-解码器预训练配方。

### Step 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选总计约mask_rate词元的spans。返回(corrupted_input, target)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是T5约定:`<sent0> span0 <sent1> span1 ...`。corrupt输入交错不变词元和span位置sentinel词元。

### Step 2: 验证往返

给定corrupt输入和目标,重建原始句。如果corruption可逆,前向定义良好。这是健全检查——真训练不做这,但测试便宜并捕获span bookkeeping off-by-one bug。

### Step 3: BART噪声

五函数:`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合两显示结果。

## 实际应用

HuggingFace参考:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5技巧:任务名进输入文本。同模型处理几十任务因每任务文本入文本出。2026这pattern已被指令调仅解码器模型泛化,但T5首先编码它。

## 产出成果

见`outputs/skill-seq2seq-picker.md`。技能给定输入输出结构、延迟和质量目标为新任务在编码器-解码器和仅解码器间选。

## 练习题

1. **简单。**运行`code/main.py`,对30词元句应用span corruption,验证拼接非sentinel源词元和decoded target spans重产原始。
2. **中等。**实现BART `text_infill`噪声:随机span换单`<mask>`词元,解码器须推断正确span长加内容。示一例。
3. **困难。**在微小英语→pig-Latin语料库(200对)微调`flan-t5-small`。在50对保留集测BLEU。比同计算微调`Llama-3.2-1B`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 编码器-解码器 | "Seq2seq transformer" | 两栈:双向编码器输入,配交叉注意力因果解码器输出。 |
| 交叉注意力 | "源与目标对话处" | 解码器Q×编码器K/V。编码器信息入解码器唯一位置。 |
| Span corruption | "T5预训练技巧" | 随机span换sentinel词元;解码器输出spans。 |
| 去噪目标 | "BART游戏" | 输入应用噪声函数,训练解码器重建干净序列。 |
| Sentinel词元 | "`<extra_id_N>`占位符" | 特殊词元在源标记corrupt spans并在目标重标记。 |
| Flan | "指令调T5" | T5在>1800任务微调;使编码器-解码器指令跟随竞争。 |
| Beam search | "解码策略" | 每步保top-k部分序列;翻译/摘要标准。 |
| Teacher forcing | "训练时输入" | 训练时,喂真前输出词元给解码器,非采样词元。 |

## 延伸阅读

- [Raffel等(2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)——T5。
- [Lewis等(2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)——BART。
- [Chung等(2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)——Flan-T5。
- [Radford等(2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)——Whisper,典型2026编码器-解码器。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)——参考实现。