# BERT — 掩码语言建模

> GPT预测下一个词。BERT预测缺失的词。一句话的差别——以及此后五年一切嵌入相关领域的命运。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程05(完整Transformer)、阶段5课程02(文本表示)
**时间:** ~45分钟

## 问题背景

2018年每个自然语言处理任务——情感、命名实体识别、问答、蕴涵——在自己的标注数据上从零训练自己模型。没有预训练"懂英语"检查点可微调。ELMo(2018)展示可用双向LSTM预训练上下文嵌入;它有帮助但不泛化。

BERT(Devlin等2018)问:如果我们取transformer编码器,在互联网每句话上训练,强制它从两边上下文预测缺失词会如何?然后你在下游任务上微调一头。参数效率是启示。

结果:18个月内BERT及其变体(RoBERTa、ALBERT、ELECTRA)统治每个存在的自然语言处理排行榜。到2020年地球上每个搜索引擎、内容审核管道、语义搜索系统内都有BERT。

2026年仅编码器模型仍是对分类、检索和结构抽取的正确工具——它们每词元运行比解码器快5-10倍,其嵌入是每个现代检索栈骨干。ModernBERT(2024年12月)用Flash注意力机制 + RoPE + GeGLU推架构到8K上下文。

## 概念讲解

![掩码语言建模:选词元、掩码、预测原始](../assets/bert-mlm.svg)

### 训练信号

取句子:`the quick brown fox jumps over the lazy dog`。

随机掩码15%词元:

```
输入:  the [MASK] brown fox jumps [MASK] the lazy dog
目标: the  quick brown fox jumps  over  the lazy dog
```

训练模型在掩码位置预测原始词元。因编码器双向,在位置1预测`[MASK]`可用位置2+的`brown fox jumps`。这是GPT做不到的。

### BERT掩码规则

在选中预测的15%词元中:

- 80%换成`[MASK]`。
- 10%换成随机词元。
- 10%保持不变。

为何不全`[MASK]`?因`[MASK]`推理时永不出现。训练模型100%掩码位置期望`[MASK]`会在预训练和微调间造分布偏移。10%随机+10%不变使模型诚实。

### 下一句预测(NSP)——及为何被弃

原始BERT还训NSP:给定两句A和B,预测B是否接A。RoBERTa(2019)消融它显示NSP有害而非有益。现代编码器跳它。

### 2026有何变化:ModernBERT

2024 ModernBERT论文用2026原语重建块:

| 组件 | 原BERT(2018) | ModernBERT(2024) |
|------|--------------|-------------------|
| 位置 | 学习绝对 | RoPE |
| 激活 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全稠密 | 交替局部(128)+全局 |
| 上下文长 | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

且不像2018栈,它是Flash-注意力机制原生。8K序列长推理比配更好GLUE分的DeBERTa-v3快2-3×。

### 2026仍选编码器的用例

| 任务 | 为何编码器胜解码器 |
|------|-------------------|
| 检索/语义搜索嵌入 | 双向上下文=每词元更好嵌入质量 |
| 分类(情感、意图、毒性) | 一次前向;无生成开销 |
| 命名实体识别/词元标注 | 每位置输出,原生双向 |
| 零样本蕴涵(自然语言推理) | 编码器顶分类头 |
| RAG重排器 | 交叉编码器评分,比大语言模型重排器快10× |

## 动手实践

### Step 1: 掩码逻辑

见`code/main.py`。函数`create_mlm_batch`取词元ID列表、词表大小和掩码概率。返回输入ID(应用掩码)和标签(仅掩码位置,其余-100——PyTorch忽略索引约定)。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Step 2: 在微小语料库跑MLM预测

在20词词表、200句上训2层编码器+MLM头。无梯度——我们做前向健全检查。全训练需PyTorch。

### Step 3: 比掩码类型

展示三路规则如何使模型无`[MASK]`可用。在无掩码句和掩码句上预测。两者应产合理词元分布因模型在训练中见两pattern。

### Step 4: 微调头

在玩具情感数据集用分类头换MLM头。仅头训练;编码器冻结。这是每BERT应用遵循模式。

## 实际应用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调BERT。**`sentence-transformers`模型如`all-MiniLM-L6-v2`是配对比损失训练的BERT。编码器相同。损失改。

**交叉编码器重排器也是微调BERT。**`[CLS] query [SEP] doc [SEP]`对分类。查询和文档间双向注意力正是交叉编码器质量胜双编码器之处。

**2026何时不选BERT。**任何生成任务。编码器无合理方式自回归产词元。也:1B参数以下小解码器可在更灵活性下匹配质量(Phi-3-Mini、Qwen2-1.5B)。

## 产出成果

见`outputs/skill-bert-finetuner.md`。技能为新分类或抽取任务scope BERT微调(骨干选择、头spec、数据、评估、停止)。

## 练习题

1. **简单。**运行`code/main.py`并打印10000词元掩码分布。确认~15%选中,其中~80%成`[MASK]`。
2. **中等。**实现整词掩码:如果词tokenize为子词,全掩子词或不掩。测500句语料库MLM准确率是否改进。
3. **困难。**在公共数据集10000句上训微小(2层,d=64)BERT。微调`[CLS]`词元做SST-2情感。比匹配参数仅解码器基线——谁胜?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MLM | "掩码语言建模" | 训练信号:随机换15%词元为`[MASK]`,预测原始。 |
| 双向 | "两边看" | 编码器注意力无因果掩码——每位置见每其他位置。 |
| `[CLS]` | "池化词元" | 每序列前置特殊词元;其最终嵌入用作句子级表示。 |
| `[SEP]` | "段分隔符" | 分隔配对序列(如查询/文档、句A/B)。 |
| NSP | "下一句预测" | BERT第二预训练任务;RoBERTa示无用,2019后弃。 |
| 微调 | "适配任务" | 编码器大多冻结;顶上训小头做下游任务。 |
| 交叉编码器 | "重排器" | 输入查询和文档BERT,输出相关分。 |
| ModernBERT | "2024刷新" | 配RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K上下文重建编码器。 |

## 延伸阅读

- [Devlin等(2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)——原始论文。
- [Liu等(2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)——如何正确训BERT;杀NSP。
- [Clark等(2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)——换词元检测在匹配计算下胜MLM。
- [Warner等(2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)——ModernBERT论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)——典型编码器参考。