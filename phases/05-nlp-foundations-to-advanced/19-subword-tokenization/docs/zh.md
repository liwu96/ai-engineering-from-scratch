# 子词词元化——BPE、WordPiece、Unigram、SentencePiece

> 词分词器卡在未见过的词上。字符分词器使序列长度爆炸。子词分词器取两者折中。每个现代大语言模型都采用其中一种。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程01(文本处理)、阶段5课程04(GloVe/FastText/子词)
**时间:** ~60分钟

## 问题背景

词汇50000词。用户输入"untokenizable"。分词器返`[UNK]`。模型无信号关于词。更糟:语料库90分位文档40稀有词,意味每文档40位丢失信息。

子词词元化解决此。常见词留单词元。稀有词分解有意义片:`untokenizable`→`un`、`token`、`izable`。训练数据覆盖一切因任何字符串终是字节序列。

2026每个前沿大语言模型发货于三算法(BPE、Unigram、WordPiece)之一,包于三库(tiktoken、SentencePiece、HF Tokenizers)之一。不能发货语言模型不选一个。

## 概念讲解

![BPE vs Unigram vs WordPiece,逐字符](../assets/subword-tokenization.svg)

**BPE(Byte-Pair Encoding,字节对编码)。** 从字符级词汇开始。计数每相邻对。合并最频对成新词元。重复直到目标词汇大小。主导算法:GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 同算法但原始字节(256基词元)而非Unicode字符。保证零`[UNK]`词元——任何字节序列编码。GPT-2用50,257词元(256字节+50,000合并+1特殊)。

**Unigram。** 从巨大词汇开始。分配每词元unigram概率。迭代剪除移除最少增语料库对数似然词元。推理概率化:可采样词元化(通过子词正则化数据增强有用)。T5、mBART、ALBERT、XLNet、Gemma用。

**WordPiece。** 合并最大化训练语料库似然而非原始频率对。BERT、DistilBERT、ELECTRA用。

**SentencePiece vs tiktoken。** SentencePiece是直接原始Unicode文本*训练*词汇(BPE或Unigram)库,编码空格为`▁`。tiktoken是OpenAI预构建词汇快*编码器*;不训练。

经验法则:

- **训练新词汇:** SentencePiece(多语言,无预分词)或HF Tokenizers。
- **GPT词汇快推理:** tiktoken(cl100k_base, o200k_base)。
- **两者:** HF Tokenizers——一库,训练+服务。

## 动手实践

### Step 1:BPE从零实现

见`code/main.py`。循环:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码三事实。`</w>`标记词尾使"low"(后缀)和"lower"(前缀)保持不同。频率加权让高频对早赢。合并列表有序——推理按训练顺序施合并。

### Step 2:用学习合并编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素O(n·|merges|)。生产实现(tiktoken、HF Tokenizers)用合并排名查找配优先队列近线性时间跑。

### Step 3:SentencePiece实践

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意:无预分词需,空格编码为`▁`,`character_coverage`控稀有字符保vs映射`<unk>`激。

### Step 4:tiktoken做OpenAI兼容词汇

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快(Rust后端)。GPT-4/5词元化字节计数、成本估计、上下文窗口预算精确匹配。

## 2026仍发货陷阱

- **分词器漂移。** 词汇A训,词汇B部署。词元ID差;模型输出垃圾。CI查`tokenizer.json`哈希。
- **空格歧义。** BPE"hello" vs " hello"产不同词元。总显式指定`add_special_tokens`和`add_prefix_space`。
- **多语言欠训练。** 英文重语料库产非拉丁脚本分裂5-10×多词元词汇。同提示词日语/阿拉伯语GPT-3.5成本5-10×多。o200k_base部分修复此。
- **Emoji分裂。** 单emoji可取5词元。预算上下文时检查emoji处理。

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 从零训单语言模型 | HF Tokenizers(BPE) |
| 训多语言模型 | SentencePiece(Unigram, `character_coverage=0.9995`) |
| 服务OpenAI兼容API | tiktoken(`o200k_base`用于GPT-4+) |
| 领域特定词汇(代码、数学、蛋白) | 领域语料库训定制BPE,合并基词汇 |
| 边缘推理,小模型 | Unigram(小词汇工作更好) |

词汇大小是缩放决策,非常数。粗启发:<1B参数32k,1-10B50-100k,多语言/前沿200k+。

## 产出成果

存`outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: 为给定语料库和部署目标选分词器算法、词汇大小、库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料库(大小、语言、领域)和部署目标(从零训练/微调/API兼容推理),输出:

1. 算法。BPE、Unigram或WordPiece。一句话理由。
2. 库。SentencePiece、HF Tokenizers或tiktoken。理由。
3. 词汇大小。舍入近1k。理由绑模型大小和语言覆盖。
4. 覆盖设置。`character_coverage`、`byte_fallback`、特殊词元列表。
5. 验证计划。保留集平均每词词元、OOV率、压缩比、往返解码等价。

拒绝在有稀有脚本内容语料库训character-coverage<0.995分词器。拒绝发货词汇无冻结`tokenizer.json`哈希检查CI。标记任何单语言分词器<16k词汇可能欠规格。
```

## 练习题

1. **简单。** 在`code/main.py`小语料库训500合并BPE。编码三保留词。多少产精确1词元vs>1词元?
2. **中等。** 比100英文Wikipedia句子`cl100k_base`、`o200k_base`和你训vocab=32k SentencePiece BPE词元计数。报每压缩比。
3. **困难。** 同语料库训BPE、Unigram和WordPiece。测用每小情感分类器下游准确率。选择移针>1点F1吗?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| BPE | Byte-Pair Encoding | 最频字符对贪婪合并直到目标词汇大小。 |
| 字节级BPE | 永无未知词元 | 原始256字节上BPE;GPT-2/Llama用此。 |
| Unigram | 概率分词器 | 用对数似然从大候选集剪;T5、Gemma用。 |
| SentencePiece | 空格那个 | 原始文本训BPE/Unigram库;空格编码`▁`。 |
| tiktoken | 快那个 | OpenAI Rust后端BPE编码器预构建词汇。无训练。 |
| 合并列表 | 魔数 | `(a, b)→ab`有序合并列表;推理按序施。 |
| 字符覆盖 | 多稀有太稀有? | 分词器须覆盖训练语料库字符分数;典型~0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch(2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)——BPE论文。
- [Kudo(2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959)——Unigram论文。
- [Kudo, Richardson(2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)——库。
- [Hugging Face—Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary)——简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken)——食谱+编码列表。