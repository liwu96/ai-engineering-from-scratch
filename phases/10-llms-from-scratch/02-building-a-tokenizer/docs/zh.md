# 从零构建分词器

> 第01课给你一个玩具。这节课给你一把利器。

**类型：** 构建
**语言：** Python
**前置要求：** 第10阶段，第01课（分词器：BPE、WordPiece、SentencePiece）
**时间：** 约90分钟

## 学习目标

- 构建一个生产级的BPE分词器，处理Unicode、空格归一化和特殊词元
- 实现字节级回退，使分词器能够编码任何输入（包括表情符号、中日韩字符和代码）而不产生未知词元
- 添加预分词正则表达式模式，在应用BPE合并之前按词边界分割文本
- 在语料库上训练自定义分词器，并在多语言文本上评估其压缩比率与tiktoken的对比

## 问题背景

你的第01课BPE分词器在英文文本上运行良好。现在试试日文。或者表情符号。或者混合制表符和空格的Python代码。

它会崩溃。

不是因为BPE错了——而是实现不完整。生产级分词器处理任何编码的原始字节，在分割之前归一化Unicode，管理永不合并的特殊词元，将预分词与子词分割链式连接，并且执行速度足够快，不会成为处理15万亿词元的训练管道的瓶颈。

GPT-2的分词器有50,257个词元。Llama 3有128,256个。GPT-4大约有100,000个。这些不是玩具数字。这些词汇表背后的合并表是在数百GB文本上训练的，而周围的机制——归一化、预分词、特殊词元注入、对话模板格式化——决定了分词器是只能处理"hello world"还是能处理整个互联网。

你将构建这些机制。

## 概念讲解

### 完整流程

生产级分词器不是一种算法。它是五个阶段的管道，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[归一化]
    B --> C[预分词]
    C --> D[BPE合并]
    D --> E[特殊词元]
    E --> F[词元ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有特定的工作：

| 阶段 | 功能 | 重要性 |
|------|------|--------|
| 归一化 | NFKC Unicode，可选小写，可选去除重音符号 | "fi"连字(U+FB01)变成"fi"（两个字符）。没有这一步，相同词汇会得到不同的词元。 |
| 预分词 | 在BPE之前将文本分割成块 | 防止BPE跨词边界合并。"the cat"不应该产生词元"e c"。 |
| BPE合并 | 对字节序列应用学习的合并规则 | 核心压缩。将原始字节转换为子词词元。 |
| 特殊词元 | 注入`<s>`、`</s>`、`<pad>`、对话模板标记 | 这些词元有固定ID。它们不参与BPE合并。模型需要它们来构建结构。 |
| ID映射 | 将词元字符串转换为整数ID | 模型看到的是整数，不是字符串。 |

### 字节级BPE

第01课的分词器在UTF-8字节上操作。这是正确的选择。但我们跳过了重要的一点：当这些字节不是有效的UTF-8时会发生什么？

字节级BPE通过将每个可能的字节值（0-255）视为有效词元来解决这个问题。你的基础词汇表正好是256个条目。任何文件——文本、二进制、损坏的——都可以被分词而不产生未知词元。

GPT-2添加了一个技巧：将每个字节映射到可打印的Unicode字符，使词汇表保持人类可读。字节0x20（空格）在他们的映射中变成字符"G"。这纯粹是装饰性的。算法不关心。

真正的力量：字节级BPE处理地球上的每种语言。中文字符每个是3个UTF-8字节。日文可以是3-4字节。阿拉伯文、天城文、表情符号——都只是字节序列。BPE算法在这些字节序列中找到模式的方式与在英文ASCII字节中找到模式的方式完全相同。

### 预分词

在BPE接触你的文本之前，你需要将其分割成块。这防止合并算法创建跨词边界的词元。

GPT-2使用正则表达式模式分割文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩略词处分割（"don't"变成"don" + "'t"），带可选前导空格的词、数字、标点符号和空格。前导空格保持附着在词上——所以"the cat"变成[" the", " cat"]，而不是["the", " ", "cat"]。

Llama使用SentencePiece，它完全跳过正则表达式。它将原始字节流视为一个长序列，让BPE算法自己找出边界。这更简单，但给BPE更多自由来创建跨词词元。

选择很重要。GPT-2的正则防止分词器学习将一个词结尾的"the"和下一个词开头的"the"合并。SentencePiece允许这样做，有时产生更有效的压缩但可解释性较差的词元。

### 特殊词元

每个生产级分词器都为结构标记保留词元ID：

| 词元 | 用途 | 使用者 |
|------|------|--------|
| `<s>` / `<bos>` | 序列开始 | Llama 3, GPT |
| `</s>` / `<eos>` | 序列结束 | 所有模型 |
| `<pad>` | 批次对齐填充 | BERT, T5 |
| `<unk>` | 未知词元（字节级BPE消除了这个） | BERT, WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户回合标记 | Llama 3 |
| `<\|assistant\|>` | 助手回合标记 | Llama 3 |

特殊词元不会被BPE分割。它们在合并算法运行之前被精确匹配，替换为固定ID，周围文本正常分词。

### 对话模板

这是大多数人困惑的地方，也是大多数实现崩溃的地方。

当你向对话模型发送消息时，API接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到JSON。它看到一个平铺的词元序列。对话模板使用特殊词元将消息转换为该平铺序列。每个模型的方式都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板出错，模型就会产生垃圾输出。它是按一种确切格式训练的。任何偏差——缺少换行、交换词元、额外空格——都会使输入超出训练分布。

### 速度

Python对于生产分词来说太慢了。

tiktoken（OpenAI）用Rust编写，带有Python绑定。HuggingFace分词器也是Rust。SentencePiece是C++。这些比纯Python实现快10-100倍。

作为参考：以每秒100万词元（快速Python）的速度分词Llama 3预训练的15万亿词元需要174天。以每秒1亿词元（Rust）的速度，只需要1.7天。

你用Python构建是为了理解算法。在生产中，你会使用编译实现，只接触Python包装器。

## 动手实践

### 第1步：字节级编码

基础。将任何字符串转换为字节序列，将每个字节映射到可打印字符以显示，并逆转该过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试以查看字节数：

```python
texts = [
    ("英文", "hello"),
    ("中文", "你好"),
    ("表情符号", "🔥"),
    ("混合", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)}字符 -> {len(b)}字节 -> {b}")
```

"hello"是5个字节。"你好"是6个字节（每字符3个）。火焰表情符号是4个字节。字节级分词器不关心是什么语言。字节就是字节。

### 第2步：带正则的预分词

使用GPT-2正则表达式模式将文本分割成块。每个块由BPE独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex`模块支持Unicode属性转义（`\p{L}`表示字母，`\p{N}`表示数字）。标准库`re`模块不支持，所以我们回退到ASCII字符类。对于生产多语言分词器，请安装`regex`。

试试看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附着在词上。缩略词在撇号处分割。标点符号变成自己的块。BPE永远不会跨这些边界合并。

### 第3步：字节序列上的BPE

第01课的核心算法，但现在在预分词的块上独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第4步：特殊词元处理

特殊词元需要精确匹配和固定ID。它们完全绕过BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第5步：完整分词器类

将所有内容链在一起：归一化、在特殊词元上分割、预分词、BPE合并、映射到ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第6步：多语言测试

真正的测试。用英文、中文、表情符号和代码来测试。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"输入:   {text}")
    print(f"词元:  {len(ids)}个ID")
    print(f"解码: {decoded}")
    print()
```

中文字符每个产生3个字节。表情符号产生4个字节。这些都不会使分词器崩溃。都不会产生未知词元。这就是字节级BPE的力量。

## 实际应用

### 对比真实分词器

加载Llama 3、GPT-4和Mistral的实际分词器。看看每个如何处理相同的多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)}词元): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)}词元): {pieces[:20]}...")
```

你会看到相同文本的不同词元数。Llama 3的128K词汇表更积极地合并常见模式。GPT-4的100K处于中间。Mistral的32K产生更多词元，但嵌入层更小。

权衡总是相同的：更大的词汇表意味着更短的序列但更多参数。

## 产出成果

这节课产出一个用于构建和调试生产分词器的提示。见`outputs/prompt-tokenizer-builder.md`。

## 练习题

1. **简单：** 添加一个`get_token_bytes(id)`方法，显示任何词元ID的原始字节。用它来检查你最常见的合并词元实际代表什么。
2. **中等：** 实现Llama风格的预分词器，在空格和数字上分割但保留前导空格。在相同语料库上将其与GPT-2正则方法进行比较。
3. **困难：** 添加一个对话模板方法，接受`{"role": ..., "content": ...}`消息列表，并为Llama 3对话格式产生正确的词元序列。针对HuggingFace实现进行测试。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 字节级BPE | "在字节上操作的分词器" | 基础词汇表为256个字节值的BPE——处理任何输入都不产生未知词元 |
| 预分词 | "BPE之前的分割" | 基于正则或规则的拆分，防止BPE跨词边界合并 |
| NFKC归一化 | "Unicode清理" | 规范分解后接兼容性组合——"fi"连字变成"fi"，全角"A"变成"A" |
| 对话模板 | "消息如何变成词元" | 将角色/内容消息列表转换为平铺词元序列的确切格式——模型特定，必须匹配训练格式 |
| 特殊词元 | "控制词元" | 绕过BPE的保留词元ID——`<s>`、`</s>`、`<pad>`、对话标记——精确匹配后合并 |
| 词元生成率 (Fertility) | "每词词元数" | 输出词元与输入词的比率——GPT-4英文为1.3，韩文更高，更高意味着浪费上下文 |
| tiktoken | "OpenAI分词器" | Rust BPE实现，带有Python绑定——比纯Python快10-100倍 |
| 合并表 | "词汇表" | 训练期间学习的字节对合并的有序列表——这就是分词器的学习知识 |

## 延伸阅读

- [OpenAI tiktoken源码](https://github.com/openai/tiktoken) —— GPT-3.5/4使用的Rust BPE实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持BPE、WordPiece、Unigram的Rust分词器库
- [Llama 3论文（Meta, 2024）](https://arxiv.org/abs/2407.21783) —— 128K词汇表和分词器训练详情
- [SentencePiece（Kudo & Richardson, 2018）](https://arxiv.org/abs/1808.06226) —— 语言无关的分词
- [GPT-2分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原始字节到Unicode映射
