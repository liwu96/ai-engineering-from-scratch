# 机器翻译

> 翻译是为NLP研究买单三十年并继续买单的任务。

**类型：** 构建
**语言：** Python
**前置要求：** 第5阶段 · 10（注意力机制），第5阶段 · 04（GloVe、FastText、子词）
**时间：** 约75分钟

## 问题背景

模型用一种语言读取句子，用另一种语言产生句子。长度变化。词序变化。一些源词映射到多个目标词，反之亦然。习语拒绝一对一映射。"I miss you" 法语是 "tu me manques" —— 字面意思是 "你对我缺乏"。没有词级对齐能幸存。

机器翻译是迫使NLP发明编码器-解码器、注意力、Transformer、最终整个LLM范式的任务。每一步进步都因为翻译质量可测量，人机差距顽固。

本课程跳过历史课，教授2026年的工作流水线：预训练多语言编码器-解码器（NLLB-200或mBART）、子词分词、束搜索、BLEU和chrF评估，以及仍发货到生产的那 handful of 失效模式。

## 概念讲解

![MT流水线：分词 → 编码 → 带注意力解码 → 反分词](../assets/mt-pipeline.svg)

现代机器翻译是Transformer编码器-解码器在平行文本上训练。编码器以其语言的分词读取源。解码器通过交叉注意力（第10课）使用编码器输出，逐子词生成目标。解码使用束搜索避免贪婪解码陷阱。输出经反分词、大小写恢复，对照参考打分。

三个操作选择驱动真实机器翻译质量。

- **分词器。** 混合语言语料库上训练的SentencePiece BPE。跨语言共享词汇表是NLLB实现零样本语言对的原因。
- **模型大小。** NLLB-200蒸馏600M可装在笔记本上。NLLB-200 3.3B是发表的生产默认值。54.5B是研究上限。
- **解码。** 一般内容束宽4-5。长度惩罚避免过短输出。需要术语一致性时的束搜索约束解码。

## 动手实践

### 步骤1：预训练机器翻译调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里三件事重要。`src_lang` 告诉分词器使用哪种脚本和切分。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是NLLB特定技巧；mBART和M2M-100使用自己的约定，不可互换。

### 步骤2：BLEU和chrF

BLEU测量输出和参考之间的n-gram重叠。四个参考n-gram大小（1-4）、精确率的几何平均、对过短输出的简短惩罚。分数在[0, 100]。常用。难以解释：30 BLEU是"可用"；40是"好"；50是"卓越"；1 BLEU以下差异是噪音。

chrF测量字符级F分数。对形态丰富语言更敏感，BLEU在这些语言上低估匹配。常与BLEU一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它标准化分词，使分数可跨论文比较。自己实现BLEU计算是产生误导性基准的方式。

### 三层评估层次（2026年）

现代机器翻译评估使用三个互补指标族。发货时至少两个。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释、对改写不敏感。用于遗留比较和回归检测。
- **学习**（COMET、BLEURT、BERTScore）。在人类判断上训练的神经模型；比较翻译与源和参考的语义相似性。COMET自2023年以来与机器翻译研究关联最高，是2026年质量重要时的生产默认值。
- **LLM作为裁判**（无参考）。提示大模型在流畅性、充分性、语气、文化适当性上打分。GPT-4作为裁判在设计良好的评分标准时与人类一致~80%。用于没有参考的开放式内容。

2026年实用栈：`sacrebleu` 用于BLEU和chrF，`unbabel-comet` 用于COMET，提示LLM用于最终面向人类的信号。在信任生产数据前，对每个指标校准50-100人工标注示例。

无参考指标（COMET-QE、BLEURT-QE、LLM作为裁判）让你无需参考评估翻译，这对没有参考翻译的长尾语言对很重要。

### 步骤3：生产中出错的地方

上面的工作流水线80%时间流畅翻译，剩下20%静默失效。命名失效模式：

- **幻觉。** 模型发明源中没有的内容。不熟悉的领域词汇中常见。症状：输出流畅但声称源未陈述的事实。缓解：领域术语约束解码、受监管内容人工审查、对输出比输入长得多进行监控。
- **离目标生成。** 模型翻译成错误语言。NLLB在罕见语言对上惊人地容易这样。缓解：验证 `forced_bos_token_id`，始终用语言ID模型检查输出解码。
- **术语漂移。** "Sign up" 在文档1变成 "s'inscrire"，文档2变成 "créer un compte"。对于UI文本和面向用户的字符串，一致性比原始质量更重要。缓解：词汇约束解码或后编辑字典。
- **语气不匹配。** 法语 "tu" vs "vous"，日语礼貌级别。模型选择在训练中出现更频繁的形式。对于面向客户的内容这通常错误。缓解：如果模型支持用语气词元提示前缀，或在纯正式语料库上微调小模型。
- **短输入长度爆炸。** 非常短的输入句子常产生过长翻译，因为长度惩罚在~5个源词元以下失效。缓解：与源长度成比例的硬最大长度限制。

### 步骤4：领域微调

预训练模型是通才。法律、医疗或游戏对话翻译从领域平行数据微调中显著受益。配方并不异国情调：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千高质量平行示例击败几十万嘈杂网页抓取。训练数据质量是单一最大生产杠杆。

## 实际应用

2026年机器翻译生产栈：

| 用例 | 推荐起点 |
|------|---------|
| 任意到任意，200语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行，廉价推理，英语-法语/德语/西班牙语 | Helsinki-NLP / Marian模型 |
| 延迟关键浏览器端 | ONNX量化Marian（~50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini带翻译提示 |

截至2026年，LLM在几种语言对上优于专用机器翻译模型，特别是在习语内容和长上下文上。权衡是每词元成本和延迟。当上下文长度、风格一致性或通过提示的领域适应比吞吐量更重要时选择LLM。

## 产出成果

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估机器翻译输出以发货。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定源文本和候选翻译，输出：

1. 自动分数估计。你期望的BLEU和chrF范围。说明是否有参考。
2. 五点人工可验证检查清单：（a）内容保留（无幻觉），（b）正确语言，（c）语域/语气匹配，（d）与词汇表一致的术语一致性（如提供），（e）无截断或长度爆炸。
3. 一个要探测的领域特定问题。例如，法律：命名实体和法规引用。医疗：药物名称和剂量。UI：占位变量 `{name}`。
4. 信心标志。"Ship" / "Ship with review" / "Do not ship"。与步骤2中发现问题的严重程度挂钩。

拒绝无输出语言ID检查发货翻译。拒绝无参考评估，除非用户显式选择无参考评分（COMET-QE、BLEURT-QE）。标记任何超过1000词元的内容为可能需要分块翻译。
```

## 练习题

1. **简单。** 使用 `nllb-200-distilled-600M` 将5句英语段落翻译成法语，再回译成英语。测量往返与原始的距离。你应该看到语义保留，用词选择漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现翻译输出的语言ID检查。集成到机器翻译调用中，使离目标生成在返回前被捕获。
3. **困难。** 在5,000对领域语料库上微调 `nllb-200-distilled-600M`。在留出集上测量微调前后的BLEU。报告哪些句子改进，哪些退化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BLEU | 翻译分数 | 带简短惩罚的n-gram精确率。[0, 100]。 |
| chrF | 字符F分数 | 字符级F分数。对形态丰富语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的Transformer编码器-解码器。2017+默认。 |
| NLLB | 不让任何语言落后 | Meta的200语言机器翻译模型族。 |
| Constrained decoding | 受控输出 | 强制特定词元或n-gram在输出中出现/不出现。 |
| Hallucination | 发明内容 | 模型输出不受源支持。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告BLEU的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调演练。
