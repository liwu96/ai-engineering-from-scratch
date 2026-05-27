# ASCII艺术和视觉Jailbreak

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753)。掩有害请求安全相关token、用同字母ASCII艺术渲染换、发cloaked提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2全不鲁棒识ASCII艺术token。攻绕PPL(perplexity filters)、Paraphrase防御、和Retokenization。相关：ViTC benchmark测非语义视觉提示识；StructuralSleight泛化到Uncommon Text-Encoded Structures(树、图、嵌JSON)作编码攻家族。

**类型:** 构建
**语言:** Python(stdlib、ArtPrompt token掩harness)
**前置要求:** 阶段18课程12(PAIR)、阶段18课程13(MSJ)
**时间:** ~60分钟

## 学习目标

- 描述ArtPrompt攻：词识步、ASCII艺术换、终cloaked提示。
- 解释何标准防御(PPL、Paraphrase、Retokenization)败于ArtPrompt。
- 定义ViTC并描述何测。
- 描述StructuralSleight作泛化到任Uncommon Text-Encoded Structures。

## 问题背景

改述和角色扮演攻(课程12)和长context攻(课程13)操文级模式。ArtPrompt操于识级：模型不解析禁token。解析字符渲染图像。安全filter见无害标点。模型见词。

## 概念讲解

### ArtPrompt, 两步

步1. 词识。给有害请求、攻者用LLM识安全相关词(如"bomb"于"how to make a bomb")。

步2. Cloaked提示生成。换每识词为ASCII艺术渲染(7x5或7x7字符块成字母形)。模型收标点和空格网格、足够力模型可识为词；安全filter仅见网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5全败。攻成功率75%以上于benchmark子集。

### 为何标准防御败

- **PPL(perplexity filter)。** ASCII艺术高perplexity — 但所有新输入也。阻ArtPrompt阈值也阻合法结构输入。
- **Paraphrase。** 改述提示毁ASCII艺术。实、改述LLM常保或重构艺术。
- **Retokenization。** 不同tokenizer分不改模型视觉识字母形。

底问题是安全filter是token或语义级；ArtPrompt操于视觉识级。

### ViTC benchmark

非语义视觉提示识。测模型读ASCII艺术、wingdings、和其他非文本语义视觉内容能力。ArtPrompt效与ViTC精度相关：模型读视觉文越好、ArtPrompt工越好。此是能力安全tradeoff。

### StructuralSleight

泛化ArtPrompt：Uncommon Text-Encoded Structures (UTES)。树、图、嵌JSON、CSV-in-JSON、diff风格代码块。若结构于训安全数据罕见但模型可解析、可藏有害内容。

防御意：安全须泛化跨模型可解析结构表示。集大且长。

### 图像模态类比

视觉LLM(GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1)延攻面。ArtPrompt风格攻用实图比ASCII艺术强因图编码器产更富信号。

### Phase 18何处

课程12-14描述三正交攻向量：迭代细化(PAIR)、context长度(MSJ)、和编码(ArtPrompt/StructuralSleight)。课程15转模型中心攻到系统边界攻(indirect prompt injection)。课程16描述防御工具响应。

## 使用

`code/main.py`玩具ArtPrompt。可cloak有害查询特定词用ASCII艺术glyph、验cloaked串过keyword filter、和(可选)解码cloaked串回用简识器。

## 交付成果

本lesson产`outputs/skill-encoding-audit.md`。给jailbreak防御报告、枚举编码攻家族覆(ASCII艺术、base64、leet-speak、UTF-8 homoglyph、UTES)和何防御层捕每。

## 练习题

1. 跑`code/main.py`。验cloaked串过简keyword filter。报需字符级改。
2. 实第二编码：同目标词base64。比filter绕率对ArtPrompt和恢复难。
3. 读Jiang等人2024第4.3节(五模型结果)。提Claude ArtPrompt抗高于Gemini于同benchmark原因。
4. 设计前生成防御检提示ASCII艺术形域。测合法代码、表、和数学符号假阳性率。
5. StructuralSleight列10编码结构。草通防御处全10并估每防御提示算成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ArtPrompt | "ASCII艺术攻" | 两步jailbreak掩安全词用ASCII艺术渲染 |
| Cloaking | "藏词" | 换禁token为视觉表示模型读但filter不 |
| UTES | "罕见结构" | Uncommon Text-Encoded Structure — 树、图、嵌JSON等走私内容 |
| ViTC | "视觉文能力" | 测模型读非语义视觉编码能力benchmark |
| Perplexity filter | "PPL防御" | 拒高perplexity提示；败因合法结构输入也高分 |
| Retokenization | "tokenizer移防御" | 用不同tokenizer前处理提示；败因识视觉 |
| Homoglyph | "看像字符" | Unicode字符看同拉丁字母；绕子串查 |

## 延伸阅读

- [Jiang等人 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII艺术jailbreak论文
- [Li等人 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES泛化
- [Chao等人 — PAIR (课程12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 补迭代攻
- [Anil等人 — Many-shot Jailbreaking (课程13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 补长度攻