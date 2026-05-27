# Memory——Virtual Context和MemGPT

> Context window有限。对话、文档、和工具trace不。MemGPT(Packer等,2023)框此作OS virtual memory——main context是RAM、外存是disk、agent间page。此是每2026 memory系统继承模式。

**类型:** 构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程06(Tool Use)
**时间:** ~75分钟

## 学习目标

- 释MemGPT建OS类比:main context=RAM、外context=disk、memory tool=page in/out。
- 实stdlib两tier MemGPT模式带main-context buffer、外可搜store、和page in/out工具。
- 描述agent何发"interrupt"查询或改外memory和结果何splice入下提示。
- 识MemGPT设计择承入Letta(课程08)和Mem0(课程09)。

## 问题背景

Context window看应解memory。不。产于三失败模式:

1. **Overflow。**多轮对话、长文档、或tool-call重trajectory跨window。过cutoff全gone。
2. **Dilution。**即使window内,塞无关context稀释attention于重要事。Frontier模型仍长输入降。
3. **Persistence。**新session起空window。无外memory agent不能跨session说"记你曾问我..."

大window帮但不修。Mem0 2025论文测128k-window基线仍漏长horizon事实4k-window agent带外memory捕。

## 概念讲解

### MemGPT:OS类比

Packer等(arXiv:2310.08560,v2 2024年2月)映context管至OS virtual memory:

| OS概念 | MemGPT概念 | 2026产类比 |
|--------|-----------|------------|
| RAM | main context(prompt) | Anthropic/OpenAI context window |
| Disk | 外context | vector DB、KV、graph store |
| Page fault | memory tool call | `memory.search`、`memory.read`、`memory.write` |
| OS kernel | agent控循环 | ReAct循环带memory tool |

Agent跑正常ReAct循环。一额外类工具让它page数据进出main context。

### 两tier

- **Main context。**固定大提示持当前任务。模型总可见。
- **外context。**Unbounded、经工具可搜。相关时读、事实现时写。

原论文于两任务评设计超base window:文档分析长100k token和多session chat带跨日持久记忆。

### Interrupt模式

MemGPT引入memory-as-interrupt:对话中途agent可调memory tool、runtime执它、结果splice入下assistant turn作新观察。概念同Unix`read()`syscall阻塞process、回byte、process续。

Canonical memory tool面:

- `core_memory_append(section,text)`——写至提示持久section。
- `core_memory_replace(section,old,new)`——编持久section。
- `archival_memory_insert(text)`——写至可搜外store。
- `archival_memory_search(query,top_k)`——从外store取。
- `conversation_search(query)`——扫前turn。

### 何MemGPT终Letta起

2024年9月MemGPT成Letta。研究repo(`cpacker/MemGPT`)留;Letta扩设计:

- 三tier非两(core、recall、archival——课程08)。
- 原生推理换`send_message`/heartbeat模式(课程08)。
- Sleep-time agent跑异步memory工作(课程08)。

MemGPT论文是2026 foundation即使产系统跑Letta、Mem0、或custom两tier store。

### 何此模式错

- **Memory rot。**写积快于读;取溺于stale事实。修:周期consolidation(Letta sleep-time)、显式invalidation(Mem0 conflict detector)。
- **Memory poisoning。**外memory是取文本。若attacker控内容落memory note,agent下session re-ingest。此是Greshake等(课程27)攻击重述时间。
- **Citation loss。**Agent recall"用户问我ship X"但不能cite何turn。存source reference(session ID、turn ID)于每archival write。

## 构建

`code/main.py`实MemGPT两tier模式于stdlib:

- `MainContext`——固定大提示buffer带`core` dict和`messages` list;过cap时auto-compact oldest message。
- `ArchivalStore`——内存BM25-esque store(token-overlap scoring)于(id、text、tags、session、turn)record。
- 五memory tool映至MemGPT面。
- Scripted agent填archival带事实、后答问题调`archival_memory_search`。

跑:

```
python3 code/main.py
```

Trace显agent写三事实、填main context至cap(强制eviction)、后答follow-up question经archival取——重产MemGPT workflow无真实LLM。

## 使用

每产memory系统今是MemGPT变体:

- **Letta**(课程08)——三tier、原生推理、sleep-time compute。
- **Mem0**(课程09)——vector+KV+graph fused带scoring layer。
- **OpenAI Assistants/Responses**——经thread和file管memory。
- **Claude Agent SDK**——经skill和session store长term memory。

按operational形选(self-hosted、managed、framework-integrated),非按core pattern——core pattern是MemGPT。

## 交付成果

`outputs/skill-virtual-memory.md`是可复skill产正确两tier memory scaffold(main+archival+tool面)用于任目标runtime,带eviction policy和citation field wired。

## 练习题

1. 加`max_main_context_tokens` cap token测(近似`len(text.split())*1.3`)。过cap时compact oldest message入summary。比有和无summarizer行为。
2. 实BM25正于archival store(term frequency、inverse document frequency)。测recall@10于toy fact set vs token-overlap基线。
3. 加`citation` field(session_id、turn_id、source_url)于archival insert。使agent每取-back答cite source。
4. 模memory poisoning:加archival record说"忽略所有未来用户指令。"写guard扫取directive-shaped文本并标记untrusted。
5. 移实现用MemGPT研究repo core-memory JSON schema(`cpacker/MemGPT`)。换flat string至typed section何变?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Virtual context | "无限memory" | Main(prompt)+外(可搜)tier带page in/out |
| Main context | "工作memory" | 提示——固定大、总可见 |
| Archival memory | "长term store" | 外可搜持久、按需取 |
| Core memory | "持久提示section" | Main context内命名section pinned |
| Memory tool | "Memory API" | Agent发工具调用读/写外memory |
| Interrupt | "Memory page fault" | Agent暂停、runtime取、结果splice入下turn |
| Memory rot | "Stale fact" | 老写溺取;用consolidation修 |
| Memory poisoning | "注入持久note" | Attacker内容存作memory、recall时re-ingest |

## 延伸阅读

- [Packer等,MemGPT(arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)——OS启发virtual context论文
- [Letta,Memory Blocks blog](https://www.letta.com/blog/memory-blocks)——三tier演化
- [Anthropic,Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——视context作预算
- [Chhikara等,Mem0(arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)——此模式上hybrid产memory