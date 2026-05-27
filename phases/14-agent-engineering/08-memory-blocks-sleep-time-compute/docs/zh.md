# Memory Blocks和Sleep-Time Compute(Letta)

> MemGPT 2024成Letta。2026演化加两想法:离散功能memory block模型可直编、和sleep-time agent异步consolidate memory当主agent idle。此何scale memory超一对话。

**类型:** 构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程07(MemGPT)
**时间:** ~75分钟

## 学习目标

- 名Letta用三memory tier(core、recall、archival)和每角色。
- 释memory-block模式:Human block、Persona block、和user-defined block作第一类typed object。
- 描述sleep-time compute何、何坐critical path外、和何可跑更强模型于主agent。
- 实scripted两agent循环主agent响应和sleep-time agent间consolidate block。

## 问题背景

MemGPT(课程07)解virtual-memory控流。三产问题涌现:

1. **Latency。**每memory操作坐critical path。若agent需prune、summarize、或reconcile当用户等,tail latency爆。
2. **Memory rot。**写积。矛盾事实留。取溺于stale内容。
3. **Structure loss。**Flat archival store不能表"Human block总在提示;Persona block总在提示;Task block每session换。"

Letta(letta.com)是2026 rewrite。Memory block使结构显;sleep-time compute移consolidation出critical path。

## 概念讲解

### 三tier

| Tier | Scope | 何处 | 写者 |
|------|-------|------|------|
| Core | 总可见 | 主提示内 | Agent tool call+sleep-time rewrite |
| Recall | 对话历史 | 可取 | 自动turn logging |
| Archival | 任事实 | Vector+KV+graph | Agent tool call+sleep-time ingest |

Core是MemGPT core。Recall是对话buffer带evicted tail。Archival是外store。分清MemGPT两tier overloading。

### Memory block

Block是typed、持久、可编core tier section。原MemGPT论文定义两:

- **Human block**——用户事实(name、role、preference、goal)。
- **Persona block**——agent自概念(identity、tone、constraint)。

Letta泛化至任user-defined block:`Task` block用于当前goal、`Project` block用于codebase事实、`Safety` block用于硬constraint。每block有`id`、`label`、`value`、`limit`(char cap)、`description`(使模型知何时编)。

Block经工具面可编:

- `block_append(label,text)`
- `block_replace(label,old,new)`
- `block_read(label)`
- `block_summarize(label)`——condense近limit block。

### Sleep-time compute

2025 Letta加:跑第二agent于background、critical path外。Sleep-time agent处理对话transcript和codebase context、写`learned_context`入共享block、并consolidate或invalidate archival record。

属性fall out:

- **无latency成本。**主响应不等memory op。
- **更强模型许。**Sleep-time agent可是更贵、更慢模型因它不latency-constrained。
- **自然consolidation window。**Dedup、summarize、invalidate矛盾事实当用户不等。

形匹配人何工:你做任务、你sleep于它、长term memory overnight settle。

### Letta V1和原生推理

Letta V1(`letta_v1_agent`,2026)弃`send_message`/heartbeat和inline`Thought:` token代原生推理。Responses API(OpenAI)和带extended thinking Messages API(Anthropic)发推理于分channel、跨轮透传(产于跨provider加密)。控循环仍ReAct。Thought trace是结构非prompt-shaped。

### 何此模式错

- **Block bloat。**无限`block_append`快撞limit。Wire block summarizer于push过cap写前。
- **Silent drift。**Sleep-time agent rewrite block主agent未注意。Version block并surface diff于trace。
- **Poisoned consolidation。**Sleep-time agent处理attacker-reachable内容入core。课程27适用于sleep-time面。

## 构建

`code/main.py`实:

- `Block`——id、label、value、limit、description。
- `BlockStore`——CRUD+`near_limit(label)` helper。
- 两scripted agent——`PrimaryAgent` serve turn、`SleepTimeAgent`间consolidate。
- Trace显三turn对话带block write、加sleep pass summarize block并invalidate stale fact。

跑:

```
python3 code/main.py
```

Transcript显分:primary turn快并产raw write;sleep pass compact并cleanup。

## 使用

- **Letta**(letta.com)用于参考实现。Self-host或managed cloud。
- **Claude Agent SDK skill**作block-shaped knowledge——skill是命名、versioned、可取instruction block agent按需load。
- **Custom build**用于team欲控storage backend。用Letta API contract使可后迁移。

## 交付成果

`outputs/skill-memory-blocks.md`生Letta形block系统带sleep-time hook用于任runtime,含安全规则和citation wiring。

## 练习题

1. 加`block_summarize`工具`near_limit`回true时换block值用模型生summary。何trigger threshold最小both summarization call和block overflow?
2. 实sleep-time dedup于archival:两record text>90% token overlap collapse至一。仅sleep pass做、critical path上勿。
3. Version block。每写记录old值和diff。露`block_history(label)`使operator可debug"何agent忘X。"
4. 视sleep-time agent作untrusted writer。当它们触Persona或Safety block、require第二agent review于commit前。
5. 移example用Letta API(`letta_v1_agent`)。Block schema何变,原生推理何改trace形?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Memory block | "可编提示section" | Typed、持久、LLM可编core memory segment |
| Human block | "用户memory" | 用户事实、pinned于core |
| Persona block | "Agent identity" | 自概念、tone、constraint、pinned于core |
| Sleep-time compute | "异步memory工作" | 第二agent于critical path外consolidation |
| Core/Recall/Archival | "Tier" | 三层memory分:总可见/对话/外 |
| Block limit | "Cap" | 每block char限;强制summarization |
| 原生推理 | "Thinking channel" | Provider级推理输出、非prompt级`Thought:` |
| Learned context | "Sleep output" | Sleep-time agent写入共享block事实 |

## 延伸阅读

- [Letta,Memory Blocks blog](https://www.letta.com/blog/memory-blocks)——block模式
- [Letta,Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute)——异步consolidation
- [Letta,Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent)——原生推理rewrite
- [Packer等,MemGPT(arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)——起源