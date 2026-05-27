# Skill Library和终身学习(Voyager)

> Voyager(Wang等,TMLR 2024)视可执行代码作skill。Skill命名、可取、可组、并环境反馈refine。此是Claude Agent SDK skill、skillkit、和2026 skill-library模式参考架构。

**类型:** 构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程07(MemGPT)、阶段14课程08(Letta Blocks)
**时间:** ~75分钟

## 学习目标

- 名Voyager三成分——automatic curriculum、skill library、iterative prompting——和每角色。
- 释何Voyager使action space代码非primitive command。
- 实stdlib skill library带注册、取、组合、和失败驱动refinement。
- 映Voyager模式于2026 Claude Agent SDK skill和skillkit ecosystem。

## 问题背景

每session从scratch重建能力agent做三错:

1. **浪费token。**每任务重elicit同推理。
2. **失进度。**Session A学correct不转至Session B。
3. **长horizon composition失败。**复杂任务需能力hierarchy;one-shot提示不能表。

Voyager答:视每可复能力作命名代码块存于library、similarity可取、与其他skill可组、并执行反馈refine。

## 概念讲解

### 三成分

Voyager(arXiv:2305.16291)结构agent于:

1. **Automatic curriculum。**Curiosity驱动proposer按agent当前skill set和环境态pick下任务。探索bottom-up。
2. **Skill library。**每skill是可执行代码。任务成功时新skill加。Skill按query-to-description similarity取。
3. **Iterative prompting mechanism。**失败时,agent收执行错、环境反馈、和self-verification output,后refine skill。

Minecraft评(Wang等,2024):3.3x更多unique item、8.5x更快stone tool、6.4x更快iron tool、2.3x更长map traversal vs基线。数Minecraft-specific但模式转。

### Action space=代码

多agent发primitive command。Voyager发JavaScript function。Skill是:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

从sub-skill组。Keyed于description和embedding存。取作program非prompt。

此是2026 Claude Agent SDK skill:命名、可取代码块加instruction agent按需load。

### Skill retrieval

新任务"造diamond pickaxe。"Agent:

1. Embed task description。
2. Query skill library top-k相似skill。
3. 取`craftIronPickaxe`、`mineDiamond`、`placeCraftingTable`等。
4. 从取primitive+新logic组新skill。

此是MCP resource(阶段13)和Agent SDK skill实模式:取于knowledge/code面、scoped当前任务。

### Iterative refinement

Voyager feedback loop:

1. Agent写skill。
2. Skill跑于环境。
3. 三信号之一回:`success`、`error`(带stack trace)、`self-verification failure`。
4. Agent用信号作context rewrite skill。
5. Loop直到成功或max round。

此是Self-Refine(课程05)用于代码生成带环境扎根验证。CRITIC(课程05)是同模式带外工具作verifier。

### Curriculum和探索

Voyager curriculum module按agent有何和未做提任务如"lake旁建shelter"。Proposer用环境态+skill inventory pick任务just above当前能力——探索sweet spot。

于产agent此译作"何missing" operator:给当前skill library和domain、何skill未cover?Team典型手动实curriculum review。

### 何此模式错

- **Skill library rot。**同skill 10次加带略异description。加write deduplication;取仅回一。
- **Composed-skill drift。**Parent skill依赖refined child。Version skill;pinned v1 parent不magic pick up v3。
- **Retrieval quality。**Skill description上vector retrieval library过数百降。Supplement tag filter和硬constraint("仅`category=tooling`skill")。

## 构建

`code/main.py`实stdlib skill library:

- `Skill`——name、description、code(作string)、version、tag、dependency。
- `SkillLibrary`——register、search(token overlap)、compose(dep topological sort)、和refine(update version bump)。
- Scripted agent注册三primitive skill、组第四、撞失败、并refine。

跑:

```
python3 code/main.py
```

Trace显library write、取、组合、失败执行、和v2 refinement——Voyager loop端到端。

## 使用

- **Claude Agent SDK skill**(Anthropic)——2026 reference:每skill有description、code、和instruction;agent session按需load。
- **skillkit**(npm:skillkit)——跨agent skill管理用于32+ AI coding agent。
- **Custom skill library**——domain-specific(SQL skill用于data agent、Terraform skill用于infra agent)。Voyager模式scale down。
- **OpenAI Agents SDK `tools`**——低端;每tool是轻skill。

## 交付成果

`outputs/skill-skill-library.md`生Voyager形skill library带注册、取、versioning、和refinement wired用于任目标runtime。

## 练习题

1. 加dependency-cycle detector于`compose()`。Skill A依赖B依赖A时何?Error vs warning?
2. 实per-skill version pinning。Parent skill组child `crafting@1`时、refine至`crafting@2`必不silent upgrade parent。
3. 换token-overlap retrieval用sentence-transformers embedding(或BM25 stdlib impl)。测50-skill toy library retrieval@5。
4. 加"curriculum" agent:给当前library和domain description、提5 missing skill。每周调。
5. 读Anthropic Claude Agent SDK skill docs。移toy library至SDK skill schema。Discoverability何变?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Skill | "可复能力" | 命名代码块+description、similarity可取 |
| Skill library | "Agent how-to memory" | Skill持久store、可搜可组 |
| Curriculum | "任务proposer" | Bottom-up goal generator驱动当前能力gap |
| Composition | "Skill DAG" | Skill调skill;执行上topological sort |
| Iterative refinement | "自correct loop" | 环反馈+错+self-verification fold入下version |
| Action-space-as-code | "Programmatic action" | 发function非primitive command用于时间扩展行为 |
| Dedup on write | "Skill collapse" | Near-duplicate description collapse至一canonical skill |

## 延伸阅读

- [Wang等,Voyager(arXiv:2305.16291)](https://arxiv.org/abs/2305.16291)——原skill-library论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)——skill作2026 productization
- [Anthropic,Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——skill和subagent实践
- [Madaan等,Self-Refine(arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)——Voyager下refinement loop