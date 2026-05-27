# Benchmark——WebArena和OSWorld

> WebArena测跨四self-hosted app web-agent能力。OSWorld测跨Ubuntu、Windows、macOS desktop-agent能力。Release时(2023–2024)两显best-in-class agent和人big gap。Gap narrowing;failure mode未变。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14课程19(SWE-bench、GAIA)
**时间:** ~60分钟

## 学习目标

- 描述WebArena四self-hosted app和何execution-based评重要。
- 释何OSWorld用真实OS screenshot而非accessibility API。
- 名两primary OSWorld失败模式:GUI grounding和operational knowledge。
- Summarize OSWorld-G和OSWorld-Human加于base benchmark上何。

## 问题背景

Generalist agent可调tool。可它们drive browser跨20 click complete shopping checkout?可它们config Linux box仅keyboard和mouse?这些是WebArena和OSWorld答问题。

## 概念讲解

### WebArena(Zhou等,ICLR 2024)

- 四self-hosted web app 812长horizon task:shopping site、forum、GitLab-like dev tool、business CMS。
- 加utility:map、calculator、scratchpad。
- 评execution-based经gym API——order placed否、issue closed否、CMS page updated否?
- Release时:best GPT-4 agent 14.41% success vs human 78.24%。

Self-hosted framing重要——benchmark不flaky因target app pinned和reproducible。

### Extension

- **VisualWebArena**——visually grounded task success依赖interpret image(screenshot作first-class observation)。
- **TheAgentCompany**(2024年12月)——加terminal+coding;更像真实remote-work environment。

### OSWorld(Xie等,NeurIPS 2024)

- Ubuntu、Windows、macOS 369真实computer task。
- 真应用free-form keyboard和mouse控。
- 1920×1080 screenshot作observation。
- Release时:best model 12.24% vs human 72.36%。

### Primary失败模式

1. **GUI grounding。**Pixel→element mapping。Model struggle reliable localize UI element于1920×1080。
2. **Operational knowledge。**何menu有setting、何keyboard shortcut、何preference pane。人年建knowledge tail。

### Follow-up

- **OSWorld-G**——564-sample grounding suite+Jedi training set。Decompose grounding从planning使你可分离测它们。
- **OSWorld-Human**——手动curate gold action trajectory。显top agent用1.4-2.7x更多step必要(trajectory-efficiency gap)。

### 何这重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use(课程21)全train WebArena和OSWorld形workload。Benchmark是target;产model是ship答。

### 何benchmarking错

- **Screenshot-only eval。**OSWorld screenshot-driven;eval用DOM或accessibility API agent于OSWorld miss grounding challenge。
- **忽略trajectory length。**仅score success-rate miss 1.4-2.7x step inefficiency OSWorld-Human surface。
- **Stale self-hosted app。**WebArena app pin特定version;update不re-curation break comparability。

## 构建

`code/main.py`实toy web-agent harness:

- 最小"shopping app"state machine:list_item、add_to_cart、checkout。
- 3 task gold trajectory。
- Scripted agent试每task。
- Execution-based evaluator(state check)和trajectory-efficiency metric(step vs gold)。

跑:

```
python3 code/main.py
```

Output:每task success rate和trajectory efficiency、mirror OSWorld-Human methodology。

## 使用

- **WebArena Verified**self-host内cluster用于continuous评。
- **OSWorld**VM fleet用于desktop agent。
- **Computer-use agent**(课程21)——Claude、OpenAI CUA、Gemini——全train此类workload。
- **你product flow**——capture你top 20 task gold trajectory;周run agent对它们。

## 交付成果

`outputs/skill-web-desktop-harness.md`建web/desktop agent harness带execution-based eval和trajectory efficiency metric。

## 练习题

1. 扩toy harness第二app(forum)。写3 task加gold trajectory。
2. 加每task trajectory-efficiency reporting。你toy上、agent 1x、2x、或3x over gold?
3. 实"distractor"tool——gold trajectory从未用。Scripted agent被tempt否?
4. 读OSWorld-G。何你己eval分离grounding失败从planning失败?
5. 读WebArena app README。何你upgrade一pinned app version时何break?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| WebArena | "Web agent benchmark" | 4 self-hosted app 812 task;gym-style评 |
| VisualWebArena | "Visual WebArena" | Visually grounded WebArena;screenshot是observation |
| OSWorld | "Desktop agent benchmark" | 真Ubuntu/Windows/macOS 369 task |
| GUI grounding | "Pixel-to-element mapping" | Model 1920x1080 localize UI element |
| Operational knowledge | "OS know-how" | 何menu、何shortcut、何preference pane |
| OSWorld-G | "Grounding suite" | 564 grounding-only sample+training set |
| OSWorld-Human | "Gold trajectory" | Manual expert action sequence测efficiency |
| Trajectory efficiency | "Step over gold" | Agent step count除人minimum |

## 延伸阅读

- [Zhou等,WebArena(arXiv:2307.13854)](https://arxiv.org/abs/2307.13854)——四app web benchmark
- [Xie等,OSWorld(arXiv:2404.07972)](https://arxiv.org/abs/2404.07972)——跨OS desktop benchmark
- [Anthropic,Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)——Claude benchmark-shaped capability
- [OpenAI,Computer-Using Agent](https://openai.com/index/computer-using-agent/)——OSWorld和WebArena数