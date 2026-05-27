# 毕业项目 17 —— 个人AI Tutor (自适应、多模态、带记忆)

> Khanmigo (Khan Academy)、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat、和Synthesis Tutor皆于2026规模出货自适应多模态辅导。同形态是Socratic policy (绝不直接dump答案)、每次交互后更新learner model (Bayesian knowledge tracing style)、语音 + 文本 + photo-math输入、课程图检索、spaced-repetition调度、和年龄适宜内容硬安全过滤。毕业项目是出货特定科目tutor (K-12代数或intro Python)、两周10 learners efficacy study、并通过内容安全审计。

**类型:** 毕业项目
**语言:** Python (backend、learner model)、TypeScript (web app)、SQL (课程图via Postgres + Neo4j)
**前置要求:** 第5阶段(NLP)、第6阶段(语音)、第11阶段(LLM工程)、第12阶段(多模态)、第14阶段(智能体)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间:** 30小时

## 问题背景

自适应辅导曾是ed-tech研究小众。2026已是消费产品。Khanmigo部署大多数美国学区。Duolingo Max达千万MAUs。Google LearnLM / Gemini for Education于Google Classroom power tutoring。Quizlet Q-Chat坐flashcards旁。Synthesis Tutor于curious-kids辅导走红。同元素: 多模态输入(打字、说、拍方程)、Socratic pedagogy(先问后解释)、每次交互后更新learner model、和严格年龄适宜安全。

将建特定群体之一。测量bar是实际efficacy study: 两周10 learners前后测分数。语音loop须自然(毕业项目03 sub-stack)。记忆须隐私尊重。安全filter须通过K-12 COPPA-aware red-team。

## 概念讲解

四组件。**Tutor policy**是Socratic loop: learner要答案时policy问引导问题; 正确时移下概念; 卡住时提供scaffolded hint。**Learner model**是Bayesian knowledge tracing (或简单变种)每交互后更新每课程节点掌握概率。**课程图**是Neo4j概念带prerequisite edges; policy walk图选下概念。**记忆**是episodic + semantic store (agentmemory-style)持过去交互、错误、和偏好。

UX多模态。文本输入打字答案。语音输入via LiveKit + Whisper (复用毕业项目03)。数学问题照片输入via dots.ocr或PaliGemma 2。语音输出via Cartesia Sonic-2。安全用Llama Guard 4加年龄适宜过滤(block成人内容、暴力、自伤害)和COPPA-aware记忆retention policy。

Efficacy study是deliverable。10 learners、前后测、两周。报学习增益delta和置信区间。比非自适应baseline(同内容线性交付无tutor policy)。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 科目选择: K-12代数或intro Python (择一深度)
- Tutor policy: LangGraph over Claude Sonnet 4.7 (带prompt caching)
- Learner model: Bayesian knowledge tracing (classic)或FSRS spacing
- 课程图: Neo4j概念 + prerequisite edges + OER内容
- 记忆: agentmemory-style持久向量 + episodic + semantic store
- 语音: LiveKit Agents 1.0 + Cartesia Sonic-2 (复用毕业项目03 sub-stack)
- 数学照片: dots.ocr或PaliGemma 2方程识别
- 安全: Llama Guard 4 + 自定义年龄适宜过滤
- 评估: Bloom-level问题生成、前后测试harness、efficacy study tooling

## 动手实践

1. **课程图。** 建Neo4j 50-150概念节点(如K-12代数从"数轴"到"二次公式")带prerequisite edges。每节点附OER内容(Open Textbook、OpenStax)。

2. **Learner model。** Initialize Bayesian knowledge tracing带prior: guess、slip、learn-rate。每交互后每概念掌握更新。每learner持久。

3. **Tutor policy。** LangGraph节点: `read_signal` (learner答案正确/部分/卡住?)、`select_concept` (walk课程图选最高优先概念)、`scaffold` (Socratic提示)、`update_mastery`。

4. **记忆。** 每交互写episodic store。错误和偏好提升semantic memory。COPPA-aware retention policy: 1年后auto-delete、parent-accessible。

5. **语音路径。** LiveKit Agents worker附tutor policy。ASR via Whisper-v3-turbo。TTS via Cartesia Sonic-2。Barge-in支持(复用毕业项目03 mechanics)。

6. **数学照片路径。** 上传或捕获图像; 运行dots.ocr或PaliGemma 2识方程; feed作结构输入给tutor。

7. **安全。** 每模型输出经Llama Guard 4 + 年龄适宜过滤(block自伤害、成人内容、暴力)。记忆访问scoped by learner ID; parental access面删除。

8. **Efficacy study。** 10 learners、前测(标准化30问baseline)、两周tutor交互(每周3 session)、后测。比同内容10 learners非adaptive baseline cohort。

9. **周报。** 每learner自动生成PDF summary探索主题、掌握轨迹、和推荐下步。

## 使用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 产出成果

`outputs/skill-ai-tutor.md`是deliverable。特定科目自适应tutor带多模态输入、learner model、记忆、安全、和测量efficacy。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 学习增益delta | 10 learner两周study前后测delta |
| 20 | Socratic忠实度 | Transcript样本rubric评分 |
| 20 | 多模态UX | 语音 + 照片 + 文本端到端一致性 |
| 20 | 安全 + 隐私态势 | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | 课程广度和图质量 | 概念覆盖 + prerequisite图一致性 |
| **100** | | |

## 练习题

1. 有无adaptive learner model运行efficacy study(随机概念顺序)。报delta。预期adaptive赢、但大小是有趣数。

2. 加多模态probe: 同概念问题文本、语音、和照片交付。测learner是否偏好模态更快收敛。

3. 建parent仪表板: 练习主题、掌握轨迹、下概念、安全事件(任何guardrail hit)。COPPA-aligned。

4. 加语言切换模式: tutor接受西班牙语输入并西班牙语教。测X-Guard覆盖。

5. 压力记忆隐私: 验证learner A经voice-clip重摄入攻击不可见learner B数据。记录尝试访问并alert。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Socratic policy | "问而非dump" | Tutor问引导问题而非给答案 |
| Bayesian knowledge tracing | "BKT" | 每概念掌握概率经典learner-model方程 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 spaced-repetition scheduler、优于SM-2 |
| 课程图 | "概念DAG" | 带prerequisite edges Neo4j概念 |
| Episodic记忆 | "每次交互log" | 每交互存储供later检索 |
| Semantic记忆 | "学习pattern store" | 从episodic提升紧凑错误和偏好 |
| COPPA | "儿童隐私法" | 美国法限制13岁以下儿童数据收集 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考消费K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 备选参考
- [Synthesis Tutor](https://www.synthesis.com) — startup参考
- [FSRS算法](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈