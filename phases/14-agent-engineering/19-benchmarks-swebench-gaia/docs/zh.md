# Benchmark——SWE-bench、GAIA、AgentBench

> 三benchmark锚2026 agent评。SWE-bench测代码patching。GAIA测generalist tool use。AgentBench测多环境reasoning。知它们成分、它们contamination story、和它们不测何。

**类型:** 学习
**语言:** Python(stdlib)
**前置要求:** 阶段14课程06(Tool Use)
**时间:** ~60分钟

## 学习目标

- 名SWE-bench测试harness(FAIL_TO_PASS)并释何gates于unit test。
- 释何SWE-bench Verified(OpenAI、500 task)存在和何它remove。
- 描述GAIA设计:人简、AI难;三difficulty level。
- 名AgentBench八环境和OSS LLM主要blocker。
- Summarize SWE-bench+ contamination发现和其implication。

## 问题背景

Leaderboard告你何模型一benchmark赢。不告你:

- Benchmark是否contaminate(训练solution、test leak)。
- Benchmark是否测你care何(code vs browsing vs generalist)。
- Evaluator是否robust(AST match、state check、人review)。

知三锚benchmark和失败模式于quote数前。

## 概念讲解

### SWE-bench(Jimenez等,ICLR 2024 oral)

- 12流行Python repo 2,294真实GitHub issue。
- Agent得:pre-fix commit codebase+natural-language issue description。
- Agent产:patch。
- Evaluator:apply patch、跑repo test suite。Patch必须flip FAIL_TO_PASS test(前fail现pass)不break PASS_TO_PASS test。

SWE-agent(Yang等,2024)达release 12.5%经强调agent-computer interface(file editor command、search syntax model understand)。

### SWE-bench Verified

OpenAI,2024年8月。人curate 500-task subset。Remove ambiguous issue、unreliable test、和fix unclear task。"你agent ship真patch否?"主要benchmark。

### Contamination

- SWE-bench issue超94%前most model cutoff。
- **SWE-bench+**发现32.67%成功patch leak solution于issue text(model见fix于description)、和31.08%因弱test coverage suspicious。
- Verified更clean但不contamination-free。

实际implication:SWE-bench 50%模型可SWE-bench+ 35%。若claim SWE-bench performance常报两。

### GAIA(Mialon等,2023年11月)

- 466 question;300 retain用于huggingface.co/gaia-benchmark private leaderboard。
- 设计哲学:"conceptually simple for human(92%)but hard for AI(GPT-4 with plugin:15%)。"
- 测reasoning、multi-modality、web、tool use。
- 三difficulty level;Level 3需跨modalities长tool chain。

GAIA是你跑测"generalist capability"用。勿confuse code-specific benchmark。

### AgentBench(Liu等,ICLR 2024)

- 8环境跨code(Bash、DB、KG)、game(Alfworld、LTP)、web(WebShop、Mind2Web)、和open-ended generation。
- Multi-turn、每split~4k-13k turn。
- 主要发现:long-term reasoning、decision-making、和instruction following是OSS LLM catch up commercial blocker。

### 何这些不测

- 真实operational cost(token、wall-clock)。
- Adversarial condition安全行为。
- 你domain performance(用己eval、课程30)。
- Tail failure(benchmark平均;产operator care worst 1%)。

### 何benchmarking错

- **Single-number fixation。**SWE-bench 50%告你少比P50/P75/P95 cost+step分布。
- **Contaminated claim。**Report SWE-bench不提Verified或SWE-bench+ misleading。
- **Benchmark-as-development-target。**Optimize benchmark diverge产usefulness。

## 构建

`code/main.py`实toy SWE-bench-like harness:

- 合成bug-fix task(3 task)。
- Scripted"agent"提patch。
- Test runner check FAIL_TO_PASS(bug现fix)和PASS_TO_PASS(无break)。
- GAIA-style difficulty classifier基于question decomposition depth。

跑:

```
python3 code/main.py
```

Output显每task+每difficulty resolution rate并make evaluator rule concrete。

## 使用

- **SWE-bench Verified**用于code agent。常报Verified score。
- **GAIA**用于generalist agent。用private leaderboard split。
- **AgentBench**用于多环境比较。
- **Custom eval**(课程30)用于你product实际形。

## 交付成果

`outputs/skill-benchmark-harness.md`建任何codebase-task pair SWE-bench-style harness带FAIL_TO_PASS/PASS_TO_PASS gating。

## 练习题

1. 移toy harness跑真实repo(pick你一)。写3 known bug FAIL_TO_PASS test。
2. 加step-count metric。你3 task上、何agent step每resolution?
3. 读SWE-bench+论文。实solution-leakage check(issue text pattern match diff)。
4. Download GAIA question从public split。Trace GPT-4-class agent何做。何tool需?
5. 读AgentBench per-environment breakdown。何环境mirror你product surface?"SOTA"何看?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| SWE-bench | "Code agent benchmark" | 2,294 GitHub issue;patch必须flip FAIL_TO_PASS test |
| SWE-bench Verified | "Clean SWE-bench" | 500人curate task、OpenAI |
| FAIL_TO_PASS | "Fix gate" | 前fail test patch后须pass |
| PASS_TO_PASS | "No-regression gate" | Pass test仍须pass |
| GAIA | "Generalist benchmark" | 466人易/AI难 multi-tool question |
| AgentBench | "Multi-env benchmark" | 8环境;长horizon multi-turn |
| Contamination | "训练set leak" | Benchmark task present model training |
| SWE-bench+ | "Contamination audit" | 32.67% solution leakage found成功SWE-bench patch |

## 延伸阅读

- [Jimenez等,SWE-bench(arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)——原benchmark
- [OpenAI,SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)——curated subset
- [Mialon等,GAIA(arXiv:2311.12983)](https://arxiv.org/abs/2311.12983)——generalist benchmark
- [Liu等,AgentBench(arXiv:2308.03688)](https://arxiv.org/abs/2308.03688)——多环境suite