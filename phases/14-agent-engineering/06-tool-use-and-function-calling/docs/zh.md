# Tool Use 和 Function Calling

> Toolformer(Schick等,2023)开启了自监督工具标注。Berkeley Function Calling Leaderboard V4(Patil等,2025)定义2026标准:40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。单轮已解决。记忆、动态决策、和长horizon工具链未解决。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段13课程01(Function Calling Deep Dive)
**时间:** ~60分钟

## 学习目标

- 解释Toolformer自监督训练信号:仅在执行减少next-token loss时保留工具标注。
- 命名BFCL V4五评估category和每个测什么。
- 实现stdlib工具注册配schema验证、argument coercion、和执行sandboxing。
- 诊断三2026开放问题:长horizon工具链、动态决策、和记忆。

## 问题背景

早期工具用问:模型可预测正确函数调用否?现代工具用问:模型可跨40步链工具、带记忆、带部分可观测性、带工具失败恢复、无幻觉不存在工具否?

Toolformer建立基线:模型可自监督学何时调工具。BFCL V4定义2026评估目标。它们间gap是生产智能体生活空间。

## 概念讲解

### Toolformer (Schick等,NeurIPS 2023)

想法:让模型标注自己预训练语料配候选API调用。对每个候选,执行它。仅当含工具结果减少下token loss时保留标注。在过滤语料上fine-tune。

覆盖工具:calculator、QA system、search engine、translator、calendar。自监督信号纯关于工具是否帮助预测文本——无人标签。

Scale结果:规模时工具用涌现。小模型伤于工具标注;大模型获。这是为何2026前沿模型强工具用内建而多7B模型需显式tool-use fine-tuning可靠。

### Berkeley Function Calling Leaderboard V4 (Patil等,ICML 2025)

BFCL是2026 de facto评估。V4成分:

- **Agentic(40%)**——全智能体轨迹:记忆、multi-turn、动态决策。
- **Multi-Turn(30%)**——配工具链交互对话。
- **Live(10%)**——用户提交真实提示(更硬分布)。
- **Non-Live(10%)**——合成测试例。
- **Hallucination(10%)**——测何时不应调工具。

V3引入态基评估:工具序列后,检查API实际态(如"文件创否?")而非匹配工具调用AST。V4加web search、记忆、和format sensitivity category。

Key 2026发现:单轮function calling近解决。失败集中于记忆(跨轮载context)、动态决策(按前结果选工具)、长horizon链(20+步后drift)、和幻觉检测(无工具fit时拒调)。

### Tool schema

每provider有schema。细节异但形同:

```
name: string
description: string (何做、何时用)
input_schema: JSON Schema (properties、required、types、enums)
```

Anthropic用`input_schema`直接。OpenAI用`function.parameters`。两接受JSON Schema。Description承重——模型读它选对工具。坏工具描述是#1 root cause wrong-tool-picked失败。

### Argument validation

不信任工具调用。验证:

1. **Type coercion。**模型可回字符串"5"于schema说int处。若unambiguous coerce;否则reject。
2. **Enum validation。**Schema说`status in {"open","closed"}`模型发`"in_progress"`时,reject配描述错。
3. **Required fields。**Missing required field->立错观察回模型,非crash。
4. **Format validation。**Date、email、URL——用具体parser验非regex。

每验证失败应回结构观察使模型可正确形重试。

### Parallel tool calls

现代provider支持一轮parallel tool calls。循环:

1. 模型发3工具调用配异`tool_use_id`。
2. Runtime执它们(若独立则并行)。
3. 每结果回作`tool_result` block由`tool_use_id` correlate。

工程规则:视correlation ID承重。换它们得wrong-tool-to-wrong-result routing。

### Sandboxing

工具执行是sandbox边界。见课程09详。短版:每工具应定义read/write面、网络访问、timeout、memory cap。泛`run_shell(cmd)`是red flag;特定`git_status()`更安全。

## 动手实践

`code/main.py`实现生产形工具注册:

- JSON Schema subset validator(仅stdlib)。
- 工具注册配description、input schema、timeout、和executor。
- Argument coercion和enum validation。
- Parallel tool dispatch配correlation ID。
- 错观察作结构字符串。

运行:

```
python3 code/main.py
```

Trace显mini agent一轮调三工具,一故意malformed call被拒配描述错模型可act on。

## 实际应用

每provider有其工具schema——Anthropic、OpenAI、Gemini、Bedrock。若需multi-provider用翻译层(OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter)。BFCL是参考benchmark——工具用是产品中心时shipping前跑于智能体。

## 产出成果

`outputs/skill-tool-registry.md`为给定任务域生工具catalog、schema、和注册。含description-quality check(每工具description告模型何时用否?)。

## 练习题

1. 加"no-op"工具让模型显式拒用任其他工具。BFCL式幻觉测试测。
2. 实argument coercion用于int-as-string和float-as-string。Coercion何始藏真实bug?
3. 加每工具timeout和circuit breaker(3连续失败后拒工具60s)。何改模型恢复方式?
4. 读BFCL V4描述。选一category(如"multi-turn")并跑10 example prompt过智能体。报告pass rate。
5. 移stdlib validator至Pydantic或Zod。何Pydantic/Zod捕toy漏?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Function calling | "Tool use" | 结构输出工具调用配验证schema |
| Toolformer | "自监督工具标注" | Schick 2023——留工具调用其结果减next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark:40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | "模型函数签名" | name、description、argument JSON Schema |
| tool_use_id | "Correlation ID" | 绑工具调用至结果;并行dispatch必需 |
| Hallucination detection | "知何时不调" | V4 category:无工具fit时拒调 |
| Argument coercion | "String-to-int修" | 可预测schema-mismatch窄修;ambiguous时reject |
| Sandboxing | "工具执行边界" | 每工具read/write面、网络、timeout、memory cap |

## 延伸阅读

- [Schick等,Toolformer(arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)——自监督工具标注
- [Berkeley Function Calling Leaderboard(V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)——2026 eval benchmark
- [Anthropic,Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Agent SDK生产工具schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)——function tool type和Guardrail