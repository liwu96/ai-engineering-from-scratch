# OpenAI Agents SDK——Handoff、Guardrail、Tracing

> OpenAI Agents SDK是建Responses API上轻量multi-agent框架。五primitive:Agent、Handoff、Guardrail、Session、Tracing。Handoff是名`transfer_to_<agent>`工具。Guardrail trip于input或output。Tracing默认on。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程06(Tool Use)
**时间:** ~75分钟

## 学习目标

- 名OpenAI Agents SDK五primitive。
- 释handoff:何建模作tool、何名形模型见、何context transfer。
- 分input guardrail、output guardrail、和tool guardrail;释`run_in_parallel`vs blocking mode。
- 实stdlib runtime带handoff+guardrail+span-style tracing。

## 问题背景

不能cleanly delegate agent end up塞全入一提示。无guardrail agent ship PII、policy-violating output、或loop forever。OpenAI SDK codify三primitive使multi-agent work tractable。

## 概念讲解

### 五primitive

1. **Agent。**LLM+instruction+tool+handoff。
2. **Handoff。**Delegation至另一agent。模型视作名`transfer_to_<agent_name>`工具。
3. **Guardrail。**Input(仅首agent)、output(仅末agent)、或tool调用(每function tool)上验。
4. **Session。**跨turn自动对话历史。
5. **Tracing。**LLM generation、tool call、handoff、guardrail内置span。

### Handoff作工具

模型见`transfer_to_billing_agent`于其tool list。调用它信号runtime:

1. Copy对话context(或经`nest_handoff_history` beta collapse)。
2. Initialize目标agent带其instruction。
3. 续run用目标agent。

此是supervisor模式(课程13/课程28)产品化。

### Guardrail

三flavor:

- **Input guardrail。**首agent input上跑。拒unsafe或out-of-scope请求于任LLM call前。
- **Output guardrail。**末agent output上跑。Catch PII leak、policy violation、malformed response。
- **Tool guardrail。**Per-function-tool跑。验argument、check permission、audit execution。

Mode:

- **Parallel**(default)。Guardrail LLM跑主LLM旁。Lower tail latency。若trip、主LLM工作discard(token waste)。
- **Blocking**(`run_in_parallel=False`)。Guardrail LLM先跑。若trip、无token waste于主call。

Tripwire raise`InputGuardrailTripwireTriggered`/`OutputGuardrailTripwireTriggered`。

### Tracing

默认on。每LLM generation、tool call、handoff、和guardrail emit span。`OPENAI_AGENTS_DISABLE_TRACING=1`opt out。`add_trace_processor(processor)`fan span至己backend旁OpenAI's。

### Session

`Session`存对话历史于backend(SQLite、Redis、custom)。`Runner.run(agent,input,session=session)`自动load和append。

### 何此模式错

- **Handoff drift。**Agent A handoff至Agent B handoff回Agent A。加hop counter。
- **Guardrail bypass。**Tool guardrail仅function tool fire;built-in tool(file reader、web fetch)需分离policy。
- **Over-tracing。**Span敏感内容。配OTel GenAI content-capture rule(课程23)——外存、ID reference。

## 构建

`code/main.py`实SDK形于stdlib:

- `Agent`、`FunctionTool`、`Handoff`(作带transfer语义function tool)。
- `Runner`带input/output/tool guardrail、handoff dispatch、和hop counter。
- 简span emitter示trace形。
- Triage agent按用户query handoff至billing或support;一input guardrail trip。

跑:

```
python3 code/main.py
```

Trace显两成功handoff、一input guardrail trip、和span tree mirror真SDK emit。

## 使用

- **OpenAI Agents SDK**用于OpenAI-first product。
- **Claude Agent SDK**(课程17)用于Claude-first product。
- **LangGraph**(课程13)当你欲显state和durable resume。
- **Custom**当你需exact控(voice、multi-provider、federated deployment)。

## 交付成果

`outputs/skill-agents-sdk-scaffold.md`scaffold Agents SDK app带triage agent、handoff、input/output/tool guardrail、session store、和trace processor。

## 练习题

1. 加handoff hop counter:N transfer后refuse。Trace行为。
2. 实`nest_handoff_history`作option——transfer前collapse前message入一summary。
3. 写blocking output guardrail。比latency于会trip prompt vs pass。
4. Wire`add_trace_processor`至JSON logger。每span何形emit?
5. 读SDK docs。移stdlib toy至`openai-agents-python`。何你model错?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent | "LLM+instruction" | SDK Agent type;own tool和handoff |
| Handoff | "Transfer" | 模型调delegate至另一agent tool |
| Guardrail | "Policy check" | Input/output/tool调用上验 |
| Tripwire | "Guardrail trip" | Guardrail reject时raise exception |
| Session | "History store" | Run间持久对话memory |
| Tracing | "Span" | LLM+tool+handoff+guardrail上内置可观测 |
| Blocking guardrail | "Sequential check" | Guardrail先跑;trip无token waste |
| Parallel guardrail | "Concurrent check" | Guardrail旁跑;lower latency、trip浪费token |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)——primitive、handoff、guardrail、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude-flavored counterpart
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——何时reach handoff at all
- [OpenTelemetry GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——Agents SDK span map标准