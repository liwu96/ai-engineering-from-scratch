# LangGraph — 代理状态机

> 手写ReAct循环是`while True`。LangGraph写ReAct循环是可checkpoint、可interrupt、可branch、可time-travel的图。代理未变。Harness变了。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段11课程09(函数调用),阶段11课程14(模型上下文协议)
**时间:** ~75分钟

## 问题背景

你发函数调用代理。三转后出错:模型试返回500工具、用户中途改意或代理未经人批准退款。`while True:`循环无钩。不可暂停、不可回退、不可分支进"若模型择他工具"。一过demo,代理成黑盒或工或不工。

次步一见即明。代理已是状态机—系统提示词加消息历史加待理工具调用加次动作。显化状态机:"模型思考"、"工具运行"、"人批准"节点和其间条件转换边。图显化,harness得四免费物:checkpointing(步间存状态)、interrupts(为人暂停)、streaming(流token和中间事件)、time-travel(回退至前态试别分支)。

LangGraph是发此抽象库。非LangChain意代理框架("这是AgentExecutor,好运")。是一等状态、一等持久化、一等interrupt图运行时。代理循环是画非手写。

## 概念讲解

![LangGraph StateGraph:节点、边和checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph`有三物。

1. **状态。**TypedDict或Pydantic模型流经图。每节点收全状态返部分更新,LangGraph用每字段*reducer*合—`operator.add`为应累列表,默overwrite。
2. **节点。**Python函数`state -> partial_state`。每是离散步:"调模型"、"跑工具"、"总结"。
3. **边。**节点间转换。静态边去一处。条件边取路由函数`state -> next_node_name`使图可按模型输出分支。

编译图。Compile绑拓扑、attach checkpointer(可选但产必需)、返runnable。用初始状态和`thread_id`调用。每执行步持久化checkpoint键`(thread_id, checkpoint_id)`。

### 四超能力

**Checkpointing。**每节点过渡写新状态至存(测用内存、产用Postgres/Redis/SQLite)。用同`thread_id`再调图恢复。图从暂停处续。

**Interrupts。**用`interrupt_before=["human_review"]`标记节点执行停于该节点前。状态持久化。API响用户"待批准"。后请求同`thread_id`带`Command(resume=...)`恢复执行。

**Streaming。**`graph.stream(state, mode="updates")`生状态delta。`mode="messages"`流模型节点内LLM token。`mode="values"`生全快照。你择何UI面。

**Time-travel。**`graph.get_state_history(thread_id)`返全checkpoint日志。传任前`checkpoint_id`至`graph.invoke`从那点fork。适debug("若模型择工具B何如?")和复产trace回归测。

### Reducer是要点

每状态字段有reducer。多默好—新值overwrite旧。但消息列表需`operator.add`使新消息append非replace。并行边经reducer合更新。若两节点都更新`messages`且忘`Annotated[list, add_messages]`,第二静胜且失半转。Reducer是库唯一微妙物;对则余compose。

### 四节点ReAct图

产ReAct代理是四节点两边:

1. `agent`—用当前消息历史调LLM。返assistant消息(可含tool_calls)。
2. `tools`—执最后assistant消息中任tool_calls,append工具结果为tool消息。
3. 从`agent`条件边路由至`tools`若最后消息有tool_calls,否则至`END`。
4. 从`tools`静态边回至`agent`。

此即。你得全ReAct循环(思考→动作→观察→思考→…)带checkpointing、interrupts、streaming,约40行代码。

### StateGraph vs Send(fanout)

`Send(node_name, state)`让节点dispatch并行子图。例:代理决同时查三retriever。每`Send`spawn目标节点并行执行;其输出经状态reducer合。这是LangGraph无线程原语表达orchestrator-workers模式法。

### 子图

编译图可为他图节点。外图见单节点;内图有自己的状态和checkpoint。这是团队建supervisor-worker代理法:supervisor图路由用户intent至每域worker子图。

## 构建

### 步骤1:状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`是reducer使消息列表累非overwrite。忘它是最常见LangGraph bug。

### 步骤2:带thread跑

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每更新是dict `{node_name: state_delta}`。前端可流这些至UI使用户见"代理思考…调search_web…得结果…答。"

### 步骤3:加人机交互interrupt

标记节点使执行停于其运行前。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 每工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"]设。检提tool calls。
# 若批准:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 若拒:写拒消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、checkpoint和thread跨interrupt持久化。执行间外无物内存。

### 步骤4:time-travel debug

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从前checkpoint fork
target = history[3].config  # 三步回
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从那点前重播
```

传`None`作输入从给定checkpoint重播;传值作为更新append至那checkpoint状态前恢复。这是如何无重跑全对话复坏代理run。

### 步骤5:换产checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis和Postgres已发。`MemorySaver`为测。跨重启持久化需真存。

## 技能

> 你建代理为图,非`while True`循环。

达LangGraph前,做60秒设计:

1. **命节点。**每离散决策或副作用动作是节点。"代理思考"、"工具运行"、"审员批准"、"响应流"。若你不可列它们,任务非代理形。
2. **声明状态。**最小TypedDict带每列表字段reducer。勿stuff一切入`messages`;hoist任务特定字段(工作`plan`、`budget`计数器、`retrieved_docs`列表)至顶层。
3. **画边。**静态除非次步依赖模型输出。每条件边需带命分支路由函数。
4. **先择checkpointer。**测用`MemorySaver`,他用Postgres/Redis/SQLite。勿无发—无checkpointer意无resume、无interrupt、无time-travel。
5. **工具运行前决interrupt,非后。**批准在副作用节点进边使你可前取消害;验证在模型出边使你可低成本拒坏调。
6. **默stream。**UI用`mode="updates"`,模型节点内token级streaming用`mode="messages"`,eval全快照用`mode="values"`。

拒发无checkpointer LangGraph代理。拒发interrupt副作用*后*者。拒发`messages`字段无`add_messages` reducer。

## 练习题

1. **易。**实上四节点ReAct图带计算器工具和web-search工具。验`list(app.get_state_history(config))`对两转对话返至少四checkpoint。
2. **中。**加`planner`节点于`agent`前跑并写结构`plan: list[str]`入状态。使`agent`标记plan步完。若`plan`跨checkpoint resume失(错reducer)测败。
3. **难。**建supervisor图路由三子图(`researcher`、`writer`、`reviewer`)用`Send`。每子图有自己的状态和checkpointer。加外图`interrupt_before=["writer"]`使人可批准研究简报。确认从前checkpoint time-travel仅重跑forked分支。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph图" | compile前添加节点和边的builder对象。 |
| Reducer | "字段何合" | 节点返该字段更新时应用函数`(old, new) -> merged`;默overwrite,`add_messages`append。 |
| Thread | "对话ID" | 限定一session所有checkpoint的`thread_id`字符串。 |
| Checkpoint | "暂停状态" | 节点过渡后持久化全图状态快照,键`(thread_id, checkpoint_id)`。 |
| Interrupt | "为人暂停" | `interrupt_before`/`interrupt_after`停执行于节点边界;用`Command(resume=...)`恢复。 |
| Time-travel | "从前步fork" | `graph.invoke(None, config_with_old_checkpoint_id)`从那checkpoint前重播。 |
| Send | "并行子图dispatch" | 节点可返spawn目标节点N并行执行的构造器。 |
| Subgraph | "编译图作节点" | 编译StateGraph用作他图节点;保自己的状态范围。 |

## 延伸阅读

- [LangGraph文档](https://langchain-ai.github.io/langgraph/) — StateGraph、reducers、checkpointers和interrupts典参考。
- [LangGraph概念:状态、reducers、checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课用心智模型,直源。
- [LangGraph持久化和Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis存、checkpoint命名空间和thread IDs细节。
- [LangGraph人机交互](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)`和edit-state模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每LangGraph代理实模式;读reasoning trace理。
- [Anthropic — 建效代理(2024年12月)](https://www.anthropic.com/research/building-effective-agents) — 何图形(chain、router、orchestrator-workers、evaluator-optimizer)何时择。
- 阶段11课程09(函数调用) — 每LangGraph代理节点重用工具调用原语。
- 阶段11课程14(模型上下文协议) — 外工具发现经MCP适配器入LangGraph `ToolNode`。
- 阶段11课程17(代理框架权衡) — 何择LangGraph而非CrewAI、AutoGen或Agno。