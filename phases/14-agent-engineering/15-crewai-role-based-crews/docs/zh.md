# CrewAI——Role-Based Crew和Flow

> CrewAI是2026 role-based multi-agent框架。四primitive:Agent、Task、Crew、Process。两顶级形:Crew(autonomous、role-based collaboration)和Flow(事件驱动、deterministic)。Docs直:"用于任产ready应用、从Flow起。"

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程12(Workflow Pattern)、阶段14课程14(Actor Model)
**时间:** ~75分钟

## 学习目标

- 名CrewAI四primitive(Agent、Task、Crew、Process)和每own何。
- 分Sequential、Hierarchical、和planned Consensus process;每workload pick一。
- 分Crew(autonomous role-based)和Flow(事件驱动deterministic)、并释docs产推荐。
- Wire tool用`@tool`装饰器和`BaseTool`子类;reason结构输出vs free text。
- 名四CrewAI memory类型和何时每payoff。
- 实stdlib三agent crew(researcher、writer、editor)产brief。
- Spot三CrewAI失败模式:prompt-bloat、manager-LLM tax、brittle handoff。

## 问题背景

Team采纳multi-agent framework撞同墙。"Autonomous collaboration"demo看大。然后customer file bug你需deterministic replay。或finance问LLM-routed crew每run成本何。或on-call需知何agent 3 AM stall。

Free-form LLM-routed crew clean答那些无。纯DAG全答但失brainstorming agent需exploratory形。

CrewAI分诚实trade。Crew用于collaborative、role-based、exploratory工作。Flow用于事件驱动、code-owned、auditable产。同框架、两形、每surface pick。

## 概念讲解

### 四primitive

CrewAI面小。记此余是config。

- **Agent。**`role+goal+backstory+tool+(optional)llm`。Backstory承重。它形tone、judgment、何agent stop。Tool是agent可调function(更下)。
- **Task。**`description+expected_output+agent+(optional)context+(optional)output_pydantic`。可复工作单元。`expected_output`是contract。`context`列上游task其output传入。`output_pydantic`强结构形。
- **Crew。**Container。Own`agent`列表、`task`列表、`process`、和可选`memory`+`verbose`+`manager_llm`设置。
- **Process。**执行策略。Sequential、Hierarchical、Consensus(planned)。Pick run形。

Agent不直见彼此。Task reference agent。Crew sequence task。Process决何pick下task。那是整mental model。

> **验证于**CrewAI 0.86(2026-05)。新版可rename或merge process type;依赖特定形前查[CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。**Task按declaration序跑。Task N output可用作`context`于Task N+1。最低成本。最可预测。序固定时用。
- **Hierarchical。**Manager Agent(分离LLM call)间router specialist。CrewAI从你`manager_llm` config或default spawn manager。Manager每轮pick下task并可refuse或re-route。当你有四或更多specialist并序真依赖前output时用。
- **Consensus。**Planned、现public API未实。Docs留名用于future voting-based process。勿今依赖。

Hierarchical加每轮LLM call(manager)于每specialist call上。五步run token cost可三倍。仅当你需routing时付。

### Crew vs Flow

此是2026 docs lead framing。

- **Crew。**LLM驱动autonomy。Framework runtime pick形。好用于:研究、brainstorming、首draft、何path是答部分处。难replay。难test。便宜prototype。
- **Flow。**事件驱动graph你own。`@start`标entry。`@listen(topic)`标step当另step emit该topic时fire。每step是plain Python(可内调Crew)。好用于:产。Observable。Testable。Deterministic。

Docs 2026产推荐:从Flow起。当autonomy earn cost时从Flow step内fold Crew作`Crew.kickoff()` call。Flow给audit trail、Crew给exploration。组勿pick。

### Tool集成

三方式给Agent tool。Pick最简fit。

1. **`@tool`装饰器。**Pure function成tool。签名是schema;docstring是LLM见description。最用于one-off helper。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool`子类。**Class-based tool带显args schema、async support、retry。Tool有态(client、cache)或需结构args时用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkit。**CrewAI发first-party adapter:`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一import wired。

结构输出用Pydantic。Task上pass`output_pydantic=MyModel`。CrewAI验LLM response于model并coerce或retry。配紧`expected_output` string。Draft free-text output fine;结构输出是下游Flow可consume。

### Memory hook

CrewAI发四memory类型out of box。它们组:Crew可同时enable全四。

> **验证于**CrewAI 0.86(2026-05)。近release route一切经统一`Memory`系统wrap这四store。下conceptual model仍hold,但public class面可collapse至单`Memory` entry-point于新版;查[CrewAI memory docs](https://docs.crewai.com/concepts/memory)当前API。

- **Short-term。**单run内对话buffer。末wipe。
- **Long-term。**跨run持久。存于vector DB(Chroma default、swappable)。取similarity于当前task。
- **Entity。**Per-entity事实。"Customer X enterprise plan。"Keyed entity非similarity。跨run存。
- **Contextual。**Assembly-time取。Agent需时pull相关memory、非preloaded。

Crew上用`memory=True`或per-type config enable。Backed于你config embedding provider(default OpenAI、swappable至local)。Memory是CrewAI earn其keep处vs更薄framework;纯LangGraph需你wire每这些。

### 何CrewAI fit

- 三至六agent带命名role和collaborative workflow。Draft、review、plan、brainstorm。
- Routing何LLM下步judgment是value部分(Hierarchical)。
- 何team happier读`role+goal+backstory`比读graph definition。

### 何CrewAI不fit

- 确定DAG带严序。用LangGraph(课程13)。Graph形是正确抽象;CrewAI role framing是friction。
- Sub-second latency预算。Hierarchical加round trip。甚至Sequential serialize prompt含backstory和前output。
- Single-agent循环。跳framework;agent循环(课程1)加tool registry更短。

课程17(Agent Framework Tradeoff)矩阵lay out此。短版:CrewAI坐"collaborative role-based"角。

### 依赖形

LangChain无关。Python 3.10至3.13。用`uv`。Star数:见[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)(2026-05 snapshot)。AWS Bedrock集成document;vendor benchmark report QA workload substantial speedup vs LangGraph,但methodology(dataset、hardware、eval metric)未publish,故视framework-vendor数directional only。

### 何此模式错

- **Backstory prompt-bloat。**每agent 2000-word backstory和五agent crew烧context budget首tool call前。Keep backstory 200 word下。Agent间reuse phrase;勿五次repeat house style。
- **Manager-LLM token税。**Hierarchical process加manager LLM call于每specialist call前。五task crew是六LLM call非五、manager call载全task list加前output。输出依赖routing时换Sequential。
- **Brittle handoff。**Task N`expected_output`是"outline"。Task N+1读作`context`并试parse三section。LLM产四。下游Agent ad-lib。Task N上用`output_pydantic`修使Task N+1读typed object非free text。
- **Crew-as-prod。**Free-form Crew ship产无Flow wrapper。Output variability高;replay不可能;on-call不能diff坏run比好run。用Flow wrap。

## 构建

`code/main.py`实stdlib两形加三agent crew。

形:

- `Agent`、`Task` dataclass匹配CrewAI面。
- `SequentialCrew.kickoff(input)`按declaration序跑task、thread output作`context`。
- `HierarchicalCrew.kickoff(topic)`加manager Agent每轮pick下specialist、"done"停。
- `Flow`带`@start`和`@listen(topic)`装饰器、小event loop、和trace。
- `tool(name)`装饰器mirror CrewAI`@tool`形。
- `Memory`带`short_term`、`long_term`、`entity` store;mock similarity用numpy。
- Mock LLM response硬code string keyed role加input prefix。无网络。Deterministic。

具体demo:researcher、writer、editor crew产brief于"agent engineering 2026"。Researcher pull(mock)source。Writer draft。Editor tighten。同crew经Flow跑示deterministic形。

跑:

```bash
python3 code/main.py
```

Trace覆盖:sequential crew thread output经`context`、hierarchical crew manager pick(researcher、writer、editor、后"done")、flow跑同三步带显topic(`researched`、`drafted`、`edited`)、tool call经`@tool` route、和long-term memory跨两kickoff存。

Crew trace fluid;manager可principle reorder。Flow trace fixed。那选是lesson。

## 使用

- **CrewAI Flow**用于产。即使Flow是一step调`Crew.kickoff()`。Flow给audit boundary。
- **CrewAI Crew(Sequential)**用于clear-ordering collaborative工作、尤其首draft和review循环。
- **CrewAI Crew(Hierarchical)**当routing依赖output并你有四或更多specialist。
- **LangGraph**(课程13)用于显state machine、durable resume、严序。
- **AutoGen v0.4**(课程14)用于actor-model concurrency和fault isolation。
- **OpenAI Agents SDK**(课程16)用于OpenAI-first product带handoff和guardrail。
- **Claude Agent SDK**(课程17)用于Claude-first product带subagent和session store。

## 交付成果

`outputs/skill-crew-or-flow.md`为任务pick Crew vs Flow并scaffold最小实现。硬reject Crew-without-backstory、Flow-without-explicit-topic、Hierarchical under三specialist。

## 陷阱

- **Backstory作flavor。**它形output。每agent测三variant;variance真实。Pick一、freeze。
- **跳`expected_output`。**每task无contract、下游task pickup何LLM产。Crew跑;audit失败。
- **Memory always-on。**每run Long-term write。Vector DB增。Retrieval得噪。Scope write于task何fact持久。
- **Manager prompt drift。**Hierarchical manager prompt implicit。Routing得怪时、verbose mode dump并读。
- **Tool side effect于Crew。**Crew可调tool更多次于预期。POST、DELETE、payment属Flow step、非Crew tool。

## 练习题

1. 转Sequential crew至Flow。数touchpoint variability drop。Note何readability降。
2. 加entity memory于crew:customer事实跨kickoff存。验取pull正确entity。
3. 实Hierarchical process manager refuse route至editor直到writer output至少三paragraph。Trace retry。
4. Wire`BaseTool`子类用于(mock)web search。比trace形vs`@tool`装饰器版。
5. 加`output_pydantic=Brief`于editor task、`Brief`有`title`、`summary`、`section`。使writer task output malformed JSON一次;验CrewAI retry behavior于trace。
6. 读CrewAI docs intro。移toy至真`crewai` API。Stdlib版跳何guarantee?
7. Wire AgentOps或Langfuse(课程24)至真run。Stdlib版miss何trace?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent | "Persona" | Role+goal+backstory+tool |
| Task | "工作单元" | Description+expected output+assignee+可选结构输出 |
| Crew | "Agent team" | Agent+Task+Process container |
| Process | "执行策略" | Sequential/Hierarchical/Consensus(planned) |
| Flow | "Deterministic workflow" | 事件驱动、code-owned、testable |
| Backstory | "Persona prompt" | Agent tone和judgment shaper |
| `@tool` | "Function tool" | 装饰器转function入Agent可调tool |
| `BaseTool` | "Class tool" | Class-based tool带args schema、retry、async support |
| Entity memory | "Per-entity事实" | Memory scoped至customer/account/issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory跨kickoff存 |
| Contextual memory | "Just-in-time取" | Agent需时pull memory |
| Manager LLM | "Router agent" | Hierarchical process extra LLM pick下task |
| `expected_output` | "Task contract" | String告Agent(和audit)何形回 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction):concept和推荐产path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows):事件驱动形、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools):`@tool`、`BaseTool`、built-in toolkit
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory):short-term、long-term、entity、contextual
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents):何时multi-agent帮何时不
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview):state-machine alternative