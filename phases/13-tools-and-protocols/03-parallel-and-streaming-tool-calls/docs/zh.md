# 并行工具调用和带工具流

> 三独立天气查找序列化是三轮。并行跑总时间塌至最慢单调用。每前沿提供者现于一轮发多工具调用。收益真;管道微妙。本课走两半:并行扇出和流参数重组,强调id关陷阱。

**类型:** 构建
**语言:** Python(stdlib,线程池+流框架)
**前置要求:** 阶段13课程02(函数调用深究)
**时间:** ~75分钟

## 学习目标

- 释`parallel_tool_calls: true`何存及何时禁。
- 流参数块间关至正工具调用id于并行扇出。
- 重组部分`arguments`字符串入完整JSON不早解析。
- 跑三城天气bench示序vs并延迟。

## 问题背景

无并行调用,agent答"Bengaluru、Tokyo、Zurich天气何"做此:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
宿主 -> 运执行器,回结果
LLM -> call get_weather(Tokyo)
宿主 -> 运执行器,回结果
LLM -> call get_weather(Zurich)
宿主 -> 运执行器,回结果
LLM -> 终文本答
```

三轮LLM,每亦付执行器延迟。约4倍理想wall-clock。

带并行调用:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
宿主 -> 并发运三执行器,回三结果
LLM -> 终文本答
```

一轮LLM。执行器时间是三最大而非和。OpenAI、Anthropic、Gemini产bench于扇出负载示60至70% wall-clock减。

代价是关复杂。当三调用乱序完成,结果须载匹配`tool_call_id`使模型可对齐。当结果流,须组装部分参数片段入完整JSON后执。Gemini 3加unique id正解两同工具并行调用不可区分实问题。

## 概念讲解

### 启并行

- **OpenAI。**`parallel_tool_calls: true`默认开。设`false`强串行。
- **Anthropic。**并行经`disable_parallel_tool_use: false`(Claude 3.5及以上默认)。设`true`串行。
- **Gemini。**总并行能;`tool_config.function_calling_config.mode = "AUTO"`让模型决。

禁并行当工具有序依赖(`create_file`后`write_file`)、一调用输出另调用输入、或速率限器不可handle 扇出。

### Id关

模型发每调用有`id`。宿主回每结果须含同id。无此,结果歧义。

- **OpenAI。**每工具角色消息上`tool_call_id`。
- **Anthropic。**每`tool_result`块上`tool_use_id`。
- **Gemini。**每`functionResponse`上`id`(Gemini 3及以上;Gemini 2按名匹配破同名并行调用)。

### 并发运行调用

宿主于己线程、协程或远程worker运行每调用执行器。简框架用线程池;产用`asyncio.gather`或结构并发asyncio。完成序不可预——id是标识。

一常见bug:按调用列表序而非完成序回结果。这通常工因模型仅关`tool_call_id`,但若结果丢或重,乱序提交使调试难。偏好完成序带显id回。

### 流工具调用

当模型流,`arguments`片段到。三并行调用三分离流块于一线交错。需每id一累加器。

按提供者形:

- **OpenAI。**每块是`choices[0].delta.tool_calls[i].function.arguments`(部分字符串)。块载`index`(调用列表位置)。你按index累,初现时读`id`,于`finish_reason = "tool_calls"`解析JSON。
- **Anthropic。**流事件是`message_start`,后每块一`content_block_start`带类型`tool_use`(含id、name、空input)。`content_block_delta`事件载`input_json_delta`块。`content_block_stop`闭每块。
- **Gemini。**`streamFunctionCallArguments`(Gemini 3及以上)发带`functionCallId`块使调用干净交错。Gemini 3前,流回一时一完整调用。

### 部分JSON和parse-early陷阱

`arguments`完成前不可解析。部分JSON如`{"city": "Beng`无效会raise。正门是提供者调用终信号:OpenAI`finish_reason = "tool_calls"`、Anthropic`content_block_stop`或Gemini流终事件。仅后试`json.loads`。更健壮法用增量JSON解析器结构完成时yield事件;OpenAI流指南荐此用于示活"思考"指示器UX。括号计作为完整性测不可靠(引号内括号或转义内容致假正)且仅应作非正式调试启发。

### 乱序完成

```
call_A: 快API,先回
call_B: 慢API,次回
call_C: 中API,三回
```

宿主回仍须引id:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复序对OpenAI或Anthropic正确无关。Gemini接任序只要id匹配。

### Bench:序vs并

`code/main.py`框架模拟三执行器带400、600、800 ms延迟。序跑1800 ms总。并跑max(400,600,800)=800 ms。差是常数非比例,故省随工具数增。

实世警:并行调用压下游API。10-way 扇出至速率限服务会失败。阶段13课程17覆盖gateway级backpressure;重试语义规划于未来phase。

### 流扇出 wall-clock

若模型本身流,可于一调用参数完成即开始执,而非等全调用终。这是OpenAI文档优化但非全SDK露。本课框架做此:模拟流yield完整参数对象时,宿主启动该调用。

## 使用

`code/main.py`有两半。首用`concurrent.futures.ThreadPoolExecutor`序和并跑三模拟天气调用并打印wall-clock时间。二半重播假流响应——三并行调用`arguments`块于一流交错——并用`StreamAccumulator`按id重组。无LLM,无网络,仅重组逻辑。

看点:

- 序计时器撞1.8秒。并计时器撞0.8秒于同假延迟。
- 累加器处乱序到块通过按id缓冲并仅JSON完整时解析。
- 执行器于id参数终即刻启动,非全流终后。

## 交付成果

本课产`outputs/skill-parallel-call-safety-check.md`。给工具注册,skill审计何工具可安全并行、何有序依赖、何会压垮下游速率限——回带每工具`parallel_safe`旗修注册。

## 练习题

1. 跑`code/main.py`并变模拟延迟。确并vs序比约`max/sum`(实跑略偏理想因线程调度、序列化、框架开销)。何延迟分布并行停重要?

2. 扩累加器处"调用流中取消"案通过丢其缓冲并发`cancelled`事件。何提供者显文档此案?查Anthropic`content_block_stop`语义和OpenAI`finish_reason: "length"`行为。

3. 换线程池为`asyncio.gather`。bench两。应见async小赢因低上下文切换成本,但仅执行器做真I/O。

4. 择两工具不应并行(如`create_file`后`write_file`)。加`ordering_dependency`图至注册并于图门并行扇出。这是依赖调度最小机制,未来agent-engineering phase形式化。

5. 读OpenAI并行函数调用节和Anthropic`disable_parallel_tool_use`文档。识Anthropic荐禁并行的一实世工具类型。(提示:同资源后果突变。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 并行工具调用 | "一轮扇出" | 模型于一assistant消息发多工具调用 |
| `parallel_tool_calls` | "OpenAI旗" | 启或禁多调用发射 |
| `disable_parallel_tool_use` | "Anthropic逆" | 退出旗;默认并行启 |
| 工具调用id | "关handle" | 结果消息须echo的每调用标识符 |
| 累加器 | "流缓冲" | 部分`arguments`块每id字符串缓冲 |
| 乱序完成 | "快者先" | 并行调用不可预序完成;id是胶 |
| 依赖图 | "序约束" | 输出馈入输入的工具;不可并行 |
| Parse-early陷阱 | "JSON.parse炸" | 试解析不完整`arguments`字符串 |
| `streamFunctionCallArguments` | "Gemini 3特性" | 带每调用unique id的流参数块 |
| 完成序回复 | "勿等全" | 按到序回结果,keyed by id |

## 延伸阅读

- [OpenAI—Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling)——默认行为和退出旗
- [Anthropic—Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use)——`disable_parallel_tool_use`和结果批
- [Google—Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling)——Gemini 3起id关并行调用
- [OpenAI—Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming)——OpenAI流块参数重组
- [Anthropic—Streaming messages](https://docs.anthropic.com/en/api/messages-streaming)——带`input_json_delta`的`content_block_delta`