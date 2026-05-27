# 为什么需要多智能体？

> 单个智能体遇到瓶颈。聪明的做法不是更大的智能体——而是更多的智能体。

**类型:** 学习
**语言:** TypeScript
**前置要求:** 第14阶段 (智能体工程)
**时间:** ~60分钟

## 学习目标

- 识别单智能体天花板（上下文溢出、 expertise 混合、顺序瓶颈）并解释何时拆分为多个智能体是正确的选择
- 比较编排模式（管道、并行扇出、监督者、层级）并为给定任务结构选择正确的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性的权衡

## 问题背景

你在第14阶段构建了一个单智能体。它有效。它可以读取文件、运行命令、调用API，并对结果进行推理。然后你将它指向一个真实代码库：200个文件、三种语言、依赖基础设施的测试，以及需要在编写代码前研究外部API的需求。

智能体卡住了。不是因为LLM笨，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了40次工具调用前读取的内容。它试图同时成为研究员、编码员和审查员，结果三者都做得不好。

这就是单智能体天花板。每当任务需要以下功能时，你就会遇到它：

- **比一个窗口能容纳的更多上下文** —— 读取50个文件会超过20万token
- **不同阶段的不同 expertise** —— 研究需要与代码生成不同的提示
- **可以并行进行的工作** —— 为什么要顺序读取三个文件，而不是同时读取？

## 概念讲解

### 单智能体天花板

单个智能体是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            单智能体                     │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                  │  │
│  │                                   │  │
│  │  研究笔记                         │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                       │  │
│  │  + 审查反馈                       │  │
│  │  + API文档                        │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图涵盖                   │
│  研究 + 编码 + 审查 + 测试              │
│                                         │
│  结果：每件事都做得平庸                 │
└─────────────────────────────────────────┘
```

三件事会出问题：

1. **上下文饱和** —— 工具结果堆积。到第30轮时，智能体已消耗15万token的文件内容、命令输出和先前推理。第5轮的关键细节丢失了。

2. **角色混乱** —— 一个说"你是研究员、编码员、审查员和测试员"的系统提示会产生一个半研究、半编码、从不完成审查的智能体。

3. **顺序瓶颈** —— 智能体先读取文件A，然后文件B，然后文件C。三次串行LLM调用。三次串行工具执行。没有并行性。

### 多智能体解决方案

拆分工作。给每个智能体一个工作、一个上下文窗口和一个针对该工作调整的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    编排器                                │
│                                                          │
│  "构建用户管理的REST API"                                │
│                                                          │
│         ┌──────────┬──────────┬──────────┐              │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│   │ 研究员   │ │  编码员  │ │  审查员  │ │  测试员  │   │
│   │          │ │          │ │          │ │          │   │
│   │ 读取     │ │ 编写     │ │ 检查     │ │ 运行     │   │
│   │ 文档,    │ │ 代码     │ │ 代码     │ │ 测试,    │   │
│   │ 发现     │ │ 基于     │ │ 质量,    │ │ 报告     │   │
│   │ 模式     │ │ 研究+规范│ │ 发现     │ │ 结果     │   │
│   │          │ │          │ │ 缺陷     │ │          │   │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│         │           │            │             │        │
│         └───────────┴────────────┴─────────────┘        │
│                          │                                │
│                     合并结果                              │
└──────────────────────────────────────────────────────────┘
```

每个智能体拥有：
- 一个专注的系统提示（"你是代码审查员。你唯一的工作是发现缺陷。"）
- 它自己的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际这样做的系统

**Claude Code子智能体** —— 当Claude Code用`Task`生成子智能体时，它创建一个具有限定任务的子智能体。父智能体保持其上下文干净。子智能体执行专注工作并返回摘要。

**Devin** —— 运行规划智能体、编码智能体和浏览器智能体。规划器将工作分解为步骤。编码器编写代码。浏览器研究文档。每个都有单独的上下文。

**多智能体编码团队（SWE-bench）** —— SWE-bench上表现最好的系统使用研究员读取代码库、规划器设计修复、编码器实现它。单智能体系统得分较低。

**ChatGPT深度研究** —— 并行生成多个搜索智能体，每个探索不同角度，然后综合结果。

### 频谱

多智能体不是二元的。它是一个频谱：

```
简单 ─────────────────────────────────────────── 复杂

 单智能体      子智能体       管道        团队         集群

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共享  │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ 状态 │
             └───┘          └───┘───┘  │  消息  │    └───────┘
                                       │  总线  │
 1个循环      父智能体 +      阶段      │       │    N个对等体,
 1个上下文    子任务          阶段      └───────┘    涌现
                                       显式         行为
                                       角色
```

**单智能体** —— 一个循环，一个提示。适合简单任务。

**子智能体** —— 父智能体为专注的子任务生成子智能体。父智能体维护计划。子智能体回报。这就是Claude Code所做的。

**管道** —— 智能体顺序运行。智能体A的输出成为智能体B的输入。适合分阶段工作流：研究 -> 代码 -> 审查 -> 测试。

**团队** —— 智能体在共享消息总线上并行运行。每个都有角色。一个编排器协调。当需要同时需要不同技能时很好。

**集群** —— 许多相同或几乎相同的智能体共享状态。没有固定的编排器。智能体从队列中挑选工作。适合高吞吐量并行任务。

### 四种多智能体模式

#### 模式1：管道

```
输入 ──▶ 智能体A ──▶ 智能体B ──▶ 智能体C ──▶ 输出
          (研究)      (编码)      (审查)
```

每个智能体转换数据并向前传递。易于推理。一个阶段的失败会阻塞其余阶段。

#### 模式2：扇出/扇入

```
                ┌──▶ 智能体A ──┐
                │              │
输入 ──▶ 拆分 ├──▶ 智能体B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体C ──┘
```

将工作拆分到并行智能体，然后合并结果。适合分解为独立子任务的任务。

#### 模式3：编排器-工作器

```
                    ┌──────────┐
                    │  编排器  │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作器A  │   │ 工作器B  │
           └──────────┘   └──────────┘
```

一个智能编排器决定做什么，委托给工作器，并综合结果。编排器本身是一个具有生成工作器工具的LLM智能体。

#### 模式4：对等集群

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息  │    ┌───────────┐     │ 消息
           └───▶│   共享    │◄────┘
                │   状态    │
           ┌───▶│  / 队列   │◄────┐
           │    └───────────┘     │
      消息  │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可扩展到许多智能体。

### 何时不使用多智能体

多智能体增加复杂性。智能体之间的每条消息都是潜在的故障点。调试从"阅读一个对话"变为"跟踪跨五个智能体的消息"。

**在以下情况下保持单智能体：**
- 任务适合一个上下文窗口（少于~10万token的工作数据）
- 你不需要不同阶段的不同系统提示
- 顺序执行足够快
- 任务足够简单，拆分它增加的 overhead 比价值多

**复杂性成本：**
- 每个智能体边界都是有损压缩步骤：智能体A的完整上下文被压缩成给智能体B的消息
- 协调逻辑（谁做什么、何时、以何顺序）是它自己的错误源
- 延迟增加：N个智能体意味着至少N次串行LLM调用，如果它们需要来回通信则更多
- 成本倍增：每个智能体独立消耗token

经验法则：如果任务需要少于20次工具调用且适合10万token，保持单智能体。

## 动手实践

### 步骤1：超载的单智能体

这是一个试图做所有事情的单智能体。它有一个巨大的系统提示和一个包含研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `你是一个全栈开发者。你必须：
1. 研究需求
2. 编写代码
3. 审查代码中的缺陷
4. 编写测试
在单个对话中完成所有这些。`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `研究: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `基于以下研究:\n${contextWindow.join("\n")}\n\n现在为以下任务编写代码: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `基于所有先前上下文:\n${contextWindow.join("\n")}\n\n审查代码。`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随每个阶段增长。到审查步骤时，它包含研究笔记和代码和先前推理。
- 系统提示是通用的。它不能为每个阶段调整。
- 没有并行运行。

### 步骤2：专家智能体

现在拆分它。每个智能体得到一个工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "研究员",
  "你是一个技术研究员。阅读文档，发现模式，并总结发现。只输出实现所需的事实。"
);

const coder = createSpecialist(
  "编码员",
  "你是一个高级TypeScript开发者。给定需求和研究笔记，编写干净、经过测试的代码。仅此而已。"
);

const reviewer = createSpecialist(
  "审查员",
  "你是一个代码审查员。发现缺陷、安全问题和逻辑错误。要具体。引用行号。"
);
```

每个专家都有一个专注的提示。每个都得到只包含它需要的输入的干净上下文窗口。

### 步骤3：通过消息协调

用显式消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "研究员",
    to: "编码员",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "编码员")
    .map((m) => `[来自 ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "编码员",
    to: "审查员",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "审查员")
    .map((m) => `[来自 ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "审查员",
    to: "编排器",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究员5万token的文档阅读永远不会进入审查员的上下文。

### 步骤4：比较

```typescript
async function compare() {
  const task = "为Express.js API构建速率限制中间件";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Token数: ${single.tokensUsed}`);
  console.log(`工具调用: ${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Token数: ${multi.tokensUsed}`);
  console.log(`工具调用: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总token（三个智能体，三次单独的LLM调用），但每个智能体的上下文保持干净。每个阶段的质量提高，因为系统提示是专业化的。

## 使用它

本课产生一个可重用的提示，用于决定何时采用多智能体。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习题

1. 添加第四个专家：一个"测试员"智能体，从编码员接收代码，从审查员接收审查反馈，然后编写测试
2. 修改管道，使审查员可以将反馈发送回编码员进行修订循环（最多2轮）
3. 将顺序管道转换为扇出：并行运行研究员和"需求分析器"智能体，然后在传递给编码员之前合并它们的输出

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 集群 | "AI智能体的蜂巢思维" | 具有共享状态且没有固定领导者的对等智能体集合。行为从本地交互中涌现。 |
| 编排器 | "老板智能体" | 其工具包括生成和管理其他智能体的智能体。它规划和委托，但可能不做实际工作。 |
| 协调器 | "交通警察" | 基于规则在智能体之间路由消息的非智能体组件（通常只是代码）。 |
| 共识 | "智能体达成一致" | 多个智能体必须在继续之前达成一致的协议。用于需要解决冲突输出的情况。 |
| 涌现行为 | "智能体自己搞清楚了" | 从智能体交互中产生但未明确编程的系统级模式。可能有用或有害。 |
| 扇出/扇入 | "智能体的Map-Reduce" | 将任务拆分到并行智能体（扇出），然后合并它们的结果（扇入）。 |
| 消息传递 | "智能体相互交谈" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code子智能体文档](https://docs.anthropic.com/en/docs/claude-code) - Claude Code如何用Task委托
- [CrewAI文档](https://docs.crewai.com/) - 基于角色的多智能体框架
