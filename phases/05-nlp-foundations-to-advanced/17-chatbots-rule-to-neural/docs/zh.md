# 聊天机器人——从规则到神经到大语言模型智能体

> ELIZA模式匹配回复。DialogFlow映射意图。GPT权重回答。Claude运行工具验证。每时代解决前一最糟失败。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段5课程13(问答)、阶段5课程14(信息检索)
**时间:** ~75分钟

## 问题背景

用户说"我想改航班。"系统得弄清他们要什么、缺什么信息、如何获取、如何完成动作。然后用户说"等等,如果我取消呢?"系统得记上下文、换任务、保状态。

对话对ML系统难。输入开放。输出多轮连贯。系统可能作用于世界(改航班、刷卡)。每错步用户可见。

聊天机器人架构经四范式循环,每因前一失败太显引入。本课按序走。2026生产景观是最后两混合。

## 概念讲解

![聊天机器人演进:规则→检索→神经→智能体](../assets/chatbot.svg)

**规则基(ELIZA、AIML、DialogFlow)。** 手写模式匹配用户输入产响应。意图分类器路由预定义流。槽填充状态机收需信息。设计窄范围内工作优秀。范围外立即失败。仍发货于安全关键域(银行认证、航空预订)幻觉不许。

**检索基。** FAQ风格系统。编码每对(话语、响应)。运行时,编码用户消息检索最近存储响应。想Zendesk经典"相似文章"功能。比规则处理转述更好。无生成,无幻觉。

**神经(seq2seq)。** 对话日志训编码器-解码器。从零生成响应。流畅但倾向泛输出("我不知道")和事实漂移。永不可靠在主题。Google、Facebook和Microsoft2016-2019失望聊天机器人原因。

**大语言模型智能体。** 语言模型包在计划、调工具、验证结果循环中。非长提示聊天机器人。智能体循环:计划→调工具→观测结果→决定下步。检索优先锚定(RAG)防幻觉。工具调用让它实际做事。这是2026架构。

四范式非顺序替换。2026生产聊天机器人路由四者:认证和破坏动作规则基、FAQ检索、自然措辞神经生成、歧义开放查询大语言模型智能体。

## 动手实践

### Step 1:规则基模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA20行。反射技巧("I feel sad"→"Why do you feel sad")是Weizenbaum 1966标准心理治疗演示。仍启发。

### Step 2:检索基(FAQ)

此示例片段需`pip install sentence-transformers`(拉torch)。课程可跑`code/main.py`用stdlib Jaccard相似度,课程无外依赖跑。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

阈值基拒绝是关键设计选择。如最佳匹配不够近,返`None`让系统升级。

### Step 3:神经生成(基线)

用小指令调编码器-解码器(FLAN-T5)或微调对话模型。2026自身生产不可用(矛盾、离题漂移、事实胡说),但发货于混合系统自然措辞。DialoGPT风格解码器仅模型需显式轮分隔器和EOS处理产连贯回复;FLAN-T5 text2text管道教学示例开箱工作。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Step 4:大语言模型智能体循环

2026生产形态:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

命名三物。工具是大语言模型可调可调用函数。循环终止当大语言模型返最终答案而非工具调用。步预算防歧义任务无限循环。

真实生产加:检索优先锚定(每大语言模型调用前注入相关文档)、护栏(无确认拒破坏动作)、可观测性(记每步)、评估(智能体行为保规范自动检)。

### Step 5:混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式:破坏动作确定性规则、罐装FAQ检索、其余大语言模型智能体。这是2026客服系统发货。

## 实际应用

2026栈:

| 用例 | 架构 |
|------|------|
| 预订、支付、认证 | 规则基状态机+槽填充 |
| 客服FAQ | curated答案检索 |
| 开放帮助聊天 | 大语言模型智能体配RAG+工具调用 |
| 内部工具/IDE助手 | 大语言模型智能体配工具调用(搜索、读、写) |
| 伴侣/角色聊天机器人 | 调大语言模型配人设系统提示词,知识检索 |

生产永用混合路由。无单架构好处理每请求。路由层本身典型小意图分类器。

## 仍发货失败模式

- **自信捏造。** 大语言模型智能体声称完成未完成动作。缓解:验证结果、记工具调用、永不让大语言模型无成功工具返声称做了什么。
- **提示词注入。** 用户插入文本覆盖系统提示词。排名LLM01于OWASP大语言模型应用2025 Top 10。两风味:直接注入(贴进聊天)和间接注入(藏于智能体读文档、邮件或工具输出)。

  攻击率因场景变。前沿模型通用工具和编码基准测量成功率~0.5-8.5%。特定高风险设置(AI编码智能体自适应攻击、脆弱编排)达~84%。生产CVE包括EchoLeak(CVE-2025-32711, CVSS 9.3)——Microsoft 365 Copilot零点击数据外泄漏洞由攻击者控邮件触发。

  缓解:全程视用户输入为不可信;工具调用前清理;隔离工具输出与主提示词;用Plan-Verify-Execute(PVE)模式智能体先计划,再对计划验证每动作执行(止工具结果注入新非计划动作);破坏动作需用户确认;工具范围用最小权限。

  无量提示词工程全消此风险。需外部运行时防御层(大语言模型Guard、白名单验证、语义异常检测)。
- **范围蔓延。** 智能体离题因工具调用返切相关信息。缓解:窄工具契约;保持系统提示词聚焦;加离题率评估。
- **无限循环。** 智能体持续调同工具。缓解:步预算、工具调用去重、大语言模型评判"我们在进展吗"。
- **上下文窗口耗尽。** 长对话推最早轮出上下文。缓解:总结旧轮、按相似度检索相关过去轮或用长上下文模型。

## 产出成果

存`outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: 为给定用例设计聊天机器人栈。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定产品上下文(用户需求、合规约束、可用工具、数据量),输出:

1. 架构。规则基、检索、神经、大语言模型智能体或混合(指定哪路径去哪)。
2. 大语言模型选择如适用。命名模型族(Claude、GPT-4、Llama-3.1、Mixtral)。匹配工具用质量和成本。
3. 锚定策略。RAG源、检索方法(见课程14)、工具契约。
4. 评估计划。任务成功率、工具调用正确率、离题率、保留对话幻觉率。

拒绝为任何破坏动作(支付、账户删除、数据修改)推荐纯大语言模型智能体无结构确认流。拒绝跳提示词注入审计如智能体有写访问任何。
```

## 练习题

1. **简单。** 实现上述规则基响应10模式咖啡店订餐机器人。测边缘情况:双订单、修改、取消、不清意图。
2. **中等。** 构混合FAQ+大语言模型回退。SaaS产品50罐装FAQ条、文档站检索大语言模型回退。测100真支持问题拒绝率和准确率。
3. **困难。** 实现上述智能体循环三工具(搜索、读用户数据、发邮件)。跑50测试场景评估含提示词注入尝试。报离题率、失败任务率和任何注入成功。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 意图 | 用户要什么 | 类别标签(book_flight, reset_password)。路由到处理器。 |
| 槽 | 信息片段 | 机器人需参数(日期、目的地)。槽填充是问序列。 |
| RAG | 检索加生成 | 检索相关文档,再锚定大语言模型响应。 |
| 工具调用 | 函数调用 | 大语言模型发结构调用配名+参数。运行时执行返结果。 |
| 智能体循环 | 计划、行动、验证 | 控制器跑大语言模型调用交织工具调用直到任务完成。 |
| 提示词注入 | 用户攻击提示词 | 恶意输入试图覆盖系统提示词。 |

## 延伸阅读

- [Weizenbaum(1966). ELIZA—A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf)——原始规则基聊天机器人论文。
- [Thoppilan等(2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239)——Google晚期神经聊天机器人论文,大语言模型智能体接管前。
- [Yao等(2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)——命名智能体循环模式论文。
- [Anthropic构建有效智能体指南](https://www.anthropic.com/research/building-effective-agents)——2024生产指导2026仍持。
- [Greshake等(2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)——提示词注入论文。
- [OWASP大语言模型应用2025 Top 10—LLM01提示词注入](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)——排名使提示词注入成最安关切。
- [AWS—Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/)——实践编排层防御含Plan-Verify-Execute和用户确认流。
- [EchoLeak(CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection)——间接提示词注入典型零点击数据外泄CVE。写访问智能体需运行时防御参考案例。