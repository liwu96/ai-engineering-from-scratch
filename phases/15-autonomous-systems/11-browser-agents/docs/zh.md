# 浏览器智能体和长时程网络任务

> ChatGPT智能体（2025年7月）将Operator和深度研究合并为一个浏览器/终端智能体，在BrowseComp上创下68.9%的SOTA。OpenAI于2025年8月31日关闭Operator——产品层整合。Anthropic的Vercept收购将Claude Sonnet在OSWorld上从<15%提升到72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始WebArena中11.3个百分点的假阴性率，发布了258个任务的Hard子集。数字是真实的。攻击面也是：OpenAI准备负责人表示浏览器智能体中的间接提示注入"不是可以完全修补的bug"。记录的2025-2026攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）、Perplexity Comet中的一键劫持。

**类型:** 学习
**语言:** Python (stdlib, 间接提示注入攻击面模型)
**前置要求:** 第15阶段 · 10 (权限模式), 第15阶段 · 01 (长时程智能体)
**时间:** ~45分钟

## 问题背景

浏览器智能体是读取不可信内容并采取有实际影响的动作的长时程智能体。智能体访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是潜在命令通道。2025-2026攻击语料库显示这不是假设：Tainted Memories让攻击者通过精心制作的页面将恶意指令绑定到智能体的内存；HashJack在智能体访问的URL片段中隐藏命令；Perplexity Comet劫持一击命中。

防御图景不舒服。OpenAI准备负责人说出了安静的部分：间接提示注入"不是可以完全修补的bug"。这是因为攻击存在于智能体的阅读与行动边界，这在架构上是模糊的——原则上模型读取的每个token都可以被读取为指令。

本课命名攻击面，命名基准全景（BrowseComp、OSWorld、WebArena-Verified），并建模最小间接提示注入场景，以便你可以在第14和18课中推理真实防御。

## 概念讲解

### 2026全景，每个系统一段话

**ChatGPT智能体 (OpenAI)。** 2025年7月启动。统一Operator（浏览）和Deep Research（多小时研究）。2025年8月31日关闭独立Operator。BrowseComp上SOTA 68.9%；OSWorld和WebArena-Verified上强劲数字。

**Claude Sonnet + Vercept (Anthropic)。** Anthropic的Vercept收购专注于计算机使用能力。将OSWorld上的Claude Sonnet从<15%提升到72.5%。Claude Computer Use作为工具API发布。

**带Browser Use的Gemini 3 Pro (DeepMind)。** Browser Use集成发布计算机使用控制；FSF v3（2026年4月，第20课）专门跟踪ML R&D领域的自主性。

**WebArena-Verified (ServiceNow, ICLR 2026)。** 修复文档良好的问题：原始WebArena有约11.3%假阴性率（标记失败但实际解决的任务）。Verified发布用人工策划的成功标准重新评分，并添加258个任务的Hard子集（ICLR 2026论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp与OSWorld与WebArena

| 基准 | 测量什么 | 时间范围 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务性网络任务 | 分钟 |
| Hard子集 | 多页面状态转换的WebArena-Verified任务 | 数十分钟 |

不同轴。高BrowseComp分数说明智能体查找事实；不说明智能体能预订航班。OSWorld分数更接近"它在我的桌面上能用吗"。WebArena-Verified更接近"它能完成流程吗"。任何生产决策需要匹配任务分布的基准。

### 攻击面，命名

1. **间接提示注入。** 不可信页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024 Kai Greshake等人，2025 Tainted Memories论文，2026 HashJack（Cato Networks）。
2. **URL片段/查询注入。** 爬取的URL的 `#fragment` 或查询字符串包含命令。从不可见渲染；仍在智能体上下文中。
3. **内存绑定攻击。** 页面指示智能体写入持久内存（第12课涵盖持久状态）。下一会话，内存触发payload而无可见触发。
4. **认证会话上的CSRF形状攻击。** Tainted Memories类别：智能体在某处登录；攻击者页面发出智能体用用户cookie执行的状态改变请求。
5. **一键劫持。** 视觉上无害的按钮搭载智能体遵循的payload。Comet类别。
6. **智能体主机表面中的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器智能体堆栈很宽。

### 为什么"不能完全修补"

攻击与智能体的能力同构。智能体必须读取不可信内容才能完成工作。智能体读取的任何内容都可以包含指令。智能体遵循的任何指令可能与用户的实际请求未对齐。防御（信任边界、分类器、工具允许列表、有实际影响的动作上的HITL）提高攻击成本并减少爆炸半径。它们不关闭类别。

这与Lob定理（第8课）的推理模式相同：智能体不能证明下一个token是安全的；它只能设置一个不安全token更可检测的系统。

### 实际交付的防御姿态

- **读/写边界。** 阅读本身从不产生实际影响。写入（提交表单、发布内容、调用带副作用的工具）如果发起内容来自信任边界之外需要新的人工批准。
- **每任务工具允许列表。** 智能体可以浏览；除非该工具明确为任务启用，否则智能体不能发起电汇。第13课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅以限定凭证运行。无生产认证、无个人电子邮件。保留每个HTTP请求的日志用于审计。
- **内容清理器。** 获取的HTML在连接到模型上下文前剥离已知不良模式。（减少容易攻击；不阻止复杂payload。）
- **有实际影响的动作上的HITL。** 提议-然后-提交模式（第15课）。
- **内存上的金丝雀token。** 如果内存条目触发，用户看到它（第14课）。

## 动手实践

`code/main.py` 针对三个合成页面建模微小的浏览器智能体运行。一个页面是良性的，一个在可见文本中有直接提示注入blob，一个有URL片段注入（不可见但在智能体上下文中）。脚本显示(a)天真智能体会做什么，(b)读/写边界捕获什么，(c)清理器捕获什么，(d)两者都未捕获什么。

## 产出成果

`outputs/skill-browser-agent-trust-boundary.md` 划定提议的浏览器智能体部署的范围：它触及哪些信任区域、授权写入什么、首次运行前必须存在哪些防御。

## 练习题

1. 运行 `code/main.py`。识别清理器捕获但读/写边界未捕获的攻击，以及仅读/写边界捕获的攻击。

2. 扩展清理器以检测一类HashJack风格URL片段注入。测量合法片段良性URL上的假阳性率。

3. 选择一个你熟悉的真实浏览器智能体工作流（例如"预订航班"）。列出每个读取和每个写入。标记哪些写入需要HITL及原因。

4. 阅读WebArena-Verified ICLR 2026论文。识别原始WebArena评分不可靠的一类任务，并解释Verified子集如何解决它。

5. 为浏览器智能体设置设计内存金丝雀。你会存储什么、在哪里、什么触发警报？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 间接提示注入 | "不良页面文本" | 智能体读取的页面中不可信内容包含智能体执行的指令 |
| Tainted Memories | "内存攻击" | 智能体将攻击者提供的指令写入持久内存；下一会话触发 |
| HashJack | "URL片段攻击" | URL片段/查询字符串中隐藏的payload在智能体上下文中但不可见渲染 |
| 一键劫持 | "不良按钮" | 可见功能搭载智能体执行的后续payload |
| BrowseComp | "网络搜索基准" | 在开放网络上查找特定事实；分钟级时间范围 |
| OSWorld | "桌面基准" | 完整OS控制；多步GUI任务 |
| WebArena-Verified | "固定网络任务基准" | ServiceNow重新评分的WebArena，带Hard子集 |
| 读/写边界 | "副作用门" | 阅读本身从不产生实际影响；如果内容超出信任则写入需要新批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator和深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator血统和成为ChatGPT智能体的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026固定子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包括计算机使用智能体的攻击面讨论。
