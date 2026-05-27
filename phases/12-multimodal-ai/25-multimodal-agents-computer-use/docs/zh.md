# 多模态Agent和计算机使用(顶点)

> 2026前沿产品是多模态agent读截屏、点按钮、导航网UI、填表、端到端完工作流。SeeClick和CogAgent (2024)证GUI grounding原语。Ferret-UI加移动。ChartAgent引图表视觉工具用。VisualWebArena和AgentVista (2026)是前沿追基准——甚至Gemini 3 Pro和Claude Opus 4.7 AgentVista难任务评分~30%。此顶点拉全第12阶段线:感知(高分辨率VLM)、推理(LLM带工具用)、grounding(坐标输出)、长视记忆、评估。

**类型:** 顶点
**语言:** Python (stdlib，行动schema + agent环骨架)
**前置要求:** 第12阶段·05(LLaVA)，第12阶段·09(Qwen-VL JSON)，第14阶段(Agent工程)
**时间:** ~240分钟

## 学习目标

- 设计多模态agent环:感知→推理→行动→观察→重复。
- 建GUI grounding输出schema(点坐标、打文本、滚、拖)VLM可发JSON。
- 比仅截屏agent vs accessibility-tree agent vs混agent。
- 设多模态agent基准评估小VisualWebArena切片。

## 问题背景

预订站工作流:"找我4月15日东京航班，走廊座低于$800，订。"

多模态agent需:

1. 取浏览器截屏。
2. 解截屏+URL+目标为计划。
3. 发结构行动:点(x,y)、打"东京"(元素E)、下滚、选(单选按钮)。
4. 施行动浏览器。
5. 观察新态(下截屏)。
6. 重复至任务完。

每步是多模态VLM调用。VLM输出须可解析JSON。错跨步积，恢复重要。

## 概念讲解

### GUI grounding——原语

GUI grounding是:给截屏和自然语言指令，输出(x, y)坐标点击(或其他行动)。

SeeClick (arXiv:2401.10935)是首开源规模结果:微调VLM合成+真GUI数据，输出坐标作纯文token。工。

CogAgent (arXiv:2312.08914)加1120x1120高分辨率编码密UI。分:~84%网导航。

Ferret-UI (arXiv:2404.05719)聚焦移动UI，集iOS accessibility数据。

输出格式通常JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "搜索按钮"}
```

`element_desc`帮恢复:若坐标截屏漂移，语义提示系统重ground。

### 行动schema

典型行动schema有6-10行动类型:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent每步发一行动。浏览器包装执行返新态。

### 仅截屏vs accessibility-tree

两输入模式:

- 仅截屏:全图像，无结构信息。最通用；任何app工。
- Accessibility tree:结构DOM / iOS accessibility信息。Grounding更可靠;树可用处工。
- 混:两者，树原子行动可靠定位器截屏语义上下文。

生产Agent混可用时用。浏览器自动化(Selenium + accessibility)总有树；桌面应用有时有。

### 视记忆

20步工作流生成20截屏。VLM上下文速填。三压缩策略:

- 总结链:每5步后，总结何发生，弃旧截屏。
- 跳帧:保首、末、每3截屏。
- 工具记录日志:执行行动，保何做文日志；不看旧截屏。

Claude计算机用API用日志模式。简，更可靠。

### 视觉工具用

ChartAgent (arXiv:2510.04514)引图表理解视觉工具用:裁、缩、OCR、调外部检测。Agent可输出"裁区(100, 200, 300, 400)后调OCR"作工具调。工具返文；VLM继续推理。

此模式泛:set-of-mark提示、域注、外部检测工具全适同"输出工具调，收结构响应"schema。

### 2026基准

- ScreenSpot-Pro.GUI grounding ~1k网截屏。开源SOTA Qwen2.5-VL-72B ~85%。前沿~90%。
- VisualWebArena.端到端网任务(店、论坛、分类)。开源SOTA ~20%。Gemini 3 Pro ~27%。
- AgentVista (arXiv:2602.23166).2026最难基准。12域真实工作流。前沿模评分27-40%；开源10-20%。
- WebArena / WebShop.旧基准；前沿饱和。

### 何仍难

Agent性能瓶颈:

1. 细尺度视觉grounding。"点小X"移动分辨率常失败。
2. 视规划。10行动后，agent偏目标。
3. 错恢复。点击失败(错按钮)，检测+恢复罕见训练数据。
4. 跨页上下文。跳标签或长表失态。

研方向:记忆架构、显重规划、多模态验证(截屏配行动成功)。

### 顶点构建

顶点任务:建计算机用agent:

1. 读预订站mock页HTML+截屏。
2. 计多步序:搜索→选择→填表→提交。
3. 发JSON行动配行动schema。
4. 固10任务切片评估。

课提供易延真实浏览器骨架码。

## 使用它

`code/main.py`是顶点骨架:

- 行动schema JSON定义(10行动)。
- Mock浏览器态作dict。
- Agent环骨架:收态、发行动、施、环。
- 10任务mini基准(合成页)测端到端成功率。
- 行动失败错恢复hook。

## 发货它

这课产`outputs/skill-multimodal-agent-designer.md`。给计算机用产品(域、行动集、评估目标)，设计全agent环、记忆策略、grounding模式、预期基准分。

## 练习题

1. 延行动schema加`screenshot_region`工具(裁+缩)。何任务受益？

2. 读AgentVista (arXiv:2602.23166)。描述最难任务类和何前沿模型仍失败。

3. 视记忆压缩:设计总结链保≤4截屏活，任数日志。

4. 建错恢复hook:行动失败(按钮未找)，agent何做下？

5. 比仅截屏Claude 4.7和混截屏+accessibility-tree Qwen2.5-VL于10网任务。何赢何任务？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| GUI grounding | "点坐标" | 模型输出指令目标截屏(x,y) |
| 行动schema | "工具定义" | 有效行动JSON描述(点、打、滚、拖) |
| Accessibility tree | "结构DOM" | 浏览器/iOS API机读UI层次 |
| 混agent | "截屏+树" | 用图像和结构信息；比任更可靠 |
| 视觉工具用 | "缩/裁/检测" | Agent中计划调外部视觉工具(OCR、检测) |
| 总结链 | "记忆压缩" | 周期文总结代长截屏历史 |
| VisualWebArena | "E2E网bench" | 2024端到端网任务基准 |
| AgentVista | "2026难bench" | 12域真实工作流；甚至Gemini 3 Pro评分~30% |

## 延伸阅读

- [Cheng等—SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong等—CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You等—Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh等—VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)