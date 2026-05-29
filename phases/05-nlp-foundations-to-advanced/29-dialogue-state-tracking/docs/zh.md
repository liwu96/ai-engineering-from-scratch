# 对话状态追踪

> "我想要北区便宜餐厅...实际改成中等...加意大利菜。"三轮，三次状态更新。DST保持槽值字典同步，以使预订正常工作。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段5课程17(聊天机器人)、阶段5课程20(结构化输出)
**时间:** ~75分钟

## 问题背景

任务导向对话系统中,用户目标编码为槽值对集:`{cuisine: italian, area: north, price: moderate}`。每用户轮可增、改或删槽。系统须读全对话并正确输出当前状态。

错单槽则系统订错餐厅、排错航班或收错卡。DST是用户所说和后端执行之间枢纽。

2026为何仍重要尽管有大语言模型:

- 合规敏感领域(银行、医疗、航空预订)需确定性槽值,非自由生成。
- 工具用智能体调API前仍需槽解析。
- 多轮修正比看起来难:"实际不,改成周四。"

现代管道:经典DST概念+大语言模型抽取器+结构化输出护栏。

## 概念讲解

![DST:对话历史→槽值状态](../assets/dst.svg)

**任务结构。** 模式定义领域(餐厅、酒店、出租车)及其槽(菜系、区域、价格、人数)。每槽可空、填闭集值(price: {cheap, moderate, expensive})或自由值(name: "The Copper Kettle")。

**两种DST形式。**

- **分类。** 每(槽,候选值)对预测yes/no。闭词汇槽工作。2020前标准。
- **生成。** 给对话,生成槽值作自由文本。开词汇槽工作。现代默认。

**指标。** 联合目标准确率(JGA)——*每*槽正确轮分数。全或无。MultiWOZ 2.4排行榜2026顶约83%。

**架构。**

1. **规则基(槽正则+关键词)。** 窄领域强基线。可调试。
2. **TripPy/BERT-DST。** BERT编码复制基生成。大语言模型前标准。
3. **LDST(LLaMA+LoRA)。** 领域槽提示指令微调大语言模型。MultiWOZ 2.4达ChatGPT级质量。
4. **无本体(2024–26)。** 跳模式;直接生成槽名和值。处理开领域。
5. **提示+结构化输出(2024–26)。** 大语言模型配Pydantic模式+约束解码。5行代码,生产就绪。

### 经典失败模式

- **跨轮共指。**"用第一个选项。"需解析哪个选项。
- **覆写vs追加。**用户说"加意大利。"替换菜系还是追加?
- **隐确认。**"好酷"——接受预订了吗?
- **修正。**"实际改成7点。"须更新时间不清其他槽。
- **指前系统话语。**"是的,那个。"哪个"那个"?

## 动手实践

### Step 1:规则基槽抽取器

见`code/main.py`。正则+同义词字典覆窄领域70%规范话语:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

规范词汇外脆弱。确定性槽确认工作。

### Step 2:状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三不变:

- 永不重置用户未触槽。
- 显否定("不管菜系")须清。
- 用户修正("实际...")须覆写,非追加。

### Step 3:大语言模型驱DST配结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor+Pydantic保证有效状态对象。无正则,无模式不匹配,无幻觉槽。

### Step 4:JGA评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准:系统多少轮全槽对?MultiWOZ 2.4,2026顶系统:80-83%。你领域内系统应超你窄词汇否则大语言模型基线赢你。

### Step 5:处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测修正时,覆写最后更新槽而非追加。无大语言模型帮助难做好。现代模式:总让大语言模型从历史重生成全状态而非增量更新——自然处理修正。

## 陷阱

- **全历史重生成成本。** 每轮让大语言模型重生成状态花O(n²)总词元。帽历史或摘要旧轮。
- **模式漂移。** 后加新槽破旧训练数据。版本你模式。
- **大小写敏感。** "Italian"vs"italian"vs"ITALIAN"——到处规范化。
- **隐继承。** 用户之前指定"4人",新请求不同时间不应清人数。总传全历史。
- **自由vs闭集。** 名、时间、地址需自由槽;菜系和区域闭。模式混两者。

## 实际应用

2026栈:

| 情况 | 方法 |
|------|------|
| 窄领域(一或二意图) | 规则基+正则 |
| 宽领域,有标数据 | LDST(LLaMA+LoRA MultiWOZ风格数据) |
| 宽领域,无标签,生产就绪 | 大语言模型+Instructor+Pydantic模式 |
| 语音/语音 | ASR+规范化器+大语言模型-DST |
| 多领域预订流 | 模式引导大语言模型配每领域Pydantic模型 |
| 合规敏感 | 规则基主,大语言模型后备配确认流 |

## 产出成果

存`outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: 设计对话状态追踪器——模式、抽取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定用例(领域、语言、词汇开放、合规需求),输出:

1. 模式。领域列表,每领域槽,每槽开vs闭词汇。
2. 抽取器。规则基/seq2seq/大语言模型配Pydantic。理由。
3. 更新策略。重生成全状态/增量;修正处理;否定处理。
4. 评估。保留对话集联合目标准确率,槽级精确召回,最难槽混淆。
5. 确认流。何时显问用户确认(破坏性动作、低置信抽取)。

拒绝合规敏感槽无规则基二次检查仅大语言模型DST。拒绝不能用户修正时回滚槽DST。标记无版本标签模式。
```

## 练习题

1. **简单。** 构`code/main.py`规则基状态追踪器3槽(菜系、区域、价格)。测10手编对话。测JGA。
2. **中等。** 同数据集配Instructor+Pydantic+小大语言模型。比JGA。查最难轮。
3. **困难。** 实现两者路由:规则基主,规则基发<2槽带置信时大语言模型后备。测联合JGA和每轮推理成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| DST | 对话状态追踪 | 跨对话轮维持槽值字典。 |
| 槽 | 用户意图单位 | 后端需命名参数(菜系、日期)。 |
| 领域 | 任务区 | 餐厅、酒店、出租车——槽集。 |
| JGA | 联合目标准确率 | 每槽正确轮分数。全或无。 |
| MultiWOZ | 基准 | 多领域WOZ数据集;标准DST评估。 |
| 无本体DST | 无模式 | 直接生成槽名和值,无固定列表。 |
| 修正 | "实际..." | 覆写前填槽的轮。 |

## 延伸阅读

- [Budzianowski等(2018). MultiWOZ—A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278)——规范基准。
- [Feng等(2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970)——LLaMA+LoRA指令微调DST。
- [Heck等(2020). TripPy—A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877)——复制基DST主力。
- [King, Flanigan(2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753)——EM基无监督任务导向对话。
- [MultiWOZ排行榜](https://github.com/budzianowski/multiwoz)——规范DST结果。