# 红队：PAIR和自攻

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR — Prompt Automatic Iterative Refinement — 是规范自黑盒jailbreak。攻LLM带红队系统提示迭代提出目标LLM jailbreak、积累尝试和响应于自聊历史作in-context反馈。PAIR典型20查询内成功、比GCG(Zou等人token级梯度搜)效高数阶且不需白盒访问。PAIR现是JailbreakBench (arXiv:2404.01318)和HarmBench标准基线、伴随GCG、AutoDAN、TAP、和Persuasive Adversarial Prompt。

**类型:** 构建
**语言:** Python(stdlib、玩具目标mock PAIR循环)
**前置要求:** 阶段18课程01(instruction-following)、阶段14(agent工程)
**时间:** ~75分钟

## 学习目标

- 描述PAIR算法：攻者系统提示、迭代细化、in-context反馈。
- 解释PAIR为何比GCG在目标黑盒时严格更效。
- 名四其他自攻基线(GCG、AutoDAN、TAP、PAP)和每区分特征。
- 描述JailbreakBench和HarmBench评估协议和每下"攻成功率"何意。

## 问题背景

红队曾是手动活动。少数专测试者构对抗提示并录工者。此不scale：攻成功率需统计样本、目标是每模型发布移动目标。PAIR操作化红队为优化问题、目标黑盒。

## 概念讲解

### PAIR算法

输入:
- 目标LLM T(攻模型)。
- 法官LLM J(评响应是否jailbreak)。
- 攻LLM A(红队优化器)。
- 目标字符串G: "响应[harmful instruction]。"
- 预算K(通常20查询)。

循环, k in 1..K:
1. A提示目标G和迄今(prompt, response)对历史。
2. A发新提示p_k。
3. 提p_k于T; 收响应r_k。
4. J评(p_k, r_k)于目标。
5. 若分>=阈值、停 — jailbreak找。
6. 否、附(p_k, r_k)于A历史; 续。

实证结果(NeurIPS 2023): >50%攻成功率于GPT-3.5-turbo、Llama-2-7B-chat; 平均成功查询10-20范围。

### 为何PAIR效

GCG (Zou等人 2023)搜对抗token后缀用梯度; 需白盒模型访问并产不可读后缀。PAIR黑盒产自然语言攻跨模型转。PAIR in-context反馈让攻者每拒学; GCG无等价(每新token更新需重发现前进步)。

### 相关自攻

- **GCG (Zou等人 2023, arXiv:2307.15043)。** Token级梯度搜对抗后缀。白盒、可转、产不可读串。
- **AutoDAN (Liu等人 2023)。** 提进化搜、层级目标导。
- **TAP (Mehrotra等人 2024)。** Tree-of-attacks带剪枝 — 分支多PAIR风格rollout。
- **PAP (Zeng等人 2024)。** Persuasive Adversarial Prompts — 编人说服技为提示模板。

### JailbreakBench和HarmBench

双(2024)标准化评估:

- JailbreakBench (arXiv:2404.01318)。100有害行为跨10 OpenAI-policy类别。攻成功率(ASR)主度量。需法官(GPT-4-turbo、Llama Guard、或StrongREJECT)。
- HarmBench (Mazeika等人 2024)。510行为跨7类别、语义和功能害测。比18攻于33模型。

ASR通常报于固定查询预算。比攻需配预算; 200查询90% ASR不可比20查询85% ASR。

### 何重于2026部署

每前沿实验室现发布前跑PAIR和TAP于产模型。ASR轨迹现于模型卡(课程26)和安全案例附录(课程18)。攻非异 — 是标准基础设施。

### Phase 18何处

课程12是自攻基础。课程13(Many-Shot Jailbreaking)是补长exploit。课程14(ASCII Art / Visual)是编码攻。课程15(Indirect Prompt Injection)是2026产攻面。课程16覆防御工具对应(Llama Guard、Garak、PyRIT)。

## 使用

`code/main.py`玩具PAIR循环。目标是mock分类器拒"显"有害提示(keyword-filter)。攻者是规则细化器试改述、角色框架、和编码。法官评响应。观攻者~5-15迭代成功于keyword filter并败于语义filter。

## 交付成果

本lesson产`outputs/skill-attack-audit.md`。给红队评估报告、审计：何攻跑(PAIR、GCG、TAP、AutoDAN、PAP)、每何预算、何法官、何有害行为集(JailbreakBench、HarmBench、内)。

## 练习题

1. 跑`code/main.py`。测三内攻策略平成功查询。解释每exploit何目标防御假设。
2. 实第四攻策略(如译另一语言、base64编码)。报keyword-filter目标和semantic-filter目标新平成功查询。
3. 读Chao等人2023图5(PAIR vs GCG比)。述两GCG尽管PAIR效优偏场景。
4. JailbreakBench报固定目标集ASR。设计补度攻多样性(成功提示方差)。解释何多样性重于防御评估。
5. TAP (Mehrotra 2024)延PAIR带分支+剪枝。草TAP风格延`code/main.py`并述算成本vs成功率trade-off。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| PAIR | "自jailbreak" | Prompt Automatic Iterative Refinement; 攻LLM + 法官LLM循环 |
| GCG | "梯度jailbreak" | 白盒token级梯度搜对抗后缀 |
| 攻成功率(ASR) | "k查询% jailbreak" | 主度量; 需报查询预算和法官身份 |
| 法官LLM | "评分者" | LLM评响应是否满有害目标 |
| JailbreakBench | "评估" | 标准化有害行为集带标签类别 |
| HarmBench | "更bench" | 510行为、功能+语义害测 |
| TAP | "攻树" | PAIR带分支+剪枝; 高算高ASR |

## 延伸阅读

- [Chao等人 — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR论文、NeurIPS 2023
- [Zou等人 — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG论文
- [Chao等人 — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika等人 — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更广评估