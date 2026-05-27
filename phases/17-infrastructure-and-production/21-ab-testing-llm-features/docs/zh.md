# LLM特性A/B Testing——GrowthBook、Statsig、和Vibes问题

> 传统A/B testing非为非确定性LLM建。关键区：eval答"模型能做任务否？" A/B test答"用户关心否？" 双需；vibe check发过。2026测何：提示工程(措辞)、模型选(GPT-4 vs GPT-3.5 vs OSS;精度vs成本vs延迟)、生成参数(temperature、top-p)。真实案例：聊天机器人reward-model变种+70%对话长度和+30%留存；Nextdoor AI主题行实验reward-function精修后+1% CTR；Khan Academy Khanmigo延迟vs数学精度轴迭代。平台分：**Statsig** (OpenAI 2025年9月$1.1B收购)——序列测试、CUPED、all-in-one。**GrowthBook**——开源、warehouse原生、Bayesian + Frequentist + Sequential engine、CUPED、SRM检查、Benjamini-Hochberg + Bonferroni修正。你选基于warehouse-SQL偏好和"OpenAI收购"组织是否重要。

**类型:** 学习
**语言:** Python(stdlib、玩具序列测试模拟器)
**前置要求:** 阶段17课程13(可观测)、阶段17课程20(渐进部署)
**时间:** ~60分钟

## 学习目标

- 区eval("模型能做任务否")和A/B test("用户关心否")。
- 列举三可测轴(提示、模型、参数)并每选指标。
- 解释CUPED、序列测试、Benjamini-Hochberg多重比修正。
- 基warehouse-SQL姿态和企业收购立场选Statsig或GrowthBook。

## 问题背景

手调系统提示。觉好。发。转化噪声变。怪指标。或发新模型转化未移——模型回退或变太小检？不知、因发无A/B。

Eval答模型标记集任务否。非答用户偏好输出否。仅控线实验答、仅实验够功率、控非确定性、修多重比。

## 概念讲解

### Eval vs A/B test

**Eval**——离线、标记集、judge (rubric或LLM-as-judge或人)。答："输出正确/有用/安全否此固定分布？"

**A/B test**——线、活用户、随机。答："新变种移用户级重要指标否？"

双需。Eval发前捕回退；A/B后确认产品影响。

### 测何

1. **提示工程**——措辞、系统提示结构、例。指标：任务成功、用户留存、cost/request。
2. **模型选**——GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：精度(任务) + cost/request + 延迟P99。多目标。
3. **生成参数**——temperature、top-p、max_tokens。指标：任务特定(输出多样性vs确定性)。

### CUPED——方差减

Controlled-experiments Using Pre-Experiment Data。Regress out前周期方差后周期比前。典型方差减：30-70%。有效样本量免费升。

实现：Statsig和GrowthBook双实现。

### 序列测试

经典A/B定样本量假设。序列测试("peek-and-decide")控重复看假阳率。Always-valid序列过程(mSPRT、Howard confidence sequence)让你早停显赢家。

### 多重比修正

20 A/B test 95%信度一假阳偶然。Bonferroni修正每测试紧α；Benjamini-Hochberg控假发现率。GrowthBook双实现。

### SRM——样本比错配

Assignment hash随机用户变种。若50/50分47/53、破——SRM检查flag。双平台实现。

### Statsig vs GrowthBook

**Statsig**：
- OpenAI 2025年9月$1.1B收购。托管、SaaS。
- 序列测试、CUPED、held-out population。
- All-in-one：feature flag + experimentation + observability。
- 最佳：队要包产品、不关心OpenAI ownership。

**GrowthBook**：
- 开源(MIT)；warehouse原生(Snowflake/BigQuery/Redshift直读)。
- 多engine：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH修正。
- 自建或托管云。
- 最佳：warehouse-SQL店、数据队控指标层、要OSS。

### 非确定性复杂功率

同提示产异输出。传统功率算假设IID观察。LLM非确定性、有效样本量低于名义。安全边界乘所需样本量~1.3-1.5x。

### 真实案例结果

- 聊天机器人reward model变种：+70%对话长度、+30%留存。
- Nextdoor主题行：reward-function精修后+1% CTR。
- Khan Academy Khanmigo：迭代延迟vs数学精度权衡。

### 反模式：vibe发

每资深工程师可名"觉好"发无A/B特性。多回退产品指标队月未注意。A/B是驱动函数。

### 你应记数

- Statsig OpenAI收购：$1.1B、2025年9月。
- GrowthBook：开源MIT；Bayesian + Frequentist + Sequential。
- CUPED方差减：30-70%。
- LLM非确定性 → +30-50%样本量buffer。

## 使用

`code/main.py`模序列A/B test定和序列边界。示序列早停。

## 交付成果

本lesson产`outputs/skill-ab-plan.md`。给特性改、负载、基线、选平台、gate、样本量。

## 练习题

1. 跑`code/main.py`。期望5%升基线3%转化、何样本量80%功率？
2. 医疗监管on-prem客户选Statsig或GrowthBook。
3. 设计A/B测GPT-4 vs GPT-3.5 cost-per-resolved-ticket。何主指标、guardrail指标、次？
4. Canary过但A/B示-1.2%转化。发否？写升级准则。
5. CUPED前周期60%后周期方差。算有效样本量升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Eval | "离线测试" | 标记集模型能力评估 |
| A/B test | "实验" | 活用户随机比 |
| CUPED | "方差减" | 前周期回归减方差 |
| 序列测试 | "peek-ok测试" | Always-valid过程允早停 |
| 多重比 | "族错误" | 多测试涨假阳 |
| Bonferroni | "紧修正" | α除测试数 |
| Benjamini-Hochberg | "BH FDR" | 假发现率控、少保守 |
| SRM | "坏分" | 样本比错配；assignment bug |
| Statsig | "OpenAI owned" | 商业all-in-one、2025收购 |
| GrowthBook | "OSS那个" | MIT warehouse原生平台 |
| mSPRT | "序列概率比测试" | 经典序列过程 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)