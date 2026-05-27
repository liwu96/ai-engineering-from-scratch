# HTN规划和进化搜索

> Symbolic规划处plan可证正确case。进化代码搜索处fitness function machine-checkable case。ChatHTN(2025)和AlphaEvolve(2025)示何解锁当配LLM。

**类型:** 构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程02(ReWOO和Plan-and-Execute)
**时间:** ~75分钟

## 学习目标

- 释Hierarchical Task Network:task、method、operator、precondition、effect。
- 描述ChatHTN hybrid循环——symbolic搜索带LLM fallback decomposition。
- 释AlphaEvolve进化循环和何仅工于programmatic evaluator。
- 实toy HTN planner加toy进化搜索于stdlib。

## 问题背景

ReWOO(课程02)、Plan-and-Execute、和ReAct覆盖多agent规划。两case它们不cover好:

1. **可证正确plan。**调度、flight pathing、compliance workflow——plan必须构造sound。流LLM plan有时hallucinate step unacceptable。
2. **机器checkable fitness function优化。**矩阵乘、调度heuristic、compiler pass——goal非"正确plan"而是"最佳plan"。

HTN规划和AlphaEvolve解两异问题。两用LLM作amplifier非replacement。

## 概念讲解

### Hierarchical Task Network

HTN是:

- **Task**——compound(被分解)和primitive(直执行)。
- **Method**——分解compound task至subtask方式、带precondition。
- **Operator**——primitive action带precondition和effect。
- **State**——fact set。

规划:给goal task和initial state、找分解入primitive operator其precondition序列满足。

HTN比LLM老仍可证正确plan reference。

### ChatHTN(Gopalakrishnan等,2025)

ChatHTN(arXiv:2505.11814)交错symbolic HTN和LLM query:

1. 试分解当前compound task用existing method。
2. 若无method应用、问LLM:"何你分解`task`于态`s`?"
3. 译LLM response入candidate subtask。
4. 验于operator schema;reject invalid decomposition。
5. Recurse。

论文中心claim:每产plan可证sound因LLM suggestion仅入作candidate decomposition、非直plan edit。Symbolic layer own correctness;LLM扩method library。

Online method learning(OpenReview`gwYEDY9j2x`,2025 follow-up)加learner generalize LLM产decomposition经regression——减LLM query频率达75%。

### AlphaEvolve(Novikov等,2025)

AlphaEvolve(arXiv:2506.13131,DeepMind,2025年6月)是异beast:Gemini 2.0 Flash/Pro ensemble orchestrate进化代码搜索。

循环:

1. seed program+programmatic evaluator(回fitness score)起。
2. LLM ensemble提mutation。
3. Mutation跑过evaluator。
4. 留最佳;再mutate。

发表win:

- 56年首个4x4复矩阵乘Strassen改进(48 scalar multiplication)。
- 经Borg调度heuristic恢复0.7% Google compute。
- Frontier workload FlashAttention 32% speedup。

硬constraint:fitness function必须machine-checkable。Prose answer进化搜索不收敛。

### 何用何

| 问题类 | 用 | 何 |
|--------|-----|-----|
| 硬constraint调度 | HTN+ChatHTN | 可证soundness |
| Compiler优化 | AlphaEvolve | Machine-checkable fitness |
| 多步task执行 | ReAct/ReWOO | LLM in loop、无formal guarantee |
| 带测试代码改进 | AlphaEvolve | 测试是evaluator |
| Policy-bound自动化 | HTN | precondition encode policy |

### 何此模式错

- **HTN无operator。**无precondition/effect schema soundness claim崩溃。ChatHTN"LLM提decomposition"需schema reject invalid move。
- **AlphaEvolve无真实evaluator。**"问LLM代码是否更好"非fitness function。Evaluator必须deterministic和快。
- **Over-engineering。**多agent任务不需任。先Reach ReAct或ReWOO。

## 构建

`code/main.py`实两toy:

- stdlib HTN planner带operator、method、precondition、effect、和`LLMFallback`踢当无method匹配compound task。"LLM"是scripted decomposer使planner可离线跑。
- stdlib算术program进化搜索:长expression其output减`|f(x)-target|`于测试set。Evaluator是deterministic。

跑:

```
python3 code/main.py
```

Trace显HTN planner分解compound task(带mid-plan LLM fallback)和进化循环收敛目标expression。

## 使用

- **HTN planner**——`pyhop`、`SHOP3`、或建己用于domain-specific policy enforcement。
- **ChatHTN**——研究代码;模式(symbolic+LLM fallback)移净于任HTN planner。
- **AlphaEvolve**——DeepMind论文;模式(ensemble+evaluator)可重现。OpenEvolve和类似开源fork涌现。
- **Agent framework**——无直发HTN或AlphaEvolve。建作subagent或background worker。

## 交付成果

`outputs/skill-hybrid-planner.md`生hybrid planner scaffold(HTN或进化)带LLM role显scoped。

## 练习题

1. 扩HTN planner带backtracking:operator postcondition runtime失败时、rollback并试下method。
2. 加LLM-method cache于ChatHTN:LLM分解task`T`于态pattern`P`时、存结果。下次call先re-check method library。
3. 换进化搜索evaluator用真实测试suite。Evolve sort function过20 test case;报告generation收敛。
4. 读AlphaEvolve evaluator设计note。Design evaluator用于你关心domain(SQL query optimization、test-suite minimization、deployment YAML)。
5. 合:用HTN分解compound task入subtask、后每subtask primitive operator用进化搜索。何shine、何over-engineer?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| HTN | "Hierarchical planner" | Task decomposition带operator、precondition、effect |
| Method | "分解rule" | Compound task入subtask分解方式 |
| Operator | "Primitive action" | Concrete step带precondition和effect |
| ChatHTN | "LLM+HTN" | Symbolic planner问LLM当无method匹配 |
| AlphaEvolve | "进化代码搜索" | Ensemble LLM mutate代码;deterministic evaluator选 |
| Fitness function | "Evaluator" | Deterministic、machine-checkable输出分 |
| Online method learning | "Cached LLM decomposition" | 存+generalize LLM plan减query cost |

## 延伸阅读

- [Gopalakrishnan等,ChatHTN(arXiv:2505.11814)](https://arxiv.org/abs/2505.11814)——symbolic+LLM hybrid planner
- [Novikov等,AlphaEvolve(arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)——带LLM mutation进化代码搜索
- [Anthropic,Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——何时reach planner vs简循环