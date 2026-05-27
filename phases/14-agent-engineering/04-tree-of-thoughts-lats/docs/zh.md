# Tree of Thoughts 和 LATS:深思搜索

> 单chain-of-thought轨迹无回退空间。ToT(Yao等,2023)将推理转为带每节点自评估的树。LATS(Zhou等,2024)在MCTS下统一ToT与ReAct和Reflexion。Game of 24从4%(CoT)至74%(ToT);LATS HumanEval 92.7% pass@1。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 阶段14课程01(Agent Loop)、阶段14课程03(Reflexion)
**时间:** ~75分钟

## 学习目标

- 将推理框架为搜索:节点是"thought"、边是"扩展"、值是"何有希望"。
- 实现stdlib ToT式BFS树搜索配自评估评分。
- 扩展到玩具LATS MCTS循环配select/expand/simulate/backpropagate。
- 决定何时搜索值得token倍增(Game of 24、代码生成)和何时单轨迹足够(简单问答)。

## 问题背景

Chain-of-thought是线性walk。如果第一步错,每后续步在坏前提上工作。Game of 24(用四个数字配+−×÷造24)上,GPT-4 CoT达4%准确率。模型早选错子表达式且不可恢复。

推理需要的是能够提多候选、评估它们、选有希望、死端出现时回退。那是搜索。Tree of Thoughts和LATS是两种canonical公式。

## 概念讲解

### Tree of Thoughts (Yao等,NeurIPS 2023)

每个节点是coherent中间步骤("一个thought")。每个节点可扩展为K个子thought。LLM配评分提示自评估每个节点。搜索探索树——BFS、DFS或beam。

```
                     (root: "从4 6 4 1找24")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自评估是承重piece。论文展示三种变体:`sure/likely/impossible`分类、`1..10`数值分、和候选间投票。全三种Game of 24大幅胜CoT(配GPT-4 4% -> 74%)。

### LATS (Zhou等,ICML 2024)

LATS在MCTS下统一ToT、ReAct和Reflexion。LLM玩三角色:

- **Policy**:提候选下动作(ReAct式)。
- **Value function**:评分部分轨迹(ToT式自eval)。
- **Self-reflector**:失败时,写自然语言反思(Reflexion式)并用它重新播种未来rollout。

环境反馈(observation)混入value function使搜索被真实工具结果inform,而非仅模型意见。论文时结果:HumanEval pass@1 92.7%配GPT-4(SOTA)、WebShop平均75.9配GPT-3.5(接近基于梯度fine-tuning)。

### MCTS,最小

每迭代四phase:

1. **Select**——从root用UCT(树upper confidence bound)walk至leaf。
2. **Expand**——经policy生成K个child。
3. **Simulate**——从child用policy rollout、用value function(或环境reward)评分leaf。
4. **Backpropagate**——沿path向上更新visit count和value estimate。

UCT公式:`Q(s,a)+c*sqrt(ln N(s)/N(s,a))`。首项exploitation;次项exploration。按任务调`c`。

### 成本现实

搜索爆炸token。ToT Game of 24用CoT 100–1000x token。LATS类似。这不免费;保留搜索用于:

- 单轨迹显不足任务(Game of 24、复杂代码)。
- Wall-clock不如correctness重要任务。
- 有便宜可靠value function任务(代码unit test、数学显式target)。

如果你的任务有单正确答案和噪evaluator,搜索常使事更糟——它找"好评分"错答案。

### 2026定位

大多数生产智能体不运行LATS。它们运行ReAct配工具支撑验证(CRITIC,课程05)。搜索出现在专用niche:

- 运行测试作value function编码智能体(HumanEval式)。
- 探多query路径深研究智能体。
- LangGraph subgraph内规划重workflow。

AlphaEvolve(课程11)是2025极端:代码上进化搜索、机器可checkable fitness、前沿突破(56年首个4x4 matmul改进)。

## 动手实践

`code/main.py`实现:

- stylized"选算术op"任务上小ToT BFS。
- 同任务上玩具LATS MCTS循环(Select/Expand/Simulate/Backpropagate)配UCT selection。
- 组合符号分加自eval分的value function。

运行:

```
python3 code/main.py
```

Trace显示ToT每节点BFS扩三候选,比较LATS经MCTS收敛最佳rollout。两者打印token数。

## 实际应用

LangGraph发布ToT式探索作subgraph模式;LangChain团队LATS blog(2024年5月)是参考tutorial。LlamaIndex发布`TreeOfThoughts`智能体。大多数2026生产智能体此模式活在`if task_complexity>threshold: use_search()` gate后——见课程05 evaluator-optimizer模式。

## 产出成果

`outputs/skill-search-policy.md`按任务形、预算、和evaluator fidelity选线性ReAct、ToT、LATS、和进化搜索。

## 练习题

1. 配UCT c=0.1 vs c=2.0运行玩具LATS。Trace何变?
2. 将value function换为更噪scorer(加random jitter)。MCTS仍找最佳leaf否?它容最小signal-to-noise是何?
3. 实现beam-search ToT(每层留top-k)并比BFS。紧token预算何更好?
4. 阅读LATS Section 5.1。重现HumanEval轨迹数:达报告pass@1需多少rollout?
5. 阅读LATS论文"何LATS help less"讨论。写一段决策规则映射任务形至搜索策略。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Tree of Thoughts | "分支CoT" | Yao等——配自评估的thought节点树 |
| LATS | "LLM的MCTS" | Zhou等——在MCTS下统一ToT+ReAct+Reflexion |
| UCT | "Upper confidence bound" | 选公式平衡exploitation(Q)和exploration(ln N/n) |
| Value function | "此态何好" | 提示LLM分或环境reward;喂backprop |
| Policy | "Action proposer" | ReAct式generator;发候选下thought/action |
| Rollout | "模拟轨迹" | 用policy从节点walk至leaf、用value评分 |
| Backpropagate | "更新祖先" | 推leaf reward沿path向上、更新visit count和Q |
| Search cost | "Token爆炸" | Game of 24 100-1000x CoT;采用前预算 |

## 延伸阅读

- [Yao等,Tree of Thoughts(arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)——标准论文
- [Zhou等,LATS(arXiv:2310.04406)](https://arxiv.org/abs/2310.04406)——配Reflexion反馈的MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——搜索subgraph模式
- [AlphaEvolve(arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)——配程序evaluator的进化搜索