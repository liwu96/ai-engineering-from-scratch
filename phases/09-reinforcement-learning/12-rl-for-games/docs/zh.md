# 强化学习游戏——AlphaZero、MuZero和大语言模型推理时代

> 1992:TD-Gammon纯TD败人类冠军backgammon。2016:AlphaGo败Lee Sedol。2017:AlphaZero从零主宰象棋、将棋和围棋。2024:DeepSeek-R1证明同配方,GRPO替PPO,于推理有效。游戏是本阶段每突破驱动基准。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段9课程05(DQN)、阶段9课程08(PPO)、阶段9课程09(RLHF)、阶段9课程10(MARL)
**时间:** ~120分钟

## 问题背景

游戏有强化学习要的一切。干净奖励(胜/负)。无限情节(自博弈重置)。完美仿真(游戏*是*仿真器)。离散或小连续动作空间。多智能体结构强制对抗鲁棒。

游戏也是每主要强化学习突破测试方式。TD-Gammon(backgammon, 1992)。Atari-DQN(2013)。AlphaGo(2016)。AlphaZero(2017)。OpenAI Five(Dota 2, 2019)。AlphaStar(StarCraft II, 2019)。MuZero(学习模型, 2019)。AlphaTensor(矩阵乘法, 2022)。AlphaDev(排序算法, 2023)。DeepSeek-R1(数学推理, 2025)——游戏强化学习技术于文本最新示范。

此顶点通过单统一镜头调研三地标架构——AlphaZero、MuZero和GRPO:**自博弈+搜索+策略改进**。每泛化前;GRPO特应用于大语言模型推理,词元作动作,数学验证作胜信号。

## 概念讲解

![AlphaZero ↔ MuZero ↔ GRPO:同循环,不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 对自玩游戏
    policy_target = search.improved_policy(trajectory) # 搜索改原始策略
    policy_net.update(policy_target, value_target)     # 搜索输出监督
```

**AlphaZero(2017)。**Silver等。给定已知规则游戏(象棋、将棋、围棋):

- 策略-价值网络:单塔`f_θ(s) → (p, v)`。`p`是合法移动先验。`v`是期望游戏结果。
- 蒙特卡洛树搜索(MCTS):每移动,展可能延续树。用`(p, v)`作先验+自举。UCB(PUCT)选节点:`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自博弈:智能体-vs-智能体玩游戏。移动`t`,MCTS访问分布`π_t`成策略训目标。
- 损失:`L = (v - z)² - π · log p + c · ||θ||²`。`z`是游戏结果(+1 / 0 / -1)。

零人类知识。零手工启发。单配方数千万自博弈游戏后主宰象棋、将棋和围棋。

**MuZero(2019)。**Schrittwieser等。移规则已知要求。

- 不固定环境,学*潜动力学模型*`(h, g, f)`:
  - `h(s)`:编码观察到潜状态。
  - `g(s_latent, a)`:预测下一潜状态+奖励。
  - `f(s_latent)`:预测策略先验+价值。
- MCTS在*学习潜空间*跑。同搜索,同训循环。
- 于围棋、象棋、将棋*和*Atari工作——单算法,无规则知识。

**随机MuZero(2022)。**加随机动力学和机会节点;扩到backgammon类游戏。

**Muesli、Gumbel MuZero(2022-2024)。**样本效率和确定性搜索改进。

**GRPO(2024-2025)。**DeepSeek-R1配方。同AlphaZero形循环,应用于语言模型推理:

- "游戏":答数学/编程/推理问题。"胜"=验证器(测例过、数值答匹配)返1。
- 策略:大语言模型。动作:词元。状态:提示词+响应至今。
- 无critic(PPO式V_φ)。替之,每提示词从策略采样`G`完成。算每奖励。用**组相对优势**`A_i = (r_i - mean_r) / std_r`作REINFORCE式更新信号。
- KL惩罚到参考策略防漂移(如RLHF)。
- 完整损失:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型、无critic、无MCTS。组相对基线替三者。推理基准匹配或超PPO-RLHF质量于计算分数。

**R1配方完整。**DeepSeek-R1(DeepSeek 2025)是一论文两模型:

- **R1-Zero。**从DeepSeek-V3基模型起。无SFT。直接应用GRPO配两奖励组件:*准确奖励*(规则基——终答解析到正确数/代码过单元测)和*格式奖励*(完成包思维链于` '..', '..' `标签)。数千步,平均响应长从~100到~10,000词元增长,数学基准分爬到近o1-preview水平。模型从零学推理。缺点:思维链常不可读、混语言、缺风格打磨。
- **R1。**用四阶段管道修R1-Zero可读性:
  1. **冷启动SFT。**收集数千干净格式长CoT演示。基模型于上监督微调。给可读起点。
  2. **推理导向GRPO。**应用GRPO配准确+格式奖励加*语言一致性*奖励防代码切换。
  3. **拒绝采样+SFT第二轮。**从强化学习检查点采样~600K推理轨迹,仅保留终答正确且CoT可读,结合~200K非推理SFT例(写作、问答、自认知)。基模型再微调。
  4. **全谱GRPO。**再一轮强化学习覆盖推理(规则基奖励)和通用对齐(有用/无害偏好基奖励)。

结果开放权重上AIME和MATH-500匹配o1,小到蒸馏。同论文也发六蒸馏密集模型(Qwen-1.5B到Llama-70B)于R1推理轨迹SFT——学生无强化学习。强强化学习教师蒸馏一致击败学生规模从头强化学习。

**为何推理用GRPO替PPO。**DeepSeekMath论文(2024年2月)三原因:(1)无价值网络训,内存减半;(2)组基线自然处理推理任务产稀疏轨迹末奖励;(3)每提示词归一化使不同难度问题优势可比,单PPO critic不能。

**搜索自由vs搜索基。**游戏分支:

- *完美信息长视界游戏*(围棋、象棋):仍搜索基。AlphaZero / MuZero主导。
- *大语言模型推理*:生产无MCTS;完整展开GRPO,推理计算Best-of-N。过程奖励模型(PRM)暗示步级搜索加回。

## 动手实践

`code/main.py`代码实现**微型GRPO**——多组样本bandit。算法同大语言模型;仅策略和环境更简。教*损失*和*组相对优势*,2025创新。

### Step 1:小验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真GRPO验证器跑单元测或检查数学相等。

### Step 2:策略:每提示词K答案词元上softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于提示词条件大语言模型最终层输出。

### Step 3:组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL惩罚:拉theta向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是2024 DeepSeek技。无critic需。"基线"是组均值,归一化用组标准差。

### Step 4:比REINFORCE基线(无价值)

同设置,同计算,朴素REINFORCE。GRPO更快更稳收敛。

### Step 5:观察熵和KL

同RLHF诊断:均值KL到参考、策略熵、奖励时序。稳定则训完。

## 陷阱

- **验证器博弈奖励骇。**GRPO继承RLHF风险:如验证器错或可利用,大语言模型会找漏洞。鲁棒验证器(多测例、形式证明)重要。
- **组大小太小。**组基线方差如`1/√G`。`G = 4`下,优势信号噪声;标准选`G = 8`到`64`。
- **长度偏。**不同长大语言模型完成有不同log概率。按词元数归一化,或用序列级log概率,或截到最大长。
- **纯自博弈循环。**AlphaZero式训一般和博弈可困主导循环。多样对手池(联赛博弈,课程10)缓解。
- **搜索-策略不匹配。**AlphaZero训策略模仿搜索输出。如策略网太小不能表示搜索分布,训停滞。
- **计算门槛。**MuZero / AlphaZero需巨大计算。单消融常数百GPU时。存在微型演示(如Connect Four AlphaZero)学习。
- **验证器覆盖。**单元测过bug解强化bug。设计捕获边缘情况验证器。

## 实际应用

2026游戏强化学习格局,按域:

| 域 | 主导方法 |
|------|----------|
| 两玩家零和棋盘游戏(围棋、象棋、将棋) | AlphaZero / MuZero / KataGo |
| 不完美信息卡牌游戏(扑克) | CFR +深度学习(DeepStack、Libratus、Pluribus) |
| Atari /像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大多人策略(Dota、StarCraft) | PPO +自博弈+联赛(OpenAI Five、AlphaStar) |
| 大语言模型数学/代码推理 | GRPO(DeepSeek-R1、Qwen-RL、开放复现) |
| 大语言模型对齐 | DPO / RLHF-PPO(非GRPO;验证器是偏好非可验证) |
| 机器人学 | PPO + DR(非游戏强化学习,但用同策略梯度工具) |
| 组合问题 | AlphaZero变体(AlphaTensor、AlphaDev) |

*配方*——自博弈、搜索增强改进、策略蒸馏——跨文本、像素和物理控制。GRPO是最年轻实例;更多将至。

## 产出成果

存`outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: 设计给定域游戏强化学习或推理强化学习训管道(AlphaZero / MuZero / GRPO)。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标(完美信息游戏/不完美信息/Atari/大语言模型推理/组合),输出:

1. 环境适配。已知规则?马尔可夫?随机?多智能体?指导AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS(PUCT配学习先验)、Gumbel采样、Best-of-N或无。
3. 自博弈计划。对称自博弈/联赛/离线数据/验证器生成。
4. 目标信号。游戏结果/验证器奖励/偏好/学习模型。含鲁棒计划。
5. 诊断。对基线胜率、ELO曲线、验证器通过率、KL到参考。

拒不完美信息游戏AlphaZero(路由CFR)。拒无可信验证器GRPO。拒无固定基线对手集游戏强化学习管道(否则自博弈ELO未校准)。
```

## 练习题

1. **简单。**实现`code/main.py`中GRPO bandit。2提示词×4答案词元每训。<1,000更新`G=8`收敛。
2. **中等。**插PPO(裁剪)和朴素REINFORCE。比同bandit上样本效率和奖励方差GRPO。
3. **困难。**扩到长2"推理链":智能体发两词元,验证器奖励对。测GRPO如何跨两步序列处理信用分配。(提示:每*完整序列*算组优势,传到两词元位置。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MCTS | "学习网树搜索" | 蒙特卡洛树搜索;UCB1/PUCT选择配学习`(p, v)`先验。 |
| AlphaZero | "自博弈+MCTS" | 策略-价值网训匹配MCTS访问和游戏结果。 |
| MuZero | "学习模型AlphaZero" | 同循环但通过学习动力学于潜空间。 |
| GRPO | "无critic PPO" | 组相对策略优化;REINFORCE配组均值基线+KL。 |
| PUCT | "AlphaZero UCB" | `Q + c · p · √N / (1 + N_a)`——平衡价值估计和先验。 |
| 自博弈 | "智能体对过去自" | 零和标准;对称训信号。 |
| 联赛博弈 | "种群基自博弈" | 过去+当前+剥削者采样作对手。 |
| 验证器奖励 | "可验证强化学习" | 奖励来自确定性检查器(测过、答匹配)。 |
| 过程奖励 | "PRM" | 每推理步评分,非仅终答。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)——引入GRPO和组相对基线论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)——完整四阶段R1配方加R1-Zero消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400)——CFR +深度学习规模。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343)——开启一切论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)——配自定义奖励函数应用GRPO生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math)——多尺度R1配方开放复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf)——自博弈、搜索和"设计奖励"教材框架,R1于大语言模型尺度实例化。