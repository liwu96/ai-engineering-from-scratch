# 深度Q网络(DQN)

> 2013:Mnih在原始像素上训一个Q-learning网络,七Atari游戏败每个经典强化学习智能体。2015:扩到49游戏,Nature发表,点燃深度强化学习时代。DQN是Q-learning加三个使函数近似稳定的技巧。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段3课程03(反向传播)、阶段9课程04(Q-learning、SARSA)
**时间:** ~75分钟

## 问题背景

表格Q-learning每(状态,动作)对需单独Q值。棋盘~10⁴³状态。Atari帧210×160×3=100,800特征。表格强化学习千状态死,更不用说亿。

修复事后看明显:神经网络替换Q表,`Q(s, a; θ)`。但事后明显花数十年。朴素函数近似配Q-learning在"致命三重奏"下发散——函数近似+自举+异策略学习。Mnih等(2013, 2015)识别三工程技巧稳定学习:

1. **经验回放**解相关转移。
2. **目标网络**冻结自举目标。
3. **奖励裁剪**归一化梯度幅度。

Atari上DQN首次单架构单超参集从原始像素解数十控制问题。自建"深度强化学习"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——叠此三技巧基顶。

## 概念讲解

![DQN训练循环:环境、回放缓冲、在线网、目标网、贝尔曼TD损失](../assets/dqn.svg)

**目标。**DQN最小化神经Q函数一步TD损失:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ`=在线网络,梯度下降每步更新。`θ^-`=目标网络,周期从`θ`复制(~10,000步)。`D`=过去转移回放缓冲。

**三技巧,重要性顺序:**

**经验回放。**~10⁶转移环形缓冲。每训练步均匀随机采样小批次。打断时间相关(连续帧近相同),让网络多次学稀有奖励转移,解相关连续梯度更新。无它,Atari上神经网同策略TD发散。

**目标网络。**贝尔曼方程两边用同网络`Q(·; θ)`使目标每更新移动——"追自己尾"。修复:保持第二网络`Q(·; θ^-)`冻结权重。每`C`步,复制`θ → θ^-`。稳定回归目标数千梯度步。软更新`θ^- ← τ θ + (1-τ) θ^-`(DDPG、SAC用)更平滑变体。

**奖励裁剪。**Atari奖励幅度1到1000+变。裁剪到`{-1, 0, +1}`止单游戏主导梯度。奖励幅度重要时错;Atari仅符号重要时好。

**双重DQN。**Hasselt(2016)修复最大偏:在线网*选*动作,目标网*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

插入替换,一致更好。默认用。

**其他改进(Rainbow, 2017):**优先回放(高TD误差转移采样更多)、Dueling架构(分离`V(s)`和优势头)、噪声网络(学习探索)、n步回报、分布Q(C51/QR-DQN)、多步自举。每加几百分点;增益大致可加。

## 动手实践

代码stdlib-only无numpy——手卷单隐藏层MLP微小连续GridWorld,每训练步微秒跑。算法与规模Atari DQN相同。

### Step 1:回放缓冲

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari~50,000容量;玩具环境5,000够。

### Step 2:微小Q网络(手工MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传:线性→ReLU→线性。整个网络。

### Step 3:DQN更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状是课程04Q-learning两差别: 可微`Q(·; θ)`反传而非索引表, 目标用`Q(·; θ^-)`。

### Step 4:外循环

每情节,`Q(·; θ)`上ε-贪婪行动,推转移入缓冲,采样小批次,取梯度步,周期同步`θ^- ← θ`。模式:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16维独热状态微小GridWorld上,智能体~500情节学近最优策略。Atari,扩到200M帧加CNN特征提取器。

## 陷阱

- **致命三重奏。**函数近似+异策略+自举可发散。DQN目标网+回放缓解;不删任一。
- **探索。**ε须衰减,典型训练前~10%从1.0到0.01。无足够早探索Q网收敛局部盆地。
- **过估。**噪声Q上`max`向上偏。生产始终用双重DQN。
- **奖励尺度。**裁剪或归一化奖励;梯度幅度比例于奖励幅度。
- **回放缓冲冷启动。**缓冲几千转移前不训。~20样本早梯度过拟合。
- **目标同步频率。**太频≈无目标网;太不频≈陈旧目标。Atari DQN用10,000环境步。经验:训练视界~1/100同步。
- **观察预处理。**Atari DQN堆4帧使状态马尔可夫。任何速度信息环境需帧堆叠或循环状态。

## 实际应用

2026年,DQN很少最优但仍参考异策略算法:

| 任务 | 选择方法 | 何不DQN? |
|------|----------|----------|
| 离散动作Atari式 | Rainbow DQN或Muesli | 同框架,更多技巧。 |
| 连续控制 | SAC/TD3(阶段9课程07) | DQN无策略网络。 |
| 同策略/高通量 | PPO(阶段9课程08) | 无回放缓冲;更易扩。 |
| 离线强化学习 | CQL/IQL/决策Transformer | 保守Q目标,无自举爆。 |
| 大离散动作空间(推荐) | 动作嵌入DQN或IMPALA | 行;装饰重要。 |
| 大语言模型强化学习 | PPO/GRPO | 序列级,非步级;不同损失。 |

教训仍传。回放和目标网络现于SAC、TD3、DDPG、SAC-X、AlphaZero自玩缓冲、每个离线强化学习方法。奖励裁剪作为PPO优势归一化存活。架构蓝图。

## 产出成果

存`outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: 产离散动作强化学习任务DQN训练配置(缓冲、目标同步、ε调度、奖励裁剪)。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定离散动作环境(观察形、动作数、视界、奖励尺度),输出:

1. 网络。架构(MLP/CNN/Transformer)、特征维、深。
2. 回放缓冲。容量、小批次大小、预热大小。
3. 目标网络。同步策略(硬每C步或软τ)。
4. 探索。ε起/末/调度长。
5. 损失。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. 双重DQN。默认开除非显式原因禁。

拒绝发货无目标网络、无回放缓冲、或ε持1的DQN。拒绝连续动作任务(路由SAC/TD3)。标记任何奖励范围>10倍每步均值需裁剪或尺度归一化。
```

## 练习题

1. **简单。**跑`code/main.py`。绘每情节回报曲线。多少情节运行均值超-10?
2. **中等。**禁目标网络(贝尔曼目标两边用在线网)。测训练不稳定——回报振荡或发散?
3. **困难。**加双重DQN:在线网选`argmax a'`,目标网评估。噪声奖励GridWorld上,1000情节后比较`Q(s_0, best_a)`偏vs真`V*(s_0)`有vs无双重DQN。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| DQN | "深度Q-learning" | 配神经Q函数、回放缓冲、目标网络Q-learning。 |
| 经验回放 | "洗牌转移" | 每梯度步均匀采样环形缓冲;解相关数据。 |
| 目标网络 | "冻结自举" | 贝尔曼目标中用周期复制Q;稳定训练。 |
| 致命三重奏 | "为何强化学习发散" | 函数近似+自举+异策略=无收敛保证。 |
| 双重DQN | "最大偏修复" | 在线网选动作,目标网评估。 |
| Dueling DQN | "V和A头" | 分解Q=V+A-mean(A);相同输出,更好梯度流。 |
| Rainbow | "所有技巧" | DDQN+PER+dueling+n步+噪声+分布合一。 |
| PER | "优先回放" | TD误差幅度比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)——点燃深度强化学习2013 NeurIPS workshop论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)——Nature论文,49游戏DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)——DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581)——Dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)——叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)——清晰现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf)——DQN目标网络和回放缓冲设计驯服"致命三重奏"教材处理。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/)——消融研究用参考单文件DQN;配本课从头版本读好。