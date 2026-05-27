# 采样方法

> 采样是AI如何探索可能性的空间。

**类型:** 构建
**语言:** Python
**前置要求:** 第1阶段, 课程06-07 (概率, 贝叶斯定理)
**时间:** ~120分钟

## 学习目标

- 仅用均匀随机数从零实现逆CDF、拒绝采样和重要性采样
- 为语言模型token生成构建温度、top-k和top-p(nucleus)采样
- 解释重参数化技巧及其为何使VAE中采样反向传播成为可能
- 运行Metropolis-Hastings MCMC从未归一化目标分布采样

## 问题背景

语言模型处理完你的提示产生50,000个logits向量。词汇表每个token一个。现在它得选一个。如何?

如果它总选最高概率token，每个响应相同。确定性。无聊。如果均匀随机选，输出是胡言乱语。答案在这两极端间某处，由采样控制。

采样不限于文本生成。强化学习通过采样轨迹估计策略梯度。VAE通过从学习分布采样并反向传播随机性学习潜在表示。扩散模型通过采样噪声并迭代去噪生成图像。Monte Carlo方法估计无闭式解积分。MCMC算法探索无法枚举的高维后验分布。

每个生成AI系统是采样系统。采样策略决定输出质量、多样性和可控性。本课程从零构建每个主要采样方法，从均匀随机数开始到现代LLM和生成模型使用的技术。

## 概念讲解

### 为什么采样重要

采样在AI和ML中扮演四个基本角色:

**生成。** 语言模型、扩散模型和GAN都通过采样产生输出。采样算法直接控制创造力、一致性和多样性。温度、top-k和nucleus采样是工程师日常调节的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout采样神经元去激活。数据增强采样随机变换。重要性采样在强化学习(PPO, TRPO)中重加权样本减少梯度方差。

**估计。** ML中许多量无闭式解。数据分布上的期望损失、能量基模型的配分函数、贝叶斯推断的证据。Monte Carlo估计通过样本平均近似所有这些。

**探索。** MCMC算法在贝叶斯推断中探索后验分布。进化策略采样参数扰动。Thompson采样在bandits中平衡探索和利用。

核心挑战: 你只能直接从简单分布采样(均匀、正态)。其他一切，你需要方法将简单样本转换为目标分布样本。

### 均匀随机采样

每个采样方法从这里开始。均匀随机数生成器产生[0, 1)值，等长子区间有相等概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对 0 <= a <= b <= 1

性质:
  E[U] = 0.5
  Var(U) = 1/12
```

要均匀从n项离散集采样，生成U返回floor(n * U)。要从连续范围[a, b]采样，计算a + (b - a) * U。

关键洞见: 单个均匀随机数含精确足够随机性从任意分布产生一个样本。技巧是找正确变换。

### 逆CDF方法(逆变换采样)

累积分布函数(CDF)映射值到概率:

```
F(x) = P(X <= x)

性质:
  F非递减
  F(-inf) = 0
  F(+inf) = 1
  F映射实轴到[0, 1]
```

逆CDF映射概率回值。如果U ~ Uniform(0, 1)，则X = F_inverse(U)服从目标分布。

```
算法:
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为何有效:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布例子:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

解 F(x) = u:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

因(1 - U)和U同分布:
  x = -ln(u) / lambda
```

这当你能闭式写出F_inverse时完美工作。正态分布无闭式逆CDF，用其他方法(Box-Muller, 或数值近似)。

**离散版本:** 对离散分布，构建CDF为累积和，生成U，找累积和首次超过U的索引。这就是课程06中 `sample_categorical` 如何工作。

### 拒绝采样

当你不能逆CDF但能估计目标PDF到常数时，拒绝采样工作。

```
目标分布: p(x)  (可估计, 可能未归一化)
提议分布: q(x)  (可采样)
界限: M 使 p(x) <= M * q(x) 对所有x

算法:
  1. 采样 x ~ q(x)
  2. 采样 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x)), 接受x
  4. 否则, 拒绝并回步骤1

接受率 = 1/M
```

界限M越紧，接受率越高。低维(1-3)拒绝采样效果好。高维接受率指数下降因为大部分提议体积被拒绝。这是拒绝采样的维度诅咒。

**例子: 从截断正态采样。** 用截断范围均匀提议。M是正态PDF在该范围最大值。

**例子: 从半圆采样。** 在边界矩形均匀提议。如果点落在半圆内接受。这是Monte Carlo如何计算pi: 接受率等于面积比pi/4。

### 重要性采样

有时你不需要目标分布p(x)的样本。你需要估计p(x)下期望，你有不同分布q(x)的样本。

```
目标: 估计 E_p[f(x)] = f(x) * p(x) dx的积分

重写:
  E_p[f(x)] = f(x) * (p(x)/q(x)) * q(x) dx的积分
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x) 是重要性权重。

估计器:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在强化学习中关键。PPO(近端策略优化)中，你收集旧策略pi_old下轨迹但想优化新策略pi_new。重要性权重是pi_new(a|s) / pi_old(a|s)。PPO裁剪这些权重防止新策略偏离旧策略太远。

重要性采样估计器方差取决于q与p多相似。如果q与p差异大，少数样本得巨大权重主导估计。自归一化重要性采样除权重和减少此问题:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo估计

Monte Carlo估计通过随机样本平均近似积分。大数定律保证收敛。

```
目标: 估计 I = g(x)在域D上积分

方法:
  1. 从D均匀采样 x_1, ..., x_N
  2. I ~ (D体积 / N) * sum(g(x_i))

误差: O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这为何Monte Carlo方法在高维主导，那里网格积分不可能。

**估计pi:**

```
从[-1, 1] x [-1, 1]均匀采样(x, y)
计数多少落在单位圆内: x^2 + y^2 <= 1
pi ~ 4 * (圆内计数) / (总计数)
```

**估计期望:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛真实期望。
估计器方差 = Var(f(X)) / N
```

### Markov链Monte Carlo (MCMC): Metropolis-Hastings

MCMC构建Markov链其平稳分布是目标分布p(x)。足够步后，链样本(近似)p(x)样本。

```
目标: p(x)  (已知到归一化常数)
提议: q(x'|x)  (给定当前状态如何提议下一状态)

Metropolis-Hastings算法:
  1. 从某x_0开始
  2. 对 t = 1, 2, ..., T:
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受率:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率min(1, alpha)接受:
        - 如果 u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - 否则: x_{t+1} = x_t
  3. 丢弃前B样本(burn-in)
  4. 返回剩余样本
```

对称提议(q(x'|x) = q(x|x'))时，比简化为p(x')/p(x)。这是原始Metropolis算法。

**为何有效。** 接受规则确保细致平衡: 在x并移到x'的概率等于在x'并移到x的概率。细致平衡暗示p(x)是链平稳分布。

**实践考虑:**
- Burn-in: 链达平衡前丢弃早期样本
- Thinning: 每k个样本保留一个减少自相关
- 提议尺度: 太小链移动慢(高接受，慢探索); 太大大多提议被拒(低接受，原地不动)
- 高维Gaussian提议最优接受率约0.234

### Gibbs采样

Gibbs采样是多变量分布MCMC特例。不一次提议所有维度移动，每步从条件分布更新一个变量。

```
目标: p(x_1, x_2, ..., x_d)

算法:
  每迭代t:
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs采样要求你能从每个条件分布p(x_i | x_{-i})采样。这对许多模型直接:
- 贝叶斯网络: 条件源自图结构
- Gaussian混合: 条件是Gaussian
- Ising模型: 每自旋条件只依赖邻居

接受率总1(每提议接受)因为从精确条件采样自动满足细致平衡。

**局限。** 变量高度相关时Gibbs采样混合慢因为一次更新一变量无法通过分布作大对角移动。

### 温度采样(LLM用)

语言模型输出词汇表每个token logits z_1, ..., z_V。Softmax转为概率。温度在softmax前重缩放logits:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 标准softmax (原始分布)
T -> 0:  argmax (确定性, 总选最高logit)
T -> inf: 均匀 (所有token等可能)
T < 1.0: 锐化分布 (更自信, 更少多样)
T > 1.0: 平坦分布 (更少自信, 更多多样)
```

**为何有效。** T < 1除logits放大logits间差异。如z_1 = 2, z_2 = 1，除T = 0.5得z_1/T = 4, z_2/T = 2，差距更大。Softmax后最高logit token得更大份额。

**实践:**
- T = 0.0: 贪婪解码, 事实QA最佳
- T = 0.3-0.7: 略创意, 代码生成好
- T = 0.7-1.0: 平衡, 一般对话好
- T = 1.0-1.5: 创意写作, 头脑风暴
- T > 1.5: 越来越随机, 很少有用

温度不改哪些token可能。它改每个token分配概率质量。

### Top-k采样

Top-k采样限制候选集到最高概率k个token，重归一化从受限集采样。

```
算法:
  1. 计算所有V token softmax概率
  2. 按概率排序token(降序)
  3. 只保留top k token
  4. 重归一化: p_i' = p_i / sum(p_j for j in top-k)
  5. 从重归一化分布采样

k = 1:  贪婪解码
k = V:  无过滤 (标准采样)
k = 40: 典型设置, 移除不可能token长尾
```

Top-k防止模型选择词汇分布长尾中极不可能token(错字、胡话)。问题: k固定无关上下文。模型自信(一个token 95%概率)时，k=40仍允许39替代。模型不确定(概率分散1000 token)时，k=40切断可能选项。

### Top-p (Nucleus)采样

Top-p采样动态调整候选集大小。不保持固定数token，保持累积概率超过p的最小token集。

```
算法:
  1. 计算所有V token softmax概率
  2. 按概率排序token(降序)
  3. 找最小k使top-k概率和 >= p
  4. 只保留这k token
  5. 重归一化并采样

p = 0.9:  保持覆盖90%概率质量token
p = 1.0:  无过滤
p = 0.1:  非常限制, 近贪婪
```

模型自信时nucleus采样保持少token(可能2-3)。不确定时保持多(可能200)。这自适应行为是nucleus采样一般比top-k产生更好文本。

**常见组合:**
- 温度0.7 + top-p 0.9: 好通用设置
- 温度0.0(贪婪): 确定任务最佳
- 温度1.0 + top-k 50: Fan et al. (2018)原始论文设置

Top-k和top-p可组合。先应用top-k，然后在剩余集上top-p。

### 重参数化技巧(VAE用)

变分自编码器(VAE)通过编码输入到潜在空间分布、从该分布采样、解码样本回来学习。问题: 不能反向传播采样操作。

```
标准采样(不可微):
  z ~ N(mu, sigma^2)

  随机性阻塞梯度流。
  d/d_mu [从N(mu, sigma^2)采样] = ???
```

重参数化技巧分离随机性与参数:

```
重参数化采样:
  epsilon ~ N(0, 1)          (固定随机噪声, 无参数)
  z = mu + sigma * epsilon   (参数确定性函数)

  现在 z是mu和sigma确定性可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度流经mu和sigma。
```

这有效因为N(mu, sigma^2)与mu + sigma * N(0, 1)同分布。关键洞见: 移动随机性到无参数源(epsilon)，然后表达样本为参数可微变换。

**在VAE训练循环:**
1. 编码器输出每个输入mu和log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码z重构输入
5. 反向传播步骤4, 3, 2, 1(可能因步骤3可微)

无重参数化技巧VAE不能用标准反向传播训练。这单洞见使VAE实用。

### Gumbel-Softmax(可微类别采样)

重参数化技巧对连续分布(Gaussian)有效。离散类别分布需不同方法。Gumbel-Softmax提供类别采样可微近似。

**Gumbel-Max技巧(不可微):**

```
从log概率log(p_1), ..., log(p_k)类别分布采样:
  1. 每类别采样 g_i ~ Gumbel(0, 1)
     (g = -log(-log(u)), 其中 u ~ Uniform(0, 1))
  2. 返回 argmax(log(p_i) + g_i)

这产生精确类别样本。
```

**Gumbel-Softmax(可微近似):**

```
用软softmax替代硬argmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (温度)控制近似:
  tau -> 0:  接近one-hot向量 (硬类别)
  tau -> inf: 接近均匀 (1/k, 1/k, ..., 1/k)
  tau = 1.0: 软近似
```

Gumbel-Softmax产生离散样本连续松弛。输出是概率向量(软one-hot)而非硬one-hot。梯度流经softmax。训练前向可用"直通"估计器: 前向用硬argmax但反向用软Gumbel-Softmax梯度。

**应用:**
- VAE离散潜变量
- 神经架构搜索(选择离散操作)
- 硬注意力机制
- 离散行动强化学习

### 分层采样

标准Monte Carlo采样可能偶然留下样本空间间隙。分层采样通过分割空间为层并从每层采样强制均匀覆盖。

```
标准Monte Carlo:
  从[0, 1]均匀采样N点
  某区域可能聚集，其他间隙

分层采样:
  分割[0, 1]为N等层: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  每层均匀采样一点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样总比标准Monte Carlo方差低或等:

```
Var(分层) <= Var(标准Monte Carlo)

f(x)平滑变化时改进最大。
分段常数函数分层采样精确。
```

**应用:**
- 数值积分(拟Monte Carlo)
- 训练数据划分(确保每折类别平衡)
- 带分层重要性采样(结合两技术)
- NeRF沿相机射线用分层采样

### 与扩散模型联系

扩散模型通过采样过程生成图像。前向过程T步向图像加Gaussian噪声直到纯噪声。反向过程学习去噪，逐步恢复原图。

```
前向过程(已知):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  T步后: x_T ~ N(0, I)  (纯噪声)

反向过程(学习):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每去噪步是采样步。
```

与本课程方法联系:
- 每去噪步用重参数化技巧(采样噪声，应用确定性变换)
- 噪声调度{alpha_t}控制温度退火形式
- 训练用Monte Carlo估计近似ELBO(证据下界)
- 扩散模型祖先是Markov链(每步只依赖当前状态)

整个图像生成过程是迭代采样: 从噪声开始，每步采样略微少噪版本条件于学习去噪模型。

## 动手实践

### 步骤1: 均匀和逆CDF采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成10,000指数样本验证均值1/lambda。

### 步骤2: 拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态分布抽取。直方图验证形状。

### 步骤3: 重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀提议估计正态分布下E[X^2]。与已知答案(mu^2 + sigma^2)比较。

### 步骤4: Monte Carlo估计pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布(两Gaussian混合)采样。可视化链轨迹。

### 步骤6: Gibbs采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤7: 温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变token logits集输出分布。

### 步骤8: Top-k和top-p采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤9: 重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度流经重参数化样本但不经直接采样。

### 步骤10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示减小温度使输出趋近one-hot向量。

完整实现及所有可视化在 `code/sampling.py`。

## 实际应用

NumPy和SciPy生产版本:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数均值: {exponential_samples.mean():.4f} (期望2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF在1.96: {normal.cdf(1.96):.4f}")
print(f"逆CDF在0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采样token索引: {token}")
```

大规模MCMC用专用库:
- PyMC: 带NUTS(自适应HMC)全贝叶斯建模
- emcee: 集成MCMC采样器
- NumPyro/JAX: GPU加速MCMC

你从零构建这些。现在知道库调用在做什么。

## 练习题

1. 实现Cauchy分布逆CDF采样。CDF是F(x) = 0.5 + arctan(x)/pi。生成10,000样本绘直方图与真实PDF比较。注意重尾(极端值远离中心)。

2. 用拒绝采样从Beta(2, 5)分布用Uniform(0, 1)提议生成样本。绘接受样本与真实Beta PDF比较。理论接受率多少?

3. 用Monte Carlo估计1,000、10,000和100,000样本估计sin(x)从0到pi积分。比较每级误差。验证误差O(1/sqrt(N))缩放。

4. 实现Metropolis-Hastings从2D分布p(x, y)正比于exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)采样。绘样本和链轨迹。实验不同提议标准差。

5. 构建完整文本生成demo: 给10词词汇表logits，生成20 token序列用 贪婪、 温度=0.7、 top-k=3、 top-p=0.9。5次运行比较输出多样性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 采样 | "抽取随机值" | 根据概率分布生成值。所有生成AI背后的机制 |
| 均匀分布 | "所有等可能" | [a, b]每值有相等概率密度1/(b-a)。所有采样方法起点 |
| 逆CDF | "概率变换" | F_inverse(U)将均匀样本转为任意已知CDF分布样本。精确高效 |
| 拒绝采样 | "提议和接受/拒绝" | 从简单提议生成，以正比于目标/提议比概率接受。精确但浪费样本 |
| 重要性采样 | "重加权样本" | 用q(x)样本通过每样本加权p(x)/q(x)估计p(x)下期望。PPO等RL核心 |
| Monte Carlo | "平均随机样本" | 用样本平均近似积分。误差O(1/sqrt(N))与维度无关 |
| MCMC | "收敛随机游走" | 构建平稳分布是目标的Markov链。Metropolis-Hastings是基础算法 |
| Metropolis-Hastings | "接受上山，有时下山" | 提议移动，基于密度比接受。细致平衡保证收敛到目标分布 |
| Gibbs采样 | "一次一变量" | 保持其他固定从条件分布更新每变量。100%接受率 |
| 温度 | "自信旋钮" | Softmax前除logits。T<1锐化(更自信)，T>1平坦(更多样) |
| Top-k采样 | "保留k个最佳" | 清零除最高概率k个token外所有，重归一化，采样。固定候选集大小 |
| Nucleus采样(top-p) | "保留可能者" | 保持累积概率超过p的最小token集。自适应候选集大小 |
| 重参数化技巧 | "移出随机性" | 写z = mu + sigma * epsilon其中epsilon ~ N(0,1)。使采样可微。VAE训练必需 |
| Gumbel-Softmax | "软类别采样" | 用Gumbel噪声+带温度softmax对类别采样可微近似 |
| 分层采样 | "强制覆盖" | 分割样本空间为层，从每层采样。总比朴素Monte Carlo方差低 |
| Burn-in | "预热期" | MCMC早期样本在链达平稳分布前丢弃 |
| 细致平衡 | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p是Markov链平稳分布充分条件 |
| 扩散采样 | "迭代去噪" | 从噪声开始通过学习去噪步生成数据。每步是条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC基础详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始Gumbel-Softmax论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p)采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE论文引入重参数化技巧
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM连接采样到图像生成