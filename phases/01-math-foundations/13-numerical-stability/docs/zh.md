# 数值稳定性

> 浮点数是有漏洞的抽象。它会在训练期间咬你，而且你不会预见到。

**类型:** 构建
**语言:** Python
**前置要求:** 第1阶段, 课程01-04
**时间:** ~120分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的softmax和log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心有限差分验证解析梯度与数值梯度
- 解释为什么训练时bfloat16优于float16，损失缩放如何防止梯度下溢

## 问题背景

你的模型训练三小时后损失变成NaN。你添加打印语句。logits在第9000步还好。在第9001步它们是 `inf`。到第9002步每个梯度都是 `nan`，训练死亡。

或者: 你的模型训练完成但精度比论文声称的低2%。你检查一切。架构匹配。超参数匹配。数据匹配。问题是论文用了float32而你用了float16但没有正确的缩放。32位累积的舍入误差静静吞噬了你的精度。

或者: 你从零实现交叉熵损失。小logits时工作。当logits超过100，返回 `inf`。softmax溢出因为 `exp(100)` 大于float32能表示的。每个ML框架用两行技巧处理这个。你不知道技巧存在。

数值稳定性不是理论关注。它是训练运行成功与静默失败的区别。你将调试的每个严重MLbug最终归结到浮点数。

## 概念讲解

### IEEE 754: 计算机如何存储实数

计算机按IEEE 754标准将实数存储为浮点值。浮点数有三部分: 符号位、指数和尾数(有效数字)。

```
Float32布局 (32位总共):
[1符号] [8指数] [23尾数]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度(多少有效数字)。指数决定范围(数值多大或多小)。

```
格式     位   指数  尾数  十进制位  范围(约)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32给你约7位十进制精度。意味着能区分1.0000001和1.0000002，但不能区分1.00000001和1.00000002。7位后都是舍入噪声。

float16给你约3位。能表示的最大数是65,504。这对ML来说惊恐地小——logits、梯度、激活值经常超过这个。

bfloat16是Google对float16范围问题的答案。它与float32相同的8位指数(相同范围，到3.4e38)但只有7位尾数(比float16精度更低)。训练神经网络时，范围比精度更重要，所以bfloat16通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字0.1无法在二进制浮点中精确表示。在二进制中，它是循环分数:

```
0.1二进制 = 0.0001100110011001100110011... (永远循环)
```

Float32截断到23位尾数。存储值约0.100000001490116。类似，0.2存储为约0.200000002980232。它们的和是0.300000004470348，不是0.3。

```
Python中:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对ML重要因为:

1. 损失比较如 `if loss < threshold` 可能给出错误答案
2. 累积许多小值(数千步的梯度更新)偏离真实总和
3. 校验码和可复现性测试用 `==` 比较浮点会失败

修复: 永不用 `==` 比较浮点。用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数，有效数字抵消，剩下舍入噪声提升为首位数字。

```
a = 1.0000001    (float32存储为 1.00000011920929)
b = 1.0000000    (float32存储为 1.00000000000000)

真实差异:  0.0000001
计算值:         0.00000011920929

相对误差: 19.2%
```

单次减法有19%相对误差。ML中这发生在:

- 计算均值大数据的方差: `E[x^2] - E[x]^2` 当E[x]大
- 减去几乎相等的对数概率
- 用太小epsilon计算有限差分梯度

修复: 重排公式避免减去大且几乎相等的数。方差用Welford算法或先中心化数据。对数概率全程在对数空间工作。

### 溢出与下溢

溢出发生在结果太大无法表示。下溢发生在太小(比最小可表示正数更接近零)。

```
Float32边界:
  最大:  3.4028235e+38
  最小正数(正规): 1.175e-38
  最小正数(非正规): 1.401e-45
  溢出:  任何 > 3.4e38 变成 inf
  下溢: 任何 < 1.4e-45 变成 0.0
```

`exp()` 函数是ML中溢出主要来源:

```
exp(88.7)  = 3.40e+38   (勉强放入float32)
exp(89.0)  = inf         (溢出)
exp(-87.3) = 1.18e-38   (勉强不下溢)
exp(-104)  = 0.0         (下溢为零)
```

`log()` 函数方向相反:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (好)
log(1e-46) = -inf        (输入下溢到0, 然后 log(0) = -inf)
```

ML中 `exp()` 出现于softmax、sigmoid和概率计算。`log()` 出现于交叉熵、对数似然和KL散度。组合 `log(exp(x))` 没有正确技巧是雷区。

### Log-Sum-Exp技巧

直接计算 `log(sum(exp(x_i)))` 数值危险。如果任何 `x_i` 大，`exp(x_i)` 溢出。如果所有 `x_i` 很负，每个 `exp(x_i)` 下溢到零，`log(0)` 是 `-inf`。

技巧: 指数化前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效: 减去 `max(x)` 后，最大指数是 `exp(0) = 1`。不可能溢出。和中至少一项是1，所以和至少1，`log(1) = 0`。不可能下溢到 `-inf`。

证明:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (加减c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (提出exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设 `c = max(x)` 消除溢出。

这技巧在ML中无处不在:
- Softmax归一化
- 交叉熵损失计算
- 序列模型中对数概率求和
- 高斯混合
- 变分推断

### 为什么Softmax需要最大值减法技巧

Softmax将logits转为概率:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

无技巧时logits [100, 101, 102] 导致溢出:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些溢出float32 (最大 ~3.4e38)? 否, 2.69e43 < 3.4e38? 实际:
exp(88.7) 已经在float32极限。
exp(100) 在float32中是 inf。
```

有技巧时，减去max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算安全。这不是优化。这是正确性要求。

### NaN和Inf: 检测与预防

`nan` (非数值)和 `inf` (无穷)在计算中病毒式传播。梯度更新中一个 `nan` 使权重 `nan`，使随后每个输出 `nan`。一步内训练死亡。

`inf` 如何出现:
- 大正数的 `exp()`
- 除零: `1.0 / 0.0`
- `float32` 累积溢出

`nan` 如何出现:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及现有 `nan` 的算术

检测:

```python
import math

math.isnan(x)       # x是nan时True
math.isinf(x)       # x是+inf或-inf时True
math.isfinite(x)    # x既非nan也非inf时True
```

预防策略:

1. Clamp `exp()` 输入: `exp(clamp(x, -80, 80))`
2. 分母加epsilon: `x / (y + 1e-8)`
3. `log()` 内加epsilon: `log(x + 1e-8)`
4. 用稳定实现(log-sum-exp, 稳定softmax)
5. 梯度裁剪防止权重爆炸
6. 调试时每次前向后检查 `nan`/`inf`

### 数值梯度检查

解析梯度(来自反向传播)可能有bug。数值梯度检查用有限差分计算梯度验证它们。

中心差分公式:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是O(h^2)精度，远优于前向差分 `(f(x+h) - f(x)) / h` 只有O(h)。

选择h: 太大近似错误。太小灾难性抵消摧毁答案。`h = 1e-5` 到 `1e-7` 典型。

检查: 计算解析与数值梯度的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则:
- relative_error < 1e-7: 完美，梯度正确
- relative_error < 1e-5: 可接受，可能正确
- relative_error > 1e-3: 有问题
- relative_error > 1: 梯度完全错误

实现新层或损失函数时总是检查梯度。PyTorch提供 `torch.autograd.gradcheck()`。

### 混合精度训练

现代GPU有专用硬件(Tensor Cores)计算float16矩阵乘法比float32快2-8倍。混合精度训练利用这个:

```
1. 维护float32权重主副本
2. 前向用float16 (快)
3. 损失用float32计算 (防止溢出)
4. 反向用float16 (快)
5. 梯度缩放到float32
6. 更新float32主权重
```

纯float16训练问题: 梯度经常很小(1e-8或更小)。Float16将低于~6e-8的任何值下溢为零。模型停止学习因为所有梯度更新是零。

修复是损失缩放:

```
1. 损失乘大缩放因子(如1024)
2. 反向计算(loss * 1024)的梯度
3. 所有梯度大1024倍(推到float16下溢之上)
4. 更新权重前梯度除1024
5. 净效果: 相同更新，但不下溢
```

动态损失缩放自动调整缩放因子。从大值(65536)开始。如果梯度溢出到 `inf`，减半。如果N步无溢出，加倍。

### bfloat16 vs float16: 为什么bfloat16训练胜出

```
float16:   [1符号] [5指数]  [10尾数]
bfloat16:  [1符号] [8指数]  [7尾数]
```

float16更精度(10尾数位 vs 7)但有限范围(最大 ~65,504)。bfloat16更少精度但与float32相同范围(最大 ~3.4e38)。

训练神经网络:

- 激活和logits训练峰值时经常超过65,504。float16溢出; bfloat16处理。
- float16需要损失缩放但bfloat16通常不需要因为其范围覆盖梯度幅度谱。
- bfloat16是float32简单截断: 丢弃尾数底部16位。转换简单且指数无损。

float16用于推理，值有界且精度更重要。bfloat16用于训练，范围更重要。这就是为什么TPU和现代NVIDIA GPU(A100, H100)有原生bfloat16支持。

### 梯度裁剪

梯度爆炸发生在梯度通过多层指数增长(RNN、深网络、Transformer常见)。单大梯度一步可破坏所有权重。

两种裁剪类型:

**按值裁剪:** 每个梯度元素独立clamp。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可改变梯度向量方向。

**按范数裁剪:** 缩放整个梯度向量使范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向。这是 `torch.nn.utils.clip_grad_norm_()` 做的。标准选择。

典型值: Transformer `max_norm=1.0`，RL `max_norm=0.5`，简单网络 `max_norm=5.0`。

梯度裁剪不是hack。是安全机制。没有它，单个异常batch可产生足够大的梯度毁掉数周训练。

### 归一化层作为数值稳定器

批归一化、层归一化和RMS归一化通常被呈现为帮助训练收敛的正则化器。它们也是数值稳定器。

无归一化，激活可在层间指数增长或收缩:

```
层1: 值在[0, 1]
层5: 值在[0, 100]
层10: 值在[0, 10,000]
层50: 值在[0, inf]
```

归一化每层重新中心化和缩放激活:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (典型1e-5)防止所有激活相同时除零。学习参数 `gamma` 和 `beta` 让网络恢复任何需要的尺度。

这保持值在网络中数值安全范围，防止前向溢出和反向梯度爆炸。

### 常见ML数值bug

**Bug: 几轮后损失NaN。**
原因: logits增长太大，softmax溢出。或学习率太高权重发散。
修复: 用稳定softmax(最大值减法)，降学习率，加梯度裁剪。

**Bug: 损失卡在log(num_classes)。**
原因: 模型输出接近均匀概率。常意味梯度消失或模型根本没学习。
修复: 检查数据标签正确，验证损失函数，检查死ReLU。

**Bug: 验证精度比预期低1-3%。**
原因: 混合精度无正确损失缩放。梯度下溢静默清零小更新。
修复: 启用动态损失缩放，或切换到bfloat16。

**Bug: 某层梯度范数是0.0。**
原因: 死ReLU神经元(所有输入负)，或float16下溢。
修复: 用LeakyReLU或GELU，用梯度缩放，检查权重初始化。

**Bug: 模型在一个GPU工作但另一GPU结果不同。**
原因: 非确定性浮点累积顺序。GPU并行归约在不同硬件以不同顺序求和，浮点加法不可结合。
修复: 接受小差异(1e-6)，或设 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug: 损失计算中 `exp()` 返回 `inf`。**
原因: 原始logits传给 `exp()` 无最大值减法技巧。
修复: 用 `torch.nn.functional.log_softmax()` 内部实现log-sum-exp。

**Bug: 从float32切换到float16后训练发散。**
原因: float16不能表示低于6e-8的梯度幅度或高于65,504的激活。
修复: 用带损失缩放的混合精度(AMP)，或用bfloat16替代。

## 动手实践

### 步骤1: 展示浮点精度极限

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差异: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2: 实现朴素vs稳定softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤3: 实现稳定log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤4: 实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5: 梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际应用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保持: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("好", [1.0, 2.0, 3.0])
check_tensor("坏",  [1.0, float('nan'), 3.0])
check_tensor("丑", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况在 `code/numerical.py`。

## 产出成果

本课程产生:
- `code/numerical.py` 包含稳定softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md` 用于诊断训练中NaN/Inf和数值问题

这些稳定实现在第3阶段构建训练循环和第4阶段实现注意力机制时会再次出现。

## 练习题

1. **灾难性抵消。** 用朴素公式 `E[x^2] - E[x]^2` 在float32中计算[1000000.0, 1000001.0, 1000002.0]的方差。然后用Welford在线算法计算。与真实方差(0.6667)比较误差。

2. **精度搜寻。** 找Python中最小正float32值 `x` 使 `1.0 + x == 1.0`。这是机器epsilon。验证它匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp边界情况。** 测试 `logsumexp_stable` 函数: (a) 所有值相等, (b) 一个值远大于其余, (c) 所有值很负(-1000)。验证给出正确结果而朴素版本失败。

4. **神经网络层梯度检查。** 实现单线性层 `y = Wx + b` 及其解析反向传播。用 `numerical_gradient` 验证3x2权重矩阵正确性。

5. **损失缩放实验。** 模拟float16训练: 创建范围[1e-9, 1e-3]随机梯度，转float16，测量多少变零。然后应用损失缩放(乘1024)，转float16，缩回，再次测量零比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值(inf, nan)的国际标准。每个现代CPU和GPU实现它。 |
| 机器epsilon | "精度极限" | 给定浮点格式中使1.0 + e != 1.0的最小值e。float32约1.19e-7。 |
| 灾难性抵消 | "减法精度损失" | 减去几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 溢出 | "数太大" | 结果超过最大可表示值变成inf。exp(89)溢出float32。 |
| 下溢 | "数太小" | 结果比最小可表示正数更接近零变成0.0。exp(-104)下溢float32。 |
| Log-sum-exp技巧 | "先减最大值" | 通过提出exp(max(x))计算log(sum(exp(x)))防止溢出下溢。用于softmax、交叉熵和对数概率数学。 |
| 稳定softmax | "不爆炸的softmax" | 指数化前减去max(logits)。数值相同结果，不可能溢出。 |
| 梯度检查 | "验证反向传播" | 比较反向传播解析梯度与有限差分数值梯度捕捉实现bug。 |
| 混合精度 | "float16前向，float32反向" | 用低精度浮点做速度关键操作，高精度浮点做数值敏感操作。典型加速2-3倍。 |
| 损失缩放 | "防止梯度下溢" | 反向前乘损失大常数使梯度在float16可表示范围，权重更新前除相同常数。 |
| bfloat16 | "Brain浮点" | Google16位格式，8指数位(与float32相同范围)和7尾数位(比float16精度更低)。训练首选。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量使范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数值" | 未定义操作(0/0, inf-inf, sqrt(-1))的特殊浮点值。传播到所有后续算术。 |
| Inf | "无穷" | 溢出或除零的特殊浮点值。可组合产生NaN(inf - inf, inf * 0)。 |
| 数值梯度 | "暴力导数" | 通过评估f(x+h)和f(x-h)除以2h近似导数。慢但验证可靠。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考, 密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA论文引入float16训练损失缩放
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch混合精度实践指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google为何为TPU选此格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差算法