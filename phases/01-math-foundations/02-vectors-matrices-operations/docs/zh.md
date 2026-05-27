# 向量、矩阵与操作

> 每个神经网络本质上都是带额外步骤的矩阵乘法。

**类型：** 构建
**语言：** Python, Julia
**前置要求：** 第 1 阶段，第 01 课（线性代数直觉）
**时间：** 约 60 分钟

## 学习目标

- 构建带逐元素操作、矩阵乘法、转置、行列式和逆的矩阵类
- 区分逐元素乘法与矩阵乘法，并解释何时应用每种
- 仅使用从零构建的矩阵类实现单个稠密神经网络层（`relu(W @ x + b)`）
- 解释广播规则和神经网络框架中的偏置加法工作原理

## 问题背景

你想构建一个神经网络。你阅读代码并看到：

```
output = activation(weights @ input + bias)
```

那个 `@` 是矩阵乘法。`weights` 是矩阵。`input` 是向量。如果你不知道这些操作做什么，这一行就是魔法。如果你知道，它就是三层操作中的整个前向传播。

你的模型处理的每张图片都是像素值矩阵。每个词嵌入都是向量。每个神经网络的每一层都是矩阵变换。不理解矩阵操作就无法构建 AI 系统，就像不理解变量就无法编写代码。

这节课从零构建这种熟练度。

## 概念讲解

### 向量：有序数字列表

向量是带方向和大小的数字列表。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        -- 2D 向量
w = [1, 0, -2]    -- 3D 向量
```

2D 向量 `[3, 4]` 指向平面上坐标 (3, 4)。它的长度（大小）是 5（3-4-5 三角形）。

### 矩阵：数字网格

矩阵是 2D 网格。行和列。m x n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2x3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量变换为输出向量。有 784 输入和 128 输出的层使用 128x784 的权重矩阵。

### 为什么形状重要

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  权重       输入       输出

内部维度：784 = 784  -- 有效
```

如果你在 PyTorch 中得到形状不匹配错误，这就是原因。

### 操作映射

| 操作 | 做什么 | 神经网络用途 |
|------|--------|--------------|
| 加法 | 逐元素组合 | 向输出添加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法 | 变换向量 | 层前向传播 |
| 转置 | 翻转行列 | 反向传播 |
| 行列式 | 单数字摘要 | 检查可逆性 |
| 逆 | 撤销变换 | 求解线性系统 |
| 单位矩阵 | 什么都不做矩阵 | 初始化、残差连接 |

### 逐元素 vs 矩阵乘法

这种区别不断让初学者困惑。

逐元素：相乘匹配位置。两个矩阵必须有相同形状。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行和列的点积。内部维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同操作，不同结果，不同规则。

### 广播

当你向输出矩阵添加偏置向量时，形状不匹配。广播拉伸较小数组以适应。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播拉伸向量跨行：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架自动做这个。理解它防止形状看起来错误但代码运行时的困惑。

## 动手实践

### 步骤 1：向量类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步骤 2：带核心操作的矩阵类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("矩阵是奇异的，不存在逆")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步骤 3：看效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步骤 4：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"输入形状: {inputs.shape}")
print(f"权重形状: {weights.shape}")
print(f"输出形状: {output.shape}")
print(f"输出: {output.data}")
```

这是一个稠密层：`output = relu(W @ x + b)`。每个神经网络中的每个稠密层都完全做这个。

## 实际应用

NumPy 用更少代码和数量级更快的速度做上面所有事情。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (逐元素) =\n", A * B)
print("A @ B (矩阵乘法) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"输出:\n{output}")
```

Python 中的 `@` 操作符调用 `__matmul__`。NumPy 用 C 和 Fortran 编写的优化 BLAS 例程实现它。相同数学，100 倍更快。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动跨两行广播 1D 偏置。这就是每个神经网络框架中偏置加法的工作原理。

## 产出成果

本节课生成一个通过几何直觉教授矩阵操作的提示词。见 `outputs/prompt-matrix-operations.md`。

这里构建的矩阵类是我们在第 3 阶段第 10 课构建的迷你神经网络框架的基础。

## 练习题

1. **验证逆。** 相乘 `A @ A.inverse_2x2()` 并确认得到单位矩阵。用三个不同 2x2 矩阵尝试。行列式为零时会发生什么？

2. **实现 3x3 逆。** 扩展矩阵类，用伴随矩阵法计算 3x3 矩阵的逆。对照 NumPy 的 `np.linalg.inv` 测试。

3. **构建两层网络。** 仅用你的矩阵类（不用 NumPy），创建两层神经网络：输入 (3) -> 隐藏 (4) -> 输出 (2)。初始化随机权重，运行前向传播，验证所有形状正确。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 向量 | "一个箭头" | 有序数字列表。在 AI 中：高维空间中的点。 |
| 矩阵 | "数字表格" | 线性变换。它将向量从一个空间映射到另一个。 |
| 矩阵乘法 | "只是相乘数字" | 第一个矩阵的每行与第二个矩阵的每列之间的点积。顺序重要。 |
| 转置 | "翻转它" | 交换行列。将 m x n 矩阵变成 n x m。反向传播中关键。 |
| 行列式 | "矩阵的某个数字" | 测量矩阵如何缩放面积（2D）或体积（3D）。零意味着变换压碎一个维度。 |
| 逆 | "撤销矩阵" | 反转变换的矩阵。仅当行列式不为零时存在。 |
| 单位矩阵 | "无聊的矩阵" | 矩阵等价的乘 1。用于残差连接（ResNets）。 |
| 广播 | "魔法形状修复" | 通过沿缺失维度重复来拉伸较小数组以匹配较大数组。 |
| 逐元素 | "常规乘法" | 相乘匹配位置。两个数组必须有相同形状（或可广播）。 |

## 延伸阅读

- [3Blue1Brown：线性代数本质](https://www.3blue1brown.com/topics/linear-algebra) - 这里涵盖的每个操作的可视直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [斯坦福 CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 特定线性代数的简明参考
