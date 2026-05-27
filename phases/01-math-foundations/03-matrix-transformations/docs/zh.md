# 矩阵变换

> 矩阵是一台重塑空间的机器。了解它对每个点的作用，你就能理解整个变换。

**类型:** 构建
**语言:** Python, Julia
**前置要求:** 第1阶段，第01-02课（线性代数直觉、向量与矩阵运算）
**时间:** ~75分钟

## 学习目标

- 构建旋转、缩放、剪切和反射矩阵，并应用于2D和3D点
- 通过矩阵乘法组合多个变换，并验证顺序的重要性
- 从特征方程计算2x2矩阵的特征值和特征向量
- 解释为什么特征值决定PCA方向、RNN稳定性和谱聚类行为

## 问题背景

你读到PCA，看到"找到协方差矩阵的特征向量"。你读到模型稳定性，看到"检查所有特征值的模是否小于1"。你读到数据增强，看到"应用随机旋转"。在你理解矩阵对空间的几何作用之前，这些都没有意义。

矩阵不只是数字网格。它们是空间机器。旋转矩阵旋转点。缩放矩阵拉伸它们。剪切矩阵倾斜它们。神经网络对数据应用的每个变换都是这些操作之一，或者是它们的组合。这节课让这些操作具体化。

## 概念讲解

### 变换即矩阵

2D中的每个线性变换都可以写成2x2矩阵。矩阵准确地告诉你基向量[1, 0]和[0, 1]最终在哪里。其他一切都由此而来。

```mermaid
graph LR
    subgraph Before["标准基"]
        e1["e1 = [1, 0] (沿x轴)"]
        e2["e2 = [0, 1] (沿y轴)"]
    end
    subgraph Transform["矩阵 M"]
        M["M = 列是新的基向量"]
    end
    subgraph After["变换 M 后"]
        e1p["e1' = 新x基"]
        e2p["e2' = 新y基"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### 旋转

2D旋转角度theta保持距离和角度不变。它沿圆弧移动每个点。

```mermaid
graph LR
    subgraph Before["旋转前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["旋转45度"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["旋转后"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

在3D中，你围绕一个轴旋转。每个轴都有自己的旋转矩阵：

```
Rz(theta) = | cos  -sin  0 |     绕z轴旋转
            | sin   cos  0 |     (x-y平面旋转，z保持不变)
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   绕x轴旋转
            | 0  cos  -sin   |   (y-z平面旋转，x保持不变)
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     绕y轴旋转
            |   0   1   0  |     (x-z平面旋转，y保持不变)
            | -sin  0  cos |
```

### 缩放

缩放独立地沿每个轴拉伸或压缩。

```mermaid
graph LR
    subgraph Before["缩放前"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["缩放 sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["缩放后"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### 剪切

剪切沿一个轴倾斜，同时保持另一个轴固定。它将矩形变成平行四边形。

```mermaid
graph LR
    subgraph Before["剪切前"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["沿x剪切, k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["剪切后"]
        Ap["A(1, 0) 不变"]
        Bp["B'(1, 1) 偏移"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

剪切矩阵：
- `Shx = [[1, k], [0, 1]]` 将x沿k * y偏移
- `Shy = [[1, 0], [k, 1]]` 将y沿k * x偏移

### 反射

反射沿轴或线镜像点。

```mermaid
graph LR
    subgraph Before["反射前"]
        A["A(2, 1)"]
    end
    subgraph Reflect["沿y轴反射"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["反射后"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

反射矩阵：
- 沿y轴反射：`[[-1, 0], [0, 1]]`
- 沿x轴反射：`[[1, 0], [0, -1]]`

### 组合：链式变换

应用变换A然后B等同于相乘它们的矩阵：`result = B @ A @ point`。顺序很重要。先旋转后缩放与先缩放后旋转产生不同的结果。

```mermaid
graph LR
    subgraph Path1["先旋转90度，然后缩放(2, 0.5)"]
        P1["(1, 0)"] -->|"旋转90"| P2["(0, 1)"] -->|"缩放"| P3["(0, 0.5)"]
    end
```

组合：`S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["先缩放(2, 0.5)，然后旋转90度"]
        Q1["(1, 0)"] -->|"缩放"| Q2["(2, 0)"] -->|"旋转90"| Q3["(0, 2)"]
    end
```

组合：`R @ S = [[0, -0.5], [2, 0]]`

结果不同。矩阵乘法不满足交换律。

### 特征值和特征向量

大多数向量在被矩阵作用时会改变方向。特征向量是特殊的：矩阵只缩放它们，从不旋转它们。缩放因子就是特征值。

```
A @ v = lambda * v

v是特征向量（存活下来的方向）
lambda是特征值（它拉伸的程度）

示例: A = | 2  1 |
         | 1  2 |

特征向量 [1, 1] 对应特征值 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     （相同方向，缩放3倍）

特征向量 [1, -1] 对应特征值 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  （相同方向，不变）
```

矩阵在[1, 1]方向上拉伸空间3倍，在[1, -1]方向上保持不变。每个其他方向都是这两个方向的混合。

### 特征分解

如果一个矩阵有n个线性独立的特征向量，它可以被分解：

```
A = V @ D @ V^(-1)

V = 特征向量作为列的矩阵
D = 特征值的对角矩阵
V^(-1) = V的逆矩阵

这表示：旋转到特征向量坐标系，沿每个轴缩放，再旋转回来。
```

### 为什么特征值重要

**PCA。** 协方差矩阵的特征向量是主成分。特征值告诉你每个成分捕获了多少方差。按特征值排序，保留前k个，你就有了降维。

**稳定性。** 在循环网络和动态系统中，模大于1的特征值导致输出爆炸。模小于1导致它们消失。这就是梯度消失/爆炸问题，一句话总结。

**谱方法。** 图神经网络使用邻接矩阵的特征值。谱聚类使用拉普拉斯矩阵的特征值。特征向量揭示了图的结构。

### 行列式作为体积缩放因子

变换矩阵的行列式告诉你它在2D中缩放面积或在3D中缩放体积的程度。

```
det = 1:   面积保持（旋转）
det = 2:   面积加倍
det = 0:   空间压缩到更低维度（奇异）
det = -1:  面积保持但方向翻转（反射）

| det(旋转) | = 1        （总是）
| det(缩放 sx, sy) | = sx * sy
| det(剪切) | = 1           （面积保持）
| det(反射) | = -1       （方向翻转）
```

## 动手实践

### 第1步：从头开始的变换矩阵（Python）

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"将(1,0)旋转45度: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"将(1,1)缩放(2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"将(1,1)沿kx=1剪切: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"将(2,1)沿y轴反射: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### 第2步：变换的组合

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"先旋转90度再缩放: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"先缩放再旋转90度: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"相同? {result1 == result2}")
```

### 第3步：从头开始的特征值（2x2）

对于2x2矩阵`[[a, b], [c, d]]`，特征值解特征方程：`lambda^2 - (a+d)*lambda + (ad - bc) = 0`。

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"矩阵: {A}")
print(f"特征值: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### 第4步：行列式作为体积缩放因子

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(旋转45度) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(缩放2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(剪切kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(反射y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(奇异)     = {det_2x2(singular):.1f}")
print("奇异：列成比例，空间坍缩成一条线。")
```

## 实际应用

NumPy用优化的例程处理所有这些。

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"将(1,0)旋转45度: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"在旋转(45度)之后缩放(2,3): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量 (列):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\n特征分解 A = V @ D @ V^-1:")
print(f"原始:\n{B}")
print(f"重构:\n{reconstructed}")
```

### 用NumPy进行3D旋转

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D点: {point_3d}")
print(f"绕z轴旋转90度: {np.round(rotated_z, 4)}")
print(f"绕x轴旋转90度: {np.round(rotated_x, 4)}")
```

## 产出成果

这节课为PCA（第2阶段）和神经网络权重分析建立了几何基础。这里构建的特征值/特征向量代码与生产ML系统中降维、谱聚类和稳定性分析使用的算法相同。

## 练习题

1. 对单位正方形（角点在[0,0]、[1,0]、[1,1]、[0,1]）应用旋转、缩放和剪切。打印每种变换后的角点。验证旋转保持角点之间的距离。

2. 使用特征方程手工计算矩阵[[4, 2], [1, 3]]的特征值。然后用你的从头开始的函数和NumPy验证。

3. 创建三个变换的组合（旋转30度，按[1.5, 0.8]缩放，kx=0.3剪切）并应用于排列成圆的8个点。打印前后坐标。计算组合矩阵的行列式，并验证它等于各个行列式的乘积。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|----------|
| 旋转矩阵 | "旋转东西" | 一个正交矩阵，沿圆弧移动点，同时保持距离和角度。行列式总是1。 |
| 缩放矩阵 | "让东西变大" | 一个对角矩阵，独立地沿每个轴拉伸或压缩。行列式是缩放因子的乘积。 |
| 剪切矩阵 | "倾斜东西" | 一个矩阵，将一坐标比例地移动到另一坐标，将矩形变成平行四边形。行列式是1。 |
| 反射 | "镜像东西" | 一个矩阵，沿轴或平面翻转空间。行列式是-1。 |
| 组合 | "做两件事" | 相乘变换矩阵以链式操作。顺序很重要：B @ A表示先应用A，然后B。 |
| 特征向量 | "特殊方向" | 矩阵只缩放而不旋转的方向。变换的指纹。 |
| 特征值 | "它拉伸的程度" | 矩阵缩放其特征向量的标量因子。可以是负的（翻转）或复数（旋转）。 |
| 特征分解 | "分解矩阵" | 将矩阵写成V @ D @ V^(-1)，将其分离成基本的缩放方向和幅度。 |
| 行列式 | "从矩阵得到的单个数字" | 变换缩放2D面积或3D体积的因子。零意味着变换不可逆。 |
| 特征方程 | "特征值的来源" | det(A - lambda * I) = 0。根是特征值的多项式。 |

## 延伸阅读

- [3Blue1Brown: 线性变换](https://www.3blue1brown.com/lessons/linear-transformations) -- 矩阵如何重塑空间的视觉直觉
- [3Blue1Brown: 特征向量和特征值](https://www.3blue1brown.com/lessons/eigenvalues) -- 特征向量几何意义的最佳视觉解释
- [MIT 18.06 第21讲：特征值和特征向量](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- Gilbert Strang的经典讲解
