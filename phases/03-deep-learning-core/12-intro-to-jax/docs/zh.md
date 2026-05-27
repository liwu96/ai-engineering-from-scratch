# JAX简介

> PyTorch变异张量。TensorFlow建图。JAX编译纯函数。那最后者改你如何思深度学习。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段03课程01-10，基础NumPy
**时间:** ~90分钟

## 学习目标

- 用JAX函数API(jax.numpy、jax.grad、jax.jit、jax.vmap)写纯函数神经网络代码
- 解释PyTorch急切变异和JAX函数编译模型间关键设计差
- 应用jit编译和vmap向量化加速训练循环比朴素Python
- 在JAX训简网络对比显式状态管理与PyTorch面向对象方法

## 问题背景

你知如何在PyTorch建神经网络。你定义`nn.Module`，调`.backward()`，步优化器。它工作。数百万用它。

但PyTorch有约束焙入其DNA: 它急切追踪操作，一次一个，在Python。每`tensor + tensor`是分内核启。每训练步重解释同Python代码。这工作好直到你需跨2,048 TPU训5400亿参数模型。然后开销杀你。

Google DeepMind在JAX训Gemini。Anthropic在JAX训Claude。这些非小操作 -- 它们是地球上最大神经网络训练跑。它们选JAX因它把训练循环作可编译程序，非Python调用序。

JAX是NumPy带三超能力: 自动微分、JIT编译到XLA和自动向量化。你写函数处理一样本。JAX给你函数处理批、算梯度、编译到机器代码、跨多设备跑。全无改原函数。

## 概念讲解

### JAX哲学

JAX是函数框架。无类、无变异状态、无`.backward()`方法。替代:

| PyTorch | JAX |
|---------|-----|
| `nn.Module`类带状态 | 纯函数: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | JIT编译经XLA |
| `for x in batch:`手动循环 | `jax.vmap(f)`自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)`自动并行 |
| 变异`model.parameters()` | 不变pytree数组 |

这非风格偏好。它是编译器约束。JIT编译需纯函数 -- 同输入总产同输出，无副作用。那限制是使100x加速可能。

### jax.numpy: 熟悉面

JAX在加速器重实现NumPy API:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

同函数名。同广播规则。同切片语义。但数组住GPU/TPU，每操作编译器可追踪。

一关键差: JAX数组不变。无`a[0] = 5`。替代: `a = a.at[0].set(5)`。这觉笨一周，然后它启 -- 不变性是使变换如`grad`、`jit`和`vmap`可组。

### jax.grad: 函数Autodiff

PyTorch附梯度到张量(`.grad`)。JAX附梯度到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`取函数返算梯度新函数。无`.backward()`调用。无计算图存张量。梯度只是你可调、组或JIT编译另函数。

这任意组:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二导。三导。雅可比。海森。全组`grad`。PyTorch可做这(`torch.autograd.functional.hessian`)，但它栓上。JAX中，它是基。

约束: `grad`仅工作纯函数。无print语句内(它们跑追踪时，非执行)。无外状态变异。无随机数生无显式键管理。

### jit: 编译到XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

首调用，JAX追踪函数 -- 它录哪些操作发生，不执行它们。然后它交那追踪给XLA(加速线性代数)，GoogleTPU和GPU编译器。XLA熔操作，消冗余内存拷，生优化机器代码。

后续调用跳Python全。编译代码在加速器以C++速度跑。

当JIT帮:
- 训练步(同计算重复数千次)
- 推理(同模型，不同输入)
- 任何函数调用超一次带相似形状输入

当JIT害:
- 带Python控制流函数依赖值(`if x > 0`其中x是追踪数组)
- 一次计算(编译开销超运行时)
- 调试(追踪藏实际执行)

控制流限制是真实。`jax.lax.cond`替`if/else`。`jax.lax.scan`替`for`循环。这些非可选 -- 它们是编译代价。

### vmap: 自动向量化

你写函数处理一样本:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`举它处理批:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`意: 不批过`params`(共享)，批过`x`轴0。无手动`for`循环。无重塑。无批维度穿。JAX算出批维度向量化全计算。

这非语法糖。`vmap`生熔向量化代码跑10-100x快Python循环。且它组`jit`和`grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

每例梯度。一行。这在PyTorch近乎不可能无hack。

### pmap: 跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`跨全可用设备(GPU/TPU)复制函数分批。函数内，`jax.lax.pmean`和`jax.lax.psum`跨设备同步梯度。

Google用`pmap`(及其继`shard_map`)跨数千TPU v5e芯片训Gemini。编程模型: 写单设备版，裹`pmap`，完。

### Pytrees: 通用数据结构

JAX操作"pytrees" -- 列、元组、字典和数组组。你模型参数是pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每JAX变换 -- `grad`、`jit`、`vmap` -- 知如何遍pytrees。`jax.tree.map(f, tree)`应用`f`到每叶。这是优化器如何一次更新全参数:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

无`.parameters()`方法。无参数注册。树结构是模型。

### 函数vs面向对象

PyTorch存状态内对象:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX用带显式状态纯函数:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params传入。无存。无变异。这使每函数可测、可组、可编译。它也意味你管params自己 -- 或用库如Flax或Equinox。

### JAX生态

JAX给你原语。库给你易用:

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | `nn.Module`带显式状态 |
| **Equinox** (Patrick Kidger) | 神经网络层 | Pytree基，Pythonic |
| **Optax** (DeepMind) | 优化器+LR调度 | 可组梯度变换 |
| **Orbax** (Google) | 检查点 | 存/恢复pytrees |
| **CLU** (Google) | 指标+日志 | 训练循环工具 |

Optax是标准优化器库。它分梯度变换(Adam、SGD、裁剪)从参数更新，使组简单:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU支持 | 一级(Google建两者) | 社区维护(torch_xla) |
| GPU支持 | 好(CUDA经XLA) | 最佳(原生CUDA) |
| 调试 | 难(追踪+编译) | 易(急切，逐行) |
| 生态 | 研聚焦(Flax, Equinox) | 巨大(HuggingFace, torchvision, 等) |
| 招聘 | 小众(Google/DeepMind/Anthropic) | 主流(处处) |
| 大规模训练 | 优(XLA, pmap, mesh) | 好(FSDP, DeepSpeed) |
| 原型速度 | 慢(函数开销) | 快(变异和走) |
| 生产推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 谁用它 | DeepMind(Gemini), Anthropic(Claude) | Meta(Llama), OpenAI(GPT), Stability AI |

诚答案: 用PyTorch除非你有特定理由用JAX。那些理由是 -- TPU访问、需每例梯度、大规模多设备训练、或在Google/DeepMind/Anthropic工作。

### JAX中随机数

JAX无全局随机状态。每随机操作需显式PRNG键:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这初觉烦。但它保跨设备和编译可复现 -- PyTorch `torch.manual_seed`在多GPU设置不能保的性质。

## 构建

### 步骤1: 设置和数据

我们将在JAX和Optax训MNIST上3层MLP。784输入，两隐藏层256和128神经元，10输出类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤2: 初始化参数

无类。仅返pytree函数:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He初始化，手动做。三PRNG键从一种子分。每权重是嵌字典中不变数组。

### 步骤3: 前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。Params入，预测出。无`self`，无存状态。`loss_fn`从零算交叉熵 -- softmax、log、负均值。

### 步骤4: JIT编译训练步

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`一次返损失值和梯度。`@jax.jit`装饰编译两函数到XLA。首调用后，每训练步跑不触Python。

### 步骤5: 训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | 损失: {epoch_loss / n_batches:.4f} | "
          f"训精度: {train_acc:.4f} | 测精度: {test_acc:.4f}")
```

10 epochs。~97%测试精度。首epoch慢(JIT编译)。Epochs 2-10快。

注意缺失: 无`.zero_grad()`、无`.backward()`、无`.step()`。全更新是一组函数调用。梯度算、Adam变换、应用到参数 -- 全在`train_step`内。

## 使用

### Flax: Google标准

Flax是最常见JAX神经网络库。它加回`nn.Module`，但带显式状态管理:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

同PyTorch结构，但`params`分模型。`model.init()`创params。`model.apply(params, x)`跑前向传播。模型对象无状态。

### Equinox: Pythonic替代

Equinox(由Patrick Kidger)表示模型作pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身是pytree。无`.apply()`需。参数只是模型叶。这更近JAX思。

### Optax: 可组优化器

Optax解耦梯度变换从更新:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减 -- 全组变换链。每变换看梯度，修它，传下。无单体优化器类。

## 交付成果

**安装:**

```bash
pip install jax jaxlib optax flax
```

GPU支持:

```bash
pip install jax[cuda12]
```

TPU(Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能陷阱:**

- 首JIT调用慢(编译)。基准前预热。
- 避JIT内Python循环JAX数组。用`jax.lax.scan`或`jax.lax.fori_loop`。
- `jax.debug.print()`JIT内工作。常`print()`不。
- 用`jax.profiler`或TensorBoard分析。XLA编译可藏瓶颈。
- JAX默认预分配75% GPU内存。设`XLA_PYTHON_CLIENT_PREALLOCATE=false`禁。

**检查点:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课程产:**
- `outputs/prompt-jax-optimizer.md` -- 选对JAX优化器配置提示词
- `outputs/skill-jax-patterns.md` -- JAX函数模式技能覆盖

## 练习题

1. 加dropout到MLP。JAX中，dropout需PRNG键 -- 穿键过前向传播为每dropout层分。比测试精度有无。

2. 用`jax.vmap`算32 MNIST图像批每例梯度。算每例梯度范数。哪例有最大梯度，为何？

3. 替换手动前向函数为泛`mlp_forward(params, x)`为任意层数工作。用`jax.tree.leaves`自动定深度。

4. 基准训练步有无`@jax.jit`。时每100步。你硬件上加速多大？首调用编译开销多大？

5. 组梯度裁剪实现`optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`。训练有无裁剪。绘训练梯度范数看效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| XLA | "使JAX快东西" | 加速线性代数 -- 融操作生优化GPU/TPU内核从计算图编译器 |
| JIT | "即时编译" | JAX首调用追踪函数，编译到XLA，然后在后续调用跑编译版 |
| 纯函数 | "无副作用" | 输出仅依赖输入函数 -- 无全局状态，无变异，无无显式键随机 |
| vmap | "自动批" | 变换处理一样本函数到处理批，无重写 |
| pmap | "自动并行" | 跨多设备复制函数分输入批 |
| Pytree | "数组嵌字典" | 列、元组、字典和数组组JAX可遍和变换 |
| 追踪 | "录计算" | JAX用抽象值执行函数建计算图，不算真实结果 |
| 函数autodiff | "函数grad" | 组函数算导数，非附梯度存到张量 |
| Optax | "JAX优化器库" | 可组梯度变换库 -- Adam、SGD、裁剪、调度 -- 链一起 |
| Flax | "JAX nn.Module" | Google JAX神经网络库，加层抽象同时保状态显式 |

## 延伸阅读

- JAX文档: https://jax.readthedocs.io/ -- 官方文档，grad、jit和vmap极教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury等, 2018) -- 原论文解释设计哲学
- Flax文档: https://flax.readthedocs.io/ -- Google JAX神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax Pythonic替代
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 标准优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) -- JAX陷阱和模式实用指南，T5作者之一