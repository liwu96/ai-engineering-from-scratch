# 投机解码——草稿、验证、重复

> 自回归解码串行。每词元等前词元。投机解码断链:便宜模型草稿N词元,昂贵模型一次前向验证全部N。当草稿正确你付一大前向得N生成。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段7课程07(GPT因果LM)、阶段7课程12(KV Cache & Flash注意力机制)
**时间:** ~60分钟

## 问题背景

70B LLM H100采样一词元取~30 ms。3B草稿模型取~3 ms。若让3B草稿5词元前,后跑70B*一次*验证全部5,总共`5×3 + 30 = 45 ms`最多5接受词元——vs直线生成`5×30 = 150 ms`。此投机解码全pitch:trade少量额外GPU内存(草稿模型)换2–4×更低解码延迟。

技巧需保分布。投机采样,Leviathan等(2023)和Chen等同时引入,保证输出序列**identically distributed**大模型自己产。无质量折衷。仅更快。

四家族草稿-验证对主导2026推理:

1. **朴素投机(Leviathan 2023)。**分离草稿模型(如Llama 3 1B)+验证器(如Llama 3 70B)。
2. **Medusa(Cai 2024)。**验证器上多解码头并预测位置`t+1..t+k`。无分离草稿模型。
3. **EAGLE家族(Li 2024, 2025)。**轻量草稿复用验证器隐藏状态;比朴素接受率近;典型3–4×。
4. **Lookahead解码(Fu 2024)。**Jacobi迭代;完全无草稿模型。自投机。小众但无依赖。

2026每个生产推理栈默认投机解码。vLLM、TensorRT-LLM、SGLang、llama.cpp全支持至少朴素+EAGLE-2。

## 概念讲解

### 核心算法

给定验证器`M_q`和更便宜草稿`M_p`:

1. 让`x_1..x_k`是已解码前缀。
2. **草稿**:用`M_p`自回归提议`d_{k+1}, d_{k+2}, ..., d_{k+N}`配草稿概率`p_1..p_N`。
3. **并行验证**:对`x_1..x_k, d_{k+1}, ..., d_{k+N}`跑`M_q`一次,得位置`k+1..k+N+1`验证器概率`q_1..q_{N+1}`。
4. **左到右接受/拒绝每草稿词元**:每`i`,按概率`min(1, q_i(d_i) / p_i(d_i))`接受。
5. 位置`j`首拒:从"残差"分布`(q_j - p_j)_+`归一化采样`t_j`。j后所有草稿丢弃。
6. 接受全部N:从`q_{N+1}`采样一额外词元`t_{N+1}`(免费奖励词元)。

残差分布技巧是保持输出分布精确如`M_q`从头采样的数学洞察。

### 何定加速

让`α` = 每草稿词元期望接受率。让`c` = 草稿/验证器成本比例。每步:

- 朴素生成每词元一大模型调用。
- 投机高α时每`(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)`词元一大模型调用。

`α = 0.75`和`N = 5`典型拇指规则:3×少大模型调用。草稿成本5×便宜。总墙钟降~2.5×。

**α取决于:**

- 草稿近似验证器多好。同家族/同训数据显著提升α。
- 解码策略。贪心草稿对贪心验证器:高α。温度采样:难匹配;接受降。
- 任务类型。代码和结构输出接受多(可预测);自由形式创意写作接受少。

### Medusa——无草稿模型草稿

Medusa换草稿模型为验证器上额外输出头。位置`t`:

```
共享主干 → 隐藏 h_t
    ├── head_0: 预测t+1词元(标准LM头)
    ├── head_1: 预测t+2词元
    ├── head_2: 预测t+3词元
    ├── head_3: 预测t+4词元
```

每头输出自己logits。推理从每头采样得候选序列,后配树注意力方案一次前向验证考虑所有候选续同时。

优点:无第二模型。缺点:加可训参数;需监督微调阶段(~1B词元);接受率比好草稿朴素投机略低。

### EAGLE——复用隐藏状态更好草稿

EAGLE-1/2/3(Li等,2024–2025)让草稿模型微小transformer(典型1层)摄入验证器最后层隐藏状态。因草稿见验证器特征表示,其预测与验证器输出分布强相关。接受率从~0.6(朴素)爬到0.85+。

EAGLE-3(2025)加候选续树搜索。vLLM和SGLang默认EAGLE-2/3作Llama 3/4和Qwen 3投机路径。

### KV cache舞蹈

验证一次前向喂N草稿词元进验证器。此扩验证器KV cache N入口。若些草稿被拒,需回滚cache到接受前缀长。

生产实现(vLLM `--speculative-model`, TensorRT-LLM LookaheadDecoder)配scratch KV缓冲处理。先写,接受时commit。非概念难,但繁琐。

## 动手实践

见`code/main.py`。实现核心投机采样算法(拒步+残差分布)配:

- "大模型"是手工编码分布确定性softmax(可解析验证接受数学)。
- "草稿模型"是大模型扰动。
- 接受/拒循环产与直接采样同边缘分布。

### Step 1: 拒步

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`是均匀随机数。`q_prob`是验证器草稿词元概率。`p_prob`是草稿模型概率。Leviathan定理是此Bernoulli决策,后拒时从残差采样,精保安验证器分布。

### Step 2: 残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

元素减`p`从`q`,负值clamp零,归一化。任何拒从此采样。

### Step 3: 一投机步

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五接受→一奖励→一验证器pass产六词元。

### Step 4: 测接受率

变草稿质量水平跑10,000投机步。绘接受率vs草稿和验证器分布KL散度。应见干净单调关系。

### Step 5: 验证分布等价

经验:投机循环产词元直方图应匹配直接从验证器采样直方图。此Leviathan定理实践。chi-square测试确认采样误差内。

## 实际应用

生产:

```bash
# vLLM配EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM配朴素草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 2026中最快Medusa路径。`faster-whisper`包Whisper-large配小草稿投机解码。

**选草稿:**

| 策略 | 何时选 | 加速 |
|------|--------|------|
| 朴素草稿(1B/3B Llama家族) | 快原型,无训练 | 1.8–2.3× |
| Medusa头 | 可微调验证器 | 2–3× |
| EAGLE-2/3 | 生产,最大速度 | 3–4× |
| Lookahead | 无草稿,无训练,无额外参数 | 1.3–1.6× |

**何时不投机解码:**

- 单序列生成1–5词元。开销主导。
- 极创意/高温采样(α降)。
- 内存受限部署(草稿模型加VRAM)。

## 产出成果

见`outputs/skill-spec-decode-picker.md`。技能为新推理工作负载选投机解码策略(朴素/Medusa/EAGLE/lookahead)和调参数(N, 草稿温度)。

## 练习题

1. **简单。**运行`code/main.py`。确认50,000词元投机词元分布匹验证器直接采样分布chi-square p > 0.05内。
2. **中等。**绘`α = 0.5, 0.7, 0.85`加速(每大模型前向词元)作N函数。识别每α最优N。(提示:每验证调用期望词元 = `(1 - α^{N+1}) / (1 - α)`。)
3. **困难。**实现微小Medusa:取课程14毕业GPT,加3额外LM头预测位置t+2, t+3, t+4。tinyshakespeare配联合多头损失训。比接受率vs截断同模型朴素草稿。
4. **困难。**实现回滚:从10词元前缀KV cache起,喂5草稿词元,模拟位置3拒。验证cache读正确匹"前缀+首2接受草稿"下迭代。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 草稿模型 | "便宜那个" | 提候选词元更小模型;常比验证器便宜10–50×。 |
| 验证器 | "大那个" | 保分布目标模型;每投机步跑一次。 |
| 接受率(α) | "草稿多常对" | 每词元验证器接受草稿概率。典型0.7–0.9。 |
| 残差分布 | "拒后备" | `(q - p)_+`归一化;拒时采样保验证器分布。 |
| 奖励词元 | "免费那个" | 当全部N草稿接受,从验证器下一步分布多采样一。 |
| Medusa | "无草稿投机" | 验证器上多LM头并预测位置t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 条件化验证器最后层隐藏状态微小transformer草稿。 |
| Lookahead解码 | "Jacobi迭代" | 配定点迭代自投机;无草稿模型。 |
| 树注意力 | "一次验证多候选" | 考虑多草稿续分支验证同时。 |
| KV回滚 | "撤销拒草稿" | Scratch KV缓冲;接受commit,拒丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias(2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)——核心算法和等价定理。
- [Chen等(2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)——同时引入;干净Bernoulli拒证明。
- [Cai等(2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)——Medusa论文;树注意力验证。
- [Li等(2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)——EAGLE-1;隐藏状态条件草稿。
- [Li等(2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)——EAGLE-2;动态树深。
- [Li等(2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)——EAGLE-3。
- [Fu等(2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)——lookahead,无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html)——规范生产参考配四策略全接。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE)——EAGLE-1/2/3参考代码。