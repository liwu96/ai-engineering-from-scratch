# 评估——FID、CLIP分数、人类偏好

> 每生成模型排行榜引FID、CLIP分数、和人类偏好竞技场胜率。每数有失败模式坚定研究员可博弈。如不知失败模式,不能分真改进和博弈跑。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段8课程01(分类)、阶段2课程04(评估指标)
**时间:** ~45分钟

## 问题背景

生成模型判*样本质量*和*条件跟随*。无闭式度量。模型须渲染10,000图像;某物须赋数;须信数跨模型族、分辨率、架构。三指标存2014-2026考验:

- **FID(Fréchet Inception距离)。**Inception网络特征空间两分布——真实和生成——间距离。低好。
- **CLIP分数。**生成图像CLIP-图像嵌入和提示词CLIP-文本嵌入余弦相似度。高好。度量提示词跟随。
- **人类偏好。**同提示词两模型对抗,人(或GPT-4级模型)选优,聚合Elo分数。

也会见:IS(Inception分数,多退役)、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每修正前一失败。

## 概念讲解

![FID、CLIP、偏好:三轴,不同失败模式](../assets/evaluation.svg)

### FID——样本质量

Heusel等(2017)。步:

1. N真图像和N生成图像提Inception-v3特征(2048维)。
2. 每池配Gaussian:算均值`μ_r, μ_g`和协方差`Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释:特征空间两多元Gaussian间Fréchet距离。低=更相似分布。

失败模式:
- **小N偏。**FID是特征分布上均方——小N低估协方差,给假低FID。总用N ≥ 10,000。
- **依赖Inception。**Inception-v3训ImageNet。远ImageNet域(脸、艺、文图像)产无意义FID。用域特特征提取器。
- **博弈。**过拟合Inception先验给低FID无视觉质量改进。配CMMD击败(下)。

### CLIP分数——提示词跟随

Radford等(2021)。生成图像+提示词:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30k生成图像平均→模型间可比标量。

失败模式:
- **CLIP自盲点。**CLIP组合推理弱("红立方上蓝球"常败)。模型CLIP分数好但真不跟复杂提示词。
- **短提示词偏。**短提示词野多CLIP-图像匹。长提示词机械CLIP分数低。
- **提示词博弈。**提示词含"高质量、4k、杰作"CLIP分数涨无改进图文绑定。

CMMD(Jayasumana等, 2024)修些:用CLIP特征而非Inception,最大均值差异而非Fréchet。更好检测微妙质量差。

### 人类偏好——真值

选提示词池。模型A和B生成。示配人(或强LLM判)。聚合胜Elo或Bradley-Terry分数。基准:

- **PartiPrompts(Google)**: 1,600多样提示词,12类。
- **HPSv2**: 107k人标注,广用自动代理。
- **ImageReward**: 137k提示词-图像偏好配,MIT许可。
- **PickScore**: Pick-a-Pic 2.6M偏好训。
- **Chatbot-Arena式图像竞技场**: https://imagearena.ai/ 等。

失败模式:
- **判方差。**非专家偏好异专家。用两者。
- **提示词分布。**精选提示词偏一族。总文档。
- **LLM判奖励博弈。**GPT-4判被骗漂亮但错输出。人三角验证。

## 合用

生产评估报告应含:

1. 10-30k样本vs留出真分布FID(样本质量)。
2. 同样本vs提示词CLIP分数/CMMD(跟随)。
3. vs前模型盲竞技场胜率(总偏好)。
4. 失败模式分析:50随机采样输出,标已知问题(手解剖、文渲染、一致对象数)。

任何单指标是谎。三佐证指标+定性审是主张。

## 动手实践

`code/main.py`实现合成"特征向量"上FID、CLIP分数式、Elo聚合(用4维向量作Inception特征替)。见:

- 小N和大N上FID算——偏。
- 特征池间余弦相似度"CLIP分数"。
- 合成偏好流Elo更新规则。

### Step 1: 四行FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP式余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 FID。**启发N<10k不可靠。报告低N FID论文博弈。
- **跨分辨率比FID。**Inception 299×299 resize改特征分布。仅匹分辨率比。
- **报告一种子。**最少跑3种。报告标准差。
- **负提示词CLIP分数涨。**些管道过拟合提示词涨CLIP。查视觉饱和。
- **提示词重叠Elo偏。**如两模型训时见基准提示词,Elo无意义。用留出提示词集。
- **人评付费众偏。**Prolific、MTurk标注者偏年轻/技术友好。混招募艺/设计专家。

## 实际应用

2026生产评估协议:

| 柱 | 最小 | 推荐 |
|------|------|------|
| 样本质量 | 10k vs留出真FID | + 5k CMMD + 子类FID |
| 提示词跟随 | 30k CLIP分数 | + HPSv2 + ImageReward + VQA式问答 |
| 偏好 | 200 vs基线盲配 | + 2000配人 + LLM判 + Chatbot Arena |
| 失败分析 | 50手标 | 500手标 + 自动安全分类器 |

报告四柱=主张。任何单=营销。

## 产出成果

存`outputs/skill-eval-report.md`。技能取新模型检查点+基线输出全评估计划:样本大小、指标、失败模式探、签准。

## 练习题

1. **简单。**跑`code/main.py`。同合成分布比N=100 vs N=1000 FID。报告偏幅度。
2. **中等。**合成CLIP式特征实现CMMD(见Jayasumana等, 2024公式)。比质量差敏感度vs FID。
3. **困难。**复HPSv2设置:Pick-a-Pic子集取1000图像-提示词配,偏好微调小CL基评分器,测留出集一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| FID | "Fréchet Inception距离" | 真vs生成Inception特征Gaussian拟合Fréchet距离。 |
| CLIP分数 | "文图相似度" | CLIP图像和文本嵌入余弦相似度。 |
| CMMD | "FID替代" | CLIP特征MMD;更少偏,无Gaussian假设。 |
| IS | "Inception分数" | Exp KL(p(y|x) || p(y));现模型关差,退役。 |
| HPSv2/ImageReward/PickScore | "学偏好代理" | 人偏好训小模型;用作自动判。 |
| Elo | "象棋分" | 配对胜Bradley-Terry聚合。 |
| PartiPrompts | "基准提示词集" | 12类1,600 Google策管提示词。 |
| FD-DINO | "自监替代" | DINOv2特征FD;ImageNet外域更好。 |

## 生产注:评估也是推理工作负载

10k样本跑FID意味生10k图像。L4上50步SDXL基1024²单请求推理~11小时。评估预算实,框架恰离线推理场景(最大化吞吐,忽略TTFT):

- **硬批次,忘延迟。**离线评估=内存匹最大大小静态批次。80GB H100上`pipe(...).images`配`num_images_per_prompt=8`比单请求墙钟快4-6×。
- **缓存真特征。**真参考集上Inception(FID)或CLIP(CLIP分数、CMMD)特征提取跑*一次*,存`.npz`。每评估不重算。

CI/回归门:每PR500样本子集跑FID+CLIP分数(~30分);每晚全10k FID+HPSv2+Elo。

## 延伸阅读

- [Heusel等(2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500)——FID论文。
- [Jayasumana等(2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603)——CMMD。
- [Radford等(2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)——CLIP。
- [Wu等(2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341)——HPSv2。
- [Xu等(2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977)——ImageReward。
- [Yu等(2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789)——PartiPrompts。
- [Stein等(2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675)——失败模式综述。