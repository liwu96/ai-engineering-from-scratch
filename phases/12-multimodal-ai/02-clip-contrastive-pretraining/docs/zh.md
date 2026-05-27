# CLIP和对比视觉语言预训练

> OpenAI CLIP (2021)证单念大至撑后五年:于同向量空间对齐图像编码器和文编码器仅用噪网络图caption对和对比loss。零监督标签。400M对。结果embedding空间零分类、图文检索和接入每2026 VLM作其视觉塔。SigLIP 2 (2025)替softmax为sigmoid并于低成本scale超CLIP。本课走InfoNCE至sigmoid pairwise loss数学并用stdlib Python建训练步。

**类型:** 构建
**语言:** Python(stdlib,InfoNCE + sigmoid loss实)
**前置要求:** 阶段12课程01(ViT patch),阶段7(Transformers)
**时间:** ~180分钟

## 学习目标

- 从互信息推导InfoNCE loss并实数值稳定向量化版。
- 释何sigmoid pairwise loss(SigLIP)scale至batch 32768+无softmax需all-gather开销。
- 通过构文模板(`a photo of a {class}`)和取cosine相似argmax跑零ImageNet分类。
- CLIP/SigLIP预训练给你四杠杆:batch size、temperature、prompt template、数据质量。

## 问题背景

CLIP前视觉是监督。集标签数据集(ImageNet:1.2M图,1000类),训CNN,发它。标签贵、标签偏标注者可共识、标签不迁新任务无微调。

图caption网有十亿plus松标对免费。金毛猎犬公园图带alt text "my dog Max in the park"带监督信号—文述图。问题:可你转此有用训?

CLIP答:视图caption对作配任务。给N图和N caption batch,学配每图至己caption对N-1 distractors。监督是"此两物属一起;此N-1不。"无类标签。无人标注。仅对比loss。

结果embedding空间做多于CLIP训。ImageNet零工因"a photo of a cat"embed近从未显标猫图。这是生每2026 VLM赌。

## 概念讲解

### 双编码器

CLIP有两塔:

- 图像编码器`f`:ViT或ResNet,每图出D维向量。
- 文编码器`g`:小transformer,每caption出D维向量。

两塔normalize输出至单位长。相似是`cos(f(x), g(y)) = f(x)^T g(y)`因两者单位范。

N(图,caption)对batch,建形`(N, N)`相似矩阵`S`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

`tau`是可学temperature(CLIP初至0.07;log空间学)。

### InfoNCE loss

CLIP用行和列对称交叉熵:

```
loss_i2t = CE(S, labels=identity)     # 每图正配是己caption
loss_t2i = CE(S^T, labels=identity)   # 每caption正配是己图
loss = (loss_i2t + loss_t2i) / 2
```

这是InfoNCE。CE softmax强每图配其caption多batch每他caption。"负例"是batch每他项。大batch = 多负例 = 强信号。CLIP于batch 32k训;规模重要。

### Temperature

`tau`控softmax锐。低tau → 锐分布,hard negative mining效。高tau → 柔,所有样本贡。CLIP学log(1/tau),clip防坍。SigLIP 2定初tau并用可学bias替。

### 何sigmoid scale更好(SigLIP)

Softmax需全相似矩阵同步。分布训练你必须all-gather每embedding至每副本,后softmax。这于world size通信二次方。

SigLIP替softmax为逐元素sigmoid:每对`(i, j)`,loss是"是配对?"二元分类。正类标签是对角,余负。loss是:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1`若`i == j`,否则0。每对loss独立。无all-gather需。每GPU算其本块和。SigLIP 2于batch 32k-512k便宜scale,CLIP需比例更多通信。

### 零分类

给N类名,每类建文模板:

```
"a photo of a {class}"
```

用文编码器embed每模板。用图像编码器embed你图。Argmax cosine相似 = 预测类。无目标类训。

提示模板重要。CLIP原论文每类用80模板(plain、artistic、photo、painting等)并平均embedding。+3 ImageNet点。现代用法常择一或两模板。

### 线性probe和微调

零是基线。线性probe(冻CLIP特征上训一线性层为目标类)于域内任务超零。全微调域内超线性probe但可伤零迁。三 regime三权衡。

### SigLIP 2:NaFlex和密特征

SigLIP 2 (2025)加:
- NaFlex:单模型理变纵横比和分辨率。
- 更好密特征为分割和深度估计,目标VLM冻backbone用。
- 多语:训于100+语言CLIP仅英语。
- 1B参数scale CLIP顶于400M。

于2026开VLM,SigLIP 2 SO400m/14是默视觉塔。CLIP仍纯图文检索默,其LAION-2B训分布匹你查询模式。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN (Google, 2021):CLIP同念,1.8B对scale,90%噪。证噪数据scale。OpenCLIP (LAION):LAION-400M / 2B上CLIP开复现,多scale,去开checkpoint。EVA-CLIP:掩图像建模初始化;VLM强backbone。BASIC:Google CLIP+ALIGN混。同族,异数据和调。

### 零天花板

CLIP类模型顶于约76% ImageNet零(CLIP-G, OpenCLIP-G)。超需或大得多数据(SigLIP 2得80%+)或架构改(监督头,多参数)。benchmark饱和;真值是下游VLM消费embedding空间。

## 使用

`code/main.py`实:

1. Toy双编码器(hash基图像特征,文char特征)使你见InfoNCE形无numpy。
2. 纯Python InfoNCE loss(log-sum-exp数值稳定)。
3. Sigmoid pairwise loss为比。
4. 零分类例程:算对一组文提示cosine相似,argmax预测。

跑它并观loss曲线。绝对数toy;形匹真CLIP训练器发。

## 交付成果

本课产`outputs/skill-clip-zero-shot.md`。给一组图像(经路径)和目标类列表,它用CLIP模板建文提示,用声checkpoint(如`openai/clip-vit-large-patch14`)embed两边,返top-1/top-5预测带相似分。技能拒不在提示列表类声称。

## 练习题

1. 手实batch 4对InfoNCE。构4x4相似矩阵,跑softmax,取对角,算交叉熵。验你Python实对此手算。

2. SigLIP用bias参数`b`加于temperature:`S'[i,j] = S[i,j]/tau + b`。当batch大类不平衡(每行多负例于正)`b`何角?读SigLIP节3(arXiv:2303.15343)。

3. 建猫vs狗零分类器。试两提示模板:`a photo of a {class}`和`a picture of a {class}`。于100测图测精度。模板ensemble超单?

4. 计512-GPU跑batch 32k softmax InfoNCE vs sigmoid pairwise通信成本。何scale O(N),何O(N^2)?引SigLIP节4。

5. 读OpenCLIP scaling-law论文(arXiv:2212.07143, Cherti et al.)。从图复现其数据scaling结论:于定模型大小,ImageNet零精度和训数据大小何log-linear关系?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| InfoNCE | "对比loss" | Batch相似矩阵上交叉熵;每项正配是配项,负例是每他 |
| Sigmoid loss | "SigLIP loss" | 每对二元交叉熵;无softmax,无all-gather,分布训练便宜scale |
| Temperature | "tau" | Softmax/sigmoid前缩logits标量;控分布锐 |
| 零shot | "无微调分类" | 用文提示构类embedding和cosine相似分类;无目标类训 |
| 提示模板 | "a photo of a ..." | 类名周文脚手架;改零精度1-5点 |
| 双编码器 | "两塔" | 一图像编码器+一文编码器,出共享D维空间 |
| Hard negative | "难distractor" | 足够近正例负例使模型须工分离 |
| 线性probe | "冻+一层" | 仅冻特征上训线性分类器;测特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2能力ingest任纵横比和分辨率图像无resize |
| Temperature scaling | "log参数化tau" | CLIP参数化`log(1/tau)`使梯度行为;clip防坍至近零tau |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 噪网络数据scale。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling law。