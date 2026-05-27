# Transfusion:一Transformer内自回归文+扩散图像

> Chameleon和Emu3全赌离散token。它们工,但量化瓶颈显—图像质量plateau低于连续空间扩散模型。Transfusion (Meta, Zhou et al., 2024年8月)取反向赌:保持图像连续,完全丢VQ-VAE,并训一带两loss单transformer。文token得next-token-prediction。图像patch得flow-matching/扩散loss。两目标优化同权重。Stable Diffusion 3(MMDiT)下架构是近cousin。本课读Transfusion论点、建toy两loss trainer、并trace让一transformer做两工attention mask。

**类型:** 构建
**语言:** Python(stdlib,MNIST scale toy上两loss trainer)
**前置要求:** 阶段12课程11(Chameleon),阶段8(生成AI)
**时间:** ~180分钟

## 学习目标

- 线于一backbone跑两loss(NTP于文token、扩散MSE于图像patch)transformer。
- 释何图像patch间bidirectional attention加文token上causal attention是正mask择。
- 比Transfusion类(连续图像,扩散loss)与Chameleon类(离散图像,NTP)于算、质量和代码复杂度。
- 命MMDiT贡献:每块模态特定权重,residual stream joint attention。

## 问题背景

离散vs连续图像token辩论老于LLM。连续表示(原始像素,VAE latent)保细节。离散token(VQ索引)fit transformer原生词汇但量化步失细节。

Chameleon / Emu3离散:一loss,一架构,但图像保真tokenizer质量cap。

扩散模型连续:异常图像质量,但LLM分离模型,复杂noise-schedule工程,且无干净与文生成集成。

Transfusion问:可我们有两者?保持图像连续,仍训一模型,用两loss stitch入一梯度步。

## 概念讲解

### 两loss架构

单decoder-only transformer处理含:

- 文token(离散,从BPE vocab)序列。
- 图像patch(连续,16x16像素块经线性embedding投入隐藏维—同ViT编码器输入)。
- `<image>`和`</image>`标签标记连续patch何处。

Forward pass跑一次。Loss每token择两head之一:

- 文token:vocab-logits head上标准交叉熵。
- 图像patch:连续patch上扩散loss—预测每patch加噪。

梯度流经共享transformer body。两loss同时改进共享权重。

### Attention mask:causal文+bidirectional图像

文token须causal—你不可让文token attend未来文,teacher forcing破。图像patch,然,代表一snapshot;它们应于同图像块彼此bidirectional attend。

Mask:

```
M[i, j] = 1 if:
  (i是文且j是文且j <= i)   # 文causal
  OR (i是图像且j是图像且same_image_block(i, j))   # 图像内bidirectional
  OR (i是文且j是图像且j < i_image_end)   # 文attend前图像
  OR (i是图像且j是文且j < i_image_start)   # 图像attend前文
```

训和推理实为block-triangular mask。

### Transformer内扩散loss

扩散loss标准:加噪至图像patch,问模型预测噪(或干净patch,equivalently)。Transfusion版用flow matching—预测从噪至干净velocity field。

训间:
1. 每图像patch x0,采样随机timestep t。
2. 采样噪ε,计算xt = (1-t)*x0 + t*ε(flow matching线性插值)。
3. Transformer预测v_theta(xt, t);loss = MSE(v_theta(xt, t), ε - x0)。
4. Backprop与同序列文NTP loss并肩。

推理,生成是:
- 文token:标准自回归采样。
- 图像patch:条件于前文token扩散采样loop(典型10-30步)。

### MMDiT:Stable Diffusion 3变种

Stable Diffusion 3 (Esser et al., 2024年3月)发MMDiT(多模扩散Transformer)与Transfusion同时。架构是siblings。

MMDiT关键异:

- 每块模态特定权重。每transformer块有文token vs图像patch分离Q、K、V和MLP权重。Attention是joint(跨模态);余模态特定。
- Rectified flow训。特定flow-matching变种已知采样和比DDPM简数学。
- Scale。MMDiT是SD3 backbone(2B和8B参数变种)。Transfusion论文scale至7B。

两者收敛同核心念:一transformer跑文NTP和连续图像表示扩散。

### 何胜Chameleon类

连续扩散和离散NTP图像生成间质量缝可测。Transfusion论文报:

- 7B参数,FID同大小Chameleon类模型超3-5点。
- 无tokenizer训需—图像编码器简(线性投至隐藏,同ViT输入层)。
- 推理可并行图像patch denoise,不同于自回归图像token。

Downside:Transfusion是双loss模型,使训动态更棘。Loss权重需调。NTP和扩散间调度错配可致一头主导。

### 何坐下游

Janus-Pro(课程12.15)精Transfusion念经解耦视觉编码器为理解和生成—一用SigLIP,一用VQ—共享transformer body。Show-o(课程12.14)换扩散为离散扩散(masked prediction)。统生成族Transfusion后快分支。

2026产VLM emit图像—Gemini 3 Pro、GPT-5、Claude Opus 4.7图像生成路径—几乎确用某此族后代。细节proprietary。

## 使用

`code/main.py`建toy Transfusion于tiny MNIST类问题:

- 文caption是述数字(0-9)短整数序列。
- 图像是4x4字节网格。
- 共享权重线性投影pair作transformer stand-in;文NTP loss,噪patch MSE loss。
- 训loop交替两loss,attention mask显。
- 生成产一文caption和4x4图像于单forward pass。

Transformer是toy。两loss plumbing、attention mask构造和推理loop是真artifact。

## 交付成果

本课产`outputs/skill-two-loss-trainer-designer.md`。给新多模训任务(文+图像,文+音频,文+视频),它设计两loss调度(loss权重、mask形、共享vs模态特定块)并flag实现风险。

## 练习题

1. Transfusion类模型训70%文token和30%图像patch。图像扩散loss量级~10x文NTP loss。何loss权重balance它们?

2. 序列`[T, T, <image>, P, P, P, P, </image>, T]`实block-triangular mask。标每entry 0或1。

3. MMDiT有模态特定QKV权重。何参数开销加vs Transfusion全共享transformer?7B参数,值吗?

4. 生成:给文提示,模型NTP跑50 token,后hit `<image>`,后256 patch上扩散跑20 denoise步。总forward pass何?

5. 读SD3论文节3。述rectified flow和何收敛推理步少于DDPM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 两loss训 | "NTP +扩散" | 单transformer同梯度步优化文token交叉熵和连续图像patch MSE |
| Flow matching | "Rectified flow" | 扩散变种预测从噪至数据velocity field;数学简于DDPM |
| MMDiT | "多模DiT" | Stable Diffusion 3架构:joint attention,模态特定MLP和norm |
| Block-triangular mask | "Causal文+bidirectional图像" | 文间causal但图像区域内bidirectional attention mask |
| 连续图像表示 | "无VQ" | 图像patch作实值向量,非整数codebook索引 |
| Velocity prediction | "v-parameterization" | 网络输出是噪和数据间velocity field,非噪本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)