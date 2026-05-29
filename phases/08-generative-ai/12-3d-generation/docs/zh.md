# 3D生成

> 3D是杠杆最大的模态——从2D升至3D。2023年的突破是3D高斯泼溅。2024-2026年的生成管道叠加多视图扩散与3D重建，从单一提示词或照片生成对象和场景。

**类型:** 学习
**语言:** Python
**前置要求:** 阶段4(视觉)、阶段8课程07(潜扩散)
**时间:** ~45分钟

## 问题背景

3D内容棘手:

- **表示。**网格、点云、体素网格、有符号距离场(SDFs)、神经辐射场(NeRFs)、3D高斯。各有权衡。
- **数据稀缺。**ImageNet有14M图像。最大干净3D数据集(Objaverse-XL, 2023)有~10M对象,多低质。
- **内存。**512³体素网格是128M体素;有用场景NeRF需1M样本/射线。生成比重建难。
- **监督。**2D图像有像素。3D通常有少量2D视图须提升到3D。

2026栈分两问题。先,扩散模型生*2D多视图图像*。次,配*3D表示*(通常高斯泼溅)那些图像。

## 概念讲解

![3D生成:多视图扩散+3D重建](../assets/3d-generation.svg)

### 表示:3D高斯泼溅(Kerbl等, 2023)

场景作~1M 3D高斯云。每有59参数:位置(3)、协方差(6,或四元数4+尺度3)、不透明度(1)、球谐颜色(3阶48,0阶3)。

渲染=投影+alpha复合。快(4090上1080p~100 fps)。可微。对真照片梯度下降拟合。场景消费GPU5-30分钟拟合。

上两2023-2024创新:
- **生成高斯泼溅。**LGM、LRM、InstantMesh模型直接从一或几图像预测高斯云。
- **4D高斯泼溅。**动态场景配每帧偏移高斯。

### 多视图扩散

微调预训练图像扩散模型从文本提示词或单图像生同对象多一致视图。Zero123(Liu等, 2023)、MVDream(Shi等, 2023)、SV3D(Stability, 2024)、CAT3D(Google, 2024)。常输出对象周4-16视图,经高斯泼溅或NeRF提升到3D。

### 文本到3D管道

| 模型 | 输入 | 输出 | 时间 |
|------|------|------|------|
| DreamFusion(2022) | 文本 | NeRF via SDS | ~1小时每资产 |
| Magic3D | 文本 | 网格+纹理 | ~40分钟 |
| Shap-E(OpenAI, 2023) | 文本 | 隐式3D | ~1分钟 |
| SJC/ProlificDreamer | 文本 | NeRF/网格 | ~30分钟 |
| LRM(Meta, 2023) | 图像 | triplane | ~5秒 |
| InstantMesh(2024) | 图像 | 网格 | ~10秒 |
| SV3D(Stability, 2024) | 图像 | 新视图 | ~2分钟 |
| CAT3D(Google, 2024) | 1-64图像 | 3D NeRF | ~1分钟 |
| TripoSR(2024) | 图像 | 网格 | ~1秒 |
| Meshy 4(2025) | 文本+图像 | PBR网格 | ~30秒 |
| Rodin Gen-1.5(2025) | 文本+图像 | PBR网格 | ~60秒 |
| Tencent Hunyuan3D 2.0(2025) | 图像 | 网格 | ~30秒 |

2025-2026方向:配PBR材料适游戏引擎直接文本到网格模型。多视图扩散中间步仍是通对象最佳配方。

### NeRF(背景)

神经辐射场(Mildenhall等, 2020)。微小MLP取`(x, y, z, 视方向)`输出`(色, 密度)`。沿射线积分渲染。质量胜网格新视图合成但渲染慢100-1000x。大多实时用被高斯泼溅换但研究仍主。

## 动手实践

`code/main.py`实现玩具2D"高斯泼溅"拟合:合成目标图像(平滑渐变)作2D高斯泼溅和。梯度下降优化位置、颜色、协方差匹目标。见两核心操作:前向渲染(泼溅+alpha复合)和梯度下降拟合。

### Step 1: 2D高斯泼溅

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: 泼溅求和渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真3D高斯泼溅深度排序高斯按序alpha复合。2D玩具仅求和。

### Step 3: 梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视图不一致。**如独立生4视图且对象结构不同,3D拟合糊。修复:配共享注意力多视图扩散。
- **背侧幻觉。**单图像→3D须发明未见侧。质量变野。
- **高斯泼溅爆炸。**无约束训练涨到10M泼溅过拟合。致密化+剪枝启发(3D-GS原论文)必需。
- **拓扑问题。**隐式场(SDFs)网格常有洞或自交。发货前跑网格重整(如blender voxel remesh)。
- **训练数据许可。**Objaverse混合许可;商业用每模型变。

## 实际应用

| 任务 | 2026选 |
|------|--------|
| 照片场景重建 | 高斯泼溅(3DGS、Gsplat、Scaniverse) |
| 游戏文本到3D对象 | Meshy 4或Rodin Gen-1.5(PBR输出) |
| 图像到3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少图像新视图合成 | CAT3D、SV3D |
| 动态场景重建 | 4D高斯泼溅 |
| Avatar/穿衣人 | Gaussian Avatar、HUGS |
| 研究/SOTA | 上周发啥 |

发货生产游戏或电商管道3D:Meshy 4或Rodin Gen-1.5输出PBR网格直入Unity/Unreal。

## 产出成果

存`outputs/skill-3d-pipeline.md`。技能取3D简(输入:文本/一图像/几图像;输出:网格/泼溅/NeRF;用:渲染/游戏/VR)输出:管道(多视图扩散+拟合,或直接网格模型)、基模型、迭代预算、拓扑后处理、需材料道。

## 练习题

1. **简单。**跑`code/main.py`配4、16、64高斯。报告终MSE vs目标。
2. **中等。**扩到颜色高斯(RGB)。确认重构匹目标颜色模式。
3. **困难。**用gsplat或Nerfstudio,50照捕获重建真实对象。报告拟合时间和留出视图终SSIM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 3D高斯泼溅 | "3DGS" | 场景作3D高斯云;可微alpha复合渲染。 |
| NeRF | "神经辐射场" | MLP输出3D点颜色+密度;射线积分渲染。 |
| Triplane | "三2D平面" | 3D分解成三2D轴对齐特征网格;比体积便宜。 |
| SDS | "分数蒸馏采样" | 用2D扩散分数作伪梯度训3D模型。 |
| 多视图扩散 | "一次多视图" | 输出一致相机视图批的扩散模型。 |
| PBR | "物理渲染" | 具有反照率、粗糙度、金属和法线贴图的材料。 |
| 致密化 | "长泼溅" | 3DGS训练启发:高梯度区分割/克隆泼溅。 |

## 生产注:3D无共享基底

不像图像(潜扩散+DiT)和视频(时空DiT),3D 2026无单一主运行时。生产决策树叉表示:

- **NeRF/triplane。**推理是射线步进+每样本MLP前向。512²渲染需百万MLP前向。激进批次射线样本;SDPA/xformers适用。
- **多视图扩散+LRM重建。**两阶段管道。阶段1(多视图DiT)恰课程07扩散服务器。阶段2(LRM Transformer)视图上一次前向。总延迟轮廓"扩散+一次"——按阶段选服务原语。
- **SDS/DreamFusion。**每资产优化,非推理。建作业,非请求处理器。

大多2026产品,正确答"请求跑多视图扩散模型,异步重建到3DGS,实时查看发3DGS"。此干净分GPU推理服务器(快)和离线优化器(慢)工作负载。

## 延伸阅读

- [Mildenhall等(2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)——NeRF。
- [Kerbl等(2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)——3DGS。
- [Poole等(2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)——SDS。
- [Liu等(2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328)——Zero123。
- [Shi等(2023). MVDream](https://arxiv.org/abs/2308.16512)——多视图扩散。
- [Hong等(2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400)——LRM。
- [Gao等(2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314)——CAT3D。
- [Stability AI(2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d)——SV3D。