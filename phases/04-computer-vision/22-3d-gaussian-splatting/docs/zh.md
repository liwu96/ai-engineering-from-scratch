# 3D Gaussian Splatting从零构建

> 场景是百万3D高斯云。每有位置、方向、尺、不透明度和依赖视角的色。光栅化它们、反向过光栅化、完。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段4课程13(3D视觉与NeRF)，阶段1课程12(张量操作)，阶段4课程10(扩散基础可选)
**时间:** ~90分钟

## 学习目标

- 解释为何3D Gaussian Splatting 2026替NeRF为照真3D重建生产默认
- 述每高斯六参(位置、转四元数、尺、不透明、球谐色、可选特征)和每贡献几浮
- 从零实现2D Gaussian splatting光栅器用`alpha`合成，后示3D况投到同循环
- 用`nerfstudio`、`gsplat`或`SuperSplat`从20-50照重建场景并导到`KHR_gaussian_splatting` glTF扩展或OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` schema

## 问题背景

NeRF存场景为MLP权重。每渲染像素是光线百MLP查询。训时、渲秒、权不可编 — 若想移场景中椅，需重训。

3D Gaussian Splatting (Kerbl、Kopanas、Leimkühler、Drettakis, SIGGRAPH 2023)替全。场景是显式3D高斯集。渲染是GPU光栅化100+ fps。训分钟。编辑直：移部分高斯你移椅。2026 Khronos Group ratified glTF高斯泼溅扩展、OpenUSD 26.03船高斯泼溅schema、Zillow和Apartments.com用它们渲染地产、大多新3D重建研论是核心3DGS想变种。

心模简，数有够移部分多介始光栅化跳投和球谐。这课建全 — 先2D版，后3D扩。

## 概念讲解

### 高斯载何

一3D高斯是空间参blob带这些属性：

```
position         mu         (3,)    世界坐标中心
rotation         q          (4,)    编方向单位四元数
scale            s          (3,)    每轴log-尺(渲染时指数化)
opacity          alpha      (1,)    后sigmoid不透明[0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视依赖色
```

转 + 尺建3x3协方差：`Sigma = R S S^T R^T`。那是高斯3D形。球谐让色随视角变 — 高光、微光泽、视依赖辉 — 无存每视纹理。SH度3你得每色通道16系数，单色48浮每高斯。

场景典型1-5百万高斯。每存约60浮(3 + 4 + 3 + 1 + 48 + misc)。那是500万高斯场景240 MB — 远小于带每点纹理等价点云，阶小于高分辨率重渲染NeRF MLP权重。

### 光栅化，非光线行进

```mermaid
flowchart LR
    SCENE["百万3D高斯<br/>(位置、转、尺、<br/>不透明、SH色)"] --> PROJ["投到2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["配到瓦<br/>(16x16屏空间)"]
    TILES --> SORT["深排<br/>每瓦"]
    SORT --> ALPHA["Alpha合成<br/>前到后"]
    ALPHA --> PIX["像素色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五步，全GPU友好。无每像素MLP查询。单RTX 3080 Ti 147 fps渲染600万泼溅。

### 投步骤

世界位`mu`带3D协方差`Sigma`的3D高斯投到屏位`mu'`带2D协方差`Sigma'`的2D高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视变换(相机转 + 平移)
J = mu'处透视投雅可比
```

2D高斯脚印是椭圆，轴`Sigma'`特征向量。椭圆内每像素收高斯贡献，权`exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha合成规则

一像素，覆它的高斯后到前排(或等价前到后反转公式)。色用1980s每半透明光栅器同方程合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到i透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   本贡献
c_i = eval_SH(SH_i, view_direction)    视依赖色
```

这**是NeRF体积渲染同方程**，仅于显式稀高斯集而非光线密样本。那身份是渲染质量匹NeRF因 — 两者积同辐射场方程。

### 为何可微

每步 — 投、瓦配、alpha合成、SH评估 — 对高斯参可微。给真图像，算渲染像素损、反向过光栅器、梯度降更新全`(mu, q, s, alpha, c_lm)`。约30,000迭高斯找正位、尺和色。

### 密化与剪

固定高斯集不能覆复杂场景。训含两适应机制：

- **克隆**高斯于当位置当梯度量高但尺小 — 重建需更多细节此。
- **分裂**大尺高斯为两小当梯度高 — 一大高斯太滑合区域。
- **剪**不透明度低于阈值高斯 — 它们不贡献。

密化每N迭跑。场景典型从~100k初始高斯(SfM点种子)长到训终1-5M。

### 球谐一段

视依赖色是单位球函数`c(direction)`。球谐是球Fourier基。截于度`L`得每通道`(L+1)^2`基函数。新视角评估色是学SH系数和视角基评估点积。度0 = 一系数 = 常色。度3 = 16系数 = 足捕Lambertian明、高光和微反射。SD Gaussian Splatting论文默认度3。

### 2026生产栈

```
1. 捕         smartphone / DJI无人机 / 手持扫
2. SfM / MVS       COLMAP或GLOMAP导相机位姿 + 稀点
3. 训3DGS      nerfstudio / gsplat / inria官方 / PostShot (~10-30 min RTX 4090)
4. 编            SuperSplat / SplatForge(清浮子、分割)
5. 导          .ply -> glTF KHR_gaussian_splatting或.usd (OpenUSD 26.03)
6. 观            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D和生成变种

- **4D Gaussian Splatting** — 高斯时间函数；用于体积视频(Superman 2026、A$AP Rocky "Helicopter")。
- **生成泼溅** — 文到泼溅模型(Marble by World Labs)幻觉整场景。
- **3D Gaussian Unscented Transform** — NVIDIA NuRec自动驾驶模拟变种。

## 构建

### 步骤1: 2D高斯

先建2D光栅器。3D况投后降为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   坐标
    returns: (G, H, W)  每像素每高斯密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum`做每(高斯,像素)对二次型`diff^T Sigma^-1 diff`。

### 步骤2: 2D泼溅光栅器

Alpha合成前到后。2D深无意义，故用学每高斯标量排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     [0, 1]
    depths:    (G,)     每高斯标量排序用
    image_size: (H, W)
    returns:   (H, W, 3) 渲图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

非快 — 真实现用瓦基CUDA核 — 但正数和全可微。

### 步骤3: 可训2D泼溅场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D单角
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit`和`colour_logits`皆无约束参数渲染时经正激活映射。这是每3DGS实现标准模式。

### 步骤4: 2D高斯拟合目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红圆
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝方
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"步 {step:3d}  mse {loss.item():.4f}")
```

200步64高斯定入两形。那是全想 — 显式几何原梯度降。

### 步骤5: 从2D到3D

3D扩保同循环。加：

1. 每高斯转是四元数而非单角。
2. 协方差是`R S S^T R^T`，`R`四元数建、`S = diag(exp(log_scale))`。
3. 投`(mu, Sigma) -> (mu', Sigma')`用相机外参和`mu'`处透视投雅可比。
4. 色变球谐展开；视角评估。
5. 深排从真相机空间z而非学标量。

每生产实现(`gsplat`、`inria/gaussian-splatting`、`nerfstudio`)于GPU瓦基CUDA核正做。

### 步骤6: 球谐评估

SH基达度3每通道16项。评估：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   末维RGB通道
    dirs:      (..., 3)       单向量
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 度3项此略；全16系数版码文件
    return result
```

学`sh_coeffs`存那高斯"每方向色"。渲染时评估当前视角得3向量RGB。

## 使用

真3DGS工作，用`gsplat` (Meta)或`nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto`是nerfstudio 3DGS训器。RTX 4090典型场景跑10-30分钟。

2026导选项重要：

- `.ply` — 原高斯云(可移、最大文件)。
- `.splat` — PlayCanvas / SuperSplat量化格式。
- glTF `KHR_gaussian_splatting` — Khronos标准，跨查看器可移(2026 Feb RC)。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD原生，NVIDIA Omniverse和Vision Pro管道。

4D / 动态场景，`4DGS`和`Deformable-3DGS`同机制扩时变均值和不透明度。

## 交付成果

本课程产：

- `outputs/prompt-3dgs-capture-planner.md` — 给场景类型计划捕会(照数、相机路径、光)提示词
- `outputs/skill-3dgs-export-router.md` — 给下游查看器或引擎择正导格式(.ply / .splat / glTF / USD)技能

## 练习题

1. **(易)** 跑上2D泼溅训器于异合成图像。变`num_splats`在`[16, 64, 256]`并绘每MSE vs步。识回点递减。

2. **(中)** 扩2D光栅器支持每高斯RGB色依赖标"视角角"通过度2谐波。训于两目标图像并验模型重建两者。

3. **(难)** 克`nerfstudio`并训`splatfacto`于你有任场景20照捕(桌、植物、脸、房)。导glTF `KHR_gaussian_splatting`并在查看器开(Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9)。报训时间、高斯数和渲染fps。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 3DGS | "高斯泼溅" | 显式场景表示为百万3D高斯带每高斯位置、转、尺、不透明、SH色 |
| 协方差 | "高斯形" | `Sigma = R S S^T R^T`；一高斯方向和各向异性尺 |
| Alpha合成 | "后到前混合" | NeRF体积渲染同方程，现于显式稀集 |
| 密化 | "克隆和分裂" | 重建欠拟合处适加新高斯 |
| 剪 | "删低不透明" | 移训时崩到近零不透明高斯 |
| 球谐 | "视依赖色" | 球上Fourier基；存色为视角函数 |
| Splatfacto | "nerfstudio 3DGS" | 2026训3DGS最易路 |
| `KHR_gaussian_splatting` | "glTF标准" | Khronos 2026扩展使3DGS跨查看器和引擎可移 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl等, SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产质量CUDA光栅器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训配方
- [Khronos KHR_gaussian_splatting扩展](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026可移格式
- [OpenUSD 26.03发布注](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D Gaussian Splatting 2026状态](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 业概