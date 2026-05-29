# Qwen-VL族和动态FPS视频

> Qwen-VL族—Qwen-VL(2023)、Qwen2-VL(2024)、Qwen2.5-VL(2025)、Qwen3-VL(2025)—是2026最有影响开视觉语言模型lineage。每代做单决架构赌开生态十二月内复制:经M-RoPE原生动态分辨率、动态FPS采样带绝对时对齐、ViT window attention和结构代理输出格式。至Qwen3-VL,配方稳:2D-RoPE ViT编码器带原生纵横比输入、MLP projector入大Qwen3语言基和训阶段强调OCR、grounding和代理行为为一等目标。本课按时间读族使你理解每个旋钮的作用和位置。

**类型:** 学习
**语言:** Python(stdlib,M-RoPE encoder + dynamic-FPS sampler)
**前置要求:** 阶段12课程06(patch-n'-pack)
**时间:** ~120分钟

## 学习目标

- 计M-RoPE三轴旋(temporal, height, width)并释何需全三。
- 为视频择动态FPS采样策略并推理token-per-second vs事件检测精度。
- 按序命四Qwen-VL代升级和何每enabled。
- 线Qwen2.5-VL类JSON代理输出格式并从VLM响应解析结构工具调用。

## 问题背景

Qwen-VL于2023年8月发为LLaVA-1.5和BLIP-2直响应。Qwen队target缝三fold:分辨率、视频和结构输出。

分辨率:LLaVA-1.5于336x336跑。照片好,中文发票或密电子表格截图无用。Qwen-VL第一创新448x448和grounded bounding-box输出,使模型指物。

视频:Video-LLaMA stack每帧编码器并feed它们至LLM。短clip工,多分钟视频时轴是信号不工。Qwen队欲单编码器理时间。

结构输出:LLaVA发自由文。代理需JSON。Qwen-VL于显JSON输出格式训含bounding-box坐标作文。

每Qwen-VL代扩这些三轴之一。

## 概念讲解

### Qwen-VL(2023年8月)

第一代:OpenCLIP ViT-bigG/14作编码器(2.5B参数),LLama兼容Q-Former(256 query一步),Qwen-7B基。贡献:

- 448x448分辨率(时开VLM SOTA)。
- Grounding:于显coordinate-token输出图文对训。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 中文+英文多语训从开始。

Benchmark时:英文与GPT-4V竞争,中文主导。Grounding监督是真headline。

### Qwen2-VL(2024年9月)—M-RoPE和原生分辨率

Qwen2-VL替固定分辨率+Q-Former stack为原生动态分辨率ViT编码器。关键改:

- 原生动态分辨率。ViT接受任28可除HxW(patch 14带2x空间merge)。1120x672图像(40x24 merged patch)产960视觉token。无resize、无tiling、无thumbnail。
- M-RoPE(多模RoPE)。每token携3D位置(t, h, w)替1D。图像t=0,视频t = frame_index。RoPE按每轴频率旋query/key向量。无位置embedding表。
- MLP projector。丢Q-Former;merged patch token上用2-layer MLP。
- 带动态FPS视频。视频默1-2 FPS采样,但模型接受任帧数。

结果:Qwen2-VL-7B于多 multimodal benchmark匹GPT-4o并于DocVQA超(94.5 vs 88.4)。架构改是决动。

### Qwen2.5-VL(2025年2月)—动态FPS +绝对时间

Qwen2.5-VL大移是视频。动态FPS非仅"需时多采样帧。"论文形式化:

- 绝对时间token。代位置索引(帧0, 1, 2...),用实际timestamp。"At 0:04, the cat jumps。"模型见`<time>0.04</time>` token interleaved帧token。
- 动态FPS。慢footage 1 FPS采样,动作4+ FPS。用户或训练者择;M-RoPE适。
- ViT Window attention。空间attention是windowed(block内局部)吞吐;每几层global attention。
- 显JSON输出格式。于工具调用数据训:"{\"tool\": \"click\", \"coords\": [380, 220]}"。代理ready out of box。
- MRoPE-v2 scaling。位置scale max输入大小使10分钟视频不耗频范围。

Benchmark:Qwen2.5-VL-72B于多视频benchmark超GPT-4o,于文档匹Gemini 2.0,并设GUI grounding开模型SOTA(ScreenSpot:84%精度vs GPT-4o 38%)。

### Qwen3-VL(2025年11月)

Qwen3-VL是增量升级consolidate而非reinvent:更大LLM backbone(Qwen3-72B),扩训数据,改进OCR,经Qwen3"thinking mode"更强推理。ViT和M-RoPE存。论文专注数据和训改进于架构。

Lineage takeaway:2025 Qwen-VL架构稳。附加代scale算和数据,非原语。

### M-RoPE数学

经典RoPE用配坐标旋位置`m`维`d` query `q`:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE分隐藏维三band。设`d = 96`。分32维temporal、32 height、32 width。每band旋己轴位置。Patch(t=5, h=10, w=20)得旋`R_t(5)`、`R_h(10)`、`R_w(20)`用于三band。

文token用`t = text_index, h = 0, w = 0`(或normalized择),保兼容。视频帧用`t = frame_time, h = row, w = col`。单图用`t = 0`。

益:一位置编码理文、图像和视频无branching代码或异位置表。

### 动态FPS采样逻辑

给视频时长`T`秒和目标token预算`B`:

1. 计你可负担max FPS:`fps_max = B / (T * tokens_per_frame)`。
2. 从`{1, 2, 4, 8}`择满足`fps <= fps_max`目标FPS。
3. 若motion高(optical-flow heuristic或显用户请求),择高FPS。若motion低,择低。
4. 于择FPS uniform采样;帧间insert `<time>t</time>` token。

Qwen2.5-VL隐训此逻辑;推理时用户经`fps`参数控。60秒动作序列4 FPS 81 token每帧 = 19440 token,32k context可理。

### 结构代理输出

Qwen2.5-VL代理训显target结构工具调用:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性:模型输出上JSON.parse。比自由"click at (1024, 512)"需regex和歧义处理。此shift是何Qwen2.5-VL ScreenSpot分从Qwen2-VL 55%跳84%。

## 使用

`code/main.py`实:

- Mix文、图像patch和视频帧packed sequence M-RoPE位置计算。
- 动态FPS sampler:给(duration, budget, motion_level),择FPS并发帧timestamp。
- Toy Qwen2.5-VL JSON输出解析器理带coordinate字段工具调用响应。

跑它,后感5分钟视频上换fixed-FPS为dynamic-FPS差。

## 交付成果

本课产`outputs/skill-qwen-vl-pipeline-designer.md`。给视频任务(监控、代理、动作识别、accessibility),它发Qwen2.5-VL配(帧预算、FPS策略、window-attention标志、代理输出模式)和延迟估计。于你deploy Qwen-VL族模型为视频产时用此。

## 练习题

1. 计patch(t=3, h=5, w=7)隐藏48(每band 16, base theta 10000)M-RoPE旋。示每band前三pair旋角。

2. 10分钟安防摄像1 FPS产何帧?384分辨率3x pool,总token何?Qwen2.5-VL默32k context理它?

3. 择30秒网球rally、30秒食谱demo和30秒UI代理录FPS。动态FPS逻辑每理。

4. Qwen2.5-VL完全丢Q-Former。何2025简单MLP工但2023不?(提示:数据scale和编码器质量。)

5. 解析三Qwen2.5-VL JSON工具调用输出入Python dict。何malformed JSON失和Qwen cookbook荐何恢复策略?

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| M-RoPE | "多模RoPE" | 隐藏维temporal、height、width band 3D旋位置embedding |
| 动态FPS | "智能采样" | 每视频基于motion、时长和token预算择帧采样率 |
| 绝对时间token | "Timestamp token" | 序列interleaved `<time>t</time>`使模型见实际秒非帧索引 |
| Window attention | "局部attention" | 速度限制小window空间self-attention;周期加global attention |
| 结构代理输出 | "JSON模式" | 训数据监督教VLM发带coords和工具名parseable JSON |
| min_pixels / max_pixels | "分辨率bound" | 每请求Qwen2.5-VL控bounding总像素数因此token数 |
| Grounding | "指它" | 文token输出bounding-box坐标;自Qwen-VL v1用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)