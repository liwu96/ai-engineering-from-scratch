# 水印——SynthID、Stable Signature、C2PA

> 三技术构2026 AI生成内容溯源。SynthID (Google DeepMind) — 图水印发2023年8月、文+视频2024年5月(Gemini + Veo)、文开源2024年10月经Responsible GenAI Toolkit、统多媒体detector 2025年11月配Gemini 3 Pro。文水印调下token样概率不显；图/视频水印存压缩、裁剪、filter、帧率改。Stable Signature (Fernandez等人, ICCV 2023, arXiv:2303.15435) — 微调latent diffusion decoder使每输出含固消息；裁剪(10%内容)生成图检>90% FPR<1e-6。后续"Stable Signature is Unstable" (arXiv:2405.07145, 2024年5月) — 微调除水印保质量。C2PA — 加签、篡显metadata标准(C2PA 2.2 Explainer 2025)。水印和C2PA补：metadata可剥但载更富溯源；水印存transcoding但载少信息。

**类型:** 构建
**语言:** Python(stdlib、token水印嵌入 + 检)
**前置要求:** 阶段10课程04(样)、阶段01课程09(信息论)
**时间:** ~75分钟

## 学习目标

- 描述token级水印(SynthID-text风格)和何检机制。
- 描述Stable Signature和2024除攻破它。
- 陈C2PA角色和何补水印。
- 描述关键限：模型特定信号、改述鲁棒、和意保攻(arXiv:2508.20228)。

## 问题背景

2023-2024见deepfake和AI生成内容于政治和消费context scale进。水印是提技溯源信号：创时标代、后检。2025证据：无水印无条件鲁、但层C2PA metadata合供可用溯源故事。

## 概念讲解

### 文水印(SynthID-text风格)

Kirchenbauer等人 2023机制、Google产化:

1. 每解码步、hash前K tokens产词汇"绿"和"红"集伪随机分。
2. 样偏绿集绿logits加δ。
3. 代含绿tokens多于几率产。

检：每前缀重hash、代中数绿tokens、算z-score。z-score水印文>0、人文~0。

属性:
- 读者不显(δ够小质量损小)。
- vocab分函数访检。
- 不改述鲁 — 重写文毁信号。

SynthID-text 2024年10月经Google Responsible GenAI Toolkit开源。

### Stable Signature (图)

Fernandez等人 ICCV 2023。微调latent diffusion decoder使每生成图latent表示含固二消息嵌。检经神经decoder latent解码。裁剪(10%内容)图检>90% FPR<1e-6。

2024年5月"Stable Signature is Unstable" (arXiv:2405.07145)：decoder微调除水印保图质量。对抗后生成微调便宜；水印对抗鲁棒限。

### SynthID统detector (2025年11月)

配Gemini 3 Pro：多媒体detector读文、图、音、视频SynthID信号于API一。统Google溯源栈。

### C2PA

Content Provenance and Authenticity联盟。加签篡显metadata标准。C2PA 2.2 Explainer (2025)。C2PA manifest录溯源声明(何创、何时、何变换)签创者key。

补水印:
- Metadata可剥；水印不可(易)。
- Metadata富(全溯源链)；水印载bits。
- C2PA依赖平台采；水印自嵌。

Google集双于Search、Ads、和"About this image"。

### 限

- **模型特定。** SynthID水印SynthID启模型代。非SynthID模型代无水印、故"无SynthID信号"非真证。
- **改述。** 文水印意保改述不存。
- **变换攻。** arXiv:2508.20228 (2025)示意保攻毁文水印和多图水印。
- **微调移。** "Stable Signature is Unstable"示后生成微调移嵌水印。

### EU AI Act Article 50

AI生成内容标签透明Code(首draft 2025年12月、二draft 2026年3月、期望终2026年6月[欧委会状页](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content))。Code仍draft于2026年4月timeline可改。需技层监管层。Deepfake须标。

### Phase 18何处

课程22-23是模型发何(私数据、溯源信号)。课程27覆训数据治。课程24是需此技措监管框架。

## 使用

`code/main.py`玩具文水印建。Tokens整数0..N-1；水印样偏hash定义绿集。Detector算绿token z-score。可观1000-token代检、观改述毁信号、并测人文假阳性率。

## 交付成果

本lesson产`outputs/skill-provenance-audit.md`。给内容部署溯源声明、审计：水印机制(若有)、C2PA签链(若有)、每对抗鲁棒、和每模态覆。

## 练习题

1. 跑`code/main.py`。报水印1000-token代vs人写文z-score。识95%置信阈值假阳性率。
2. 实改述攻替30% token同义词。重测z-score。
3. 读Kirchenbauer等人2023第6节鲁棒性。何文水印改述败但图水印裁剪存？
4. 设计用SynthID-text + C2PA metadata部署。述消费者见溯源链。识每组件一失败模式。
5. 2024"Stable Signature is Unstable"结果示微调移图水印。设计部署控限此攻 — 例如、需微调checkpoint签发。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| SynthID | "Google水印" | 跨模态溯源信号；文、图、音、视频 |
| Token水印 | "Kirchenbauer风格" | 偏样文水印经绿token z-score检 |
| Stable Signature | "图水印" | 微调decoder水印；ICCV 2023 |
| C2PA | "metadata标准" | 加签篡显溯源metadata |
| 改述鲁棒 | "改词破否" | 文水印属性；目前限 |
| 微调移 | "对抗去水印" | 经decoder微调移图水印攻 |
| 跨模态detector | "统SynthID" | 2025年11月跨模态统API |

## 延伸阅读

- [Kirchenbauer等人 — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token水印机制
- [Fernandez等人 — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 除攻
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — metadata标准