# 文档和图表理解

> 文档非照片。PDF、科学论文、发票或手写表有布局、表、图、脚注、头、语义结构纯图像理解不能捕。VLM前栈是流水线:Tesseract OCR + LayoutLMv3 +表提取启发。VLM浪换OCR-free模型——Donut (2022)、Nougat (2023)、DocLLM (2023)——直发结构标注。2026前沿仅"喂页面图像Claude Opus 4.7 2576px原生"，结构标注输出免费来。这课读文档AI三时代弧。

**类型:** 构建
**语言:** Python (stdlib，布局感知文档解析骨架)
**前置要求:** 第12阶段·05(LLaVA)，第5阶段(NLP)
**时间:** ~180分钟

## 学习目标

- 解释文档AI三时代:OCR流水线、OCR-free、VLM原生。
- 描述LayoutLMv3三输入流:文本、布局(bbox)、图像patch，带统一掩。
- 比Donut (OCR-free，图像→标注)、Nougat (科学论文→LaTeX)、DocLLM (布局感知生成)、PaliGemma 2 (VLM原生)。
- 为新任务(发票、科学论文、手写表、中文收据)选文档模型。

## 问题背景

"理解此PDF"欺骗难。信息在:

- 文本内容(90%信号)。
- 布局(头、脚注、边栏、双列格式)。
- 表(行、列、合单元格)。
- 图和图表。
- 手写注。
- 字体和排(标题vs正文)。

原始OCR倾文本失其余。关心发票系统需知"总:$1,245"来自右下，非脚注。

## 概念讲解

### 时代1——OCR流水线(2021前)

经典栈:

1. PDF→每页图像。
2. Tesseract(或商OCR)提取文本带每词边界框。
3. 布局分析识块(头、表、段)。
4. 表结构识别解析表。
5. 域规+正则提取字段。

干净打印文本工。手写、偏斜扫描、复杂表、非英文脚本失败。每失败模式需定制异常路。

### TrOCR (2021)

TrOCR (Li等，arXiv:2109.10282)换Tesseract经典CNN-CTC用transformer编码器解码器训合成+真文图像。手写和多语文净赢。仍流水线(检测后TrOCR后布局)，但OCR步剧改。

### 时代2——OCR-free (2022-2023)

首OCR-free模型说:跳检测全，直映图像像素结构输出。

Donut (Kim等，arXiv:2111.15664):
- 编码器解码器transformer，编码器Swin-B。
- 输出JSON表单理解，标注总结，或任务特定schema。
- 无OCR，无布局，无检测。

Nougat (Blecher等，arXiv:2308.13418):
- 专训科学论文。
- 输出LaTeX / markdown。
- 处理方程、多列布局、图。
- 每arXiv解析器调用模型。

这些是专家，非通用。Donut科学论文失败；Nougat发票失败。

### LayoutLMv3 (2022)

不同轨。LayoutLMv3 (Huang等，arXiv:2204.08387)保OCR加布局理解:

- 三输入流:OCR文token、每token 2D边界框、图像patch。
- 掩训练目标跨三模态(掩文本、掩patch、掩布局)。
- 下游:分类、实体提取、表QA。

LayoutLMv3是OCR基文档理解峰。表单发票强。需OCR上游。VLM前标准文档基准最佳准确。

### DocLLM (2023)

DocLLM (Wang等，arXiv:2401.00908)是LayoutLM生成兄弟。生成自由答案条件布局token。文档QA更好；仍依赖OCR输入。

### 时代3——VLM原生(2024+)

2024 VLM变好足换流水线全。喂全页图像高分辨率VLM，问问题，得答案。

- LLaVA-NeXT 336-tile AnyRes小文档工。
- Qwen2.5-VL动态分辨率原生处理2048+像素。
- Claude Opus 4.7支持2576px文档。
- PaliGemma 2 (2025年4月)专训文档+手写。

VLM原生和OCR流水线差距速闭。2026，VLM原生赢于:

- 场景文本(手写+打印，混脚本)。
- 复杂表带合单元格。
- 数学方程嵌入文本。
- 图带文本注。

OCR流水线仍赢于:

- 纯扫描负载大规模每页延迟重要。
- 流水线可靠性(确定失败vs VLM幻觉)。
- 监管环境需可审计OCR输出。

### Claude 4.7 / GPT-5前沿

2576像素原生输入，前沿VLM文档理解近人准确。2026初基准数:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3,流水线LayoutLMv3 ~83。
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78。
- VisualMRC: Claude 4.7 ~94。

闭模差距多分辨率和基础LLM规模。开源7B模型落后几点但追。

### 数学方程和LaTeX输出

科学论文需精确LaTeX方程输出。Nougat训此。带LaTeX目标训VLM (Qwen2.5-VL-Math, Nougat衍)产可用LaTeX。无显LaTeX训，VLM产可读但不精确转录。

2026科学论文流水线:链Nougat PDF，VLM处理棘页。

### 手写

仍最难子任务。混打印+手写(医笔记、填表)是OCR流水线仍VLM成本胜处。仅手写VLM改进(Claude 4.7, PaliGemma 2)。

### 2026配方

新文档AI项目:

- 纯打印发票规模:LayoutLMv3 +规则，成本效。
- 混文档(科学+手写+表):VLM原生(PaliGemma 2或Qwen2.5-VL)。
- 全arXiv摄取:Nougat数学，VLM图。
- 监管:OCR流水线+VLM验证器交叉检查。

## 使用它

`code/main.py`:

- 玩具布局感知分词器:给(文本，bbox)对，产LayoutLMv3式输入。
- Donut式任务schema生成器:表单JSON模板。
- 每页token预算比OCR流水线、Donut、Nougat、VLM原生。

## 发货它

这课产`outputs/skill-document-ai-stack-picker.md`。给文档AI项目(域、规模、质、监管)，选OCR流水线、OCR-free专家、VLM原生。

## 练习题

1. 项目10M发票每天。何栈最小化每页成本不失准确？

2. 何LayoutLMv3表单QA超纯CLIP-VLM但场景文本表现差？Bbox流放弃何？

3. Nougat生成LaTeX。提VLM原生输出LaTeX保真胜Nougat测试例，和Nougat胜例。

4. 读PaliGemma 2论文(Google, 2024)。何关键训练数据增加提文档准确vs PaliGemma 1？

5. 设计监管安全混:OCR流水线主，VLM次交叉检查。何解分歧？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| OCR流水线 | "Tesseract式" | 阶式栈:检测→OCR→布局→规则；确定，脆弱 |
| OCR-free | "Donut式" | 图像到输出transformer跳显OCR；单模型 |
| 布局感知 | "LayoutLM" | 输入包括每token bbox坐标；跨模态统一掩 |
| VLM原生 | "前沿VLM" | 直喂页面图像Claude/GPT/Qwen VLM高分辨率；无流水线 |
| DocVQA | "文档bench" | 文档VQA标准；最引分数 |
| 标注输出 | "LaTeX / MD" | 结构输出格式代自由文本；使下游自动化 |

## 延伸阅读

- [Li等—TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher等—Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang等—LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim等—Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang等—DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)