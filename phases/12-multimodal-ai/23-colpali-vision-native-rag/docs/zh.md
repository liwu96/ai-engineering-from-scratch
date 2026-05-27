# ColPali和视觉原生文档RAG

> 传统RAG解析PDF成文本，分chunk，嵌chunk，存向量。每步失信号:OCR倾图数据、chunking断表行、文本嵌入忽视图。ColPali (Faysse等，2024年7月)问更简问题:何提取文本？直嵌页面图像经PaliGemma，用ColBERT式晚交互检索，保全布局、图、字体、格式信号文档带。发基准:富视觉文档端到端准确比文RAG好20-40%。ColQwen2、ColSmol、VisRAG延模式。这课读视觉原生RAG论并建小ColPali类索引器。

**类型:** 构建
**语言:** Python (stdlib，多向量索引器+MaxSim评分器)
**前置要求:** 第11阶段(LLM工程——RAG基础)，第12阶段·05(LLaVA)
**时间:** ~180分钟

## 学习目标

- 解释双编码器检索(每文档一向量)和晚交互检索(每文档多向量)差。
- 描述ColBERT MaxSim操作和ColPali何从文token泛图像patch。
- 建小ColPali类索引器:页→patch嵌入→MaxSim查询term嵌入→top-k页。
- 比ColPali + Qwen2.5-VL生成器vs文RAG + GPT-4于发票/财务报告用例。

## 问题背景

PDF文RAG倾文档大部。财务报告Q3收入增长通常在图表；医学报告发现注图像；法律合同签块是布局事实，非文本事实。

文RAG流水线:

1. PDF→文本经OCR / pdftotext。
2. 文本→300-500 token chunk。
3. Chunk→双编码器嵌入(一向量)。
4. 用户查询→嵌入→余弦相似→top-k chunk。
5. Chunk +查询→LLM。

五失步。图表未捕。表跨chunk断。多列布局平。图注消失。

ColPali修:跳OCR，直嵌页图像。用ColBERT式晚交互检索使模型可注细粒patch查询时。

## 概念讲解

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832)是文本检索方法。代每文档一向量，它产每token一向量。查询时:

- 查询token得己嵌入(N_q向量)。
- 文档token得嵌入(N_d向量，典型缓存)。
- 分=查询token和文档token余弦相似最大和:Σ_i max_j cos(q_i, d_j)。

这是MaxSim操作。每查询token"选"最佳匹配文档token。终分是和。

优:强召回，处term级语义。缺:N_d向量每文档，存贵。

### ColPali

ColPali (Faysse等，arXiv:2407.01449)用ColBERT模式图像。

- 每页经PaliGemma (ViT +语言)编码patch嵌入:N_p向量每页。
- 每用户查询(文本)编码查询token嵌入:N_q向量。
- 分=Σ_i max_j cos(q_i, p_j)，即MaxSim于查询文token和页图像patch。
- 按总分检索top-k页。

文档摄入时:每页PaliGemma嵌入，存全patch嵌入。查询时:嵌入查询token，MaxSim算全存页嵌入，返top-k页。

优:端到端富视觉文档击文RAG 20-40%。每patch向量捕局布局内容。

缺:N_p patch × 4字节浮点 × D维向量每页=存速长。PQ / OPQ量化缓解。

### ColQwen2和ColSmol

ColQwen2 (illuin-tech, 2024-2025)换PaliGemma Qwen2-VL。好基编码器，好检索。

ColSmol是更小变本地/边用。~1B参数ColSmol检索器消费GPU运。

### VisRAG

VisRAG (Yu等，arXiv:2410.10594)是不同变:代patch上MaxSim，每页VLM池成单向量后双编码器检索。快索引+小存，弱召回。

质量vs成本权衡:ColPali质量，VisRAG缩。

### M3DocRAG

M3DocRAG (Cho等，arXiv:2411.04952)延多模态检索多页多文档推理。跨文档检索页，组多页上下文VLM。

### ViDoRe——基准

ColPali伴基准。视觉文档检索评估。任务包括财务报告、科学论文、行政文档、医学记录、手册。指标:nDCG@5。

ColPali-v1 ViDoRe评分~80% nDCG@5；同文档文RAG评分~50-60%。

### 端到端RAG流水线

视觉原生RAG:

1. 摄入:PDF→页图像→PaliGemma编码→存全patch嵌入。
2. 查询:用户文本→查询token嵌入→MaxSim全索引页→top-k页。
3. 生成:top-k页图像+查询→VLM (Qwen2.5-VL或Claude)→答案。

无OCR任何处。图、图表、字体、布局全流入答案。

### 存储数学

50页财务报告729 patch每页128维嵌入:

- ColPali: 50 * 729 * 128 * 4字节 = ~18 MB原始，~4 MB PQ后。
- 文RAG: 50 chunk * 768维 * 4字节 = ~150 kB。

ColPali每文档存~30x多。缩，OPQ / PQ带~5-10x，通常可忍。

### 何文RAG仍赢

- 纯文文档无布局信号(wiki文、聊日志)。文RAG简存便宜。
- 百万页档案存主宰成本。
- 严监管要求检索旁可提取OCR文本。

2026其他一切——财务报告、科学论文、法律合同、医学记录、UX文档——视觉原生RAG赢。

## 使用它

`code/main.py`:

- 玩具patch编码器:映"页"(特征向量小网格)patch嵌入数组。
- MaxSim评分器:算ColBERT式分于查询token嵌入集和页patch集。
- 索5玩具页，运3查询，返top-k带分。

## 发货它

这课产`outputs/skill-vision-rag-designer.md`。给文档RAG项目，选ColPali / ColQwen2 / VisRAG /文RAG并尺寸存。

## 练习题

1. 200页年报729 patch每页，128维嵌入，4字节浮点。算原始存和PQ压缩(8x)存。

2. MaxSim是Σ_i max_j cos(q_i, p_j)。何此和捕简均相似不能？

3. ColPali索引页作patch集。何若代单词级索引(如ColBERT)？权衡？

4. 设计端到端流水线1M页语料500ms查询延迟预算。选ColQwen2 / VisRAG并证。

5. 读M3DocRAG (arXiv:2411.04952)。描述多页注意模式和何不同于单页ColPali检索。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 晚交互 | "ColBERT式" | 检索用每token或每patch嵌入+MaxSim，非单文档向量 |
| MaxSim | "Max-over-patches" | 每查询token选最高相似文档token；跨查询和 |
| 双编码器 | "单向量" | 每文档一向量；更快但失粒度 |
| 多向量 | "每文档多向量" | 每文档/页存N_p向量；存成本长但召回改进 |
| Patch嵌入 | "页特征" | VLM编码器每图像patch一向量，每页缓存 |
| ViDoRe | "视觉文档bench" | ColPali视觉文档检索基准套 |
| PQ量化 | "乘积量化" | 维持向量相似压缩缩存~8x |

## 延伸阅读

- [Faysse等—ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia—ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu等—VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho等—M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)