# 毕业项目 04 —— 多模态文档QA (Vision-First PDF、表、图)

> 2026文档QA前沿移离OCR-then-text朝vision-first late interaction。ColPali、ColQwen2.5、和ColQwen3-omni视每PDF页为图像、multi-vector late interaction embed、让query直接attend patches。于财务10-Ks、科学论文、和手写笔记此pattern beat OCR-first大幅。建pipeline端到端于10k页并发side-by-side对OCR-then-text。

**类型:** 毕业项目
**语言:** Python (pipeline)、TypeScript (viewer UI)
**前置要求:** 第4阶段(计算机视觉)、第5阶段(NLP)、第7阶段(transformers)、第11阶段(LLM工程)、第12阶段(多模态)、第17阶段(基础设施)
**涉及阶段:** P4 · P5 · P7 · P11 · P12 · P17
**时间:** 30小时

## 问题背景

企业坐于OCR pipelines mangle PDFs：旋转表扫描10-Ks、方程密集科学论文、仅图 sensible charts、手写annotations。视此text-first意失半signal。2026答是raw page images late-interaction multi-vector retrieval。ColPali (Illuin Tech)引入；ColQwen2.5-v0.2和ColQwen3-omni推accuracy。于ViDoRe v3、vision-first retrieval score OCR-then-text上显著margin — gap于charts、tables、和handwriting widen。

Trade-off是storage和latency。ColQwen embedding每页~2048 patch vectors、非单1024-dim vector。Raw storage balloon。DocPruner (2026)带50% pruning无measurable accuracy loss。你将index 10k页、测ViDoRe v3 nDCG@5、serve answers under 2s、并直比OCR-then-text baseline。

## 概念讲解

Late interaction意每query token score每patch token、每query token maximum score sum。获fine-grained matching无需single pooled vector。Multi-vector index (Vespa、Qdrant multi-vector、或AstraDB)存per-patch embeddings并retrieval time run MaxSim。

Answerer是vision-language model取query加top-k retrieved pages作图像并写带evidence regions (bounding boxes或page references) answer。Qwen3-VL-30B、Gemini 2.5 Pro、和InternVL3是2026前沿choice。于equations和scientific notation、OCR fallback (Nougat、dots.ocr) splice入作optional text channel。

Evaluation是二维矩阵。一轴：content type (plain text paragraphs、dense tables、bar/line charts、handwritten notes、equations)。另一轴：retrieval approach (vision-first late interaction vs OCR-then-text vs hybrid)。每cell得nDCG@5和answer accuracy。报告是deliverable。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- Page rendering：PyMuPDF (fitz) at 180 DPI、portrait-normalized
- Late-interaction model：ColQwen2.5-v0.2或ColQwen3-omni (vidore team on Hugging Face)
- Index：Vespa带multi-vector field、或Qdrant multi-vector、或AstraDB带MaxSim
- Pruning：DocPruner 2026 policy (keep high-variance patches、50% compression at < 0.5% accuracy loss)
- OCR fallback (equations / dense tables)：dots.ocr或Nougat
- VLM answerer：Qwen3-VL-30B自hosted或Gemini 2.5 Pro hosted；InternVL3 fallback
- Evaluation：ViDoRe v3 benchmark、M3DocVQA于multi-page reasoning
- Viewer UI：Next.js 15带canvas overlay于evidence regions

## 动手实践

1. **摄入。** Walk 10-Ks、科学论文、和扫描文档10k PDF pages corpus。Render每页1536x2048 PNG。Persist `{doc_id, page_num, image_path}`。

2. **Embed。** 每page image Run ColQwen2.5-v0.2。Output shape ~2048 patch embeddings dim 128。Apply DocPruner keep highest-signal half。Write Vespa multi-vector field或Qdrant multi-vector。

3. **Query。** 每incoming query、embed query tower (token-level embeddings)。Run MaxSim对index：每query token、take max dot-product over page patch embeddings、sum。Return top-k pages。

4. **Synthesize。** Call Qwen3-VL-30B带query和top-5 page images。Prompt："仅用所供页回答。Cite每claim于(doc_id, page)并名region (figure、table、paragraph)。"

5. **Evidence regions。** Post-process answer提cited regions。若VLM emit bounding boxes (Qwen3-VL does)、viewer render overlay。

6. **OCR fallback。** 识equation-dense页 (image variance heuristic)、Run Nougat或dots.ocr并pass OCR text作图像旁extra channel。

7. **Eval。** Run ViDoRe v3 (retrieval nDCG@5)和M3DocVQA (multi-page QA accuracy)。同corpus同synthesizer Run OCR-then-text pipeline。Produce content-type × approach matrix。

8. **UI。** Streamlit prototype first；Next.js 15产viewer带page-by-page evidence-region overlay。

## 使用它

```
$ doc-qa ask "2024 EMEA segment operating margin change何?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin从18.2%移到16.8%、140bp decline。
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open带highlighted bounding boxes overlaid on p.88 Table 4
```

## 产出成果

`outputs/skill-doc-qa.md`描述deliverable：vision-first multimodal document QA system tuned于特定corpus并于ViDoRe v3对OCR-then-text baseline评估。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline和published leaderboard |
| 20 | Evidence-region grounding | 实含answer span cited regions比例 |
| 20 | Storage和latency engineering | DocPruner compression ratio、index p95、answer p95 |
| 20 | Multi-page reasoning | hand-labeled 100问multi-page set accuracy |
| 15 | Source-inspection UX | Viewer clarity、overlay fidelity、side-by-side comparison tools |
| **100** | | |

## 练习题

1. 同corpus测ColQwen2.5-v0.2 vs ColQwen3-omni。何页一right另一miss？Index加"content class" tag按type route。
2. 激prune embeddings (75%、90%)。找compression cliff：ViDoRe nDCG@5 drop OCR baseline下点。
3. 建hybrid：parallel run OCR-then-text和ColQwen、RRF fuse、cross-encoder rerank。Hybrid beat任alone？何help most？
4. 换Qwen3-VL-30B为smaller VLM (Qwen2.5-VL-7B)。测accuracy-per-dollar curve。
5. 加handwritten-note支持。Render handwriting corpus、ColQwen embed、测retrieval。比handwriting OCR pipeline。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Late interaction | "ColPali-style retrieval" | Query tokens独立对page patches score；MaxSim aggregates |
| Multi-vector | "Per-patch embedding" | 每document多vectors、非一pooled vector |
| MaxSim | "Late-interaction scoring" | 每query token take max similarity over document vectors；sum |
| DocPruner | "Patch compression" | 2026 pruning keep 50% patches negligible accuracy loss |
| ViDoRe v3 | "Document-retrieval benchmark" | 2026 standard测visual-document retrieval |
| Evidence region | "Cited bounding box" | Source page localize answer span bbox |
| OCR fallback | "Equation channel" | Vision旁用于equation-或table-heavy页text pipeline |

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 参考late-interaction doc retrieval
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — 产ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — 参考serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback