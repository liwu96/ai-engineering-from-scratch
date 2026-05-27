# 毕业项目 08 —— 生产RAG聊天机器人 (监管垂直领域)

> Harvey、Glean、Mendable、和LlamaCloud于2026皆运行同产形态。用docling或Unstructured和ColPali摄入。混合搜索。用bge-reranker-v2-gemma重排。用Claude Sonnet 4.7合成、prompt caching于60-80%命中率。用Llama Guard 4和NeMo Guardrails守卫。用Langfuse和Phoenix观测。于200问golden set用RAGAS评分。于监管领域(法律、临床、保险)建一、毕业项目是通过golden set、red team、和漂移仪表板。

**类型:** 毕业项目
**语言:** Python (流水线 + API)、TypeScript (聊天UI)
**前置要求:** 第5阶段(NLP)、第7阶段(transformers)、第11阶段(LLM工程)、第12阶段(多模态)、第17阶段(基础设施)、第18阶段(安全)
**涉及阶段:** P5 · P7 · P11 · P12 · P17 · P18
**时间:** 30小时

## 问题背景

监管领域RAG(法律合同、临床试验协议、保险政策)是2026最多出货产形态因ROI明显且stakes具体。Harvey (Allen & Overy)建于法律。Mendable出货developer-docs风味。Glean覆盖企业搜索。模式是: 高保真摄入、带重排混合检索、带引用强制和prompt caching合成、多层安全守卫、和持续漂移监控。

难点不在模型。在jurisdiction-aware合规(HIPAA、GDPR、SOC2)、引用级审计、成本控制(prompt caching于高命中率买60-90%折扣)、经RAGAS faithfulness幻觉检测、和源文档更新而索引未跟上时漂移检测。毕业项目要求于200问golden set发货全部、并配red-team套件。

## 概念讲解

流水线两面。**摄入**: docling或Unstructured解析结构文档; ColPali处理视觉丰富; chunks得摘要、标签、和角色基访问标签。向量入pgvector + pgvectorscale (50M vectors下) 或 Qdrant Cloud; sparse BM25并行运行。**对话**: LangGraph处理记忆和多turn; 每query运行混合检索、用bge-reranker-v2-gemma-2b重排、用Claude Sonnet 4.7 (prompt-cached)合成、输出经Llama Guard 4和NeMo Guardrails、发引用锚定响应。

评估栈四层。**Golden set** (200标注Q/A带引用)正确性。**Red team** (jailbreaks、PII提取尝试、离域问题)安全。**RAGAS**每turn自动faithfulness / answer relevance / context precision。**漂移仪表板** (Arize Phoenix)周监控检索质量和幻觉评分。

Prompt caching是成本杠杆。Claude 4.5+和GPT-5+支持缓存系统提示 + 检索context。60-80%命中率、每query成本降3-5x。流水线须设计稳定prefix (系统提示 + 重排context先) 达高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入: Unstructured.io或docling结构文档; ColPali视觉丰富PDF
- 向量DB: pgvector + pgvectorscale 50M vectors下; Qdrant Cloud否则
- Sparse: Tantivy BM25带字段权重
- 编排: LlamaIndex Workflows (摄入) + LangGraph (对话)
- 重排器: bge-reranker-v2-gemma-2b自托管或Voyage rerank-2托管
- LLM: Claude Sonnet 4.7带prompt caching; fallback Llama 3.3 70B自托管
- 评估: RAGAS 0.2在线、DeepEval幻觉和jailbreak套件
- 可观测性: Langfuse自托管带annotation queue; Arize Phoenix漂移
- Guardrails: Llama Guard 4输入/输出classifier、NeMo Guardrails v0.12 policy、Presidio PII scrub
- 合规: chunks角色基访问标签; GDPR/HIPAA jurisdiction tags

## 动手实践

1. **摄入。** 用Unstructured或docling解析corpus (1000-10000文档严肃build)。扫描/视觉重页路由ColPali。产chunks带摘要、role-labels、jurisdiction tags。

2. **索引。** 密embeddings (Voyage-3或Nomic-embed-v2)入pgvector + pgvectorscale。BM25侧索引Tantivy。角色和jurisdiction filters作payload。

3. **混合检索。** 先filter by role+jurisdiction; 然并行dense + BM25; merge用reciprocal rank fusion; top-20给reranker; top-5给synth。

4. **带prompt caching合成。** 系统提示 + 静态policy在cache header; 重排context作cache extension; 用户问题作uncached suffix。稳态目标60-80%缓存命中率。

5. **Guardrails。** Llama Guard 4输入; NeMo Guardrails rails阻塞离域问题或policy禁止主题; Presidio scrub输出偶然PII; 引用强制post-filter。

6. **Golden set。** 200 Q/A对领域专家标注(answer、citations)。评分智能体exact-citation match、answer correctness、faithfulness (RAGAS)。

7. **Red team。** 50对抗提示: jailbreaks (PAIR、TAP)、PII exfiltration尝试、离域、跨jurisdiction泄漏。评分pass/fail和严重性。

8. **漂移仪表板。** Arize Phoenix周跟踪检索质量(nDCG、citation faithfulness)。5%降alert。

9. **成本报告。** Langfuse: prompt-caching命中率、tokens per query、$/query按阶段分解。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 产出成果

`outputs/skill-production-rag.md`描述deliverable。监管领域聊天机器人部署带合规标签、经rubric评分、带实时漂移监控观测。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | Golden set (200 Q/A)在线评分 |
| 20 | 引用正确性 | 可验源anchor答案比例 |
| 20 | Guardrail覆盖 | Llama Guard 4 pass rate + jailbreak套件结果 |
| 20 | 成本 / 延迟工程 | Prompt-cache命中率、p95延迟、$/query |
| 15 | 漂移监控仪表板 | Phoenix实时仪表板带周检索质量趋势 |
| **100** | | |

## 练习题

1. 建第二corpus slice于不同jurisdiction (如HIPAA配GDPR)。演示role+jurisdiction过滤防20问跨jurisdiction probe cross-leak。

2. 于一周产traffic测prompt-cache命中率。识何query破cache prefix。重构。

3. 加多turn记忆带10k-token summary buffer。测对话增长时faithfulness降否。

4. 换Claude Sonnet 4.7为Llama 3.3 70B自托管。测$/query和faithfulness delta。

5. 加"unsure"模式: 若top重排分数低于阈值、智能体说"我无自信引用"而非答。测假自信降。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Prompt caching | "Cached system + context" | Claude/OpenAI feature: 缓存prefix token于命中折扣60-90% |
| RAGAS | "RAG evaluator" | 自动评分faithfulness、answer relevance、context precision |
| Golden set | "标注eval" | 200+专家标注Q/A带引用; ground truth |
| Jurisdiction tag | "合规标签" | GDPR/HIPAA/SOC2范围附于chunks; 检索过滤强制 |
| 引用faithfulness | "Grounded answer rate" | 可检索源span支撑claim比例 |
| 漂移 | "检索质量衰减" | nDCG或citation评分周变化; alert阈值5% |
| Red team | "对抗eval" | 发前jailbreak、PII提取、离域probe |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律产栈
- [Glean企业搜索](https://www.glean.com) — 企业级RAG参考
- [Mendable文档](https://mendable.ai) — developer-docs RAG参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — canonical RAG评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026安全classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail框架