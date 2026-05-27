# 毕业项目 02 —— RAG over Codebase (跨仓库语义搜索)

> 2026年每个严肃工程组织都运行理解含义而非仅字符串的内部代码搜索。Sourcegraph Amp、Cursor的codebase answers、Augment的enterprise graph、Aider的repomap、Pinterest的internal MCP — 同形态。摄入多仓库、tree-sitter解析、函数和类级chunk嵌入、混合搜索、重排、带引用回答。此毕业项目要求你构建一个处10仓库2M行代码并于每次git push存活增量重索引。

**类型:** 毕业项目
**语言:** Python (摄入)、TypeScript (API + UI)
**前置要求:** 第5阶段(NLP基础)、第7阶段(transformers)、第11阶段(LLM工程)、第13阶段(工具)、第17阶段(基础设施)
**涉及阶段:** P5 · P7 · P11 · P13 · P17
**时间:** 30小时

## 问题背景

2026年每个前沿编码agent带codebase retrieval层因context window独不解跨仓库问。Claude 1M-token context助；不减ranked retrieval需。原始chunk naive cosine搜索毒结果于生成代码、monorepo重复、和rarely-imported symbols长尾。产答是AST-aware chunk上混合(dense + BM25)搜索带重排、backed by符号引用graph。

你学此经索引实fleet — 非一tutorial repo — 并测MRR@10、citation faithfulness、和增量freshness。失败模式是infrastructural：100k-file monorepo、push重touch半文件、需跨四仓库正确回答查询。

## 概念讲解

AST-aware摄入pipeline用tree-sitter解析每文件、提function和class节点、并于node边界而非固定token窗口chunk。每chunk得三表示：dense embedding (Voyage-code-3或nomic-embed-code)、sparse BM25 terms、和短自然语言summary。Summary加第三可检索模态 — 用户问"X何authorized"并summary提"authz"、即使代码仅`check_permission`。

Retrieval混合。Query火dense和BM25搜索、merge top-k、并hand union于cross-encoder重排(Cohere rerank-3或bge-reranker-v2-gemma-2b)。重排list于长context synthesizer (Claude Sonnet 4.7带prompt caching、或Llama 3.3 70B自host)带指令cite每claim于file和line range。无citation答案被post-filter拒。

增量freshness是基础设施问题。Git push触发diff：何文件改、何symbols改。仅affected chunks重embed。Affected cross-file symbol edges (imports、method calls)重算。Index持一致无需每commit重process 2M lines。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter带17语言grammar (Python、TS、Rust、Go、Java、C++、等)
- Dense embeddings：Voyage-code-3 (hosted)或nomic-embed-code-v1.5 (自host)、bge-code-v1 fallback
- Sparse index：Tantivy (Rust)带BM25F、symbol name vs body field-weighted
- Vector DB：Qdrant 1.12带hybrid search、或pgvector + pgvectorscale于50M vectors下团队
- Chunk summary model：Claude Haiku 4.5或Gemini 2.5 Flash、prompt-cached
- 重排：Cohere rerank-3或bge-reranker-v2-gemma-2b自hosted
- Orchestration：LlamaIndex Workflows于摄入、LangGraph于query agent
- Synthesizer：Claude Sonnet 4.7 (1M context)带prompt caching
- Symbol graph：Neo4j (managed)或kuzu (embedded)于import和call edges
- Observability：Langfuse spans per retrieval + synthesis step

## 动手实践

1. **摄入walker。** 每push hook迭代git history。收改文件。每文件、tree-sitter解析、提function和class nodes带全source span。发chunk records `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk summarizer。** Batch chunks于Haiku 4.5 calls带system preamble prompt caching。Prompt："一句话summary此function、命名public contract和side effects。"存summary于chunk旁。

3. **Embedding pool。** 两parallel queues：dense (Voyage-code-3 batch 128)和summary (同model、但于summary string)。写vectors于Qdrant带payload `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 index。** Field-weighted Tantivy index：symbol name weight 4、symbol body weight 1、summary weight 2。启"找named X function"queries于"找does X function"旁。

5. **Symbol graph。** 每chunk、record edges：imports (此文件用repo Z symbol Y)、calls (此function calls class C method M)、inheritance。存于kuzu。Query时用跨repo boundaries扩retrieval。

6. **Query agent。** LangGraph三nodes。`retrieve`火dense + BM25 parallel、按(repo, path, symbol)去重。`rerank`于top-50跑cross-encoder并keep top-10。`synth`于context调用Claude Sonnet 4.7带reranked chunks、cache system prompt、需file:line citations。

7. **Citation enforcement。** 解模型输出；无`(repo/path:start-end)` anchor claim被flag重ask或drop。回cited-only answer于用户。

8. **Incremental re-index。** 每webhook、算symbol-level diff。仅重embed text改chunks。imports改chunks重算symbol edges。测：50-file push于2M-LOC fleet 60秒内re-indexed。

9. **Eval。** 标100跨仓库问gold file:line答案。测MRR@10、nDCG@10、citation faithfulness (可验anchor claim比例)、和p50/p99 latency。

## 使用它

```
$ code-rag ask "S3 multipart abort何wired入我retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts触发`AbortMultipartOnFail`于
  services/uploader/retry.go:122-148、decrement per-bucket
  retry budget定义于config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 产出成果

可交付skill `outputs/skill-codebase-rag.md`。给repo corpus、建摄入pipeline、混合index、和query agent、并回任跨仓库问cited answer。评分标准：

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | Retrieval quality | MRR@10和nDCG@10于100问held-out集 |
| 20 | Citation faithfulness | 可验file:line anchor answer claim比例 |
| 20 | Latency和scale | p95 query latency于indexed corpus size 10k QPS |
| 20 | Incremental indexing correctness | 50-file commit git push到searchable时间 |
| 15 | UX和answer formatting | Citation clickability、snippet previews、follow-up affordance |
| **100** | | |

## 练习题

1. 换Voyage-code-3为nomic-embed-code自hosted。测MRR@10 delta。报gap是否重排enabled后close。
2. 注入20%生成代码(LLM-produced boilerplate)入corpus并重评估。观retrieval poisoning。payload加"generated" flag并down-weight those hits。
3. 基准Qdrant hybrid search vs pgvector + pgvectorscale于你corpus size。报batch size 1 p99。
4. 加sampling-based drift check：周、重run 100问eval。Alert于MRR@10 drop > 5%。
5. 延cross-language symbol resolution：Python function经gRPC调Go service。用symbol graph link。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| AST-aware chunking | "Function-level splits" | 于tree-sitter node边界而非固定token窗口切代码 |
| Hybrid search | "Dense + sparse" | BM25和vector search parallel run、merge top-k、rerank |
| Cross-encoder rerank | "Second-stage rank" | 一起score每(query, candidate)对模型、比cosine更准 |
| Prompt caching | "Cached system prompt" | 2026 Claude / OpenAI feature discount重复prefix tokens高达90% |
| Symbol graph | "Code graph" | imports、calls、inheritance跨文件和repo edges |
| Citation faithfulness | "Grounded answer rate" | 用户可click anchor读referenced span验证claim比例 |
| Incremental re-index | "Push-to-search time" | git push到改symbols可query wall-clock |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 产跨repo代码智能
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 此毕业项目参考deep-dive
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3细节
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder参考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内平台参考