# Hybrid Memory——Vector+Graph+KV(Mem0)

> Mem0(Chhikara等,2025)视memory作三store并行——vector用于语义similarity、KV用于快fact lookup、graph用于entity-relationship reasoning。Scoring layer于取fusion三。此是2026产标准用于外memory。

**类型:** 构建
**语言:** Python(stdlib)
**前置要求:** 阶段14课程07(MemGPT)、阶段14课程08(Letta Blocks)
**时间:** ~75分钟

## 学习目标

- 释何单store(vector only、graph only、KV only)不足agent memory。
- 名Mem0三并行store和每优化何。
- 描述Mem0 fusion scoring——relevance、importance、recency——和何是weighted sum非hierarchy。
- 实toy三store memory于stdlib带`add()`写全三和`search()` fusion结果。

## 问题背景

单store错于三query类之一:

- **语义similarity**——"上周我们论agent drift何?"Vector胜;KV和graph漏。
- **Fact lookup**——"用户电话何?"KV胜;vector浪费、graph overkill。
- **关系reasoning**——"何customer共享billing entity?"Graph胜;vector和KV不能答。

产agent发全三于一session。单store memory总错于其二。Mem0贡献是线全三后单`add`/`search`面带scoring function fusion。

## 概念讲解

### 三store并行

Mem0(arXiv:2504.19413,2025年4月)于`add(text,user_id,metadata)`:

1. 从text extract candidate fact(LLM驱动步)。
2. 写每fact至vector store(embedding)用于语义search。
3. 写每fact至KV store keyed(user_id,fact_type,entity)用于O(1)lookup。
4. 写每fact至graph store(Mem0g)作typed edge用于关系query。

于`search(query,user_id)`:

1. Vector store回top-k按embedding cosine。
2. KV store回direct hit keyed于query-derived(user_id,type,entity)。
3. Graph store回从query entity reachable subgraph。
4. Scoring layer fusion三。

### Fusion scoring

```
score = w_relevance * relevance(q,record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance**——vector cosine、KV exact match、graph path weight。
- **Importance**——write time tag或learn(些fact更重:name、ID、policy)。
- **Recency**——last write或read时间 exponential decay。

Weight按product调。Chat agent高`w_recency`;compliance agent高`w_importance`;retrieval agent高`w_relevance`。

### Mem0g和temporal reasoning

Mem0g加conflict detector。当新fact矛盾existing edge,existing edge标记invalid但不删。Temporal query("用户三月city何?")走valid-at-time subgraph。

此是compliance-grade behavior Letta invalidation pattern泛化。

### Benchmark数

Mem0论文报告(2025):

- **LoCoMo**(长form conversation memory):91.6
- **LongMemEval**(长horizon episodic memory):93.4
- **BEAM 1M**(1M-token memory benchmark):64.1

比基线(full-context 128k LLM、flat vector store、flat KV)全失10+点。Benchmark不justify择——operational形需——但数显fusion设计非rounding error。

### Scope taxonomy

Mem0按scope分memory:

- **User memory**——跨session持久、keyed于`user_id`。
- **Session memory**——一thread内持久。
- **Agent memory**——每agent instance态。

每写选一scope。取可跨scope query带per-scope weight。无思混scope是何得"assistant告Alice Bob project"事故。

### 何此模式错

- **Embedding drift。**首百query看对vector result corpus增时降。加periodic top-N-used record re-embedding。
- **KV schema creep。**`(user_id,type,entity)`看简直到每team加己`type`。Quarterly audit type set。
- **Graph explosion。**一noisy extractor每message加50 edge。Cap每`add` call graph write;丢低confidence edge。

## 构建

`code/main.py`实三store模式于stdlib:

- `VectorStore`——naive token-overlap similarity作embedding stand-in。
- `KVStore`——dict keyed(user_id,fact_type,entity)。
- `GraphStore`——typed edge(subject,relation,object,valid)。
- `Mem0`——顶facade带`add()`、`search()`、fusion scoring、和scope-aware retrieval。
- 于multi-user、multi-session对话工作trace。

跑:

```
python3 code/main.py
```

输出显三分离recall path加fused top-k。Flip scoring weight于`main()`顶并watch ranking变。

## 使用

- **Mem0(Apache 2.0)**——产ready。Self-host用Postgres+Qdrant+Neo4j、或用managed cloud。
- **Letta**——三tier core/recall/archival;带己vector和graph backend。
- **Zep**——商业替代带temporal KG和fact extraction。
- **Custom build**——需exact extractor控(compliance)或fusion weight(voice agent recency dominate)。

## 交付成果

`outputs/skill-hybrid-memory.md`生三store memory scaffold带fusion scorer、scope taxonomy、和temporal invalidation wired。

## 练习题

1. 换toy vector similarity用真实embedding model(sentence-transformers、Ollama、OpenAI embedding)。测recall@10于合成长对话。Ranking 1000 write后drift否?
2. 加temporal query:`search(query,as_of=timestamp)`。仅回timestamp时或前valid record。何store需最多工作?
3. 实conflict detector:若incoming fact矛盾graph edge、invalidate old edge并log both。测"用户住Berlin"->"用户住Lisbon。"
4. 移fusion scorer含`user_feedback` dimension(thumbs-up于取record)。何防gaming(agent仅回它已like record)?
5. 读Mem0 docs(`docs.mem0.ai`)。移toy至`mem0` client call。比相同20 test query retrieval quality。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Hybrid memory | "Vector加graph加KV" | 三store并行写、取fusion |
| Fact extraction | "Memory ingestion" | LLM步断text(entity,relation,fact)tuple |
| Fusion scoring | "Relevance ranking" | Relevance、importance、recency weighted sum |
| Scope | "Memory namespace" | user/session/agent——定何见何 |
| Mem0g | "Memory graph" | Typed edge带temporal validity用于关系query |
| Temporal invalidation | "Soft delete" | 标矛盾edge invalid;永不删 |
| Embedding drift | "Retrieval rot" | Vector quality corpus增时降;periodically re-embed |

## 延伸阅读

- [Chhikara等,Mem0(arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)——原论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview)——产API、SDK、managed cloud
- [Packer等,MemGPT(arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)——virtual-context前身
- [Letta,Memory Blocks blog](https://www.letta.com/blog/memory-blocks)——三tier sibling设计