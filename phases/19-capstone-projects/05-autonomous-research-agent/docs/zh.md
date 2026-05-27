# 毕业项目 05 —— 自研智能体 (AI-Scientist Class)

> Sakana's AI-Scientist-v2发全论文。Agent Laboratory行实验。Allen AI共享traces。2026形态是plan-execute-verify tree search于实验、budgeted cost、sandboxed code execution、vision-feedback LaTeX writer、和automated NeurIPS-style reviewer ensemble。毕业项目是建一个、$30每论文内run end to end、并survive Sakana documented sandbox-escape red team。

**类型:** 毕业项目
**语言:** Python (agent + sandbox)、LaTeX (output)
**前置要求:** 第2阶段(ML)、第3阶段(深度学习)、第7阶段(transformers)、第10阶段(LLMs from scratch)、第14阶段(agent)、第15阶段(自主)、第16阶段(多agent)、第18阶段(安全)
**涉及阶段:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间:** 40小时

## 问题背景

自研智能体2026过阈值。Sakana AI's AI-Scientist-v2发于Nature带generated papers workshop peer review clear。ShinkaEvolve (ICLR 2026)延line evolving hypotheses。AMD's Agent Laboratory ship reproducible traces。Agents非magic — 是candidate experiments tree上plan-execute-verify loop run、cost caps、seed-bound sandboxes、和automated review。Craft在loop、budget、和safety story。

你经对窄域seed idea (例如100M-parameter transformer attention-sparsity ablations) implement one学loop。Value非first run discover new。Value在infrastructure：tree-search、experiment sandbox、writer-reviewer loop、red-team report。Sakana team documented sandbox-escape failures；你agent须pass同red team。

## 概念讲解

Agent是best-first tree search。Nodes是experiment specifications：(hypothesis、config、code、expected outcome)。Expand step children small edits propose (swap optimizer、shift batch size、ablate component)。每child fresh sandbox hard resource cap run。Results feed回score function rank nodes (novelty × quality × remaining budget)。Tree budget exhaust grow、best branch write up。

Writer多模态。Generate LaTeX draft、compile、render figures、和feed rendered PDF back Claude Opus 4.7's vision mode critique layout、figure legibility、和claim-evidence alignment。Reviewer ensemble five LLM judges emit NeurIPS-style scores (novelty、rigor、clarity、reproducibility、impact)；若average threshold下drop、paper writer critique return。

Safety load-bearing。每experiment E2B或Daytona sandbox run无network egress、bounded wall-clock、和pinned resource limits。Agent code-generation step policy layer pass block syscall escape sandbox。Red-team report reproduce Sakana-documented attack surface (fork bombs、filesystem escapes、LLM-written network calls)。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- Orchestration：LangGraph带checkpointing和human-approval gates
- Tree search：custom best-first over experiment nodes (AB-MCTS-style from Sakana v2)
- Sandbox：E2B per experiment、Docker-in-Docker fallback；resource caps via cgroups
- Literature：Semantic Scholar Graph API + OpenAlex + local FAISS cache of abstracts
- Writer：LaTeX template + Claude Opus 4.7 (vision mode)于figure critique和layout
- Reviewer：5 judges ensemble (Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max) weighted aggregation
- Experiment framework：PyTorch 2.5于physical experiments、W&B于logging
- Observability：Langfuse于agent traces、$30 hard budget per paper

## 动手实践

1. **Seed和domain scoping。** 取seed idea (如"investigate sub-1B transformers attention map sparsity patterns")。定义search space：models、datasets、compute budget。

2. **Literature pass。** Query Semantic Scholar + OpenAlex 50 most-cited relevant papers；本地cache abstracts；generate 1-page domain digest。

3. **Tree scaffolding。** Initialize root seed hypothesis。Implement `expand(node) -> children` small-edit proposals (每child one config change)。Implement `score(node)` weighted novelty × quality × budget term。

4. **Sandbox wrapping。** 每experiment `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (或equivalent E2B policy) run。Seeds sandbox写；outputs read-only mount back out。

5. **Plan-execute-verify loop。** `plan` children propose。`execute` sandbox run、logs和metrics capture。`verify` metrics unit checks run (loss decrease? ablation isolate effect?)。Failed nodes failure reason tree store。

6. **Writer。** Budget后、select best branch。matplotlib figures render。Claude Opus 4.7 branch trace context LaTeX draft generate。Compile。Compiled PDF Opus 4.7 vision critique feed back。Iterate。

7. **Reviewer ensemble。** Five judges draft (novelty、rigor、clarity、reproducibility、impact) NeurIPS-style rubrics score。若mean < 4.0/5、writer critique return。3 rewrites hard stop。

8. **Red team。** Build或integrate sandbox adversarial tasks set：fork bombs、network exfiltration attempts、filesystem escapes、LLM-written shell metacharacters。Confirm all blocked。Findings write up。

9. **Reproducibility。** 每paper tree-search trace JSON、seeds、W&B run links、sandbox configs、和README end to end reproduce ship。

## 使用它

```
$ ai-scientist run --seed "sub-1B transformers attention sparsity" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 产出成果

`outputs/skill-ai-scientist.md`是deliverable。给seed idea + domain + $30 budget、run全pipeline并emit reviewable paper reproducibility bundle。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | Paper quality | Blind rubric review对published workshop papers |
| 20 | Experimental rigor | Baselines、seeds、ablations；每claim results table cell backed |
| 20 | Cost和compute discipline | $30/paper ceiling enforced、Langfuse-traced |
| 20 | Safety | Sandbox red team passes；network policy和kill-switch verified |
| 15 | Reproducibility | One-command rerun identical seeds reproduces paper |
| **100** | | |

## 练习题

1. 同域三不同seed ideas pipeline run。比tree-search parts overlap。Identify duplicated wasted compute。
2. $5以上nodes experiment execution前human-in-the-loop gate add。测total cost drop多少。
3. 单judge swap reviewer ensemble。held-out known-bad papers set测false-accept rate。
4. Network-exfiltration red team test introduce：agent writes code tries `curl` external address。`--network=none` policy blocks确认。Log attempt。
5. Flat random baseline (same budget、no expansion strategy) tree-search compare。Report novelty × quality gain。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Tree search | "AB-MCTS-style expansion" | novelty×quality×budget score experiment nodes best-first exploration |
| Sandbox | "Experiment isolation" | 无network、bounded CPU/memory、pinned seeds、read-only inputs container |
| Vision critique | "Render-then-read" | Compile paper PDF、feed PDF back VLM layout和claim-evidence critique |
| Reviewer ensemble | "Automated peer review" | Multiple LLM judges NeurIPS rubric paper score；weighted aggregate gates pipeline |
| Novelty score | "Is this new?" | 50-paper literature cache proximity penalizes heuristic |
| Cost ceiling | "$ budget" | Per paper total spend hard cap；Langfuse counters + pre-run estimates |
| Red team | "Sandbox-escape audit" | Policy wrong sandbox escape adversarial tasks |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考产研agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 参考orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — 参考experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — reviewer ensemble encodes rubric