# 毕业项目 10 —— 多智能体软件工程团队

> SWE-AF工厂架构、MetaGPT角色基提示、AutoGen 0.4类型actor图、Cognition Devin、和Factory Droids皆收敛于同2026形态: architect计划、N coder并行worktree工作、reviewer门控、tester验证。并行worktree将wall-clock转为吞吐。共享状态和移交协议成失败面。毕业项目是建团队、于SWE-bench Pro评估、并报何移交破及频率。

**类型:** 毕业项目
**语言:** Python / TypeScript (智能体)、Shell (worktree脚本)
**前置要求:** 第11阶段(LLM工程)、第13阶段(工具)、第14阶段(智能体)、第15阶段(自主)、第16阶段(多智能体)、第17阶段(基础设施)
**涉及阶段:** P11 · P13 · P14 · P15 · P16 · P17
**时间:** 40小时

## 问题背景

单智能体编码harness于大任务触及天花板。非因任何个体智能体弱、但因200k-token context无法持架构计划加四并行代码库slice加reviewer评论加测试输出。多智能体工厂拆问题: architect持计划、coder持并行worktree实现、reviewer门控、tester验证。SWE-AF"工厂"架构、MetaGPT角色、AutoGen类型actor图 — 三framing描述同形态。

失败面是移交。Architect计划coder无法实现。Coder产冲突diff。Reviewer批准幻觉修复。Tester race仍写coder。将建此类团队之一、于50 SWE-bench Pro issue运行、跟踪每移交、并发布post-mortem。

## 概念讲解

角色是类型智能体。**Architect** (Claude Opus 4.7)读issue、写计划、并拆子任务带明确接口。**Coders** (Claude Sonnet 4.7、N并行实例、各于`git worktree` + Daytona sandbox)独立实现子任务。**Reviewer** (GPT-5.4)读合并diff或批准或请求具体变更。**Tester** (Gemini 2.5 Pro)隔离运行测试套并报pass/fail带artifacts。

通信经共享任务板(file-backed或Redis)。每角色消费允许处理任务。移交是A2A-protocol-typed消息。协调关注: merge-conflict解决(coordinator角色或自动三路merge)、共享状态同步(coder开始后计划冻结; replan是独立事件)、reviewer门控(reviewer不可批准自己变更或提议变更)。

Token放大是隐藏成本。每角色边界加摘要提示和移交context。40-turn单智能体run变四角色160总turn。Rubric特定权衡token效率vs单智能体baseline因问题非"多智能体工作否"而"每美元赢否"。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排: LangGraph带共享状态 + 每智能体子图
- 消息: A2A protocol (Google 2025)类型智能体间消息
- 模型: Opus 4.7 (architect)、Sonnet 4.7 (coders)、GPT-5.4 (reviewer)、Gemini 2.5 Pro (tester)
- Worktree隔离: 每coder `git worktree add` + Daytona sandbox
- Merge coordinator: 自定义三路merge + LLM-mediated冲突解决
- 评估: SWE-bench Pro (50 issues)、SWE-AF场景、HumanEval++单元测试
- 可观测性: Langfuse带角色标签span、每智能体token会计
- 部署: K8s每角色独立Deployment + HPA on backlog

## 动手实践

1. **任务板。** File-backed JSONL带类型消息: `plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **Architect。** 读GitHub issue、运行Opus 4.7带计划模板需明确子任务接口(files touched、public functions、test impact)。发一`plan_request`带子任务DAG。

3. **Coders。** N并行worker、各从板claim一子任务。各spawn fresh `git worktree add` branch加Daytona sandbox。实现子任务。发`diff_ready`带patch + test deltas。

4. **Merge coordinator。** 所有coder-done时、三路merge N branches入staging branch。仅文件级overlap时LLM-mediated冲突解决。

5. **Reviewer。** GPT-5.4读合并diff。不可批准自己author的diff。发`approved` (no-op)或`review_feedback`带具体变更请求路由回相关coder。

6. **Tester。** Gemini 2.5 Pro干净sandbox运行测试套。捕获artifacts。发`test_passed`或`test_failed`带stacktraces。失败测试loop回own失败子任务coder。

7. **移交会计。** 每跨角色边界消息得Langfuse span带payload大小和所用模型。算每子任务token放大 (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens)。

8. **评估。** 于50 SWE-bench Pro issue运行。比pass@1和$-per-solved-issue与单智能体baseline (一Sonnet 4.7于单worktree)。

9. **Post-mortem。** 每失败issue、识破移交(计划太模糊、merge冲突、reviewer假批准、tester flake)。产移交失败histogram。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 产出成果

`outputs/skill-multi-agent-team.md`是deliverable。给issue URL和并行度、团队产merge-ready PR带每角色token会计。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配50-issue subset、pass@1 |
| 20 | 并行加速 | Wall-clock vs单智能体baseline |
| 20 | Review质量 | 注入bug probe假批准率 |
| 20 | Token效率 | 每解决issue总tokens vs单智能体 |
| 15 | 协调工程 | Merge-conflict解决、移交失败histogram |
| **100** | | |

## 练习题

1. mid-run注入明显bug(主body前额外`return None`)。测reviewer假批准率。调reviewer prompt至假批准低于5%。

2. 减至二coder (architect + coder + reviewer + tester、coder顺序运行二子任务)。比wall-clock和pass rate。

3. 替merge coordinator为single-writer约束(子任务touch不交集文件)。测architect规划负担。

4. 换reviewer从GPT-5.4为Claude Opus 4.7。测假批准率和token成本delta。

5. 加第五角色: documenter (Haiku 4.5)。Review后产changelog entry。测文档质量是否justify额外token spend。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 并行worktree | "隔离branch" | 每coder `git worktree add`产fresh working tree |
| 任务板 | "共享消息总线" | File或Redis存智能体订阅类型消息 |
| 移交 | "角色边界" | 从一角色context跨至另一角色的任何消息 |
| Token放大 | "多智能体开销" | 跨角色总tokens / 同任务单智能体tokens |
| A2A protocol | "智能体间" | Google 2025类型智能体间消息spec |
| Merge coordinator | "集成器" | 运行三路merge并mediates冲突的组件 |
| 假批准 | "Reviewer幻觉" | Reviewer批准带已知bug的diff |

## 延伸阅读

- [SWE-AF工厂架构](https://github.com/Agent-Field/SWE-AF) — 参考2026多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 角色基多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft类型actor框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — 智能体间消息spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔离substrate
- [SWE-bench Pro](https://www.swebench.com) — 评估目标