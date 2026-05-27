# 毕业项目 09 —— 代码迁移智能体 (仓库级语言 / 运行时升级)

> Amazon MigrationBench (Java 8到17)和Google App Engine Py2-to-Py3 migrator设2026 bar。Moderne OpenRewrite于规模做确定性AST重写。Grit用codemod-style DSL靶向同问题。产模式结合两者: 确定性substrate安全重写加智能体层处理歧义案例、每分支sandbox build、和测试harness翻绿后PR开。毕业项目是迁移50真repo并发布带失败分类学的pass rate。

**类型:** 毕业项目
**语言:** Python (智能体)、Java / Python (目标)、TypeScript (仪表板)
**前置要求:** 第5阶段(NLP)、第7阶段(transformers)、第11阶段(LLM工程)、第13阶段(工具)、第14阶段(智能体)、第15阶段(自主)、第17阶段(基础设施)
**涉及阶段:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间:** 30小时

## 问题背景

大规模代码迁移是2026编码智能体最干净产应用之一。Ground truth明显(迁移后测试套pass否?)、奖励真实(Java-8 fleet迁移是headcount级项目)、benchmark公开(MigrationBench 50-repo subset)。Moderne OpenRewrite处理确定性侧。智能体层处理OpenRewrite recipe不能的一切: 歧义重写、build-system漂移、long-tail语法、传递依赖破坏。

将建智能体取Java 8 repo (或Python 2 repo)产green-CI迁移分支。将测pass rate、test-coverage保持、每repo成本、并建失败分类学。与确定性-only baseline side-by-side告知智能体价值实际所在。

## 概念讲解

流水线两层。**确定性substrate** (OpenRewrite Java、libcst Python)安全运行大量机械重写: imports、method signatures、null-safety edits、try-with-resources、deprecated API replacements。快且产可审计diff。**智能体层** (OpenAI Agents SDK或LangGraph over Claude Opus 4.7和GPT-5.4-Codex)处理recipe不能案例: build-file升级(Maven/Gradle/pyproject)、传递依赖冲突、test flakes、自定义annotations。

每repo得Daytona sandbox预装目标运行时。智能体迭代: run build、分类失败、应用修复、rerun。硬限: 每repo 30分钟、$8每repo、20智能体turn。若全测试pass且coverage delta非负、分支开PR。否则repo入失败类带证据。

失败分类学是deliverable。跨50 repo何broken? 传递deps? 自定义annotations? Build tool版本? Test flakes无关迁移? 每类得计数和范例diff。未来recipe作者可靶向前三。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性substrate: OpenRewrite (Java) 或 libcst (Python)
- 智能体: OpenAI Agents SDK或LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainers每分支、预装目标运行时(Java 17 / Python 3.12)
- Build系统: Maven、Gradle、uv (Python)
- Benchmark: Amazon MigrationBench 50-repo subset (Java 8到17)、Google App Engine Py2-to-Py3 repos
- 测试harness: 并行runner、coverage via Jacoco (Java)或coverage.py (Python)
- 可观测性: Langfuse + 每repo trace bundle带每diff chunk
- 仪表板: failure-taxonomy仪表板带每类计数和范例diff

## 动手实践

1. **Recipe pass。** 先运行OpenRewrite (Java)或libcst (Python) recipes。捕获70-80%机械迁移。作"recipe" commit提交。

2. **Build trial。** Daytona sandbox: 安装目标运行时、运行build。若绿跳至测试。若红移交智能体。

3. **智能体loop。** LangGraph带工具: `run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败(dep、syntax、test、build-tool)并应用靶向修复。Rerun。

4. **预算cap。** 每repo 30分钟wall-clock、$8成本、20智能体turn。任何breach halt并入"budget_exhausted"带当前diff。

5. **测试 + coverage门。** Build绿后运行测试套。比coverage与base repo。若coverage降超2%入"coverage_regression"。

6. **PR开。** 成功时push branch、开PR带diff和summary何recipe应用及何commit智能体author。

7. **失败分类学。** 每失败repo标记类: `dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。建仪表板。

8. **50-repo run。** 执行于MigrationBench subset。报每类pass rate、cost-per-repo、coverage-preservation、和compare-vs-deterministic-only baseline。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 产出成果

`outputs/skill-migration-agent.md`是deliverable。给repo、执行确定性recipe然后智能体loop产绿迁移分支、或repo入分类学类。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage保持 | Mean coverage delta vs base |
| 20 | 每迁移repo成本 | passing runs $/repo |
| 20 | 智能体 / 确定性工具集成 | OpenRewrite处理vs智能体author修复比例 |
| 15 | 失败分析write-up | 分类学完整性带范例 |
| **100** | | |

## 练习题

1. 仅OpenRewrite运行migrate流水线(无智能体)。比pass rate与全流水线。识智能体alone差异案例。

2. 实现"lint-clean"检查: 迁移后运行style linter (spotless Java、ruff Python)。若新lint error出PR fail。测coverage-preserved-but-style-regressed rate。

3. 加"minimal-diff"优化器: 智能体branch pass测试后、二pass trim不必要变更。报diff-size降。

4. 延至第三迁移: Node 18到Node 22。复用sandbox wrapping; 换recipe layer为自定义codemod。

5. 测time-to-first-green-build (TTFGB)作UX metric。目标: p50低于10分钟。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 确定性substrate | "Recipe engine" | OpenRewrite / libcst: 安全保证声明式AST重写 |
| Codemod | "代码修改程序" | 机械改源代码重写规则 |
| Build drift | "工具版本偏差" | 主版本间微妙Maven / Gradle / uv行为变化 |
| 失败类 | "分类桶" | repo未迁移标注原因: dep、syntax、test、build-tool、budget |
| Coverage delta | "Coverage保持" | base到迁移分支测试coverage %变化 |
| 智能体turn | "工具调用轮" | 智能体loop一plan -> act -> observe cycle |
| Budget exhaustion | "触及天花板" | repo耗尽30-min / $8 / 20-turn限未pass |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 确定性substrate参考
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — 备选codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK参考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 备选迁移benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python确定性substrate
- [Daytona sandboxes](https://daytona.io) — 参考每分支sandbox