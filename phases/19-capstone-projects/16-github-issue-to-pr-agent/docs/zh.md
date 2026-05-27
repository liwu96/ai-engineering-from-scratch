# 毕业项目 16 —— GitHub Issue到PR Autonomous智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex云、和Google Jules皆ship同2026产品形态: 标issue、得PR。云sandbox跑智能体、验证测试pass、并post review-ready PR带rationale。难点是自动复现repo build环境、防credential泄漏、强制每repo预算、并确保智能体不可force-push。毕业项目建自托管版本并于成本和pass rate比托管备选。

**类型:** 毕业项目
**语言:** Python (智能体)、TypeScript (GitHub App)、YAML (Actions)
**前置要求:** 第11阶段(LLM工程)、第13阶段(工具)、第14阶段(智能体)、第15阶段(自主)、第17阶段(基础设施)
**涉及阶段:** P11 · P13 · P14 · P15 · P17
**时间:** 30小时

## 问题背景

异步云编码智能体是独立产品类不同于交互编码智能体(毕业项目01)。UX是GitHub label。标issue `@agent fix this`、worker云sandbox spin up、clone repo、run tests、edit files、verify、并开PR带智能体rationale于body。无交互loop、无terminal。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex云、Google Jules、和Factory Droids皆收敛于此。

工程挑战具体: 环境复现(智能体须从头build repo无cached dev image)、flaky tests(须重跑或隔离)、credential scoping(最小fine-grained权限GitHub App)、预算强制每repo每天、和无force-push policy。毕业项目测pass rate、成本、和安全vs托管备选。

## 概念讲解

触发是GitHub webhook (issue label或PR comment)。Dispatcher enqueue工作到ECS Fargate或Lambda。Worker拉repo入Daytona或E2B sandbox带从repo推断通用Dockerfile (语言、框架)。智能体运行mini-swe-agent或SWE-agent v2 loop对Claude Opus 4.7或GPT-5.4-Codex。迭代: 读代码、提修复、应用patch、run tests。

验证是gating步骤。完整CI须sandbox内pass后PR开。Coverage delta算; 若negative超阈值、PR开但得标签`needs-review`。智能体post rationale作PR描述加`@agent` thread reviewer可ping follow-ups。

安全经两不同GitHub面scoped: App提供短期installation token带`workflows: read`和窄repo contents/PR scopes; branch protection (非app权限)强制"无直接写`main`"和"无force-push" — app绝不加bypass list。`.github/workflows`路径scoped只读访问非真实GitHub App primitive、故智能体文件edit allow-list须worker强制。每repo每天预算ceilings于dispatcher强制(如每天每repo最多5 PRs、每PR $20)。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发: GitHub App带fine-grained token; webhook receiver via Lambda或Fly.io
- Worker: ECS Fargate task (或GitHub Actions self-hosted runner)
- Sandbox: Daytona devcontainer或E2B sandbox每任务
- 智能体loop: mini-swe-agent baseline或SWE-agent v2 over Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: tree-sitter repo-map + ripgrep
- 验证: sandbox内完整CI + coverage delta gate
- 可观测性: Langfuse带每PR trace archive从PR body链接
- 预算: 每repo每天美元ceiling; 每repo每天最多PRs

## 动手实践

1. **GitHub App。** Fine-grained installation token: issues read+write、pull_requests write、contents read+write、workflows read。Branch protection (唯一能做此面)强制"无直接push `main`"和"无force-push"; app不在bypass list。Worker强制"无写`.github/workflows`"作proposed diff allow-list检查、因GitHub App权限非路径scoped。

2. **Webhook接收器。** Lambda函数接受issue label / PR comment webhook。按label `@agent fix this`过滤。Enqueue到SQS。

3. **Dispatcher。** Pop SQS任务。强制每repo每天预算。Spin up ECS Fargate task带repo URL、issue body、和fresh Daytona sandbox。

4. **环境推断。** 检语言(Python、Node、Go、Rust)和package manager (uv、pnpm、go mod、cargo)。若不存在则飞generate Dockerfile。

5. **智能体loop。** mini-swe-agent或SWE-agent v2带Claude Opus 4.7。工具: ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限: $20成本、30分钟wall-clock、30智能体turns。

6. **验证。** Loop结束后、sandbox内运行完整测试套。经jacoco / coverage.py算coverage delta。若CI红: halt、不开PR。若coverage降超2%: 开PR带`needs-review`标签。

7. **PR发布。** Push智能体branch。经GitHub API开PR带: title、rationale、diff summary、trace URL、成本、turns。

8. **Credential卫生。** Worker运行带短期GitHub App installation token。日志secret scrub后archival。

9. **评估。** 30 seeded内部issues难度多样。测pass rate、PR质量(diff size、style、coverage)、成本、延迟。比同issues上Cursor Background Agents和AWS Remote SWE Agents。

## 使用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 产出成果

`outputs/skill-issue-to-pr.md`是deliverable。GitHub App + 异步云worker将标注issues转review-ready PRs带bounded cost和scoped credentials。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 30 issues pass rate | 端到端成功(CI绿 + coverage OK) |
| 20 | PR质量 | Diff size、coverage delta、style conformance |
| 20 | 每解决issue成本和延迟 | 每PR $和wall-clock |
| 20 | 安全 | Scoped token、每repo预算、无force-push、credential卫生 |
| 15 | Operator UX | Rationale评论、retry affordance、@-mention follow-up |
| **100** | | |

## 练习题

1. 加"fix flaky test"模式: label `@agent stabilize-flake TestX` sandbox内run test 50次并提最小稳定化change。

2. 比三共享issues成本vs Cursor Background Agents。报何工具何处赢。

3. 实现预算仪表板: 每repo每天成本、每用户成本。异常alert。

4. 建"dry-run"模式开draft PR无run CI、以便reviewers便宜检查计划。

5. 加retention policy: 7天未merge PR branches自动删。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GitHub App | "Scoped bot identity" | App带fine-grained权限 + 短期installation token |
| 异步云智能体 | "Background agent" | 云sandbox非terminal运行非交互worker |
| 环境推断 | "Dockerfile synthesis" | 检语言 + package manager、若缺generate Dockerfile |
| 验证 | "CI-in-sandbox" | 开PR前worker内运行完整测试套 |
| Coverage delta | "Coverage保持" | base到智能体branch测试coverage %变化 |
| 每repo预算 | "每日ceiling" | Dispatcher强制美元和PR-count cap |
| Rationale | "PR body解释" | 智能体何变及为何summary; PR body必 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — canonical异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业备选
- [OpenAI Codex (云)](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google托管版
- [Factory Droids](https://www.factory.ai) — 备选商业参考
- [GitHub App文档](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona云sandboxes](https://daytona.io) — 参考sandbox