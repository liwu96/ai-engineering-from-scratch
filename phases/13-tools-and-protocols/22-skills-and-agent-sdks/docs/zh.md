# Skills和Agent SDK——Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP说"何工具存"。Skill说"何做task"。2026栈叠两。Anthropic Agent Skills(开标准,2025年12月)发货作SKILL.md带渐进披露。OpenAI Apps SDK是MCP加widget metadata。AGENTS.md(现60,000+ repo)坐repo根作项目级agent上下文。本课命名每覆盖何并建最小SKILL.md+AGENTS.md bundle跨agent旅。

**类型:** 学习
**语言:** Python(stdlib,SKILL.md解析器和loader)
**前置要求:** 阶段13课程07(MCP server)
**时间:** ~45分钟

## 学习目标

- 分三层:AGENTS.md(项目上下文)、SKILL.md(复用know-how)、MCP(工具)。
- 写带YAML frontmatter和渐进披露SKILL.md。
- 文件系统式加载skill入agent runtime。
- 合skill和MCP server及AGENTS.md使一包于Claude Code、Cursor、Codex工。

## 问题背景

工程师蒸馏release-notes写workflow入多步提示:"读最新合并PR。按area group。每summarize。按团队风格写changelog entry。Post至Slack draft。"放入Notion doc给team。

现欲从Claude Code、Cursor、Codex CLI用此workflow。每agent异式load指令:Claude Code slash-command、Cursor rule、Codex`.codex.md`。工程师复制workflow三次并维三copy。

AGENTS.md和SKILL.md共修复此:

- **AGENTS.md**坐repo根。每兼容agent session start读。"此项目何工?何约定?何命令跑test?"
- **SKILL.md**是便携bundle:YAML frontmatter(name、description)+markdown body+可选resource。支持skill agent按需按名load。
- **MCP**(阶段13课程06-14)处理skill需调工具。

三层,一便携artifact。

## 概念讲解

### AGENTS.md(agents.md)

2025年底发,2026年4月60,000+ repo采纳。Repo根一文件。格式:

```markdown
# Project: my-service

## 约定
- TypeScript带strict mode。
- Python侧用Pydantic模型。
- Tests `pnpm test`跑。

## Build和run
- `pnpm dev`本地dev server。
- `pnpm build`产bundle。
```

Agent session start读此并用它为该项目校行为。2026每coding agent支持AGENTS.md:Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md格式

Anthropic Agent Skills(2025年12月开标准发):

```markdown
---
name: release-notes-writer
description: 为最新合并PR按项目风格写changelog entry。
---

# Release notes writer

调用时,跑这些步:

1. List last tag后合并PR。用`gh pr list --base main --state merged`。
2. 按label group:feature、fix、chore、docs。
3. 每group每PR,写一行:`- <title> (#<num>)`。
4. Draft release notes并stage CHANGELOG.md。

若用户说"ship",跑`git tag vX.Y.Z`和`gh release create`。

## Notes

- 勿含无PR commit。
- 从公开changelog跳"chore"条。
```

Frontmatter声明skill identity。Body是skill load时模型见提示。

### 渐进披露

Skill可引用sub-resource agent仅需时fetch。例:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md说"见style-guide.md用于style规则"。Agent仅skill active run时pull style-guide.md。这避bloat提示于模型不需细节。

### 文件系统发现

Agent runtime scan知目录SKILL.md文件:

- `~/.anthropic/skills/*/SKILL.md`
- 项目`./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

Load按folder名和frontmatter `name`。Claude Code、Anthropic Claude Agent SDK、SkillKit(cross-agent)皆follow此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`(TypeScript)和`claude-agent-sdk`(Python)session start load skill,露作runtime内callable"agent"。Agent loop用户调用时dispatch至skill。

### OpenAI Apps SDK

2025年10月发;直建于MCP。统OpenAI前Connectors和Custom GPT Actions至单developer surface。Apps SDK app是:

- MCP server(工具、资源、提示)。
- 加ChatGPT UI widget metadata。
- 加可选MCP Apps`ui://`resource用于交互面。

同协议,更富UX。

### 跨agent便携经SkillKit

工具如SkillKit和类似跨agent分发层译单SKILL.md入32+ AI agent(Claude Code、Cursor、Codex、Gemini CLI、OpenCode等)原生格式。一真源;多consumer。

### 三层栈

| 层 | 文件 | 何时load | 目 |
|----|------|----------|----|
| AGENTS.md | repo根 | session start | 项目级约定 |
| SKILL.md | skills目录 | skill调用 | 复用workflow |
| MCP server | 外进程 | 需工具 | 可调动作 |

三层合:agent session start读AGENTS.md,用户调skill,skill指令含MCP工具调用,agent经MCP client dispatch。

## 使用

`code/main.py`发stdlib SKILL.md解析器和loader。发现`./skills/`下skill,解析YAML frontmatter加markdown body,产按skill名keyed dict。后模拟agent loop按名调用`release-notes-writer`。

看点:

- YAML frontmatter用最小stdlib解析器解析(无`pyyaml`依赖)。
- Skill body存原文;agent调用时prepend至系统提示。
- 渐进披露经`read_subresource`函数demo按需pull引用文件。

## 交付成果

本课产`outputs/skill-agent-bundle.md`。给workflow,skill产合SKILL.md+AGENTS.md+MCP-server-blueprint bundle,跨agent便携。

## 练习题

1. 跑`code/main.py`。`skills/`下加第二skill并验loader pick。

2. 为此课程repo写AGENTS.md。含test命令、风格约定、和Phase 13心智模型。

3. 将你团队内部文档多步workflow移植入SKILL.md。验Claude Code load。

4. 手译skill入Cursor和Codex原生rule格式。计格式间diff——此是SkillKit自动化翻译面。

5. 读Anthropic Agent Skills blog post。识Claude Agent SDK一特性课程loader未覆盖。(提示:agent sub-invocation。)

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| SKILL.md | "Skill文件" | YAML frontmatter加markdown body,agent runtime load |
| AGENTS.md | "Repo根agent上下文" | Session start读项目级约定文件 |
| 渐进披露 | "Lazy-load sub-resource" | Skill body引用文件仅需时pull |
| Frontmatter | "顶YAML块" | `---`分隔符metadata(name、description) |
| Claude Agent SDK | "Anthropic skill runtime" | `@anthropic-ai/claude-agent-sdk`,load skill路由 |
| OpenAI Apps SDK | "MCP+widget meta" | OpenAI dev surface建于MCP加ChatGPT UI hook |
| Skill发现 | "文件系统scan" | Walk知dir寻SKILL.md,按名key |
| 跨agent便携 | "一skill多agent" | 经SkillKit式工具译一SKILL.md至32+ agent |
| Agent Skill | "便携know-how" | MCP工具概念外复用task模板 |
| Apps SDK | "MCP加ChatGPT UI" | Connectors和Custom GPT于MCP统 |

## 延伸阅读

- [Anthropic—Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)——2025年12月发
- [Anthropic—Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)——SKILL.md格式参考
- [OpenAI—Apps SDK](https://developers.openai.com/apps-sdk)——ChatGPT MCP基developer平台
- [agents.md](https://agents.md/)——AGENTS.md格式和采纳列表
- [Anthropic—anthropics/skills GitHub](https://github.com/anthropics/skills)——官方skill例