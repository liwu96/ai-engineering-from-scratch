# API 与密钥

> 每个 AI API 的工作方式都一样：发送请求，获取响应。细节会变，模式不变。

**类型：** 构建
**语言：** Python, TypeScript
**前置要求：** 第 0 阶段，第 01 课
**时间：** 约 30 分钟

## 学习目标

- 使用环境变量和 `.env` 文件安全存储 API 密钥
- 使用 Anthropic Python SDK 和原始 HTTP 进行大语言模型 API 调用
- 比较基于 SDK 和原始 HTTP 的请求/响应格式以进行调试
- 识别和处理常见 API 错误，包括身份验证和速率限制

## 问题背景

从第 11 阶段开始，你将调用大语言模型 API（Anthropic、OpenAI、Google）。在第 13-16 阶段，你将构建在循环中使用这些 API 的智能体。你需要了解 API 密钥的工作原理、如何安全存储它们以及如何发起第一次 API 调用。

## 概念讲解

```mermaid
sequenceDiagram
    participant C as 你的代码
    participant S as API 服务器
    C->>S: HTTP 请求（带 API 密钥）
    S->>C: HTTP 响应（JSON）
```

每个 API 调用包含：
1. 端点（URL）
2. API 密钥（身份验证）
3. 请求体（你想要什么）
4. 响应体（你得到什么）

## 动手实践

### 步骤 1：安全存储 API 密钥

永远不要把 API 密钥放在代码中。使用环境变量。

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

或使用 `.env` 文件（添加到 `.gitignore`）：

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### 步骤 2：第一次 API 调用（Python）

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "用一句话解释什么是神经网络？"}]
)

print(response.content[0].text)
```

### 步骤 3：第一次 API 调用（TypeScript）

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "用一句话解释什么是神经网络？" }],
});

console.log(response.content[0].text);
```

### 步骤 4：原始 HTTP（不使用 SDK）

```python
import os
import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}
body = json.dumps({
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "用一句话解释什么是神经网络？"}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])
```

这就是 SDK 底层所做的工作。理解原始 HTTP 调用有助于调试。

## 实际应用

对于本课程：

| API | 何时需要 | 免费额度 |
|-----|----------|----------|
| Anthropic (Claude) | 第 11-16 阶段（智能体、工具） | 注册送 $5 |
| OpenAI | 第 11 阶段（对比） | 注册送 $5 |
| Hugging Face | 第 4-10 阶段（模型、数据集） | 免费 |

你现在不需要全部设置。课程需要时再设置。

## 产出成果

本节课生成：
- `outputs/prompt-api-troubleshooter.md` - 诊断常见 API 错误

## 练习题

1. 获取 Anthropic API 密钥并发起第一次 API 调用
2. 尝试原始 HTTP 版本并比较响应格式与 SDK 版本
3. 故意使用错误的 API 密钥并阅读错误消息

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| API 密钥 | "API 的密码" | 唯一字符串，标识你的账户并授权请求 |
| 速率限制 | "他们在限流我" | 每分钟/小时最大请求数，防止滥用并确保公平使用 |
| Token | "一个词"（API 语境） | 计费单位：输入和输出 token 分开计数和收费 |
| 流式传输 | "实时响应" | 逐字获取响应，而不是等待完整响应 |
