# AI Engineering 课程翻译指南

## 翻译原则

### 1. 代码不翻译
- 所有代码块保持原样
- 变量名、函数名保持英文
- 注释可以翻译，但如果原文无注释则不加

### 2. Markdown格式保持
- 标题层级 (#, ##, ###) 保持
- 列表 (-, *) 保持格式
- 表格保持结构，只翻译内容
- Mermaid图表中的文本要翻译

### 3. 术语统一

| 英文术语 | 中文翻译 |
|---------|----------|
| AI/ML | AI/ML (不翻译) 或 人工智能/机器学习 |
| Vector | 向量 |
| Matrix | 矩阵 |
| Neural Network | 神经网络 |
| Deep Learning | 深度学习 |
| Transformer | Transformer (保留英文) |
| Embedding | 嵌入/词嵌入 |
| Token | Token/词元 |
| Attention | 注意力机制 |
| Loss Function | 损失函数 |
| Gradient Descent | 梯度下降 |
| Backpropagation | 反向传播 |
| Epoch | 轮次/迭代轮数 |
| Batch | 批次 |
| Hyperparameter | 超参数 |
| Fine-tuning | 微调 |
| Inference | 推理 |
| Training | 训练 |
| Dataset | 数据集 |
| Model | 模型 |
| Layer | 层 |
| Activation Function | 激活函数 |
| Optimization | 优化 |
| Regularization | 正则化 |
| Overfitting | 过拟合 |
| Underfitting | 欠拟合 |
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 Score | F1分数 |
| LLM (Large Language Model) | 大语言模型 |
| Prompt | 提示词/提示 |
| Agent | 智能体/代理 |
| RAG (Retrieval-Augmented Generation) | RAG/检索增强生成 |
| MCP (Model Context Protocol) | MCP |

### 4. 结构保持

每个课程文档都有固定结构：
1. 标题 (# Lesson Title)
2. 引言 (> motto)
3. 元数据 (**Type:**, **Languages:** 等)
4. ## Learning Objectives / 学习目标
5. ## The Problem / 问题背景
6. ## The Concept / 概念讲解
7. ## Build It / 动手实践
8. ## Use It / 实际应用
9. ## Ship It / 产出成果
10. ## Exercises / 练习题
11. ## Key Terms / 关键术语 (如果有)

### 5. 输出位置
- 原文: `phases/NN-phase/MM-lesson/docs/en.md`
- 译文: `phases/NN-phase/MM-lesson/docs/zh.md`

### 6. Quiz翻译
- question → 问题
- options → 选项数组
- explanation → 解释
- 保持JSON结构不变

## 检查清单

翻译完成后检查：
- [ ] 所有标题已翻译
- [ ] 所有段落已翻译
- [ ] 表格内容已翻译
- [ ] 图表文本已翻译
- [ ] 代码块未翻译（保持原样）
- [ ] 术语翻译一致
- [ ] 文件保存为 zh.md

## 示例

### 原文示例
```markdown
# Linear Algebra Intuition

> Every AI model is just matrix math wearing a fancy hat.

**Type:** Learn
**Languages:** Python, Julia
**Prerequisites:** Phase 0

## The Problem

Open any ML paper. Within the first page, you'll see vectors...
```

### 译文示例
```markdown
# 线性代数直觉

> 每个AI模型本质上都是戴着花哨帽子的矩阵数学。

**类型:** 学习
**语言:** Python, Julia
**前置要求:** 第0阶段

## 问题背景

打开任何机器学习论文。在第一页之内，你就会看到向量...
```
