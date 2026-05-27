# 数据管理

> 数据是燃料。你如何管理它决定了你的速度。

**类型：** 构建
**语言：** Python
**前置要求：** 第 0 阶段，第 01 课
**时间：** 约 45 分钟

## 学习目标

- 使用 Hugging Face `datasets` 库加载、流式和缓存数据集
- 在 CSV、JSON、Parquet 和 Arrow 格式之间转换并解释其权衡
- 使用固定随机种子创建可复现的训练/验证/测试划分
- 使用 `.gitignore`、Git LFS 或 DVC 管理大型模型和数据集文件

## 问题背景

每个 AI 项目都从数据开始。你需要找到数据集、下载它们、在格式之间转换、为训练和评估划分它们，并对它们进行版本控制使实验可复现。每次都手动做是缓慢且容易出错的。你需要一个可重复的工作流程。

## 概念讲解

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets 库"]
    B --> C["加载 / 流式"]
    C --> D["本地缓存<br/>~/.cache/huggingface/"]
    B --> E["格式转换<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["数据划分<br/>train / val / test"]
    F --> G["你的训练管道"]
```

Hugging Face `datasets` 库是 AI 工作中加载数据的标准方式。它开箱即用处理下载、缓存、格式转换和流式传输。

## 动手实践

### 步骤 1：安装 datasets 库

```bash
pip install datasets huggingface_hub
```

### 步骤 2：加载数据集

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])
```

这会下载 IMDB 电影评论数据集。首次下载后，它从 `~/.cache/huggingface/datasets/` 的缓存加载。

### 步骤 3：流式大型数据集

有些数据集太大无法放在磁盘上。流式逐行加载它们而无需下载完整内容。

```python
dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

流式给你一个 `IterableDataset`。数据到达时处理行。无论数据集大小，内存使用保持恒定。

### 步骤 4：数据集格式

`datasets` 库内部使用 Apache Arrow。你可以根据管道需要转换为其他格式。

```python
dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

格式对比：

| 格式 | 大小 | 读取速度 | 最适合 |
|------|------|----------|--------|
| CSV | 大 | 慢 | 人类可读性、电子表格 |
| JSON | 大 | 慢 | API、嵌套数据 |
| Parquet | 小 | 快 | 分析、列式查询 |
| Arrow | 小 | 最快 | 内存处理（datasets 内部使用） |

对于 AI 工作，Parquet 是最佳存储格式。Arrow 是你在内存中使用的。CSV 和 JSON 用于交换。

### 步骤 5：数据划分

每个 ML 项目需要三个划分：

- **训练**：模型从这学习（通常 80%）
- **验证**：你在训练期间检查进度（通常 10%）
- **测试**：训练完成后最终评估（通常 10%）

有些数据集预划分。当没有时，自己划分：

```python
dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"训练: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")
```

始终设置种子以保证可复现性。相同种子每次产生相同划分。

### 步骤 6：下载和缓存模型

模型是大文件。`huggingface_hub` 库处理下载和缓存。

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"缓存于: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"完整模型在: {model_dir}")
```

模型缓存到 `~/.cache/huggingface/hub/`。下载后，后续运行立即加载。

### 步骤 7：处理大文件

模型权重和大数据集不应该进 git。三个选项：

**选项 A：.gitignore（最简单）**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**选项 B：Git LFS（在 git 中跟踪大文件）**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS 在你的仓库中存储指针，实际文件在单独服务器上。GitHub 免费给你 1 GB。

**选项 C：DVC（数据版本控制）**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "用 DVC 跟踪训练数据"
```

DVC 创建指向你数据的小 `.dvc` 文件。数据本身存在于 S3、GCS 或另一个远程存储后端。

| 方法 | 复杂度 | 最适合 |
|------|--------|--------|
| .gitignore | 低 | 个人项目、可以重新获取的下载数据 |
| Git LFS | 中 | 通过 git 共享模型权重的团队 |
| DVC | 高 | 可复现实验、大数据集、团队 |

对于本课程，`.gitignore` 足够了。当你需要在机器之间复现确切实验时使用 DVC。

### 步骤 8：存储模式

**本地存储** 适用于 ~10 GB 以下的数据集。HF 缓存自动处理这个。

**云存储** 适用于更大或跨机器共享的：

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC 直接与 S3 和 GCS 集成：

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

对于本课程，本地存储足够了。当你在远程 GPU 实例上微调时，云存储变得相关。

## 本课程使用的数据集

| 数据集 | 课程 | 大小 | 教授什么 |
|--------|------|------|----------|
| IMDB | 分词、分类 | 84 MB | 文本分类基础 |
| WikiText | 语言建模 | 181 MB | 下一 token 预测 |
| SQuAD | 问答系统 | 35 MB | 问答、跨度 |
| Common Crawl（子集） | 嵌入 | 变化 | 大规模文本处理 |
| MNIST | 视觉基础 | 21 MB | 图像分类基础 |
| COCO（子集） | 多模态 | 变化 | 图像-文本对 |

你现在不需要下载所有这些。每节课指定需要什么。

## 实际应用

运行工具脚本验证一切正常：

```bash
python code/data_utils.py
```

这下载一个小数据集，转换它，划分它，并打印摘要。

## 产出成果

本节课生成：
- `code/data_utils.py` - 可复用的数据加载和缓存工具
- `outputs/prompt-data-helper.md` - 用于为任务找到正确数据集的提示词

## 练习题

1. 加载带 `mrpc` 配置的 `glue` 数据集并检查前 5 个示例
2. 流式传输 `c4` 数据集并计数 10 秒内能处理多少示例
3. 将数据集转换为 Parquet 并比较与 CSV 的文件大小
4. 创建 70/15/15 训练/验证/测试划分，固定种子并验证大小

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 数据集划分 | "训练数据" | ML 生命周期不同阶段使用的命名子集（训练/验证/测试） |
| 流式 | "惰性加载" | 从远程源逐行处理数据而无需下载完整数据集 |
| Parquet | "压缩 CSV" | 为分析查询和存储效率优化的列式文件格式 |
| Arrow | "快速 dataframe" | `datasets` 库内部用于零拷贝读取的内存列式格式 |
| Git LFS | "大文件的 git" | 将大文件存储在 git 仓库外同时保留版本控制中指针的扩展 |
| DVC | "数据的 git" | 与云存储集成的数据集和模型版本控制系统 |
| 缓存 | "已下载" | 先前获取数据的本地副本，默认存储在 ~/.cache/huggingface/ |
