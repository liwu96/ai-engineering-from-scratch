# 终端与 Shell

> 终端是 AI 工程师的家。在这里要感到舒适。

**类型：** 学习
**语言：** --
**前置要求：** 第 0 阶段，第 01 课
**时间：** 约 35 分钟

## 学习目标

- 使用管道、重定向和 `grep` 从命令行过滤和处理训练日志
- 创建带多个窗格的持久 tmux 会话，用于并发训练和 GPU 监控
- 用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统与 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地和远程机器之间传输文件

## 问题背景

你在终端中度过的时间比任何编辑器都多。训练运行、GPU 监控、日志跟踪、远程 SSH 会话、环境管理。每个 AI 工作流程都接触 shell。如果你在这里慢，你哪里都慢。

这节课涵盖 AI 工作需要的终端技能。没有 Unix 历史。没有深入 Bash 脚本。只需要你需要的。

## 概念讲解

```mermaid
graph TD
    subgraph tmux["tmux 会话：训练"]
        subgraph top["顶行"]
            P1["窗格 1：训练运行<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["窗格 2：GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | 内存: 14/24G"]
        end
        P3["窗格 3：日志 + 实验<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时运行。一个终端。你可以分离，回家，SSH 回来，重新连接。训练继续运行。

## 动手实践

### 步骤 1：了解你的 shell

检查你运行的是哪个 shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`。两者都好。本课程的命令在两者中都适用。

要知道的关键事项：

```bash
# 移动
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# 历史搜索（你会学到的最有用快捷键）
# Ctrl+R 然后输入之前命令的一部分
# 再按 Ctrl+R 循环匹配项

# 清除终端
clear   # 或 Ctrl+L

# 取消运行中的命令
# Ctrl+C

# 暂停运行中的命令（用 fg 恢复）
# Ctrl+Z
```

### 步骤 2：管道和重定向

管道连接命令。这是你处理日志、过滤输出和链接工具的方式。你会不断使用这个。

```bash
# 统计 "loss" 在日志中出现多少次
cat train.log | grep "loss" | wc -l

# 从训练输出中提取 loss 值
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# 实时跟踪日志文件更新，过滤错误
tail -f train.log | grep --line-buffered "ERROR"

# 按最终准确率排序实验
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# 重定向 stdout 和 stderr 到不同文件
python train.py > output.log 2> errors.log

# 重定向两者到同一文件
python train.py > train_full.log 2>&1
```

你需要的三个重定向：

| 符号 | 作用 |
|------|------|
| `>` | 写 stdout 到文件（覆盖） |
| `>>` | 追加 stdout 到文件 |
| `2>` | 写 stderr 到文件 |
| `2>&1` | 发送 stderr 到 stdout 相同的地方 |
| `\|` | 发送一个命令的 stdout 作为下一个的 stdin |

### 步骤 3：后台进程

训练运行需要数小时。你不想一直开着终端。

```bash
# 后台运行（输出仍到终端）
python train.py &

# 后台运行，免疫挂起（关闭终端不会杀死它）
nohup python train.py > train.log 2>&1 &

# 检查后台运行什么
jobs
ps aux | grep train.py

# 把后台作业带到前台
fg %1

# 杀死后台进程
kill %1
# 或找到 PID 杀死它
kill $(pgrep -f "train.py")
```

`&`、`nohup` 和 `screen`/`tmux` 的区别：

| 方法 | 终端关闭后存活？ | 可以重新连接？ |
|------|----------------|--------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（检查日志文件） |
| `screen` / `tmux` | 是 | 是 |

任何超过几分钟的，用 tmux。

### 步骤 4：tmux

tmux 让你创建带多个窗格的持久终端会话。这是管理训练运行最有用的单一工具。

```bash
# 安装
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# 启动命名会话
tmux new -s training

# 水平分割
# Ctrl+B 然后 "

# 垂直分割
# Ctrl+B 然后 %

# 在窗格之间导航
# Ctrl+B 然后方向键

# 分离（会话继续运行）
# Ctrl+B 然后 d

# 重新连接
tmux attach -t training

# 列出会话
tmux ls

# 杀死会话
tmux kill-session -t training
```

典型的 AI 工作流程会话：

```bash
tmux new -s train

# 窗格 1：启动训练
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " 分割，然后运行 GPU 监控
watch -n1 nvidia-smi

# Ctrl+B, % 垂直分割，跟踪日志
tail -f logs/experiment.log

# 现在用 Ctrl+B, d 分离
# SSH 退出，去喝咖啡，回来
# tmux attach -t train
```

### 步骤 5：用 htop 和 nvtop 监控

```bash
# 系统进程（比 top 好）
htop

# GPU 进程（如果你有 NVIDIA GPU）
# 安装：sudo apt install nvtop (Ubuntu) 或 brew install nvtop (macOS)
nvtop

# 不用 nvtop 快速 GPU 检查
nvidia-smi

# 每秒观看 GPU 使用更新
watch -n1 nvidia-smi

# 看哪些进程在使用 GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

`htop` 你会用的键绑定：
- `F6` 或 `>` 按列排序（按内存排序找内存泄漏）
- `F5` 切换树视图（看子进程）
- `F9` 杀死进程
- `/` 搜索进程名

### 步骤 6：远程 GPU 机器的 SSH

当你租用云 GPU（Lambda、RunPod、Vast.ai）时，你通过 SSH 连接。

```bash
# 基本连接
ssh user@gpu-box-ip

# 用特定密钥
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# 复制文件到远程
scp model.pt user@gpu-box-ip:~/models/

# 从远程复制文件
scp user@gpu-box-ip:~/results/metrics.json ./

# 同步整个目录（多文件时比 scp 快）
rsync -avz ./data/ user@gpu-box-ip:~/data/

# 端口转发（本地访问远程 Jupyter/TensorBoard）
ssh -L 8888:localhost:8888 user@gpu-box-ip
# 现在在浏览器打开 localhost:8888

# SSH 配置方便使用
# 添加到 ~/.ssh/config：
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# 然后只需：
# ssh gpu
```

### 步骤 7：AI 工作的有用别名

把这些添加到你的 `~/.bashrc` 或 `~/.zshrc`：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

或复制你想要的。关键别名：

```bash
# 一眼查看 GPU 状态
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# 杀死所有 Python 训练进程
alias killtraining='pkill -f "python.*train"'

# 快速虚拟环境激活
alias ae='source .venv/bin/activate'

# 观察训练 loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整集合见 `code/shell_aliases.sh`。

### 步骤 8：常见 AI 终端模式

这些在实践中反复出现：

```bash
# 运行训练，记录所有，完成时通知
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# 并排比较两个实验日志
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# 找最大模型文件（清理磁盘空间）
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# 从 Hugging Face 下载模型
wget https://huggingface.co/model/resolve/main/model.safetensors

# 解压数据集
tar xzf dataset.tar.gz -C ./data/

# 统计所有 Python 文件行数（看你的项目有多大）
find . -name "*.py" | xargs wc -l | tail -1

# 检查磁盘空间（训练数据快速填满磁盘）
df -h
du -sh ./data/*

# 训练前检查环境变量
env | grep -i cuda
env | grep -i torch
```

## 实际应用

这是本课程中每个工具的使用场景：

| 工具 | 何时使用 |
|------|----------|
| tmux | 每次训练运行（第 3 阶段+） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 调试慢训练、OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 重复命令节省时间 |

## 练习题

1. 安装 tmux，创建带三个窗格的会话，在其中一个运行 `htop`，另一个运行 `watch -n1 date`，第三个运行 Python 脚本。分离并重新连接。
2. 把 `code/shell_aliases.sh` 中的别名添加到你的 shell 配置，用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 创建假训练日志，然后用 `grep`、`tail` 和 `awk` 提取 loss 值。
4. 为你可以访问的服务器设置 SSH 配置条目（或用 `localhost` 练习语法）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Shell | "终端" | 解释你命令的程序（bash、zsh、fish） |
| tmux | "终端复用器" | 让你在一个窗口中运行多个终端会话，并分离/重新连接的程序 |
| 管道 | "那个竖线" | `\|` 操作符，发送一个命令的输出作为另一个的输入 |
| PID | "进程 ID" | 分配给每个运行进程的唯一数字，用于监控或杀死它 |
| nohup | "不挂起" | 运行免疫挂起信号的命令，所以关闭终端不会杀死它 |
| SSH | "连接到服务器" | Secure Shell，在远程机器上运行命令的加密协议 |
