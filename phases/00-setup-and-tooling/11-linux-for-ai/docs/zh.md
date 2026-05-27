# AI 中的 Linux

> 大多数 AI 在 Linux 上运行。你需要知道足够的知识以免被困住。

**类型：** 学习
**语言：** --
**前置要求：** 第 0 阶段，第 01 课
**时间：** 约 30 分钟

## 学习目标

- 导航 Linux 文件系统并从命令行执行基本文件操作
- 用 `chmod` 和 `chown` 管理文件权限以解决"权限被拒绝"错误
- 用 `apt` 安装系统包并为 AI 工作设置新的 GPU 机器
- 识别 macOS 到 Linux 的差异，这些差异经常让在远程机器上工作的开发者绊倒

## 问题背景

你在 macOS 或 Windows 上开发。但当你 SSH 到云 GPU 盒子、租用 Lambda 实例或启动 EC2 机器时，你进入 Ubuntu。终端是你唯一的界面。没有 Finder，没有 Explorer，没有 GUI。如果你无法从命令行导航文件系统、安装包和管理进程，你就被困住了，为空闲的 GPU 小时付费，同时谷歌搜索"如何在 Linux 中解压文件"。

这是生存指南。它涵盖了在远程 Linux 机器上进行 AI 工作所需的内容。仅此而已。

## 文件系统布局

Linux 把所有内容组织在单个根 `/` 下。没有 `C:\` 或 `/Volumes`。你实际会接触的目录：

```mermaid
graph TD
    root["/"] --> home["home/你的用户名/<br/>你的文件 — 克隆仓库、运行训练"]
    root --> tmp["tmp/<br/>临时文件，重启时清除"]
    root --> usr["usr/<br/>系统程序和库"]
    root --> etc["etc/<br/>配置文件"]
    root --> varlog["var/log/<br/>日志 — 出问题时检查"]
    root --> mnt["mnt/ 或 /media/<br/>外部驱动器和卷"]
    root --> proc["proc/ 和 /sys/<br/>虚拟文件 — 内核和硬件信息"]
```

你的主目录是 `~` 或 `/home/你的用户名`。几乎所有事情都在这里发生。

## 基本命令

这 15 个命令涵盖你在远程 GPU 盒子上 95% 的工作。

### 移动

```bash
pwd                         # 我在哪里？
ls                          # 这里有什么？
ls -la                      # 这里有什么，包括带详细信息的隐藏文件？
cd /path/to/dir             # 去那里
cd ~                        # 回家
cd ..                       # 上一级
```

### 文件和目录

```bash
mkdir my-project            # 创建目录
mkdir -p a/b/c              # 一步创建嵌套目录

cp file.txt backup.txt      # 复制文件
cp -r src/ src-backup/      # 复制目录（递归）

mv old.txt new.txt          # 重命名文件
mv file.txt /tmp/           # 移动文件

rm file.txt                 # 删除文件（没有回收站，没了）
rm -rf my-dir/              # 删除目录及其中所有内容
```

`rm -rf` 是永久的。没有撤销。按回车前仔细检查路径。

### 读取文件

```bash
cat file.txt                # 打印整个文件
head -20 file.txt           # 前 20 行
tail -20 file.txt           # 最后 20 行
tail -f log.txt             # 实时跟踪日志文件（Ctrl+C 停止）
less file.txt               # 滚动文件（q 退出）
```

### 搜索

```bash
grep "error" training.log           # 查找包含 "error" 的行
grep -r "learning_rate" .           # 搜索当前目录下所有文件
grep -i "cuda" config.yaml          # 不区分大小写搜索

find . -name "*.py"                 # 查找当前目录下所有 Python 文件
find . -name "*.ckpt" -size +1G     # 查找大于 1GB 的检查点文件
```

## 权限

Linux 中每个文件都有所有者和权限位。当脚本无法执行或你无法写入目录时，你会遇到这个。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             所有者权限：读、写、执行
#     ^^^          组权限：读、执行
#        ^^        其他所有人：只读
```

常见修复：

```bash
chmod +x train.sh           # 使脚本可执行
chmod 755 deploy.sh         # 所有者：全部，其他：读+执行
chmod 644 config.yaml       # 所有者：读+写，其他：只读

chown user:group file.txt   # 更改文件所有者（需要 sudo）
```

当说"权限被拒绝"时，几乎总是权限问题。`chmod +x` 或 `sudo` 会修复大多数情况。

## 包管理（apt）

Ubuntu 使用 `apt`。这是你安装系统级软件的方式。

```bash
sudo apt update             # 刷新包列表（总是先做这个）
sudo apt install -y htop    # 安装包（-y 跳过确认）
sudo apt install -y build-essential  # C 编译器、make 等。许多 Python 包需要
sudo apt install -y tmux    # 终端复用器（断开连接后保持会话存活）

apt list --installed        # 安装了什么？
sudo apt remove htop        # 卸载
```

在全新 GPU 盒子上你会安装的常见包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户和 sudo

你通常以普通用户登录。有些操作需要 root（管理员）访问。

```bash
whoami                      # 我是谁？
sudo command                # 以 root 运行单个命令
sudo su                     # 成为 root（exit 返回，少用）
```

在云 GPU 实例上，你通常是唯一用户且已有 sudo 访问。不要什么都以 root 运行。只在需要时用 sudo。

## 进程和 systemd

当你的训练挂起，或你需要检查运行什么时：

```bash
htop                        # 交互进程查看器（q 退出）
ps aux | grep python        # 查找运行中的 Python 进程
kill 12345                  # 优雅停止 PID 12345 的进程
kill -9 12345               # 强制杀死（优雅不行时用）
nvidia-smi                  # GPU 进程和内存使用
```

systemd 管理服务（后台守护进程）。如果你运行推理服务器会用到它：

```bash
sudo systemctl start nginx          # 启动服务
sudo systemctl stop nginx           # 停止
sudo systemctl restart nginx        # 重启
sudo systemctl status nginx         # 检查是否运行
sudo systemctl enable nginx         # 开机自动启动
```

## 磁盘空间

GPU 盒子的磁盘空间通常有限。模型和数据集很快填满它。

```bash
df -h                       # 所有挂载驱动器的磁盘使用
df -h /home                 # /home 的磁盘使用

du -sh *                    # 当前目录每个项目的大小
du -sh ~/.cache             # 缓存大小（pip、huggingface 模型在这里）
du -sh /data/checkpoints/   # 检查检查点有多大

# 找出最大的空间占用者
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见空间节省方法：

```bash
# 清除 pip 缓存
pip cache purge

# 清除 apt 缓存
sudo apt clean

# 删除不需要的旧检查点
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你会从命令行下载模型、传输文件和访问 API。

```bash
# 下载文件
wget https://example.com/model.bin                   # 下载文件
curl -O https://example.com/data.tar.gz              # 用 curl 做同样的事
curl -s https://api.example.com/health | python3 -m json.tool  # 访问 API，漂亮打印 JSON

# 机器间传输文件
scp model.bin user@remote:/data/                     # 复制文件到远程机器
scp user@remote:/data/results.csv .                  # 从远程复制文件到本地
scp -r user@remote:/data/checkpoints/ ./local-dir/   # 复制目录

# 同步目录（大传输时比 scp 快，失败时恢复）
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

任何大文件用 `rsync` 而不是 `scp`。它只传输更改的字节并处理中断连接。

## tmux：保持会话存活

当你 SSH 到远程盒子时，关闭笔记本会杀死你的训练运行。tmux 防止这个。

```bash
tmux new -s train           # 启动名为 "train" 的新会话
# ... 启动训练，然后：
# Ctrl+B，然后 D            # 分离（训练继续运行）

tmux ls                     # 列出会话
tmux attach -t train        # 重新连接会话

# tmux 内部：
# Ctrl+B，然后 %            # 垂直分割窗格
# Ctrl+B，然后 "            # 水平分割窗格
# Ctrl+B，然后方向键        # 窗格间切换
```

总是在 tmux 中运行长时间训练作业。总是。

## WSL2 给 Windows 用户

如果你在 Windows 上，WSL2 给你真正的 Linux 环境而无需双启动。

```bash
# 在 PowerShell（管理员）中
wsl --install -d Ubuntu-24.04

# 重启后，从开始菜单打开 Ubuntu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真正的 Linux 内核。这节课的所有内容在里面都适用。你的 Windows 文件在 WSL 内部是 `/mnt/c/Users/你的名字/`。

GPU 透传在 Windows 端安装 NVIDIA 驱动时有效。安装 Windows NVIDIA 驱动（不是 Linux 的），CUDA 在 WSL2 内可用。

## 陷阱：macOS 到 Linux

如果你从 macOS 来，这些会让你绊倒：

| macOS | Linux | 说明 |
|-------|-------|------|
| `brew install` | `sudo apt install` | 包名有时不同。`brew install htop` vs `sudo apt install htop` 同样有效，但 `brew install readline` vs `sudo apt install libreadline-dev` 不同。 |
| `open file.txt` | `xdg-open file.txt` | 但远程盒子没有 GUI。用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | SSH 上没有剪贴板管道。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认用 zsh。大多数 Linux 服务器用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`、`/usr/local/bin/` | 二进制文件在不同位置。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS sed `-i` 后需要空字符串。Linux 不需要。 |
| 不区分大小写文件系统 | 区分大小写文件系统 | Linux 上 `Model.py` 和 `model.py` 是两个不同文件。 |
| 行尾 `\n` | 行尾 `\n` | 相同。但 Windows 用 `\r\n`，会破坏 bash 脚本。运行 `dos2unix` 修复。 |

## 快速参考卡

```
导航：     pwd, ls, cd, find
文件：     cp, mv, rm, mkdir, cat, head, tail, less
搜索：     grep, find
权限：     chmod, chown, sudo
包：       apt update, apt install
进程：     htop, ps, kill, nvidia-smi
服务：     systemctl start/stop/restart/status
磁盘：     df -h, du -sh
网络：     curl, wget, scp, rsync
会话：     tmux new/attach/detach
```

## 练习题

1. SSH 到任何 Linux 机器（或打开 WSL2）并导航到你的主目录。创建一个项目文件夹，用 `touch` 在里面创建三个空文件，然后用 `ls -la` 列出它们。
2. 用 apt 安装 `htop`，运行它，识别哪个进程使用最多内存。
3. 启动 tmux 会话，在里面运行 `sleep 300`，分离，列出会话，重新连接。
4. 用 `df -h` 检查可用磁盘空间，然后用 `du -sh ~/.cache/*` 查找缓存中什么占用空间。
5. 用 `scp` 把文件从本地机器传输到远程机器，然后用 `rsync` 做同样传输并比较体验。
