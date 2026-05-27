# 具身VLA: RT-2, OpenVLA, π0, GR00T

> 模型第一次从网站读食谱并在厨房机器人执行是RT-2 (Google DeepMind，2023年7月)。RT-2离散行动文token，共微调VLM于网数据加机器人行动数据，证明网规模视觉语言知识转移机器人控制。OpenVLA (2024年6月)交付开源7B参考。Physical Intelligence π0系列(2024-2025)加流匹配行动专家。NVIDIA GR00T N1 (2025年3月)交付双系统(System 1 / System 2)控制人形机器人规模。VLA原语——视觉语言行动，单模型看、读、行动——是本阶段理解模型和第15阶段自主系统桥。

**类型:** 学习
**语言:** Python (stdlib，行动分词器+VLA推理骨架)
**前置要求:** 第12阶段·05(LLaVA)，第15阶段(自主系统，参考)
**时间:** ~180分钟

## 学习目标

- 描述行动分词:离散bin编码(RT-2)、FAST效行动token、连续流匹配行动(π0)。
- 解释何共微调网+机器人数据保通用知识转移到新任务。
- 比OpenVLA(开源7B Llama+VLM)、π0(流匹配)、GR00T N1(双系统)于同机器人任务。
- 命Open X-Embodiment数据集及其作RT-X训练语料角色。

## 问题背景

从自然语言指令做家务机器人自1970年代研目标。2020答:视觉语言行动(VLA)模型。同VLM架构VQA用，但输出行动(关节力、末端执行器姿、离散命令)非文本。

VLA特定挑战:

1. 行动空间连续(关节角、力)和高维(7-DOF臂+3-DOF夹爪=10维30 Hz)。
2. 机器人特定训练数据稀。Open X-Embodiment ~1M轨迹；网文图5B+。
3. 控制频率重要。30 Hz控制环意33ms预算每行动。
4. 安全。错行动损硬件、人、财产。

## 概念讲解

### 行动分词(RT-2)

RT-2技:代表每关节目标作量化文token。离散化归[-1, 1]范围256 bin，映每bin词汇ID。10-DOF行动变10 token每控步。

共微调PaLM-X VLM混:

- 网文图对(标注、VQA)。
- 机器人demo，行动作token。

模型见"拾红立方"(语言)→图像(视觉)→10-token行动序(离散关节目标)。网预训练保通用知识转移:RT-2可循"向快移物移"虽"快移"不在训练数据。

论文推理3-5 Hz，限于VLM自回归解码。

### OpenVLA——开源7B参考

OpenVLA (Kim等，2024年6月)是开源权重RT-2等价。7B Llama背，DINOv2 + SigLIP双视觉编码器，行动分词256 bin。

训Open X-Embodiment (970k轨迹22机器人)。配LoRA微调支持适新机器人。

推理:A100量化4-5 Hz。慢操作快够，高频控不够。

### FAST分词器——更快行动解码

Pertsch等(2024)示离散bin分词低效——大多行动聚bin空间小区。FAST (频域行动序分词器)经DCT压缩行动序量化系数。

30步行动轨迹变~10 FAST token代300离散bin token。推理速度3-5x无质失。

### π0和流匹配行动

Physical Intelligence π0 (Black等，2024年10月)换离散行动token流匹配行动专家:

- 小行动transformer读VLM隐藏态并出连续50步行动序经rectified flow。
- 行动头流匹配损失训；VLM预训练不变。
- 推理:全行动序~5去噪步发，效50 Hz控制。

π0声称:宽套操作任务击败OpenVLA和Octo。连续行动表示保平滑量化破坏。

π0.5和π0-FAST是增量升级。π0-FAST合FAST分词流匹配。

### GR00T N1——人形双系统

NVIDIA GR00T N1 (2025年3月)建人形机器人(>30 DOF，全身):

- System 2:大VLM读场景+指令，产高层子目标~1 Hz。
- System 1:小行动头transformer产低层50-100 Hz关节命令条件子目标。

分映Kahneman快慢思:System 2计划，System 1行动。益:慢VLM级规划不阻塞快控制；System 1留小延迟。

GR00T N1.7 (2025末)改进数据缩。GR00T用Omniverse sim-to-real数据微调。

### Open X-Embodiment

训练数据。RT-X (2023年10月)组22数据集覆1M轨迹22机器人。Open X-Embodiment是每人用语料:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每样:(机器人态，相机视，指令，行动序)。
- 训练卫生:统一行动空间，归关节范围，缩相机。

OpenVLA和π0训Open X-Embodiment。特定机器人域隙LoRA微调100-1000任务特定demo闭。

### 共微调vs仅机器人

共微调混网VQA数据机器人轨迹。比重要:太多VQA模型忘行动；太多机器人数据模型失通用知识。

RT-2比:~1:1。OpenVLA:~0.5:1网机器人。π0:类似。精比超参每数据集尺寸调。

仅机器人训练产任务特定模型失败分布外指令。共微调是"拾红立方(demo中)"和"从左拾第三大物(新措辞)"差距。

### 安全和行动限

每生产VLA配:

- 硬关节限(不能力过规)。
- 速度限(软裁)。
- 工作空间界(末端执行器不能离桌)。
- 新任务人在环批准。

这些坐VLA外控制层检。VLA输出是建议，非命令。

## 使用它

`code/main.py`:

- 实现256-bin行动分词和去分词。
- 草FAST分词器基DCT +量化。
- 比token数每行动步跨(离散bin, FAST,连续流)。
- 印RT-2 → OpenVLA → π0 → GR00T谱摘要。

## 发货它

这课产`outputs/skill-vla-action-format-picker.md`。给机器人任务(操作、导航、人形全身)，选离散bin + RT-2、FAST + OpenVLA、流匹配 + π0、或双系统 + GR00T。

## 练习题

1. 10-DOF臂30 Hz控制率。256 bin离散分词发每秒何token？7B VLM能跟上？

2. FAST分词压缩30步轨迹~10 token。何高频运动轨迹(如鼓)用户失？

3. π0流匹配头~5步去噪。比吞吐OpenVLA自回归解码4-5 Hz。

4. GR00T System 1 / System 2分映Kahneman。提不同分(System 3?)可能帮双足走。

5. 读Open X-Embodiment第4节数据集策。命三策规则防域泄漏。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| VLA | "视觉语言行动" | 取图像+指令出行动命令模型 |
| 行动分词 | "离散bin" | 量化连续关节目标256 bin每维，每词汇ID |
| FAST分词器 | "频域行动token" | DCT+量化压缩30步轨迹~10 token |
| 共微调 | "混网+机器人" | 网VQA数据旁机器人demo训保通用知识 |
| 流匹配行动头 | "π0连续输出" | 小transformer经rectified flow出50步行动序 |
| System 1 / System 2 | "双系统控制" | 大VLM慢计划，小行动头快行动；GR00T模式 |
| Open X-Embodiment | "RT-X数据集" | 1M轨迹跨机器人数据集；训练语料 |

## 延伸阅读

- [Brohan等—RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim等—OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black等—π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA—GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab—RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)