# 说话人识别与验证

> ASR问"他们说了什么?"说话人识别问"谁说的?"数学看起来一样——嵌入加余弦——但每个生产决策取决于单个EER数字。

**类型:** 构建
**语言:** Python
**前置要求:** 阶段6课程02(频谱图与Mel)、阶段5课程22(嵌入模型)
**时间:** ~45分钟

## 问题背景

用户说密码短语。要知道:这是他们声称的人吗(*验证*,1:1),还是注册库中第一人(*识别*,1:N)?或都不是——这是未知说话人吗(*开放集*)?

2018前:GMM-UBM + i-vector。合理EER但对渠道移(电话vs笔记本)和情绪脆弱。2018–2022:x-vector(TDNN骨干配角度边缘训)。2022+:ECAPA-TDNN和WavLM-large嵌入。2026该领域由三模型和一指标主导。

指标是**EER**——等错误率。设决策阈值使假接受率=假拒绝率。交叉点是EER。用于每篇论文、每个排行榜、每个采购电话。

## 概念讲解

![注册+验证管道配嵌入+余弦+EER](../assets/speaker-verification.svg)

**管道。**注册:录目标说话人5–30秒;算固定维嵌入(ECAPA-TDNN 192维,WavLM-large 256维)。验证:得测试话语嵌入;算余弦相似度;比阈值。

**ECAPA-TDNN(2020,2026仍主导)。**强调通道注意力、传播和聚合-时延神经网络。1D卷积块配压缩激励、多头注意力池化,后接线性层到192维。VoxCeleb 1+2(2,700说话人,1.1M话语)上配加性角度边缘损失(AAM-softmax)训。

**WavLM-SV(2022+)。**用AAM损失微调预训WavLM-large自监督骨干。更高质量但更慢——300+ MB vs 15 MB。

**x-vector(基线)。**TDNN +统计池化。经典;CPU/边缘仍用。

**AAM-softmax。**标准softmax配角度空间加边缘`m`:`cos(θ + m)`对正确类。强制类间角度分离。典型`m=0.2`,尺度`s=30`。

### 评分

- **余弦**注册和测试嵌入间。阈值基决策。
- **PLDA(概率LDA)。**投嵌入到潜空间同说话人vs不同说话人有闭式似然比。加于余弦顶给+10–20% EER降。2020前标准;现仅闭集用。
- **分数归一化。**`S-norm`或`AS-norm`:每分数对冒名顶替均值和标准差归一化。跨域评必备。

### 应知数字(2026)

| 模型 | VoxCeleb1-O EER | 参数 | 吞吐量(A100) |
|------|-----------------|------|--------------|
| x-vector(经典) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1分割+嵌入 | 0.65% | 6 M | 100× RT |
| ReDimNet(2024) | 0.39% | 24 M | 100× RT |

### 说话人分离

"谁何时说"于多说话人片段。管道:VAD →分段→每段嵌入→聚类(凝聚或谱)→平滑边界。现代栈:`pyannote.audio` 3.1,将说话人分割+嵌入+聚类打包一次调用。2026 AMI SOTA DER约15%(2022从23%降)。

## 动手实践

### Step 1:MFCC统计玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26维
```

远非SOTA——仅教学。`code/main.py`用作合成说话人数据概念证明。

### Step 2:余弦相似度+阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3:相似度对EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返(eer, threshold_at_eer)。报两者。

### Step 4:SpeechBrain生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 注册:平均3-5干净样本嵌入
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA典型阈值;你数据调
```

### Step 5:pyannote分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 实际应用

2026栈:

| 情况 | 选 |
|------|------|
| 闭集1:1验证,边缘 | ECAPA-TDNN +余弦阈值 |
| 开集验证,云 | WavLM-SV + AS-norm |
| 分离(会议,播客) | `pyannote/speaker-diarization-3.1` |
| 反欺骗(重放/深度伪造检测) | AASIST或RawNet2 |
| 小嵌入式(KWS +注册) | Titanet-Small(NeMo) |

## 陷阱

- **渠道不匹配。**VoxCeleb(web视频)训模型≠电话音频。总目标渠道评。
- **短话语。**测试音频低于3秒EER剧烈退化。
- **噪声注册。**一噪声注册污染锚点。用≥3干净样本平均。
- **跨条件固定阈值。**总目标域保留开发集调阈值。
- **未归一化嵌入余弦。**先L2归一化;否则幅度主导。

## 产出成果

存`outputs/skill-speaker-verifier.md`。选模型、注册协议、阈值调计划和欺诈防护。

## 练习题

1. **简单。**跑`code/main.py`。构合成"说话人"(不同音调配置),注册,算100对试验EER。
2. **中等。**SpeechBrain ECAPA于30 VoxCeleb1话语(5说话人×6每)。算余弦vs PLDA EER。
3. **困难。**配`pyannote.audio`构完整注册→分离→验证管道。AMI开发集评DER。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| EER | 头条指标 | 假接受=假拒绝阈值。 |
| 验证 | 1:1 | "这是Alice吗?" |
| 识别 | 1:N | "谁在说?" |
| 开放集 | 未知可能 | 测试集可含未注册说话人。 |
| 注册 | 登记 | 算说话人参考嵌入。 |
| AAM-softmax | 损失 | 配加性角度边缘softmax;强制聚类分离。 |
| PLDA | 经典评分 | 概率LDA;嵌入顶似然比评分。 |
| DER | 分离指标 | 分离错误率——漏+假警+混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)——经典深嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143)——2020–2026主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)——SV和分离自监督骨干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)——生产分离+嵌入栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)——跨模型当前EER排名。