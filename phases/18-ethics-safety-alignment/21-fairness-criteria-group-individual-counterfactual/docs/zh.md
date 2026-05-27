# 公平准则——组、个体、反事实

> 三家族构公平文献。组公平：人口parity、equalized odds、条件用精度平等 — 保组平率平均。个体公平(Dwork等人 2012)：相似个体收相似决策；决策图Lipschitz条件。反事实公平(Kusner等人 2017)：个体决策公平若敏感属性反事实改时不变。2024理论结果(NeurIPS 2024)：存在固CF-vs-精度trade-off；模型无关方法转最优但非公平预测器为CF带界精度损。Backtracking counterfactuals (arXiv:2401.13935, 2024年1月)：新范式避需法保属性介入。哲学和解(ICLR Blogposts 2024)：因果图、满某些组公平度蕴含反事实公平。

**类型:** 学习
**语言:** Python(stdlib、三准则比)
**前置要求:** 阶段18课程20(偏)、阶段02(经典ML)
**时间:** ~60分钟

## 学习目标

- 陈三组公平准则(人口parity、equalized odds、条件用精度平等)和一不可能结果。
- 描述个体公平经Dwork等人 2012 Lipschitz公式。
- 描述反事实公平及其因果图依赖。
- 解释backtracking counterfactuals和何避保属性介入问题。

## 问题背景

课程20是偏测。课程21是定义公平标准测应服务。三家族给结构不同标准 — 模型可组公平而个体非公平、反事实公平而组非公平。选标准是政策决策；无标准全优。

## 概念讲解

### 组公平

- **人口parity。** P(Y=1 | A=a) = P(Y=1 | A=a') 所有组。平接受率。
- **Equalized odds。** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。组间平TPR和FPR。
- **条件用精度平等。** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。组间平预测值。

不可能(Chouldechova, Kleinberg-Mullainathan-Raghavan 2017)：此三不基率不等时可同满。

### 个体公平

Dwork等人 2012。决策图f任务特定相似度量d个体公平若|f(x) - f(x')| <= L * d(x, x')某Lipschitz常数L。相似个体得相似决策。

需定义d。政策问、非统计。

### 反事实公平

Kusner等人 2017。个体i决策反事实公平若、人口因果模型下、i敏感属性反事实改时决策不变。

需因果DAG。DAG是建模选。反事实公平仅如DAG有据。

### CF-vs-精度trade-off

NeurIPS 2024理论：存在反事实公平和预测精度固trade-off。模型无关方法可转最优但非公平预测器为CF、界精度成本。精度成本依赖最优非公平预测器敏感属性系数大小。

### Backtracking counterfactuals

arXiv:2401.13935 (2024年1月)。传统反事实需敏感属性介入 — "若此人异性别决策改否。" 法、此有问：保属性不可分类法介入。

Backtracking counterfactuals翻方向：非介入属性、问个体实特征何组合产反事实结果。此避法异议。

### 哲学和解

ICLR Blogposts 2024。手因果图、满某些组公平度蕴含反事实公平。三家族非正交；是同因果结构不同面。

此不解不可能定理(不基率仍阻同组公平)。但示"组"和"个体/反事实"显对部分是未明因果模型artifact。

### Phase 18何处

课程20是偏测。课程21是公平定义。课程22是隐私(差隐私)。课程23是水印。此是分配相关课程补欺骗相关课程7-11。

## 使用

`code/main.py`玩具二元分类数据带敏感属性和不基率。简分类器算人口parity、equalized odds、和条件用精度平等。观三度量分。施人口parity重加权并观其对另两成本。

## 交付成果

本lesson产`outputs/skill-fairness-criterion.md`。给公平声明或政策、识何准则声明、模型能否满剩余准则于声明不基率、和声明依赖何因果DAG。

## 练习题

1. 跑`code/main.py`。报默认数据三组度量。施人口parity目标重加权并重报。
2. 实Dwork等人 2012个体公平度量用L2于非敏感特征。报何对违Lipschitz常数L=1。
3. 读Kusner等人 2017。构简历评分简两特征因果DAG并识其意反事实公平条件。
4. 2024 backtracking-counterfactuals论文避保属性介入。述此何重于法合规场景。
5. ICLR 2024和解争组和反事实公平是同结构面。选`code/main.py`三准则两并述使其等价因果假设。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 人口parity | "平率" | P(Y=1 | A=a)组间平 |
| Equalized odds | "平TPR/FPR" | 组间平真阳假阳率 |
| 条件用精度 | "平PPV/NPV" | 组间平预测值 |
| 个体公平 | "Lipschitz条件" | 相似个体得相似决策 |
| 反事实公平 | "因果改不变" | 反事实属性改决策不变 |
| Backtracking counterfactual | "经实解释" | 反事实从结果后推理、非从属性前 |
| 不可能定理 | "三冲突" | Chouldechova / KMR 2017：组准则不基率不等时互斥 |

## 延伸阅读

- [Dwork等人 — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — 个体公平
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — 反事实公平
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — 不可能
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — 保属性介入新范式