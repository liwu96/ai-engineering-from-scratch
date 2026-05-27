# Society of Mind和Multi-Agent Debate

> Minsky 1986 premise — intelligence society specialist — get rediscover every decade。In 2023 Du et al. turn concrete algorithm:multiple LLM instance propose answer、read each other answer、critique、and update。Over N round converge consensus beat zero-shot CoT和reflection six reasoning和factuality task。Two finding matter:both **multiple agent**和**multiple round** contribute independently。Society beat single-agent monologue;multi-round exchange beat one-shot voting。

**类型:** 学习+构建
**语言:** Python(stdlib)
**前置要求:** 阶段16课程04(Primitive Model)
**时间:** ~60分钟

## 问题背景

Self-consistency — sample one model many time and take majority answer — cheapest reasoning improvement bolt on。It work、but saturate fast。You can double sample and not see another meaningful jump。

Debate break saturation。Instead N independent sample one model、N agent read each other reasoning and revise。Correlation between sample drop(they no longer i.i.d.)、and convergence point often correct where i.i.d. voting confidently wrong。

## 概念讲解

### Du et al. 2023 algorithm

From arXiv:2305.14325(ICML 2024):

1. Each N agent produce initial answer question。
2. For round r = 2..R:each agent show other agent round r-1 answer and ask "considering these、give updated answer。"
3. After R round、majority-vote final answer。

Paper test MMLU、GSM8K、biography、MATH、and factuality benchmark。Debate consistently beat CoT和Self-Reflection。

### Two independent knob

Ablation same paper:

- **Agent count alone**(1 round、majority vote N)beat single-agent most task、but plateau。
- **Round count alone**(1 agent seeing own prior reasoning)barely help — reflection known weakness。
- **Both together**produce big jump。Multi-round exchange between multiple agent drive gain。

### 何work

Two mechanism:

1. **Exposure disagreement。**When agent see another agent reasoning chain different conclusion、it has either justify or update。Either way、context round r+1 richer round r。
2. **Correlated error reduction。**In self-consistency、all sample come same model、so error correlate — you average confidently wrong answer。Different model or different seed decorrelate。Different *debated view* decorrelate further。

### Heterogeneous debate

A-HMAD and related follow-up use *different base model* different agent。Llama + Claude + GPT debate reduce monoculture collapse(Lesson 26)because correlated error one model family not share other。

Downside:weak model participate debate can drag consensus toward wrong answer(see "Should we be going MAD?"、arXiv:2311.17371)。

### NLSOM — 129-agent extension

Zhuge et al.("Mindstorms Natural Language-Based Societies of Mind、"arXiv:2305.17066)scale idea 129-member society。Result:specialization和self-organization emerge scale、and system outperform single-agent task like visual question answering。

### Failure mode

- **Sycophancy cascade。**All agent defer whichever agent sound most confident。Debate collapse loudest voice。Prompting adversarial role("one agent must argue counter-position")help。
- **Topic drift。**Debate over many round drift original question。Mitigation:re-inject question every round。
- **Compute blowup。**N agent × R round = N·R LLM call、each context grow。5-agent、5-round debate 25 call growing context。Cost per question can exceed 10× single CoT call。

## 构建

`code/main.py` run 3-agent × 3-round debate math question where each agent start different(possibly wrong)answer。Agent scripted — each "update" by average neighbor answer weight scripted confidence。Convergence visible round-by-round log。

Demo show two key effect:

- Single round exchange move agent closer correct answer。
- Extra round past round 2 show diminishing return(match Du et al. plateau)。

跑:

```
python3 code/main.py
```

## 使用

`outputs/skill-debate-configurator.md` configure debate new task:number agent、number round、heterogeneity(same model vs mixed)、role assignment(symmetric vs one-adversarial)。It also estimate token cost before you run。

## 交付成果

If you ship debate:

- **Cap round 3。**Du et al. show 3 round capture most gain。More cost、非quality。
- **Cap agent 5。**Beyond 5、context bloat和cost dominate。
- **Heterogeneous default。**At least two different base model pool。
- **Adversarial slot。**One agent prompt disagree regardless。Break sycophancy。
- **Log every round。**Debate system hide intermediate round cannot debug or audit。

## 练习题

1. Run `code/main.py`、then set round count 5 and watch diminishing return。何round additional convergence stop?
2. Add fourth agent adversarial role:always disagree current majority。Does this break or improve convergence?
3. Plot(print)agreement score per round(fraction agent majority answer)。When hit 1.0 and equivalent "correct"?
4. Read Du et al. Section 4 ablation。Replicate "agent-only" vs "round-only" vs "both" result using code。
5. Read "Should we be going MAD?"(arXiv:2311.17371)and list two debate variant beyond round-robin — e.g. judge-led、chain-of-debate、adversarial。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Society of Mind | "Minsky idea" | Intelligence interacting specialist;1986 framing now operationalize LLM debate。 |
| Multi-agent debate | "Agent argue" | N agent propose、critique each other、revise over R round、majority-vote。 |
| Consensus | "They agree" | 非epistemic truth — just fraction-on-majority-answer。Can confidently wrong。 |
| Round | "Exchange step" | One round = each agent read other and update once。 |
| Heterogeneous debate | "Mix model family" | Use different base model decorrelate error。 |
| Sycophancy cascade | "Everyone agree loud one" | Debate failure agent defer most confident agent regardless correctness。 |
| NLSOM | "129-agent society" | Natural-language society mind;Zhuge et al. scaled version。 |
| Correlated error | "Same model、same bug" | Why self-consistency saturate;debate different view decorrelate。 |

## 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — reference paper、ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmark debate variant
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al. code、demo、and ablation detail