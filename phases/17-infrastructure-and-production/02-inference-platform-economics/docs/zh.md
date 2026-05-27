# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026推理市场不再是GPU时间租赁。分叉为定制芯片(Groq、Cerebras、SambaNova)、GPU平台(Baseten、Together、Fireworks、Modal)、和API优先市场(Replicate、DeepInfra)。Fireworks2026年5月1日涨GPU $1/hr价，10T+ tokens/day上$4B估值告诉你量驱模型工作。Baseten2026年1月$300M Series E闭$5B。竞争定位规则简单：Fireworks优延迟、Together优目录广度、Baseten优企业打磨、Modal优Python原生DX、Replicate优多模态覆盖、Anyscale优分布式Python。本lesson给你可递创始人的矩阵。

**类型:** 学习
**语言:** Python(stdlib、玩具每调用经济比较器)
**前置要求:** 阶段17课程01(托管LLM平台)、阶段17课程04(vLLM Serving Internals)
**时间:** ~60分钟

## 学习目标

- 命名三市场段(定制芯片、GPU平台、API优先)并映射每供应商到段。
- 解释为何"每token"API定价模型压缩向服务引擎成本曲线而非硬件。
- 跨至少三供应商计算每请求有效成本并解释每分钟(Baseten、Modal)何时赢每token。
- 识别给定负载(无服务器突发、稳高通量、微调变体、多模态)哪个平台是正确默认。

## 问题背景

你评估托管超规模平台。决定需更窄更快provider——Fireworks延迟、Together广度、Baseten微调定制模型。现你有六真实选择定价页不对齐。Fireworks示$/M token；Baseten示$/分钟；Modal示$/秒；Replicate示$/预测。不建模负载无头对头比。

更糟，每定价页后商业模式不同。Fireworks跑自己定制引擎(FireAttention)共享GPU；每token率反映其利用曲线。Baseten给Truss+专用GPU；每分钟反映独占。Modal是真Python无服务器——秒级计费亚秒冷启动。同输出(LLM响应)、三不同成本函数。

本lesson建模六告诉你每何时赢。

## 概念讲解

### 三段

**定制芯片**——Groq(LPU)、Cerebras(WSE)、SambaNova(RDU)。典型同模型GPU基集群解码5-10x快。每token价高(Groq 2025末Llama-70B~$0.99/M)但对延迟敏感用例无敌。Groq是语音Agent和实时翻译生产选。

**GPU平台**——Baseten、Together、Fireworks、Modal、Anyscale。跑NVIDIA(H100、H200、B200 2026)或有时AMD。"裸GPU租赁"(RunPod、Lambda)和"超规模托管服务"(Bedrock)间经济层。

**API优先市场**——Replicate、DeepInfra、OpenRouter、Fal。广目录、付每预测或付每秒、强调首次调用时间。

### Fireworks——延迟优化GPU平台

- FireAttention引擎(定制)；营销等配置比vLLM 4x低延迟。
- 批层~50%无服务器率非交互负载。
- 微调模型等同基座模型率——对提供商收LoRA溢价真差异化。
- 2026中：2026年5月1日生效GPU租赁$1/hr涨。规模量价可谈。
- 金融信号：$4B估值、10T+ tokens/day处理。

### Together——广度优化

- 200+模型含上游发布天内开源发布。
- 同LLM模型比Replicate 50-70%便宜——"AI Native Cloud"定位量加目录。
- 推理+微调+训练一API。

### Baseten——企业打磨优化

- Truss框架：模型打包依赖、secret、服务配置一manifest。
- GPU范围从T4到B200。每分钟计费合理冷启动缓解。
- SOC 2 Type II、HIPAA-ready。常见金融医疗选。
- $5B估值、2026年1月Series E($300M from CapitalG、IVP、NVIDIA)。

### Modal——Python原生优化

- 纯Python Infrastructure-as-code。函数装饰`@modal.function(gpu="A100")`一命令部署。
- 每秒计费。冷启动2-4s预热；小模型<1s。
- $87M Series B $1.1B估值(2025)。独立调研最强开发者体验分。

### Replicate——多模态广度

- 付每预测。图像、视频、音频模型默认平台。
- 集成生态(Zapier、Vercel、CMS plugins)。
- LLM每token率竞争力差但多模态多样赢。

### Anyscale——Ray原生

- 基Ray；RayTurbo是Anyscale专推理引擎(vLLM竞)。
- 最适分布式Python工作负载推理步是大图一节点。
- 托管Ray集群；Ray AIR和Ray Serve紧集成。

### 每token vs每分钟——何时赢

每token当负载延迟不敏突发适——付所仅用。每分钟当利用高可预测——专用GPU~30%稳利用以上每分钟(Baseten、Modal)开始赢每token(Fireworks、Together)。以下每token赢因避付空闲。

粗规则：专用GPU~30%稳利用以上，每分钟(Baseten、Modal)开始赢每token(Fireworks、Together)。以下每token赢避付空闲。

### 定制引擎是真护城河

vLLM和SGLang以上每平台声称定制引擎。FireAttention、RayTurbo、Baseten推理栈。定制引擎声称影营销——诚实框架是vLLM + SGLang代表~80%生产开源推理，平台层差异化DX、归属、SLA。

### 你应记住数

- Fireworks GPU租赁：2026年5月1日生效$1/hr涨。
- Fireworks声称：等配置比vLLM 4x低延迟。
- Together：LLM比Replicate 50-70%便宜。
- Baseten估值：$5B(Series E，2026年1月，$300M轮)。
- Modal估值：$1.1B(Series B，2025)。
- 每分钟~30%稳利用以上赢每token。

## 使用

`code/main.py`比较六供应商合成负载跨定价模型。报告$/day和有效$/M token。跑找每token和每分钟平衡。

## 交付成果

本lesson产`outputs/skill-inference-platform-picker.md`。给负载profile、SLA、预算，选主要推理平台并命名备选。

## 练习题

1. 跑`code/main.py`。70B模型一H100稳利用何时Baseten(每分钟)赢Fireworks(每token)？推导交叉比规则拇指。
2. 产品服图像生成+聊天+语音转文本。每模态选平台并命名统一gateway模式。
3. Fireworks主要模型涨$1/hr价。若40%流量移批层(50% off)建模混合成本影响。
4. 监管客户需SOC 2 Type II + HIPAA +专用GPU。哪三平台可行哪个FinOps赢？
5. 比Llama 3.1 70B Fireworks无服务器、Together按需、Baseten专用、Replicate API每1000预测成本。10预测/天哪最便宜？10,000？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 定制芯片 | "非GPU芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——解码优化 |
| FireAttention | "Fireworks引擎" | 定制attention kernel；营销比vLLM 4x低延迟 |
| Truss | "Baseten格式" | 模型打包manifest；依赖+secret+服务配置 |
| 每token | "API定价" | 消耗token收费；付无空闲 |
| 每分钟 | "专用定价" | 墙钟GPU时间收费；高利用赢 |
| 每预测 | "Replicate定价" | 模型调用收费；图像/视频常见 |
| RayTurbo | "Anyscale引擎" | Ray专推理；Ray集群上vLLM竞 |
| 批层 | "50% off" | 非交互队列降率；Fireworks、OpenAI常见 |
| 微调基座率 | "Fireworks LoRA" | LoRA请求基座模型率收费(差异化) |

## 延伸阅读

- [Fireworks定价](https://fireworks.ai/pricing) — 每token率、批层、GPU租赁。
- [Baseten定价](https://www.baseten.co/pricing/) — 每分钟率、承诺容量、企业层。
- [Modal定价](https://modal.com/pricing) — 每秒GPU率和免费层。
- [Together AI定价](https://www.together.ai/pricing) — 模型目录和每token率。
- [Anyscale定价](https://www.anyscale.com/pricing) — RayTurbo和托管Ray定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — AI推理API供应商2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商景观