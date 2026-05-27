# 自建服务选——llama.cpp、Ollama、TGI、vLLM、SGLang

> 四engine 2026自建推理主导。基于硬件、规模、生态选。**llama.cpp** CPU最快——广模型支持、量化线程全控。**Ollama** dev-laptop一命令装、llama.cpp慢~15-30% (Go + CGo + HTTP序列化)、prod类负载3x吞吐gap。**TGI 2025年12月11日进maintenance mode**——仅bug fix、raw吞吐比vLLM慢~10%但历史可观测和HF-ecosystem集成顶。maintenance态新项目风险长投——SGLang或vLLM新项目安全默认。**vLLM**通用生产默认——v0.15.1 (2026年2月)加PyTorch 2.10、RTX Blackwell SM120、H200优化。**SGLang** agentic多轮/前缀重专——生产400,000+ GPU (xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS)。硬件约束：CPU-only → llama.cpp仅。AMD / 非NVIDIA → vLLM仅(TRT-LLM NVIDIA-locked)。2026管道模式：dev = Ollama、staging = llama.cpp、prod = vLLM或SGLang。同GGUF/HF权重贯穿。

**类型:** 学习
**语言:** Python(stdlib、engine决策树walker)
**前置要求:** 全阶段17engine课(04、06、07、09、18)
**时间:** ~45分钟

## 学习目标

- 给硬件(CPU / AMD / NVIDIA Hopper / Blackwell)、规模(1用户 / 100 / 10,000)、负载(通用聊天 / agent / 长上下文)选engine。
- 命名2026 TGI maintenance-mode态(2025年12月11日)和为何新项目偏vLLM或SGLang。
- 描述dev/staging/prod管道同GGUF或HF权重贯穿。
- 解释为何"CPU only"强制llama.cpp和"AMD"排除TRT-LLM。

## 问题背景

队新自建LLM项目启。一工程师说Ollama、另一说vLLM、三说"TGI开箱即用？"全不同上下文对。无全对。

2026选择树重要：硬件首、规模次、负载三。2025特定事件——TGI 2025年12月11日进maintenance——改新项目默认。

## 概念讲解

### 五engine

| Engine | 最佳 | 注 |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / 最少依赖 / 广模型支持 | CPU最快、全控 |
| **Ollama** | Dev laptop、单用户、一命令装 | llama.cpp慢15-30%；3x prod吞吐gap |
| **TGI** | HF生态、监管行业 | **2025年12月11日Maintenance mode** |
| **vLLM** | 通用生产、100+用户 | 广生产默认；v0.15.1 2026年2月 |
| **SGLang** | Agentic多轮、前缀重负载 | 生产400,000+ GPU |

### 硬件首决策

**CPU only** → llama.cpp。Ollama也工但慢。无他engine CPU竞争。

**AMD GPU** → vLLM (AMD ROCm支持)。SGLang也工。TRT-LLM NVIDIA-locked、排除。

**NVIDIA Hopper (H100 / H200)** → vLLM或SGLang或TRT-LLM。三顶级。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM吞吐领(阶段17课程07)。vLLM和SGLang近跟。

**Apple Silicon (M-series)** → llama.cpp (Metal)。Ollama包此。

### 规模次决策

**1用户 / 本地dev** → Ollama。一命令、首token秒。

**10-100用户 / 小队** → vLLM单GPU。

**100-10k用户 / 生产** → vLLM production-stack (阶段17课程18)或SGLang。

**10k+用户 / 企业** → vLLM production-stack + 解耦(阶段17课程17) + LMCache (阶段17课程18)。

### 负载三决策

**通用聊天 / Q&A** → vLLM广默认赢。

**Agentic多轮(工具、规划、记忆)** → SGLang RadixAttention (阶段17课程06)主导。

**RAG前缀重用** → SGLang。

**代码生成** → vLLM行；SGLang cache稍好。

**长上下文(128K+)** → vLLM + 分块prefill；SGLang + 层级KV。

### TGI maintenance陷阱

Hugging Face TGI 2025年12月11日进maintenance mode——仅bug fix前。历史：顶级可观测、HF-ecosystem集成最佳(model card、安全工具)、raw吞吐略后vLLM。

2026新项目：默认远TGI。存TGI部署续但应终迁。SGLang和vLLM安全默认。

### 管道模式

Dev (Ollama) → staging (llama.cpp) → prod (vLLM)。同GGUF或HF权重贯穿。工程师laptop快迭代；staging镜像生产量化；prod服务目标。

### Ollama caveat

Ollama dev好。共享生产非好：Go HTTP序列化加开销、并发管理比vLLM简、OpenTelemetry支持滞。Ollama强处用——一用户、一命令——共享换vLLM。

### 自建vs托管分离决策

阶段17课程01(托管hyperscaler)、02(推理平台)覆托管。本lesson假定已决自建。自建因：数据居住、定制微调、规模总成本所有权、域模型托管无。

### 你应记数

- TGI maintenance mode：2025年12月11日。
- vLLM v0.15.1：2026年2月；PyTorch 2.10；Blackwell SM120支持。
- SGLang生产footprint：400,000+ GPU。
- Ollama吞吐gap vs llama.cpp：15-30%慢；prod负载3x。

## 使用

`code/main.py`决策树walker：给硬件 + 规模 + 负载、选engine并解释。

## 交付成果

本lesson产`outputs/skill-engine-picker.md`。给约束、选engine写迁计划。

## 练习题

1. 跑`code/main.py`你硬件 / 规模 / 负载。输出匹配直觉否？
2. Infra 12 H100s和8 MI300X AMD。何engine？为何TRT-LLM排除？
3. 队2026想用TGI因"我们熟"。论证迁case。
4. Ollama dev到vLLM prod：量化、配置、可观测何变？
5. RAG产品P99前缀长度8K租户高重用。选engine并栈阶段17课程11 + 18。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| llama.cpp | "CPU那个" | 广模型支持、CPU最快 |
| Ollama | "laptop那个" | 一命令装、dev级吞吐 |
| TGI | "HF serving" | 2025年12月maintenance mode |
| vLLM | "默认那个" | 2026广生产基线 |
| SGLang | "agentic那个" | 缀重、RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell吞吐领、仅NVIDIA |
| GGUF | "llama.cpp格式" | 包K-quant变种 |
| Production-stack | "vLLM K8s" | 阶段17课程18参考部署 |
| 管道模式 | "dev→stage→prod" | Ollama → llama.cpp → vLLM同权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)