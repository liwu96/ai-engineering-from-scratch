# Edge推理——Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> Edge核心约束是内存带宽、非计算。移动DRAM 50-90 GB/s；数据中心HBM3 2-3 TB/s——30-50x gap。解码内存绑、gap决定。2026格局四方分。Apple M4/A18 Neural Engine峰值38 TOPS统一内存(无CPU↔NPU拷)。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 45 TOPS。WebGPU + WebLLM M3 Max跑Llama 3.1 8B (Q4) ~41 tok/s(原生70-80%)；17.6k GitHub stars、OpenAI兼容API、~70-75%移动覆盖。NVIDIA Jetson Orin Nano Super (8GB)装Llama 3.2 3B / Phi-3；AGX Orin vLLM跑gpt-oss-20b ~40 tok/s；Jetson T4000 (JetPack 7.1)是AGX Orin 2x。TensorRT Edge-LLM支持EAGLE-3、NVFP4、分块prefill——CES 2026 Bosch、ThunderSoft、MediaTek示。

**类型:** 学习
**语言:** Python(stdlib、玩具带宽绑解码模拟器)
**前置要求:** 阶段17课程04(vLLM Serving)、阶段17课程09(生产量化)
**时间:** ~60分钟

## 学习目标

- 解释为何移动LLM推理内存带宽绑、计算次。
- 列举四方edge目标(Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson)并匹配用例。
- 命名2026 WebGPU覆盖gap(Firefox Android追赶)和Safari iOS 26着陆。
- 每目标选量化格式(Core ML INT4 + FP16 ANE、QNN INT8/INT4 Hexagon、WebGPU Q4浏览器、NVFP4 Jetson Thor)。

## 问题背景

客户要设备上聊天机器人：语音优先、隐私默认、离线工作。MacBook Pro M3 Max、Llama 3.1 8B Q4跑~55 tok/s——行。iPhone 16 Pro、同模型3 tok/s——不行。中端Android Snapdragon 8 Gen 3、7 tok/s。浏览器WebGPU Chrome Android v121+、4-8 tok/s设备依赖。

吞吐方差非移植问题。是带宽gap乘量化格式乘NPU用户空间可访。Edge推理2026是四方不同问题四方不同解。

## 概念讲解

### 带宽是真天花板

解码每token读全权重。一7B模型Q4 3.5 GB。50 GB/s读3.5 GB 70 ms——理论天花板~14 tok/s。90 GB/s(高端移动DRAM)天花板~25 tok/s。计算无助于此数下。

数据中心HBM3 3 TB/s读3.5 GB 1.2 ms——天花板830 tok/s。同模型、同权重。不同内存子系统。

### Apple Neural Engine (M4 / A18)

- 峰值38 TOPS。统一内存(CPU和ANE共享池)——无拷开销。
- Core ML + `.mlmodel`编译模型访、或Metal Performance Shaders (MPS) PyTorch。
- Llama.cpp Metal backend MPS、非ANE直；原生ANE需Core ML转换。
- 2026 iOS应用最佳路：Core ML INT4权重 + FP16激活。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 峰值45 TOPS。SoC内CPU和GPU集成但分离内存域。
- QNN (Qualcomm Neural Network) SDK和AI Hub PyTorch/ONNX转换。
- Chat模板、Llama 3.2、Phi-3 AI Hub一级artifact发。

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。软件滞Apple/Qualcomm后；OpenVINO进但niche。
- Windows ARM copilot应用最佳；AMD/Intel桌面原生本地优先。

### WebGPU + WebLLM

- 浏览器WebGPU compute shaders跑模型；无安装。
- Llama 3.1 8B Q4 M3 Max ~41 tok/s——同backend原生70-80%。
- WebLLM 17.6k GitHub stars；OpenAI兼容JS API；Apache 2.0。
- 2026覆盖：Chrome Android v121+、Safari iOS 26 GA、Firefox Android追赶。整体~70-75%移动覆盖。

### NVIDIA Jetson家族

- Orin Nano Super (8GB)：装Llama 3.2 3B、Phi-3好tok/s。
- AGX Orin：vLLM跑gpt-oss-20b ~40 tok/s。
- Thor / T4000 (JetPack 7.1)：AGX Orin性能2x、EAGLE-3和NVFP4支持。
- TensorRT Edge-LLM (2026)支持EAGLE-3投机解码、NVFP4权重、分块prefill——数据中心优化移植edge。

### 每目标量化选

| 目标 | 格式 | 注 |
|--------|--------|-------|
| Apple ANE | INT4权重 + FP16激活 | Core ML转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 用`mlc_llm convert_weight` + 编译`.wasm`；GGUF不支持 |
| Jetson Orin Nano | Q4 GGUF或TRT-LLM INT4 | 内存绑 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM路径 |

### Edge长上下文陷阱

Llama 3.1 128K上下文是数据中心特性。手机8 GB RAM、4 GB模型 + 32K token 2 GB KV cache + OS开销 = OOM。Edge部署上下文保4K-8K除非激进KV量化(Q4 KV)接受。

### 语音是杀手应用

语音Agent延迟敏感(首token < 500 ms)。本地推理完全消网络延迟。配speech-to-text(Whisper Turbo变种edge跑)和edge推理成生产质量语音循环。

### 你应记数

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 ~41 tok/s。
- AGX Orin：vLLM gpt-oss-20b ~40 tok/s。
- 数据中心-edge带宽gap：30-50x。
- WebGPU移动覆盖：~70-75%(Firefox Android滞)。

## 使用

`code/main.py`算带宽绑数学跨edge目标理论解码吞吐天花板。比观察基准示带宽非计算瓶颈处。

## 交付成果

本lesson产`outputs/skill-edge-target-picker.md`。给平台(iOS/Android/browser/Jetson)、模型、延迟/内存预算、选量化格式和转换管道。

## 练习题

1. 跑`code/main.py`。7B模型Q4 Snapdragon 8 Gen 3 (~77 GB/s带宽)、算解码天花板。比观察6-8 tok/s——runtime高效否？
2. WebGPU Android需Chrome v121+。设计老浏览器fallback——同OpenAI兼容API服务器端。
3. iOS应用需4K上下文流。哪模型/格式组iPhone 16 4 GB活内存下？
4. Jetson AGX Orin gpt-oss-20b 40 tok/s。Jetson Nano仅装3B。若产品双靶、如何统一推理栈？
5. 论"WebLLM 2026生产就绪否"。引覆盖、性能、Firefox Android gap。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| ANE | "Apple神经引擎" | M系列和A系列设备NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；QNN SDK访 |
| WebGPU | "浏览器GPU" | W3C标准化浏览器GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器LLM runtime" | MLC-LLM项目；Apache 2.0；OpenAI兼容JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000家族 |
| TRT Edge-LLM | "edge TensorRT" | 2026 TensorRT-LLM edge移植；EAGLE-3 + NVFP4 |
| 统一内存 | "共享池" | CPU和NPU见同RAM；无拷开销 |
| 带宽绑 | "内存限" | 解码读权重字节/秒门 |
| Core ML | "Apple转换" | ANE原生模型Apple框架 |
| QNN | "Qualcomm栈" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计和基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon预转换模型。