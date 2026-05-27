# Omni模型:Qwen2.5-Omni和Thinker-Talker分

> GPT-4o 2024年5月产品demo颠覆非因底层模型但因产品形——语音界面你说话、模型看相机见、250ms内回复。开源生态2024和2025余下竞达产品面。Qwen2.5-Omni (2025年3月)是参考开源设计:Thinker(大文本生成transformer)加Talker(并行语音生成transformer)，链流式语音token。Mini-Omni简，Moshi匹延迟，GLM-4-Voice延中文。这课读Thinker-Talker架构和延迟预算使流式实时对话工。

**类型:** 构建
**语言:** Python (stdlib，流式流水线延迟模拟器+VAD循环)
**前置要求:** 第12阶段·19(音频LLM)，第12阶段·16(任意到任意)
**时间:** ~180分钟

## 学习目标

- 分推理流水线Thinker(文本推理)和Talker(语音合成)并解释何并行流工。
- 算对话交互首音频字节时(TTFAB)预算，逐组件。
- 描述TMRoPE时间对齐位置编码跨Thinker内视觉、音频、文本。
- 三实时对话模式:半双工、turn-taking、全双工。

## 问题背景

实时语音助手须做多，快:

1. 听用户。实时语音分词，语音活动检测(VAD)知何时说完。
2. 可选看。相机输入2-4 FPS，流入Thinker伴音频。
3. 思。组回答条件对话历史。
4. 说。合成音频token，解码波形，流用户扬声器。

每步加延迟。对话感需总往返< 500ms——下，用户停注意滞后。GPT-4o称~250ms。Moshi ~160ms。Qwen2.5-Omni ~350-500ms。

每组件须流。无"批一切后解码。"

## 概念讲解

### Thinker和Talker

Qwen2.5-Omni分解:

- Thinker:7B-80B文本生成transformer。消费交错文本+图像+音频token。输出文本token代表何说。
- Talker:更小语音生成transformer(200M-1B)。消费Thinker文本输出token加最近语音上下文token。输出离散语音token (residual-VQ索引)。
- 语音解码器:流波形解码器(SNAC, MoVQGAN族)取语音token音频样实时。

分重要。Thinker须大好推理。Talker可小因其工作局——转文本语音token。大Talker非更表达；是更慢。

并行运:

1. Thinker发文token t_i。
2. Talker消费t_i(经流)并发语音token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器消费语音token至来并发音频样。
4. Thinker文token t_{i+3}时，Talker已流t_0..t_{i+2}音频。

### TMRoPE——时间对齐多模态位置

Thinker须合图像帧(到，如4 FPS)、音频帧(到50帧/秒)、文本对话历史。朴素序(全图像、全音频、全文本)失时间对齐。

TMRoPE配绝对时间戳每token。视觉token t=2.3s。音频token t=2.32s。文本token用户"停" t=2.35s。RoPE旋注意时间戳；模型见它们时间并。

这是"他说你好时挥手"工基础设施——模型见视频帧和音频同时刻。

### 流式语音合成

语音token须流。Mini-Omni (Xie & Wu, 2024)引"语言模型可听、思时流说":Thinker输出token和Talker输出token交同序。Talker射Thinker承下文token即。无批边界。

Moshi (Défossez等，2024年10月)是最快开源实现。160ms TTFAB单A100。架构:单7B transformer发文和语音token交替位置，带"内心独白"分思流说流。这是效Thinker + Talker融合单模型带小心训练。

### VAD和turn-taking

语音活动检测运输入侧。两模式:

- 半双工:用户说，模型听。模型说，用户听。清切换经VAD静检(~200ms)。
- 全双工:两者可同说。模型可backchannel ("嗯嗯")或打断。更难。Moshi支持。

Qwen2.5-Omni默认半双工支持，带turn-taking经静阈值。全双工需应用层处理。

### Qwen3-Omni (2025年11月)

后继。Qwen3-80B Thinker，更大Talker，改进TMRoPE-v2。延迟近GPT-4o 250ms。开源权重。OmniBench基准Gemini 2.0 Live竞争。

### 生产延迟预算

典型流交互:

- Mic→音频token:40-80ms。
- 预填(提示+历史):100-200ms于7B，70B更多。
- 首Thinker文token:40ms。
- Talker处理首文token:20ms。
- 首语音token承:40ms。
- Residual-VQ解码:30ms。
- 语音波形解码:50-80ms。

总TTFAB:7B 320-510ms，70B 600-900ms。前沿质量通常意70B+；故前沿延迟差距。

### Token率数学

16kHz语音50 Hz基础语音token，你需50语音token每秒输出。Talker须发≥50 tok/s跟上。典型LLM吞吐H100上30-80 tok/s，小(200-300M) Talker快够；7B Talker落后。

这是何存小专用Talker模型而非"仅用主模型。"

## 使用它

`code/main.py`:

- 模拟Thinker-Talker流水线带mock token发率。
- 算TTFAB可配置模型大小和mic采样率。
- 示半双工turn-taking带VAD静阈值。

## 发货它

这课产`outputs/skill-omni-streaming-budget.md`。给实时语音产品目标TTFAB和特征集(视觉入、双语、全双工)，选Qwen2.5-Omni、Qwen3-Omni、Moshi、或Mini-Omni并尺寸Thinker/Talker。

## 练习题

1. 目标TTFAB 300ms。7B Thinker和300M Talker，写每组件延迟。

2. Qwen2.5-Omni用TMRoPE。描述模型见何提示用户t=1s开始说和相机t=1.2s捕手势。

3. 全双工支持需模型听时发音频。提训练数据格式教此。

4. 读Moshi论文第4节。描述"内心独白"分并何避Thinker-Talker分。

5. 吞吐预算:Talker须何快发token跟上16kHz语音50基础层token/秒？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Thinker | "推理脑" | 大文本生成transformer产何说 |
| Talker | "语音生成口" | 小transformer从Thinker文本产离散语音token |
| TTFAB | "延迟预算" | 首音频字节时:从用户语音结束到首音频样出 |
| TMRoPE | "时间对齐RoPE" | 用绝对时间戳跨视觉、音频、文本位置编码 |
| 半双工 | "Turn-taking" | 用户和模型交替；VAD静检用户完 |
| 全双工 | "同时" | 模型可说和听同；backchannel能 |
| 内心独白 | "Moshi分" | 单模型设计思流说流交 |

## 延伸阅读

- [Xu等—Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team—Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu—Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez等—Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng等—GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)