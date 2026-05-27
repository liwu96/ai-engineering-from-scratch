# 毕业项目 12 —— 视频理解流水线 (场景、QA、搜索)

> Twelve Labs产品化Marengo + Pegasus。VideoDB出货CRUD-for-video API。AI2 Molmo 2发布开放VLM checkpoints。Gemini长context原生处理小时级视频。TimeLens-100K定义规模时间grounding。2026流水线已定: 场景分割、每场景caption + embedding、transcript对齐、多向量索引、和带(start, end)时间戳加帧预览回答query。毕业项目是摄入100小时、击公开benchmark、并测计数和动作问题幻觉。

**类型:** 毕业项目
**语言:** Python (流水线)、TypeScript (UI)
**前置要求:** 第4阶段(CV)、第6阶段(语音)、第7阶段(transformers)、第11阶段(LLM工程)、第12阶段(多模态)、第17阶段(基础设施)
**涉及阶段:** P4 · P6 · P7 · P11 · P12 · P17
**时间:** 30小时

## 问题背景

长视频QA是2026规模带宽最 hungry多模态问题。Gemini 2.5 Pro可原生读2小时视频、但摄入100小时视频入可query corpus仍需场景级索引。产形态结合场景分割(TransNetV2或PySceneDetect)、每场景VLM captioning (Gemini 2.5、Qwen3-VL-Max、或Molmo 2)、transcript对齐(Whisper-v3-turbo带word时间戳)、和多向量索引并存caption、帧embedding、和transcript。Query流水线带(start, end)时间戳加帧预览回答。

Benchmark公开(ActivityNet-QA、NeXT-GQA)加自定义100-query集。计数和动作类型问题幻觉是已知hard失败类; 毕业项目显式测量。

## 概念讲解

摄入时三流水线并行运行。**场景分割**切视频成场景。**VLM captioning**每场景生成caption和keyframe embedding。**ASR对齐**产word级时间戳。三流按(scene_id、time range)连接。每场景得三向量类型入多向量索引(Qdrant): caption embedding、keyframe embedding、transcript embedding。

Query时、自然语言问题对三向量fired; 结果RRF合并; 时间grounding适配器(TimeLens-style)于top场景内细化(start, end)窗口。VLM synthesizer (Gemini 2.5 Pro或Qwen3-VL-Max)取query + top场景 +裁剪帧并带引用时间戳和帧预览回答。

幻觉测量重要。计数("多少人进房间?")和动作类型("厨师先倒再搅否?")问题 notoriously unreliable。分开报告准确性与描述问题。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割: TransNetV2 (state-of-the-art 2024-26) 或 PySceneDetect
- ASR: Whisper-v3-turbo via faster-whisper带word时间戳
- VLM captioner + answerer: Gemini 2.5 Pro或Qwen3-VL-Max或Molmo 2
- 时间grounding: TimeLens-100K-trained适配器或VideoITG
- 索引: Qdrant多向量支持(caption / frame / transcript)
- UI: Next.js 15带HTML5 video player和场景thumbnails
- 评估: ActivityNet-QA、NeXT-GQA、自定义100问手标注集
- 幻觉benchmark: 计数和动作类型子集带手标签

## 动手实践

1. **摄入walker。** 接YouTube URLs或本地MP4s。需时downscale到720p。持久`{video_id, file_path}`。

2. **场景分割。** 运行TransNetV2或PySceneDetect产`[{scene_id, start_ms, end_ms, keyframe_path}]`。目标100小时: ~6k-8k场景。

3. **ASR pass。** 于audio运行Whisper-v3-turbo; 导word级时间戳; 分成每场景transcript切片。

4. **VLM captioning。** 每场景、调Gemini 2.5 Pro (或Qwen3-VL-Max)带keyframe和短caption模板。产caption +帧embedding。

5. **多向量索引。** Qdrant collection三命名向量。Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **Query。** 自然语言问题fired三dense queries; merge用reciprocal rank fusion; top-k=5场景。

7. **时间grounding。** 于top场景运行TimeLens-style适配器细化(start, end)窗口。

8. **VLM synth。** 调Gemini 2.5 Pro带query + top-3场景clips (作图像或短clip) + transcripts。需`(video_id, start_ms, end_ms)`引用。

9. **评估。** 运行ActivityNet-QA和NeXT-GQA。建100问自定义集。报整体准确率 + 每类分解(计数、动作、描述)。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 产出成果

`outputs/skill-video-qa.md`是deliverable。给YouTube URL或上传视频、流水线索引场景并带时间戳引用回答问题。

| 权重 | 标准 | 何测 |
|:-:|---|---|
| 25 | 时间grounding IoU | held-out grounding集Intersection-over-union |
| 20 | QA准确率 | NeXT-GQA和自定义100问 |
| 20 | 摄入吞吐 | 每美元消耗视频小时数 |
| 20 | UI和引用UX | 时间戳链接、thumbnail strip、jump-to-frame |
| 15 | 幻觉率 | 计数和动作类型准确率分开 |
| **100** | | |

## 练习题

1. 于captioning pass换Gemini 2.5 Pro为Qwen3-VL-Max。报人工评分50场景样本caption质量delta。

2. 减每场景帧embedding为单池化向量而非多向量。测检索回归。

3. 建"计数strict"模式: synthesizer提取每计数实例带时间戳、用户点击验证。测用户验证降幻觉否。

4. Benchmark摄入成本: 三VLM选择video-hours-per-dollar。择sweet spot。

5. 加speaker-diarized transcript: 于audio运行pyannote speaker diarization并embed每speaker transcripts。演示"Alice关于X说什么?" queries。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 场景分割 | "镜头检测" | 于镜头边界切视频成场景 |
| 多向量索引 | "Caption +帧 + transcript" | Qdrant collection带每表示命名向量 |
| 时间grounding | "何时确切发生" | 细化query答案(start, end)窗口 |
| 帧embedding | "视觉表示" | keyframe向量embedding; 用于场景视觉相似 |
| RRF fusion | "Reciprocal rank fusion" | 跨多排序列表merge策略; classic hybrid-retrieval trick |
| 计数幻觉 | "错误计数" | VLM "多少X"问题已知失败模式 |
| ActivityNet-QA | "Video-QA benchmark" | 长视频QA准确率benchmark |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 规模时间grounding
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — CRUD-for-video API参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — classic开放备选
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估benchmark