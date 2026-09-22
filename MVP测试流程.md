# HR Storyboard + Reference Set MVP 测试流程

> 测试日期：2026-09-03
> 目标：在真实 ComfyUI 中验证 Qwen3.8 分镜规划、H3 Ref2VA 编码和 HR Endless Sampler 两个 physical chunks。

## 1. 重启与节点检查

重启 ComfyUI 后确认以下节点存在：

- `HR Qwen3.8 Director Config`
- `HR MiniMax H3 Reference Set`
- `HR MiniMax H3 Storyboard Planner`
- `HR MiniMax H3 Reference Conditioning`
- `HR Endless Sampler`
- `HR Endless Sampler Preview`

如果任一节点缺失，先保存完整启动错误，不要进入 GPU 测试。

## 2. 最小接线

```text
HR Qwen3.8 Director Config.director_config
  ├─→ HR MiniMax H3 Storyboard Planner.director_config
  └─→ HR Endless Sampler.director_config

Load Image(s)
  └─→ HR MiniMax H3 Reference Set.ref_image_N

HR MiniMax H3 Reference Set.reference_set
  ├─→ HR MiniMax H3 Storyboard Planner.reference_set
  └─→ HR MiniMax H3 Reference Conditioning.reference_set

HR MiniMax H3 Storyboard Planner.prompt
  ├─→ HR MiniMax H3 Reference Conditioning.prompt
  └─→ HR Endless Sampler.prompt

HR MiniMax H3 Storyboard Planner.planned_frames
  └─→ HR MiniMax H3 Reference Conditioning.length

HR MiniMax H3 Reference Conditioning.positive
  └─→ Guider positive

HR MiniMax H3 Reference Conditioning.latent
  └─→ HR Endless Sampler.latent_image

HR MiniMax H3 Reference Conditioning.reference_set
  └─→ HR Endless Sampler.reference_set
```

其余 sampler/noise/sigmas/model/clip/vae 和 Preview 连接沿用现有可用工作流。

## 3. 测试 A：只运行 Planner

参数：

```text
duration_seconds = 5.0
fps = 24
style = cinematic realism
shot_density = medium
```

预期：

- `planned_frames = 124`；
- `prompt` 有六个 H3 字段；
- `[Shot 1]` 没有时间码；
- 后续 `[Shot N] At MM:SS.mmm,` 时间严格递增；
- `story_plan` 中 shots 连续覆盖 `[0,124)`；
- Qwen worker 完成后显存释放。

## 4. 测试 B：Planner + Conditioning

先只接 1 张图片：

```text
ref_image_size = match
ref_scale = 1.0
width = 1344
height = 768
```

预期：

- Conditioning 返回 positive 和 nested AV latent；
- positive 含一个 image 类型的 `minimax_refs`；
- latent 视频/音频时长对应 124 帧；
- `<Picture 1>` 与该图片一致。

## 5. 测试 C：两个 Chunk

为了强制两个 physical chunks：

```text
duration_seconds = 5.0
planned_frames = 124
chunk_frames = 56
video_continuation = 22
debug = true
debug_stop_chunk = 2
```

预期：

1. Planner 的 Qwen3.8 worker 退出；
2. H3 Chunk 1 完成；
3. Qwen3.8 观察 Chunk 1 并生成 Chunk 2 prompt；
4. Qwen worker 再次退出；
5. H3 Chunk 2 成功加载并采样；
6. 没有 Qwen/H3 同驻导致的 OOM；
7. chunk prompt 的 Shot 时间码是当前 chunk 的本地时间；
8. Timeline 包含两个已完成 chunks。

## 6. 测试 D：多图

连接 2–3 张不同尺寸图片，保持 Reference Set 中顺序不变。

检查：

- Planner 的 `image_subjects` 编号连续；
- prompt 中 `<Picture N>` 对应实际顺序；
- Conditioning 的 `minimax_refs` 顺序一致；
- Sampler 不报告图片数量不匹配。

## 7. 测试 E：视频和音频协议

在图片测试通过后再执行：

- `ref_video_0` + `ref_video_audio_0`；
- `ref_video_1` + `ref_video_audio_1`，但允许 `ref_video_audio_0` 留空；
- 一条独立 `ref_audio_0`。

检查：

- 同步音轨只与同索引视频配对；
- tokenizer 顺序是图片、视频音轨、对应视频、独立音频；
- `minimax_refs` 中视频携带对应 `audio_latent`；
- Sampler 能在后续 chunk 重建视频/音频语义 presentation。

注意：当前 Planner 只分析图片；视频和音频会传给 H3 Conditioning/Sampler，但不会参与首版故事分镜的 VLM 分析。

## 8. 失败时保留

请保留：

- ComfyUI 启动日志；
- 完整异常和 traceback；
- Planner 输出的 prompt/story_plan/shot_report；
- `${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_chunk_prompts.txt`；
- 实际节点截图；
- GPU 型号、显存和关键参数；
- 失败发生在 Planner、Conditioning、Chunk 1、导演观察还是 Chunk 2。
