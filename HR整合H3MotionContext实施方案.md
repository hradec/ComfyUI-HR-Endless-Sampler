# HR Endless Sampler 整合 H3 Motion Context 精华实施方案

> 制定时间：2026-09-11 10:46（Asia/Singapore）  
> 状态：实施前方案  
> 目标项目：`ComfyUI-MiniMax-H3-Sampler-Unlimited`

## 一、实施目标

保持 HR Endless Sampler 的现有架构与生命周期：

- 单次执行内串行 physical chunk 采样；
- 低显存运行；
- 当前 Video1、Audio1 和 keyframe continuation；
- replay、retake、checkpoint、continuation plan；
- Preview、Timeline、Save/Load；
- Qwen/Gemma 导演链路。

只吸收 `ComfyUI-H3-Motion-Context` 的三项成熟能力：

1. H3 Layout 行为契约检查；
2. 独立于视频窗口的音频续接窗口；
3. 可量化的 chunk seam 诊断。

明确不引入：

- Motion Context 的跨运行 Save/Load Latent；
- 浏览器 Chain 状态机；
- 像素层 Trim；
- 固定 clip slot；
- 第二套 continuation 生命周期。

## 二、当前 HR 已具备的基础

### 2.1 Latent 直连

HR 当前保存 `previous_video` 与 `previous_audio`，下一 chunk 可以直接切取上一 chunk 的完成 latent，不需要经过：

```text
VAE decode → pixels/audio → VAE encode
```

这与 Motion Context 最重要的 latent-direct 原则一致。

### 2.2 音频 keyframe 定位

当前 `_conditioning_for_chunk()` 已使用：

```python
audio_start = audio_end_frame - audio_context.shape[-1] / FRAME_RESCALE
```

并生成：

```python
{
    "resolved_frame_index": audio_start,
    "audio_latent": audio_context,
}
```

HR 已支持负数、小数音频位置；仍需补充独立音频窗口、grid overhang 集中处理、layout 行为验证和实际接缝测量。

### 2.3 Physical chunk 裁剪

HR 已在 latent 层完成 physical prefix、context overlap、output trim 和最终视频/音频拼接。因此不能移植 Motion Context 的像素层 Trim，否则会重复裁剪。

## 三、设计原则

### 3.1 先测量，再改变默认行为

第一阶段只增加 contract 和诊断，不改变生成结果。

第二阶段加入独立音频窗口，但兼容默认先使用：

```text
audio_continuation_frames = 0
```

其含义为：保持旧行为，音频窗口继续跟随 `context_keyframes`。真实 GPU A/B 验证完成后，再考虑向新工作流推荐 `24`。

### 3.2 区分三种音频长度

1. **Physical audio range**：本 chunk 实际采样的音频 latent 范围，决定最终输出结构。
2. **Physical overlap audio**：与视频 physical overlap 对应、采样后被裁掉的音频部分。
3. **Conditioning audio context**：从上一 chunk 音频尾部截取并作为 keyframe 给当前 chunk 的条件，由新参数控制。

新参数不得改变 chunk 输入 latent shape、输出裁剪、最终音频长度或 chunk ownership。

### 3.3 诊断只读

Seam Probe 不得修改 latent、conditioning、replay revision 或 sampler output；不得默认增加 VAE decode，也不得长期保留完整 chunk。

## 四、阶段一：H3 Layout Contract

### 4.1 新文件

新增：

```text
h3_layout_contract.py
```

应按照 HR 当前 ComfyUI 接口重新实现，不直接复制外部插件源码。

### 4.2 行为检查

首次实际进入 H3 chunk 采样前，构造最小 fake latent，检查 `comfy.ldm.minimax.model.PackedLayout`。

#### A. 普通视频 keyframe

验证 frame 0 和 frame 3 的 cond rows 相对 target origin 距离分别为：

```text
0
3 * FRAME_RESCALE
```

#### B. Reference compensation

添加假的 reference block 后，frame 3 keyframe 相对 target origin 的距离仍必须是 `3 * FRAME_RESCALE`。

#### C. 负数、小数音频位置

构造 `[1, 32, 2, 40]` 音频 latent，并使用：

```python
resolved_frame_index = end_frame - 40 / FRAME_RESCALE
```

验证：

- `cond_audio` segment 存在；
- 行数为 `2 * 40`；
- 音频窗口终点准确；
- 窗口起点位于 target origin 之前；
- 没有整数截断或负数钳制。

#### D. 混合 keyframe

验证同一 `minimax_keyframes` 中的视频 latent、音频 latent、原有 last-frame/Add Guide keyframe可以共存。

#### E. Constructor wrapper

检测 `PackedLayout.__init__.__wrapped__` 和已知 patch marker。发现 wrapper 时先警告，随后以行为检查决定是否继续，不能仅凭存在 wrapper 就失败。

### 4.3 调用位置

在 `HREndlessSampler.execute()` 中完成基础 latent 验证与 chunk plan 后、首次 sampling 前调用：

```python
ensure_h3_layout_contract()
```

成功与失败结果均按进程缓存。

### 4.4 测试

新增：

```text
tests/test_h3_layout_contract.py
```

覆盖：

1. 正常 layout 通过；
2. keyframe index 被整数化时失败；
3. 负数位置被钳制时失败；
4. 音频窗口锚点方向错误时失败；
5. reference compensation 消失时失败；
6. 音频 row 数变化时失败；
7. 失败状态被缓存；
8. wrapper 存在但行为正确时通过；
9. 缺少 `PackedLayout` 时明确失败。

### 4.5 验收

- 默认采样结果不变；
- 首次运行只打印一次检查结果；
- contract 测试和原有回归测试通过；
- 行为异常时拒绝采样并报告实际值与预期值。

## 五、阶段二：独立音频续接窗口

### 5.1 新增输入

在 `video_continuation_res` 后追加：

```text
audio_continuation_frames
```

建议 schema：

```python
io.Int.Input(
    "audio_continuation_frames",
    default=0,
    min=0,
    max=240,
    step=1,
)
```

语义：

```text
0  = 兼容旧行为，跟随 context_keyframes
24 = 使用上一 chunk 最后 1 秒音频
48 = 使用上一 chunk 最后 2 秒音频
```

### 5.2 Geometry helper

新增纯函数：

```python
def _audio_context_geometry(
    previous_video_t,
    previous_audio_t,
    requested_frames,
    audio_end_frame,
):
    ...
```

返回 requested/effective frames、audio steps、source range、previous overhang、start/end frame 和 end coordinate。

### 5.3 时间换算

内部以 H3 40Hz 音频 latent 为准：

```python
audio_steps = round(requested_frames / fps * 40)
```

实施前必须核对 ComfyUI H3 layout 的 frame index 是否固定按原生 24FPS 解释。不得在未确认的情况下把任意输出 `fps` 直接带入 conditioning 坐标公式。

### 5.4 Previous audio overhang

```python
previous_pixel_frames = _pixel_frames(previous_video.shape[2])
ideal_audio_steps = previous_pixel_frames * FRAME_RESCALE
overhang = previous_audio.shape[-1] - ideal_audio_steps
```

合法值通常为 `0`、`+1/3`、`-1/3`。若绝对值达到或超过半个 audio step，应拒绝或明确回退，不得静默猜测。

### 5.5 精确结束定位

```python
end_frame = physical_video_context_end
end_frame += previous_overhang / FRAME_RESCALE
end_coord = round(FRAME_RESCALE * end_frame)
end_frame = end_coord / FRAME_RESCALE
audio_start_frame = end_frame - audio_steps / FRAME_RESCALE
```

生成：

```python
{
    "resolved_frame_index": audio_start_frame,
    "audio_latent": previous_audio[..., -audio_steps:].clone(),
}
```

### 5.6 与现有模式的关系

- `context_keyframes > 0`：视频可用 5/22/39/56 帧，音频可独立使用 24/48 帧。
- `context_keyframes == 0`：保留五帧 Video1 boundary keyframe；显式设置独立音频窗口时允许音频向前延伸。
- Video1 已携带 Audio1 时，默认不重复添加独立音频；只有用户显式配置时才允许共存，并记录日志。
- 不改变 `guide_overlap` 几何、warm-start noise、retained output 或 output trim。

### 5.7 Replay fingerprint

加入：

```python
"audio_continuation_frames": int(audio_continuation_frames)
```

并建议记录模式：

```text
legacy_follow_video
independent
```

每 chunk 的实际有效 step 数写入 metadata。

### 5.8 Chunk metadata

示例：

```json
{
  "audio_continuation": {
    "mode": "independent",
    "requested_frames": 24,
    "effective_frames": 24,
    "audio_steps": 40,
    "source_start": 167,
    "source_end": 207,
    "start_frame": -2.0,
    "end_frame": 22.0,
    "previous_overhang_steps": 0.3333
  }
}
```

不得保存 tensor。

## 六、阶段三：Seam Diagnostics

### 6.1 第一版范围

新增可选输入：

```text
seam_diagnostics = false
```

第一版只做 latent 与结构诊断，不强制 VAE decode。

### 6.2 边界结构记录

记录：

- 相邻 chunk 编号；
- 全局 output boundary；
- physical frame start；
- trim frames；
- video/audio context frames 与 steps；
- audio anchor start/end；
- previous audio overhang；
- continuation 模式；
- Video1/guide overlap 状态。

### 6.3 Latent 指标

在 tensors 仍在内存时计算标量：

- MAE；
- RMSE；
- max absolute error；
- cosine similarity；
- NaN/Inf；
- 音频长度差。

这些是诊断指标，不是 bit-identical 正确性断言。

### 6.4 独立 Replay Probe

第一版稳定后新增：

```text
HR Endless Seam Probe
```

它从 replay cache 加载相邻 chunk，可选接入视频/音频 VAE，输出文字报告。避免在采样期间增加显存。

音频诊断包括：

- 滑动归一化互相关；
- 可信峰跟踪；
- 平均 lag 与范围；
- broadband RMS step；
- floor/room-tone step；
- 周期信号 alias 警告。

默认参数：

```text
window_ms = 50
search_ms = 40
credible_correlation = 0.6
```

## 七、Timeline 与 Preview

### 7.1 Timeline

暂不升级 `TIMELINE_SCHEMA_VERSION`，增加可选顶层字段：

```json
"seams": []
```

`normalize_timeline()` 应：

- 接受旧 timeline 缺少 seams；
- 严格清洗数值和字符串；
- 拒绝 tensor、NaN 和 Infinity；
- 只保存小型结构与标量。

### 7.2 Preview

第一版只在边界 hover 信息中展示：

```text
Video context
Audio context
Audio anchor
Audio overhang
Latent seam cosine
```

Preview 不是数据源，所有内容来自 sampler seam record。

### 7.3 Save/Load

验证普通视频、sidecar、EXR sidecar 和 Load Video 均能保存/恢复可选 seams，同时继续加载没有 seams 的旧文件。

## 八、Replay、Retake 与 Continuation

### 8.1 Replay

以下参数变化必须导致 replay 不兼容：

```text
context_keyframes
guide_overlap
video_continuation
video_continuation_res
audio_continuation_frames
```

旧 cache 只有在新参数为兼容模式 `0` 且语义可证明等价时才允许复用。

### 8.2 Retake

- `video_only`：保留原音频 revision，诊断标记音频与视频来源不同。
- `isolated_av`：重算 `N-1 → N` 与 `N → N+1` 两个边界。
- `continuous_av`：从选中 chunk 开始重建后续 AV continuation 与 seam records。

### 8.3 Continuation checkpoint

从 checkpoint 最终 latent 计算独立 audio tail；只新增 checkpoint 到新 segment 的边界，不重写历史 seams。assemble 时正确偏移新 timeline 的全局 frame。

## 九、测试矩阵

### 9.1 Geometry 组合

| 视频上下文 | 音频上下文 | 预期 |
|---:|---:|---|
| 0 | 0 | 保持旧 synthetic-prefix 行为 |
| 5 | 0 | 音频跟随旧 5 帧 |
| 5 | 24 | 视频 2 steps，音频 40 steps |
| 22 | 24 | 视频 7 steps，音频 40 steps |
| 22 | 48 | 视频 7 steps，音频 80 steps |
| 0 | 24 | 五帧 Video1 boundary + 独立音频 |
| 39 | 24 | 长视频头、短音频窗 |
| 5 | 超出上一 chunk | clamp 并记录有效值 |

覆盖上一 chunk 帧数对 3 的三种余数，验证 overhang 为 `0`、`+1/3`、`-1/3`。

### 9.2 Conditioning

断言：

- 原 `minimax_refs` 不变；
- last-frame keyframe 保留；
- 冲突 keyframe 遵循现有 HR 规则；
- 音频 keyframe允许负数、小数；
- 音频窗口精确结束于 boundary；
- Video1 Audio1 与独立 audio keyframe 不被意外替换。

### 9.3 Output geometry

新功能不得改变 chunk video/audio 输入 shape、output trim、最终 video/audio steps 或 NestedTensor 结构。

### 9.4 回归测试

至少执行：

```bash
python tests/test_chunk_director_helpers.py
python tests/test_retake_director.py
python tests/test_continuation.py
python tests/test_video_io.py
python -m compileall ...
git diff --check
```

完整 discover 若因真实 ComfyUI 外部模块缺失失败，必须单独报告环境依赖错误，不能把部分通过描述成完整通过。

## 十、真实 GPU 验收

### 10.1 基准内容

1. 持续音乐：检测节拍、25ms/8.3ms 偏差；
2. 连续环境声：检测 room-tone 重启；
3. 连续对白/呼吸：检测音色、韵律和 lip-sync 边界。

### 10.2 配置对照

| 组 | 视频上下文 | 音频上下文 | Video1 | 目的 |
|---|---:|---:|---:|---|
| A | 5 | legacy | on | 当前基线 |
| B | 5 | 24 | on | 低视频成本、1 秒音频 |
| C | 22 | 24 | on | Motion Context 推荐组合 |
| D | 5 | 48 | on | 2 秒音频 |
| E | 0 | 24 | on | synthetic prefix + 独立音频 |
| F | 22 | legacy | on | 当前长 keyframe 基线 |

记录峰值 VRAM/RAM、H3 时间、conditioning rows、音频 correlation/lag、broadband/floor step、视频边界差和主观评分。

### 10.3 建议成功阈值

```text
mean correlation >= 0.6
credible windows >= 70%
abs(mean lag) <= 12 ms
floor step < 0.25
```

同时要求无明显重复动作、冻结、亮度或颜色突变，输出长度与基线一致，且不破坏 12GB 工作流。

## 十一、分批提交策略

### Commit 1

```text
feat: validate MiniMax H3 continuation layout semantics
```

只包含 layout contract、测试和首次调用，不改变生成。

### Commit 2

```text
feat: add independent H3 audio continuation window
```

包含 schema、geometry helper、conditioning、fingerprint、metadata 和专项测试，默认保持兼容。

### Commit 3

```text
feat: record HR Endless chunk seam diagnostics
```

包含 latent 指标、replay metadata、timeline seams 和 normalization tests，默认关闭。

### Commit 4

```text
feat: add replay-based HR Endless seam probe
```

包含独立诊断节点、音频互相关、可选视频边界检查、报告和 fixture tests。

## 十二、最终验收条件

1. 旧工作流在兼容默认下行为不变；
2. HR physical chunk 生命周期不变；
3. 最终 AV shape 与时长不变；
4. 音频窗口可独立于视频窗口；
5. 支持 24/48 帧音频 tail；
6. 音频 keyframe精确结束于 continuation boundary；
7. layout 语义异常时拒绝采样；
8. replay fingerprint涵盖新行为；
9. retake 三种模式正确；
10. continuation checkpoint 三种音频策略正确；
11. seam 诊断默认不增加 VAE 工作；
12. 旧 Timeline 和视频 sidecar 可继续加载；
13. 真实 12GB GPU 基线仍可运行；
14. 音乐、环境声、对白各完成至少一次两 chunk A/B；
15. 只有测量证明 24 帧音频窗口更优后，才考虑调整推荐默认值。
