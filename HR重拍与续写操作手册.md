# HR Endless Sampler 重拍与续写操作手册

> 文档日期：2026-09-10  
> 适用项目：ComfyUI-MiniMax-H3-Sampler-Unlimited（`mickeylan` 分支）  
> 功能状态：代码实现完成，等待真实 ComfyUI 与 GPU 环境验收

本文只介绍 **HR Endless Sampler** 节点族的分块重拍与持久续写，不涉及独立 Story Director 项目的故事规划流程。

---

## 1. 功能概览

### 1.1 分块重拍

分块重拍允许复用最近一次完整生成的缓存，只重新采样选中的 physical chunk。支持：

- 只重拍画面并保留原音频；
- 独立重拍选中段的画面和音频；
- 从最早选中段开始连续重拍到结尾；
- 为同一 chunk 保存多个 revision；
- 切换原版或重拍版本后，无采样重新拼接。

相关节点：

- `HR Endless Segment Retake Director`
- `HR Endless Sampler`
- `HR Endless Retake Assemble`

### 1.2 持久续写

持久续写允许把最近一次完整生成固化为 checkpoint，再以其最后一段音视频状态作为新片段的连续性起点。

支持：

- checkpoint 跨 ComfyUI 重启保存；
- 新提示词续写；
- 继续声音、新音频段或静音；
- 继承、替换或合并参考媒体；
- 将旧结果和新结果重新拼接。

相关节点：

- `HR Endless Continuation Checkpoint`
- `HR Endless Continuation Plan`
- `HR Endless Sampler`
- `HR Endless Continuation Assemble`

---

## 2. 重要概念

### 2.1 Physical chunk

HR Endless Sampler 不会一次采样完整长视频，而是把 H3 音视频 latent 划分成多个 physical chunks，逐块完成完整扩散过程。

重拍选择的是这些 physical chunks，不是 Story Director 的故事段或逻辑镜头。

### 2.2 Replay cache

每次正常完整生成时，Sampler 会保存最近一次运行的临时 replay cache，其中包括：

- 每个 chunk 的采样结果；
- output 和 denoised latent；
- 实际 H3 prompt；
-观察图片；
-分块范围和 fingerprint；
-重拍 revision。

重拍直接依赖这份“最近一次运行缓存”。运行另一套普通 Sampler 工作流可能替换它。

### 2.3 Continuation checkpoint

Continuation checkpoint 是从完整 replay cache 建立的持久副本，默认保存在：

```text
ComfyUI/output/hr_endless_sampler/continuations/<checkpoint-id>/
├─ manifest.json
└─ final_state.pt
```

它和临时 replay cache 不同：正常重启 ComfyUI 或清理临时缓存后仍然可以使用。

### 2.4 Revision

重拍不会覆盖原始 chunk，而是创建新 revision：

```text
原版：revision 0
第一次重拍：revision 1
第二次重拍：revision 2
...
```

可在重拍导演节点中选择每个 chunk 当前启用的版本，再通过 Retake Assemble 重新拼接。

---

## 3. 使用前准备

确认 ComfyUI 已加载以下节点：

```text
HR Endless Sampler
HR Endless Segment Retake Director
HR Endless Retake Assemble
HR Endless Continuation Checkpoint
HR Endless Continuation Plan
HR Endless Continuation Assemble
HR Endless Sampler Save Video
```

还需要一套能正常生成 MiniMax H3 音视频的基础工作流，包括：

- MODEL/Guider；
- Noise；
- Sampler；
- Sigmas；
- MiniMax H3 CLIP；
- MiniMax H3 视频 VAE；
- MiniMax H3 音频 VAE；
- nested AV latent；
- H3 格式 prompt；
- 可选参考媒体。

### 首次验收建议

- 使用较短时长；
- 让基线包含 2–3 个 chunks；
- 打开 `debug` 以保留更详细日志；
- 不设置 `debug_stop_chunk`；
- 先保存基线成片，便于逐帧和逐音轨对比；
- 记录 FPS、总帧数、`chunk_frames`、`video_continuation`、分辨率和 seed。

---

# 第一部分：分块重拍

## 4. 第一步：生成完整基线

使用正常 HR Endless Sampler 工作流完成一次生成：

```text
Noise ───────────────┐
Guider ──────────────┤
Sampler ─────────────┤
Sigmas ──────────────┤
H3 AV Latent ────────┤
CLIP ────────────────┤
H3 Prompt ───────────┤
Video VAE ───────────┤
Reference Set ───────┤
                     ▼
              HR Endless Sampler
                     ├─ output
                     ├─ denoised_output
                     ├─ chunk_prompts
                     └─ timeline
```

基线要求：

1. 所有 chunks 全部完成；
2. `debug_stop_chunk=0`；
3. 运行中没有中止或 OOM；
4. replay cache 状态为 complete；
5. 重拍前不要用其他普通工作流覆盖最近一次 Sampler 缓存。

建议同时保存：

```text
HR Endless Sampler.output → VAE Decode → HR Endless Sampler Save Video
HR Endless Sampler.timeline → HR Endless Sampler Save Video.timeline
```

基线视频是判断重拍是否正确的对照样本。

---

## 5. 第二步：配置重拍导演

添加节点：

```text
HR Endless Segment Retake Director
```

该节点不需要连接基线 Sampler 的输出。它通过后端接口读取最近一次 HR Endless Sampler replay cache。

节点面板应显示：

-缓存状态；
-已完成 chunk 数；
-每个 chunk 的编号和时间范围；
-观察图片；
-原最终 H3 prompt；
-提示词编辑框；
-重拍模式；
-revision 选择器。

如果面板没有内容：

1. 点击“刷新缓存”；
2. 确认基线生成已完整结束；
3. 检查 ComfyUI 控制台；
4. 确认未运行另一套覆盖缓存的工作流；
5. 刷新浏览器后重试。

---

## 6. 第三步：选择重拍模式

### 6.1 `video_only`：仅重拍画面

适合：

-人物动作错误；
-构图或运镜不理想；
-画面出现瑕疵；
-原对白、声音和节奏正确。

行为：

-重新采样选中 chunk 的视频；
-使用原版本的音频输出；
-保存新的视频 revision。

首次测试建议优先使用该模式，因为最容易对比“画面变化、音频不变”。

### 6.2 `isolated_av`：独立重拍音画

适合：

-选中段的画面和音频都需要修改；
-后续段落无需跟随改变；
-希望减少重新采样成本。

行为：

-只重新生成选中 chunk 的视频和音频；
-未选择 chunks 使用缓存中的活动版本；
-后续 chunk 不基于新结果重新生成。

风险：选中段结束状态发生明显变化时，和后续旧 chunk 的边界可能不连续。

### 6.3 `continuous_av`：连续重拍音画

适合：

-修改会改变人物位置、动作状态或场景；
-需要保证后续连续性；
-愿意接受更高的重新采样成本。

行为：

-找到最早选中的 chunk；
-从该 chunk 开始一直重新采样到最后一个 chunk；
-每个后续 chunk 都继承新生成的前一块状态。

例如选择 Chunk 2，而基线共有 5 个 chunks：

```text
Chunk 1：复用活动版本
Chunk 2：重拍
Chunk 3：连续重拍
Chunk 4：连续重拍
Chunk 5：连续重拍
```

---

## 7. 第四步：选择 chunk 并修改提示词

在重拍导演面板中：

1. 勾选一个或多个完整 chunks；
2. 查看观察图和原最终 H3 prompt；
3. 在文本框填写该 chunk 的新 H3 prompt；
4. 留空则沿用缓存中的原最终 H3 prompt；
5. 选择重拍模式。

提示词建议：

-只修改确实需要变化的内容；
-保留人物、服装、场景和镜头连续性描述；
-保留 H3 所需的结构和引用标签；
-避免改变不属于当前 chunk 的全局时间码；
-`video_only` 模式下不要期待提示词改变原音频。

重拍导演输出：

```text
retake plan
plan JSON
```

---

## 8. 第五步：连接重拍计划

连接：

```text
HR Endless Segment Retake Director.retake plan
                  │
                  ▼
HR Endless Sampler.retake_plan
```

重拍使用的 HR Endless Sampler 应保持与基线一致：

-相同 FPS；
-相同总 latent 长度；
-相同宽高；
-相同 `chunk_frames`；
-相同基础 prompt 结构；
-相同模型与 conditioning；
-相同参考媒体；
-相同 Director 配置；
-相同影响分块计划的参数。

Sampler 会比较 replay fingerprint。配置不兼容时应停止并报错，而不是错误复用旧缓存。

必须留空：

```text
HR Endless Sampler.continuation_plan
```

`retake_plan` 和 `continuation_plan` 不能同时使用。

---

## 9. 第六步：执行重拍

运行连接了 `retake_plan` 的 HR Endless Sampler。

预期行为：

-未选择的 chunk 从缓存复用；
-选中的 chunk 按模式重新采样；
-`continuous_av` 会继续采样到最后一块；
-新结果保存为 revision，不覆盖原版；
-输出可立即解码和保存。

观察控制台，记录：

-识别出的重拍模式；
-实际采样的 chunk 编号；
-复用的 chunk 编号；
-新 revision 保存结果；
-显存峰值；
-是否发生 cache fingerprint 错误。

---

## 10. 第七步：切换 revision

重拍完成后，在 `HR Endless Segment Retake Director` 点击刷新。

每个有重拍记录的 chunk 应出现：

```text
原版
重拍 1 · video_only
重拍 2 · isolated_av
...
```

选择器决定该 chunk 的活动版本。

例如：

```text
Chunk 1 → 原版
Chunk 2 → 重拍 2
Chunk 3 → 原版
Chunk 4 → 重拍 1
```

切换 revision 不会自动重新采样。

---

## 11. 第八步：无采样重新拼接

添加：

```text
HR Endless Retake Assemble
```

该节点无需输入，会读取 replay cache 中所有 chunks 的活动 revision。

输出：

```text
output
denoised_output
timeline
```

推荐连接：

```text
HR Endless Retake Assemble.output
        → MiniMax H3 VAE Decode
        → HR Endless Sampler Save Video.images

HR Endless Retake Assemble.timeline
        → HR Endless Sampler Save Video.timeline
```

如果保存音频，还需从解码链中连接对应 AUDIO。

执行 Assemble 后应验证：

-总帧数和基线一致；
-音频长度和视频同步；
-每个 chunk 使用所选 revision；
-切回原版后结果可恢复；
-此步骤没有触发 H3 采样。

---

## 12. 推荐的首次重拍验收

准备一个包含 3 个 chunks 的短基线。

### 测试 A：video_only

```text
选择：Chunk 2
模式：video_only
提示词：明显改变动作或镜头方向
```

检查：

-Chunk 1、3 未重新采样；
-Chunk 2 画面明显变化；
-Chunk 2 音频与原版完全一致；
-revision 可切换；
-Assemble 无采样完成。

### 测试 B：isolated_av

```text
选择：Chunk 2
模式：isolated_av
```

检查 Chunk 2 音画均变化，同时重点观察 Chunk 2→3 边界。

### 测试 C：continuous_av

```text
选择：Chunk 2
模式：continuous_av
```

检查实际重新采样 Chunk 2 和 Chunk 3，并比较与 isolated 模式的边界连续性。

---

# 第二部分：持久续写

## 13. 续写工作流总览

续写建议分三次 Queue：

```text
第一次 Queue：完整生成原始片段
第二次 Queue：创建持久 checkpoint
第三次 Queue：生成新片段并 Assemble
```

不要假定在同一次 Queue 中，Checkpoint 节点一定会在 Sampler 完整写完 replay cache 后才执行。

---

## 14. 第一步：完整生成原始片段

与重拍基线一样，先完成一次正常 HR Endless Sampler 生成。

要求：

-所有 chunks 完成；
-`debug_stop_chunk=0`；
-replay cache 为 complete；
-原始结果已经检查并保存；
-创建 checkpoint 前不要运行其他普通 Sampler 工作流。

记录：

-原始 FPS；
-原始总帧数；
-最后一个 chunk；
-原 Reference Set；
-原音频状态；
-原视频末尾画面。

---

## 15. 第二步：创建 Continuation Checkpoint

添加：

```text
HR Endless Continuation Checkpoint
```

输入：

- `name`：检查点名称；
- `reference_set`：可选，建议连接原生成使用的 Reference Set。

示例：

```text
name = 第一集完成版
```

连接可选参考：

```text
原 HR MiniMax H3 Reference Set.reference_set
                     │
                     ▼
HR Endless Continuation Checkpoint.reference_set
```

单独 Queue 该节点。

输出：

```text
checkpoint
checkpoint info
```

`checkpoint info` 应包含：

- `checkpoint_id`；
-名称；
-创建时间；
-FPS；
-总帧数；
-时长；
-最终 chunk；
-状态和 fingerprint。

执行后确认目录存在：

```text
output/hr_endless_sampler/continuations/<checkpoint-id>/manifest.json
output/hr_endless_sampler/continuations/<checkpoint-id>/final_state.pt
```

注意：`final_state.pt` 可能很大，因为会保存音视频 latent。不要手动只移动其中一个文件。

---

## 16. 第三步：准备新片段

续写 Sampler 的 `latent_image` 只表示“新增片段”的长度，不应包含原视频长度。

例如原视频 10 秒，现在续写约 5 秒：

```text
原视频：保存在 checkpoint 中
新 latent：只创建约 5 秒，对齐 H3 17k+5 帧网格
```

建议首次测试只续写 1–2 个 chunks。

新片段需要：

-新的 H3 prompt；
-新的 H3 nested AV latent；
-与 checkpoint 相同的 FPS；
-正常 Noise/Guider/Sampler/Sigmas；
-视频 VAE；
-按策略使用的 Reference Set。

---

## 17. 第四步：创建 Continuation Plan

添加：

```text
HR Endless Continuation Plan
```

连接：

```text
HR Endless Continuation Checkpoint.checkpoint
                    │
                    ▼
HR Endless Continuation Plan.checkpoint
```

填写：

- `prompt`：新片段的完整 H3 prompt；
- `audio_mode`：声音策略；
- `reference_policy`：参考媒体策略；
- `reference_set`：可选的新参考媒体。

输出：

```text
continuation plan
plan JSON
```

---

## 18. 音频策略

### 18.1 `continue`

使用旧片段最后的音频尾部作为新片段 Audio1 连续条件。

适合：

-同一场景继续；
-连续环境声；
-对白或音乐需要自然衔接。

### 18.2 `new_segment`

保持视频尾部连续性，但不延续旧音频的实际内容，使用静音尾部作为第一块音频边界条件。

适合：

-画面连续但开始新对白；
-进入新的音乐或声音段落；
-避免旧语音内容被重复延续。

### 18.3 `mute`

不但不继承旧音频内容，还将新生成片段的 assembled audio 清零。

适合：

-只验收画面连续性；
-后期单独配音；
-排查音频路径问题。

`new_segment` 与 `mute` 不相同：前者允许新片段正常生成自己的音频，后者会把新片段输出音频置为静音。

---

## 19. 参考媒体策略

### 19.1 `inherit`

只使用 checkpoint 中保存的原 Reference Set。

前提：创建 checkpoint 时连接了原 `reference_set`。

适合人物、服装、场景和声音参考保持不变的续集。

### 19.2 `replace`

只使用 Continuation Plan 当前连接的新 Reference Set。

适合新章节或更换角色、场景、视频和音频参考。

如果未连接新 Reference Set，则有效参考可能为空。

### 19.3 `inherit_plus_replace`

在原 Reference Set 后追加新 Reference Set。

适合保留主角参考，同时加入新角色、场景或声音。

合并后仍受 H3 限制：

-图片最多 9 张；
-参考视频最多 3 个；
-视频音轨与视频按索引配对；
-独立音频最多 3 条。

首次测试建议使用 `inherit` 或 `replace`，待单次续写通过后再测试合并策略。

---

## 20. 第五步：连接续写 Sampler

核心连接：

```text
Continuation Plan.continuation plan
                 │
                 ▼
HR Endless Sampler.continuation_plan
```

新片段的基础采样连接与普通 Sampler 相同：

```text
新 Noise ───────────────┐
新 Guider ──────────────┤
Sampler ────────────────┤
Sigmas ─────────────────┤
新 H3 AV Latent ────────┤
CLIP ───────────────────┤
新 Prompt ──────────────┤
Video VAE ──────────────┤
Continuation Plan ──────┤
                        ▼
                 HR Endless Sampler
```

关键参数：

- `fps` 必须与 checkpoint 完全一致；
- `latent_image` 是新片段长度；
- `video_continuation` 使用已验证的值，例如 22；
- `vae` 必须连接；
- `retake_plan` 必须留空；
-新 prompt 建议同时供新 conditioning 使用；
-参考媒体由 Continuation Plan 的策略决定，避免无意重复连接。

执行时 Sampler 会：

1. 加载 checkpoint；
2. 用 plan 中的 prompt 替换本次 prompt；
3. 读取 checkpoint 最后一个已采样 chunk；
4. 将其视频尾部作为 Video1；
5. 按音频模式处理 Audio1；
6. 为新片段第一块建立 continuation；
7. 只输出新增片段。

---

## 21. 第六步：拼接旧片段与新片段

添加：

```text
HR Endless Continuation Assemble
```

连接：

```text
HR Endless Continuation Checkpoint.checkpoint
                    │
                    ▼
HR Endless Continuation Assemble.checkpoint

续写 HR Endless Sampler.output
                    │
                    ▼
HR Endless Continuation Assemble.continuation_output

续写 HR Endless Sampler.denoised_output
                    │
                    ▼
HR Endless Continuation Assemble.continuation_denoised
```

输出：

```text
output
denoised_output
```

再连接解码和保存：

```text
Continuation Assemble.output
          → MiniMax H3 VAE Decode
          → HR Endless Sampler Save Video
```

当前 Continuation Assemble 不输出合并后的 `HRENDLESS_TIMELINE`。首次测试应通过帧数、音频长度和实际成片检查拼接结果。

---

## 22. 续写工作流接线图

```text
【原片段生成】
Normal HR Endless Sampler
        │ 写入完整 replay cache
        ▼

【建立检查点】
Reference Set（可选） ───────┐
                             ▼
HR Endless Continuation Checkpoint
        ├─ checkpoint ───────────────┬─────────────────────────────┐
        └─ checkpoint info           │                             │
                                     ▼                             │
                          HR Endless Continuation Plan              │
                                     │ continuation plan            │
                                     ▼                             │
新 Noise/Guider/Sampler/Sigmas ─→ HR Endless Sampler               │
新 H3 AV Latent ────────────────→ latent_image                      │
新 Conditioning ────────────────→ guider                            │
Video VAE ──────────────────────→ vae                               │
                                     │ output                       │
                                     ├──────────────────────────┐   │
                                     │ denoised_output           │   │
                                     └──────────────────────┐   │   │
                                                            ▼   ▼   ▼
                                              HR Endless Continuation Assemble
                                                        ├─ output
                                                        └─ denoised_output
```

---

## 23. 推荐的首次续写验收

### 测试 A：基本连续续写

基线生成 2 个 chunks，然后：

```text
checkpoint name = 基线短片
新片段 = 1 个 chunk
audio_mode = continue
reference_policy = inherit
```

检查：

-Checkpoint 创建成功；
-关闭并重启 ComfyUI 后仍能通过已保存工作流使用该 checkpoint 输出；
-续写第一块正确使用旧片段 Video1/Audio1；
-旧片尾到新片头画面连续；
-音频没有明显断裂或重复；
-Assemble 后总长度等于旧片段加新片段。

### 测试 B：新音频段

```text
audio_mode = new_segment
reference_policy = inherit
```

检查视频仍连续，同时新音频不重复旧对白。

### 测试 C：静音续写

```text
audio_mode = mute
```

检查新片段的 audio latent 为静音，旧片段音频保持不变。

### 测试 D：替换参考媒体

```text
audio_mode = new_segment
reference_policy = replace
```

连接新的 Reference Set，检查新片段使用新参考，但仍从旧视频末尾继续。

### 测试 E：合并参考媒体

```text
reference_policy = inherit_plus_replace
```

检查合并后的图片、视频和音频未超过上限，引用顺序符合预期。

---

## 24. 多次链式续写

理论流程：

```text
片段 A → Checkpoint A
Checkpoint A → 续写 B → Assemble A+B
续写运行完成后 → 创建下一个 Checkpoint
新 Checkpoint → 续写 C
```

但当前 checkpoint 是从“最近一次 Sampler replay cache”创建。首次真实测试时应先确认第二个 checkpoint 保存的是预期基底和尾部状态，再进行第三段，不要直接进行长链生成。

建议：

1. 每次续写后保存成片；
2. 核对新 replay cache 的帧数和最终 chunk；
3. 创建新 checkpoint；
4. 保存 `checkpoint info`；
5. 再开始下一轮。

---

# 第三部分：故障排查

## 25. 重拍导演显示没有缓存

可能原因：

-还没有完整运行 HR Endless Sampler；
-基线被 `debug_stop_chunk` 提前终止；
-最近一次运行失败；
-临时缓存被清理；
-另一个工作流覆盖了 last-run cache。

处理：重新完整生成短基线，然后立即刷新重拍导演。

---

## 26. Retake plan 不匹配当前缓存

典型错误：

```text
Retake plan does not match the current last-run cache
```

原因：生成 retake plan 后，缓存或影响 fingerprint 的设置发生变化。

处理：

1. 恢复基线工作流原参数；
2. 确保目标 replay cache 是当前 last-run cache；
3. 在重拍导演中刷新；
4. 重新选择 chunks 并生成计划。

不要手工编辑 cache identity。

---

## 27. 重拍后边界不连续

优先检查：

-是否用了 `isolated_av`；
-重拍动作是否改变了 chunk 结束状态；
-后续 chunk 是否仍是旧版本；
-`video_continuation` 是否与基线相同。

如修改会影响后续状态，改用 `continuous_av`。

---

## 28. Checkpoint 创建失败

典型原因：

-没有可读 replay cache；
-replay cache 未 complete；
-某个 chunk 缺少 tensor；
-最终状态缺少 output/denoised 音视频；
-磁盘空间不足。

必须从完整、当前格式的 replay cache 创建 checkpoint。

---

## 29. Continuation FPS 不匹配

典型错误：

```text
Continuation checkpoint FPS does not match the new segment
```

新续写 Sampler 的 `fps` 必须与 checkpoint manifest 完全相同。不要用改变 FPS 的方式调整续写速度。

---

## 30. 续写 prompt 为空

Continuation Plan 和 Sampler 都会拒绝空 prompt。填写新片段完整 H3 prompt，并确保连接到新 conditioning 链。

---

## 31. Reference Set 超过上限

`inherit_plus_replace` 会追加两组参考媒体。合并后如超过 H3 限制，应删减参考，而不是依赖运行时截断。

检查：

```text
图片 ≤ 9
视频 ≤ 3
独立音频 ≤ 3
视频音轨与视频同索引
```

---

## 32. OOM 或显存未释放

记录失败阶段：

-加载 H3；
-第一块 conditioning；
-第一块采样；
-Director 观察；
-第二块采样；
-重拍 revision 写盘；
-续写第一块。

尝试：

-减少 `chunk_frames`；
-降低 `video_continuation_res`；
-关闭不必要的其他 ComfyUI 工作流和模型；
-确认 Qwen/Gemma disposable worker 已退出；
-先用短 latent 验证逻辑；
-不要同时连接 retake 和 continuation plan。

不要为了绕过 OOM 随意改变基线 fingerprint 后继续使用旧重拍计划。

---

## 33. 磁盘空间管理

重拍 revision 和 continuation checkpoint 都会保存 tensor，可能占用大量空间。

常见位置：

```text
系统临时目录/comfyui-hr-endless-sampler/last_run_replay/
ComfyUI/output/hr_endless_sampler/continuations/
```

当前 continuation 没有专用删除 UI。删除 checkpoint 时应整目录删除对应 `<checkpoint-id>`，并注意 `index.json` 可能仍保留旧条目；不熟悉格式时先备份，不要在运行中手工修改。

---

# 第四部分：验收记录模板

## 34. 重拍测试记录

```text
日期：
GPU / VRAM：
ComfyUI 版本/提交：
插件提交：
H3 模型：
分辨率：
FPS：
总帧数：
chunk_frames：
video_continuation：
chunks 数量：
重拍模式：
选择 chunks：
prompt override：
基线生成：成功 / 失败
重拍执行：成功 / 失败
revision 切换：成功 / 失败
无采样 Assemble：成功 / 失败
音频保持/变化符合模式：是 / 否
边界连续：是 / 否
峰值 VRAM：
错误与日志：
```

## 35. 续写测试记录

```text
日期：
GPU / VRAM：
ComfyUI 版本/提交：
插件提交：
原片 FPS：
原片帧数：
原片 chunks：
checkpoint ID：
checkpoint 重启后可用：是 / 否
新片段帧数：
新片段 chunks：
audio_mode：
reference_policy：
续写采样：成功 / 失败
Video1 连续：是 / 否
Audio1 符合策略：是 / 否
Assemble：成功 / 失败
合并总帧数：
音视频同步：是 / 否
峰值 VRAM：
错误与日志：
```

---

## 36. 当前验收边界

截至 2026-09-10：

-重拍代码闭环已实现；
-续写代码闭环已实现；
-单元和模拟测试覆盖了部分缓存、计划、revision、checkpoint 和拼接逻辑；
-真实 ComfyUI、真实 H3、真实 GPU 的完整重拍与续写流程尚未验收。

第一次真实测试请保留：

-完整 ComfyUI 控制台日志；
-工作流 JSON；
-节点截图；
-基线与重拍/续写输出；
-`chunk_prompts`；
-Timeline sidecar；
-GPU 型号和峰值显存；
-失败发生的准确阶段。
