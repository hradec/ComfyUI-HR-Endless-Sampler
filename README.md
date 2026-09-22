# ComfyUI-MiniMax-H3-Sampler-Unlimited (mickeylan fork)
## HR Endless Sampler 中文增强与低显存长视频工具集

> ⚠️ **注意**：这是 [hradec/ComfyUI-HR-Endless-Sampler](https://github.com/hradec/ComfyUI-HR-Endless-Sampler) 的中文用户/低显存优化分支。

本项目以 **HR Endless Sampler 节点族**为核心：在保留 MiniMax H3 低显存 physical chunk 连续采样的基础上，加入多导演、统一参考媒体、实时预览、Timeline、Save/Load、断点重跑、分块重拍和持久续写。

https://github.com/user-attachments/assets/5da194ea-4d29-4fd3-9b1c-edd537b88431

- video generated with HR Endless Sampler at 1080p 625 frames on a 16GB GPU

## 🎯 本分支特色

本分支专为 **中文用户** 和 **低显存（12GB）用户** 设计：

| 特性 | 原版 | 本分支 |
|------|------|--------|
| Gemma 4 导演 | ✅ 支持 | ✅ 支持 |
| Qwen3.5 导演 | ❌ 不支持 | ✅ 支持 |
| Qwen3.6 导演 | ❌ 不支持 | ✅ 支持 |
| Qwen3.8 导演 | ❌ 不支持 | ✅ 支持 |
| 12GB VRAM 支持 | ❌ Gemma 12B 太大 | ✅ Qwen 27B MoE + UD-IQ2-mtp |
| 中文提示词 | ⚠️ 需要翻译 | ✅ 原生支持 |
| MoE CPU Offload | ❌ 不支持 | ✅ 支持 |
| 分块重拍与 Revision | ❌ 不支持 | ✅ 已实现，待实机验收 |
| 持久续写 Checkpoint | ❌ 不支持 | ✅ 已实现，待实机验收 |
| 统一参考媒体输入 | ❌ 分散接线 | ✅ 图片/视频/音轨/独立音频 |
| Replay/断点重跑 | ⚠️ 基础能力 | ✅ 最近 5 次完整生成可选，缓存、重拍和续写共用 |
| 生成中控制 | ❌ 不支持 | ✅ 停止保留、停止删除、删除后从头重排 |

### 为什么选择 Qwen3.6/3.8？

- **Qwen3.6/3.8 是 27B MoE 模型**，可使用 UD-IQ2-mtp 量化降低显存占用
- MoE 架构只激活部分参数，适合显存受限环境
- 本分支用户实测 Qwen3.6/3.8 可在 12GB VRAM 上运行；稳定参数仍取决于 GGUF、CUDA、参考媒体、分辨率和 chunk 大小
- Gemma 4 12B 在该 12GB 测试环境中不可用，因此保留为旧工作流默认后端，不作为 12GB 推荐方案
- Qwen3.6/3.8 支持内置 MTP 推测解码和 MoE offload

### 12GB VRAM 推荐配置

```text
director_backend = qwen3.8           # 27B MoE + UD-IQ2-mtp
director_mtp = true                  # 内置 MTP
director_reasoning_effort = medium   # 平衡质量与速度
chunk_frames = 56-62                 # 1080p 推荐值
video_continuation = 22
pytorch_memory_fraction = 0.82
```

---

`HR Endless Sampler` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long video/audio latents, currently supports 
Minimax H3 only. The plan is to add support to LTX 2.5 in the near future.

`HR Endless Sampler` is able to render videos of any length by automatically 
splitting the inference into small chunks of the same long latent. It uses 
a user-selected local Gemma 4 or Qwen3.5 multimodal director to analyze the
original prompt and references, plan action timing for every shot and chunk,
then inspect previous rendered frames and write the next H3 prompt while
maintaining continuity and coherence. Gemma 4 remains the default for old
workflows.

Using `HR Endless Sampler Preview` node (based on the amazing KJ Live preview node) 
allows to visualize the whole video as it is infered, with a timeslider that displays
the video shots and each chunk. You can even visualize each chunk prompt Gemma created
by holding the mouse pointer over a chunk bar. 

<img width="350"  alt="image" src="https://github.com/user-attachments/assets/ac2c7bf3-cc07-45d9-b78b-760c4580338e" />
<img width="350"  alt="image" src="https://github.com/user-attachments/assets/6308733b-101a-43b2-869e-ebfc97605e5e" />

The `HR Endless Sampler Save Video` and `HR Endless Sampler Load Video` also display the timeslider with all the features of the preview node. They also have an extra button "Macthing Videos" that display a list of the last videos with the same filename prefix, so we can quickly compare previous renders with newer ones, also seeing the chunks, prompts, time to render, etc.:

<img width="350" alt="image" src="https://github.com/user-attachments/assets/9e1e1312-0493-4a10-a750-1e92b94451a7" />
<img width="350" alt="image" src="https://github.com/user-attachments/assets/d0c37c83-06e0-4582-b240-ae8199194d2d" />
<img width="350" alt="image" src="https://github.com/user-attachments/assets/d0a39920-647b-4891-90cb-b172c5e73c16" />
<img width="600" alt="image" src="https://github.com/user-attachments/assets/54e1f553-b7e5-470e-ade8-aafe333ce075" />


## Quick HELP as I don't have a workflow template yet!
The way to use is pretty straight forward - just replace the normal "Sampler" node by this one, and add the preview node behind it so it can show the preview as the inference happens. You just have to add the extra inputs:
- `clip` - just connect the model clip
- `vae` - just connect the video vae model
- `images` - connect the images you used with minimax guiding - for ref2va, those would be the reference images
- `prompt` - connect the text of the full prompt you are using with minimax
- `fps` - should always be 24, but if you have a node that sets the fps, you can connect it here too.
<img width="349" height="422" alt="Image" src="https://github.com/user-attachments/assets/ebb106f4-804b-4465-8ffd-6a26a94ef6a2" />

## Included nodes

### HR Endless Sampler 主节点族

| Node | Purpose |
| --- | --- |
| `HR Endless Sampler` | 串行采样长音视频 latent，生成 chunk prompts、成品 latent 和 Timeline，并接收重拍或续写计划。 |
| `HR Endless Sampler Preview` | 实时累计预览、chunk 播放、Shot 标记、提示词/耗时悬停、逐帧控制、性能图表和刷新恢复。 |
| `HR Endless Sampler Save Video` | 保存普通视频、VHS 格式或 float EXR 序列，同时保留 Timeline、提示词、渲染耗时和可选音频。 |
| `HR Endless Sampler Load Video` | 浏览或上传成品媒体，恢复交互式 Timeline，并输出 VIDEO/IMAGE/AUDIO、尺寸、FPS、帧数和文件名。 |
| `HR MiniMax H3 Continuation Analyzer` | 仅使用本地 Qwen3.5 分析普通视频最后 22 帧，输出 H3 prompt、尾帧上下文、分析 JSON 和预览；可选参考图只提供给 Qwen，不会成为 H3 reference。 |
| `HR MiniMax H3 Continuation Apply` | 接收现有 MiniMax H3 conditioning 与 latent，使用 Analyzer 上下文编码并合并同位置 keyframe；原有 `cross_attn`、token tags 和 refs 原样保留，不重新 tokenization。输出可接 SelfLift，也保留 `external_continuation` 给 HR Endless Sampler。 |
| `HR Video Bridge Extract` | 提取视频 A 最后 22 帧、视频 B 最前 22 帧及对应音频，建立桥接源数据。 |
| `HR Video Bridge Director` | 使用隔离的 Qwen3.6/3.8 分析人物参考图和 A/B 边界，输出结构化过渡计划及 H3 提示词。 |
| `HR Video Bridge Conditioning` | 将人物图和 B 头作为 references、A 尾作为首端 continuation，输出可接 HR Endless Sampler 的 conditioning、latent 和 external continuation。 |
| `HR Video Bridge Assemble` | 自动搜索 A/Bridge/B 的低差异接缝，组装帧、音频、Timeline 和接缝报告。 |
| `HR Endless Segment Retake Director` | 浏览最近一次完整 replay cache，选择 physical chunks、编辑 H3 prompt 并生成重拍计划。 |

接线：`Analyzer.H3 prompt → 现有 MiniMax H3 conditioning`；`conditioning.positive + latent → Apply`；`Analyzer.context → Apply`；`Apply.positive + latent → SelfLift`。
| `HR Endless Retake Assemble` | 根据每个 chunk 当前选中的原版/重拍 revision，无采样重新拼接 output、denoised output 和 Timeline。 |
| `HR Endless Continuation Checkpoint` | 将最近一次完整 replay 固化为可跨重启保存的续写 checkpoint。 |
| `HR Endless Continuation Plan` | 设置新提示词、音频策略以及参考媒体继承/替换/合并策略。 |
| `HR Endless Continuation Assemble` | 将 checkpoint 中的旧音视频 latent 与本次续写结果拼接。 |

### Sampler 辅助节点

| Node | Purpose |
| --- | --- |
| `HR Qwen Director Config` | 为 Sampler 和提示词编译器共享本地 Qwen3.5/3.6/3.8 model/mmproj/runtime 配置。 |
| `HR H3 Prompt Skill Compiler` | 将普通故事编译成带原子事件所有权、Shot 起止状态、禁止重复项和差异化机位的 H3 六字段提示词；自动使用连接的 Director Config 后端。 |
| `HR MiniMax H3 Reference Set` | 统一输入最多 9 张图片、3 个视频及对应音轨、3 条独立音频。 |
| `HR MiniMax H3 Reference Conditioning` | 创建 MiniMax H3 Ref2VA conditioning 和 nested AV latent。 |

提示词编译接线：人物图片 → `HR MiniMax H3 Reference Set`；同一个 `HR Qwen Director Config` 同时连接 `HR H3 Prompt Skill Compiler` 和 `HR Endless Sampler`；Compiler 的 `H3 prompt` 接 Reference Conditioning 与 Sampler 的 `prompt`。用户在 Director Config 选择 Qwen3.5 时两者都使用 3.5，选择 Qwen3.6/3.8 时同理。

视频桥接接线：两个 `HR Endless Sampler Load Video.images/audio/fps` → `HR Video Bridge Extract`；人物图片 → `HR MiniMax H3 Reference Set`；Extract + Reference Set + `HR Qwen Director Config` → Director → Conditioning；Conditioning 的 `positive` 接 Guider，`latent`、`external_continuation`、`bridge_reference_set`、`prompt` 接 `HR Endless Sampler`；Sampler latent 经 VAE Decode 后与 Extract 输出一起接 `HR Video Bridge Assemble`。详细说明见 [`视频自然过渡节点实施计划.md`](视频自然过渡节点实施计划.md)。

The Save and Load players use the same colored chunk timeline and shot brackets
as the live Preview node, but omit the live sampling graphs. Hovering a chunk
shows its H3 prompt and the sampler/Gemma/miscellaneous timing breakdown.

## Chunk retake（分块重拍）

先让 `HR Endless Sampler` 完整生成一次基线，随后在 `HR Endless Segment Retake Director` 中刷新最近一次 replay cache、选择 chunks、修改提示词并选择模式：

- `video_only`：只重拍画面，保留缓存中的原音频；
- `isolated_av`：只重拍选中 chunks 的画面和音频；
- `continuous_av`：从最早选中的 chunk 连续重拍到结尾，后续块继承新的前块状态。

将 Director 的 `retake plan` 接入原 `HR Endless Sampler.retake_plan` 后重新运行。每次成功重拍都会创建 revision，不覆盖原版。在 Director 中选择各 chunk 的活动 revision，再运行 `HR Endless Retake Assemble` 即可无采样重新拼接。

> 重拍代码闭环已实现，但尚未完成真实 ComfyUI + H3 + GPU 的完整验收。首次测试请保留基线视频、日志、Timeline 和 replay cache。

详细接线和验收步骤见 [`HR重拍与续写操作手册.md`](HR重拍与续写操作手册.md)。

## Durable continuation（持久续写）

续写建议分三次 Queue：

1. 用 `HR Endless Sampler` 完整生成原片段；
2. 单独运行 `HR Endless Continuation Checkpoint`，把完整 replay 固化到 `output/hr_endless_sampler/continuations/`；
3. 用 `HR Endless Continuation Plan` 设置新提示词、音频和参考策略，将其接入新的 `HR Endless Sampler.continuation_plan`，最后用 `HR Endless Continuation Assemble` 拼接旧结果与新结果。

音频策略：

- `continue`：继承旧片末尾 Audio1；
- `new_segment`：保持视频连续，但不延续旧音频内容；
- `mute`：将新增片段输出音频静音。

参考媒体策略：

- `inherit`：使用 checkpoint 保存的 Reference Set；
- `replace`：只使用新连接的 Reference Set；
- `inherit_plus_replace`：在原参考媒体后追加新参考媒体。

续写 Sampler 的 `latent_image` 只表示新增片段长度，FPS 必须与 checkpoint 相同；`retake_plan` 和 `continuation_plan` 不能同时连接。

> 续写代码闭环已实现，但尚未完成真实 ComfyUI + H3 + GPU 的完整验收。

详细接线、策略说明和测试模板见 [`HR重拍与续写操作手册.md`](HR重拍与续写操作手册.md)。

## Main settings

`chunk_frames` is the number of frames sampled in one H3 call. Use the largest
value that fits in VRAM. Smaller chunks use less VRAM, but need more handoffs.
For example, 39 frames is a practical 1080p starting point on a 16 GB GPU.
H3 uses a `5 + 17k` frame grid, so the effective size is aligned to that grid.

Qwen3.6/3.8 chunk directing also maintains an automatic event ownership ledger. Completed actions are persisted in replay state and injected into later H3 prompts as forbidden replays; active and pending events remain separately tracked. This is intended to reduce the chance that a later physical chunk restages an action or shot that already appeared. Qwen3.5 remains unchanged, and this mitigation still requires real multi-chunk GPU validation.

`video_continuation` is the number of completed frames carried from the last
chunk into the next one. H3 sees them as a synchronized `<Video N>` and
`<Audio N>` reference. `22` frames is a good default for continuity. `5` is
the minimum. Larger values use more VRAM. If it is larger than the current
chunk, the sampler caps it to the chunk size.

`video_continuation_res` controls only the spatial size of the clean Video1
latent passed to H3. `full` reuses the generated latent exactly and performs no
extra encode. A smaller preset decodes the completed Video1 tail once, resizes
all of its frames to the selected 32-pixel-aligned canvas, and VAE-encodes that
sequence as a smaller `minimax_refs` block. This reduces H3 reference-attention
VRAM and may allow a larger `chunk_frames`, particularly at 1080p. It trades
some fine continuation detail and adds one VAE encode between chunks. The
synchronized Audio1 latent and full-resolution five-frame boundary keyframe
are unchanged. Qwen and Gemma also remain at the normal stock H3
reference-video presentation size; this setting does not reduce what they see.
For every continuation chunk, the console reports the Video1 video/audio and
boundary-keyframe shapes, raw MiB, packed H3 row counts, reduction versus a
full-resolution Video1, and continuation rows relative to the target AV rows.
Packed rows are the useful comparison: the latent itself is small, while every
additional row expands much larger per-layer attention and activation buffers.
The sampler separately reports the complete Qwen cross-attention tensor retained
for H3. That value includes the prompt, original references, and Video1 semantic
presentation and does not change when only `video_continuation_res` changes.

The sampler also uses the previous chunk's final five frames as a small H3
boundary keyframe. This is automatic. It helps adjacent chunks meet cleanly.

`director_backend` explicitly selects `gemma4`, `qwen3.5`, `qwen3.6`, or
`qwen3.8`. The existing `qwen3.5` value remains compatible with old workflows.
`director_model` and `director_mmproj` select local files discovered recursively
beneath `models/llama_cpp` and `models/LLM/GGUF`. For each Qwen backend, `auto`
selects only a same-directory model/projector pair from that exact Qwen series.
Explicit Qwen selections must match the selected series and directory. URLs, external paths, and non-GGUF files
are rejected. Qwen never downloads a model.

`cache_gemma_preproduction` saves Gemma's static preproduction context in
system RAM. This can make later Gemma requests much faster because they do not
need the full source prompt and shot plan again. Linux uses `/dev/shm` when it
has enough free RAM; otherwise the normal temporary directory is used. The
cache uses several GiB of RAM, never VRAM. It is optional and does not change
the generated video.

`gemma4_mtp` enables native MTP where the selected director supports it.
Gemma uses its four-token configuration. Qwen3.5 does not support MTP.
Qwen3.6 and Qwen3.8 use embedded NextN/MTP layers for both text timing and visual
MTMD requests. `director_mtp_draft_tokens` controls their draft length. A native Qwen
MTP failure is retried once in a fresh non-MTP worker; invalid model-authored
JSON is not repeatedly regenerated. Turn the setting off to compare ordinary
decoding on the same workflow.
The console reports generated tokens/second and, in MTP mode, the assistant's
draft-token acceptance rate, proposal count, verification work, rollback
replays, and checkpoint time. This is real speculative decoding: the matching
Gemma assistant proposes as many as four tokens and the 12B target verifies
them together.

`pytorch_memory_fraction` sets a process-wide ceiling for PyTorch's CUDA
allocator when the sampler starts. The default `0.85` leaves 15% of physical
VRAM outside PyTorch's cache so allocator pressure happens before a large H3
temporary consumes the final driver pages. This is especially useful with
ComfyUI's `cudaMallocAsync` backend, where `garbage_collection_threshold` is
ignored. The setting remains active until another sampler run changes it; use
`1.0` for PyTorch's normal unrestricted limit. The console and final run report
show the effective fraction.

`debug` adds detailed prompt and memory information to the console.

`debug_stop_chunk` stops after a selected 1-based chunk. `0` means render the
whole video.

`debug_start_chunk` reruns from a selected 1-based chunk. It is useful for
testing a later shot without sampling all earlier chunks again. The first run
creates a temporary replay cache; later compatible runs reuse its noise,
completed chunks, and continuation boundary. Set it back to `0` to clear that
temporary cache on the next render.

If the main prompt changes during a replay, the sampler keeps the saved physical
frames and noise but asks Gemma to make a new preproduction plan from the new
prompt. It also rebuilds the Gemma KV cache. This lets prompt changes such as
moving dialogue earlier in a shot affect the rerun chunk.

## How the sampler works

The sampler runs chunks in order. A chunk finishes all H3 sampling steps before
the next chunk begins. The completed tail becomes the next chunk's Video1/Audio1
continuation reference.

Before Chunk 1, Gemma reads the complete prompt and plans the timing of every
source shot. It knows every physical chunk boundary before H3 starts. This gives
Gemma a full view of a long action instead of making it guess each chunk in
isolation.

For every chunk, Gemma receives:

- the complete original prompt and the relevant timing plan;
- the frames and shots the chunk must produce;
- the latest generated stills from the previous chunk, sampled at 2 FPS plus
  its exact final frame; and
- the previous chunk's Gemma prompt and end state, when they still match the
  current source prompt.

Gemma writes one short H3 `detailed_description` for that chunk. It keeps exact
dialogue inside `<d>...</d>`, preserves each original global `[Shot N]` label,
and recalculates only cut timecodes on the current chunk's local clock. H3
receives only that final description, not Gemma's JSON notes or planning data.

Generated Video1 frames shown to Qwen and Gemma use the same spatial canvas as
ComfyUI's native H3 reference-video path: a nominal 768-pixel short edge with a
768×1344 pixel-area cap, aligned to 32 pixels and never enlarged. For example,
a 1920×1088 continuation is presented at 1344×768 rather than using a smaller
sampler-only observation format. Temporal presentation remains 2 FPS plus the
exact final frame for Gemma. The separate H3 `minimax_refs` continuation latent
remains at the generated video's full latent resolution.

The sampler saves the latest Gemma transcript after every chunk, even with
`debug` off:

```text
${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_chunk_prompts.txt
${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_images/
```

The text file includes the preproduction plan, each request to the selected
director, its JSON response, any correction request, and the final prompt sent
to H3. The image directory contains the stills that the director saw. The legacy
`last_gemma_*` filenames remain unchanged for workflow/tool compatibility. A new
render replaces both.

## Qwen3.5, Qwen3.6, and Qwen3.8 setup

Place the local model and projector beneath `models/LLM/GGUF`, for example:

```text
models/LLM/GGUF/qwen3.5-9B/
├── Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf
└── mmproj-Huihui-Qwen3.5-9B-abliterated.gguf
```

Select the matching `qwen3.5`, `qwen3.6`, or `qwen3.8` value in
`director_backend`; `auto` then discovers only that series. Qwen uses a disposable llama.cpp worker
with a 256-token batch. Qwen3.5 uses a 65536-token context; Qwen3.6 and Qwen3.8
use 32768 to match Gemma 4's director context. Qwen3.8 reads the GGUF's native
chat template, supports `xhigh`, `medium`, and `low` reasoning effort, and adapts
that template for its mmproj. Qwen3.6 and Qwen3.8 can optionally pass `cpu_moe` or
`n_cpu_moe`. The Gemma preproduction KV cache remains unsupported. The same
directing contract accepts Chinese source prompts, writes H3 visual/action/
camera prose in English, and preserves original dialogue, lyrics, visible text,
required shot markers, and prior chunk continuity context.

The Qwen worker exits before H3 sampling resumes, so its llama.cpp CUDA context
cannot remain allocated beside H3. The plugin does not use Transformers,
Hugging Face fallback loading, or any Qwen network request. Runtime support for
a particular GGUF/mmproj pair still depends on the installed pinned
`llama-cpp-python` build.

## Gemma 4 setup

The sampler uses the official Google Gemma 4 12B QAT Q4 GGUF model through
`llama-cpp-python`. When `gemma4_mtp` is enabled, the matching native MTP
assistant proposes up to four tokens at a time. Install the CUDA 12.5
dependencies with ComfyUI's Python:

```bash
~/comfyui/tools/python.sh -m pip install -r requirements.txt
```

On first use, the sampler downloads Gemma, its projector, and the matching
465 MB Q8 MTP assistant to:

```text
models/llama_cpp/gemma-4-12b-it-qat-q4_0/
```

Gemma runs in a separate process between H3 chunks. H3, Qwen, and the video VAE
are unloaded before Gemma runs, and the Gemma process exits before H3 sampling
resumes. This is intentional: it releases Gemma's CUDA allocations before H3
needs VRAM again. If native MTP is enabled but the platform wheel lacks its
required symbols, the Gemma pass stops with an explicit error; it never
silently labels ordinary decoding as MTP. Disable `gemma4_mtp` to deliberately
use the original decoder.

The fast MTP checkpoint path is currently experimental in upstream llama.cpp.
Gemma therefore runs inside a disposable worker. If native MTP aborts that
worker, the sampler preserves the exact request and retries **only that failed
Gemma operation once** with the original non-MTP decoder. It does not disable
MTP for the rest of the render, and it does not discard the preproduction KV
cache or any other request data: the next Gemma operation attempts MTP again.
Ordinary prompt/schema errors remain visible and are not mistaken for an MTP
crash. The upstream failure is tracked in
[llama.cpp issue #27439](https://github.com/ggml-org/llama.cpp/issues/27439).

The editable Gemma instructions are in
[`gemma4_prompts.txt`](gemma4_prompts.txt). The sampler reads this file again
before preproduction and before each chunk. You can adjust the wording without
editing Python, but keep the named section headers and `{{placeholders}}`.

## Preview node

Place `HR Endless Sampler Preview` in the model path before the guider used by
the sampler. Connect the actual H3 model through the preview node, then use its
model output for the guider and sampler.

The preview plays every completed chunk in order. It can restore the current
preview after a browser refresh. Its timeline uses a different color for each
chunk and shows brackets for source shots. Hover a chunk color to see Gemma's
H3 prompt, H3 render time, Gemma processing time, and total processing time for
that chunk. Only the prompt prose is colored: each shot section uses the same
color as its shot bracket, including prompts containing a single shot.

Use the small play/pause button, Space, or the timeline to control playback.
Focus the preview and use Left/Right for frame stepping. The lower-right label
shows the output frame and, when available, the shot and chunk number.

`tiny_vae: none` uses the fast H3 Latent2RGB preview. Select
`taeh3.safetensors` for a more representative preview. Tiny-VAE preview costs
more time and VRAM. `max_resolution: 0` keeps the latent preview resolution.
The preview FPS can be changed while it is playing and does not affect sampling.

## Save and load finished videos

`HR Endless Sampler` now has a fourth `timeline` output. Connect it, together
with decoded `images`, to `HR Endless Sampler Save Video`. The Save node has
its own lighter version of the preview player: it plays the finished render,
keeps the colored chunk bar and shot brackets, shows each chunk's Gemma prompt
and saved timing details on hover, and colors each prompt's shot sections to
match the shot brackets. It supports play/pause, timeline seeking, and
Left/Right frame stepping. The status line also shows the full sampler render
time saved with the video. It intentionally has no sampler graphs.

`video/h264-mp4` uses ComfyUI's native video encoder and does not require Video
Helper Suite. It supports the Save node's CRF, 8/10-bit pixel-format choice,
audio muxing, and embedded timeline metadata. The other ordinary formats call
the installed **Video Combine 🎥🅥🅗🅢** encoder directly. Their `pixel_format`
uses the corresponding VHS options (`auto` keeps the format's VHS default),
and `crf` is passed through whenever that encoder supports CRF. Other VHS
format-specific settings retain their VHS defaults. Connect decoded `audio`
to mux its soundtrack into ordinary video output. When VHS is installed, the
format menu includes every Video Combine choice, including animated GIF and
WebP, plus native H.264 and the Endless-only EXR option.

`video/exr` writes an OpenEXR image sequence. Choose `half` for 16-bit float
or `float` for 32-bit float, plus `none`, `rle`, `zip1`, or `zip16`
compression. `exr_gamma` exposes the encoder's remaining gamma option; leave
it at `1.0` for raw values. EXR saving never clamps the tensor it receives. For the H3 VAE's
actual decoder values—including values below 0 or above 1—connect the sampler
`output` latent and the H3 `vae` to the optional Save-node `latent` and `vae`
inputs. That path bypasses only H3's final display clamp before writing EXR.
The resulting EXR contains those raw VAE RGB values; it does not silently apply
an sRGB-to-linear color conversion. An EXR sequence cannot contain audio, so a
connected `audio` input is written beside it as a 32-bit float WAV sidecar and is
included in the Save/Load node's browser preview.

Native H.264 and VHS formats that support container metadata write the compact
timeline to the `hr_endless_sampler_timeline` tag. Every export also receives an
adjacent sidecar:

```text
your_render.mp4.hr_endless_sampler_timeline.json
```

The sidecar is always written because transcoding services can remove custom
video metadata. EXR uses the same sidecar for the full sequence manifest and
also embeds the timeline in its first EXR frame. `HR Endless Sampler Load
Video` opens a saved video or the first EXR frame/sidecar in the same player;
it prefers the sidecar and falls back to embedded video metadata. Its `fps` is
only a playback-rate override; `0` uses the rate stored with the render.

The Load node includes **Browse output…** and **Upload video…** controls.
Browse output opens a folder dialog rooted at ComfyUI's `output` directory; it
can navigate subfolders, lists supported video files and standalone EXRs, and
shows each saved Endless EXR sequence as one item instead of hundreds of frame
files. Small Name, Size, and Date buttons change the ordering; Date is the
default, with the newest item first. Upload video copies a file from the browser machine into
`output/hr_endless_sampler_uploads/` using 16 MiB chunks, then fills the node's
path automatically. Choosing or uploading immediately probes the media, reads
its timeline metadata, and fills the player without queueing the workflow.
Queue the node when downstream nodes need its timeline, filename, FPS, native
ComfyUI `VIDEO`, decoded `IMAGE` frame batch, `AUDIO`, frame count, width, or
height outputs. The browser-only preview remains lightweight; the full frame
and audio decode happens when the workflow queues the Load node.

Both Save and Load also have a **Matching videos ▾** dropdown, ordered newest
first. On Save it lists `filename_prefix*` from the matching output folder. On
Load it derives the prefix from the current `video` path by removing the
generated `_<number>_…` suffix—for example,
`video/render_00042_.mp4` searches for `video/render*`. Selecting an entry
switches the player immediately; on Load it also replaces the serialized
`video` value.

## Prompt format

Use MiniMax's normal shot format. The first shot has no timecode. Later shots
use a strictly increasing cut time:

```text
[Shot 1] The tiger runs through the jungle.
[Shot 2] At 00:02.833, the camera cuts inside the temple.
```

Set `fps` to the same frame rate used by those timecodes. H3 normally uses
24 FPS. The sampler converts cut times to frames, keeps each real cut at the
correct position inside its physical chunk, and gives H3 the corresponding
local timecode.

For Ref2VA, keep the reference images in the same order as the original H3
conditioning. The sampler keeps those identity/style references for every
chunk. The generated Video1 continuation reference is added separately.

## Memory and performance

The first chunk can fit while the second chunk fails. Later chunks include the
Video1/Audio1 continuation tail, so they use more VRAM than Chunk 1. Choose a
`chunk_frames` and `video_continuation` pair that fits Chunk 2 as well.

Chunking reduces the temporal part of H3's memory use. It cannot make an
arbitrary resolution fit: one full-resolution H3 sampling step must still fit
in VRAM.

The console shows chunk progress, H3 step progress, and Gemma preparation
progress with live generated tokens/second. The end-of-run report includes H3, Qwen, VAE, and Gemma time,
plus peak RAM and VRAM use.

## Current limits and verification status

- The released sampling backend currently supports MiniMax H3 only; LTX 2.5 is still planned.
- Multi-chunk H3 rendering needs the H3 video VAE.
- Chunked denoise masks are not supported.
- Gemma/Qwen observes generated video frames, not generated audio. It preserves dialogue and sound instructions from the source prompt, but does not judge the resulting soundtrack.
- 重拍和持久续写已经完成代码、缓存协议及模拟测试，但尚未完成真实 ComfyUI + H3 + GPU 的端到端验收。
- Reference Set 的真实视频、同步音轨和独立音频路径仍需实机联调。
- 12GB VRAM 可用性来自本分支用户对特定 Qwen3.6/3.8 UD-IQ2-mtp 配置的实测，不代表所有模型、分辨率和参考媒体组合都能稳定运行。
- 上游 llama.cpp issue #27439 截至 2026-09-10 仍为 open；必须保留 disposable worker 和 operation-local non-MTP fallback。

## TIPS TO RENDER 1080p with 16GB of VRAM:  
 - These tips are from my workflow using ref2va with 5 images at 720p resolution as reference. 
 - To render a 625 frames video at 1080p with only 16GB of VRAM, I use 56 `chunk_frames` and 22 `video_continuation` frames. This configuration may OOM without  `KJNodes MiniMax H3 Low VRAM Attention`. `KJNodes MiniMax H3 Low VRAM Attention` helps reduce VRAM memory peaks which is specially necessary during all chunks after Chunk1, since those chunks have an extra 22 frames to deal with. I set it to `4` and it renders 1080p without problems.
 - If you don't want to use the amazing `KJNodes MiniMax H3 Low VRAM Attention`, you still can render 1080p by reducing `chunk_frames` to 39.
 - off course this all changes depending on how many (and resolution) reference images/videos/audio you are using. 

## References

- [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md)
- [MiniMax H3 base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Google Gemma 4 12B QAT Q4 GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
