# ComfyUI-HR-Endless-Sampler 
## (the older ComfyUI MiniMax H3 Sampler Unlimited)

`HR Endless Sampler` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long video/audio latents, currently supports 
Minimax H3 only. The plan is to add support to LTX 2.5 in the near future.

`HR Endless Sampler` is able to render videos of any length by automatically 
splitting the inference into small chunks of the same long latent. It uses 
Gemma4 12B QAT internally to analyze the original prompt and all references, 
plan the action-timing for each shot and each chunk, then analyzes previous 
rendered frames and writes new small prompts for each chunk, maintaining 
the continuity and coherence of the entire video. 

Using `HR Endless Sampler Preview` node (based on the amazing KJ Live preview node) 
allows to visualize the whole video as it is infered, with a timeslider that displays
the video shots and each chunk. You can even visualize each chunk prompt Gemma created
by holding the mouse pointer over a chunk bar. 

## Quick HELP as I don't have a workflow template yet!
The way to use is pretty straight forward - just replace the normal "Sampler" node by this one, and add the preview node behind it so it can show the preview as the inference happens. You just have to add the extra inputs:
- clip - just connect the model clip
- vae - just connect the video vae model
- images - connect the images you used with minimax guiding - for ref2va, those would be the reference images
- prompt - connect the text of the full prompt you are using with minimax
- fps - should always be 24, but if you have a node that sets the fps, you can connect it here too.
<img width="349" height="422" alt="Image" src="https://github.com/user-attachments/assets/ebb106f4-804b-4465-8ffd-6a26a94ef6a2" />

The `chunk_frames parameter` is the number of frames you want each chunk to have. You probably want to use the max you can that fits in your vram. If you get OOM, just reduce it. With 16GB of VRAM, I have to use 39 frames to render full 1080p videos.

The `video_continuation` is the number of frames you want to use for minimax to "see" from the previous chunk. 5 is the minimum, but from my tests 22 seems to be the one that works best. The more the better, but this will add to your VRAM. In my 16GB VRAM setup, for 1080p render I have to use `video_continuation`=22 with `chunk_frames`=39.

`debug` just shows a bunch of memory and gemma prompt debugging in the console.

`debug_stop_chunk` allows you to stop a render at a certain chunk. It's good if you want to quickly render just the start of the video for testing, without changing anything in the workflow. 

That's pretty much it. 


## Version 0.9.0

Version 0.9 introduces the serial-continuation architecture. It keeps the
normal ComfyUI sampler workflow while sampling one deterministic video/audio
latent as model-aligned chunks, then returns one correctly assembled result.
It includes:

- bounded synchronized Video1/Audio1 continuation in the current H3 backend,
  with additional continuation experiments retained for development;
- a one-time Gemma 4 preproduction action schedule for every complete source
  shot, followed by visual chunk directing using that schedule, the full source
  prompt, prior rendered stills, prior prompt state, and a local-marker
  correction pass so H3 receives chunk-local—not full-video—cut times;
- a self-managed local Gemma 4 12B QAT GGUF runtime with high-detail vision,
  deterministic capture/replay fixtures, persistent last-run prompts/images,
  and process-isolated GPU cleanup before H3 resumes;
- chunk and sampling progress, debug prompt/VRAM diagnostics, a final H3/Qwen/
  VAE/Gemma timing and RAM/VRAM report, and an optional stop-after-chunk
  diagnostic control; and
- `HR Endless Sampler Preview`: a browser-refresh-safe, ordered accumulated
  preview with Tiny-VAE or Latent2RGB decoding, chunk colors, shot brackets,
  frame/shot/chunk labels, timeline transport, keyboard frame stepping, and
  interactive sampling graphs.

The architecture is intended for video models with a continuable temporal
latent. This release implements the MiniMax H3 backend; LTX support is planned.
The H3-specific grid, joint-video/audio layout, and continuation rules below
are backend details, not limits on the sampler's intended scope.

### Why this exists?
1. Replace one sampler node to render longer videos without loop workflows or manual clip concatenation.
2. Use smaller temporal chunks to trade unused temporal VRAM for higher resolution, where the active backend supports it.

### How is this possible?
1. The sampler samples at most `chunk_frames` at once.
2. Its current H3 backend carries a bounded completed tail through the native Video1/Audio1 continuation mechanism, while Gemma maintains prompt and action continuity.
3. The assembled latent is decoded normally after the final chunk, producing one continuous result.

## Use

1. Replace `SamplerCustomAdvanced` with `HR Endless Sampler`.
2. Reconnect the same noise, guider, sampler, sigmas, and latent inputs.
3. For the current H3 backend, connect the same H3 `clip` used by the original conditioning node and paste/connect its original `prompt`.
4. Set `fps` to the prompt timeline's frame rate. H3 normally uses 24.
5. For I2VA/FL2VA, connect the original first frame followed by the optional
   last frame as one image batch. For image-only Ref2VA, connect every reference
   image in its original order. T2VA and audio-only Ref2VA need no images.
6. Set `chunk_frames` to the largest chunk that fits in VRAM. `124` is the
   conservative default; the value is snapped down to H3's `17k + 5` grid.
7. Set `video_continuation` to `5, 22, 39, 56, ...`. The current H3 backend
   always uses this Ref2VA path. It decodes exactly that
   bounded previous tail for Qwen, adds it as the next
   synchronized `<Audio N>` + `<Video N>` reference for the DiT, and adds the
   documented `[video continuation]` sections to later chunk prompts. Connect
   the H3 video `vae`. The DiT AV reference stays latent; only the video is
   decoded for Qwen and limited to a 512x512 pixel-area budget. ComfyUI's H3
   tokenizer represents audio references as labels rather than waveform input,
   so no audio VAE or temporary audio decode is needed.

   Video1 continuation also supplies the previous chunk's exact final five-frame latent tail as one native video
   keyframe clip. It anchors that clip across local frames 0-4, the mandatory
   discarded packing prefix. This adds two video-latent conditioning time steps
   but no target frames, physical-overlap allocation, VAE re-encode, or separate
   audio keyframe; synchronized audio remains available through Video1.

8. Before Chunk 1, Gemma 4 performs one text-only preproduction pass over the
    complete unsplit source shots and the exact retained output ownership of all
    physical chunks. It writes a compact source-relative schedule for every
    shot: contiguous serial `visual_beats` covering the complete duration plus
    optional overlapping dialogue, sound, and sustained-action `overlays`.
    Gemma defaults to concurrency within a shot; it uses serial visual beats
    only for an explicitly ordered, causal, or state-dependent progression.
    This gives Gemma the whole action and every future chunk boundary at once,
    rather than asking it to invent timing independently at every handoff. The
    same pass creates a stable table of only the character names
    explicitly mapped to existing `<Subject N>` labels in the original prompt.
    The sampler validates the shot identities, table shape, and interval coverage;
    an invalid plan receives one complete Gemma correction request, then stops
    before sampling if it remains unusable. It never invents a timing fallback.

    Gemma then directs the complete local `detailed_description` for every
    sampled chunk, including Chunk 1 and chunks containing two source shots. It
    receives a front-loaded list of the relevant visual beats and concurrent
    overlays that actually intersect its retained frames, followed by the complete relevant
    schedule, unchanged full user prompt,
    complete unsplit source bodies for every relevant shot, exact
    global/current/previous frame ranges, only the required local real-cut
    `[Shot N] At MM:SS.mmm,` markers, the continuation conditioning actually
    available to H3, and chronological 2 FPS stills plus the exact final frame
    from the previous sampled chunk. A physical chunk that begins in the middle
    of a source shot starts with plain continuation prose—never a synthetic
    `[Shot 1]` cue. The real rendered stills stay authoritative if H3
    has drifted from the plan: Gemma should continue the next unfinished
    immediate beat rather than replay completed action or compress every later
    beat. The immutable character table is supplied to every chunk: whenever
    Gemma writes a listed name into H3 descriptive prose, it writes `Name
    (<Subject N>)` beside it. It never changes dialogue to add those labels.
    There is no sentence splitter or immutable action ledger. Connect the
    H3 video `vae` for a multi-chunk render, and install this node's
    requirements:

    ```bash
    ~/comfyui/tools/python.sh -m pip install -r /NVME/comfyui/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Sampler-Unlimited/requirements.txt
    ```

    On first eligible handoff, it downloads Google's official 6.98 GB GGUF plus
    its 175 MB projector directly to `models/llama_cpp/gemma-4-12b-it-qat-q4_0`.
    Gemma always uses every possible GPU layer while H3 and Qwen are unloaded,
    inside a disposable child process. The worker exits before the next H3
    inference, which also releases llama.cpp CUDA allocations that PyTorch's
    cache manager cannot own or flush. The current observation is visual only:
    it does not decode or judge the generated soundtrack.

    Observation images are placed in chronological order before the user text,
    as required by Gemma 4's recommended modality order. A project-local MTMD
    handler enables Gemma 4's dynamic visual-token range from 70 through the
    official 1120-token maximum; `n_batch` and `n_ubatch` are both 1120 so one
    maximum-detail non-causal image block fits intact. This preserves up to
    2,580,480 source pixels per still instead of llama.cpp's 280-token/default
    645,120-pixel ceiling. Smaller observation frames are not forced to the
    maximum budget.

    The editable Gemma system and chunk-request messages are in
    [`gemma4_prompts.txt`](gemma4_prompts.txt). The sampler rereads that file
    before the preproduction pass and every chunk, so changing and saving it
    affects the next run/chunk without a Python-code edit. Keep its `[SYSTEM]`,
    `[OBSERVATION]`, `[PREPRODUCTION_SYSTEM]`, and `[PREPRODUCTION]` section
    headers and the documented `{{lowercase_placeholders}}`.
    Its instructions distill the official H3 base/full-reference rules for shot
    markers, concrete playback-order description, camera language, speaker IDs,
    exact `<d>` dialogue, audio continuity, and stable reference labels. Gemma
    uses the documented mapped-speaker form `<Subject N> (Sx) says,
    <d>...</d>`—never `Name (<Subject N>) (Sx)`—so the speaker token remains
    visible to H3. It may add a modest non-cut camera movement when it supports
    the original intent, but must introduce every such move with `In a
    continuous movement,` and continue the established view; it may not invent
    a fresh angle, setup, framing, perspective, transition, or cut.
    It returns a factual progress summary, a per-shot `timing_plan`, a Gemma-only
    `end_state`, and the complete H3-facing local shot sequence. For every later
    chunk, Gemma receives all three prior values alongside the chronological
    stills from the same chunk. The rendered stills remain authoritative for
    what H3 actually accomplished. Only `detailed_description` reaches H3;
    Gemma-only metadata is withheld. A legacy `[end state]` paragraph inside a
    returned description is extracted and logged before H3 encoding. The exact
    sampler-calculated local shot markers are also presented in a separate
    copy-only block; full-video/source timecodes are explicitly forbidden as
    H3 markers. If Gemma nevertheless returns a wrong/missing/renumbered
    marker, or tries to defer a current scheduled beat, the sampler gives Gemma
    one correction request containing the literal required tokens and current
    beat IDs, and uses that second, complete Gemma JSON response. Each response
    attests to current-beat coverage with quoted evidence from its own H3
    description; a current dialogue overlay must include its exact `<d>` line.
    Both model responses and the correction request remain in the capture and
    console report. If the correction is still invalid, marker, coverage,
    formatting, and dialogue findings remain warnings: the sampler logs them and still sends
    Gemma's usable final description unchanged to H3. It never substitutes a
    static algorithmic prompt for a usable Gemma response. A response with no
    usable description stops sampling instead.
    With `debug` enabled, the node also creates a temporary directory such as
    `/tmp/hr-endless-sampler-gemma4-...` and logs its path. Every chunk gets a
    `prompt_NNN_chunk_NNN` fixture containing the exact base64-JPEG worker
    request, separate inspectable JPEGs, rendered system/observation prompts,
    and Gemma response or error. Move useful fixtures into `tests/fixtures/`
    and replay them after editing `gemma4_prompts.txt`, without running H3:

    ```bash
    ~/comfyui/tools/python.sh tests/replay_gemma_capture.py tests/fixtures/<fixture-directory>
    ```

    Replay deliberately uses the current editable prompt templates while
    preserving the captured images, exact frame map, complete source intent,
    chunk/shot ranges, conditioning facts, and worker generation settings. This
    makes prompt iterations take one Gemma request
    instead of another complete video render.
9. `chunk_prompts` is populated during normal sampling too. Enable `debug` to
   additionally print those prompts in the ComfyUI console and log VRAM at each
   VAE/Qwen/DiT boundary and immediately before and after every DiT evaluation.
   The report includes physical free memory, PyTorch allocation/cache/peaks,
   known model residency, and visible GPU tensor payloads. A failed evaluation
   emits an abbreviated CUDA allocator summary before re-raising the error.
   Gemma-directed chunks include its factual visual-progress summary,
   confidence, chunk-local description, raw JSON, and any validation warnings
   in the same output.
   Independently of `debug`, every normal sampler run replaces
   `${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_chunk_prompts.txt` and
   flushes a readable Gemma-to-H3 transcript before each chunk samples. The
   preproduction system/request/JSON/action schedule is recorded before Chunk 1.
   Each 200-character-separated chunk then records the exact user request sent
    beside its observation images, every raw Gemma JSON response and any
    local-marker correction request, validation warnings, and the exact finalized
    structured prompt subsequently encoded for H3.
   The sibling `last_gemma_images/` directory is also deleted and recreated at
   run start. It receives the exact JPEG payload of every chronological still
   passed to Gemma, named by target chunk and exact global source frame.
   An interrupted or failed run therefore retains every prompt reached so far.
   Every sampling run, even with `debug` disabled, ends with a structured timing
   and memory baseline: configuration, rendered range, H3 denoising, Qwen
   encoding, each VAE purpose, Gemma 4, per-chunk timing, peak
   ComfyUI-process/system RAM, average/peak whole-device VRAM, and PyTorch's
   allocator high-water mark. `Peak Time`, printed immediately after `Peak`, is
   the estimated wall time for which device VRAM was closer to the run's peak
   than its average (above the midpoint between those two values).
10. For a short diagnostic render, set `debug_stop_chunk` to a 1-based chunk
   number. `0` is the normal setting and samples the complete video.
11. Decode the returned AV latent normally.

Existing H3 first/last-frame guides are assigned to the chunk containing their
frame. Ref2VA references remain attached to every chunk.

After each chunk prompt is encoded, the node explicitly unloads Qwen and the
connected H3 video VAE before starting the DiT. ComfyUI's dynamic loader does
not include MiniMax reference rows in its normal sampling-memory estimate and
can otherwise leave both conditioning models resident at a chunk boundary.
This cleanup trades model reload time for maximum sampling headroom.

## Accumulated live preview

Add `HR Endless Sampler Preview` between the model loader/model patches and
the guider used by `HR Endless Sampler`. Its widget plays the
chunks generated so far as one growing preview. Each sampler callback replaces
one complete, ordered frame group for the active chunk. A single browser
playhead finishes every frame in chunk 1 before entering chunk 2, and so on;
live updates cannot splice frames from one chunk into another.

`tiny_vae: none` uses H3's fast Latent2RGB approximation. Select
`taeh3.safetensors` from `models/vae_approx` for a more representative RGB
preview. Tiny-VAE decoding is slower and uses additional VRAM, so increase
`frame_stride` if preview overhead is too high. `fps` controls playback timing
and should initially match the sampler's prompt FPS. Its arrow controls change
by one FPS; changing it during an active job immediately reschedules browser
playback at the new rate without affecting sampling. `max_resolution: 0` preserves the
preview decoder's native resolution. The widget shows the active `Chunk N/N`,
preview resolution, FPS, seconds per step, elapsed time, ETA,
sigma/latent-change history, and step-time history. The configured continuation
frames are shown only once because the preview removes the overlap from later
chunks.

The backend retains the latest frame group for every chunk of the current
execution. After a browser refresh, the widget immediately restores that
snapshot through a local ComfyUI endpoint and then continues receiving live
updates. Hover either graph to draw a vertical sampling-step marker and inspect
any retained preview step from the active chunk; moving off the graph resumes
the accumulated transport at its current playhead. New denoising previews only replace
their chunk atomically; they never alter the currently playing frame group.

A five-pixel transport line below the image uses a different color for every
chunk. Click or drag it to pause and seek, use the small play/pause button (or
Space), and focus the preview then press Left/Right to step through individual
available preview images. Horizontal brackets immediately below the line show
each original prompt shot and its inclusive frame range, for example `S3
118–229`; hovering a bracket shows the full range when its label is narrow. The
outlined yellow label in the lower-right shows `S#/C#/frame`, identifying the
shot, chunk, and zero-based output-frame index. It falls back to the bare frame
number when range metadata is unavailable. H3 temporally compresses several output frames into most
latent preview images, so frame stepping and the label usually advance by four
frames even when `frame_stride` is one.

The console also displays a chunk progress line above the stock sampler's step
progress. A later chunk starts only after the previous chunk has completed; the
current H3 backend supplies its completed continuation tail through Video1 and
the five-frame boundary keyframe clip.

## Shot timing

The prompt parser follows MiniMax's documented format:

```text
[Shot 1] Opening shot with no timestamp.
[Shot 2] At 00:04.167, the camera cuts to...
```

The first shot is untimed. Every later shot must be sequential and use a
strictly increasing `MM:SS.mmm` cut time. The sampler converts those values to
integer global frames using `fps`.

Each physical chunk keeps only source shots that intersect its sampled frame
window, renumbers them locally, and uses the canonical H3 grammar. A genuine
cut remains `[Shot N] At MM:SS.mmm, ...`, where the time is relative to the
physical chunk start, including any carried guide prefix. If that prefix ends
at a source cut, the previous local shot is a compact preservation block—not a
repeat of its completed action—so H3 still receives the real upcoming cut.

The former custom master-range/timeslice storyboard grammar and synthetic
`Shot ends` events were removed. They calculated the intended frame correctly
but were not MiniMax's documented shot syntax and could make H3 move a cut
early or late. When a chunk begins inside a long source shot, automatic Gemma
now supplies short semantic continuation prose after observing the previous
generated chunk. Later chunks are still presented as continuations rather than
fresh I2VA/FL2VA requests: original picture anchors are removed from Qwen while
Ref2VA identity/style references remain available.

The parser recognizes both `integrated_multimodal_description:` and
`detailed_description:`. Other MiniMax prompt sections are preserved.

## Limits

- This reduces temporal sampling memory. It does not make a single frame of an
  arbitrary resolution fit in VRAM; spatial tiling would change H3's global
  attention and is outside this node.
- Video Ref2VA sources cannot be reconstructed from an image input. Image and
  audio Ref2VA presentations are supported; a conditioning containing a video
  reference fails clearly instead of silently dropping its Qwen video tokens.
- A multi-chunk render needs the H3 video `vae`. Gemma directs Chunk 1 from
  text, then observes the last sampled chunk and writes the next short H3
  description; it does not ask H3 to infer its location in a repeated
  full-shot prompt.
- The current H3 Video1 continuation path requires image/audio Ref2VA
  conditioning that this node can reconstruct plus the H3 video VAE. It is
  ignored for the first chunk because no generated history exists yet.
- Native `video_continuation` adds the bounded synchronized video/audio tail to
  DiT attention and the previous final five-frame video tail as a visual
  keyframe across the discarded packing prefix, without adding a separate
  audio keyframe.
- Earlier keyframe-overlap, latent warm-start, Qwen-history, and prompt-only
  branches remain in source for development experiments but are deliberately
  hidden from the released node UI while Video1 continuation is evaluated.
- Gemma chunk directing needs `llama-cpp-python==0.3.35`, built with the
  CUDA wheel in `requirements.txt`. It loads a separate 7 GB model only between
  relevant chunks in a disposable worker process, then exits that worker before
  H3 sampling. This gives stronger VRAM isolation at the cost of Python startup
  time per handoff. Dependency/download failures stop clearly; a malformed
  observation falls back to the canonical source-shot prompt for that chunk.
  Gemma receives complete unsplit source-shot prose and is free to rephrase it
  while preserving intent, exact dialogue, explicit sound effects, and real
  cuts. Audio observation and shot-aware physical chunk boundaries remain
  future work.
- Chunked denoise masks are not supported.

Prompt format references: [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md),
[MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md),
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
