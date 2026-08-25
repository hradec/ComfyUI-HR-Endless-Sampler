# SamplerCustomAdvanced-Unlimited development memory

## Goal

This custom node adds `SamplerCustomAdvanced-Unlimited`, a MiniMax H3-specific
replacement for ComfyUI's stock `SamplerCustomAdvanced`. Its purpose is to
sample a long joint video/audio latent as smaller temporal chunks so the H3 DiT
does not have to hold the complete video sequence in VRAM at once.

The node keeps the stock sampler inputs and delegates every individual chunk to
the stock `SamplerCustomAdvanced`. It adds:

- `clip`: the same MiniMax H3 CLIP/Qwen model used by the upstream conditioning
  node;
- `prompt`: the original MiniMax-formatted prompt;
- `fps`: the frame rate used to convert prompt timestamps to global frames;
- `chunk_frames`: the maximum number of H3 frames sampled at once;
- `guide_overlap_enable` + `guide_overlap`: an independent retained latent
  warm-start. The previous tail initializes the first new target positions;
  those positions are fully denoised, kept, and omitted from prompt wording;
- `context_keyframes_enable` + `context_keyframes`: the completed tail supplied
  as native H3 video/audio keyframe conditioning over the same truthful opening
  target overlap, which is trimmed after sampling;
- `video_continuation_enable` + `video_continuation`: the independent bounded
  previous AV tail exposed as a new native `<Audio N>` + `<Video N>` reference;
  when context keyframes are disabled it also adds the previous exact final
  five-frame latent tail as one visual keyframe clip across the discarded
  packing prefix;
- `qwen_full_history`: experimental Qwen-only view of every completed frame,
  sampled at 2 FPS, without adding that history to DiT reference attention;
- automatic Gemma 4 chunk directing: it directs Chunk 1 from the complete
  source prompt, then observes chronological stills from each previous sampler
  chunk and writes the complete H3 description for the next local slice;
- `prompt_preview_only`: returns the canonical prompt plan without noise,
  per-chunk conditioning, VAE/DiT loading, or diffusion;
- `debug`: logs each chunk's rewritten prompt and frame ranges to the ComfyUI
  console and enables detailed VRAM snapshots around conditioning and sampling;
- an always-on final report: reports wall time for H3, Qwen, each VAE decode
  path, and Gemma 4, plus peak RAM and VRAM use for the sampler execution;
- `debug_stop_chunk`: returns after the selected 1-based serial chunk for fast
  boundary diagnostics; zero keeps the normal complete run;
- optional `images`: pixel images needed to rebuild Qwen visual conditioning;
- optional `vae`: the H3 video VAE, required by decoded-video modes and by
  Gemma visual directing after Chunk 1.

The sampler node id and display name are both:

```text
SamplerCustomAdvanced-Unlimited
```

An optional model-patch node provides the accumulated live preview:

```text
MiniMaxH3UnlimitedPreview -> MiniMax H3 Unlimited Preview
```

## Files

- `__init__.py` registers the node through `NODE_CLASS_MAPPINGS` and
  `NODE_DISPLAY_NAME_MAPPINGS`.
- `nodes.py` contains chunk planning, prompt rewriting, Qwen conditioning
  rebuilding, continuation guides, sampling, output assembly, and the narrow
  preview-session calls around each stock sampler invocation.
- `gemma4.py` owns the self-downloading Gemma model paths, MTMD vision runtime,
  official MiniMax prompt-guide loading, strict JSON response validation,
  image conversion, capture/replay, and free chunk-local H3 prompt directing.
- `gemma4_prompts.txt` contains the editable runtime `[SYSTEM]` and
  `[OBSERVATION]` messages passed to Gemma. It documents its supported
  placeholders and includes the relevant H3 structured-prompt conventions.
- `vendor/minimax-h3-prompt-writing/` contains reviewed byte-for-byte copies of
  MiniMax's official skill and its base/Ref2VA reference guides, which are fed
  to Gemma at runtime.
- `dependency.md` records the mutable upstream URLs, reviewed hashes, mode
  routing, and manual update procedure for those vendored prompt dependencies.
- `AGENTS.md` requires a new development session to read this file and the
  dependency record before changing the integration.
- `preview.py` contains the H3 preview model wrapper, Latent2RGB and optional
  tiny-VAE decoding, asynchronous static-WebP frame-group encoding, bounded
  server-side history, exact output-frame metadata, and local preview
  events/state restoration.
- `web/unlimited_preview.js` owns the preview widget and the browser-side chunk
  playlist, colored transport, prompt-shot range brackets, keyboard stepping,
  and frame-number overlay.
- `README.md` contains concise user-facing setup and limitations.
- `memory.md` is this implementation and design handoff.

## Development history and decisions

The first implementation used independent, strictly serial stock sampler calls.
It completed an approximately 356-frame Ref2VA render, represented as 362
frames on H3's valid temporal grid in logged tests, with roughly 30 percent
less peak VRAM than sampling the complete latent at once. This proved the basic
memory-saving design, but later chunks sometimes replayed actions or appeared
to contain two interpretations of the same shot joined together.

The serial implementation was then extended with exact per-chunk prompt logging,
sentence/clause ownership for shots that span boundaries, native accumulated
previews, and clearer continuation instructions. A proposed change that removed
a trailing phrase such as `The tiger stops` from a later prompt was reverted
after the same duplicate-shot symptom was found inside the first chunk. That
observation showed the phrase alone was not the root cause. The current parser
no longer assigns action units by proportional word position. It keeps the
complete active-shot prose and adds a timecoded continuation endpoint plus the
shot's cut/end events. Later testing established that this fixes shot-level
clock arithmetic but does not define where individual actions inside the prose
occur.

A second architecture was tested to see whether one global latent and one
diffusion schedule could preserve more state. It evaluated every temporal
window during each diffusion step. This experiment produced deterministic
changes at window boundaries and could not supply completed continuation frames
to later windows. It was abandoned and the serial architecture was restored.

The restored path was compared directly with serial commit `4be0ad9`: prompt
rewriting, completed-tail conditioning, trimming, and stock sampler delegation
were unchanged when the serial path was restored. Later sampler additions are
`debug_stop_chunk`, the outer chunk progress bar, and configurable
`context_keyframes` (then named `context_frames`). The experimental global-window commit remains on the separate
`improve-windowed-sampling` branch and an associated stash is inert; neither is
part of the active `restore-serial-continuation` branch.

The accumulated preview subsequently gained real Tiny-VAE selection,
native-resolution mode, sampling telemetry, two graphs, live elapsed time, and
paired chunk/step visibility. The separate `Previewer: ...` browser row was
removed; decoder selection and fallback are still reported in the ComfyUI log.

## Repository state at this handoff

The GitHub remote is:

```text
git@github.com:hradec/ComfyUI-MiniMax-H3-Sampler-Unlimited.git
```

Relevant published history:

- `main` contains the original serial implementation and the merged continuity
  and preview work through merge commit `9714162`;
- `restore-serial-continuation` is the active branch and is published through
  `da537bd` (`Improve continuation controls and preview`);
- `improve-windowed-sampling` preserves the rejected global-window experiment
  at `cdbbd24` for forensic comparison only.

`stash@{0}` contains a global-window follow-up experiment and is not applied.
The local `prompt.txt` is user diagnostic material, remains untracked, and must
not be added to commits unless explicitly requested.

The three independent continuation-toggle experiments described below are
currently uncommitted working-tree changes on top of `da537bd`, ready for a GPU
comparison before they are published.

## MiniMax H3 latent structure

H3 samples a nested AV latent containing two tensors:

```text
video: [B, 24, T_video, H/16, W/16]
audio: [B, 32, 2, T_audio]
```

The video temporal axis is not one latent step per pixel frame. H3 uses the
repeating frame coverage:

```text
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
```

Therefore a valid `17k + 5` pixel-frame sequence has `5k + 2` video latent
steps. Examples:

| Pixel frames | Video latent steps |
| ---: | ---: |
| 5 | 2 |
| 22 | 7 |
| 39 | 12 |
| 124 | 37 |
| 243 | 72 |
| 362 | 107 |

H3 audio latents run at 40 Hz while video normally runs at 24 FPS. The expected
audio length is:

```text
round(pixel_frames * 40 / 24)
```

The node validates both the video grid and the matching audio duration before
chunking. Invalid nested shapes fail clearly rather than producing an
incorrectly synchronized latent.

## Temporal chunk planning

`chunk_frames` is snapped down to H3's `17k + 5` grid. The default is 124
frames, or 37 video latent steps.

The first chunk contributes its complete latent. In later chunks,
`context_keyframes` supplies a separate native H3 video/audio keyframe condition
over a matching physical target overlap. Valid enabled values are 5, 22, 39,
56, and so on, corresponding to 2, 7, 12, 17, and so on video latent steps.
Those opening frames are sampled again at their truthful timeline positions and
discarded after sampling. When context keyframes are disabled, later chunks use
the minimum synthetic five-frame packing prefix needed to preserve H3's
temporal phase, and only that prefix is discarded.

`chunk_frames` remains the total amount sampled in one stock call, including
the physical context overlap or mandatory packing prefix. Increasing
`context_keyframes` adds separate H3 conditioning/attention rows and increases
the trimmed physical overlap, reducing the new content produced by a later
call. `guide_overlap` does not change geometry: it replaces retained target
latent values inside the existing allocation.

With the default five-frame context-keyframe overlap, two 124-frame chunks deliver:

```text
124 + (124 - 5) = 243 frames
```

In latent steps this is:

```text
37 + (37 - 2) = 72 video latent steps
```

The configured context-keyframe overlap latent steps are removed before final assembly. This
keeps the result on H3's native grid and makes the returned video latent exactly
the same shape as the upstream requested latent.

### Audio rounding at boundaries

The configured context-keyframe overlap duration normally corresponds to a fractional number of
40 Hz audio steps. For the default five frames it is 8.333 steps, so a join uses
eight or nine context steps. The planner calculates cumulative global audio
boundaries and chooses the matching integer length for every boundary. This
avoids accumulating audio/video drift across a long sequence and works the same
way for longer context values.

The previous chunk's audio endpoint can overhang or underhang its final pixel
frame by a fraction of one 40 Hz step because of rounding. The audio guide's
start position includes this signed offset, matching H3's native timeline
layout rather than assuming every chunk ends on an exact audio boundary.

## Noise behavior

The input noise object generates noise once for the complete requested latent
on the CPU/intermediate device. Each chunk receives the corresponding video and
audio slices through a small fixed-noise wrapper.

This was chosen instead of asking the input noise object to generate each chunk
independently because equally shaped chunks generated from the same seed could
repeat the same noise. Slicing one full deterministic noise value preserves a
stable global noise field and supports custom ComfyUI noise implementations.

The sampler seed passed to H3 is incremented for each chunk. The fixed noise
itself still comes from the original global seed; the incremented sampling seed
keeps H3's conditioning-side stochastic behavior distinct between chunks.

## Continuation from the previous generation

The node does not use the first frame of the previous generation. It uses the
previous sampled chunk's final latent tail:

```text
context_video_steps = ((context_keyframes - 5) // 17) * 5 + 2
video_context = previous_video[:, :, -context_video_steps:]
audio_context = previous_audio[..., -context_audio_steps:]
```

These tensors are cloned so they do not keep long-lived views into a larger
chunk tensor.

The video tail is added to positive conditioning as a native H3 guide at local
frame zero:

```python
{"resolved_frame_index": 0, "latent": video_context}
```

The matching audio tail is added as a separate native audio guide. Its start is
chosen so its endpoint aligns with the configured video prefix, including the
previous chunk's fractional audio-grid overhang.

The prefix is guide conditioning, not a decode/re-encode round trip. Reusing
the sampler output directly has three benefits:

- it avoids needing the video VAE and audio VAE for generated continuation;
- it avoids VAE quality loss at every link in the chain;
- it is cheaper than decoding the previous video to images and encoding it
  again as Ref2VA media.

After sampling, the configured video and audio context steps are removed. Only
new content is appended to the final output.

### Failed approach: one latent sampled through temporal windows

The experimental implementation kept one full noisy latent and evaluated all
temporal windows during every denoising step. Its intended benefit was to retain
one global sigma schedule and sampler invocation while limiting each model
forward to one temporal window.

The first variant blended predictions in overlapping window regions. It caused
visible duplicate transitions at exact overlap boundaries, observed at frames
51 and 102 in a diagnostic render. Both neighboring windows were predicting
the overlap from different temporal and prompt contexts, so averaging them did
not make them one coherent prediction.

The second variant assigned each output region to only one window and treated
the overlap as context. It removed overlap averaging but moved the visible
changes to the single-owner output boundaries, observed at frames 56 and 107.
It could also place Shot 2 before its requested global cut.

The fundamental problem affects both variants: during diffusion step N, the
next window is sampled before the previous window has completed steps N+1
through the end of the sigma schedule. Its tail is still noisy and cannot be
the final completed frames required by MiniMax's continuation method.
Passing a partially denoised overlap is not equivalent to conditioning on a
completed previous clip. The later window can independently reinterpret the
end of the prior shot, producing the observed replay, fade, early cut, or
glued-double-shot behavior.

The console exposed this architecture clearly: `Chunk 1/N` through `Chunk N/N`
repeated inside every diffusion step. In the restored serial path, chunk 1 runs
its complete step progress before chunk 2 begins its own complete progress.

The global-window architecture was removed instead of retaining another blend,
weighting, or boundary heuristic. Serial stock sampler calls are the current
behavior because each continuation begins from actual completed video and audio
latents.

### Attention state is not continuation memory

ComfyUI's sampler does not retain an attention buffer that can be detached from
one H3 invocation and injected into another. Attention keys, queries, values,
and intermediate activations are temporary results tied to a particular latent,
timestep, prompt presentation, and model forward. The stock sampler's persistent
state is the evolving latent and sampler algorithm state, not a reusable model
memory describing characters or props.

Within a chunk, H3 maintains appearance through the complete current latent,
prompt tokens, keyframes, and Ref2VA references. Across serial chunks this node
can carry only explicit model inputs: the completed latent tail, synchronized
audio tail, prompt conditioning, and reusable references. This is why an
invented detail such as a tiara can drift if it is absent or unclear in those
inputs.

H3-compatible temporal inputs use its `17k + 5` frame grid: 5, 22, 39, 56
frames, and so on. Each numeric control retains that native grid and has a
separate Boolean enable; runtime multiplies `int(enable) * value` to obtain zero
when disabled. `context_keyframes` owns native keyframe conditioning,
`guide_overlap` owns retained latent warm-start initialization, and
`video_continuation` owns the native AV reference.

### Independent continuation experiments

Four sampler controls isolate the continuation mechanisms so GPU tests can
compare them without silently coupling their effects. Context and warm-start
default to enabled at 5 frames; Video1 defaults to disabled with 5 selected.

`guide_overlap` is now a retained latent warm-start, despite its legacy name.
For each later chunk, the selected previous video-latent tail is copied into
the first retained target positions after the mandatory context/packing prefix.
Fresh target noise is unchanged, the copied positions are fully denoised and
kept, and neither the prompt nor native keyframes identify them as overlap.
This deliberately tests whether latent initialization carries grading and
other state without spending target duration on repeated output.

`context_keyframes` selects the real completed video/audio keyframe-condition
tail. It is packed as native `minimax_keyframes` at the opening local timeline,
while the same global frames occupy the physical target interval. That truthful
overlap is trimmed from output. The warm-start begins after the keyframe
overlap—or after the mandatory synthetic five-frame prefix when keyframes are
disabled—so copied warm-start positions survive.

An attempted optimization temporarily removed that matching physical overlap:
a 22-frame completed tail was still anchored as `minimax_keyframes` at local
frame zero, but only the five-frame packing prefix was trimmed. A real GPU
preview showed the previous chunk reconstructed through most of the next chunk.
Inspection of ComfyUI's `PackedLayout` confirmed why: native keyframes are fixed
anchors on the target timeline, not detached historical memory. The experiment
was reverted. A contiguous multi-frame context keyframe must now have an equal
physical target overlap that is trimmed after sampling.

`video_continuation` is a third independent frame count. When nonzero, the
full-reference continuation experiment clones exactly that final AV tail from
the completed previous chunk. It does not change physical target geometry.
The H3 VAE decodes that bounded tail, and Qwen is
shown frames sampled at the same 2 FPS cadence used by ComfyUI's stock H3
Ref2VA node. The matching generated-audio tail is selected using cumulative
global 40 Hz endpoints rather than rounding the isolated duration; this avoids
an occasional one-step AV error for fractional durations such as 22/24 second.
The clean bounded pair is independently appended to `minimax_refs` as a native
`video_audio` block, so the DiT can attend to both continuation streams without
merging them into the noisy target latent.

When `video_continuation` is enabled while `context_keyframes` is disabled, the
node also clones the previous sampler chunk's exact final two video-latent
tokens, which represent its final five pixel frames, as one native video
keyframe clip anchored at local frame 0. The clip spans local frames 0-4: the
complete mandatory five-frame synthetic packing prefix immediately before
retained local frame 5. That prefix was already required and is still
discarded, so the boundary clip adds two video-latent conditioning time steps but no
physical overlap, target-frame allocation, or VAE re-encode. No separate audio
keyframe is added; the synchronized bounded `<Audio N>` latent remains in the
Video1 reference block. This replaced a failed one-image experiment: a real GPU
test showed that the single final-frame keyframe could preserve a chunk 2-to-3
camera join but did not prevent chunk 2 from choosing a different opening camera
state. Two separate runs with ordinary five-frame video/audio context keyframes
plus a 22-frame Video1 reference did preserve the first chunk boundary, which
motivated isolating the five-frame visual trajectory here.

Commit `2254120` checkpoints the working five-frame-boundary-plus-Video1 state.
A subsequent temporary isolation run set `INCLUDE_VIDEO1_REFERENCE = False`:
the five-frame visual boundary remained, while the decoded Qwen Video1 item,
DiT `minimax_refs` Video1/Audio1 block, and injected prompt labels were all
disabled. Its initial visual result was nearly identical to the complete path,
showing that the five-frame boundary was doing most of the immediately visible
join work, but this does not prove the Video1 integration is correct. The switch
is restored to `True`; the gates remain available for another comparison. This
experiment exists because an initial Video1-only run had no first-boundary
consistency despite official/community reports that Ref2VA video extension can
work without extra keyframes. Do not assume a model limitation until the exact
last-run prompt capture and native reference packing have been audited.

The generated video and synchronized soundtrack labels use the next available
video and audio ordinals. Following ComfyUI's native ordering, Qwen receives the
`<Audio N>` label immediately before the decoded `<Video N>` frames. Its H3
tokenizer does not ingest a waveform for that audio item; the actual audio
latent goes directly to the DiT inside the combined reference block. Therefore
this path needs no audio VAE and does not pretend that Qwen can hear generated
audio. A later chunk prompt gains concise definitions and retention entries for
both labels, a `[video continuation]` task type, and a detailed-description
instruction placing both synchronized endpoints at the same position on the
active shot's timeline. The original prompt is used afresh for every chunk, so the dynamic sections cannot
accumulate. The implementation prefers `detailed_description` when a hybrid
diagnostic prompt also contains `integrated_multimodal_description`; this
prevents `[Shot N]` mentions in Ref2VA analysis sections from being mistaken
for timeline markers.

`qwen_full_history` deliberately changes only Qwen conditioning. Before each
later chunk, the already assembled output from the beginning through the last
completed chunk is decoded and sampled at 2 FPS. That video presentation is
appended to Qwen's reference items, but no matching `minimax_refs` block and no
prompt rewrite are added. This directly tests whether long visual history in
Qwen improves consistency without increasing DiT reference attention. The
currently sampled chunk cannot be included because it is still noise when its
Qwen prompt is encoded.

`prompt_preview_only` runs the same canonical planning function used by normal
sampling and returns every active chunk's exact prompt plus sampled/output frame
ranges. It exits before full-noise generation, per-chunk Qwen encoding, preview
setup, VAE decoding, model loading, or DiT sampling. Noise, sampler, sigmas, and
CLIP are lazy inputs, so ComfyUI does not evaluate those branches for this mode.
The toggle is an optional schema input with a `False` execution default so
workflows saved before it was introduced continue to validate and sample
normally instead of failing with `Required input is missing`.
The guider and H3 latent remain required because the planner needs reference
types/ordinals and the exact nested AV temporal shape. The two latent outputs
are unchanged placeholders and are not valid generated results in this mode.

The mode is an explicit serialized boolean rather than being inferred solely
from connected output sockets. ComfyUI caches a node by its inputs and upstream
ancestry, not by the set of downstream output sockets currently requested. An
automatic text-only execution returning placeholder latents could otherwise be
cached and later reused after a latent output was connected. Including
`prompt_preview_only` in the node inputs gives preview and sampling executions
different cache signatures. `chunk_prompts` is populated from the same plan in
both modes; `debug` controls console/VRAM logging rather than text availability.

Both decoded-video modes require the H3 video VAE and a Ref2VA conditioning
whose original Qwen presentation can be reconstructed by this node. Decoded
pixel tensors are execution-local and released after CLIP encoding. Dynamic
video presentations are sampled at 2 FPS and downscaled to at most a 512x512
pixel-area budget before Qwen encoding. This does not resize the clean DiT
reference latent or any original Ref2VA image. The native mode keeps only its
bounded full-resolution clean latent for the sampler call. The full-history
mode keeps no additional DiT latent, although its temporary VAE decode and Qwen
token count grow with completed duration. When both experiments are enabled,
Qwen receives the bounded native video followed by the full-history video; only
the bounded one is referenced by the prompt and DiT.

### Conditioning-model eviction and the Qwen token wall

A 1920x1088 diagnostic using `chunk_frames=56`, `context_keyframes=22` (then
implemented as a repeated physical prefix), native
video continuation, and no Qwen full history completed chunk one but OOMed at
chunk two's first DiT QKV projection. The decoded video and Qwen prompt had
already completed. The console immediately before the failure reported
`0 models unloaded`.

The first hypothesis was residual VAE/Qwen model residency. VAE decode and CLIP
encode call ComfyUI's dynamic loader but do not unload their models after
returning, and dynamic loading avoids evicting other dynamic models when
possible. MiniMax H3 also does not currently report `minimax_refs` through
`extra_conds_shapes`, so generic sampling-memory estimates omit the packed
reference rows.

After every chunk's prompt encoding, the sampler now calls ComfyUI's targeted
`unload_model_and_clones` first for the CLIP patcher and then for the connected
VAE patcher. This runs before stock `SamplerCustomAdvanced` prepares the DiT,
including for chunk one when an upstream node may have left the VAE loaded. The
encoded cross-attention, token tags, and small clean reference latent remain;
decoded pixel frames have already left scope. Targeted eviction also releases
the associated CUDA cache without unloading unrelated models. The tradeoff is
additional model reload/offload time at every chunk boundary.

A second render confirmed both targeted unload calls before chunks one and two
but failed at the identical QKV operation. Allocated and peak memory changed by
only about 5 MiB (`9383/12123 MiB` before eviction versus `9378/12118 MiB`
afterward). This disproved retained conditioning-model weights as the immediate
cause of that OOM, although explicit eviction remains useful headroom hygiene.

The successful 56+22 guide and failing 56+22 native reference have the same
14,280 clean visual-condition rows at 1920x1088. The native path additionally
presents two decoded full-resolution frames to Qwen. Qwen turns that pair into
approximately 2,040 more packed vision rows. The installed INT8 linear backend
scales output in parts, retains every part, and then allocates a second complete
buffer with `torch.cat`; this small sequence increase crosses its sharp QKV
allocation threshold.

Dynamic Qwen video frames are therefore resized by aspect-preserving pixel area
to at most `512 * 512` before tokenization. A 1920x1088 pair becomes 672x384,
reducing its merged vision rows from roughly 2,040 to 252. The separate clean
video latent still enters the DiT at the original 1920x1088 latent resolution.
Debug logging reports the actual frame count and Qwen presentation resolution
for every dynamic video item.

### Debug VRAM accounting

Debug mode records memory at execution setup, each chunk start, before and
after VAE decode/Qwen encoding, after targeted conditioning-model eviction,
immediately before the stock sampler, and after it returns. A temporary
`APPLY_MODEL` wrapper also records memory directly before and after every DiT
evaluation. This is earlier than the normal sampler callback, so the first
failed evaluation in an OOMing chunk still produces a snapshot and an
abbreviated CUDA allocator summary.

Each snapshot distinguishes device memory used by all processes, physically
free memory, PyTorch allocated/active/reserved memory, inactive cache, and
per-chunk peaks. It also lists the loaded sizes reported by the known H3 DiT,
Qwen/CLIP, and video-VAE patchers and every patcher in ComfyUI's resident-model
registry, including each patcher's active patch-key count so failed LoRA
attachment is visible. Selected node-owned and model-call tensors are shown as
logical GPU payload sizes. Those payload values can overlap when tensors are
views or are reachable under more than one conditioning field; the
PyTorch/device totals remain the authoritative allocation figures.

The lightweight model-call monitor is installed for every sampling execution
and is removed in the same `finally` cleanup that restores the guider
conditioning. With `debug` disabled it only records silent before/after-DiT
memory observations; detailed model/tensor residency logs and allocator failure
summaries remain debug-only.

Every sampler execution, including non-debug runs and partial/error exits,
writes a structured final timing and memory baseline. It includes the effective
continuation configuration, rendered and planned ranges, resolution, step count,
sampler wall time, completed-chunk average, projected full-run time after a debug
stop, and a component table. The table separately accumulates H3 sampling, Qwen
encoding/tokenization, VAE decoding of the prior chunk for Gemma, continuation
Video1 VAE decode, VAE decoding of Qwen full history, and Gemma 4's
whole local handoff.

Physical VRAM is sampled device-wide once per second, as well as at sampler
stage boundaries and immediately before and after each DiT evaluation. The
report shows the average of the explicit stage/DiT snapshots, H3-specific and
later-chunk H3 averages, the device-wide peak, and PyTorch allocated/reserved
high-water. `Peak Time` appears immediately after `Peak`: it estimates the
wall-clock time for which physical VRAM was closer to Peak than Average, which
is precisely usage above `(Average + Peak) / 2`. Threshold crossings are
linearly interpolated between adjacent samples. This timeline also observes
GPU allocations made by the isolated Gemma worker, which PyTorch's allocator
statistics inside ComfyUI cannot see. Peak ComfyUI-process RSS and system RAM
usage are reported separately.

Each multi-line debug VRAM snapshot immediately redraws active tqdm bars after
it is printed. The sampler's outer `chunk` bar is redrawn before ComfyUI's
inner `steps` bar, leaving the current sampling step visible below the memory
report instead of several terminal screens above it.

An initial four-step diagnostic passed the previously failing second-chunk
first evaluation while the GPU stayed near its physical capacity, but that
process rejected every H3 LoRA key and reported `MiniMaxH3 ... 0 patches
attached`. A restarted run with the latest sampler then reported 208 H3 patch
keys, presented the bounded Qwen video as two 672x384 frames, and also passed
chunk two. Finally, the same workflow without the LoRA and with the normal 20
steps did not OOM. Reducing the step count does not change one DiT evaluation's
tensor shapes, and the LoRA was not common to the successful tests. The shared
change was therefore the updated continuation path: bounded Qwen-video
downscaling plus explicit Qwen/VAE eviction. This confirms that the former
full-resolution Qwen vision rows were what pushed the chunk-two INT8 QKV
allocation past the 16 GB limit.

Attention-patch isolation tests produced the same conclusion. With Sage
Attention plus SOL-attn and Spectrum disabled, chunk two peaked at about
12,057 MiB active and 14,144 MiB reserved. With Spectrum Apply MiniMax H3
enabled, the comparable chunk peaked at about 12,050 MiB active and 14,208 MiB
reserved and continued into chunk three. The roughly 64 MiB reserved-memory
difference is negligible beside the 12 GiB DiT forward and does not grow by
chunk. Spectrum was therefore not the source of the former chunk-two OOM.

The official base guide clarified that ordinary first/last-frame workflows are
I2VA/FL2VA/L2VA tasks with a fixed alignment instruction and three core prompt
fields. The six-section format and `[video continuation]` task type belong to
the full-reference guide. For that reason, neither the guide-only path nor the
Qwen-history-only experiment receives the new Ref2VA summary text.

## Existing H3 conditioning

The sampler receives a guider whose `original_conds` already contains converted
conditioning dictionaries. The implementation clones these dictionaries for
each chunk and restores the original guider state in a `finally` block.

Only positive conditioning is changed. Negative or other guider branches remain
untouched.

Existing `minimax_keyframes` use global resolved frame positions. For each
chunk, keyframes intersecting that chunk are copied and remapped to local frame
positions:

```text
local position = global position - chunk global start
```

This allows an upstream first-frame or last-frame guide to affect only the
chunk containing that global frame. Existing `minimax_refs`, control metadata,
and other conditioning fields are preserved.

## MiniMax shot prompt handling

The implementation follows the official MiniMax prompt guides:

- `[Shot 1]` is the opening shot and has no timestamp.
- Every later shot uses `[Shot N] At MM:SS.mmm,`.
- Shot numbers must start at one and increase sequentially.
- Cut timestamps must be strictly increasing.

The parser recognizes both official description fields:

```text
integrated_multimodal_description:
detailed_description:
```

The first is used by T2VA/I2VA/FL2VA/L2VA prompts. The second is used by
full-reference prompts. Prefix sections such as `subject_definitions`,
`summary`, and `retention_analysis`, and suffix sections such as
`overall_soundscape` and `non_diegetic_music`, are preserved.

For each physical chunk, the current parser:

1. converts every absolute `MM:SS.mmm` timestamp to a global frame using
   `round(seconds * fps)`;
2. keeps only source shots whose intervals intersect the *sampled* physical
   window, including an already-completed predecessor when carried guide frames
   straddle a source cut;
3. renumbers those selected shots locally and writes the documented H3 form
   `[Shot N] At MM:SS.mmm, ...`, with the time measured from the physical chunk
   start so a cut lands after any carried guide prefix;
4. uses only a compact preservation line for a predecessor represented solely
   by carried frames, rather than replaying its completed source action;
5. uses original source bodies in its deterministic preview/fallback output.
   During a normal render, Gemma instead directs the complete local description
   for every chunk from the same shot/frame facts.

For example, at 24 FPS:

```text
global cut frame: 100
chunk global start: 50
local cut frame: 50
local cut time: 00:02.083
```

During normal sampling, H3 no longer receives the deterministic repeated source
shot, a custom time range, a reference-endpoint clock, or a synthetic `shot
ends` command. Gemma directs every chunk—including Chunk 1—from the complete
source intent and exact slice facts. On later chunks it also observes retained
frames from the previous generated output and writes only events appropriate to
the new slice. The sampler itself owns canonical timecodes only for genuine
source cuts that physically occur in the current chunk.

Proportional sentence and clause slicing was removed after practical tests
showed that word position does not represent action timing. It could leave a
chunk with a fragment such as `The tiger stops`, causing H3 to restart or
invent a shot without the original action and camera context.

### Retired prompt-timing experiments (historical)

An initial experiment placed verbose complete-duration, reference interval,
target interval, and stop instructions before the full shot prose. MiniMax
respected the reference-defined start and did not replay completed action, but
ignored the requested target endpoint and compressed the remaining prose into
the available latent duration. That experiment was removed.

A retired legacy experiment preserved the complete active-shot prose and might append
a simple shot-relative command after the final active split shot: `At
MM:SS.mmm, shot ends.`. The earlier bracketed form, `[MM:SS.mmm] Shot ends.`,
was replaced after recognizing that MiniMax's documented timeline-event grammar
uses `At MM:SS.mmm`. Earlier active shots do not receive the suffix when another
timed shot follows in the chunk. The following chunk-local `[Shot N] At ...`
cut already defines their endpoint. The old behavior could put a shot-relative
`00:04.667` immediately before a chunk-local `00:01.083` cut. This behavior is
removed when the sampler was simplified around Gemma semantic continuation.

The subsequently retired default `storyboard` prompt experiment followed a
MiniMax-like storyboard shape: every selected
source shot gets one master header, a completed-reference micro-range when
available, a new-generated micro-range, the complete original shot prose on
the full master range, and an explicit endpoint. For example:

```text
Shot 1 | 00:02.000-00:07.000 | Duration 5s | Reference <Video 1>
00:02.708-00:03.625: <Video 1> is the already completed portion of this shot.
00:03.625-00:04.333: Continue from the end of <Video 1>. Generate the part of the overall shot that occurs during this interval.
00:02.000-00:07.000: <full original shot prompt>
00:07.000: Shot ends.
```

The header and full-shot range use the source prompt's global timeline rather
than a chunk-local clock. This deliberately tells H3 that the prose belongs to
the whole shot, while this invocation has a bounded place inside it. Chunk 1
also receives a generated-range line so it does not compress a long opening
shot into its first short latent. Later native Ref2VA chunks label the exact
decoded `<Video N>` range; guide-only chunks instead label the same interval as
`Provided opening frames`. In a chunk that crosses a cut, the planner writes a
separate storyboard block for every source shot whose new generated interval
*or* bounded reference interval intersects the chunk. Thus a reference ending
exactly at a cut still identifies its completed preceding shot, even when the
new generated range starts at the following shot. A reference-only predecessor
contains only its header and compact completed-reference range, not the full
already-finished action prose or a second `Shot ends` event; full prose and the
endpoint are retained only for a shot with new output. A reference interval is
associated only with a shot it actually overlaps.

For a shot with new output, its original multi-line prose is flattened into one
line before insertion. Newlines become sentence boundaries (a period is added
only when the preceding line did not already end in terminal punctuation), and
the line is written as `master-range: ONLY GENERATE THE TIMESLICE generated-range
OF THIS FULL PROMPT: <flattened full prompt>`. This removes newline/list
structure that could distract H3 while making the requested bounded interval
the first instruction attached to the complete source prompt.

Both retired prompt formats performed interval math in integer frames and exposed only
`MM:SS.mmm` values to MiniMax. A native continuation prompt identifies `<Video
N>` concisely in `subject_definitions`, `summary`, and `retention_analysis`.
For the retired `legacy` path, the first active shot said `<Video N>
and its synchronized <Audio N> ends At MM:SS.mmm`, using the reference
endpoint's position on a cumulative timeline anchored at the first active
source shot. Every following `[Shot N] At ...` cut
and the final `At MM:SS.mmm, shot ends.` command use that same clock. The events
therefore remain monotonic and tell MiniMax which part of the complete shot
description the reference has already covered and how much remains. Immediately
after the endpoint, the prompt says `Continue from this timecode; all subsequent
timecodes use the same timeline.` The wording deliberately avoids `relative to
this timecode`, which could imply that later values are offsets from a reset
clock rather than positions on the cumulative clock. An earlier implementation mistake
used the endpoint's chunk-local position instead (for example `00:00.208` for
a five-frame guide overlap at 24 FPS). That described only the overlap inside
the new sampling window and did not locate the reference within the shot. In a
long shot spanning chunks 5, 6, and 7, H3 consequently received the same full
action prose with no shot-progress anchor and replayed the action shortly after
the chunk 6 handoff. The endpoint is now calculated as `content_start -
shot_start`; the bounded reference length and guide-overlap length remain
unchanged. A subsequent preview exposed another contradiction: the reference
ended at `00:03.792` on the active-shot clock while a following cut was still
written as chunk-local `00:01.083`. Following cuts are now measured from the
same first-active-shot origin, and a later shot's endpoint is cumulative from
that origin rather than resetting to that shot's duration. The previous verbose
paragraph mapping every part of the bounded reference to several time ranges
remains removed.

In those retired variants, the configured latent overlap was excluded from the new action range because it
is continuation context and is trimmed from the assembled output. This matters
for sparse descriptions: a single sentence can span more than one chunk, and
removing it from later chunks leaves MiniMax with no concrete shot description
and can cause an unintended cut or a new interpretation based only on reference
images. The continuation instruction and completed opening frames tell MiniMax
not to restart the overlapping sentence's action.

The removed proportional division explains apparently isolated phrases in old
debug output. For example, `The tiger stops` was selected because its inferred
sentence/clause span overlapped the chunk, not because the parser understood
its meaning. That behavior was ultimately removed because prose length is not
a reliable action clock.

### Confirmed limitation: complete prose has no action-level clock

An August 2026 diagnostic at 24 FPS confirmed that the current native
continuation timestamps were arithmetically correct while H3 still replayed
different parts of one long shot. The shot began at global frame 118 and
contained this ordered action:

```text
Tila closeup and pointing -> dialogue -> continuous zoom out ->
Heman dismounts and walks right
```

Chunk 4 sampled frames 102-140 and contributed frames 107-140. Its continuation
reference ended at `00:01.625` on the preceding active-shot clock and the Tila
cut was correctly placed at `00:02.083`, global frame 118. The prompt explicitly
said `Without a cut` before the zoom-out and dismount, but the generated chunk
showed only a quick Tila closeup followed by an invented hard cut to Heman
already walking. That hard cut was model behavior, not a second cut inserted by
the planner.

Chunk 5 sampled frames 136-174 and contributed frames 141-174. Because the Tila
shot started at frame 118, its bounded `<Video 1>` endpoint was correctly
reported as:

```text
(141 - 118) / 24 = 00:00.958
```

Nevertheless, the complete shot prose followed that endpoint instruction,
including the already shown `Tila ... points to the right` action. H3 replayed
the pointing in chunk 5. Chunk 6 contributed from frame 175, correctly placed
the endpoint at `(175 - 118) / 24 = 00:02.375`, received the same complete
prose again, and this time resumed with the zoom-out. The final shot endpoint
was consistently `00:04.667`, corresponding to the shot's complete 112-frame
duration.

This is not an accidental duplicate `[Shot N]` block, overlap calculation
error, or non-monotonic timecode. It is a prompt-planning limitation. The
continuation endpoint tells H3 how far the reference has progressed through the
shot's clock, but the prose contains no timestamps mapping `point`, `speak`,
`zoom`, `dismount`, or `walk` to that clock. H3 must infer that mapping from a
short visual reference and can choose a different action on each continuation.
Complete-shot repetition is therefore insufficient for a shot that spans
several chunks and contains multiple sequential actions.

### Retired Gemma action ledger and implemented free chunk-prompt director

The first Gemma implementation mechanically split an original source shot at
sentences, semicolons, and newlines and labeled the resulting pieces
`S<shot>.A<action>`. Gemma had to report an ordered completed prefix and rewrite
only the first unfinished ledger item. This was too restrictive and the split
was not semantic. For example, one ledger item combined camera zooming, Heman
dismounting, and Heman walking. Gemma could correctly see Heman already on the
ground while the compound item remained in progress, then reintroduce “begins
to dismount” and make H3 reset his pose. The action ledger, stable IDs,
completed-prefix state, and algorithmic prose splitting are removed.

Gemma is now the free prompt director for every normal sampled chunk, including
Chunk 1, a chunk starting exactly at a new source shot, and a chunk containing
parts of two or more shots. It uses `n_gpu_layers=-1` while H3, Qwen, and the
VAE are explicitly unloaded, in a disposable worker that exits before H3
resumes. Later chunks require the H3 video VAE to decode visual evidence; the
first chunk is a text-only Gemma request.

Every request contains the complete unchanged user prompt plus deterministic
facts that only the sampler should own:

- current physical sampled range and retained output range in exact global
  frames;
- a front-loaded current-chunk shot map, naming every local/source shot with
  retained output, its exact source global start, physical local start, and
  owned global/local frame span, followed by a direct request naming whether
  each shot contributes its opening, middle, ending, or complete portion;
- a per-shot timing contract that states complete source duration, the exact
  source-relative frame/percentage range owned now, and the requirement to
  state concrete covered versus deferred events before directing H3;
- previous physical sampled/retained ranges;
- the exact global frame number of every chronological attached still;
- 2 FPS stills from the complete previous sampled chunk plus its exact final
  decoded frame;
- the exact `detailed_description` Gemma directed for that same previous
  sampled chunk, presented immediately beside its still manifest so Gemma can
  compare intended action beats with what H3 actually rendered;
- the same chunk's Gemma-only `timing_plan` and concise `end_state`, linked to
  those stills and passed only to the next Gemma request, never to H3;
- all source shots intersecting the previous and current ranges, with complete
  original shot bodies preserved unsplit;
- the portion of each source shot assigned to this chunk;
- exact required local `[Shot 1]` and `[Shot N] At MM:SS.mmm,` markers for real
  source cuts, repeated in an explicit copy-only H3-local marker block that
  distinguishes them from the full-video/global source timestamps; and
- the actual opening conditioning available to H3: keyframe prefix, bounded
  `<Video N>`/`<Audio N>`, fully denoised latent warm-start, or none.

Gemma interprets what the previous generated sequence already accomplished,
retains the source shot's complete intent rather than its exact wording, and
freely rephrases, reorganizes, clarifies, or enhances concrete H3 instructions
for only the current frame slice. It must not compress a whole continuing shot
into one chunk, replay a visibly completed action, change dialogue or explicit
sound events, invent cuts/outcomes, or contradict reference meanings. For a
multi-shot chunk it writes the complete local sequence itself, including every
required real-cut marker. The response JSON contains `confidence`, a factual
`analysis` summary, a per-shot factual `timing_plan`, a concise Gemma-only
`end_state`, and the complete chunk-local `detailed_description` value. The
sampler preserves all other structured prompt sections and replaces only the
description value, so H3 never sees the planning or end-state metadata. A
legacy `[end state]` paragraph mistakenly appended inside
`detailed_description` is deterministically extracted into `end_state`, removed
before H3 encoding, and recorded as a warning next to the raw JSON. If the
initial response has any missing, reordered, renumbered, or wrong-time marker,
the worker gives Gemma exactly one additional correction turn. That turn shows
the validation errors and literal local tokens, while retaining the first JSON
as assistant context; Gemma must return another complete JSON object itself.
No description text is patched or generated by sampler code. Both raw JSON
attempts and the correction request are kept in the capture/transcript, and a
console warning identifies the initial marker failure even when the correction
succeeds. If the one correction response is still invalid, marker,
section-shape, and dialogue checks remain warning-only diagnostics and its
usable description is sent unchanged. The sampler must never replace a usable
Gemma output with an algorithmically generated source-prompt fallback. A
response with no usable `detailed_description` instead stops sampling; it
cannot truthfully be turned into an H3 prompt and is recorded with its raw JSON
when one was parsed.

`gemma4_prompts.txt` remains editable and is reread for every request. In
addition, the official MiniMax-maintained H3 prompt-writing `SKILL.md` and one
mode-specific reference are vendored under `vendor/minimax-h3-prompt-writing/`
and appended directly to Gemma's system message: base modes receive
`references/base-en.txt`; Ref2VA receives `references/ref-en.txt`. The
chunk-local contract takes precedence over their full-video examples. Their
mutable upstream URLs, reviewed hashes, and update procedure are recorded in
`dependency.md`; render-time network fetching is intentionally forbidden for
reproducibility. `AGENTS.md` requires future development sessions to read both
this memory and the dependency record.

Every normal sampling execution also replaces the fixed temp capture
`${TMPDIR}/comfyui-minimax-h3-unlimited/last_gemma_chunk_prompts.txt`. After
Gemma directs a chunk, the sampler appends and closes a complete Gemma-to-H3
transcript entry before starting its DiT call. The exact system message returned
by the disposable worker, including the vendored MiniMax documentation, appears
once near the top of the file. A 200-character `=` line precedes every
`=== Chunk ...` header; each chunk then contains the exact rendered
observation/user request sent beside its images, every raw Gemma JSON, any
marker-correction request, `=== GEMMA VALIDATION WARNINGS ===`, and the exact
final structured H3 prompt encoded by Qwen. A parsed raw JSON response is
retained even if it has no usable description and sampling must stop. This
capture is always active even when
`debug` is off and remains useful after an interrupt or sampler failure.
`prompt_preview_only` intentionally does not overwrite the most recent real
sampling-run capture.

The same run deletes and recreates the sibling `last_gemma_images/` directory.
`Gemma4ContinuityDirector` writes the exact quality-88 JPEG byte payloads that
it places in the worker's multimodal request, even when `debug` is off. Files
use `chunk_NNN_source_frame_NNNNNN.jpg`, where the chunk number identifies the
Gemma target request and the frame number is the exact global frame decoded
from the immediately preceding H3 chunk. Saving happens before the disposable
worker starts, so images remain available when Gemma or H3 later fails. This is
separate from debug-only replay fixtures, contains only the concise still set,
and is reset only by another real sampler execution—not prompt preview.

The llama context was raised from 8192 to 16384 for the official guide, full
source prompt, visual tokens, and response. This is intentionally not 256K on
the 16 GB RTX 4070 Ti SUPER: the real llama.cpp debug log showed an 8192-cell
F16 KV allocation of 2560 MiB, so this backend scales to roughly 5 GiB at 16K
before adding the 6638 MiB model and approximately 527 MiB compute buffer. A
256K allocation would not fit even though the GGUF metadata declares a native
262144-token context.

Gemma 4's observation path now follows Google's image-first modality order:
the chronological stills precede the observation text in the user content
array. The pinned `llama-cpp-python==0.3.35` MTMD structure exposes
`image_min_tokens` and `image_max_tokens`, but its public `MTMDChatHandler`
constructor does not. `gemma4.py` therefore creates a narrow project-local
subclass inside the disposable worker and sets the official dynamic range to
70-1120 visual tokens. Its MTMD batch maximum, llama logical `n_batch`, and
physical `n_ubatch` are all at least 1120 so a maximum-detail non-causal image
chunk is not divided by the previous 512-token batch setting.

The exact installed official projector was probed without running inference.
Its unset/default range reported 92,160-645,120 pixels, with the 48x48 patch
size making the maximum 280 visual tokens. Explicit budgets map to 161,280
pixels at 70, 645,120 at 280, 1,290,240 at 560, and 2,580,480 at 1120. Thus the
latest captured 672x384 observations already fit below the old ceiling, while
a 2048x1152 observation would previously have been reduced to roughly
1070x600. The new dynamic maximum can retain that complete 2K pixel area;
smaller inputs are not forcibly upscaled to 1120 tokens because the minimum
remains 70.

Debug captures now use schema v2 and directories named
`prompt_NNN_chunk_NNN`. Each stores the exact request/base64 JPEGs, JPEG files
named by global frame, rendered messages, manifest, and result/error. Replay
uses the current editable prompts and vendored MiniMax guide against identical
frames and chunk facts. Old `observation_NNN_shot_NN` schema-v1 action-ledger
captures remain useful historical evidence but cannot reproduce the new input
contract and are rejected with an explicit migration message.

`gemma4.py` downloads the exact files from
`https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf` to the persistent
ComfyUI model path `models/llama_cpp/gemma-4-12b-it-qat-q4_0`. `hf_hub_download`
uses that local directory directly, avoiding a duplicate seven-gigabyte global
Hugging Face cache copy. The first release is intentionally visual only: no H3
audio VAE is connected, so Gemma cannot determine whether speech was completed.
That is a future extension, together with richer action segmentation and
full-current-shot semantic history when the immediately previous chunk is not
enough.

The required binding is exactly `llama-cpp-python==0.3.35`: it exposes the
generic MTMD Python handler used for Gemma 4's `mmproj`. The development
environment has 0.3.35 installed through ComfyUI's isolated
`tools/python.sh` interpreter.

### Planned experiment: align chunks to shot boundaries

A separate sampler toggle should test shot-aware chunk planning. When enabled,
the planner should prefer ending a sampler call at a source-prompt shot
boundary instead of placing one physical chunk across two shots. A shot that
fits within the effective temporal capacity would then be sampled normally as
one complete single-shot call. Gemma would only be needed to direct continuity
when one shot itself is too long for a single call, rather than intervening at
every ordinary cut.

Long shots should be divided into balanced intra-shot segments. In particular,
if a shot exceeds the available capacity by only one to five frames, sampling
one maximum-sized segment followed by a nearly empty segment would give H3 no
useful temporal runway. The planner should instead split that shot roughly in
half and sample two meaningful smaller segments. More generally, it should
choose the minimum required segment count and distribute the shot duration
across those segments so the final remainder is not pathologically short.
Gemma's adaptive AV observation would run only at those intra-shot joins.

This design must use effective *new-output* capacity, not blindly compare shot
duration with the `chunk_frames` widget. Later calls currently spend only the
minimum five-frame local packing prefix; a future restored physical guide
overlap would consume additional new-output capacity. The complete sampled
window must still stay at or below the requested VRAM cap.

There is also an unresolved H3-grid constraint. Valid H3 sampling windows use
the `17k + 5` pixel-frame grid, and the output advance between ordinary
continuation chunks is correspondingly quantized. Prompt cuts can occur at any
integer frame after timestamp conversion, so an arbitrary shot endpoint may
not be a legal physical chunk endpoint. Exact shot alignment therefore cannot
be implemented by simply changing `frame_end`; it needs a proven padding,
ownership, and trimming scheme that preserves the final AV latent grid without
reintroducing the rejected overlapping-window blend/ownership artifacts. The
toggle should remain experimental until that mapping is validated. A fallback
may align only grid-compatible cuts and retain the existing planner for others.

Gemma is intentionally an internal sampler dependency. It does not use
ComfyUI's `CLIP`, Generate Text node, an LLM socket/input, Ollama, or a server
process. The implemented director uses Google's official Gemma 4 12B
instruction QAT Q4 GGUF at
`https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf` through
`llama-cpp-python`'s generic MTMD handler inside a short-lived child process.
The user had already
benchmarked this model at roughly 150 tokens per second on this GPU, so the
between-chunk visual planning latency is expected to be practical; multimodal
input length, temporary model swapping, and GPU-layer count still affect the
integrated result. The actual download, model/projector loading, validated
visual observation, and immediate release are described in the implemented
Gemma section above.

The source prompt's cut positions are converted to integer frames for exact
internal intersection math, but generated prompts do not expose global frame
numbers. With native video continuation enabled, actual future cuts, the
bounded continuation endpoint, and the final split-shot endpoint use a single
cumulative timeline beginning at the first active source shot. Guide-only
continuation retains the established chunk-local cuts and shot-duration suffix.
A cut placed only a few new frames after a continuation boundary can still be
difficult for a generative model even when the timestamp conversion and latent
handoff are correct.

## Replacing only prompt-owned conditioning

The original prompt has already been processed by MiniMax's Qwen3-VL text
encoder when it reaches the guider. Arbitrary text cannot be inserted directly
into the final hidden-state tensor. The node therefore needs the original
`clip` and prompt.

For each chunk it tokenizes and encodes the rewritten prompt, then replaces
only these positive-conditioning fields:

```text
cross_attn
minimax_token_tags
```

These are the fields owned by the Qwen prompt presentation. The node does not
replace latent references, keyframes, controls, sampler settings, or unrelated
metadata.

## What the `images` input means

The optional `images` input is used only to reconstruct Qwen/CLIP visual
conditioning. It does not create H3 video-VAE or audio-VAE latents.

For ordinary I2VA/FL2VA conditioning, the image batch is interpreted as:

```text
images[0] -> <Picture 1> / original first-frame image
images[1] -> <Picture 2> / optional original last-frame image
```

The first image is resized to the generation canvas with the same stretched
geometry-anchor behavior used by the stock H3 node. Later images use
aspect-preserving center crop, matching the stock last-frame behavior.

For image-only Ref2VA, each batch item is a separate picture reference:

```text
images[0] -> <Picture 1>
images[1] -> <Picture 2>
images[2] -> <Picture 3>
```

The image references are reconstructed in the same order as the existing
`minimax_refs` payload. Audio-only reference blocks and the dynamic continuation
soundtrack contribute their `<Audio N>` labels to Qwen but need no waveform
there because their audio latents already exist in `minimax_refs`.

The optional video VAE is used only to show continuation/history frames to
Qwen. No audio VAE is needed for generated continuation:

- the `images` input supplies only pixels for Qwen visual tokens;
- upstream keyframes and Ref2VA blocks are already VAE-encoded in the guider;
- previous generated video and audio are already available as sampled latents.

Ordinary I2VA/FL2VA images are presented to Qwen only for the first chunk.
Presenting them again would relabel the original image as a fresh local
`<Picture 1>` and repeat its zero-second alignment instruction, competing with
the previous chunk's latent continuation guide. Later ordinary chunks remove
those picture-alignment lines and replace remaining picture-label prose with a
neutral reference to the already established subject and scene. Image-only
Ref2VA inputs are different: they describe reusable identity, scene, motion, or
style references and remain presented to every chunk.

## Why an IMAGE batch is not a Ref2VA video

An IMAGE batch normally represents separate `<Picture N>` references. A
Ref2VA `<Video N>` has additional temporal structure:

- all frames belong to one ordered media item;
- Qwen samples the video at 2 FPS and inserts timestamps;
- the video VAE encodes the complete temporal sequence as one reference latent;
- an optional synchronized soundtrack is encoded by the audio VAE;
- the prompt refers to the sequence as `<Video N>`, not as many pictures.

Consequently, the node refuses existing `video` or `video_audio` Ref2VA blocks
when rebuilding Qwen conditioning from `images`. Silently treating those frames
as separate pictures would make the Qwen presentation disagree with the DiT
reference payload.

Correct support for new Ref2VA videos would require dedicated inputs such as:

- one or more separately delimited `ref_video` IMAGE sequences;
- the MiniMax video VAE;
- optional reference-video audio and the MiniMax audio VAE;
- an explicit choice between a full reusable reference and a reference aligned
  one-to-one with the target's global timeline.

For a timeline-aligned source, each chunk would have to slice the raw frames,
snap them to H3's temporal grid, sample the same slice at 2 FPS for Qwen,
VAE-encode the slice, and replace the corresponding `minimax_refs` video block.
If it has synchronized audio, that audio must be sliced and encoded on the same
timeline. For identity, appearance, general motion, camera, or style references,
the complete reference video should normally remain available to every chunk
instead of being timeline-sliced.

None of this is needed for continuation from the immediately preceding
generated chunk; direct latent-tail guides are the more accurate path.

## Stock sampler delegation and output assembly

Every planned chunk is placed in a temporary latent dictionary and passed to:

```python
SamplerCustomAdvanced.execute(...)
```

This preserves ComfyUI's normal sampler, sigma, callback, preview, model
loading, offloading, dtype, and backend behavior. The custom node does not
implement a separate diffusion sampler.

Chunks run strictly in sequence. The next stock sampler call is not created
until the previous call has finished and its final video/audio latent tail has
been cloned into native H3 continuation conditioning. A dedicated tqdm line
shows `Chunk N/total` above the stock sampler's normal per-step progress.

`debug_stop_chunk` truncates the execution plan only after the complete global
prompt timeline has been calculated. This means a diagnostic partial result
uses exactly the same prompt intervals and continuation data as those chunks
would use during a full run. The default value is zero and does not truncate.

Both stock outputs are collected:

```text
output
denoised_output
```

Continuation prefixes are trimmed from both, and their video and audio streams
are concatenated independently. The final tensors are wrapped in a new
`NestedTensor`. Their shapes match the original upstream H3 AV latent exactly.

Non-H3 or non-nested latents fall through to the stock sampler unchanged.

## Accumulated live preview

`MiniMax H3 Unlimited Preview` is a separate model-patch node. Its MODEL output
must feed the guider passed to `SamplerCustomAdvanced-Unlimited`. Keeping the
preview at the model boundary follows ComfyUI's existing outer-sampler wrapper
contract and avoids adding UI or transport behavior to the sampler itself.

At the start of one Unlimited execution, `nodes.py` finds wrappers registered
under the private `minimax_h3_unlimited_preview` key and opens a short-lived
preview session. Before each stock sampler call it supplies:

```text
chunk index and count
sampled global frame range
assembled-output frame range
number of carried video latent steps to hide
```

The wrapper observes the normal stock sampler callback. It restores the H3
video stream from either a nested AV value or ComfyUI's flattened sampler
representation, removes the configured carried latent steps from later
previews, and decodes the remaining latent positions. The original callback
still runs, so ComfyUI's normal progress and preview behavior are preserved.

Two decode modes are available:

- `tiny_vae: none` applies the H3 latent-format RGB factors in one tensor
  operation. This is the cheapest path but only an approximation.
- A compatible 24-channel flat decoder such as `taeh3.safetensors` is built
  from its checkpoint-defined layer indices and decodes one latent position at
  a time. This limits peak activation memory while producing a more useful RGB
  preview. The decoder weights exist only for the current Unlimited execution
  and are released when it finishes.

`max_resolution: 0` keeps the decoder's native output size. Positive values cap
the longest preview side. The wrapper sends encoded resolution, FPS, rolling
step duration, ETA inputs, sigma schedule, latent-change magnitude, and
step-time samples to the local widget. The browser draws the two telemetry
graphs for the active chunk while retaining completed chunk media in its
playlist. It also measures total elapsed sampling time from the reset event,
updates it once per second, and freezes the final value on completion. Decoder
selection is intentionally not given a separate UI row.

Every sampling event carries the execution id, chunk count, sigma schedule, and
backend elapsed time. The backend keeps a bounded in-memory snapshot containing
the current execution metadata and only the latest encoded frame group for each
chunk. A local read-only endpoint returns that snapshot to a newly created
widget, so browser refresh restores all prior chunks immediately instead of
waiting for and adopting only the next live event. No latent or decoded tensor
is retained, and only the eight most recently used preview-node ids are kept.

For the active chunk, the browser keeps each received step frame group until the
next chunk starts. Hovering either graph maps the pointer's horizontal position
to a sampling boundary, draws the same vertical marker on both graphs, updates
their values, pauses the chunk playlist, and loops that step's frame group.
Leaving the graph restarts the playlist from the first available chunk.
Intermediate previews may be absent if the bounded encoder intentionally drops
an outdated job while a newer step is waiting.

Each chunk update is an atomic object containing an ordered array of static WebP
frames and their individual durations. The browser snapshots that object when
it enters a chunk and uses one explicit `(chunk index, frame index)` cursor. It
must finish the snapshot's final frame before advancing to the next available
chunk index. If denoising replaces a chunk while it is playing, the new group is
used only on the next playlist pass. There is no browser-managed timer competing
with an independently looping animated image, so frames from chunk 4 cannot be
inserted into chunk 2 or chunk 3.

`frame_stride` chooses every Nth H3 latent position. Per-frame durations
still use H3's `(1, 4, 4, 4, 4)` pixel-frame coverage, so skipped positions do
not speed up playback. Cumulative millisecond rounding keeps the total group
duration equal to the represented pixel-frame duration at the selected `fps`.

The preview frontend records that backend/source FPS with every frame group,
then scales those stored duration values to the current `fps` widget value at
playback time. Changing the widget during an active inference immediately
cancels and reschedules the displayed frame at the new rate; it neither waits
for another preview message nor changes H3 sampling. The widget uses a one-FPS
step so its arrows provide practical live speed control, while direct numeric
entry can still specify a fractional rate.

PIL WebP compression runs on a bounded background worker. When encoding falls
behind, the queued intermediate step is replaced by the latest one instead of
blocking diffusion. Tiny-VAE and Latent2RGB decoding happen before that worker;
therefore tiny-VAE preview can slow sampling, and `frame_stride` is the direct
control for reducing that cost.

Each live event transfers only the current chunk's latest static-WebP frame
group. The browser stores one group per chunk, replaces the active group as
denoising progresses, and loops over every available chunk in numeric order. A
refresh performs one snapshot transfer containing the retained latest groups;
normal live events do not resend earlier chunks. A reset event clears stale
media at the next execution, and a completion event leaves the assembled
preview playing.

Preview loading, decoding, encoding, and event-send errors are non-fatal. An
invalid tiny decoder is disabled for that execution and falls back to
Latent2RGB. Preview state contains no network client or outbound request; it
uses ComfyUI's local websocket event path plus a read-only local snapshot route.

## LoRA scheduling finding

ComfyUI exposes hook/keyframe nodes that can schedule LoRA strength across
diffusion percentages. In principle that can apply an ordinary compatible LoRA
only during selected steps, and a serial sampler would repeat the same schedule
for every chunk.

The installed hook LoRA loader did not map the tested MiniMax H3 Turbo LoRA
format. The console printed `lora key not loaded` for the H3 blocks and reported
`0 patches attached`. A known working run through the ordinary H3-compatible
LoRA loader reported `208 patches attached`. With a six-step Turbo schedule,
zero patches means the model is effectively being sampled for only six steps
without the distillation LoRA and can produce chaotic or under-denoised output.

For the tested Turbo LoRA, use the ordinary loader path that reports the
expected attached patches and keep it active for all intended Turbo steps. Do
not diagnose such a result as a serial continuation failure until the model-load
line confirms that the LoRA actually attached.

## Current limitations

- Chunking reduces temporal DiT memory, not the memory required for one
  high-resolution frame. It cannot guarantee arbitrary spatial resolution.
- The upstream H3 latent nodes may impose their own user-interface duration
  maximum even though the sampler planner itself works with longer valid
  latents.
- Chunked denoise masks are rejected. Slicing and synchronizing nested AV masks
  needs a separate, explicitly designed path.
- New Ref2VA video media is unsupported for the reasons above. Existing image
  and audio reference presentations can be rebuilt.
- Prompt cut timing remains generative guidance. Rewriting a cut to the correct
  local timestamp does not make diffusion frame-exact in the way a deterministic
  video editor would be.
- Gemma visual direction observes 2 FPS stills plus the exact final frame and
  its own exact prior `detailed_description`, `timing_plan`, and `end_state`
  from the immediately preceding sampled chunk, but has no generated-audio
  input. It can therefore misjudge fine motion, rapid dialogue, or an action
  whose relevant beginning is no longer visible. Marker/dialogue validation is
  warning-only so prompt defects remain visible in the transcript: a usable
  Gemma description reaches H3 unchanged except for the deliberate extraction
  of legacy Gemma-only `[end state]` metadata, while a missing/empty description
  stops sampling instead of being silently replaced by a canonical static
  prompt. Quality must still be tested against the known Tila-pointing/zoom-out
  case.
- Gemma observations now run in a disposable child process. This adds Python
  startup overhead at each observation, but process exit is the only reliable
  way to reclaim llama.cpp/ggml CUDA backend state before Qwen and H3 resume;
  PyTorch's cache controls cannot release allocations owned by llama.cpp.
- Shot-aware physical chunk boundaries are not implemented. Prompt cuts are
  arbitrary integer frames, while H3 sampler windows are constrained to its
  `17k + 5` temporal grid; exact alignment requires a validated padding and
  latent-ownership scheme.
- A cut placed only a few new frames after a continuation prefix can be
  unstable. Increasing `chunk_frames` or choosing boundaries farther from cuts
  gives H3 more temporal runway without changing the requested global cut.
- Identity details invented during generation are not guaranteed to persist
  across chunks unless they are visible in the carried frames, described in the
  prompt, or fixed by reusable reference media.
- Full-reference prefix analysis sections remain present even when their
  detailed shots are outside the current chunk. They preserve label meanings,
  but may still describe the overall video rather than only the local window.

## Future experiments (not implemented)

Keep deferred ideas in this section so exploratory discussion is not mistaken
for current sampler behavior. None of the items below should be implemented
without an explicit decision and a checkpoint of the working version.

1. **Gemma-selected sparse continuity keyframes.** Decode the completed
   previous chunk once, let Gemma compare its frame-indexed visual history with
   the original reference images, and ask it to identify sharp frames that best
   preserve generated details such as hair ornaments, costume changes, props,
   character arrangement, pose, and environment state. The current Gemma path
   observes only 2 FPS stills, so this experiment needs an all-frame or
   carefully designed indexed-contact-sheet presentation. Gemma should return
   validated frame numbers and reasons, with a configurable small keyframe
   budget.

   Native `minimax_keyframes` remain target-timeline anchors. Therefore only a
   selected source frame that lies inside the next chunk's physical context can
   be placed at its truthful global/local time. Each sparse frame must be
   decoded and VAE-encoded as its own one-frame keyframe entry; combining sparse
   images into one video latent would falsely describe them as consecutive.
   The final eligible overlap frame should normally remain a mandatory boundary
   anchor. A useful UI could be `keyframe_mode: contiguous | gemma_sparse` plus
   `gemma_keyframe_count`.

2. **Gemma-selected dynamic appearance reference.** If Gemma finds an important
   continuity frame earlier than the physical context overlap, do not misplace it as a
   timeline keyframe. Test exposing one such frame as a new image reference and
   adding its persistent details to Gemma's continuation prose instead. This
   would be a semantic/appearance reference, not a statement that the old frame
   occurs again in the new target interval. Its Qwen and DiT token/VRAM cost
   must be measured.

3. **Shot-boundary-aware chunk planning.** Prefer whole-shot sampler calls when
   a shot fits, and split only overlong shots into balanced segments so a
   one-to-five-frame remainder never becomes a nearly empty final call. The
   detailed design and unresolved H3-grid ownership problem are documented in
   `Planned experiment: align chunks to shot boundaries` above.

4. **Audio-aware Gemma continuity.** Give Gemma a representation of generated
   audio so it can judge completed dialogue, interrupted speech, sound effects,
   and synchronized action rather than relying on visuals and text alone. This
   requires an explicit audio decode/presentation design and careful model
   swapping; sparse visual keyframes must not silently imply sparse audio
   anchors.

5. **Longer semantic state memory.** Let Gemma maintain a compact validated
   state summary across the complete current shot, rather than inferring all
   persistent state from only the immediately previous chunk. Track invented
   appearance details, object state, spatial relationships, completed actions,
   dialogue progress, and camera state without feeding the complete latent
   history to H3.

6. **Action-level micro-timing fallback.** If semantic continuation prose still
   causes H3 to compress or replay a long action, test having Gemma convert only
   the current slice into a few short local timed beats. Keep this optional: the
   earlier hand-written global/local timecode formats were retired because H3
   frequently ignored endpoints or replayed complete shot descriptions.

7. **Continuation-mechanism quality/VRAM matrix.** Use the independent numeric
   controls to compare overlap-only, contiguous keyframes,
   Video1-plus-five-frame-visual-boundary, and combined modes at identical seeds, chunk
   sizes, resolutions, and prompts.
   Record continuity quality, new-output frames per call, peak VRAM, and total
   runtime before choosing stronger defaults.

8. **Model-generic Endless Sampler.** Investigate reliable runtime detection of
   H3, LTX, Wan, and other model families, then expose only continuation methods
   that their conditioning and temporal latent formats actually support. Do not
   generalize H3's `17k + 5` grid, AV nesting, keyframes, or Ref2VA conventions
   to another architecture without model-specific adapters and tests.

9. **Preview saving.** Preserve the sampler preview's timeline data with the
   finished video so a saved render can later reopen in the same interactive
   player with colored chunk ranges, shot brackets, frame/shot/chunk labels,
   prompts, and available timing graphs. Do not use VHS Video Combine's
   `meta_batch` input for this: `VHS_BatchManager` controls batched execution and
   the persistent FFmpeg process; it is not a general metadata payload.

   Proposed design: have the sampler output a versioned `H3_TIMELINE` object;
   pass the completed VHS filename and that object to a downstream metadata
   node; stream-copy/remux the video with compact JSON in a dedicated
   `minimax_h3_timeline` container tag; and always write an adjacent
   `.h3timeline.json` sidecar because container metadata can be stripped. Add a
   finished-video player/load node that reads the embedded tag, falls back to
   the sidecar, and reuses the preview timeline UI. Store structural metadata,
   prompts, and small numeric series—not preview images, since the finished
   video already contains the frames. Loading a saved video should preserve
   seeking, play/pause, arrow-key frame stepping, chunk colors, shot brackets,
   and browser-refresh recovery.

10. **Gemma visual-memory catalog and on-demand H3 references.** After each
    completed chunk, have Gemma update a persistent structured catalog for the
    exact sparse observation JPEGs already saved for the last render. Catalog
    entries should identify the image/source frame, shot and chunk, visible
    characters and objects, appearance details, poses, costumes, props,
    environment state, possible future continuity value, confidence, and any
    older entry they supersede. Feed the compact catalog back to Gemma when
    planning every later chunk so it can remember useful visual evidence from
    more than only the immediately preceding chunk.

    Let Gemma return validated structured retrieval requests rather than
    informal prose, for example an image ID plus the continuity reason. The
    sampler can load a small bounded number of requested saved images, append
    them to that chunk's Qwen/H3 image-reference conditioning, assign the next
    available `<Picture N>` identifiers, and make those mappings explicit in
    the generated H3 prompt. Original user references must retain priority.
    Prefer targeted character/detail crops where practical because a full old
    frame may accidentally reinforce its obsolete camera, background, or pose.
    Measure the added Qwen, DiT-token, VRAM, and consistency costs.

    Compare a one-pass flow, where Gemma selects references from catalog text
    and writes the final prompt, with a two-pass flow, where it first requests
    candidates and then sees the loaded images before finalizing the prompt.
    The latter is more reliable but adds another Gemma invocation. This remains
    a future experiment: the current implementation retains all sparse 2-FPS
    plus exact-final-frame observations on disk, but still shows Gemma only the
    immediately previous chunk's selected observations during live planning.

11. **Gemma hierarchical chunk and shot retry director.** Treat each sampled
    chunk as provisional until Gemma has inspected its retained generated
    frames against the exact frame interval it owns, the complete source-shot
    intent, immutable dialogue/cut markers, and the preceding accepted visual
    context. Gemma should return a strictly validated verdict such as
    `accept`, `retry`, or `uncertain`, a score/confidence, completed intent,
    missing or incorrect events, continuity/camera failures, and a rewritten
    chunk-local description. A retry must use a documented deterministic
    alternate noise/seed as well as the revised prompt; repeating identical
    inputs would normally reproduce the same candidate. Keep a small bounded
    candidate set and accept the best scored valid result rather than allowing
    unbounded retries.

    Chunk acceptance alone must not finalize a longer source shot. Once the
    final chunk belonging to a shot has completed, let Gemma inspect the
    complete sequence of accepted retained frames for that shot and compare it
    with the source-shot intent. A failed shot-level review must roll back to a
    deliberate checkpoint at the shot start, invalidate every chunk in that
    shot and all downstream continuation chunks, then regenerate the shot with
    the accumulated factual feedback. This is necessary because later chunks
    condition on the old chunk's latent tail, Video1 reference, and Gemma
    observations; replacing a chunk after successors have been accepted cannot
    be safely spliced into the chain.

    Store Gemma's candidate prompts, evidence, verdicts, and successful/failed
    phrasing in a reset-per-render, inspectable retry journal such as
    `${TMPDIR}/comfyui-minimax-h3-unlimited/last_gemma_retry_memory.md`.
    Do not let runtime Gemma mutate this repository's `memory.md`: that file is
    the human-maintained development handoff. The review remains visual-only
    until an explicit audio-observation design exists, and existing code must
    continue to enforce dialogue/cut invariants independently of Gemma's
    judgment.

12. **Two-stage spatial-resolution sampling.** Experiment with sampling the
    early/high-noise portion of a joint H3 video/audio latent at a lower video
    resolution, spatially upscaling only the partially denoised video latent at
    a selected sigma, adding appropriately distributed high-frequency noise,
    then completing the remaining sigmas at the requested resolution while
    carrying audio state forward. This needs a custom two-call/sigma-split
    sampler rather than the stock single-shape sampler path, plus tests for H3
    spatial-grid validity, prompt/AV conditioning continuity, deterministic
    noise handling, and image quality.

    Treat it as a speed/quality experiment, not an OOM solution: once phase two
    runs one full-resolution H3 forward pass, its attention and activation peak
    is close to a normal full-resolution step. A resolution that cannot fit one
    ordinary H3 step will still OOM during phase two. Naively interpolating the
    low-resolution latent also cannot recreate the native full-resolution
    diffusion trajectory, so it may cause missing detail, texture instability,
    or temporal artifacts.

## Verification performed

The implementation was checked with the Python environment used by the running
ComfyUI installation.

Completed checks include:

- Python compilation of `__init__.py`, `nodes.py`, and `preview.py`;
- loading through ComfyUI's real `load_custom_node` path;
- schema registration with the expected node id and all new inputs;
- a planner matrix covering 212 H3 durations and eight chunk settings;
- exact reconstructed video/audio shapes for mocked multi-chunk sampling;
- mocked execution of configured-tail guide + overlap, five-frame guide +
  overlap, fully off, bounded native video, Qwen-full-history, and combined
  configurations with exact final AV shapes;
- bounded native references appearing in both Qwen presentation and the DiT
  `minimax_refs` list, while full-history references appear only in Qwen;
- 2 FPS decoded-video frame selection and half-second timestamps matching the
  installed stock H3 Ref2VA implementation;
- non-mutation and final restoration of the original Ref2VA reference list;
- targeted Qwen/VAE unload ordering before every stock sampler invocation;
- dynamic Qwen video downscaling with preserved aspect ratio, 32-pixel canvas
  alignment, correct timestamps, and unchanged DiT reference dimensions;
- native continuation insertion into subject definitions, combined summary
  task types, retention analysis, and the local first-shot instruction;
- native keyframe lengths of 5, 22, 39, 56, and 107 frames smaller than their
  corresponding chunk, with exact reconstructed output sizes and synchronized
  audio conditioning;
- mocked `context_keyframes=22` serial execution proving that later chunks receive
  the previous completed seven-step video tail and matching audio tail;
- restoration of the guider's original conditioning after sampling;
- serial proof with the default five-frame context that chunk two receives
  chunk one's completed two-step video tail and synchronized audio tail before
  its sampler call starts;
- `debug_stop_chunk=2` partial video/audio output sizes and unchanged prompt
  timing against the complete global plan;
- remapping of global first/last-frame keyframes;
- replacement of `cross_attn` and `minimax_token_tags` for every chunk;
- official base-prompt and full-reference description field parsing;
- sparse mid-shot continuation retaining its concrete shot sentence while
  rewriting a leading `camera cut to` as an already-established continuing
  view;
- conversion of global frame 100 to local frame 50 / `00:02.083` at 24 FPS;
- strict rejection of malformed or non-increasing shot markers;
- reconstruction of image/audio Ref2VA presentation order;
- clear rejection of video Ref2VA conditioning from an IMAGE-only input;
- registration of `MiniMaxH3UnlimitedPreview` and its web directory through
  ComfyUI's real custom-node loader;
- CPU loading and decoding with the installed 24-channel
  `taeh3.safetensors` checkpoint;
- Latent2RGB wrapper callback delivery and reset/chunk/complete event order;
- preview sample-start/progress telemetry, native-resolution dimensions, and
  active decoder reporting;
- JavaScript syntax and lifecycle review for the elapsed timer, completion
  freeze, removal cleanup, and removal of the separate previewer status row;
- JavaScript lifecycle review for snapshot restoration after refresh and
  interactive graph-step preview selection with a synchronized vertical marker;
- static WebP frame-group creation, sorted cache restoration, and exact
  124-frame/119-frame playback durations at 24 FPS.
- Gemma worker startup through ComfyUI's exact Python executable, ComfyUI-root
  import-path injection, result-envelope parsing, nonzero worker status, and
  propagation of a worker-side observation error back to the sampler process.
- exact Gemma debug-fixture request, JPEG, rendered-prompt, response, and replay
  serialization without loading the real model;
- independent continuation-control validation for disabled/enabled keyframe,
  retained warm-start, and Video1 paths;
- native numeric widget schemas restored to `min=5, step=17`, each paired with
  a Boolean enable whose runtime value is `int(enable) * frames`;
- retained warm-start placement after the context/packing trim boundary for
  disabled context, five-frame context, and 22-frame context, proving that the
  copied latent positions survive output assembly.
- Gemma capture schema v2 serialization with exact global frame filenames,
  complete chunk/shot facts, current editable prompt templates, and response
  replay data;
- selection of the vendored official Ref2VA guide without also injecting the
  base-mode guide, plus byte-for-byte SHA-256 agreement between all three
  vendored files and their upstream sources on 2026-08-21;
- multi-shot marker/timestamp and dialogue diagnostics for Gemma's complete
  chunk-local description, preserving usable output unchanged while recording
  warning(s); and
- a deterministic one-retry marker-correction path: an initial global-timecode
  marker response receives the literal local tokens, returns a second complete
  Gemma JSON response, preserves both attempts in payload/capture/replay, and
  submits only the corrected model-authored description to H3; and
- retained-output-only visual observation: a 39-frame previous physical window
  with a five-frame trimmed prefix yields exact chronological frames
  `5, 17, 29, 38`, rather than showing Gemma the trimmed prefix; and
- no-context Video1 boundary layout: the exact final two video-latent tokens
  are cloned as a five-frame visual keyframe clip across discarded local frames
  0-4, while enabling ordinary context keyframes suppresses that special
  boundary clip. The earlier one-image local-frame-4 experiment remains recorded
  as a failed visual-continuity test above.

Manual GPU/log observations also established:

- the original serial implementation completed a long multi-chunk render with
  materially lower peak VRAM;
- the failed global approach changed windows during each sampler step and made
  artifacts at its exact overlap/ownership boundaries;
- the restored implementation runs a complete sigma schedule per chunk;
- later serial conditioning receives the previous chunk's completed final two
  video latent positions rather than an unfinished per-step overlap;
- the tested scheduled Hook LoRA path attached zero H3 patches, while the known
  working ordinary loader attached 208.
- in the failing 56-frame + 22-frame-guide run, chunk 1 peaked at 13,848.2 MiB
  physical VRAM with 2,088.9 MiB free. After the first Gemma handoff,
  `Llama.close()` released the 6,637.69 MiB model buffer, 2,688 MiB of KV
  buffers, and its MTMD projector, but the next pre-sampler baseline was about
  518 MiB higher. That closely matched llama.cpp's reported 527 MiB CUDA
  compute buffer. PyTorch itself was down to 21.3 MiB active and 128 MiB
  reserved, and ComfyUI's managed-model registry was empty, so another
  `torch.cuda.empty_cache()` could not recover the missing memory.
- the same failing chunk 2 reached 15,904.2 MiB physical VRAM with only 32.9 MiB
  free and failed in the eager quantized `int8_linear` temporary
  `torch.cat`. This is a peak-workspace failure, not evidence that the complete
  H3 model, latent, or attention state was intentionally retained between
  chunks.
- llama.cpp's public `llama_backend_free()` is documented as an end-of-program
  backend call and does not provide a supported per-observation CUDA-pool flush.
  Resetting a CUDA device inside ComfyUI would also invalidate PyTorch and other
  live CUDA owners. Gemma therefore remains local Python/llama.cpp but is now
  process-isolated so worker exit authoritatively returns all of its CUDA memory.

A full MiniMax H3 GPU render was not run as part of the lightweight automated
verification. The mocked path exercised chunk planning, prompt replacement,
conditioning mutation, continuation guide construction, trimming, and final
shape assembly without loading the large model.

## Documentation used

- MiniMax-maintained H3 prompt-writing skill now vendored and fed directly to
  Gemma at runtime:
  <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md>
- Direct MiniMax base-mode dependency referenced by that skill:
  <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt>
- Direct MiniMax Ref2VA dependency referenced by that skill:
  <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt>
- MiniMax H3 team Reddit AMA preserved as an informal source of team comments
  and future research leads; individual claims should be verified against code
  or official documentation before becoming implementation assumptions:
  <https://www.reddit.com/r/StableDiffusion/comments/1vh9rtw/ama_minimax_h3_team_ask_us_anything_about_our/>
- MiniMax base video prompt guide:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md>
- MiniMax full-reference prompt guide:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md>
- MiniMax CLI H3 storyboard/reference template used when designing the
  `storyboard` continuation format:
  <https://github.com/MiniMax-AI/cli/blob/main/skill/h3-video/references/h3-video.md>
- Naxdy's third-party **Minimax H3 Prompt Enhancer** system prompt:
  <https://gist.github.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace>. It is
  not official MiniMax documentation and does not establish model behavior, but
  it is a useful design reference: it treats reference order as semantic, keeps
  `<Video N>` and `<Audio N>` ordinals independent, uses the six-section Ref2VA
  brief, and recommends short sequential timed beats with observable end
  states. The current sampler behavior was not changed from this source; its
  beat guidance remains relevant to the optional micro-timing experiment.
- deAPI's third-party T2VA-oriented prompting article:
  <https://deapi.ai/blog/minimax-h3-prompting-guide-how-to-write-structured-prompts-for-text-to-video>.
  It is a practical restatement of the structured fields, increasing
  `MM:SS.mmm` cut times, concrete observable language, and camera-motion prose
  already covered by the official guides. Its useful extra heuristic is to
  budget dialogue at roughly 2.5 words per second and, where lip-sync quality
  matters, prefer one speaker per shot. It does not document serial chunk
  continuation, Ref2VA latent handling, or H3 internals, so its published model
  specifications and workflow claims are not treated as implementation facts.
- Official Gemma 4 12B QAT Q4 model and projector used by the local continuity
  director:
  <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf>
- `llama-cpp-python`, whose 0.3.35 generic MTMD vision handler is required for
  the Gemma integration:
  <https://github.com/abetlen/llama-cpp-python>
- llama.cpp's CUDA backend source, consulted for the lifetime of CUDA/VMM buffer
  pools and the absence of a safe public per-context pool reset:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu>
- NVIDIA Unified Memory reference consulted for the rejected idea of keeping
  active H3 attention/reference tensors in system RAM:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html>
- PyTorch CUDA memory-management notes consulted for allocator/cache behavior:
  <https://docs.pytorch.org/docs/stable/notes/cuda.html>
- Installed ComfyUI implementations inspected during development:
  `comfy_extras/nodes_custom_sampler.py`,
  `comfy_extras/nodes_minimax_h3.py`,
  `comfy/model_base.py`,
  `comfy/ldm/minimax/model.py`, and
  `comfy/ldm/minimax/vae.py`.
