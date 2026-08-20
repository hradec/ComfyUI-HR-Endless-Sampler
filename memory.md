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
- `context_frames`: the completed H3-valid tail carried into every later
  chunk, defaulting to five frames;
- `guide_overlap`: three-way selector for the configured completed-tail guide
  and overlap, a fixed five-frame guide and overlap, or fully off;
- `video_continuation`: experimental bounded native Ref2VA continuation using
  the previous synchronized `context_frames` AV tail as a new `<Audio N>` +
  `<Video N>` reference;
- `qwen_full_history`: experimental Qwen-only view of every completed frame,
  sampled at 2 FPS, without adding that history to DiT reference attention;
- automatic Gemma 4 semantic continuation: when a later chunk begins within
  an already generated source shot, it observes the previous sampler chunk and
  writes only that shot's next short H3 continuation prose;
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
  automatic Gemma continuation when a long shot crosses a chunk boundary.

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
  immutable source-action ledger, strict JSON observation validation, image
  conversion, and source-constrained H3 continuation-prose generation.
- `gemma4_prompts.txt` contains the editable runtime `[SYSTEM]` and
  `[OBSERVATION]` messages passed to Gemma. It documents its supported
  placeholders and includes the relevant H3 structured-prompt conventions.
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
`context_frames`. The experimental global-window commit remains on the separate
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

The first chunk contributes its complete latent. Every later chunk contains a
`context_frames` continuation prefix followed by new frames. Valid values are
5, 22, 39, 56, and so on. They correspond to 2, 7, 12, 17, and so on video
latent steps. `context_frames` must be smaller than the effective snapped
`chunk_frames` value.

`chunk_frames` remains the total amount sampled in one stock call, including
the repeated prefix. Increasing `context_frames` therefore reduces the new
content produced by later chunks and can increase the number of sampler calls;
it does not silently increase the temporal VRAM cap.

With the default five-frame context, two 124-frame chunks deliver:

```text
124 + (124 - 5) = 243 frames
```

In latent steps this is:

```text
37 + (37 - 2) = 72 video latent steps
```

The configured context latent steps are removed before final assembly. This
keeps the result on H3's native grid and makes the returned video latent exactly
the same shape as the upstream requested latent.

### Audio rounding at boundaries

The configured context duration normally corresponds to a fractional number of
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
context_video_steps = ((context_frames - 5) // 17) * 5 + 2
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

H3 accepts longer native continuation clips only on its `17k + 5` frame grid:
5, 22, 39, 56 frames, and so on. `context_frames` now exposes these lengths.
The planner derives the matching video latent steps and synchronized audio tail,
excludes the complete carried interval from new prompt action ownership, trims
it from both sampler outputs, and tells the preview to hide it. Longer context
may preserve more visual history, but it repeats more computation, creates more
chunks for a fixed `chunk_frames`, and can overconstrain motion or replay prior
action.

### Independent continuation experiments

Three sampler controls isolate the continuation mechanisms so GPU tests can
compare them without silently coupling their effects. Their defaults preserve
the published serial path: `guide_overlap=context_frames`,
`video_continuation=false`, and `qwen_full_history=false`.

`guide_overlap` has three modes. `context_frames` injects the configured
completed video and audio tails as native H3 keyframes, samples the same global
interval as overlap, and trims it afterward. `5 frames` performs that identical
guide + overlap process with H3's minimum five-frame tail, regardless of the
larger configured value. Both are the original continuation mechanism at
different strengths; there is no overlap-only mode.

`off` carries no prior video latent, audio latent, or global noise interval into
the next chunk and adds no continuation keyframe. H3's temporal phase still
requires every independently sampled local chunk to begin on its `17k + 5`
grid. The node therefore prepends newly allocated empty five-frame video/audio
latents with separately generated noise and discards their sampled result.
This is local packing, not previous-chunk overlap. It lets the remainder join
the single output latent at the correct temporal phase while ensuring that any
previous-result information comes only from the separately selected native
video-continuation or Qwen-history mechanism.

`video_continuation` implements the full-reference continuation experiment for
later chunks. It clones only the final `context_frames` latent positions from
the completed previous chunk. The H3 VAE decodes that bounded tail, and Qwen is
shown frames sampled at the same 2 FPS cadence used by ComfyUI's stock H3
Ref2VA node. The matching generated-audio tail is selected using cumulative
global 40 Hz endpoints rather than rounding the isolated duration; this avoids
an occasional one-step AV error for fractional durations such as 22/24 second.
The clean bounded pair is independently appended to `minimax_refs` as a native
`video_audio` block, so the DiT can attend to both continuation streams without
merging them into the noisy target latent.

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

A 1920x1088 diagnostic using `chunk_frames=56`, `context_frames=22`, native
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

The monitor is installed only while a debug execution is active and is removed
in the same `finally` cleanup that restores the guider conditioning. Non-debug
sampling therefore keeps the stock model-call path and logging volume.

Every sampler execution, including non-debug runs and partial/error exits,
also writes one compact final timing and high-water report. It separately
accumulates H3 sampling, Qwen encoding/tokenization, VAE decoding of the prior
chunk for Gemma, VAE decoding of bounded continuation context, VAE decoding of
Qwen full history, and Gemma 4's whole local handoff. It records peak ComfyUI
process RSS and system RAM use, along with sampled whole-device VRAM use and
PyTorch allocated/reserved high-water. PyTorch's native peak counter can be
reset by the per-chunk debug monitor, so the final report retains the largest
value observed after each timed phase as well. This is intended for quickly
comparing which optional continuation path consumes time or memory without
reading every debug snapshot.

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
5. uses the original source body for a new shot, and Gemma-authored semantic
   continuation prose for a shot that began in a previous chunk.

For example, at 24 FPS:

```text
global cut frame: 100
chunk global start: 50
local cut frame: 50
local cut time: 00:02.083
```

When a chunk begins in the middle of a long shot, it no longer receives the
complete source shot, a custom time range, a reference-endpoint clock, or a
synthetic `shot ends` command. Gemma observes the previously generated video
and writes only the next H3-ready continuation description. The sampler itself
keeps canonical timecodes only for genuine source cuts that physically occur in
the current chunk.

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

### Implemented default: Gemma 4 semantic continuation director

There is no `gemma4_continuity` toggle, continuation-format selector, or
Gemma GPU-layer widget. Gemma is automatic when a later chunk's new output
begins inside an existing source shot. It uses `n_gpu_layers=-1` while H3 and
Qwen are explicitly unloaded, then destroys its llama.cpp context before H3
sampling resumes. A new shot that starts exactly at a chunk output boundary is
left to its original author-written prompt because no footage of that shot has
yet been generated.

For one eligible handoff, the sampler:

1. builds an immutable action ledger from the original source shot, splitting
   only at sentence and semicolon boundaries; each item receives a stable
   `S<shot>.A<action>` identifier;
2. decodes the immediately preceding *sampled* chunk with the H3 VAE and sends
   its chronological 2 FPS stills to Gemma, rather than only the five or
   twenty-two carried guide frames;
3. supplies the source shot range, FPS, accepted action prefix, and number and
   approximate duration of the new frames that belong to this source shot;
4. receives JSON containing the ordered completed prefix, first unfinished ID,
   confidence, factual final-state observation, and a one-to-four-sentence
   `continuation_description` written in H3's concrete playback-order style;
5. validates that action IDs are a non-regressing ordered source prefix, then
   substitutes only `continuation_description` as the active shot body.

Thus H3 receives canonical source-shot markers and only the next semantic
action prose. It is no longer asked to map a complete long-shot prompt onto a
custom master clock or to infer action progress from `ONLY GENERATE THE
TIMESLICE` wording. Source cuts are still deterministic sampler calculations
and remain normal `[Shot N] At ...` H3 markers; Gemma must not emit markers,
timestamps, cuts, reference labels, or `shot ends` text in its continuation
body. The original source prompt remains the authority: Gemma is instructed not
to invent actions, camera work, dialogue, characters, objects, or outcomes.

An unavailable/mismatched `llama-cpp-python` runtime or a failed model download
stops explicitly. A malformed Gemma response is logged and falls back to the
canonical source-shot continuation for that chunk. Debug `chunk_prompts`
includes the ledger state, observation, confidence, Gemma's actual H3
continuation prose, and raw validated JSON immediately before the actual H3
prompt.

The Gemma instructions are deliberately not embedded in `gemma4.py`.
`gemma4_prompts.txt` has a `[SYSTEM]` message and an `[OBSERVATION]` template;
the latter receives source-shot number/range/FPS, immutable ledger, accepted
action IDs, and new-continuation frame/duration through documented
`{{lowercase_placeholders}}`. The file is reparsed before every Gemma handoff,
so an experimenter can edit and save it while ComfyUI is running and the next
eligible chunk will use the new wording. It validates both section headers and
every placeholder; an invalid template is reported as a recoverable observation
failure and that chunk uses the canonical source fallback. The system message
includes H3 phrasing rules because Gemma now authors the H3-facing body.

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
environment currently has 0.3.16, which is deliberately rejected with an
installation message until `requirements.txt` is installed using ComfyUI's
isolated `tools/python.sh` interpreter.

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
duration with the `chunk_frames` widget. Later calls spend part of their sampled
window on `context_frames` guide/overlap, or on the minimum local packing prefix
when guide overlap is off. The complete sampled window must still stay at or
below the requested VRAM cap.

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
`llama-cpp-python`'s in-process generic MTMD handler. The user had already
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
- Automatic Gemma continuation observes only 2 FPS stills from the immediately
  preceding sampled chunk, uses sentence/semicolon action units, and has no
  generated-audio input. It can therefore misjudge fine motion, rapid dialogue,
  or an action whose relevant beginning is no longer visible. Strict ID
  validation prevents malformed responses from corrupting the action plan, and
  a failure uses the canonical original source-shot body for that chunk. Quality
  must still be tested against the known Tila-pointing/zoom-out case.
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
- context lengths of 5, 22, 39, 56, and 107 frames with exact reconstructed
  output sizes and synchronized audio trimming;
- mocked `context_frames=22` serial execution proving that later chunks receive
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

A full MiniMax H3 GPU render was not run as part of the lightweight automated
verification. The mocked path exercised chunk planning, prompt replacement,
conditioning mutation, continuation guide construction, trimming, and final
shape assembly without loading the large model.

## Documentation used

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
  beat guidance is relevant to future Gemma action-ledger segmentation.
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
