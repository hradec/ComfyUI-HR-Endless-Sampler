# HR Endless Sampler development memory

## Goal

This custom node adds `HR Endless Sampler`, an extensible replacement for
ComfyUI's stock `SamplerCustomAdvanced`. Its purpose is to sample a long video
latent as smaller temporal chunks so the active model backend does not have to
hold the complete sequence in VRAM at once. MiniMax H3 is the currently
implemented backend; LTX support is planned.

The node keeps the stock sampler inputs and delegates every individual chunk to
the stock `SamplerCustomAdvanced`. It adds:

- `clip`: the CLIP/Qwen model used by the active backend's upstream conditioning
  node;
- `prompt`: the original backend-formatted prompt;
- `fps`: the frame rate used to convert prompt timestamps to global frames;
- `chunk_frames`: the maximum number of backend frames sampled at once;
- `video_continuation`: the current H3 backend's bounded previous AV tail
  exposed as native `<Audio N>` + `<Video N>` reference, plus its exact final
  five-frame visual boundary keyframe clip across the discarded packing prefix;
- automatic Gemma 4 chunk directing: it directs Chunk 1 from the complete
  source prompt, then observes chronological stills from each previous sampler
  chunk and writes the complete H3 description for the next local slice;
- `debug`: logs each chunk's rewritten prompt and frame ranges to the ComfyUI
  console and enables detailed VRAM snapshots around conditioning and sampling;
- an always-on final report: reports wall time for H3, Qwen, each VAE decode
  path, and Gemma 4, plus peak RAM and VRAM use for the sampler execution;
- `debug_stop_chunk`: returns after the selected 1-based serial chunk for fast
  boundary diagnostics; zero keeps the normal complete run;
- optional `images`: pixel images needed to rebuild Qwen visual conditioning;
- optional `vae`: the active backend video VAE; the current H3 implementation
  needs it for Video1 and Gemma visual directing after Chunk 1.

Native context-keyframe, retained guide-overlap, Qwen full-history, and
prompt-preview-only branches remain in source for development, but are hidden
and forced off in the released UI. Video1 continuation is hidden-and-forced on.

The sampler node id and display name are both:

```text
HREndlessSampler -> HR Endless Sampler
```

An optional model-patch node provides the accumulated live preview:

```text
HREndlessSamplerPreview -> HR Endless Sampler Preview
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
- `video_io.py` owns the versioned `HRENDLESS_TIMELINE` transport type, finished
  video/EXR export, metadata sidecars and container tags, and the persistent
  state endpoint used by the finished-video player.
- `web/finished_video_player.js` provides the graph-free finished-media player
  used by `HR Endless Sampler Save Video` and `HR Endless Sampler Load Video`.
  It reuses the preview's chunk/shot timeline language without retaining live
  sampler frame groups in the browser.
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

## Finished-video timeline save/load

`HREndlessSampler` now has an additive fourth `HRENDLESS_TIMELINE` output. It
contains only JSON-safe structural render metadata: source FPS, exact output
frame count, completed chunk ranges, source-shot ranges, and the final
Gemma-authored `detailed_description` for each chunk. The range objects are
the same objects used by the live preview, so a saved timeline cannot drift
from the interactive chunk colors, brackets, or hover prompt that were shown
during the render. Replay-restored chunks carry their saved Gemma description
into the finished timeline as well. A prompt-preview-only execution creates a
structural planned timeline but it must not be confused with a sampled result.

`HR Endless Sampler Save Video` receives decoded `IMAGE` frames plus the
optional timeline. `video/h264-mp4` uses ComfyUI's native
`VideoFromComponents`/`save_to` API, including its H.264 MP4 encoder, AAC audio
mux, 8/10-bit output, CRF, and container-metadata path; this default format has
no Video Helper Suite dependency. Other ordinary formats delegate directly to
the installed `Video Combine 🎥🅥🅗🅢` implementation. This reuses its format
discovery, FFmpeg invocation, format defaults, metadata route, and
codec-specific pixel-format/CRF handling. The node presents the union of VHS
pixel formats; `auto` does not override the selected VHS format's own default,
while an explicit `pixel_format` is forwarded as VHS's `pix_fmt` widget value.
`crf` is forwarded when the selected VHS JSON format uses it; other VHS
format-specific widgets retain their upstream defaults.

Save also accepts an optional standard ComfyUI `AUDIO` input. For ordinary
VHS video formats, it is passed straight through as VHS's `audio` argument so
VHS performs the final synchronized FFmpeg mux and returns its audio-bearing
file. The format list includes both animated image formats that Video Combine
adds itself (`image/gif`, `image/webp`) and every current entry returned by
VHS's `get_video_formats()`, followed by Endless's extra `video/exr` entry.
GIF/WebP do not have an audio container and therefore cannot carry that input.

The video encoder receives the timeline as the `hr_endless_sampler_timeline`
metadata field. Native ComfyUI writes it for H.264; VHS writes it through the
FFmpeg container metadata path for formats that enable VHS metadata. Save
always also writes an adjacent versioned JSON sidecar named
`<media>.hr_endless_sampler_timeline.json`. The sidecar is authoritative on
load because websites, editors, and remuxers can strip arbitrary MP4/MKV/WebM
metadata. The Load node prefers it and falls back to the container tag. The
new finished-media player uses a normal browser `<video>` element, so it seeks
and plays the actual final media rather than retaining every decoded image. It
keeps the colored chunk transport, shot brackets, hover prompt, play/pause,
keyboard arrow stepping, and frame/shot/chunk overlay, but deliberately omits
the live sampler's sigma and step-time graphs.

`video/exr` is an EXR image sequence rather than a video container. The node
uses ComfyUI's installed PyAV/FFmpeg `exr` encoder because it exposes every
available local option: 16-bit `half` or 32-bit `float`, and `none`, `rle`,
`zip1`, or `zip16` compression, plus its `gamma` control (default `1.0`). Each EXR is emitted from `float32` tensor
values with no normalization, gamma conversion, or `0..1` clamp. The primary
first EXR receives the same timeline string header when ComfyUI's EXR metadata
helper is available; the mandatory adjacent sidecar holds the full ordered
frame manifest. A temporary H.264 proxy is made only for the Save/Load browser
player because browsers cannot display EXR directly; the EXR sequence remains
the raw master and no proxy value is used for output data.

EXR cannot embed a soundtrack. When its optional `AUDIO` input is connected,
Save writes a sibling 32-bit float WAV from the original waveform and records its
filename in the EXR sidecar manifest. The temporary browser proxy is muxed with
the same audio on save; Load reads the WAV back and muxes it into a fresh
temporary proxy after a ComfyUI/browser restart. This preserves the EXR master
as an unclamped image sequence while keeping its soundtrack durable and
playable.

The Load node's browser UI has two explicit acquisition paths. **Browse
output…** calls `/hr_endless_sampler_video/browse_output`, a read-only directory
listing restricted by resolved-path containment to ComfyUI's output root. It
supports nested folder navigation, normal video files, standalone EXRs, and
collapses a sidecar-backed Endless EXR manifest into one sequence item while
hiding its individual frames. The picker has Name, Size, and Date sort buttons;
Date descending (newest first) is the initial ordering, while selecting the
active column again reverses its direction. **Upload video…** divides the local browser file
into 16 MiB pieces and posts them sequentially to
`/hr_endless_sampler_video/upload_chunk`. The backend validates the upload id,
extension, filename and chunk plan, assembles it atomically, chooses a
non-overwriting filename, and stores it beneath
`output/hr_endless_sampler_uploads/`. The returned output-relative path is put
into the existing serialized `video` widget so it persists with the workflow.
Queueing the Load node remains necessary only to produce downstream outputs.

Selection no longer requires a workflow run merely to populate the player.
The browser posts the chosen serialized path, FPS override, and node id to
`/hr_endless_sampler_video/load_preview`. The route runs the shared
`_load_video_payload` probe in a worker thread, stores the same player state as
normal execution for refresh recovery, and returns it directly to the
requesting node. This loads video playback, chunk colors, shot brackets, and
saved Gemma prompts immediately. Normal node execution calls that same helper
so its timeline/filename/FPS outputs cannot diverge from the immediate preview.

Save and Load both expose a **Matching videos ▾** popup. The backend
`/hr_endless_sampler_video/matching_output` endpoint resolves the requested
prefix inside `output/`, returns video/EXR-sequence matches sorted by mtime
descending, and never searches outside the prefix's containing folder. Save
uses its current serialized `filename_prefix` directly. Load starts from its
serialized `video` path and removes the final generated `_<number>_…` suffix;
the greedy prefix capture preserves earlier numeric parts of a legitimate
filename. Selecting a match immediately uses the shared preview-load route;
Load also replaces its `video` widget, while Save treats the choice as a player
preview without changing its future save prefix.

EXR browser proxies now use the same native ComfyUI H.264 encoder as normal
`video/h264-mp4` saves. They no longer require VHS merely to make a temporary
browser-playable proxy.

ComfyUI's stock MiniMax H3 VAE clamps its decoder pixels inside
`MiniMaxH3VideoVAE._finalize_pixels` before its ordinary `IMAGE` output. To
make a genuinely unclamped H3 EXR possible, Save additionally accepts the
sampler's nested H3 latent and its video VAE. On `video/exr` it temporarily
replaces only that final H3 display clamp with the same mean/std conversion
without `clamp`, calls the normal VAE device/temporal decode path, restores the
original method in `finally`, and writes the resulting raw float tensors. This
is H3-specific by design and fails clearly for another VAE; it does not claim
that raw H3 decoder RGB has automatically become scene-linear color.

The experimental masked-AV continuation briefly applied a fixed `1.055x`
linear-light pre-gain to only its first/middle/final sparse boundary keyframe
images. This was an isolation test prompted by measured same-shot drift (chunk
luma `0.1672 -> 0.1585 -> 0.1519`), but it was later removed because any
display correction in H3 conditioning can compound drift. Sparse keyframes
now use the raw decoded pixels.

The finished Save Video player treats a newly completed save as authoritative
even when the user previously opened another matching render in that same
player. Save now publishes one revision-tagged player state through both the
existing socket event and ComfyUI's normal executed-node UI payload. The
frontend deduplicates those two deliveries, invalidates any older asynchronous
manual-preview request, cache-busts the new media URL, and updates the displayed
filename. Refresh restoration recognizes the same saved-state origin.

The Save Video player also has a bottom-left `Compare with` control. It opens
the same filename-prefix matching list upward and loads the chosen render into
a muted, frame-synchronized overlay without replacing the Save node's durable
primary player state. The primary/newly saved or top `Matching videos`
selection owns the first side and audio; `Compare with` owns the other side.
A draggable wipe line starts vertical. Right-clicking the line cycles through
horizontal, diagonal-left, diagonal-right, and back to vertical. The backend
preview-load route accepts `store_state: false` for this transient comparison,
so browser refresh still restores the real primary saved video rather than the
comparison choice.

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
the completed previous chunk. It does not change physical target geometry. A
request equal to or larger than the effective chunk size is clamped to that
size: a predecessor cannot supply a longer tail than the physical chunk it
actually sampled.
The H3 VAE decodes that bounded tail, and Qwen is
shown frames sampled at the same 2 FPS cadence used by ComfyUI's stock H3
Ref2VA node. The matching generated-audio tail is selected using cumulative
global 40 Hz endpoints rather than rounding the isolated duration; this avoids
an occasional one-step AV error for fractional durations such as 22/24 second.
The clean bounded pair is independently appended to `minimax_refs` as a native
`video_audio` block, so the DiT can attend to both continuation streams without
merging them into the noisy target latent.

`video_continuation_res` is an explicit spatial-quality/VRAM control for that
DiT Video1 block. `full` preserves the cloned generated latent exactly. The
smaller presets range from the stock 1344x768 H3 reference-video canvas down to
448x256, always in 32-pixel-aligned dimensions. For a reduced preset, the node
decodes the complete bounded tail once, uses that decode for Qwen, resizes all
chronological frames in pixel space, and VAE-encodes them back into a smaller
24-channel reference latent. It rejects any encode that changes the temporal
latent length. The synchronized Audio1 latent and the full-resolution
five-frame boundary keyframe do not change. Qwen and Gemma independently keep
the stock H3 reference-video presentation canvas, so reducing the DiT reference
does not also hide details from the semantic directors. The replay fingerprint,
last-run prompt header, debug log, and final run report record the selected
preset because changing it changes H3 conditioning and invalidates cached
continuation state. Every continuation chunk also emits an always-on payload
report with separate Video1 video, Audio1, and full-resolution boundary-keyframe
tensor shapes/raw MiB; packed row counts; Video1 row reduction against its
full-resolution equivalent; and total continuation rows relative to target AV
rows. Raw latent bytes are explicitly not presented as the expected VRAM
savings because H3 expands each row into much larger per-layer attention and
activation allocations.

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
video presentations are sampled at 2 FPS and use the same canvas policy as
ComfyUI's stock H3 reference-video input: a nominal 768-pixel short edge with a
`768 * 1344` area cap, 32-pixel alignment, and no enlargement. A 1920x1088
continuation is therefore presented at 1344x768 to both Qwen and Gemma. This
does not resize the clean DiT
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

An earlier emergency optimization resized dynamic Qwen video frames by
aspect-preserving area to at most `512 * 512` before tokenization. A 1920x1088
pair became 672x384, reducing its merged vision rows from roughly 2,040 to 252.
That avoided the observed Qwen token wall, but it also made internally generated
Video1 conditioning materially lower resolution than a video supplied through
ComfyUI's stock H3 Ref2VA node. It could hide small generated continuity details
from both Qwen and Gemma. The default path now uses the stock 768-short-edge,
`768 * 1344`-area reference-video canvas instead. The separate clean video
latent still enters the DiT at the original 1920x1088 latent resolution. Debug
logging reports the actual frame count and presentation resolution. A future
explicit optimization control may restore lower presentation sizes for tight
VRAM workflows; it must never be a silent behavioral difference.

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
3. treats `[Shot 1]` as a real opening-shot cue, not generic chunk syntax. It
   writes it only if a source shot genuinely begins at physical local frame
   zero. A physical chunk that begins inside an already-running source shot
   starts with unmarked continuation prose; later real cuts retain their local
   `[Shot N] At MM:SS.mmm,` form, with the time measured from the physical
   chunk start so a cut lands after any carried guide prefix;
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
- a front-loaded current-chunk shot map, naming every global source shot with
  retained output, its exact global start, physical local start, and
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
- only the required markers for real source boundaries, repeated in an
  explicit copy-only block. Every `[Shot N]` preserves the original global
  source-shot number, while only its `At` timecode is recalculated on the
  physical chunk-local clock. A source shot gets an opening marker only when it
  truly starts at the physical chunk's first sampled frame; a middle-shot
  chunk begins as unmarked continuation prose; and
- the actual opening conditioning available to H3: keyframe prefix, bounded
  `<Video N>`/`<Audio N>`, fully denoised latent warm-start, or none.

### Implemented preproduction shot-timing director

Per-chunk visual direction can preserve continuity while still making a poor
long-shot pacing decision. A 39-frame diagnostic showed this clearly: the
second source shot lasted 81 frames, but the director assigned its first 48
percent solely to entering/running, then asked H3 to perform the harness pull,
sudden deceleration, full skid, roar, and complete stop in the next 34 frames.
The images passed into Chunk 5 showed that the tiger was still sliding, yet the
next request trusted the earlier planned `end_state` that said the stop was
complete. The interval arithmetic, source coverage, and image selection were
correct; the timing decision was not. Giving each physical chunk the complete
shot prose and an isolated percentage was insufficient for Gemma to see the
whole action as one plan.

The sampler now makes one additional **text-only Gemma preproduction request
before Chunk 1**. It receives every complete source shot that will be rendered,
the complete unchanged user prompt, and every active physical chunk's sampled
and retained output ranges. It returns a compact JSON schedule per source shot:

```json
{
  "source_shot": 2,
  "shot_start_frame": 68,
  "shot_end_frame": 148,
  "visual_beats": [
    {"start_frame": 0, "end_frame": 24, "action": "..."},
    {"start_frame": 24, "end_frame": 48, "action": "..."}
  ],
  "overlays": [
    {"start_frame": 20, "end_frame": 44, "type": "dialogue", "content": "..."}
  ]
}
```

The same preproduction JSON owns an immutable `character_name_table`, for
example `{"character_name": "Heman", "subject": "<Subject 1>"}`. Gemma
may add a row only when the original prompt explicitly maps that character name
to the existing label; it must not infer a label from image appearance,
reference order, actor name, or a generic noun. The table can be empty, aliases
may point to one subject label, and duplicate character-name rows are rejected.
It is rendered into every later chunk-director request. Gemma must preserve the
table and write each listed name in H3 descriptive prose as `Name (<Subject
N>)`, but never put the label inside `<d>...</d>` because that would alter the
immutable spoken dialogue. The table itself remains Gemma-only metadata; H3
receives it only through those natural parenthetical name bindings in the final
Gemma-authored `detailed_description`.

The normal descriptive-name binding has one essential H3 dialogue exception:
when a mapped character speaks in a direct clause, the final H3-facing clause
must use the immediate speaker token `<Subject N> (Sx) says: <d>...</d>`, not
`Name (<Subject N>) (Sx) says`. The latter can hide the Subject-plus-speaker
grammar that H3 expects and was observed in a run where Heman's Shot 5 dialogue
did not render. The response validator detects the malformed nested form and
gives Gemma one model-authored correction request; sampler code never rewrites
the dialogue itself. The same run showed that an invented phrase such as
`The camera follows the chase` can be interpreted by H3 as a new camera cut.
Gemma may still add modest camera movement, but every non-source-cut movement,
follow, pan, zoom, track, shake, or reposition must begin exactly `In a
continuous movement,` and explicitly continue the established view. It must
not invent a new angle, camera setup, framing, perspective, transition, or
cut. The actual Chunk 2 cut in that run was independently confirmed to be the
genuine source Shot 1-to-Shot 2 boundary at global frame 68; Gemma's prompt
did not request the five-headed-dragon visual seen after it.

`start_frame` and `end_frame` are source-shot-relative and end-exclusive.
`visual_beats` are the one serial timeline: validation requires each shot to
have at least one non-empty visual beat, with an exact contiguous cover from
relative frame zero through the complete source-shot duration. `overlays` are
optional non-empty dialogue, sound, or sustained-action intervals that fit
inside the shot but may overlap the visual timeline and one another. This is
intentional: a prompt that says a frightened character shields themselves while
the room collapses, a roar is heard, and they speak describes one concurrent
moment unless it supplies genuine sequencing language. The Gemma system prompt
now defaults to concurrency; it makes a serial visual progression only for an
explicit connector (`then`, `after`, `once`, `only then`, `finally`, `a moment
passes`), a causal/state dependency, or a required camera progression. Mere
description order is not sequencing. Dialogue receives a realistic overlapping
interval rather than being automatically scheduled after all visible action.

The sampler attaches the known global source boundary to a validated schedule;
Gemma does not echo it, because an inclusive/exclusive endpoint echo caused an
otherwise valid real schedule to be rejected. Gemma sometimes likewise writes
the final source-relative frame index where the JSON contract wants the
exclusive endpoint, while still using matching half-open boundaries between all
earlier visual beats. The validator accepts only that unambiguous final
one-frame spelling and extends the same final Gemma-authored visual action
through the known final source frame; it does not invent or replace action
text. A malformed first response receives one full Gemma correction turn
containing the error and literal schema; sampler code never fills missing
visual beats or writes a synthetic schedule. If the correction still fails,
sampling stops before Chunk 1 because there is no truthful algorithmic timing
fallback.

The validated schedule is Gemma-only data. Before every later visual directing
call, the sampler renders only the complete schedule(s) relevant to that
chunk's target source shot(s) and supplies them next to the ordinary
source-relative timing contract. The same text now begins with a derived
**mandatory current-slice beat coverage** section: it intersects every
Gemma-authored visual beat and overlay with the retained output frame interval,
labels each as `S#.V#` or `S#.O#`, and says whether it begins or continues in
this chunk. Gemma must explicitly include every listed action in its H3-facing
description, even when its start is only a few frames before the chunk ends; it
may defer only the later outcome outside the slice. Every response now includes
a `coverage` JSON array with one status (`begins`, `continues`, or `completes`)
and one exact description-evidence phrase for each current ID. `deferred` is
invalid for an ID in the current slice. A dialogue overlay additionally must
include its exact `<d>...</d>` line in the H3 description. Failure triggers one
complete Gemma-authored **chunk-contract correction** turn; both raw JSON
attempts and the correction are logged, and no algorithmic prompt replacement
is ever used.

This was added after two real pacing failures: first, a Chunk 1 request
contained Shot 1's planned camera-arc beat at frames 32-49, while Gemma
incorrectly deferred frames 32-67 and omitted the arc from the final 0-38
prompt. Later, Shot 6 correctly assigned global frames 379-412 to a collapse,
roar, and the start of Tila's dialogue, but Gemma marked the dialogue deferred
and waited for the one remaining Shot 6 frame in Chunk 13. The assignment math
was correct; the serial-only schedule encouraged Gemma to treat speech as an
after-action. The overlay model and correction contract address that failure.
H3 never receives the JSON, beat labels, or planning prose directly; it still
receives only the final Gemma-authored local `detailed_description`. The visual
director uses the schedule to keep actions on pace, but chronological rendered
stills remain authoritative when H3 has drifted. It should identify the drift
and continue the immediate unfinished beat rather than replaying a completed
beat or cramming every remaining outcome into the current slice.

The preproduction pass reuses the existing process-isolated full-GPU Gemma
worker and the reviewed vendored H3 rule set. It uses a 2048-token JSON budget
(ordinary local prompt direction remains at 1024), is included in the existing
Gemma timing total, and has no VAE decode/observation images. H3, Qwen, and the
video VAE are explicitly unloaded before it runs. The temp
`last_gemma_chunk_prompts.txt` capture now starts with the exact preproduction
system prompt, request, raw JSON attempts/correction, and validated readable
schedule, followed by the existing per-chunk transcripts. `gemma4_prompts.txt`
therefore owns four editable sections: `[PREPRODUCTION_SYSTEM]`,
`[PREPRODUCTION]`, `[SYSTEM]`, and `[OBSERVATION]`.

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
the worker gives Gemma an append-only correction turn. That turn shows the
validation errors and exact required global-label/chunk-local-time tokens while
retaining the first JSON as assistant context; Gemma must return another
complete JSON object itself. Structural marker failures can receive up to ten
focused repairs, while creative coverage retains the one-rewrite policy. No
description text is patched or generated by sampler code. Every raw JSON
attempt and correction request is kept in the capture/transcript, and a console
warning identifies the initial marker failure even when correction succeeds.
The sampler must never replace a usable Gemma output with an algorithmically
generated source-prompt fallback. A
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

The context was initially raised from 8192 to 16384 for the official guide,
full source prompt, visual tokens, and response. That first configuration used
the default F16 K/V cache: its real llama.cpp log showed 2560 MiB at 8192 cells,
or roughly 5 GiB at 16K before the 6638 MiB model and 527 MiB compute buffer.
The resulting brief 20K test aborted during full-size SWA initialization. This
historical F16 result was superseded on 2026-08-29 by the explicit 32K Q8_0
K/V, non-full-SWA configuration documented below; it must not be used as an
upper bound for the current runtime.

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
historical captured 672x384 observations already fit below the old ceiling, while
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

### Mid-shot marker correction

An August 2026 prompt review identified that every chunk-local description was
being prefixed with `[Shot 1]`, including chunks whose first retained frame lay
inside an existing source shot. Although the marker was intended merely as a
local index, it is documented H3 shot-opening syntax and could tell MiniMax to
restart the shot at each sampler handoff. The marker contract now represents
actual source boundaries instead: the first source shot receives `[Shot 1]`
only when its start equals the physical sampled-window start. If that source
shot started earlier, Gemma must begin with ordinary continuation prose and no
marker. If carried prefix frames contain the end of a preceding source shot and
a new source shot starts later in the physical window, the new cut remains
`[Shot 2] At MM:SS.mmm,`; the unmarked opening continuation is its implicit
local Shot 1. Marker validation counts and compares only explicitly required
tokens, so it also catches a spurious model-authored `[Shot 1]`. The same rule
is applied to the deterministic `prompt_preview_only` planner and to the
Video1 retention wording, which previously mentioned `[Shot 1]` outside the
description.

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

`debug_start_chunk` is the companion replay control. A nonzero value activates
the disposable last-run cache under ComfyUI's temp directory. If no compatible
cache exists, the requested run deliberately begins at Chunk 1 and records the
complete source AV latent, fixed full-video/audio noise, every synthetic-prefix
noise tensor, every completed chunk's sampled AV tail and trimmed outputs, and
the prior Gemma description/timing/end-state. Later runs with the same latent
geometry, continuation settings, and physical chunk plan restore that exact
state through the requested predecessor and sample only the requested suffix.
This makes prompt-directed tests at a later chunk comparable without changing
their noise or Video1/keyframe boundary. The cached Gemma preproduction plan is
reused only while the main source-prompt hash is unchanged. An edited main
prompt retains the physical replay state but deliberately invalidates its old
Gemma production plan and stale predecessor Gemma text, then produces a new
plan and clean KV snapshot from the edited source before Chunk 1. Because each
sampler execution resets the clean Gemma KV snapshot, an unchanged-prompt
replay rebuilds that snapshot from its restored timing plan without asking
Gemma to plan the source shots again. With `debug_start_chunk` at zero, a
compatible interrupted ordinary render resumes automatically; use the live
Preview's `clear` control when the next execution must deliberately restart at
Chunk 1. Cached earlier assembled output stays in CPU RAM while the new suffix
samples and is moved back to the normal latent device only for final output
assembly.

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

`HR Endless Sampler Preview` is a separate model-patch node. Its MODEL output
must feed the guider passed to `HR Endless Sampler`. Keeping the
preview at the model boundary follows ComfyUI's existing outer-sampler wrapper
contract and avoids adding UI or transport behavior to the sampler itself.

At the start of one Unlimited execution, `nodes.py` finds wrappers registered
under the private `hr_endless_sampler_preview` key and opens a short-lived
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

The chunk-color transport bar keeps its original click/drag help in the native
hover tooltip. When the pointer is over one colored chunk segment, the same
tooltip appends that chunk's human number and the exact Gemma-authored final
`detailed_description` sent to H3. The sampler sends this metadata immediately
after Gemma directs a chunk and before its DiT call begins, rather than waiting
for the first encoded preview image. The preview cache folds it into the reset
timeline state, so a browser refresh retains the descriptions for completed
and currently sampling chunks. Chunks not directed yet explicitly say that
their Gemma direction is still pending.

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

- **Gemma JSON skeleton prompting.** Test supplying a compact, complete empty
  JSON object for the global production bible, each per-shot plan, and chunk
  response so Gemma fills required fields rather than recalling their schema.
  Keep immutable dialogue as source input rather than prefilled prose. Measure
  whether this reduces missing fields, malformed JSON, and correction turns
  without making the response longer or anchoring Gemma to bad placeholder
  content.

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
   anchors. In particular, attach the synchronized audio from the immediately
   previous generated chunk beside its chronological observation frames and
   prior Gemma description. Ask Gemma to determine, for every required speech
   line, whether it never started, is currently in progress, finished, or was
   cut off, and carry that factual dialogue state into the next chunk prompt.
   Preserve the exact audio time range and its relationship to global/chunk
   frames so Gemma does not mistake silence at one sparse visual observation
   for evidence about the whole chunk.

   Reuse the same audio evidence in the planned chunk/shot review director.
   A missing, prematurely cut, repeated, wrong-speaker, or badly synchronized
   line should be able to fail an otherwise visually acceptable candidate and
   trigger the future bounded chunk-redo path with a revised prompt. This is a
   TODO only; the current Gemma observation and retry evaluation remain visual.

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

9. **Preview saving — implemented.** The sampler now outputs a versioned
   `HRENDLESS_TIMELINE` object, and the Save/Load nodes persist it in the
   `hr_endless_sampler_timeline` container field plus a mandatory
   `<media>.hr_endless_sampler_timeline.json` sidecar. This intentionally does
   **not** use VHS Video Combine's `meta_batch`: `VHS_BatchManager` controls
   batch execution and FFmpeg-process lifetime, not arbitrary metadata.

   The finished-media player reuses the preview's structural UI—colored chunks,
   shot brackets, hover prompt, seeking, play/pause, and arrow-key stepping—on
   the actual saved video. It intentionally does not save the live sampler's
   WebP preview frames or the two sampler graphs: the final media is the source
   of pixels, and the graph samples are not meaningful after a completed
   render. Browser-refresh state is retained server-side for the most recent
   Save/Load nodes just like the live preview. The currently saved timeline
   contains structural metadata and prompts, not the optional live telemetry
   series. Extending the format with compact final timing/memory telemetry is a
   future additive schema decision, not an excuse to reintroduce preview frame
   storage. The Save Video player also exposes a browser-side `Download video`
   button for GitHub issue #13. It stays disabled until the player has a saved
   media URL, downloads the exact currently displayed video (including a
   selected matching render), and leaves ComfyUI's output copy untouched. For
   an EXR sequence it downloads the playable MP4 proxy shown by the player,
   rather than misleadingly downloading only the first EXR frame.

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
    `${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_retry_memory.md`.
    Do not let runtime Gemma mutate this repository's `memory.md`: that file is
    the human-maintained development handoff. The review remains visual-only
    until the audio-aware continuity experiment above supplies synchronized
    previous-chunk audio and validated dialogue state. Once implemented, audio
    evidence must participate in both chunk and whole-shot acceptance; existing
    code must continue to enforce dialogue/cut invariants independently of
    Gemma's judgment.

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

13. **Explicit Qwen/Gemma presentation-resolution optimization.** The
    default Qwen and Gemma presentation of generated Video1 frames now matches
    ComfyUI's stock H3 reference-video canvas (768-pixel nominal short edge,
    `768 * 1344` area cap) so internal continuation does not silently lose more
    small details than an externally supplied reference video. Later, test an
    opt-in semantic-presentation resolution control for memory-constrained
    workflows. This is separate from the implemented `video_continuation_res`,
    which reduces only H3's clean DiT Video1 block. Keep temporal
    sampling at 2 FPS, preserve aspect ratio/32-pixel alignment, report the
    actual presentation dimensions, and do not silently couple it to the DiT
    `minimax_refs` resolution.
    Compare prompt accuracy, tiny generated-detail retention, Qwen token count,
    encoding time, and Chunk 2+ VRAM/OOM behavior at stock, 768, and legacy
    `512 * 512`-area presentation budgets.

14. **Feathered H3 audio continuation — initial optional mode implemented.** The
    new `Masked AV overlap (experimental)` method implements the latent copy,
    nested AV mask, audio feather, and authoritative latent-tail replacement
    described below, while allowing every valid H3 overlap (`5, 22, 39, ...`).
    The exact 39/90/141-frame shared-clock variant remains a useful comparison
    because it avoids fractional 24-FPS/40-Hz endpoints.

    The design came from the continuation
    method described in the Reddit discussion at
    <https://www.reddit.com/r/StableDiffusion/comments/1w25d7g/comment/p6td943/>
    and implemented conceptually by
    <https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef>. At 24 FPS,
    use a 39-frame physical AV context because it is both a valid H3 video run
    and exactly 65 H3 audio-latent ticks at 40 Hz; later exact shared boundaries
    are 90, 141, 192, and so on. Copy the preceding completed AV latent tail
    directly into the next target prefix. Preserve the video prefix, preserve
    most of the audio prefix, and apply an eight-tick (0.2-second) half-cosine
    audio denoise-mask release from `0 = preserve` to `1 = generate` at the end
    of the audio context.

    When assembling, do not PCM-crossfade the audio and do not discard the new
    chunk's feathered prefix as the current sampler discards its synthetic
    packing prefix. Make the newer chunk's complete audio authoritative across
    the protected overlap: replace the matching tail of the accumulated audio
    latent with the new chunk's prefix, then append its newly generated audio.
    The referenced workflow blends only video pixels after VAE decode. Keep
    that visual blend separate from this audio experiment because the current
    priority is diagnosing and correcting long-chain color/detail degradation.

    Compare this exact method against the current Video1 synchronized-audio
    reference, which conditions H3 semantically but does not provide a masked,
    seam-ending target-audio timeline. Measure dialogue restart/repetition,
    waveform discontinuity, beat/ambience continuity, AV drift, VRAM, new
    frames per chunk, and total render time. A 56-frame target with 39 frames
    of physical context produces only 17 new frames per call, so expose the
    tradeoff explicitly rather than silently changing chunk progress.

    The installed ComfyUI snapshot is currently `v0.33.0-7-g55b6a9b1`. The
    initial implementation uses ComfyUI's generic nested-stream sampler mask;
    it does not copy the reference repository's GPL runtime patch for H3's
    model-internal fractional timestep rows. Reassess native H3 AV-mask support
    and compare it with the generic sampler-mask result after a future ComfyUI
    update. The reference repository remains a behavioral/testing source, not
    copied source code.

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
- dynamic Qwen/Gemma video presentation matching stock H3's 768-short-edge,
  `768 * 1344` area cap with preserved aspect ratio, 32-pixel canvas alignment,
  correct timestamps, and unchanged full-resolution DiT reference dimensions;
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

## HR Endless Sampler naming and released UI

The public sampler is now named **HR Endless Sampler** (`HREndlessSampler`),
and its paired live widget is **HR Endless Sampler Preview**
(`HREndlessSamplerPreview`). The preview's browser event, restore endpoint,
wrapper key, and web-extension identity were renamed together, so a ComfyUI
restart is required after this change. The architecture is intentionally
model-extensible: MiniMax H3 is the current backend, while LTX support is
planned; H3-specific latent, prompt, and reference rules remain backend code
rather than the product identity.

The released sampler UI now exposes only the stable Video1 continuation path:
`video_continuation` remains visible and is always enabled internally. The
former `video_continuation_enable` widget is hidden and forced true. The
experimental native context-keyframe, retained guide-overlap, Qwen full-history,
and prompt-preview-only widgets are hidden and forcibly disabled, including for
serialized values from an old workflow. Their implementation remains in source
for controlled development experiments, but it cannot change normal render
behavior through the current node UI. `debug`, `debug_stop_chunk`, and
`debug_start_chunk` are the diagnostic controls; `cache_gemma_preproduction`
is the optional Gemma performance control immediately before them.

Gemma's isolated preproduction planner and its first chunk-directing call can
legitimately take several minutes before H3's first denoising step. They were
previously silent when `debug` was off, making the sampler and preview appear
stalled. `_PreparationProgress` now logs every preparation phase regardless of
debug: initial chunk setup, one-time Gemma shot planning, each chunk's Gemma
direction, Qwen conditioning encode, and the point H3 inference starts. While
a Gemma worker is still running it emits an elapsed-time heartbeat every
15 seconds. The accumulated preview receives durable `phase` events, shows the
current phase beside its existing elapsed timer, and restores it after a
browser refresh. The purpose is observability only; no scheduling, worker,
conditioning, or sampling behavior was changed.

Each per-chunk Gemma directing handoff additionally uses an indeterminate live
tqdm console line. Gemma's isolated worker cannot expose meaningful generated
token progress, so the line deliberately loops without a percentage or ETA and
shows elapsed time instead. It is refreshed every second and closes before the
normal outer chunk and inner H3-step bars begin; the existing concise
console/preview heartbeat remains every 15 seconds.

## Gemma clean-preproduction KV cache and corrective chat turns

`cache_gemma_preproduction` is an opt-in, render-local acceleration path. The
normal Gemma workflow still creates one disposable llama.cpp worker for the
preproduction schedule and one worker per chunk so its CUDA allocations cannot
survive into H3 sampling. Before the first chunk, when the toggle is enabled,
the preproduction worker builds a second, clean *directorial* conversation:
the normal chunk-director system rules, compact H3 working summary, complete
original prompt, complete source-shot bodies, character-to-subject table,
physical chunk map, and validated full timing plan. It acknowledges that
memory, and its native llama.cpp state is saved immediately before Chunk 1.

Each chunk worker restores that same clean state, then appends only its dynamic
chunk request: current ownership/marker contract, current-slice timing-plan
coverage, actual continuation inventory, prior chunk's chronological stills,
and prior Gemma description/timing/end-state. It is intentionally not a
rolling all-chunk conversation; rendered evidence is still passed explicitly
per chunk, while the source intent and global plan stay stable. The cache is
deleted/reset at every new sampler execution, so it cannot leak a previous
render's prompt into the next one. Linux prefers `/dev/shm`; Windows and an
unusable RAM disk fall back to the platform temporary directory. A failed
export/restore only logs a warning and falls back to the ordinary complete
self-contained request.

The saved format contains native llama.cpp KV state plus token history, not
`Llama.save_state()`'s large historical logits matrix. An appended user turn
will always evaluate fresh suffix tokens before sampling, so historical logits
are unnecessary; omitting them keeps the RAM-disk snapshot close to KV size.
At the current 32K Q8_0 configuration it is still several GiB of system RAM,
not VRAM. Check available RAM-disk capacity before increasing the context
further.

The pinned `MTMDChatHandler` clears llama.cpp KV on every high-level
`create_chat_completion`, which used to make a correction retry re-evaluate the
entire original multimodal prompt and its images. The local handler now has an
append-only method that asks its own Jinja chat template for the exact
assistant-closing/user-opening suffix, evaluates that suffix and any new
images without reset, then generates the correction JSON. This applies to both
preproduction-plan corrections and chunk-contract corrections. A minimal
runtime/test fallback retains the old full-request behavior only when that
append capability is absent.

The complete vendored MiniMax skill and base/ref guides remain the reviewed
dependency source, but they are no longer injected into every Gemma request.
`minimax_h3_prompt_summary.txt` is the compact editable runtime distillation.
Update it together with a semantic review of the vendored sources whenever
their upstream hashes change.

### Native Gemma 4 MTP acceleration (2026-08-27)

Gemma was decoding at roughly 58 tokens/second in the sampler while the same
target reached about 120-150 tokens/second in llama.cpp's web chat. The web
configuration was not equivalent: it loaded the matching Gemma 4 QAT MTP
assistant with `draft-mtp` and proposed up to four tokens per target pass.
The main target was already fully GPU-offloaded, so CPU fallback was not the
cause of the gap.

`llama-cpp-python==0.3.35` already contains the low-level MTP/NextN structures
and symbols. Its advanced `examples/server/server.py` implements the complete
native MTP provider, while the ordinary high-level `Llama` class only wires a
simple Python draft callback. `gemma4_mtp.py` adapts the reference provider's
linked-context, single-sequence path to the disposable Gemma worker: it exposes
the target decode's NextN hidden rows, drives the Q8 assistant for at most four
greedy draft tokens, and lets the target model and JSON grammar verify every
proposal. The multi-user server scheduler is intentionally not vendored.

The node automatically downloads
`gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf` (about 465 MB) from
`Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF` beside the official
target and projector. This is a GGUF conversion of Google's matching official
QAT assistant checkpoint, not a standalone model. This first implementation
treated a missing symbol or initialization error as a warning and silently
fell back to normal decoding. That policy was later retired because it made a
controlled MTP/non-MTP speed comparison ambiguous.

The isolated worker emits lightweight generated-token counters while each
Gemma response is being decoded. The parent consumes those records live and
adds `N tokens, X.X tokens/sec` to the Gemma preparation bar and preview phase.
The clock starts at the first generated token, so the displayed rate measures
decode throughput and deliberately excludes model loading, image/prompt
prefill, and time-to-first-token. Progress records never enter Gemma's captured
JSON or final H3 prompt.

The requirements keep the Python version pinned at 0.3.35 but now use the
`cu125` wheel channel. That release channel contains both Linux and Windows
wheels; the earlier `cu121` channel did not publish a Windows wheel. Existing
Gemma preproduction state files remain target-only. A restored state evaluates
the new chunk suffix before sampling, which recreates the MTP pending hidden
state without storing a second assistant KV snapshot.

#### Native MTP correction and comparison toggle (2026-08-27)

The first adapter above proved to be an invalid performance comparison. It
attached the assistant after the ordinary high-level `Llama` target context had
already been created. The latest run showed about 30.5 accepted output
tokens/second even though the target itself still decoded at roughly 54.7
tokens/second; assistant work and repeated CUDA graph resets consumed most of
the remaining wall time. The target log also showed `n_rs_seq=0`. The first
attempt to correct this copied the older llama-cpp-python advanced server's
request for `n_rs_seq=4`. That was also wrong for the installed Gemma 4 model:
llama.cpp explicitly reported that the model does not support recurrent
partial rollback, clamped the value to zero, and the adapter stopped before
Chunk 1 with `Gemma target context was not created for native MTP rollback`.

The corrected implementation follows current llama.cpp rather than requiring
unsupported recurrent snapshot slots. `create_native_mtp_llama()` still loads
the target from birth with `load_mtp=true`, unified KV, and all NextN outputs,
then creates the linked `LLAMA_CONTEXT_TYPE_MTP` Q8 assistant and proposes at
most four tokens. The target and draft contexts deliberately use
`n_rs_seq=0`. Before each target verification batch, the adapter exports the
target's `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state. If the target rejects part
of the draft, the adapter restores that SWA/recurrent checkpoint, removes the
ordinary attention suffix, and replays only the physically accepted pending
token plus accepted draft prefix. This is the same rollback structure used by
current `examples/speculative-simple/speculative-simple.cpp`; merely calling
llama-cpp-python's normal `kv_cache_seq_rm()` is unsafe for a hybrid model.

The adapter also wraps the existing high-level generator rather than replacing
the chat/sampling implementation. JSON grammar, temperature, penalties, and
the MTMD chat handler therefore remain owned by llama-cpp-python. The wrapper
tracks an early-stopped generator so no unaccepted speculative suffix can leak
into the clean preproduction KV state. It forwards `Generator.send()` and
restores the original `eval`, `generate`, draft callback, logits mode, and KV
method during teardown.

Real-GPU validation on 2026-08-27 used the official Q4 target and Q8 MTP
assistant with a four-token draft. It generated a coherent three-sentence
answer after accepting 32 of 100 proposed draft tokens across 25 proposals,
which proves that real partial-rejection rollback was exercised rather than
only testing initialization. The unit suite also has a focused rejection test
that verifies checkpoint restore, attention trim, accepted-prefix replay,
token counters, and proposal cleanup. Worker teardown now prints checkpoint
time, rollback count, and replayed-token count in addition to the existing MTP
acceptance and assistant-rate statistics.

The sampler exposes `gemma4_mtp` immediately above its debug controls. It is
enabled by default. `true` selects the native four-token path; `false` creates
the original non-MTP high-level Gemma runtime and does not load or download the
assistant. An enabled native setup is fail-loud: it must never silently fall
back and contaminate a speed comparison. Both modes retain the same live
accepted-output tokens/second display. The selected mode is recorded in the
console and at the top of `last_gemma_chunk_prompts.txt`.

## Documentation used

- MiniMax-maintained H3 prompt-writing skill now vendored as the reviewed
  source for the compact Gemma runtime summary:
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
- llama-cpp-python's native MTP reference server implementation adapted for the
  local single-sequence worker:
  <https://github.com/abetlen/llama-cpp-python/blob/3691546f1c9e0c1bf93323dff02230bd959cf562/examples/server/server.py>
- Matching Gemma 4 12B QAT Q8_0 MTP assistant GGUF:
  <https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF>
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

## 2026-08-27 — Native Gemma MTP performance diagnosis and correction

The first working four-token MTP implementation was functionally real but
performed worse than expected: the render log showed roughly 18–22 accepted
output tokens/second and GPU utilization usually below 30%. The Q8 assistant
was not the bottleneck. Exact replay of captured Chunk 6 confirmed that MTP
proposed four tokens at a time, accepted 1,191 of 1,748 proposals (68.1%), and
generated assistant proposals at roughly 3,800–4,200 tokens/second.

Two implementation costs were corrected first. The on-device checkpoint
choice below was later retired by the 2026-08-28 stability fix documented
after this section:

- Target hybrid-state checkpoints temporarily used both
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` and
  `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. The former host checkpoint path cost
  13–14 seconds per response; the device path reduced it to roughly
  0.44–0.67 seconds, but was subsequently proven capable of aborting the
  worker on a real multimodal chunk and is no longer used.
- The MTP verifier no longer copies every target verification-logit row into
  NumPy because the director never requests log probabilities. Sampler debug
  no longer enables llama.cpp's native verbose diagnostics, and every MTMD
  handler is likewise constructed with `verbose=False`.

Those changes improved throughput but did not explain the remaining low GPU
use. Fine-grained timers around the exact Chunk 6 capture found the decisive
bottleneck:

- Q8 assistant generation: 0.427 seconds;
- batched target verification: 1.015 seconds for 437 calls / 2,185 rows;
- accepted-prefix target replay: 0.765 seconds for 206 calls / 473 rows;
- target synchronization: 0.026 seconds;
- **target sampling: 45.060 seconds for 1,628 calls**.

The target sampler was CPU-bound because llama-cpp-python's `json_object`
response format installs a JSON grammar over Gemma 4's very large vocabulary
for every generated token. This also explains why the user's native llama.cpp
web-chat configuration reached 120–150 tokens/second: that ordinary chat did
not apply the chunk director's strict JSON grammar.

The director now uses instructed JSON with post-generation parsing and schema/
semantic validation as its normal path. If Gemma returns malformed JSON, the
node prints a warning and retries once using the strict `json_object` grammar.
Chat-style semantic correction remains unchanged and also uses the fast path
unless syntax recovery is actually needed. The node never silently substitutes
an algorithmic prompt for Gemma's response.

Replaying the same captured Chunk 6 after this change produced valid JSON
without invoking the recovery grammar. Total wall time fell from about 65
seconds to 24.55 seconds, including a semantic correction pass. Live accepted
output rates were 97.2 tokens/second for the initial 879-token response and
125.0 tokens/second for the 835-token correction. MTP accepted 1,246 of 1,868
proposed tokens (66.7%) over 467 four-token proposals; target sampling fell to
0.261–0.285 seconds. Short speculative verification batches and CPU
orchestration mean GPU utilization need not match a long continuous server
decode, so accepted tokens/second and the detailed phase timers are the useful
comparison metrics.

The implementation was validated with the project's isolated ComfyUI Python:
32 Gemma capture tests, 22 chunk-director helper tests, and all 62 discovered
unit tests passed. New tests guarantee that valid JSON uses no grammar and that
malformed JSON activates the constrained recovery pass only.

### 2026-08-28 — Failed host-checkpoint MTP workaround

A real multimodal Chunk 2 capture reproducibly killed the disposable Gemma
worker with status `-6`. The native stack ended at
`ggml_backend_tensor_copy -> llama_io_write_device ->
llama_state_seq_get_data_ext` while saving the target's speculative rollback
checkpoint. Replaying the exact same request with `gemma4_mtp=false` succeeded,
which isolated the failure from the prompt, observation images, MTMD encoding,
and Gemma model itself.

The crash came from using `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for every hybrid
target checkpoint. llama.cpp upstream removed that flag from its speculative
server and `speculative-simple` example in PR #24108 because on-device
checkpoints are not fully compatible with meta/device buffers and use
unaccounted device memory. Issue #27439 also records that this path may throw or
abort before the public C API can report a normal failure.

An initial workaround changed checkpoints and restores to host-only
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state. It did avoid the original device
copy stack in the captured request, but real render measurements showed
13.5–25.5 seconds of checkpoint serialization per response and accepted output
fell to roughly 56 tokens/second. It also did not make MTP generally stable: a
later Chunk 2 worker still exited with status `-6` immediately after native MTP
initialization. The host-only workaround was therefore removed.

References:

- <https://github.com/ggml-org/llama.cpp/pull/24108>
- <https://github.com/ggml-org/llama.cpp/issues/27439>

### 2026-08-28 — Fast MTP with operation-local process fallback

The original fast `PARTIAL_ONLY | ON_DEVICE` checkpoint path is restored, but
it remains confined to the disposable Gemma worker. Before launching an MTP
worker, the parent retains a serialized copy of the exact request. If the
worker aborts, exits without a result, returns a non-zero status after a result,
or reports a native `Gemma4MTPError`, the parent prints a prominent warning and
retries the same operation using fresh original non-MTP workers.

The fallback applies only to the operation whose MTP worker failed. Its copied
request changes only `gemma4_mtp` from true to false; the preproduction cache
and every other request field are preserved. The director's MTP setting is not
changed, so the next preproduction or chunk operation attempts native MTP
again. Normal Gemma response/schema errors do not trigger the fallback; they
remain visible and fail normally.

This policy restores the previously observed 100–150 token/second fast path
when native MTP behaves, while treating a native child-process crash as a
recoverable acceleration failure rather than losing the entire H3 render.

On 2026-08-28, another exact Chunk 2 capture exposed a two-stage failure: the
native worker could not load the 444 MiB MTP assistant, then the first non-MTP
fallback aborted inside MTMD `clip_encode` with a CUDA error while encoding the
first 1344x768 observation still. The parent had reported 14,001.8 MiB
physically free immediately before directing, while an isolated replay showed
the same native request reaching about 14,800 MiB in its worker alone. The
captured request itself replayed successfully once more GPU headroom was
available, confirming a transient/resource-sensitive native failure rather
than corrupt prompt or image data.

One fallback was not enough to preserve a long render in that case. Worker
crash recovery now keeps one immutable serialized request and allows ten fresh
operation-local retries. The initial attempt honors `gemma4_mtp`; after an MTP
worker failure, all retries for that one operation use the original non-MTP
decoder. Every retry receives a new deep copy so subprocess cleanup cannot
erase image, cache, or request fields. An operation that starts with MTP off is
also allowed ten fresh non-MTP retries. The sampler fails only after the
initial attempt and all ten retries exit; the next independent operation still
honors the sampler's MTP toggle normally.

## 2026-08-28 — Per-chunk timing in all timeline players

Each completed sampler chunk records H3 sampling/render time, Gemma directing
time, and whole chunk-processing time. Chunk 1 additionally records the
one-time Gemma shot-timing preproduction pass. That preproduction duration is
attributed to both Chunk 1's Gemma total and Chunk 1's processing total, so the
first hover exposes the real startup cost instead of beginning its clock only
after planning is complete.

The live Preview tooltip presents the breakdown on one compact clock-formatted
line: `Chunk processing: mm:ss ( sampler:mm:ss + gemma4:mm:ss + misc:mm:ss )`.
`misc` is the non-negative remainder after H3 and all attributed Gemma work;
it therefore covers Qwen, VAE work, model swapping, conditioning, assembly,
and other per-chunk overhead. Chunk 1 adds a second line identifying how much
of its Gemma value was preproduction. The backend cache preserves all fields
across a browser refresh.

The same fields survive timeline normalization, video metadata/sidecar saving,
and loading, so the Save Video and Load Video players display identical hover
details. The timeline also stores the sampler's complete wall time as
`render_total_seconds`; both finished-video players show that total in their
bottom status line. Older videos without timing metadata remain loadable and
simply omit the unavailable values.

## 2026-08-28 — Load Video media and geometry outputs

`HR Endless Sampler Load Video` retains its original timeline, filename, and
FPS output slots and appends native ComfyUI `VIDEO`, decoded `IMAGE` batch,
`AUDIO`, frame-count, width, and height outputs. Appending the sockets preserves
existing workflow link indices. Ordinary video files are decoded once during
workflow execution and their components are shared by the VIDEO and IMAGE/
AUDIO outputs. EXR sequences retain their float image tensors and optional
float WAV sidecar audio when producing the same outputs.

Selecting a file in the custom browser still loads the timeline player without
queueing or decoding the complete ordinary video. Full media decoding happens
only when the Load node itself executes.

## 2026-08-28 — Shot-colored chunk prompt tooltips

The Preview, Save Video, and Load Video timeline players now use a styled HTML
tooltip instead of the browser's plain-text `title`. Help and timing lines stay
neutral. Only Gemma's `detailed_description` prose is colored, with each shot
section using the exact palette color of its source-shot bracket. A one-shot
prompt is still colored with that shot's bracket color.

Gemma's `[Shot N]` tokens preserve the global source-shot numbers. Their `At`
timestamps alone use the physical chunk-local clock. The UI therefore maps a
marked prompt section directly to the same global timeline shot. When a chunk
starts in the middle of an existing shot, its unmarked continuation prose
receives the already-active global shot's color.

## 2026-08-28 — Explicit PyTorch allocator ceiling

The sampler exposes `pytorch_memory_fraction`, defaulting to `0.85`. At the
very beginning of every sampler execution it calls
`torch.cuda.set_per_process_memory_fraction` for the model patcher's CUDA
device and logs the resulting GiB ceiling and active allocator backend. The
setting is process-wide and remains active until a later sampler run changes
it; `1.0` restores the normal unrestricted PyTorch limit. CPU/non-CUDA
execution skips it explicitly.

This was added after a 1920x1088 diagnostic completed two Chunk 2 evaluations
and failed on the third INT8 QKV concatenation. Only about 330 MiB was actively
allocated between evaluations, but `cudaMallocAsync` retained 14,720 MiB and
left only 104.9 MiB physically free. ComfyUI does not otherwise call
`set_per_process_memory_fraction`; it automatically enables
`cudaMallocAsync` on supported CUDA systems unless launched with
`--disable-cuda-malloc`. PyTorch's `garbage_collection_threshold` is ignored by
that backend. The explicit 85% ceiling is intended to create allocator
pressure while driver-side headroom still exists, rather than waiting for the
next large contiguous allocation to encounter a physically full device.

The backend input remains serialized and callable, but its native widget is
hidden from the sampler UI to keep the normal controls concise. The frontend
uses both the legacy zero-layout footprint and the Nodes 2.0 hidden flag
without changing the widget type or value, preserving existing workflows.
`video_continuation` now defaults to 22 frames in both the node schema and the
Python execute fallback; 5 remains the minimum selectable value.

## 2026-08-28 — Debug continuation VRAM preflight

Debug mode now performs a disposable three-step continuation simulation before
the real first chunk of a multi-chunk Video1 render. It uses the configured
target chunk shape, Video1 frame count and spatial-resolution preset, a
synchronized black audio/video reference, the normal sparse Qwen video
presentation, and the five-frame full-resolution boundary keyframe. This makes
the probe exercise the continuation attention rows that make Chunk 2 more
expensive instead of merely repeating the cheaper Chunk 1 layout. It also
holds simulated sampled and denoised Chunk 1 AV outputs on ComfyUI's configured
intermediate device during the probe; this costs no VRAM when that device is
CPU, but reproduces the retained-output cost on configurations that keep it on
the GPU.

The probe uses the first three transitions of the workflow's real sigma
schedule and the already-generated fixed chunk noise. Its output is never sent
to the preview, timeline, replay cache, or accumulated result. CPU and CUDA RNG
states are restored afterward. Cleanup clears every local reference, restores
the original guider conditioning, unloads H3/Qwen/VAE model owners, runs Python
garbage collection, forces ComfyUI's cache release, and empties the CUDA cache.
The console reports the preflight peak plus post-cleanup active/reserved memory
relative to the baseline and warns if active allocations remain materially
above it. This intentionally adds startup time only when `debug` is enabled.

## 2026-08-29 — Retained-boundary shot markers and persistent character state

A 13-chunk diagnostic exposed a false cut in Chunk 8. Source Shot 6 began at
global frame 359, while Chunk 8 physically sampled frames 357–412 but retained
only frames 362–412. The real cut therefore existed entirely inside the five
discarded packing frames and was already established by the Video1/boundary
conditioning. The old marker calculation compared the cut only with the
physical sampled start and incorrectly forced `[Shot 2] At 00:00.083,` into
Chunk 8. Gemma's initially correct unmarked continuation was rejected and its
model-authored correction inserted that duplicate cut. Marker ownership now
starts at the retained output boundary: cuts before `output_start` receive no
H3 marker, cuts exactly at or after it retain their truthful physical-local
timecode.

The same render showed Tila reappearing against a green meadow instead of
remaining mounted inside the ancient temple. Immediate-previous-chunk stills
cannot solve this when a character has been off-screen for several chunks.
Every Gemma chunk response therefore now owns a persistent
`last_seen_character_state` table with one entry per immutable named Subject:
last observed global frame/source shot, environment, pose and position,
ongoing state/action, and spatial relationships. The table represents rendered
evidence only through the prior generated chunk, never a forecast of the chunk
currently being directed. Gemma updates visible characters from the attached
chronological stills and copies absent characters' entries exactly, including
across replay-cache checkpoints. The replay format was bumped to 2 so older
checkpoints cannot silently omit this continuity memory.

For H3 conditioning, the sampler selects only table entries whose `<Subject N>`
actually appears in the current Gemma detailed description. Their environment
and physical state are inserted as a dynamic `retention_analysis` addendum;
off-screen characters are excluded to avoid inviting accidental appearances.
The detailed description remains responsible for current action and explicit
state changes. Gemma is also instructed that a camera cut changes framing, not
location or mounted/seated/standing state, and that an audible source not
explicitly revealed by the source prompt remains off-screen. This last rule
addresses the observed Shot 6 failure where H3 visualized a scheduled dragon
roar by cutting to the referenced roaring tiger.

## 2026-08-29 — Recoverable Gemma responses and interrupted-render checkpoints

A long render was interrupted after Gemma returned syntactically valid JSON
without its required `detailed_description` field. The former response path
treated that as a hard `Gemma4ObservationError`: only native disposable-worker
crashes had the ten-attempt recovery policy. This made an ordinary model output
mistake unnecessarily abort the entire H3 render.

Chunk-response validation now requests a complete **model-authored** replacement
in the same Gemma chat when the JSON lacks a usable `detailed_description` (or
another hard response-shape requirement). It preserves the current request,
chronological images, preproduction cache context, and prior answer in KV. Up
to ten bounded schema-repair turns are allowed; there is no algorithmic prompt
fallback. Existing one-pass marker/coverage/dialogue contract correction
behavior is retained, and an invalid response during that correction also uses
the bounded repair path. The transcript records every invalid JSON response and
the correction instruction whenever a repaired answer succeeds.

Serial runs now create the existing CPU/disk `last_run_replay` checkpoint for
every multi-chunk render, not only runs launched with `debug_start_chunk` set.
The checkpoint stores the initial AV latent/noise plus each successfully
completed chunk's sampled AV tail, assembled output, Gemma state, timings, and
independent prefix noise. Its manifest tracks `recording`, `interrupted`,
`debug_stop`, and `complete` states atomically with each chunk save.

With `debug_start_chunk=0` (the normal UI default), a new execution now first
checks that manifest. A compatible `recording` or `interrupted` render resumes
automatically at the next unsampled chunk; a completed run, intentional debug
stop, old unmarked cache, changed latent/chunk/continuation configuration, or
missing checkpoint begins a fresh render and replaces stale state. A nonzero
`debug_start_chunk` remains an explicit override for rerunning a chosen suffix.
The legacy manually created replay cache could not recover the already
interrupted diagnostic run, because it was not active for normal runs before
this change.

The live Preview transport has a small `clear` button directly below its gray
frame counter. It deletes only the fixed temporary `last_run_replay` recovery
cache, making the next compatible execution start at Chunk 1; rendered output
files and the visible preview are not deleted. The browser polls a dedicated
cache-status route, so the button is disabled when no completed chunk cache
exists and while any sampler execution is active. The backend enforces the
same active-run lock, preventing a stale browser state from deleting a cache
that the sampler is currently reading or writing. Its native tooltip explains
the operation and reports the number of cached completed chunks.

### 32K Q8 preproduction-cache configuration

The earlier 20,480-context failure was not an upper limit of this GPU: the
sampler had left llama.cpp's K/V cache at its F16 defaults and its worker log
reported a full-size SWA cache. The same machine's native llama.cpp Gemma
server successfully uses a 262K context with quantized cache configuration.

The sampler now explicitly configures the pinned llama-cpp-python runtime with
`n_ctx=32768`, `type_k=GGML_TYPE_Q8_0`, `type_v=GGML_TYPE_Q8_0`, and
`swa_full=False`. Q8_0 halves the K/V storage versus F16 while retaining more
precision than Q4_0; it is the chosen default for prompt-directing quality.
Disabling forced full-size SWA mirrors the native server's normal default.
Native MTP's linked assistant context inherits these target-cache settings.

The clean preproduction KV cache is never removed merely to recover context
space. It carries the prompt, source-shot plan, and prior preproduction state
that Gemma needs to direct chunks coherently. A genuine 32K overflow remains a
visible Gemma error to be fixed by reducing/summarizing the appended request,
not by silently re-running Gemma without its preproduction memory. Existing
preproduction cache files are versioned by their runtime metadata and are
recreated at the start of each render, so the changed context size cannot load
an incompatible old state.

## 2026-08-29 — Real Gemma integration replay and MTP JSON recovery

`tests/test_gemma4_live.py` is an opt-in real-GPU integration test. It reads
the user's untracked `prompt.txt` as a 625-frame production, runs the actual
Gemma preproduction timing planner, finds a recent exact observation capture
under `/tmp/hr-endless-sampler-gemma4-*`, restores its chronological JPG
stills, and directs that real chunk. It never downloads the model and remains
skipped in the ordinary unit suite. Enable it with
`HR_ENDLESS_SAMPLER_RUN_GEMMA4_LIVE_TEST=1`; use
`HR_ENDLESS_SAMPLER_GEMMA4_LIVE_MTP=0` for the original-decoder comparison.

The captured Chunk 10 replay isolated two failures. First, experimental MTP
could consume its complete 1,024-token output budget without producing a
parseable JSON object. The old path then generated two more append responses
and a grammar response in the same broken MTP operation. That both delayed the
render and produced misleading 20–30 token/s readings: local diagnostics had
already shown grammar sampling dominating target time, while the clean
original decoder remained around its former 50–60 token/s class. A new typed
`Gemma4MTPOutputError` now exits the disposable worker after the first
no-complete-JSON MTP answer. The parent retries the exact operation in a fresh
non-MTP worker, retaining every request and preproduction-cache field. The
next independent chunk still attempts MTP normally.

Second, a valid non-MTP JSON answer could fail the H3 coverage contract and
then append the same large correction contract up to ten times, eventually
overflowing 32K context. Only the first correction now repeats the complete
contract; later schema repairs are compact deltas listing the seven required
JSON fields and the remaining errors. Dialogue speaker reminders are limited
to dialogue belonging to the current source shot(s), instead of copying every
line from the whole production. The live MTP-enabled test now survives the
native preproduction abort, the Chunk 10 empty-MTP response, the operation-local
original-decoder fallback, and one creative coverage correction, completing in
105 seconds. The normal suite passes 99 tests with the live test skipped.

A sampler run started before this revision exposed a hot-reload boundary:
ComfyUI's long-lived parent retained the older worker-result decoder while the
new disposable worker imported the edited file from disk. The worker returned
the new typed MTP-output error, but the stale parent reduced it to a generic
`Gemma4ObservationError` and stopped before the retry loop could recognize it.
The retry loop now also recognizes the exact MTP no-complete-JSON message when
it arrives as the generic class. This is intentionally narrow and does not
retry unrelated creative/schema errors. A ComfyUI restart is still required
after editing backend Python so the active parent and workers use one revision.

The next real Chunk 2 run exposed a separate output-budget/parser bug after
the MTP abort recovered correctly. The original decoder generated exactly
1,024 tokens on every attempt. Its outer seven-field response was truncated
inside `last_seen_character_state` before it reached `detailed_description`.
The old JSON extractor searched every later `{` after the incomplete root and
incorrectly accepted one complete nested character-state object as the whole
response. Validation consequently reported a missing detailed description and
asked ten times for a replacement that hit the identical 1,024-token limit.

Chunk JSON generations and their append-only corrections now receive a 2,048
token budget. The extractor permits prose/channel text before the first `{`,
but once the instructed root begins it either decodes that complete outer
object or reports an incomplete/malformed top-level response; it never salvages
a nested object. The exact saved six-image Chunk 2 capture was replayed through
the original decoder: it returned a 3,847-character complete response, used
one normal coverage correction, and finished with a usable H3 description and
no validation warnings.

## 2026-08-29 — Native MTP checkpoint regression and reference port

The later near-100% native-MTP failure rate was not a Gemma response problem.
Historical render logs showed that the original 16K/default-KV MTP runtime
successfully completed hundreds of device checkpoints per response at roughly
120–150 output tokens/second and 62–75% draft acceptance. Recent 32K/Q8_0 runs
instead repeatedly aborted after exactly two generated tokens inside
`llama_state_seq_get_data_ext()`, reporting either `not enough space in the
buffer` or an invalid backend-buffer assertion. `gemma4_mtp.py` had not changed
between those runs; its explicit `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` checkpoint
flag was the incompatible part exposed by the larger/quantized runtime.

Current llama.cpp removed on-device speculative checkpoints because their
extra device allocation is not accounted at context startup and is not fully
compatible with device/meta buffers. The local adapter now matches the current
reference lifecycle: save and restore `PARTIAL_ONLY` state in host memory,
retain one growable host checkpoint allocation across proposals, restore that
hybrid state after partial draft rejection, remove the speculative attention
suffix, and replay only the accepted target prefix. Four-token native MTP,
linked NextN contexts, and the disposable-worker operation-local original
decoder retry remain unchanged. All 101 normal unit tests pass; the real GPU
Chunk 2 replay still has to be run after ComfyUI releases the approximately
14.95 GiB it currently owns.

## 2026-08-29 — Global shot labels with chunk-local cut times

A live Chunk 3 exposed two independent sources of ambiguity. The physical
chunk began inside global Source Shot 2 and contained the one real transition
into global Source Shot 3. Renumbering that transition as local `[Shot 2] At
00:01.958,` asked Gemma and H3 to treat one label as two identities: global
Shot 3 in the production plan but local Shot 2 in the generated prompt. Gemma
then replayed Shot 2's final action and also copied global `[Shot 3] At
00:06.208,`, producing two markers and two apparent restarts.

The prompt contract is now simpler: `[Shot N]` always retains N from the
original/global source prompt, while only `At MM:SS.mmm` is reset to the
physical chunk-local clock. If a chunk begins midway through global Shot 2,
that continuation is unmarked. If global Shot 3 begins 1.958 seconds into the
same physical chunk, its exact marker is `[Shot 3] At 00:01.958,`. A shot that
starts exactly at the physical chunk opening uses its global `[Shot N]` marker
without an `At` time.

Every initial request and marker correction includes this sampler-owned
transition map. Python validates both marker count and the complete exact
global-label/local-time token sequence. Marker violations are structural, so
they receive focused append-only Gemma repair turns—up to the existing
ten-repair bound—until the exact sequence is valid; Python never edits or
substitutes H3 prose. Other creative coverage warnings retain the one-rewrite
policy. Replay fingerprints include the marker-mode version so cached prompts
using the old local renumbering cannot silently survive this change. The normal
suite passes 102 tests with the opt-in live GPU test skipped.

## 2026-08-30 — Stable UTF-8 Gemma worker protocol

GitHub issue #9 reported a Windows `UnicodeDecodeError` while the parent
sampler read a Gemma worker pipe as UTF-8. Byte `0x93` is a typical CP-1252
opening quote, showing that the child Python process could inherit the Windows
console encoding even though the parent explicitly requested UTF-8 decoding.

All three Gemma subprocess paths now use one shared worker launcher. Its child
environment forces `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`, while the parent
pipe retains `encoding="utf-8"` and uses `errors="replace"` as a final guard
against stray invalid bytes emitted by native code. This prevents a locale or
one malformed diagnostic byte from terminating a long render. A regression
test verifies the command, environment, and text-decoder configuration. The
normal suite now passes 103 tests with the opt-in live GPU replay skipped.

## 2026-08-30 — Preproduction character continuity and H3-facing retention

The first dynamic retention implementation derived its H3 text in Python from
Gemma's persistent `last_seen_character_state`, but filtered that table by
subjects already named in the current `detailed_description`. This was a
circular failure: if Gemma emphasized Tila and omitted Heman once, Python also
removed Heman from `retention_analysis`, making H3 even more likely to remove
him. The injected block also exposed the internal phrase `Gemma last-seen
continuity state relevant to this chunk`, which is bookkeeping language and
has no place in an H3 prompt.

Preproduction now authors a `continuity_slices` table alongside every source
shot's visual beats and overlays. Its half-open source-relative intervals must
exactly match the physical chunk output ownership for that shot. Each interval
lists every mapped character physically participating there—even when a
closeup puts that character temporarily off-screen—with a concise planned
entry state and expected exit state covering location/environment, mounted or
standing state, pose/action, and important spatial relationships. Python
validates the exact slice geometry, immutable name-to-subject mappings,
duplicate entries, usable entry/exit text, and complete omission across the
whole shot of a character explicitly named in its source prompt. Gemma—not a
Python name scan—decides which physical slices that character actually occupies.
The original monolithic timing-plan response budget was raised to 16,384
tokens after a live 19-chunk plan proved that 4,096 tokens truncated the root
JSON repeatedly. That monolithic response has since been replaced by the
global-bible plus independent-shot architecture documented below.

For each live chunk, the director now receives the relevant preproduction
entry/exit table plus the persistent observed table and chronological prior
stills. Gemma is instructed to compare the latest rendered state with the
planned entry, report drift in its Gemma-only analysis, and write
`detailed_description` as a bridge from what H3 actually rendered toward the
planned slice-exit state without resetting completed progress. Every planned
participating character must remain present in both the directed prose and a
new model-authored `retention_analysis` JSON value.

That `retention_analysis` value is passed to H3 verbatim beneath the existing
field header. It is short physical prose—where each participating character is
and how they are posed or moving—and contains no Gemma, last-seen, table,
frame-analysis, plan, or future-transition language. Python no longer builds
or filters retention prose from `detailed_description`. It validates that each
planned subject appears, requests append-only Gemma correction when the field
is missing or leaks internal language, persists the exact value in replay
state, and exposes that same value in live/saved timeline tooltips. The normal
suite passes 108 tests with the optional live GPU replay skipped.

## 2026-08-30 — Debug capture of every raw Gemma worker response

When sampler debug is enabled, every complete response object returned by
llama.cpp is appended before parsing to
`${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_raw_output.txt`. The file is
reset at the beginning of each real debug render and removed when the next
render runs without debug, so it cannot silently describe an older run. Each
entry identifies the UTC time, disposable worker PID, operation/chunk, and
whether it was the initial response, an append-only JSON repair, or the final
grammar fallback. The serialized object preserves visible content, reasoning
content, token usage, and finish reason. This is deliberately separate from
`last_gemma_chunk_prompts.txt`, which remains the curated request/response and
final-H3-prompt transcript. The raw file makes output-budget truncation and
malformed intermediate attempts inspectable even when a later repair succeeds.

## 2026-08-30 — Nonfatal iterative preproduction repair and paused VRAM probe

A debug raw capture proved that the raised 16,384-token preproduction budget
was working: Gemma stopped normally after 4,616 tokens with a complete root
JSON object. The render instead failed because the validator required every
character named anywhere in a source shot to appear in every physical
continuity slice. That was too mechanical. For example, Shot 7 names Tiamat,
but Tiamat is still hidden behind the gate during the shot's opening slice.
The validator now allows a character to be absent from slices where Gemma says
the character has not appeared, while still rejecting a plan that omits that
named character from the entire source shot.

The same capture showed that the first repair correctly added the missing
tiger state to Shot 4 but independently omitted Tiamat from all of Shot 7.
Preproduction used to abort immediately after that single imperfect semantic
repair. It now keeps requesting complete Gemma-authored timing-plan
replacements, using the existing ten-repair ceiling. Python continues to
validate and never invents continuity states or substitutes a fallback plan.

The disposable three-step continuation VRAM preflight has been temporarily
disabled because its delay became intrusive during prompt iteration. Its code
is preserved behind `ENABLE_DEBUG_MEMORY_PREFLIGHT = False` in `nodes.py`, so
the memory experiment can be restored without reconstructing it.

## 2026-08-30 — Global production bible plus independent shot preproduction

The one-time Gemma preproduction is no longer one large JSON response that
contains every visual beat and every physical continuity slice. That design
made a local omission—such as missing Tiamat from Shot 7—force Gemma to rewrite
the complete multi-shot plan. On the 625-frame, seven-shot diagnostic prompt,
the successful run generated roughly 4,600 tokens three times and spent 2:55
in preproduction even though only two localized continuity defects needed
repair.

Preproduction now has two explicit levels:

1. The global production-bible pass reads the complete source prompt and all
   immutable sampler-owned source-shot boundaries. It records every explicit
   `Name is <Subject N>` mapping, each shot's complete intent and environment,
   supplied cut/camera design, and every participating mapped character's
   physical opening and closing state. It does not allocate frame-level beats
   or physical chunk slices. Python independently extracts unambiguous mapping
   declarations from the original prompt and rejects a global table that omits
   or changes one. The global response budget is 4,096 tokens.
2. Each source shot receives a separate text request containing the unchanged
   validated global bible, that one authoritative source-shot description, its
   exact duration, and only its physical ownership intervals. Gemma creates
   that shot's contiguous visual beats, concurrent dialogue/sound/action
   overlays, and per-slice character entry/exit states. Python validates one
   shot at a time. A malformed or semantically incomplete Shot 7 response gets
   an append-only Shot 7 correction turn; accepted global preproduction and
   Shots 1-6 are not regenerated or edited. Each shot response has its own
   4,096-token budget and the existing ten-correction ceiling.

The assembled `GemmaShotTimingPlan` persists the immutable global-bible JSON,
the separately validated shot schedules, every raw response/correction, and
the exact global and per-shot requests. The debug preproduction transcript
therefore remains sufficient to reproduce what Gemma saw and authored.

Chunk directing receives the same separation. Without a KV cache, the chunk
request includes the global production bible plus only the finalized source
shot plan(s) intersecting that chunk. With `cache_gemma_preproduction`, the
clean KV snapshot holds the global bible, authoritative source descriptions,
original prompt, reference meanings, and physical ownership map—but not every
shot's detailed timing schedule. Each cached chunk turn receives only its
relevant finalized shot plan and mandatory current-slice intersections. This
keeps the global production immutable while preventing unrelated shot detail
from crowding or confusing a local directing call.

Replay-cache format 4 invalidates saved monolithic preproduction plans. The
console now announces global phase 1 and each independently planned source
shot in phase 2. The normal suite passes 112 tests with the optional live GPU
test skipped because an H3 render was actively saturating the only GPU.

## 2026-08-31 — Completed chunks become full-VAE video-and-audio previews

The live sampler preview still uses TinyVAE or Latent2RGB while H3 is
denoising, because loading the full video VAE inside the active DiT step would
defeat the preview's low-overhead purpose and can cause an OOM. Once a physical
chunk finishes, the sampler now unloads H3/Qwen and performs one authoritative
full video-VAE decode. It strips the five discarded packing-prefix pixel
frames and atomically replaces that chunk's approximate browser group with all
retained output frames. Finalized groups deliberately ignore the live
`frame_stride`: their frame-number list is contiguous, their per-frame
durations use cumulative FPS rounding, and Left/Right therefore advances by
exactly one real decoded output frame rather than jumping through H3 latent
time representatives.

This is not an extra whole-chunk video decode in the ordinary continuation
pipeline. The CPU-resident decoded tensor is reused for the next chunk's sparse
Gemma visual observation and color diagnostics, then released before H3
sampling begins again. Video1 intentionally retains its existing independent
bounded-latent decode: cropping the finalized full-chunk decode could carry
different temporal VAE boundary context and would change conditioning rather
than merely improve preview. The last chunk's existing diagnostic decode is
likewise replaced by this earlier authoritative finalization. If a full decode
fails, the render continues with the last approximate preview and the next
required continuation stage may retry its normal decode.

MiniMax H3 uses a separate audio VAE, so `HR Endless Sampler` now exposes an
optional `audio_vae` input immediately after its video `vae`. When connected,
each completed physical audio latent is decoded with the same normalization as
ComfyUI's standard `VAEDecodeAudio`, trimmed past the discarded latent prefix,
cropped to the exact retained video duration, and encoded as stereo/mono PCM16
WAV for the browser. The finalized video frames and WAV are sent and cached in
one `chunk_final` event, so a browser refresh restores them together. The
custom image player uses an invisible HTML audio transport, seeks it whenever
the timeline or arrow keys move, pauses it during frame inspection, changes
its playback rate with the live FPS control, and resynchronizes small drift at
frame boundaries. Browser autoplay policy may initially suppress sound; the
preview's Play button or a click on the viewport is an explicit media gesture
that enables it. Without `audio_vae`, finalized full-VAE video still replaces
the latent preview but remains silent.

The server cache treats `chunk_final` as a replacement for the earlier live
`chunk` payload, preserving full frames and audio across browser refreshes.
Focused tests cover contiguous final frames even when live `frame_stride` is
larger than one, WAV encoding, exact audio prefix/duration trimming, and cache
replacement. The complete suite passes 115 tests with one opt-in live GPU test
skipped.

## 2026-08-31 — Per-chunk H3 retention injection disabled

Live renders showed that the dynamically appended per-chunk
`retention_analysis` prose could make H3 break visual continuity, introduce a
cut, or omit characters instead of preserving them. Two different additions
were involved: Gemma's compact per-chunk character entry-state value and the
sampler-authored `<Video N>`/`<Audio N>` `fully_preserved` continuation lines.
Both are now disabled behind
`INCLUDE_PER_CHUNK_RETENTION_ANALYSIS = False` in `nodes.py`.

The source prompt's original, global `retention_analysis` is preserved exactly
and remains the only retention text encoded for H3. Video/audio continuation
is still declared through the documented `subject_definitions` and `summary`
sections; only the extra retention prose is absent. Gemma's internal planning,
observed last-seen state, and proposed retention value remain available in the
debug transcript for future experiments, but the proposed value is explicitly
labelled as not sent to H3. New chunk timeline/replay metadata also omits it,
and preview/save/load tooltips hide the retention subsection when no older
recorded value exists. A regression test verifies that a global retention
block survives while both kinds of dynamic text are excluded.

## 2026-08-31 — Crackle-free preview transport and mute control

The first finalized-audio browser transport called `syncAudio` for every image
frame. It contained a direct logic bug: when `shouldPlay` was true but the
media element was already playing, the shared `else` branch paused it. The next
frame started it again, creating rapid pause/resume cycling that sounded like
crackling. It could also assign `currentTime` whenever visual WebP decoding
drifted by more than 200 ms, introducing additional audible seek clicks.

Ordinary playback now leaves a running audio element running. It seeks only
when playback starts/restarts, the user seeks or frame-steps, a finalized chunk
replaces its latent preview, or playback moves to a different chunk. For a
finalized chunk, audio is the stable clock: visual frame delays are calculated
from its continuous media time, and a late decoded image may be skipped to
catch up instead of moving audio backward. Live latent previews without audio
retain their ordinary duration-driven frame loop. Playback-rate assignments
are also made only when the requested FPS ratio actually changes.

The preview transport now has a compact speaker button immediately to the
right of the timeline. It displays a crossed speaker while muted, changes the
HTML media element's mute state without stopping or seeking it, and can also be
toggled with the `M` key while the preview has focus.

## 2026-08-31 — Dialogue is partitioned across retained chunk ownership

The former dialogue coverage contract repeated a complete `<d>...</d>` line in
every physical chunk intersected by its preproduction overlay. A line planned
for source-relative frames 140-349 was therefore written in full to both a
chunk ending at frame 242 and the next chunk beginning at retained frame 243.
Although the general continuity prompt said not to replay completed material,
the mandatory validator explicitly demanded the full line again, so H3 could
restart and visibly repeat the speech.

Independent source-shot preproduction now creates immutable
`dialogue_segments` for every dialogue overlay. Their half-open frame ranges
must exactly equal the intersections between the planned utterance and the
sampler's retained physical ownership intervals; sampled overlap and the five
discarded prefix frames do not own dialogue. Gemma chooses natural word or
clause boundaries according to the time available in each segment. Every
segment repeats the original language tag for valid H3 syntax, while Python
validates that all segment speech concatenates to every original word and
punctuation token exactly once and in order. Missing, duplicated, reordered,
or paraphrased words make only that source-shot plan enter Gemma's existing
correction loop.

A long source `<d>` block is not required to remain one enormous overlay.
Gemma may divide it into multiple chronological overlays, each containing one
contiguous source fragment, which is especially useful for monologues spanning
many small chunks. Python validates the complete ordered overlay collection
against the source speech, including stable speaker ID and language, before it
validates each overlay's retained-slice segments. This replaced an overly
strict first implementation that demanded every overlay `content` equal an
entire source `<d>` block; Gemma naturally produced per-interval overlay
fragments, so that rule caused ten futile MTP corrections followed by the same
non-MTP retry failure during a one-minute monologue test. Later overlays inside
the same original speech are now flagged as continuation even when they each
contain only one physical segment.

Chunk mandatory coverage now exposes only the assigned dialogue segment, with
a unique suffix such as `S1.O2.D2`. Later pieces are explicitly described as
continuing without pausing or restarting; their H3 prompt contains only their
remaining partial `<d>` line, never the complete source utterance. The chunk
validator accepts exact source fragments, still checks mapped speaker syntax,
and requests a correction if Gemma expands the fragment back to the full line.
Replay cache format 5 invalidates preproduction schedules made under the old
full-line-per-intersection contract.

Future experiment: if H3 clips or misses the first phoneme at a dialogue
boundary, test a controlled one- or two-word textual overlap. Keep it disabled
initially because any overlap deliberately repeats audible content and must be
evaluated against synchronized `<Audio N>` continuation.

## 2026-08-31 — Optional masked AV continuation with feathered audio

`HR Endless Sampler` now exposes `video_continuation_method` immediately after
`video_continuation`. Existing workflows and the default retain `Video1
reference (current)` unchanged. The second choice, `Masked AV overlap
(experimental)`, is intentionally independent of Ref2VA semantic continuation:
it does not decode/re-encode a generated Video1, append a generated
`minimax_refs` block, show generated Video1 frames to Qwen, inject `<Video N>`
or `<Audio N>`, or add `[video continuation]` summary text.

In masked mode, `video_continuation` is the physical AV overlap inside the
configured `chunk_frames` target. For example, `chunk_frames=56` and
`video_continuation=22` samples 56 total frames in every later H3 call and
retains 34 new frames. There is no additional five-frame synthetic packing
prefix: the 22-frame overlap itself satisfies H3's temporal phase. This removes
the current method's separate 22-frame clean Video1/Audio1 attention block and
its five-frame boundary keyframe rows, so later calls should use less VRAM.
The cost is temporal throughput: the current Video1 method retains 51 new
frames from a 56-frame call, while the masked 22-frame method retains 34 and
therefore needs more serial chunks.

For each later chunk, the sampler copies the previous completed video/audio
latent tail into the target's opening physical interval. A nested ComfyUI
denoise mask hard-preserves every video-overlap token. It hard-preserves most
audio-overlap ticks, then applies an eight-tick (0.2-second at H3's 40 Hz audio
latent rate) half-cosine release whose values end at exactly `1 = generate`.
Any valid H3 frame count (`5, 22, 39, ...`) is accepted as long as it is smaller
than the effective chunk. Fractional 24-FPS/40-Hz durations such as 22 frames
use the planner's cumulative global audio boundaries, so individual joins may
carry 36 or 37 ticks without accumulating duration drift. The 39-frame setting
remains the exact shared-grid comparison: it is always 65 audio ticks.

Visual assembly keeps the previously accepted video overlap and appends the
new chunk only after the protected prefix. Audio assembly is different because
the final feather ticks were deliberately regenerated: the new sampled audio
prefix replaces the matching latent tail of all accumulated output, then only
the post-overlap audio is appended. The same operation is applied to the
denoised output. Tail replacement handles an overlap spanning more than one
stored chunk part. Replay-cache format 6 records each chunk's feathered sampled
and denoised audio prefix and reapplies the replacements in order when resuming,
so an interrupted render cannot silently restore the old hard audio seam.

The masked branch gives Gemma an explicit factual conditioning description:
the opening frames are a copied physical AV overlap, not a new reference or a
new shot. Its deterministic planned prompt likewise treats the prefix as
opening frames while never introducing Video1 labels. `video_continuation_res`
remains serialized but applies only to the Video1 method.

This first version uses the installed ComfyUI generic nested-stream sampler
mask. The inspected external implementation additionally carries separate
fractional video/audio timesteps through H3's internal DiT rows and supplies a
runtime compatibility layer for older ComfyUI builds. That GPL implementation
was not copied into this Apache-licensed repository. The new mode is therefore
labelled experimental until a real GPU comparison establishes audio quality,
visual continuity, VRAM, and whether the local ComfyUI mask behavior is
sufficient. CPU regressions verify overlap geometry, copied AV values, the
exact half-cosine ramp, cross-part audio-tail replacement, prompt isolation,
schema/default compatibility, and replay-method fingerprint separation.

## 2026-08-31 — Full-horizon preproduction under debug stops

A one-minute monologue exposed that `debug_stop_chunk` was incorrectly
shortening Gemma's immutable preproduction horizon as well as the actual H3
render. A successful run planned all seven physical chunks and divided the
dialogue naturally; a later `debug_stop_chunk=3` run told Gemma that the
production had only three physical chunks. Gemma compressed nearly all speech
into retained frames 0-650, then satisfied a structural correction by filling
frames 651-1449 with invented silence. The chunk director correctly copied the
bad mandatory dialogue segment, so the defect originated in preproduction.

Gemma preproduction now always receives the complete physical `plan`, complete
source-shot range, full chunk count, and every retained output-ownership
interval. `debug_stop_chunk` continues to limit only sampling, preview output,
and chunk-local execution. A restored timing plan is revalidated against this
complete request; a formerly truncated schedule regenerates preproduction while
preserving compatible cached physical chunks and noise.

Shot-plan validation also has a permissive dialogue-density ceiling: each
dialogue overlay may contain at most four whitespace-delimited spoken words per
second plus a four-word short-phrase allowance. The prompt targets a natural
two-to-three words per second and tells Gemma to extend long speech across later
ownership slices rather than compressing it and inventing silence. This guard
is intentionally generous enough for short exclamations and is a structural
failure that triggers the existing model-authored per-shot correction path.
Regression coverage reproduces a seven-chunk full plan limited to three debug
chunks and rejects an exact-word but physically over-compressed dialogue plan.

## 2026-09-01 — Sparse overlap anchors and explicit dialogue timing

A live masked-AV test with a 39-frame overlap confirmed that the deterministic
`At 00:01.625,` prefix was present in every final H3 prompt after Chunk 1, but
H3 did not use that general prose cue to delay its audio head. Dialogue still
began inside the physical overlap and its onset was removed when the overlap
prefix was trimmed. The one exact previous-final-frame keyframe improved visual
geometry at chunk joins, but the first fully generated retained frame could
still change background texture, contrast, or small objects.

The masked mode now encodes three independent one-frame native MiniMax guides
from the exact decoded previous-chunk tail: the first, middle, and final frame
of the protected physical overlap. They are anchored at the corresponding
local overlap positions (for 39 frames: 0, 19, and 38; for 22: 0, 10, and 21).
Each image is encoded separately so MiniMax receives sparse keyframes rather
than one contiguous guide clip. The fully masked video overlap is unchanged.

Gemma's preproduction shot planner must now reserve at least 0.5 seconds for
every `...` or Unicode ellipsis sequence in immutable dialogue, because H3
turns that punctuation into an audible dramatic pause. The Python dialogue
density guard deducts this pause budget before checking the permissive
four-words-per-second ceiling. The planner also reserves at least 0.25 seconds
after a completed utterance instead of scheduling its final word on the last
frame of a physical chunk or source shot.

Every live dialogue segment now carries a sampler-calculated exact physical
chunk-local cue such as `At 00:01.625,`. Gemma must copy that cue immediately
before the assigned H3 speaker clause. The cue time comes from the immutable
preproduction segment's actual global start minus the physical sampled chunk
start, so it includes a masked overlap or synthetic packing prefix without
guessing. Chunk validation warns and invokes the existing Gemma-authored
correction pass when the exact cue is absent. Replay-cache format 8 invalidates
older cached prompts and timing plans that predate these timing rules.

### Follow-up result: dialogue `At` cues are unsupported

The next live seven-chunk render proved that the dialogue cue was present in
every final H3 prompt but was not interpreted as a speech-start command. Chunk
1 used `At 00:00.000,` and began near the end of its assigned speech. Later
chunks used `At 00:01.625,`; Chunk 2 still began inside the protected overlap
and lost its first words, while other chunks ignored the time entirely. The
official MiniMax prompt guides define `At MM:SS.mmm` only in `[Shot N] At ...`
real-cut markers and provide no dialogue-timecode syntax.

Both experimental mechanisms were therefore removed: no general masked-AV
prompt prefix and no dialogue-level `At` cue remain. Real source-shot cuts keep
their sampler-calculated `[Shot N] At ...` markers. Replay-cache format 9
invalidates prompts/preproduction created under the failed timing experiment.

The official speaker contract is now reinforced instead. Global preproduction
creates one immutable `speaker_voice_profiles` entry for every stable `(Sx)`
vocal source. It preserves a supplied description or chooses one concise
pitch/timbre/cadence/rate profile when none exists. Every independent H3 chunk
with that speaker must repeat the exact profile outside `<d>`. A later segment
of the same source utterance must also use the exact phrase `continues
uninterrupted`; Python validation requests a Gemma-authored correction if the
profile or continuation phrase is missing. Only the language tag and exact
spoken content remain inside `<d>`.

The first/middle/final sparse visual overlap anchors remain enabled because the
test showed substantially better background, position, and state continuity.
The same test made chunk-to-chunk color variation more visible, suggesting the
decoded-image-to-VAE-keyframe round trip or multiple image anchors can impose
slightly different color statistics. Keep this as a separate visual experiment
from the new voice/audio prompt test rather than changing both at once.

## 2026-09-01 — Browser refresh must preserve finalized preview media

The live latent-preview encoder, the server-state restore request, and browser
websocket events are asynchronous. A delayed `chunk` event could therefore
arrive after a completed `chunk_final` event and replace its authoritative
full-VAE frames and decoded WAV audio. The preview would look correct before a
browser reload but could fall back to the fast latent decode without audio
afterward. Finalized chunks are now monotonic within one execution: both the
server cache and browser reject a later latent-only downgrade for the same
chunk. A new sampler execution still resets all groups normally.

Interrupted-render recovery had a second, independent issue: its checkpoint
stores authoritative AV latents, but the preview restore path deliberately ran
only Latent2RGB/tiny-VAE and published an ordinary `chunk` event. Resume now
decodes every completed checkpoint through the connected full video VAE and
audio VAE, applies the original video-frame and audio-latent prefix trims, and
publishes `chunk_final`. The final restored physical decode is retained for the
next Gemma observation to avoid decoding it twice, then the VAE models and CUDA
cache are released before normal preparation continues. A preview-only decode
failure falls back to the latent preview for that individual chunk and never
aborts the recoverable render.

## 2026-09-01 — Final chunk-dialogue punctuation normalization

After Gemma validation but before H3 encoding, Python now inspects only the
last `<d>...</d>` block in the chunk description. If its spoken text ends with
`...`, a Unicode ellipsis, a comma, semicolon, colon, dash, or another terminal
punctuation character outside `.`, `!`, and `?`, that terminal punctuation run
is replaced with one period. Earlier dialogue blocks, all spoken words, the
language tag, valid terminal marks, and dialogue with no punctuation are left
unchanged. This is intentionally post-validation: Gemma and the timing planner
continue to reconstruct the immutable source dialogue exactly, while the final
H3 prompt gets a closed terminal stop instead of an open pause at a physical
chunk boundary. The exact normalized H3 prompt is retained for the next chunk
and written to the usual transcript.

## 2026-09-01 — Preview mirrors retroactive masked-audio ownership

The masked AV latent assembly already used the new chunk's generated overlap
to replace the matching tail of accumulated audio, then trimmed that overlap
from the audio appended as the new chunk. The browser preview did neither side
of that transaction correctly: the preceding finalized WAV was never updated,
while the new chunk WAV was decoded only after the whole overlap. Speech that
began in the eight feathered audio ticks consequently disappeared from the
preview even though it remained in the final assembled latent.

For masked AV preview, the audio VAE now decodes the complete physical chunk
once. Python splits its PCM at the exact overlap-frame sample boundary. The
post-overlap portion belongs to the current preview group; the overlap portion
retroactively replaces the same-duration tail across one or more already
finalized preview groups. This can span multiple short groups when the overlap
is longer than one chunk's retained output. Each changed group is sent as an
audio-only update, and the server mutates its cached `chunk_final` payload so a
browser refresh keeps the corrected WAV without resending its images. The
frontend queues an update that races ahead of a large `chunk_final` event and
applies it when those authoritative frames arrive. Interrupted-render preview
restoration performs the same full-decode, split, and retroactive replacement
in chronological order.

## 2026-09-01 — Durable temporary videos for completed chunks

The sampler now reuses the authoritative full-video-VAE frames and decoded
audio that already finalize the live preview to write one crash-useful H.264
MP4 per completed chunk. It does not perform another VAE decode. The queued
ComfyUI graph is inspected through its hidden dynamic prompt: when this
sampler's timeline output is connected directly to `HR Endless Sampler Save
Video`, the intermediate writer reuses that node's literal
`filename_prefix`. A prefix supplied by a simple string primitive is also
resolved. With no connected Save Video prefix, no implicit output is written.

Files live beside the eventual final save and use deterministic names such as
`prefix_TEMP_chunk_003-of-010_frames_000073-000111.mp4`. Each file contains
only that chunk's retained output frames, optional synchronized AAC audio, and
an embedded plus sidecar Endless timeline. The local timeline retains the
global source-frame range and clips/remaps intersecting shot brackets, so an
intermediate can be opened in the existing Load Video player without implying
that it begins at global frame zero.

Encoding runs through one serial background worker, allowing the next H3
chunk to start while CPU H.264/AAC encoding proceeds. Sampler shutdown waits
for every submitted file, including after an exception, so a render failure
does not abandon already completed chunks. A genuinely fresh render deletes
only stale `prefix_TEMP_chunk_*.mp4` files and their sidecars for its resolved
prefix. Automatic interrupted-render resume preserves prior files and writes
or replaces the restored/missing chunk files atomically through `.part.mp4`
paths. The completed temporary set remains available until the next fresh run
with the same prefix.

## 2026-09-02 — Corrected decoded IMAGE and AUDIO sampler outputs

`HR Endless Sampler` now appends `images` (`IMAGE`) and `audio` (`AUDIO`)
outputs after its original four outputs. The latent, denoised latent, prompt,
and timeline socket indices are unchanged, preserving existing workflow
connections. The new media outputs reuse each chunk's authoritative full-VAE
finalization decode; they do not invoke a second whole-video VAE decode.

Completed video chunks receive adaptive exposure correction in linear RGB.
The first implementation compared the protected physical overlap against the
same finalized timestamps. That measurement always returned approximately
`1.0`, because keyframes/masking already make those discarded frames match;
the visible exposure shift begins only in the first newly generated frame.
The next experiments compared multi-frame windows and then extrapolated the
linear-light trend from the final eight frames into the first five generated
frames. Both could infer gains below `1.0` from ordinary motion and actively
worsen the measured darkening. The next deliberately simpler experiment used
exactly the previous chunk's final retained frame and the current chunk's first
retained frame. It measured one robust linear-light midtone level per image,
calculated `previous / current`, and applied that gain to the current same-shot
output. Crushed blacks and clipped highlights were excluded and the gain was
bounded to a conservative range. The protected overlap remained byte-for-byte
untouched.

Analysis of saved render `HR_Endless_Sampler_00015_.mp4` showed that boundary
matching alone was insufficient: roughly 56% of its first-to-last luma loss
occurred inside chunks, and excluding dark pixels made the earlier "robust
midtone" measurement remain nearly constant while total display brightness
fell. The active implementation now includes every pixel in full-frame linear
luma. It calculates a start gain from previous-final/current-first and an end
gain from previous-final/current-last, then interpolates exposure smoothly in
log space across the retained same-shot portion. This aligns the seam and
compensates an internal chunk fade without touching the protected overlap.
Both gains are clamped to `0.80..1.25`. Authored shot-cut boundaries still
receive no correction, and correction stops before a later cut inside a chunk.

The resulting gain is applied only to retained frames belonging to the same
source shot as the chunk boundary and stops before the next authored cut. A
chunk beginning on an intentional cut receives no correction from the prior
shot. Corrected retained frames are then sent to the browser, temporary chunk
writer, color diagnostics, Gemma's next visual observation, and final `IMAGE`
batch. Replay restoration decodes and corrects cached chunks chronologically
through the same path, so an automatically resumed run does not mix corrected
new chunks with uncorrected restored media.

The `AUDIO` output similarly reuses the full audio-VAE decode used by the final
preview. In masked-overlap mode, its decoded overlap replaces the matching tail
of the already accumulated waveform before the post-overlap samples are
appended. Thus the returned waveform includes the same trim, feathered-overlap
replacement, and seam handling heard in the finalized preview. If either VAE
is disconnected or its decode fails, the corresponding decoded output is
unavailable while latent sampling and recovery remain valid.

All returned images and audio remain on CPU. This saves redundant VAE work and
ensures Save Video can consume exactly the corrected preview media, but a
ComfyUI `IMAGE` output is one complete float32 batch: system-RAM use therefore
scales directly with resolution and video duration. Keep the latent outputs
for workflows where retaining a full decoded batch is too expensive.

### Output-only seam correction revision

The earlier interpolated start-to-end gain curve was removed. Although it
improved some whole-chunk statistics, it altered a chunk's internal lighting
progression and allowed corrected decoded pixels to become the next H3 sparse
guide. H3 now always receives raw decoded images for masked AV sparse
keyframes: the experimental 1.055x guide pre-gain is gone entirely.

The completed chunk's first newly generated frame is compared against the
previous finalized output frame in bounded linear RGB. One fixed shared gain
and three bounded RGB offsets are fitted for its same-shot output portion by
scoring RGB mean, luma mean, median luma, and display-black clipping. The
single transform is applied uniformly only after H3 has sampled; it is never
fed to H3. The logs report raw, corrected, reference, and residual RGB/luma/
black statistics. This targets an exact colour/tone statistic match at the
seam, not impossible pixel identity between two moving frames. A source cut
still receives no correction and the correction stops at the next cut.

### Display tone-curve revision

The first output-only RGB gain/lift revision could match linear average luma
while leaving a visibly different shadow floor, contrast, saturation, and
display histogram at a boundary. In one live Chunk 1 -> 2 measurement, its
linear luma mean matched (`0.04877` corrected versus `0.04877` reference),
but black clipping changed from `1.212%` to `1.843%` and the display luma
histogram distance remained `0.0940`.

It was therefore replaced with one fixed display-referred three-point tone
curve per same-shot output segment. The curve maps each newly generated
boundary frame's 5th-percentile shadow, median midtone, and 95th-percentile
highlight to the preceding finalized frame, then applies a bounded RGB channel
balance. It is monotonic and limited to a `0.75..1.30` luma ratio and
`0.94..1.06` channel balance, so it cannot invert tones or make extreme colour
changes. It is still strictly post-sampling: raw VAE frames alone remain H3
conditioning/keyframes. The focused helper test suite passed 60 tests after
this replacement.

## 2026-09-22 — The selective-LoRA panel is the editor, in both frontends

`web/selective_lora_ui.js` is now the whole feature. Earlier revisions drew the
group bars on the node canvas and then as a Nodes 2.0 DOM overlay behind the
real sliders; both fought the frontend for pixels and lost. The panel is one
DOM widget instead, so it renders the same way in the classic canvas UI and
under Nodes 2.0, and every pixel inside it is ours.

Each group row paints its share of the adapter as its own background gradient:
the bright band is the share the current strength keeps, the dim band is the
part a reduced strength drops, and the rest is an empty track. The radar draws
the adapter as trained (dashed grey) against the current settings (yellow),
scaled by a shared peak so a strength above 100% rescales instead of clipping.
The five strength widgets stay in `node.widgets` - their values must keep
serializing and `load_lora` must not change - but they are labelled with
`widget.label` and removed from both frontends with `widget.hidden = true`.
Each group therefore has exactly one editor, the panel, and the graph can never
disagree with it.

The percentage field at the right of each row is that editor. Committing a
value writes `widget.value`, rounded to the widget's own `round`/`precision`
and clamped to its `min`/`max`, then calls `widget.callback` so extensions
watching the node still see the edit. Enter commits, Escape restores the text,
the arrow keys step by the widget's real fine step (`options.step2`, which the
frontend rewrites from the declared `step`), shift and PageUp/PageDown step by
its tenfold coarse twin, a double click resets to 100%, a vertical drag of four
pixels per fine step scrubs, and blur restores the stored value in case
`change` never arrived. A strength change never refetches; only the LoRA picker
does.

Frontend behavior this relies on was read from the installed
`comfyui_frontend_package` 1.53.6 sources (extracted from its source maps) and
then confirmed in a real browser:

- `widget.hidden` is honored by both frontends, and hidden widgets still occupy
  their `widgets_values` position, because `LGraphNode.serialize` skips only
  widgets whose `serialize === false`.
- `widget.serialize` and `widget.options.serialize` are separate flags: the
  first controls workflow persistence, the second controls API prompt
  inclusion. Passing `{ serialize: false }` to `addDOMWidget` only sets the
  options copy, because that argument becomes `widget.options`; the widget
  object itself must also be flagged or a stray empty value is appended to
  `widgets_values`.
- DOM widget room comes from `--comfy-widget-min-height` /
  `--comfy-widget-height` in the element's computed style, so the panel
  declares its own height.
- Nodes 2.0 reuses the real widget objects and drives its own number field from
  `modelValue`, so a write to `widget.value` shows up in both frontends.
- ComfyUI intercepts `wheel` at document capture over every DOM widget and
  re-dispatches it to the canvas. No wheel listener in a custom panel can ever
  run, so the panel has none, and scrolling over it zooms the graph as it does
  everywhere else. An earlier revision's scroll-to-scrub was dead code.

Verified in headless Chromium through Playwright against the live server, with
every non-GET request intercepted and answered locally so no probe could write
settings, workflows or queue state. In both frontends the node's sliders are
absent while the panel is visible and interactive, real pixel samples of the
rows match the computed geometry (`#4aa3ff` at 0.45 over the panel background
at 12% of the attention row, `#f0a03c` at 0.45 across 60% of the texture row),
every editing path lands in `widget.value`, `serialize()` returns the original
six values in their original order, and `graphToPrompt()` sends exactly the six
node inputs - never `lora_groups` - carrying the strengths the panel shows.
`tests/test_selective_lora_ui.js` was rewritten to drive the real module against
a small DOM stand-in, replacing the earlier tests of the removed canvas and
overlay paths.

Weight statistics still say where an adapter writes, not what it controls. The
bars and the radar are a map of stored strength, and camera attribution still
needs the GPU ablation.

## 2026-09-22 — `lms_r64` only fails on the full model because of the AdaLN stack

`minimax_h3_lms_v1.0_r64.safetensors` was trained against a `use_adaln_curves`
checkpoint. Its 258 modules were classified against every base model in
`models/unet/` with the loader's own `classify_module`, and the counts are
decisive:

- `minimax_h3_ref2va_bf16` (535 tensors): 208 modules load, **all 50
  `blocks.N.adaln_proj.linear` modules are dropped**.
- every pruned `int8_convrot` build (932 tensors, incl. fl2va and the Singularity
  v1.3 repack): **258 of 258 load**.

All four base models have exactly 50 blocks, `0..49`, and the adapter covers that
whole stack (attention `qkv_proj` + `out_proj`, mlp `fc1` + `fc2`, eight
`token_refiner` modules, one `adaln_proj.linear` per block), so the modulation
group is the only difference between the two runs.

The shapes say why. The pruned builds delete `time_embedder` and instead
condition on a learned look-up: `adaln_t_table [1025, 8]`, interpolated per
timestep and fed to `adaln_proj.linear.weight [96768, 8]`. The full build keeps
the MLP and takes `[96768, 2688]`. The adapter's modulation modules are
`A [64, 8] x B [96768, 64]` - rank 64 on an 8-wide input, so `B·A` spans the
input space completely and is a *rewrite* of the timestep-to-modulation map, not
a nudge. `AdalnProj` is what injects the timestep into every block
(`expand=6, modalities=3, hidden=5376` -> 96768 shift/scale/gate vectors), so the
modulation group is 49.965% of the adapter (309,683,200 params) and is the part
that speaks to schedule position.

The error log is therefore not noise, it identifies the mechanism. At patch time
`comfy/weight_adapter/lora.py:270` builds `mm(A, B)` -> `[96768, 8]` = 774144
elements and tries `.reshape(weight.shape)`; the `except` at line 285 logs
`ERROR lora <key> shape '[96768, 2688]' is invalid for input of size 774144` and
returns the weight **unpatched**. A live full-model run at 19:26:55-19:27:57
produced 330 such lines, block indices `0..49` only, ~6.6 per module from weight
re-patching under offload. Nothing else is ever reported, and the adapter still
changes the render through the 208 attention/mlp/refiner modules that do load.

That is a load-compatibility fact, not a causal one: it proves the modulation
group is absent on the full model and present on the pruned one, not that it
moves the camera. The pruned builds also differ architecturally (curve
conditioning, int8 convrot). The cheap discriminating run is the panel itself -
on the pruned model set the conditioning group to 0% and leave the other four at
100%. If the initial camera position stops changing, it is the modulation LoRA;
if it persists, it is the pruned model and the adapter only amplifies it.

The error is a property of the patch, not of any loader, and our loader only
looked immune because those runs had the conditioning group at zero. Three
independent confirmations:

- The log's own patch count tracks the group. `Model ... prepared for dynamic
  VRAM loading. <N>MB Staged. <N> patches attached.` reports 258 for
  `UNETLoader -> our loader, modulation 1.0` on the pruned model at 18:51 (zero
  errors - the shapes match there), 208 for every run with modulation 0 (19:13
  pruned, 19:19 and 19:20 full bf16), and 258 for the 19:26 full run, which is
  the first time any `ERROR lora` appears in the log. 208 is attention 100 +
  feed_forward 100 + token_refiner 8; the 50 modulation keys are the difference.
- `/history` holds the executed graphs. In `e7bf4c26` (19:26) the chain is
  `UNETLoader minimax_h3_ref2va_bf16 -> LoraLoaderModelOnly minimax_h3_lms_v1.0_r64
  @1.0 -> HREndlessSamplerSelectiveLora minimax_h3_taomate_3step -> sampler`, and
  the older selective node for `lms_r64` (attention 0, modulation 0) is orphaned,
  `consumed by None`, so it never ran. The standard loader cannot be told to skip
  a group, so it attaches all 258 and the 50 doomed ones fail per weight reload.
- Reproduced with no loader in the picture at all: calling
  `LoRAAdapter.calculate_weight` on a `[96768, 2688]` weight logs the identical
  line and returns the weight unmodified (`/tmp/hrlora/why_no_error.py`), and
  our loader's grouping of the same file on the full model is 50 modulation +
  100 attention + 100 feed_forward + 8 token_refiner, i.e. 208 attached at
  modulation 0 and 258 at modulation 1.

Because a failed patch is discarded and behaves exactly like an absent one,
`LoraLoaderModelOnly` at 1.0 on the full model is functionally the same as this
node with modulation 0 - just louder. So loader choice is ruled out as the cause
of the pruned-versus-full difference: every full-BF16 run so far, under either
loader, has had the modulation stack switched off.

## 2026-09-22 — The preview's L curve now covers latent previews too

`web/unlimited_preview.js` used to apply its browser-only `sRGB^0.45` curve only
to finalized decodes: `renderInverseGammaDisplay` took a `finalized` flag, the
playback and paused-frame paths passed `Boolean(group.finalized)`, step-graph
inspection passed `false`, and a test pinned the rule with the note that the
curve "must never alter latent step previews". The owner asked for the L button
to cover the latent preview, so that gate is gone.

The `finalized` parameter is deleted rather than ignored: `displaySource` now
takes `(source, valid, displayed)`, all three of its callers dropped their flag,
and `renderInverseGammaDisplay()` applies the curve whenever the toggle is on and
whatever image is on screen - live latent frames, tiny-VAE or Latent2RGB
previews, step-graph inspection, and decoded chunks alike. The toggle itself, the
persisted `inverseGammaDisplay` player state and the SVG filter are unchanged, so
saved workflows and browser restores are unaffected.

Why this is only a decision, not a correctness fix: the latent preview frames
arrive from `_latent_rgb_frames` (Latent2RGB) or a tiny VAE decode, and under
`linear_color_compute` the server restores decoded previews to display sRGB
while the latent path carries no such restore. Applying the same curve to both
therefore shows the same experiment on both surfaces rather than a color-managed
result - which is exactly what the button claims to be.

Verified in real Chromium (Playwright) with the filter markup and
`renderInverseGammaDisplay` lifted verbatim out of the module, so no ComfyUI
server was involved: the filter reports `color-interpolation-filters: sRGB` and
`exponent: 0.45`, the toggle sets `image.style.filter` to
`url("#hr-endless-inverse-gamma-1")`, a solid `rgb(64)` frame renders as
`rgb(136,136,136)` and `rgb(128)` as `rgb(186,186,186)` (both the
`pow(value/255, 0.45)` result within rounding), the curve survives an `image.src`
swap of the kind a latent frame update performs, and toggling off restores
`rgb(128,128,128)` with `filter: none`.
`tests/test_chunk_director_helpers.py::test_preview_inverse_gamma_toggle_applies_to_every_preview_image`
replaces the old assertion: it pins the three argument-free call sites, requires
`displaySource` to have three parameters, and forbids `Boolean(group.finalized),`
from returning to any display path.

## 2026-09-22 — The finished-video player's L button, and the filter id it needed

The owner asked for the preview's `L` button on the other nodes. Save Video, Load
Video and Video Compare turn out to share one extension and one player
(`HREndlessSampler.FinishedVideoPlayer` in `web/finished_video_player.js`), so
this is one implementation, not three. Its viewport holds two `<video>` elements:
`media`, the primary render, and `compareMedia`, inset over it with a `clip-path`
that `renderComparisonWipe` rewrites on every drag, right-click orientation
change and comparison selection. Compare nodes drive that same pair from
`compare_media_url` instead of a matching-list picker.

The markup, the `color-interpolation-filters: sRGB` declaration and the three
`0.45` gamma curves are the preview's, verbatim. The button sits in the viewport's
top-right corner exactly as the preview's does, but at `z-index: 7` instead of
`2`: this viewport also contains an interactive wipe overlay whose 28-pixel hit
line is a live pointer target, and a lower button would lose clicks that land on
it. `renderInverseGammaDisplay` writes the filter to **both** videos, which is
what makes an active wipe (primary on one side, comparison on the other) curve as
one image; it deliberately does not touch `clip-path`, so the clipped side stays
clipped. The toggle is persisted as `inverseGammaDisplay` alongside `frame` and
`muted` in `node.properties.hr_endless_sampler_player`, so it survives a refresh
and a workflow save like the preview's does.

An earlier probe had left the `clip-path` question open, and it was a wrong
expectation rather than a real failure: it stacked a filtered `rgb(64)` under a
filtered `rgb(128)` and then compared the `rgb(128)` it sampled on the clipped
side against the `rgb(64)` prediction. Re-probed with the two videos as distinct
solid colours (`/tmp/hrlora/video_clip_probe.mjs`, real Chromium, no server), all
six checks pass and the two properties are independent: `186` inside the clip
polygon, `136` outside it (the bottom video), and `128`/`64` with the curve off.
`filter` and `clip-path` compose; the clip does not suppress the curve, and the
curve does not defeat the clip.

### The filter id could not be `node.id`

The first live run pinned the id as `hr-endless-inverse-gamma--1`, and that is a
fact about `onNodeCreated`, not about the probe: `LiteGraph.createNode` runs
`onNodeCreated` before `graph.add` assigns the real id (and `LGraph.configure`
sets it after `createNode` has already run), so every one of the four node types
reports `idAtCreation: -1` and only gets `1..4` afterwards
(`/tmp/hrlora/filter_id_probe.mjs`). Writing `${node.id}` into the filter id would
therefore give every player in the page the same id.

Chromium's behavior for that was measured rather than assumed
(`/tmp/hrlora/dangling_filter_probe.mjs`, four of four checks): with duplicate
ids, `url(#id)` resolves to the **first element in tree order** - a first copy
with the `0.45` curve won over a second copy carrying an identity curve, `186`
versus `128`; removing the resolved element silently **rebinds** to the next
match; and with no match at all Chromium **ignores** the filter, rendering the
element unfiltered (`128`), rather than hiding it. So a shared id does not blank
the media out, and the curve could stop applying with nothing but a stale-looking
button to show for it.

The player therefore takes an id from a module-level serial,
`hr-endless-player-inverse-gamma-<n>`, which is correct at creation time by
construction. `unlimited_preview.js` still writes `${node.id}`, and two preview
nodes in a live page do put the same `hr-endless-inverse-gamma--1` into the
document (`/tmp/hrlora/preview_id_probe.mjs`). That is benign where it is used -
each node contributes its own identical filter element, so the reference always
matches something, and all the definitions are the same - but duplicate element
ids are invalid HTML, and aligning the preview with the player is a separate
decision that this change did not take.

### Verification

Live, in the real frontend, against a real server, with every non-GET request
intercepted by the harness (`/tmp/hrlora/live_l_button_probe.mjs`, 17 checks,
none failing). Because the browser's real `/view` route serves
`/NVME/comfyui/output` — not `ComfyUI/output`, which is why every file in
`ComfyUI/output` 404s — the probe points the players at deterministic solid-colour
H.264 clips it serves itself over GET. Scenario A, a real `HREndlessSamplerSaveVideo`
node with one video: the button is in the document, off to start, a real mouse
click turns it on, the referenced id exists exactly once, all 49 sampled pixels of
the content box become `curve(128) = 186`, and a second click restores the
untouched decode byte for byte. Scenario B, a real `HREndlessSamplerVideoCompare`
node with the wipe active: the primary reads `128` on the left and the comparison
`64` on the right before the click, and `186`/`186`/`136`/`136` after it — each
half curved inside its own clipped region; the clip path string is unchanged by
the filter write; the second node gets `-2` rather than reusing `-1`; and after
dragging the divider to `0.961` so the 28-pixel hit line sits exactly on the
button centre at `0.961`, a click there still toggles the curve and leaves the
dragged clip path alone.

`tests/test_chunk_director_helpers.py::test_finished_video_player_inverse_gamma_toggle_covers_both_videos`
pins the new surface: the button text, the serial-based id and why (with a comment
naming the `-1` cause), `sRGB`, the argument-free render function, a filter write
to **both** videos, the untouched `compareMedia.style.clipPath` write, `z-index:7`,
the persisted and restored flag, and exactly two call sites. The module's failure
set is unchanged by this work: the same six failures and one error that the
uncommitted tree already had (the four camera-sentence tests, the one-based seed
multiples, the schema ordering, and the compute-precision error), with both curve
tests green.

## 2026-09-23 — Chunk dialogue splits at the clock again, not at the sentence

The owner asked for the one-sentence-per-chunk dialogue allocation to be
reverted: "in fact the code tries to put one sentence per chunk. I think we need
to revert that and split the sentence as we used to do before." The
sentence-preserving allocator was TaoMate-only plumbing — `sentence_word_owners`
in `python/dialogue_timing.py`, threaded through
`_legacy_dialogue_for_range(sentence_chunks=...)`, `_legacy_shot_body_for_range`,
`_legacy_opening_body_with_timed_dialogue`, `_planned_chunk_prompts(...,
taomate=...)` and the sampler call site at `nodes.py:4825` (mirrored by
`python/preproduction.py:151`) — and all of it is now removed rather than
defaulted off, so no per-method fork survives.

### Why the revert was warranted

The evidence is the successful 68-minute run itself (history entry
`671c6b35-a954-4972-ae71-79c0b77a5ad4`). Node `141 HREndlessSampler` has **no
`pre_production` input wired**, so `legacy = pre_production is None` is `True`
(`nodes.py:4836`) and the sampler plans its own prompts; node `2600
HREndlessLegacyChunkPrompts` is not in that graph at all. Replaying the run's
real prompt (`graph["138"]`), its real `TaoMateStreaming.request_plan(427, 2417,
124, 39)` and its real `_planned_chunk_prompts` reproduces the 13 `h3_prompt`s
the run recorded in `outputs["144"]...["timeline"]` exactly, so the planner under
test is the planner that ran.

That replay exposes two defects, both attributable to the sentence rule:

1. The final chunk is a 17-frame stub (frames 1433–1449, a 0.708 s window) handed
   1.510 s of speech — **slack −0.802 s**, the only negative slack in the run.
2. It severed "…talk about death and | suffering and why light is bad for you…"
   across a seam purely to keep sentences whole.

One earlier in-session claim was wrong and is corrected here:
`sentence_word_owners` compares *stretched* durations, so 4.630 s × 1.694 =
7.843 s against the 7.750 s threshold (1.5 × the 5.167 s nominal). The sentence
could legally have been moved into chunk 12 — a knife-edge split, not an
impossibility, and not proof that the rule was working as intended.

### The revert

- `_legacy_dialogue_for_range` (`nodes.py` ~1335) lost its `sentence_chunks`
  parameter and its `sentence_word_owners` branch; the proportional
  `cumulative_weights`/`desired_weight` walk is back, with a comment recording
  that sentence-preserving allocation was tried and reverted because it starved
  the short tail chunk. The overflow `ValueError` (~1361) is unchanged.
- `owns_prefix` no longer keys off `sentence_chunks`, so prefix carry-over
  applies under TaoMate again.
- `python/dialogue_timing.py` (2159 bytes) is deleted; `sentence_word_owners` was
  its only symbol.
- The two TaoMate sentence tests were rewritten rather than deleted, so both
  behaviours stay pinned: `test_dialogue_boundary_splits_a_sentence_when_the_chunk_clock_requires_it`
  and `test_short_final_chunk_receives_its_proportional_share`. Both pass
  explicit `dialogue_ranges`, because without them each chunk still receives the
  full utterance — which is what the neighbouring proportional tests rely on.

### Verification

`py_compile` is clean on `nodes.py`, `python/preproduction.py` and
`tests/test_chunk_director_helpers.py`, and no occurrence of `sentence_chunks`,
`sentence_word_owners`, `dialogue_timing`, `taomate=taomate` or
`taomate=use_taomate` remains anywhere outside `vendor/`. The new tests pass. The
module's failure set is byte-identical to the uncommitted tree's: 132 tests, the
same six failures and one error (the four camera-sentence tests, the one-based
seed multiples, the schema ordering, and the compute-precision error).
`node tests/test_legacy_chunk_prompts.js` and
`tests/test_global_prompt_sections.py` both pass.

Replaying the reverted planner end to end at 1,450 frames (60.417 s; 13 chunks:
124 then 119 × 11 then 17) gives **min slack +0.133 s**, up from −0.802 s. Chunk
13 becomes a 0.708 s window carrying 0.575 s of speech, "about."; sentences now
split at seams (chunk 1 ending "…whole minute. That's an", chunk 12 "This is
torture. Finally! something worth talking", chunk 13 "about."), and the
"dialogue already in progress" beat is emitted again. Adjacent durations behave
sensibly too: 1,416 → 12 chunks / 102-frame tail / +0.328 s, 1,433 → 12 / 119 /
+0.978, 1,467 → 13 / 34 / −0.093, 1,484 → 13 / 51 / +0.615.

What this does *not* establish is the GPU-visible result. Only the prompt text
was replayed; `tests/test_taomate.py` and `tests/test_video_io.py` (slow CPU
forwards) were not re-run, so the rendered effect of the revert is unverified.

Per the standing Gemma/MTP rule, issue
<https://github.com/ggml-org/llama.cpp/issues/27439> was rechecked before
touching `python/preproduction.py`: it remains `open`, labeled `bug-unconfirmed`
and `stale`, with zero comments and last updated 2026-09-20, and PyPI still
reports `llama-cpp-python` 0.3.35 (2026-08-17) as the newest release. The
disposable worker and the operation-local non-MTP retry are preserved unchanged;
`dependency.md` records the recheck.

Nothing here was committed or staged. The untracked `prompt.txt` and the owner's
own dirty `python/taomate_audio_teacher.py` were left untouched.

## 2026-09-26 — TaoMate reuses its teacher audio when prompt and seed are unchanged

The owner asked for an audio cache: "implement an audio cache for the teacher
audio, when the prompt and seed doesn't change, so the sampler can re-use the
audio. We also need to reset the cache when the cache button is disable and the
render starts. With taomate mode we have no video cache, but we still can make
use of a global teacher audio cache, can't we?" Four decisions were put to the
owner and all four recommended options were chosen: (1) the key covers the
source prompt, the seed **and** every other audio-affecting setting, (2) the
whole audio-first pass is restored, not just the latents, (3) the entry lives on
disk in the existing temp root, (4) the existing preview cache button becomes
TaoMate's teacher-audio switch.

### Why the owner's key is coherent

`gemma_director` is created with `seed=replay_noise_seed` (`nodes.py:5840`), so
the entire audio-first chain — Gemma's per-chunk prompts → teacher audio →
Whisper transcript → the next prompt — is reproducible from the source prompt
plus the noise seed. That is what makes reuse of a recorded pass legitimate
rather than a guess. The unavoidable caveat is the same one that already applies
to every Gemma path: the documented MTP instability can make a *fresh* run
diverge. A cache hit therefore freezes the direction the cached render recorded,
which is the requested behaviour, not an accident.

### What was built

- `nodes.py` gained `TEACHER_AUDIO_CACHE_DIRNAME` (`teacher_audio`) beside
  `last_run_replay`, a shared `_remove_cache_directory`, the
  `_TeacherAudioCache` class (manifest plus one `teacher_audio.pt` payload), the
  state global `_TEACHER_AUDIO_CACHE_STATE`, and `audio_cache` in
  `_replay_cache_ui_status_unlocked()` so the button can report it.
- The key is `_teacher_audio_key(chunk_prompts, chunk_ticks)`: the ordered
  per-chunk prompts the render will use (the "legacy chunk prompts" provider's
  editable table when one is connected, otherwise the legacy planner's output from
  `planned_prompts`), plus each chunk's audio length in ticks. Both lists exist
  before the pass starts. `_teacher_audio_key_difference` therefore reports
  "Chunk 4's prompt changed" — the unit the owner asked for. Cache format 3.
- **Why the first version of this key was wrong, and dangerous.** It hashed the
  source prompt only. The owner edits individual chunks in the
  `HREndlessLegacyChunkPrompts` node, which reaches the sampler as
  `pre_production` (the provider's 5.8 KB table at node 2600), not as `prompt`.
  Under the source-prompt key that edit would not have invalidated anything: the
  next render would have silently served the old chunk-4 audio. The key now
  includes the provider table, so an edited chunk always redraws.
- The key was `source prompt hash + chunk ticks` for one revision, which also
  ignored `video_continuation` (73 vs 56 produce identical request plans and thus
  identical audio windows). **This replaced a much stricter key**
  (the whole replay fingerprint, seed, sigmas, codec, model digest and toggles).
  The owner's rule: "drop everything but the chunk prompt" — users iterate on
  video parameters far more than on dialogue, and an audio they like should
  survive those changes. The per-chunk prompts themselves cannot be part of the
  key, because in the audio-first pass chunk N's prompt is authored from chunk
  N-1's transcript and so only exists while the pass runs; on a hit the cached
  prompts are restored, which is what the rule asks for. Changing `chunk_frames`
  still redraws the audio, because it changes the chunk count and their tick
  lengths. The cache button remains the way to force a fresh pass.
- **Why the strict key had to go.** On 2026-09-26 the 18:00 render missed with
  "prompt, seed, or a teacher-audio setting changed" although the Cached pass was
  exactly reusable: the source prompt hash read from the in-flight run's graph was
  identical (`989243807b5743e5`), as were the seed, sigmas, codec, model digest
  and toggles. The only difference was `video_continuation` 73 → 56, and
  `TaoMateStreaming.request_plan` produces byte-identical plans for both: same six
  requests, same frame windows, same audio ticks `[0, 433, 858, 1283, 1708, 2133, 2417]`,
  differing only in the video sub-chunk `phases`, which the teacher never reads.
  The render paid the full ~15-minute teacher pass for a parameter the audio pass
  does not consume. The miss now names what changed
  (`_teacher_audio_key_difference`), and the cache format is 2, so an old entry
  reports "format is obsolete" instead of pretending a prompt changed.
- Deliberate consequences of the looser key, so nobody rediscovers them as bugs: a
  changed seed, sigma schedule, KV codec, LoRA/model, resolution or compute
  precision now reuses the stored audio rather than regenerating it. That is the
  requested behaviour ("if they have an audio they like, they can stick to it"),
  and the button is the escape hatch. The payload is still validated for shape —
  chunk count and each chunk's tick length — because a restored milestone whose
  width differs from its chunk window would silently misalign the video pass.
- The payload holds only the sampler's own finished products: per-chunk
  guidance milestones, the clean teacher latent, the per-chunk teacher prompts,
  the Whisper transcripts and the decoded observation waveforms. Nothing there
  replaces a model or sampler input. The manifest is written *after* the payload
  so a manifest can never claim a pass that cannot be loaded.
- Reuse restores `audio_first_milestones` (list → int-keyed dict),
  `audio_first_ready`, `audio_requested_prompts`, `audio_transcripts` and
  `audio_observations`, republishes each chunk's prompt/subtitle/audio to the
  preview, and decodes the cached clean latent through the same
  `decode_audio_timeline` call the fresh path uses. Teacher inference, its Gemma
  audio requests, the per-chunk decodes and the transcripts are all skipped.
- The fresh path now saves the pass *before* decoding it, so a preview decode
  failure cannot cost the next render the whole teacher pass, and a storage
  failure is logged and the render continues rather than dying.
- Reset semantics match the replay cache contract: with the button off, the
  stored pass is cleared as the render starts and the run records a fresh
  replacement. `web/unlimited_preview.js` no longer disables the button in
  TaoMate, its tooltip explains that it now governs the teacher audio, and
  `cacheReuseLabel` shows "reusing cached teacher audio (N chunks)" from the new
  `audio_cache.run_state`.

### Trap that had to be checked

`execute_chunk` pops `self.audio_first_milestones.pop(self.current_chunk_number - 1)`
(`python/taomate.py:1103`) and `current_chunk_number` is *also* set inside
`prepare_audio_first`, which a cache hit skips. Had the video loop not set it
per chunk (`nodes.py:7537`), every reuse would have popped the wrong key. It
does, so the reuse path is safe; keep that in mind before moving either line.
Similarly, `audio_observations` must round-trip as *tuples* — the payload
validator rejects anything else, so `weights_only=True` load behaviour is now
pinned by a test.

### Verified, and not verified

`tests/test_teacher_audio_cache.py` (new, 11 tests) pins the round trip, the
JSON-serializability of the key, key sensitivity per input, identifier-independent
model signatures, rejection of changed keys / obsolete formats / truncated
payloads / mismatched chunk counts, missing and corrupt entries reporting instead
of raising, and the sampler wiring (including "a storage failure never stops the
render"). `tests/test_chunk_director_helpers.py` needed one update: it asserts the
exact status payload, so the new `audio_cache` block was added to that
expectation, and it now states the audio state it pins instead of depending on
test order. Full sweep is green: chunk-director helpers 147, TaoMate 41,
teacher-audio cache 11, every other python test exits 0, all eight JS tests pass.

**Not verified:** no live GPU render was run in this session. The fresh path's
new save call and the reuse path only execute inside a real TaoMate render, so
the first real render is the actual acceptance test. Reuse is also unmeasured —
only the code path was proven, not the wall-clock saving.

### Deliberate limits

- Only the audio-first path is cached. With
  `TOGGLE_TAOMATE_DIVERGENCY_AUDIO_FIRST` disabled, the per-chunk teacher audio
  is interleaved into the video loop in `execute_chunk` and is not cached.
- Replacing the base H3 weights while keeping the same LoRA, prompt and seed is
  not detected. The video replay cache has the same hole, so the cache button
  remains the manual escape hatch; adding a base-weight digest was rejected as
  an expensive scan of a lazily loaded model.
- A non-TaoMate render with the button off does *not* clear this entry; only a
  TaoMate audio-first render does. That scoping was chosen so an unrelated video
  render cannot destroy a good pass.
- Which chunks were retried for transcript quality (`audio_retried_chunks`) is
  not restored, because it only colours preview bands.

Per the standing Gemma/MTP rule, issue
<https://github.com/ggml-org/llama.cpp/issues/27439> was rechecked before
touching these Gemma-dependent paths: still `open`, labeled `bug-unconfirmed`
and `stale`, zero comments, last updated 2026-09-20, unsafely documented
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` restore path with no linked fix, and PyPI still
reports `llama-cpp-python` 0.3.35 (2026-08-17) as the newest release. The
disposable worker and the operation-local non-MTP retry are preserved unchanged;
`dependency.md` records the recheck.

Nothing here was committed or staged. The untracked `prompt.txt` was left
untouched, and the owner's own pre-existing uncommitted `nodes.py` work (the two
`_release_captured_graph_pools` calls and their helper) was preserved as found.

## 2026-09-26 — Source prose is owned by one chunk, and speech rate is directed

The owner reported that chunk 2 of a render was correct (camera line plus its own
dialogue) while chunk 3 "introduced back the state of subject 1 as described in
the global prompt", and asked whether a per-chunk table of the prompt's actions
could decide where each action goes. They then added two requirements: detect
parallel versus sequential actions ("Tila smiles, as the camera zooms in" versus
"The camera zooms to tilas face, and she smiles"), and stop treating speech rate
as fixed, because H3 can be told how fast a subject speaks. They supplied a new
`prompt.txt` (single-shot monologue) and moved the old seven-shot prompt to
`prompt-heman.txt` so both could serve as fixtures.

### Why the repetition happened

`_legacy_shot_body_for_range` selected a sentence for **every** chunk whose
proportional word window overlapped it (`offset < end_weight and offset + weight
> start_weight`), and `_legacy_opening_body_with_timed_dialogue` handed the chunk
where a shot opens the shot's **entire** visual prose. A long description
sentence therefore reached the opening chunk and then every later chunk whose
window reached it, while a shorter one in the middle could be skipped by the
chunk in between. Reproduced exactly, with the shot opening at chunk 3: the state
sentence appeared in chunk 3 and every chunk after it, and not in chunk 2. There
was no ownership anywhere in the visual path; the dialogue path already had it in
`_dialogue_owner_spans`.

### What was built

- `_action_clauses` splits a sentence into clauses and keeps each clause's raw
  connector, so rejoining reproduces the source punctuation byte for byte. The
  split group must **capture**: without the capture `re.split` returns no
  separators and the pairing silently drops half the text, which is how the first
  version lost the middle of a sentence.
- `_action_units` classifies each clause: `camera_state`, `camera_cut`,
  `camera_move`, `subject`, `other`. The camera vocabulary follows the vendored
  guides and shares `CAMERA_MOTION` with `_legacy_static_camera_prompt` instead
  of keeping a second copy of the regex.
- `_action_relation` decides parallel or sequential: sequential markers win over
  parallel ones, a camera cut is always a sequential boundary, a participial
  clause and a continuous camera move are parallel unless the text sequenced
  them, and an unmarked pair defaults to sequential (ordering two beats is
  recoverable, merging them into one time window is not).
- `_action_groups` returns the ownable parallel groups, the repeating camera
  states, and the opening setup clauses. A setup clause (`<Picture N>`,
  "starts with", "in silence for", "exact first frame") is neither repeated nor
  splittable, so it is pinned to the shot's opening chunk; this is what keeps
  `test_continuation_drops_complete_opening_picture_instruction_before_slicing`
  meaningful.
- A parallel group's weight is the **max** of its members, and sequential groups
  chain by sum, so a camera move running alongside its subject no longer charges
  the shot twice. That inflated estimate was a second cause of prose bleeding
  into later chunks.
- `_action_owner_index` is the dialogue walk applied to groups: whole groups, cut
  at the boundary nearest the output clock, at least one group per planned chunk
  while groups remain, never split and never dropped. `_legacy_action_plan`
  builds the table per shot across every planned chunk and returns
  window → prose; `_legacy_shot_body_for_range` and the opening wrapper now emit
  only their owned text. `_planned_chunk_prompts` builds the tables once and
  passes them in; `python/preproduction.py:154` inherits this because it calls
  that function directly.
- `_action_plan_rows` feeds the debug chunk prompt with shot, category,
  relation and owning window per unit.

### Three refusals replaced

1. **Speech rate.** `base-en.txt` §4.4 documents a speaking rate as a supported
   delivery attribute and requires delivery to sit **outside** `<d>`.
   `_dialogue_word_schedule` already stretches words into their window
   (`scale = available / total_weight`), so only the guard assumed a fixed rate.
   The guard is now a factor: 1.0–1.10 "speaking a little faster than usual",
   1.10–1.25 "speaking quickly", 1.25–1.40 "speaking very quickly, rushing the
   words", past 1.40 the render still stops with the shot named. The owner's
   Shot 3 case is 2.98 s in 2.75 s = **1.08×**, comfortably inside the first
   band. Slack between 0.60 and 1.0 asks for "speaking slowly, with pauses";
   below 0.60 the shot simply has sparse dialogue and no instruction is emitted,
   because telling H3 to drag each word out would make that worse.
2. **Fewer words than chunks.** The old guard raised when a shot had fewer words
   than output chunks. `_dialogue_owner_spans` is documented to leave the
   trailing ranges silent, which is exactly a long shot with one line, so the
   guard became a log line. Without this the seven-shot prompt could not plan at
   all: its Shot 7 is 43 s with 8 words.
3. **Delivery verbs.** `LEGACY_DIALOGUE_PATTERN` required the literal word
   "says", so `Heman (S1) shout:` and `Tila (S2) shouts:` made the prompt
   unparseable. The header now accepts the delivery wording the guides allow
   between the speaker ID and `<d>`.

### Product change to watch

The first video no longer receives the shot's complete description: ownership
gives every chunk its own units, including the first one. The previous behaviour
was deliberate ("the first H3 video establishes the shot from the user's complete
source description") and its test was rewritten to the ownership contract rather
than deleted. For both fixtures the change is invisible, because each opening
shot fits inside its first chunk, but a long first shot now spreads its prose
progressively. Reverting just that rule is a one-line change in
`_legacy_action_plan`.

### Verified, and not verified

`tests/test_chunk_director_helpers.py` is 159 tests, all passing, including five
rewritten pinned tests and five new ones: no action unit announced by two chunks,
parallel group unsplit, sequential order and connector preserved, camera cut as a
sequential boundary, still-camera state repeating while actions do not, setup
pinned to its opener, and the rate bands. `test_taomate.py` 41, teacher-audio
cache 11, every other python test exits 0, all eight JS tests pass. Across both
real prompts the ownership table reports **32 action groups with 0 announced
twice**, and the seven-shot prompt now plans at both geometries (14 and 17
chunks) instead of stopping.

**Not verified:** no live GPU render. The prompts were planned, not sampled, so
the rendered result of the new distribution and of the delivery instructions is
unmeasured. Known remaining weakness: `_dialogue_word_schedule` spreads a shot's
words across the whole shot, so a long shot with one line still gives each chunk
a word or two (Shot 7 of the seven-shot prompt), and the fix for that is to clip
the utterance span rather than stretch it — not attempted here.

Nothing was committed or staged. Both prompt files were read as fixtures and left
untouched. The Gemma/MTP upstream check recorded earlier in this same session
still applies; this change touches prompt construction only and no llama.cpp,
worker, cache or retry behaviour.

## 2026-09-26 — Dialogue cuts land on phrase terminals, and retries change the rate

The owner watched chunk 3 lose the word "silence" on every take and asked for
three things: never split a phrase between chunks so a chunk always ends with
punctuation, make a retry raise the speaking-rate instruction instead of changing
the text, and use "speaking much faster than before" for that retry because the
take's own audio is in the retained KV cache, so the model can measure "before".

### The live log evidence

`/NVME/comfyui/ComfyUI/user/comfyui.log` and its two predecessors, render of
17:12-17:19, six chunks, `chunk_frames=260`, `video_continuation=73` (geometry
recovered by matching `TaoMateStreaming.request_plan` against the teacher's
logged audio ticks). Every chunk whose slice opened with a carried seam word lost
exactly that word, before and after the retry:

| chunk | expected head | heard | lost |
|---|---|---|---|
| 2 | `and why light is bad…` | `why light is bad…` | "and" |
| 3 | `silence comforting…` | `comforting…` | "silence" |
| 4 | `is shining. The birds are singing. Disturbing…` | `Disturbing…` | "is shining." **and** the new sentence "The birds are singing." |

Chunk 3 take 1 was scheduled cleanly — "silence" at 21.458s-22.039s in a window
ending 32.083s — so it was never a timing slip. The slice opened mid-phrase and
the prose said "The dialogue is already in progress at this video opening."; H3
obeyed by starting at the next complete unit. `_recut_dialogue_prompts` rebuilt
the slice from `leftover[span_start:span_end + 1]` with `span_start = 0`, so the
same head word came back and the retry could never fix it — chunk 3's retried
take dropped "silence" again, chunk 4's dropped two sentences again.

### Changes

- `_dialogue_owner_spans` now cuts only on a phrase terminal: it picks the
  terminal nearest the output clock instead of the nearest word. `_dialogue_phrase_ids`
  already marks a phrase end at `. ! ? …`, so every slice finishes on punctuation.
  A phrase that cannot fit its window overruns it rather than being split, which
  is what the speaking-rate delivery now covers. `_seam_orphans_a_phrase_word`
  became unnecessary and was deleted.
- `_escalate_speech_rate(prompt)` replaces whichever band the prompt carries with
  "Speaking much faster than before", outside `<d>`, or attaches it to the
  speaker header when no band was emitted. The spoken words are untouched.
- Both retry paths (`retry_audio_first_candidate` for audio-first, and
  `retry_chunk_teacher_audio` for the per-chunk teacher) now keep the dialogue and
  escalate the delivery. The "approved spoken prefix" branch and its text
  re-authoring are gone: a retry means the take lost or garbled words, not that
  the words were wrong, so the words stay.
- Deleted as dead once the retry stopped rewriting text: `_recut_dialogue_prompts`,
  `_retry_take_prompts`, `_publish_revised_dialogue`, `_audio_transcript_approved_tail_prompt`
  (124 lines). They remain in git history if the prefix-acceptance idea returns.

### Verification

On the live monologue prompt every chunk slice now ends on a phrase terminal —
"chunks ending mid-phrase: 0" — and chunk 2 ends with the whole phrase
`...Personally, I find silence comforting.` while chunk 3 starts on a fresh
sentence instead of a borrowed fragment. Escalation on chunk 3 turns "Speaking
slowly" into "Speaking much faster than before" with the spoken words byte-identical.
`tests/test_chunk_director_helpers.py` is 158 tests passing: five obsolete
re-cut tests were replaced by three escalation tests and a
phrase-terminal test, and six tests that pinned mid-phrase splits were rewritten
to the phrase-aligned contract, including the property that a phrase which
overruns its window may only be restated with words its own chunk already spoke.
TaoMate 41, teacher-audio cache 11, every other python test exits 0, all eight JS
tests pass.

**Not verified:** no live GPU render of these changes yet; the cuts and the
escalation were verified by planning, not by sampling.

### Per-chunk teacher audio KV records and a resumable pass

The owner asked "aren't we caching each chunk kv? audio only kv-cache is not that
big", and they were right that no KV was cached anywhere. **The size estimate in
the first draft of this note was wrong by 10x**: measured from a real record, the
audio attention has **56 heads** and the teacher's commit captured **866 tokens for
a 433-tick chunk** (two per tick) across 50 layers, so a chunk's rows are ~250 MB
at `turboquant3` (3 bits, codes shape (tokens, 56, 48)) and a 60-second video is
**~1.4 GB** — a three-minute video ~5-6 GB. That is the real price of exact prefix
reuse, and it needs a disk budget decision rather than a cheerful assumption. It is
still ~10x smaller than the video KV, but not "not that big".

### Why a browser refresh shows a week-old cache

Driving the owner's report "when I refresh the browser it loads a really old cache
(which apparently is never getting deleted) and doesn't show the last rendered
cache (the audio?)":

- On load the widget asks `/hr_endless_sampler_preview/state?node_id=…` and, when
  ComfyUI no longer has that in-memory snapshot, falls back to
  `/hr_endless_sampler_preview/cached_preview` (`web/unlimited_preview.js:2319-2327`,
  route at `nodes.py:559`).
- That route rebuilds the player from the **finalized replay checkpoint** —
  `_LastRunReplayCache()` at `nodes.py:1257` → `build_cached_final_preview_snapshot`
  in `preview.py`, which publishes each cached chunk's frames *and* audio.
- TaoMate renders never write that checkpoint
  (`replay_cache_enabled = _replay_cache_enabled() and not use_taomate`), so the
  one on disk is frozen at **2026-09-19 16:32**, and its manifest says
  `status="interrupted"` with one completed chunk: a refresh therefore plays a
  week-old half-finished run, audio included.
- The last render's teacher audio lives in `teacher_audio/`, which that path never
  reads, so the newest audio cannot appear. When the entry is also incomplete (only
  `kv/` records, as while a pass is still running) there is nothing recent to show
  at all.

- `serialize_stored_kv` / `deserialize_stored_kv` (`python/taomate.py`) describe
  the three storage forms: quantized (codes + scales), compressed (codec frames
  as bytes) and raw. `tests/test_taomate_kv_records.py` proves a save/load round
  trip is bit-identical for turboquant, int8, zstd-lossless and none, codec bytes
  included, through the sampler's own `weights_only=True` loader.
- `TaoMateKVCache.staged_snapshot()` and `prime_staged()` snapshot one chunk's
  rows before `commit()` consumes them and later accept them as if that chunk had
  just been generated; priming two chunks reproduces a two-chunk history exactly.
- `Base10AudioTeacher.on_kv_commit` hands each chunk's rows to the caller before
  the commit, and `pending_prime` commits stored rows the moment the cache exists,
  which is the first attention forward of the resumed chunk.
- `prepare_audio_first(..., start_index, prefix_segments, prefix_origin,
  prime_records, on_kv_commit)` resumes a pass: chunks before `start_index` keep
  their clean rows as the teacher's context, and `_TeacherAudioCache` stores one
  record per chunk beside the payload.
- Per-pass state that a resume must restore, or the continuation lands on
  different coordinates: `audio_time_origin` (the rope origin, set once from the
  first chunk's text length) and `audio_anchor` (the renorm mean/std). Both are in
  the payload now.
- `_teacher_audio_resume_index` returns the first chunk whose prompt hash differs,
  and only when the chunk tick lengths are unchanged and every earlier record
  exists. Otherwise the render falls back to a full pass, which is always safe.
- **A trap worth remembering:** `save()` used to start with `self.clear()`, which
  would have deleted the KV records written during that same pass. It now replaces
  only the manifest and the payload, and a fresh pass clears the records
  explicitly through `clear_kv_records()`.
- Past twelve chunks the pass calls `drop_audio_history`, so records from before
  the last drop boundary no longer describe the KV: the save path discards them,
  and the gap rule then refuses a resume that would need them.

**Not verified:** a live resumed render. The serializer, priming, decision and
prefix validation are tested offline; only a render can confirm that the resumed
pass produces the same audio for the settled chunks. What to look for in the log:
`keeping Chunks 1-3 from the stored pass and resuming the teacher at Chunk 4 of 6`,
`primed 3 stored audio KV chunks before resuming`, and a teacher progress bar with
fewer steps than the chunk count.

### One last cache, newest source on refresh

The owner's rule after the diagnosis: "only one last cache stays. If a render that
doesn't use cache starts, that last is deleted since the new non-cached render will
create a new one." Implemented as three changes.

- **Refresh shows the newest cache.** `_newest_cached_preview_source()` compares the
  teacher-audio manifest against the replay checkpoint and
  `/hr_endless_sampler_preview/cached_preview` builds from the winner, falling back
  to the checkpoint if the audio snapshot cannot be built. The teacher-audio payload
  now carries a `preview` block (fps plus per-chunk start/end/prompt/subtitle), and
  `build_teacher_audio_preview_snapshot` in `preview.py` replays the live
  audio-first sequence (`reset` → `chunk_audio_first` → `audio_chunk_complete` →
  `complete`) so a refreshed player shows the last TaoMate render's chunks, prompts
  and audio. No frames, by design: a TaoMate render keeps none.
- **Button off deletes every last cache as the render starts.** The rule lives once,
  next to where the button is read: `_remove_replay_cache()` +
  `_remove_teacher_audio_cache()`. The audio-first block no longer clears anything
  itself, so there is a single place that decides this.
- **An incompatible cache warns instead of failing silently.** When the button is on
  and a stored entry exists but cannot be used, the state is `incompatible` and the
  preview shows a yellow line naming the reason and saying the render overwrites it.
  A missing entry stays a plain `miss`.
- KV records stay inside the cache folder (`teacher_audio/kv/`) and are removed with
  it, as asked; no separate pruning.

### Follow-up worth doing

The delivery band is still computed per **shot** (`total_weight / available_seconds`
over the shot's dialogue window), while the cut is now per phrase. A chunk can
therefore own more speech than its own window while the shot-level rate stays
under 1.0, and nothing asks that chunk to speak faster until the retry does.
Computing the band per chunk from the words it owns would prevent the drop
instead of repairing it. The owner also noted that the monologue's chunks all
carry "speaking slowly, with pauses"; that band came from the shot-level slack,
and a per-chunk rate would change it.

## 2026-09-26 — Add 3-bit and 2-bit TurboQuant KV modes

The owner asked to expose 4-, 3-, and 2-bit TurboQuant choices, rename the
existing `turboquant` choice to `turboquant4`, and make `turboquant3` the
sampler default.

- Added the fixed Gaussian Lloyd-Max centroids used by the upstream
  `pyturboquant` codebook for 2 and 3 bits, reusing the existing 4-bit table.
  The 3-bit packer writes each 8-index group into three bytes; 2- and 4-bit
  packers use 4- and 2-index groups respectively. All three unpack back to the
  128-wide H3 attention head.
- The sampler menu and execution default are now `turboquant3`; `none`,
  `zstd lossless`, and `int8` remain. Saved quantized records using the former
  name `turboquant` deserialize as `turboquant4`.
- A resumed audio cache can contain an older TurboQuant width than the newly
  selected mode. At that boundary, the old and new rows are reconstructed on
  CPU and re-encoded in the active mode; same-width rows continue to append
  packed data directly.
- Expected retained K/V storage, including two FP32 norms per vector pair, is
  26.6% for 4 bits, 20.3% for 3 bits, and 14.1% for 2 bits relative to BF16.
- The first full TaoMate suite run found three tiny-head test fixtures inheriting
  the new 3-bit default. The user-facing sampler default is the requested 3-bit
  mode; the no-argument backend fallback was restored to its prior setting so
  these internal fixtures keep their old behavior.

Validation: `tests/test_taomate_kv_records.py` (10 tests) and
`tests/test_taomate.py` (41 tests) pass. The record tests verify exact packed
index round trips, stored byte counts, legacy 4-bit records, mixed-width cache
continuation, and save/load. `py_compile` and `git diff --check` pass. No live
render comparison has been run for the new lower-precision modes.

## 2026-09-26 — Why a stopped render kept losing its cache, and the fixes

The owner stopped an audio render, restarted ComfyUI, started a new one and watched
it recompute everything, not delete the old video checkpoint, and then saw a
refreshed browser restore that week-old checkpoint instead of the audio render.
Three separate mechanisms produced that, in this order.

- **The cache button lived in the process only.** `REPLAY_CACHE_ENABLED` is a module
  global set at import, so a ComfyUI restart silently flipped it back to *enabled*:
  the delete branch (`if not _replay_cache_enabled()` → `_remove_replay_cache()` and
  `_remove_teacher_audio_cache()`) never ran, and the banner the owner saw said
  cache reuse was on. The endpoint also refuses a change while a render is active
  (`409`, "Cache policy can change only between sampler renders"), so clicking it
  mid-render bounced back. Fix: the choice is written to
  `<temp>/comfyui-hr-endless-sampler/cache_policy.json` by
  `_save_cache_policy` and restored by `_load_cache_policy()` at import.
- **A stopped pass had no identity.** KV records were written per chunk during the
  pass, but the manifest and payload only appeared in `save()`, which runs after the
  whole teacher pass. Stopping at Chunk 6 of 6 therefore left 1.4 GB of records with
  no manifest: invisible to the status, invisible to
  `_newest_cached_preview_source()` (so the refresh fell back to the Sep 19 replay
  checkpoint), and a miss for the next render, which then called
  `clear_kv_records()` and deleted them.
- **A resume could not have worked anyway.** Two defects: the resume index needed an
  *edited* prompt (an identical key returned 0, so a stopped pass restarted at Chunk
  1), and `prepare_audio_first` reset `audio_first_milestones = {}` after the caller
  had restored the prefix, while `execute_chunk` pops one entry per chunk with
  `audio_first_milestones.pop(...)` — a resumed render would have raised `KeyError`
  on its first video chunk.

Implemented, in the order the plan table listed them:

- **Pass identity from the first chunk.** `_TeacherAudioCache.begin_pass(key,
  chunk_count, resume_index)` writes `manifest.json` with
  `{key, chunks, recorded, status: "recording", deleted}` before sampling. A fresh
  pass (`resume_index == 0`) drops the previous payload and records first, exactly as
  the old `clear_kv_records()` call did; a resumed pass keeps the prefix and drops
  the records after the resume point, because those rows describe the take this
  render is replacing. `save()` finally writes `status: "complete"`.
- **The settled prefix is written as it grows.** `save_chunk_state(index, payload)`
  rewrites the payload (same mapping as `save()`, without `audio_latent`) at the end
  of every chunk, from the `teacher_audio_payload()` closure both writers share. The
  write is deliberately not atomic: a process killed inside it leaves a payload the
  loaders reject, which costs one reuse, while a temp file plus rename would cost a
  second copy of every tensor.
- **Resume index semantics.** `_teacher_audio_resume_index` now keeps
  `min(settled, edited, first deleted)` chunks: an identical key on an unfinished
  pass continues at its stop point, an edited prompt still resumes at the first
  changed chunk, and a chunk deleted from the timeline forces that chunk (and every
  chunk after it) to be generated again. A *finished* pass whose key is identical
  still returns 0 — that case is the whole-pass reuse `load_if_compatible` already
  handles, and returning 0 there avoids resuming from a payload the loader rejected.
  `load_if_compatible` refuses a payload whose manifest lists deleted chunks, naming
  the chunk.
- **Prefix milestones survive the resume.** `prepare_audio_first` keeps the caller's
  restored entries (`index < start_index`) instead of clearing the dict.
- **A cache a render cannot use is deleted at render start, never left behind.** The
  owner's rule, stated twice and now enforced in both places: *a cache is either
  reused or deleted*. TaoMate renders carry no video cache and cannot continue one,
  so the replay checkpoint goes at render start even with the button on (it is the
  checkpoint a refresh kept restoring). A non-TaoMate render with the button on whose
  checkpoint fails `load_if_compatible` (fingerprint mismatch, or missing prior
  chunks) now clears it instead of leaving it for the refresh and overwriting it at
  the end. The teacher-audio side already follows the rule through
  `begin_pass(..., resume_index=0)`: a miss or an incompatible pass drops the stored
  payload and records before sampling, then records under the new key. (Approved as
  row 7 of the plan table plus this follow-up.)
- **Per-chunk delete from the timeline.** `POST
  /hr_endless_sampler_preview/teacher_audio_chunk?chunk=<0-based index>`, backed by
  `_TeacherAudioCache.delete_chunk` (rows plus the manifest's `deleted` list), and a
  right-click item "Delete teacher audio cached chunk" in the preview menu, shown
  when `taomateMode` and the chunk has stored audio. The status now reports
  `recorded`, `kv_chunks` and `interrupted`, and the button tooltip says
  "N of M chunks are stored so far (the render was stopped)".

Known ceiling, named in the code: a pass past twelve chunks drops its earliest audio
history and their records with it, so `_teacher_audio_resume_index` still restarts
those long passes at Chunk 1. Priming from the live twelve-chunk window is the
upgrade path.

Validation: `tests/test_teacher_audio_cache.py` (22 tests, five new: stopped-pass
identity and refresh, fresh-pass replacement, resume prefix, button persistence,
unusable-cache deletion), `tests/test_taomate_kv_records.py` (12 tests, two new:
interrupted-pass resume at the stop point, delete forcing a new take),
`tests/test_chunk_director_helpers.py` (158), `tests/test_taomate.py` (41), all JS
suites. No live render has exercised the new resume path yet: the next stopped render
is the first real check. The owner's week-old Sep 19 replay checkpoint was removed by
hand after the rule was implemented, since no render had run in between.

## 2026-09-26 — The resume path's arity crash, and what the first live pass proved
The first render that exercised the new resume logic died with
`_TeacherAudioCache.load_prefix() missing 1 required positional argument:
'chunk_ticks'` at 22:03. The call site still passed one argument, so the *decision*
had worked (it wanted to keep a prefix) and the restore crashed on the very next
line. Fixes:

- The per-chunk ticks list is now built once as `teacher_audio_chunk_ticks` and feeds
  both `_teacher_audio_key(...)` and `load_prefix(resume_index,
  teacher_audio_chunk_ticks)`; the duplication that let the two drift apart is gone.
- `_write_manifest` gained `created=`, so the manifest's creation time names the pass
  instead of the last rewrite. `save_chunk_state`, `save` and `delete_chunk` pass the
  stored value through.
- Regression test `test_every_cache_call_matches_its_method_signature`: it walks
  `nodes.py` with `ast`, collects every `.method(...)` call whose name belongs to the
  cache module classes, and asserts the positional count matches one of the arities
  that name has in this module (two classes share `load_if_compatible` and
  `delete_chunk`, hence the set). This is the check that would have caught the crash.

What the 21:56–22:01 pass left on disk, read back with the real loader
(`/tmp/codex/verify_resume.py`): manifest `status="recording"`, `chunks=6`,
`recorded=4`, `deleted=[]`, key with six prompts and ticks
`[433, 425, 425, 425, 425, 284]`; five KV records on disk but only four settled
chunks, because the render stopped between Chunk 5's KV commit and its completion —
which is exactly the case `settled = min(recorded, available records)` exists for.
The resume index for an identical key is 4, with 4 prime records and 4 prefix
milestones whose widths match the first four ticks, `audio_time_origin = 654`, and 4
preview chunks at 24 fps. So the next render with the same prompt keeps chunks 1-4 and
generates only 5 and 6.

Note for later: `audio_anchor` is always `None` in practice because
`TOGGLE_TAOMATE_DIVERGENCY_RENORM_TEACHER_AUDIO` is hard-coded `False`. If that
constant is ever enabled, a resumed pass would re-anchor on its own first generated
chunk instead of the original Chunk 1, and the prefix's packed rows are no longer
available to recompute it from.

## 2026-09-26 — Preview button states survive a refresh

The owner reported that every preview button resets on a browser refresh. The cause:
`persistPlayerState()` only wrote `node.properties.hr_endless_sampler_preview_player`
(the `L` gamma display, subtitles, mute and the frame position), and LiteGraph
serializes node properties only when the *workflow is saved*, so an unsaved refresh
or a ComfyUI restart started from the defaults.

- The same mapping is now mirrored into `localStorage` under
  `hr_endless_sampler_preview_player` by `writeStoredPlayerState()`, and read back by
  `readStoredPlayerState()` when the node is created, merged *over* the workflow
  copy: every press writes storage, so storage is the newer authority.
  `rememberPlayerState(state)` is the single writer, used by `persistPlayerState()`
  and by the restored-frame path in `restorePersistedFrame()`.
- Both copies keep the same shape, so a workflow saved by an older build keeps
  working, and a browser that blocks or corrupts storage falls back to the defaults
  with one warning instead of throwing inside node creation.
- The play button is part of the same mapping now: `paused` is stored by
  `persistPlayerState()`, `setPaused()` stores it the moment it is pressed (so a
  press is remembered even before a frame is displayed), and
  `restorePersistedFrame()` honors it — an explicit `paused: false` resumes playing
  from the stored frame, while a state written before this key existed (or by an
  older build) still restores paused at the frame, exactly as it did then. Only the
  *soundtrack* still needs one click after a refresh, because browsers grant audio
  playback only from a user gesture; the frames resume on their own.
- The cache button was already server-side (`cache_policy.json`), so it needed no
  change. Fullscreen is deliberately not restored: the browser grants it only from a
  user gesture. The playback-fps widget is a node widget, so it keeps following the
  workflow rather than browser storage.

Validation: `tests/test_preview_player_state.js` (new) executes both helpers in a
`vm` context against a fake `localStorage` and checks the round trip, the exact
storage key, the failure fallbacks and the wiring; all nine JS suites and
`tests/test_teacher_audio_cache.py` (22) pass.

## 2026-09-27 — The delete-chunk item was invisible, and its route was stale

The owner reported the timeline's "delete audio chunk" item as missing. Two causes:

- The item only appeared when *that* chunk had a stored record, so right-clicking any other
  chunk showed nothing at all. It is now offered for **every** TaoMate chunk while the cache
  button is on; when the chunk has no stored audio it is disabled and its tooltip names the
  chunks that do have it. It is a state the owner can see, not a hidden condition.
- The server route behind it (`POST /hr_endless_sampler_preview/teacher_audio_chunk`) answered
  **404** on the live process: the running module predates the route again. Python changes need
  a restart; the JS is served `no-store`, so that half was already current.

Evidence worth keeping: `GET` on that POST-only route is the cheapest liveness probe (405 =
present, `404 = missing`), and the live status showed `kv_chunks: [0]` for a Chunk-1-only pass —
the tail chunks legitimately have no rows because they were kept, not generated.

`tests/test_taomate_timeline.js` now pins the always-offered item, the disabled state and its
explanation; all nine JS suites are green.

## 2026-09-27 — A table item is consumed the moment a chunk gets it

Owner's own proposal, implemented: remove an item from the table as soon as it lands in a
chunk prompt, so it can never reach another chunk.

- `_planned_chunk_prompts` keeps one `emitted_units` ledger per render and threads it into
  every chunk body (`emitted=`).
- `_legacy_shot_body_for_range` claims each owned prose piece by `(slot, text)` and each
  block's in-between prose by `(block, gap)`. A repeat is dropped and named in a warning
  ("already went into an earlier chunk"), so a duplicated sentence is now impossible and
  visible if some future path tries it.
- Repeated *by design*, therefore not claimed: the still-camera state and the opening setup
  (pieces with a negative slot — they are states, not items) and the dialogue words a chunk
  restates to bridge a physical prefix.

Verified directly (`/tmp/codex/verify_ledger.py`): with a shared ledger the second chunk
carrying the same pieces emits the owned sentence once and keeps the camera state.
`tests/test_chunk_director_helpers.py` (168, one new) plus `tests/test_teacher_audio_cache.py`
(30), `tests/test_taomate_kv_records.py` (18), `tests/test_taomate.py` (41) and the global
prompt section checks are green.

Also added this session: `_log_chunk_prompts` writes every chunk's final prompt to the log and
to `<temp>/comfyui-hr-endless-sampler/last_chunk_prompts.txt` (`CHUNK_PROMPT_LOG_FILENAME`),
so the next render can be read instead of guessed at.

## 2026-09-27 — The speaking-rate band only rides a cut delivery

Owner rule, from a real miss: "Speaking slowly, with pauses" appeared in front of a delivery
the chunk holds whole ("That's an act of agression"). The band exists for a delivery a chunk
boundary *cut* — a phrase that continues here from the previous chunk — and a whole delivery
must never gain one.

The band was computed once per source shot (`speech_rate = total_weight / available_seconds`)
and then stamped on **every** piece, which is why an unrelated delivery inherited it.
`_legacy_dialogue_pieces` now tracks per piece whether any of its words arrived through the
chunk's physical prefix (`piece_is_cut`) and only then prefixes the band (or when the slice
starts inside a phrase, `carried_prefix`). The shot-level rate check still stops the render
past a plausible human pace; it just no longer decorates deliveries that were not cut.

Verified with the mocked weights the tests use (`/tmp/codex/verify_band2.py`,
`/tmp/codex/verify_band3.py`): a 48s-in-40s shot asks for "speaking quickly" only on the
continuing chunk, and the chunk holding the whole phrase carries no instruction at all.
`tests/test_chunk_director_helpers.py` (167: one new test for the cut case, and the old
"asks for a faster delivery" test now pins that a whole over-long line is spoken as written
without any band) plus the other Python suites are green.

## 2026-09-27 — The chunk prompt keeps the global prompt's line breaks

Owner rule: the newlines and empty lines in the global prompt are there for a reason, so the
cutter may not flatten them. It did: every piece was joined with a space and the sentence
splitter consumed blank lines as separators.

- `_legacy_prose_layout(body)` maps each prose sentence to `(slot, leading whitespace)`, the
  whitespace being the run the source wrote before that sentence (its line breaks and empty
  lines). `_legacy_prose_slots` is the same map stripped to the slot.
- `_action_units` stamps both onto every unit; the plan's pieces carry the unit's
  `separator` (leading whitespace + connector, so `, and then` keeps its own spacing) and the
  joined `text` is assembled from those same pieces, so the text other callers read is
  unchanged.
- Each dialogue piece carries the whitespace *before* its header as its separator and keeps
  everything the source wrote between the previous block and that header inside the piece
  (the in-between prose plus the break before the header). The first block claims only the
  break into its own header, never the leading prose — that belongs to the chunk's prose
  piece, and claiming it twice duplicated the opening sentence.
- The emission joins with those separators, collapsing runs longer than one empty line
  (`\n{3,}` → `\n\n`) so a dropped piece cannot leave a wall of blank lines.

Verified with a four-part prompt (`/tmp/codex/verify_newlines.py`): the chunk prompt keeps all
four blank lines, has no tripled newline, emits the opening sentence once, and orders
delivery / in-between prose / delivery / trailing prose as written.
`tests/test_chunk_director_helpers.py` (166, one new) plus the other three Python suites are
green.

## 2026-09-27 — Dialogue is a unit in the action table

Owner decision, implemented: dialogue belongs in the action table next to camera, action and
direction, so the chunk body follows the sequence of events the global prompt writes instead
of "all owned prose, then all dialogue".

- `_legacy_prose_slots(body)` maps each prose sentence to its **slot**: how many spoken
  blocks precede it (`0` before the first, `k` between block `k-1` and `k`, the trailing slot
  after the last). It is built with the same `ACTION_SENTENCE_BREAK` the table unitizes with,
  so the two can never disagree about what a sentence is.
- `_action_units(..., sentence_slots)` stamps each unit with that slot; `_action_groups`
  passes it through; the slot rides on the piece, not on the plan text, so ownership is
  untouched.
- `_legacy_action_outputs(...)` returns both the joined text (byte for byte what the other
  callers read) and `[(slot, text, separator), ...]` per window. `_legacy_action_plan` is now
  a thin wrapper over it.
- `_legacy_dialogue_pieces(...)` is the index-carrying twin of `_legacy_dialogue_for_range`,
  which stays the public two-value API, so the slicer's words, seams and delivery bands are
  unchanged.
- `_legacy_shot_body_for_range(..., action_pieces=, sentence_slots=)` merges prose pieces
  (rank 0) and deliveries (rank 1, keyed by their block index) and sorts by slot, so the
  emitted order is the prompt's order. Callers that still pass only the joined `action_plan`
  get the old arrangement through `_legacy_plan_pieces`.

Verified through the real builder (`/tmp/codex/verify_table_order.py`): with prose after the
last delivery it now follows the delivery instead of being hoisted; leading prose stays
before it; the between-deliveries case keeps its order. `tests/test_chunk_director_helpers.py`
(165, three new ordering tests) plus `tests/test_teacher_audio_cache.py` (30),
`tests/test_taomate_kv_records.py` (18), `tests/test_taomate.py` (41) and the global-prompt
section checks are green. Not yet exercised by a live render.

## 2026-09-27 — Prose between two deliveries now stays between them

The owner's rule, from a real miss: in

```
… then <Subject 1> (S1) says: <d>[English]Someone asked me to keep talking for a whole minute.</d>

after saying, she pauses and  makes a disaproving face moving her head side to side.

<Subject 1> (S1) then says: <d>[English] That's an act of agression.</d>
```

the middle prose must stay between the two deliveries. It did not: `_legacy_visual_body`
stripped only the `<d>` blocks, so that prose became an action-table unit that the chunk
author emitted *before* both deliveries — and the delivery band ("speaking slowly, with
pauses") then sat where the prose belonged.

- `_legacy_dialogue_gaps(body)` returns the prose between each pair of spoken blocks, keyed
  by the block it precedes.
- `_legacy_visual_body` strips those gaps as well as the blocks, so the action table can
  never claim them — one definition, used by the plan and by the direct callers.
- `_legacy_dialogue_for_range` emits each gap immediately before the delivery it precedes
  (`"<gap> <header><d>…</d>"`), so the gap lands between the two phrases in whichever chunk
  speaks the second one, and exactly once.

Verified with the owner's own text through the real chunk builder
(`/tmp/codex/verify_gap2.py`): in the produced body phrase 1 < gap < phrase 2, and the gap
appears once. `tests/test_chunk_director_helpers.py` (164, two new) plus
`tests/test_teacher_audio_cache.py` (30), `tests/test_taomate_kv_records.py` (18),
`tests/test_taomate.py` (41) are green.

## 2026-09-27 — Cache warnings moved from the node to ComfyUI toasts

The owner asked for the yellow text on the preview node to become a ComfyUI warning at the
top of the UI, dismissed with its X. Both yellow sites used the same colour (`#e0c060`):
the incompatible-cache notice in `renderCacheReuseLabel` and the reason for a refused cache
policy change in the button handler.

- `warnToast(summary, detail)` calls `app.extensionManager.toast.add({severity: "warn",
  summary, detail, life: 8000})`. It skips a repeat of the *identical* message, because the
  cache status is polled every three seconds and would otherwise raise a toast each time.
- When the frontend has no toast API it writes `console.warn` and returns false, so a cache
  problem is never silent; no yellow node text remains (`grep e0c060` is empty).
- `tests/test_preview_player_state.js` runs the helper against a fake
  `app.extensionManager.toast` and asserts the payload, the repeat suppression and the
  missing-API path.

## 2026-09-27 — The cursor could not come back on an audio-only preview

`restorePersistedFrame()` matched the stored frame against each chunk's `frameNumbers`, and
a TaoMate preview publishes audio with **no** frames (`frame_numbers: []`), so the search
found nothing, the restore gave up and the timeline cursor stayed at the first chunk.

The restore now falls back to the chunk whose **frame range** holds the stored position —
`output_start`/`output_end` from the published chunk, else the range table — sets the cursor
there and clears the pending target. It deliberately skips any group that *does* have frames,
so a real frame match still wins when the frames arrive later.

Verified by running the real function in a harness (`/tmp/codex/run_restore.cjs`): the
audio-only shape restores with `playing=0` on the chunk spanning frames 260-514 for a stored
frame of 261, and the frame shape still shows the exact frame. All nine JS suites green,
`tests/test_preview_player_state.js` pins the fallback and its skip condition.

## 2026-09-27 — A refresh always started playing

`resetExecution()` unconditionally set `paused = false`, and every restore path runs a
reset: the browser's restored snapshot arrives as a `reset` action, so a panel that was
paused came back playing. `restorePersistedFrame()` could not fix it either — it applies
the stored `paused` only when it finds the stored frame, and an audio-only preview has no
frames to find.

`resetExecution` now keeps the stored play state when the panel is restoring
(`restoringCache` is true for exactly that window) and still forces `paused = false` for a
live reset, which is a new run. The restore's `finally` also repaints the transport, so
the restored play button is visible instead of waiting for the next event; the reconnect
test harness needed the matching `renderTransport` stub for that.

Validation: all nine JS suites green, including `tests/test_preview_player_state.js`
(two new pins: the restore condition and the repaint) and `tests/test_preview_reconnect.js`.

## 2026-09-27 — The key now hashes the owner's own chunk prompts

Asked again, plainly: only the chunk prompts and the chunk count may rebuild the audio.
Until now the key hashed `planned_prompts`, i.e. the *derived* prose (action ownership,
dialogue windows, delivery bands), so anything that re-derived that text — the
continuation's presence, a resolution change, an edited table entry — moved the hashes.

`_teacher_audio_key(...)` is now called with `teacher_audio_key_prompts`: the owner's own
chunk table (`manual_prompts.prompts()[:replay_sample_end]`, the legacy/manual chunk
prompts node) when one exists, falling back to the planned prompts only for
Gemma-authored layouts that have no table. The second argument stays
`teacher_audio_chunk_ticks`, which already encodes how many chunks the layout has and how
long each one is, i.e. what `chunk_frames` decides. Nothing else reaches the key — no
model, LoRA, seed, sigmas, resolution, continuation or KV precision.

Cost, stated before the run: stored passes keyed on the *derived* text no longer match, so
the first render after this change rebuilds the audio once; every later render reuses it
unless a chunk's own text changes.

Validation: `tests/test_teacher_audio_cache.py` (29, the key test now pins the owner-table
call and asserts `video_continuation` never appears at the call site),
`tests/test_chunk_director_helpers.py` (162), `tests/test_taomate_kv_records.py` (16),
`tests/test_taomate.py` (41).

## 2026-09-27 — The audio key, the dialogue windows, and a decision line

The owner stopped a video render, switched `kv_cache_compression` to `turboquant4` and
saw the audio rebuild, and asked why the key is not "chunk prompt only". Findings, all
evidence-based:

- The precision is not in the key at all. The key line now printed by every render
  (`teacher audio key: N chunks, ticks [...], prompt hashes [...], cache button on|off`)
  shows only the six ticks and the six prompt hashes.
- The dialogue assignment does **not** use the continuation's *value*: `dialogue_ranges`
  is built from each request's output window (`frame_start + output_trim_frames`,
  `frame_end`, nodes.py:2761) and each shot body is sliced against
  `content_start … chunk["frame_end"]` (nodes.py:2789), all of which the planner keeps
  identical for continuation 39, 56 and 73 (see the 2026-09-27 plan entry above). Only
  `continuation > 0` matters, through the `<Video N>` reference label (nodes.py:2775).
- The hashes for Chunks 1 and 4 in the 01:41 render differ from the pass stored at
  22:40, but the pass immediately before it (00:46) had its key overwritten, so whether
  the text changed between those two runs cannot be shown from the outside.
- Why the owner could not see any of this: the running module predates the decision
  logging, so a rebuild printed neither the branch nor the reason. Fixed by one
  unconditional line after the branch chain —
  `teacher audio cache decision: reused|resumed|miss|incompatible|disabled (<reason>)`,
  with the reason naming the chunk whose prompt moved. The module must be restarted for
  it to appear (the live one predates both log lines).

Validation: `tests/test_teacher_audio_cache.py` (29, one new pinning the decision line),
`tests/test_chunk_director_helpers.py` (162), `tests/test_taomate_kv_records.py` (16),
`tests/test_taomate.py` (41).

## 2026-09-27 — One <d> group per phrase, and the toggle for the first attempt

The owner's own fix for a take that drops a phrase: give every phrase its own
`<d>[language] …</d>` group, because MiniMax speaks one group as one delivery and a
group holding several phrases can lose the ones it has no room for. Their example:
`<d>[English] This is that. But that is not this.</d>` must become
`<d>[English] This is that.</d> <d>[English] But that is not this.</d>`.

- `_split_dialogue_phrases(prompt)` rewrites each `<d>` group into one group per phrase,
  repeating the group's own language tag (a bare `<d>` stays bare) and leaving every
  character outside the groups byte-identical. Phrases end where `_dialogue_phrase_ids`
  ends them (`.`, `!`, `?`, `…`, `...`), so the split and the chunk cutter agree on what
  a phrase is. A group that holds one phrase is returned untouched.
- `_dialogue_retry_prompt(prompt)` = `_split_dialogue_phrases(_escalate_speech_rate(...))`,
  so a retry re-groups the phrases *and* asks for the faster delivery; the words stay
  identical either way. Both retries use it: `retry_audio_first_candidate` and
  `retry_chunk_teacher_audio` (the log line now says "one group per phrase and a faster
  delivery").
- `TOGGLE_SPLIT_DIALOGUE_PHRASES = True` decides whether the *first* attempt is authored
  split as well (answered as "make a toggle, default to 2"): True splits every chunk's
  dialogue as it is authored, False keeps the owner's grouping and lets only a retry
  split. The toggle does not enter the audio cache key — the key hashes `planned_prompts`,
  not the authored prompt — so flipping it never invalidates a stored pass.

Validation: the owner's example and the edge cases were checked with the real helper
(`/tmp/codex/verify_split.py`, `/tmp/codex/verify_retry.py`: two languages in one chunk,
a bare `<d>`, a lone phrase, ellipses, prose with no dialogue, and the retry keeping the
dialogue words while replacing the delivery band). Tests: `tests/test_chunk_director_helpers.py`
(162, five new), `tests/test_teacher_audio_cache.py` (28), `tests/test_taomate_kv_records.py`
(16), `tests/test_taomate.py` (41), all nine JS suites green. Not yet exercised by a live
retry.

## 2026-09-27 — video_continuation 39 vs 56 vs 73: what actually changes

The owner saw video artifacts at `video_continuation=56` and `73` that disappear at
`39`, and asked how the chunk distribution differs. Measured with the plugin's own
planner (`/tmp/codex/plan_table.py`, `/tmp/codex/plan_phases.py`; source 1450 frames at
24 fps ⇒ 427 video tokens, 2417 audio ticks, `chunk_frames=260`):

- **The request-level chunks are identical** for 39, 56 and 73: 6 chunks of
  `[260, 260, 260, 260, 260, 175]` frames and `[433, 425, 425, 425, 425, 284]` audio
  ticks. So the artifacts are *not* a chunking, prompt or audio-cache effect, and the
  audio cache key is unaffected by the continuation setting.
- **Only the internal streaming grid changes**, i.e. `TaoMateStreaming.plan`'s phases
  inside each chunk: 49 phases at 39, 34 at 56, 26 at 73.
- Chunk 1 specifically: 9 sub-chunks at 39, 6 at 56, 5 at 73; its opening sub-chunk grows
  `12 → 17 → 22` new frames, i.e. a `17 → 22 → 27`-frame sampling window, and the video
  steps derived from that window (`_video_steps`) jump `2 → 7 → 7`. At 39 every
  sub-chunk in chunk 1 stays at the minimum (17-frame window, 2 steps); at 56 and 73 the
  opening window crosses onto the next H3 grid step. With 73 three of chunk 1's five
  sub-chunks need 7 steps (25 total) against 18 at 39.

That is the one geometry that moves exactly where the owner reports artifacts, and it is
why 39 is clean. No code was changed for this analysis; it is recorded so the next
session starts from the numbers instead of re-deriving them.

## 2026-09-26 — Why no button state ever reached storage

The owner changed the buttons, refreshed, and got defaults again — twice. The mirror was
never written, and the reason was inside `persistPlayerState()`:

```js
if (Number.isFinite(pendingFrame) && frame !== pendingFrame) return;
```

That guard exists so a restored frame survives until its chunk arrives, but it returned
*before* `rememberPlayerState(...)`, and it returned for **every** button. In a TaoMate
preview there are no video frames at all, so `displayedFrameNumber()` never becomes the
stored target and the guard blocked forever: L, subtitles, mute and play were dropped
no matter what the owner pressed, which is exactly the reported symptom.

The guard now only protects the *frame field*: a pending target is kept as the stored
frame, and the other four values are stored on every press. `tests/test_preview_player_state.js`
runs the real function body in a `vm` with a fake closure and checks all three cases
(dropped-then-fixed pending frame, no pending frame, arrived frame).

Two operational notes for the next diagnosis: ComfyUI serves extension JS with
`Cache-Control: no-store`, so a browser refresh alone picks up `web/*.js` changes and no
ComfyUI restart is needed for them; Python changes still need a restart, and the restart
must come *after* the edit (a restart at 22:50 misses an edit made at 22:52).

## 2026-09-26 — Button states after the restart: what was really broken

The owner restarted ComfyUI, refreshed, and every preview button looked reset. Two
separate causes, neither of them a missing feature:

- The browser had been running the *old* extension script until that refresh, so the
  button presses that preceded it never reached `rememberPlayerState` and browser
  storage was empty. The first refresh after the fix therefore shows defaults by
  construction; every press from then on stores the state. ComfyUI serves extension
  files with `Cache-Control: no-store`, verified with `curl -D`, so a plain refresh
  does load the new script.
- The cache button *was* persisting: `<temp>/comfyui-hr-endless-sampler/cache_policy.json`
  was rewritten by the owner's click at 22:55:01 (`enabled: false`) and the endpoint
  logged `replay cache is disabled for future sampler runs`. Earlier attempts did not
  land because the endpoint refuses a policy change while a sampler render is active
  and the widget used to log the refusal and flip the button back **silently**, which
  reads exactly like "it did not save".

That silent revert is fixed: the click handler keeps the reason and prints it in the
cache label, so a refused change says why. Self-check for the owner, in the browser
console: `localStorage.getItem("hr_endless_sampler_preview_player")` returns the stored
state, or `null` when no press has reached the new handler yet.

Also worth knowing when reading the preview after a restart: the on-disk audio cache
was a one-chunk leftover from the render the restart killed (`status="recording"`,
`recorded=1`), so a refresh legitimately restores only Chunk 1 until a render resumes
it and records the rest.

## 2026-09-26 — The audio cache key, and the stale process that hid it

The owner changed the resolution (and later the attention backend, the LoRA source and
the source prompt) and the audio rebuilt, which contradicts the agreed key: the audio
cache is keyed by the per-chunk prompts plus each chunk's audio tick lengths, and by
nothing upstream — no model, LoRA, seed, sigmas, guidance, resolution or attention
backend. That rule is real: `_teacher_audio_key(chunk_prompts, chunk_ticks)` takes both
lists as arguments, the single call site passes exactly
`planned_prompts[:replay_sample_end]` and `teacher_audio_chunk_ticks`, and
`test_the_key_is_the_chunk_prompts_and_the_chunk_structure` pins that call site. The
22:12 and 22:20 renders prove the sensitivity is prompt-only: they resumed at Chunk 4
*only* because the owner had edited Chunk 4's text.

What actually happened is that the running ComfyUI served **stale code**. The process
loaded the plugin at ~21:31 and never reloaded, so everything added after that was
missing from it: the timeline delete route, the delete-an-unusable-cache rule, the
full-plan timeline, the resume-reason label, the contiguous-row cap and the whole
keep-the-tail feature. Evidence, gathered from the live server and log:

- `GET /hr_endless_sampler_preview/teacher_audio_chunk` answered **404** while
  `cached_preview`, `replay_cache` and `state` answered 200 — the route exists in the
  file and in no loaded module.
- The live `/replay_cache` status reported `audio_cache.run_state: "idle"` for the
  22:40 render, and the log for that render contains no cache-decision line at all
  (neither `reused`, `resumes`, `has no reusable`, `cannot reuse` nor `cache button is
  off`), i.e. that process never evaluated the cache for the render.
- Fetching `/object_info` does **not** reload custom nodes in this setup: the 404
  survived it. Only restarting ComfyUI picks up an edited plugin.

Lesson for the next session: before diagnosing the sampler's behaviour, check that the
running process has the code. A route probe against a route added in the same edit
batch is the cheapest test, and `comfyui.log`'s cache-decision lines are the second.

Also added: every render now logs its teacher-audio key —
`teacher audio key: N chunks, ticks [...], prompt hashes [...], cache button on|off` —
so a rebuild either names its reason or shows that the prompts or the chunk lengths
really moved. `tests/test_teacher_audio_cache.py` (28) stays green.

## 2026-09-26 — An edited chunk now keeps the chunks after it

Owner decision, taken knowingly against the seam argument: when an edited chunk's
audio changes, the chunks *after* it keep the audio they already had, and only the
edited chunk is generated again. Before this, a resume regenerated the whole suffix
(every later chunk was generated attending over the edited chunk's clean audio) and
the discarded rows went with it.

- `_teacher_audio_kept_tail(cached_prefix, resume_index, chunk_count)` offers the tail
  only when the stored pass settled **every** chunk and the edited chunk is not the
  last one. An interrupted pass cannot offer a complete tail, and a hole inside it
  would force a chunk to be generated whose stored rows are gone, so those passes
  regenerate the suffix exactly as before.
- `prepare_audio_first(..., generate_count=1)` generates just the edited chunk and
  keeps `audio_first_milestones` for everything outside `[start_index,
  generated_end)`, which is what lets the video pass pop one guidance state per chunk
  while chunks after the edit are never sampled again.
- The pass products stay in chunk order: the tail's prompts, transcripts,
  observations and clean latents are appended after the pass returns, and
  `full_teacher_audio` is extended with the kept latents, so the payload, the preview
  and the video pass all read position i as chunk i. `teacher_audio_payload()` merges
  the same tail as soon as the edited chunk settles — before that, the previous
  payload is still untouched on disk, so a stop during the edited chunk loses nothing.
- `_teacher_audio_resume_reason` composes the sentence the preview shows (a resume can
  keep chunks on both sides of the one it regenerates), and the widget prints
  `run_reason` verbatim instead of building its own wording.
- The attention rows of the kept tail are deliberately dropped (`begin_pass` discards
  records from the resume point on): they describe the take the edit replaced, and
  nothing needs them because those chunks are not sampled again. A later edit inside
  the tail therefore starts from the first chunk without rows, which is the chunk
  after the last one this pass generated.

Validation against the owner's real stored pass (finished, 6/6 chunks, all rows;
`/tmp/codex/verify_tail.py`): editing Chunk 4 reports `resume at Chunk 4; kept teacher
audio for Chunks 1-3 and Chunks 5-6; generating Chunk 4` with kept tail widths
`[425, 284]`; editing Chunk 1 finds no resume and regenerates everything; editing
Chunk 6 generates Chunk 5 and keeps Chunk 6, because the rows of a skipped chunk are
never written and one chunk has to be regenerated to rebuild the history. Tests:
`tests/test_taomate_kv_records.py` (16), `tests/test_teacher_audio_cache.py` (28),
`tests/test_chunk_director_helpers.py` (158), `tests/test_taomate.py` (41), all nine JS
suites. The mixed pass itself has not run live yet.

## 2026-09-26 — A refresh mid-pass dropped the chunks the pass had not reached

The owner edited Chunk 4's prompt, restarted the render and reported that chunks 5 and
6 were gone. Reproduced with the real loader: the stored payload's `preview` block only
described the *settled* chunks (`payload preview block chunks: 4`, `restored reset
chunk_count: 4 (plan has 6)`, `total_frames 1025` instead of 1450), so a refresh showed
a four-chunk timeline and the plan's last two chunks disappeared from it.

- `teacher_audio_payload()` now describes the whole plan: every planned chunk up to
  `replay_sample_end` keeps its `index`, frame range, prompt and subtitle, plus
  `settled` telling whether its audio exists. `_teacher_audio_chunk_prompt` picks the
  generated prompt when there is one and the plan's `audio_teacher_prompt`/`h3_prompt`
  otherwise, so a pending chunk is still described.
- `build_teacher_audio_preview_snapshot` publishes the plan's ranges for the reset and
  only emits `chunk_audio_first`/`audio_chunk_complete` for chunks that are settled;
  `cached_chunk_count`/`reused_chunk_numbers` count the settled prefix instead of every
  range. Its guard now compares observations against the settled count, and a payload
  without the `settled` key keeps the old meaning (every listed chunk has audio).
- Consequence for the pass itself, unchanged and deliberate: an edited chunk k still
  regenerates k..N, because the later chunks were generated attending over chunk k's
  clean audio. The timeline keeps their slots and refills them as the pass reaches them.

Validation: `/tmp/codex/verify_plan_timeline.py` (a four-chunk plan with two settled
chunks restores `chunk_count=4`, `total_frames=96`, audio only for chunks 0-1, and the
pending chunks keep their planned prompts; the finished variant publishes all four),
`tests/test_teacher_audio_cache.py` (27, two new), `tests/test_taomate_kv_records.py`
(13), `tests/test_chunk_director_helpers.py` (158), `tests/test_taomate.py` (41), all
nine JS suites.

## 2026-09-26 — First real resume, and the two things it exposed

The 22:12 render logged `resumes teacher audio at Chunk 4 of 6, keeping 3 settled
chunks because Chunk 4's prompt changed` and then continued the pass from there — the
resume path works live. Two defects showed up next to it:

- **The preview reported the plan's chunk count as the kept count.** The resumed
  branch called `_set_teacher_audio_cache_state("resumed", reason, replay_sample_end)`
  while the third argument is `reused_chunks`, so the label read "kept 6 settled
  teacher audio chunks; the teacher audio continues at Chunk 7" on a six-chunk plan.
  It now passes `teacher_audio_resume`, and the label names both sides of the work:
  "kept teacher audio for Chunks 1-3; generating Chunks 4-6", with a separate wording
  for the case where every planned chunk was kept.
- **The rows can lag the payload by one chunk**, because a chunk's stored rows land
  with the *next* chunk's forward while its payload entry is written when it settles.
  The resume point is now `min(settled, contiguous rows)` instead of failing the whole
  prefix when a row is missing: on the live cache that is 4 kept chunks out of a
  5-chunk payload, where the old rule returned 0 and recomputed everything.

Validation: `tests/test_taomate_kv_records.py` (13, new test for the lagging rows),
`tests/test_teacher_audio_cache.py` (25, new test pinning the reported kept count),
`tests/test_chunk_director_helpers.py` (158), `tests/test_taomate.py` (41), all nine JS
suites. The real cache was read back with the real loader after the change:
`settled=5`, rows `[0, 1, 2, 3]`, resume index 4, four prime records, five prefix
milestones, `audio_time_origin=654`.

## 2026-09-27 — A cached pass is audio and rows together, or it is nothing

The owner refreshed the browser and reported the timeline's "Delete teacher audio cached
chunk" item showing but greyed out, although a render had just reused the stored audio.
The live status route explained it: `recorded: 6, chunks: 6` with `kv_chunks: [0]` — the
payload held all six chunks of teacher audio while `kv/` held one single record, because
the pass that wrote it regenerated Chunk 1 and `begin_pass(keep_payload=True)` had wiped
*every* stored row before that. The menu enables the item (and the delete route guards it)
on "this chunk has a KV record", so Chunks 2-6 were greyed, and the tooltip blamed "no
stored teacher audio", which was false. An earlier note in this session that the running
process was stale was wrong: ComfyUI answers 404 for a GET on a POST-only route, and
`POST {}` returning the route's own validation error proves the route is live.

The owner's rule settled it: the cache is one object, so audio and its attention rows are
stored, kept and deleted together. Three changes:

- `_TeacherAudioCache.begin_pass` drops only the rows of the chunks this pass samples
  again. With `keep_payload` (the tail-keeping edit path) that is exactly `resume_index`;
  a resume without a kept tail still drops everything from `resume_index` on.
- `_teacher_audio_kept_tail(prefix, resume_index, chunk_count, available_rows)` refuses a
  tail whose chunks have no stored rows, so audio is never kept without the rows that
  describe it; both call sites pass `teacher_audio_cache.kv_chunk_indices()`.
- The timeline wording: the greyed item says the audio and its rows are cached together
  and this chunk has neither, and the enabled item promises "keeps every other chunk and
  generates this one again" instead of the old "and every chunk after it".

Consequences: a healthy pass keeps rows for every chunk through any single-chunk edit, so
a later delete (which removes that chunk's row and resumes at it) still regenerates that
chunk alone and keeps the rest — the pair invariant also makes the old silent ignore
impossible, where a delete was capped below the deleted chunk and the kept tail copied its
audio straight back. The route guard and the menu's enable test were left alone because
under the pair rule a chunk's row implies the rows before it. The cache on disk right now
is legacy (audio 6, rows 1): with the tail rule, editing Chunk 1 of it regenerates all six
chunks instead of keeping Chunks 2-6, and one full teacher pass heals it.

Validation: `tests/test_taomate_kv_records.py` (19; the kept-tail test rewritten to assert
that only the generated chunk's row goes, plus a new test that a tail without rows is not
kept), `tests/test_teacher_audio_cache.py` (30), `tests/test_taomate.py` (41),
`tests/test_chunk_director_helpers.py` (168), and all nine JS suites. The running ComfyUI
process predates these Python edits, so a restart is required before the new rule is live.

## 2026-09-27 — First attempts keep the owner's <d> groups, and the pace is the chunk's own

Two owner reports from the 03:31 render of `prompt.txt` (one 60.42s shot, one
`<d>` group holding the whole 38s monologue, cut into six chunks): the per-phrase
`<d>` split broke the dialogue generation, and no chunk prompt carried a
speaking-speed instruction any more, "not even cut dialogues".

The phrase split was the first attempt's own doing: `nodes.py` re-grouped every
`<d>` block into one group per phrase before the teacher ever saw it. The owner
wants the prompt's groups spoken as written, so that call is gone: the teacher
prompt is appended untouched, and `_dialogue_retry_prompt` is now the only place
that re-groups (still behind `TOGGLE_SPLIT_DIALOGUE_PHRASES`, still adding "Speaking
much faster than before"), which is what a retry is for — phrases the take lost.

The missing band was the cut test. `delivery` was computed per *shot* (the log
showed `needs 0.63x` for all six chunks), inserted only when the chunk boundary cut
a phrase, and the phrase-ownership rule now places every seam between phrases, so
nothing counted as cut and no chunk received anything. Two fixes in
`_legacy_dialogue_pieces`:

- A piece that speaks only part of its source block is a cut delivery, whether the
  seam fell inside a phrase or between two of them (`len(selected) <
  len(weighted_words)`), so a sliced monologue is banded on every chunk it spans.
- The band's value is the chunk's own: `_dialogue_delivery_instruction` now runs on
  the words this chunk speaks divided by the time this chunk gives them
  (`selected_weight / (current_range[1] - current_range[0]) / fps`), not on the
  shot's ratio. On the owner's prompt that is 0.42x/0.56x for the sparse chunks
  (below the 0.60 floor: sparse dialogue, no instruction), 0.89x, 0.70x and 0.61x
  for the rest ("Speaking slowly, with pauses,"), and the log prints one line per
  banded chunk with those numbers. The shot-level line now only reports the
  estimate that can stop the render.

Validation: the offline bake probe (`/tmp/codex/band_probe.py`) rebuilds the
owner's six chunk prompts from `prompt.txt` and showed one whole group per chunk
with no band before the change and bands on Chunks 2, 5 and 6 after it;
`/tmp/codex/band_probe2.py` pins the per-chunk rule (2s of words in 20s gets
nothing, 18s in 20s gets "Speaking slowly, with pauses,"), and
`/tmp/codex/band_probe3.py` the cut case (24s of words in a 20s window gets
"Speaking quickly,"). Tests: `tests/test_chunk_director_helpers.py` (169, the
authoring-path test rewritten to prove the first attempt never re-groups and a new
per-chunk pace test), `tests/test_teacher_audio_cache.py` (30),
`tests/test_taomate_kv_records.py` (19), `tests/test_taomate.py` (41), nine JS
suites. The running ComfyUI process predates all of it: a restart is required, and
the chunk prompts in the node's editor have to be re-baked to gain the bands.

## 2026-09-27 — The timeline cursor follows the mouse while a chunk is still filling

The owner's complaint: while a chunk is being rendered, dragging the timeline
"does not make it follow the mouse", the cursor stays behind it because the chunk
has fewer sub-chunks than its span, and when the pointer leaves the last available
sub-chunk the cursor jumps.

The cause was in `seekTimeline`: the pointer's fraction was applied to the frames
that *exist* (`Math.round(fraction * (group.frames.length - 1))`), so the cursor
could never reach past the newest decoded frame of a half-filled chunk, and a
pointer outside an available chunk made the code pick the nearest rendered chunk and
snap the cursor to its edge — which is the jump the owner saw.

- `seekTimeline` now records the pointer's own position as `timelineScrub = {chunk,
  fraction}` after showing the closest available frame, and only calls `show` when
  the pointed chunk has frames: a chunk the pass has not reached keeps the
  displayed picture where it is instead of dragging it to the nearest rendered one.
- The new `scrubPlayheadPosition(spans, total, scrub)` converts that position to a
  percentage, and `renderTransport` draws the playhead from it when a drag owns the
  cursor — even when nothing is rendered under the pointer.
- The drag's position is cleared by `show`, by `playFrameGroup` (playback moves the
  position itself) and by `resetExecution`, so the cursor returns to the displayed
  frame as soon as anything else seeks.

Validation: `tests/test_taomate_timeline.js` gained "Timeline scrub cursor checks"
(positions for a half-filled second chunk, clamped fractions, out-of-range chunks,
a zero span, and source assertions that the nearest-rendered-chunk jump is gone),
and all nine JS suites pass.

## 2026-09-27 — The cache-button-off render crashed on its own decision line

After the restart the owner's first render died 29s in with
`UnboundLocalError: cannot access local variable 'teacher_audio_reason'` at the
unconditional `teacher audio cache decision` log line. The cache button was off, so
the `if not _replay_cache_enabled():` branch ran — the branch that never loads the
cache — and `teacher_audio_reason` was bound only in the `else`.

The off branch now names its own reason (`the cache button is off, so this render
records a fresh teacher audio pass`) before logging, which also makes the decision
line read `disabled` with a cause instead of an empty reason. An AST sweep of the
same function confirms `resume_index`, `stored_pass`, `prefix_records`,
`cached_prefix`, `prefix_reason` and `kept_chunks` are read only inside the branch
that binds them, so this was the only leaked name.

Validation: `tests/test_teacher_audio_cache.py` (31; the new
`test_the_decision_line_has_a_reason_whatever_branch_ran` asserts the off branch
binds the reason before the cache load and before the decision line),
`tests/test_chunk_director_helpers.py` (169), `tests/test_taomate.py` (41).

## 2026-09-27 — A sub-chunk with no speech is no longer told the words

The owner watched sub-chunks whose audio is silent still move the mouth in the video.
The cause is in `python/taomate.py`: a request's phases all share one encoded
conditioning ("encoded once per request", `execute_chunk`), so a phase with silence in
its own window still read the chunk's whole `<d>` dialogue.

Owner's rules for the fix: a sub-chunk that speaks keeps the chunk's dialogue exactly as
it is; only a sub-chunk with no speech loses it; the removal adds no text of its own;
and it applies to both flows, the audio-first pass (whose transcripts are already
finished) and the in-chunk teacher path (whose transcript arrives during that chunk,
before its sub-chunks).

- `nodes.py` gained `_prompt_without_dialogue`, `_subchunk_speaks` and
  `_silent_subchunk_prompts`. Speech is decided from the take's own word times (Whisper
  seconds from the chunk's audio start, which matches `decode_audio_preview_segment`
  because it discards the decoded context prefix) against the phase's own audio window
  (`phase["audio_start"]..phase["audio_end"]`, 40 ticks per second). Any overlap counts
  as speech, so only a certain silence loses the dialogue, and a transcript that never
  arrived keeps today's prompt.
- `DIALOGUE_LEAD` and `DIALOGUE_LEAD_BARE` remove the clause that hands a line to a
  speaker together with the line. That is what keeps "starts the video in silence (no
  mouth movent) for 00:01.00," and the prose around it intact while the delivery and its
  words go. Prose that merely mentions speaking is left alone, line breaks survive, and
  only `\n{3,}` collapses, as everywhere else in this plugin. The first pattern anchors
  on the `<Subject N>` tag, the second on a clause standing alone at a line start or
  after a finished sentence; a first attempt anchored nowhere ate the tail of the
  preceding sentence.
- The silent sub-chunks of a chunk share one dialogue-less text, so a single extra
  encode inside the CLIP window the chunk already opens serves all of them, and each
  one's conditioning comes from `_conditioning_for_chunk` with the same arguments as the
  chunk's own (only the text differs).
- `execute_chunk` takes `phase_conditionings`, remembers the caller's conditioning and
  assigns `guider.original_conds` for every phase: a silent phase's own text where one
  exists, the chunk's again for the others, so a silent phase cannot leak into the
  sub-chunk that follows it.

Validation: `tests/test_chunk_director_helpers.py` (171; the removal on the owner's own
chunk 1, plus the per-sub-chunk decision including a word straddling a boundary, an
empty take, a missing transcript and a prompt with nothing to remove),
`tests/test_taomate.py` (42; a new test that phase 2 of a four-phase request is sampled
with its own text and the rest with the chunk's, and that the guider still holds the
chunk's conditioning afterwards), `tests/test_teacher_audio_cache.py` (31),
`tests/test_taomate_kv_records.py` (19). Not verifiable offline: that H3 really closes
the mouth once the dialogue is gone — that needs a render.

## 2026-09-27 — A retry isolates only the phrase the take failed

The owner tested by hand what the retry should do: chunk 2's three phrases in one
`<d>` group, the take losing the opening phrase — splitting *only* that phrase away
and leaving the other two grouped made MiniMax speak the whole dialogue, while the
retry's "one group per phrase" produced a take that was worse than the first one.

So the retry no longer re-groups everything:

- `_dialogue_phrase_groups(spoken)` is the phrase walk `_split_dialogue_phrases` used,
  extracted so both share one definition of a phrase (terminals and ellipses).
- `_dialogue_phrase_was_spoken(phrase, heard_words)` judges a phrase word by word with
  the existing tolerant `_dialogue_same_word`, so a stale take only fails a phrase when
  a word of it is really missing, never for spelling or a number written as digits.
- `_split_out_damaged_phrases(prompt, transcript)` rebuilds every `<d>` group as the
  healthy runs the prompt already had, with each failed phrase on its own. A phrase
  damaged in the middle separates its neighbours too, because a later phrase cannot be
  moved in front of it.
- `_dialogue_retry_prompt(prompt, transcript)` applies that behind
  `TOGGLE_SPLIT_DIALOGUE_PHRASES` and then escalates the delivery, and it is the only
  place that re-groups; both retry call sites now hand it their own `recognized`
  transcript, and the log names the phrases it isolated. With no usable transcript, or
  when every phrase was spoken, the grouping is left exactly as written — a take that
  failed every phrase is the one case where the whole dialogue still goes one group per
  phrase.

Replayed on the real chunk 2 (heard `and why light is bad for you and love is a curse.
People are strangely afraid of silence. Personally, I find silence comforting.`): the
failed opening phrase stands alone and the other two stay one group, which is what the
owner's manual test produced.

Validation: `tests/test_chunk_director_helpers.py` (173; the failed-last-phrase case,
a garbled middle phrase, the no-transcript / nothing-damaged / all-damaged cases, and
source assertions for both call sites), `tests/test_taomate.py` (42),
`tests/test_teacher_audio_cache.py` (31), `tests/test_taomate_kv_records.py` (19), nine
JS suites.

## 2026-09-27 — Ctrl-hover names every teacher-audio attempt, tinted per take

The owner asked the preview tooltip for a second key: Shift already shows the full
prompt of a chunk, and Ctrl must show every prompt the chunk's teacher audio was
sampled with — the first take and each retry — cut to the `detailed_description`
section only, each on a colored background so the first take and the retries are told
apart at a glance.

The prompt of record could not answer that: a retry overwrites `audio_teacher_prompt`
(and `h3_prompt`) with the wording it is about to use, so the take it rejected was gone.
The attempts are now a list beside it:

- `nodes.py` gained `_remember_audio_teacher_attempt(range_item, retry_prompt)`, called
  at the two retry sites (`retry_audio_first_candidate`, `retry_chunk_teacher_audio`)
  *before* the prompt of record is overwritten. It seeds the list from that prompt when
  the range has none yet, so the first entry is the rejected take and every later entry
  is one retry, in sampling order. Nothing else writes the list, so a chunk that was
  never retried carries none and the tooltip shows its own prompt as a single attempt.
- `teacher_audio_payload()` writes `"attempts"` per preview chunk and
  `preview.build_teacher_audio_preview_snapshot` maps it back to
  `audio_teacher_attempts`, so the sequence survives a browser refresh and a process
  restart through the teacher-audio disk payload. `preview.set_audio_prompt` now takes
  an optional `attempts` and publishes `audio_teacher_attempts` on `chunk_metadata`,
  and `preview._cache_payload` keeps that key, so a refresh mid-render keeps it too.
- `web/unlimited_preview.js`: `chunkPromptDescription(chunk, fallback=true)` — the
  attempt view passes `false` so it never falls back to Gemma's description, which would
  make two attempts read identically; a prompt without that section shows its real text
  instead of an empty block. `audioTeacherAttempts(range)` is the one reader of the list
  (single prompt when the list is absent). Ctrl owns the prompt area and wins over
  Shift; the keydown/keyup handler is now `updatePromptKeys` for both keys, the help line
  names Ctrl, and the tooltip's signature carries the attempts so an open tooltip
  refreshes when a retry lands.
- Coloring: one tint per attempt index, cycled — blue `#45b7d1`, orange `#f4b942`,
  purple `#b27ce8`, green `#78d078` — as a 3px left border plus a 14%-alpha background,
  with `Take 1 (first take):` / `Retry N:` labels in the same hue. The prompt body is
  built by the new shared `coloredPromptBody`, so the attempt blocks and the existing
  description block keep the identical per-shot coloring and bold dialogue tags.

Out of scope on purpose: the finished-video player's tooltip (`createFinishedChunkTooltip`)
and the timeline sidecar (`video_io.normalize_timeline`) were left alone at the owner's
choice. Known gap: a *resumed* pass republishes its kept chunks from the stored
`prompts` list, which holds only each chunk's prompt of record, so a chunk kept across a
resume shows one attempt until it is generated again.

Validation: `tests/test_preview_prompt.js` (new `checkAttemptView`: attempt reading,
blank filtering, the strict section extractor, one tinted block per attempt, the labels
in order, the raw-text fallback, Ctrl over Shift, live delivery), `tests/test_chunk_director_helpers.py`
(174; the tooltip source assertions updated and
`test_a_retry_keeps_the_take_it_replaced_and_reaches_the_preview` added),
`tests/test_teacher_audio_cache.py` (32; `test_a_refresh_keeps_every_teacher_audio_attempt`),
`tests/test_taomate.py` (42), `tests/test_taomate_kv_records.py` (19),
`tests/test_video_io.py` (25), `tests/test_audio_transcription.py` (2), all nine JS
suites. The running ComfyUI process predates these edits, so a restart is required
before Ctrl-hover is live.

## 2026-09-27 — Each retry also shows the hearing that caused it

Follow-up to the Ctrl-hover attempt view: a retry prompt only says what was asked for,
not what went wrong, so the owner asked for the transcript that triggered each retry,
on a darker shade of that retry's own background so the pair reads as one retry.

- `_remember_audio_teacher_attempt(range_item, retry_prompt, transcript="")` now appends
  the hearing to `audio_teacher_retry_transcripts` in the same call that appends the
  retry, so the two lists stay aligned by construction: entry N belongs to retry N, and
  the take itself owns no entry. An unavailable transcript keeps its slot as `""`, so a
  hearing can never drift onto a different retry.
- Both retry sites pass `recognized["text"]` — the transcript of the take being rejected,
  which is exactly the mismatch the retry answers.
- `teacher_audio_payload()` writes `retry_transcripts` beside `attempts`;
  `preview.set_audio_prompt(..., retry_transcripts=None)` publishes
  `audio_teacher_retry_transcripts` (only when at least one entry is non-empty),
  `preview._cache_payload` keeps the list including its empty slots, and
  `build_teacher_audio_preview_snapshot` restores it from the payload.
- `web/unlimited_preview.js`: the new `audioTeacherRetryTranscripts(range)` reader, and
  under each `Retry N:` block an extra
  `Heard instead (caused this retry):` block holding the hearing, styled
  `background:color-mix(in srgb, <that retry's accent> 32%, black)` with the same 3px
  accent left border — a darker shade of the retry's tint, so the hearing and its prompt
  are tied together and still told apart. The attempts signature carries the transcripts,
  so an open tooltip refreshes when a retry lands.

Validation: `tests/test_preview_prompt.js` (the reading, an empty slot suppressed, the
hearing block's darker shade derived from the retry's own accent, and proof it never
borrows the first take's tint), `tests/test_chunk_director_helpers.py` (174; the helper's
alignment and empty-slot cases, the published key, and the payload source assertion),
`tests/test_teacher_audio_cache.py` (32; the snapshot keeps `retry_transcripts`),
`tests/test_taomate.py` (42), `tests/test_taomate_kv_records.py` (19),
`tests/test_video_io.py` (25), all nine JS suites. Restart ComfyUI (Python changed) and
hard-refresh the browser (JS changed).

## 2026-09-27 — Every prompt view frames the prompt, and cascades its sections

After liking the framed background of the Ctrl attempt blocks, the owner asked for the
same treatment on the normal tooltip and the Shift one: the prompt on a really dark shade
of its chunk color, and, cascaded inside, one block per prompt section named by the
prompt's own field labels (summary, detailed_description, overall_soundscape, ...).

- `web/unlimited_preview.js` gained `promptSections(text)`: it matches the authored field
  headers (`^[ \t]*(subject_definitions|summary|retention_analysis|detailed_description|
  integrated_multimodal_description|overall_soundscape|non_diegetic_music|
  non_diegetic_audio)[ \t]*:`) and returns each section's range plus its header end, with
  a leading section when text precedes the first header. Every section keeps its header
  verbatim, so the prompt text is only ever grouped, never rewritten; when no header
  exists (the normal view shows the description body alone) there is exactly one section.
- `promptSectionBlock(text, chunk, shotRanges, colors, chunkColor)` draws the outer frame
  on `color-mix(in srgb, <chunk color> 18%, black)` with a 3px chunk-color left border,
  and one cascaded block per section on `color-mix(in srgb, <chunk color> 32%, black)`
  with a 2px border and a bold label in `color-mix(in srgb, <chunk color> 55%, white)`.
  The section header is emphasised where it stands rather than repeated above the text.
- Coloring is not recomputed per section: `coloredShotPromptSegments` still maps the
  markers over the whole prompt (so a marker keeps the shot it always had), and the new
  `appendColoredRange(parent, segments, offsets, start, end)` renders a character range
  of that mapping, including the literal `<d>` bolding. Sections are therefore frames
  around one continuous color mapping, not a re-coloring of each piece.
- The Ctrl attempt view is deliberately untouched: its per-attempt tint is what tells the
  attempts apart, and each attempt is already shown as the description section alone.

Validation: `tests/test_preview_prompt.js` — `collectPromptSpans` now walks the nests (and
falls back to the finished player's flat tooltip), the literal-text invariant still holds
in both views, and `checkPromptBlocks` pins the outer shade/border, one section block per
label, the lighter section shade, the in-place bold headers, and that the sections cover
the prompt exactly. All nine JS suites, `tests/test_chunk_director_helpers.py` (174),
`tests/test_teacher_audio_cache.py` (32). The DOM was also dumped offline for both views
(normal: frame + one unlabelled section; Shift: frame + subject_definitions, summary,
detailed_description, overall_soundscape) before the scratch script was removed. JS only,
so a hard browser refresh is enough.

## 2026-09-27 — A drag shows the frame that is really at the pointed position

The owner's report: while the sub-chunks of a chunk were rendering, dragging across the
whole chunk swept the *rendered* sub-chunk through the drag, instead of placing it where it
was rendered with black everywhere the later sub-chunks had not arrived yet. The cursor
itself was correct; the picture it produced was not.

The cause was one line in `seekTimeline`: the pointer's fraction of the chunk span was
applied to the frames that exist
(`show(targetChunk, Math.round(fraction * frames.length - 1))`), so a chunk holding 20 of
its 80 frames stretched those 20 across the whole span.

- `scrubFrameIndex(chunkIndex, fraction)` resolves the pointer's position instead of a
  frame count: the position is `floor(fraction * span)` chunk-local, `span` is the chunk's
  planned length, and the frame that answers it is the group's decoded frame whose
  *absolute* `frameNumbers` entry equals the planned start plus that offset. Frame numbers
  are the same coordinate as `chunkRanges[i].start/end` (the transport's own frame label
  already matches them), so a group that starts later, that carries overlap-replacement
  frames or that holds latent steps with gaps is located by its own numbers, not by
  counting from zero. No entry means the pass has not rendered that position: `-1`.
- `seekTimeline` shows that frame when one exists, and `showEmptyPosition(chunk)` when none
  does. The empty position sets `playing = chunk`, `playingFrame = -1` (so the transport
  reports `—/N` and no frame label), displays `EMPTY_FRAME_SOURCE` — the same black asset
  the teacher-audio placeholder groups use, now one shared constant instead of two inline
  copies — and leaves `timelineScrub` to keep the playhead under the pointer.
- `restorePlayback()` keeps that black position across later redraws while paused, and
  starts from the chunk's first frame if playback is asked for, so pressing play never
  leaves a black picture behind.

Owner's choice, applied everywhere: a chunk the pass has not reached at all is black now
too, replacing the earlier "keep the last displayed picture" behaviour, so a position on
the timeline always shows what exists there.

Validation: `tests/test_taomate_timeline.js` — `scrubFrameIndex` is run through its own
sources: a 20-of-80 chunk keeps frame 19 at position 19/80 and answers nothing at 20/80,
0.5 and 1; a group numbered 8,9,12,13 is found by its numbers with its gap answering
nothing; a later chunk is read at its own start with `chunkRanges[i].start`; a group with
no frame numbers falls back to its own list; a missing chunk answers nothing; plus source
assertions that the fraction-times-length mapping is gone, that a frame-less position goes
black, that it reports no frame, that it survives a redraw and that play recovers it. All
nine JS suites and `tests/test_chunk_director_helpers.py` (174) pass. JS only, so a hard
browser refresh is enough.

## 2026-09-27 — One chunk card from Chunk N, and a color per prompt section

Two follow-ups on the tooltip framing, from the owner: the background must start at the
`Chunk N` line rather than at the prompt heading, it must be much darker, and every prompt
section needs its own distinct color with a darker shade of that color behind it.

- `createColoredChunkTooltip`'s `line()` now takes its container, and `show()` builds one
  **chunk card** immediately after the help line. The card holds the chunk number, the
  timing lines, the retention analysis, the prompt heading and everything under it, on
  `border-left:3px solid <chunk color>` over
  `color-mix(in srgb, <chunk color> 7%, black)` — much darker than the previous 18% frame.
  The help line deliberately stays outside, because it describes the tooltip, not the chunk.
- The old chunk-colored prompt frame is gone: the card *is* that frame, one level up.
  `promptSectionBlock` became `promptSectionBlocks`, returning one block per section
  (`background:color-mix(in srgb, <section color> 14%, black)`, 2px border in the section
  color, header in `color-mix(<section color> 55%, white)`).
- New `PROMPT_SECTION_COLORS` gives every field its own stable color, so a section is
  recognizable in any view and any prompt: `subject_definitions` #f08a4b, `summary`
  #f4b942, `retention_analysis` #d6cf57, `detailed_description` and
  `integrated_multimodal_description` #45b7d1, `overall_soundscape` #72c472,
  `non_diegetic_music` #a98bea, `non_diegetic_audio` #ef6f91, and prose before the first
  label keeps the neutral #b8b8d0.
- The Ctrl attempt view keeps its per-attempt tints and its hearing blocks (still a 32%
  shade of the retry's own color, deliberately darker than a section) and now sits inside
  the same card, so every view starts its background at `Chunk N`.

Validation: `tests/test_preview_prompt.js` — walkers now traverse the tree instead of one
level (`elementTree`, `leavesOf`, `promptBlocks`, `contentBlocks`, `cardOf`, `boxAccent`),
and the checks pin that the card starts at `Chunk 1` while the help line stays outside,
that the timing lines and the heading are inside it, that the card is the 7% chunk shade
with a 3px chunk-color border, that a section's background is derived from its *own*
border color at 14%, and that the three Shift sections keep three distinct colors
(#f4b942 / #45b7d1 / #72c472). All nine JS suites and
`tests/test_chunk_director_helpers.py` (174) pass. The rendered tree was dumped offline for
the normal, Shift and Ctrl views before the scratch script was removed. JS only, so a hard
browser refresh is enough.

## 2026-09-27 — A sub-chunk's frames are merged into its chunk, not thrown away

Owner's report: dragging the timeline over a chunk that had finished sub-chunks *and* not
yet rendered areas made the preview blink black; the arrow keys were fine. The scrub maths
were not at fault — the finished sub-chunks were not in the client any more.

A TaoMate phase reports only its own frames, numbered where they really sit
(`preview.py`: `output_start = chunk.get("subchunk_start", chunk["output_start"])`, fed by
`nodes.py` `set_subchunk(index, frames(phase["video_start"]), phase["frame_end"] - 1, ...)`),
and the `chunk` handler **replaced** the chunk's group with them. So the moment sub-chunk 2
landed, the group held only frames 17…38 of an 80-frame chunk: sub-chunk 1's frames were
gone, the corrected scrub said "nothing rendered" for 0…16, and dragging across the chunk
therefore alternated picture and black.

- New module-level `mergeFrameGroup(oldGroup, frames, durations, numbers)` writes the
  incoming frames into the chunk's group by frame number, keeping `frames`, `durations` and
  `frameNumbers` parallel and sorted. Entries are written last so a phase's
  replacement-overlap frames overwrite the earlier version, and a frame whose number is
  missing from a payload cannot be placed, so that payload becomes the group (an unnumbered
  event must not silently drop what is already rendered).
- The `chunk` handler merges every non-finalized event and lets `chunk_final` replace the
  group outright, since the full-VAE decode is authoritative for the whole chunk.
  `outputStart`/`outputEnd` now follow the merged extremes, so the audio clock keeps a
  correct frame-number base for a group that starts later in the chunk (the audio-first
  placeholder covers the whole span already, so its black frames stay as the holes).
- No clock or sampling change: playback already maps frames to the audio clock by frame
  number, so a hole between two sub-chunks is simply skipped.

Validation: `tests/test_preview_prompt.js` — `checkFrameMerge` covers the merge order, the
replacement overlap taking the new version, durations staying parallel, an unordered event,
an unrendered position staying unrendered, the first sub-chunk of a chunk, an unnumbered
payload, plus source assertions that live events merge and the final one replaces.
`tests/test_preview_audio.js` needed the helper inside its own function slice (its
assertions are unchanged and still pass). An offline walk of a 4-sub-chunk, 80-frame chunk
(scratch script, then removed) showed 21 → 39 → 60 → 80 frames with 0 empty positions at the
end, each frame resolving at its own position. All nine JS suites,
`tests/test_chunk_director_helpers.py` (175), `tests/test_teacher_audio_cache.py` (32),
`tests/test_taomate.py` (42), `tests/test_taomate_kv_records.py` (19),
`tests/test_video_io.py` (25). JS only, so a hard browser refresh is enough.

## 2026-09-27 — A position shows the image that covers it, and the arrow keys stop seeing black

Owner's follow-up: the arrow keys that were fine before showed black between frames, and the
drag did too. Their diagnosis was right: the preview decodes one image per H3 temporal
token, not per frame, so the image numbered 1 stands for frames 1…4 — and the scrub maths
treated the frames a token covers as unrendered. The arrows regressed for a second reason:
merging a live sub-chunk into the teacher-audio placeholder kept the placeholder's black
images in the group, so stepping walked through them.

- `scrubFrameIndex` now returns the image that *covers* the pointed position: the last
  decoded frame whose number is at or before it (the numbers are ordered, so a scan keeps
  the last one). "Nothing" is returned only when the position is past the chunk's rendered
  end (`group.outputEnd`) or before its first decoded image, which is what keeps the
  un-rendered tail black.
- `outputEnd` now keeps the phase that finished: `max(data.output_end, last merged frame
  number)`, because an image's own token frames are rendered too. Shrinking it to the last
  image would have blacked out up to three frames of a phase that just completed.
- The `chunk` handler passes `null` as the group to merge when the old group is an
  audio-first placeholder: the placeholder is span coverage with a black picture, not
  frames, so a real sub-chunk replaces it (its audio fields still carry over through
  `Object.assign`). A chunk with teacher audio but no video yet keeps its placeholder and
  its black picture exactly as before.

Validation: `tests/test_taomate_timeline.js` — the scrub checks were rewritten for the
covering rule (four token images covering 0-3/4-7/8-11/12-15: position 3 shows image 0,
4 shows image 1, 15 shows image 3, 16 and beyond answer nothing; a later chunk read by its
own numbers; a position before its first image black; a group without numbers unchanged),
plus source assertions for the rendered end. `tests/test_preview_prompt.js` asserts the
placeholder is replaced rather than merged into, and that `outputEnd` keeps the phase end.
An offline walk (scratch, then removed) over a token-sparse 333-frame chunk rendered
through frame 60 gave 61 pictured positions and the first black exactly at the rendered
end. All nine JS suites, `tests/test_chunk_director_helpers.py` (175),
`tests/test_teacher_audio_cache.py` (32). JS only, so a hard browser refresh is enough.

## 2026-09-27 — The prompt tooltip scrolls when its text is taller than the panel

The owner asked that a tooltip whose text is larger than its area scroll with the mouse
wheel. It already capped at `max-height:min(70vh,640px)`, but with `overflow:hidden` and
`pointer-events:none`, so anything past the cap was silently cut off and the wheel could
not reach it.

Owner's choice: the tooltip takes the mouse, so its scrollbar is draggable and its text is
selectable, rather than forwarding the wheel from the document while staying transparent.

- Tooltip style: `overflow-y:auto;overflow-x:hidden` with `scrollbar-width:thin` and
  `scrollbar-color:#6a6a6a transparent`, `pointer-events:auto`, `user-select:text`.
  Everything else about the look is unchanged.
- Taking the mouse creates the problem the old `pointer-events:none` was hiding: the
  timeline's `mouseleave` closed the tooltip the moment the pointer moved towards it, so
  the scrollbar could never be reached. `createColoredChunkTooltip` now tracks its own
  `hovered` state from `mouseenter`/`mouseleave` and reads a `leaveGuard`: the extension
  registers `event => Boolean(event.relatedTarget && timelineShell.contains(event.relatedTarget))`,
  so leaving for the timeline hands the hover back without closing, and leaving for
  anything else closes it. `timelineShell`'s `mouseleave` returns early while
  `chunkTooltip.isHovered()`. The tooltip also stops `pointerdown` from reaching the panel
  or the canvas, so scrolling or selecting text can never start a node drag or a pan.
- `hide()` clears `hovered` and resets `scrollTop`, so a tooltip that opens again starts at
  the top, while a rebuild of the content it is already showing (a retry landing, a timing
  update) keeps the reader where they scrolled to — the reset is deliberately in `hide()`
  and not in `show()` for that reason.

The finished-video player's tooltip (`createFinishedChunkTooltip`) is untouched, as the
owner scoped this to the preview node.

Validation: `tests/test_preview_prompt.js` — the element fake gained
`addEventListener`/`dispatchEvent`, and `checkTooltipScroll` pins the style flags, that a
shown tooltip is not hovered, that entering marks it hovered, that leaving for the timeline
keeps it open (guard true) while leaving elsewhere closes it at scrollTop 0, and that a
rebuild keeps scrollTop, plus source assertions for the guard, the timeline's early return
and the press stop. All nine JS suites and `tests/test_chunk_director_helpers.py` (175)
pass. JS only, so a hard browser refresh is enough.

## 2026-09-27 — Ctrl+wheel over the timeline scrolls the prompt tooltip

The owner reported the tooltip scroll doing nothing and Ctrl+wheel zooming the page instead.
The cause is in the frontend, not in the tooltip: `document.body` carries
`K(document.body, "wheel", …, { capture: true, passive: false })` which, whenever Ctrl or
Meta is held, calls `preventDefault()` and `stopPropagation()` and then switches workflow
tabs. The tooltip's own listener therefore never ran, its native scroll was cancelled, and
the browser's ctrl+wheel page zoom took over.

- `createColoredChunkTooltip` now takes the wheel from a `window` **capture** listener,
  which runs before that `document.body` capture handler: while the tooltip is showing a
  prompt taller than itself it calls `preventDefault()` + `stopImmediatePropagation()` and
  scrolls `scrollTop` itself (line-mode deltas ×16, clamped to
  `scrollHeight - clientHeight`), so the frontend's tab switch, the graph's zoom and the
  browser's ctrl+zoom are all kept out of it.
- The guard is deliberately *not* the pointer being on the tooltip: the tooltip only exists
  while the pointer is on the timeline, and moving off the timeline hides it immediately, so
  the pointer is never over it. What decides is that a tooltip is showing content that
  overflows; a tooltip that fits, or none at all, leaves the wheel to the graph.
- `remove()` detaches the window listener, so a deleted node leaves nothing behind.

Validation: `tests/test_preview_prompt.js` — the fake harness gained a `window` with
listeners, a `remove()` on elements and a `dispatchWindowEvent`, and `checkTooltipScroll`
now pins the wheel: a tooltip that fits and a hidden tooltip let the event through
untouched, while a showing, overflowing tooltip swallows it (`preventDefault` +
`stopImmediatePropagation`) and moves the text by 120px, converts a 3-line delta to 48px
and clamps at the end. All nine JS suites and `tests/test_chunk_director_helpers.py` (175)
pass. JS only, so a hard browser refresh is enough.

## 2026-09-27 — The retry splits a group in two, and the pace is the shot's own

Two follow-ups from the owner's renders of the monologue prompt.

**A retry never makes more than two groups.** The retry that isolates the failed phrase
produced three groups when the *middle* phrase failed (18:44: `People are strangely afraid
of silence.` and `Personally, I find silence comforting.` both failed), and the owner heard
audio worse than the take it was meant to fix. So `_split_out_damaged_phrases` now splits
each group at exactly one place: after the first phrase when the first phrase failed,
otherwise before the last phrase. Words, order, language tags and the retry's faster
delivery are untouched, a group with one phrase or nothing damaged is still left alone,
and a take that failed everything still only becomes two groups.

**The pace bands are relative to the shot.** The owner's objection: chunk 2 is the densest
chunk of the shot and was told "speaking slowly, with pauses", while the sparsest chunks
were told nothing — the absolute bands were backwards. `_dialogue_delivery_instruction(rate,
pace)` now compares the chunk's own words-per-second with the shot's own pace (the
`total_weight / available_seconds` the cutter already computed), with the owner's approved
dead band:

| ratio to the shot's pace | instruction |
|---|---|
| ≥ 1.50 | speaking very quickly, rushing the words |
| 1.25 – 1.50 | speaking quickly |
| 1.10 – 1.25 | speaking a little faster than usual |
| 0.90 – 1.10 | nothing |
| < 0.90 | speaking slowly, with pauses |

On the owner's prompt (shot pace 0.63x) that is chunk 1 slowly, chunk 2 quickly, chunk 3
nothing, chunk 4 slowly, chunk 5 a little faster, chunk 6 nothing. `ACTION_MAX_SPEECH_RATE`
(1.40) is still the *absolute* pace no human reaches, and the render stops there; the band
list now ends in `float("inf")` so a ratio past the last edge keeps the strongest phrase
instead of falling through to nothing, which is what the first cut of this change did.

Validation: `tests/test_chunk_director_helpers.py` (175; new relative band test, a
whole-delivery test that proves the instruction rides only cut pieces, the cut-piece test
with its two bands, and `test_a_retry_never_makes_more_than_two_groups`),
`tests/test_taomate.py` (42), `tests/test_teacher_audio_cache.py` (32),
`tests/test_taomate_kv_records.py` (19), nine JS suites.

## 2026-09-27 — A word's tail no longer keeps the dialogue in a silent sub-chunk

The owner watched the 19:58 render: sub-chunk 1.4 lost its dialogue and the video stayed
silent ✔, but 1.3 still carried the dialogue and moved the mouth with no audio driving it.
Reading the cached take's word times against the sub-chunk windows showed why: 1.3 covers
4.42s-6.55s and the only word in it was the tail of `minute.` (4.10-4.70, 0.28s) followed by
1.85s of silence, and `_subchunk_speaks` counted *any* overlap as speech.

The owner's rule now: a word belongs to the sub-chunk it **starts** in — the voice begins
there — and a word that started earlier only counts while its sound fills at least
`ACTION_MIN_SPEECH_SHARE` (0.25) of this sub-chunk. On chunk 1 that leaves 1.1, 1.2, 1.5 and
1.6 speaking and 1.3 and 1.4 dialogue-free, which is the render the owner asked for. A
transcript that never arrived still keeps the prompt, and a take with no words at all is
still silent everywhere.

Validation: `tests/test_chunk_director_helpers.py` (175) — the sub-chunk test now pins a
word starting inside a window, a word whose tail alone falls in the next one (removed), a
word that really runs across the boundary (kept), an empty take and a missing transcript —
plus `tests/test_taomate.py` (42), `tests/test_teacher_audio_cache.py` (32) and
`tests/test_taomate_kv_records.py` (19).

## 2026-09-27 — kv_remember: the plugin had been dropping one remembered sub-chunk

The owner asked for a `kv_remember` widget after suspecting the camera jump at a chunk seam
came from what the KV cache feeds a new sub-chunk, and they were right about the cause: the
vendored upstream policy in `python/taomate_upstream/cache.py:211-219` is

```python
if block_count <= 2:
    return
recent_start = max(1, block_count - 2)
selection = [(0, True)] + [(index, False) for index in range(recent_start, block_count)]
```

— the video-only sink plus the **last two** committed sub-chunk blocks — while
`VideoSubchunkKVCache.prepare_commit` had been trimming to `[(0, True), (count - 1, False)]`,
i.e. the sink plus **one** recent block. That deviation shipped with the TaoMate backend
itself (`6939e1a`, `9e84be8`, both 2026-09-26), so it is not a recent regression; it also
made the plugin evict earlier than upstream (at three blocks upstream keeps all three, the
plugin already dropped the middle one).

- `parse_kv_remember("1,-2,-1")` validates the new widget once per render and raises a clear
  `ValueError` for `0`, words or an empty spec, so a typo stops the render instead of
  quietly narrowing the window.
- `kv_remember_selection(entries, block_count)` resolves positive entries from the start
  (1 = the first sub-chunk) and negative ones from the newest (-1, -2, …), keeps the first
  block video-only exactly as upstream ages it into a sink, skips blocks the render has not
  reached and de-duplicates. For the default it reproduces upstream's selection exactly at
  every block count.
- `VideoSubchunkKVCache` now carries the parsed spec and trims to it; upstream's
  `block_count <= 2` guard is kept, so the first sub-chunks stay unaged as before.
- The widget sits above `kv_cache_compression`, and `kv_cache_compression` now defaults to
  `turboquant4` in both the node schema and the legacy signature.

Validation: `tests/test_taomate.py` (46) — the default equals upstream's formula at every
block count, `"1,-1"` reproduces the old behaviour, `"-2,-1"` drops the sink,
`"-3,-2,-1"` keeps three, malformed specs raise, and `prepare_commit` applies the spec
through `_retain_commit_rows`; the existing cross-rollover comparison against the upstream
cache now expects the sink plus two recents and drops its old identity check, because the
plugin trims before each append while upstream trims after it (the retained bytes are what
the test compares). `tests/test_chunk_director_helpers.py` (175),
`tests/test_teacher_audio_cache.py` (32), `tests/test_taomate_kv_records.py` (19) pass too.

## 2026-09-28 — No pace band on the head of a cut delivery, and it goes with the line

Two corrections to the pace band.

**The head of a delivery is not a cut delivery.** The owner: "it wasn't a cut dialogue that
the cutter created, it was the start of the big dialogue that the cutter cuts — as a rule,
no speech speed should be added before the start of a dialogue that has been cut." The
condition I had added earlier (`len(selected) < len(weighted_words)`, i.e. "this piece
holds only part of its block") fires on the *first* chunk of a monologue too, so chunk 1
was told "speaking slowly, with pauses". It is now `bool(selected_index)` — a piece earns an
instruction only when its first spoken word is not its block's first word, or it carries
words from the previous chunk — so the head stays clean and the continuations keep theirs
(on the owner's prompt: chunk 1 none, 2 quickly, 3 none, 4 slowly, 5 a little faster,
6 none).

**A removed dialogue takes its delivery band with it.** A silenced sub-chunk kept
"Speaking slowly, with pauses, <Subject 1> (S1) then says:" in front of nothing, because
the band sits *before* the clause `_prompt_without_dialogue` removes. `SPEECH_RATE_PREFIX`
(`(?:speaking [^.\n]*?, )?`) is now part of both the `<Subject N>` and the bare lead
patterns, so the band, the clause and the line go together.

Validation: `tests/test_chunk_director_helpers.py` (175) — the cut-delivery test now
asserts the head chunk gains no instruction while the continuation keeps "Speaking very
quickly, rushing the words,", and the silent-prompt test asserts a banded prompt loses the
band, the header and the line. `tests/test_taomate.py` (46),
`tests/test_teacher_audio_cache.py` (32), `tests/test_taomate_kv_records.py` (19) pass.

## 2026-09-27 — Fullscreen playback floats the chrome over the whole picture and fades it

Owner's request for the preview node, then for every fullscreen player in the plugin: with
the picture fullscreen, the timeslider, the info under it and the top buttons should fade out
after three seconds of a still pointer and come back on movement, and the video should stay
fullscreen with everything floating on top of it. The owner chose the whole transport bar
with the status line and the top buttons, translucent bars, the graphs overlaid and fading
with the rest, and chrome kept visible while playback is paused.

Two panels have their own fullscreen: `web/unlimited_preview.js` (the preview node) and
`web/finished_video_player.js` (the Save/Load Video nodes). They already duplicate their
panel and tooltip code, so the helper is duplicated too rather than adding a shared module
both would have to import.

- New file-scope `createControlsAutohide(elements, { delayMs = 3000, isFullscreen,
  isPlaying, doc = document })` in both files: `mousemove` and `fullscreenchange` show the
  elements and arm a three second timer, which fades them to `opacity:0` +
  `pointer-events:none` (a faded control must not answer a click) only while the player is
  fullscreen *and* playing; `isVisible()` and `stop()` (clears the timer, detaches both
  listeners, restores) are exposed, and `arm()` is what a state change calls.
- The transport bar and the status line now live in a `controls` wrapper (`display:flex;
  flex-direction:column`), which is invisible in the normal layout and is what fullscreen
  floats; the finished player puts its comparison footer in it too. `root` gained
  `position:relative`.
- `applyFullscreenLayout()` runs on every `fullscreenchange` (from the existing handler,
  alongside `updateFullscreenButton()`): the viewport becomes `position:absolute;inset:0`
  so the picture is full-bleed and `object-fit:contain` keeps its shape, `controls` is
  anchored to the bottom, and (preview only) `graphs` gets the `min(430px,44vw)` top-left
  corner. Over that picture the bars turn translucent (`rgba(20,20,20,.72)` transport,
  `rgba(24,24,24,.72)` status, `rgba(18,18,18,.72)` graphs) with `backdrop-filter:blur(3px)`
  and are restored to their exact opaque colors on exit.
- Play state re-arms from the player itself: the preview's `setPaused()` calls `arm()`
  (`isPlaying: () => !paused`), and the finished player listens to its `media` element's
  `play`/`pause`/`ended` events (`isPlaying: () => !media.paused && !media.ended`), so
  pausing brings the chrome back and a paused fullscreen never hides it.

Validation: new `tests/test_preview_controls_autohide.js` runs **both** copies of the helper
for real under `vm` with fake timers and a fake document: nothing arms outside fullscreen,
fullscreen arms exactly 3000ms and hides, movement restores and re-arms, a paused player
arms nothing, leaving fullscreen restores and clears, `stop()` detaches both listeners, the
transition is set, and source assertions cover the wiring of both panels (bottom-floated
translucent chrome, the graphs floating with the rest in the preview, the comparison footer
in the save node, each panel's own play state). All ten JS suites pass, plus
`tests/test_chunk_director_helpers.py` (175). The change is JS only, so a hard browser
refresh is enough.

## 2026-09-27 — The sub-chunk view shows what that slice of the take heard

Owner's follow-up on the Ctrl+Shift view: each sub-chunk should also show the teacher audio
transcript belonging to its own time slice. The take's Whisper transcript is per chunk
(`{"text", "words": [{word, start, end}]}`, seconds from the chunk's audio start) and was
never published to the preview, so there was nothing to show.

- New `_subchunk_transcript(phase, chunk, transcript)` in `nodes.py` cuts those words to one
  phase's own window — `(audio_start - chunk_audio_start) / AUDIO_LATENT_FPS` (**40**, not
  25: the first test expectation was wrong about this) to `(audio_end - chunk_audio_start) /
  AUDIO_LATENT_FPS`, with the same overlap rule as `_subchunk_speaks`, so a word straddling
  a boundary belongs to both windows. Words are joined in order.
- `_publish_subchunk_prompts(..., chunk, transcript)` now publishes `subchunk_transcripts`
  (one entry per phase, `""` where the slice holds no word) beside the silent-sub-chunk
  keys, and is called for every chunk rather than only for chunks with a silent sub-chunk.
  Both call sites pass the transcript they already have: `audio_transcripts[index]` in the
  video pass, `recognized` in the audio-first pass. A transcript that never arrived or holds
  no words publishes **nothing**, so the tooltip never claims a silence it cannot know.
- `preview.set_subchunk_prompts(chunk, numbers, prompt, transcripts=None)` publishes it (and
  publishes transcripts alone for a chunk that speaks throughout), `_cache_payload` keeps
  the empty slots so a transcript cannot drift onto another sub-chunk,
  `build_teacher_audio_preview_snapshot` restores it, and `teacher_audio_payload()` carries
  `subchunk_transcripts` so a reload keeps it.
- `web/unlimited_preview.js`: the Ctrl+Shift branch adds
  `Teacher audio transcript (this sub-chunk's window):` and, under it, the words in their
  own plain block (grey border, no shot colours — it is speech, not prompt text); an empty
  entry reads `nothing heard in this window`, which is exactly why that sub-chunk's prompt
  dropped the dialogue. A chunk without a transcript list shows no line at all. The
  transcript joins the tooltip signature and the `chunk_metadata` handler, so a late
  transcript refreshes an open tooltip.

Validation: `tests/test_chunk_director_helpers.py` (175; the window cutting with one word
per window, a straddling word in both, a silent window, an absent/wordless transcript
publishing nothing, the transport publishing both keys and transcripts alone, the cache
keeping the empty slot), `tests/test_teacher_audio_cache.py` (32; a reload keeps
`subchunk_transcripts` with its empty slot), `tests/test_preview_prompt.js` (the label, the
plain block, the empty-window note, no line without a list), plus all ten JS suites and
`tests/test_taomate.py` (42), `tests/test_global_prompt_sections.py`. Python changed, so
this one needs a ComfyUI restart and a hard browser refresh.

## 2026-09-27 — The timeline shows where each chunk's sub-chunks are, in luminance only

Owner's request: represent the sub-chunks inside a chunk on the timeline as a variation of
the chunk color's luminance, positioned by the frames each sub-chunk really owns (their
choice: the real `taomate_phase_work` counts, with the pending part of a filling chunk
showing the same treatment).

**Second cut, after the owner saw it.** The first version ran a sawtooth — the chunk color
brightening towards each sub-chunk's middle and back at its boundaries — and the owner
rejected it twice: the transitions should not be smooth, and the split should be *odd
sub-chunks in the chunk's own color, even sub-chunks a step brighter*, each band flat with a
**sharp** step at every boundary. That is what ships.

The timeline paints every chunk into one `linear-gradient` on `timelineTrack`, so the
structure arrives as extra stops, not as elements:

- New file-scope helpers in `web/unlimited_preview.js`: `mixHex(from, to, amount)`,
  `subchunkEdges(work)` (each sub-chunk's end as a fraction of the chunk; `null` when the
  chunk has one sub-chunk or none of the counts is known) and `subchunkTimelineStops(stops,
  { color, start, end, work, alpha, renderedFraction })`. Bands are `edges` (or one band for
  the whole chunk), the odd ones keep `color`, the even ones take
  `mixHex(color, "#ffffff", 0.3)`, and each segment emits its color at both ends — so a
  boundary carries both colors at the same percentage, which is what makes the step sharp.
  Percentages are rounded to a thousandth so a long timeline keeps a readable string.
- The loop in `renderTransport` calls it for both halves of a filling chunk (the rendered
  part keeps its `e8` alpha, the pending part its `color-mix(… 50%, black)` recipe, in the
  same bands), and for a chunk with no progress (`e8`, or `35` while it has no frames). A
  band the render has half filled is split a hundredth of a percent before the boundary, so
  the rendered/pending step stays sharp and inside its own band. A chunk without known
  sub-chunk frames produces exactly the two flat stops it always did.
- Nothing else changed: the chunk colors, the completed/pending split, the cached and
  overlap overlays, the shot brackets and the playhead are untouched.

Validation: `tests/test_taomate_timeline.js` runs the real stop-building loop, so the
sub-chunk cases go through the panel code: `mixHex` positions, edges for equal and unequal
work (`[31, 93]` of 124 frames puts the boundary at the quarter mark), `null` edges for a
single-sub-chunk or unknown chunk, the two-band stop list of a pair, the four alternating
bands of a 4-sub-chunk chunk (its starts and ends back in the chunk color), a filling chunk
whose dimmed part keeps the alternation with the render step inside the band, and the
finished-sub-chunk case below. An offline dump of a real four-sub-chunk chunk (`[21, 22, 22,
21]`, 8 stops, boundaries at 24.419/50/75.581%) confirmed the sharp steps before the scratch
script was removed. All ten JS suites and `tests/test_chunk_director_helpers.py` (175) pass.
JS only, so a hard browser refresh is enough — no ComfyUI restart for this one.

**Third cut: a finished sub-chunk still ramped.** The owner: "there's still a gradient when
a subchunk is showing images". The split of rendered from pending was `renderedFraction < to`
(exclusive), but a sub-chunk that finishes puts the render boundary *exactly* on the band
edge — so that band took the rendered color from its start and the dimmed one at its end and
the gradient ramped across the whole sub-chunk. Reproduced offline for
`work: [31,31,31,31], renderedFraction: 0.25`, which printed `#45b7d1e8 0%` straight to
`color-mix(… 50%, black) 25%`. `subchunkTimelineStops` now splits when
`renderedFraction <= to`, keeps the band rendered up to `max(from, renderedFraction - 1e-4)`,
starts the pending color at the boundary and only paints the band's pending tail when the
boundary is strictly inside it (the clamped edge also stops a non-monotonic stop list when a
boundary sits a hair above a band edge). Same dump now: `#45b7d1e8 0%`, `#45b7d1e8 24.99%`,
`color-mix(… 50%, black) 25%`, then the pending bands — flat everywhere.
`tests/test_taomate_timeline.js` carries that state as a regression case
(`draw(62, …, [62, 62])`).

## 2026-09-28 — The tooltip follows the sub-chunk, so the chord is gone

The owner's conclusion: the Ctrl+Shift chord is unnecessary. A chunk with sub-chunks should
read sub-chunk by sub-chunk in the ordinary tooltip — no key for that sub-chunk's
`detailed_description`, Shift for its whole prompt — and the chord is removed. Their two
extra answers: holding Ctrl+Shift does **nothing** (the prompt text is cleared, the card keeps
its numbers), and the sub-chunk's teacher-audio transcript belongs in **every** view, below
the prompt text that view showed.

- `setChunkTooltip` now resolves the pointed sub-chunk *before* anything else and builds the
  tooltip's text from it: `fullPrompt = subchunkBody?.prompt || range.h3_prompt` and
  `description = chunkPromptDescription({ h3_prompt: fullPrompt, gemma_detailed_description:
  range.gemma_detailed_description })` — the object form keeps the Gemma fallback that the
  range form used to give. Everything downstream (`showFullPrompt`, the prompts list, the
  signature) then uses those, so a silent sub-chunk shows its dialogue-less body with no
  further branching, and a chunk without sub-chunks is byte-for-byte today's behaviour
  (`subchunkAt` returns `{number: 1, count: 1}`).
- Views: `showAttempts = ctrl` (Ctrl+Shift included, as the owner asked — Ctrl still owns the
  attempts), `showFullPrompt = !showAttempts && shift`, `showNothing = ctrl && shift`, which
  prints `Prompt text hidden.` where the prompt would be. `showSubchunk` is gone from the
  code and from both test suites.
- `Chunk N.M of N: …` heading stays, now shown by default whenever the pointed chunk has more
  than one sub-chunk, and the transcript line (`Teacher audio transcript (this sub-chunk's
  window):`, its own plain block, `nothing heard in this window` when empty) moved out of the
  removed branch to the end of the card, so it lands under the detailed_description of every
  view, including the Ctrl attempts one.
- The local holding the named sub-chunk is called `pointed`, not `position`: inside `show()`
  a `const position` shadowed the tooltip's own `position(event)` and broke the final
  placement call. The test harness caught it before the browser did.

Validation: `tests/test_preview_prompt.js` — `checkSubchunkView` now covers `subchunkAt`
boundaries and `subchunkPrompt` (unchanged), the plain view showing the pointed sub-chunk's
description with the heading and transcript, the Shift view adding its sections up to the
sub-chunk's whole prompt, the empty-window note, no transcript line without a transcript,
the cleared Ctrl+Shift view (no prompt block, numbers kept, transcript still there), a chunk
without sub-chunks showing no sub-chunk line, and source assertions that the chord is gone
and the prompt comes from the sub-chunk.
`tests/test_chunk_director_helpers.py` (175) asserts the same on the source.
All ten JS suites, `tests/test_taomate.py` (46) pass. JS only, so a hard browser refresh is
enough.

## 2026-09-28 — The plain tooltip is a Shift section, not a grey block

Owner: "make the normal tooltip the same style as the shift version… it's currently
different." It was: the plain view drew the description as one neutral-grey block
(`PROMPT_SECTION_NEUTRAL`) with a `chunk detailed_description:` heading above it, while Shift
framed every section in its own palette colour with a bold in-place header. The owner also
settled what each view carries: **Shift shows everything including the retention_analysis**,
the plain view shows the detailed_description and the transcript.

- New file-scope `promptSectionField(label)` (the label without its colon, lowercased, now
  shared with `promptSectionColor`) and `PROMPT_DESCRIPTION_FIELDS`
  (`detailed_description`, `integrated_multimodal_description`); `promptSectionBlocks(text,
  chunk, shotRanges, colors, fields=null)` keeps only the sections a caller asks for.
- The plain view renders that one section of the *same* prompt the Shift view uses, so it
  now carries the section's own colour (`#45b7d1`), its bold `detailed_description:` header
  and the same 14% background. Its grey heading line is gone, because the section names
  itself. A prompt with no such field (a legacy chunk, or Gemma's own description) falls
  back to the unframed body exactly as before, and an empty one still shows the waiting
  line. The plain branch no longer trims the section's trailing newline — that is the
  literal prompt text, as in Shift.
- The retention analysis moved out of the plain view: it is shown when `showFullPrompt`
  (Shift's everything) or `showAttempts` (Ctrl's teacher takes), and never in the plain or
  cleared views.

Validation: `tests/test_preview_prompt.js` — the plain view now asserts one framed section
in `#45b7d1` named by its own header with no grey heading left, its literal text is
`detailed_description: <body>\n`, the retention analysis appears only with Shift, and a
description without the field still shows its body. The finished player's own tooltip keeps
its flat description (its expectations are scoped to the preview). An offline dump of both
trees confirmed the plain and Shift framings match. All ten JS suites,
`tests/test_chunk_director_helpers.py` (175). JS only, so a hard browser refresh is enough.

## 2026-09-28 — The sub-chunk number rides on the chunk title, in every view

Owner: drop the `Chunk 1.N of T: the chunk prompt` line and put `1.N / T` on the `Chunk 1`
colored title instead — in all views. Done by moving the resolution of the pointed sub-chunk
above the title: `setChunkTooltip` already computes it for the prompt text, and `show()`
draws `Chunk ${chunkNumber}.${pointed.number} / ${pointed.count}` (or the plain
`Chunk ${chunkNumber}` for a chunk with one sub-chunk, `pointed` being `subchunk.count > 1`)
before any view branch, so the plain, Shift, Ctrl and cleared views all carry it. The
heading's "no speech in its window, so the dialogue is dropped" hint went with the line —
the missing dialogue in the section body is what shows it now.

Validation: `tests/test_preview_prompt.js` pins the title as `Chunk 1.3 / 4` in the plain and
Shift views, in the Ctrl attempts view, and (unchanged) `Chunk 1` for a chunk without
sub-chunks; an offline dump of the four views printed `Chunk 1.3 / 4` as the second line of
each. All ten JS suites and `tests/test_chunk_director_helpers.py` (175) pass.

## 2026-09-28 — Each teacher take shows what it heard, and the retry's own is now published

Owner's batch on the tooltip: drop the `Chunk prompt (audio and video):` line in the Shift
view, call the first attempt `Teacher audio prompt:` (same tint), put a transcript below every
teacher-audio prompt, remove the old block that sat under the retry (it held the *first*
take's words) and show the retry's own words instead, and paint the hearing that *caused* a
retry red. Follow-ups: no label at all when Shift has a single prompt, and each hearing is
labelled `Teacher audio heard:`.

The blocker was data: the preview only ever received the transcript that *caused* a retry
(the rejected take's words), never the retry's own. Both flows do know it —
`publish_audio_first_chunk` and `publish_chunk_teacher_audio` transcribe the committed take
and know `audio_index in audio_retried_chunks`.

- New `_remember_audio_teacher_transcript(range_item, attempt, transcript)` keeps a list
  aligned with the prompt list (entry i = what take i heard, `""` for one not yet
  transcribed), padding to one slot per attempt. `_remember_audio_teacher_attempt` records
  the rejected take's hearing through it at `max(0, len(attempts) - 2)` — the take a retry
  replaced — so a range whose first prompt is the retry still lands in slot 0.
- The key is `audio_teacher_attempt_transcripts` (was `audio_teacher_retry_transcripts`),
  and both transcription sites now record the committed take's own hearing at attempt `1`
  when the chunk was retried, else `0`. The rename follows through `set_audio_prompt(…,
  attempt_transcripts=)`, the `chunk_metadata` key, `_cache_payload`,
  `teacher_audio_payload` (payload key `attempt_transcripts`) and
  `build_teacher_audio_preview_snapshot`.
- JS: the reader is `audioTeacherAttemptTranscripts`; the attempts view labels the first
  attempt `Teacher audio prompt:`, and under **each** attempt's description prints
  `Teacher audio heard:` with that take's own words — red (`#e05252` border, 26% red
  background, `#ffe3e3` text) when the attempt is followed by another, i.e. when it caused
  that retry, and in the attempt's own tint for the accepted one; an empty slot reads
  `nothing heard in this window`. The old `Heard instead (caused this retry):` block is gone.
  The Shift view prints no label line for a single prompt; two prompts keep `Audio prompt:` /
  `Video prompt:`, since those are what tell them apart.

Validation: `tests/test_chunk_director_helpers.py` (175) — the aligned slots, the retry's own
slot filled afterwards, the padded list, the renamed keys and payload; the old
`"Chunk prompt (audio and video):"` assertion became an absence plus the two-label case.
`tests/test_preview_prompt.js` — the attempt labels, one hearing per attempt with the right
words, red for the causing one, the tint for the accepted one, the empty-slot note, no
leftover old block, and the Shift view carrying no label for one prompt.
`tests/test_teacher_audio_cache.py` (32) keeps the renamed key through a reload. An offline
dump of both views confirmed the layout. All ten JS suites, `tests/test_taomate.py` (46) and
`tests/test_taomate_kv_records.py` (19) pass. Python changed, so this needs a ComfyUI restart
and a hard browser refresh.
