# SamplerCustomAdvanced-Unlimited development memory

## Goal and current architecture

This custom node replaces ComfyUI's stock `SamplerCustomAdvanced` for long
MiniMax H3 audio/video generation. It keeps the stock sampler contract and adds
the H3 prompt inputs needed to select prompt content by global time.

The current implementation performs one stock sampler call over the complete
requested AV latent. Inside every model prediction it evaluates smaller,
overlapping temporal windows and blends them before returning to the sampler:

```text
global x at sigma N
  -> predict window 1
  -> predict window 2
  -> predict window 3
  -> blend complete global prediction
  -> stock sampler updates global x once
global x at sigma N-1
```

This is deliberately different from the original implementation, which ran a
complete sigma schedule independently for every chunk and passed five finished
frames forward as continuation conditioning. Independent runs reset the noisy
latent and multistep solver history, allowing later chunks to reinterpret
details and sometimes repeat actions. The single-trajectory design preserves
the evolving latent and solver history for the whole video.

The node adds these inputs to the stock sampler interface:

- `clip`: the MiniMax H3 Qwen/CLIP model used by the upstream conditioner;
- `prompt`: the original MiniMax-formatted prompt;
- `fps`: converts prompt timestamps to global frames;
- `chunk_frames`: the largest temporal H3 transformer window;
- `debug`: logs and returns each rewritten window prompt;
- optional `images`: reconstructs Qwen visual presentation tokens.

The registered nodes are:

```text
SamplerCustomAdvanced-Unlimited
MiniMaxH3UnlimitedPreview -> MiniMax H3 Unlimited Preview
```

## Files and ownership

- `nodes.py` owns input validation, H3 window planning, prompt rewriting,
  Qwen re-encoding, execution-scoped guider setup, and stock sampler delegation.
- `windowed.py` owns H3 AV window evaluation, condition selection, global
  position layouts, overlap fusion, and per-window memory estimation.
- `preview.py` owns the optional model-patcher preview wrapper, Latent2RGB or
  tiny-VAE decoding, WebP encoding, and local websocket events.
- `web/unlimited_preview.js` owns the browser preview widget.
- `README.md` is the user-facing setup and limitation reference.

The implementation does not monkey-patch `MiniMaxH3`, `MiniMaxH3Model`, or
ComfyUI's sampler functions. The context handler is installed only on a cloned
`guider.model_options` dictionary for one node execution and restored in a
`finally` block.

## Why sampler state matters

ComfyUI's sampler maintains an evolving noisy latent `x` throughout the sigma
schedule. Multistep samplers also retain values such as `old_denoised` or
derivative histories. Those tensors are valid only for the same latent shape,
coordinates, and adjacent sigma steps.

They cannot be transferred from a finished chunk at sigma zero to a new chunk
starting at sigma maximum. The former continuation implementation therefore
had no true cross-chunk sampler memory even though it allocated a full-length
input and assembled a full-length output.

MiniMax H3 also has no causal KV cache or recurrent scene state. It uses
bidirectional attention and recomputes Q/K/V tensors for every transformer
layer and denoising step. Saving attention activations would be mathematically
stale for a new sigma and would retain the large tensors windowing is intended
to release.

The supported way to preserve state is consequently to keep one global `x`
and window only the expensive H3 prediction performed at each sigma.

## MiniMax H3 latent structure

H3 samples a nested AV latent:

```text
video: [B, 24, T_video, H/16, W/16]
audio: [B, 32, 2, T_audio]
```

Video latent positions cover the repeating pixel-frame pattern:

```text
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
```

A valid `17k + 5` pixel-frame sequence therefore has `5k + 2` video latent
positions.

| Pixel frames | Video latent positions |
| ---: | ---: |
| 5 | 2 |
| 22 | 7 |
| 39 | 12 |
| 124 | 37 |
| 243 | 72 |
| 362 | 107 |

Audio runs at 40 Hz while the normal video timeline is 24 FPS:

```text
T_audio = round(pixel_frames * 40 / 24)
```

`_chunk_plan` validates both streams and creates windows whose video starts
remain aligned to the five-token phase. The default 124-frame window is 37
video latent positions. Later windows start two latent positions before the
previous endpoint, giving five overlapping pixel frames. Audio boundaries are
derived from cumulative global frame positions, so the matching overlap
alternates between eight and nine audio positions where rounding requires it.

No overlap is trimmed from the result. The same global latent positions appear
in multiple model evaluations and their predictions are blended.

## One global stock sampler call

`nodes.py` calls `SamplerCustomAdvanced.execute` exactly once with the original
full nested latent and original noise object. This preserves ComfyUI's normal:

- noise generation;
- sigma schedule and selected solver;
- sampler callbacks and cancellation;
- model loading, low-VRAM offload, dtype, and backend policy;
- `output` and `denoised_output` contracts.

ComfyUI packs the nested video and audio into one flat sampler latent. The
custom context handler receives that flat `x` from `calc_cond_batch`, unpacks
it using the original full shapes, and extracts synchronized window views.

The handler allocates full-sized prediction accumulators and one-dimensional
video/audio weight counts for the duration of a model prediction. These are
execution-local tensors and are discarded after the prediction returns. It
does not create persistent model or module-level tensor caches.

## H3 window evaluation and fusion

For each planned window, `MiniMaxH3WindowedContextHandler`:

1. slices the global video latent along dimension 2;
2. slices the matching audio interval along dimension 3;
3. packs those two local tensors into ComfyUI's sampler representation;
4. selects the positive conditioning encoded for that exact window;
5. rebuilds `latent_shapes` and the H3 `PackedLayout` for the local AV shapes;
6. calls ComfyUI's normal conditional model evaluation;
7. unpacks the video/audio prediction;
8. accumulates it into the global locations using pyramid weights.

After every window, each accumulated position is divided by its total weight
and the complete AV prediction is packed and returned. Non-overlapping
positions reduce to their sole prediction. Overlapping positions smoothly
combine both neighboring predictions.

The overlap is the information bridge. At the next sigma, both windows read
the shared global `x` containing the previously blended overlap, allowing
appearance and motion information to propagate across boundaries over the
denoising trajectory. This is still an approximation of full attention:
tokens in distant non-overlapping windows never attend directly.

## Global H3 temporal positions

A local `PackedLayout` normally resets target video and audio positions to the
start of the window. That would tell H3 every window occurs at time zero and
would break global keyframe alignment.

The handler builds a normal H3 `PackedLayout` and offsets only its target rows:

```text
video offset = FRAME_RESCALE * pixel_frame_at(video_window_start)
audio offset = global audio window start
```

Text and Ref2VA reference rows retain their normal layout coordinates.
Existing keyframes retain their original global `resolved_frame_index`, so
their condition rows already land on the global target timeline. All keyframes
and references remain available to every window, matching the conditioning a
single full H3 call would receive.

The video and audio offsets are intentionally separate. Five video frames span
`5 * 40 / 24 = 8.333...` audio positions, so a rounded audio boundary is not
always identical to the fractional H3 video time coordinate.

Layouts are cached only on the context-handler instance for the current
sampler execution. They contain structural CPU metadata, not model activations.

## Memory estimation

The sampler must retain the complete global latent and any solver history, but
these tensors are much smaller than H3's transformer activations. Before model
loading, an execution-scoped `PREPARE_SAMPLING` wrapper replaces the flat
full-latent shape used for VRAM estimation with the largest packed AV window
shape. This keeps ComfyUI's low-VRAM loader aligned with the tensors the H3
transformer actually evaluates.

The wrapper changes estimation only. The real sampler still receives and
updates the full latent.

## Prompt timing and window conditioning

The parser accepts MiniMax's documented markers:

```text
[Shot 1] ...
[Shot 2] At MM:SS.mmm, ...
```

Shot numbering must start at one and timestamps must increase strictly. Both
`integrated_multimodal_description:` and `detailed_description:` are supported.

For each window the parser:

1. converts global timestamps to frames with `fps`;
2. retains shots intersecting the complete window, including its overlap;
3. renumbers the local opening shot to `[Shot 1]` without a timestamp;
4. keeps later cuts at their global timestamps to match global target RoPE;
5. distributes long-shot sentence or clause units across their global range;
6. adds a continuation instruction when a window begins midway through a shot.

For example, a global cut at frame 100 remains `00:04.167` at 24 FPS in a
window starting at frame 50. The old independent-chunk implementation rewrote
that cut to local frame 50 / `00:02.083` because each chunk reset its target
positions to zero; doing so with global target positions would be incorrect.

Every rewritten prompt is encoded before sampling. Window-specific positive
conditions receive an internal integer marker. During each model call the
handler selects only the conditions matching the active window. Other guider
branches remain unmarked and are reused.

Only prompt-owned positive fields are replaced:

```text
cross_attn
minimax_token_tags
```

References, keyframes, controls, hooks, and other conditioning metadata are
preserved. Each duplicated positive condition receives a new UUID so ComfyUI
does not treat distinct window prompts as the same condition.

## The `images` input

`images` supplies pixels only for reconstructing Qwen visual presentation. It
does not create H3 video-VAE or audio-VAE latents.

For I2VA/FL2VA, the first item is the original first frame and an optional
second item is the last frame. These images are presented to Qwen only for the
first window. Later prompts remove zero-time picture anchors, while the
upstream H3 keyframe latents remain available globally.

For image-only Ref2VA, every batch item is a separate reusable picture
reference and is presented to every window in the order described by
`minimax_refs`. Audio reference blocks already own their encoded audio latents
and require no waveform input here.

Existing Ref2VA `video` and `video_audio` blocks cannot be rebuilt from an
IMAGE batch. A Qwen `<Video N>` presentation has timestamps, one ordered media
item, a video-VAE latent, and optional synchronized audio. The node rejects
that mismatch rather than silently presenting video frames as unrelated
`<Picture N>` items.

## Preview behavior

`MiniMax H3 Unlimited Preview` remains a model-patcher wrapper around the stock
outer-sampler callback. The single-trajectory sampler registers one preview
item covering the complete requested timeline. After each sigma step, the
callback receives the full denoised AV estimate and replaces that item.

Latent2RGB is inexpensive. Tiny-VAE preview decodes one selected latent
position at a time and can become slow for long videos, so `frame_stride` is
the intended cost control. WebP compression remains on a bounded background
worker and local websocket send failures remain non-fatal.

`max_resolution` defaults to `0`, meaning the decoder output is not downscaled.
The backend reports the actual encoded width and height plus the selected
previewer name, making a tiny-VAE fallback visible instead of silent. The
browser also displays playback FPS, rolling seconds per step, ETA, a combined
sigma/latent-change graph, and a step-time graph. Latent change is calculated
from a bounded 65,536-value sample of the video latent so the graph does not
retain a second full-length latent in CPU memory.

The context handler reports the active transformer window to the preview as
`Chunk N/total`. It also owns a temporary `tqdm` chunk bar created before the
stock sampler starts, so ComfyUI's ordinary denoising-step bar is placed below
it. The chunk bar resets on window zero for each model evaluation and is closed
in the sampler node's existing `finally` cleanup.

## Compatibility and restoration

Non-H3 and non-nested latents fall through to the stock sampler unchanged.
The public node id, existing inputs, and three outputs are preserved. The
`chunk_frames` and `chunk_prompts` names remain for workflow compatibility,
although they now refer to transformer windows.

The original `guider.original_conds` and `guider.model_options` are restored in
a `finally` block after success, cancellation, or failure. No context handler,
prompt marker, or layout cache leaks into later queue executions.

Multi-window denoise masks are rejected because synchronized packed AV mask
windowing has not been implemented. A single-window request retains stock mask
behavior.

## Known limits

- Full latent and solver memory still grow linearly with duration.
- A complete spatial frame window must fit; there is no spatial tiling.
- Fixed local attention windows approximate, rather than reproduce,
  full-sequence attention.
- Very long global RoPE positions may exceed the model's trained duration.
- Small or occluded details can still drift without explicit prompt/reference
  support.
- Prompt timestamps remain generative guidance rather than deterministic cuts.

## Verification

Lightweight verification uses the Python environment installed with ComfyUI.
The current checks cover:

- Python compilation of `__init__.py`, `nodes.py`, `preview.py`, and
  `windowed.py`;
- ComfyUI custom-node loading and schema registration;
- the H3 duration/window planner over multiple valid durations;
- exact video/audio coverage and eight/nine-position audio overlaps;
- identity reconstruction after window slicing, weighted fusion, and packing;
- selection of the prompt conditioning matching each window;
- local `latent_shapes` replacement;
- global video and audio target offsets in `PackedLayout`;
- per-window packed VRAM-estimation shape;
- prompt timestamp rewriting and strict marker validation;
- existing preview backend and JavaScript syntax checks.

A full H3 GPU render is still required to measure actual peak VRAM, speed, seam
quality, and long-range consistency on the target system.

## References inspected

- MiniMax base prompt guide:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md>
- MiniMax Ref2VA prompt guide:
  <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md>
- Installed ComfyUI implementation:
  `comfy/samplers.py`, `comfy/context_windows.py`,
  `comfy_extras/nodes_custom_sampler.py`, `comfy/model_base.py`, and
  `comfy/ldm/minimax/model.py`.
