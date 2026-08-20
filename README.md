# ComfyUI MiniMax H3 Sampler Unlimited

`SamplerCustomAdvanced-Unlimited` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long MiniMax H3 video/audio latents.

### Why this exists?
1. by replacing just the Sampler node, Minimax H3 now produces longer than 15secs videos using the same amount of VRAM seamlessly. No complicated loop workflows and video clip concatenations. Just use a normal workflow with the new node and it just works!
2. since we can now do longer videos with the same VRAM, why not SAVE VRAM with smaller chunks and use the left over to increase the resolution? Yep, we can do that now! We can do 2K video inference with only 16GB of VRAM now. No upscale necessary... just render 2K straight. 

### How is this possible?
1. The new Sampler node samples at most `chunk_frames` at a time.
2. After the first chunk, the final `context_frames` video frames (five by default) and the matching generated-audio tail are supplied to the next chunk through H3's native guide conditioning.
3. The repeated latent prefix is removed before the chunks are joined, so both output latents have the same video and audio shapes requested by the upstream H3 conditioning node.
4. The video/audio decoding stage just decodes the latent as usual, creating a longer than 15secs video and/or higher resolution than possible with low-vram. 

## Use

1. Replace `SamplerCustomAdvanced` with `SamplerCustomAdvanced-Unlimited`.
2. Reconnect the same noise, guider, sampler, sigmas, and latent inputs.
3. Connect the same MiniMax H3 `clip` used by the original conditioning node,
   and paste/connect its original `prompt`.
4. Set `fps` to the prompt timeline's frame rate. MiniMax H3 normally uses 24.
5. For I2VA/FL2VA, connect the original first frame followed by the optional
   last frame as one image batch. For image-only Ref2VA, connect every reference
   image in its original order. T2VA and audio-only Ref2VA need no images.
6. Set `chunk_frames` to the largest chunk that fits in VRAM. `124` is the
   conservative default; the value is snapped down to H3's `17k + 5` grid.
7. Set `context_frames` to the completed tail carried into later chunks. Valid
   values are `5, 22, 39, 56, ...`; the default is `5`, and it must remain
   smaller than `chunk_frames`. Longer context can improve continuity but
   leaves fewer new frames per chunk and therefore increases runtime.
8. Choose the `guide_overlap` mode:

   - `context_frames` uses the configured completed video/audio tail as both
     guide and overlap;
   - `5 frames` uses the same guide + overlap mechanism with H3's minimum
     five-frame tail;
   - `off` carries no previous-chunk guide or overlap. A newly generated local
     five-frame packing prefix is discarded so the combined output remains a
     valid H3 latent; it contains no previous-chunk frames.
9. `video_continuation` is an experimental Ref2VA alternative. It decodes only
   the bounded `context_frames` tail for Qwen, adds that tail as the next
   synchronized `<Audio N>` + `<Video N>` reference for the DiT, and adds the
   documented `[video continuation]` sections to later chunk prompts. Connect
   the H3 video `vae`. The DiT AV reference stays latent; only the video is
   decoded for Qwen and limited to a 512x512 pixel-area budget. ComfyUI's H3
   tokenizer represents audio references as labels rather than waveform input,
   so no audio VAE or temporary audio decode is needed.
10. `qwen_full_history` is an independent experiment. Before each later chunk,
    Qwen sees decoded 2 FPS frames from all completed output so far. It does not
    rewrite the prompt or add that history as a DiT reference. Connect the H3
    `vae`.
11. Gemma 4 automatically directs a later chunk only when its new frames begin
    inside an already generated source shot. It observes the immediately
    previous sampled chunk at 2 FPS, compares it with an immutable action
    ledger made from the original shot, and writes H3-ready prose for only the
    next short continuation. It does not run for Chunk 1 or for a new shot that
    starts exactly at a chunk boundary. Connect the H3 video `vae` whenever a
    long shot will cross a chunk boundary, and install this node's requirements:

    ```bash
    ~/comfyui/tools/python.sh -m pip install -r /NVME/comfyui/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Sampler-Unlimited/requirements.txt
    ```

    On first eligible handoff, it downloads Google's official 6.98 GB GGUF plus
    its 175 MB projector directly to `models/llama_cpp/gemma-4-12b-it-qat-q4_0`.
    Gemma always uses every possible GPU layer while H3 and Qwen are unloaded,
    then its llama.cpp context is destroyed before the next H3 inference. The
    current observation is visual only: it does not decode or judge the
    generated soundtrack.

    The actual editable Gemma system and observation messages are in
    [`gemma4_prompts.txt`](gemma4_prompts.txt). The sampler rereads that file
    before every Gemma handoff, so changing and saving it affects the next
    eligible chunk without a Python-code edit. Keep its `[SYSTEM]` and
    `[OBSERVATION]` section headers and the documented `{{lowercase_placeholders}}`.
    Gemma returns a validated action-progress record plus short H3 continuation
    prose; the actual text it gave H3 is included in `chunk_prompts` when the
    handoff occurs. `prompt_preview_only` intentionally shows the static
    canonical plan because no generated chunk exists for Gemma to observe.
12. Enable `prompt_preview_only` to return every exact planned prompt and frame
   range through `chunk_prompts` without generating noise, rebuilding per-chunk
   conditioning, loading the DiT/VAE for this node, or running inference. Noise,
   sampler, sigmas, and CLIP are lazy in this mode. The two latent outputs are
   unchanged placeholders and should not be used while this toggle is enabled.
13. `chunk_prompts` is populated during normal sampling too. Enable `debug` to
   additionally print those prompts in the ComfyUI console and log VRAM at each
   VAE/Qwen/DiT boundary and immediately before and after every DiT evaluation.
   The report includes physical free memory, PyTorch allocation/cache/peaks,
   known model residency, and visible GPU tensor payloads. A failed evaluation
   emits an abbreviated CUDA allocator summary before re-raising the error.
   Gemma-directed chunks include the accepted action IDs, observation summary,
   confidence, and raw JSON in the same output.
   Every sampling run also ends with a compact timing and memory report: H3
   denoising, Qwen encoding, each VAE decode purpose, and Gemma 4 time, plus
   peak ComfyUI-process/system RAM and GPU VRAM. The VRAM line distinguishes
   sampled whole-GPU use from PyTorch's allocator high-water mark.
14. For a short diagnostic render, set `debug_stop_chunk` to a 1-based chunk
   number. `0` is the normal setting and samples the complete video.
15. Decode the returned AV latent normally.

Existing H3 first/last-frame guides are assigned to the chunk containing their
frame. Ref2VA references remain attached to every chunk.

After each chunk prompt is encoded, the node explicitly unloads Qwen and the
connected H3 video VAE before starting the DiT. ComfyUI's dynamic loader does
not include MiniMax reference rows in its normal sampling-memory estimate and
can otherwise leave both conditioning models resident at a chunk boundary.
This cleanup trades model reload time for maximum sampling headroom.

## Accumulated live preview

Add `MiniMax H3 Unlimited Preview` between the model loader/model patches and
the guider used by `SamplerCustomAdvanced-Unlimited`. Its widget plays the
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
progress. A later chunk starts only after the previous chunk has completed, so
its native H3 continuation guide contains the previous result's finished final
`context_frames` video frames and synchronized audio tail.

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
- A long source shot that crosses a chunk boundary needs the H3 video `vae`.
  Gemma observes the last sampled chunk and writes only the next short H3
  continuation description; it does not ask H3 to infer its location in a
  repeated full-shot prompt.
- Longer `context_frames` uses more of each chunk for repeated guide context,
  increasing the number of sampler calls and total runtime.
- `video_continuation` and `qwen_full_history` require image/audio Ref2VA
  conditioning that this node can reconstruct plus the MiniMax H3 video VAE.
  They are ignored for the first chunk because no generated history exists yet.
- Native `video_continuation` adds only the bounded synchronized video/audio
  tail to DiT attention.
  `qwen_full_history` adds no DiT reference, but its Qwen token and temporary
  VAE-decode cost grows with the completed duration. Dynamic Qwen video frames
  are downscaled by area before encoding; original Ref2VA images are unchanged.
- `guide_overlap: off` is the clean native-continuation experiment: the
  previous result reaches the next chunk only through `video_continuation`
  and/or `qwen_full_history`, according to those independent switches.
- Automatic Gemma continuation needs `llama-cpp-python==0.3.35`, built with the
  CUDA wheel in `requirements.txt`. It loads a separate 7 GB model only between
  relevant chunks, then releases it before H3 sampling. Dependency/download
  failures stop clearly; a malformed observation falls back to the canonical
  source-shot prompt for that chunk. The immutable source ledger is currently
  sentence/semicolon based, so write long sequential shots as clear individual
  sentences. Audio observation and shot-aware physical chunk boundaries remain
  future work.
- Chunked denoise masks are not supported.

Prompt format references: [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
