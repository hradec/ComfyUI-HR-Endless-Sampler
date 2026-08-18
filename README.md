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
   `<Video N>` reference for the DiT, and adds the documented
   `[video continuation]` sections to later chunk prompts. Connect the H3 `vae`.
   The DiT reference stays at latent resolution; only Qwen's decoded copy is
   limited to a 512x512 pixel-area budget to control packed-token memory.
10. `qwen_full_history` is an independent experiment. Before each later chunk,
    Qwen sees decoded 2 FPS frames from all completed output so far. It does not
    rewrite the prompt or add that history as a DiT reference. Connect the H3
    `vae`.
11. Enable `debug` to print the exact rewritten prompt and sampled/output frame
   range for every chunk in the ComfyUI console and return the same text through
   the `chunk_prompts` output. Debug mode also logs VRAM at each VAE/Qwen/DiT
   boundary and immediately before and after every DiT evaluation. The report
   includes physical free memory, PyTorch allocation/cache/peaks, known model
   residency, and visible GPU tensor payloads. A failed DiT evaluation emits an
   additional abbreviated CUDA allocator summary before re-raising the error.
12. For a short diagnostic render, set `debug_stop_chunk` to a 1-based chunk
   number. `0` is the normal setting and samples the complete video.
13. Decode the returned AV latent normally.

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
chunks generated so far as one growing preview. The active chunk is replaced
at each sampler callback; completed earlier chunks remain in the browser and
are not decoded or transferred again.

`tiny_vae: none` uses H3's fast Latent2RGB approximation. Select
`taeh3.safetensors` from `models/vae_approx` for a more representative RGB
preview. Tiny-VAE decoding is slower and uses additional VRAM, so increase
`frame_stride` if preview overhead is too high. `fps` controls playback timing
and should match the sampler's prompt FPS. `max_resolution: 0` preserves the
preview decoder's native resolution. The widget shows the active `Chunk N/N`,
preview resolution, FPS, seconds per step, elapsed time, ETA,
sigma/latent-change history, and step-time history. The configured continuation
frames are shown only once because the preview removes the overlap from later
chunks.

If the browser is refreshed during sampling, the widget reconnects on the next
preview event and continues showing the running job. Hover either graph to draw
a vertical sampling-step marker and inspect any retained preview step from the
active chunk; moving off the graph returns to the live accumulated playback.
New denoising previews replace their cached chunk at the next playback boundary,
so an update cannot restart a chunk or splice it into the preceding chunk.

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
strictly increasing `MM:SS.mmm` cut time. For each chunk, times are converted
to global frames with `fps`, shots outside the chunk are removed, local shots
are renumbered from one, and cuts are written back as chunk-local timestamps.
For example, at 24 fps a global cut at frame 100 becomes local frame 50
(`00:02.083`) in a chunk starting at frame 50.

Later chunks are presented as continuations rather than fresh I2VA/FL2VA
requests: the original picture anchors are removed from their Qwen prompt and
the source images are not presented again. Ref2VA identity/style references
remain available to every chunk. A chunk that starts midway through a shot is
explicitly told to continue from its latent opening frames without replaying
the earlier action.

When one shot is longer than a chunk, its sentences are distributed across the
chunk timelines instead of sending the complete shot description to every
sampler call. A sentence that spans a chunk boundary remains in both prompts so
a continuation never loses the shot's only concrete description. The carried
frames are context only and do not consume a second copy of the prompt action
range.

The parser recognizes both `integrated_multimodal_description:` and
`detailed_description:`. Other MiniMax prompt sections are preserved.

## Limits

- This reduces temporal sampling memory. It does not make a single frame of an
  arbitrary resolution fit in VRAM; spatial tiling would change H3's global
  attention and is outside this node.
- Video Ref2VA sources cannot be reconstructed from an image input. Image and
  audio Ref2VA presentations are supported; a conditioning containing a video
  reference fails clearly instead of silently dropping its Qwen video tokens.
- When a chunk starts partway through a long shot, its description is retained
  with a continuation instruction. Natural-language timing remains generative,
  so very long action-dense shots may still benefit from a larger chunk size.
- Longer `context_frames` uses more of each chunk for repeated guide context,
  increasing the number of sampler calls and total runtime.
- `video_continuation` and `qwen_full_history` require image/audio Ref2VA
  conditioning that this node can reconstruct plus the MiniMax H3 video VAE.
  They are ignored for the first chunk because no generated history exists yet.
- Native `video_continuation` adds only the bounded tail to DiT attention.
  `qwen_full_history` adds no DiT reference, but its Qwen token and temporary
  VAE-decode cost grows with the completed duration. Dynamic Qwen video frames
  are downscaled by area before encoding; original Ref2VA images are unchanged.
- `guide_overlap: off` is the clean native-continuation experiment: the
  previous result reaches the next chunk only through `video_continuation`
  and/or `qwen_full_history`, according to those independent switches.
- Chunked denoise masks are not supported.

Prompt format references: [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
