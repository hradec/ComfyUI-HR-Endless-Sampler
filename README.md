# ComfyUI MiniMax H3 Sampler Unlimited

`SamplerCustomAdvanced-Unlimited` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long MiniMax H3 video/audio latents.

### Why this exists?
1. by replacing just the Sampler node, Minimax H3 now produces longer than 15secs videos using the same amount of VRAM seamlessly. No complicated loop workflows and video clip concatenations. Just use a normal workflow with the new node and it just works!
2. since we can now do longer videos with the same VRAM, why not SAVE VRAM with smaller chunks and use the left over to increase the resolution? Yep, we can do that now! We can do 2K video inference with only 16GB of VRAM now. No upscale necessary... just render 2K straight. 

### How is this possible?
1. The new Sampler node samples at most `chunk_frames` at a time.
2. After the first chunk, the final five video frames and the matching generated-audio tail are supplied to the next chunk through H3's native guide conditioning.
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
7. Enable `debug` to print the exact rewritten prompt and sampled/output frame
   range for every chunk in the ComfyUI console and return the same text through
   the `chunk_prompts` output.
8. Decode the returned AV latent normally.

Existing H3 first/last-frame guides are assigned to the chunk containing their
frame. Ref2VA references remain attached to every chunk.

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
and should match the sampler's prompt FPS. The five continuation frames are
shown only once because the preview removes the overlap from later chunks.

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
sampler call. The five carried frames are context only and do not consume a
second copy of the prompt action range.

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
- Chunked denoise masks are not supported.

Prompt format references: [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
