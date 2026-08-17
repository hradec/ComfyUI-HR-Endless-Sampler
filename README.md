# ComfyUI MiniMax H3 Sampler Unlimited

`SamplerCustomAdvanced-Unlimited` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long MiniMax H3 video/audio latents.

It samples at most `chunk_frames` at a time. After the first chunk, the final
five video frames and the matching generated-audio tail are supplied to the
next chunk through H3's native guide conditioning. The repeated latent prefix
is removed before the chunks are joined, so both output latents have the same
video and audio shapes requested by the upstream H3 conditioning node.

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
7. Decode the returned AV latent normally.

Existing H3 first/last-frame guides are assigned to the chunk containing their
frame. Ref2VA references remain attached to every chunk.

## Shot timing

The prompt parser follows MiniMax's documented format:

```text
[Shot 1] Opening shot with no timestamp.
[Shot 2] At 00:04.167, the camera cuts to...
```

The first shot is untimed. Every later shot must be sequential and use a
strictly increasing `MM:SS.mmm` cut time. For each chunk, times are converted
to global frames with `fps`, shots outside the chunk are removed, and cuts are
written back as chunk-local timestamps. For example, at 24 fps a global cut at
frame 100 becomes local frame 50 (`00:02.083`) in a chunk starting at frame 50.

The parser recognizes both `integrated_multimodal_description:` and
`detailed_description:`. Other MiniMax prompt sections are preserved.

## Limits

- This reduces temporal sampling memory. It does not make a single frame of an
  arbitrary resolution fit in VRAM; spatial tiling would change H3's global
  attention and is outside this node.
- Video Ref2VA sources cannot be reconstructed from an image input. Image and
  audio Ref2VA presentations are supported; a conditioning containing a video
  reference fails clearly instead of silently dropping its Qwen video tokens.
- When a chunk starts partway through a long shot, the full text of that active
  shot is retained. The previous chunk's latent tail provides its actual visual
  and audio starting state.
- Chunked denoise masks are not supported.

Prompt format references: [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
