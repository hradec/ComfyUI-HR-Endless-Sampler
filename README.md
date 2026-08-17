# ComfyUI MiniMax H3 Sampler Unlimited

`SamplerCustomAdvanced-Unlimited` is a drop-in replacement for ComfyUI's
`SamplerCustomAdvanced` for long MiniMax H3 video/audio latents.

The sampler keeps one full-length AV latent and one sigma trajectory. At each
denoising step, only a bounded temporal window is sent through the H3
transformer. Overlapping window predictions are blended into one full-length
prediction before the stock sampler updates the global latent.

```text
one global noisy AV latent
          |
          +-- H3 window 1 --+
          +-- H3 window 2 --+--> blend --> one global sampler update
          +-- H3 window 3 --+
          |
       next sigma step
```

This preserves the sampler's evolving latent and multistep solver history
across the complete video. The full latent and solver buffers still grow with
duration, but the much larger H3 attention activations are limited to
`chunk_frames`.

## Use

1. Replace `SamplerCustomAdvanced` with `SamplerCustomAdvanced-Unlimited`.
2. Reconnect the same noise, guider, sampler, sigmas, and latent inputs.
3. Connect the same MiniMax H3 `clip` used by the original conditioning node,
   and paste or connect its original `prompt`.
4. Set `fps` to the prompt timeline's frame rate. MiniMax H3 normally uses 24.
5. For I2VA/FL2VA, connect the original first frame followed by the optional
   last frame as one image batch. For image-only Ref2VA, connect every reference
   image in its original order. T2VA and audio-only Ref2VA need no images.
6. Set `chunk_frames` to the largest transformer window that fits in VRAM.
   `124` is the conservative default; values are snapped down to H3's
   `17k + 5` frame grid.
7. Enable `debug` to log and return the exact prompt used for every model
   window through `chunk_prompts`.
8. Decode the returned AV latent normally.

Existing H3 first/last-frame guides and Ref2VA references remain available in
every window at their global H3 temporal positions.

## Full-length live preview

Add `MiniMax H3 Unlimited Preview` between the model loader/model patches and
the guider used by `SamplerCustomAdvanced-Unlimited`. After each global sampler
step, its widget replaces the preview with the current denoised estimate of the
complete video.

`tiny_vae: none` uses H3's fast Latent2RGB approximation. Select a compatible
24-channel decoder such as `taeh3.safetensors` from `models/vae_approx` for a
more representative preview. Tiny-VAE decoding processes the whole current
video and can become expensive for long generations; increase `frame_stride`
to reduce the overhead. `fps` controls playback timing and should match the
sampler's prompt FPS. `max_resolution: 0` keeps the preview decoder's native
output size. The widget identifies the active decoder and shows the encoded
resolution, FPS, rolling seconds per step, ETA, sigma/latent-change history,
and step-time history. It also reports the active `Chunk N/total` transformer
window. The console shows a matching chunk progress line above the normal
sampler-step progress line and resets it for each model evaluation.

## Shot timing

The prompt parser follows MiniMax's documented format:

```text
[Shot 1] Opening shot with no timestamp.
[Shot 2] At 00:04.167, the camera cuts to...
```

The first shot is untimed. Every later shot must be sequential and use a
strictly increasing `MM:SS.mmm` cut time. For each window, global times are
converted to frames with `fps`, shots outside the window are removed, local
shots are renumbered from one, and later cuts retain their global timestamps.
The target latent rows also retain global H3 temporal positions, so a cut at
frame 100 remains `00:04.167` at 24 FPS in a window starting at frame 50.

Overlapping windows receive the prompt content relevant to their complete
range. A later window that starts midway through a shot is explicitly told to
continue the established shot. For ordinary I2VA/FL2VA, source pictures and
their zero-time prompt anchors are presented to Qwen only for the first
window. Ref2VA identity, scene, motion, and style references remain available
to every window.

When one shot spans several windows, sentence-level action units are assigned
proportionally across its global interval instead of repeating the complete
action description in every window.

The parser recognizes both `integrated_multimodal_description:` and
`detailed_description:`. Other MiniMax prompt sections are preserved.

## Limits

- Windowing approximates full-sequence attention. Distant frames cannot attend
  directly; information propagates through the shared overlap across denoising
  steps. Important appearance details should still be stated in the prompt or
  supplied as references.
- The complete AV latent and sampler-specific history remain allocated, so
  memory still grows linearly with duration. This is not literally unlimited.
- Temporal windowing does not make an arbitrary spatial resolution fit in
  VRAM. A single `chunk_frames` window must still fit.
- Very long global RoPE positions extend beyond H3's usual training duration
  and may reduce quality even when memory is sufficient.
- Multi-window denoise masks are not supported.
- Video Ref2VA sources cannot be reconstructed from an IMAGE input. Image and
  audio Ref2VA presentations are supported; video-reference conditioning fails
  clearly instead of silently changing Qwen's media presentation.
- Prompt cut timing is generative guidance, not a deterministic editing cut.

Prompt format references: [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
and [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
