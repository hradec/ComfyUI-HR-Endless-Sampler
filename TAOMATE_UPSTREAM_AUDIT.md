# TaoMate-H3 integration audit — 2026-09-19

Compared the installed adapter against upstream `origin/main`, fetched again at
`b933d8e98358241085f421d1a7d3c0944d506b42`. This is a source/CPU-test audit,
not a claim of numerical or perceptual parity on the full GPU model.

## Audio publication mismatch and correction

Upstream `runner._publish` removes each group's transport prefix, concatenates
the clean audio latents, then calls `runtime.decode_audio_latents` once.
`model/output.py:decode_audio` does not normalize each waveform's volume.
It does not crossfade or feather group junctions.

Our sampler instead decoded each group with a short transport halo, applied
ComfyUI's waveform-standard-deviation normalization independently, trimmed each
waveform, then concatenated the waveforms. These operations can introduce decoder
boundary artifacts, different gain between groups, and independent rounding at
cuts. Exact equality to the teacher's *latents* did not test this publication path.
This is a plausible seam source, not proof that it explains every audible issue.

The TaoMate final AUDIO output now decodes the concatenated, already prefix-trimmed
audio latents in one call, with no added waveform normalization. Retained streaming
KV is released before this decode. AudioSR, if enabled, processes this continuous
decode. Finished browser previews receive slices of the same final waveform using
absolute sample boundaries and inclusive frame-range endpoints.

During rendering, per-group audio previews remain provisional (now without local
gain normalization). They are replaced at successful finalization, including a
normal debug stop. We do not repeatedly decode the growing full timeline after
every group. Intermediate per-group MP4 files still contain provisional audio;
final save nodes receiving the sampler AUDIO output receive the continuous decode.
Interrupted renders do not reach final publication.

## Reviewed correspondence and remaining differences

| Area | Upstream | Installed adapter |
|---|---|---|
| Geometry, KV retention, attention routing, sigma selection | Native implementations | Vendored files byte-identical; ComfyUI adapter surrounds them |
| Teacher continuation | Frozen previous 40 audio ticks; nine forwards; states 3/6/9 | Same copied packed-row and denoising logic; frozen-tail invariance tested |
| Teacher execution order | Prepare all groups' audio before video; save states to disk | Prepare each group's teacher immediately before its video; no saved teacher-state files |
| Teacher model | Separate BF16 base teacher | Supplied model **with LoRA**, explicitly requested by user; loaded precision retained |
| Teacher forward | Upstream transformer | Adapter to native ComfyUI projections, timestep embedding, blocks and output head; isolated padding sequence omitted |
| Video denoising | Positive-only Euler loop, separate video/audio clocks | Supplied ComfyUI sampler/options, one complete solver call per phase; teacher audio at model evaluations with interpolation at internal sigma points; final exact teacher audio before clean KV; CFG remains allowed |
| Schedule | Published retained states and configured shifts | Supplied video sigmas/step count now retained; teacher grid includes every supplied endpoint and uses incoming model shifts |
| Published teacher audio | Exact latent equality check | Check retained, permitting only known native carry/un-carry roundoff before replacing with exact teacher values |
| Sub-chunk geometry | 39/34/34/17 frames, then 34/34/34/17; globally rounded audio ticks | Same for full groups; also supports partial final groups |
| Temporal positions | Fixed first media origin, prompt right-alignment, global AV positions | Native PackedLayout adapted to same T2AV placement; also carries native reference/keyframe layouts |
| Clean KV capture | Extra clean forward, cache-only output optimization | Extra clean forward retained; native output heads still execute |
| Video latent renormalization | First sub-chunk's per-feature mean/std | Same formula before clean KV capture |
| Cache retention | First video anchor plus two recent AV commits; audio reset every 12 groups | Same retention/reset policy; CPU cache instead of GPU |
| Attention/hardware | Hopper FlashAttention-3, distributed execution, W8A8 acceleration | PyTorch SDPA, native ComfyUI loading/offload/precision; distributed/W8A8 runtime not ported |
| Noise | Separate upstream video/audio seed sequences and CPU draw layout | Existing sampler noise source sliced per group; teacher/student share group audio noise |
| Text | Pre-encode all supplied prompts | ComfyUI per-group prompt/conditioning pipeline, including manual/local/Gemma providers |
| Conditioning scope | Published streaming T2AV path | References, keyframes, CFG and other patches allowed at user request; no upstream quality-parity claim |
| Video decode | Join latents and publish whole timeline | Per-group native video-VAE decode with transport halo; optional output-only color correction |
| Delivery timing | Whole-output video retiming and audio tempo adjustment to exact nominal five-second groups | Existing native frame timeline and selected fps; exact-duration publication filters not ported |
| Audio publication | Single continuous latent decode | Corrected here; live previews provisional until finalization |

## Source reuse verified

`geometry.py`, `cache.py`, `attention_hook.py`, `denoise_schedule.py`,
`architecture.py`, and `packed_sequence.py` are byte-identical to their pinned
upstream files. `denoise.py` differs only in replacing the distributed
`ParallelContext` import with `Any`. The renormalization formula also matches.
Full model/runtime/publication files were not copied wholesale; their ComfyUI
adaptations are identified above.

## Validation and limits

CPU checks cover native H3 forwards, clean KV, teacher audio injection/carry
scaling, the frozen one-second teacher tail, globally rounded audio geometry,
continuous temporal decoding without gain normalization, and gap-free preview
audio replacement. Full-model listening tests at five-second junctions remain
necessary. A continuous decode cannot repair speech discontinuity already in the
teacher latent itself.

The earlier explanation that the teacher LoRA was the stronger suspect was not
established by measurements. Publication should have been checked first.
Audio-first execution and a base teacher remain separate differences, not proven
seam fixes.
