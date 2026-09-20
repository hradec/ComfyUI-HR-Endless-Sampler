# TaoMate-H3 upstream source

2026-09-19 audit: upstream main remains at the commit below. Source comparisons
confirmed the documented copies. Final audio now follows upstream's concatenate-
latents-before-decode publication; live group previews remain provisional until
replaced at finalization. See `../../TAOMATE_UPSTREAM_AUDIT.md` for differences.

Schedule follow-up: the adapter now honors incoming video sigmas and step count.
The copied denoising loop is unchanged; teacher schedules combine the upstream
nine-interval grid with exact incoming endpoints and inherit model shifts.
This replaces fixed states 3/6/9 in production and is an intentional departure
from the fixed upstream three-step student. The selected ComfyUI sampler and
options are also retained, with one full solver call per phase. Teacher audio
is supplied at model evaluations, interpolated for internal solver sigma points;
the final clean teacher latent is installed before KV capture. The teacher's
own integration still uses the copied upstream Euler loop.

Source: https://github.com/TaoLiveAIGC/TaoMate-H3
Commit: b933d8e98358241085f421d1a7d3c0944d506b42
Retrieved: 2026-09-18

These files are copied **without modifications**:

- `geometry.py`: `src/taomate_h3/streaming/geometry.py`
- `cache.py`: `src/taomate_h3/streaming/cache.py`
- `attention_hook.py`: `src/taomate_h3/streaming/attention_hook.py`
- `denoise_schedule.py`: `src/taomate_h3/denoise_schedule.py`
- `LICENSE` and `NOTICE`: repository root

The parent `taomate.py` is our ComfyUI adapter. Its
`_renorm_clean_video_rows` method is copied from upstream
`src/taomate_h3/streaming/runtime.py`, with annotations removed and the anchor
attribute renamed. The adapter subclasses the upstream hook to use PyTorch
SDPA instead of Hopper-only FlashAttention-3, and the upstream cache to store
KV on CPU and move the active layer to its execution device.

This is an experimental port with native H3 reference/keyframe conditioning with an audio teacher using the supplied model and LoRA. This is not a claim
of parity with upstream's separate BF16 base teacher, distributed/W8A8 execution,
or throughput benchmarks. ComfyUI's native
VAE decoding uses a two-token transport halo between requests; it is excluded
from inference and trimmed from generated output. Prompts are authored per
five-second request using Endless's existing prompt-provider interface. No persistent-KV
disk replay is supported yet.

SHA-256:

```
56cb44213cd05d0a917cbcf06ae56f1488c6dae7229a035b9c4915d191f027dd attention_hook.py
f57d13d5aac4d52c2a18729d65a426477d73b6f2c1d32d9ba5f174ad8470d7a2 cache.py
987b5fe292636f7126f6ed48f54bf40a9b28045f7d9c55cd060eb9712c1bc583 denoise_schedule.py
c27ea8b26e20a93ba4436938f733d3a5896163e32028843773283b0ec4c023d5 geometry.py
```

Audio teacher source reuse (same pinned commit):
- `architecture.py` and `packed_sequence.py` are copied unchanged from
  `src/taomate_h3/model/`.
- `denoise.py` is copied from that directory; its sole change replaces the
  distributed `ParallelContext` import with `Any`. The ComfyUI adapter supplies
  a single-device context. The branch timestep planner and nine-forward Euler
  loop run as copied, with no rewritten denoising algorithm.
- `../taomate_audio_teacher.py` adapts these packed rows to native ComfyUI H3
  blocks, audio projection and audio head. It omits the isolated padding
  sequence. The rollout and milestone selection follow `teacher.py`.
- Per user instruction the teacher keeps the supplied LoRA active and inherits
  loaded precision. It is prepared per request, rather than as external BF16
  disk artifacts. Student and teacher reuse exactly the same request audio noise.

SHA-256 of the vendored teacher files:
```
41813d8d5e8b7cf56dec951ca63c6ddbf730dadeabb3bbc15064dbc5a946f359 architecture.py
23ef18db9df1f382c1e9ddb0fedc3789a3e49d331568572667082f3d660d8921 packed_sequence.py
2a909322e5472ca6284085826020566cdf44e596169771092768cac5da92a681 denoise.py
```

CPU cache retention follow-up: `../taomate.py` overrides `_retain_commit_rows`
with the copied upstream method, replacing its separate `next_history` dictionary
with the existing history dictionary. Each old layer is released after replacement,
so retention does not duplicate the complete 50-layer cache in system RAM. Row
selection, dtype, sink and recent-history policy remain unchanged. The vendored
`cache.py` is untouched; a CPU test checks exact upstream parity and early release.

CPU commit follow-up: after all 50 clean layers are staged, validate commit
completeness and retain only the first video anchor plus the newest old commit
before calling upstream commit. The incoming commit supplies the second recent.
Skip identity retention selections afterward. This avoids concatenating expired
history, while keeping all history available throughout the clean forward.
Tests compare exact KV and metadata across seven commits and an audio-history
reset, and check that eviction precedes upstream concatenation.

Lossless CPU storage experiment: `TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV`
enables Blosc2 byte-shuffle + Zstd level 1 in the adapter. Both staged and
retained KV use 32 MiB compressed blocks, restored per layer at native dtype.
The compressed commit copies upstream append/metadata bookkeeping, after the
existing early eviction. Runtime logs actual stored/raw bytes; the synthetic
31% saving is not a measured H3-cache guarantee. Vendored files remain unchanged.

`TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV` selects raw Zstd CPU storage
with byte shuffle and nvCOMP CUDA decompression/unshuffle for attention history.
Compression and commit/retention still run on CPU. False retains the previous
Blosc2 CPU decompression path. This is adapter-only; upstream files are unchanged.
