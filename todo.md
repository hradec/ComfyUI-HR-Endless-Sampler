# TODO

## TaoMate KV-cache memory: compression first, disk prefetch afterward

Deferred experiments; no cache-storage implementation requested yet.

- [ ] First benchmark lossless compression on actual KV layers: compression
  ratio, compression/decompression time, peak RAM, and bit-exact restoration.
  Preserve the current precision and first-video-anchor plus two recent AV
  sub-chunks. Do not substitute lossy FP8/INT8 quantization.
- [ ] Then investigate NVMe-backed per-layer KV storage with a bounded prefetch
  queue: read the next layer while the GPU processes the current layer, release
  consumed buffers, and immediately enqueue later layers. Compare individual
  layers with small layer groups and measure queue depth versus RAM and stalls.
- [ ] Evaluate pinned-memory buffers and asynchronous CUDA transfers, with
  explicit synchronization before consumption or buffer reuse. Include clean
  KV commits, retention updates, cancellation, temporary-file cleanup and disk
  capacity checks without recreating a whole-cache RAM copy.
- [ ] Measure end-to-end render time and disk bandwidth; prefetch can hide I/O
  only when storage keeps up with computation. Include incoming clean KV and
  staging buffers in peak estimates, not just the retained three sub-chunks.

## taomate full audio

Deferred investigation; no inference changes requested yet.

- [ ] Test generating a complete shot's teacher audio from the original prompt
  in one audio-only H3 pass. The current test's Shot 1 lasts 60 seconds.
- [ ] Verify H3's supported audio-only duration and assess consistency beyond
  the approximately 15-second video/audio limit raised by the user. Do not
  assume a 60-second pass works simply because video tokens are absent.
- [ ] Measure dialogue completeness, voice consistency, pronunciation, timing,
  VRAM use, and runtime on long shots.
- [ ] If viable, retain the intermediate teacher audio states and final clean
  latent, then slice them to guide each video sub-chunk.
- [ ] Evaluate Whisper transcription and/or forced alignment of the original
  dialogue to the decoded teacher audio. Use measured word timestamps to build
  five-second video prompts instead of estimating dialogue duration in advance;
  detect omitted, altered, or repeated words.
- [ ] If full-shot generation fails, investigate audio generation in overlapping
  chunks. Explicitly test for the metallic voice degradation previously observed
  when repeatedly continuing from generated audio, as well as audible seams.

Compare against the current five-second teacher baseline with continuous final
audio decoding. This proposal differs from upstream's per-group teacher generation.

## Low-resolution KV reference bank

- [ ] Explore a separate, lossless low-resolution video/audio KV reference bank
  for high-resolution TaoMate generation. Keep it distinct from ordinary past
  timeline history, so target frames do not collide with same-time reference
  K/V positions.
- [ ] Reuse the existing multi-source attention hook to route reference K/V
  separately from text, streaming history, and current target media. Do not
  decode or re-encode reference frames through the VAE.
- [ ] Define and test temporal/spatial position and attention-weight policies.
  Validate subject, camera, and motion retention against raw latent/pixel
  references, and measure whether the reference overpowers target generation.
