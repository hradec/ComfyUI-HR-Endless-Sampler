# TaoMate caching TODO

Design note and work plan for reusing finished TaoMate work across renders.
Nothing here is implemented except the teacher-audio cache, which is finished.
No implementation has been requested yet; the storage policy is still open.

Related sections in `todo.md` describe a *different* problem: moving the live
retained KV to NVMe with a prefetch queue to cut render-time RAM. This document
persists *finished* sub-chunks so a later render can resume from them. Both need
a per-layer blob format, but the retention policies are unrelated.

## Where TaoMate stands today

- **Teacher audio: cached and shipped.** `_TeacherAudioCache` in `nodes.py`
  keeps one finished audio-first pass in
  `$TMPDIR/comfyui-hr-endless-sampler/teacher_audio`, keyed by the source prompt
  hash, the noise seed, the plan/latent geometry, the sigmas, the compute
  precision, the KV codec, the preproduction fingerprint, the model/LoRA digest
  and the TaoMate toggles. A hit restores the per-chunk guidance milestones, the
  per-chunk teacher prompts, the Whisper transcripts and the decoded observation
  waveforms, and decodes the cached clean latent through the normal path.
  The preview cache button governs it; with the button off the stored pass is
  cleared as the render starts and the run records a fresh one. In the observed
  render the pass cost 14:58 of wall clock.
- **Video: no cache at all.** Two gates, plus the UI:
  - `nodes.py:5544`: `replay_cache_enabled = _replay_cache_enabled() and not use_taomate`
  - `nodes.py:5278`: `raise ValueError("TaoMate-H3 currently starts from Chunk 1; persistent KV replay is not yet supported")`
  - `web/unlimited_preview.js` forces `cachedChunkCount = 0` in TaoMate, so
    cached-chunk underlines, the reuse label and the delete-from-chunk action
    are all unavailable.
- **The non-TaoMate replay cache already stores the per-chunk state** that a
  TaoMate resume would also need. `replay_cache.save_chunk(...)` at
  `nodes.py:8044` writes sampled video/audio latents, `previous_frame_count`,
  assembled output/denoised latents, masked audio prefixes, the output and
  denoised templates, the Gemma description, timing plan, end state, retention
  analysis and last-seen character state, the H3 prompt, the per-chunk prefix
  noises, the decoded frames plus a browser proxy. `save_timing_plan` covers the
  Gemma preproduction plan and the manifest tracks `status`/`completed_chunks`.

## Why that checkpoint is not enough for TaoMate

Chunk N does not condition on the previous chunk's latents; it attends over the
persistent KV of everything before it. `TaoMateStreaming.sample`
(`python/taomate.py:940`) runs the supplied solver, renormalizes, then performs
**one clean forward** — `model(result, result.new_zeros([result.shape[0]]), ...)`
at `python/taomate.py:958` — and `cache.commit()` at `python/taomate.py:962`.
That forward is the `clean H3 pass` line in the render profile, and it is what
writes the chunk's rows into the store.

The store itself:

- `TaoMateKVCache._history[layer]` holds one entry per transformer layer, with
  commit bookkeeping in `_commit_token_counts`, `_commit_token_tags` and
  `_block_index` (`python/taomate_upstream/cache.py:84-93`).
- Two codecs, `CompressedKV` (`python/taomate.py:82`, shuffled bytes in bounded
  Blosc or Zstd frames, optional raw-Zstd frames) and `QuantizedKV`
  (`python/taomate.py:175`, `int8` or `turboquant` codes plus fp16 scales).
- `TaoMateKVCache.storage_bytes()` reports retained physical versus raw bytes.
- **No serialization exists for any of it**: no `save`/`load`/`state_dict` on the
  cache classes or the codecs, and no `torch.save` anywhere in
  `python/taomate.py` or `python/taomate_upstream/`.

## Chosen design: one record per sub-chunk, loaded and rebuilt in order

The owner's proposal, validated against the code. Each completed sub-chunk
becomes an independent record on disk; a re-render loads the records it needs,
rebuilds the retained cache in the same order, and resumes real sampling.

- **The capture unit already exists.** In `TaoMateKVCache._commit`
  (`python/taomate.py:362`) the loop does `current = self._staged.pop(layer_name)`
  (`:379`), and `_staged` holds exactly one commit's rows per layer, already in
  its storage codec from `store_pair` (`python/taomate.py:267`). No row math is
  needed to isolate a sub-chunk.
- **History is a per-layer concatenation of commits**, with boundaries recorded
  by `_commit_token_counts` / `_commit_token_tags`
  (`python/taomate_upstream/cache.py:186-204`).
- **Retention selects whole commits**: commit 0 kept as a video-only sink plus
  the last two complete commits (`retain_sink_and_recent_commits`,
  `cache.py:206-219`; row selection in `_retain_commit_rows`, `cache.py:246-285`).
  Retention is therefore a deterministic function of the commit sequence.
- **The quantized path appends without decompressing**:
  `QuantizedKV.concat(previous, current)` at `python/taomate.py:383-384`, so a
  rebuild is CPU `concat`/`index_select` over codes with no GPU forward.
- **Restore is bookkeeping replay**: for each needed record in order, fill
  `_staged`, `_staged_tags` and `_staged_block`, call `commit()`, then call
  `retain_sink_and_recent_commits()` at the same points the live path calls it.
  That rebuilds `_block_index` too, so `begin_clean_commit`'s ordering check
  (`cache.py:123-131`) keeps passing.
- **Bonus over the current design**: records key on their own chunk, so editing
  a late prompt invalidates only the records that actually changed, instead of
  the single global replay fingerprint used today.

## Work items

- [ ] Write the record serializer for `CompressedKV` and `QuantizedKV`, plus the
  uncompressed `none` mode. Keep the payload inside the `torch.load(weights_only=True)`
  allowlist, the way the teacher-audio cache does: concatenate the Zstd/Blosc
  frames into one `uint8` tensor with an offset table for the compressed case,
  and store `key_codes`/`key_scales`/`value_codes`/`value_scales` plus
  `mode`/`shape`/`dtype` for the quantized case. Record `stored_bytes` and
  `raw_bytes` so the manifest can report sizes without decompressing.
- [ ] Add the capture hook at the end of `TaoMateKVCache._commit`, writing
  `{block_index, token_count, token_tags, layers, chunk, phase}` per sub-chunk.
  Capture before the retention call so a record is the full commit, and let the
  rebuild apply the video-only sink filter through the normal retention path
  rather than storing a second copy.
- [ ] Add the restore helper and drive it from the resume path. The remaining
  prefix inputs are already in the replay checkpoint (`sampled_video`,
  `sampled_audio`, templates, Gemma state, prefix noises) plus the teacher
  milestones, which the audio cache now supplies.
- [ ] Key each record by chunk/phase index, plan geometry (video/audio token
  counts), codec mode, the KV contract from `kv_contract_from_model(model)`
  (`cache.py:26`), the model/LoRA digest used by the audio cache, and the
  **chunk's own prompt hash** plus seed. A contract or codec mismatch must be a
  rejection, not a silent load.
- [ ] Wire the resume: remove the `not use_taomate` gate at `nodes.py:5544` and
  the refusal at `nodes.py:5278`, skip the recorded sub-chunks, and keep
  `TaoMateStreaming.begin_chunk` / layout and its per-chunk prompt plumbing in
  step with the skipped prefix.
- [ ] Re-enable the TaoMate cached-chunk UI in `web/unlimited_preview.js`
  (reuse label, cached-chunk underlines, delete-from-chunk) and report the
  video records alongside `audio_cache` in the replay-cache status payload.
- [ ] Validate on load: `history_tokens` must equal the recorded value after the
  rebuild, and the restored retained rows must match a full run — the same style
  of proof as the existing `TaoMate teacher clean audio exact match: True` check
  in `python/taomate.py`.

## Measured numbers, from the 2026-09-26 render

Source: `/NVME/comfyui/ComfyUI/user/comfyui.log`, render started 00:44:08,
11 chunks × 5 sub-chunks = 55 commits, codec `turboquant`, sampled with
`TaoMate KV cache memory by completed sub-chunk` and the per-sub-chunk profile
lines.

- One complete commit: **8.728 GiB stored / 32.858 GiB raw** (`C1.1/11.5`).
- Retained band after eviction: **15.957–23.232 GiB stored**, 60.073–87.463 GiB
  raw, i.e. 73.4% saved, holding the sink plus two recent commits.
- Per-sub-chunk profile, representative sub-chunks:
  - `group 1 sub-chunk 1/5`: total 217.992 s; denoise solver 161.158 s; clean H3
    pass 55.755 s; KV stage/compress 19.136 s; KV commit 0.001 s.
  - `group 2 sub-chunk 2/5`: total 268.350 s; KV restore 16.247 s; denoise solver
    197.168 s; KV stage/compress 30.624 s; clean H3 pass 67.626 s; KV retention
    1.246 s; KV commit 3.399 s.
- Therefore all 55 records cost roughly **350–480 GiB** at `turboquant`, and a
  `zstd lossless` run is larger still because only `int8`/`turboquant` achieve
  the 73.4% saving.

## Disk budget is the binding constraint, and it is currently tight

Measured on this machine at the time of writing:

- `/tmp` is **not** tmpfs; it is on `/dev/sdb2`, 94 GB total with **1.1 GB free
  (99% used)**.
- The NFS `NVME` volume is 3.6 TB with **95 GB free (98% used)**.
- The sampler temp root currently holds 183 MB (`last_run_replay`, the Gemma
  prompt captures and image logs).

Consequences to settle before implementing:

- The teacher-audio cache (tens of MiB) is fine where it is. **KV records must
  not go to `$TMPDIR`** at 1.1 GB free.
- All 55 records (350–480 GiB) is **not possible right now** on either volume.
- The rolling window (~20–40 GiB) fits on the `NVME` volume, so the cache root
  has to become a configurable path with an explicit budget, and the run must
  fail loudly rather than fill the disk when the budget is exceeded.

## Storage policies

- **All records** — any sub-chunk is a resume point, ~350–480 GiB. Blocked by the
  current disk budget.
- **Rolling window** — keep the sink plus the last few commits and prune the
  rest, ~20–40 GiB, but only recent resume points are instant. Recommended first
  step: it caps disk at roughly what the run already holds in RAM and proves the
  capture/restore path end to end.
- **Hybrid** — rolling window for fast recent resumes, and for older points
  rebuild the prefix with the extra clean forward: 55.755–67.626 s clean pass plus
  19.136–30.624 s stage/compress per skipped sub-chunk, replacing the
  161.158–197.168 s of denoise solver it avoids.

## Rejected and known-bad approaches

- [ ] Do not persist one whole retained KV blob per checkpoint. The history is
  already an eviction-rebuilt concatenation, so it cannot be updated
  incrementally, each write is multi-gigabyte, and it gives no resume-point
  flexibility that per-sub-chunk records do not.
- [ ] Do not rebuild the prefix with the clean forward as the primary mechanism.
  It works, but it pays roughly half of re-sampling the sub-chunk (clean pass +
  stage/compress) and buys nothing over records; keep it only as the hybrid
  fallback for lookback beyond the window.
- [ ] Do not store raw BF16 rows. 32.858 GiB per commit, and 60.073–87.463 GiB
  for the retained three, is not viable on this machine.
- [ ] Do not reuse a record across codec modes. `int8`, `turboquant` and
  `zstd lossless` encode different rows, so the codec mode belongs in the key.
- [ ] Do not reuse a record across models or LoRAs. The KV contract comes from
  the model, so a weights change must invalidate rather than load into a
  mismatched contract.

## Audio-side follow-up that belongs with this work

The teacher-audio key includes the source prompt hash, while the replay cache
deliberately excludes it because editing a late prompt is the reason to replay.
So today, editing a late chunk's direction misses the audio cache and re-runs
the full 14:58 teacher pass even though only the later chunks' audio changed.
Making the audio records **chunk-granular** — keyed by the chunk's own prompt,
seed and chunk geometry instead of the global source prompt — would let one late
edit keep both caches valid together, and matches the per-record video key.

- [ ] Split the teacher-audio payload into per-chunk records with their own keys.
- [ ] Keep the global pass record as a fallback, or drop it once the per-chunk
  path is proven.

## Verification plan

- [ ] Unit tests in the style of `tests/test_teacher_audio_cache.py`: serializer
  round trip for `turboquant`, `int8`, `zstd lossless` and `none`; record key
  sensitivity per input; rejection of a mismatched codec, model contract,
  truncated record and wrong chunk count; and that restore order and
  `_block_index` checks hold.
- [ ] Integration: rebuild a recorded prefix and assert `history_tokens` and the
  retained rows equal a full run's; then a real re-render from a later chunk with
  one changed prompt, comparing the untouched prefix against the earlier render.
- [ ] Report resume wall clock versus a full run, bytes written and removed,
  peak RAM, and whether the audio cache hit rather than missed.

## Open questions for the owner

1. Which storage policy: all records, a rolling window, or the hybrid?
2. Should the teacher-audio records become chunk-granular in the same change?
3. Where should the cache root live, given `/tmp` has 1.1 GB free and the `NVME`
   volume has 95 GB, and what is the hard budget the run must respect?
4. What resume granularity should the UI offer: chunk boundaries only, or any
   sub-chunk?
