# External dependencies and update checks

- 2026-09-25 (dialogue-cut commit check): rechecked [llama.cpp #27439](https://github.com/ggml-org/llama.cpp/issues/27439) through the GitHub API; it is still `open` with labels `bug-unconfirmed` and `stale`, zero comments, last updated 2026-09-20. PyPI's JSON API still reports `0.3.35` (uploaded 2026-08-17) as the newest `llama-cpp-python`, so no new package exists to inspect for the MTP fix and the recorded 0.3.35 vendoring is unchanged. The disposable MTP worker and the operation-local non-MTP retry are preserved unchanged. The only new runtime package in this commit is `openai-whisper`, used to transcribe generated teacher audio locally; it vendors no llama.cpp.

- 2026-09-25 (Gemma audio input follow-up): [llama.cpp #27439](https://github.com/ggml-org/llama.cpp/issues/27439) is still open, and the [latest upstream llama-cpp-python release](https://github.com/abetlen/llama-cpp-python/releases) remains `v0.3.35-hip-radeon`; preserve the disposable worker and operation-local non-MTP retry. The installed JamePeng Gemma4 MTMD handler accepts `audio_url`, and the local 12B mmproj declares an audio encoder; the current project wrapper previously supplied only images and text. Direct WAV audio input is now wired for decoded teacher-audio observations, pending live-render validation.

- 2026-09-25 (Gemma audio-transcript request check): [llama.cpp #27439](https://github.com/ggml-org/llama.cpp/issues/27439) remains open; [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases) still list `v0.3.35-hip-radeon` as latest, with the previously recorded `v0.3.35` vendored llama.cpp commit `4df29be4f4c3673f428170fda944a5b19f743bb8`. Preserve the operation-local non-MTP retry and disposable worker.

- 2026-09-24 (Gemma prompt path check): [llama.cpp #27439](https://github.com/ggml-org/llama.cpp/issues/27439) remains open; the [latest abetlen llama-cpp-python release](https://github.com/abetlen/llama-cpp-python/releases) remains `v0.3.35-hip-radeon`, with the previously recorded `v0.3.35` vendored llama.cpp commit `4df29be4f4c3673f428170fda944a5b19f743bb8`. No fixed runtime is available. Preserve the disposable-worker, operation-local non-MTP retry.

- 2026-09-20 (TaoMate KV cache codecs): added sampler-selected `none`, `zstd lossless`, `int8`, and `turboquant` cache representations without a new runtime package. `turboquant` uses the 128-wide 4-bit MSE TurboQuant rotation/codebook method described in arXiv:2504.19874 and cross-checked against MIT-licensed `jorgebmann/pyturboquant` at its then-current main branch. It quantizes on GPU before CPU transfer and reconstructs one current H3 attention layer on GPU. The current SDPA path consumes reconstructed K/V, so it deliberately does not store TurboQuant's QJL residual: that residual only helps a fused approximate-inner-product attention implementation. This mode is lossy and experimental; retain `zstd lossless` for exact continuity comparisons.

- 2026-09-19 (TaoMate audio/publication audit): fetched upstream main; it remains
  `b933d8e98358241085f421d1a7d3c0944d506b42`. No vendored update required.
  Six runtime files compare byte-identical; denoise.py retains only its documented
  ParallelContext import substitution. Corrected final audio publication to join
  clean latents before decoding, without per-group waveform normalization.
  See `TAOMATE_UPSTREAM_AUDIT.md` for the remaining differences and validation limits.

- 2026-09-09 (legacy dialogue global-clock correction): llama.cpp issue
  #27439 remains open, last updated 2026-08-20, with no confirmed fix. The
  latest upstream llama-cpp-python release remains `v0.3.35-hip-radeon`
  (published 2026-08-17). This change uses phonemizer/espeak to derive
  word-exact source-clock dialogue ownership: it honors one source
  opening-silence interval, makes later chunks continue rather than restart,
  and supplies the immutable result for Gemma preproduction to verify.
  Preserve the disposable worker and operation-local non-MTP retry.

## AudioSR optional worker runtime (2026-09-07)

- Source: <https://github.com/haoheliu/versatile_audio_super_resolution>.
  The reviewed upstream `audiosr/pipeline.py` exposes `build_model` and
  `super_resolution`; this integration uses the published `audiosr==0.0.7`
  wheel, with `torchlibrosa==0.1.0` and `progressbar==2.5`.
- The main `requirements.txt` declares unpinned NumPy, Transformers, and
  librosa so it retains ComfyUI's selected versions, and installs
  `torchlibrosa==0.1.0` and `progressbar==2.5`. ComfyUI Manager then runs the root `install.py`, which
  installs `audiosr==0.0.7` into the same ComfyUI Python with `--no-deps` and
  verifies the import. Never install AudioSR's pinned NumPy <=1.23.5, librosa
  0.9.2, or Transformers 4.30.2 over ComfyUI. Full backend import and real CPU
  inference passed with torch/torchaudio 2.8.0, NumPy 2.2.6, librosa 0.11.0,
  Transformers 5.13.0, torchvision 0.23.0, scipy 1.16.3, pandas 2.3.2,
  timm 1.0.22, huggingface-hub 1.22.0, and PyTorch Lightning 2.5.5.
- The basic checkpoint comes from <https://huggingface.co/haoheliu/audiosr_basic>
  (`pytorch_model.bin`) through upstream's download helper. The worker enables
  the legacy checkpoint loader needed by this trusted upstream artifact;
  local request/result files use explicit `weights_only=True`. Do not extend
  the legacy loader to user-provided checkpoint files.
- Upstream normalizes input/output and processes mono audio. The adapter runs
  channels separately, restores source amplitude, pads to the 5.12-second grid,
  and uses bounded overlap-add for long audio. Full output is enhanced from
  original VAE decodes, independently of chunk-enhanced preview audio. Never
  substitute lossy cached preview audio for the original decoded waveform.
- Validation: five standalone tests passed and a real two-step CPU inference
  returned a finite, non-silent 48 kHz mono waveform with exact 0.5-second
  duration. GPU integration and subjective artifact reduction are unverified.
  Existing Gemma/llama.cpp runtime and fallback policies are unchanged.

- 2026-09-07 (Video1 first-frame picture reference): issue #27439 remains
  open, last updated 2026-08-20, with no close date. The latest upstream
  llama-cpp-python release remains `v0.3.35-hip-radeon`; tag `v0.3.35`
  vendors llama.cpp `4df29be4f4c3673f428170fda944a5b19f743bb8`.
  JamePeng's latest release remains `v0.3.49-cu131-win-20260831` from the
  previously reviewed 0.3.49 release set. No new fixed runtime was identified.
  This change only supplies an ordinary H3 picture reference and its summary
  contract; preserve the disposable worker and operation-local non-MTP retry.

Read this file before changing the Gemma prompt director or refreshing vendored
documentation. The files below are reviewed runtime source data used to
maintain Gemma's compact prompt summary; they are not contributor or
coding-agent instructions.

## MiniMax H3 prompt-writing skill

The project vendors MiniMax's official H3 prompt-writing skill so a render does
not depend on network availability and upstream edits cannot silently change
generation behavior halfway through a run.

| Vendored file | Mutable upstream source | SHA-256 checked 2026-08-26 |
| --- | --- | --- |
| `vendor/minimax-h3-prompt-writing/SKILL.md` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md> | `a7000443588ca3f145e3b3fd8900f14e0325dc460bd811268fac89a9dc8e56d0` |
| `vendor/minimax-h3-prompt-writing/references/base-en.txt` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt> | `2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc` |
| `vendor/minimax-h3-prompt-writing/references/ref-en.txt` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt> | `1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7` |

`gemma4.py` does **not** inject these complete files into every live Gemma
request. The files remain the reviewed runtime source material for updating
[`minimax_h3_prompt_summary.txt`](minimax_h3_prompt_summary.txt), the compact
working rules reference that Gemma receives at runtime. This avoids spending
most of its 16K context on repeated documentation while preserving the
project's pinned, auditable upstream source.

When reviewing an update, route the documents as follows:

- base T2VA/I2VA/FL2VA/L2VA conditioning receives `base-en.txt`;
- full-reference Ref2VA conditioning receives `ref-en.txt`.

The repository's `gemma4_prompts.txt` supplies the higher-priority chunk-local
contract: Gemma returns only the current chunk's description value, uses the
sampler's immutable real-cut markers, and bases continuation on prior generated
stills. The upstream guides supply MiniMax vocabulary and formatting rules;
their full-video examples must not override the chunk-local contract.

### Update procedure

1. Read the upstream `SKILL.md` and both referenced files completely.
2. Compare all three upstream hashes with the table above.
3. If anything changed, inspect the semantic diff before replacing the
   vendored copy. Pay special attention to shot/cut syntax, dialogue tags,
   reference labels, section names, and supported task modes.
4. Update all three vendored files together, update the hashes and check date
   here, and adapt `gemma4_prompts.txt` or tests when the contract changed.
5. Replay captured Gemma fixtures and run the unit tests before committing.

Read-only hash checks:

```bash
curl -Ls https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/SKILL.md | sha256sum
curl -Ls https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/references/base-en.txt | sha256sum
curl -Ls https://raw.githubusercontent.com/MiniMax-AI/MiniMax-H3/main/skills/h3-prompt-writing/references/ref-en.txt | sha256sum
```

Do not automatically refresh these mutable `main` URLs at render time. A
reviewed repository update is required for reproducible prompt behavior.

## Video Helper Suite finished-video encoder

`HR Endless Sampler Save Video` uses ComfyUI's native `VideoFromComponents`
API for `video/h264-mp4`; H.264 MP4 therefore does not require Video Helper
Suite. Other ordinary `video/*` exports delegate to the locally installed
[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite).
The integration was reviewed on 2026-08-27 against local commit
`4ee72c065db22c9d96c2427954dc69e7b908444b` (`fix(metadata): stop
double-stringifying prompt in MP4 metadata`). It uses only these public module
members from `videohelpersuite.nodes`:

- `get_video_formats()` to populate the Save node's current `video/*` list and
  discover its `pix_fmt` widget choices;
- `VideoCombine().combine_video(...)` to preserve VHS's own FFmpeg format JSON,
  CRF/pixel-format behavior, output naming, metadata path, and optional
  standard ComfyUI `AUDIO` mux path.

The finished timeline is passed as `extra_pnginfo["hr_endless_sampler_timeline"]`.
VHS serializes that value into its FFMETADATA input for formats with
`save_metadata`; the Endless node always writes its own adjacent JSON sidecar
as a reliable fallback. `meta_batch` remains deliberately unused because it
is VHS execution-control state, not a metadata transport.

Before changing this integration or updating VHS, inspect the signatures and
return layout above. In particular, verify that `combine_video` still accepts
direct format widget values (`pix_fmt`, `crf`, `audio`, `extra_pnginfo`) and returns its
file list as `result["result"][0][1]`. Re-run the small real CPU encode/
embedded-metadata smoke test and the project unit suite. `video/exr` is
independent of VHS and uses the PyAV/FFmpeg EXR encoder already supplied by
ComfyUI.

ComfyUI loads VHS beneath a path-derived custom-node package and does not add
the VHS repository directory to `sys.path`. Do not assume that
`import videohelpersuite.nodes` works. Runtime discovery first accepts that
standalone import for compatible installations, then reuses the already-loaded
module whose name ends in `.videohelpersuite.nodes`. Never load a second copy
of VHS from its file path because that duplicates module state and route/node
registration.

## Gemma 4 MTMD and MTP runtime

The local director pins JamePeng's fork of `llama-cpp-python==0.3.49` and
Google's official `google/gemma-4-12B-it-qat-q4_0-gguf` model/projector pair.
Runtime integration uses the `cu124` binary-wheel channel: this is the lowest
CUDA channel published by that fork for both Linux and Windows, and remains
driver-compatible with the existing CUDA 12.5 system. It was initially
reviewed on 2026-08-27 against:

- <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf>, especially
  image-before-text modality ordering and the supported 70, 140, 280, 560, and
  1120 visual-token budgets;
- <https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.h>, which
  defines the MTMD image-token budget and media batching fields;
- <https://github.com/ggml-org/llama.cpp/issues/21550>, which records evaluation
  failures encountered with high Gemma 4 image budgets;
- <https://github.com/abetlen/llama-cpp-python/blob/3691546f1c9e0c1bf93323dff02230bd959cf562/examples/server/server.py>,
  whose `draft-mtp` implementation is the reference for the local
  single-sequence Gemma MTP adapter;
- <https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/speculative-simple.cpp>,
  which is the current reference for checkpointing a hybrid target before
  speculative verification, restoring `PARTIAL_ONLY` state after partial
  acceptance, and replaying only the accepted prefix;
- <https://github.com/ggml-org/llama.cpp/pull/24108>, which removed
  `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` from speculative checkpoints because
  on-device state was not fully compatible with meta/device buffers and its
  memory was not accounted at startup;
- <https://github.com/ggml-org/llama.cpp/issues/27439>, which tracks the
  remaining public-C-API failure mode where an invalid on-device state can
  throw or abort instead of returning a recoverable error;
- <https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp>,
  which supplies the linked-context Gemma 4 MTP implementation and explicitly
  creates the MTP draft context with `n_rs_seq=0`;
- <https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF>,
  the Q8_0 GGUF conversion of Google's matching official QAT assistant/drafter
  checkpoint that is automatically downloaded beside the target model.

The pinned Python handler binds the MTMD fields but does not expose them in its
constructor. `gemma4.py` therefore owns a narrow subclass that sets the dynamic
70-1120 budget and keeps MTMD, logical, and physical batch capacities at least
1120. It also sends chronological images before the observation text. Before
updating `llama-cpp-python`, verify that the high-level handler's constructor,
`_init_mtmd_context`, cleanup callback, MTMD structure layout, and non-causal
image batching behavior remain compatible. Prefer an upstream public budget
API when one becomes available, then remove the local override and rerun the
real model plus unit tests.

Runtime MTP now uses JamePeng's high-level native `SpecConfig` with
`SpeculativeType.DRAFT_MTP`, the official Q8 MTP draft model, and
`draft_n_max=4`. This replaces runtime use of the former local low-level
`gemma4_mtp.py` adapter; retain that file only while its historical tests and
captures still need it. The fast path stays inside a disposable worker because
native MTP can still abort before its C API returns an error. The parent must
preserve the exact request and retry that operation up to ten times in fresh
non-MTP workers after a native worker exit. The retries change only the copied
request's MTP flag: all cache fields remain intact, and the next independent
operation attempts MTP again.

Every normal and append-only Gemma completion uses JamePeng's native Gemma
first-reasoning-block controls: `reasoning_budget=4096`, start
`<|think|>`, end `<channel|>`, and the explicit budget message that ends the
thought block and requests the visible answer. `reasoning_start_in_prompt` is
true because `Gemma4ChatHandler(enable_thinking=True)` inserts `<|think|>` into
the rendered assistant prefix before generation. This is not a generic
OpenAI-compatible setting: retain the exact Gemma tags when revising the
model/template.

### Periodic Gemma MTP upstream check

Issue <https://github.com/ggml-org/llama.cpp/issues/27439> must be checked
periodically, specifically during every Gemma/MTP development session and
before each project release. Also check the latest `llama-cpp-python` release,
its bundled llama.cpp commit, and its CUDA wheel availability. The purpose is
to detect when the native on-device state API has become safe enough to remove
the process-level non-MTP retry.

Do not infer that a newer Python package contains the fix from its version or
release date alone. Confirm that issue #27439 is resolved by an upstream code
change, confirm that the package vendors that change, and replay the saved
multimodal Chunk 2 failure capture with four-token MTP enabled. Record each
review date, versions/commits checked, and outcome below.

- 2026-08-28: issue #27439 remains open. `llama-cpp-python==0.3.35` still uses
  the runtime in which the captured on-device checkpoint abort was reproduced.
  Host-only checkpoints reduced output to roughly 56 tokens/second and a later
  Chunk 2 worker still aborted during MTP initialization. Keep fast on-device
  MTP isolated in the child worker and retain the explicit non-MTP retry.
- 2026-08-28 (latest Chunk 2 crash recheck): issue #27439 remains open with no
  linked fix or pull request. GitHub still identifies `v0.3.35-hip-radeon` as
  the latest `llama-cpp-python` release, built from package commit `3691546`;
  no newer package containing an upstream state-restore fix is available. The
  exact captured multimodal request showed an MTP load failure followed by a
  CUDA abort in the first non-MTP worker, so worker-exit recovery now preserves
  the request and permits up to ten fresh operation-local non-MTP retries.
- 2026-08-29: issue #27439 remains open and has no recorded fix or close date.
  GitHub still reports `v0.3.35-hip-radeon` (published 2026-08-17) as the
  latest `llama-cpp-python` release. The 0.3.35 changelog still pins llama.cpp
  `4df29be4f`/`adb55e514`, predating the affected issue's reproduction commit;
  no package containing a confirmed fix is available. Preserve the disposable
  worker and ten operation-local non-MTP retries unchanged.
- 2026-08-29 (render-resume/response-repair session): rechecked issue #27439;
  it remains open, labeled `bug-unconfirmed`, with no linked fix or pull
  request. GitHub's latest-release endpoint still returns
  `v0.3.35-hip-radeon`, and the current changelog still lists llama.cpp
  `4df29be4f`/`adb55e514` for 0.3.35. No newer Python package contains a
  confirmed fix. The disposable MTP worker and ten operation-local non-MTP
  retries are therefore preserved unchanged.
- 2026-08-29 (32K KV-cache alignment session): issue #27439 remains open,
  labeled `bug-unconfirmed`, with no linked fix or release that contains one.
  `llama-cpp-python` remains at 0.3.35; its published changelog still vendors
  llama.cpp `4df29be4f`/`adb55e514`. The local `Llama` constructor exposes
  `n_ctx`, `type_k`, `type_v`, and `swa_full`, so the sampler now explicitly
  matches the tested native Gemma server's memory policy: 32,768 context,
  Q8_0 K/V cache (`GGML_TYPE_Q8_0`), and no forced full-size SWA cache. The
  prior 20,480 test was F16/default-cache configuration and is not evidence
  against the 32K Q8_0 configuration. Preserve the disposable MTP worker and
  ten operation-local non-MTP retries; this change does not alter them.
- 2026-08-29 (MTP reliability regression investigation): issue #27439 remains
  open and `llama-cpp-python` remains at 0.3.35. The render log establishes a
  local configuration-dependent regression rather than a Gemma JSON failure:
  early native-MTP runs at the original 16,384-token/default-KV configuration
  completed hundreds of on-device checkpoints per response with roughly
  62-75% draft acceptance, while recent 32,768-token/Q8_0 runs repeatedly abort
  after two output tokens inside `llama_state_seq_get_data_ext()` with either
  `not enough space in the buffer` or an invalid backend-buffer assertion.
  `gemma4_mtp.py` itself did not change between those runs, but it still opts
  into `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. Upstream PR #24108 removed that flag
  from speculative checkpoints because its extra device allocation is not
  accounted at context startup and it is not fully compatible with meta/device
  buffers; current `speculative-simple` uses `PARTIAL_ONLY` without
  `ON_DEVICE`. Therefore the disposable-worker retry remains necessary but is
  containment, not proof that the fast checkpoint path is correct. Before any
  further performance tuning, replay the captured multimodal Chunk 2 request
  against an explicit matrix of 16K/default-KV, 32K/Q8_0, and host-only
  checkpoints after ComfyUI releases the GPU. Do not characterize the present
  all-operation failure rate as only upstream randomness.
- 2026-08-29 (reference-checkpoint port): the local native MTP adapter now
  follows current llama.cpp checkpoint flags by saving and restoring only
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` host state; it no longer requests
  `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`. The adapter retains and overwrites one
  growable host checkpoint buffer across proposals, while keeping the existing
  accepted-prefix replay and disposable-worker/non-MTP retry unchanged. All
  101 non-live repository tests pass. This is not yet a successful captured
  multimodal replay: the running ComfyUI process held 14,954 MiB of the GPU, so
  a separate Gemma worker could not load for the live test. Keep the fallback
  until the next real Chunk 2 operation confirms the host-checkpoint path.
- 2026-08-29 (global-shot-label prompt session): issue #27439 remains open,
  labeled `bug-unconfirmed`, with no assignee, linked fix, or pull request.
  PyPI and the repository tags still identify `llama-cpp-python==0.3.35` as
  current; tag `v0.3.35` is package commit `3691546f1c9e0c1bf93323dff02230bd959cf562`
  and its `vendor/llama.cpp` submodule is `4df29be4f4c3673f428170fda944a5b19f743bb8`.
  No package with a confirmed #27439 fix exists. Preserve the disposable
  worker and operation-local non-MTP retry unchanged.
- 2026-08-30 (Windows worker-encoding fix): issue #27439 remains open,
  labeled `bug-unconfirmed`, with no linked fix or pull request. PyPI and the
  repository tags still report `llama-cpp-python==0.3.35`, which vendors
  llama.cpp `4df29be4f4c3673f428170fda944a5b19f743bb8`; no confirmed fix is
  available. Preserve the disposable worker and operation-local non-MTP retry.
- 2026-08-30 (character-continuity planning session): issue #27439 remains
  open (last updated 2026-08-20) with no close reason or confirmed fix. The
  GitHub latest-release endpoint still reports `v0.3.35-hip-radeon` (published
  2026-08-17), so there is no newer `llama-cpp-python` package known to vendor
  a fix beyond the already recorded 0.3.35 llama.cpp commit. Preserve the
  disposable worker, request/cache fields, and operation-local non-MTP retry
  unchanged while revising only Gemma's prompt/response continuity contract.
- 2026-08-30 (preproduction output-truncation diagnosis): issue #27439 remains
  open, labeled `bug-unconfirmed`, with no linked fix or pull request. GitHub's
  latest-release endpoint still reports `v0.3.35-hip-radeon` at package commit
  `3691546`; no newer `llama-cpp-python` release containing a confirmed fix is
  available. The current render failure is independent of that upstream bug:
  MTP, the operation-local original-decoder retry, both append-only repairs,
  and the final grammar fallback each reached the local 4,096-token timing-plan
  response ceiling without closing the expanded continuity-plan JSON. Preserve
  the disposable worker and operation-local non-MTP retry unchanged.

The runtime was compared against `llama-cpp-python` tag `0.3.35` at commit
`3691546f1c9e0c1bf93323dff02230bd959cf562`; that package vendors llama.cpp at
`4df29be409b3c26e33b5d95e29415b21cba9d6a1`. Native llama diagnostics must stay
disabled even when the sampler's own `debug` option is enabled. The sampler's
debug output is intentionally limited to its progress, prompts, timing, and
memory records; enabling llama.cpp `verbose` output produces thousands of CUDA
graph/state lines and obscures the useful measurements.

Do not apply llama.cpp's JSON grammar to normal Gemma director responses.
Exact-capture profiling on 2026-08-27 showed that strict JSON grammar made the
CPU target sampler spend 45.060 seconds on 1,628 token samples while target
verification itself took only 1.015 seconds. Gemma already receives a strict
JSON output contract, and its unconstrained response is parsed and validated
after generation. The grammar is therefore a final recovery-only mechanism:
an ordinary non-MTP malformed response first gets two compact append-only
model-authored repair turns, and only then may use grammar. An MTP response
with no complete JSON must leave the disposable worker immediately and retry
the exact operation in a fresh original-decoder worker; it must not spend
several full generations in grammar recovery. A valid response must take the
fast unconstrained path; semantic validation and chat-style correction still
run afterward. The same captured chunk reduced
target sampling to 0.285 seconds and streamed at 97.2 tokens/second on the
initial response and 125.0 tokens/second on its correction. Preserve unit
coverage for both the ordinary fast path and malformed-JSON recovery when
updating llama-cpp-python or the director response contract.
Do not restore the former `n_rs_seq=4` requirement: the installed Gemma 4
runtime reports that the model does not support recurrent partial rollback and
clamps it to zero. The `gemma4_mtp` sampler toggle is the
explicit fallback: when it is false, use the original high-level runtime; when
it is true, missing native symbols or invalid target setup must stop the Gemma
pass instead of silently running and reporting the non-MTP decoder.

- 2026-08-31 (retained-slice dialogue segmentation session): llama.cpp issue
  #27439 remains open with no state reason or upstream fix and was last updated
  2026-08-20. The newest published llama-cpp-python GitHub release remains
  `v0.3.35-hip-radeon`, published 2026-08-17; no newer package with a confirmed
  fix is available. This session changes only Gemma's preproduction dialogue
  schedule and chunk contract. Preserve the disposable worker and the
  operation-local non-MTP retry without modification.

- 2026-09-01 (overlap-keyframe/dialogue-timecode session): llama.cpp issue
  #27439 remains open, labeled `bug-unconfirmed`, with no linked fix or pull
  request. GitHub's latest `llama-cpp-python` release endpoint still resolves
  to `v0.3.35-hip-radeon` at package commit `3691546`; the already recorded
  0.3.35 package vendors llama.cpp `4df29be4f4c3673f428170fda944a5b19f743bb8`.
  No newer Python package with a confirmed #27439 fix is available. Preserve
  the disposable worker, all request/cache fields, and the operation-local
  non-MTP retry unchanged while revising only Gemma's dialogue timing contract.

- 2026-09-03 (minimal chunk-prompt wording session): issue #27439 remains
  open and labeled `bug-unconfirmed`; its public report still identifies the
  unsafe `ON_DEVICE` restore path and contains no linked fix. GitHub's latest
  llama-cpp-python release endpoint did not report a newer package than the
  recorded 0.3.35 release. This session changes only Gemma's H3-facing wording
  contract: no llama.cpp, MTP, worker, cache, or retry behavior changed.
  Preserve the disposable worker and operation-local non-MTP retry.

- 2026-09-03 (JamePeng native MTP/reasoning-budget migration): issue #27439
  remains open and labeled `bug-unconfirmed`; no linked upstream fix or close
  state exists. JamePeng release `v0.3.49-cu124-linux-20260831`, tag commit
  `34c1bfbce3ad485d31e67039fa9200e6ab49882e`, provides Linux and Windows
  CUDA 12.4 wheels and exposes the required high-level `SpecConfig` external
  `DRAFT_MTP` API plus `reasoning_budget`, `reasoning_start`,
  `reasoning_end`, and `reasoning_budget_message` completion parameters.
  The local CUDA 12.5 system successfully imports its Python 3.11 Linux wheel
  and those APIs. The fast native MTP route therefore uses that public API with
  four draft tokens. Preserve the disposable worker and operation-local
  non-MTP retry until this package passes the captured multimodal Chunk 2
  replay; this API availability does not prove the #27439 abort is fixed.

The 2026-08-29 live 625-frame integration replay confirmed the distinction.
MTP produced no complete JSON after one 1,024-token response (622 of 1,608
draft tokens accepted, 38.7%); the new typed MTP-output failure immediately
retried that same multimodal Chunk 10 operation through the original decoder,
which returned usable JSON and completed the test. The former 20–30 token/s
reports came from repeatedly generating maximum-length unusable responses and
then entering the expensive grammar sampler, not from the clean non-MTP
decoder. Preserve the typed early handoff until upstream MTP is reliable.

- 2026-09-04 (JamePeng MTP configuration/performance audit): llama.cpp issue
  #27439 remains open and has no confirmed fix. JamePeng's newest published
  release remains `v0.3.49-cu124-linux-20260831` at package commit
  `34c1bfbce3ad485d31e67039fa9200e6ab49882e`; it vendors llama.cpp commit
  `9723942adc518b43c4b95dc4dce6906903eb5e09`. The installed wheel is that exact
  Python 3.11 CUDA 12.4 release. Fork `main` is currently `c31ed303`; changes
  since the release contain no Gemma MTP checkpoint/rollback fix. Its
  documented external Gemma 4 MTP setup uses
  `SpecConfig(DRAFT_MTP, draft_model_path=..., draft_n_gpu_layers="all",
  draft_backend_sampling=True)` and warns that stateful MTP is text-only and
  that public prompt-cache restoration does not restore speculative-engine
  state. In this fork `n_gpu_layers=-1` means `auto`, while `"all"` maps to the
  native all-layers value. Real local profiling found 64.6 token/s without MTP
  but only 2.3 token/s through the high-level MTP path: Gemma 4 exposes no
  target recurrent snapshot slots, so the Python verifier deep-copies roughly
  170 MiB host checkpoints during rejection rollback. A disposable one-buffer
  prototype removed that catastrophic copy cost but reached only 44.8 token/s,
  still below the original decoder. Do not describe the Python `SpecConfig`
  route as equivalent in performance to native llama.cpp server MTP. An exact
  text-only test of the fork's documented all-GPU setup drafted at normal GPU
  speed but failed on its first partial rejection with `Failed to restore the
  exact hybrid checkpoint for speculative rejection`; the same failure occurs
  without the MTMD handler, so it is not caused by this project's image prompt
  path. Preserve the currently usable target-loading policy, disposable worker,
  and operation-local non-MTP retry until the fork fixes this rollback path.
  A subsequent identical 100-token long-prompt comparison with
  `draft_n_max=2` completed at 1.98-1.99 token/s (66.3% draft-token
  acceptance), versus roughly 2.3 token/s at `draft_n_max=4`; changing draft
  GPU placement between `auto` and `all` did not materially affect that result.
  Keep four draft tokens.

- 2026-09-04 (preproduction request-structure review): llama.cpp issue #27439
  remains open, labeled `bug-unconfirmed`, with no linked branch or pull
  request. JamePeng's latest GitHub release endpoint reports
  `v0.3.49-cu131-win-20260831`; the applicable CUDA 12.4 Linux release remains
  the already reviewed `v0.3.49-cu124-linux-20260831` from the same 0.3.49
  release set. Upstream `abetlen/llama-cpp-python` still reports
  `v0.3.35-hip-radeon` as its latest release. No newly published package is
  confirmed to fix #27439. Preserve the disposable worker, request/cache
  fields, and operation-local non-MTP retry unchanged.

- 2026-09-04 (SGLang feasibility review): llama.cpp issue #27439 remains open,
  labeled `bug-unconfirmed`, with no linked fix or pull request. The latest
  upstream `abetlen/llama-cpp-python` release is still
  `v0.3.35-hip-radeon`; JamePeng's latest release is still the 0.3.49
  `v0.3.49-cu131-win-20260831` artifact set, while this project uses its
  matching CUDA 12.4 Linux build. No newly published llama-cpp-python package
  contains a confirmed fix, so preserve the disposable worker and exact
  operation-local non-MTP retry.

  SGLang `main` now has native Gemma 4 multimodal inference and Frozen-KV MTP,
  including the multimodal `Gemma4ForConditionalGeneration` target. That path
  consumes Hugging Face-format target and assistant checkpoints; it is not a
  drop-in runtime for this project's Q4_0 GGUF target, separate MTMD projector,
  and Q8_0 GGUF assistant. Its current Gemma 4 12B cookbook targets H200/B200
  class GPUs, and the documented QAT `q4_0-unquantized` checkpoint keeps BF16
  weights rather than providing the approximately 6.6 GiB GGUF footprint used
  here. On this 16 GiB RTX 4070 Ti SUPER, no reviewed SGLang configuration is
  presently equivalent to the existing 32K multimodal worker plus MTP while
  leaving enough memory for reliable startup and inference. SGLang also has no
  reviewed public cross-process KV-state export/import equivalent to the
  current preproduction snapshot. Do not replace `gemma4.py` with an SGLang
  backend until a separately isolated environment passes target-only and MTP
  multimodal capture replays, clean worker teardown, 32K-context memory checks,
  append-only correction turns, and the captured Chunk 2 test on this GPU.

- 2026-09-04 (alternative Gemma 4 multimodal/MTP runtime review): llama.cpp
  issue #27439 remains open, labeled `bug-unconfirmed`, with no linked fix or
  pull request. The newest upstream `abetlen/llama-cpp-python` release remains
  `v0.3.35-hip-radeon`; JamePeng's newest release remains the 0.3.49
  `v0.3.49-cu131-win-20260831` artifact set, while this project uses the
  matching CUDA 12.4 Linux build. Preserve the disposable worker and exact
  operation-local non-MTP retry.

  Three exact 12B alternatives were verified. Hugging Face Transformers has an
  official `google/gemma-4-12B-it` plus
  `google/gemma-4-12B-it-assistant` multimodal assisted-generation path and is
  the smallest independent implementation to prototype. Current vLLM `main`
  supports the 12B unified multimodal target and matching MTP assistant, but
  open issue #48503 reports a CUDA-graph capture crash; `--enforce-eager` is a
  workaround which also removes the principal fast path. Native current
  `llama-server` supports MTMD multimodal requests plus external GGUF MTP and
  can reuse this project's target, projector, and assistant artifacts, but it
  remains subject to llama.cpp's open speculative/state bugs and therefore
  must be isolated and replay-tested rather than assumed reliable.

  LiteRT-LM exposes Gemma 4 MTP, but its published 12B `.litertlm` package does
  not contain the required drafter payload (open issue #2498). TensorRT-LLM
  supports multimodal MTP for E2B, E4B, 26B-A4B, and 31B, but explicitly does
  not support the 12B unified target/assistant architecture. Neither is an
  exact replacement for the current 12B workflow. Any prototype must first
  pass target-only and MTP multimodal capture replays, full worker teardown,
  32K-context memory checks, correction turns, and captured Chunk 2 before it
  can replace the existing path or its fallback.
- 2026-09-23 (chunk-prompt allocation revert session): rechecked issue #27439
  through the GitHub API; it remains `open`, labeled `bug-unconfirmed` and
  `stale`, with zero comments and last updated 2026-09-20. PyPI's JSON API still
  reports `0.3.35` (uploaded 2026-08-17) as the newest `llama-cpp-python`
  release, so no new package exists to inspect for a fix and the previously
  recorded 0.3.35 vendoring (llama.cpp `4df29be4f`/`adb55e514`) is unchanged.
  Nothing to update; the disposable MTP worker and the operation-local non-MTP
  retry are preserved unchanged.

## TaoMate-H3 streaming core (2026-09-18)

Vendored from https://github.com/TaoLiveAIGC/TaoMate-H3 at
`b933d8e98358241085f421d1a7d3c0944d506b42`, the checked-out upstream main
revision for this integration. `python/taomate_upstream/PROVENANCE.md` records
source paths, SHA-256 hashes and adaptation boundaries. Geometry, clean-KV
retention, attention routing and the three-step schedule are copied unchanged;
LICENSE and NOTICE accompany them. No new pip dependency is required.
The adapter uses native ComfyUI H3 projections/RoPE with PyTorch SDPA, CPU KV
storage, and joint audio generation. Upstream's Base10 teacher and distributed
runtime are not ported. This backend is experimental and T2AV-only. CPU tests
exercise native H3 forwards and 50-layer clean commits; full model GPU quality
and throughput still need validation.

- TaoMate adapter follow-up: removed the policy gates rejecting native image,
  video and audio references/keyframes, nonempty latents, CFG values other than
  1, and other diffusion wrappers. Native PackedLayout now carries condition
  rows; only generated AV enters persistent KV. CFG branches retain separate
  histories. Keyframe timing excludes the decoding-only halo. Native spatial
  padding is preserved. Upstream vendored files are unchanged. CPU checks
  include real conditioned H3 forwards and independent positive/negative
  commits; full-model GPU/reference quality remains unverified. The native
  batch/shape contract, existing chunk-mask limitation, and lack of KV disk
  resume still apply.

- TaoMate prompt/request correction: rechecked upstream main at the same
  b933d8e commit. Its config requires one prompt per nominal five-second
  request, and runtime reuses that conditioning across four internal phases.
  The adapter now groups those phases into one outer sampler/prompt chunk,
  preserving request-level text/reference positions while target AV advances.
  Manual prompt generation and dialogue timing share this request plan. Output
  decoding occurs after assembling the request, with its transport halo.

- Prompt-allocation revert (2026-09-23): the one-sentence-per-chunk dialogue
  allocator that used `sentence_word_owners` in the now-deleted
  `python/dialogue_timing.py` is gone. `_legacy_dialogue_for_range` is back to
  splitting the utterance proportionally across the chunk clock, so a seam may
  fall inside a sentence again, for every continuation method including TaoMate;
  the `sentence_chunks`/`taomate=` plumbing was removed rather than defaulted
  off, so no per-method fork remains. This is planner-local: no vendored
  TaoMate file, request-plan geometry, or upstream revision changes, and no
  upstream update is needed.

- TaoMate audio-teacher integration: copied upstream model/architecture.py and
  model/packed_sequence.py unchanged, and model/denoise.py with only its
  distributed-context import replaced by a typing alias. Provenance and hashes
  are recorded in python/taomate_upstream/PROVENANCE.md. The native ComfyUI
  adapter executes the copied nine-forward Base10 loop without video tokens,
  projections, or heads. It captures states 3/6/9, retains a frozen 40-tick
  previous clean tail, injects per-phase audio after student Euler updates,
  commits the guided clean AV KV, and checks exact published clean audio.
  Native carried-audio scaling is accounted for, with only known carry/un-carry
  roundoff removed before exact assembly verification. Audio KV drops every
  12 requests as upstream specifies.
  User explicitly requested the SAME model WITH its LoRA for this experiment:
  do not remove the LoRA, require a separate BF16 model, or describe the teacher
  as equivalent to upstream's BF16 base-model artifact. Loaded precision is
  retained (including the sampler's explicit fp32 override). No new package
  dependency or model download is needed. Teacher work adds nine audio-only
  forwards per request; audio is then used by every video phase. Preparation
  uses the exact same audio noise as the student rather than upstream's
  separate artifact-producer seed sequence.
  Validation: 234 tests passed, one skipped, including native H3 teacher
  forwards with CPU SDPA, frozen-tail invariance, milestone audio carry,
  exact output assembly and mismatch rejection. Full-model GPU rendering and
  LoRA-teacher speech quality are not yet verified.

## Lossless TaoMate KV storage experiment

Uses installed `blosc2==3.12.1` (declared `blosc2>=3.12.1` in requirements),
byte-shuffle with native tensor element size and Zstd level 1. No upstream
TaoMate update is needed for this adapter-local CPU storage change. Decode
restores the exact tensor bytes; there is no BF16/FP32 conversion or truncation.

TaoMate duration clarification (issue #4, contributor reply 2026-09-20 UTC):
https://github.com/TaoLiveAIGC/TaoMate-H3/issues/4#issuecomment-5747481599
The five-second prompt-management interval is a configurable demo choice;
model duration remains within the original model's supported range. Adapter
request grouping now follows chunk_frames, retaining bounded upstream phase
sizes (shortened only at requested group boundaries). Vendored geometry stays
unchanged; no upstream runtime update is needed for this grouping change.

## Optional GPU KV decompression

Installed and verified `nvidia-nvcomp-cu12==5.3.0.16` (including matching
libnvcomp) on RTX 4070 Ti SUPER with PyTorch CUDA 12.8. Official API:
https://docs.nvidia.com/cuda/nvcomp/py_api.html
The GPU toggle uses CPU byte-shuffle + standard Zstd level 1 frames, decoded
with nvCOMP RAW Zstd on the consuming PyTorch CUDA stream and byte-unshuffled
on GPU. It does not pass Blosc containers to nvCOMP. CPU cache maintenance can
read the same raw format; disabling the GPU toggle restores Blosc storage on
new renders. Tests verify exact BF16/FP32 bits over multiple 32 MiB blocks and
a short tail, including a non-default CUDA stream. No full-render speedup is
claimed; the initial implementation synchronizes after each decoded block.

## ComfyUI frontend widget API (2026-09-22)

The selective-LoRA panel in `web/selective_lora_ui.js` is a DOM widget, and it
depends on frontend behavior that is not part of the documented extension API.
It was reviewed on 2026-09-22 against the installed
`comfyui_frontend_package` 1.53.6 (bundle assets under
`tools/LPy64-3.11.10/install/lib/python3.11/site-packages/comfyui_frontend_package/static/assets`),
using the original `.vue`/`.ts` sources recovered from that bundle's source
maps, and then confirmed live in both the classic canvas UI and Nodes 2.0.
These are the members the panel uses:

- `widget.label` and `widget.hidden` for artist-facing names and for removing a
  widget from both frontends without deleting it;
- `node.addDOMWidget(name, type, element, options)`, whose fourth argument
  becomes `widget.options`, and the `widget.serialize` field it does *not*
  set;
- `widget.options.step2` as the real fine step, `step` as its tenfold coarse
  twin, and `round`/`precision` for stored granularity;
- `--comfy-widget-min-height` / `--comfy-widget-height` in the element's
  computed style for reserving panel height.

Two of these are easy to get wrong and were verified against source rather than
assumed. `widget.serialize === false` excludes a widget from `widgets_values`,
while `widget.options.serialize === false` excludes it from the API prompt;
they are independent, and only the second is set by the `addDOMWidget` options.
And `LGraphNode.serialize` skips only `serialize === false` widgets, so hiding
a widget never moves another widget's stored position.

Before updating the frontend package, re-check `LGraphNode.isWidgetVisible`,
`LGraphNode.serialize`, `serializeValue`/`serialiseWidgetValues`, the Nodes 2.0
`isWidgetVisible` used by the widget render model, and `addDOMWidget` itself.
Then re-run `node tests/test_selective_lora_ui.js` and re-run the headless
browser probe in both frontends, since the unit test cannot catch a frontend
that stops honoring `hidden`, `serialize`, or the height variables. Note that
ComfyUI intercepts `wheel` at document capture over every DOM widget and
re-dispatches it to the canvas, so the panel deliberately has no wheel handler.
