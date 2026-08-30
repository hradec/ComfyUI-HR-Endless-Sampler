# External dependencies and update checks

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

The local director pins `llama-cpp-python==0.3.35` and Google's official
`google/gemma-4-12B-it-qat-q4_0-gguf` model/projector pair. Runtime integration
uses the `cu125` wheel channel because release 0.3.35 publishes both Linux and
Windows CUDA wheels on that channel. It was reviewed on 2026-08-27 against:

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

`gemma4_mtp.py` uses the 0.3.35 low-level MTP/NextN bindings rather than
starting an HTTP server. It intentionally adapts only the linked-context,
single-sequence path used by the disposable director worker. When updating the
binding, verify `load_mtp`, `LLAMA_CONTEXT_TYPE_MTP`, `ctx_other`, the NextN
embedding functions, `PARTIAL_ONLY` target checkpoints and accepted-prefix
replay after a rejected draft, JSON grammar generation, and teardown order.
The fast path currently adds `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`, but it is
allowed only inside the disposable worker: llama.cpp can abort before its C API
returns an error. The parent must preserve the exact request and retry that
operation once without MTP after a native worker exit. The retry changes only
the copied request's MTP flag: all cache fields remain intact, and the next
independent operation attempts MTP again. Generated target logits stay inside
llama.cpp—the director
never requests logprobs, so copying every verification row into NumPy is
prohibited on this path.

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
- 2026-08-30: issue #27439 remains open with the `bug-unconfirmed` label and no
  published fix. PyPI still reports `llama-cpp-python==0.3.35` as the newest
  release (uploaded 2026-08-17). Preserve the disposable worker and
  operation-local non-MTP retry while adding another local llama.cpp director.

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
after generation. The grammar is therefore a recovery-only mechanism: use it
for one retry when the ordinary response is not syntactically valid JSON. A
valid response must take the fast unconstrained path; semantic validation and
chat-style correction still run afterward. The same captured chunk reduced
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
