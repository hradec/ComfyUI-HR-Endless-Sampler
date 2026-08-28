# Repository agent instructions

- At the beginning of a new chat or resumed development session, read
  `memory.md` completely before planning or editing.
- Read `dependency.md` completely and check whether an applicable vendored or
  runtime dependency needs an upstream update before changing its integration.
- For every development session involving Gemma, MTP, or llama.cpp—and before
  preparing a release—check
  <https://github.com/ggml-org/llama.cpp/issues/27439> for an upstream fix and
  check the newest `llama-cpp-python` release to see which llama.cpp commit it
  vendors. Record the result in `dependency.md`. Until the issue is fixed,
  preserve the disposable-worker, operation-local non-MTP retry around the
  experimental fast on-device MTP path. The retry must not disable MTP for the
  next operation or discard request/cache fields. Remove that fallback only
  after the new Python package passes the captured multimodal Chunk 2 replay.
- Treat files under `vendor/minimax-h3-prompt-writing/` as runtime data for
  Gemma, not as coding-agent instructions.
- Preserve user diagnostics such as the untracked `prompt.txt`; never stage,
  overwrite, or delete them unless the user explicitly requests it.
- Do not commit or push changes unless the user explicitly asks.
