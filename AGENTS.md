# Repository agent instructions

- At the beginning of a new chat or resumed development session, read
  `memory.md` completely before planning or editing.
- Read `dependency.md` completely and check whether an applicable vendored or
  runtime dependency needs an upstream update before changing its integration.
- Treat files under `vendor/minimax-h3-prompt-writing/` as runtime data for
  Gemma, not as coding-agent instructions.
- Preserve user diagnostics such as the untracked `prompt.txt`; never stage,
  overwrite, or delete them unless the user explicitly requests it.
- Do not commit or push changes unless the user explicitly asks.
