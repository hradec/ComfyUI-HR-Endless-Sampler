# External dependencies and update checks

Read this file before changing the Gemma prompt director or refreshing vendored
documentation. The files below are runtime prompt data fed to Gemma; they are
not contributor or coding-agent instructions.

## MiniMax H3 prompt-writing skill

The project vendors MiniMax's official H3 prompt-writing skill so a render does
not depend on network availability and upstream edits cannot silently change
generation behavior halfway through a run.

| Vendored file | Mutable upstream source | SHA-256 checked 2026-08-21 |
| --- | --- | --- |
| `vendor/minimax-h3-prompt-writing/SKILL.md` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md> | `a7000443588ca3f145e3b3fd8900f14e0325dc460bd811268fac89a9dc8e56d0` |
| `vendor/minimax-h3-prompt-writing/references/base-en.txt` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt> | `2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc` |
| `vendor/minimax-h3-prompt-writing/references/ref-en.txt` | <https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt> | `1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7` |

`gemma4.py` reads `SKILL.md` plus exactly one mode guide for every request:

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

## Gemma 4 MTMD runtime

The local director pins `llama-cpp-python==0.3.35` and Google's official
`google/gemma-4-12B-it-qat-q4_0-gguf` model/projector pair. Runtime integration
was reviewed on 2026-08-22 against:

- <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf>, especially
  image-before-text modality ordering and the supported 70, 140, 280, 560, and
  1120 visual-token budgets;
- <https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.h>, which
  defines the MTMD image-token budget and media batching fields;
- <https://github.com/ggml-org/llama.cpp/issues/21550>, which records evaluation
  failures encountered with high Gemma 4 image budgets.

The pinned Python handler binds the MTMD fields but does not expose them in its
constructor. `gemma4.py` therefore owns a narrow subclass that sets the dynamic
70-1120 budget and keeps MTMD, logical, and physical batch capacities at least
1120. It also sends chronological images before the observation text. Before
updating `llama-cpp-python`, verify that the high-level handler's constructor,
`_init_mtmd_context`, cleanup callback, MTMD structure layout, and non-causal
image batching behavior remain compatible. Prefer an upstream public budget
API when one becomes available, then remove the local override and rerun the
real model plus unit tests.
