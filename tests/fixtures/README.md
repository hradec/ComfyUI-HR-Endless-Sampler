# Gemma 4 chunk-prompt fixtures

With sampler `debug` enabled, each real Gemma chunk request is captured under a
temporary directory logged as `Gemma capture directory`. Move a complete
`prompt_NNN_chunk_NNN` directory here when it represents a useful continuity
regression.

Replay it after editing `gemma4_prompts.txt`, without running H3:

```bash
~/comfyui/tools/python.sh tests/replay_gemma_capture.py tests/fixtures/<fixture-directory>
```

`request.json` contains the exact base64 JPEG payload and semantic request sent
to the disposable worker. The separate JPEG and prompt files are inspection
copies. Replay intentionally uses the repository's current prompt templates so
prompt changes can be compared against identical visual evidence, exact frame
mapping, complete source-shot intent, and chunk conditioning inputs.
