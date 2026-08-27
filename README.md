# ComfyUI-HR-Endless-Sampler 
## (the older ComfyUI MiniMax H3 Sampler Unlimited)

`HR Endless Sampler` is a chunked replacement for ComfyUI's
`SamplerCustomAdvanced` for long video/audio latents, currently supports 
Minimax H3 only. The plan is to add support to LTX 2.5 in the near future.

`HR Endless Sampler` is able to render videos of any length by automatically 
splitting the inference into small chunks of the same long latent. It uses 
Gemma4 12B QAT internally to analyze the original prompt and all references, 
plan the action-timing for each shot and each chunk, then analyzes previous 
rendered frames and writes new small prompts for each chunk, maintaining 
the continuity and coherence of the entire video. 

Using `HR Endless Sampler Preview` node (based on the amazing KJ Live preview node) 
allows to visualize the whole video as it is infered, with a timeslider that displays
the video shots and each chunk. You can even visualize each chunk prompt Gemma created
by holding the mouse pointer over a chunk bar. 

## Quick HELP as I don't have a workflow template yet!
The way to use is pretty straight forward - just replace the normal "Sampler" node by this one, and add the preview node behind it so it can show the preview as the inference happens. You just have to add the extra inputs:
- `clip` - just connect the model clip
- `vae` - just connect the video vae model
- `images` - connect the images you used with minimax guiding - for ref2va, those would be the reference images
- `prompt` - connect the text of the full prompt you are using with minimax
- `fps` - should always be 24, but if you have a node that sets the fps, you can connect it here too.
<img width="349" height="422" alt="Image" src="https://github.com/user-attachments/assets/ebb106f4-804b-4465-8ffd-6a26a94ef6a2" />

## Main settings

`chunk_frames` is the number of frames sampled in one H3 call. Use the largest
value that fits in VRAM. Smaller chunks use less VRAM, but need more handoffs.
For example, 39 frames is a practical 1080p starting point on a 16 GB GPU.
H3 uses a `5 + 17k` frame grid, so the effective size is aligned to that grid.

`video_continuation` is the number of completed frames carried from the last
chunk into the next one. H3 sees them as a synchronized `<Video N>` and
`<Audio N>` reference. `22` frames is a good default for continuity. `5` is
the minimum. Larger values use more VRAM. If it is larger than the current
chunk, the sampler caps it to the chunk size.

The sampler also uses the previous chunk's final five frames as a small H3
boundary keyframe. This is automatic. It helps adjacent chunks meet cleanly.

`cache_gemma_preproduction` saves Gemma's static preproduction context in
system RAM. This can make later Gemma requests much faster because they do not
need the full source prompt and shot plan again. Linux uses `/dev/shm` when it
has enough free RAM; otherwise the normal temporary directory is used. The
cache uses several GiB of RAM, never VRAM. It is optional and does not change
the generated video.

`debug` adds detailed prompt and memory information to the console.

`debug_stop_chunk` stops after a selected 1-based chunk. `0` means render the
whole video.

`debug_start_chunk` reruns from a selected 1-based chunk. It is useful for
testing a later shot without sampling all earlier chunks again. The first run
creates a temporary replay cache; later compatible runs reuse its noise,
completed chunks, and continuation boundary. Set it back to `0` to clear that
temporary cache on the next render.

If the main prompt changes during a replay, the sampler keeps the saved physical
frames and noise but asks Gemma to make a new preproduction plan from the new
prompt. It also rebuilds the Gemma KV cache. This lets prompt changes such as
moving dialogue earlier in a shot affect the rerun chunk.

## How the sampler works

The sampler runs chunks in order. A chunk finishes all H3 sampling steps before
the next chunk begins. The completed tail becomes the next chunk's Video1/Audio1
continuation reference.

Before Chunk 1, Gemma reads the complete prompt and plans the timing of every
source shot. It knows every physical chunk boundary before H3 starts. This gives
Gemma a full view of a long action instead of making it guess each chunk in
isolation.

For every chunk, Gemma receives:

- the complete original prompt and the relevant timing plan;
- the frames and shots the chunk must produce;
- the latest generated stills from the previous chunk, sampled at 2 FPS plus
  its exact final frame; and
- the previous chunk's Gemma prompt and end state, when they still match the
  current source prompt.

Gemma writes one short H3 `detailed_description` for that chunk. It keeps exact
dialogue inside `<d>...</d>`, preserves real shot cuts, and uses local H3
timecodes for cuts. H3 receives only that final description, not Gemma's JSON
notes or planning data.

The sampler saves the latest Gemma transcript after every chunk, even with
`debug` off:

```text
${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_chunk_prompts.txt
${TMPDIR}/comfyui-hr-endless-sampler/last_gemma_images/
```

The text file includes the preproduction plan, each request to Gemma, Gemma's
JSON response, any correction request, and the final prompt sent to H3. The
image directory contains the stills that Gemma saw. A new render replaces both.

## Gemma 4 setup

The sampler uses the official Google Gemma 4 12B QAT Q4 GGUF model through
`llama-cpp-python`. Install the dependencies with ComfyUI's Python:

```bash
~/comfyui/tools/python.sh -m pip install -r requirements.txt
```

On first use, the sampler downloads Gemma and its projector to:

```text
models/llama_cpp/gemma-4-12b-it-qat-q4_0/
```

Gemma runs in a separate process between H3 chunks. H3, Qwen, and the video VAE
are unloaded before Gemma runs, and the Gemma process exits before H3 sampling
resumes. This is intentional: it releases Gemma's CUDA allocations before H3
needs VRAM again.

The editable Gemma instructions are in
[`gemma4_prompts.txt`](gemma4_prompts.txt). The sampler reads this file again
before preproduction and before each chunk. You can adjust the wording without
editing Python, but keep the named section headers and `{{placeholders}}`.

## Preview node

Place `HR Endless Sampler Preview` in the model path before the guider used by
the sampler. Connect the actual H3 model through the preview node, then use its
model output for the guider and sampler.

The preview plays every completed chunk in order. It can restore the current
preview after a browser refresh. Its timeline uses a different color for each
chunk and shows brackets for source shots. Hover a chunk color to see Gemma's
H3 prompt for that chunk.

Use the small play/pause button, Space, or the timeline to control playback.
Focus the preview and use Left/Right for frame stepping. The lower-right label
shows the output frame and, when available, the shot and chunk number.

`tiny_vae: none` uses the fast H3 Latent2RGB preview. Select
`taeh3.safetensors` for a more representative preview. Tiny-VAE preview costs
more time and VRAM. `max_resolution: 0` keeps the latent preview resolution.
The preview FPS can be changed while it is playing and does not affect sampling.

## Prompt format

Use MiniMax's normal shot format. The first shot has no timecode. Later shots
use a strictly increasing cut time:

```text
[Shot 1] The tiger runs through the jungle.
[Shot 2] At 00:02.833, the camera cuts inside the temple.
```

Set `fps` to the same frame rate used by those timecodes. H3 normally uses
24 FPS. The sampler converts cut times to frames, keeps each real cut at the
correct position inside its physical chunk, and gives H3 the corresponding
local timecode.

For Ref2VA, keep the reference images in the same order as the original H3
conditioning. The sampler keeps those identity/style references for every
chunk. The generated Video1 continuation reference is added separately.

## Memory and performance

The first chunk can fit while the second chunk fails. Later chunks include the
Video1/Audio1 continuation tail, so they use more VRAM than Chunk 1. Choose a
`chunk_frames` and `video_continuation` pair that fits Chunk 2 as well.

Chunking reduces the temporal part of H3's memory use. It cannot make an
arbitrary resolution fit: one full-resolution H3 sampling step must still fit
in VRAM.

The console shows chunk progress, H3 step progress, Gemma preparation progress,
and an end-of-run report. The report includes H3, Qwen, VAE, and Gemma time,
plus peak RAM and VRAM use.

## Current limits

- The released backend currently supports MiniMax H3 only.
- Multi-chunk H3 rendering needs the H3 video VAE.
- Chunked denoise masks are not supported.
- The sampler can reconstruct image and audio Ref2VA inputs. It cannot turn an
  image input back into an original video Ref2VA source.
- Gemma observes generated video frames, not generated audio. It preserves
  dialogue and sound instructions from the source prompt, but does not judge
  the resulting soundtrack.

## References

- [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md)
- [MiniMax H3 base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- [MiniMax H3 full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [Google Gemma 4 12B QAT Q4 GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
