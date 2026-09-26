import asyncio
import copy
import functools
import gc
import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from aiohttp import web
from phonemizer import phonemize
from phonemizer.separator import Separator

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from tqdm.auto import tqdm
from tqdm import tqdm as _cli_tqdm

try:
    from server import PromptServer
except ImportError:  # Unit tests can import the node helpers without ComfyUI's server.
    PromptServer = None

from .gemma4 import (
    Gemma4DependencyError,
    Gemma4ObservationError,
    Gemma4PreproductionCache,
    _timing_plan_from_payload,
    _timing_plan_payload,
    _validate_timing_plan,
    reset_gemma4_live_output_log,
    reset_gemma4_raw_output_log,
)
from .preview import begin_preview_execution, build_cached_final_preview_snapshot
from .python.audio_sr import AudioSR, fix_audio
from .python.preproduction import HRPreProduction, ENABLE_GEMMA4_MTP
from .video_io import HREndlessTimeline, IntermediateChunkVideoWriter, load_replay_preview_proxy, normalize_timeline, save_replay_preview_proxy


AUDIO_LATENT_FPS = 40
VIDEO_FPS = 24
MIN_VIDEO_STEPS = 2
CANVAS_MULTIPLE = 32
H3_REFERENCE_VIDEO_SHORT_EDGE = 768
H3_REFERENCE_VIDEO_MAX_PIXELS = 768 * 1344
COLOR_DIAGNOSTIC_MAX_FRAMES = 16
COLOR_DIAGNOSTIC_MAX_SIDE = 256
COLOR_CORRECTION_MAX_FRAMES = 8
COLOR_CORRECTION_MAX_SIDE = 256
COLOR_CORRECTION_MIN_TONE_RATIO = 0.75
COLOR_CORRECTION_MAX_TONE_RATIO = 1.30
COLOR_CORRECTION_MIN_RGB_BALANCE = 0.94
COLOR_CORRECTION_MAX_RGB_BALANCE = 1.06
PHONEME_SECONDS = 0.055
WORD_ONSET_SECONDS = 0.075
COMMA_PAUSE_SECONDS = 0.15
SENTENCE_PAUSE_SECONDS = 0.30
ELLIPSIS_PAUSE_SECONDS = 0.50
minimax_visual_cond_noise_aug = 0.99
VIDEO_CONTINUATION_RESOLUTIONS = (
    "full",
    "0.98mp (1344x768 native)",
    "0.90mp (1280x736)",
    "0.80mp (1216x672)",
    "0.70mp (1152x640)",
    "0.60mp (1056x608)",
    "0.50mp (960x544)",
    "0.40mp (864x480)",
    "0.30mp (736x416)",
    "0.20mp (608x352)",
    "0.10mp (448x256)",
)
VIDEO_CONTINUATION_METHOD_VIDEO1 = "Video1 reference (current)"
VIDEO_CONTINUATION_METHOD_MASKED_AV = "Masked AV overlap (experimental)"
VIDEO_CONTINUATION_METHOD_TAOMATE = "TaoMate-H3 streaming (experimental)"
VIDEO_CONTINUATION_METHODS = (
    VIDEO_CONTINUATION_METHOD_VIDEO1,
    VIDEO_CONTINUATION_METHOD_MASKED_AV,
    VIDEO_CONTINUATION_METHOD_TAOMATE,
)
# Temporary A/B switch: do not copy/protect the selected 39/90/... previous AV
# run inside the new chunk latent. Native-boundary mode instead uses the
# selected continuation duration as a disposable packing prefix.
ENABLE_MASKED_AV_OVERLAP = False
# Test a fully frozen AV prefix: Video already uses an all-zero denoise mask,
# and audio now does too. Set this back to 8 to restore the former 0.2-second
# audio seam feather experiment.
MASKED_AV_AUDIO_FEATHER_TICKS = 0
# Inspection switch for the frozen masked-AV experiment. Normally the locked
# physical AV prefix is removed from this chunk so the retained output has no
# duplicate frames. Leave the complete prefix in this chunk temporarily to
# inspect exactly what H3 emitted during its locked temporal interval.
# ponytail: this intentionally repeats the preceding AV tail; restore True
# after diagnosing the prefix behavior.
TRIM_MASKED_AV_PREFIX = True
COLOR_CORRECTION_MODES = ("disable", "chunk boundaries", "entire shots", "chunk boundaries + entire shots")
VIDEO_CONTINUATION_CANVASES = {
    label: tuple(int(value) for value in re.search(r"\((\d+)x(\d+)", label).groups())
    for label in VIDEO_CONTINUATION_RESOLUTIONS[1:]
}
DEFAULT_PYTORCH_MEMORY_FRACTION = 0.85
VRAM_DEBUG_WRAPPER_KEY = "hr_endless_sampler_vram_debug"
# The current llama-cpp-python Gemma 4 MTP verifier is slower than ordinary
# decoding and can fail hybrid-state rollback. Keep its implementation intact,
# but reject stale workflow values until the upstream path is usable again.
# Temporarily disable the disposable three-step continuation memory probe.  It
# remains implemented below so the experiment can be restored by changing this
# single flag after its startup cost is useful again.
ENABLE_DEBUG_MEMORY_PREFLIGHT = True
# Experimental A/B switch. False regenerates complete AV noise for every
# chunk with seed * chunk_number; True preserves one sliced full-sequence noise.
TOGGLE_SINGLE_NOISE = True
# Set this to False only for the isolation experiment that retains the
# native visual boundary keyframe while suppressing Video1/Audio1 in Qwen,
# DiT references, and prompt text.
INCLUDE_VIDEO1_REFERENCE = True
# Experimental H3-conditioning A/B switch.  The normal source uses the raw
# prior VAE decode, while this path feeds the finalized display-color-corrected
# tail back through H3's pixel-space continuation inputs.  It deliberately
# does not alter the native latent boundary, which has no pixel-space
# color-correction equivalent.
USE_COLOR_CORRECTED_H3_CONTEXT = True
# Per-chunk retention prose proved counterproductive with H3: it can behave
# like a fresh scene/shot constraint and override the visually established
# continuation. Preserve only the user's global retention_analysis for now.
# Gemma may still author and validate its internal value for diagnostics and
# future experiments, but it is not inserted into H3 prompts or timeline data.
INCLUDE_PER_CHUNK_RETENTION_ANALYSIS = False
GEMMA_PROMPT_LOG_DIRNAME = "comfyui-hr-endless-sampler"
GEMMA_PROMPT_LOG_FILENAME = "last_gemma_chunk_prompts.txt"
GEMMA_IMAGE_LOG_DIRNAME = "last_gemma_images"
REPLAY_CACHE_DIRNAME = "last_run_replay"
REPLAY_CACHE_FORMAT = 10
_REPLAY_CACHE_ACTIVITY_LOCK = threading.Lock()
_REPLAY_CACHE_ACTIVE_RUNS = 0
# Preview's cache button controls whether future sampler executions use or
# record the disposable replay checkpoint. It deliberately starts enabled to
# preserve the established automatic interrupted-render recovery behavior.
REPLAY_CACHE_ENABLED = True
DETAILED_DESCRIPTION_FIELD = re.compile(r"detailed_description\s*:", re.IGNORECASE)
INTEGRATED_DESCRIPTION_FIELD = re.compile(r"integrated_multimodal_description\s*:", re.IGNORECASE)
SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
SUBJECT_DEFINITIONS_FIELD = re.compile(r"(?im)^\s*subject_definitions\s*:\s*$")
SUMMARY_FIELD = re.compile(r"(?im)^(\s*summary\s*:\s*)(.*)$")
RETENTION_FIELD = re.compile(r"(?im)^\s*retention_analysis\s*:\s*$")
PICTURE_LABEL = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)
DIALOGUE_BLOCK = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)
SUBJECT_SPEAKER = re.compile(r"(<Subject\s+\d+>)\s*\((S\d+)\)", re.IGNORECASE)


def _preserve_global_prompt_sections(chunk_prompt, global_prompt):
    """Keep global subjects, summary and retention verbatim across chunk directors."""
    headers = r"subject_definitions|summary|retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music|non_diegetic_audio"
    for field in ("subject_definitions", "summary", "retention_analysis"):
        section = re.compile(r"(?im)^[ \t]*" + field + r"[ \t]*:[\s\S]*?(?=^[ \t]*(?:" + headers + r")[ \t]*:|\Z)")
        original = section.search(global_prompt)
        replacement = original.group(0) if original else ""
        if section.search(chunk_prompt):
            chunk_prompt = section.sub(lambda match: replacement, chunk_prompt)
        elif original:
            # Missing sections go before the description without altering source text.
            following = headers.split("|")[headers.split("|").index(field) + 1:]
            position = re.search(r"(?im)^[ \t]*(?:" + "|".join(following) + r")[ \t]*:", chunk_prompt)
            offset = position.start() if position else 0
            separator = "" if replacement.endswith("\n") else "\n"
            chunk_prompt = chunk_prompt[:offset] + replacement + separator + chunk_prompt[offset:]
    return chunk_prompt


def _description_field(prompt, start=0):
    return DETAILED_DESCRIPTION_FIELD.search(prompt, start) or INTEGRATED_DESCRIPTION_FIELD.search(prompt, start)


def _normalize_last_dialogue_terminal_punctuation(description):
    """Give the final H3 dialogue block one unambiguous terminal stop.

    Chunk-owned dialogue can legitimately be split from source prose at a
    comma, semicolon, ellipsis, or similar punctuation. MiniMax may interpret
    that open-ended punctuation as a pause or unfinished continuation at the
    physical chunk boundary. Preserve every earlier block and every spoken
    word, but normalize the *last* block's unsupported terminal punctuation to
    one period. A single period, exclamation mark, question mark, combinations
    of ``!``/``?``, and dialogue with no terminal punctuation remain unchanged.
    """
    if not isinstance(description, str) or not description:
        return description
    matches = list(DIALOGUE_BLOCK.finditer(description))
    if not matches:
        return description
    match = matches[-1]
    value = match.group(1)
    stripped = value.rstrip()
    trailing_space = value[len(stripped):]
    if not stripped:
        return description

    terminal_start = len(stripped)
    while terminal_start > 0 and unicodedata.category(stripped[terminal_start - 1]).startswith("P"):
        terminal_start -= 1
    if terminal_start == len(stripped):
        return description
    terminal = stripped[terminal_start:]
    supported = all(character in ".!?" for character in terminal)
    is_ellipsis = terminal == "…" or len(terminal) > 1 and set(terminal) == {"."}
    if supported and not is_ellipsis:
        return description

    normalized = stripped[:terminal_start] + "." + trailing_space
    return description[:match.start(1)] + normalized + description[match.end(1):]


def _begin_last_gemma_prompt_log(chunk_frames, context_keyframes, guide_overlap,
                                 video_continuation, video_continuation_method,
                                 video_continuation_res, fps, chunk_count,
                                 cache_gemma_preproduction=False,
                                 gemma4_mtp=False):
    """Replace the fixed temp capture so it always represents the latest run."""
    path = Path(tempfile.gettempdir()) / GEMMA_PROMPT_LOG_DIRNAME / GEMMA_PROMPT_LOG_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "HR Endless Sampler last-run Gemma chunk prompts\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Configuration: chunk_frames={chunk_frames}, context_keyframes={context_keyframes}, "
            f"guide_overlap={guide_overlap}, video_continuation={video_continuation}, "
            f"video_continuation_method={video_continuation_method}, "
            f"video_continuation_res={video_continuation_res}, "
            f"fps={fps:g}, chunks={chunk_count}, "
            f"cache_gemma_preproduction={bool(cache_gemma_preproduction)}, "
            f"gemma4_mtp={bool(gemma4_mtp)}\n\n",
            encoding="utf-8",
        )
    except OSError as error:
        logging.warning("HR Endless Sampler could not initialize Gemma prompt log %s: %s", path, error)
        return None
    logging.info("HR Endless Sampler writing last-run Gemma prompts to %s", path)
    return path


def _append_last_gemma_prompt(path, chunk_header, chunk_prompt, *, system_prompt=None,
                              observation_prompt=None, gemma_response=None,
                              validation_warnings=()):
    """Flush one complete Gemma-to-H3 transcript entry immediately."""
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as prompt_file:
            if system_prompt:
                prompt_file.write("=== GEMMA SYSTEM PROMPT ===\n")
                prompt_file.write(system_prompt.rstrip())
                prompt_file.write("\n\n")
            prompt_file.write("=" * 200)
            prompt_file.write("\n")
            prompt_file.write(chunk_header.rstrip())
            prompt_file.write("\n\n=== GEMMA REQUEST ===\n")
            prompt_file.write((observation_prompt or "not available").rstrip())
            prompt_file.write("\n\n=== GEMMA RESPONSE ===\n")
            prompt_file.write((gemma_response or "not available").rstrip())
            if validation_warnings:
                prompt_file.write("\n\n=== GEMMA VALIDATION WARNINGS ===\n")
                prompt_file.write("\n".join(f"- {warning}" for warning in validation_warnings))
            prompt_file.write("\n\n=== FINAL H3 PROMPT ===\n")
            prompt_file.write((chunk_prompt or "not sampled: Gemma returned no usable detailed_description; no algorithmic fallback was applied.").rstrip())
            prompt_file.write("\n\n")
    except OSError as error:
        logging.warning("HR Endless Sampler could not append Gemma prompt log %s: %s", path, error)


def _append_gemma_timing_plan(path, timing_plan, *, system_prompt=None,
                              planning_prompt=None, gemma_response=None,
                              validation_warnings=()):
    """Flush the one-time preproduction request before the first chunk entry."""
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as prompt_file:
            if system_prompt:
                prompt_file.write("=== GEMMA PREPRODUCTION SYSTEM PROMPT ===\n")
                prompt_file.write(system_prompt.rstrip())
                prompt_file.write("\n\n")
            prompt_file.write("=" * 200)
            prompt_file.write("\n=== GEMMA SHOT TIMING PREPRODUCTION ===\n\n")
            prompt_file.write("=== GEMMA REQUEST ===\n")
            prompt_file.write((planning_prompt or "not available").rstrip())
            prompt_file.write("\n\n=== GEMMA RESPONSE ===\n")
            prompt_file.write((gemma_response or "not available").rstrip())
            if validation_warnings:
                prompt_file.write("\n\n=== GEMMA VALIDATION WARNINGS ===\n")
                prompt_file.write("\n".join(f"- {warning}" for warning in validation_warnings))
            prompt_file.write("\n\n=== VALIDATED SHOT TIMING PLAN ===\n")
            prompt_file.write((timing_plan or "not available: sampling stopped before Chunk 1.").rstrip())
            prompt_file.write("\n\n")
    except OSError as error:
        logging.warning("HR Endless Sampler could not append Gemma timing plan log %s: %s", path, error)


def _reset_last_gemma_image_log():
    """Replace only the fixed image subdirectory for the latest sampled run."""
    path = Path(tempfile.gettempdir()) / GEMMA_PROMPT_LOG_DIRNAME / GEMMA_IMAGE_LOG_DIRNAME
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        logging.warning("HR Endless Sampler could not reset Gemma image log %s: %s", path, error)
        return None
    logging.info("HR Endless Sampler writing last-run Gemma images to %s", path)
    return path


def _replay_cache_root():
    """Return the bounded, disposable cache for debug chunk replays."""
    return Path(tempfile.gettempdir()) / GEMMA_PROMPT_LOG_DIRNAME / REPLAY_CACHE_DIRNAME


def _remove_replay_cache():
    """Remove only the sampler's fixed temporary replay cache."""
    path = _replay_cache_root()
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except OSError as error:
        logging.warning("HR Endless Sampler could not clear replay cache %s: %s", path, error)


def _replay_cache_activity(active):
    """Track sampler executions so the UI cannot erase an in-use checkpoint."""
    global _REPLAY_CACHE_ACTIVE_RUNS
    with _REPLAY_CACHE_ACTIVITY_LOCK:
        _REPLAY_CACHE_ACTIVE_RUNS = max(
            0,
            _REPLAY_CACHE_ACTIVE_RUNS + (1 if active else -1),
        )


def _replay_cache_enabled():
    """Read the preview-controlled replay policy for a new sampler execution."""
    with _REPLAY_CACHE_ACTIVITY_LOCK:
        return bool(REPLAY_CACHE_ENABLED)


def _guard_replay_cache(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        _replay_cache_activity(True)
        try:
            return function(*args, **kwargs)
        finally:
            _replay_cache_activity(False)
    return guarded


def _replay_cache_ui_status_unlocked():
    """Build cache status while the caller owns the activity lock."""
    active = _REPLAY_CACHE_ACTIVE_RUNS > 0
    root = _replay_cache_root()
    manifest = {}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError, OSError, json.JSONDecodeError):
        pass
    try:
        completed_chunks = max(0, int(manifest.get("completed_chunks", 0)))
    except (TypeError, ValueError):
        completed_chunks = 0
    status = str(manifest.get("status", ""))
    cached_chunks = []
    directory = root / "chunks"
    if directory.is_dir():
        for path in directory.glob("chunk_*.pt"):
            try:
                cached_chunks.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    cached_chunks.sort()
    # A cancelled first chunk has no completed chunk file, but it still leaves
    # an initial/noise checkpoint that must be removable before the next run.
    # Once marked interrupted, all cache writes are complete, so this cache is
    # also safe to clear while the interrupted node is unwinding.
    has_cache = (root / "manifest.json").is_file() and (
        completed_chunks > 0 or status == "interrupted"
    )
    return {
        "has_cache": has_cache,
        "enabled": bool(REPLAY_CACHE_ENABLED),
        "active": active,
        "status": status,
        "completed_chunks": completed_chunks,
        "cached_chunks": cached_chunks,
    }


def _replay_cache_ui_status():
    """Return the small, non-tensor cache state needed by the preview button."""
    with _REPLAY_CACHE_ACTIVITY_LOCK:
        return _replay_cache_ui_status_unlocked()


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_preview/replay_cache")
    async def hr_endless_sampler_replay_cache_status(_request):
        status = await asyncio.to_thread(_replay_cache_ui_status)
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_preview/cached_preview")
    async def hr_endless_sampler_cached_preview(request):
        """Rebuild a dormant preview from the disk replay cache on demand."""
        try:
            max_resolution = int(request.rel_url.query.get("max_resolution", "0"))
            quality = int(request.rel_url.query.get("quality", "75"))
        except ValueError:
            return web.json_response({"error": "invalid preview dimensions"}, status=400)
        try:
            fps = float(request.rel_url.query.get("fps", "24"))
        except ValueError:
            return web.json_response({"error": "invalid preview fps"}, status=400)
        if fps <= 0:
            return web.json_response({"error": "preview fps must be greater than zero"}, status=400)
        snapshot = await asyncio.to_thread(
            _cached_replay_preview_snapshot,
            request.rel_url.query.get("node_id", ""),
            max_resolution=max_resolution,
            quality=quality,
            fps=fps,
        )
        # The restored frame payload can be large; serialize it off-loop too.
        return await asyncio.to_thread(web.json_response, snapshot or {}, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_preview/replay_cache_enabled")
    async def hr_endless_sampler_set_replay_cache_enabled(request):
        """Never wait for the replay lock on aiohttp's event loop."""
        return await asyncio.to_thread(_set_replay_cache_enabled_response, request)

    def _set_replay_cache_enabled_response(request):
        """Apply cache policy under the shared lock in a worker thread."""
        requested = request.query.get("enabled", "")
        if requested not in {"0", "1"}:
            return web.json_response(
                {"error": "enabled must be 0 or 1"},
                status=400,
                headers={"Cache-Control": "no-store"},
            )
        with _REPLAY_CACHE_ACTIVITY_LOCK:
            if _REPLAY_CACHE_ACTIVE_RUNS > 0:
                return web.json_response(
                    {**_replay_cache_ui_status_unlocked(), "error": "Cache policy can change only between sampler renders."},
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
            global REPLAY_CACHE_ENABLED
            REPLAY_CACHE_ENABLED = requested == "1"
            status = _replay_cache_ui_status_unlocked()
        logging.info(
            "HR Endless Sampler replay cache is %s for future sampler runs.",
            "enabled" if status["enabled"] else "disabled",
        )
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_preview/replay_cache_chunk")
    async def hr_endless_sampler_delete_replay_cache_chunk(request):
        """Run disk mutation and lock acquisition off the event loop."""
        return await asyncio.to_thread(_delete_replay_cache_chunk_response, request)

    def _delete_replay_cache_chunk_response(request):
        """Delete only the requested checkpoint under the cache lock."""
        try:
            chunk_number = int(request.query.get("chunk", ""))
        except (TypeError, ValueError):
            chunk_number = 0
        if chunk_number < 1:
            return web.json_response(
                {"error": "chunk must be a positive integer"},
                status=400,
                headers={"Cache-Control": "no-store"},
            )
        with _REPLAY_CACHE_ACTIVITY_LOCK:
            if _REPLAY_CACHE_ACTIVE_RUNS > 0:
                return web.json_response(
                {**_replay_cache_ui_status_unlocked(), "error": "Cached chunks can change only between sampler renders."},
                    status=409,
                    headers={"Cache-Control": "no-store"},
                )
            cache = _LastRunReplayCache()
            status = _replay_cache_ui_status_unlocked()
            # ``completed_chunks`` becomes the first missing chunk after a
            # sparse deletion. Later checkpoint files remain individually
            # deletable, so availability must be based on the file itself.
            if not cache.has_chunk(chunk_number):
                return web.json_response(
                    {**status, "error": f"Cached Chunk {chunk_number} is unavailable."},
                    status=404,
                    headers={"Cache-Control": "no-store"},
                )
            try:
                cache.delete_chunk(chunk_number)
            except (OSError, RuntimeError, ValueError) as error:
                return web.json_response(
                    {**_replay_cache_ui_status_unlocked(), "error": str(error)},
                    status=500,
                    headers={"Cache-Control": "no-store"},
                )
            status = _replay_cache_ui_status_unlocked()
        logging.info(
            "HR Endless Sampler deleted cached Chunk %d; intact later checkpoints remain available for replay.",
            chunk_number,
        )
        return web.json_response(
            {**status, "deleted_from_chunk": chunk_number},
            headers={"Cache-Control": "no-store"},
        )


def _replay_cpu_copy(value):
    """Detach replay state from VRAM before persisting it to the temp cache."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _replay_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_replay_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_replay_cpu_copy(item) for item in value]
    return value


def _replay_load_tensor_file(path):
    """Load only ordinary tensors/containers written by this process."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # ``weights_only`` was added long before supported ComfyUI builds, but
        # retain a compatibility path for an older isolated Python runtime.
        return torch.load(path, map_location="cpu")


def _replay_write_tensor_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(_replay_cpu_copy(value), temporary)
    temporary.replace(path)


def _replay_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _replay_plan_signature(plan):
    """Keep only deterministic JSON-friendly physical chunk geometry."""
    keys = (
        "frame_start", "frame_end", "video_start", "video_end", "audio_start", "audio_end",
        "context_video_t", "context_audio_t", "output_trim_frames", "synthetic_prefix",
    )
    return [{key: chunk.get(key) for key in keys} for chunk in plan]


def _replay_fingerprint(video, audio, plan, *, fps, chunk_frames,
                        context_keyframes, guide_overlap, video_continuation,
                        video_continuation_res, ref2va,
                        video_continuation_method=VIDEO_CONTINUATION_METHOD_VIDEO1):
    """Describe the immutable tensor/layout inputs required for an exact replay.

    The source prompt intentionally is not part of this signature: editing it
    is the reason to replay a later chunk.  Its hash is still recorded for
    diagnostics in the cache manifest.
    """
    return {
        "chunk_prompt_marker_mode": "global-shot-labels-local-time-dialogue-segments-v3",
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "video_dtype": str(video.dtype),
        "audio_dtype": str(audio.dtype),
        "fps": float(fps),
        "chunk_frames": int(chunk_frames),
        "context_keyframes": int(context_keyframes),
        "guide_overlap": int(guide_overlap),
        "video_continuation": int(video_continuation),
        "video_continuation_method": str(video_continuation_method),
        "video_continuation_res": str(video_continuation_res),
        "ref2va": bool(ref2va),
        "plan": _replay_plan_signature(plan),
    }


class _LastRunReplayCache:
    """Persistent-on-disk, bounded state needed to restart a serial chunk run.

    The cache never replaces model/sampler inputs.  It pins the original
    source/noise tensors and all completed serial state so a changed Gemma
    prompt can be evaluated from a later physical chunk without rerunning the
    earlier H3 calls.
    """

    def __init__(self):
        self.root = _replay_cache_root()

    @property
    def manifest_path(self):
        return self.root / "manifest.json"

    @property
    def initial_path(self):
        return self.root / "initial_tensors.pt"

    @property
    def timing_path(self):
        return self.root / "preproduction_timing_plan.json"

    def chunk_path(self, chunk_number):
        return self.root / "chunks" / f"chunk_{int(chunk_number):04d}.pt"

    def preview_path(self, chunk_number):
        return self.root / "chunks" / f"chunk_{int(chunk_number):04d}.preview.mp4"

    def clear(self):
        _remove_replay_cache()

    def create(self, fingerprint, source_prompt, initial_tensors):
        self.clear()
        self.root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "format": REPLAY_CACHE_FORMAT,
            "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "fingerprint": fingerprint,
            "source_prompt_sha256": hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
            "status": "recording",
            "completed_chunks": 0,
        }
        _replay_write_json(self.manifest_path, manifest)
        _replay_write_tensor_file(self.initial_path, initial_tensors)
        logging.info("HR Endless Sampler is recording replay state in %s", self.root)

    def _update_manifest(self, **changes):
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not update replay manifest: {error}") from error
        manifest.update(changes)
        _replay_write_json(self.manifest_path, manifest)

    @staticmethod
    def automatic_resume_chunk(manifest, chunk_count):
        """Return the next chunk, or one-past-end for interrupted finalization."""
        if manifest.get("status") not in {"recording", "interrupted"}:
            return None
        try:
            completed = int(manifest.get("completed_chunks", -1))
        except (TypeError, ValueError):
            return None
        if completed < 0 or completed > int(chunk_count):
            return None
        return completed + 1

    def load_if_compatible(self, fingerprint):
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != REPLAY_CACHE_FORMAT:
                return None, "cache format is obsolete"
            cached_fingerprint = dict(manifest.get("fingerprint") or {})
            provider = cached_fingerprint.get("pre_production", {})
            # Accept older manual-prompt caches that included the editor text.
            if isinstance(provider, dict) and provider.get("provider") == "legacy-chunk-prompts":
                cached_fingerprint["pre_production"] = {key: value for key, value in provider.items() if key != "text"}
            if cached_fingerprint != fingerprint:
                return None, "latent/chunk layout or continuation settings changed"
            initial = _replay_load_tensor_file(self.initial_path)
        except FileNotFoundError:
            return None, "no replay cache exists"
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            return None, f"could not load replay cache: {error}"
        return {"manifest": manifest, "initial": initial}, None

    def load_chunk(self, chunk_number):
        return _replay_load_tensor_file(self.chunk_path(chunk_number))

    def has_chunk(self, chunk_number):
        return self.chunk_path(chunk_number).is_file()

    def load_timing_plan(self):
        try:
            payload = json.loads(self.timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not load cached Gemma preproduction plan: {error}") from error
        return _timing_plan_from_payload(payload)

    def save_timing_plan(self, timing_plan, *, source_prompt=None):
        _replay_write_json(self.timing_path, _timing_plan_payload(timing_plan))
        if source_prompt is not None:
            self._update_manifest(
                source_prompt_sha256=hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
            )

    def save_chunk(self, chunk_number, state, *, fps=None):
        state = dict(state)
        preview_frames = state.pop("corrected_video_frames", None)
        # Browser proxies always store display sRGB. The checkpoint itself
        # keeps H3's inverse-gamma compute frames for an exact replay.
        state["preview_proxy_color_space"] = "srgb"
        state.pop("decoded_video_frames", None)
        preview_audio = state.pop("decoded_preview_audio", None)
        preview_audio_rate = state.pop("decoded_preview_audio_rate", None)
        state.pop("decoded_preview_overlap_audio", None)
        preview_offset = max(0, int(state.get("decoded_preview_offset", 0)))
        try:
            preview_count = max(0, int(state["decoded_preview_end"]) - int(state["decoded_preview_start"]) + 1)
        except (KeyError, TypeError, ValueError):
            preview_count = 0
        # The latent checkpoint is authoritative; a browser proxy failure must
        # never make an otherwise resumable render unrecoverable.
        _replay_write_tensor_file(self.chunk_path(chunk_number), state)
        if isinstance(preview_frames, torch.Tensor) and preview_count and fps is not None:
            proxy_frames = preview_frames[preview_offset:preview_offset + preview_count]
            try:
                save_replay_preview_proxy(
                    self.preview_path(chunk_number),
                    proxy_frames,
                    fps,
                    audio_waveform=preview_audio,
                    audio_sample_rate=preview_audio_rate,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                try:
                    self.preview_path(chunk_number).unlink()
                except FileNotFoundError:
                    pass
                logging.warning(
                    "HR Endless Sampler saved Chunk %d's latent checkpoint but could not encode its browser proxy: %s",
                    int(chunk_number),
                    error,
                )
        self._update_manifest(status="recording", completed_chunks=int(chunk_number))

    def begin_from(self, chunk_number):
        """Mark a restored cache as actively recording its rerun suffix."""
        self._update_manifest(status="recording", completed_chunks=max(0, int(chunk_number) - 1))

    def mark_interrupted(self, completed_chunks):
        self._update_manifest(
            status="interrupted",
            completed_chunks=max(0, int(completed_chunks)),
            interrupted=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    def mark_debug_stop(self, completed_chunks):
        self._update_manifest(status="debug_stop", completed_chunks=max(0, int(completed_chunks)))

    def mark_complete(self, completed_chunks):
        self._update_manifest(status="complete", completed_chunks=max(0, int(completed_chunks)))

    def truncate_from(self, chunk_number):
        directory = self.root / "chunks"
        if not directory.exists():
            return
        for path in directory.glob("chunk_*.pt"):
            try:
                cached_number = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if cached_number >= int(chunk_number):
                path.unlink()
                try:
                    self.preview_path(cached_number).unlink()
                except FileNotFoundError:
                    pass

    def delete_chunk(self, chunk_number):
        """Remove one checkpoint while preserving later cache entries for replay."""
        chunk_number = int(chunk_number)
        if chunk_number < 1 or not self.has_chunk(chunk_number):
            raise ValueError(f"Cached Chunk {chunk_number} is unavailable")
        self.chunk_path(chunk_number).unlink()
        try:
            self.preview_path(chunk_number).unlink()
        except FileNotFoundError:
            pass
        self.mark_interrupted(chunk_number - 1)


def _cached_replay_preview_snapshot(node_id, *, max_resolution, quality, fps):
    """Return a browser snapshot from finalized CPU media in the dormant cache.

    ponytail: the replay cache represents one deliberately global last render,
    so a preview node restores that one cache rather than inventing per-node
    cache ownership. Per-workflow cache namespaces would be the upgrade path.
    """
    if not str(node_id):
        return None
    def progress(completed, total, message):
        """Send lightweight restoration progress without taking a preview lock."""
        if _PROMPT_SERVER is not None:
            _PROMPT_SERVER.send_sync("hr_endless_sampler_cache_restore", {"node_id": str(node_id), "completed": completed, "total": total, "message": message})
    with _REPLAY_CACHE_ACTIVITY_LOCK:
        if not REPLAY_CACHE_ENABLED or _REPLAY_CACHE_ACTIVE_RUNS:
            return None
        status = _replay_cache_ui_status_unlocked()
        if not status["has_cache"] or not status["cached_chunks"]:
            return None
        cache = _LastRunReplayCache()
        try:
            manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        plan = manifest.get("fingerprint", {}).get("plan", [])
        if not isinstance(plan, list) or not plan:
            return None

        # Keep the complete physical geometry visible, including unrendered
        # suffix chunks of an interrupted cache. The encoder below publishes
        # only checkpoints that carry a browser-only preview proxy.
        if not all(isinstance(item, dict) for item in plan):
            return None
        try:
            chunk_ranges = _preview_ranges_for_plan(
                plan,
                show_replaced_tail=(
                    manifest.get("fingerprint", {}).get("video_continuation_method")
                    == VIDEO_CONTINUATION_METHOD_MASKED_AV
                    and not bool(manifest.get("fingerprint", {}).get("keep_continuation_prefix"))
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

        cached_chunks = []
        for position, number in enumerate(status["cached_chunks"]):
            progress(position, 2 * len(status["cached_chunks"]), f"Retrieving cached previous run: loading chunk {number}")
            if number < 1 or number > len(chunk_ranges):
                continue
            try:
                state = cache.load_chunk(number)
            except (OSError, RuntimeError, ValueError):
                continue
            try:
                preview_frames, preview_audio, preview_audio_rate = load_replay_preview_proxy(
                    cache.preview_path(number)
                )
                preview_start = int(state["decoded_preview_start"])
                preview_end = int(state["decoded_preview_end"])
            except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                continue
            if int(preview_frames.shape[0]) != preview_end - preview_start + 1:
                continue
            # Before this marker existed, linear_color_compute proxies were
            # encoded directly from inverse-gamma compute RGB. Restore their
            # intended display transfer when a browser reloads an old cache.
            if manifest.get("fingerprint", {}).get("linear_color_compute") and state.get("preview_proxy_color_space") != "srgb":
                preview_frames = _convert_image_transfer(preview_frames, _inverse_gamma_compute_to_srgb)
            range_item = chunk_ranges[number - 1]
            range_item["start"] = preview_start
            range_item["end"] = preview_end
            description = state.get("gemma_description")
            if isinstance(description, str) and description.strip():
                range_item["gemma_detailed_description"] = description.strip()
            h3_prompt = state.get("h3_prompt")
            if isinstance(h3_prompt, str) and h3_prompt.strip():
                range_item["h3_prompt"] = h3_prompt.strip()
            if INCLUDE_PER_CHUNK_RETENTION_ANALYSIS:
                retention = _chunk_retention_analysis(state.get("gemma_retention_analysis"))
                if retention:
                    range_item["gemma_retention_analysis"] = retention
            for key in (
                "h3_render_seconds",
                "gemma_seconds",
                "gemma_preproduction_seconds",
                "chunk_total_seconds",
            ):
                value = state.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    range_item[key] = float(value)
            cached_chunks.append({
                "index": number - 1,
                "frames": preview_frames,
                "output_start": preview_start,
                "audio": preview_audio,
                "audio_sample_rate": preview_audio_rate,
                "gemma_detailed_description": range_item.get("gemma_detailed_description"),
                "gemma_retention_analysis": range_item.get("gemma_retention_analysis"),
                "h3_prompt": range_item.get("h3_prompt"),
                "h3_render_seconds": range_item.get("h3_render_seconds"),
                "gemma_seconds": range_item.get("gemma_seconds"),
                "gemma_preproduction_seconds": range_item.get("gemma_preproduction_seconds"),
                "chunk_total_seconds": range_item.get("chunk_total_seconds"),
            })
        if not cached_chunks:
            return None

        shot_ranges = []
        if cache.timing_path.is_file():
            try:
                timing_plan = cache.load_timing_plan()
                for shot in timing_plan.shots:
                    shot_ranges.append({
                        "shot": int(shot.source_shot),
                        "start": int(shot.shot_start_frame),
                        "end": int(shot.shot_end_frame) - 1,
                        "source_end": int(shot.shot_end_frame) - 1,
                        "light_change": bool(shot.light_change),
                    })
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                logging.warning("HR Endless Sampler could not restore cached shot brackets.")
        source_fps = manifest.get("fingerprint", {}).get("fps", fps)
        return build_cached_final_preview_snapshot(
            node_id,
            chunk_ranges,
            shot_ranges,
            cached_chunks,
            fps=source_fps,
            max_resolution=max_resolution,
            quality=quality,
            progress_callback=lambda done, total: progress(total + done, 2 * total, f"Retrieving cached previous run: preparing preview {done}/{total}"),
        )


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t))


def _video_steps(frames):
    return ((frames - 5) // 17) * 5 + MIN_VIDEO_STEPS


def _audio_steps(frames):
    return round(frames * AUDIO_LATENT_FPS / VIDEO_FPS)


def _bounded_video_steps(frame_count, max_chunk_frames, field_name, allow_equal=False):
    if frame_count == 0:
        return 0
    if frame_count < 5 or (frame_count - 5) % 17:
        raise ValueError(f"{field_name} must be 0 or use MiniMax H3's 17k+5 frame grid: 5, 22, 39, 56, ...")
    if frame_count > max_chunk_frames or (frame_count == max_chunk_frames and not allow_equal):
        comparison = "no greater than" if allow_equal else "smaller than"
        raise ValueError(f"{field_name} ({frame_count}) must be {comparison} the effective chunk size ({max_chunk_frames})")
    return _video_steps(frame_count)


def _continuation_controls(context_keyframes, guide_overlap, video_continuation, max_chunk_frames):
    """Normalize legacy widgets, validate overlaps, and bound the Video1 tail."""
    legacy_context_keyframes = context_keyframes
    if video_continuation is True:
        video_continuation = legacy_context_keyframes
    elif video_continuation is False:
        video_continuation = 0
    if guide_overlap is True or guide_overlap in ("context_frames", "context_keyframes"):
        guide_overlap = legacy_context_keyframes
    elif guide_overlap is False or guide_overlap == "5 frames":
        context_keyframes = 5
        guide_overlap = 5
    elif guide_overlap == "off":
        context_keyframes = 0
        guide_overlap = 0

    # A continuation reference can be as long as the previous physical chunk,
    # but never longer. Clamping makes a small-chunk workflow convenient: a
    # stable preferred tail such as 22 can stay connected while testing a
    # 5- or 22-frame chunk without creating an impossible reference request.
    if isinstance(video_continuation, int) and not isinstance(video_continuation, bool):
        video_continuation = min(video_continuation, max_chunk_frames)

    values = {
        "context_keyframes": (context_keyframes, False),
        "guide_overlap": (guide_overlap, False),
        "video_continuation": (video_continuation, True),
    }
    for name, (value, allow_equal) in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer: 0, 5, 22, 39, 56, ...")
        _bounded_video_steps(value, max_chunk_frames, name, allow_equal=allow_equal)
    return context_keyframes, guide_overlap, video_continuation, _video_steps(context_keyframes) if context_keyframes else 0


def _timestamp_frame(minutes, seconds, milliseconds, fps):
    return round((int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0) * fps)


def _frame_timestamp(frame, fps):
    total_milliseconds = round(frame / fps * 1000.0)
    minutes, milliseconds = divmod(total_milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _drop_picture_anchors(prompt):
    field = _description_field(prompt)
    if field is None:
        return PICTURE_LABEL.sub("the established subject and scene", prompt)
    prefix = "\n".join(line for line in prompt[:field.start()].splitlines() if "picture" not in line.lower())
    if prefix:
        prefix += "\n"
    return prefix + PICTURE_LABEL.sub("the established subject and scene", prompt[field.start():])


def _audio_reference_speakers(description):
    """Return prior visual speakers and languages from final H3 dialogue prose."""
    speakers = {}
    for dialogue in DIALOGUE_BLOCK.finditer(str(description or "")):
        prefix = str(description)[:dialogue.start()]
        prior_dialogue = prefix.lower().rfind("</d>")
        local_prefix = prefix[prior_dialogue + 4:] if prior_dialogue >= 0 else prefix
        matches = list(SUBJECT_SPEAKER.finditer(local_prefix))
        if not matches:
            continue
        subject, speaker = matches[-1].groups()
        language = re.match(r"\s*\[([^]]+)\]", dialogue.group(1))
        key = (subject, speaker.upper())
        speakers.setdefault(key, set())
        if language is not None:
            speakers[key].add(language.group(1).strip())
    return tuple((subject, speaker, tuple(sorted(languages))) for (subject, speaker), languages in speakers.items())


def _audio_reference_definition(audio_label, audio_speakers=()):
    """Describe previous-chunk audio using MiniMax's official reference roles."""
    base = f"{audio_label} is the previous video's full audio reference for seamless continuation in this video"
    if not audio_speakers:
        return base + "."
    subjects = [f"{subject} ({speaker})" for subject, speaker, _languages in audio_speakers]
    languages = sorted({language for _subject, _speaker, values in audio_speakers for language in values})
    subject_text = ", ".join(subjects[:-1]) + (" and " if len(subjects) > 1 else "") + subjects[-1]
    language_text = ""
    if languages:
        layer = "layer" if len(languages) == 1 else "layers"
        language_text = f", containing spoken {' and '.join(languages)} vocal {layer}"
    return f"{base} and the voice-timbre reference for {subject_text}{language_text}."


def _audio_reference_retention(audio_label, audio_speakers=()):
    """Describe reference-only audio continuity without requesting signal copying."""
    if not audio_speakers:
        return f"{audio_label}: reference - its audible continuity guides the new audio without copying the original signal."
    subjects = [subject for subject, _speaker, _languages in audio_speakers]
    subject_text = ", ".join(subjects[:-1]) + (" and " if len(subjects) > 1 else "") + subjects[-1]
    timbre = "timbre guides" if len(subjects) == 1 else "timbres guide"
    return f"{audio_label}: reference - its audio continuity guides this video, and its vocal {timbre} the dialogue delivery of {subject_text} without copying the original signal."


def _video_continuation_prompt(prompt, video_label, audio_label=None, storyboard=False, audio_speakers=()):
    source_lines = []
    if video_label is not None:
        source_lines.append(f"{video_label} is the continuation source for this video.")
    if audio_label is not None:
        source_lines.append(_audio_reference_definition(audio_label, audio_speakers))
    if not source_lines:
        return prompt
    source_line = "\n".join(source_lines)
    subject = SUBJECT_DEFINITIONS_FIELD.search(prompt)
    if subject is not None:
        next_section = SUMMARY_FIELD.search(prompt, subject.end()) or RETENTION_FIELD.search(prompt, subject.end()) or _description_field(prompt, subject.end())
        insert_at = next_section.start() if next_section is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + "\n" + source_line + "\n\n" + prompt[insert_at:].lstrip()
    else:
        field = _description_field(prompt)
        insert_at = field.start() if field is not None else 0
        prompt = prompt[:insert_at] + f"subject_definitions:\n{source_line}\n\n" + prompt[insert_at:]

    if video_label is not None:
        summary = SUMMARY_FIELD.search(prompt)
        summary_text = f"[video continuation] Continue directly from the end of {video_label}."
        if summary is not None:
            existing = summary.group(2).strip()
            task = re.match(r"\[([^]]+)\]\s*(.*)", existing)
            if task is not None:
                types = [value.strip() for value in task.group(1).split("+")]
                if "video continuation" not in [value.lower() for value in types]:
                    types.insert(0, "video continuation")
                existing = f"[{' + '.join(types)}] {task.group(2).strip()}".rstrip()
                replacement = summary.group(1) + existing + f" Continue directly from the end of {video_label}."
            else:
                replacement = summary.group(1) + summary_text + (" " + existing if existing else "")
            prompt = prompt[:summary.start()] + replacement + prompt[summary.end():]
        else:
            retention = RETENTION_FIELD.search(prompt)
            field = _description_field(prompt)
            insert_at = retention.start() if retention is not None else field.start() if field is not None else len(prompt)
            prompt = prompt[:insert_at].rstrip() + f"\n\nsummary: {summary_text}\n\n" + prompt[insert_at:].lstrip()

    # The full previous audio is a real H3 reference, not Gemma's speculative
    # per-chunk character-state prose. Keep this one concise reference line
    # even while the experimental per-chunk retention analysis stays disabled.
    if audio_label is not None:
        audio_retention_line = _audio_reference_retention(audio_label, audio_speakers)
        retention = RETENTION_FIELD.search(prompt)
        if retention is not None:
            field = _description_field(prompt, retention.end())
            insert_at = field.start() if field is not None else len(prompt)
            prompt = prompt[:insert_at].rstrip() + "\n" + audio_retention_line + "\n\n" + prompt[insert_at:].lstrip()
        else:
            field = _description_field(prompt)
            insert_at = field.start() if field is not None else len(prompt)
            prompt = prompt[:insert_at].rstrip() + f"\n\nretention_analysis:\n{audio_retention_line}\n\n" + prompt[insert_at:].lstrip()

    if not INCLUDE_PER_CHUNK_RETENTION_ANALYSIS:
        return prompt

    # A continuation chunk can begin in the middle of a source shot.  Do not
    # mention ``[Shot 1]`` in a non-description field: H3 can still interpret
    # that token as a fresh-shot cue even when Gemma correctly begins its
    # detailed description with plain continuation prose.
    if video_label is not None:
        continuation_location = "the opening storyboard block" if storyboard else "the opening local continuation sequence"
        retention_line = f"{video_label} (appears in {continuation_location}): fully_preserved - its ending is used as the continuation starting point for this video."
        retention = RETENTION_FIELD.search(prompt)
        if retention is not None:
            field = _description_field(prompt, retention.end())
            insert_at = field.start() if field is not None else len(prompt)
            prompt = prompt[:insert_at].rstrip() + "\n" + retention_line + "\n\n" + prompt[insert_at:].lstrip()
        else:
            field = _description_field(prompt)
            insert_at = field.start() if field is not None else len(prompt)
            prompt = prompt[:insert_at].rstrip() + f"\n\nretention_analysis:\n{retention_line}\n\n" + prompt[insert_at:].lstrip()
    return prompt


def _parse_prompt_shots(prompt, total_frames, fps):
    field = _description_field(prompt)
    description_start = field.end() if field is not None else 0
    description_end_match = DESCRIPTION_END.search(prompt, description_start)
    description_end = description_end_match.start() if description_end_match is not None else len(prompt)
    markers = list(SHOT_MARKER.finditer(prompt, description_start, description_end))
    if not markers:
        return markers, [], description_end

    shot_starts = []
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != index + 1:
            raise ValueError("MiniMax shot numbers must start at 1 and increase sequentially")
        if marker.group(2) is None:
            if index:
                raise ValueError("MiniMax shot markers after the opening shot must use 'At MM:SS.mmm,'")
            shot_starts.append(0)
        else:
            if not index:
                raise ValueError("MiniMax [Shot 1] must not have a timestamp")
            shot_starts.append(_timestamp_frame(marker.group(2), marker.group(3), marker.group(4), fps))
    if any(right <= left for left, right in zip(shot_starts, shot_starts[1:])):
        raise ValueError("MiniMax shot timestamps must be strictly increasing")

    shots = []
    for index, marker in enumerate(markers):
        shot_end = shot_starts[index + 1] if index + 1 < len(markers) else total_frames
        segment_end = markers[index + 1].start() if index + 1 < len(markers) else description_end
        shots.append((index, shot_starts[index], shot_end, prompt[marker.end():segment_end]))
    return markers, shots, description_end


def _preview_shot_ranges(prompt, total_frames, preview_end, fps):
    _markers, shots, _description_end = _parse_prompt_shots(prompt, total_frames, fps)
    ranges = []
    for shot_index, shot_start, shot_end, _body in shots:
        if shot_start >= preview_end or shot_end <= 0:
            continue
        ranges.append({
            "shot": shot_index + 1,
            "start": max(0, shot_start),
            "end": min(preview_end, shot_end) - 1,
            "source_end": shot_end - 1,
        })
    return ranges


def _prompt_for_chunk(prompt, frame_start, frame_end, total_frames, fps, content_start=None, continuation=False,
                      drop_picture_anchors=False, continuation_video_label=None, continuation_audio_label=None,
                      has_opening_frames=True, body_overrides=None):
    """Build one canonical H3 prompt for a physical sampler chunk.

    Source cuts retain their global ``[Shot N]`` labels while ``At
    MM:SS.mmm,`` is recalculated on the physical chunk timeline. We
    deliberately do not give H3 our former
    master-range, timeslice, reference-range, or synthetic shot-end language.
    This remains the deterministic preview/fallback planner. During normal
    sampling, Gemma replaces the complete local description for every chunk.
    """
    content_start = frame_start if content_start is None else content_start
    if drop_picture_anchors:
        prompt = _drop_picture_anchors(prompt)
    markers, shots, description_end = _parse_prompt_shots(prompt, total_frames, fps)
    if not markers:
        if continuation_video_label is not None:
            return _video_continuation_prompt(prompt, continuation_video_label, continuation_audio_label)
        return prompt

    # Start from the physical window rather than only new output. If carried
    # opening frames end exactly at a source cut, include a compact preceding
    # block so the following canonical marker can place that real cut at the
    # correct local time without replaying the completed prior shot.
    selected = [shot for shot in shots if shot[1] < frame_end and shot[2] > frame_start]
    if not selected:
        raise ValueError(f"No prompt shots overlap sampled frames {frame_start} through {frame_end - 1}")

    rewritten = []
    for index, (shot_index, shot_start, shot_end, body) in enumerate(selected):
        # A ``[Shot N]`` marker is not generic chunk syntax. It is a real
        # source-shot label, so a physical window beginning inside that shot
        # starts with ordinary continuation prose. A later real cut retains
        # its global source-shot number and receives only a recalculated
        # chunk-local timecode. This planner is normally preview-only, but
        # keeping it consistent with Gemma prevents misleading fallback text.
        global_shot_number = shot_index + 1
        if index == 0:
            if shot_start == frame_start:
                marker_text = f"[Shot {global_shot_number}]"
            elif shot_start > frame_start:
                marker_text = (
                    f"[Shot {global_shot_number}] At "
                    f"{_frame_timestamp(shot_start - frame_start, fps)},"
                )
            else:
                marker_text = ""
        else:
            marker_text = (
                f"[Shot {global_shot_number}] At "
                f"{_frame_timestamp(shot_start - frame_start, fps)},"
            )

        if shot_end <= content_start:
            # This block represents only carried guide/reference frames from a
            # predecessor. Its source action is deliberately absent.
            body = " Preserve the supplied opening frames from this completed preceding shot; do not replay its action."
        else:
            override = None if body_overrides is None else body_overrides.get(shot_index)
            if override is not None:
                body = " " + override.strip()
            elif continuation and shot_start < content_start:
                opening = "supplied opening frames" if has_opening_frames else "established continuation source"
                body = (
                    f" Continue directly from the {opening}; do not restart or replay earlier actions. "
                    + body.lstrip()
                )
        rewritten.append((marker_text + " " if marker_text else "") + body.rstrip() + " ")
    rewritten_prompt = prompt[:markers[0].start()] + "".join(rewritten) + prompt[description_end:]
    if continuation_video_label is not None or continuation_audio_label is not None:
        rewritten_prompt = _video_continuation_prompt(
            rewritten_prompt,
            continuation_video_label,
            continuation_audio_label,
        )
    return rewritten_prompt


def _dialogue_language_code(language):
    """Map the H3 dialogue language label to espeak's language code."""
    return {"arabic": "ar", "chinese": "cmn", "english": "en-us", "french": "fr-fr", "german": "de", "italian": "it", "japanese": "ja", "korean": "ko", "portuguese": "pt-br", "russian": "ru", "spanish": "es"}.get(language.strip().lower(), "en-us")


@functools.lru_cache(maxsize=4096)
def _dialogue_phoneme_count(word, language):
    """Count espeak phones for one spoken word, preserving deterministic timing."""
    spoken = re.sub(r"(^[^\w']+|[^\w']+$)", "", word, flags=re.UNICODE)
    if not spoken:
        return 0
    phones = phonemize(spoken, language=_dialogue_language_code(language), backend="espeak", separator=Separator(phone=" ", word="|"), strip=True, preserve_punctuation=False)
    return max(1, len([phone for phone in phones.replace("|", " ").split() if phone]))


def _dialogue_word_weights(speech, language):
    """Return phoneme and punctuation weights for a word-exact utterance."""
    words = list(re.finditer(r"\S+", speech))
    if not words:
        return ()
    weights = []
    for word in words:
        token = word.group()
        if re.search(r"(?:\.{3,}|…+)", token):
            pause = ELLIPSIS_PAUSE_SECONDS
        elif re.search(r"[.!?][\"')]*$", token):
            pause = SENTENCE_PAUSE_SECONDS
        elif re.search(r"[,;:][\"')]*$", token):
            pause = COMMA_PAUSE_SECONDS
        else:
            pause = 0.0
        weights.append(WORD_ONSET_SECONDS + PHONEME_SECONDS * _dialogue_phoneme_count(token, language) + pause)
    return tuple(zip(words, weights))


def _dialogue_word_tokens(text):
    """Return the comparable words of one dialogue text, ignoring case and punctuation."""
    return re.findall(r"\w+(?:['’]\w+)?", str(text or "").casefold())


def _dialogue_prefix_tokens(block_words, spoken_words):
    """Return the block tokens carrying the first `spoken_words` comparable words."""
    tokens = []
    consumed = 0
    for word in block_words:
        tokens.append(word.group())
        consumed += len(_dialogue_word_tokens(word.group()))
        if consumed >= spoken_words:
            break
    return tokens


_DIALOGUE_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000,
}


def _dialogue_number_value(word):
    """Return the value of digits or a plain English number word, else None."""
    token = str(word).casefold().replace(",", "").replace("-", " ")
    if token.isdigit():
        return int(token)
    total = 0
    current = 0
    seen = False
    for part in token.split():
        value = _DIALOGUE_NUMBER_WORDS.get(part)
        if value is None:
            return None
        seen = True
        if value >= 100:
            # "hundred"/"thousand" scale what came before them.
            current = max(1, current) * value
            if value >= 1000:
                total += current
                current = 0
        else:
            current += value
    return total + current if seen else None


def _dialogue_same_word(script_word, heard_word):
    """Report whether a script word and a transcript word are the same word.

    The script's own spelling never decides a retry: a word that differs from
    what was heard by a single letter, or that is the same number written as
    digits rather than words, is the same word. Every larger difference still
    counts, so a missing, extra or genuinely different word stays visible.
    """
    if script_word == heard_word:
        return True
    number = _dialogue_number_value(script_word)
    if number is not None and number == _dialogue_number_value(heard_word):
        return True
    if min(len(script_word), len(heard_word)) < 4 or abs(len(script_word) - len(heard_word)) > 1:
        return False
    if len(script_word) == len(heard_word):
        # One wrong letter, such as the doubled consonant a script spells singly.
        return sum(1 for left, right in zip(script_word, heard_word) if left != right) == 1
    # One extra or missing letter, such as a plural the transcript dropped.
    longer, shorter = (script_word, heard_word) if len(script_word) > len(heard_word) else (heard_word, script_word)
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1:] == shorter:
            return True
    return False


def _dialogue_words_equal(script_words, heard_words):
    """Compare one dialogue word list with a recorded transcript word list."""
    return len(script_words) == len(heard_words) and all(_dialogue_same_word(script, heard) for script, heard in zip(script_words, heard_words))


def _teacher_take_matches_dialogue(dialogue_text, transcript):
    """Report whether a take spoke its dialogue, ignoring cosmetic differences."""
    return _dialogue_words_equal(_dialogue_word_tokens(dialogue_text), _dialogue_word_tokens(transcript))


def _dialogue_phrase_ids(speech):
    """Index every spoken word's phrase so a seam can never orphan one word of it."""
    phrases = []
    phrase = 0
    for word in re.finditer(r"\S+", speech):
        phrases.append(phrase)
        # Terminals and ellipses end a phrase exactly where the weight table
        # already places its pause punctuation.
        if re.search(r"(?:\.{3,}|…+|[.!?])[\"')]*$", word.group()):
            phrase += 1
    return tuple(phrases)


def _seam_orphans_a_phrase_word(phrase_ids, first_word, last_word, word_count):
    """Report whether a seam hands a single word of the phrase it cuts to either side."""
    if last_word + 1 >= word_count or phrase_ids[last_word] != phrase_ids[last_word + 1]:
        return False
    phrase = phrase_ids[last_word]
    left = last_word
    while left >= first_word and phrase_ids[left] == phrase:
        left -= 1
    right = last_word + 1
    while right < word_count and phrase_ids[right] == phrase:
        right += 1
    return last_word - left == 1 or right - last_word - 1 == 1


def _dialogue_owner_spans(weights, phrase_ids, range_seconds):
    """Give each range the words nearest its clock position, never a lone phrase word."""
    total_weight = sum(weights)
    total_seconds = sum(range_seconds)
    cumulative_weights = []
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
        cumulative_weights.append(weight_sum)
    spans = []
    first_word = 0
    elapsed_seconds = 0.0
    for range_index, duration in enumerate(range_seconds[:-1]):
        elapsed_seconds += duration
        desired_weight = total_weight * elapsed_seconds / total_seconds if total_seconds else total_weight
        remaining_ranges = len(range_seconds) - range_index - 1
        # Keep words contiguous while putting each weight boundary as close as
        # possible to the output-time boundary. Fewer words than ranges leaves
        # the trailing ranges silent rather than overflowing the word list.
        upper_bound = min(len(weights), max(first_word + 1, len(weights) - remaining_ranges))
        if first_word >= upper_bound:
            spans.append((first_word, first_word - 1))
            continue
        final_word = min(range(first_word, upper_bound), key=lambda word_index: abs(cumulative_weights[word_index] - desired_weight))
        # The chunk's own clock wins over sentence punctuation, but a phrase is
        # never cut so that one side of the seam holds a single word of it:
        # hand that word back to the earlier chunk instead, which keeps the
        # words out of the following chunks (the short tail cannot absorb them)
        # and stops the retries from starving the end of the dialogue. The last
        # range is the only one allowed to end empty because of this rule.
        while (
            _seam_orphans_a_phrase_word(phrase_ids, first_word, final_word, len(weights))
            and final_word + 1 <= len(weights) - remaining_ranges + 1
        ):
            final_word += 1
        spans.append((first_word, final_word))
        first_word = final_word + 1
    spans.append((first_word, len(weights) - 1))
    return tuple(spans)


def _last_dialogue_word_feather_ticks(prompt, maximum_ticks):
    """Return the final rendered dialogue word's estimated duration in audio ticks."""
    maximum_ticks = max(0, int(maximum_ticks))
    dialogues = list(DIALOGUE_BLOCK.finditer(str(prompt or "")))
    if not dialogues or not maximum_ticks:
        return 0
    # Use the final exact H3 dialogue block, including its punctuation pause,
    # rather than re-timing the source shot's complete dialogue.
    dialogue = dialogues[-1].group(1)
    language_match = re.match(r"\s*\[([^]]+)\]\s*(.*)", dialogue, re.DOTALL)
    if language_match is None:
        return 0
    weighted_words = _dialogue_word_weights(language_match.group(2), language_match.group(1))
    if not weighted_words:
        return 0
    last_word_seconds = weighted_words[-1][1]
    return min(maximum_ticks, max(1, int(round(last_word_seconds * AUDIO_LATENT_FPS))))


def _dialogue_word_schedule(speech, language, start_seconds, end_seconds):
    """Assign phoneme-weighted word intervals across one source dialogue span."""
    weighted_words = _dialogue_word_weights(speech, language)
    if not weighted_words:
        return ()
    available = max(0.0, end_seconds - start_seconds)
    total_weight = sum(weight for _word, weight in weighted_words)
    scale = available / total_weight if total_weight else 0.0
    cursor = start_seconds
    schedule = []
    for word, weight in weighted_words:
        next_cursor = cursor + weight * scale
        schedule.append((word, cursor, next_cursor))
        cursor = next_cursor
    return tuple(schedule)


def _legacy_dialogue_for_range(body, shot_start, shot_end, frame_start, frame_end, fps, dialogue_ranges=(), sampled_start=None):
    """Assign whole phoneme-timed words once on the global source-shot clock."""
    pattern = re.compile(r"(?P<header>(?:<Subject\s+\d+>|[\w'-]+)\s*\(S\d+\)\s*says\s*:\s*)<d>\s*(?P<language>\[[^]\n]+\])\s*(?P<speech>.*?)</d>", re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(body))
    if len(matches) != len(re.findall(r"<d>", body, re.IGNORECASE)) or len(matches) != len(re.findall(r"</d>", body, re.IGNORECASE)):
        raise ValueError("Legacy dialogue needs complete '<Subject N> (SN) says: <d>[Language] words</d>' blocks (a character name may replace <Subject N>).")
    if not matches:
        return body, []
    silence = re.search(
        r"(?i)\b(?:start(?:s)?\s+(?:the\s+)?video|begin(?:s)?)\s+in\s+silence\s+for\s+"
        r"(?P<clock>\d+(?::\d+)?(?:\.\d+)?)\s*(?:seconds?|s)?",
        body,
    )
    clock_text = "" if silence is None else silence.group("clock")
    clock_parts = clock_text.split(":")
    opening_silence = float(clock_parts[-1] or 0)
    if len(clock_parts) == 2:
        opening_silence += 60.0 * float(clock_parts[0] or 0)
    # ponytail: the utterance owns the source shot after an explicit opening
    # silence. Forced alignment against generated audio remains the upgrade.
    dialogue_end = (shot_end - shot_start) / fps
    source_dialogues = [(match, _dialogue_word_weights(match.group("speech"), match.group("language").strip("[]"))) for match in matches]
    total_weight = sum(weight for _match, weighted_words in source_dialogues for _word, weight in weighted_words)
    if not total_weight:
        return pattern.sub("", body), []
    available_seconds = max(0.0, dialogue_end - opening_silence)
    if total_weight > available_seconds:
        raise ValueError(f"Legacy dialogue phoneme estimate estimates {total_weight:.2f}s of speech, but the shot allows only {available_seconds:.2f}s after opening silence. Increase the shot duration, shorten its dialogue, or connect a pre-production director.")
    speech_start = shot_start + int(math.ceil(opening_silence * fps))
    owned_ranges = tuple((max(shot_start, speech_start, start), min(shot_end, end)) for start, end in dialogue_ranges if start < shot_end and end > speech_start)
    word_owners = {}
    word_start_frames = {}
    word_end_frames = {}
    if owned_ranges:
        all_words = [(match_index, word_index, weight) for match_index, (_match, weighted_words) in enumerate(source_dialogues) for word_index, (_word, weight) in enumerate(weighted_words)]
        if len(all_words) < len(owned_ranges):
            raise ValueError(f"Legacy dialogue has {len(all_words)} words for {len(owned_ranges)} output chunks. Add dialogue, use longer chunks, or shorten the shot so every dialogue chunk can contain a complete word.")
        # Keep words contiguous while putting each weight boundary as close as
        # possible to the output-time boundary. Every dialogue chunk receives a
        # whole word, and each local phoneme schedule finishes at its seam.
        # `_dialogue_owner_spans` owns the sentence rule as well, so this cut
        # and the retry re-cut below can never drift apart.
        range_seconds = [(end - start) / fps for start, end in owned_ranges]
        phrase_ids = []
        for _match, weighted_words in source_dialogues:
            phrase_ids.extend(_dialogue_phrase_ids(" ".join(word.group() for word, _weight in weighted_words)))
        spans = _dialogue_owner_spans([item[2] for item in all_words], tuple(phrase_ids), range_seconds)
        for range_index, (span_start, span_end) in enumerate(spans):
            for word_index in range(span_start, span_end + 1):
                word_owners[all_words[word_index][:2]] = range_index
        for range_index, (range_start, range_end) in enumerate(owned_ranges):
            owned_words = [item for item in all_words if word_owners[item[:2]] == range_index]
            if not owned_words:
                continue
            range_scale = range_seconds[range_index] / sum(item[2] for item in owned_words)
            range_cursor = range_start / fps
            for match_index, word_index, weight in owned_words:
                word_start_frames[(match_index, word_index)] = max(range_start, int(math.floor(range_cursor * fps)))
                range_cursor += weight * range_scale
                word_end_frames[(match_index, word_index)] = min(range_end, int(math.ceil(range_cursor * fps)))
    # Expand natural word and pause weights across the remaining planned shot;
    # this prevents the dialogue from finishing in early chunks and leaving a
    # silent tail simply because a long source shot was chunked.
    scale = available_seconds / total_weight
    cursor = opening_silence
    dialogue = []
    carries_boundary_words = False
    sampled_start = frame_start if sampled_start is None else int(sampled_start)
    current_range = (max(shot_start, speech_start, frame_start), min(shot_end, frame_end))
    current_range_index = owned_ranges.index(current_range) if current_range in owned_ranges else None
    for match_index, (match, weighted_words) in enumerate(source_dialogues):
        speech = match.group("speech")
        selected = []
        selected_index = None
        for word_index, (word, weight) in enumerate(weighted_words):
            cursor += weight * scale
            # A word is owned where its spoken sound finishes. This lets a word
            # cross a chunk's physical prefix instead of being cut at its tail.
            word_end_frame = shot_start + int(math.ceil(cursor * fps))
            if owned_ranges:
                # The physical opening prefix is sampled again and replaces
                # the preceding chunk's decoded tail. Give H3 the words that
                # begin there, followed by this chunk's retained words, so
                # its phoneme timing continues through the boundary.
                owns_output = current_range_index is not None and word_owners[(match_index, word_index)] == current_range_index and current_range[0] < word_end_frames[(match_index, word_index)] <= current_range[1]
                owns_prefix = current_range_index and word_owners[(match_index, word_index)] == current_range_index - 1 and sampled_start <= word_start_frames[(match_index, word_index)] < current_range[0]
                selected_here = owns_prefix or owns_output
                carries_boundary_words = carries_boundary_words or bool(owns_prefix)
            else:
                selected_here = max(frame_start, shot_start) < word_end_frame <= min(frame_end, shot_end)
            if selected_here:
                if selected_index is None:
                    selected_index = word_index
                selected.append(word)
        if selected:
            words = re.sub(r"\s+", " ", speech[selected[0].start():selected[-1].end()]).strip()
            # A slice that starts inside a phrase keeps its leading seam space,
            # so H3 does not glue it onto a new word at the boundary.
            phrases = _dialogue_phrase_ids(speech)
            carried_prefix = bool(selected_index) and phrases[selected_index] == phrases[selected_index - 1]
            dialogue.append(match.group("header") + "<d>" + match.group("language") + _dialogue_segment_text(words.split(), carried_prefix) + "</d>")
    if dialogue and carries_boundary_words:
        # The selected words are a source-clock slice, not a fresh utterance.
        # State this once outside <d> so H3 continues the prefix's phonemes.
        dialogue.insert(0, "The dialogue is already in progress at this video opening.")
    return pattern.sub("", body), dialogue


def _legacy_shot_body_for_range(body, shot_start, shot_end, frame_start, frame_end, fps=24.0, dialogue_ranges=(), sampled_start=None):
    """Pre-Gemma proportional word slicing, restored from 12a771b's parent."""
    if frame_start > shot_start:
        # Remove explicit opening-image setup BEFORE slicing can truncate it.
        # Protect spoken text even when it mentions a picture or first frame.
        parts = re.split(r"(<d>.*?</d>)", body, flags=re.IGNORECASE | re.DOTALL)
        opening = re.compile(r"(?:^\s*|(?<=[.!?])\s+)(?:starts?\s+with|begins?\s+with|use|using)\s+<Picture\s+\d+>\s+as\s+(?:the\s+)?(?:exact\s+)?(?:first|opening|initial)\s+frame\b[^.!?]*(?:[.!?]|$)\s*", re.IGNORECASE)
        body = "".join(part if index % 2 else opening.sub("", part) for index, part in enumerate(parts))
    visual_body, dialogue = _legacy_dialogue_for_range(body, shot_start, shot_end, frame_start, frame_end, fps, dialogue_ranges, sampled_start)
    if visual_body != body:
        if frame_start > shot_start:
            # Opening-only setup belongs only to the source shot's beginning.
            # Its former proportional visual slice could reintroduce a one-second
            # silence in later chunks even while their dialogue words progressed.
            visual_body = re.sub(
                r"(?is)(?:<Subject\s+\d+>|[\w'-]+)\s+start(?:s)?\s+(?:the\s+)?video\s+in\s+silence\s+for\s+"
                r"\d+(?::\d+)?(?:\.\d+)?\s*(?:seconds?|s)?\s*,?\s*(?:then\s*)?",
                "",
                visual_body,
            )
        visual = _legacy_shot_body_for_range(visual_body, shot_start, shot_end, frame_start, frame_end, fps) if visual_body.strip() else ""
        return " ".join([visual.strip()] + dialogue).strip() or "Continue the established shot without additional dialogue."
    if frame_start <= shot_start and frame_end >= shot_end:
        return body
    # ponytail: this estimates action timing by word count, not scene semantics;
    # connect a pre-production provider for visual, model-authored direction.
    units = []
    for sentence in re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+|\n\s*\n+", body):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Keep complete visual sentences: fixed word blocks can leave a
        # dangling camera instruction immediately before dialogue prose.
        units.append(sentence)
    weights = [max(1, len(re.findall(r"\w+", unit))) for unit in units]
    total_weight = sum(weights)
    start_weight = total_weight * (max(frame_start, shot_start) - shot_start) / (shot_end - shot_start)
    end_weight = total_weight * (min(frame_end, shot_end) - shot_start) / (shot_end - shot_start)
    selected = []
    offset = 0
    for unit, weight in zip(units, weights):
        if offset < end_weight and offset + weight > start_weight:
            selected.append(unit)
        offset += weight
    return " " + " ".join(selected) if selected else " Continue the established shot and its ongoing action."


def _legacy_opening_body_with_timed_dialogue(body, shot_start, shot_end, frame_start, frame_end, fps=24.0, dialogue_ranges=(), sampled_start=None):
    """Keep opening visual prose whole while assigning only its dialogue to time."""
    visual_body, dialogue = _legacy_dialogue_for_range(body, shot_start, shot_end, frame_start, frame_end, fps, dialogue_ranges, sampled_start)
    return " ".join([visual_body.strip()] + dialogue).strip()


def _legacy_static_camera_prompt(prompt, previous_prompt=None):
    """Reinforce a still camera only when both neighboring descriptions allow it."""
    def description_bounds(text):
        """Ignore reference metadata when checking camera instructions."""
        field = _description_field(text)
        start = field.end() if field is not None else 0
        end = DESCRIPTION_END.search(text, start)
        return start, end.start() if end is not None else len(text)

    # ponytail: explicit English camera-motion vocabulary, not semantic parsing.
    # Unknown/implicit movement needs a director; never infer it from dialogue.
    motion = re.compile(r"\b(?:pan(?:s|ning)?|tilt(?:s|ing)?|zoom(?:s|ing)?|dolly|dollies|dollying|truck(?:s|ing)?|orbit(?:s|ing)?|arc(?:s|ing)?|tracking\s+shot|handheld)\b|\bcamera\b[^.!?\n]{0,100}\b(?:move[sd]?|moving|movement|follow[s]?|track[s]?|shake[s]?|shaking|rotate[s]?|rotating|push(?:es)?|pull[s]?|crane[s]?)\b", re.IGNORECASE)
    for text in (prompt, previous_prompt or ""):
        start, end = description_bounds(text)
        description = re.sub(r"<d>.*?</d>", "", text[start:end], flags=re.IGNORECASE | re.DOTALL)
        description = re.sub(r"\b(?:no|without)\s+camera\s+movement\b", "", description, flags=re.IGNORECASE)
        if motion.search(description):
            return prompt
    start, _ = description_bounds(prompt)
    # The legacy source prompt often repeats this whole-shot instruction in
    # every sliced chunk. The local static-frame instruction below is the
    # single command H3 needs for a chunk without detected camera motion.
    description_end = DESCRIPTION_END.search(prompt, start)
    description_end = len(prompt) if description_end is None else description_end.start()
    # ponytail: explicit English setup clauses only; keep dialogue verbatim
    # and preserve action after a colon. Semantic camera direction needs a director.
    parts = re.split(r"(<d>.*?</d>)", prompt[start:description_end], flags=re.IGNORECASE | re.DOTALL)
    setup = re.compile(r"(?:^\s*|(?<=[.!?\]])\s+)(?:the\s+)?(?:camera\b|(?:first|opening|initial)\s+frame\b|(?:wide|medium|close[- ]?up|establishing)\s+shot\b)[^.!?:]*(?:[.!?:]|$)\s*", re.IGNORECASE)
    description = "".join(part if index % 2 else setup.sub("", part) for index, part in enumerate(parts))
    prompt = prompt[:start] + description + prompt[description_end:]
    # Preserve the required opening shot marker before its descriptive prose.
    marker = SHOT_MARKER.match(prompt, start + len(prompt[start:]) - len(prompt[start:].lstrip()))
    insert_at = marker.end() if marker is not None else start
    sentence = "The camera stays fixed in the stablished frame."
    if prompt[insert_at:].lstrip().startswith(sentence):
        return prompt
    return prompt[:insert_at] + " " + sentence + " " + prompt[insert_at:].lstrip()


def _planned_chunk_prompts(prompt, plan, active_plan, fps, guide_frames, video_continuation,
                           audio_continuation, ref2va, video_number, audio_number, legacy=False,
                           include_dialogue_prefix=True):
    total_frames = plan[-1]["frame_end"]
    guide_enabled = guide_frames > 0
    planned = []
    previous_bodies = {}
    source_shots = _parse_prompt_shots(prompt, total_frames, fps)[1] if legacy else ()
    dialogue_ranges = [(chunk["frame_start"] + chunk.get("output_trim_frames", 0), chunk["frame_end"]) for chunk in plan] if legacy else ()
    for index, chunk in enumerate(active_plan):
        continuation = index > 0
        content_start = chunk["frame_start"] + chunk.get("output_trim_frames", 0)
        continuation_video_label = f"<Video {video_number}>" if continuation and video_continuation else None
        continuation_audio_label = f"<Audio {audio_number}>" if continuation and audio_continuation else None
        body_overrides = None
        # The first H3 video establishes the shot from the user's complete
        # source description, but dialogue still follows the output clock.
        # Later videos also slice visual action for continuation.
        if legacy:
            body_overrides = {}
            for number, start, end, body in source_shots:
                if start >= chunk["frame_end"] or end <= content_start:
                    continue
                # Camera establishment belongs to the first chunk of EACH shot.
                opening = content_start <= start
                body_for_range = _legacy_opening_body_with_timed_dialogue if opening else _legacy_shot_body_for_range
                sampled_start = chunk["frame_start"] if include_dialogue_prefix else content_start
                local_body = body_for_range(body, start, end, content_start, chunk["frame_end"], fps, dialogue_ranges, sampled_start)
                if not opening:
                    prior_body = previous_bodies.get(number, "")
                    local_body = _legacy_static_camera_prompt(local_body, prior_body)
                body_overrides[number] = local_body
            previous_bodies = body_overrides
        chunk_prompt = _prompt_for_chunk(
            prompt,
            chunk["frame_start"],
            chunk["frame_end"],
            total_frames,
            fps,
            content_start=content_start,
            continuation=continuation,
            drop_picture_anchors=continuation and not ref2va,
            continuation_video_label=continuation_video_label,
            continuation_audio_label=continuation_audio_label,
            has_opening_frames=guide_enabled,
            body_overrides=body_overrides,
        )
        chunk_prompt = _preserve_global_prompt_sections(chunk_prompt, prompt)
        debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt)
        planned.append((chunk_prompt, debug_prompt))
    return planned


def _preview_ranges_for_plan(plan, keep_physical_prefix=False, show_replaced_tail=False):
    """Return sequential display ranges, optionally retaining packed prefixes."""
    ranges = []
    retained_prefix_offset = 0
    for index, chunk in enumerate(plan):
        trim_frames = max(0, int(chunk.get("output_trim_frames", 0)))
        normal_start = int(chunk["frame_start"]) + trim_frames
        frame_count = int(chunk["frame_end"]) - normal_start
        keep_prefix = bool(keep_physical_prefix and index and trim_frames)
        start = normal_start + retained_prefix_offset
        if keep_prefix:
            frame_count += trim_frames
        ranges.append({"chunk": index + 1, "start": start, "end": start + frame_count - 1})
        if keep_prefix:
            retained_prefix_offset += trim_frames
    if show_replaced_tail:
        for index in range(1, len(ranges)):
            ranges[index - 1]["replacement_overlap_frames"] = max(
                0,
                int(plan[index].get("output_trim_frames", 0)),
            )
    return ranges


def _preview_subtitle(prompt):
    """Extract plain dialogue text for a chunk's synchronized caption."""
    subtitles = []
    for match in DIALOGUE_BLOCK.finditer(str(prompt or "")):
        text = re.sub(r"^\s*\[[^]]+\]\s*", "", match.group(1)).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            subtitles.append(text)
    return " ".join(subtitles)


def _dialogue_segment_text(words, carried_prefix=False):
    """Format a local dialogue slice, keeping one space on each side of a split."""
    text = " ".join(words).strip()
    # A phrase split across chunks never glues the two halves together: the
    # slice that continues the previous chunk's phrase keeps one leading space,
    # and the slice whose phrase continues in the next chunk keeps one trailing
    # space. Only a real phrase terminal closes the slice without it.
    if carried_prefix:
        text = " " + text
    return text if re.search(r"(?:\.{3,}|…+|[.!?])[\"')\]]*$", text) else text + " "


def _audio_transcript_approved_tail_prompt(prompt, transcript):
    """Keep a take that spoke an exact case- and punctuation-insensitive dialogue prefix."""
    expected = _dialogue_word_tokens(_preview_subtitle(prompt))
    heard = _dialogue_word_tokens(transcript.get("text", ""))
    if not heard or len(heard) >= len(expected) or not _dialogue_words_equal(expected[:len(heard)], heard):
        return None
    parsed = _dialogue_block_words(prompt)
    if parsed is None:
        return None
    # Keep the seam space the plan already put after the language tag, so the
    # shortened take still reads as a continuation.
    return _replace_dialogue_block_words(prompt, _dialogue_prefix_tokens(parsed[3], len(heard)), parsed[1] != parsed[1].rstrip())


def _dialogue_block_words(prompt):
    """Return the one simple dialogue block's language, words and speaker marker."""
    dialogues = list(DIALOGUE_BLOCK.finditer(prompt))
    if len(dialogues) != 1:
        return None
    dialogue = dialogues[0]
    language = re.match(r"(\s*\[([^]]+)\]\s*)(.*)", dialogue.group(1), re.DOTALL)
    speakers = re.findall(r"\(S\d+\)", prompt[:dialogue.start()], re.IGNORECASE)
    words = list(re.finditer(r"\S+", language.group(3))) if language is not None else []
    if language is None or not speakers or not words:
        return None
    return dialogue, language.group(1), language.group(2), words, speakers[-1].casefold()


def _replace_dialogue_block_words(prompt, words, carried_prefix=False):
    """Replace one dialogue block's spoken words while retaining all prompt prose."""
    parsed = _dialogue_block_words(prompt)
    if parsed is None:
        return None
    dialogue, language_prefix, _language, _old_words, _speaker = parsed
    if not words:
        # A chunk that inherits no word stays silent: drop the whole
        # "<Subject N> (S1) says: <d>...</d>" clause rather than leaving H3 an
        # empty dialogue block to improvise in.
        clause = re.search(r"\s*(?:<Subject\s+\d+>|[\w'-]+)\s*\(S\d+\)\s*says\s*:\s*$", prompt[:dialogue.start()], re.IGNORECASE)
        start = clause.start() if clause is not None else dialogue.start()
        return prompt[:start] + prompt[dialogue.end():]
    replacement = language_prefix.rstrip() + _dialogue_segment_text(words, carried_prefix)
    return prompt[:dialogue.start(1)] + replacement + prompt[dialogue.end(1):]


def _transcript_is_certain(transcript):
    """Report whether Whisper's own word confidences let a take be judged at all."""
    words = transcript.get("words") or ()
    if not words:
        return True
    return sum(float(word.get("probability", 0)) >= 0.5 for word in words) >= max(1, len(words) // 2)


def _recut_dialogue_prompts(prompts, plan, fps, dialogue_index, first_index, kept_words=0, shrink_frames=0):
    """Re-author the leftover dialogue over the leftover chunks with the initial cut.

    `dialogue_index` supplies the leftover words and the speaker, `first_index`
    is the first chunk that receives a new slice, `kept_words` skips a prefix the
    take already spoke, and `shrink_frames` makes the initial cut believe that
    chunk's window is shorter, which is how a diverged take asks for fewer words.
    """
    if len(prompts) != len(plan) or not 0 <= dialogue_index < len(prompts) or not dialogue_index <= first_index <= len(prompts):
        return None
    parsed = _dialogue_block_words(prompts[dialogue_index])
    if parsed is None:
        return None
    _dialogue, _prefix, language, words, speaker = parsed
    leftover = [word.group() for word in words][kept_words:]
    chain = []
    for index in range(dialogue_index, len(prompts)):
        later = _dialogue_block_words(prompts[index])
        if later is None or later[2].casefold() != language.casefold() or later[4] != speaker:
            break
        if index > dialogue_index:
            leftover.extend(word.group() for word in later[3])
        if index >= first_index:
            chain.append(index)
    if not leftover:
        return None
    output = list(prompts)
    keeping_prefix = bool(kept_words) and first_index > dialogue_index
    if keeping_prefix:
        # A take that spoke a clean prefix keeps exactly that prefix as its own
        # dialogue, so the plan and the accepted audio stay in step. The seam
        # space the plan already put after the language tag is kept with it.
        output[dialogue_index] = _replace_dialogue_block_words(prompts[dialogue_index], _dialogue_prefix_tokens(words, kept_words), parsed[1] != parsed[1].rstrip())
        if output[dialogue_index] is None:
            return None
    if not chain:
        return output if keeping_prefix else None
    ranges = []
    for index in chain:
        chunk = plan[index]
        start = int(chunk["frame_start"]) + int(chunk.get("output_trim_frames", 0))
        end = int(chunk["frame_end"])
        if index == dialogue_index and shrink_frames:
            end = max(start + 1, end - int(shrink_frames))
        ranges.append((start, end))
    speech = " ".join(leftover)
    phrase_ids = _dialogue_phrase_ids(speech)
    spans = _dialogue_owner_spans(
        [weight for _word, weight in _dialogue_word_weights(speech, language)],
        phrase_ids,
        [(end - start) / float(fps) for start, end in ranges],
    )
    # A re-cut slice keeps its leading seam space whenever it starts inside a
    # phrase: the first one continues the chunk before it, and every later one
    # continues the slice before it.
    before = _dialogue_block_words(output[chain[0] - 1]) if chain[0] else None
    carried_first = bool(before and before[3]) and not re.search(r"(?:\.{3,}|…+|[.!?])[\"')]*$", before[3][-1].group())
    for position, index in enumerate(chain):
        span_start, span_end = spans[position]
        if position:
            carried_prefix = bool(span_start) and phrase_ids[span_start] == phrase_ids[span_start - 1]
        else:
            carried_prefix = carried_first
        replacement = _replace_dialogue_block_words(prompts[index], leftover[span_start:span_end + 1], carried_prefix)
        if replacement is None:
            return None
        output[index] = replacement
    return output


def _retry_take_prompts(prompts, plan, fps, index):
    """Re-cut a diverged chunk and its leftovers, shortening that chunk until its words change.

    Missing or wrong words usually mean H3 ran out of time for them, so the same
    cut is asked for a shorter chunk until it hands that chunk fewer words.
    """
    original = _preview_subtitle(prompts[index])
    chunk = plan[index]
    window_frames = int(chunk["frame_end"]) - int(chunk["frame_start"]) - int(chunk.get("output_trim_frames", 0))
    for shrink_frames in range(1, max(2, window_frames)):
        revised = _recut_dialogue_prompts(prompts, plan, fps, index, index, shrink_frames=shrink_frames)
        if revised is not None and _preview_subtitle(revised[index]) != original:
            return revised, shrink_frames
    return _recut_dialogue_prompts(prompts, plan, fps, index, index), 0


def _publish_revised_dialogue(planned_prompts, revised, first_index, preview_chunk_ranges, preview_execution):
    """Adopt re-authored dialogue in the plan, the captions and the live preview."""
    for prompt_index, revised_prompt in enumerate(revised):
        planned_prompts[prompt_index] = (revised_prompt, planned_prompts[prompt_index][1])
        preview_chunk_ranges[prompt_index]["h3_prompt"] = revised_prompt.strip()
        preview_chunk_ranges[prompt_index]["subtitle"] = _preview_subtitle(revised_prompt)
        if preview_execution is not None and prompt_index >= first_index:
            preview_execution.set_audio_prompt(prompt_index, revised_prompt.strip(), preview_chunk_ranges[prompt_index]["subtitle"])


def _reconcile_native_dialogue(previous_prompt, current_prompt, transcript):
    """Move only unmistakably spoken or missing boundary words between prompts."""
    if isinstance(transcript, dict):
        recognized_words = transcript.get("words", ())
        if not recognized_words or any(word.get("probability", 0) < 0.5 for word in recognized_words):
            return current_prompt
        transcript = str(transcript.get("text", ""))
    previous = list(DIALOGUE_BLOCK.finditer(previous_prompt))
    current = list(DIALOGUE_BLOCK.finditer(current_prompt))
    if len(previous) != 1 or len(current) != 1 or not transcript:
        return current_prompt
    previous_tag = re.match(r"\s*(\[[^]]+\])\s*(.*)", previous[0].group(1), re.DOTALL)
    current_tag = re.match(r"\s*(\[[^]]+\])\s*(.*)", current[0].group(1), re.DOTALL)
    previous_speakers = re.findall(r"\(S\d+\)", previous_prompt[:previous[0].start()], re.IGNORECASE)
    current_speakers = re.findall(r"\(S\d+\)", current_prompt[:current[0].start()], re.IGNORECASE)
    if previous_tag is None or current_tag is None or not previous_speakers or not current_speakers or previous_speakers[-1].lower() != current_speakers[-1].lower() or previous_tag.group(1).lower() != current_tag.group(1).lower():
        return current_prompt
    # ponytail: exact single-speaker, word matches only; an ASR substitution or
    # multiple speakers needs a richer aligner, not a guess. Case and punctuation
    # never decide anything: Whisper writes "." or "..." where the dialogue slice
    # ends in a space, and those tokens used to disable this correction entirely.
    normalize = lambda word: re.sub(r"[^\w']", "", word.casefold())
    previous_words = [word for word in previous_tag.group(2).split() if normalize(word)]
    current_words = [word for word in current_tag.group(2).split() if normalize(word)]
    previous_clean = [normalize(word) for word in previous_words]
    current_clean = [normalize(word) for word in current_words]
    heard_clean = _dialogue_word_tokens(transcript)
    if not previous_clean or not current_clean or not heard_clean:
        return current_prompt
    if _dialogue_words_equal(previous_clean[:len(heard_clean)], heard_clean) and len(heard_clean) < len(previous_clean):
        corrected = previous_words[len(heard_clean):] + current_words
    elif _dialogue_words_equal(previous_clean, heard_clean[:len(previous_clean)]) and _dialogue_words_equal(current_clean[:len(heard_clean) - len(previous_clean)], heard_clean[len(previous_clean):]) and len(heard_clean) > len(previous_clean):
        consumed = len(heard_clean) - len(previous_clean)
        if consumed >= len(current_words):
            return current_prompt
        corrected = current_words[consumed:]
    else:
        return current_prompt
    replacement = current_tag.group(1) + " " + " ".join(corrected)
    return current_prompt[:current[0].start(1)] + replacement + current_prompt[current[0].end(1):]


def _native_prompt_for_stage(planned_prompt, prompt_stage, previous_prompt=None, transcript=None):
    """Answer an audio or video request from the native dialogue plan."""
    if prompt_stage not in ("audio", "video"):
        raise ValueError("Native chunk prompt stage must be audio or video")
    # ponytail: the native planner deliberately uses identical AV wording;
    # stage-specific wording belongs in an attached pre-production provider.
    return _reconcile_native_dialogue(previous_prompt, planned_prompt, transcript) if previous_prompt and transcript else planned_prompt


def _debug_chunk_header(index, chunk, content_start):
    return (
        f"=== Chunk {index + 1}: sampled frames {chunk['frame_start']}-{chunk['frame_end'] - 1}; "
        f"output frames {content_start}-{chunk['frame_end'] - 1} ==="
    )


def _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report=None):
    report = "" if not gemma_report else f"\n\n{gemma_report}"
    return f"{_debug_chunk_header(index, chunk, content_start)}{report}\n{chunk_prompt}"


def _gemma_shot_records(shots, range_start, range_end, sampled_start, fps, target):
    selected = [shot for shot in shots if shot[1] < range_end and shot[2] > range_start]
    records = []
    for selected_index, (shot_index, shot_start, shot_end, body) in enumerate(selected):
        global_shot_number = shot_index + 1
        record = {
            "shot_number": global_shot_number,
            "shot_start": shot_start,
            "shot_end": shot_end,
            "source_body": body,
        }
        if target:
            target_start = max(range_start, shot_start)
            if selected_index == 0:
                # Never synthesize a marker merely because a new physical
                # chunk starts in the middle of a source shot. When a real cut
                # occurs after carried prefix frames, retain that next source
                # shot's global number and give only its timecode a chunk-local
                # value.
                # A real cut that falls entirely inside a discarded physical
                # packing prefix has already been established by the opening
                # Video1/keyframe context. Repeating it in H3 prose creates a
                # second cut in retained output. Marker ownership therefore
                # begins at the retained range, not merely sampled_start.
                if shot_start < range_start:
                    required_marker = None
                elif shot_start == sampled_start:
                    required_marker = f"[Shot {global_shot_number}]"
                elif shot_start > sampled_start:
                    required_marker = (
                        f"[Shot {global_shot_number}] At "
                        f"{_frame_timestamp(shot_start - sampled_start, fps)},"
                    )
                else:
                    required_marker = None
            else:
                required_marker = (
                    f"[Shot {global_shot_number}] At "
                    f"{_frame_timestamp(shot_start - sampled_start, fps)},"
                )
            record.update({
                "target_start": target_start,
                "target_end": min(range_end, shot_end),
                "required_marker": required_marker,
            })
        else:
            record.update({
                "covered_start": max(range_start, shot_start),
                "covered_end": min(range_end, shot_end),
            })
        records.append(record)
    return records


def _deterministic_dialogue_segments(body, shot_start, shot_end, chunks, fps):
    """Return exact phoneme-timed dialogue fragments owned by output chunks."""
    segments = []
    dialogue_ranges = [(item["output_start"], item["output_end"]) for item in chunks]
    for chunk_index, chunk in enumerate(chunks, 1):
        start = max(shot_start, int(chunk["output_start"]))
        end = min(shot_end, int(chunk["output_end"]))
        if start >= end:
            continue
        _visual, dialogue = _legacy_dialogue_for_range(body, shot_start, shot_end, start, end, fps, dialogue_ranges)
        for item in dialogue:
            if "<d>" not in item:
                continue
            segments.append({"chunk": chunk_index, "start_frame": start, "end_frame": end, "content": item})
    return segments


def _gemma_source_shot_records(shots, range_start, range_end):
    """Return complete source-shot facts for Gemma's preproduction pass."""
    return [
        {
            "shot_number": shot_index + 1,
            "shot_start": shot_start,
            "shot_end": shot_end,
            "source_body": body,
        }
        for shot_index, shot_start, shot_end, body in shots
        if shot_start < range_end and shot_end > range_start
    ]


def _gemma_preproduction_chunks(active_plan):
    """Use only final output ownership, never synthetic/trimmed source frames."""
    return [
        {
            "sampled_start": chunk["frame_start"],
            "sampled_end": chunk["frame_end"],
            "output_start": chunk["frame_start"] + chunk.get("output_trim_frames", 0),
            "output_end": chunk["frame_end"],
        }
        for chunk in active_plan
    ]


def _gemma_preproduction_request(prompt, shots, full_plan, fps, ref2va):
    """Build preproduction from the complete render, never a debug subset.

    ``debug_stop_chunk`` limits sampling and preview output, but it must not
    shorten Gemma's immutable production horizon. A truncated horizon makes
    the shot planner squeeze all dialogue/actions into the visible debug
    chunks and invent a silent remainder for the source shot.
    """
    source_shots = _gemma_source_shot_records(
        shots,
        full_plan[0]["frame_start"],
        full_plan[-1]["frame_end"],
    )
    chunks = _gemma_preproduction_chunks(full_plan)
    for shot in source_shots:
        shot["deterministic_dialogue_segments"] = _deterministic_dialogue_segments(
            shot["source_body"], shot["shot_start"], shot["shot_end"], chunks, fps,
        )
    return source_shots, {
        "chunk_count": len(full_plan),
        "fps": fps,
        "prompt_mode": "ref" if ref2va else "base",
        "source_shots": source_shots,
        "chunks": chunks,
        "original_prompt": prompt,
    }


def _gemma_conditioning_context(continuation, context_keyframes, guide_overlap, video_continuation,
                                video_label, audio_label, include_video1_reference=True,
                                video_continuation_method=VIDEO_CONTINUATION_METHOD_VIDEO1,
                                masked_av_overlap=False):
    if not continuation:
        return "First chunk: original image/reference conditioning only; there is no previous generated chunk."
    if video_continuation_method == VIDEO_CONTINUATION_METHOD_MASKED_AV and masked_av_overlap:
        return (
            f"a {video_continuation}-frame physical video/audio overlap copied from the previous completed "
            "chunk; its video is fixed, most audio is fixed, and the final audio ticks are feathered into "
            "new generation. This is opening continuity, not a Video/Audio reference and must not be named "
            "or described as a new shot"
        )
    if video_continuation_method == VIDEO_CONTINUATION_METHOD_MASKED_AV:
        return (
            f"a {video_continuation}-frame video/audio boundary copied from the previous completed chunk into the "
            "discarded temporal packing prefix; this is duplicated opening continuity, not a Video reference "
            "or a new shot; native video and audio keyframes provide the boundary without a separate Audio reference"
        )
    sources = []
    if context_keyframes:
        sources.append(
            f"native fixed video/audio opening keyframes covering {context_keyframes} completed frames"
        )
    if video_continuation:
        if include_video1_reference:
            sources.append(
                f"a bounded {video_continuation}-frame continuation reference as {video_label}"
            )
        if audio_label is not None:
            sources.append(
                f"the whole previous physical chunk's audio as {audio_label}"
            )
    if guide_overlap:
        sources.append(
            f"a {guide_overlap}-frame latent warm-start that is fully denoised and retained, not a fixed keyframe"
        )
    return "; ".join(sources) if sources else "No fixed opening frames or native Video/Audio continuation reference."


def _chunk_retention_analysis(retention_analysis):
    """Normalize Gemma's concise H3-facing entry-state value without adding prose."""
    return retention_analysis.strip() if isinstance(retention_analysis, str) else ""


def _reinforce_static_shot_grade(description, shots, content_start):
    """Carry a source shot's first-picture grade into its static continuation."""
    camera = "The camera remains static in the established framing."
    # ponytail: require an explicit picture/first-frame relationship in one
    # clause. Ambiguous prose is left alone rather than guessing an asset.
    source = next((body for _index, start, end, body in shots if start < content_start < end), None)
    if source is None or not re.search(r"(?i)camera\s+(?:stays|remains|is)\s+static", source):
        return description
    pictures = []
    for clause in re.split(r"[.!?;\n]", source):
        if re.search(r"(?i)\bfirst\s+frame\b", clause):
            pictures.extend(PICTURE_LABEL.findall(clause))
    pictures = list(dict.fromkeys(pictures))
    if len(pictures) != 1:
        return description
    # Only the opening continuing segment may inherit this shot's reference;
    # a later real cut has its own camera and lighting instructions.
    marker = SHOT_MARKER.search(description)
    end = marker.start() if marker is not None else len(description)
    opening = description[:end]
    if camera not in opening:
        return description
    reinforcement = f" Use the lighting and color grading from {pictures[0]}."
    # Compare letters and digits only: punctuation, spacing and case can
    # vary in Gemma's response without making this a new instruction.
    normalized_opening = "".join(character for character in opening.casefold() if character.isalnum())
    normalized_reinforcement = "".join(character for character in reinforcement.casefold() if character.isalnum())
    if re.search(re.escape(normalized_reinforcement) + r"(?!\d)", normalized_opening):
        return description
    return opening.replace(camera, camera + reinforcement, 1) + description[end:]


def _prompt_with_gemma_description(prompt, description, drop_picture_anchors=False,
                                   continuation_video_label=None, continuation_audio_label=None,
                                   retention_analysis="", audio_speakers=(), summary=None):
    if drop_picture_anchors:
        prompt = _drop_picture_anchors(prompt)
    field = _description_field(prompt)
    description_start = field.end() if field is not None else 0
    description_end_match = DESCRIPTION_END.search(prompt, description_start)
    description_end = description_end_match.start() if description_end_match is not None else len(prompt)
    if field is None:
        marker = SHOT_MARKER.search(prompt, description_start, description_end)
        if marker is None:
            raise ValueError("MiniMax prompt has no description field or [Shot 1] marker")
        replace_start = marker.start()
    else:
        replace_start = description_start
    rewritten = prompt[:replace_start] + " " + description.strip() + " " + prompt[description_end:]
    chunk_retention_analysis = _chunk_retention_analysis(retention_analysis)
    if INCLUDE_PER_CHUNK_RETENTION_ANALYSIS and chunk_retention_analysis:
        retention = RETENTION_FIELD.search(rewritten)
        description_field = _description_field(
            rewritten,
            retention.end() if retention is not None else 0,
        )
        insert_at = description_field.start() if description_field is not None else len(rewritten)
        if retention is not None:
            rewritten = rewritten[:insert_at].rstrip() + "\n" + chunk_retention_analysis + "\n\n" + rewritten[insert_at:].lstrip()
        else:
            rewritten = (
                rewritten[:insert_at].rstrip()
                + "\n\nretention_analysis:\n"
                + chunk_retention_analysis
                + "\n\n"
                + rewritten[insert_at:].lstrip()
            )
    if continuation_video_label is not None or continuation_audio_label is not None:
        rewritten = _video_continuation_prompt(
            rewritten,
            continuation_video_label,
            continuation_audio_label,
            audio_speakers=audio_speakers,
        )
    if summary is not None:
        # Replace the entire prior summary after reference definitions have
        # been inserted, keeping Gemma's current-chunk paragraph authoritative.
        summary_field = re.search(r"(?im)^\s*summary\s*:", rewritten)
        if summary_field is not None:
            next_field = re.search(r"(?im)^\s*(?:retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:", rewritten[summary_field.end():])
            end = summary_field.end() + next_field.start() if next_field else len(rewritten)
            rewritten = rewritten[:summary_field.start()].rstrip() + "\n\nsummary: " + summary.strip() + "\n\n" + rewritten[end:].lstrip()
        else:
            field = RETENTION_FIELD.search(rewritten) or _description_field(rewritten)
            insert_at = field.start() if field else len(rewritten)
            rewritten = rewritten[:insert_at].rstrip() + "\n\nsummary: " + summary.strip() + "\n\n" + rewritten[insert_at:].lstrip()
    return rewritten


def _chunk_summary_prompt(prompt, original_prompt, continuation, picture_label=None, video_label=None, audio_label=None, boundary_keyframe=False):
    """Keep the source opening summary only for chunk one; rebuild later roles."""
    section = re.compile(r"(?im)^[ \t]*summary[ \t]*:[\s\S]*?(?=^[ \t]*(?:retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music)[ \t]*:|\Z)")
    current = section.search(prompt)
    if not continuation:
        # Copy the complete original section verbatim, including multiline prose.
        original = section.search(original_prompt)
        replacement = original.group(0) if original else ""
    else:
        tasks, roles = [], []
        if video_label is not None:
            tasks.append("video continuation")
            roles.append(f"Continue directly from the end of {video_label}.")
        if audio_label is not None:
            tasks.append("audio reference")
            roles.append(f"Use {audio_label}, the previous video's audio, as the audio reference.")
        if picture_label is not None or boundary_keyframe:
            tasks.append("keyframe completion")
            roles.append(f"{picture_label} serves as first frame of target video." if picture_label is not None else "Continue from the supplied previous-video boundary keyframe.")
        replacement = "summary: " + ("[" + " + ".join(tasks) + "] " if tasks else "") + (" ".join(roles) or "Continue the current scene.") + "\n\n"
    if current is not None:
        return prompt[:current.start()] + replacement + prompt[current.end():]
    field = RETENTION_FIELD.search(prompt) or _description_field(prompt)
    start = field.start() if field else 0
    return prompt[:start] + replacement + prompt[start:]


def _first_frame_picture_prompt(prompt, picture_label):
    """Add the ordinary picture's first-frame role to the current summary."""
    field = re.search(r"(?im)^\s*summary\s*:", prompt)
    end_field = re.search(r"(?im)^\s*(?:retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:", prompt[field.end():] if field else prompt)
    start = field.end() if field else (end_field.start() if end_field else len(prompt))
    end = start + end_field.start() if field and end_field else (len(prompt) if field else start)
    summary = prompt[start:end].strip()
    # Keep the authored tasks and prose, adding only the requested reference contract.
    tasks = re.match(r"\[([^\]]*)\]", summary)
    if tasks and "keyframe completion" not in tasks.group(1).lower():
        summary = "[" + tasks.group(1) + " + keyframe completion]" + summary[tasks.end():]
    elif not tasks:
        summary = "[keyframe completion] " + summary
    sentence = f"{picture_label} serves as first frame of target video."
    if sentence not in summary:
        summary = summary.rstrip() + " " + sentence
    prefix = prompt[:field.start()] if field else prompt[:start]
    return prefix.rstrip() + "\n\nsummary: " + summary.strip() + "\n\n" + prompt[end:].lstrip()


def _last_frame_picture_reference(vae, frames, width, height):
    """Encode the final full decoded frame as a native image reference."""
    if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("Video1 first-frame picture requires decoded previous-chunk frames")
    # Preserve the full frame's float values when it already matches the canvas.
    image = frames[-1:, ..., :3] if tuple(frames.shape[1:3]) == (height, width) else _reference_image(frames[-1:], width, height)
    latent = vae.encode(image)
    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError("Video1 first-frame picture encode did not return an H3 image latent")
    return {"type": "image", "data": image}, {"kind": "image", "latent_h": image.shape[1] // 16, "latent_w": image.shape[2] // 16, "latent": latent}


def _gemma_report(chunk_number, result):
    report = (
        f"=== Gemma 4 chunk prompt director: Chunk {chunk_number} ===\n"
        f"confidence: {result.confidence}\n"
        f"progress summary: {result.analysis or 'none'}\n"
        f"timing plan: {result.timing_plan or 'none'}\n"
        f"Gemma-only end state: {result.end_state or 'none'}\n"
        f"Gemma proposed retention_analysis (not sent to H3): {result.retention_analysis or 'none'}\n"
        "Gemma-only last-seen character state:\n"
        f"{json.dumps(list(result.last_seen_character_state), ensure_ascii=False, indent=2)}\n"
        f"H3 detailed_description: {result.detailed_description}\n"
        f"Gemma JSON attempts:\n{_gemma_response_transcript(result)}"
    )
    if len(result.attempts) > 1:
        report += (
            f"\nGemma response repair/correction: H3 uses Gemma's attempt {len(result.attempts)} response."
        )
    if result.validation_warnings:
        report += "\nvalidation warnings:\n- " + "\n- ".join(result.validation_warnings)
    return report


def _gemma_response_transcript(result):
    """Keep every model JSON response, including a model-authored contract correction."""
    attempts = tuple(result.attempts)
    if not attempts:
        return result.raw_json
    sections = []
    for index, attempt in enumerate(attempts, 1):
        if attempt.correction_prompt:
            sections.append(
                "=== GEMMA CHUNK-CONTRACT CORRECTION REQUEST ===\n"
                + attempt.correction_prompt.rstrip()
            )
        section = f"=== GEMMA ATTEMPT {index}: {attempt.kind} ===\n{attempt.raw_json.rstrip()}"
        if attempt.validation_warnings:
            section += "\nvalidation findings:\n- " + "\n- ".join(attempt.validation_warnings)
        sections.append(section)
    return "\n\n".join(sections)


def _gemma_timing_plan_transcript(result):
    """Keep raw preproduction JSON and any full-model correction visible."""
    attempts = tuple(result.attempts)
    if not attempts:
        return result.raw_json
    sections = []
    for index, attempt in enumerate(attempts, 1):
        if attempt.correction_prompt:
            sections.append(
                "=== GEMMA TIMING-PLAN CORRECTION REQUEST ===\n"
                + attempt.correction_prompt.rstrip()
            )
        section = f"=== GEMMA TIMING-PLAN ATTEMPT {index}: {attempt.kind} ===\n{attempt.raw_json.rstrip()}"
        if attempt.validation_warnings:
            section += "\nvalidation findings:\n- " + "\n- ".join(attempt.validation_warnings)
        sections.append(section)
    return "\n\n".join(sections)


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _reference_image(image, width, height):
    source_height, source_width = image.shape[1:3]
    scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    target_width = max(CANVAS_MULTIPLE, round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(image, target_width, target_height, "disabled")


def _reference_video_canvas(width, height):
    """Match ComfyUI's native MiniMax H3 reference-video presentation.

    H3 reference videos use a 768-pixel nominal short edge with a 768x1344
    area cap, aligned to 32 pixels. Smaller sources are never enlarged. This
    keeps internally generated Video1 frames equivalent to videos supplied to
    the stock H3 Ref2VA conditioning node instead of applying an undocumented
    lower-resolution sampler-only path.
    """
    ratio = width / height
    if ratio >= 1.0:
        nominal_width = H3_REFERENCE_VIDEO_SHORT_EDGE * ratio
        nominal_height = H3_REFERENCE_VIDEO_SHORT_EDGE
    else:
        nominal_width = H3_REFERENCE_VIDEO_SHORT_EDGE
        nominal_height = H3_REFERENCE_VIDEO_SHORT_EDGE / ratio
    if nominal_width * nominal_height > H3_REFERENCE_VIDEO_MAX_PIXELS:
        scale = math.sqrt(H3_REFERENCE_VIDEO_MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    target_width = max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if width * height < target_width * target_height:
        target_width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return target_width, target_height


def _prompt_tokens(clip, prompt, images, positive, width, height, continuation, video_items=()):
    refs = positive[0].get("minimax_refs") if positive else None
    image_list = [] if images is None else [images[index:index + 1] for index in range(images.shape[0])]
    if refs:
        ref_items = []
        image_index = 0
        for ref in refs:
            kind = ref["kind"]
            if kind == "image":
                if image_index >= len(image_list):
                    raise ValueError("HR Endless Sampler needs every Ref2VA reference image in the images input")
                ref_items.append({"type": "image", "data": _reference_image(image_list[image_index], width, height)})
                image_index += 1
            elif kind == "audio":
                ref_items.append({"type": "audio"})
            elif kind in ("video", "video_audio"):
                raise ValueError("HR Endless Sampler cannot rebuild video Ref2VA conditioning from an images input")
        if image_index != len(image_list):
            raise ValueError("HR Endless Sampler received more images than the Ref2VA conditioning uses")
        ref_items.extend(video_items)
        return clip.tokenize(prompt, minimax_ref_items=ref_items)

    if video_items:
        raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")

    prompt_images = []
    for index, image in enumerate(() if continuation else image_list):
        prompt_images.append(_resize(image, width, height, "disabled" if index == 0 else "center"))
    return clip.tokenize(prompt, images=prompt_images)


def _encode_prompt(clip, prompt, images, positive, width, height, continuation, video_items=()):
    conditioning = clip.encode_from_tokens_scheduled(_prompt_tokens(clip, prompt, images, positive, width, height, continuation, video_items))
    if len(conditioning) != 1:
        raise ValueError("HR Endless Sampler expects one MiniMax H3 conditioning segment")
    return conditioning[0]


def _linear_ref2va_image_refs(refs, images, vae):
    """Re-encode Ref2VA image blocks from their original sRGB image inputs."""
    if vae is None:
        raise ValueError("linear_color_compute needs the video VAE to rebuild Ref2VA image references")
    image_list = [] if images is None else [images[index:index + 1] for index in range(images.shape[0])]
    rebuilt = []
    image_index = 0
    for ref in refs:
        updated = ref.copy()
        if ref.get("kind") == "image":
            if image_index >= len(image_list):
                raise ValueError("linear_color_compute needs every Ref2VA reference image in the images input")
            width = int(ref["latent_w"]) * 16
            height = int(ref["latent_h"]) * 16
            linear_image = _convert_image_transfer(image_list[image_index], _srgb_to_inverse_gamma_compute_rgb)
            updated["latent"] = vae.encode(_resize(linear_image, width, height, "disabled"))
            image_index += 1
        rebuilt.append(updated)
    if image_index != len(image_list):
        raise ValueError("linear_color_compute received more images than the Ref2VA conditioning uses")
    return rebuilt


def _chunk_plan(video_t, audio_t, chunk_frames, overlap_frames=5):
    max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
    max_chunk_t = _video_steps(max_chunk_frames)
    context_video_t = _bounded_video_steps(overlap_frames, max_chunk_frames, "physical_overlap")

    if video_t < MIN_VIDEO_STEPS or (video_t - MIN_VIDEO_STEPS) % 5:
        raise ValueError("HR Endless Sampler expects a MiniMax H3 video latent on the 17k+5 frame grid")

    total_frames = _pixel_frames(video_t)
    if audio_t != _audio_steps(total_frames):
        raise ValueError("HR Endless Sampler expects a MiniMax H3 audio latent matching the video duration")

    plan = []
    video_end = 0
    audio_end = 0
    output_frames = 0
    remaining = video_t
    while remaining:
        if not plan:
            chunk_t = min(max_chunk_t, remaining)
            video_start = 0
            new_video_t = chunk_t
            chunk_frame_count = _pixel_frames(chunk_t)
        else:
            new_video_t = min(max_chunk_t - context_video_t, remaining)
            chunk_t = new_video_t + context_video_t
            video_start = video_end - context_video_t
            chunk_frame_count = _pixel_frames(chunk_t)

        output_frames += chunk_frame_count if not plan else chunk_frame_count - overlap_frames
        next_audio_end = _audio_steps(output_frames)
        chunk_audio_t = _audio_steps(chunk_frame_count)
        new_audio_t = next_audio_end - audio_end
        context_audio_t = 0 if not plan else chunk_audio_t - new_audio_t
        audio_start = 0 if not plan else audio_end - context_audio_t

        plan.append({
            "video_start": video_start,
            "video_end": video_start + chunk_t,
            "audio_start": audio_start,
            "audio_end": next_audio_end,
            "context_video_t": 0 if not plan else context_video_t,
            "context_audio_t": context_audio_t,
            "output_trim_frames": 0 if not plan else overlap_frames,
            "frame_start": 0 if not plan else output_frames - chunk_frame_count,
            "frame_end": output_frames,
        })
        video_end += new_video_t
        audio_end = next_audio_end
        remaining -= new_video_t

    return plan


def _chunk_plan_without_overlap(video_t, audio_t, chunk_frames, overlap_frames=5):
    """Plan a zero/noise synthetic prefix of the requested H3-valid duration."""
    plan = _chunk_plan(video_t, audio_t, chunk_frames, overlap_frames)
    for index in range(1, len(plan)):
        chunk = plan[index].copy()
        chunk["video_start"] += chunk["context_video_t"]
        chunk["audio_start"] += chunk["context_audio_t"]
        chunk["synthetic_prefix"] = True
        plan[index] = chunk
    return plan


def _full_noise_prefix(video_noise, audio_noise, video_start, audio_start, video_count, audio_count):
    """Reuse the preceding full-sequence noise tokens for a packed prefix."""
    if video_count <= 0 or audio_count <= 0 or video_start < video_count or audio_start < audio_count:
        raise ValueError("Synthetic prefix requires enough preceding full-sequence AV noise")
    if video_start > video_noise.shape[2] or audio_start > audio_noise.shape[-1]:
        raise ValueError("Synthetic prefix noise starts outside the full sequence")
    return video_noise[:, :, video_start - video_count:video_start], audio_noise[..., audio_start - audio_count:audio_start]


def _per_chunk_noise_seed(seed, chunk_index):
    """Return the requested one-based seed multiple for an independent chunk."""
    return (int(seed) * (int(chunk_index) + 1)) & 0xffffffffffffffff


def _apply_audio_context_feather(audio_mask, context_audio_t,
                                 feather_ticks=MASKED_AV_AUDIO_FEATHER_TICKS):
    """Protect an audio prefix and release its final ticks with a half cosine.

    MiniMax/ComfyUI denoise masks use 0 for preserved latent values and 1 for
    fully generated values. The final feather value is exactly 1, so the new
    audio owns the seam endpoint instead of meeting it with a hard latent cut.
    """
    context_audio_t = max(0, min(int(context_audio_t), int(audio_mask.shape[-1])))
    feather_ticks = max(0, min(int(feather_ticks), context_audio_t))
    hard_ticks = context_audio_t - feather_ticks
    if hard_ticks:
        audio_mask[..., :hard_ticks] = 0.0
    if feather_ticks:
        positions = torch.arange(
            1,
            feather_ticks + 1,
            device=audio_mask.device,
            dtype=audio_mask.dtype,
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * positions / float(feather_ticks))
        shape = [1] * audio_mask.ndim
        shape[-1] = feather_ticks
        audio_mask[..., hard_ticks:context_audio_t] = ramp.view(*shape)
    return audio_mask


def _native_audio_boundary_target(chunk_video, chunk_audio, previous_audio, context_audio_t, feather_ticks=None, previous_video=None):
    """Seed AV prefixes and release them over the same final-word interval."""
    context_audio_t = int(context_audio_t)
    if context_audio_t <= 0 or context_audio_t >= chunk_audio.shape[-1]:
        raise ValueError("Native audio boundary must be smaller than the sampled audio chunk")
    if previous_audio is None or previous_audio.shape[-1] < context_audio_t:
        raise ValueError("Previous chunk is too short for the native audio boundary")

    target_audio = chunk_audio.clone()
    target_audio[..., :context_audio_t] = previous_audio[..., -context_audio_t:].to(
        device=target_audio.device,
        dtype=target_audio.dtype,
    )
    video_mask = torch.ones(
        (chunk_video.shape[0], 1, chunk_video.shape[2], chunk_video.shape[3], chunk_video.shape[4]),
        device=chunk_video.device,
        dtype=torch.float32,
    )
    audio_mask = torch.ones(
        (target_audio.shape[0], 1, target_audio.shape[2], target_audio.shape[3]),
        device=target_audio.device,
        dtype=torch.float32,
    )
    feather_ticks = context_audio_t if feather_ticks is None else max(0, min(int(feather_ticks), context_audio_t))
    hard_ticks = context_audio_t - feather_ticks
    # Keep the copied audio exact until the final spoken word. A linear ramp then
    # hands that word's end to H3 while retaining a fully generated next tick.
    if hard_ticks:
        audio_mask[..., :hard_ticks] = 0.0
    if feather_ticks:
        positions = torch.linspace(0.0, 1.0, feather_ticks, device=audio_mask.device, dtype=audio_mask.dtype)
        audio_mask[..., hard_ticks:context_audio_t] = positions.view(1, 1, 1, feather_ticks)
    if previous_video is not None:
        video_t = previous_video.shape[2]
        if video_t <= 0 or video_t >= chunk_video.shape[2] or previous_video.shape[:2] + previous_video.shape[3:] != chunk_video.shape[:2] + chunk_video.shape[3:]:
            raise ValueError("Native video boundary must match the target and leave a generated suffix")
        chunk_video = chunk_video.clone()
        chunk_video[:, :, :video_t] = previous_video.to(device=chunk_video.device, dtype=chunk_video.dtype)
        # Use temporal frame positions, not evenly spaced token indices: H3
        # tokens cover 1,4,4,4,4 frames. Align the final token with audio's end.
        frame_ends = torch.tensor([_pixel_frames(i + 1) - 1 for i in range(video_t)], device=video_mask.device, dtype=torch.float32)
        positions = frame_ends / max(float(frame_ends[-1]), 1.0) * (context_audio_t - 1)
        if feather_ticks > 1:
            weights = ((positions - hard_ticks) / (feather_ticks - 1)).clamp(0, 1)
        else:
            weights = torch.zeros_like(positions)
        video_mask[:, :, :video_t] = weights.view(1, 1, video_t, 1, 1)
        return chunk_video, target_audio, comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return target_audio, comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def _masked_av_overlap_target(chunk_video, chunk_audio, previous_video, previous_audio,
                              context_video_t, context_audio_t,
                              feather_ticks=MASKED_AV_AUDIO_FEATHER_TICKS):
    """Copy the completed AV tail into a physical target prefix and mask it."""
    context_video_t = int(context_video_t)
    context_audio_t = int(context_audio_t)
    if context_video_t <= 0 or context_audio_t <= 0:
        raise ValueError("Masked AV continuation requires a non-empty physical video/audio overlap")
    if context_video_t >= chunk_video.shape[2] or context_audio_t >= chunk_audio.shape[-1]:
        raise ValueError("Masked AV continuation overlap must be smaller than the sampled chunk")
    if previous_video is None or previous_video.shape[2] < context_video_t:
        raise ValueError("Previous chunk is too short for the requested masked video overlap")
    if previous_audio is None or previous_audio.shape[-1] < context_audio_t:
        raise ValueError("Previous chunk is too short for the requested masked audio overlap")

    target_video = chunk_video.clone()
    target_audio = chunk_audio.clone()
    target_video[:, :, :context_video_t] = previous_video[:, :, -context_video_t:].to(
        device=target_video.device,
        dtype=target_video.dtype,
    )
    target_audio[..., :context_audio_t] = previous_audio[..., -context_audio_t:].to(
        device=target_audio.device,
        dtype=target_audio.dtype,
    )

    video_mask = torch.ones(
        (
            target_video.shape[0],
            1,
            target_video.shape[2],
            target_video.shape[3],
            target_video.shape[4],
        ),
        device=target_video.device,
        dtype=torch.float32,
    )
    audio_mask = torch.ones(
        (target_audio.shape[0], 1, target_audio.shape[2], target_audio.shape[3]),
        device=target_audio.device,
        dtype=torch.float32,
    )
    video_mask[:, :, :context_video_t] = 0.0
    _apply_audio_context_feather(audio_mask, context_audio_t, feather_ticks)
    return target_video, target_audio, comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def _replace_masked_av_video_prefix(chunk_video, replacement_video, context_video_t):
    """Replace a frozen masked-AV prefix with a shape-identical VAE re-encode."""
    context_video_t = int(context_video_t)
    expected_shape = (chunk_video.shape[0], chunk_video.shape[1], context_video_t,
                      chunk_video.shape[3], chunk_video.shape[4])
    if tuple(replacement_video.shape) != tuple(expected_shape):
        raise ValueError(
            "Color-corrected masked-AV VAE re-encode has shape %s; expected %s"
            % (tuple(replacement_video.shape), tuple(expected_shape))
        )
    updated = chunk_video.clone()
    updated[:, :, :context_video_t] = replacement_video.to(
        device=updated.device,
        dtype=updated.dtype,
    )
    return updated


def _replace_decoded_frame_tail(parts, replacement):
    """Replace final decoded frames across chunk parts without changing duration."""
    remaining = int(replacement.shape[0])
    if remaining <= 0:
        return
    available = sum(int(part.shape[0]) for part in parts)
    if available < remaining:
        raise ValueError(
            f"Cannot replace {remaining} decoded frames; only {available} finalized frames are available"
        )
    source_end = remaining
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        take = min(int(part.shape[0]), remaining)
        source_start = source_end - take
        replacement_slice = replacement[source_start:source_end].to(device=part.device, dtype=part.dtype)
        parts[index] = replacement_slice.clone() if take == int(part.shape[0]) else torch.cat((part[:-take], replacement_slice), dim=0)
        remaining -= take
        source_end = source_start
        if remaining == 0:
            break


def _replace_stream_tail(parts, replacement):
    """Replace an accumulated stream tail without changing its total length."""
    remaining = int(replacement.shape[-1])
    if remaining <= 0:
        return
    available = sum(int(part.shape[-1]) for part in parts)
    if available < remaining:
        raise ValueError(
            f"Cannot replace {remaining} accumulated audio ticks; only {available} are available"
        )
    source_end = remaining
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        take = min(int(part.shape[-1]), remaining)
        source_start = source_end - take
        replacement_slice = replacement[..., source_start:source_end].to(
            device=part.device,
            dtype=part.dtype,
        )
        if take == int(part.shape[-1]):
            parts[index] = replacement_slice.clone()
        else:
            parts[index] = torch.cat((part[..., :-take], replacement_slice), dim=-1)
        remaining -= take
        source_end = source_start
        if remaining == 0:
            break


def _video_continuation_boundary_guide(previous_video, chunk, context_keyframes, use_video_continuation):
    if not use_video_continuation or context_keyframes:
        return None, 0
    if not chunk.get("synthetic_prefix"):
        raise ValueError("Native video boundary keyframe needs a discarded synthetic packing prefix")
    guide_t = _video_steps(chunk["output_trim_frames"])
    if previous_video.shape[2] < guide_t:
        raise ValueError("Previous chunk is too short for the requested native video boundary keyframe")
    return previous_video[:, :, -guide_t:].clone(), 0


def _srgb_to_linear_rgb(images):
    """Return float32 linear RGB while preserving out-of-display-range values."""
    rgb = images[..., :3].to(dtype=torch.float32)
    # ``where`` evaluates both branches. Clamp only the unused power operand
    # so negative pixels do not create NaNs; the selected linear branch keeps
    # their original values without clipping them.
    positive = ((rgb.clamp_min(-0.055) + 0.055) / 1.055).pow(2.4)
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        positive,
    )


def _linear_rgb_to_srgb(linear):
    """Convert float32 linear RGB to sRGB without clipping float values."""
    # See _srgb_to_linear_rgb: protect only the unselected fractional-power
    # operand. The selected low branch retains negative linear RGB values and
    # the high branch preserves highlights above one for EXR output.
    positive = linear.clamp_min(0.0).pow(1.0 / 2.4)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * positive - 0.055,
    )


def _srgb_to_inverse_gamma_compute_rgb(images):
    """Apply the requested float32 sRGB^(1/2.4) compute transfer."""
    return images.to(dtype=torch.float32).clamp_min(0.0).pow(1.0 / 2.4)


def _inverse_gamma_compute_to_srgb(rgb):
    """Restore display sRGB values from inverse-gamma compute RGB."""
    return rgb.to(dtype=torch.float32).clamp_min(0.0).pow(2.4)


def _convert_image_transfer(images, converter):
    """Apply an RGB transfer curve while retaining any non-RGB image channels."""
    if images is None:
        return None
    converted = images.to(dtype=torch.float32).clone()
    converted[..., :3] = converter(converted[..., :3])
    return converted


def _bounded_overlap_color_sample(frames):
    """Bound adaptive boundary analysis without changing temporal alignment."""
    if frames.ndim != 4 or frames.shape[-1] < 3 or not frames.shape[0]:
        raise ValueError("Overlap color correction requires non-empty [T,H,W,C] frames")
    frame_count = int(frames.shape[0])
    sample_count = min(frame_count, COLOR_CORRECTION_MAX_FRAMES)
    indices = torch.linspace(0, frame_count - 1, steps=sample_count, device=frames.device).round().long()
    sample = frames.index_select(0, indices)
    height, width = int(sample.shape[1]), int(sample.shape[2])
    scale = min(1.0, COLOR_CORRECTION_MAX_SIDE / max(height, width))
    if scale < 1.0:
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        sample = F.interpolate(
            sample[..., :3].permute(0, 3, 1, 2).to(dtype=torch.float32),
            size=(resized_height, resized_width),
            mode="area",
        ).permute(0, 2, 3, 1)
    return sample


def _display_color_statistics(frames):
    """Return display-referred seam statistics without clipping float pixels."""
    sample = _bounded_overlap_color_sample(frames)[..., :3]
    luma = (sample * sample.new_tensor((0.2126, 0.7152, 0.0722))).sum(dim=-1)
    quantiles = torch.quantile(
        luma.reshape(-1),
        luma.new_tensor((0.05, 0.50, 0.95)),
    )
    return {
        "rgb_mean": sample.mean(dim=(0, 1, 2)),
        "luma_mean": luma.mean(),
        "luma_median": quantiles[1],
        "luma_p05": quantiles[0],
        "luma_p95": quantiles[2],
        "black_clip": (luma <= (1.0 / 255.0)).to(torch.float32).mean(),
    }


def _serializable_color_statistics(statistics):
    """Convert small tensor seam statistics into log-safe Python values."""
    return {
        "rgb_mean": [float(value.detach().cpu()) for value in statistics["rgb_mean"]],
        "luma_mean": float(statistics["luma_mean"].detach().cpu()),
        "luma_median": float(statistics["luma_median"].detach().cpu()),
        "luma_p05": float(statistics["luma_p05"].detach().cpu()),
        "luma_p95": float(statistics["luma_p95"].detach().cpu()),
        "black_clip": float(statistics["black_clip"].detach().cpu()),
    }


def _monotonic_tone_points(statistics):
    """Build monotonic float anchors without limiting the signal to SDR range."""
    low = float(statistics["luma_p05"].detach().cpu())
    middle = float(statistics["luma_median"].detach().cpu())
    high = float(statistics["luma_p95"].detach().cpu())
    span = max(0.0001, high - low)
    points = [low - span, low, middle, high, high + span]
    result = [points[0]]
    for value in points[1:]:
        result.append(max(result[-1] + 0.0001, value))
    return result


def _apply_display_tone_curve(images, source_points, target_points):
    """Map float luma through a monotonic curve while preserving RGB hue."""
    rgb = images[..., :3].to(dtype=torch.float32)
    luma = (rgb * rgb.new_tensor((0.2126, 0.7152, 0.0722))).sum(dim=-1)
    source = rgb.new_tensor(source_points)
    target = rgb.new_tensor(target_points)
    # `bucketize` selects the containing two-point segment, including values
    # at either end. The strictly monotonic points above prevent zero division.
    segment = torch.bucketize(luma.contiguous(), source[1:-1]).clamp(0, len(source_points) - 2)
    source_lo = source[segment]
    source_hi = source[segment + 1]
    target_lo = target[segment]
    target_hi = target[segment + 1]
    # Deliberately allow extrapolation beyond the sampled anchors: decoded H3
    # floats may contain valid negative detail or highlights above one.
    fraction = (luma - source_lo) / (source_hi - source_lo).clamp_min(1e-6)
    mapped_luma = target_lo + (target_hi - target_lo) * fraction
    safe_luma = torch.where(
        luma.abs() < 1e-4,
        torch.where(luma < 0.0, luma.new_full((), -1e-4), luma.new_full((), 1e-4)),
        luma,
    )
    ratio = (mapped_luma / safe_luma).clamp(
        COLOR_CORRECTION_MIN_TONE_RATIO,
        COLOR_CORRECTION_MAX_TONE_RATIO,
    )
    return rgb * ratio.unsqueeze(-1)


def _fixed_output_color_transform(reference_frames, generated_frames, whole_range=False):
    """Fit a seam or shot-wide float tone transform without clipping pixels."""
    reference_source = reference_frames if whole_range else reference_frames[-1:]
    generated_source = generated_frames if whole_range else generated_frames[:1]
    reference = _display_color_statistics(reference_source)
    generated = _display_color_statistics(generated_source)
    source_points = _monotonic_tone_points(generated)
    target_points = _monotonic_tone_points(reference)
    source = _bounded_overlap_color_sample(generated_source)
    curved_rgb = _apply_display_tone_curve(source, source_points, target_points)
    channel_balance = (reference["rgb_mean"] / curved_rgb.mean(dim=(0, 1, 2)).clamp_min(1e-5)).clamp(
        COLOR_CORRECTION_MIN_RGB_BALANCE,
        COLOR_CORRECTION_MAX_RGB_BALANCE,
    )
    return {
        "source_points": source_points,
        "target_points": target_points,
        "rgb_balance": [float(value.detach().cpu()) for value in channel_balance],
        "reference": _serializable_color_statistics(reference),
        "raw": _serializable_color_statistics(generated),
    }


def _apply_fixed_output_color_transform(images, transform):
    """Apply fixed output grading without clipping non-display float pixels."""
    corrected = images.clone()
    rgb = _apply_display_tone_curve(
        corrected,
        transform["source_points"],
        transform["target_points"],
    )
    balance = rgb.new_tensor(transform["rgb_balance"]).reshape(1, 1, 1, 3)
    corrected[..., :3] = (rgb * balance).to(dtype=corrected.dtype)
    return corrected


def _correct_decoded_chunk_color(previous_finalized_frames, decoded_chunk_frames, overlap_frames,
                                 correction_frames=None, enabled=True):
    """Apply ColorMatchV2's output-only MKL transfer to one decoded chunk."""
    if not enabled:
        return decoded_chunk_frames, None, 0
    overlap = min(max(0, int(overlap_frames)), int(decoded_chunk_frames.shape[0]))
    generated_available = int(decoded_chunk_frames.shape[0]) - overlap
    correction_count = generated_available if correction_frames is None else min(
        generated_available, max(0, int(correction_frames)))
    if previous_finalized_frames is None or correction_count <= 0:
        return decoded_chunk_frames, None, 0
    if not int(previous_finalized_frames.shape[0]):
        return decoded_chunk_frames, None, 0

    # Match ColorMatchV2 exactly: each target frame uses its own ColorMatcher
    # MKL transfer against the last finalized frame before this chunk.
    from color_matcher import ColorMatcher
    reference = previous_finalized_frames[-1].detach().to(device="cpu", dtype=torch.float32).numpy()
    corrected = decoded_chunk_frames.clone()
    failures = 0
    for frame_index in range(correction_count):
        target_index = overlap + frame_index
        target = decoded_chunk_frames[target_index].detach().to(device="cpu", dtype=torch.float32).numpy()
        try:
            matched = ColorMatcher().transfer(src=target, ref=reference, method="mkl")
        except Exception as error:
            # ColorMatchV2 keeps the original target when a frame cannot be matched.
            logging.warning("HR Endless Sampler MKL color match skipped frame %d: %s", target_index, error)
            failures += 1
            continue
        corrected[target_index] = torch.from_numpy(matched).to(dtype=corrected.dtype).clamp_(0, 1)

    raw_stats = _serializable_color_statistics(_display_color_statistics(decoded_chunk_frames[overlap:overlap + 1]))
    corrected_stats = _serializable_color_statistics(_display_color_statistics(corrected[overlap:overlap + 1]))
    reference_stats = _serializable_color_statistics(_display_color_statistics(previous_finalized_frames[-1:]))
    return corrected, {
        "method": "mkl",
        "reference": reference_stats,
        "raw": raw_stats,
        "corrected": corrected_stats,
        "failures": failures,
        "residual": {
            "rgb_mean": [corrected_value - reference_value for corrected_value, reference_value in zip(corrected_stats["rgb_mean"], reference_stats["rgb_mean"])],
            "luma_mean": corrected_stats["luma_mean"] - reference_stats["luma_mean"],
            "luma_median": corrected_stats["luma_median"] - reference_stats["luma_median"],
            "black_clip": corrected_stats["black_clip"] - reference_stats["black_clip"],
        },
    }, correction_count


def _final_shot_color_correction(images, source_shots, timing_plan, chunk_ranges):
    """Match every stable-lighting shot frame to that shot's first frame.

    The ordinary per-chunk seam correction remains responsible for exact
    adjacent-boundary continuity. This second pass uses that same MKL transfer
    over the completed shot. When Gemma supplies a lighting decision, only a
    declared lighting change suppresses it.
    """
    if images is None or not source_shots or not chunk_ranges:
        return images, []
    if images.ndim != 4 or not int(images.shape[0]):
        return images, []

    lighting = {} if timing_plan is None else {
        int(shot.source_shot): bool(shot.light_change) for shot in timing_plan.shots
    }
    corrected = images.clone()
    reports = []
    frame_count = int(images.shape[0])
    for shot_index, shot_start, shot_end, _body in source_shots:
        source_shot = int(shot_index) + 1
        start = max(0, int(shot_start))
        end = min(frame_count, int(shot_end))
        if start >= end:
            continue
        if lighting.get(source_shot, timing_plan is not None):
            reports.append((source_shot, "skipped: source prompt permits lighting change"))
            continue

        anchor = images[start:start + 1]
        corrected_ranges = 0
        for chunk in chunk_ranges:
            chunk_start = max(start, int(chunk.get("start", start)))
            chunk_end = min(end, int(chunk.get("end", end - 1)) + 1)
            if chunk_start >= chunk_end:
                continue
            # Preserve the source shot's first frame exactly; every other
            # frame, including the remainder of its first chunk, is graded.
            chunk_start = max(chunk_start, start + 1)
            if chunk_start >= chunk_end:
                continue
            # Use the same ColorMatchV2 MKL path as the chunk decode stage.
            matched, _transform, matched_count = _correct_decoded_chunk_color(
                anchor, images[chunk_start:chunk_end], 0, enabled=True,
            )
            if matched_count:
                corrected[chunk_start:chunk_end] = matched
                corrected_ranges += 1
        reports.append((
            source_shot,
            "applied to %d chunk portion(s) after the first frame" % corrected_ranges
            if anchor is not None else "skipped: no rendered anchor",
        ))
    return corrected, reports


def _correct_ready_entire_shot_frames(frames, frame_start, source_shots, timing_plan, anchors):
    """Grade one ready display chunk from each source shot's fixed first frame."""
    if frames is None or frames.ndim != 4 or not int(frames.shape[0]) or not source_shots:
        return frames
    lighting = {} if timing_plan is None else {
        int(shot.source_shot): bool(shot.light_change) for shot in timing_plan.shots
    }
    corrected = frames.clone()
    frame_start = int(frame_start)
    for shot_index, shot_start, shot_end, _body in source_shots:
        source_shot = int(shot_index) + 1
        shot_start = int(shot_start)
        shot_end = int(shot_end)
        local_start = max(0, shot_start - frame_start)
        local_end = min(int(frames.shape[0]), shot_end - frame_start)
        if local_start >= local_end or lighting.get(source_shot, timing_plan is not None):
            continue
        anchor = anchors.get(source_shot)
        if anchor is None:
            anchor = frames[local_start:local_start + 1].clone()
            anchors[source_shot] = anchor
            local_start += 1
        if local_start >= local_end:
            continue
        matched, _transform, _matched_count = _correct_decoded_chunk_color(
            anchor, frames[local_start:local_end], 0, enabled=True,
        )
        corrected[local_start:local_end] = matched
    return corrected


def _log_output_color_correction(chunk_label, transform, corrected_frame_count):
    """Log a compact, comparable record of one output-only seam correction."""
    if transform is None:
        logging.info(
            "HR Endless Sampler %s output color correction: skipped (%d same-shot frames).",
            chunk_label,
            corrected_frame_count,
        )
        return
    reference = transform["reference"]
    raw = transform["raw"]
    corrected = transform["corrected"]
    residual = transform["residual"]
    if transform.get("method") == "mkl":
        logging.info(
            "HR Endless Sampler %s output color correction: ColorMatchV2 MKL applied to %d same-shot frames (%d skipped); "
            "first-frame luma %.5f -> %.5f, reference %.5f.",
            chunk_label, corrected_frame_count, transform["failures"], raw["luma_mean"],
            corrected["luma_mean"], reference["luma_mean"],
        )
        return
    source_points = ", ".join("%.4f" % value for value in transform["source_points"])
    target_points = ", ".join("%.4f" % value for value in transform["target_points"])
    balance = ", ".join("%.4f" % value for value in transform["rgb_balance"])
    raw_rgb = ", ".join("%.5f" % value for value in raw["rgb_mean"])
    corrected_rgb = ", ".join("%.5f" % value for value in corrected["rgb_mean"])
    reference_rgb = ", ".join("%.5f" % value for value in reference["rgb_mean"])
    residual_rgb = ", ".join("%+.5f" % value for value in residual["rgb_mean"])
    logging.info(
        "HR Endless Sampler %s output color correction: fixed display tone curve (%s) -> (%s), "
        "RGB balance (%s), %d same-shot frames; first-frame RGB raw (%s) -> corrected (%s), "
        "reference (%s), residual (%s).",
        chunk_label,
        source_points,
        target_points,
        balance,
        corrected_frame_count,
        raw_rgb,
        corrected_rgb,
        reference_rgb,
        residual_rgb,
    )
    logging.info(
        "HR Endless Sampler %s output color correction luma: raw mean/median/black %.5f/%.5f/%.3f%% -> "
        "corrected %.5f/%.5f/%.3f%%, reference %.5f/%.5f/%.3f%%; residual %+.5f/%+.5f/%+.3f%%.",
        chunk_label,
        raw["luma_mean"], raw["luma_median"], raw["black_clip"] * 100.0,
        corrected["luma_mean"], corrected["luma_median"], corrected["black_clip"] * 100.0,
        reference["luma_mean"], reference["luma_median"], reference["black_clip"] * 100.0,
        residual["luma_mean"], residual["luma_median"], residual["black_clip"] * 100.0,
    )


def _decoded_frame_tail(parts, frame_count):
    """Collect only the requested CPU-frame tail without concatenating the render."""
    remaining = max(0, int(frame_count))
    selected = []
    for part in reversed(parts):
        if remaining <= 0:
            break
        take = min(remaining, int(part.shape[0]))
        if take:
            selected.append(part[-take:])
            remaining -= take
    if not selected:
        return None
    selected.reverse()
    return selected[0] if len(selected) == 1 else torch.cat(selected, dim=0)


def _decoded_frame_before_tail(parts, tail_frames):
    """Return the final retained decoded frame before a tail replacement."""
    tail_frames = max(0, int(tail_frames))
    available = sum(int(part.shape[0]) for part in parts)
    if available <= tail_frames:
        return None
    return _decoded_frame_tail(parts, tail_frames + 1)[:1]


def _masked_av_boundary_guides(vae, previous_decoded_frames, overlap_frames):
    """Anchor every image of the short keyframe-only boundary.

    MiniMax accepts independent one-frame guides at arbitrary pixel positions.
    Encoding every boundary image separately avoids the temporal VAE's clip
    encoding, which can reinterpret motion or brightness between frames.
    """
    overlap_frames = int(overlap_frames)
    if overlap_frames <= 0:
        raise ValueError("Masked AV boundary keyframes need a positive overlap")
    if previous_decoded_frames is None or int(previous_decoded_frames.shape[0]) < overlap_frames:
        raise ValueError(
            "Masked AV boundary keyframes need the complete decoded previous-chunk overlap"
        )
    positions = range(overlap_frames)
    tail_start = int(previous_decoded_frames.shape[0]) - overlap_frames
    guides = []
    for position in positions:
        frame = previous_decoded_frames[tail_start + position:tail_start + position + 1].to(
            dtype=torch.float32,
        )
        guides.append({
            "resolved_frame_index": position,
            "latent": vae.encode(frame),
        })
    return tuple(guides)


def _h3_context_frames(raw_frames, corrected_frames):
    """Choose the prior decoded frames for the experimental H3 color A/B test."""
    if USE_COLOR_CORRECTED_H3_CONTEXT and corrected_frames is not None:
        if raw_frames is None or tuple(raw_frames.shape) == tuple(corrected_frames.shape):
            return corrected_frames, "color-corrected"
        logging.warning(
            "HR Endless Sampler color-corrected H3 context shape %s differs from raw shape %s; using raw frames.",
            tuple(corrected_frames.shape),
            tuple(raw_frames.shape),
        )
    return raw_frames, "raw"


def _conditioning_for_chunk(original_conds, frame_start, frame_end, encoded_prompt, video_context=None,
                            audio_context=None, audio_end_frame=5.0, video_refs=(), video_context_start=0,
                            video_contexts=(), replacement_refs=None):
    conds = {name: [item.copy() for item in values] for name, values in original_conds.items()}
    positive = conds.get("positive")
    if positive is None:
        raise ValueError("HR Endless Sampler requires a standard guider with positive conditioning")

    # H3's value is the clean fraction: 1.0 disables visual condition noise.
    # Apply to both CFG branches without mutating the upstream guider.
    for values in conds.values():
        for cond in values:
            cond["minimax_visual_cond_noise_aug"] = minimax_visual_cond_noise_aug
    cross_attn, prompt_metadata = encoded_prompt
    for cond in positive:
        cond["cross_attn"] = cross_attn
        token_tags = prompt_metadata.get("minimax_token_tags")
        if token_tags is not None:
            cond["minimax_token_tags"] = token_tags
        else:
            cond.pop("minimax_token_tags", None)
        if replacement_refs is not None:
            cond["minimax_refs"] = [*replacement_refs, *video_refs]
        elif video_refs:
            cond["minimax_refs"] = [*cond.get("minimax_refs", ()), *video_refs]
        keyframes = []
        for keyframe in cond.get("minimax_keyframes", ()):
            position = keyframe["resolved_frame_index"]
            if frame_start <= position < frame_end:
                local_keyframe = keyframe.copy()
                local_keyframe["resolved_frame_index"] = position - frame_start
                keyframes.append(local_keyframe)

        if video_context is not None:
            keyframes.append({"resolved_frame_index": video_context_start, "latent": video_context})
        for video_keyframe in video_contexts:
            keyframes.append(video_keyframe.copy())
        if audio_context is not None:
            audio_start = audio_end_frame - audio_context.shape[-1] / FRAME_RESCALE
            keyframes.append({"resolved_frame_index": audio_start, "audio_latent": audio_context})
        if keyframes:
            cond["minimax_keyframes"] = keyframes
        else:
            cond.pop("minimax_keyframes", None)
    return conds


def _decode_video_frames(vae, latent):
    """Decode MiniMax pixels without clipping float detail before final output."""
    stage = getattr(vae, "first_stage_model", None)
    original_finalize = getattr(stage, "_finalize_pixels", None)
    can_preserve_float_range = (
        callable(original_finalize)
        and hasattr(stage, "pixel_mean")
        and hasattr(stage, "pixel_std")
    )
    if can_preserve_float_range:
        def finalize_without_clamp(part):
            """Apply H3 pixel normalization while retaining HDR/negative values."""
            return part * stage.pixel_std.to(device=part.device, dtype=torch.float32) + stage.pixel_mean.to(device=part.device, dtype=torch.float32)

        # ComfyUI's H3 VAE otherwise clamps inside _finalize_pixels, before
        # the sampler can perform its float-space seam or final shot grading.
        stage._finalize_pixels = finalize_without_clamp
    try:
        frames = vae.decode(latent)
    finally:
        if can_preserve_float_range:
            stage._finalize_pixels = original_finalize
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if not frames.shape[0]:
        raise ValueError("MiniMax H3 video VAE decoded no frames")
    return frames


def _enhance_decoded_audio(waveform, sample_rate, overlap=None, *, enabled=False, device="cuda:0", seed=42):
    """Enhance a preview copy after original decoded audio has been accumulated."""
    if not enabled or waveform is None or sample_rate is None:
        return waveform, sample_rate, overlap
    # AudioSR owns its GPU only for this pass; original H3 latents stay untouched.
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)
    original = torch.cat((overlap, waveform), dim=-1) if overlap is not None else waveform
    enhanced = fix_audio({"waveform": original, "sample_rate": int(sample_rate)}, device=str(device), seed=seed, interrupt_check=comfy.model_management.throw_exception_if_processing_interrupted)
    split = round(overlap.shape[-1] * enhanced["sample_rate"] / sample_rate) if overlap is not None else 0
    return enhanced["waveform"][..., split:], enhanced["sample_rate"], enhanced["waveform"][..., :split] if overlap is not None else None


def _decode_audio_preview(audio_vae, latent, *, trim_latent_steps=0, output_frames=None, fps=VIDEO_FPS, normalize=True):
    """Decode and trim one H3 chunk using ComfyUI's ordinary audio normalization."""
    waveform = audio_vae.decode(latent).movedim(-1, 1)
    if normalize:
        deviation = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
        deviation[deviation < 1.0] = 1.0
        waveform = waveform / deviation
    sample_rate = int(getattr(
        audio_vae,
        "audio_sample_rate_output",
        getattr(audio_vae, "audio_sample_rate", 44100),
    ))
    trim_samples = max(0, round(int(trim_latent_steps) * sample_rate / AUDIO_LATENT_FPS))
    waveform = waveform[..., trim_samples:]
    if output_frames is not None:
        expected_samples = max(0, round(int(output_frames) * sample_rate / float(fps)))
        waveform = waveform[..., :expected_samples]
    return waveform.detach().to(device="cpu", dtype=torch.float32), sample_rate


def _connected_save_video_prefix(dynprompt, sampler_node_id):
    """Find the literal filename prefix on a Save Video fed by this timeline output."""
    if dynprompt is None or sampler_node_id is None:
        return None
    try:
        graph = dynprompt.get_original_prompt()
    except AttributeError:
        graph = getattr(dynprompt, "original_prompt", None)
    if not isinstance(graph, dict):
        return None
    sampler_node_id = str(sampler_node_id)

    def literal_string(value):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, (list, tuple)) or len(value) < 1:
            return None
        source = graph.get(str(value[0]))
        if not isinstance(source, dict):
            return None
        inputs = source.get("inputs")
        if not isinstance(inputs, dict):
            return None
        for name in ("value", "text", "string", "filename_prefix"):
            candidate = inputs.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    matches = []
    for node_id, node in graph.items():
        if not isinstance(node, dict) or node.get("class_type") != "HREndlessSamplerSaveVideo":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        timeline = inputs.get("timeline")
        if not isinstance(timeline, (list, tuple)) or len(timeline) < 2:
            continue
        try:
            is_our_timeline = str(timeline[0]) == sampler_node_id and int(timeline[1]) == 3
        except (TypeError, ValueError):
            is_our_timeline = False
        if not is_our_timeline:
            continue
        prefix = literal_string(inputs.get("filename_prefix"))
        if prefix:
            matches.append((str(node_id), prefix))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    unique_prefixes = list(dict.fromkeys(prefix for _node_id, prefix in matches))
    if len(unique_prefixes) > 1:
        logging.warning(
            "HR Endless Sampler timeline is connected to multiple Save Video prefixes; "
            "temporary chunk videos will use %s only.",
            unique_prefixes[0],
        )
    return unique_prefixes[0]


def _decode_replay_preview_media(vae, audio_vae, sampled_video, sampled_audio, *,
                                 output_trim_frames, context_audio_t, output_frames,
                                 fps=VIDEO_FPS, masked_audio_overlap_frames=0,
                                 replace_audio_tail_ticks=False,
                                 keep_masked_av_prefix=False):
    """Decode one cached physical chunk exactly like live finalization.

    Replay checkpoints store authoritative video/audio latents plus a lossy
    browser proxy. Resume output must therefore run the real VAEs again; proxy
    pixels are never allowed into sampling, color correction, or IMAGE output.
    """
    decoded_frames = _decode_video_frames(vae, sampled_video).detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    retained_start = 0 if keep_masked_av_prefix else max(
        0,
        min(int(output_trim_frames), int(decoded_frames.shape[0])),
    )
    retained_count = max(0, int(output_frames)) + (
        max(0, int(output_trim_frames)) if keep_masked_av_prefix else 0
    )
    retained_frames = decoded_frames[retained_start:retained_start + retained_count]
    if retained_count and int(retained_frames.shape[0]) != retained_count:
        raise ValueError(
            f"cached video decoded {int(decoded_frames.shape[0])} frames, but "
            f"{retained_count} retained frames were required after trimming {retained_start}"
        )

    audio_waveform = None
    audio_sample_rate = None
    overlap_waveform = None
    if audio_vae is not None and sampled_audio is not None:
        overlap_frames = max(0, int(masked_audio_overlap_frames))
        if replace_audio_tail_ticks:
            full_waveform, audio_sample_rate = _decode_audio_preview(
                audio_vae, sampled_audio, trim_latent_steps=0, output_frames=None, fps=fps,
            )
            overlap_samples = max(0, round(int(context_audio_t) * audio_sample_rate / AUDIO_LATENT_FPS))
            retained_samples = max(0, round(retained_count * audio_sample_rate / float(fps)))
            overlap_waveform = full_waveform[..., :overlap_samples]
            audio_waveform = full_waveform[..., overlap_samples:overlap_samples + retained_samples]
        elif overlap_frames:
            full_waveform, audio_sample_rate = _decode_audio_preview(
                audio_vae,
                sampled_audio,
                trim_latent_steps=0,
                output_frames=overlap_frames + retained_count,
                fps=fps,
            )
            overlap_samples = max(0, round(overlap_frames * audio_sample_rate / float(fps)))
            retained_samples = max(0, round(retained_count * audio_sample_rate / float(fps)))
            overlap_waveform = full_waveform[..., :overlap_samples]
            audio_waveform = full_waveform[..., overlap_samples:overlap_samples + retained_samples]
        else:
            audio_waveform, audio_sample_rate = _decode_audio_preview(
                audio_vae,
                sampled_audio,
                trim_latent_steps=context_audio_t,
                output_frames=retained_count,
                fps=fps,
            )
    return decoded_frames, retained_frames, audio_waveform, audio_sample_rate, overlap_waveform


def _bounded_color_sample(frames):
    """Bound diagnostic work while retaining the first and final frames."""
    if frames.ndim != 4 or frames.shape[-1] < 3 or not frames.shape[0]:
        raise ValueError("Color diagnostics require a non-empty [T,H,W,C] image tensor")
    frame_count = int(frames.shape[0])
    sample_count = min(frame_count, COLOR_DIAGNOSTIC_MAX_FRAMES)
    indices = torch.linspace(
        0,
        frame_count - 1,
        steps=sample_count,
        device=frames.device,
    ).round().to(dtype=torch.long).unique(sorted=True)
    sample = frames.index_select(0, indices)[..., :3].detach().to(dtype=torch.float32)
    height, width = int(sample.shape[1]), int(sample.shape[2])
    longest = max(height, width)
    if longest > COLOR_DIAGNOSTIC_MAX_SIDE:
        scale = COLOR_DIAGNOSTIC_MAX_SIDE / longest
        target_height = max(1, round(height * scale))
        target_width = max(1, round(width * scale))
        sample = F.interpolate(
            sample.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="area",
        ).permute(0, 2, 3, 1)
    return sample


def _color_sample_statistics(sample):
    """Measure display-referred RGB without pretending it is scene-linear."""
    rgb_mean = sample.mean(dim=(0, 1, 2))
    luma = (
        sample[..., 0] * 0.2126
        + sample[..., 1] * 0.7152
        + sample[..., 2] * 0.0722
    )
    channel_max = sample.amax(dim=-1)
    channel_min = sample.amin(dim=-1)
    saturation = torch.where(
        channel_max > 1e-6,
        (channel_max - channel_min) / channel_max,
        torch.zeros_like(channel_max),
    )
    quantiles = torch.quantile(
        luma.reshape(-1),
        torch.tensor((0.05, 0.50, 0.95), device=luma.device, dtype=luma.dtype),
    )
    # This fixed SDR diagnostic histogram may omit out-of-range samples, but
    # must never modify the decoded tensor just to make the graph fit [0, 1].
    histogram = torch.histc(luma, bins=32, min=0.0, max=1.0)
    histogram = histogram / histogram.sum().clamp_min(1.0)
    p05, p50, p95 = (float(value) for value in quantiles.detach().cpu())
    tonal_span = p95 - p05
    return {
        "rgb_mean": [float(value) for value in rgb_mean.detach().cpu()],
        "luma_mean": float(luma.mean().detach().cpu()),
        "luma_std": float(luma.std(unbiased=False).detach().cpu()),
        "luma_p05": p05,
        "luma_p50": p50,
        "luma_p95": p95,
        "midtone_balance": (p50 - p05) / tonal_span if tonal_span > 1e-6 else 0.5,
        "saturation_mean": float(saturation.mean().detach().cpu()),
        "black_clip_percent": float((luma <= (1.0 / 255.0)).to(torch.float32).mean().mul(100).detach().cpu()),
        "white_clip_percent": float((luma >= (254.0 / 255.0)).to(torch.float32).mean().mul(100).detach().cpu()),
        "luma_histogram": [float(value) for value in histogram.detach().cpu()],
    }


def _video_color_diagnostics(frames):
    """Return bounded whole-chunk and exact-boundary display-color statistics."""
    sample = _bounded_color_sample(frames)
    result = _color_sample_statistics(sample)
    result.update({
        "frame_count": int(frames.shape[0]),
        "sampled_frame_count": int(sample.shape[0]),
        "first_frame": _color_sample_statistics(sample[:1]),
        "last_frame": _color_sample_statistics(sample[-1:]),
    })
    return result


def _safe_video_color_diagnostics(frames, chunk_number=None):
    """Keep optional diagnostics from ever interrupting a render."""
    try:
        return _video_color_diagnostics(frames)
    except Exception as error:
        label = "" if chunk_number is None else f" for Chunk {chunk_number}"
        logging.warning(
            "HR Endless Sampler could not calculate color diagnostics%s; sampling will continue: %s",
            label,
            error,
        )
        return None


def _boundary_color_diagnostics(previous, current):
    """Compare adjacent decoded boundary frames using color and tone statistics."""
    previous_frame = previous["last_frame"]
    current_frame = current["first_frame"]
    return {
        "rgb_mean_delta": [
            current_value - previous_value
            for previous_value, current_value in zip(
                previous_frame["rgb_mean"], current_frame["rgb_mean"]
            )
        ],
        "luma_mean_delta": current_frame["luma_mean"] - previous_frame["luma_mean"],
        "luma_std_delta": current_frame["luma_std"] - previous_frame["luma_std"],
        "saturation_mean_delta": (
            current_frame["saturation_mean"] - previous_frame["saturation_mean"]
        ),
        "midtone_balance_delta": (
            current_frame["midtone_balance"] - previous_frame["midtone_balance"]
        ),
        "luma_histogram_distance": 0.5 * sum(
            abs(current_value - previous_value)
            for previous_value, current_value in zip(
                previous_frame["luma_histogram"], current_frame["luma_histogram"]
            )
        ),
    }


def _format_color_diagnostics(chunk_number, diagnostics):
    rgb = ", ".join(f"{value:.4f}" for value in diagnostics["rgb_mean"])
    return (
        f"HR Endless Sampler color diagnostics Chunk {chunk_number}: "
        f"RGB mean=({rgb}); luma mean={diagnostics['luma_mean']:.4f}; "
        f"contrast sigma={diagnostics['luma_std']:.4f}; "
        f"luma p05/p50/p95={diagnostics['luma_p05']:.4f}/"
        f"{diagnostics['luma_p50']:.4f}/{diagnostics['luma_p95']:.4f}; "
        f"midtone balance={diagnostics['midtone_balance']:.4f}; "
        f"saturation={diagnostics['saturation_mean']:.4f}; "
        f"black/white clip={diagnostics['black_clip_percent']:.3f}%/"
        f"{diagnostics['white_clip_percent']:.3f}% "
        f"({diagnostics['sampled_frame_count']} of {diagnostics['frame_count']} retained frames sampled)"
    )


def _format_boundary_color_diagnostics(previous_chunk, current_chunk, diagnostics):
    rgb = ", ".join(f"{value:+.4f}" for value in diagnostics["rgb_mean_delta"])
    context = diagnostics.get("context") or "unclassified boundary"
    return (
        f"HR Endless Sampler boundary color diagnostics Chunk {previous_chunk}->{current_chunk} "
        f"[{context}]: "
        f"delta RGB mean=({rgb}); delta luma={diagnostics['luma_mean_delta']:+.4f}; "
        f"delta contrast sigma={diagnostics['luma_std_delta']:+.4f}; "
        f"delta saturation={diagnostics['saturation_mean_delta']:+.4f}; "
        f"delta midtone={diagnostics['midtone_balance_delta']:+.4f}; "
        f"luma histogram distance={diagnostics['luma_histogram_distance']:.4f}"
    )


def _color_boundary_context(shots, boundary_frame):
    for shot_index, shot_start, _shot_end, _body in shots:
        if shot_start == boundary_frame:
            return f"source cut into Shot {shot_index + 1} at global frame {boundary_frame}"
    for shot_index, shot_start, shot_end, _body in shots:
        if shot_start < boundary_frame < shot_end:
            return f"same Shot {shot_index + 1} at global frame {boundary_frame}"
    return f"unclassified global frame {boundary_frame}"


def _same_shot_correction_frames(shots, boundary_frame, chunk_end_frame):
    """Return generated frames safe to exposure-match before the next source cut."""
    boundary_frame = int(boundary_frame)
    chunk_end_frame = max(boundary_frame, int(chunk_end_frame))
    for _shot_index, shot_start, _shot_end, _body in shots:
        if int(shot_start) == boundary_frame:
            # The new output begins on an intentional source cut. Preserve the
            # new shot's authored grade instead of matching the preceding shot.
            return 0
    for _shot_index, shot_start, shot_end, _body in shots:
        if int(shot_start) < boundary_frame < int(shot_end):
            return max(0, min(chunk_end_frame, int(shot_end)) - boundary_frame)
    # Unstructured prompts have no reliable cut map. Correct the complete new
    # output slice because there is no evidence that its first frame is a cut.
    return chunk_end_frame - boundary_frame


def _record_color_diagnostics(records, chunk_index, diagnostics, boundary_context=None):
    """Store one decoded chunk and emit its adjacent-boundary diagnostics."""
    previous = records.get(chunk_index - 1)
    if previous is not None:
        boundary = _boundary_color_diagnostics(previous, diagnostics)
        if boundary_context:
            boundary["context"] = boundary_context
        diagnostics["boundary_from_previous"] = boundary
        logging.info(
            _format_boundary_color_diagnostics(chunk_index, chunk_index + 1, boundary)
        )
    records[chunk_index] = diagnostics
    logging.info(_format_color_diagnostics(chunk_index + 1, diagnostics))


def _decoded_video_frames(vae, latent, include_final=False, start_frame=0, return_final_frame=False,
                          return_color_diagnostics=False):
    frames = _decode_video_frames(vae, latent)
    retained_start = max(0, min(int(start_frame), frames.shape[0]))
    color_diagnostics = (
        _safe_video_color_diagnostics(frames[retained_start:])
        if return_color_diagnostics and retained_start < frames.shape[0]
        else None
    )
    result = _sample_decoded_video_frames(
        frames,
        include_final=include_final,
        start_frame=start_frame,
        return_final_frame=return_final_frame,
    )
    if return_color_diagnostics:
        return (*result, color_diagnostics) if isinstance(result, tuple) else (result, color_diagnostics)
    return result


def _sample_decoded_video_frames(frames, include_final=False, start_frame=0, return_final_frame=False):
    """Create the stock-resolution 2 FPS Qwen/Gemma presentation from decoded frames."""
    final_frame = frames[-1:].detach().to(device="cpu", copy=True) if return_final_frame else None
    start_frame = max(0, min(int(start_frame), frames.shape[0]))
    sample_indices = list(range(start_frame, frames.shape[0], VIDEO_FPS // 2))
    if include_final and frames.shape[0] > start_frame and sample_indices[-1] != frames.shape[0] - 1:
        sample_indices.append(frames.shape[0] - 1)
    sampled_frames = frames[sample_indices]
    height, width = sampled_frames.shape[1:3]
    target_width, target_height = _reference_video_canvas(width, height)
    if (target_width, target_height) != (width, height):
        sampled_frames = _resize(sampled_frames, target_width, target_height, "disabled")
    if return_final_frame:
        return sampled_frames, sample_indices, final_frame
    return sampled_frames, sample_indices


def _decoded_video_item(vae, latent):
    return _decoded_video_item_from_frames(_decode_video_frames(vae, latent))


def _decoded_video_item_from_frames(frames):
    qwen_frames, sample_indices = _sample_decoded_video_frames(frames)
    return {
        "type": "video",
        "data": qwen_frames,
        "timestamps": [index / 2.0 for index in range(len(sample_indices))],
    }


def _continuation_reference_canvas(video_continuation_res, source_width, source_height):
    """Resolve the selected H3 reference canvas without enlarging the source."""
    if video_continuation_res == "full":
        return source_width, source_height
    try:
        target_width, target_height = VIDEO_CONTINUATION_CANVASES[video_continuation_res]
    except KeyError as error:
        raise ValueError(
            f"Unknown video_continuation_res {video_continuation_res!r}; choose one of "
            + ", ".join(VIDEO_CONTINUATION_RESOLUTIONS)
        ) from error
    if source_height > source_width:
        target_width, target_height = target_height, target_width
    if source_width * source_height <= target_width * target_height:
        return source_width, source_height
    return target_width, target_height


def _encode_resized_continuation_reference(vae, decoded_frames, video_continuation_res):
    """VAE-encode a smaller pixel-space Video1 while preserving its full timeline."""
    source_height, source_width = decoded_frames.shape[1:3]
    target_width, target_height = _continuation_reference_canvas(
        video_continuation_res,
        source_width,
        source_height,
    )
    if (target_width, target_height) == (source_width, source_height):
        return None, (source_width, source_height)
    resized_frames = _resize(decoded_frames, target_width, target_height, "disabled")
    try:
        reference_latent = vae.encode(resized_frames)
    finally:
        del resized_frames
    if reference_latent.ndim != 5 or reference_latent.shape[1] != 24:
        raise ValueError("MiniMax H3 continuation VAE encode did not return a 24-channel video latent")
    return reference_latent, (target_width, target_height)


def _video_ref_block(latent, audio_latent=None):
    ref_audio_t = 0 if audio_latent is None else audio_latent.shape[-1]
    return {
        "kind": "video_audio" if ref_audio_t else "video",
        "latent_t": latent.shape[2],
        "latent_h": latent.shape[3],
        "latent_w": latent.shape[4],
        "ref_audio_t": ref_audio_t,
        "latent": latent,
        "audio_latent": audio_latent,
    }


def _audio_ref_block(audio_latent):
    """Build one native standalone MiniMax H3 audio reference block."""
    return {
        "kind": "audio",
        "ref_audio_t": audio_latent.shape[-1],
        "audio_latent": audio_latent,
    }


def _continuation_payload_metrics(reference_latent, reference_audio, boundary_latent,
                                  target_video, target_audio, full_reference_video):
    """Measure raw conditioning tensors and the packed rows that drive H3 cost."""
    def tensor_bytes(value):
        return 0 if value is None else value.numel() * value.element_size()

    def video_rows(value):
        if value is None:
            return 0
        return int(value.shape[2]) * (int(value.shape[3]) // 2) * (int(value.shape[4]) // 2)

    def audio_rows(value):
        return 0 if value is None else int(value.shape[-1]) * 2

    reference_video_rows = video_rows(reference_latent)
    reference_audio_rows = audio_rows(reference_audio)
    boundary_rows = video_rows(boundary_latent)
    target_rows = video_rows(target_video) + audio_rows(target_audio)
    full_reference_rows = video_rows(full_reference_video)
    return {
        "video_bytes": tensor_bytes(reference_latent),
        "audio_bytes": tensor_bytes(reference_audio),
        "boundary_bytes": tensor_bytes(boundary_latent),
        "total_bytes": tensor_bytes(reference_latent) + tensor_bytes(reference_audio) + tensor_bytes(boundary_latent),
        "video_rows": reference_video_rows,
        "audio_rows": reference_audio_rows,
        "boundary_rows": boundary_rows,
        "total_rows": reference_video_rows + reference_audio_rows + boundary_rows,
        "target_rows": target_rows,
        "full_reference_rows": full_reference_rows,
    }


def _log_continuation_payload(chunk_number, chunk_count, preset, reference_latent,
                              reference_audio, boundary_latent, target_video,
                              target_audio, full_reference_video):
    metrics = _continuation_payload_metrics(
        reference_latent,
        reference_audio,
        boundary_latent,
        target_video,
        target_audio,
        full_reference_video,
    )
    mib = 1024 ** 2
    full_fraction = (
        100.0 * metrics["video_rows"] / metrics["full_reference_rows"]
        if metrics["full_reference_rows"] else 0.0
    )
    target_fraction = (
        100.0 * metrics["total_rows"] / metrics["target_rows"]
        if metrics["target_rows"] else 0.0
    )
    boundary_shape = "none" if boundary_latent is None else str(tuple(boundary_latent.shape))
    audio_shape = "none" if reference_audio is None else str(tuple(reference_audio.shape))
    logging.info(
        "HR Endless Sampler chunk %d/%d Video1 H3 payload [%s]:\n"
        "  video: shape=%s, raw=%0.3f MiB, packed_rows=%d (%0.1f%% of full-resolution Video1 rows=%d)\n"
        "  audio: shape=%s, raw=%0.3f MiB, packed_rows=%d\n"
        "  full-resolution boundary keyframe: shape=%s, raw=%0.3f MiB, packed_rows=%d\n"
        "  total continuation conditioning: raw=%0.3f MiB, packed_rows=%d (%0.1f%% of target AV rows=%d)\n"
        "  Note: raw payload is small; packed rows drive the much larger per-layer attention/activation allocation.",
        chunk_number,
        chunk_count,
        preset,
        tuple(reference_latent.shape),
        metrics["video_bytes"] / mib,
        metrics["video_rows"],
        full_fraction,
        metrics["full_reference_rows"],
        audio_shape,
        metrics["audio_bytes"] / mib,
        metrics["audio_rows"],
        boundary_shape,
        metrics["boundary_bytes"] / mib,
        metrics["boundary_rows"],
        metrics["total_bytes"] / mib,
        metrics["total_rows"],
        target_fraction,
        metrics["target_rows"],
    )


def _debug_preflight_sigmas(sigmas, maximum_steps=3):
    """Return a short, truthful prefix of the configured sampling schedule."""
    step_count = min(max(0, int(maximum_steps)), max(0, int(sigmas.shape[-1]) - 1))
    return sigmas[..., :step_count + 1], step_count


def _debug_preflight_continuation_payload(video, audio, continuation_frames,
                                          video_continuation_res, width, height):
    """Build shape-faithful black continuation data for the debug VRAM probe.

    The values are intentionally meaningless; only the temporal/spatial rows,
    synchronized audio span, Qwen presentation, and temporal boundary anchor
    need to match a real continuation chunk for its allocation behavior.
    """
    reference_t = _video_steps(continuation_frames)
    reference_audio_t = _audio_steps(continuation_frames)
    target_width, target_height = _continuation_reference_canvas(
        video_continuation_res,
        width,
        height,
    )
    reference_video = video.new_zeros(
        (video.shape[0], video.shape[1], reference_t, target_height // 16, target_width // 16)
    )
    if reference_video.shape[3:] == video.shape[3:]:
        # The real full-resolution path aliases these names to the same latent;
        # do not accidentally make the debug probe one reference more costly.
        full_reference_video = reference_video
    else:
        full_reference_video = video.new_zeros(
            (video.shape[0], video.shape[1], reference_t, video.shape[3], video.shape[4])
        )
    reference_audio = audio.new_zeros((*audio.shape[:-1], reference_audio_t))
    boundary_video = video.new_zeros(
        (video.shape[0], video.shape[1], _video_steps(5), video.shape[3], video.shape[4])
    )

    # A real Video1 is decoded in full, then sparsely presented to Qwen at 2
    # FPS. Construct only those retained presentation frames: the discarded
    # intermediate decoded frames do not survive into H3 sampling.
    qwen_frame_count = len(range(0, continuation_frames, VIDEO_FPS // 2))
    qwen_width, qwen_height = _reference_video_canvas(width, height)
    qwen_frames = torch.zeros(
        (qwen_frame_count, qwen_height, qwen_width, 3),
        dtype=torch.float32,
        device=video.device,
    )
    qwen_video_item = {
        "type": "video",
        "data": qwen_frames,
        "timestamps": [index / 2.0 for index in range(qwen_frame_count)],
    }
    return {
        "reference_video": reference_video,
        "full_reference_video": full_reference_video,
        "reference_audio": reference_audio,
        "boundary_video": boundary_video,
        "qwen_items": [{"type": "audio"}, qwen_video_item],
    }


def _allocator_active_reserved(device):
    backend = _memory_backend(device)
    if backend is None:
        return 0, 0
    if device.type == "cuda":
        backend.synchronize(device)
    stats = backend.memory_stats(device)
    return (
        int(stats.get("active_bytes.all.current", stats.get("allocated_bytes.all.current", 0))),
        int(stats.get("reserved_bytes.all.current", 0)),
    )


def _release_debug_preflight(device, *patchers):
    """Drop every disposable preflight owner and flush reclaimable VRAM."""
    for patcher in patchers:
        if patcher is not None:
            comfy.model_management.unload_model_and_clones(patcher)
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)
    backend = _memory_backend(device)
    if backend is not None:
        if device.type == "cuda":
            backend.synchronize(device)
            backend.empty_cache()
        gc.collect()


def _tensor_bytes(value, device):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.device == device else 0
    if getattr(value, "is_nested", False):
        return sum(_tensor_bytes(item, device) for item in value.unbind())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, device) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, device) for item in value)
    return 0


def _memory_backend(device):
    if device.type == "cuda":
        return torch.cuda
    if device.type in ("xpu", "npu", "mlu"):
        return getattr(torch, device.type)
    return None


def _set_pytorch_memory_fraction(fraction, device):
    """Apply an explicit process-wide PyTorch CUDA allocator ceiling."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("pytorch_memory_fraction must be greater than 0 and at most 1.0")
    if not torch.cuda.is_available():
        logging.info(
            "HR Endless Sampler PyTorch memory fraction %.2f was not applied because CUDA is unavailable.",
            fraction,
        )
        return None

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        logging.info(
            "HR Endless Sampler PyTorch memory fraction %.2f was not applied to non-CUDA device %s.",
            fraction,
            cuda_device,
        )
        return None
    if cuda_device.index is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())

    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=cuda_device)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Could not set PyTorch CUDA memory fraction to {fraction:.2f} on {cuda_device}: {error}"
        ) from error

    total_bytes = int(torch.cuda.get_device_properties(cuda_device).total_memory)
    allocator_backend = torch.cuda.get_allocator_backend()
    logging.info(
        "HR Endless Sampler PyTorch CUDA allocator limit: %.1f%% on %s "
        "(%.2f GiB of %.2f GiB, backend=%s).",
        fraction * 100.0,
        cuda_device,
        total_bytes * fraction / (1024 ** 3),
        total_bytes / (1024 ** 3),
        allocator_backend,
    )
    return {
        "fraction": fraction,
        "device": str(cuda_device),
        "limit_bytes": int(total_bytes * fraction),
        "total_bytes": total_bytes,
        "backend": allocator_backend,
    }


def _vram_report(stage, device, components=(), tensors=None):
    mib = 1024 ** 2
    total = comfy.model_management.get_total_memory(device)
    comfy_free, torch_cache_free = comfy.model_management.get_free_memory(device, torch_free_too=True)
    lines = [f"HR Endless Sampler VRAM [{stage}] on {device}:"]
    backend = _memory_backend(device)
    if backend is not None:
        stats = backend.memory_stats(device)
        if device.type == "cuda":
            physical_free, physical_total = backend.mem_get_info(device)
        else:
            physical_total = total
            physical_free = comfy_free - torch_cache_free
        active = stats.get("active_bytes.all.current", 0)
        allocated = stats.get("allocated_bytes.all.current", active)
        reserved = stats.get("reserved_bytes.all.current", 0)
        peak_active = stats.get("active_bytes.all.peak", 0)
        peak_reserved = stats.get("reserved_bytes.all.peak", 0)
        lines.append(
            f"  device: {physical_total / mib:.1f} MiB total, {(physical_total - physical_free) / mib:.1f} MiB used by all processes, "
            f"{physical_free / mib:.1f} MiB physically free"
        )
        lines.append(
            f"  torch: {allocated / mib:.1f} MiB allocated, {active / mib:.1f} MiB active, {reserved / mib:.1f} MiB reserved, "
            f"{max(0, reserved - active) / mib:.1f} MiB cached/inactive"
        )
        lines.append(f"  peak: {peak_active / mib:.1f} MiB active, {peak_reserved / mib:.1f} MiB reserved")
    else:
        lines.append(f"  device: {total / mib:.1f} MiB total")
    lines.append(f"  ComfyUI usable free: {comfy_free / mib:.1f} MiB ({torch_cache_free / mib:.1f} MiB in the torch cache)")

    component_parts = []
    for name, patcher in components:
        if patcher is not None:
            component_parts.append(
                f"{name}={patcher.loaded_size() / mib:.1f} MiB loaded "
                f"({'dynamic' if patcher.is_dynamic() else 'standard'}, {patcher.load_device}, {len(patcher.patches)} patch keys)"
            )
    if component_parts:
        lines.append("  known models: " + "; ".join(component_parts))

    resident_parts = []
    for patcher in comfy.model_management.loaded_models():
        resident_parts.append(
            f"{patcher.model.__class__.__name__}={patcher.loaded_size() / mib:.1f} MiB/{len(patcher.patches)} patches"
        )
    lines.append("  ComfyUI model registry: " + ("; ".join(resident_parts) if resident_parts else "empty"))

    if tensors:
        tensor_parts = []
        for name, value in tensors.items():
            size = _tensor_bytes(value, device)
            if size:
                tensor_parts.append(f"{name}={size / mib:.1f} MiB")
        lines.append("  visible GPU tensor payloads: " + ("; ".join(tensor_parts) if tensor_parts else "none"))
    logging.info("\n".join(lines))


def _refresh_console_progress():
    """Redraw live tqdm bars after a multi-line debug log snapshot.

    ComfyUI's CLI progress handler owns the sampling ``steps`` bar while this
    node owns the outer ``chunk`` bar. A logging call moves the terminal cursor
    below both bars, so without a redraw the next visible progress line can be
    far above a long VRAM report. Refresh chunk bars first and step bars last.
    ``tqdm.auto`` and ComfyUI's plain ``tqdm`` can expose different classes,
    hence both instance registries are checked.
    """
    bars = []
    seen = set()
    for tqdm_class in (tqdm, _cli_tqdm):
        for bar in tuple(getattr(tqdm_class, "_instances", ())):
            if id(bar) not in seen and not getattr(bar, "disable", False):
                seen.add(id(bar))
                bars.append(bar)
    for bar in sorted(bars, key=lambda item: getattr(item, "unit", "") != "chunk"):
        try:
            bar.refresh()
        except (AttributeError, OSError, ValueError):
            # A tqdm instance can be closed while ComfyUI is handling an
            # interrupt; a cosmetic redraw must never affect sampling.
            pass


class _VRAMMonitor:
    def __init__(self, timing, device, components, chunk_count, debug=False):
        self.timing = timing
        self.device = device
        self.components = components
        self.chunk_count = chunk_count
        self.debug = debug
        self.chunk = 0
        self.call = 0
        self.scope = None

    def set_chunk(self, index):
        self.chunk = index
        self.call = 0
        self.scope = None
        backend = _memory_backend(self.device)
        if self.debug and self.device.type == "cuda" and backend is not None:
            backend.reset_peak_memory_stats(self.device)

    def set_scope(self, label):
        self.scope = str(label) if label else None
        self.call = 0
        backend = _memory_backend(self.device)
        if self.debug and self.device.type == "cuda" and backend is not None:
            backend.reset_peak_memory_stats(self.device)

    def report(self, stage, tensors=None, sample_group="all"):
        self.timing.observe_memory(sample_group=sample_group, chunk_index=self.chunk)
        if self.debug:
            _vram_report(stage, self.device, self.components, tensors)
            _refresh_console_progress()

    def __call__(self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        self.call += 1
        scope = self.scope or f"chunk {self.chunk + 1}/{self.chunk_count}"
        label = f"{scope} DiT evaluation {self.call}"
        tensors = {"model input": x, "cross attention": c_crossattn, "model conditions": kwargs}
        self.report(label + " before", tensors, sample_group="dit")
        try:
            result = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        except Exception:
            self.report(label + " FAILED", tensors, sample_group="dit")
            if self.debug and self.device.type == "cuda":
                logging.info("HR Endless Sampler CUDA allocator after failure:\n%s", torch.cuda.memory_summary(self.device, abbreviated=True))
            raise
        self.report(label + " after", {"model output": result}, sample_group="dit")
        return result


class _FixedNoise:
    def __init__(self, seed, samples):
        self.seed = seed
        self.samples = samples

    def generate_noise(self, _latent):
        return self.samples


def _run_debug_memory_preflight(*, guider, sampler, sigmas, chunk_latent, chunk_noise,
                                clip, vae, images, positive, original_conds, chunk_prompt,
                                chunk_seed, video, audio, continuation_frames,
                                video_continuation_res, video_number, audio_number,
                                width, height, chunk_count, vram_monitor=None,
                                preview_execution=None):
    """Probe continuation-chunk VRAM with three disposable denoising steps."""
    preflight_sigmas, preflight_steps = _debug_preflight_sigmas(sigmas, 3)
    if preflight_steps == 0:
        logging.info("HR Endless Sampler debug VRAM preflight skipped: the sigma schedule has no steps.")
        return

    device = guider.model_patcher.load_device
    baseline_active, baseline_reserved = _allocator_active_reserved(device)
    mib = 1024 ** 2
    payload = None
    preflight_encoded = None
    preflight_conds = None
    preflight_refs = []
    preflight_result = None
    simulated_completed_output = None
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state(device)

    phase = (
        f"Debug VRAM preflight: simulating a continuation chunk for {preflight_steps} steps "
        f"with {continuation_frames} Video1 frames"
    )
    logging.info("HR Endless Sampler: %s.", phase)
    if preview_execution is not None:
        preview_execution.set_phase(phase, chunk=0)
    if vram_monitor is not None:
        vram_monitor.set_scope("debug VRAM preflight")

    try:
        payload = _debug_preflight_continuation_payload(
            video,
            audio,
            continuation_frames,
            video_continuation_res,
            width,
            height,
        )
        video_label = f"<Video {video_number}>"
        audio_label = f"<Audio {audio_number}>"
        preflight_prompt = _video_continuation_prompt(chunk_prompt, video_label, audio_label)
        preflight_encoded = _encode_prompt(
            clip,
            preflight_prompt,
            images,
            positive,
            width,
            height,
            True,
            payload["qwen_items"],
        )
        payload["qwen_items"].clear()
        comfy.model_management.unload_model_and_clones(clip.patcher)
        if vae is not None:
            comfy.model_management.unload_model_and_clones(vae.patcher)
        comfy.model_management.soft_empty_cache(force=True)

        preflight_refs.append(
            _video_ref_block(payload["reference_video"], payload["reference_audio"])
        )
        target_video, target_audio = chunk_latent["samples"].unbind()
        intermediate_device = comfy.model_management.intermediate_device()
        simulated_completed_output = (
            torch.empty_like(target_video, device=intermediate_device),
            torch.empty_like(target_audio, device=intermediate_device),
            torch.empty_like(target_video, device=intermediate_device),
            torch.empty_like(target_audio, device=intermediate_device),
        )
        preflight_conds = _conditioning_for_chunk(
            original_conds,
            0,
            _pixel_frames(chunk_latent["samples"].unbind()[0].shape[2]),
            preflight_encoded,
            video_context=payload["boundary_video"],
            video_refs=preflight_refs,
            video_context_start=0,
        )
        guider.original_conds = preflight_conds
        _log_continuation_payload(
            2,
            chunk_count,
            f"debug preflight/{video_continuation_res}",
            payload["reference_video"],
            payload["reference_audio"],
            payload["boundary_video"],
            *chunk_latent["samples"].unbind(),
            payload["full_reference_video"],
        )
        payload["full_reference_video"] = None
        if vram_monitor is not None:
            vram_monitor.report(
                "debug VRAM preflight immediately before sampler",
                {
                    "chunk latent": chunk_latent,
                    "chunk noise": chunk_noise,
                    "conditioning": preflight_conds,
                    "simulated completed output": simulated_completed_output,
                },
            )

        preflight_result = guider.sample(
            chunk_noise,
            chunk_latent["samples"],
            sampler,
            preflight_sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=chunk_seed,
        )
        active, reserved = _allocator_active_reserved(device)
        backend = _memory_backend(device)
        stats = {} if backend is None else backend.memory_stats(device)
        peak_active = int(stats.get("active_bytes.all.peak", active))
        peak_reserved = int(stats.get("reserved_bytes.all.peak", reserved))
        logging.info(
            "HR Endless Sampler debug VRAM preflight passed: %d/%d steps; "
            "peak %.1f MiB active, %.1f MiB reserved. Discarding the simulation before Chunk 1.",
            preflight_steps,
            max(0, int(sigmas.shape[-1]) - 1),
            peak_active / mib,
            peak_reserved / mib,
        )
    finally:
        guider.original_conds = original_conds
        preflight_result = None
        simulated_completed_output = None
        preflight_conds = None
        preflight_encoded = None
        preflight_refs.clear()
        if payload is not None:
            payload.clear()
        payload = None
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        _release_debug_preflight(
            device,
            guider.model_patcher,
            clip.patcher,
            vae.patcher if vae is not None else None,
        )
        post_active, post_reserved = _allocator_active_reserved(device)
        active_delta = post_active - baseline_active
        reserved_delta = post_reserved - baseline_reserved
        message = (
            "HR Endless Sampler debug VRAM preflight cleanup: "
            f"active {post_active / mib:.1f} MiB ({active_delta / mib:+.1f} MiB vs baseline), "
            f"reserved {post_reserved / mib:.1f} MiB ({reserved_delta / mib:+.1f} MiB vs baseline)."
        )
        if active_delta > 16 * mib:
            logging.warning("%s Disposable allocations remain above baseline.", message)
        else:
            logging.info("%s Disposable allocations were released.", message)
        if vram_monitor is not None:
            vram_monitor.set_chunk(0)
            vram_monitor.report("after debug VRAM preflight cleanup")


class _ChunkProgress:
    def __init__(self, count):
        self.count = count
        self.bar = None

    def start(self, index):
        if self.bar is None:
            self.bar = tqdm(
                total=self.count,
                desc=f"Chunk {index + 1}/{self.count}",
                unit="chunk",
                leave=False,
                position=0,
                dynamic_ncols=True,
                disable=not comfy.utils.PROGRESS_BAR_ENABLED,
            )
        self.bar.n = index
        self.bar.set_description_str(f"Chunk {index + 1}/{self.count}")
        self.bar.refresh()

    def finish(self, index):
        self.bar.n = index + 1
        self.bar.refresh()

    def close(self):
        if self.bar is not None:
            self.bar.close()


class _PreparationProgress:
    """Keep long pre-sampling work visible without relying on debug logging.

    Gemma runs in a deliberately isolated subprocess and can take minutes to
    load, consume the long prompt, and emit its plan.  That is valid work, but
    without a heartbeat ComfyUI looks frozen before it ever opens the sampler's
    normal progress bar.  The same concise phase reaches both the console and
    the accumulated preview widget.
    """

    def __init__(self, phase, preview_execution=None, *, chunk=None, interval=15.0,
                 live_console_bar=False):
        self.phase = str(phase)
        self.preview_execution = preview_execution
        self.chunk = chunk
        self.interval = max(1.0, float(interval))
        self.live_console_bar = bool(live_console_bar)
        self.started = None
        self._stop = threading.Event()
        self._thread = None
        self._bar = None
        self._pulse = 0
        self._last_report = None
        self._token_generation = 0
        self._tokens = 0
        self._tokens_per_second = None

    @staticmethod
    def _elapsed(seconds):
        rounded = max(0, round(seconds))
        minutes, seconds = divmod(rounded, 60)
        return f"{minutes}:{seconds:02d}"

    def _message(self, status="still working"):
        elapsed = self._elapsed(time.perf_counter() - self.started)
        throughput = ""
        if self._tokens_per_second is not None:
            throughput = (
                f"; {self._tokens} tokens, "
                f"{self._tokens_per_second:.1f} tokens/sec"
            )
        return f"{self.phase} — {status} ({elapsed} elapsed{throughput})"

    def _refresh_bar(self):
        if self._bar is None:
            return
        # Gemma's isolated worker does not expose an accurate token count. A
        # looping bar is therefore deliberately indeterminate: it confirms
        # active work without inventing a percentage or ETA.
        if self._bar.total:
            self._pulse = (self._pulse + 1) % int(self._bar.total)
            self._bar.n = self._pulse
        set_postfix = getattr(self._bar, "set_postfix_str", None)
        if callable(set_postfix) and self._tokens_per_second is not None:
            set_postfix(
                f"{self._tokens} tokens, {self._tokens_per_second:.1f} tokens/sec",
                refresh=False,
            )
        self._bar.refresh()

    def update_token_progress(self, tokens, tokens_per_second, generation=1):
        """Receive a live decode-rate record from the isolated Gemma worker."""
        self._token_generation = int(generation)
        self._tokens = max(0, int(tokens))
        self._tokens_per_second = max(0.0, float(tokens_per_second))
        self._refresh_bar()
        if self.preview_execution is not None:
            self.preview_execution.set_phase(
                self._message("generating"),
                chunk=self.chunk,
            )

    def _emit(self, status="still working", *, force=False):
        self._refresh_bar()
        now = time.perf_counter()
        report_due = (
            force
            or not self.live_console_bar
            or self._last_report is None
            or now - self._last_report >= self.interval
        )
        if not report_due:
            return
        message = self._message(status)
        logging.info("HR Endless Sampler: %s", message)
        if self.preview_execution is not None:
            self.preview_execution.set_phase(message, chunk=self.chunk)
        self._last_report = now
        _refresh_console_progress()

    def _run(self):
        tick_interval = 1.0 if self.live_console_bar else self.interval
        while not self._stop.wait(tick_interval):
            self._emit()

    def __enter__(self):
        self.started = time.perf_counter()
        if self.live_console_bar:
            self._bar = tqdm(
                total=30,
                desc=self.phase,
                unit="gemma",
                leave=False,
                position=0,
                dynamic_ncols=True,
                disable=not comfy.utils.PROGRESS_BAR_ENABLED,
                bar_format="{desc}: |{bar:24}| {elapsed} elapsed{postfix}",
            )
        self._emit(force=True)
        self._thread = threading.Thread(
            target=self._run,
            name="hr-endless-sampler-preparation-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval + 1.0))
        self._emit("complete" if _exc_type is None else "stopped", force=True)
        if self._bar is not None:
            self._bar.close()


class _SamplerTiming:
    """Accumulate wall-clock work and an always-on physical-memory timeline."""

    _ORDER = (
        ("H3 sampling", "h3_sampling"),
        ("Qwen encode/tokenize", "qwen"),
        ("Gemma 4", "gemma4"),
        ("Video VAE decode: final preview/Gemma", "vae_previous_chunk"),
        ("Video VAE decode: continuous TaoMate output", "vae_video_final"),
        ("TaoMate audio teacher inference", "taomate_audio_first"),
        ("Audio VAE decode: final preview", "vae_audio_preview"),
        ("AudioSR: chunk previews", "audiosr_preview"),
        ("AudioSR: full original audio", "audiosr_final"),
        ("VAE decode: final color diagnostics", "vae_color_final"),
        ("Final output shot color grade", "output_shot_color"),
        ("VAE continuation decode/resize/encode", "vae_context"),
        ("VAE decode: Qwen full history", "vae_history"),
    )

    _REPORT_LABELS = {
        "h3_sampling": "H3 sampling",
        "qwen": "Qwen",
        "gemma4": "Gemma 4",
        "vae_previous_chunk": "Final video preview/Gemma VAE decode",
        "vae_video_final": "Continuous TaoMate video VAE decode",
        "taomate_audio_first": "TaoMate audio teacher inference",
        "vae_audio_preview": "Final audio preview VAE decode",
        "audiosr_preview": "AudioSR chunk previews",
        "audiosr_final": "AudioSR full original audio",
        "vae_color_final": "Final color-diagnostic VAE decode",
        "output_shot_color": "Final output shot color grade",
        "vae_context": "Video1 VAE decode",
        "vae_history": "Qwen full-history VAE decode",
    }

    def __init__(self, device, poll_interval=1.0):
        self.started = time.perf_counter()
        self.device = torch.device(device)
        self.seconds = {key: 0.0 for _label, key in self._ORDER}
        self.calls = {key: 0 for _label, key in self._ORDER}
        self.chunk_started = {}
        self.chunk_seconds = {}
        self.max_process_rss = 0
        self.max_system_ram_used = 0
        self.system_ram_total = 0
        self.max_device_used = 0
        self.device_total = 0
        self.max_torch_allocated = 0
        self.max_torch_reserved = 0
        self.max_torch_allocator_peak = 0
        self.max_torch_reserved_peak = 0
        self._process = psutil.Process()
        self._backend = _memory_backend(self.device)
        self._memory_lock = threading.Lock()
        self._physical_samples = []
        self._snapshot_count = 0
        self._snapshot_sum = 0
        self._dit_snapshot_count = 0
        self._dit_snapshot_sum = 0
        self._later_dit_snapshot_count = 0
        self._later_dit_snapshot_sum = 0
        self._poll_interval = max(0.25, float(poll_interval))
        self._poll_stop = threading.Event()
        self._poll_thread = None

        # This high-water mark is scoped to this sampler execution. The debug
        # wrapper may reset PyTorch's native counter per chunk, so we retain
        # the largest value observed after every timed phase as well.
        if self._backend is not None:
            try:
                self._backend.reset_peak_memory_stats(self.device)
            except (AttributeError, RuntimeError):
                pass
        self.observe_memory()

    def start_memory_poll(self):
        if self._backend is not None and self._poll_thread is None:
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_memory,
                name="hr-endless-sampler-memory",
                daemon=True,
            )
            self._poll_thread.start()

    def start_chunk(self, index):
        self.chunk_started[index] = time.perf_counter()

    def finish_chunk(self, index):
        started = self.chunk_started.pop(index, None)
        if started is not None:
            elapsed = time.perf_counter() - started
            self.chunk_seconds[index] = elapsed
            return elapsed
        return None

    def add(self, key, started):
        elapsed = time.perf_counter() - started
        self.seconds[key] += elapsed
        self.calls[key] += 1
        self.observe_memory()
        return elapsed

    def elapsed(self):
        return max(0.0, time.perf_counter() - self.started)

    def _observe_ram(self):
        try:
            memory = psutil.virtual_memory()
            process_rss = self._process.memory_info().rss
            with self._memory_lock:
                self.max_process_rss = max(self.max_process_rss, process_rss)
                self.max_system_ram_used = max(self.max_system_ram_used, memory.used)
                self.system_ram_total = max(self.system_ram_total, memory.total)
        except (OSError, psutil.Error):
            pass

    def _record_physical(self, used, total, sample_group=None, chunk_index=None):
        now = time.perf_counter()
        with self._memory_lock:
            self._physical_samples.append((now, used))
            self.max_device_used = max(self.max_device_used, used)
            self.device_total = max(self.device_total, total)
            if sample_group is not None:
                self._snapshot_count += 1
                self._snapshot_sum += used
                if sample_group == "dit":
                    self._dit_snapshot_count += 1
                    self._dit_snapshot_sum += used
                    if chunk_index is not None and chunk_index > 0:
                        self._later_dit_snapshot_count += 1
                        self._later_dit_snapshot_sum += used

    def _observe_physical(self, sample_group=None, chunk_index=None):
        if self._backend is None:
            return
        try:
            if self.device.type == "cuda":
                physical_free, physical_total = self._backend.mem_get_info(self.device)
            else:
                physical_total = comfy.model_management.get_total_memory(self.device)
                physical_free = comfy.model_management.get_free_memory(self.device)
            self._record_physical(
                physical_total - physical_free,
                physical_total,
                sample_group=sample_group,
                chunk_index=chunk_index,
            )
        except (AttributeError, RuntimeError):
            pass

    def _poll_memory(self):
        while not self._poll_stop.wait(self._poll_interval):
            self._observe_ram()
            self._observe_physical()

    def _stop_memory_poll(self):
        if self._poll_thread is not None:
            self._poll_stop.set()
            self._poll_thread.join(timeout=max(2.0, self._poll_interval * 2.0))
            self._poll_thread = None

    def observe_memory(self, sample_group=None, chunk_index=None):
        """Best-effort memory snapshot; monitoring must never affect sampling."""
        self._observe_ram()
        if self._backend is None:
            return
        try:
            stats = self._backend.memory_stats(self.device)
            allocated = stats.get("allocated_bytes.all.current", stats.get("active_bytes.all.current", 0))
            reserved = stats.get("reserved_bytes.all.current", 0)
            allocator_peak = stats.get("allocated_bytes.all.peak", allocated)
            reserved_peak = stats.get("reserved_bytes.all.peak", reserved)
            with self._memory_lock:
                self.max_torch_allocated = max(self.max_torch_allocated, allocated)
                self.max_torch_reserved = max(self.max_torch_reserved, reserved)
                self.max_torch_allocator_peak = max(self.max_torch_allocator_peak, allocator_peak)
                self.max_torch_reserved_peak = max(self.max_torch_reserved_peak, reserved_peak)
        except (AttributeError, RuntimeError):
            pass
        self._observe_physical(sample_group=sample_group, chunk_index=chunk_index)

    @staticmethod
    def _duration(seconds):
        minutes, seconds = divmod(seconds, 60.0)
        if minutes:
            return f"{int(minutes)}m {seconds:05.2f}s"
        return f"{seconds:.2f}s"

    @staticmethod
    def _memory_size(value):
        return f"{value / (1024 ** 3):.2f} GiB"

    @staticmethod
    def _clock_duration(seconds):
        rounded = max(0, round(seconds))
        hours, remainder = divmod(rounded, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _peak_interval(v0, v1, duration, threshold):
        if duration <= 0:
            return 0.0
        above0 = v0 > threshold
        above1 = v1 > threshold
        if above0 == above1:
            return duration if above0 else 0.0
        if v1 == v0:
            return 0.0
        crossing = max(0.0, min(1.0, (threshold - v0) / (v1 - v0)))
        return duration * (crossing if above0 else 1.0 - crossing)

    def _physical_summary(self):
        with self._memory_lock:
            samples = sorted(self._physical_samples)
            snapshot_count = self._snapshot_count
            snapshot_sum = self._snapshot_sum
            dit_count = self._dit_snapshot_count
            dit_sum = self._dit_snapshot_sum
            later_count = self._later_dit_snapshot_count
            later_sum = self._later_dit_snapshot_sum
            peak = self.max_device_used
            total = self.device_total
        if snapshot_count:
            average = snapshot_sum / snapshot_count
        elif samples:
            average = sum(value for _timestamp, value in samples) / len(samples)
        else:
            average = 0
        threshold = (average + peak) / 2.0
        peak_time = 0.0
        if peak > average:
            for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
                peak_time += self._peak_interval(v0, v1, t1 - t0, threshold)
        return {
            "average": average,
            "snapshot_count": snapshot_count or len(samples),
            "dit_average": dit_sum / dit_count if dit_count else 0,
            "dit_count": dit_count,
            "later_dit_average": later_sum / later_count if later_count else 0,
            "later_dit_count": later_count,
            "peak": peak,
            "total": total,
            "threshold": threshold,
            "peak_time": peak_time,
        }

    def _projected_time(self, full_chunks):
        completed = sorted(self.chunk_seconds)
        if not completed or full_chunks <= len(completed):
            return None
        first = self.chunk_seconds[completed[0]]
        later = [self.chunk_seconds[index] for index in completed if index != completed[0]]
        later_average = sum(later) / len(later) if later else first
        return first + later_average * max(0, full_chunks - 1)

    def report(self, status, completed_chunks, run):
        self._stop_memory_poll()
        self.observe_memory()
        total = time.perf_counter() - self.started
        measured = sum(self.seconds.values())
        physical = self._physical_summary()
        rendered_frames = run["rendered_frames"]
        rendered = "none" if rendered_frames <= 0 else f"frames 0-{rendered_frames - 1}"
        configuration = (
            f"chunk_frames={run['chunk_frames']}, context_keyframes={run['context_keyframes']}, "
            f"guide_overlap={run['guide_overlap']}, video_continuation={run['video_continuation']}, "
            f"video_continuation_method={run.get('video_continuation_method', VIDEO_CONTINUATION_METHOD_VIDEO1)}, "
            f"video_continuation_res={run.get('video_continuation_res', 'full')}, "
            f"pytorch_memory_fraction={run.get('pytorch_memory_fraction', 1.0):.2f}"
        )
        lines = [
            "HR Endless Sampler run report:",
            "",
            "Baseline from this run:",
            f"  Configuration: {configuration}",
            f"  Rendered: {completed_chunks} chunk{'s' if completed_chunks != 1 else ''}, {rendered} ({status})",
            f"  Resolution: {run['width']}x{run['height']}",
            f"  Sampling: {run['sampling_steps']} steps",
            f"  Full planned sequence: {run['full_frames']} frames, {run['full_chunks']} chunks"
            + ("; completed" if completed_chunks == run["full_chunks"] and status == "complete" else f"; stopped after chunk {completed_chunks}"),
            "",
            "VRAM baseline:",
        ]
        if physical["total"]:
            average_percent = 100.0 * physical["average"] / physical["total"]
            peak_percent = 100.0 * physical["peak"] / physical["total"]
            lines.append(
                f"  Average across all {physical['snapshot_count']} physical-VRAM snapshots: "
                f"{self._memory_size(physical['average'])} / {self._memory_size(physical['total'])} - {average_percent:.1f}%"
            )
            if physical["dit_count"]:
                lines.append(
                    f"  Average during H3 DiT evaluations: {self._memory_size(physical['dit_average'])} - "
                    f"{100.0 * physical['dit_average'] / physical['total']:.1f}%"
                )
            if physical["later_dit_count"]:
                lines.append(
                    f"  Average during later-chunk DiT evaluations: {self._memory_size(physical['later_dit_average'])} - "
                    f"{100.0 * physical['later_dit_average'] / physical['total']:.1f}%"
                )
            lines.append(
                f"  Peak: {self._memory_size(physical['peak'])} - {peak_percent:.1f}%"
            )
            lines.append(
                f"  Peak Time: {self._duration(physical['peak_time'])} "
                f"(VRAM closer to Peak than Average; above {self._memory_size(physical['threshold'])})"
            )
            lines.append(
                f"  PyTorch VRAM high-water: allocated {self._memory_size(self.max_torch_allocator_peak)}, "
                f"reserved {self._memory_size(self.max_torch_reserved_peak)}"
            )
        if self.system_ram_total:
            lines.append(
                "  Peak RAM: "
                f"ComfyUI process RSS {self._memory_size(self.max_process_rss)}; "
                f"system {self._memory_size(self.max_system_ram_used)} / {self._memory_size(self.system_ram_total)} used"
            )
        lines.extend([
            "",
            "Time baseline:",
            f"  Unlimited sampler wall time: {self._clock_duration(total)}",
            f"  Average per completed chunk: {self._duration(total / completed_chunks) if completed_chunks else 'n/a'}",
        ])
        taomate_kv_cache = run.get("taomate_kv_cache")
        if taomate_kv_cache is not None:
            stored = int(taomate_kv_cache.get("stored_bytes", 0))
            raw = int(taomate_kv_cache.get("raw_bytes", 0))
            peak_stored = int(taomate_kv_cache.get("peak_stored_bytes", stored))
            peak_raw = int(taomate_kv_cache.get("peak_raw_bytes", raw))
            cache_seconds = taomate_kv_cache.get("seconds") or {}
            lines.extend([
                "",
                "TaoMate KV cache:",
                f"  CPU RAM retained: {self._memory_size(stored)} stored / {self._memory_size(raw)} uncompressed; "
                f"peak {self._memory_size(peak_stored)} stored / {self._memory_size(peak_raw)} uncompressed "
                f"({taomate_kv_cache.get('compression', 'none')})",
                f"  KV-cache operations: {self._duration(sum(cache_seconds.values()))}",
            ])
            for name, seconds in sorted(cache_seconds.items()):
                lines.append(f"    {name}: {self._duration(seconds)}")
        lines.extend([
            "",
            "Breakdown:",
            "  Component                                  Total         Average/call",
            "  -----------------------------------------  ------------  ------------",
        ])
        for _label, key in self._ORDER:
            calls = self.calls[key]
            if calls:
                lines.append(
                    f"  {self._REPORT_LABELS[key]:<41}  {self._duration(self.seconds[key]):>12}  "
                    f"{self._duration(self.seconds[key] / calls):>12}"
                )
        lines.append(f"  Other sampler overhead: {self._duration(max(0.0, total - measured))}")
        color_diagnostics = run.get("color_diagnostics") or {}
        if color_diagnostics:
            ordered = sorted(color_diagnostics.items())
            first_chunk, first = ordered[0]
            latest_chunk, latest = ordered[-1]
            rgb_delta = [
                latest_value - first_value
                for first_value, latest_value in zip(first["rgb_mean"], latest["rgb_mean"])
            ]
            boundary_entries = [
                (chunk_number, metrics.get("boundary_from_previous"))
                for chunk_number, metrics in ordered
                if metrics.get("boundary_from_previous")
            ]
            same_shot_boundaries = [
                item for item in boundary_entries
                if str(item[1].get("context") or "").startswith("same Shot ")
            ]
            comparison_boundaries = same_shot_boundaries or boundary_entries
            lines.extend([
                "",
                "Decoded color/contrast baseline:",
                "  Display-referred VAE RGB; scene changes also affect these values, so this is not a gamma estimate.",
                f"  Analyzed retained chunks: {', '.join(str(index + 1) for index, _metrics in ordered)}",
                f"  Chunk {first_chunk + 1} -> Chunk {latest_chunk + 1} whole-chunk drift: "
                f"delta RGB=({', '.join(f'{value:+.4f}' for value in rgb_delta)}), "
                f"luma={latest['luma_mean'] - first['luma_mean']:+.4f}, "
                f"contrast sigma={latest['luma_std'] - first['luma_std']:+.4f}, "
                f"saturation={latest['saturation_mean'] - first['saturation_mean']:+.4f}, "
                f"midtone={latest['midtone_balance'] - first['midtone_balance']:+.4f}",
            ])
            if comparison_boundaries:
                largest_luma_chunk, largest_luma = max(
                    comparison_boundaries,
                    key=lambda item: abs(item[1]["luma_mean_delta"]),
                )
                largest_hist_chunk, largest_hist = max(
                    comparison_boundaries,
                    key=lambda item: item[1]["luma_histogram_distance"],
                )
                lines.append(
                    f"  Largest {'same-shot ' if same_shot_boundaries else ''}boundary luma shift: "
                    f"Chunk {largest_luma_chunk}->{largest_luma_chunk + 1} "
                    f"({largest_luma['luma_mean_delta']:+.4f}; {largest_luma.get('context', 'unclassified')})"
                )
                lines.append(
                    f"  Largest {'same-shot ' if same_shot_boundaries else ''}boundary luma-histogram distance: "
                    f"Chunk {largest_hist_chunk}->{largest_hist_chunk + 1} "
                    f"({largest_hist['luma_histogram_distance']:.4f}; "
                    f"{largest_hist.get('context', 'unclassified')})"
                )
            if latest_chunk + 1 < completed_chunks:
                lines.append(
                    f"  Chunks after {latest_chunk + 1} were not decoded inside the sampler before this run ended; "
                    "their final pixels are decoded downstream."
                )
        projection = self._projected_time(run["full_chunks"])
        if projection is not None:
            lines.append(
                f"  Projected full-sequence sampler time: approximately {self._clock_duration(projection)} "
                f"for {run['full_chunks']} chunks"
            )
        logging.info("\n".join(lines))


class HREndlessSampler(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSampler",
            display_name="HR Endless Sampler",
            category="model/sampling/custom",
            description="Samples a long video latent as continuation-guided temporal chunks. Replace SamplerCustomAdvanced and set the largest chunk that fits in VRAM. The current chunking backend is MiniMax H3.",
            inputs=[
                io.Noise.Input("noise", lazy=True),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler", lazy=True),
                io.Sigmas.Input("sigmas", lazy=True),
                io.Latent.Input("latent_image"),
                io.Clip.Input("clip", lazy=True, tooltip="The CLIP used to encode the original conditioning for the current model backend."),
                io.String.Input("prompt", force_input=True,
                                tooltip="The original model prompt. MiniMax H3 currently uses [Shot 1] and [Shot N] At MM:SS.mmm, markers."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001,
                               tooltip="FPS used to convert source-prompt cut timestamps to exact frame positions."),
                io.Int.Input("chunk_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Maximum chunk frames, snapped down to H3's 17k+5 grid. In TaoMate this sets the prompt/audio-teacher group size; inference still uses small sub-chunks."),
                io.Image.Input("images", optional=True,
                               tooltip="Original backend conditioning images as a batch. For MiniMax H3 Ref2VA, keep reference images in their original order."),
                io.Int.Input("video_continuation", default=39, min=5, max=3600, step=17,
                               tooltip="Completed continuation tail length. Video1 reference uses it for a synchronized Video1/Audio1 reference; Masked AV uses it for native boundary keyframes and the matching decoded/latent tail replacement inside chunk_frames."),
                io.Combo.Input(
                    "video_continuation_method",
                    options=list(VIDEO_CONTINUATION_METHODS),
                    default=VIDEO_CONTINUATION_METHOD_TAOMATE,
                    tooltip=(
                        "Video1 reference keeps the current Ref2VA <Video N>/<Audio N> path plus its five-frame "
                        "packing prefix. Masked AV creates native video/audio boundary keyframes covering this "
                        "many completed frames, then replaces the preceding decoded/latent tails with the new "
                        "prefix. It creates no Video1 or Audio1 reference text. The boundary is included inside "
                        "chunk_frames, so it must be smaller than chunk_frames and reduces new frames per chunk. "
                        "TaoMate-H3 uses small streaming phases and CPU KV memory, preserving the supplied sampler and sigmas. "
                        "chunk_frames controls TaoMate prompt/audio group size. video_continuation controls its first "
                        "sub-chunk; following sub-chunks use video_continuation minus 5 frames and the cadence ends on 17 frames. "
                        "Set video_continuation to 39 to preserve the prior 39/34/34/17 cadence. "
                        "Uses native H3 references/keyframes; CFG 1 is recommended for the TaoMate LoRA. "
                        "Audio teacher uses the same model/LoRA and captures the selected sigma points; no disk replay."
                    ),
                ),
                io.Combo.Input(
                    "video_continuation_res",
                    options=list(VIDEO_CONTINUATION_RESOLUTIONS),
                    default="full",
                    tooltip=(
                        "Spatial resolution of the H3 Video1 continuation reference. full preserves the generated "
                        "latent exactly. Smaller choices decode, resize, and VAE-encode only Video1 before H3 "
                        "sampling, reducing reference attention/VRAM so more frames may fit in chunk_frames. "
                        "Qwen and Gemma keep the normal native H3 reference-video presentation resolution."
                    ),
                ),
                io.Boolean.Input("audio_feathered_overlap", default=False, tooltip="Copy previous video/audio latent tails into the opening latent and smoothly release both during the final spoken word. Both tails remain keyframes when disabled. This changes joint AV inference and requires a fresh cache."),
                io.Vae.Input("vae", optional=True,
                             tooltip="Video VAE required by the current MiniMax H3 continuation and Gemma visual-directing backend."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="MiniMax H3 audio VAE. When connected, each completed chunk's final decoded audio is synchronized with its full-VAE browser preview."),
                io.Combo.Input("color_correction", options=list(COLOR_CORRECTION_MODES), default="disable", tooltip="Disable grading, match chunk boundaries with ColorMatchV2's MKL transfer, match entire shots after generation, or apply both. This changes preview/final pixels only; H3 continuation latents remain native and unchanged. Changing this setting requires a fresh cache."),
                io.Boolean.Input("linear_color_compute", default=False, tooltip="Use float32 inverse-gamma compute color: convert input reference RGB with sRGB^(1/2.4) before H3/VAE encoding, then restore preview and IMAGE output with value^2.4. Changes inference behavior and requires a fresh cache."),
                io.Combo.Input("compute_precision", options=["default", "fp32 (full precision but more VRAM needed)"], default="default", tooltip="default preserves the incoming model precision. fp32 forces diffusion computation to 32-bit for video and audio, increasing memory use and runtime. VAE and text encoder precision are unchanged."),
                io.Combo.Input("kv_cache_compression", options=["none", "zstd lossless", "int8", "turboquant"], default="turboquant", tooltip="TaoMate only: none keeps exact BF16 KV in RAM. zstd lossless stores exact BF16 bytes with Zstd. int8 uses per-vector signed INT8 plus a scale. turboquant uses a GPU 4-bit rotated-vector codec. The last two are lossy and experimental."),
                io.Boolean.Input("audio_sr", default=False, tooltip="Apply AudioSR to each decoded chunk for preview, then separately to the assembled original decoded audio for the final 48 kHz AUDIO output. Requires audio_vae and python/audio_sr.py --install. Adds processing time; does not change H3 reference latents."),
                HRPreProduction.Input("pre_production", optional=True, tooltip="Connect Gemma 4 for model-directed prompts or Legacy Chunk Prompts for editable, pre-baked prompts. Unconnected uses native legacy prompts without an LLM."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log every chunk prompt, raw Gemma response, and detailed VRAM snapshots. chunk_prompts is returned whether debug is enabled or not."),
                io.Int.Input("debug_stop_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Stop after this 1-based chunk number and return the partial result. 0 samples every chunk."),
                io.Int.Input("debug_start_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Resume or rerun from this 1-based chunk using the automatically recorded last-run recovery cache. Leave this at 0 to automatically continue a compatible interrupted render; nonzero forces a specific chunk for debugging."),
            ],
            outputs=[
                io.Latent.Output(display_name="output", tooltip="Raw assembled latent from every sampled chunk. Display color correction is applied only to decoded preview, image, and video output, never to this latent."),
                io.Latent.Output(display_name="denoised_output", tooltip="Raw assembled denoised latent from every sampled chunk. Display color correction is applied only to decoded preview, image, and video output, never to this latent."),
                io.String.Output(display_name="chunk_prompts", tooltip="Exact planned prompt and frame ranges for every active chunk."),
                HREndlessTimeline.Output(display_name="timeline", tooltip="Finished chunk, shot, and Gemma prompt metadata for HR Endless Sampler Save Video."),
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Color-corrected full-VAE decoded frames assembled from the finalized chunks. "
                        "These are the same authoritative frames shown by the completed preview; keeping "
                        "the complete float image batch uses system RAM proportional to video length."
                    ),
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip=(
                        "Corrected full audio-VAE output assembled with the sampler's continuation trim, "
                        "overlap replacement, and seam handling. This matches the finalized preview audio."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def check_lazy_status(cls, noise=None, sampler=None, sigmas=None, clip=None, **_kwargs):
        lazy_inputs = {"noise": noise, "sampler": sampler, "sigmas": sigmas, "clip": clip}
        return [name for name, value in lazy_inputs.items() if value is None]

    @classmethod
    @_guard_replay_cache
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0, chunk_frames=124, images=None,
                video_continuation=39, video_continuation_method=VIDEO_CONTINUATION_METHOD_TAOMATE,
                video_continuation_res="full", vae=None, audio_vae=None,
                debug=False, debug_stop_chunk=0, debug_start_chunk=0,
                unique_id=None, dynprompt=None, color_correction="disable",
                pre_production=None,
                linear_color_compute=False,
                audio_feathered_overlap=False,
                compute_precision="default",
                kv_cache_compression="turboquant",
                audio_sr=False,
                **_deprecated_inputs):
        use_taomate = video_continuation_method == VIDEO_CONTINUATION_METHOD_TAOMATE
        taomate_backend = None
        taomate_kv_cache = None
        audio_first_enabled = False
        if use_taomate:
            from .python.taomate import TaoMateStreaming, TOGGLE_TAOMATE_DIVERGENCY_AUDIO_FIRST, TOGGLE_TAOMATE_DIVERGENCY_AUDIO_TRANSCRIPT_RETRY
            audio_first_enabled = bool(TOGGLE_TAOMATE_DIVERGENCY_AUDIO_FIRST)
            if audio_vae is None:
                raise ValueError("TaoMate dialogue feedback requires audio_vae to decode each teacher chunk for transcription")
            TaoMateStreaming.validate(guider)
            if debug_start_chunk:
                raise ValueError("TaoMate-H3 currently starts from Chunk 1; persistent KV replay is not yet supported")
            incoming = latent_image["samples"]
            if not incoming.is_nested or len(incoming.unbind()) != 2:
                raise ValueError("TaoMate-H3 requires a nested H3 video/audio latent")
            incoming_video, incoming_audio = incoming.unbind()
            if incoming_video.ndim != 5 or incoming_video.shape[1] != 24 or incoming_audio.ndim != 4 or incoming_audio.shape[1] != 32:
                raise ValueError("TaoMate-H3 requires 24-channel video and 32-channel audio latents")
            taomate_backend = TaoMateStreaming(kv_cache_compression=kv_cache_compression)
            guider = taomate_backend.patch_guider(guider)
            logging.warning("TaoMate-H3: using supplied sampler %s and %d-step sigma schedule; fixed streaming geometry, CPU clean-KV cache and SDPA. Audio teacher uses the supplied model/LoRA and model shifts; audio-first=%s.", getattr(getattr(sampler, "sampler_function", None), "__name__", type(sampler).__name__), len(sigmas) - 1, audio_first_enabled)
        if audio_sr and audio_vae is not None:
            AudioSR.ensure_installed()
        # Accept the former input name and value for existing API callers.
        compute_dtype = _deprecated_inputs.get("compute_dtype", compute_precision) if compute_precision == "default" else compute_precision
        if compute_dtype == "fp32 (full precision but more VRAM needed)":
            compute_dtype = "fp32"
        if compute_dtype not in ("default", "fp32"):
            raise ValueError(f"Unknown compute_precision: {compute_dtype!r}")
        if compute_dtype == "fp32":
            # ponytail: reuse native casting and a shallow guider copy; no duplicate model weights.
            guider = copy.copy(guider)
            guider.model_patcher = guider.model_patcher.clone()
            guider.model_patcher.set_model_compute_dtype(torch.float32)
            if taomate_backend is not None:
                taomate_backend.audio_teacher.patcher.set_model_compute_dtype(torch.float32)
            guider.model_options = guider.model_patcher.model_options
            logging.info("HR Endless Sampler diffusion compute dtype override: torch.float32 (video and audio).")

        # ComfyUI V3 stores hidden inputs on the per-execution class clone
        # instead of passing them as execute() arguments.  Keep the arguments
        # as a compatibility fallback for older ComfyUI releases.
        color_correction = {
            "chunk end-start": "chunk boundaries",
            "all chunks at once": "entire shots",
            "chunk end-start + all at once": "chunk boundaries + entire shots",
        }.get(color_correction, color_correction)
        if color_correction not in COLOR_CORRECTION_MODES:
            raise ValueError(f"Unknown color_correction mode: {color_correction!r}")
        correct_chunk_boundaries = color_correction in ("chunk boundaries", "chunk boundaries + entire shots")
        correct_all_chunks = color_correction in ("entire shots", "chunk boundaries + entire shots")
        linear_color_compute = bool(linear_color_compute)
        audio_feathered_overlap = bool(audio_feathered_overlap)
        # Keep sRGB images for optional director observation, while H3's
        # rebuilt Ref2VA/Qwen conditioning receives float32 inverse-gamma RGB.
        h3_images = _convert_image_transfer(images, _srgb_to_inverse_gamma_compute_rgb) if linear_color_compute else images
        hidden = getattr(cls, "hidden", None)
        if unique_id is None:
            unique_id = getattr(hidden, "unique_id", None)
        if dynprompt is None:
            dynprompt = getattr(hidden, "dynprompt", None)
        # Keep experimental allocator policy out of serialized workflow widgets.
        pytorch_memory_fraction = DEFAULT_PYTORCH_MEMORY_FRACTION
        _set_pytorch_memory_fraction(pytorch_memory_fraction, guider.model_patcher.load_device)
        # Keep the former experiment code available for development, but make
        # the released UI a single, unambiguous continuation method. Ignore
        # serialized legacy values too: an old workflow must not quietly enable
        # an experimental overlap, keyframe, Qwen-history, or preview-only path.
        prompt_preview_only = False
        context_keyframes_enable = False
        context_keyframes = 5
        guide_overlap_enable = False
        guide_overlap = 5
        video_continuation_enable = True
        qwen_full_history = False
        if pre_production is not None and not callable(getattr(pre_production, "create_session", None)):
            raise ValueError("pre_production must be a callable HR Endless pre-production provider")
        cache_gemma_preproduction = bool(getattr(pre_production, "cache_gemma_preproduction", False))
        gemma4_mtp = bool(getattr(pre_production, "gemma4_mtp", False))
        if video_continuation_method not in VIDEO_CONTINUATION_METHODS:
            raise ValueError(
                f"Unknown video_continuation_method {video_continuation_method!r}; choose one of "
                + ", ".join(VIDEO_CONTINUATION_METHODS)
            )
        if video_continuation_res not in VIDEO_CONTINUATION_RESOLUTIONS:
            raise ValueError(
                f"Unknown video_continuation_res {video_continuation_res!r}; choose one of "
                + ", ".join(VIDEO_CONTINUATION_RESOLUTIONS)
            )
        debug_start_chunk = int(debug_start_chunk)
        debug_stop_chunk = int(debug_stop_chunk)
        samples = latent_image["samples"]
        if not samples.is_nested:
            if prompt_preview_only:
                raise ValueError("prompt_preview_only requires a MiniMax H3 nested video/audio latent")
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(
                sampled[0], sampled[1], "", normalize_timeline(None, fps=fps, total_frames=0), None, None
            )

        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24 or streams[1].ndim != 4 or streams[1].shape[1] != 32:
            if prompt_preview_only:
                raise ValueError("prompt_preview_only requires MiniMax H3 24-channel video and 32-channel audio latents")
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(
                sampled[0], sampled[1], "", normalize_timeline(None, fps=fps, total_frames=0), None, None
            )

        video, audio = streams
        context_keyframes = int(context_keyframes_enable) * context_keyframes
        guide_overlap = int(guide_overlap_enable) * guide_overlap
        video_continuation = int(video_continuation_enable) * video_continuation
        max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
        requested_video_continuation = video_continuation
        context_keyframes, guide_overlap, video_continuation, guide_video_t = _continuation_controls(
            context_keyframes,
            guide_overlap,
            video_continuation,
            max_chunk_frames,
        )
        if video_continuation != requested_video_continuation:
            logging.info(
                "HR Endless Sampler clamped video_continuation from %d to the effective chunk size %d.",
                requested_video_continuation,
                video_continuation,
            )
        warm_start_video_t = _video_steps(guide_overlap) if guide_overlap else 0
        keyframe_duration_frames = context_keyframes
        use_video_continuation = video_continuation > 0 and not use_taomate
        use_masked_av_mode = (
            use_video_continuation
            and video_continuation_method == VIDEO_CONTINUATION_METHOD_MASKED_AV
        )
        use_masked_av_overlap = (
            ENABLE_MASKED_AV_OVERLAP
            and use_masked_av_mode
        )
        if use_masked_av_mode and not ENABLE_MASKED_AV_OVERLAP:
            logging.info(
                "HR Endless Sampler long masked AV overlap is temporarily disabled; using only a protected "
                "native AV boundary keyframes, without full Video1 or Audio1 references."
            )
        # Both methods retain their packing prefix in outputs and previews.
        keep_continuation_prefix = use_video_continuation and (not use_masked_av_mode or not TRIM_MASKED_AV_PREFIX)
        if keep_continuation_prefix:
            logging.warning(
                "HR Endless Sampler prefix inspection is active: retaining each "
                "video/audio prefix in its own chunk instead of trimming the repeated AV tail."
            )
        if use_masked_av_mode and video_continuation >= max_chunk_frames:
            raise ValueError(
                f"video_continuation ({video_continuation}) must be smaller than the effective chunk size "
                f"({max_chunk_frames}) for {VIDEO_CONTINUATION_METHOD_MASKED_AV}"
            )
        include_video1_reference = (
            use_video_continuation
            and not use_masked_av_mode
            and INCLUDE_VIDEO1_REFERENCE
        )
        # The native audio boundary keyframe replaces the broad <Audio N>
        # reference in masked packing mode; it carries no prompt-visible role.
        include_previous_audio_reference = use_video_continuation and not use_masked_av_mode
        # A multi-frame MiniMax keyframe is anchored on the target timeline; it
        # is not detached historical memory. Keep the same completed frames in
        # the opening physical target interval and trim that truthful overlap
        # after sampling. Native masked mode uses the selected continuation
        # duration as its synthetic prefix; Video1 keeps its five-frame phase.
        if use_taomate:
            plan = taomate_backend.request_plan(video.shape[2], audio.shape[-1], chunk_frames, video_continuation)
        elif use_masked_av_overlap:
            plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames, video_continuation)
        elif context_keyframes:
            plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames, context_keyframes)
        else:
            plan = _chunk_plan_without_overlap(
                video.shape[2], audio.shape[-1], chunk_frames,
                video_continuation if use_masked_av_mode else 5,
            )
        if debug_stop_chunk > len(plan):
            raise ValueError(f"debug_stop_chunk is {debug_stop_chunk}, but this latent has only {len(plan)} chunks")
        if debug_start_chunk > len(plan):
            raise ValueError(f"debug_start_chunk is {debug_start_chunk}, but this latent has only {len(plan)} chunks")
        if debug_start_chunk and debug_stop_chunk and debug_start_chunk > debug_stop_chunk:
            raise ValueError("debug_start_chunk cannot be greater than debug_stop_chunk")
        active_plan = plan if debug_stop_chunk == 0 else plan[:debug_stop_chunk]
        _gemma_markers, gemma_shots, _gemma_description_end = _parse_prompt_shots(prompt, plan[-1]["frame_end"], fps)
        manual_prompts = pre_production if callable(getattr(pre_production, "get_chunk_prompt", None)) else None
        if manual_prompts is not None and len(manual_prompts.prompts()) != len(plan):
            raise ValueError(f"Manual prompts contain {len(manual_prompts.prompts())} chunks, but this render needs {len(plan)} {'TaoMate chunks' if use_taomate else 'chunks'}. Regenerate prompts for this layout.")
        gemma_director_needed = pre_production is not None and manual_prompts is None and bool(gemma_shots)

        original_conds = guider.original_conds
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError("HR Endless Sampler requires a standard guider with positive conditioning")
        ref2va = bool(positive[0].get("minimax_refs"))
        if len(active_plan) > 1 and (include_video1_reference or include_previous_audio_reference or qwen_full_history) and not ref2va:
            raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")
        original_refs = positive[0].get("minimax_refs", ())
        linear_original_refs = None
        if linear_color_compute and any(ref.get("kind") == "image" for ref in original_refs):
            # Ref2VA's DiT payload is distinct from Qwen's image tokens. Build
            # matching linear VAE latents once and install them per video below.
            linear_original_refs = _linear_ref2va_image_refs(original_refs, images, vae)
        video_number = 1 + sum(ref["kind"] in ("video", "video_audio") for ref in original_refs)
        audio_number = 1 + sum(ref["kind"] in ("audio", "video_audio") for ref in original_refs)
        picture_number = 1 + sum(ref["kind"] == "image" for ref in original_refs)
        planned_prompts = _planned_chunk_prompts(
            prompt,
            plan,
            active_plan,
            fps,
            video_continuation if use_masked_av_overlap else context_keyframes,
            include_video1_reference,
            include_previous_audio_reference,
            ref2va,
            video_number,
            audio_number,
            legacy=pre_production is None or (use_taomate and audio_first_enabled),
            include_dialogue_prefix=not (use_taomate and audio_first_enabled),
        )
        if manual_prompts is not None:
            planned_prompts = [(manual_prompts.get_chunk_prompt(index + 1), _debug_chunk_prompt(index, chunk, chunk["frame_start"] + chunk.get("output_trim_frames", 0), manual_prompts.get_chunk_prompt(index + 1))) for index, chunk in enumerate(active_plan)]
        elif use_taomate and audio_first_enabled:
            planned_prompts = [(_preserve_global_prompt_sections(_chunk_summary_prompt(item[0], prompt, index > 0), prompt), item[1]) for index, item in enumerate(planned_prompts)]
        if debug:
            logging.info(
                "HR Endless Sampler independent continuation controls: "
                "context_keyframes=%d, guide_overlap=%d, video_continuation=%d, "
                "video_continuation_method=%s, video_continuation_res=%s",
                context_keyframes,
                guide_overlap,
                video_continuation,
                video_continuation_method,
                video_continuation_res,
            )
            if use_video_continuation and not use_masked_av_overlap and not include_video1_reference:
                logging.info(
                    "HR Endless Sampler Video1 isolation experiment: "
                    "native visual boundary keyframe enabled; Qwen/DiT/prompt Video1 reference disabled"
                )
        if prompt_preview_only:
            prompt_preview = "\n\n".join(debug_prompt for _chunk_prompt, debug_prompt in planned_prompts)
            if debug:
                logging.info(
                    "HR Endless Sampler prompt-preview-only execution; sampling skipped:\n%s",
                    prompt_preview,
                )
            preview_timeline = normalize_timeline(
                {
                    "fps": fps,
                    "total_frames": active_plan[-1]["frame_end"],
                    "chunks": _preview_ranges_for_plan(
                        active_plan,
                        keep_physical_prefix=keep_continuation_prefix,
                    ),
                    "shots": _preview_shot_ranges(prompt, plan[-1]["frame_end"], active_plan[-1]["frame_end"], fps),
                },
                fps=fps,
                total_frames=active_plan[-1]["frame_end"],
            )
            return io.NodeOutput(latent_image, latent_image, prompt_preview, preview_timeline, None, None)

        if len(active_plan) > 1 and "noise_mask" in latent_image:
            raise ValueError("HR Endless Sampler does not support denoise masks when chunking")
        if len(active_plan) > 1 and (include_video1_reference or qwen_full_history or gemma_director_needed):
            if vae is None:
                raise ValueError("video_continuation, qwen_full_history, and Gemma chunk directing require a MiniMax H3 video VAE")

        replay_cache = None
        replay_start_index = 0
        replay_prior_chunks = []
        replay_cached_suffix = []
        replay_prefix_noises = {}
        replay_timing_plan = None
        replay_prompt_changed = False
        replay_cache_enabled = _replay_cache_enabled() and not use_taomate
        if not replay_cache_enabled and debug_start_chunk:
            logging.info(
                "HR Endless Sampler replay cache is disabled; ignoring debug_start_chunk=%d and sampling from Chunk 1.",
                debug_start_chunk,
            )
            debug_start_chunk = 0
        if not replay_cache_enabled and not use_taomate:
            logging.info(
                "HR Endless Sampler replay cache is disabled; this run will ignore the old cache and record a fresh replacement."
            )
        replay_fingerprint = _replay_fingerprint(
            video,
            audio,
            plan,
            fps=fps,
            chunk_frames=max_chunk_frames,
            context_keyframes=context_keyframes,
            guide_overlap=guide_overlap,
            video_continuation=video_continuation,
            video_continuation_method=video_continuation_method,
            video_continuation_res=video_continuation_res,
            ref2va=ref2va,
        )
        replay_fingerprint["color_correction"] = color_correction
        replay_fingerprint["linear_color_compute"] = linear_color_compute
        replay_fingerprint["audio_feathered_overlap"] = audio_feathered_overlap
        replay_fingerprint["av_feather_version"] = 2
        replay_fingerprint["minimax_visual_cond_noise_aug"] = minimax_visual_cond_noise_aug
        replay_fingerprint["pre_production"] = pre_production.fingerprint() if pre_production is not None else {"provider": "legacy-static-camera-v3"}
        # Precision comparisons must never restore chunks from a different compute mode.
        replay_fingerprint["compute_dtype"] = compute_dtype
        replay_fingerprint["video1_first_frame_picture"] = True
        replay_fingerprint["keep_continuation_prefix"] = keep_continuation_prefix
        replay_fingerprint["chunk_summary_policy"] = "original-opening-current-continuation-v1"
        replay_fingerprint["audio_sr"] = bool(audio_sr)
        replay_fingerprint["prefix_noise_source"] = "full_sequence_tail_v1" if TOGGLE_SINGLE_NOISE else "per_chunk_seed_v1"
        replay_cached_initial = None
        auto_resumed = False
        if replay_cache_enabled and debug_start_chunk == 0:
            candidate_cache = _LastRunReplayCache()
            loaded_cache, _cache_reason = candidate_cache.load_if_compatible(replay_fingerprint)
            if loaded_cache is not None:
                automatic_start = candidate_cache.automatic_resume_chunk(
                    loaded_cache["manifest"], len(active_plan)
                )
                if automatic_start is not None:
                    debug_start_chunk = automatic_start
                    auto_resumed = True
                    logging.warning(
                        "HR Endless Sampler automatically resuming the compatible interrupted render from Chunk %d "
                        "(checkpointed through Chunk %d).",
                        debug_start_chunk,
                        debug_start_chunk - 1,
                    )
                else:
                    # A completed run, an intentional debug stop, or a
                    # manifest from before lifecycle tracking is a new render
                    # rather than an automatic continuation candidate.
                    candidate_cache.clear()
        if replay_cache_enabled and debug_start_chunk:
            candidate_cache = _LastRunReplayCache()
            loaded_cache, cache_reason = candidate_cache.load_if_compatible(replay_fingerprint)
            required_prior_numbers = range(1, debug_start_chunk)
            if loaded_cache is not None and all(candidate_cache.has_chunk(number) for number in required_prior_numbers):
                try:
                    missing_initial = {
                        "video", "audio", "video_noise", "audio_noise", "noise_seed",
                    } - set(loaded_cache["initial"])
                    if missing_initial:
                        raise KeyError("initial cache is missing " + ", ".join(sorted(missing_initial)))
                    replay_prior_chunks = [
                        candidate_cache.load_chunk(number)
                        for number in required_prior_numbers
                    ]
                    # Prefix noise is independent of the full latent's normal
                    # noise slice. Preserve it too, including when replaying a
                    # run with a different current UI seed.
                    for number in range(debug_start_chunk, len(active_plan) + 1):
                        if candidate_cache.has_chunk(number):
                            state = candidate_cache.load_chunk(number)
                            if state.get("prefix_video_noise") is not None:
                                replay_prefix_noises[number - 1] = (
                                    state["prefix_video_noise"],
                                    state["prefix_audio_noise"],
                                )
                    if gemma_director_needed and candidate_cache.timing_path.is_file():
                        replay_timing_plan = candidate_cache.load_timing_plan()
                    replay_cached_initial = loaded_cache["initial"]
                    replay_cache = candidate_cache
                    replay_start_index = debug_start_chunk - 1
                    replay_cache.begin_from(debug_start_chunk)
                    logging.info(
                        "HR Endless Sampler %s: restoring cached state through Chunk %d and rerunning from Chunk %d.",
                        "automatic recovery" if auto_resumed else "replay",
                        debug_start_chunk - 1,
                        debug_start_chunk,
                    )
                    current_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    if loaded_cache["manifest"].get("source_prompt_sha256") != current_prompt_hash:
                        replay_prompt_changed = True
                        # The cached physical predecessor is still the exact
                        # desired visual/noise boundary, but its production
                        # schedule describes an older source prompt. Rebuild
                        # the schedule from the edited prompt before any
                        # replayed chunk is directed.
                        replay_timing_plan = None
                        logging.info(
                            "HR Endless Sampler replay: the source prompt changed; retaining the cached physical "
                            "state but rebuilding Gemma preproduction for the edited prompt."
                        )
                    if replay_timing_plan is not None:
                        logging.info("HR Endless Sampler replay: reusing the cached Gemma preproduction timing plan.")
                    # A right-click cache deletion leaves later completed
                    # chunks intact. Reuse that suffix only when it reaches
                    # this run's end and the VAE can reconstruct authoritative
                    # frames from its latent checkpoints. The MP4 proxies are
                    # deliberately never accepted as render output.
                    if not replay_prompt_changed and vae is not None:
                        suffix_numbers = list(range(debug_start_chunk + 1, len(active_plan) + 1))
                        if suffix_numbers and all(candidate_cache.has_chunk(number) for number in suffix_numbers):
                            replay_cached_suffix = [candidate_cache.load_chunk(number) for number in suffix_numbers]
                            logging.info(
                                "HR Endless Sampler replay will sample Chunk %d only and VAE-restore cached Chunks %d-%d.",
                                debug_start_chunk,
                                debug_start_chunk + 1,
                                len(active_plan),
                            )
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    logging.warning(
                        "HR Endless Sampler replay cache could not restore Chunk %d; recording a fresh baseline from Chunk 1: %s",
                        debug_start_chunk,
                        error,
                    )
                    replay_prior_chunks = []
                    replay_cached_suffix = []
                    replay_prefix_noises = {}
                    replay_timing_plan = None
                    replay_cached_initial = None
                    replay_cache = None
                    replay_start_index = 0
                    candidate_cache.clear()
            else:
                logging.info(
                    "HR Endless Sampler replay: %s; recording a fresh baseline from Chunk 1 before Chunk %d can be replayed.",
                    cache_reason or "the cache does not contain every preceding chunk",
                    debug_start_chunk,
                )
                candidate_cache.clear()

        gemma_prompt_log = _begin_last_gemma_prompt_log(
            max_chunk_frames,
            context_keyframes,
            guide_overlap,
            video_continuation,
            video_continuation_method,
            video_continuation_res,
            fps,
            len(active_plan),
            cache_gemma_preproduction=cache_gemma_preproduction,
            gemma4_mtp=gemma4_mtp,
        )
        reset_gemma4_raw_output_log(bool(debug and gemma_director_needed))
        reset_gemma4_live_output_log()
        gemma_image_log = _reset_last_gemma_image_log()
        timing = _SamplerTiming(guider.model_patcher.load_device)
        fixed_latent = latent_image.copy()
        fixed_latent["samples"] = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            samples,
            latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"),
        )
        if replay_cached_initial is not None:
            try:
                source_device = video.device
                video = replay_cached_initial["video"].to(device=source_device, dtype=video.dtype)
                audio = replay_cached_initial["audio"].to(device=source_device, dtype=audio.dtype)
                video_noise = replay_cached_initial["video_noise"].to(device=source_device, dtype=video.dtype)
                audio_noise = replay_cached_initial["audio_noise"].to(device=source_device, dtype=audio.dtype)
                fixed_latent["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
                full_noise = comfy.nested_tensor.NestedTensor((video_noise, audio_noise))
                replay_noise_seed = int(replay_cached_initial["noise_seed"])
            except (KeyError, RuntimeError, ValueError) as error:
                raise RuntimeError(f"HR Endless Sampler replay cache has invalid initial tensors: {error}") from error
        else:
            full_noise = noise.generate_noise(fixed_latent)
            replay_noise_seed = int(noise.seed)
        if not full_noise.is_nested or len(full_noise.unbind()) != 2:
            raise ValueError("HR Endless Sampler expected nested video and audio noise")
        if replay_cached_initial is None:
            video_noise, audio_noise = full_noise.unbind()
        # The cache toggle controls reuse only. Every multi-chunk run records
        # a fresh checkpoint so disabling reuse is also a simple way to
        # replace an unwanted interrupted cache with the current render.
        if len(active_plan) > 1 and replay_cache is None and not use_taomate:
            candidate_cache = _LastRunReplayCache()
            try:
                candidate_cache.create(
                    replay_fingerprint,
                    prompt,
                    {
                        "video": video,
                        "audio": audio,
                        "video_noise": video_noise,
                        "audio_noise": audio_noise,
                        "noise_seed": replay_noise_seed,
                    },
                )
                replay_cache = candidate_cache
            except (OSError, RuntimeError, ValueError) as error:
                logging.warning(
                    "HR Endless Sampler could not create its automatic recovery checkpoint; "
                    "sampling will continue, but an interrupted render cannot resume: %s",
                    error,
                )

        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        output_video = []
        output_audio = []
        denoised_video = []
        denoised_audio = []
        # Authoritative full-VAE media is accumulated on CPU as each chunk is
        # finalized. This reuses the preview/Gemma decode and avoids a second
        # whole-video VAE pass downstream. A failed or disconnected VAE makes
        # only its corresponding decoded output unavailable; latent sampling
        # and recovery continue normally.
        decoded_output_frames = []
        # Keep inverse-gamma compute frames for H3 re-encode paths. Seam/shot
        # matching temporarily converts these float32 frames to display sRGB.
        decoded_linear_output_frames = [] if linear_color_compute else decoded_output_frames
        decoded_output_audio = []
        decoded_video_complete = vae is not None
        decoded_audio_complete = audio_vae is not None
        decoded_audio_sample_rate = None
        audio_first_decoded = False
        audio_teacher_preview_chunks = {}
        enhanced_audio_output = None
        previous_video = None
        previous_audio = None
        previous_frame_count = None
        # The raw full video-VAE decode of the most recently completed
        # physical chunk. It is reused only where H3/Gemma need original
        # pixels; finalized browser/output frames use a separate corrected
        # copy below. Video1 keeps its bounded-latent decode path.
        previous_decoded_frames = None
        # The matching finalized display-corrected decode. It is retained only
        # for the optional pixel-space H3 context experiment below.
        previous_color_corrected_decoded_frames = None
        # Only promote this after a stock sampler call succeeds. The next
        # Gemma request can then pair the exact prior directed description with
        # stills from the same rendered chunk, never with an unsampled plan.
        previous_gemma_description = None
        previous_h3_prompt = None
        previous_gemma_timing_plan = None
        previous_gemma_end_state = None
        previous_gemma_last_seen_character_state = None
        color_diagnostics = {}
        entire_shot_color_anchors = {}
        output_template = None
        denoised_template = None
        completed_chunks = 0
        sampling_completed = False
        debug_prompts = []
        return_prompts = True
        replay_output_on_cpu = replay_start_index > 0
        if replay_prior_chunks:
            try:
                for state in replay_prior_chunks:
                    masked_audio_prefix = state.get("masked_audio_prefix")
                    masked_denoised_audio_prefix = state.get("masked_denoised_audio_prefix")
                    if masked_audio_prefix is not None:
                        _replace_stream_tail(output_audio, masked_audio_prefix)
                    if masked_denoised_audio_prefix is not None:
                        _replace_stream_tail(denoised_audio, masked_denoised_audio_prefix)
                    output_video.append(state["output_video"])
                    output_audio.append(state["output_audio"])
                    denoised_video.append(state["denoised_video"])
                    denoised_audio.append(state["denoised_audio"])
                    if state.get("debug_prompt"):
                        debug_prompts.append(str(state["debug_prompt"]))
                previous_state = replay_prior_chunks[-1]
                previous_video = previous_state["sampled_video"].to(device=video.device, dtype=video.dtype)
                previous_audio = previous_state["sampled_audio"].to(device=audio.device, dtype=audio.dtype)
                previous_frame_count = int(previous_state["previous_frame_count"])
                previous_gemma_description = previous_state.get("gemma_description")
                previous_h3_prompt = previous_state.get("h3_prompt")
                previous_gemma_timing_plan = previous_state.get("gemma_timing_plan")
                previous_gemma_end_state = previous_state.get("gemma_end_state")
                previous_gemma_last_seen_character_state = previous_state.get("gemma_last_seen_character_state")
                if replay_prompt_changed:
                    # These were authored against the old source prompt. The
                    # predecessor still remains available as chronological
                    # rendered stills, which are more reliable evidence for
                    # the first rerun chunk than stale textual instructions.
                    previous_gemma_description = None
                    previous_h3_prompt = None
                    previous_gemma_timing_plan = None
                    previous_gemma_end_state = None
                    previous_gemma_last_seen_character_state = None
                    logging.info(
                        "HR Endless Sampler replay: discarded stale prior Gemma text; "
                        "the edited plan will use the retained predecessor frames as evidence."
                    )
                output_template = previous_state.get("output_template")
                denoised_template = previous_state.get("denoised_template")
                completed_chunks = replay_start_index
            except (KeyError, RuntimeError, ValueError) as error:
                raise RuntimeError(f"HR Endless Sampler replay cache has invalid completed chunk state: {error}") from error
        gemma_director = (
            pre_production.create_session(
                debug=debug,
                seed=replay_noise_seed,
                observation_image_directory=gemma_image_log,
                render_context={"prompt": prompt, "fps": fps, "chunk_layout": plan, "latent": latent_image, "guider": guider, "images": images, "clip": clip, "vae": vae, "audio_vae": audio_vae},
            )
            if gemma_director_needed else None
        )
        if gemma_director is not None:
            logging.info(
                "HR Endless Sampler Gemma 4 mode: %s; fixed sampler-derived seed %d.",
                "native draft-MTP (4 draft tokens)" if gemma4_mtp else "original non-MTP decoding",
                replay_noise_seed & 0x7fffffff,
            )
        gemma_preproduction_timing_plan = None
        gemma_preproduction_cache = None
        gemma_preproduction_cache_ready = False
        gemma_preproduction_seconds = 0.0
        if gemma_director_needed:
            # This cache is render-local. Clear an earlier render's state even
            # when the toggle is now off, so a subsequent worker can never
            # accidentally inherit a stale source prompt or timing plan.
            stale_cache = Gemma4PreproductionCache()
            try:
                if cache_gemma_preproduction:
                    stale_cache.reset()
                    gemma_preproduction_cache = stale_cache
                    logging.info(
                        "HR Endless Sampler Gemma 4 clean preproduction KV cache enabled at %s.",
                        stale_cache.root,
                    )
                else:
                    stale_cache.clear()
            except OSError as error:
                logging.warning(
                    "HR Endless Sampler could not prepare the Gemma preproduction KV cache; "
                    "continuing without it: %s",
                    error,
                )
        gemma_system_logged = False
        preview_chunk_ranges = _preview_ranges_for_plan(
            active_plan,
            keep_physical_prefix=keep_continuation_prefix,
            show_replaced_tail=bool(use_masked_av_mode and TRIM_MASKED_AV_PREFIX),
        )
        for preview_range, planned_prompt in zip(preview_chunk_ranges, planned_prompts):
            preview_range["subtitle"] = _preview_subtitle(planned_prompt[0])
            preview_range["h3_prompt"] = planned_prompt[0].strip()
            if use_taomate and audio_first_enabled:
                preview_range["audio_teacher_prompt"] = planned_prompt[0].strip()
        if use_taomate:
            for preview_range, taomate_chunk in zip(preview_chunk_ranges, active_plan):
                preview_range["taomate_audio_first"] = bool(audio_first_enabled)
                preview_range["taomate_completed_frames"] = 0
                preview_range["taomate_phase_count"] = len(taomate_chunk.get("phases", (taomate_chunk,)))
                preview_range["taomate_phase_work"] = [
                    phase["frame_end"] - TaoMateStreaming.frames(phase["video_start"])
                    for phase in taomate_chunk.get("phases", (taomate_chunk,))
                ]
        for index, state in enumerate(replay_prior_chunks):
            description = state.get("gemma_description")
            if isinstance(description, str) and description.strip() and index < len(preview_chunk_ranges):
                preview_chunk_ranges[index]["gemma_detailed_description"] = description.strip()
                if INCLUDE_PER_CHUNK_RETENTION_ANALYSIS:
                    retention_analysis = _chunk_retention_analysis(state.get("gemma_retention_analysis"))
                    if retention_analysis:
                        preview_chunk_ranges[index]["gemma_retention_analysis"] = retention_analysis
            h3_prompt = state.get("h3_prompt")
            if isinstance(h3_prompt, str) and h3_prompt.strip() and index < len(preview_chunk_ranges):
                preview_chunk_ranges[index]["h3_prompt"] = h3_prompt.strip()
            if index < len(preview_chunk_ranges):
                for key in (
                    "h3_render_seconds",
                    "gemma_seconds",
                    "gemma_preproduction_seconds",
                    "chunk_total_seconds",
                ):
                    value = state.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                        preview_chunk_ranges[index][key] = float(value)
        preview_end = active_plan[-1]["frame_end"]
        preview_shot_ranges = _preview_shot_ranges(prompt, plan[-1]["frame_end"], preview_end, fps)
        reused_chunk_numbers = [
            *range(1, len(replay_prior_chunks) + 1),
            *range(len(active_plan) - len(replay_cached_suffix) + 1, len(active_plan) + 1),
        ]
        preview_execution = begin_preview_execution(
            guider.model_patcher,
            preview_chunk_ranges,
            preview_shot_ranges,
            reusing_cached_chunks=bool(reused_chunk_numbers),
            cached_chunk_count=len(reused_chunk_numbers),
            reused_chunk_numbers=reused_chunk_numbers,
        )
        intermediate_writer = None
        intermediate_prefix = _connected_save_video_prefix(dynprompt, unique_id)
        if intermediate_prefix is not None and vae is not None:
            try:
                intermediate_writer = IntermediateChunkVideoWriter(
                    intermediate_prefix,
                    fps,
                    len(active_plan),
                    width,
                    height,
                    fresh_run=replay_start_index == 0,
                )
                logging.info(
                    "HR Endless Sampler will save temporary finalized chunk videos using prefix %s.",
                    intermediate_prefix,
                )
                intermediate_writer.register_preview(preview_execution)
            except (OSError, RuntimeError, ValueError) as error:
                logging.warning(
                    "HR Endless Sampler could not prepare temporary chunk-video output; "
                    "sampling will continue normally: %s",
                    error,
                )
                intermediate_writer = None
        if preview_execution is not None and audio_vae is None:
            logging.warning(
                "HR Endless Sampler Preview will replace completed chunks with full video-VAE frames, "
                "but audio preview is disabled because audio_vae is not connected to HR Endless Sampler."
            )
        if replay_prior_chunks:
            logging.info(
                "HR Endless Sampler restoring %d completed cached chunks%s.",
                len(replay_prior_chunks),
                " from finalized decoded media when available" if vae is not None else " with latent preview decoding only",
            )
            restored_preview_chunks = []
            for restored_index, state in enumerate(replay_prior_chunks):
                restored_plan = active_plan[restored_index]
                restored_range = preview_chunk_ranges[restored_index]
                restored_preview_chunks.append({
                    "index": restored_index,
                    "video": state["sampled_video"],
                    "sampled_start": restored_plan["frame_start"],
                    "sampled_end": restored_plan["frame_end"] - 1,
                    "output_start": restored_range["start"],
                    "output_end": restored_range["end"],
                    "trim_steps": 0 if keep_continuation_prefix else restored_plan.get("context_video_t", 0),
                    "gemma_detailed_description": restored_range.get("gemma_detailed_description"),
                    "gemma_retention_analysis": restored_range.get("gemma_retention_analysis"),
                    "h3_prompt": restored_range.get("h3_prompt"),
                })
            if vae is None:
                decoded_video_complete = False
                decoded_audio_complete = False
                if preview_execution is not None:
                    preview_execution.restore_chunks(
                        restored_preview_chunks,
                        guider.model_patcher.model.latent_format,
                    )
            else:
                # MP4 is only the dormant browser preview. Rebuild exact float
                # frames and audio from the authoritative latent checkpoints.
                comfy.model_management.unload_model_and_clones(guider.model_patcher)
                comfy.model_management.unload_model_and_clones(clip.patcher)
                comfy.model_management.soft_empty_cache(force=True)
                latent_fallbacks = []
                for restored_index, (state, restored) in enumerate(
                    zip(replay_prior_chunks, restored_preview_chunks)
                ):
                    restored_plan = active_plan[restored_index]
                    if preview_execution is not None:
                        preview_execution.set_phase(
                            f"Restoring cached final preview Chunk {restored_index + 1}/{len(replay_prior_chunks)}",
                            chunk=restored_index,
                        )
                    try:
                        correction_overlap = max(0, int(restored_plan.get("output_trim_frames", 0)))
                        keep_masked_av_prefix = (
                            keep_continuation_prefix and restored_index > 0
                        )
                        (
                            decoded_frames,
                            _unused_retained_frames,
                            restored_audio,
                            restored_audio_rate,
                            restored_overlap_audio,
                        ) = _decode_replay_preview_media(
                            vae,
                            audio_vae,
                            state["sampled_video"],
                            state.get("sampled_audio"),
                            output_trim_frames=restored_plan.get("output_trim_frames", 0),
                            context_audio_t=(
                                0 if restored_index == 0
                                or (keep_continuation_prefix)
                                else restored_plan.get("context_audio_t", 0)
                            ),
                            output_frames=restored_plan["frame_end"] - restored_plan["frame_start"] - correction_overlap,
                            fps=fps,
                            masked_audio_overlap_frames=(
                                restored_plan.get("output_trim_frames", 0)
                                if use_masked_av_overlap and TRIM_MASKED_AV_PREFIX and restored_index > 0
                                else 0
                            ),
                            replace_audio_tail_ticks=bool(use_masked_av_mode and TRIM_MASKED_AV_PREFIX and restored_index > 0),
                            keep_masked_av_prefix=keep_masked_av_prefix,
                        )
                        retained_start = 0 if keep_masked_av_prefix else correction_overlap
                        decoded_display_frames = _convert_image_transfer(decoded_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else decoded_frames
                        correction_reference = _decoded_frame_before_tail(decoded_linear_output_frames, retained_start)
                        correction_reference = _convert_image_transfer(correction_reference, _inverse_gamma_compute_to_srgb) if linear_color_compute else correction_reference
                        correction_frame_count = _same_shot_correction_frames(
                            gemma_shots,
                            restored["output_start"],
                            restored["output_end"] + 1,
                        )
                        corrected_frames, color_transform, corrected_frame_count = _correct_decoded_chunk_color(
                            correction_reference,
                            decoded_display_frames,
                            0,
                            correction_frame_count + (retained_start if correction_frame_count else 0),
                            enabled=correct_chunk_boundaries,
                        )
                        if correct_all_chunks:
                            corrected_frames = _correct_ready_entire_shot_frames(
                                corrected_frames,
                                restored["output_start"],
                                gemma_shots,
                                gemma_preproduction_timing_plan,
                                entire_shot_color_anchors,
                            )
                        corrected_frames = _convert_image_transfer(corrected_frames, _srgb_to_inverse_gamma_compute_rgb) if linear_color_compute else corrected_frames
                        retained_count = restored["output_end"] - restored["output_start"] + 1
                        retained_linear_frames = corrected_frames[retained_start:retained_start + retained_count].clone()
                        retained_frames = _convert_image_transfer(retained_linear_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else retained_linear_frames
                        _log_output_color_correction(
                            "restored Chunk %d" % (restored_index + 1),
                            color_transform,
                            corrected_frame_count,
                        )
                        restored_preview_start = restored["output_start"]
                        restored_preview_end = restored["output_end"]
                        if retained_start and decoded_linear_output_frames:
                            _replace_decoded_frame_tail(decoded_linear_output_frames, corrected_frames[:retained_start])
                            if linear_color_compute:
                                _replace_decoded_frame_tail(decoded_output_frames, _convert_image_transfer(corrected_frames[:retained_start], _inverse_gamma_compute_to_srgb))
                            if preview_execution is not None:
                                previous_range = preview_chunk_ranges[restored_index - 1]
                                preview_execution.replace_video_tail(
                                    restored_index - 1, decoded_output_frames[-1], previous_range["start"], previous_range["end"],
                                    gemma_detailed_description=previous_range.get("gemma_detailed_description"),
                                    gemma_retention_analysis=previous_range.get("gemma_retention_analysis"),
                                )
                        decoded_linear_output_frames.append(retained_linear_frames)
                        if linear_color_compute:
                            decoded_output_frames.append(retained_frames)
                        restored_color_diagnostics = _safe_video_color_diagnostics(
                            retained_frames,
                            restored_index + 1,
                        )
                        if restored_color_diagnostics is not None:
                            _record_color_diagnostics(
                                color_diagnostics,
                                restored_index,
                                restored_color_diagnostics,
                                _color_boundary_context(gemma_shots, restored_preview_start),
                            )
                        if restored_audio is None or restored_audio_rate is None:
                            decoded_audio_complete = False
                        else:
                            if decoded_audio_sample_rate is None:
                                decoded_audio_sample_rate = int(restored_audio_rate)
                            elif decoded_audio_sample_rate != int(restored_audio_rate):
                                raise ValueError(
                                    "cached chunk audio sample rate changed from "
                                    f"{decoded_audio_sample_rate} to {int(restored_audio_rate)}"
                                )
                            if restored_overlap_audio is not None:
                                _replace_stream_tail(decoded_output_audio, restored_overlap_audio)
                            decoded_output_audio.append(restored_audio.clone())
                        restored_audio, restored_audio_rate, restored_overlap_audio = _enhance_decoded_audio(restored_audio, restored_audio_rate, restored_overlap_audio, enabled=audio_sr, device=guider.model_patcher.load_device, seed=(replay_noise_seed + restored_index) % (2 ** 32))
                        if preview_execution is not None:
                            preview_execution.finalize_chunk(
                                restored_index,
                                retained_frames,
                                restored_preview_start,
                                restored_preview_end,
                                audio_waveform=restored_audio,
                                audio_sample_rate=restored_audio_rate,
                                gemma_detailed_description=restored.get("gemma_detailed_description"),
                                gemma_retention_analysis=restored.get("gemma_retention_analysis"),
                            )
                            if restored_overlap_audio is not None and restored_audio_rate is not None:
                                preview_execution.replace_audio_tail(
                                    restored_index,
                                    restored_overlap_audio,
                                    restored_audio_rate,
                                )
                        if intermediate_writer is not None:
                            intermediate_writer.submit(
                                restored_index,
                                retained_frames,
                                restored_preview_start,
                                restored_preview_end,
                                chunk_metadata=preview_chunk_ranges[restored_index],
                                shot_ranges=preview_shot_ranges,
                                audio_waveform=restored_audio,
                                audio_sample_rate=restored_audio_rate,
                            )
                        # The immediately preceding physical chunk is also the
                        # next Gemma observation source. Retain only that final
                        # CPU decode and avoid decoding it a second time.
                        if restored_index + 1 == len(replay_prior_chunks):
                            # Keep both prior pixel versions for the next
                            # chunk: raw remains the normal H3/Gemma source;
                            # the corrected copy is only for the explicit H3
                            # color-conditioning experiment.
                            previous_decoded_frames = decoded_frames
                            previous_color_corrected_decoded_frames = corrected_frames
                        else:
                            decoded_frames = None
                    except Exception as error:
                        decoded_video_complete = False
                        decoded_audio_complete = False
                        logging.warning(
                            "HR Endless Sampler could not restore cached Chunk %d with the full VAE; "
                            "using its latent preview while sampling continues: %s",
                            restored_index + 1,
                            error,
                        )
                        if preview_execution is not None:
                            latent_fallbacks.append(restored)
                if latent_fallbacks and preview_execution is not None:
                    preview_execution.restore_chunks(
                        latent_fallbacks,
                        guider.model_patcher.model.latent_format,
                    )
                comfy.model_management.unload_model_and_clones(vae.patcher)
                if audio_vae is not None:
                    comfy.model_management.unload_model_and_clones(audio_vae.patcher)
            restored_preview_chunks = None
            # Replay restoration is intentionally released before Gemma/Qwen/
            # H3 loads. Return its freed blocks to the allocator.
            comfy.model_management.soft_empty_cache(force=True)
            logging.info(
                "HR Endless Sampler restored completed cached final preview Chunks 1-%d.",
                len(replay_prior_chunks),
            )
        preparation_message = (
            f"Preparing {len(active_plan)} chunks at {fps:g} fps; "
            + (
                f"masked AV continuation overlaps {video_continuation} frames inside each sampled chunk"
                if use_masked_av_overlap
                else f"Video1 continuation carries {video_continuation} frames at {video_continuation_res}"
            )
        )
        logging.info("HR Endless Sampler: %s.", preparation_message)
        if preview_execution is not None:
            preview_execution.set_phase(preparation_message, chunk=replay_start_index)
        chunk_progress = _ChunkProgress(len(active_plan))
        components = [
            ("MiniMax H3 DiT", guider.model_patcher),
            ("Qwen/CLIP", clip.patcher),
            ("H3 video VAE", vae.patcher if vae is not None else None),
            ("H3 audio VAE", audio_vae.patcher if audio_vae is not None else None),
        ]
        vram_monitor = _VRAMMonitor(
            timing,
            guider.model_patcher.load_device,
            components,
            len(active_plan),
            debug=debug,
        )
        guider.model_patcher.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            VRAM_DEBUG_WRAPPER_KEY,
        )
        guider.model_patcher.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            VRAM_DEBUG_WRAPPER_KEY,
            vram_monitor,
        )
        timing.start_memory_poll()
        vram_monitor.report("execution prepared", {"full latent": samples, "full noise": full_noise})

        try:
            if (
                ENABLE_DEBUG_MEMORY_PREFLIGHT
                and debug
                and replay_start_index == 0
                and len(active_plan) > 1
                and include_video1_reference
            ):
                first_chunk = active_plan[0]
                preflight_video = video[:, :, first_chunk["video_start"]:first_chunk["video_end"]]
                preflight_audio = audio[..., first_chunk["audio_start"]:first_chunk["audio_end"]]
                preflight_video_noise = video_noise[:, :, first_chunk["video_start"]:first_chunk["video_end"]]
                preflight_audio_noise = audio_noise[..., first_chunk["audio_start"]:first_chunk["audio_end"]]
                preflight_latent = fixed_latent.copy()
                preflight_latent["samples"] = comfy.nested_tensor.NestedTensor((preflight_video, preflight_audio))
                preflight_noise = comfy.nested_tensor.NestedTensor((preflight_video_noise, preflight_audio_noise))
                try:
                    _run_debug_memory_preflight(
                        guider=guider,
                        sampler=sampler,
                        sigmas=sigmas,
                        chunk_latent=preflight_latent,
                        chunk_noise=preflight_noise,
                        clip=clip,
                        vae=vae,
                        images=h3_images,
                        positive=positive,
                        original_conds=original_conds,
                        chunk_prompt=planned_prompts[0][0],
                        chunk_seed=replay_noise_seed & 0xffffffffffffffff,
                        video=video,
                        audio=audio,
                        continuation_frames=video_continuation,
                        video_continuation_res=video_continuation_res,
                        video_number=video_number,
                        audio_number=audio_number,
                        width=width,
                        height=height,
                        chunk_count=len(active_plan),
                        vram_monitor=vram_monitor,
                        preview_execution=preview_execution,
                    )
                finally:
                    preflight_latent = None
                    preflight_noise = None
                    preflight_video = None
                    preflight_audio = None
                    preflight_video_noise = None
                    preflight_audio_noise = None
                    gc.collect()

            if gemma_director is not None:
                preproduction_shots, preproduction_request = _gemma_preproduction_request(
                    prompt,
                    gemma_shots,
                    plan,
                    fps,
                    ref2va,
                )
                if len(active_plan) != len(plan):
                    logging.info(
                        "HR Endless Sampler Gemma 4 preproduction is planning the complete %d-chunk production; "
                        "debug_stop_chunk limits only this H3 render to %d chunks.",
                        len(plan),
                        len(active_plan),
                    )
                if replay_timing_plan is not None:
                    try:
                        replay_payload = _timing_plan_payload(replay_timing_plan)
                        _validate_timing_plan(
                            replay_payload,
                            preproduction_request,
                            json.dumps(replay_payload, ensure_ascii=False),
                        )
                    except Gemma4ObservationError as error:
                        logging.info(
                            "HR Endless Sampler replay: cached Gemma timing plan does not satisfy the complete "
                            "production horizon, so only preproduction will be regenerated: %s",
                            error,
                        )
                        replay_timing_plan = None
                if replay_timing_plan is not None:
                    required_shots = {int(item["shot_number"]) for item in preproduction_shots}
                    cached_shots = {int(shot.source_shot) for shot in replay_timing_plan.shots}
                    if not required_shots.issubset(cached_shots):
                        logging.info(
                            "HR Endless Sampler replay: cached Gemma timing plan does not cover every requested "
                            "source shot, so it will be regenerated."
                        )
                        replay_timing_plan = None
                if gemma_preproduction_cache is not None:
                    preproduction_request["preproduction_cache"] = gemma_preproduction_cache.worker_spec()
                try:
                    if replay_timing_plan is not None:
                        logging.info(
                            "HR Endless Sampler replay: using cached Gemma preproduction timing plan; "
                            "only chunk-local directing will run again."
                        )
                    # No VAE decode is needed for the text-only pass, but the
                    # temporary Gemma worker still needs H3/Qwen/VAE gone so
                    # its full-GPU model can load and exit cleanly.
                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    if vae is not None:
                        comfy.model_management.unload_model_and_clones(vae.patcher)
                    comfy.model_management.soft_empty_cache(force=True)
                    vram_monitor.report("before Gemma 4 shot-timing preproduction")
                    timer_started = time.perf_counter()
                    try:
                        with _PreparationProgress(
                            (
                                "Restoring cached Gemma 4 timing plan"
                                if replay_timing_plan is not None
                                else "pre-production planning"
                            ),
                            preview_execution,
                            chunk=0,
                        ) as preparation_progress:
                            gemma_preproduction_timing_plan = (
                                replay_timing_plan
                                if replay_timing_plan is not None
                                else gemma_director.plan_timing(
                                    preproduction_request,
                                    progress_callback=preparation_progress.update_token_progress,
                                )
                            )
                    finally:
                        gemma_preproduction_seconds += timing.add("gemma4", timer_started)
                    if replay_cache is not None and replay_timing_plan is None:
                        replay_cache.save_timing_plan(gemma_preproduction_timing_plan, source_prompt=prompt)
                    _append_gemma_timing_plan(
                        gemma_prompt_log,
                        "Character-name table:\n"
                        + gemma_preproduction_timing_plan.character_name_table_text()
                        + "\n\nGlobal production bible:\n"
                        + gemma_preproduction_timing_plan.production_bible_text()
                        + "\n\n"
                        + gemma_preproduction_timing_plan.for_target_shots(preproduction_shots, fps),
                        system_prompt=gemma_preproduction_timing_plan.system_prompt or gemma_director.last_timing_system_prompt,
                        planning_prompt=(
                            gemma_preproduction_timing_plan.planning_prompt
                            or gemma_director.last_timing_planning_prompt
                        ),
                        gemma_response=_gemma_timing_plan_transcript(gemma_preproduction_timing_plan),
                        validation_warnings=gemma_preproduction_timing_plan.validation_warnings,
                    )
                    logging.info(
                        "HR Endless Sampler Gemma 4 preproduction timing plan is ready for %d source shots.",
                        len(gemma_preproduction_timing_plan.shots),
                    )
                    lighting_by_shot = {
                        int(shot.source_shot): bool(shot.light_change)
                        for shot in gemma_preproduction_timing_plan.shots
                    }
                    for preview_shot in preview_shot_ranges:
                        preview_shot["light_change"] = lighting_by_shot.get(
                            int(preview_shot["shot"]), True
                        )
                    if gemma_preproduction_cache is not None and replay_timing_plan is not None:
                        # Replay restores only the validated schedule. The
                        # render-local cache was intentionally reset above, so
                        # rebuild the clean static directorial conversation
                        # from that schedule before Chunk 1. Do not ask Gemma
                        # to plan the same shots a second time.
                        cache_timer_started = time.perf_counter()
                        try:
                            with _PreparationProgress(
                                "Gemma 4 is rebuilding the clean preproduction KV cache from the replay plan",
                                preview_execution,
                                chunk=0,
                            ) as preparation_progress:
                                gemma_director.materialize_preproduction_cache(
                                    preproduction_request,
                                    gemma_preproduction_timing_plan,
                                    progress_callback=preparation_progress.update_token_progress,
                                )
                        except (Gemma4DependencyError, Gemma4ObservationError, OSError, RuntimeError, ValueError) as cache_error:
                            # This optimization is optional. Preserve the
                            # replay even if the isolated cache worker cannot
                            # export a new clean state.
                            logging.warning(
                                "HR Endless Sampler Gemma 4 could not rebuild the clean preproduction KV cache "
                                "from the replay plan; each chunk will use its ordinary full directing request: %s",
                                cache_error,
                            )
                        finally:
                            gemma_preproduction_seconds += timing.add("gemma4", cache_timer_started)
                    if gemma_preproduction_cache is not None:
                        gemma_preproduction_cache_ready = gemma_preproduction_cache.ready()
                        if gemma_preproduction_cache_ready:
                            logging.info(
                                "HR Endless Sampler Gemma 4 clean preproduction KV cache is ready (%0.2f GiB); "
                                "every chunk will restore this same pre-Chunk-1 memory.",
                                gemma_preproduction_cache.size_bytes() / (1024 ** 3),
                            )
                        else:
                            logging.warning(
                                "HR Endless Sampler Gemma 4 clean preproduction KV cache was not produced; "
                                "each chunk will receive the ordinary full directing request."
                            )
                    vram_monitor.report("after Gemma 4 shot-timing preproduction release")
                except Gemma4DependencyError:
                    raise
                except Gemma4ObservationError as error:
                    logging.warning(
                        "HR Endless Sampler Gemma 4 shot-timing preproduction failed; "
                        "sampling is stopping before Chunk 1 and no sampler-authored timing fallback will be used: %s",
                        error,
                    )
                    _append_gemma_timing_plan(
                        gemma_prompt_log,
                        None,
                        system_prompt=gemma_director.last_timing_system_prompt,
                        planning_prompt=gemma_director.last_timing_planning_prompt,
                        gemma_response=error.raw_json or f"{type(error).__name__}: {error}",
                        validation_warnings=(str(error),),
                    )
                    raise
            replay_sample_end = len(active_plan) - len(replay_cached_suffix)
            audio_transcripts = []
            audio_observations = []
            audio_requested_prompts = []
            audio_retried_chunks = set()
            teacher_audio_latents = {}
            audio_transcriber = None
            if use_taomate and audio_vae is not None:
                from .python.audio_transcription import ChunkAudioTranscriber
                audio_transcriber = ChunkAudioTranscriber()
            if use_taomate and audio_first_enabled and replay_start_index == 0 and replay_sample_end:
                first_started = time.perf_counter()
                sampling = guider.model_patcher.get_model_object("model_sampling")
                video_shift = float(sampling.shift)
                audio_shift = float(sampling.audio_shift if sampling.audio_shift is not None else 3.0)
                logging.info("TaoMate audio-first: requesting %d timed chunk prompts in audio order before video sampling.", replay_sample_end)
                text_only_teacher = any(ref["kind"] in ("video", "video_audio") for ref in original_refs)
                if text_only_teacher:
                    logging.warning("TaoMate audio teacher will encode timed text without video references, which are unavailable before video sampling.")
                def audio_teacher_conditioning(audio_index):
                    """Request and encode the next prompt after the previous audio was heard."""
                    audio_chunk = active_plan[audio_index]
                    audio_content_start = audio_chunk["frame_start"] + audio_chunk.get("output_trim_frames", 0)
                    if manual_prompts is not None:
                        teacher_prompt = manual_prompts.get_chunk_prompt(audio_index + 1, previous_audio_transcription=audio_transcripts[-1]["text"] if audio_transcripts else None, prompt_stage="audio")
                    elif gemma_director is not None:
                        # Audio is generated before video, so there are no prior
                        # video stills for this request. The transcript is the
                        # previous rendered evidence available to this pass.
                        target_shots = _gemma_shot_records(gemma_shots, audio_content_start, audio_chunk["frame_end"], audio_chunk["frame_start"], fps, target=True)
                        previous_chunk = active_plan[audio_index - 1] if audio_index else None
                        previous_output_start = previous_chunk["frame_start"] + previous_chunk.get("output_trim_frames", 0) if previous_chunk is not None else 0
                        prior_range = {"sampled_start": previous_chunk["frame_start"], "sampled_end": previous_chunk["frame_end"], "output_start": previous_output_start, "output_end": previous_chunk["frame_end"]} if previous_chunk is not None else None
                        current_range = {"sampled_start": audio_chunk["frame_start"], "sampled_end": audio_chunk["frame_end"], "output_start": audio_content_start, "output_end": audio_chunk["frame_end"]}
                        request = {
                            "require_summary": True,
                            "summary_required_tasks": [],
                            "summary_forbidden_tasks": ["video continuation"] + (["audio reference", "audio reuse"] if not any(ref["kind"] in ("audio", "video_audio") for ref in original_refs) else []),
                            "chunk_number": audio_index + 1,
                            "chunk_count": len(active_plan),
                            "fps": fps,
                            "prompt_mode": "ref" if ref2va else "base",
                            "prompt_stage": "audio",
                            "current_chunk": current_range,
                            "previous_chunk": prior_range,
                            "previous_shots": _gemma_shot_records(gemma_shots, previous_output_start, previous_chunk["frame_end"], previous_output_start, fps, target=False) if previous_chunk is not None else [],
                            "observation_frame_numbers": [],
                            "previous_audio_transcription": audio_transcripts[-1]["text"] if audio_transcripts else None,
                            "target_shots": target_shots,
                            "preproduction_timing_plan": gemma_preproduction_timing_plan.for_target_shots(target_shots, fps),
                            "production_bible": gemma_preproduction_timing_plan.production_bible_text(),
                            "mandatory_coverage": gemma_preproduction_timing_plan.mandatory_coverage(target_shots),
                            "character_name_table": gemma_preproduction_timing_plan.character_name_table_text(),
                            "planned_character_continuity": gemma_preproduction_timing_plan.continuity_for_target_shots(target_shots),
                            "current_character_subjects": [{"character_name": item.character_name, "subject": item.subject} for item in gemma_preproduction_timing_plan.current_character_subjects(target_shots)],
                            "conditioning_context": "Teacher audio prompt request. Video has not been generated yet; use the source plan and the prior audio transcription to decide the next spoken words. No previous video frames are available.",
                            "original_prompt": prompt,
                        }
                        if gemma_preproduction_cache_ready and gemma_preproduction_cache is not None:
                            request["preproduction_cache"] = gemma_preproduction_cache.worker_spec()
                            request["preproduction_current_slice"] = gemma_preproduction_timing_plan.current_slice_coverage_text(target_shots)
                        comfy.model_management.unload_model_and_clones(guider.model_patcher)
                        comfy.model_management.unload_model_and_clones(clip.patcher)
                        result = gemma_director.direct(request, None, audio=audio_observations[-1] if audio_observations else None)
                        teacher_prompt = _prompt_with_gemma_description(prompt, result.detailed_description, drop_picture_anchors=bool(audio_index and not ref2va), summary=result.summary if audio_index else None)
                        teacher_prompt = _preserve_global_prompt_sections(_chunk_summary_prompt(teacher_prompt, prompt, audio_index > 0), prompt)
                    else:
                        teacher_prompt = _native_prompt_for_stage(planned_prompts[audio_index][0], "audio", audio_requested_prompts[-1] if audio_index else None, audio_transcripts[-1] if audio_index else None)
                    audio_requested_prompts.append(teacher_prompt)
                    preview_chunk_ranges[audio_index]["audio_teacher_prompt"] = teacher_prompt.strip()
                    preview_chunk_ranges[audio_index]["h3_prompt"] = teacher_prompt.strip()
                    preview_chunk_ranges[audio_index]["subtitle"] = _preview_subtitle(teacher_prompt)
                    if preview_execution is not None:
                        preview_execution.set_audio_prompt(audio_index, teacher_prompt.strip(), preview_chunk_ranges[audio_index]["subtitle"])
                        preview_execution.set_phase("TaoMate: encoding audio prompt %d/%d" % (audio_index + 1, replay_sample_end), chunk=audio_index)
                    if text_only_teacher:
                        # The audio-only teacher has no decoded video reference
                        # before sampling; its timed text is still authoritative.
                        encoded_teacher = _encode_prompt(clip, teacher_prompt, None, (), width, height, True)
                    else:
                        encoded_teacher = _encode_prompt(clip, teacher_prompt, h3_images, positive, width, height, audio_index > 0)
                    teacher_cond = {"cross_attn": encoded_teacher[0].detach().to(device="cpu")}
                    teacher_tags = encoded_teacher[1].get("minimax_token_tags")
                    if teacher_tags is not None:
                        teacher_cond["minimax_token_tags"] = teacher_tags.detach().to(device="cpu")
                    logging.info("TaoMate audio teacher prompt %d/%d: audio ticks %d-%d, dialogue=%r", audio_index + 1, replay_sample_end, active_plan[audio_index]["audio_start"], active_plan[audio_index]["audio_end"] - 1, _preview_subtitle(teacher_prompt))
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    return teacher_cond
                if preview_execution is not None:
                    preview_execution.set_phase("TaoMate: generating full audio before video")
                audio_teacher_steps = max(0, int(sigmas.numel()) - 1)

                def publish_audio_first_chunk(audio_index, audio_latent, previous_latent, context_ticks):
                    """Decode, transcribe, then publish this teacher-audio segment."""
                    audio_range = preview_chunk_ranges[audio_index]
                    context_ticks = int(context_ticks)
                    if context_ticks and (previous_latent is None or previous_latent.shape[-1] < context_ticks):
                        raise ValueError("Teacher audio preview is missing the preceding latent context")
                    if preview_execution is not None:
                        preview_execution.set_phase("TaoMate: audio teacher %d/%d · step %d/%d · decoding and transcribing" % (audio_index + 1, replay_sample_end, audio_teacher_steps, audio_teacher_steps), chunk=audio_index)
                    audio_started = time.perf_counter()
                    try:
                        comfy.model_management.unload_model_and_clones(guider.model_patcher)
                        comfy.model_management.unload_model_and_clones(clip.patcher)
                        waveform, sample_rate = taomate_backend.decode_audio_preview_segment(audio_vae, audio_latent, previous_latent, context_ticks, AUDIO_LATENT_FPS)
                        recognized = audio_transcriber.transcribe(waveform, sample_rate)
                        audio_transcripts.append(recognized)
                        audio_observations.append((waveform.detach().to(device="cpu"), int(sample_rate)))
                        expected_dialogue = _preview_subtitle(audio_requested_prompts[audio_index])
                        expected_words = re.findall(r"\w+(?:['’]\w+)?", expected_dialogue.casefold())
                        heard_words = re.findall(r"\w+(?:['’]\w+)?", recognized["text"].casefold())
                        logging.info("TaoMate audio teacher transcript %d/%d (%s): expected=%r, heard=%r", audio_index + 1, replay_sample_end, "exact word match" if expected_words == heard_words else "review dialogue", expected_dialogue, recognized["text"])
                        if audio_index in audio_retried_chunks and expected_words != heard_words:
                            logging.warning("TaoMate audio teacher %d/%d still differs from its dialogue after one retry", audio_index + 1, replay_sample_end)
                        if preview_execution is not None:
                            if audio_index in audio_retried_chunks:
                                preview_execution.set_audio_retry(audio_index, finished=True)
                            preview_execution.publish_chunk_audio_first(audio_index, waveform, sample_rate, audio_range["start"], audio_range["end"])
                            preview_execution.set_audio_chunk_complete(audio_index)
                    finally:
                        comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                        timing.add("vae_audio_preview", audio_started)

                def retry_audio_first_candidate(audio_index, audio_latent, previous_latent, context_ticks):
                    """Keep a clean spoken prefix, otherwise re-cut the dialogue and re-sample once."""
                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                    waveform, sample_rate = taomate_backend.decode_audio_preview_segment(audio_vae, audio_latent, previous_latent, context_ticks, AUDIO_LATENT_FPS)
                    recognized = audio_transcriber.transcribe(waveform, sample_rate)
                    requested = audio_requested_prompts[audio_index]
                    native_dialogue = manual_prompts is None and gemma_director is None
                    approved_prompt = _audio_transcript_approved_tail_prompt(requested, recognized) if native_dialogue else None
                    if approved_prompt is not None:
                        # The take spoke a clean prefix of its dialogue, so its KV
                        # stays. Only the words it did not say are re-authored with
                        # the initial cut, over the chunks still to be sampled.
                        texts = [item[0] for item in planned_prompts]
                        texts[audio_index] = requested
                        revised = _recut_dialogue_prompts(texts, active_plan, fps, audio_index, audio_index + 1, kept_words=len(_dialogue_block_words(approved_prompt)[3]))
                        if revised is None:
                            logging.info("TaoMate audio teacher %d/%d accepted its spoken dialogue prefix.", audio_index + 1, replay_sample_end)
                            revised = texts
                            revised[audio_index] = approved_prompt
                        else:
                            logging.info("TaoMate audio teacher %d/%d accepted its spoken dialogue prefix and re-authored the remaining dialogue.", audio_index + 1, replay_sample_end)
                        _publish_revised_dialogue(planned_prompts, revised, audio_index, preview_chunk_ranges, preview_execution)
                        audio_requested_prompts[audio_index] = revised[audio_index]
                        preview_chunk_ranges[audio_index]["audio_teacher_prompt"] = revised[audio_index].strip()
                        return None
                    if _teacher_take_matches_dialogue(_preview_subtitle(requested), recognized["text"]):
                        return None
                    if not _transcript_is_certain(recognized):
                        logging.warning("Teacher audio transcript is uncertain; skipping prompt retry")
                        return None
                    logging.warning("TaoMate audio teacher %d/%d dialogue mismatch; retrying once: expected=%r, heard=%r", audio_index + 1, replay_sample_end, _preview_subtitle(requested), recognized["text"])
                    retry_prompt = requested
                    if native_dialogue:
                        texts = [item[0] for item in planned_prompts]
                        texts[audio_index] = requested
                        revised, shrink_frames = _retry_take_prompts(texts, active_plan, fps, audio_index)
                        if revised is not None:
                            _publish_revised_dialogue(planned_prompts, revised, audio_index, preview_chunk_ranges, preview_execution)
                            retry_prompt = revised[audio_index]
                            logging.info("TaoMate audio teacher %d/%d re-cut the leftover dialogue over the remaining chunks, %d frames shorter.", audio_index + 1, replay_sample_end, shrink_frames)
                    audio_retried_chunks.add(audio_index)
                    audio_requested_prompts[audio_index] = retry_prompt
                    preview_chunk_ranges[audio_index]["audio_teacher_prompt"] = retry_prompt.strip()
                    preview_chunk_ranges[audio_index]["h3_prompt"] = retry_prompt.strip()
                    if preview_execution is not None:
                        preview_execution.set_audio_retry(audio_index)
                        preview_execution.set_audio_prompt(audio_index, retry_prompt.strip(), _preview_subtitle(retry_prompt))
                        preview_execution.set_phase("TaoMate: retrying teacher audio %d/%d after transcript mismatch" % (audio_index + 1, replay_sample_end), chunk=audio_index)
                    comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                    encoded = _encode_prompt(clip, retry_prompt, None if text_only_teacher else h3_images, () if text_only_teacher else positive, width, height, text_only_teacher or audio_index > 0)
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    result = {"cross_attn": encoded[0].detach().to(device="cpu")}
                    tags = encoded[1].get("minimax_token_tags")
                    if tags is not None:
                        result["minimax_token_tags"] = tags.detach().to(device="cpu")
                    return result

                full_teacher_audio = taomate_backend.prepare_audio_first(audio_teacher_conditioning, audio_noise, video, active_plan[:replay_sample_end], sigmas, video_shift, audio_shift, on_status=(lambda message: preview_execution.set_phase(message)) if preview_execution is not None else None, on_progress=(lambda chunk_number, chunk_total, step, steps: preview_execution.set_phase("TaoMate: audio teacher %d/%d · step %d/%d" % (chunk_number, chunk_total, step, steps), chunk=chunk_number - 1, audio_step_ms=taomate_backend.audio_teacher.average_step_ms)) if preview_execution is not None else None, on_chunk_complete=publish_audio_first_chunk, on_audio_candidate=retry_audio_first_candidate if TOGGLE_TAOMATE_DIVERGENCY_AUDIO_TRANSCRIPT_RETRY and audio_vae is not None else None)
                del audio_transcriber
                timing.add("taomate_audio_first", first_started)
                if full_teacher_audio is not None and audio_vae is not None:
                    decode_started = time.perf_counter()
                    try:
                        comfy.model_management.unload_model_and_clones(guider.model_patcher)
                        comfy.model_management.unload_model_and_clones(clip.patcher)
                        full_audio, decoded_audio_sample_rate = taomate_backend.decode_audio_timeline(audio_vae, [full_teacher_audio])
                        decoded_output_audio = [full_audio]
                        decoded_audio_complete = True
                        audio_first_decoded = True
                        if preview_execution is not None:
                            preview_execution.publish_audio_first(full_audio, decoded_audio_sample_rate)
                    except Exception as error:
                        if not audio_first_decoded:
                            decoded_audio_complete = False
                        logging.warning("HR Endless Sampler could not decode the audio-first teacher preview; it will retry audio decode after video sampling: %s", error)
                    finally:
                        comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                        timing.add("vae_audio_preview", decode_started)
                taomate_backend.audio_first_audio_latent = None
                full_teacher_audio = None
                if vram_monitor is not None:
                    vram_monitor.report("after TaoMate full-audio teacher prepass")
                if debug:
                    logging.info("TaoMate TOGGLE_TAOMATE_DIVERGENCY_AUDIO_FIRST=True; completed teacher audio pass before video sampling.")
            for index, chunk in enumerate(
                    active_plan[replay_start_index:replay_sample_end], start=replay_start_index):
                timing.observe_memory()
                timing.start_chunk(index)
                gemma_chunk_seconds = 0.0
                h3_render_seconds = 0.0
                vram_monitor.set_chunk(index)
                vram_monitor.report(
                    f"chunk {index + 1}/{len(active_plan)} start",
                    {
                        "full latent": samples,
                        "full noise": full_noise,
                        "completed output": (output_video, output_audio),
                        "completed denoised output": (denoised_video, denoised_audio),
                        "previous chunk": (previous_video, previous_audio),
                    },
                )
                continuation = index > 0
                continuation_picture_label = f"<Picture {picture_number}>" if continuation and include_video1_reference else None
                content_start = chunk["frame_start"] + chunk.get("output_trim_frames", 0)
                chunk_label = f"Chunk {index + 1}/{len(active_plan)}"
                if preview_execution is not None:
                    preview_execution.set_phase(f"{chunk_label}: preparing continuation conditioning", chunk=index)
                logging.info("HR Endless Sampler: %s: preparing continuation conditioning.", chunk_label)
                gemma_description = None
                gemma_retention_analysis = None
                gemma_report = None
                gemma_system_prompt = None
                gemma_observation_prompt = None
                gemma_response = None
                gemma_validation_warnings = ()
                chunk_audio_prompt = None
                if gemma_director is not None:
                    observation_frames = None
                    try:
                        # H3 and Qwen must be out of VRAM before the optional
                        # VAE decode and temporary fully-GPU Gemma load.
                        comfy.model_management.unload_model_and_clones(guider.model_patcher)
                        comfy.model_management.unload_model_and_clones(clip.patcher)
                        comfy.model_management.soft_empty_cache(force=True)
                        if vram_monitor is not None:
                            vram_monitor.report(
                                f"chunk {index + 1}/{len(active_plan)} before Gemma 4 prompt directing",
                                {"previous chunk": previous_video},
                            )

                        previous_chunk = None
                        previous_shots = []
                        observation_frame_numbers = []
                        if continuation:
                            previous_plan = active_plan[index - 1]
                            previous_output_start = previous_plan["frame_start"] + previous_plan.get("output_trim_frames", 0)
                            previous_chunk = {
                                "sampled_start": previous_plan["frame_start"],
                                "sampled_end": previous_plan["frame_end"],
                                "output_start": previous_output_start,
                                "output_end": previous_plan["frame_end"],
                            }
                            previous_shots = _gemma_shot_records(
                                gemma_shots,
                                previous_output_start,
                                previous_plan["frame_end"],
                                previous_output_start,
                                fps,
                                target=False,
                            )
                            if previous_decoded_frames is None:
                                timer_started = time.perf_counter()
                                try:
                                    previous_decoded_frames = _decode_video_frames(vae, previous_video).detach().to(
                                        device="cpu",
                                        dtype=torch.float32,
                                    )
                                finally:
                                    timing.add("vae_previous_chunk", timer_started)
                            retained_start = max(
                                0,
                                min(
                                    int(previous_plan.get("output_trim_frames", 0)),
                                    int(previous_decoded_frames.shape[0]),
                                ),
                            )
                            decoded_color_diagnostics = (
                                _safe_video_color_diagnostics(previous_decoded_frames[retained_start:])
                                if retained_start < previous_decoded_frames.shape[0]
                                else None
                            )
                            observation_frames, observation_indices = _sample_decoded_video_frames(
                                previous_decoded_frames,
                                include_final=True,
                                start_frame=previous_plan.get("output_trim_frames", 0),
                            )
                            observation_frame_numbers = [
                                previous_plan["frame_start"] + frame_index
                                for frame_index in observation_indices
                            ]
                            if decoded_color_diagnostics is not None:
                                decoded_chunk_index = index - 1
                                if decoded_chunk_index not in color_diagnostics:
                                    _record_color_diagnostics(
                                        color_diagnostics,
                                        decoded_chunk_index,
                                        decoded_color_diagnostics,
                                        _color_boundary_context(gemma_shots, previous_output_start),
                                    )

                        continuation_video_label = f"<Video {video_number}>" if include_video1_reference else None
                        continuation_audio_label = f"<Audio {audio_number}>" if include_previous_audio_reference else None
                        target_shots = _gemma_shot_records(
                            gemma_shots,
                            content_start,
                            chunk["frame_end"],
                            chunk["frame_start"],
                            fps,
                            target=True,
                        )
                        request = {
                            "require_summary": True,
                            "summary_required_tasks": (["video continuation"] if continuation and include_video1_reference else []) + (["audio reference"] if continuation and include_previous_audio_reference else []),
                            "summary_forbidden_tasks": (["video continuation"] if not (continuation and include_video1_reference) else []) + (["audio reference", "audio reuse"] if not (continuation and include_previous_audio_reference) and not any(ref["kind"] in ("audio", "video_audio") for ref in original_refs) else []),
                            "chunk_number": index + 1,
                            "chunk_count": len(active_plan),
                            "fps": fps,
                            "prompt_mode": "ref" if ref2va else "base",
                            "prompt_stage": "video",
                            "current_chunk": {
                                "sampled_start": chunk["frame_start"],
                                "sampled_end": chunk["frame_end"],
                                "output_start": content_start,
                                "output_end": chunk["frame_end"],
                            },
                            "previous_chunk": previous_chunk,
                            "previous_audio_transcription": audio_transcripts[index - 1]["text"] if 0 < index <= len(audio_transcripts) else None,
                            "previous_shots": previous_shots,
                            "observation_frame_numbers": observation_frame_numbers,
                            "previous_gemma_description": previous_gemma_description,
                            "previous_gemma_timing_plan": previous_gemma_timing_plan,
                            "previous_gemma_end_state": previous_gemma_end_state,
                            "previous_last_seen_character_state": previous_gemma_last_seen_character_state,
                            "target_shots": target_shots,
                            "preproduction_timing_plan": gemma_preproduction_timing_plan.for_target_shots(
                                target_shots,
                                fps,
                            ),
                            "production_bible": gemma_preproduction_timing_plan.production_bible_text(),
                            "mandatory_coverage": gemma_preproduction_timing_plan.mandatory_coverage(
                                target_shots,
                            ),
                            "character_name_table": gemma_preproduction_timing_plan.character_name_table_text(),
                            "planned_character_continuity": gemma_preproduction_timing_plan.continuity_for_target_shots(
                                target_shots,
                            ),
                            "current_character_subjects": [
                                {
                                    "character_name": item.character_name,
                                    "subject": item.subject,
                                }
                                for item in gemma_preproduction_timing_plan.current_character_subjects(target_shots)
                            ],
                            "conditioning_context": _gemma_conditioning_context(
                                continuation,
                                context_keyframes,
                                guide_overlap,
                                video_continuation,
                                continuation_video_label,
                                continuation_audio_label,
                                include_video1_reference,
                                video_continuation_method,
                                use_masked_av_overlap,
                            ),
                            "original_prompt": prompt,
                        }
                        # Give summary authorship the actual conditioning roles
                        # and prior speaker bindings, not a guessed Video1.
                        if continuation:
                            request["conditioning_context"] += "\nThe original prompt summary describes only the first chunk's opening. Replace those opening picture/video/audio roles with the current chunk's supplied continuation references; do not repeat the original first-frame instruction."
                        if continuation and (use_masked_av_mode or context_keyframes):
                            request["summary_required_tasks"].append("keyframe completion")
                        if continuation_picture_label is not None:
                            request["summary_required_tasks"].append("keyframe completion")
                            request["conditioning_context"] += f"\nAn ordinary image reference contains the final full frame of the previous chunk. Include this exact sentence in summary: {continuation_picture_label} serves as first frame of target video."
                        if continuation and include_previous_audio_reference:
                            request["conditioning_context"] += "\n" + _audio_reference_definition(continuation_audio_label, _audio_reference_speakers(previous_gemma_description))
                        if gemma_preproduction_cache_ready and gemma_preproduction_cache is not None:
                            request["preproduction_cache"] = gemma_preproduction_cache.worker_spec()
                            request["preproduction_current_slice"] = (
                                gemma_preproduction_timing_plan.current_slice_coverage_text(target_shots)
                            )
                        if vae is not None:
                            comfy.model_management.unload_model_and_clones(vae.patcher)
                        comfy.model_management.soft_empty_cache(force=True)
                        timer_started = time.perf_counter()
                        try:
                            with _PreparationProgress(
                                f"{chunk_label}: directing chunk",
                                preview_execution,
                                chunk=index,
                                live_console_bar=True,
                            ) as preparation_progress:
                                gemma_director.render_context.update({"chunk_index": index, "chunk": chunk, "previous_video": previous_video, "previous_audio": previous_audio, "previous_frames": previous_decoded_frames, "observation_frames": observation_frames})
                                if use_taomate and not audio_first_enabled:
                                    audio_request = dict(request)
                                    audio_request["prompt_stage"] = "audio"
                                    audio_request["summary_required_tasks"] = [task for task in request["summary_required_tasks"] if task in ("audio reference", "audio reuse")]
                                    audio_request["summary_forbidden_tasks"] = list(dict.fromkeys(request["summary_forbidden_tasks"] + ["video continuation", "keyframe completion", "video editing"]))
                                    audio_request["conditioning_context"] = "Teacher audio prompt request. Previous video stills describe the scene; previous decoded audio and its transcript establish delivery and already spoken words. This request does not sample video; use only audio reference roles actually supplied by the source prompt."
                                    audio_result = gemma_director.direct(audio_request, observation_frames, progress_callback=preparation_progress.update_token_progress, audio=audio_observations[index - 1] if 0 < index <= len(audio_observations) else None)
                                    chunk_audio_prompt = _prompt_with_gemma_description(prompt, audio_result.detailed_description, drop_picture_anchors=continuation and not ref2va, summary=audio_result.summary if continuation else None)
                                    chunk_audio_prompt = _preserve_global_prompt_sections(_chunk_summary_prompt(chunk_audio_prompt, prompt, continuation), prompt)
                                result = gemma_director.direct(
                                    request,
                                    observation_frames,
                                    progress_callback=preparation_progress.update_token_progress,
                                    audio=audio_observations[index - 1] if 0 < index <= len(audio_observations) else None,
                                )
                        finally:
                            gemma_chunk_seconds = timing.add("gemma4", timer_started)
                        gemma_system_prompt = result.system_prompt or gemma_director.last_system_prompt
                        gemma_observation_prompt = result.observation_prompt or gemma_director.last_observation_prompt
                        gemma_response = _gemma_response_transcript(result)
                        gemma_description = _normalize_last_dialogue_terminal_punctuation(
                            result.detailed_description
                        )
                        if gemma_description != result.detailed_description:
                            logging.info(
                                "HR Endless Sampler chunk %d/%d normalized the final dialogue's "
                                "terminal punctuation to a period before H3 encoding.",
                                index + 1,
                                len(active_plan),
                            )
                        if continuation and ref2va:
                            gemma_description = _reinforce_static_shot_grade(gemma_description, gemma_shots, content_start)
                        gemma_retention_analysis = (
                            _chunk_retention_analysis(result.retention_analysis)
                            if INCLUDE_PER_CHUNK_RETENTION_ANALYSIS
                            else None
                        )
                        gemma_validation_warnings = result.validation_warnings
                        gemma_report = _gemma_report(index + 1, result)
                        if vram_monitor is not None:
                            vram_monitor.report(
                                f"chunk {index + 1}/{len(active_plan)} after Gemma 4 release",
                            )
                    except Gemma4DependencyError:
                        raise
                    except Gemma4ObservationError as error:
                        gemma_system_prompt = gemma_director.last_system_prompt
                        gemma_observation_prompt = gemma_director.last_observation_prompt
                        gemma_response = error.raw_json or f"{type(error).__name__}: {error}"
                        logging.warning(
                            "HR Endless Sampler Gemma 4 prompt directing for chunk %d/%d failed; "
                            "sampling is stopping and no algorithmic source-prompt fallback will be used: %s",
                            index + 1,
                            len(active_plan),
                            error,
                        )
                        _append_last_gemma_prompt(
                            gemma_prompt_log,
                            _debug_chunk_header(index, chunk, content_start),
                            None,
                            system_prompt=gemma_system_prompt if not gemma_system_logged else None,
                            observation_prompt=gemma_observation_prompt,
                            gemma_response=gemma_response,
                            validation_warnings=(str(error),),
                        )
                        raise
                    finally:
                        observation_frames = None
                        if vae is not None:
                            comfy.model_management.unload_model_and_clones(vae.patcher)
                vs, ve = chunk["video_start"], chunk["video_end"]
                aus, aue = chunk["audio_start"], chunk["audio_end"]
                context_video_t = chunk["context_video_t"]
                context_audio_t = chunk["context_audio_t"]

                chunk_video = video[:, :, vs:ve]
                chunk_audio = audio[..., aus:aue]
                chunk_video_noise = video_noise[:, :, vs:ve]
                chunk_audio_noise = audio_noise[..., aus:aue]
                prefix_video = None
                prefix_audio = None
                prefix_latent = None
                prefix_noise = None
                prefix_video_noise = None
                prefix_audio_noise = None
                chunk_noise_mask = None
                boundary_video_context = None
                boundary_audio_context = None
                if chunk.get("synthetic_prefix"):
                    prefix_video = video.new_zeros((*video.shape[:2], context_video_t, *video.shape[3:]))
                    prefix_audio = audio.new_zeros((*audio.shape[:-1], context_audio_t))
                    # Reuse the preceding AV noise tokens from the same full
                    # sequence as the new output, including on cache replay.
                    prefix_video_noise, prefix_audio_noise = _full_noise_prefix(video_noise, audio_noise, vs, aus, context_video_t, context_audio_t)
                    if use_taomate:
                        # Transport halo is only for VAE decoding, never sampled twice.
                        prefix_video = previous_video[:, :, -context_video_t:].clone()
                        prefix_audio = previous_audio[..., -context_audio_t:].clone()
                    chunk_video = torch.cat((prefix_video, chunk_video), dim=2)
                    chunk_audio = torch.cat((prefix_audio, chunk_audio), dim=-1)
                    chunk_video_noise = torch.cat((prefix_video_noise, chunk_video_noise), dim=2)
                    chunk_audio_noise = torch.cat((prefix_audio_noise, chunk_audio_noise), dim=-1)

                if continuation and use_masked_av_overlap:
                    chunk_video, chunk_audio, chunk_noise_mask = _masked_av_overlap_target(
                        chunk_video,
                        chunk_audio,
                        previous_video,
                        previous_audio,
                        context_video_t,
                        context_audio_t,
                    )
                    logging.info(
                        "HR Endless Sampler chunk %d/%d masked AV overlap: %d frames, "
                        "%d video tokens, %d audio ticks (%d feathered); no Video1/Audio1 reference",
                        index + 1,
                        len(active_plan),
                        video_continuation,
                        context_video_t,
                        context_audio_t,
                        min(MASKED_AV_AUDIO_FEATHER_TICKS, context_audio_t),
                    )
                elif continuation and use_masked_av_mode and chunk.get("synthetic_prefix"):
                    # Native tails guide the prefix without VAE re-encoding.
                    # Feathering additionally seeds and masks both targets.
                    boundary_video_context, boundary_start = _video_continuation_boundary_guide(
                        previous_video, chunk, context_keyframes, use_video_continuation,
                    )
                    boundary_audio_t = context_audio_t
                    if previous_audio is None or int(previous_audio.shape[-1]) < boundary_audio_t:
                        raise ValueError("Native audio boundary needs the prior chunk's final audio-latent ticks")
                    boundary_audio_context = previous_audio[..., -boundary_audio_t:].clone()
                    boundary_audio_feather_ticks = _last_dialogue_word_feather_ticks(
                        previous_h3_prompt or previous_gemma_description,
                        boundary_audio_t,
                    ) if audio_feathered_overlap else 0
                    if audio_feathered_overlap:
                        chunk_video, chunk_audio, chunk_noise_mask = _native_audio_boundary_target(
                            chunk_video,
                            chunk_audio,
                            boundary_audio_context,
                            boundary_audio_t,
                            boundary_audio_feather_ticks,
                            previous_video=boundary_video_context,
                        )
                    logging.info(
                        "HR Endless Sampler chunk %d/%d native-tail packing boundary: %d video frames "
                        "from the prior latent's final 1+4 token pair and %d audio-latent ticks supplied as a keyframe at local frame %d; "
                        "video/audio latent feathering is %s%s",
                        index + 1,
                        len(active_plan),
                        int(chunk.get("output_trim_frames", 0)),
                        boundary_audio_t,
                        boundary_start,
                        "enabled" if audio_feathered_overlap else "disabled",
                        " over its final %d last-word ticks" % boundary_audio_feather_ticks if audio_feathered_overlap else "",
                    )

                if continuation and warm_start_video_t:
                    warm_start = context_video_t
                    warm_count = min(
                        warm_start_video_t,
                        chunk_video.shape[2] - warm_start,
                        previous_video.shape[2],
                    )
                    warm_end = warm_start + warm_count
                    # This is deliberately not a keyframe or physical overlap.
                    # Initialize retained target positions from the completed
                    # tail, keep their fresh target noise, fully denoise them,
                    # and retain them in the assembled output.
                    if warm_count:
                        chunk_video = chunk_video.clone()
                        chunk_video[:, :, warm_start:warm_end] = previous_video[:, :, -warm_count:]
                        if debug:
                            logging.info(
                                "HR Endless Sampler chunk %d/%d retained latent warm-start: "
                                "%d previous-tail video tokens copied to kept local tokens %d-%d",
                                index + 1,
                                len(active_plan),
                                warm_count,
                                warm_start,
                                warm_end - 1,
                            )

                chunk_latent = fixed_latent.copy()
                chunk_latent["samples"] = comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))
                if chunk_noise_mask is not None:
                    chunk_latent["noise_mask"] = chunk_noise_mask
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_video_noise, chunk_audio_noise))
                if not TOGGLE_SINGLE_NOISE and index > 0 and not use_taomate:
                    # Preserve Chunk 1's original full-shot noise slices, so
                    # this experiment cannot change an existing first chunk.
                    # Later prefixes belong to their chunk's independent AV
                    # noise realization.
                    per_chunk_noise_seed = _per_chunk_noise_seed(replay_noise_seed, index)
                    chunk_noise = comfy.sample.prepare_noise(chunk_latent["samples"], per_chunk_noise_seed)
                    chunk_video_noise, chunk_audio_noise = chunk_noise.unbind()
                    if chunk.get("synthetic_prefix"):
                        prefix_video_noise = chunk_video_noise[:, :, :context_video_t]
                        prefix_audio_noise = chunk_audio_noise[..., :context_audio_t]
                    if debug:
                        logging.info(
                            "HR Endless Sampler chunk %d/%d independent AV noise seed: %d.",
                            index + 1, len(active_plan), per_chunk_noise_seed,
                        )

                guide_enabled = context_keyframes > 0
                guide_audio_t = (
                    0 if not guide_enabled
                    else _audio_steps(content_start) - _audio_steps(content_start - context_keyframes)
                )
                video_context = None if previous_video is None or not guide_enabled else previous_video[:, :, -guide_video_t:].clone()
                audio_context = None if previous_audio is None or not guide_enabled else previous_audio[..., -guide_audio_t:].clone()
                if boundary_video_context is not None:
                    video_context = boundary_video_context
                if boundary_audio_context is not None:
                    audio_context = boundary_audio_context
                video_contexts = ()
                video_context_start = 0
                audio_end_frame = float(keyframe_duration_frames)
                if boundary_audio_context is not None:
                    # The exact synthetic-prefix audio ticks begin at local
                    # zero and use the same duration as the sampled prefix.
                    audio_end_frame = boundary_audio_context.shape[-1] / FRAME_RESCALE
                elif audio_context is not None:
                    overhang = previous_audio.shape[-1] - FRAME_RESCALE * previous_frame_count
                    audio_end_frame += overhang / FRAME_RESCALE
                video_items = []
                video_refs = []
                reference_audio = None
                if continuation and include_previous_audio_reference:
                    # Audio is intentionally independent from Video1. This
                    # lets the masked-AV A/B path keep its own mode semantics
                    # while H3 receives the complete preceding chunk's audio.
                    reference_audio = previous_audio.clone()
                    video_items.append({"type": "audio"})
                    video_refs.append(_audio_ref_block(reference_audio))
                if continuation and use_masked_av_overlap:
                    if vae is None:
                        raise ValueError(
                            "Masked AV continuation needs the video VAE to encode its previous-frame boundary keyframe"
                        )
                    timer_started = time.perf_counter()
                    try:
                        if previous_decoded_frames is None:
                            previous_decoded_frames = _decode_video_frames(vae, previous_video).detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            )
                        h3_context_frames, h3_context_kind = _h3_context_frames(
                            previous_decoded_frames,
                            previous_color_corrected_decoded_frames,
                        )
                        if h3_context_kind == "color-corrected":
                            # Re-encode the whole corrected 39-frame tail,
                            # then install it in the fully frozen physical
                            # overlap. This is the direct latent equivalent of
                            # the corrected pixels sent to Qwen and the sparse
                            # H3 boundary guides below.
                            corrected_overlap_latent = vae.encode(
                                h3_context_frames[-video_continuation:]
                            )
                            chunk_video = _replace_masked_av_video_prefix(
                                chunk_video,
                                corrected_overlap_latent,
                                context_video_t,
                            )
                            chunk_latent["samples"] = comfy.nested_tensor.NestedTensor(
                                (chunk_video, chunk_audio)
                            )
                            del corrected_overlap_latent
                        video_contexts = _masked_av_boundary_guides(
                            vae,
                            h3_context_frames,
                            video_continuation,
                        )
                        # The original masked-AV experiment supplied the
                        # boundary images only to H3. If their corrected color
                        # establishes the desired look, Qwen needs the same
                        # evidence while authoring this chunk's multimodal
                        # conditioning. Keep this separate from Video1: no
                        # Video1 latent, audio reference, or continuation
                        # prompt prose is created in masked-AV mode.
                        if h3_context_kind == "color-corrected":
                            if positive and positive[0].get("minimax_refs"):
                                video_items.append(_decoded_video_item_from_frames(
                                    h3_context_frames[-video_continuation:]
                                ))
                            else:
                                logging.warning(
                                    "HR Endless Sampler could not attach color-corrected masked-AV frames to Qwen; "
                                    "the positive conditioning has no MiniMax Ref2VA reference slots."
                                )
                    finally:
                        timing.add("vae_context", timer_started)
                    logging.info(
                        "HR Endless Sampler chunk %d/%d masked AV continuation: "
                        "%s previous-chunk decoded overlap%s anchored at local frames %s",
                        index + 1,
                        len(active_plan),
                        h3_context_kind,
                        " re-encoded into the frozen latent prefix" if h3_context_kind == "color-corrected" else "",
                        ", ".join(str(item["resolved_frame_index"]) for item in video_contexts),
                    )
                    if h3_context_kind == "color-corrected" and video_items:
                        logging.info(
                            "HR Endless Sampler chunk %d/%d attached the color-corrected %d-frame masked-AV tail to Qwen.",
                            index + 1,
                            len(active_plan),
                            video_continuation,
                        )
                # Video1 also receives the last full frame as an ordinary picture;
                # this does not add a minimax_keyframes entry or change prefix noise.
                if continuation and include_video1_reference:
                    if include_video1_reference:
                        reference_latent = previous_video[:, :, -_video_steps(video_continuation):].clone()
                        full_reference_latent = reference_latent
                        if vram_monitor is not None:
                            vram_monitor.report(
                                f"chunk {index + 1}/{len(active_plan)} before continuation VAE decode",
                                {
                                    "continuation video latent": reference_latent,
                                    "continuation audio latent": reference_audio,
                                },
                            )
                        timer_started = time.perf_counter()
                        try:
                            if previous_decoded_frames is None:
                                previous_decoded_frames = _decode_video_frames(vae, previous_video).detach().to(device="cpu", dtype=torch.float32)
                            h3_context_frames, h3_context_kind = _h3_context_frames(
                                previous_decoded_frames,
                                previous_color_corrected_decoded_frames,
                            )
                            picture_item, picture_ref = _last_frame_picture_reference(vae, h3_context_frames, width, height)
                            video_items.append(picture_item)
                            video_refs.append(picture_ref)
                            if h3_context_kind == "color-corrected":
                                # This is intentionally an A/B path: the
                                # display-corrected tail is re-encoded so H3
                                # can see exactly the colors shown at the
                                # preceding output boundary.
                                decoded_reference_frames = h3_context_frames[-video_continuation:].clone()
                                reference_latent = vae.encode(decoded_reference_frames)
                                if reference_latent.ndim != 5 or reference_latent.shape[1] != 24:
                                    raise ValueError("MiniMax H3 color-corrected Video1 encode did not return a 24-channel video latent")
                            else:
                                # Preserve the established conditioning path:
                                # decode the bounded Video1 latent independently.
                                # Cropping the full finalized-chunk decode can have
                                # different temporal VAE boundary context and would
                                # therefore change generation, not merely preview.
                                decoded_reference_frames = _decode_video_frames(vae, reference_latent)
                            video_items.append(_decoded_video_item_from_frames(decoded_reference_frames))
                            resized_reference_latent, reference_canvas = _encode_resized_continuation_reference(
                                vae,
                                decoded_reference_frames,
                                video_continuation_res,
                            )
                            if resized_reference_latent is not None:
                                if resized_reference_latent.shape[2] != reference_latent.shape[2]:
                                    raise ValueError(
                                        "MiniMax H3 resized Video1 changed temporal latent length "
                                        f"from {reference_latent.shape[2]} to {resized_reference_latent.shape[2]}"
                                    )
                                reference_latent = resized_reference_latent
                            if debug:
                                logging.info(
                                    "HR Endless Sampler chunk %d/%d H3 Video1 reference (%s source): %s, "
                                    "%dx%d pixels -> latent %dx%d with %d temporal tokens",
                                    index + 1,
                                    len(active_plan),
                                    h3_context_kind,
                                    video_continuation_res,
                                    reference_canvas[0],
                                    reference_canvas[1],
                                    reference_latent.shape[4],
                                    reference_latent.shape[3],
                                    reference_latent.shape[2],
                                )
                        finally:
                            timing.add("vae_context", timer_started)
                            decoded_reference_frames = None
                        _log_continuation_payload(
                            index + 1,
                            len(active_plan),
                            video_continuation_res,
                            reference_latent,
                            reference_audio,
                            boundary_video_context,
                            chunk_video,
                            chunk_audio,
                            full_reference_latent,
                        )
                        full_reference_latent = None
                        video_refs.append(_video_ref_block(reference_latent))
                if continuation and qwen_full_history:
                    history_latent = torch.cat(output_video, dim=2)
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} before history VAE decode",
                            {"history latent": history_latent},
                        )
                    timer_started = time.perf_counter()
                    try:
                        video_items.append(_decoded_video_item(vae, history_latent))
                    finally:
                        timing.add("vae_history", timer_started)
                        del history_latent
                if debug and video_items:
                    presentations = ", ".join(
                        f"{item['data'].shape[0]} frames at {item['data'].shape[2]}x{item['data'].shape[1]}"
                        for item in video_items if item["type"] == "video"
                    )
                    logging.info(
                        "HR Endless Sampler chunk %d/%d Qwen video presentation: %s",
                        index + 1,
                        len(active_plan),
                        presentations,
                    )
                if gemma_director is not None:
                    if gemma_description is None:
                        raise RuntimeError("Gemma director completed without a detailed_description")
                    continuation_video_label = f"<Video {video_number}>" if continuation and include_video1_reference else None
                    continuation_audio_label = f"<Audio {audio_number}>" if continuation and include_previous_audio_reference else None
                    audio_speakers = _audio_reference_speakers(previous_gemma_description) if continuation_audio_label else ()
                    chunk_prompt = _prompt_with_gemma_description(
                        prompt,
                        gemma_description,
                        drop_picture_anchors=continuation and not ref2va,
                        continuation_video_label=continuation_video_label,
                        continuation_audio_label=continuation_audio_label,
                        retention_analysis=gemma_retention_analysis,
                        audio_speakers=audio_speakers,
                        summary=result.summary if continuation else None,
                    )
                    debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report)
                else:
                    chunk_prompt, debug_prompt = planned_prompts[index]
                    if gemma_report is not None:
                        debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report)
                # Native audio-first text is already finalized before teacher audio.
                # Requesting it again for video must return exactly the same text.
                if use_taomate and audio_first_enabled and gemma_director is None:
                    chunk_prompt = manual_prompts.get_chunk_prompt(index + 1, previous_audio_transcription=audio_transcripts[index - 1]["text"] if index else None, prompt_stage="video") if manual_prompts is not None else _native_prompt_for_stage(planned_prompts[index][0], "video", audio_requested_prompts[index - 1] if index else None, audio_transcripts[index - 1] if index else None)
                elif manual_prompts is not None:
                    chunk_prompt = manual_prompts.get_chunk_prompt(index + 1, previous_audio_transcription=audio_transcripts[index - 1]["text"] if index and audio_transcripts else None, prompt_stage="video")
                else:
                    chunk_prompt = _chunk_summary_prompt(chunk_prompt, prompt, continuation, picture_label=continuation_picture_label, video_label=f"<Video {video_number}>" if continuation and include_video1_reference else None, audio_label=f"<Audio {audio_number}>" if continuation and include_previous_audio_reference else None, boundary_keyframe=continuation and bool(use_masked_av_mode or context_keyframes))
                if not (use_taomate and audio_first_enabled and gemma_director is None):
                    chunk_prompt = _preserve_global_prompt_sections(chunk_prompt, prompt)
                if use_taomate and not audio_first_enabled:
                    if manual_prompts is not None:
                        chunk_audio_prompt = manual_prompts.get_chunk_prompt(index + 1, previous_audio_transcription=audio_transcripts[index - 1]["text"] if index and audio_transcripts else None, prompt_stage="audio")
                    elif gemma_director is None:
                        chunk_audio_prompt = _native_prompt_for_stage(chunk_prompt, "audio", audio_requested_prompts[index - 1] if index else None, audio_transcripts[index - 1] if index and audio_transcripts else None)
                        chunk_prompt = _native_prompt_for_stage(chunk_prompt, "video", audio_requested_prompts[index - 1] if index else None, audio_transcripts[index - 1] if index and audio_transcripts else None)
                    audio_requested_prompts.append(chunk_audio_prompt)
                    preview_chunk_ranges[index]["audio_teacher_prompt"] = chunk_audio_prompt.strip()
                    if preview_execution is not None:
                        preview_execution.set_audio_prompt(index, chunk_audio_prompt.strip(), _preview_subtitle(chunk_audio_prompt))
                if use_taomate and audio_first_enabled and pre_production is None and TOGGLE_TAOMATE_DIVERGENCY_AUDIO_TRANSCRIPT_RETRY:
                    # A rejected take revised this one native AV prompt, not a
                    # separate audio-only prompt. Video uses that exact text.
                    chunk_prompt = audio_requested_prompts[index]
                if use_taomate and audio_first_enabled and pre_production is None and chunk_prompt != audio_requested_prompts[index]:
                    raise RuntimeError("Native TaoMate audio and video chunk prompts diverged")
                debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report)
                if return_prompts:
                    debug_prompts.append(debug_prompt)
                if gemma_director is not None:
                    # Keep an exact, immediately flushed transcript of Gemma's
                    # request/response and the structured prompt encoded for H3.
                    _append_last_gemma_prompt(
                        gemma_prompt_log,
                        _debug_chunk_header(index, chunk, content_start),
                        chunk_prompt,
                        system_prompt=gemma_system_prompt if not gemma_system_logged else None,
                        observation_prompt=gemma_observation_prompt,
                        gemma_response=gemma_response,
                        validation_warnings=gemma_validation_warnings,
                    )
                    if gemma_system_prompt:
                        gemma_system_logged = True
                if debug:
                    logging.info("HR Endless Sampler debug:\n%s", debug_prompt)
                preview_chunk_ranges[index]["h3_prompt"] = chunk_prompt
                if vram_monitor is not None:
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} before Qwen encode",
                        {
                            "chunk latent": chunk_latent,
                            "chunk noise": chunk_noise,
                            "Qwen video frames": video_items,
                            "DiT video references": video_refs,
                        },
                    )
                qwen_message = f"{chunk_label}: encoding conditioning with Qwen"
                logging.info("HR Endless Sampler: %s.", qwen_message)
                if preview_execution is not None:
                    preview_execution.set_phase(qwen_message, chunk=index)
                timer_started = time.perf_counter()
                try:
                    encoded_prompt = _encode_prompt(clip, chunk_prompt, h3_images, positive, width, height, continuation, video_items)
                    per_chunk_audio_conditioning = None
                    if use_taomate and not audio_first_enabled:
                        encoded_audio_prompt = encoded_prompt if chunk_audio_prompt == chunk_prompt else _encode_prompt(clip, chunk_audio_prompt, h3_images, positive, width, height, continuation, video_items)
                        per_chunk_audio_conditioning = {"cross_attn": encoded_audio_prompt[0].detach().to(device="cpu")}
                finally:
                    timing.add("qwen", timer_started)
                if continuation and include_video1_reference:
                    qwen_cross_attn = encoded_prompt[0]
                    logging.info(
                        "HR Endless Sampler chunk %d/%d Qwen conditioning retained for H3 "
                        "(complete prompt + original references + Video1 presentation): "
                        "shape=%s, raw=%0.3f MiB. This does not change with video_continuation_res.",
                        index + 1,
                        len(active_plan),
                        tuple(qwen_cross_attn.shape),
                        qwen_cross_attn.numel() * qwen_cross_attn.element_size() / (1024 ** 2),
                    )
                retry_video_items = video_items if use_taomate and not audio_first_enabled and TOGGLE_TAOMATE_DIVERGENCY_AUDIO_TRANSCRIPT_RETRY else None
                del video_items
                if vram_monitor is not None:
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} after Qwen encode",
                        {"encoded prompt": encoded_prompt, "DiT video references": video_refs},
                    )
                comfy.model_management.unload_model_and_clones(clip.patcher)
                if vae is not None:
                    comfy.model_management.unload_model_and_clones(vae.patcher)
                if audio_vae is not None:
                    comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                previous_decoded_frames = None
                previous_color_corrected_decoded_frames = None
                if debug:
                    logging.info(
                        "HR Endless Sampler released Qwen%s%s before chunk %d/%d",
                        " and the H3 video VAE" if vae is not None else "",
                        " and the H3 audio VAE" if audio_vae is not None else "",
                        index + 1,
                        len(active_plan),
                    )
                vram_monitor.report(
                    f"chunk {index + 1}/{len(active_plan)} after Qwen/VAE release",
                    {"encoded prompt": encoded_prompt, "DiT video references": video_refs},
                )
                guider.original_conds = _conditioning_for_chunk(
                    original_conds,
                    chunk["frame_start"] + chunk["output_trim_frames"] if use_taomate else chunk["frame_start"],
                    chunk["frame_end"],
                    encoded_prompt,
                    video_context,
                    audio_context,
                    audio_end_frame,
                    video_refs,
                    video_context_start,
                    video_contexts,
                    linear_original_refs,
                )

                # Every dependency on the previous sampler container has now
                # been converted into the bounded guide/reference tensors in
                # the current conditioning. The accumulated output already
                # owns its trimmed clone, so do not keep the previous full
                # nested AV result alive through the next DiT evaluation.
                if continuation:
                    previous_video = None
                    previous_audio = None

                chunk_seed = (replay_noise_seed + index) & 0xffffffffffffffff
                # This durable record is also the sampler's finished-video
                # timeline output. It must be populated even when no live
                # preview wrapper is present in the model path.
                if isinstance(gemma_description, str) and gemma_description.strip():
                    preview_chunk_ranges[index]["gemma_detailed_description"] = gemma_description.strip()
                if isinstance(gemma_retention_analysis, str) and gemma_retention_analysis.strip():
                    preview_chunk_ranges[index]["gemma_retention_analysis"] = gemma_retention_analysis.strip()
                if preview_execution is not None:
                    preview_range = preview_chunk_ranges[index]
                    preview_keeps_prefix = bool(
                        continuation and keep_continuation_prefix
                    )
                    preview_execution.set_chunk(
                        index,
                        chunk["frame_start"],
                        chunk["frame_end"] - 1,
                        preview_range["start"],
                        preview_range["end"],
                        0 if preview_keeps_prefix or use_taomate else context_video_t,
                        gemma_description,
                        gemma_retention_analysis,
                        chunk_prompt,
                    )
                try:
                    sampling_message = f"{chunk_label}: starting H3 inference"
                    logging.info("HR Endless Sampler: %s.", sampling_message)
                    if preview_execution is not None:
                        preview_execution.set_phase(sampling_message, chunk=index)
                    chunk_progress.start(index)
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} immediately before sampler",
                            {
                                "chunk latent": chunk_latent,
                                "chunk noise": chunk_noise,
                                "conditioning": guider.original_conds,
                                "completed output": (output_video, output_audio),
                                "completed denoised output": (denoised_video, denoised_audio),
                            },
                        )
                    timer_started = time.perf_counter()
                    try:
                        if debug and index == 0:
                            from .python.h3_diagnostics import H3FirstStepDiagnostic
                            first_diagnostic = H3FirstStepDiagnostic(video_continuation_method, {"prompt": chunk_prompt, "seed": chunk_seed, "sigmas": sigmas, "noise": chunk_noise.unbind(), "target": chunk_latent["samples"].unbind(), "noise_mask": chunk_latent.get("noise_mask"), "conditioning": guider.original_conds, "chunk": chunk, "cfg": getattr(guider, "cfg", None)})
                            if use_taomate:
                                taomate_backend.diagnostic = first_diagnostic
                            else:
                                guider.model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "hr_first_step_diagnostic", first_diagnostic.forward)
                        if use_taomate:
                            taomate_backend.current_chunk_number = index + 1
                            taomate_backend.current_chunk_total = len(active_plan)
                            def publish_chunk_teacher_audio(audio_index, audio_latent):
                                """Decode and transcribe this teacher before its video phases."""
                                if preview_execution is not None:
                                    preview_execution.set_phase(f"Chunk {audio_index + 1}: decoding and transcribing teacher audio", chunk=audio_index)
                                audio_started = time.perf_counter()
                                try:
                                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                                    comfy.model_management.unload_model_and_clones(clip.patcher)
                                    previous_teacher = teacher_audio_latents.get(audio_index - 1)
                                    if previous_teacher is not None:
                                        previous_teacher = previous_teacher.to(device=audio_latent.device, dtype=audio_latent.dtype)
                                    waveform, sample_rate = taomate_backend.decode_audio_preview_segment(audio_vae, audio_latent, previous_teacher, chunk["context_audio_t"], AUDIO_LATENT_FPS)
                                    recognized = audio_transcriber.transcribe(waveform, sample_rate)
                                    audio_transcripts.append(recognized)
                                    audio_observations.append((waveform.detach().to(device="cpu"), int(sample_rate)))
                                    logging.info("TaoMate audio teacher transcript %d/%d: expected=%r, heard=%r", audio_index + 1, len(active_plan), _preview_subtitle(audio_requested_prompts[audio_index]), recognized["text"])
                                    if audio_index in audio_retried_chunks and re.findall(r"\w+(?:['’]\w+)?", _preview_subtitle(audio_requested_prompts[audio_index]).casefold()) != re.findall(r"\w+(?:['’]\w+)?", recognized["text"].casefold()):
                                        logging.warning("TaoMate audio teacher %d/%d still differs from its dialogue after one retry", audio_index + 1, len(active_plan))
                                    teacher_audio_latents.clear()
                                    teacher_audio_latents[audio_index] = audio_latent.detach().to(device="cpu")
                                    audio_teacher_preview_chunks[audio_index] = (waveform, sample_rate)
                                    if preview_execution is not None:
                                        if audio_index in audio_retried_chunks:
                                            preview_execution.set_audio_retry(audio_index, finished=True)
                                        audio_range = preview_chunk_ranges[audio_index]
                                        preview_execution.publish_chunk_audio_first(audio_index, waveform, sample_rate, audio_range["start"], audio_range["end"])
                                finally:
                                    comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                                    timing.add("vae_audio_preview", audio_started)

                            def retry_chunk_teacher_audio(audio_index, audio_latent):
                                """Reject a mismatched take and re-encode one shared AV prompt."""
                                nonlocal retry_video_items, chunk_prompt, debug_prompt
                                previous_teacher = teacher_audio_latents.get(audio_index - 1)
                                comfy.model_management.unload_model_and_clones(guider.model_patcher)
                                waveform, sample_rate = taomate_backend.decode_audio_preview_segment(audio_vae, audio_latent, previous_teacher, chunk["context_audio_t"], AUDIO_LATENT_FPS)
                                recognized = audio_transcriber.transcribe(waveform, sample_rate)
                                requested = audio_requested_prompts[audio_index]
                                native_dialogue = manual_prompts is None and gemma_director is None
                                approved_prompt = _audio_transcript_approved_tail_prompt(requested, recognized) if native_dialogue else None
                                if approved_prompt is not None:
                                    # The take spoke a clean prefix of its dialogue, so
                                    # its audio stays; only the words it did not say are
                                    # re-authored with the initial cut over the chunks
                                    # still to be sampled.
                                    texts = [item[0] for item in planned_prompts]
                                    texts[audio_index] = requested
                                    revised = _recut_dialogue_prompts(texts, active_plan, fps, audio_index, audio_index + 1, kept_words=len(_dialogue_block_words(approved_prompt)[3]))
                                    if revised is None:
                                        logging.info("TaoMate audio teacher %d/%d accepted its spoken dialogue prefix.", audio_index + 1, len(active_plan))
                                        revised = texts
                                        revised[audio_index] = approved_prompt
                                    else:
                                        logging.info("TaoMate audio teacher %d/%d accepted its spoken dialogue prefix and re-authored the remaining dialogue.", audio_index + 1, len(active_plan))
                                    _publish_revised_dialogue(planned_prompts, revised, audio_index, preview_chunk_ranges, preview_execution)
                                    audio_requested_prompts[audio_index] = revised[audio_index]
                                    chunk_prompt = revised[audio_index]
                                    debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report)
                                    preview_chunk_ranges[audio_index]["audio_teacher_prompt"] = chunk_prompt.strip()
                                    return None
                                if _teacher_take_matches_dialogue(_preview_subtitle(requested), recognized["text"]):
                                    retry_video_items = None
                                    return None
                                if not _transcript_is_certain(recognized):
                                    logging.warning("Teacher audio transcript is uncertain; skipping prompt retry")
                                    retry_video_items = None
                                    return None
                                logging.warning("TaoMate audio teacher %d/%d dialogue mismatch; retrying once: expected=%r, heard=%r", audio_index + 1, len(active_plan), _preview_subtitle(requested), recognized["text"])
                                retry_prompt = requested
                                if native_dialogue:
                                    texts = [item[0] for item in planned_prompts]
                                    texts[audio_index] = requested
                                    revised, shrink_frames = _retry_take_prompts(texts, active_plan, fps, audio_index)
                                    if revised is not None:
                                        _publish_revised_dialogue(planned_prompts, revised, audio_index, preview_chunk_ranges, preview_execution)
                                        retry_prompt = revised[audio_index]
                                        logging.info("TaoMate audio teacher %d/%d re-cut the leftover dialogue over the remaining chunks, %d frames shorter.", audio_index + 1, len(active_plan), shrink_frames)
                                audio_retried_chunks.add(audio_index)
                                audio_requested_prompts[audio_index] = retry_prompt
                                chunk_prompt = retry_prompt
                                debug_prompt = _debug_chunk_prompt(index, chunk, content_start, retry_prompt, gemma_report)
                                preview_chunk_ranges[audio_index]["audio_teacher_prompt"] = retry_prompt.strip()
                                preview_chunk_ranges[audio_index]["h3_prompt"] = retry_prompt.strip()
                                if preview_execution is not None:
                                    preview_execution.set_audio_retry(audio_index)
                                    preview_execution.set_audio_prompt(audio_index, retry_prompt.strip(), _preview_subtitle(retry_prompt))
                                    preview_execution.set_phase("TaoMate: retrying teacher audio %d/%d after transcript mismatch" % (audio_index + 1, len(active_plan)), chunk=audio_index)
                                comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                                encoded_retry = _encode_prompt(clip, retry_prompt, h3_images, positive, width, height, continuation, retry_video_items)
                                retry_video_items = None
                                comfy.model_management.unload_model_and_clones(clip.patcher)
                                tags = encoded_retry[1].get("minimax_token_tags")
                                for cond in guider.original_conds["positive"]:
                                    cond["cross_attn"] = encoded_retry[0]
                                    if tags is not None:
                                        cond["minimax_token_tags"] = tags
                                    else:
                                        cond.pop("minimax_token_tags", None)
                                retry_conditioning = {"cross_attn": encoded_retry[0].detach().to(device="cpu")}
                                if tags is not None:
                                    retry_conditioning["minimax_token_tags"] = tags.detach().to(device="cpu")
                                return retry_conditioning

                            def subchunk_start(phase):
                                """Locate live previews on this phase's actual global frame range."""
                                if preview_execution is not None:
                                    preview_execution.set_subchunk(index, taomate_backend.frames(phase["video_start"]), phase["frame_end"] - 1, phase["video_start"], chunk["phases"].index(phase) + 1)

                            def subchunk_complete(completed_frames):
                                """Fill the request timeline as each internal phase finishes."""
                                preview_chunk_ranges[index]["taomate_completed_frames"] = completed_frames
                                if preview_execution is not None:
                                    completed_phases = sum(1 for phase in chunk["phases"] if phase["frame_end"] - taomate_backend.frames(chunk["video_start"]) <= completed_frames)
                                    preview_execution.set_subchunk_progress(index, completed_frames, completed_phases)
                            sampled, denoised = taomate_backend.execute_chunk(super().execute, _FixedNoise, chunk_seed, chunk_noise, guider, sigmas, chunk_latent, chunk, on_subchunk=subchunk_complete, sampler=sampler, on_subchunk_start=subchunk_start, on_status=(lambda message: preview_execution.set_phase(message, chunk=index)) if preview_execution is not None else None, debug_timing=debug, on_audio_teacher=publish_chunk_teacher_audio if audio_vae is not None and not audio_first_enabled else None, on_audio_progress=(lambda step, steps: preview_execution.set_phase("TaoMate: audio teacher %d/%d · step %d/%d" % (index + 1, len(active_plan), step, steps), chunk=index, audio_step_ms=taomate_backend.audio_teacher.average_step_ms)) if preview_execution is not None and not audio_first_enabled else None, audio_conditioning=per_chunk_audio_conditioning, on_audio_candidate=retry_chunk_teacher_audio if TOGGLE_TAOMATE_DIVERGENCY_AUDIO_TRANSCRIPT_RETRY and audio_vae is not None and not audio_first_enabled else None)
                        else:
                            sampled, denoised = super().execute(
                                _FixedNoise(chunk_seed, chunk_noise), guider, sampler, sigmas, chunk_latent
                            )
                    finally:
                        if debug and index == 0:
                            guider.model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "hr_first_step_diagnostic")
                            if use_taomate:
                                taomate_backend.diagnostic = None
                        h3_render_seconds = timing.add("h3_sampling", timer_started)
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} sampler complete",
                            {"sampled": sampled, "denoised": denoised},
                        )
                finally:
                    if preview_execution is not None:
                        preview_execution.clear_chunk()
                # Preserve latent metadata without making the template another
                # owner of a full per-chunk nested sample. Final concatenated
                # samples are installed into these dictionaries after the loop.
                output_template = sampled.copy()
                denoised_template = denoised.copy()
                output_template.pop("samples", None)
                denoised_template.pop("samples", None)
                output_template.pop("noise_mask", None)
                denoised_template.pop("noise_mask", None)
                previous_video, previous_audio = sampled["samples"].unbind()
                previous_frame_count = chunk["frame_end"] - chunk["frame_start"]
                denoised_chunk_video, denoised_chunk_audio = denoised["samples"].unbind()

                video_trim = 0 if (
                    continuation
                    and keep_continuation_prefix
                ) else context_video_t
                audio_trim = 0 if (
                    index == 0
                    or (keep_continuation_prefix)
                ) else context_audio_t
                assembled_video = previous_video[:, :, video_trim:].clone()
                masked_audio_prefix = None
                masked_denoised_audio_prefix = None
                if continuation and use_masked_av_overlap and TRIM_MASKED_AV_PREFIX:
                    # The new chunk's feathered audio is authoritative across
                    # the overlap. Replace the matching accumulated tail, then
                    # append only audio beyond that physical overlap.
                    masked_audio_prefix = previous_audio[..., :audio_trim].clone()
                    masked_denoised_audio_prefix = denoised_chunk_audio[..., :audio_trim].clone()
                    _replace_stream_tail(output_audio, masked_audio_prefix)
                    _replace_stream_tail(denoised_audio, masked_denoised_audio_prefix)
                elif continuation and use_masked_av_mode and TRIM_MASKED_AV_PREFIX and audio_trim:
                    # Mirror the video-tail handoff directly in audio-latent
                    # space; no audio VAE round trip reaches H3.
                    masked_audio_prefix = previous_audio[..., :audio_trim].clone()
                    masked_denoised_audio_prefix = denoised_chunk_audio[..., :audio_trim].clone()
                    _replace_stream_tail(output_audio, masked_audio_prefix)
                    _replace_stream_tail(denoised_audio, masked_denoised_audio_prefix)
                assembled_audio = previous_audio[..., audio_trim:].clone()
                assembled_denoised_video = denoised_chunk_video[:, :, video_trim:].clone()
                assembled_denoised_audio = denoised_chunk_audio[..., audio_trim:].clone()
                if replay_output_on_cpu:
                    output_video.append(assembled_video.to(device="cpu"))
                    output_audio.append(assembled_audio.to(device="cpu"))
                    denoised_video.append(assembled_denoised_video.to(device="cpu"))
                    denoised_audio.append(assembled_denoised_audio.to(device="cpu"))
                else:
                    output_video.append(assembled_video)
                    output_audio.append(assembled_audio)
                    denoised_video.append(assembled_denoised_video)
                    denoised_audio.append(assembled_denoised_audio)

                # Replace the approximate latent preview with the authoritative
                # full video-VAE frames as soon as this physical chunk is
                # complete. Keep that CPU decode for the next Gemma handoff so
                # finalization does not add a duplicate whole-chunk video
                # decode. The audio VAE is separate in MiniMax H3 and is loaded
                # only for this short post-sampling phase.
                final_preview_frames = None
                final_preview_audio = None
                final_preview_audio_rate = None
                audio_teacher_preview_used = False
                final_preview_overlap_audio = None
                preview_range = preview_chunk_ranges[index]
                preview_output_start = int(preview_range["start"])
                preview_trim_frames = 0
                if vae is not None:
                    finalization_message = f"{chunk_label}: decoding final video/audio preview"
                    logging.info("HR Endless Sampler: %s.", finalization_message)
                    if preview_execution is not None:
                        preview_execution.set_phase(finalization_message, chunk=index)
                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    if use_taomate:
                        # TurboQuant has already copied retained KV to CPU RAM;
                        # force-release the H3 device residency before VAE decode.
                        comfy.model_management.unload_all_models()
                    comfy.model_management.soft_empty_cache(force=True)
                    timer_started = time.perf_counter()
                    try:
                        previous_decoded_frames = _decode_video_frames(vae, previous_video).detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        retained_start = max(0, int(chunk.get("output_trim_frames", 0)))
                        preview_keeps_prefix = bool(
                            continuation and keep_continuation_prefix
                        )
                        preview_trim_frames = 0 if preview_keeps_prefix else retained_start
                        retained_count = max(0, int(preview_range["end"]) - preview_output_start + 1)
                        # MKL is a pixel-space color matcher. Keep the decoded
                        # compute frames float32, but present sRGB values to it.
                        decoded_display_frames = _convert_image_transfer(previous_decoded_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else previous_decoded_frames
                        correction_reference = _decoded_frame_before_tail(decoded_linear_output_frames, preview_trim_frames)
                        correction_reference = _convert_image_transfer(correction_reference, _inverse_gamma_compute_to_srgb) if linear_color_compute else correction_reference
                        correction_frame_count = _same_shot_correction_frames(
                            gemma_shots,
                            content_start,
                            chunk["frame_end"],
                        )
                        (
                            corrected_decoded_frames,
                            color_transform,
                            corrected_frame_count,
                        ) = _correct_decoded_chunk_color(
                            correction_reference,
                            decoded_display_frames,
                            0,
                            correction_frame_count + (preview_trim_frames if correction_frame_count else 0),
                            enabled=correct_chunk_boundaries,
                        )
                        if correct_all_chunks:
                            corrected_decoded_frames = _correct_ready_entire_shot_frames(
                                corrected_decoded_frames,
                                preview_output_start,
                                gemma_shots,
                                gemma_preproduction_timing_plan,
                                entire_shot_color_anchors,
                            )
                        corrected_decoded_frames = _convert_image_transfer(corrected_decoded_frames, _srgb_to_inverse_gamma_compute_rgb) if linear_color_compute else corrected_decoded_frames
                        # Keep the normal raw decode and a separate finalized
                        # corrected copy. The module-level experiment switch
                        # selects the latter only for H3's pixel-space
                        # continuation input on the next chunk.
                        previous_color_corrected_decoded_frames = corrected_decoded_frames
                        final_linear_frames = corrected_decoded_frames[
                            preview_trim_frames:preview_trim_frames + retained_count
                        ].clone()
                        final_preview_frames = _convert_image_transfer(final_linear_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else final_linear_frames
                        if retained_count and int(final_preview_frames.shape[0]) != retained_count:
                            raise ValueError(
                                f"Chunk {index + 1} decoded {int(previous_decoded_frames.shape[0])} frames, "
                                f"but {retained_count} retained frames were required after trimming {retained_start}"
                            )
                        if preview_trim_frames and decoded_linear_output_frames and not use_taomate:
                            _replace_decoded_frame_tail(decoded_linear_output_frames, corrected_decoded_frames[:preview_trim_frames])
                            if linear_color_compute:
                                _replace_decoded_frame_tail(decoded_output_frames, _convert_image_transfer(corrected_decoded_frames[:preview_trim_frames], _inverse_gamma_compute_to_srgb))
                            if preview_execution is not None:
                                previous_range = preview_chunk_ranges[index - 1]
                                preview_execution.replace_video_tail(
                                    index - 1, decoded_output_frames[-1], previous_range["start"], previous_range["end"],
                                    gemma_detailed_description=previous_range.get("gemma_detailed_description"),
                                    gemma_retention_analysis=previous_range.get("gemma_retention_analysis"),
                                )
                        decoded_linear_output_frames.append(final_linear_frames)
                        if linear_color_compute:
                            decoded_output_frames.append(final_preview_frames)
                        _log_output_color_correction(
                            "Chunk %d" % (index + 1),
                            color_transform,
                            corrected_frame_count,
                        )
                        final_color_diagnostics = _safe_video_color_diagnostics(
                            final_preview_frames,
                            index + 1,
                        )
                        if final_color_diagnostics is not None:
                            _record_color_diagnostics(
                                color_diagnostics,
                                index,
                                final_color_diagnostics,
                                _color_boundary_context(gemma_shots, content_start),
                            )
                    except Exception as error:
                        decoded_video_complete = False
                        previous_decoded_frames = None
                        previous_color_corrected_decoded_frames = None
                        logging.warning(
                            "HR Endless Sampler could not decode final video preview Chunk %d; "
                            "keeping its latent preview and retrying the normal decode if the next chunk needs it: %s",
                            index + 1,
                            error,
                        )
                    finally:
                        timing.add("vae_previous_chunk", timer_started)
                        comfy.model_management.unload_model_and_clones(vae.patcher)
                        comfy.model_management.soft_empty_cache(force=True)

                if audio_vae is not None:
                    timer_started = time.perf_counter()
                    try:
                        retained_audio_frames = retained_count
                        teacher_preview_audio = audio_teacher_preview_chunks.pop(index, None) if use_taomate else None
                        if teacher_preview_audio is not None:
                            final_preview_audio, final_preview_audio_rate = teacher_preview_audio
                            audio_teacher_preview_used = True
                        elif use_taomate and audio_first_decoded and decoded_output_audio:
                            rate = int(decoded_audio_sample_rate)
                            audio_start = round(int(preview_range["start"]) * rate / float(fps))
                            audio_end = round((int(preview_range["end"]) + 1) * rate / float(fps))
                            final_preview_audio = decoded_output_audio[0][..., audio_start:audio_end]
                            final_preview_audio_rate = rate
                        elif continuation and use_masked_av_mode and TRIM_MASKED_AV_PREFIX:
                            overlap_frames = max(0, int(chunk.get("output_trim_frames", 0)))
                            full_preview_audio, final_preview_audio_rate = _decode_audio_preview(
                                audio_vae,
                                previous_audio,
                                trim_latent_steps=0,
                                output_frames=overlap_frames + retained_audio_frames,
                                fps=fps,
                            )
                            overlap_samples = max(
                                0,
                                round(
                                    audio_trim * final_preview_audio_rate / AUDIO_LATENT_FPS
                                    if not use_masked_av_overlap else overlap_frames * final_preview_audio_rate / float(fps)
                                ),
                            )
                            retained_samples = max(
                                0,
                                round(retained_audio_frames * final_preview_audio_rate / float(fps)),
                            )
                            final_preview_overlap_audio = full_preview_audio[..., :overlap_samples]
                            final_preview_audio = full_preview_audio[
                                ..., overlap_samples:overlap_samples + retained_samples
                            ]
                        else:
                            final_preview_audio, final_preview_audio_rate = _decode_audio_preview(
                                audio_vae,
                                previous_audio,
                                trim_latent_steps=audio_trim,
                                output_frames=retained_audio_frames,
                                fps=fps,
                                normalize=not use_taomate,
                            )
                    except Exception as error:
                        if not audio_first_decoded:
                            decoded_audio_complete = False
                        logging.warning(
                            "HR Endless Sampler could not decode final audio preview Chunk %d; "
                            "the final video preview will remain playable without sound: %s",
                            index + 1,
                            error,
                        )
                    finally:
                        if not (use_taomate and (audio_first_decoded or audio_teacher_preview_used)):
                            timing.add("vae_audio_preview", timer_started)
                            comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                            comfy.model_management.soft_empty_cache(force=True)

                if (final_preview_audio is None or final_preview_audio_rate is None) and not audio_first_decoded:
                    decoded_audio_complete = False
                elif final_preview_audio is not None and final_preview_audio_rate is not None:
                    if decoded_audio_sample_rate is None:
                        decoded_audio_sample_rate = int(final_preview_audio_rate)
                    elif decoded_audio_sample_rate != int(final_preview_audio_rate):
                        decoded_audio_complete = False
                        logging.warning(
                            "HR Endless Sampler decoded audio sample rate changed from %d to %d; "
                            "the assembled AUDIO output will be unavailable.",
                            decoded_audio_sample_rate,
                            int(final_preview_audio_rate),
                        )
                    if decoded_audio_complete and not audio_first_decoded:
                        try:
                            if final_preview_overlap_audio is not None:
                                _replace_stream_tail(decoded_output_audio, final_preview_overlap_audio)
                            decoded_output_audio.append(final_preview_audio.clone())
                        except (RuntimeError, ValueError) as error:
                            decoded_audio_complete = False
                            logging.warning(
                                "HR Endless Sampler could not assemble corrected audio for Chunk %d; "
                                "the latent and video outputs remain available: %s",
                                index + 1,
                                error,
                            )

                # Preview enhancement happens only after original samples were stored above.
                if audio_sr and final_preview_audio is not None:
                    if preview_execution is not None:
                        preview_execution.set_phase(f"{chunk_label}: AudioSR preview audio", chunk=index)
                    audio_sr_started = time.perf_counter()
                    final_preview_audio, final_preview_audio_rate, final_preview_overlap_audio = _enhance_decoded_audio(final_preview_audio, final_preview_audio_rate, final_preview_overlap_audio, enabled=True, device=guider.model_patcher.load_device, seed=chunk_seed % (2 ** 32))
                    timing.add("audiosr_preview", audio_sr_started)
                if (
                    preview_execution is not None
                    and final_preview_overlap_audio is not None
                    and final_preview_audio_rate is not None
                ):
                    preview_execution.replace_audio_tail(
                        index,
                        final_preview_overlap_audio,
                        final_preview_audio_rate,
                    )
                    logging.info(
                        "HR Endless Sampler chunk %d/%d preview audio: moved the generated "
                        "%d-frame overlap into the finalized preview tail before this chunk.",
                        index + 1,
                        len(active_plan),
                        int(chunk.get("output_trim_frames", 0)),
                    )
                if preview_execution is not None and final_preview_frames is not None:
                    preview_execution.finalize_chunk(
                        index,
                        final_preview_frames,
                        preview_output_start,
                        preview_range["end"],
                        audio_waveform=final_preview_audio,
                        audio_sample_rate=final_preview_audio_rate,
                        gemma_detailed_description=gemma_description,
                        gemma_retention_analysis=gemma_retention_analysis,
                    )
                chunk_progress.finish(index)
                completed_chunks = index + 1
                chunk_total_seconds = timing.finish_chunk(index) or 0.0
                chunk_preproduction_seconds = gemma_preproduction_seconds if index == 0 else 0.0
                chunk_gemma_seconds = gemma_chunk_seconds + chunk_preproduction_seconds
                # The one-time shot planner exists to prepare Chunk 1, so
                # attribute both its Gemma time and its wall time to that
                # chunk. This keeps the tooltip's sampler + Gemma + misc
                # breakdown arithmetically truthful.
                chunk_total_seconds += chunk_preproduction_seconds
                preview_chunk_ranges[index].update({
                    "h3_render_seconds": h3_render_seconds,
                    "gemma_seconds": chunk_gemma_seconds,
                    "gemma_preproduction_seconds": chunk_preproduction_seconds,
                    "chunk_total_seconds": chunk_total_seconds,
                })
                if preview_execution is not None:
                    preview_execution.set_chunk_timing(
                        index,
                        h3_render_seconds=h3_render_seconds,
                        gemma_seconds=chunk_gemma_seconds,
                        gemma_preproduction_seconds=chunk_preproduction_seconds,
                        chunk_total_seconds=chunk_total_seconds,
                    )
                if intermediate_writer is not None and final_preview_frames is not None:
                    intermediate_writer.submit(
                        index,
                        final_preview_frames,
                        preview_output_start,
                        preview_range["end"],
                        chunk_metadata=preview_chunk_ranges[index],
                        shot_ranges=preview_shot_ranges,
                        audio_waveform=final_preview_audio,
                        audio_sample_rate=final_preview_audio_rate,
                    )
                if gemma_director is not None:
                    previous_gemma_description = gemma_description
                    previous_gemma_timing_plan = result.timing_plan
                    previous_gemma_end_state = result.end_state
                    previous_gemma_last_seen_character_state = list(result.last_seen_character_state)
                # The native audio seam uses the actual H3 prompt that just
                # rendered, so a carried dialogue word is timed from its
                # emitted text rather than the source prompt.
                previous_h3_prompt = chunk_prompt
                if replay_cache is not None:
                    try:
                        replay_cache.save_chunk(
                            index + 1,
                            {
                                "sampled_video": previous_video,
                                "sampled_audio": previous_audio,
                                "previous_frame_count": previous_frame_count,
                                "output_video": assembled_video,
                                "output_audio": assembled_audio,
                                "masked_audio_prefix": masked_audio_prefix,
                                "denoised_video": assembled_denoised_video,
                                "denoised_audio": assembled_denoised_audio,
                                "masked_denoised_audio_prefix": masked_denoised_audio_prefix,
                                "output_template": output_template,
                                "denoised_template": denoised_template,
                                "gemma_description": previous_gemma_description,
                                "gemma_timing_plan": previous_gemma_timing_plan,
                                "gemma_end_state": previous_gemma_end_state,
                                "gemma_retention_analysis": gemma_retention_analysis,
                                "h3_prompt": chunk_prompt,
                                "gemma_last_seen_character_state": previous_gemma_last_seen_character_state,
                                "h3_render_seconds": h3_render_seconds,
                                "gemma_seconds": chunk_gemma_seconds,
                                "gemma_preproduction_seconds": chunk_preproduction_seconds,
                                "chunk_total_seconds": chunk_total_seconds,
                                "debug_prompt": debug_prompt,
                                "prefix_video_noise": prefix_video_noise,
                                "prefix_audio_noise": prefix_audio_noise,
                                # Persist the already-finalized CPU media with
                                # its latent checkpoint. A replay can then
                                # restore the exact corrected IMAGE/preview
                                # frames and audio without loading either VAE.
                                "decoded_video_frames": previous_decoded_frames,
                                "corrected_video_frames": _convert_image_transfer(previous_color_corrected_decoded_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else previous_color_corrected_decoded_frames,
                                "decoded_preview_start": preview_output_start,
                                "decoded_preview_end": int(preview_range["end"]),
                                "decoded_preview_offset": preview_trim_frames,
                                "decoded_preview_audio": final_preview_audio,
                                "decoded_preview_audio_rate": final_preview_audio_rate,
                                "decoded_preview_overlap_audio": final_preview_overlap_audio,
                            },
                            fps=fps,
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        logging.warning(
                            "HR Endless Sampler could not save replay state for Chunk %d; "
                            "this run will continue but cannot be resumed from that cache: %s",
                            index + 1,
                            error,
                        )
                        replay_cache = None
                final_preview_frames = None
                final_preview_audio = None
                final_preview_overlap_audio = None

                # The next chunk needs only previous_video/previous_audio and
                # the accumulated trimmed outputs. Release this chunk's input,
                # noise, denoised result, and conditioning owners now instead
                # of carrying them through the next Gemma/VAE handoff.
                guider.original_conds = original_conds
                encoded_prompt = None
                chunk_latent = None
                chunk_noise = None
                chunk_video = None
                chunk_audio = None
                chunk_video_noise = None
                chunk_audio_noise = None
                video_context = None
                audio_context = None
                video_refs.clear()
                reference_latent = None
                reference_audio = None
                prefix_video = None
                prefix_audio = None
                prefix_latent = None
                prefix_noise = None
                prefix_video_noise = None
                prefix_audio_noise = None
                sampled = None
                denoised = None
                denoised_chunk_video = None
                denoised_chunk_audio = None
            if replay_cached_suffix:
                logging.info(
                    "HR Endless Sampler restored cached Chunks %d-%d without H3 sampling.",
                    replay_sample_end + 1,
                    len(active_plan),
                )
                for suffix_offset, state in enumerate(replay_cached_suffix):
                    index = replay_sample_end + suffix_offset
                    if preview_execution is not None:
                        preview_execution.set_phase(
                            f"Restoring cached Chunk {index + 1}/{len(active_plan)} (H3 sampling skipped)",
                            chunk=index,
                        )
                    restored_plan = active_plan[index]
                    restored_range = preview_chunk_ranges[index]
                    correction_overlap = max(0, int(restored_plan.get("output_trim_frames", 0)))
                    keep_masked_av_prefix = (
                        keep_continuation_prefix and index > 0
                    )
                    (
                        raw_frames,
                        _unused_retained_frames,
                        cached_audio,
                        cached_audio_rate,
                        cached_overlap_audio,
                    ) = _decode_replay_preview_media(
                        vae,
                        audio_vae,
                        state["sampled_video"],
                        state.get("sampled_audio"),
                        output_trim_frames=restored_plan.get("output_trim_frames", 0),
                        context_audio_t=(
                            0 if index == 0 or (keep_continuation_prefix)
                            else restored_plan.get("context_audio_t", 0)
                        ),
                        output_frames=restored_plan["frame_end"] - restored_plan["frame_start"] - correction_overlap,
                        fps=fps,
                        masked_audio_overlap_frames=(
                            restored_plan.get("output_trim_frames", 0)
                            if use_masked_av_overlap and TRIM_MASKED_AV_PREFIX and index > 0
                            else 0
                        ),
                        replace_audio_tail_ticks=bool(use_masked_av_mode and TRIM_MASKED_AV_PREFIX and index > 0),
                        keep_masked_av_prefix=keep_masked_av_prefix,
                    )
                    retained_start = 0 if keep_masked_av_prefix else correction_overlap
                    # Cached chunk restoration follows the live decode path:
                    # correct in float32 display sRGB, then retain compute RGB.
                    raw_display_frames = _convert_image_transfer(raw_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else raw_frames
                    correction_reference = _decoded_frame_before_tail(decoded_linear_output_frames, retained_start)
                    correction_reference = _convert_image_transfer(correction_reference, _inverse_gamma_compute_to_srgb) if linear_color_compute else correction_reference
                    correction_frame_count = _same_shot_correction_frames(
                        gemma_shots,
                        restored_range["start"],
                        int(restored_range["end"]) + 1,
                    )
                    corrected_frames, color_transform, corrected_frame_count = _correct_decoded_chunk_color(
                        correction_reference,
                        raw_display_frames,
                        0,
                        correction_frame_count + (retained_start if correction_frame_count else 0),
                        enabled=correct_chunk_boundaries,
                    )
                    if correct_all_chunks:
                        corrected_frames = _correct_ready_entire_shot_frames(
                            corrected_frames,
                            restored_range["start"],
                            gemma_shots,
                            gemma_preproduction_timing_plan,
                            entire_shot_color_anchors,
                        )
                    corrected_frames = _convert_image_transfer(corrected_frames, _srgb_to_inverse_gamma_compute_rgb) if linear_color_compute else corrected_frames
                    retained_count = int(restored_range["end"]) - int(restored_range["start"]) + 1
                    cached_linear_frames = corrected_frames[retained_start:retained_start + retained_count].clone()
                    cached_frames = _convert_image_transfer(cached_linear_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else cached_linear_frames
                    cached_preview_start = int(restored_range["start"])
                    _log_output_color_correction(
                        "restored Chunk %d" % (index + 1),
                        color_transform,
                        corrected_frame_count,
                    )
                    masked_audio_prefix = state.get("masked_audio_prefix")
                    masked_denoised_audio_prefix = state.get("masked_denoised_audio_prefix")
                    if masked_audio_prefix is not None:
                        _replace_stream_tail(output_audio, masked_audio_prefix)
                    if masked_denoised_audio_prefix is not None:
                        _replace_stream_tail(denoised_audio, masked_denoised_audio_prefix)
                    output_video.append(state["output_video"])
                    output_audio.append(state["output_audio"])
                    denoised_video.append(state["denoised_video"])
                    denoised_audio.append(state["denoised_audio"])
                    output_template = state.get("output_template")
                    denoised_template = state.get("denoised_template")
                    if state.get("debug_prompt"):
                        debug_prompts.append(str(state["debug_prompt"]))

                    description = state.get("gemma_description")
                    if isinstance(description, str) and description.strip():
                        preview_chunk_ranges[index]["gemma_detailed_description"] = description.strip()
                    h3_prompt = state.get("h3_prompt")
                    if isinstance(h3_prompt, str) and h3_prompt.strip():
                        preview_chunk_ranges[index]["h3_prompt"] = h3_prompt.strip()
                    for key in (
                        "h3_render_seconds",
                        "gemma_seconds",
                        "gemma_preproduction_seconds",
                        "chunk_total_seconds",
                    ):
                        value = state.get(key)
                        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                            preview_chunk_ranges[index][key] = float(value)

                    if retained_start and decoded_linear_output_frames:
                        _replace_decoded_frame_tail(decoded_linear_output_frames, corrected_frames[:retained_start])
                        if linear_color_compute:
                            _replace_decoded_frame_tail(decoded_output_frames, _convert_image_transfer(corrected_frames[:retained_start], _inverse_gamma_compute_to_srgb))
                        if preview_execution is not None:
                            previous_range = preview_chunk_ranges[index - 1]
                            preview_execution.replace_video_tail(
                                index - 1, decoded_output_frames[-1], previous_range["start"], previous_range["end"],
                                gemma_detailed_description=previous_range.get("gemma_detailed_description"),
                                gemma_retention_analysis=previous_range.get("gemma_retention_analysis"),
                            )
                    decoded_linear_output_frames.append(cached_linear_frames)
                    if linear_color_compute:
                        decoded_output_frames.append(cached_frames)
                    cached_color_diagnostics = _safe_video_color_diagnostics(cached_frames, index + 1)
                    if cached_color_diagnostics is not None:
                        _record_color_diagnostics(
                            color_diagnostics,
                            index,
                            cached_color_diagnostics,
                            _color_boundary_context(gemma_shots, cached_preview_start),
                        )
                    if cached_audio is None or cached_audio_rate is None:
                        decoded_audio_complete = False
                    else:
                        if decoded_audio_sample_rate is None:
                            decoded_audio_sample_rate = int(cached_audio_rate)
                        elif decoded_audio_sample_rate != int(cached_audio_rate):
                            raise ValueError("cached decoded audio sample rate changed during replay")
                        if cached_overlap_audio is not None:
                            _replace_stream_tail(decoded_output_audio, cached_overlap_audio)
                        decoded_output_audio.append(cached_audio.clone())
                    cached_audio, cached_audio_rate, cached_overlap_audio = _enhance_decoded_audio(cached_audio, cached_audio_rate, cached_overlap_audio, enabled=audio_sr, device=guider.model_patcher.load_device, seed=(replay_noise_seed + index) % (2 ** 32))
                    if preview_execution is not None:
                        preview_execution.finalize_chunk(
                            index,
                            cached_frames,
                            cached_preview_start,
                            cached_preview_start + int(cached_frames.shape[0]) - 1,
                            audio_waveform=cached_audio,
                            audio_sample_rate=cached_audio_rate,
                            gemma_detailed_description=description,
                            gemma_retention_analysis=state.get("gemma_retention_analysis"),
                        )
                        if cached_overlap_audio is not None and cached_audio_rate is not None:
                            preview_execution.replace_audio_tail(index, cached_overlap_audio, cached_audio_rate)
                    if intermediate_writer is not None:
                        intermediate_writer.submit(
                            index,
                            cached_frames,
                            cached_preview_start,
                            cached_preview_start + int(cached_frames.shape[0]) - 1,
                            chunk_metadata=preview_chunk_ranges[index],
                            shot_ranges=preview_shot_ranges,
                            audio_waveform=cached_audio,
                            audio_sample_rate=cached_audio_rate,
                        )
                    chunk_progress.finish(index)
                    completed_chunks = index + 1
            if completed_chunks and vae is not None:
                final_chunk_index = completed_chunks - 1
                if final_chunk_index not in color_diagnostics and previous_video is not None:
                    logging.info(
                        "HR Endless Sampler decoding final Chunk %d for complete color diagnostics.",
                        final_chunk_index + 1,
                    )
                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    comfy.model_management.soft_empty_cache(force=True)
                    timer_started = time.perf_counter()
                    final_frames = None
                    try:
                        final_frames = _decode_video_frames(vae, previous_video)
                        retained_start = active_plan[final_chunk_index].get("output_trim_frames", 0)
                        if retained_start < final_frames.shape[0]:
                            final_color_diagnostics = _safe_video_color_diagnostics(
                                final_frames[retained_start:],
                                final_chunk_index + 1,
                            )
                            if final_color_diagnostics is not None:
                                _record_color_diagnostics(
                                    color_diagnostics,
                                    final_chunk_index,
                                    final_color_diagnostics,
                                    _color_boundary_context(
                                        gemma_shots,
                                        active_plan[final_chunk_index]["frame_start"]
                                        + active_plan[final_chunk_index].get("output_trim_frames", 0),
                                    ),
                                )
                    except Exception as error:
                        logging.warning(
                            "HR Endless Sampler could not decode final Chunk %d for color diagnostics; "
                            "the completed render will still be returned: %s",
                            final_chunk_index + 1,
                            error,
                        )
                    finally:
                        timing.add("vae_color_final", timer_started)
                        final_frames = None
                        comfy.model_management.unload_model_and_clones(vae.patcher)
                        comfy.model_management.soft_empty_cache(force=True)
            if use_taomate and vae is not None and output_video:
                # Publish from one continuous latent decode, as upstream does.
                # output_video contains only new media, with transport halos removed.
                taomate_backend.close()
                # The retained TurboQuant cache has been released from CPU RAM
                # above. Release H3's device residency before loading the VAE.
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache(force=True)
                if preview_execution is not None:
                    preview_execution.set_phase("TaoMate: decoding continuous video timeline")
                logging.info("TaoMate: decoding assembled video latent timeline for final output.")
                timer_started = time.perf_counter()
                decoded_output_frames.clear()
                decoded_linear_output_frames.clear()
                entire_shot_color_anchors.clear()
                try:
                    expected_frames = int(preview_chunk_ranges[completed_chunks - 1]["end"]) + 1
                    full_frames = taomate_backend.decode_video_timeline(lambda latent: _decode_video_frames(vae, latent), output_video, expected_frames)
                    # Apply the selected output-only grading to the new continuous
                    # decode, never reuse pixels or anchors from provisional previews.
                    prior_display = None
                    for final_index, final_range in enumerate(preview_chunk_ranges[:completed_chunks]):
                        start, end = int(final_range["start"]), int(final_range["end"]) + 1
                        display_frames = full_frames[start:end]
                        if linear_color_compute:
                            display_frames = _convert_image_transfer(display_frames, _inverse_gamma_compute_to_srgb)
                        correction_count = _same_shot_correction_frames(gemma_shots, start, end)
                        display_frames, transform, corrected_count = _correct_decoded_chunk_color(prior_display, display_frames, 0, correction_count, enabled=correct_chunk_boundaries)
                        if correct_all_chunks:
                            display_frames = _correct_ready_entire_shot_frames(display_frames, start, gemma_shots, gemma_preproduction_timing_plan, entire_shot_color_anchors)
                        prior_display = display_frames[-1:]
                        decoded_output_frames.append(display_frames)
                        if linear_color_compute:
                            decoded_linear_output_frames.append(_convert_image_transfer(display_frames, _srgb_to_inverse_gamma_compute_rgb))
                        _log_output_color_correction("TaoMate final Chunk %d" % (final_index + 1), transform, corrected_count)
                        if preview_execution is not None:
                            preview_execution.finalize_chunk(final_index, display_frames, start, end - 1, gemma_detailed_description=final_range.get("gemma_detailed_description"), gemma_retention_analysis=final_range.get("gemma_retention_analysis"))
                    decoded_video_complete = True
                    full_frames = None
                finally:
                    timing.add("vae_video_final", timer_started)
                    comfy.model_management.unload_model_and_clones(vae.patcher)
                    comfy.model_management.soft_empty_cache(force=True)
            # Ready chunks were progressively matched as they decoded. Retain
            # this fallback only when no source-shot anchor was available.
            if correct_all_chunks and not entire_shot_color_anchors and completed_chunks == len(plan) and decoded_video_complete and decoded_linear_output_frames:
                final_grade_started = time.perf_counter()
                provisional_frames = torch.cat(decoded_linear_output_frames, dim=0)
                provisional_display_frames = _convert_image_transfer(provisional_frames, _inverse_gamma_compute_to_srgb) if linear_color_compute else provisional_frames
                final_frames, shot_grade_reports = _final_shot_color_correction(
                    provisional_display_frames,
                    gemma_shots,
                    gemma_preproduction_timing_plan,
                    preview_chunk_ranges[:completed_chunks],
                )
                final_frames = _convert_image_transfer(final_frames, _srgb_to_inverse_gamma_compute_rgb) if linear_color_compute else final_frames
                if shot_grade_reports:
                    logging.info(
                        "HR Endless Sampler final output-only stable-lighting grade: %s.",
                        "; ".join(
                            "Shot %d %s" % (shot_number, report)
                            for shot_number, report in shot_grade_reports
                        ),
                    )
                if final_frames is not provisional_frames:
                    offset = 0
                    for index, frames in enumerate(decoded_linear_output_frames):
                        next_offset = offset + int(frames.shape[0])
                        decoded_linear_output_frames[index] = final_frames[offset:next_offset].clone()
                        display_frames = _convert_image_transfer(decoded_linear_output_frames[index], _inverse_gamma_compute_to_srgb) if linear_color_compute else decoded_linear_output_frames[index]
                        if linear_color_compute:
                            decoded_output_frames[index] = display_frames
                        if preview_execution is not None:
                            preview_execution.finalize_chunk(
                                index,
                                display_frames,
                                preview_chunk_ranges[index]["start"],
                                preview_chunk_ranges[index]["end"],
                                gemma_detailed_description=preview_chunk_ranges[index].get("gemma_detailed_description"),
                                gemma_retention_analysis=preview_chunk_ranges[index].get("gemma_retention_analysis"),
                            )
                        offset = next_offset
                timing.add("output_shot_color", final_grade_started)
            if use_taomate and audio_vae is not None and output_audio and not audio_first_decoded:
                # Upstream runner._publish joins the complete clean latent timeline
                # before audio-VAE decoding. Group previews above are provisional.
                # Release retained KV before loading the final decoder.
                taomate_kv_cache = taomate_backend.kv_cache_report()
                taomate_backend.close()
                if preview_execution is not None:
                    preview_execution.set_phase("TaoMate: decoding continuous audio timeline")
                timer_started = time.perf_counter()
                try:
                    full_audio, decoded_audio_sample_rate = taomate_backend.decode_audio_timeline(audio_vae, output_audio)
                    decoded_output_audio = [full_audio]
                    decoded_audio_complete = True
                finally:
                    timing.add("vae_audio_preview", timer_started)
                    comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                    comfy.model_management.soft_empty_cache(force=True)
            if audio_sr and decoded_audio_complete and decoded_output_audio and decoded_audio_sample_rate is not None:
                # One independent pass over the assembled original decodes, never
                # the already-enhanced per-chunk preview audio or browser proxies.
                if preview_execution is not None:
                    preview_execution.set_phase("AudioSR: enhancing full original decoded audio")
                audio_sr_started = time.perf_counter()
                full_original_audio = torch.cat(decoded_output_audio, dim=-1)
                full_audio, full_rate, _unused_overlap = _enhance_decoded_audio(full_original_audio, decoded_audio_sample_rate, enabled=True, device=guider.model_patcher.load_device, seed=replay_noise_seed % (2 ** 32))
                enhanced_audio_output = {"waveform": full_audio, "sample_rate": full_rate}
                timing.add("audiosr_final", audio_sr_started)
            if use_taomate and preview_execution is not None and decoded_audio_complete and decoded_output_audio:
                # Replace provisional browser audio with slices of the single decode.
                published_audio = enhanced_audio_output or {"waveform": decoded_output_audio[0], "sample_rate": decoded_audio_sample_rate}
                preview_execution.replace_audio_timeline(published_audio["waveform"], published_audio["sample_rate"], preview_chunk_ranges[:completed_chunks], fps)
            sampling_completed = True
        finally:
            if taomate_backend is not None:
                if taomate_kv_cache is None:
                    taomate_kv_cache = taomate_backend.kv_cache_report()
                taomate_backend.close()
            guider.original_conds = original_conds
            if vram_monitor is not None:
                guider.model_patcher.remove_wrappers_with_key(
                    comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                    VRAM_DEBUG_WRAPPER_KEY,
                )
            if preview_execution is not None:
                preview_execution.close()
            if intermediate_writer is not None:
                intermediate_writer.close()
            chunk_progress.close()
            status = "complete" if sampling_completed and debug_stop_chunk == 0 else "debug stop"
            if not sampling_completed:
                status = "incomplete"
            timing.report(
                status,
                completed_chunks,
                {
                    "chunk_frames": max_chunk_frames,
                    "context_keyframes": context_keyframes,
                    "guide_overlap": guide_overlap,
                    "video_continuation": video_continuation,
                    "video_continuation_method": video_continuation_method,
                    "video_continuation_res": video_continuation_res,
                    "pytorch_memory_fraction": float(pytorch_memory_fraction),
                    "width": width,
                    "height": height,
                    "sampling_steps": max(0, len(sigmas) - 1),
                    "rendered_frames": active_plan[completed_chunks - 1]["frame_end"] if completed_chunks else 0,
                    "full_frames": plan[-1]["frame_end"],
                    "full_chunks": len(plan),
                    "color_diagnostics": color_diagnostics,
                    "taomate_kv_cache": taomate_kv_cache,
                },
            )
            if not sampling_completed and replay_cache is not None:
                resume_chunk = min(completed_chunks + 1, len(active_plan))
                try:
                    replay_cache.mark_interrupted(completed_chunks)
                except (OSError, RuntimeError, ValueError) as error:
                    logging.warning(
                        "HR Endless Sampler could not mark its recovery checkpoint as interrupted: %s",
                        error,
                    )
                logging.error(
                    "HR Endless Sampler preserved the interrupted render through Chunk %d in %s. "
                    "Queue the same workflow again with debug_start_chunk=0 to continue automatically from Chunk %d "
                    "without rerendering completed chunks. Set a nonzero debug_start_chunk only to force a specific "
                    "debug replay point.",
                    completed_chunks,
                    replay_cache.root,
                    resume_chunk,
                )
            elif sampling_completed and debug_stop_chunk and replay_cache is not None:
                try:
                    replay_cache.mark_debug_stop(completed_chunks)
                except (OSError, RuntimeError, ValueError) as error:
                    logging.warning(
                        "HR Endless Sampler could not mark its recovery checkpoint as an intentional debug stop: %s",
                        error,
                    )

        final_output_video = torch.cat(output_video, dim=2)
        final_output_audio = torch.cat(output_audio, dim=-1)
        final_denoised_video = torch.cat(denoised_video, dim=2)
        final_denoised_audio = torch.cat(denoised_audio, dim=-1)
        # Cached earlier chunks intentionally stay in system RAM while a
        # replayed suffix samples. Return the normal device-resident latent
        # shape expected by downstream ComfyUI nodes only after assembly.
        if final_output_video.device != video.device:
            final_output_video = final_output_video.to(device=video.device)
            final_output_audio = final_output_audio.to(device=audio.device)
            final_denoised_video = final_denoised_video.to(device=video.device)
            final_denoised_audio = final_denoised_audio.to(device=audio.device)
        output_template["samples"] = comfy.nested_tensor.NestedTensor((final_output_video, final_output_audio))
        denoised_template["samples"] = comfy.nested_tensor.NestedTensor((final_denoised_video, final_denoised_audio))
        rendered_frames = preview_chunk_ranges[completed_chunks - 1]["end"] + 1 if completed_chunks else 0
        timeline = normalize_timeline(
            {
                "fps": fps,
                "total_frames": rendered_frames,
                "chunks": preview_chunk_ranges[:completed_chunks],
                "render_total_seconds": timing.elapsed(),
                "annotation": preview_execution.annotation() if preview_execution is not None else "",
                "shots": [
                    shot for shot in preview_shot_ranges
                    if int(shot.get("start", rendered_frames)) < rendered_frames
                ],
            },
            fps=fps,
            total_frames=rendered_frames,
        )
        decoded_images = None
        if decoded_video_complete and decoded_output_frames:
            decoded_images = torch.cat(decoded_output_frames, dim=0)
            if int(decoded_images.shape[0]) != rendered_frames:
                logging.warning(
                    "HR Endless Sampler assembled %d decoded frames for a %d-frame render; "
                    "the IMAGE output will be unavailable rather than returning a misaligned batch.",
                    int(decoded_images.shape[0]),
                    rendered_frames,
                )
                decoded_images = None
            else:
                logging.info(
                    "HR Endless Sampler IMAGE output: %d color-corrected full-VAE frames, "
                    "%dx%d float32 on CPU (%.1f MiB).",
                    int(decoded_images.shape[0]),
                    int(decoded_images.shape[2]),
                    int(decoded_images.shape[1]),
                    decoded_images.numel() * decoded_images.element_size() / (1024 ** 2),
                )
        decoded_audio_output = None
        if decoded_audio_complete and decoded_output_audio and decoded_audio_sample_rate is not None:
            decoded_waveform = torch.cat(decoded_output_audio, dim=-1)
            decoded_audio_output = {
                "waveform": decoded_waveform,
                "sample_rate": int(decoded_audio_sample_rate),
            }
            if enhanced_audio_output is not None:
                decoded_audio_output = enhanced_audio_output
            logging.info(
                "HR Endless Sampler AUDIO output: %.3f seconds at %d Hz (%s).",
                decoded_audio_output["waveform"].shape[-1] / float(decoded_audio_output["sample_rate"]),
                decoded_audio_output["sample_rate"],
                "AudioSR full-original pass" if enhanced_audio_output is not None else "original audio-VAE decode",
            )
        if replay_cache is not None and sampling_completed and debug_stop_chunk == 0:
            try:
                replay_cache.mark_complete(completed_chunks)
            except (OSError, RuntimeError, ValueError) as error:
                logging.warning(
                    "HR Endless Sampler could not mark the recovery checkpoint complete: %s",
                    error,
                )
        return io.NodeOutput(
            output_template,
            denoised_template,
            "\n\n".join(debug_prompts),
            timeline,
            decoded_images,
            decoded_audio_output,
        )

    sample = execute
