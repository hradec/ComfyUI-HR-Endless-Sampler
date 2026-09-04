import asyncio
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
    Gemma4ContinuityDirector,
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
from .video_io import HREndlessTimeline, IntermediateChunkVideoWriter, normalize_timeline


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
VIDEO_CONTINUATION_METHODS = (
    VIDEO_CONTINUATION_METHOD_VIDEO1,
    VIDEO_CONTINUATION_METHOD_MASKED_AV,
)
# Temporary A/B switch: leave the normal five-frame Video1 boundary keyframe
# and all other continuation conditioning active, but do not copy/protect the
# selected 39/90/... previous AV run inside the new chunk latent.
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
TRIM_MASKED_AV_PREFIX = False
VIDEO_CONTINUATION_CANVASES = {
    label: tuple(int(value) for value in re.search(r"\((\d+)x(\d+)", label).groups())
    for label in VIDEO_CONTINUATION_RESOLUTIONS[1:]
}
DEFAULT_PYTORCH_MEMORY_FRACTION = 0.85
VRAM_DEBUG_WRAPPER_KEY = "hr_endless_sampler_vram_debug"
# Temporarily disable the disposable three-step continuation memory probe.  It
# remains implemented below so the experiment can be restored by changing this
# single flag after its startup cost is useful again.
ENABLE_DEBUG_MEMORY_PREFLIGHT = False
# Set this to False only for the isolation experiment that retains the
# five-frame visual boundary keyframe while suppressing Video1/Audio1 in Qwen,
# DiT references, and prompt text.
INCLUDE_VIDEO1_REFERENCE = True
# Experimental H3-conditioning A/B switch.  The normal source uses the raw
# prior VAE decode, while this path feeds the finalized display-color-corrected
# tail back through H3's pixel-space continuation inputs.  It deliberately
# does not alter the five-frame latent boundary, which has no pixel-space
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
REPLAY_CACHE_FORMAT = 9
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
        return web.json_response(_replay_cache_ui_status(), headers={"Cache-Control": "no-store"})

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
        return web.json_response(snapshot or {}, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_preview/replay_cache_enabled")
    async def hr_endless_sampler_set_replay_cache_enabled(request):
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
            if chunk_number > status["completed_chunks"] or not cache.has_chunk(chunk_number):
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


def _replay_cached_decoded_media(state):
    """Return one checkpoint's CPU final media, or ``None`` for older caches.

    The raw decode remains necessary for the next Qwen/Gemma handoff, while
    the corrected decode is the authoritative source of preview and IMAGE
    output frames.  Keeping both prevents a resumed render from changing its
    pixel history by applying the boundary correction a second time.
    """
    raw_frames = state.get("decoded_video_frames")
    corrected_frames = state.get("corrected_video_frames")
    if not isinstance(raw_frames, torch.Tensor) or raw_frames.ndim != 4:
        return None
    if not isinstance(corrected_frames, torch.Tensor) or corrected_frames.shape != raw_frames.shape:
        return None
    try:
        preview_start = int(state["decoded_preview_start"])
        preview_end = int(state["decoded_preview_end"])
    except (KeyError, TypeError, ValueError):
        return None
    preview_count = max(0, preview_end - preview_start + 1)
    preview_offset = int(state.get("decoded_preview_offset", 0))
    if preview_offset < 0 or preview_offset + preview_count > int(corrected_frames.shape[0]):
        return None
    return {
        "raw_frames": raw_frames.detach().to(device="cpu", dtype=torch.float32),
        "corrected_frames": corrected_frames.detach().to(device="cpu", dtype=torch.float32),
        "preview_frames": corrected_frames[preview_offset:preview_offset + preview_count].detach().to(
            device="cpu", dtype=torch.float32
        ),
        "preview_start": preview_start,
        "preview_end": preview_end,
        "audio": state.get("decoded_preview_audio"),
        "audio_sample_rate": state.get("decoded_preview_audio_rate"),
        "overlap_audio": state.get("decoded_preview_overlap_audio"),
    }


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
        "chunk_prompt_marker_mode": "global-shot-labels-local-time-dialogue-segments-v2",
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
        """Return the next chunk only for an incomplete ordinary render."""
        if manifest.get("status") not in {"recording", "interrupted"}:
            return None
        try:
            completed = int(manifest.get("completed_chunks", -1))
        except (TypeError, ValueError):
            return None
        if completed < 0 or completed >= int(chunk_count):
            return None
        return completed + 1

    def load_if_compatible(self, fingerprint):
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != REPLAY_CACHE_FORMAT:
                return None, "cache format is obsolete"
            if manifest.get("fingerprint") != fingerprint:
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

    def save_chunk(self, chunk_number, state):
        _replay_write_tensor_file(self.chunk_path(chunk_number), state)
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

    def delete_chunk(self, chunk_number):
        """Remove one checkpoint while preserving later cache entries for replay."""
        chunk_number = int(chunk_number)
        if chunk_number < 1 or not self.has_chunk(chunk_number):
            raise ValueError(f"Cached Chunk {chunk_number} is unavailable")
        self.chunk_path(chunk_number).unlink()
        self.mark_interrupted(chunk_number - 1)


def _cached_replay_preview_snapshot(node_id, *, max_resolution, quality, fps):
    """Return a browser snapshot from finalized CPU media in the dormant cache.

    ponytail: the replay cache represents one deliberately global last render,
    so a preview node restores that one cache rather than inventing per-node
    cache ownership. Per-workflow cache namespaces would be the upgrade path.
    """
    if not str(node_id):
        return None
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
        # only checkpoints that carry finalized decoded media.
        chunk_ranges = []
        for index, item in enumerate(plan):
            if not isinstance(item, dict):
                return None
            try:
                start = int(item["frame_start"]) + int(item.get("output_trim_frames", 0))
                end = int(item["frame_end"]) - 1
            except (KeyError, TypeError, ValueError):
                return None
            chunk_ranges.append({"chunk": index + 1, "start": start, "end": end})

        cached_chunks = []
        for number in status["cached_chunks"]:
            if number < 1 or number > len(chunk_ranges):
                continue
            try:
                state = cache.load_chunk(number)
            except (OSError, RuntimeError, ValueError):
                continue
            media = _replay_cached_decoded_media(state)
            if media is None:
                continue
            range_item = chunk_ranges[number - 1]
            range_item["start"] = media["preview_start"]
            range_item["end"] = media["preview_end"]
            description = state.get("gemma_description")
            if isinstance(description, str) and description.strip():
                range_item["gemma_detailed_description"] = description.strip()
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
                "frames": media["preview_frames"],
                "output_start": media["preview_start"],
                "audio": media["audio"],
                "audio_sample_rate": media["audio_sample_rate"],
                "gemma_detailed_description": range_item.get("gemma_detailed_description"),
                "gemma_retention_analysis": range_item.get("gemma_retention_analysis"),
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


def _video_continuation_prompt(prompt, video_label, audio_label=None, storyboard=False):
    source_lines = []
    if video_label is not None:
        source_lines.append(f"{video_label} is the continuation source for this chunk.")
    if audio_label is not None:
        source_lines.append(f"{audio_label} is the audio from previous video that needs to be continued seamlessly.")
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
        audio_retention_line = f"{audio_label}: reference - audio from the previous video that needs to be continued seamlessly."
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
        retention_line = f"{video_label} (appears in {continuation_location}): fully_preserved - its ending is used as the continuation starting point for this chunk."
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


def _planned_chunk_prompts(prompt, plan, active_plan, fps, guide_frames, video_continuation,
                           audio_continuation, ref2va, video_number, audio_number):
    total_frames = plan[-1]["frame_end"]
    guide_enabled = guide_frames > 0
    planned = []
    for index, chunk in enumerate(active_plan):
        continuation = index > 0
        content_start = chunk["frame_start"] + chunk.get("output_trim_frames", 0)
        continuation_video_label = f"<Video {video_number}>" if continuation and video_continuation else None
        continuation_audio_label = f"<Audio {audio_number}>" if continuation and audio_continuation else None
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
        )
        debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt)
        planned.append((chunk_prompt, debug_prompt))
    return planned


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
    return source_shots, {
        "chunk_count": len(full_plan),
        "fps": fps,
        "prompt_mode": "ref" if ref2va else "base",
        "source_shots": source_shots,
        "chunks": _gemma_preproduction_chunks(full_plan),
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
        if include_video1_reference and not context_keyframes:
            sources.append(
                "one fixed five-frame video keyframe clip made from the previous chunk's exact final tail, "
                "anchored across the discarded packing prefix; it has no separate audio keyframe"
            )
    if guide_overlap:
        sources.append(
            f"a {guide_overlap}-frame latent warm-start that is fully denoised and retained, not a fixed keyframe"
        )
    return "; ".join(sources) if sources else "No fixed opening frames or native Video/Audio continuation reference."


def _chunk_retention_analysis(retention_analysis):
    """Normalize Gemma's concise H3-facing entry-state value without adding prose."""
    return retention_analysis.strip() if isinstance(retention_analysis, str) else ""


def _prompt_with_gemma_description(prompt, description, drop_picture_anchors=False,
                                   continuation_video_label=None, continuation_audio_label=None,
                                   retention_analysis=""):
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
    if continuation_video_label is not None:
        rewritten = _video_continuation_prompt(
            rewritten,
            continuation_video_label,
            continuation_audio_label,
        )
    return rewritten


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


def _chunk_plan_without_overlap(video_t, audio_t, chunk_frames):
    plan = _chunk_plan(video_t, audio_t, chunk_frames, 5)
    for index in range(1, len(plan)):
        chunk = plan[index].copy()
        chunk["video_start"] += chunk["context_video_t"]
        chunk["audio_start"] += chunk["context_audio_t"]
        chunk["synthetic_prefix"] = True
        plan[index] = chunk
    return plan


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
    if not chunk.get("synthetic_prefix") or chunk.get("output_trim_frames") != 5:
        raise ValueError("Video1 boundary keyframe needs the five-frame discarded packing prefix")
    guide_t = _video_steps(chunk["output_trim_frames"])
    if previous_video.shape[2] < guide_t:
        raise ValueError("Previous chunk is too short for the five-frame Video1 boundary keyframe")
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
                                 correction_frames=None):
    """Color-match only new same-shot decoded output, never the H3 input path."""
    overlap = min(
        max(0, int(overlap_frames)),
        int(decoded_chunk_frames.shape[0]),
    )
    generated_available = int(decoded_chunk_frames.shape[0]) - overlap
    correction_count = generated_available if correction_frames is None else min(
        generated_available,
        max(0, int(correction_frames)),
    )
    if previous_finalized_frames is None or correction_count <= 0:
        return decoded_chunk_frames, None, 0
    if not int(previous_finalized_frames.shape[0]):
        return decoded_chunk_frames, None, 0
    transform = _fixed_output_color_transform(
        previous_finalized_frames,
        decoded_chunk_frames[overlap:overlap + 1],
    )
    corrected = decoded_chunk_frames.clone()
    corrected[overlap:overlap + correction_count] = _apply_fixed_output_color_transform(
        decoded_chunk_frames[overlap:overlap + correction_count],
        transform,
    )
    corrected_stats = _serializable_color_statistics(
        _display_color_statistics(corrected[overlap:overlap + 1])
    )
    transform.update({
        "corrected": corrected_stats,
        "residual": {
            "rgb_mean": [
                corrected_value - reference_value
                for corrected_value, reference_value in zip(
                    corrected_stats["rgb_mean"], transform["reference"]["rgb_mean"]
                )
            ],
            "luma_mean": corrected_stats["luma_mean"] - transform["reference"]["luma_mean"],
            "luma_median": corrected_stats["luma_median"] - transform["reference"]["luma_median"],
            "black_clip": corrected_stats["black_clip"] - transform["reference"]["black_clip"],
        },
    })
    return corrected, transform, correction_count


def _final_shot_color_correction(images, source_shots, timing_plan, chunk_ranges):
    """Match stable-lighting shot chunks to that shot's first rendered chunk.

    The ordinary per-chunk seam correction remains responsible for exact
    adjacent-boundary continuity. This second pass uses Gemma's immutable
    source-shot lighting decision only after every chunk is complete, so it
    never changes H3/Qwen/Gemma conditioning or the serial sampling path.
    """
    if images is None or not source_shots or timing_plan is None or not chunk_ranges:
        return images, []
    if images.ndim != 4 or not int(images.shape[0]):
        return images, []

    lighting = {int(shot.source_shot): bool(shot.light_change) for shot in timing_plan.shots}
    corrected = images.clone()
    reports = []
    frame_count = int(images.shape[0])
    for shot_index, shot_start, shot_end, _body in source_shots:
        source_shot = int(shot_index) + 1
        start = max(0, int(shot_start))
        end = min(frame_count, int(shot_end))
        if start >= end:
            continue
        if lighting.get(source_shot, True):
            reports.append((source_shot, "skipped: source prompt permits lighting change"))
            continue

        anchor = None
        corrected_ranges = 0
        for chunk in chunk_ranges:
            chunk_start = max(start, int(chunk.get("start", start)))
            chunk_end = min(end, int(chunk.get("end", end - 1)) + 1)
            if chunk_start >= chunk_end:
                continue
            if anchor is None:
                # This complete physical-chunk portion defines the grade for
                # the source shot. _display_color_statistics samples it
                # cheaply, so retaining the tensor needs no extra copy.
                anchor = images[chunk_start:chunk_end]
                continue
            transform = _fixed_output_color_transform(
                anchor,
                images[chunk_start:chunk_end],
                whole_range=True,
            )
            corrected[chunk_start:chunk_end] = _apply_fixed_output_color_transform(
                images[chunk_start:chunk_end],
                transform,
            )
            corrected_ranges += 1
        reports.append((
            source_shot,
            "applied to %d later chunk portion(s)" % corrected_ranges
            if anchor is not None else "skipped: no rendered anchor",
        ))
    return corrected, reports


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


def _masked_av_boundary_guides(vae, previous_decoded_frames, overlap_frames):
    """Anchor the first, middle, and final images of the protected overlap.

    MiniMax accepts multiple independent one-frame guides at arbitrary pixel
    frame positions. Encoding each image separately avoids turning the three
    sparse anchors into one contiguous guide clip. The final anchor remains
    immediately before the first retained/generated frame; the two earlier
    anchors give H3 more evidence for background, lighting, and motion through
    the physical overlap.
    """
    overlap_frames = int(overlap_frames)
    if overlap_frames <= 0:
        raise ValueError("Masked AV boundary keyframes need a positive overlap")
    if previous_decoded_frames is None or int(previous_decoded_frames.shape[0]) < overlap_frames:
        raise ValueError(
            "Masked AV boundary keyframes need the complete decoded previous-chunk overlap"
        )
    positions = tuple(sorted({0, (overlap_frames - 1) // 2, overlap_frames - 1}))
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
                            video_contexts=()):
    conds = {name: [item.copy() for item in values] for name, values in original_conds.items()}
    positive = conds.get("positive")
    if positive is None:
        raise ValueError("HR Endless Sampler requires a standard guider with positive conditioning")

    cross_attn, prompt_metadata = encoded_prompt
    for cond in positive:
        cond["cross_attn"] = cross_attn
        token_tags = prompt_metadata.get("minimax_token_tags")
        if token_tags is not None:
            cond["minimax_token_tags"] = token_tags
        else:
            cond.pop("minimax_token_tags", None)
        if video_refs:
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


def _decode_audio_preview(audio_vae, latent, *, trim_latent_steps=0, output_frames=None, fps=VIDEO_FPS):
    """Decode and trim one H3 chunk using ComfyUI's ordinary audio normalization."""
    waveform = audio_vae.decode(latent).movedim(-1, 1)
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
                                 keep_masked_av_prefix=False):
    """Decode one cached physical chunk exactly like live finalization.

    Replay checkpoints intentionally store the authoritative video/audio
    latents rather than a second compressed browser copy. Resume preview must
    therefore run the real VAEs again; using the fast latent preview here makes
    restored chunks visibly different and drops their audio.
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
        if overlap_frames:
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
    synchronized audio span, Qwen presentation, and five-frame boundary anchor
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
        ("Audio VAE decode: final preview", "vae_audio_preview"),
        ("VAE decode: final color diagnostics", "vae_color_final"),
        ("VAE continuation decode/resize/encode", "vae_context"),
        ("VAE decode: Qwen full history", "vae_history"),
    )

    _REPORT_LABELS = {
        "h3_sampling": "H3 sampling",
        "qwen": "Qwen",
        "gemma4": "Gemma 4",
        "vae_previous_chunk": "Final video preview/Gemma VAE decode",
        "vae_audio_preview": "Final audio preview VAE decode",
        "vae_color_final": "Final color-diagnostic VAE decode",
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
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="The original model prompt. MiniMax H3 currently uses [Shot 1] and [Shot N] At MM:SS.mmm, markers."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001,
                               tooltip="FPS used to convert source-prompt cut timestamps to exact frame positions."),
                io.Int.Input("chunk_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Maximum frames sampled at once. MiniMax H3 values are snapped down to its 17k+5 frame grid."),
                io.Image.Input("images", optional=True,
                               tooltip="Original backend conditioning images as a batch. For MiniMax H3 Ref2VA, keep reference images in their original order."),
                io.Int.Input("video_continuation", default=22, min=5, max=3600, step=17,
                             tooltip="Completed continuation tail length. The selected continuation method interprets this as either a separate synchronized Video1/Audio1 reference or a physical masked AV overlap inside chunk_frames."),
                io.Combo.Input(
                    "video_continuation_method",
                    options=list(VIDEO_CONTINUATION_METHODS),
                    default=VIDEO_CONTINUATION_METHOD_VIDEO1,
                    tooltip=(
                        "Video1 reference keeps the current Ref2VA <Video N>/<Audio N> path plus its five-frame "
                        "boundary prefix. Masked AV overlap instead copies this many completed frames directly "
                        "into the next target, preserves video, feathers the audio seam, and creates no Video1 "
                        "reference or continuation-summary text. In masked mode the overlap is included inside "
                        "chunk_frames, so it must be smaller than chunk_frames and reduces new frames per chunk."
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
                io.Vae.Input("vae", optional=True,
                             tooltip="Video VAE required by the current MiniMax H3 continuation and Gemma visual-directing backend."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="MiniMax H3 audio VAE. When connected, each completed chunk's final decoded audio is synchronized with its full-VAE browser preview."),
                io.Boolean.Input("cache_gemma_preproduction", default=False,
                                 tooltip="Save one clean post-preproduction Gemma KV context in temporary RAM and restore it for each chunk. Avoids re-feeding static source intent and timing plans; needs several GiB of system RAM."),
                io.Boolean.Input("gemma4_mtp", default=False,
                                 tooltip="Experimental native Gemma 4 draft-MTP with four speculative tokens. Disabled by default because the current Python runtime is slower and has hybrid-checkpoint failures."),
                io.Float.Input(
                    "pytorch_memory_fraction",
                    default=DEFAULT_PYTORCH_MEMORY_FRACTION,
                    min=0.50,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Global PyTorch CUDA allocator limit applied when this sampler starts. 0.85 reserves "
                        "15% of VRAM outside PyTorch so cached memory is pressured before a driver-level OOM. "
                        "The setting remains active for the ComfyUI process until another run changes it; "
                        "set 1.0 for the normal unrestricted allocator limit."
                    ),
                ),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log every chunk prompt, raw Gemma response, and detailed VRAM snapshots. chunk_prompts is returned whether debug is enabled or not."),
                io.Int.Input("debug_stop_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Stop after this 1-based chunk number and return the partial result. 0 samples every chunk."),
                io.Int.Input("debug_start_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Resume or rerun from this 1-based chunk using the automatically recorded last-run recovery cache. Leave this at 0 to automatically continue a compatible interrupted render; nonzero forces a specific chunk for debugging."),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
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
                video_continuation=22, video_continuation_method=VIDEO_CONTINUATION_METHOD_VIDEO1,
                video_continuation_res="full", vae=None, audio_vae=None,
                cache_gemma_preproduction=False,
                gemma4_mtp=False,
                pytorch_memory_fraction=DEFAULT_PYTORCH_MEMORY_FRACTION,
                debug=False, debug_stop_chunk=0, debug_start_chunk=0,
                unique_id=None, dynprompt=None,
                **_deprecated_inputs):
        # ComfyUI V3 stores hidden inputs on the per-execution class clone
        # instead of passing them as execute() arguments.  Keep the arguments
        # as a compatibility fallback for older ComfyUI releases.
        hidden = getattr(cls, "hidden", None)
        if unique_id is None:
            unique_id = getattr(hidden, "unique_id", None)
        if dynprompt is None:
            dynprompt = getattr(hidden, "dynprompt", None)
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
        use_video_continuation = video_continuation > 0
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
                "HR Endless Sampler masked AV overlap is temporarily disabled; preserving masked-mode keyframes "
                "and the whole-previous-chunk Audio reference without a protected AV prefix or Video1 reference."
            )
        if use_masked_av_overlap and not TRIM_MASKED_AV_PREFIX:
            logging.warning(
                "HR Endless Sampler masked-AV prefix inspection is active: retaining each locked "
                "video/audio prefix in its own chunk instead of trimming the repeated AV tail."
            )
        if use_masked_av_overlap and video_continuation >= max_chunk_frames:
            raise ValueError(
                f"video_continuation ({video_continuation}) must be smaller than the effective chunk size "
                f"({max_chunk_frames}) for {VIDEO_CONTINUATION_METHOD_MASKED_AV}"
            )
        include_video1_reference = (
            use_video_continuation
            and not use_masked_av_mode
            and INCLUDE_VIDEO1_REFERENCE
        )
        include_previous_audio_reference = use_video_continuation
        # A multi-frame MiniMax keyframe is anchored on the target timeline; it
        # is not detached historical memory. Keep the same completed frames in
        # the opening physical target interval and trim that truthful overlap
        # after sampling. With keyframes disabled, retain the minimum synthetic
        # five-frame prefix needed to preserve H3's temporal packing phase.
        if use_masked_av_overlap:
            plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames, video_continuation)
        elif context_keyframes:
            plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames, context_keyframes)
        else:
            plan = _chunk_plan_without_overlap(video.shape[2], audio.shape[-1], chunk_frames)
        if debug_stop_chunk > len(plan):
            raise ValueError(f"debug_stop_chunk is {debug_stop_chunk}, but this latent has only {len(plan)} chunks")
        if debug_start_chunk > len(plan):
            raise ValueError(f"debug_start_chunk is {debug_start_chunk}, but this latent has only {len(plan)} chunks")
        if debug_start_chunk and debug_stop_chunk and debug_start_chunk > debug_stop_chunk:
            raise ValueError("debug_start_chunk cannot be greater than debug_stop_chunk")
        active_plan = plan if debug_stop_chunk == 0 else plan[:debug_stop_chunk]
        _gemma_markers, gemma_shots, _gemma_description_end = _parse_prompt_shots(prompt, plan[-1]["frame_end"], fps)
        gemma_director_needed = bool(gemma_shots)

        original_conds = guider.original_conds
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError("HR Endless Sampler requires a standard guider with positive conditioning")
        ref2va = bool(positive[0].get("minimax_refs"))
        if len(active_plan) > 1 and (include_video1_reference or include_previous_audio_reference or qwen_full_history) and not ref2va:
            raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")
        original_refs = positive[0].get("minimax_refs", ())
        video_number = 1 + sum(ref["kind"] in ("video", "video_audio") for ref in original_refs)
        audio_number = 1 + sum(ref["kind"] in ("audio", "video_audio") for ref in original_refs)
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
        )
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
                    "five-frame visual boundary keyframe enabled; Qwen/DiT/prompt Video1 reference disabled"
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
                    "chunks": [
                        {
                            "chunk": index + 1,
                            "start": chunk["frame_start"] + chunk.get("output_trim_frames", 0),
                            "end": chunk["frame_end"] - 1,
                        }
                        for index, chunk in enumerate(active_plan)
                    ],
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
        replay_cache_enabled = _replay_cache_enabled()
        if not replay_cache_enabled and debug_start_chunk:
            logging.info(
                "HR Endless Sampler replay cache is disabled; ignoring debug_start_chunk=%d and sampling from Chunk 1.",
                debug_start_chunk,
            )
            debug_start_chunk = 0
        if not replay_cache_enabled:
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
                    # this run's end and every state includes final decoded
                    # media; otherwise ordinary serial sampling remains the
                    # only valid way to reach a later uncached chunk.
                    if not replay_prompt_changed:
                        suffix_numbers = list(range(debug_start_chunk + 1, len(active_plan) + 1))
                        if suffix_numbers and all(candidate_cache.has_chunk(number) for number in suffix_numbers):
                            candidate_suffix = [candidate_cache.load_chunk(number) for number in suffix_numbers]
                            if all(_replay_cached_decoded_media(state) is not None for state in candidate_suffix):
                                replay_cached_suffix = candidate_suffix
                                logging.info(
                                    "HR Endless Sampler replay will sample Chunk %d only and reuse cached Chunks %d-%d.",
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
        if len(active_plan) > 1 and replay_cache is None:
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
        decoded_output_audio = []
        decoded_video_complete = vae is not None
        decoded_audio_complete = audio_vae is not None
        decoded_audio_sample_rate = None
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
        previous_gemma_timing_plan = None
        previous_gemma_end_state = None
        previous_gemma_last_seen_character_state = None
        color_diagnostics = {}
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
                previous_gemma_timing_plan = previous_state.get("gemma_timing_plan")
                previous_gemma_end_state = previous_state.get("gemma_end_state")
                previous_gemma_last_seen_character_state = previous_state.get("gemma_last_seen_character_state")
                if replay_prompt_changed:
                    # These were authored against the old source prompt. The
                    # predecessor still remains available as chronological
                    # rendered stills, which are more reliable evidence for
                    # the first rerun chunk than stale textual instructions.
                    previous_gemma_description = None
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
            Gemma4ContinuityDirector(
                debug=debug,
                gemma4_mtp=bool(gemma4_mtp),
                seed=replay_noise_seed,
                observation_image_directory=gemma_image_log,
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
        preview_chunk_ranges = [
            {
                "chunk": index + 1,
                "start": chunk["frame_start"] + chunk.get("output_trim_frames", 0),
                "end": chunk["frame_end"] - 1,
            }
            for index, chunk in enumerate(active_plan)
        ]
        for index, state in enumerate(replay_prior_chunks):
            description = state.get("gemma_description")
            if isinstance(description, str) and description.strip() and index < len(preview_chunk_ranges):
                preview_chunk_ranges[index]["gemma_detailed_description"] = description.strip()
                if INCLUDE_PER_CHUNK_RETENTION_ANALYSIS:
                    retention_analysis = _chunk_retention_analysis(state.get("gemma_retention_analysis"))
                    if retention_analysis:
                        preview_chunk_ranges[index]["gemma_retention_analysis"] = retention_analysis
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
        preview_execution = begin_preview_execution(
            guider.model_patcher,
            preview_chunk_ranges,
            preview_shot_ranges,
            reusing_cached_chunks=bool(replay_prior_chunks),
            cached_chunk_count=len(replay_prior_chunks),
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
                    "trim_steps": restored_plan.get("context_video_t", 0),
                    "gemma_detailed_description": restored_range.get("gemma_detailed_description"),
                    "gemma_retention_analysis": restored_range.get("gemma_retention_analysis"),
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
                # New checkpoints contain raw and corrected CPU decoded media,
                # so normal restore never reloads either VAE. Older/incomplete
                # checkpoints retain the original latent-VAE fallback; a bad
                # preview restore must never prevent serial sampling recovery.
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
                            use_masked_av_overlap
                            and restored_index > 0
                            and not TRIM_MASKED_AV_PREFIX
                        )
                        cached_media = _replay_cached_decoded_media(state)
                        if cached_media is not None:
                            decoded_frames = cached_media["raw_frames"]
                            corrected_frames = cached_media["corrected_frames"]
                            retained_frames = cached_media["preview_frames"]
                            restored_audio = cached_media["audio"]
                            restored_audio_rate = cached_media["audio_sample_rate"]
                            restored_overlap_audio = cached_media["overlap_audio"]
                            restored_preview_start = cached_media["preview_start"]
                            restored_preview_end = cached_media["preview_end"]
                            logging.info(
                                "HR Endless Sampler restored cached decoded media for Chunk %d without VAE decoding.",
                                restored_index + 1,
                            )
                        else:
                            (
                                decoded_frames,
                                _unused_retained_frames,
                                restored_audio,
                                restored_audio_rate,
                                restored_overlap_audio,
                            ) = (
                                _decode_replay_preview_media(
                                    vae,
                                    audio_vae,
                                    state["sampled_video"],
                                    state.get("sampled_audio"),
                                    output_trim_frames=restored_plan.get("output_trim_frames", 0),
                                    context_audio_t=(
                                        0 if restored_index == 0
                                        or (use_masked_av_overlap and not TRIM_MASKED_AV_PREFIX)
                                        else restored_plan.get("context_audio_t", 0)
                                    ),
                                    output_frames=restored["output_end"] - restored["output_start"] + 1,
                                    fps=fps,
                                    masked_audio_overlap_frames=(
                                        restored_plan.get("output_trim_frames", 0)
                                        if (
                                            use_masked_av_overlap
                                            and TRIM_MASKED_AV_PREFIX
                                            and restored_index > 0
                                        )
                                        else 0
                                    ),
                                    keep_masked_av_prefix=keep_masked_av_prefix,
                                )
                            )
                            retained_start = 0 if keep_masked_av_prefix else correction_overlap
                            correction_reference = _decoded_frame_tail(decoded_output_frames, correction_overlap)
                            correction_frame_count = _same_shot_correction_frames(
                                gemma_shots,
                                restored["output_start"],
                                restored["output_end"] + 1,
                            )
                            (
                                corrected_frames,
                                color_transform,
                                corrected_frame_count,
                            ) = _correct_decoded_chunk_color(
                                correction_reference,
                                decoded_frames,
                                correction_overlap,
                                correction_frame_count,
                            )
                            retained_count = restored["output_end"] - restored["output_start"] + 1
                            if keep_masked_av_prefix:
                                retained_count += correction_overlap
                            retained_frames = corrected_frames[retained_start:retained_start + retained_count].clone()
                            _log_output_color_correction(
                                "restored Chunk %d" % (restored_index + 1),
                                color_transform,
                                corrected_frame_count,
                            )
                            restored_preview_start = (
                                restored_plan["frame_start"]
                                if keep_masked_av_prefix
                                else restored["output_start"]
                            )
                            restored_preview_end = (
                                restored_plan["frame_end"] - 1
                                if keep_masked_av_prefix
                                else restored["output_end"]
                            )
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
                        images=images,
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
                            "chunk_number": index + 1,
                            "chunk_count": len(active_plan),
                            "fps": fps,
                            "prompt_mode": "ref" if ref2va else "base",
                            "current_chunk": {
                                "sampled_start": chunk["frame_start"],
                                "sampled_end": chunk["frame_end"],
                                "output_start": content_start,
                                "output_end": chunk["frame_end"],
                            },
                            "previous_chunk": previous_chunk,
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
                                result = gemma_director.direct(
                                    request,
                                    observation_frames,
                                    progress_callback=preparation_progress.update_token_progress,
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
                if chunk.get("synthetic_prefix"):
                    prefix_video = video.new_zeros((*video.shape[:2], context_video_t, *video.shape[3:]))
                    prefix_audio = audio.new_zeros((*audio.shape[:-1], context_audio_t))
                    cached_prefix = replay_prefix_noises.get(index)
                    if cached_prefix is not None:
                        prefix_video_noise = cached_prefix[0].to(device=video.device, dtype=video.dtype)
                        prefix_audio_noise = cached_prefix[1].to(device=audio.device, dtype=audio.dtype)
                        if prefix_video_noise.shape != prefix_video.shape or prefix_audio_noise.shape != prefix_audio.shape:
                            raise RuntimeError(
                                f"HR Endless Sampler replay cache prefix for Chunk {index + 1} has the wrong shape"
                            )
                    else:
                        prefix_latent = fixed_latent.copy()
                        prefix_latent["samples"] = comfy.nested_tensor.NestedTensor((prefix_video, prefix_audio))
                        prefix_noise = noise.generate_noise(prefix_latent)
                        if not prefix_noise.is_nested or len(prefix_noise.unbind()) != 2:
                            raise ValueError("HR Endless Sampler expected nested video and audio prefix noise")
                        prefix_video_noise, prefix_audio_noise = prefix_noise.unbind()
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

                guide_enabled = context_keyframes > 0
                guide_audio_t = (
                    0 if not guide_enabled
                    else _audio_steps(content_start) - _audio_steps(content_start - context_keyframes)
                )
                video_context = None if previous_video is None or not guide_enabled else previous_video[:, :, -guide_video_t:].clone()
                audio_context = None if previous_audio is None or not guide_enabled else previous_audio[..., -guide_audio_t:].clone()
                video_contexts = ()
                video_context_start = 0
                audio_end_frame = float(keyframe_duration_frames)
                if audio_context is not None:
                    overhang = previous_audio.shape[-1] - FRAME_RESCALE * previous_frame_count
                    audio_end_frame += overhang / FRAME_RESCALE
                video_items = []
                video_refs = []
                boundary_video_context = None
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
                # The discarded five-frame boundary keyframe is part of the
                # Video1 path only. Masked-AV mode must not silently acquire
                # it merely because its physical overlap is temporarily off.
                if continuation and include_video1_reference:
                    boundary_video_context, boundary_keyframe_index = _video_continuation_boundary_guide(
                        previous_video,
                        chunk,
                        context_keyframes,
                        use_video_continuation,
                    )
                    if boundary_video_context is not None:
                        video_context = boundary_video_context
                        video_context_start = boundary_keyframe_index
                        if debug:
                            logging.info(
                                "HR Endless Sampler chunk %d/%d Video1 boundary keyframe: "
                                "previous final five-frame latent tail anchored across discarded local frames 0-4",
                                index + 1,
                                len(active_plan),
                            )
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
                            h3_context_frames, h3_context_kind = _h3_context_frames(
                                previous_decoded_frames,
                                previous_color_corrected_decoded_frames,
                            )
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
                    chunk_prompt = _prompt_with_gemma_description(
                        prompt,
                        gemma_description,
                        drop_picture_anchors=continuation and not ref2va,
                        continuation_video_label=continuation_video_label,
                        continuation_audio_label=continuation_audio_label,
                        retention_analysis=gemma_retention_analysis,
                    )
                    debug_prompt = _debug_chunk_prompt(index, chunk, content_start, chunk_prompt, gemma_report)
                else:
                    chunk_prompt, debug_prompt = planned_prompts[index]
                    if gemma_report is not None:
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
                    encoded_prompt = _encode_prompt(clip, chunk_prompt, images, positive, width, height, continuation, video_items)
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
                    chunk["frame_start"],
                    chunk["frame_end"],
                    encoded_prompt,
                    video_context,
                    audio_context,
                    audio_end_frame,
                    video_refs,
                    video_context_start,
                    video_contexts,
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
                    preview_execution.set_chunk(
                        index,
                        chunk["frame_start"],
                        chunk["frame_end"] - 1,
                        content_start,
                        chunk["frame_end"] - 1,
                        context_video_t,
                        gemma_description,
                        gemma_retention_analysis,
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
                        sampled, denoised = super().execute(
                            _FixedNoise(chunk_seed, chunk_noise), guider, sampler, sigmas, chunk_latent
                        )
                    finally:
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
                    and use_masked_av_overlap
                    and not TRIM_MASKED_AV_PREFIX
                ) else context_video_t
                audio_trim = 0 if (
                    index == 0
                    or (use_masked_av_overlap and not TRIM_MASKED_AV_PREFIX)
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
                final_preview_overlap_audio = None
                preview_output_start = int(content_start)
                preview_trim_frames = 0
                if vae is not None:
                    finalization_message = f"{chunk_label}: decoding final video/audio preview"
                    logging.info("HR Endless Sampler: %s.", finalization_message)
                    if preview_execution is not None:
                        preview_execution.set_phase(finalization_message, chunk=index)
                    comfy.model_management.unload_model_and_clones(guider.model_patcher)
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    comfy.model_management.soft_empty_cache(force=True)
                    timer_started = time.perf_counter()
                    try:
                        previous_decoded_frames = _decode_video_frames(vae, previous_video).detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                        retained_start = max(0, int(chunk.get("output_trim_frames", 0)))
                        preview_output_start = (
                            int(chunk["frame_start"])
                            if continuation and use_masked_av_overlap and not TRIM_MASKED_AV_PREFIX
                            else int(content_start)
                        )
                        preview_trim_frames = 0 if preview_output_start == int(chunk["frame_start"]) else retained_start
                        retained_count = max(0, int(chunk["frame_end"]) - preview_output_start)
                        correction_reference = _decoded_frame_tail(decoded_output_frames, retained_start)
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
                            previous_decoded_frames,
                            retained_start,
                            correction_frame_count,
                        )
                        # Keep the normal raw decode and a separate finalized
                        # corrected copy. The module-level experiment switch
                        # selects the latter only for H3's pixel-space
                        # continuation input on the next chunk.
                        previous_color_corrected_decoded_frames = corrected_decoded_frames
                        final_preview_frames = corrected_decoded_frames[
                            preview_trim_frames:preview_trim_frames + retained_count
                        ].clone()
                        if retained_count and int(final_preview_frames.shape[0]) != retained_count:
                            raise ValueError(
                                f"Chunk {index + 1} decoded {int(previous_decoded_frames.shape[0])} frames, "
                                f"but {retained_count} retained frames were required after trimming {retained_start}"
                            )
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
                        if continuation and use_masked_av_overlap and TRIM_MASKED_AV_PREFIX:
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
                                round(overlap_frames * final_preview_audio_rate / float(fps)),
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
                            )
                    except Exception as error:
                        decoded_audio_complete = False
                        logging.warning(
                            "HR Endless Sampler could not decode final audio preview Chunk %d; "
                            "the final video preview will remain playable without sound: %s",
                            index + 1,
                            error,
                        )
                    finally:
                        timing.add("vae_audio_preview", timer_started)
                        comfy.model_management.unload_model_and_clones(audio_vae.patcher)
                        comfy.model_management.soft_empty_cache(force=True)

                if final_preview_audio is None or final_preview_audio_rate is None:
                    decoded_audio_complete = False
                else:
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
                    if decoded_audio_complete:
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
                        chunk["frame_end"] - 1,
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
                        chunk["frame_end"] - 1,
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
                                "corrected_video_frames": previous_color_corrected_decoded_frames,
                                "decoded_preview_start": preview_output_start,
                                "decoded_preview_end": int(chunk["frame_end"]) - 1,
                                "decoded_preview_offset": preview_trim_frames,
                                "decoded_preview_audio": final_preview_audio,
                                "decoded_preview_audio_rate": final_preview_audio_rate,
                                "decoded_preview_overlap_audio": final_preview_overlap_audio,
                            },
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
                    media = _replay_cached_decoded_media(state)
                    if media is None:
                        raise RuntimeError(
                            f"HR Endless Sampler cached Chunk {index + 1} lost its decoded media during replay"
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
                    for key in (
                        "h3_render_seconds",
                        "gemma_seconds",
                        "gemma_preproduction_seconds",
                        "chunk_total_seconds",
                    ):
                        value = state.get(key)
                        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                            preview_chunk_ranges[index][key] = float(value)

                    decoded_output_frames.append(media["preview_frames"])
                    cached_color_diagnostics = _safe_video_color_diagnostics(media["preview_frames"], index + 1)
                    if cached_color_diagnostics is not None:
                        _record_color_diagnostics(
                            color_diagnostics,
                            index,
                            cached_color_diagnostics,
                            _color_boundary_context(gemma_shots, media["preview_start"]),
                        )
                    cached_audio = media["audio"]
                    cached_audio_rate = media["audio_sample_rate"]
                    cached_overlap_audio = media["overlap_audio"]
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
                    if preview_execution is not None:
                        preview_execution.finalize_chunk(
                            index,
                            media["preview_frames"],
                            media["preview_start"],
                            media["preview_end"],
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
                            media["preview_frames"],
                            media["preview_start"],
                            media["preview_end"],
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
            if decoded_video_complete and decoded_output_frames and gemma_preproduction_timing_plan is not None:
                final_grade_started = time.perf_counter()
                provisional_frames = torch.cat(decoded_output_frames, dim=0)
                final_frames, shot_grade_reports = _final_shot_color_correction(
                    provisional_frames,
                    gemma_shots,
                    gemma_preproduction_timing_plan,
                    preview_chunk_ranges[:completed_chunks],
                )
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
                    for index, frames in enumerate(decoded_output_frames):
                        next_offset = offset + int(frames.shape[0])
                        decoded_output_frames[index] = final_frames[offset:next_offset].clone()
                        if preview_execution is not None:
                            preview_execution.finalize_chunk(
                                index,
                                decoded_output_frames[index],
                                preview_chunk_ranges[index]["start"],
                                preview_chunk_ranges[index]["end"],
                                gemma_detailed_description=preview_chunk_ranges[index].get("gemma_detailed_description"),
                                gemma_retention_analysis=preview_chunk_ranges[index].get("gemma_retention_analysis"),
                            )
                        offset = next_offset
                timing.add("output_shot_color", final_grade_started)
            sampling_completed = True
        finally:
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
        rendered_frames = active_plan[completed_chunks - 1]["frame_end"] if completed_chunks else 0
        timeline = normalize_timeline(
            {
                "fps": fps,
                "total_frames": rendered_frames,
                "chunks": preview_chunk_ranges[:completed_chunks],
                "render_total_seconds": timing.elapsed(),
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
            logging.info(
                "HR Endless Sampler AUDIO output: %.3f seconds of corrected full audio-VAE output at %d Hz.",
                decoded_waveform.shape[-1] / float(decoded_audio_sample_rate),
                decoded_audio_sample_rate,
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
