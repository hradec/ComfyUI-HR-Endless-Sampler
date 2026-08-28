import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch

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

from .gemma4 import (
    Gemma4ContinuityDirector,
    Gemma4DependencyError,
    Gemma4ObservationError,
    Gemma4PreproductionCache,
    _timing_plan_from_payload,
    _timing_plan_payload,
)
from .preview import begin_preview_execution
from .video_io import HREndlessTimeline, normalize_timeline


AUDIO_LATENT_FPS = 40
VIDEO_FPS = 24
MIN_VIDEO_STEPS = 2
CANVAS_MULTIPLE = 32
QWEN_VIDEO_MAX_PIXELS = 512 * 512
VRAM_DEBUG_WRAPPER_KEY = "hr_endless_sampler_vram_debug"
# Set this to False only for the isolation experiment that retains the
# five-frame visual boundary keyframe while suppressing Video1/Audio1 in Qwen,
# DiT references, and prompt text.
INCLUDE_VIDEO1_REFERENCE = True
GEMMA_PROMPT_LOG_DIRNAME = "comfyui-hr-endless-sampler"
GEMMA_PROMPT_LOG_FILENAME = "last_gemma_chunk_prompts.txt"
GEMMA_IMAGE_LOG_DIRNAME = "last_gemma_images"
REPLAY_CACHE_DIRNAME = "last_run_replay"
REPLAY_CACHE_FORMAT = 1
DETAILED_DESCRIPTION_FIELD = re.compile(r"detailed_description\s*:", re.IGNORECASE)
INTEGRATED_DESCRIPTION_FIELD = re.compile(r"integrated_multimodal_description\s*:", re.IGNORECASE)
SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
SUBJECT_DEFINITIONS_FIELD = re.compile(r"(?im)^\s*subject_definitions\s*:\s*$")
SUMMARY_FIELD = re.compile(r"(?im)^(\s*summary\s*:\s*)(.*)$")
RETENTION_FIELD = re.compile(r"(?im)^\s*retention_analysis\s*:\s*$")
PICTURE_LABEL = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)


def _description_field(prompt, start=0):
    return DETAILED_DESCRIPTION_FIELD.search(prompt, start) or INTEGRATED_DESCRIPTION_FIELD.search(prompt, start)


def _begin_last_gemma_prompt_log(chunk_frames, context_keyframes, guide_overlap,
                                 video_continuation, fps, chunk_count,
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
                        context_keyframes, guide_overlap, video_continuation, ref2va):
    """Describe the immutable tensor/layout inputs required for an exact replay.

    The source prompt intentionally is not part of this signature: editing it
    is the reason to replay a later chunk.  Its hash is still recorded for
    diagnostics in the cache manifest.
    """
    return {
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "video_dtype": str(video.dtype),
        "audio_dtype": str(audio.dtype),
        "fps": float(fps),
        "chunk_frames": int(chunk_frames),
        "context_keyframes": int(context_keyframes),
        "guide_overlap": int(guide_overlap),
        "video_continuation": int(video_continuation),
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
        }
        _replay_write_json(self.manifest_path, manifest)
        _replay_write_tensor_file(self.initial_path, initial_tensors)
        logging.info("HR Endless Sampler is recording replay state in %s", self.root)

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
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"could not update replay source-prompt hash: {error}") from error
            manifest["source_prompt_sha256"] = hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
            _replay_write_json(self.manifest_path, manifest)

    def save_chunk(self, chunk_number, state):
        _replay_write_tensor_file(self.chunk_path(chunk_number), state)

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
    source_line = f"{video_label} is the continuation source for this chunk."
    if audio_label is not None:
        source_line += f"\n{audio_label} is the synchronized soundtrack of {video_label} and the audio continuation source."
    subject = SUBJECT_DEFINITIONS_FIELD.search(prompt)
    if subject is not None:
        next_section = SUMMARY_FIELD.search(prompt, subject.end()) or RETENTION_FIELD.search(prompt, subject.end()) or _description_field(prompt, subject.end())
        insert_at = next_section.start() if next_section is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + "\n" + source_line + "\n\n" + prompt[insert_at:].lstrip()
    else:
        field = _description_field(prompt)
        insert_at = field.start() if field is not None else 0
        prompt = prompt[:insert_at] + f"subject_definitions:\n{source_line}\n\n" + prompt[insert_at:]

    summary = SUMMARY_FIELD.search(prompt)
    continuation_sources = video_label if audio_label is None else f"{video_label} and its synchronized {audio_label}"
    summary_text = f"[video continuation] Continue directly from the end of {continuation_sources}."
    if summary is not None:
        existing = summary.group(2).strip()
        task = re.match(r"\[([^]]+)\]\s*(.*)", existing)
        if task is not None:
            types = [value.strip() for value in task.group(1).split("+")]
            if "video continuation" not in [value.lower() for value in types]:
                types.insert(0, "video continuation")
            existing = f"[{' + '.join(types)}] {task.group(2).strip()}".rstrip()
            replacement = summary.group(1) + existing + f" Continue directly from the end of {continuation_sources}."
        else:
            replacement = summary.group(1) + summary_text + (" " + existing if existing else "")
        prompt = prompt[:summary.start()] + replacement + prompt[summary.end():]
    else:
        retention = RETENTION_FIELD.search(prompt)
        field = _description_field(prompt)
        insert_at = retention.start() if retention is not None else field.start() if field is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + f"\n\nsummary: {summary_text}\n\n" + prompt[insert_at:].lstrip()

    # A continuation chunk can begin in the middle of a source shot.  Do not
    # mention ``[Shot 1]`` in a non-description field: H3 can still interpret
    # that token as a fresh-shot cue even when Gemma correctly begins its
    # detailed description with plain continuation prose.
    continuation_location = "the opening storyboard block" if storyboard else "the opening local continuation sequence"
    retention_line = f"{video_label} (appears in {continuation_location}): fully_preserved - its ending is used as the continuation starting point for this chunk."
    if audio_label is not None:
        retention_line += f"\n{audio_label} (synchronized with {video_label}): fully_preserved - its ending is used as the audio continuation starting point."
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

    Source cuts remain ordinary documented ``[Shot N] At MM:SS.mmm,`` markers
    on the physical chunk timeline.  We deliberately do not give H3 our former
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
        # ``[Shot 1]`` is not generic chunk syntax.  It is H3's actual
        # shot-opening cue, so a physical window that begins inside a source
        # shot must start with ordinary continuation prose.  A later real cut
        # still uses the local ordinal it has on the physical timeline.  This
        # planner is normally preview-only, but keeping it consistent with the
        # Gemma path prevents misleading previews and fallback prompts.
        if index == 0:
            if shot_start == frame_start:
                marker_text = "[Shot 1]"
            elif shot_start > frame_start:
                marker_text = f"[Shot 2] At {_frame_timestamp(shot_start - frame_start, fps)},"
            else:
                marker_text = ""
        else:
            marker_text = f"[Shot {index + 1}] At {_frame_timestamp(shot_start - frame_start, fps)},"

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
    if continuation_video_label is not None:
        rewritten_prompt = _video_continuation_prompt(
            rewritten_prompt,
            continuation_video_label,
            continuation_audio_label,
        )
    return rewritten_prompt


def _planned_chunk_prompts(prompt, plan, active_plan, fps, guide_frames, video_continuation,
                           ref2va, video_number, audio_number):
    total_frames = plan[-1]["frame_end"]
    guide_enabled = guide_frames > 0
    planned = []
    for index, chunk in enumerate(active_plan):
        continuation = index > 0
        content_start = chunk["frame_start"] + chunk.get("output_trim_frames", 0)
        continuation_video_label = f"<Video {video_number}>" if continuation and video_continuation else None
        continuation_audio_label = f"<Audio {audio_number}>" if continuation and video_continuation else None
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
    for local_index, (shot_index, shot_start, shot_end, body) in enumerate(selected):
        record = {
            "shot_number": shot_index + 1,
            "shot_start": shot_start,
            "shot_end": shot_end,
            "source_body": body,
        }
        if target:
            target_start = max(range_start, shot_start)
            if local_index == 0:
                # [Shot 1] is a genuine shot-opening signal to H3.  Never
                # synthesize it merely because a new physical sampler chunk
                # starts in the middle of a source shot.  When a real cut
                # occurs after carried physical-prefix frames, the implied
                # continuing source material is local Shot 1 and the new
                # source shot is explicitly local Shot 2 at that real cut.
                if shot_start == sampled_start:
                    required_marker = "[Shot 1]"
                elif shot_start > sampled_start:
                    required_marker = f"[Shot 2] At {_frame_timestamp(shot_start - sampled_start, fps)},"
                else:
                    required_marker = None
            else:
                required_marker = f"[Shot {local_index + 1}] At {_frame_timestamp(shot_start - sampled_start, fps)},"
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


def _gemma_conditioning_context(continuation, context_keyframes, guide_overlap, video_continuation,
                                video_label, audio_label, include_video1_reference=True):
    if not continuation:
        return "First chunk: original image/reference conditioning only; there is no previous generated chunk."
    sources = []
    if context_keyframes:
        sources.append(
            f"native fixed video/audio opening keyframes covering {context_keyframes} completed frames"
        )
    if video_continuation:
        if include_video1_reference:
            sources.append(
                f"a bounded {video_continuation}-frame continuation reference as {video_label} "
                f"with synchronized {audio_label}"
            )
        if not context_keyframes:
            sources.append(
                "one fixed five-frame video keyframe clip made from the previous chunk's exact final tail, "
                "anchored across the discarded packing prefix; it has no separate audio keyframe"
            )
    if guide_overlap:
        sources.append(
            f"a {guide_overlap}-frame latent warm-start that is fully denoised and retained, not a fixed keyframe"
        )
    return "; ".join(sources) if sources else "No fixed opening frames or native Video/Audio continuation reference."


def _prompt_with_gemma_description(prompt, description, drop_picture_anchors=False,
                                   continuation_video_label=None, continuation_audio_label=None):
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
        f"H3 detailed_description: {result.detailed_description}\n"
        f"Gemma JSON attempts:\n{_gemma_response_transcript(result)}"
    )
    if len(result.attempts) > 1:
        report += (
            f"\nGemma chunk-contract correction: attempt 1 had local-marker or current-slice coverage validation errors; "
            f"H3 uses Gemma's attempt {len(result.attempts)} response."
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


def _video_continuation_boundary_guide(previous_video, chunk, context_keyframes, use_video_continuation):
    if not use_video_continuation or context_keyframes:
        return None, 0
    if not chunk.get("synthetic_prefix") or chunk.get("output_trim_frames") != 5:
        raise ValueError("Video1 boundary keyframe needs the five-frame discarded packing prefix")
    guide_t = _video_steps(chunk["output_trim_frames"])
    if previous_video.shape[2] < guide_t:
        raise ValueError("Previous chunk is too short for the five-frame Video1 boundary keyframe")
    return previous_video[:, :, -guide_t:].clone(), 0


def _conditioning_for_chunk(original_conds, frame_start, frame_end, encoded_prompt, video_context=None,
                            audio_context=None, audio_end_frame=5.0, video_refs=(), video_context_start=0):
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
        if audio_context is not None:
            audio_start = audio_end_frame - audio_context.shape[-1] / FRAME_RESCALE
            keyframes.append({"resolved_frame_index": audio_start, "audio_latent": audio_context})
        if keyframes:
            cond["minimax_keyframes"] = keyframes
        else:
            cond.pop("minimax_keyframes", None)
    return conds


def _decode_video_frames(vae, latent):
    frames = vae.decode(latent)
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if not frames.shape[0]:
        raise ValueError("MiniMax H3 video VAE decoded no frames")
    return frames


def _decoded_video_frames(vae, latent, include_final=False, start_frame=0, return_final_frame=False):
    frames = _decode_video_frames(vae, latent)
    final_frame = frames[-1:].detach().to(device="cpu", copy=True) if return_final_frame else None
    start_frame = max(0, min(int(start_frame), frames.shape[0]))
    sample_indices = list(range(start_frame, frames.shape[0], VIDEO_FPS // 2))
    if include_final and frames.shape[0] > start_frame and sample_indices[-1] != frames.shape[0] - 1:
        sample_indices.append(frames.shape[0] - 1)
    sampled_frames = frames[sample_indices]
    height, width = sampled_frames.shape[1:3]
    if height * width > QWEN_VIDEO_MAX_PIXELS:
        scale = math.sqrt(QWEN_VIDEO_MAX_PIXELS / (height * width))
        target_width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        sampled_frames = _resize(sampled_frames, target_width, target_height, "disabled")
    if return_final_frame:
        return sampled_frames, sample_indices, final_frame
    return sampled_frames, sample_indices


def _decoded_video_item(vae, latent):
    qwen_frames, sample_indices = _decoded_video_frames(vae, latent)
    return {
        "type": "video",
        "data": qwen_frames,
        "timestamps": [index / 2.0 for index in range(len(sample_indices))],
    }


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

    def set_chunk(self, index):
        self.chunk = index
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
        label = f"chunk {self.chunk + 1}/{self.chunk_count} DiT evaluation {self.call}"
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
        ("VAE decode: previous chunk for Gemma", "vae_previous_chunk"),
        ("VAE continuation decode", "vae_context"),
        ("VAE decode: Qwen full history", "vae_history"),
    )

    _REPORT_LABELS = {
        "h3_sampling": "H3 sampling",
        "qwen": "Qwen",
        "gemma4": "Gemma 4",
        "vae_previous_chunk": "Previous-chunk VAE decode",
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
            f"guide_overlap={run['guide_overlap']}, video_continuation={run['video_continuation']}"
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
                io.Int.Input("video_continuation", default=5, min=5, max=3600, step=17,
                             tooltip="Completed continuation tail length. MiniMax H3 currently uses the synchronized Ref2VA <Audio N> + <Video N> continuation path; values at or above chunk_frames are clamped to the effective chunk size."),
                io.Vae.Input("vae", optional=True,
                             tooltip="Video VAE required by the current MiniMax H3 continuation and Gemma visual-directing backend."),
                io.Boolean.Input("cache_gemma_preproduction", default=False,
                                 tooltip="Save one clean post-preproduction Gemma KV context in temporary RAM and restore it for each chunk. Avoids re-feeding static source intent and timing plans; needs several GiB of system RAM."),
                io.Boolean.Input("gemma4_mtp", default=True,
                                 tooltip="Use native Gemma 4 draft-MTP with four speculative tokens. Disable it to compare against the original non-MTP decoder."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log every chunk prompt and detailed VRAM snapshots to the console. chunk_prompts is returned whether debug is enabled or not."),
                io.Int.Input("debug_stop_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Stop after this 1-based chunk number and return the partial result. 0 samples every chunk."),
                io.Int.Input("debug_start_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Replay from this 1-based chunk using the temporary last-run cache. The first nonzero run records chunks from 1; 0 clears that cache on the next run."),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
                io.String.Output(display_name="chunk_prompts", tooltip="Exact planned prompt and frame ranges for every active chunk."),
                HREndlessTimeline.Output(display_name="timeline", tooltip="Finished chunk, shot, and Gemma prompt metadata for HR Endless Sampler Save Video."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, noise=None, sampler=None, sigmas=None, clip=None, **_kwargs):
        lazy_inputs = {"noise": noise, "sampler": sampler, "sigmas": sigmas, "clip": clip}
        return [name for name, value in lazy_inputs.items() if value is None]

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0, chunk_frames=124, images=None,
                video_continuation=5, vae=None, cache_gemma_preproduction=False,
                gemma4_mtp=True,
                debug=False, debug_stop_chunk=0, debug_start_chunk=0,
                **_deprecated_inputs):
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
        debug_start_chunk = int(debug_start_chunk)
        debug_stop_chunk = int(debug_stop_chunk)
        if debug_start_chunk == 0:
            # A normal run deliberately invalidates every previous replay
            # checkpoint. This avoids accidentally mixing a new render with a
            # stale continuation tail on a later debugging pass.
            _remove_replay_cache()
        samples = latent_image["samples"]
        if not samples.is_nested:
            if prompt_preview_only:
                raise ValueError("prompt_preview_only requires a MiniMax H3 nested video/audio latent")
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "", normalize_timeline(None, fps=fps, total_frames=0))

        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24 or streams[1].ndim != 4 or streams[1].shape[1] != 32:
            if prompt_preview_only:
                raise ValueError("prompt_preview_only requires MiniMax H3 24-channel video and 32-channel audio latents")
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "", normalize_timeline(None, fps=fps, total_frames=0))

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
        include_video1_reference = use_video_continuation and INCLUDE_VIDEO1_REFERENCE
        # A multi-frame MiniMax keyframe is anchored on the target timeline; it
        # is not detached historical memory. Keep the same completed frames in
        # the opening physical target interval and trim that truthful overlap
        # after sampling. With keyframes disabled, retain the minimum synthetic
        # five-frame prefix needed to preserve H3's temporal packing phase.
        if context_keyframes:
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
        if len(active_plan) > 1 and (use_video_continuation or qwen_full_history) and not ref2va:
            raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")
        original_refs = positive[0].get("minimax_refs", ())
        video_number = 1 + sum(ref["kind"] in ("video", "video_audio") for ref in original_refs)
        audio_number = 1 + sum(ref["kind"] in ("audio", "video_audio") for ref in original_refs)
        planned_prompts = _planned_chunk_prompts(
            prompt,
            plan,
            active_plan,
            fps,
            context_keyframes,
            include_video1_reference,
            ref2va,
            video_number,
            audio_number,
        )
        if debug:
            logging.info(
                "HR Endless Sampler independent continuation controls: "
                "context_keyframes=%d, guide_overlap=%d, video_continuation=%d",
                context_keyframes,
                guide_overlap,
                video_continuation,
            )
            if use_video_continuation and not include_video1_reference:
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
            return io.NodeOutput(latent_image, latent_image, prompt_preview, preview_timeline)

        if len(active_plan) > 1 and "noise_mask" in latent_image:
            raise ValueError("HR Endless Sampler does not support denoise masks when chunking")
        if len(active_plan) > 1 and (use_video_continuation or qwen_full_history or gemma_director_needed):
            if vae is None:
                raise ValueError("video_continuation, qwen_full_history, and Gemma chunk directing require a MiniMax H3 video VAE")

        replay_cache = None
        replay_start_index = 0
        replay_prior_chunks = []
        replay_prefix_noises = {}
        replay_timing_plan = None
        replay_prompt_changed = False
        replay_fingerprint = _replay_fingerprint(
            video,
            audio,
            plan,
            fps=fps,
            chunk_frames=max_chunk_frames,
            context_keyframes=context_keyframes,
            guide_overlap=guide_overlap,
            video_continuation=video_continuation,
            ref2va=ref2va,
        )
        replay_cached_initial = None
        if debug_start_chunk:
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
                    replay_cache.truncate_from(debug_start_chunk)
                    logging.info(
                        "HR Endless Sampler replay: restoring cached state through Chunk %d and rerunning from Chunk %d.",
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
                except (OSError, RuntimeError, ValueError, KeyError) as error:
                    logging.warning(
                        "HR Endless Sampler replay cache could not restore Chunk %d; recording a fresh baseline from Chunk 1: %s",
                        debug_start_chunk,
                        error,
                    )
                    replay_prior_chunks = []
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
            fps,
            len(active_plan),
            cache_gemma_preproduction=cache_gemma_preproduction,
            gemma4_mtp=gemma4_mtp,
        )
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
        if debug_start_chunk and replay_cache is None:
            replay_cache = _LastRunReplayCache()
            replay_cache.create(
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

        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        output_video = []
        output_audio = []
        denoised_video = []
        denoised_audio = []
        previous_video = None
        previous_audio = None
        previous_frame_count = None
        # Only promote this after a stock sampler call succeeds. The next
        # Gemma request can then pair the exact prior directed description with
        # stills from the same rendered chunk, never with an unsampled plan.
        previous_gemma_description = None
        previous_gemma_timing_plan = None
        previous_gemma_end_state = None
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
                if replay_prompt_changed:
                    # These were authored against the old source prompt. The
                    # predecessor still remains available as chronological
                    # rendered stills, which are more reliable evidence for
                    # the first rerun chunk than stale textual instructions.
                    previous_gemma_description = None
                    previous_gemma_timing_plan = None
                    previous_gemma_end_state = None
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
                observation_image_directory=gemma_image_log,
            )
            if gemma_director_needed else None
        )
        if gemma_director is not None:
            logging.info(
                "HR Endless Sampler Gemma 4 mode: %s.",
                "native draft-MTP (4 draft tokens)" if gemma4_mtp else "original non-MTP decoding",
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
        )
        preparation_message = (
            f"Preparing {len(active_plan)} chunks at {fps:g} fps; "
            f"Video1 continuation carries {video_continuation} frames"
        )
        logging.info("HR Endless Sampler: %s.", preparation_message)
        if preview_execution is not None:
            preview_execution.set_phase(preparation_message, chunk=0)
        chunk_progress = _ChunkProgress(len(active_plan))
        components = [
            ("MiniMax H3 DiT", guider.model_patcher),
            ("Qwen/CLIP", clip.patcher),
            ("H3 video VAE", vae.patcher if vae is not None else None),
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
            if gemma_director is not None:
                preproduction_shots = _gemma_source_shot_records(
                    gemma_shots,
                    active_plan[0]["frame_start"],
                    active_plan[-1]["frame_end"],
                )
                if replay_timing_plan is not None:
                    required_shots = {int(item["shot_number"]) for item in preproduction_shots}
                    cached_shots = {int(shot.source_shot) for shot in replay_timing_plan.shots}
                    if not required_shots.issubset(cached_shots):
                        logging.info(
                            "HR Endless Sampler replay: cached Gemma timing plan does not cover every requested "
                            "source shot, so it will be regenerated."
                        )
                        replay_timing_plan = None
                preproduction_request = {
                    "chunk_count": len(active_plan),
                    "fps": fps,
                    "prompt_mode": "ref" if ref2va else "base",
                    "source_shots": preproduction_shots,
                    "chunks": _gemma_preproduction_chunks(active_plan),
                    "original_prompt": prompt,
                }
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
                                else f"Gemma 4 is planning {len(preproduction_shots)} source shots for "
                                f"{len(active_plan)} chunks before H3 sampling"
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
            for index, chunk in enumerate(active_plan[replay_start_index:], start=replay_start_index):
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
                            timer_started = time.perf_counter()
                            try:
                                observation_frames, observation_indices = _decoded_video_frames(
                                    vae,
                                    previous_video,
                                    include_final=True,
                                    start_frame=previous_plan.get("output_trim_frames", 0),
                                )
                            finally:
                                timing.add("vae_previous_chunk", timer_started)
                            observation_frame_numbers = [
                                previous_plan["frame_start"] + frame_index
                                for frame_index in observation_indices
                            ]

                        continuation_video_label = f"<Video {video_number}>" if include_video1_reference else None
                        continuation_audio_label = f"<Audio {audio_number}>" if include_video1_reference else None
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
                            "target_shots": target_shots,
                            "preproduction_timing_plan": gemma_preproduction_timing_plan.for_target_shots(
                                target_shots,
                                fps,
                            ),
                            "mandatory_coverage": gemma_preproduction_timing_plan.mandatory_coverage(
                                target_shots,
                            ),
                            "character_name_table": gemma_preproduction_timing_plan.character_name_table_text(),
                            "conditioning_context": _gemma_conditioning_context(
                                continuation,
                                context_keyframes,
                                guide_overlap,
                                video_continuation,
                                continuation_video_label,
                                continuation_audio_label,
                                include_video1_reference,
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
                                f"{chunk_label}: Gemma 4 is directing the chunk prompt",
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
                        gemma_description = result.detailed_description
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
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_video_noise, chunk_audio_noise))

                guide_enabled = context_keyframes > 0
                guide_audio_t = (
                    0 if not guide_enabled
                    else _audio_steps(content_start) - _audio_steps(content_start - context_keyframes)
                )
                video_context = None if previous_video is None or not guide_enabled else previous_video[:, :, -guide_video_t:].clone()
                audio_context = None if previous_audio is None or not guide_enabled else previous_audio[..., -guide_audio_t:].clone()
                video_context_start = 0
                audio_end_frame = float(keyframe_duration_frames)
                if audio_context is not None:
                    overhang = previous_audio.shape[-1] - FRAME_RESCALE * previous_frame_count
                    audio_end_frame += overhang / FRAME_RESCALE
                video_items = []
                video_refs = []
                if continuation and use_video_continuation:
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
                        reference_audio_t = _audio_steps(content_start) - _audio_steps(content_start - video_continuation)
                        reference_audio = previous_audio[..., -reference_audio_t:].clone()
                        if vram_monitor is not None:
                            vram_monitor.report(
                                f"chunk {index + 1}/{len(active_plan)} before continuation VAE decode",
                                {
                                    "continuation video latent": reference_latent,
                                    "continuation audio latent": reference_audio,
                                },
                            )
                        # ComfyUI's native video+soundtrack presentation emits the
                        # audio label immediately before the matching video label.
                        video_items.append({"type": "audio"})
                        timer_started = time.perf_counter()
                        try:
                            video_items.append(_decoded_video_item(vae, reference_latent))
                        finally:
                            timing.add("vae_context", timer_started)
                        video_refs.append(_video_ref_block(reference_latent, reference_audio))
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
                    continuation_audio_label = f"<Audio {audio_number}>" if continuation and include_video1_reference else None
                    chunk_prompt = _prompt_with_gemma_description(
                        prompt,
                        gemma_description,
                        drop_picture_anchors=continuation and not ref2va,
                        continuation_video_label=continuation_video_label,
                        continuation_audio_label=continuation_audio_label,
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
                del video_items
                if vram_monitor is not None:
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} after Qwen encode",
                        {"encoded prompt": encoded_prompt, "DiT video references": video_refs},
                    )
                comfy.model_management.unload_model_and_clones(clip.patcher)
                if vae is not None:
                    comfy.model_management.unload_model_and_clones(vae.patcher)
                if debug:
                    logging.info(
                        "HR Endless Sampler released Qwen%s before chunk %d/%d",
                        " and the H3 video VAE" if vae is not None else "",
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
                if preview_execution is not None:
                    preview_execution.set_chunk(
                        index,
                        chunk["frame_start"],
                        chunk["frame_end"] - 1,
                        content_start,
                        chunk["frame_end"] - 1,
                        context_video_t,
                        gemma_description,
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
                previous_video, previous_audio = sampled["samples"].unbind()
                previous_frame_count = chunk["frame_end"] - chunk["frame_start"]
                denoised_chunk_video, denoised_chunk_audio = denoised["samples"].unbind()

                video_trim = context_video_t
                audio_trim = 0 if index == 0 else context_audio_t
                assembled_video = previous_video[:, :, video_trim:].clone()
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
                if gemma_director is not None:
                    previous_gemma_description = gemma_description
                    previous_gemma_timing_plan = result.timing_plan
                    previous_gemma_end_state = result.end_state
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
                                "denoised_video": assembled_denoised_video,
                                "denoised_audio": assembled_denoised_audio,
                                "output_template": output_template,
                                "denoised_template": denoised_template,
                                "gemma_description": previous_gemma_description,
                                "gemma_timing_plan": previous_gemma_timing_plan,
                                "gemma_end_state": previous_gemma_end_state,
                                "h3_render_seconds": h3_render_seconds,
                                "gemma_seconds": chunk_gemma_seconds,
                                "gemma_preproduction_seconds": chunk_preproduction_seconds,
                                "chunk_total_seconds": chunk_total_seconds,
                                "debug_prompt": debug_prompt,
                                "prefix_video_noise": prefix_video_noise,
                                "prefix_audio_noise": prefix_audio_noise,
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
                    "width": width,
                    "height": height,
                    "sampling_steps": max(0, len(sigmas) - 1),
                    "rendered_frames": active_plan[completed_chunks - 1]["frame_end"] if completed_chunks else 0,
                    "full_frames": plan[-1]["frame_end"],
                    "full_chunks": len(plan),
                },
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
        return io.NodeOutput(output_template, denoised_template, "\n\n".join(debug_prompts), timeline)

    sample = execute
