"""Process-isolated Gemma 4 timing planner and chunk-prompt director.

Gemma is intentionally short lived.  A text-only preproduction request first
plans the action timing of every source shot across the known physical chunk
ranges.  Each subsequent request studies that plan, the complete source intent,
and chronological stills from the completed H3 chunk, then writes the H3
``detailed_description`` for the next physical chunk.  Process exit is
deliberate: llama.cpp's CUDA backend owns allocations outside PyTorch and can
retain a backend/context high-water mark after ``Llama.close()``.  Exiting the
worker guarantees those allocations are returned before H3/Qwen use the GPU
again.
"""

from __future__ import annotations

import base64
import codecs
import gc
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image

import comfy.model_management
import folder_paths


GEMMA4_REPOSITORY = "google/gemma-4-12B-it-qat-q4_0-gguf"
GEMMA4_MODEL_FILENAME = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA4_DEBUG_CAPTURE_PREFIX = "hr-endless-sampler-gemma4-"
# Older worker/capture experiments used sibling names under this owned prefix.
# Keep cleanup broad enough to remove those disposable directories too, while
# never touching the durable ``comfyui-hr-endless-sampler`` replay/log root.
GEMMA4_OWNED_TEMP_PREFIX = "hr-endless-sampler-"
GEMMA4_MMPROJ_FILENAME = "mmproj-gemma-4-12b-it-qat-q4_0.gguf"
GEMMA4_MTP_REPOSITORY = "Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF"
GEMMA4_MTP_FILENAME = "gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"
GEMMA4_MODEL_DIRECTORY = "llama_cpp/gemma-4-12b-it-qat-q4_0"
GEMMA4_REQUIRED_VERSION = "0.3.49"
GEMMA4_IMAGE_MIN_TOKENS = 70
GEMMA4_IMAGE_MAX_TOKENS = 1120
GEMMA4_BATCH_SIZE = GEMMA4_IMAGE_MAX_TOKENS
# Match the user's known-good native llama.cpp Gemma server configuration.
# Gemma 4's channel template greedily selects the empty ``thought`` turn when
# temperature is zero, which yields no visible content at all. A normal
# sampling distribution lets it enter its final-answer channel and produce the
# requested JSON. These are intentionally shared by base, cached, correction,
# and MTP calls so the assistant observes the same target distribution.
GEMMA4_TEMPERATURE = 1.0
GEMMA4_TOP_P = 0.95
GEMMA4_TOP_K = 64
# Mirror the tested native llama.cpp Gemma configuration: a 32K text context
# with Q8_0 K/V cache. llama-cpp-python exposes GGML_TYPE_Q8_0 as 8 in the
# pinned 0.3.35 runtime. Keeping the value local also lets lightweight unit
# tests exercise runtime construction without importing libllama/CUDA.
GEMMA4_KV_CACHE_Q8_0 = 8
GEMMA4_CONTEXT_TOKENS = 32768
GEMMA4_WORKER_RETRY_LIMIT = 10
GEMMA4_RESPONSE_REPAIR_LIMIT = 10
# Gemma receives a native 4K thought budget, then emits its structured answer.
# The two pre-production passes therefore retain the full 32K completion room
# rather than treating thought tokens as their entire response allowance.
GEMMA4_CHUNK_RESPONSE_TOKENS = 8132
GEMMA4_GLOBAL_PREPRODUCTION_RESPONSE_TOKENS = 32768
GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS = 32768
GEMMA4_TIMING_RESPONSE_TOKENS = 16384
GEMMA4_REASONING_BUDGET = 2048
# Gemma 4's native template puts this marker in the prompt when thinking is
# enabled.  The first generated reasoning channel is then closed by
# ``<channel|>``.  These are model tokens, not prose strings.
GEMMA4_REASONING_START = "<|think|>"
GEMMA4_REASONING_END = "<channel|>"
GEMMA4_REASONING_BUDGET_MESSAGE = (
    "\n...Wait, I have been thinking long enough. Let me start answering the user's question.\n"
)
# This is a permissive structural ceiling, not the desired performance pace.
# Natural dramatic speech is normally slower; the small fixed allowance keeps
# brief exclamations from failing merely because their overlay is very short.
GEMMA4_MAX_DIALOGUE_WORDS_PER_SECOND = 4.0
GEMMA4_DIALOGUE_BURST_WORD_ALLOWANCE = 4
GEMMA4_DIALOGUE_ELLIPSIS_SECONDS = 0.5
# A valid JSON response is normally produced by the unconstrained decoder. If
# Gemma accidentally answers in its private thought channel, correct it as a
# real next chat turn first.  That keeps the already encoded request, images,
# and MTP state alive.  The grammar path is only a final compatibility guard:
# on Gemma 4 it is much slower and has itself returned an empty thought turn.
GEMMA4_JSON_FORMAT_REPAIR_LIMIT = 2
GEMMA4_PROMPTS_PATH = Path(__file__).with_name("gemma4_prompts.txt")
MINIMAX_PROMPT_SUMMARY_PATH = Path(__file__).with_name("minimax_h3_prompt_summary.txt")
MINIMAX_PROMPT_SKILL_PATH = Path(__file__).with_name("vendor") / "minimax-h3-prompt-writing" / "SKILL.md"
MINIMAX_PROMPT_GUIDES = {
    "base": MINIMAX_PROMPT_SKILL_PATH.parent / "references" / "base-en.txt",
    "ref": MINIMAX_PROMPT_SKILL_PATH.parent / "references" / "ref-en.txt",
}
_PROMPT_SECTION = re.compile(r"(?ms)^\[([A-Z][A-Z0-9_]*)\]\s*$\n?(.*?)(?=^\[[A-Z][A-Z0-9_]*\]\s*$|\Z)")
_PROMPT_PLACEHOLDER = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")
_SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.IGNORECASE | re.DOTALL)
_DIALOGUE_CONTROL = re.compile(r"</?(?:scenetrans|cutoff)>", re.IGNORECASE)
_LEGACY_END_STATE = re.compile(r"(?is)(?:\s|^)*\[end\s+state\]\s*(.*?)\s*$")
_SUBJECT_REFERENCE = re.compile(r"^<Subject\s+\d+>$", re.IGNORECASE)
_WORKER_RESULT_PREFIX = "MINIMAX_H3_GEMMA4_RESULT="
_WORKER_PROGRESS_PREFIX = "HR_ENDLESS_SAMPLER_GEMMA4_PROGRESS="
_PREPRODUCTION_CACHE_FORMAT = "hr-endless-sampler-gemma4-preproduction-kv-v1"
_PREPRODUCTION_CACHE_DIRECTORY = "comfyui-hr-endless-sampler/gemma4_preproduction_kv"
_PREPRODUCTION_CACHE_MIN_FREE_BYTES = 6 * 1024 ** 3
GEMMA4_RAW_OUTPUT_DIRECTORY = "comfyui-hr-endless-sampler"
GEMMA4_RAW_OUTPUT_FILENAME = "last_gemma_raw_output.txt"
GEMMA4_LIVE_OUTPUT_FILENAME = "last_gemma_live_output.txt"
_ACTIVE_RAW_OUTPUT_PATH: Path | None = None
_ACTIVE_RAW_OUTPUT_OPERATION = "unknown operation"
_ACTIVE_LIVE_OUTPUT_PATH: Path | None = None
_ACTIVE_LIVE_OUTPUT_OPERATION = "unknown operation"


def gemma4_raw_output_path() -> Path:
    """Return the fixed, latest-render raw Gemma transcript path."""
    return Path(tempfile.gettempdir()) / GEMMA4_RAW_OUTPUT_DIRECTORY / GEMMA4_RAW_OUTPUT_FILENAME


def gemma4_live_output_path() -> Path:
    """Return the append-only, currently-decoding Gemma transcript path."""
    return Path(tempfile.gettempdir()) / GEMMA4_RAW_OUTPUT_DIRECTORY / GEMMA4_LIVE_OUTPUT_FILENAME


def reset_gemma4_live_output_log() -> Path | None:
    """Create the live transcript before a sampler run starts its Gemma workers."""
    path = gemma4_live_output_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "HR Endless Sampler live Gemma token stream\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            "This file is appended while Gemma decodes; completed structured responses are in last_gemma_raw_output.txt.\n\n",
            encoding="utf-8",
        )
    except OSError as error:
        logging.warning("HR Endless Sampler could not reset live Gemma output log %s: %s", path, error)
        return None
    logging.info("HR Endless Sampler writing live Gemma output to %s", path)
    return path


def reset_gemma4_raw_output_log(enabled: bool) -> Path | None:
    """Start a debug transcript, or remove a stale transcript for a non-debug run."""
    path = gemma4_raw_output_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not enabled:
            path.unlink(missing_ok=True)
            return None
        path.write_text(
            "HR Endless Sampler raw Gemma worker responses\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            "This debug-only file records every complete llama.cpp response before JSON parsing.\n\n",
            encoding="utf-8",
        )
    except OSError as error:
        logging.warning("HR Endless Sampler could not reset raw Gemma output log %s: %s", path, error)
        return None
    logging.info("HR Endless Sampler writing raw Gemma worker responses to %s", path)
    return path


def _configure_raw_output_log(request: dict[str, Any]) -> None:
    """Enable the fixed transcript inside one disposable debug worker."""
    global _ACTIVE_RAW_OUTPUT_PATH, _ACTIVE_RAW_OUTPUT_OPERATION
    global _ACTIVE_LIVE_OUTPUT_PATH, _ACTIVE_LIVE_OUTPUT_OPERATION
    _ACTIVE_RAW_OUTPUT_PATH = gemma4_raw_output_path() if bool(request.get("debug", False)) else None
    _ACTIVE_LIVE_OUTPUT_PATH = gemma4_live_output_path()
    operation = str(request.get("operation") or "chunk_prompt")
    chunk_number = request.get("chunk_number")
    _ACTIVE_RAW_OUTPUT_OPERATION = (
        f"{operation} chunk={int(chunk_number)}"
        if chunk_number is not None
        else operation
    )
    _ACTIVE_LIVE_OUTPUT_OPERATION = _ACTIVE_RAW_OUTPUT_OPERATION


def _append_raw_gemma_response(stage: str, response: Any) -> None:
    """Persist one unparsed llama.cpp response without affecting worker success."""
    if _ACTIVE_RAW_OUTPUT_PATH is None:
        return
    try:
        with _ACTIVE_RAW_OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
            output_file.write("=" * 200)
            output_file.write("\n")
            output_file.write(
                f"{datetime.now(timezone.utc).isoformat()} | worker_pid={os.getpid()} | "
                f"{_ACTIVE_RAW_OUTPUT_OPERATION} | {stage}\n"
            )
            output_file.write(json.dumps(response, ensure_ascii=False, indent=2, default=str))
            output_file.write("\n\n")
    except OSError as error:
        logging.warning(
            "HR Endless Sampler could not append raw Gemma response to %s: %s",
            _ACTIVE_RAW_OUTPUT_PATH,
            error,
        )


def _begin_live_gemma_generation(generation: int) -> None:
    """Mark one live llama.cpp decoder turn without delaying token generation."""
    if _ACTIVE_LIVE_OUTPUT_PATH is None:
        return
    try:
        with _ACTIVE_LIVE_OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
            output_file.write("=" * 200)
            output_file.write("\n")
            output_file.write(
                f"{datetime.now(timezone.utc).isoformat()} | worker_pid={os.getpid()} | "
                f"{_ACTIVE_LIVE_OUTPUT_OPERATION} | decoder generation {generation}\n"
            )
    except OSError as error:
        logging.warning(
            "HR Endless Sampler could not start live Gemma output in %s: %s",
            _ACTIVE_LIVE_OUTPUT_PATH,
            error,
        )


def _append_live_gemma_text(text: str) -> None:
    """Flush decoded Gemma text so ``tail -f`` can inspect an active worker."""
    if not text or _ACTIVE_LIVE_OUTPUT_PATH is None:
        return
    try:
        with _ACTIVE_LIVE_OUTPUT_PATH.open("a", encoding="utf-8") as output_file:
            output_file.write(text)
    except OSError as error:
        logging.warning(
            "HR Endless Sampler could not append live Gemma output to %s: %s",
            _ACTIVE_LIVE_OUTPUT_PATH,
            error,
        )


def _gemma_reasoning_budget_kwargs() -> dict[str, Any]:
    """Return the native first-thought-block budget shared by every Gemma turn."""
    return {
        "reasoning_budget": GEMMA4_REASONING_BUDGET,
        "reasoning_start": GEMMA4_REASONING_START,
        "reasoning_end": GEMMA4_REASONING_END,
        "reasoning_budget_message": GEMMA4_REASONING_BUDGET_MESSAGE,
        "reasoning_start_in_prompt": True,
    }


class Gemma4DependencyError(RuntimeError):
    """Raised when the node cannot use its deliberately pinned local runtime."""


class Gemma4ObservationError(RuntimeError):
    """Raised for a malformed or unusable model observation."""

    def __init__(self, message: str, *, raw_json: str = ""):
        super().__init__(message)
        self.raw_json = raw_json


class Gemma4WorkerExitError(Gemma4ObservationError):
    """A disposable native worker exited instead of returning a usable result."""

    def __init__(self, message: str, *, returncode: int | None = None,
                 worker_error_type: str = "", raw_json: str = ""):
        super().__init__(message, raw_json=raw_json)
        self.returncode = returncode
        self.worker_error_type = worker_error_type


class Gemma4MTPOutputError(Gemma4ObservationError):
    """MTP produced no parseable answer, so this operation needs a clean retry.

    This is deliberately narrower than :class:`Gemma4ObservationError`.
    Creative/schema mistakes remain Gemma-authored correction turns; only an
    MTP transport/decode result containing no JSON at all crosses the worker
    boundary and activates the existing operation-local original-decoder
    retry.
    """


class Gemma4PreproductionCache:
    """One render-local clean Gemma context, stored outside the H3 process.

    The cache is deliberately reset before every sampler execution.  It is not
    a cross-render prompt cache: the saved llama.cpp state includes an exact
    preproduction plan, source prompt, model/runtime configuration, and KV
    context.  Linux prefers ``/dev/shm`` so the large state never touches the
    already busy system disk; Windows and constrained Linux systems fall back
    to the normal temporary directory.
    """

    def __init__(self):
        self.root = self._cache_root()
        self.state_path = self.root / "preproduction_state.bin"
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def _cache_root() -> Path:
        override = os.environ.get("HR_ENDLESS_SAMPLER_GEMMA_CACHE_DIR")
        candidates: list[Path] = []
        if override:
            candidates.append(Path(override))
        if os.name != "nt":
            candidates.append(Path("/dev/shm"))
        candidates.append(Path(tempfile.gettempdir()))
        for parent in candidates:
            try:
                if parent.is_dir() and os.access(parent, os.W_OK | os.X_OK):
                    # A 32K Q8_0 Gemma KV snapshot is still multi-GiB. Do not
                    # choose a tiny container /dev/shm and fail later when a
                    # normal temporary directory can hold it instead.
                    if parent == Path("/dev/shm") and shutil.disk_usage(parent).free < _PREPRODUCTION_CACHE_MIN_FREE_BYTES:
                        continue
                    return parent / _PREPRODUCTION_CACHE_DIRECTORY
            except OSError:
                continue
        # ``tempfile.gettempdir`` is expected to be usable. Keep the final
        # value deterministic so an error can name the attempted location.
        return Path(tempfile.gettempdir()) / _PREPRODUCTION_CACHE_DIRECTORY

    def reset(self) -> None:
        """Clear any earlier render state before preparing this render."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def worker_spec(self) -> dict[str, str]:
        return {
            "format": _PREPRODUCTION_CACHE_FORMAT,
            "state_path": str(self.state_path),
            "manifest_path": str(self.manifest_path),
        }

    def ready(self) -> bool:
        if not self.state_path.is_file() or not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("format") == _PREPRODUCTION_CACHE_FORMAT

    def size_bytes(self) -> int:
        try:
            return int(self.state_path.stat().st_size)
        except OSError:
            return 0

def _preproduction_cache_paths(spec: object) -> tuple[Path, Path]:
    """Validate the worker-only cache specification supplied by the sampler."""
    if not isinstance(spec, dict) or spec.get("format") != _PREPRODUCTION_CACHE_FORMAT:
        raise Gemma4ObservationError("Gemma preproduction KV cache specification is invalid")
    try:
        state_path = Path(str(spec["state_path"]))
        manifest_path = Path(str(spec["manifest_path"]))
    except (KeyError, TypeError, ValueError) as error:
        raise Gemma4ObservationError("Gemma preproduction KV cache paths are invalid") from error
    return state_path, manifest_path


@dataclass(frozen=True)
class GemmaPromptAttempt:
    """One unmodified JSON response from Gemma during a directing handoff."""

    kind: str
    raw_json: str
    validation_warnings: tuple[str, ...] = ()
    correction_prompt: str = ""


@dataclass(frozen=True)
class GemmaChunkPrompt:
    confidence: str
    analysis: str
    detailed_description: str
    raw_json: str
    timing_plan: str = ""
    end_state: str = ""
    retention_analysis: str = ""
    last_seen_character_state: tuple[dict[str, Any], ...] = ()
    system_prompt: str = ""
    observation_prompt: str = ""
    validation_warnings: tuple[str, ...] = ()
    attempts: tuple[GemmaPromptAttempt, ...] = ()


@dataclass(frozen=True)
class GemmaShotTimingBeat:
    """One serial visual beat on Gemma's source-relative shot timeline."""

    start_frame: int
    end_frame: int
    action: str


@dataclass(frozen=True)
class GemmaShotTimingOverlay:
    """A concurrent source-relative dialogue, sound, or sustained action."""

    start_frame: int
    end_frame: int
    overlay_type: str
    content: str
    dialogue_segments: tuple["GemmaDialogueSegment", ...] = ()
    continues_source_dialogue: bool = False


@dataclass(frozen=True)
class GemmaDialogueSegment:
    """One word-exact piece of a dialogue owned by one retained chunk slice."""

    start_frame: int
    end_frame: int
    content: str


@dataclass(frozen=True)
class GemmaCharacterContinuity:
    """Planned physical state for one character across one retained slice."""

    character_name: str
    subject: str
    entry_state: str
    expected_exit_state: str


@dataclass(frozen=True)
class GemmaShotContinuitySlice:
    """Preproduction character-state contract for one chunk-owned shot slice."""

    start_frame: int
    end_frame: int
    characters: tuple[GemmaCharacterContinuity, ...]


@dataclass(frozen=True)
class GemmaShotTimingShot:
    """The complete preproduction schedule for one unchanged source shot.

    ``visual_beats`` deliberately remain a contiguous timeline. ``overlays``
    are allowed to overlap them and one another, so dialogue and sound are not
    falsely treated as events that must wait until the visible action ends.
    """

    source_shot: int
    shot_start_frame: int
    shot_end_frame: int
    visual_beats: tuple[GemmaShotTimingBeat, ...]
    overlays: tuple[GemmaShotTimingOverlay, ...] = ()
    continuity_slices: tuple[GemmaShotContinuitySlice, ...] = ()
    # True only when the original prompt requires lighting or color to change
    # within this source shot. The sampler uses false as permission for a
    # final output-only shot-grade stabilization pass.
    light_change: bool = True


@dataclass(frozen=True)
class GemmaCharacterSubject:
    """One source-proven character name and its existing H3 subject label."""

    character_name: str
    subject: str


@dataclass(frozen=True)
class GemmaShotTimingPlan:
    """Validated Gemma-authored shot schedule reused by every chunk director."""

    confidence: str
    analysis: str
    shots: tuple[GemmaShotTimingShot, ...]
    character_name_table: tuple[GemmaCharacterSubject, ...]
    raw_json: str
    production_bible_json: str = ""
    shot_planning_prompts: tuple[str, ...] = ()
    system_prompt: str = ""
    planning_prompt: str = ""
    validation_warnings: tuple[str, ...] = ()
    attempts: tuple[GemmaPromptAttempt, ...] = ()

    def character_name_table_text(self) -> str:
        """Render the Gemma-owned table for the later visual directing calls."""
        if not self.character_name_table:
            return "No explicit named-character-to-subject mapping was found in the original prompt."
        return "\n".join(
            f"- {entry.character_name} -> {entry.subject}"
            for entry in self.character_name_table
        )

    def production_bible_text(self) -> str:
        """Return the immutable global preproduction context for chunk directing."""
        return self.production_bible_json or (
            "No separate global production bible is available; use the validated shot schedules."
        )

    @staticmethod
    def _matching_continuity_slice(
        shot: GemmaShotTimingShot,
        target: dict[str, Any],
    ) -> GemmaShotContinuitySlice | None:
        start = int(target["target_start"]) - shot.shot_start_frame
        end = int(target["target_end"]) - shot.shot_start_frame
        return next(
            (
                item for item in shot.continuity_slices
                if item.start_frame == start and item.end_frame == end
            ),
            None,
        )

    def current_character_subjects(
        self,
        target_shots: Sequence[dict[str, Any]],
    ) -> tuple[GemmaCharacterSubject, ...]:
        """Return every planned character participating in the current slice."""
        targets = {int(item["shot_number"]): item for item in target_shots}
        seen: set[str] = set()
        result: list[GemmaCharacterSubject] = []
        for shot in self.shots:
            target = targets.get(shot.source_shot)
            if target is None:
                continue
            continuity = self._matching_continuity_slice(shot, target)
            if continuity is None:
                continue
            for character in continuity.characters:
                key = character.character_name.casefold()
                if key in seen:
                    continue
                seen.add(key)
                result.append(GemmaCharacterSubject(character.character_name, character.subject))
        return tuple(result)

    def continuity_for_target_shots(self, target_shots: Sequence[dict[str, Any]]) -> str:
        """Render planned entry/exit states for Gemma's observed-state comparison."""
        targets = {int(item["shot_number"]): item for item in target_shots}
        blocks: list[str] = []
        for shot in self.shots:
            target = targets.get(shot.source_shot)
            if target is None:
                continue
            continuity = self._matching_continuity_slice(shot, target)
            if continuity is None:
                continue
            lines = [
                f"Source Shot {shot.source_shot}, source-relative frames "
                f"{continuity.start_frame}-{continuity.end_frame - 1}:"
            ]
            if not continuity.characters:
                lines.append("- No mapped character is physically present in this slice.")
            for character in continuity.characters:
                lines.append(
                    f"- {character.character_name} ({character.subject}) — planned entry: "
                    f"{character.entry_state}; expected by slice end: {character.expected_exit_state}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "No planned character continuity applies to this slice."

    @staticmethod
    def _beat_identifier(source_shot: int, kind: str, index: int) -> str:
        return f"S{source_shot}.{kind}{index}"

    def mandatory_coverage(self, target_shots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return model-authored beats intersecting a live retained chunk.

        This is metadata only: it lets the chunk response attest that every
        scheduled current-slice beat was actually included. The sampler never
        turns it into H3 prose or invents timing of its own.
        """
        targets_by_number = {int(target["shot_number"]): target for target in target_shots}
        required: list[dict[str, Any]] = []
        for shot in self.shots:
            target = targets_by_number.get(shot.source_shot)
            if target is None or "target_start" not in target or "target_end" not in target:
                continue
            target_start = int(target["target_start"])
            target_end = int(target["target_end"])
            for kind, entries in (("V", shot.visual_beats), ("O", shot.overlays)):
                for index, entry in enumerate(entries, 1):
                    coverage_entry = entry
                    identifier = self._beat_identifier(shot.source_shot, kind, index)
                    dialogue_segment_index = None
                    if (
                        kind == "O"
                        and entry.overlay_type == "dialogue"
                        and entry.dialogue_segments
                    ):
                        target_relative_start = target_start - shot.shot_start_frame
                        target_relative_end = target_end - shot.shot_start_frame
                        matching_segments = [
                            (segment_index, segment)
                            for segment_index, segment in enumerate(entry.dialogue_segments, 1)
                            if max(target_relative_start, segment.start_frame)
                            < min(target_relative_end, segment.end_frame)
                        ]
                        if not matching_segments:
                            continue
                        # Retained ownership intervals do not overlap, so a
                        # live source-shot slice owns at most one segment.
                        dialogue_segment_index, coverage_entry = matching_segments[0]
                        identifier += f".D{dialogue_segment_index}"
                    entry_start = shot.shot_start_frame + coverage_entry.start_frame
                    entry_end = shot.shot_start_frame + coverage_entry.end_frame
                    overlap_start = max(target_start, entry_start)
                    overlap_end = min(target_end, entry_end)
                    if overlap_start >= overlap_end:
                        continue
                    item: dict[str, Any] = {
                        "id": identifier,
                        "kind": "visual" if kind == "V" else "overlay",
                        "source_shot": shot.source_shot,
                        "source_start_frame": coverage_entry.start_frame,
                        "source_end_frame": coverage_entry.end_frame,
                        "overlap_start_frame": overlap_start - shot.shot_start_frame,
                        "overlap_end_frame": overlap_end - shot.shot_start_frame,
                        "action": coverage_entry.action if kind == "V" else coverage_entry.content,
                    }
                    if kind == "O":
                        item["overlay_type"] = entry.overlay_type
                        if dialogue_segment_index is not None:
                            item["dialogue_segment_index"] = dialogue_segment_index
                            item["dialogue_segment_count"] = len(entry.dialogue_segments)
                            item["dialogue_continuation"] = (
                                entry.continues_source_dialogue or dialogue_segment_index > 1
                            )
                    required.append(item)
        return required

    def for_target_shots(self, target_shots: Sequence[dict[str, Any]], fps: float) -> str:
        """Render relevant full schedules plus mandatory current-slice beats.

        The sampler never writes action prose from this data.  It simply
        presents Gemma's preproduction schedule back to the later visual
        director, alongside the current generated evidence.  The explicit
        intersections make a late-starting beat actionable: a director cannot
        silently defer a camera arc that begins at frame 32 merely because the
        physical chunk ends at frame 38.
        """
        targets_by_number = {int(target["shot_number"]): target for target in target_shots}
        wanted = set(targets_by_number)
        blocks: list[str] = []
        required_blocks: list[str] = []
        for shot in self.shots:
            if shot.source_shot not in wanted:
                continue
            target = targets_by_number[shot.source_shot]
            shot_frames = shot.shot_end_frame - shot.shot_start_frame
            lines = [
                f"Source Shot {shot.source_shot}: immutable preproduction timing schedule for "
                f"global frames {shot.shot_start_frame}-{shot.shot_end_frame - 1} "
                f"({shot_frames} frames, {shot_frames / fps:.3f} s)."
            ]
            lines.append("Serial visual timeline:")
            for index, beat in enumerate(shot.visual_beats, 1):
                global_start = shot.shot_start_frame + beat.start_frame
                global_end = shot.shot_start_frame + beat.end_frame - 1
                lines.append(
                    f"- [{self._beat_identifier(shot.source_shot, 'V', index)}] source-relative frames "
                    f"{beat.start_frame}-{beat.end_frame - 1} (global {global_start}-{global_end}): {beat.action}"
                )
            if shot.overlays:
                lines.append("Concurrent overlays (these may occur during the visual timeline):")
                for index, overlay in enumerate(shot.overlays, 1):
                    global_start = shot.shot_start_frame + overlay.start_frame
                    global_end = shot.shot_start_frame + overlay.end_frame - 1
                    lines.append(
                        f"- [{self._beat_identifier(shot.source_shot, 'O', index)}] {overlay.overlay_type} "
                        f"at source-relative frames {overlay.start_frame}-{overlay.end_frame - 1} "
                        f"(global {global_start}-{global_end}): {overlay.content}"
                    )
                    if overlay.overlay_type == "dialogue":
                        lines.append("  Immutable retained-slice dialogue segments:")
                        for segment_index, segment in enumerate(overlay.dialogue_segments, 1):
                            segment_global_start = shot.shot_start_frame + segment.start_frame
                            segment_global_end = shot.shot_start_frame + segment.end_frame - 1
                            lines.append(
                                f"  - [{self._beat_identifier(shot.source_shot, 'O', index)}.D{segment_index}] "
                                f"source-relative frames {segment.start_frame}-{segment.end_frame - 1} "
                                f"(global {segment_global_start}-{segment_global_end}): {segment.content}"
                            )
            else:
                lines.append("Concurrent overlays: none planned.")
            lines.append("Character continuity by retained physical slice:")
            for continuity in shot.continuity_slices:
                lines.append(
                    f"- source-relative frames {continuity.start_frame}-{continuity.end_frame - 1}:"
                )
                if not continuity.characters:
                    lines.append("  - no mapped character is physically present")
                for character in continuity.characters:
                    lines.append(
                        f"  - {character.character_name} ({character.subject}) entry: {character.entry_state}; "
                        f"expected exit: {character.expected_exit_state}"
                    )
            blocks.append("\n".join(lines))

            # The preproduction transcript renders this same method with
            # source-shot records, which intentionally have no physical target
            # slice. Only a live chunk request has these two fields and needs
            # mandatory beat intersections.
            if "target_start" not in target or "target_end" not in target:
                continue
            target_start = int(target["target_start"])
            target_end = int(target["target_end"])
            required_lines = [
                f"Source Shot {shot.source_shot}: this chunk retains global frames "
                f"{target_start}-{target_end - 1} (source-relative frames "
                f"{target_start - shot.shot_start_frame}-{target_end - shot.shot_start_frame - 1})."
            ]
            for item in self.mandatory_coverage((target,)):
                phase = "continues without restarting" if item.get("dialogue_continuation") else (
                    "begins" if item["source_start_frame"] >= item["overlap_start_frame"] else "continues"
                )
                descriptor = item["kind"]
                if item["kind"] == "overlay":
                    descriptor += f"/{item['overlay_type']}"
                required_lines.append(
                    f"- Required now [{item['id']}], {descriptor}, source-relative frames "
                    f"{item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}: "
                    f"this planned beat {phase} here — {item['action']}"
                )
            continuity = self._matching_continuity_slice(shot, target)
            if continuity is not None:
                required_lines.append("Planned character continuity for this retained slice:")
                if not continuity.characters:
                    required_lines.append("- no mapped character is physically present")
                for character in continuity.characters:
                    required_lines.append(
                        f"- {character.character_name} ({character.subject}) entry: {character.entry_state}; "
                        f"expected exit: {character.expected_exit_state}"
                    )
            required_blocks.append("\n".join(required_lines))

        if not blocks:
            return "No relevant preproduction timing schedule is available."
        if not required_blocks:
            return "\n\n".join(blocks)
        return (
            "MANDATORY CURRENT-SLICE BEAT COVERAGE — EACH LISTED BEAT MUST APPEAR IN "
            "detailed_description\n"
            "A listed beat may remain unfinished beyond this slice, but its start or continuation in the listed "
            "frames must be explicitly described now.\n\n"
            + "\n\n".join(required_blocks)
            + "\n\nCOMPLETE RELEVANT PREPRODUCTION SCHEDULE (for pacing context)\n"
            + "\n\n".join(blocks)
        )

    def current_slice_coverage_text(self, target_shots: Sequence[dict[str, Any]]) -> str:
        """Render only the live required portions of an already cached plan."""
        targets_by_number = {int(target["shot_number"]): target for target in target_shots}
        blocks: list[str] = []
        for shot_number, target in targets_by_number.items():
            if "target_start" not in target or "target_end" not in target:
                continue
            shot = next((item for item in self.shots if item.source_shot == shot_number), None)
            if shot is None:
                continue
            matching = [item for item in self.mandatory_coverage((target,)) if item["source_shot"] == shot_number]
            if not matching:
                continue
            lines = [
                f"Source Shot {shot_number}: this chunk retains source-relative frames "
                f"{int(target['target_start']) - int(target['shot_start'])}-"
                f"{int(target['target_end']) - int(target['shot_start']) - 1}."
            ]
            for item in matching:
                phase = "continues without restarting" if item.get("dialogue_continuation") else (
                    "begins" if item["source_start_frame"] >= item["overlap_start_frame"] else "continues"
                )
                descriptor = item["kind"]
                if item["kind"] == "overlay":
                    descriptor += f"/{item['overlay_type']}"
                lines.append(
                    f"- Required now [{item['id']}], {descriptor}, source-relative frames "
                    f"{item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}: "
                    f"this planned beat {phase} here — {item['action']}"
                )
            continuity = self._matching_continuity_slice(shot, target)
            if continuity is not None:
                lines.append("Planned character continuity for this retained slice:")
                if not continuity.characters:
                    lines.append("- no mapped character is physically present")
                for character in continuity.characters:
                    lines.append(
                        f"- {character.character_name} ({character.subject}) entry: {character.entry_state}; "
                        f"expected exit: {character.expected_exit_state}"
                    )
            blocks.append("\n".join(lines))
        if not blocks:
            return "No mandatory current-slice beat coverage is required."
        return (
            "MANDATORY CURRENT-SLICE BEAT COVERAGE — the full immutable schedule is already in your "
            "preproduction memory. Each listed beat must appear in detailed_description now.\n\n"
            + "\n\n".join(blocks)
        )


def _gemma_prompt_templates() -> dict[str, str]:
    """Read editable runtime templates rather than burying them in Python.

    This intentionally reads on every observation.  It makes prompt iteration
    possible without changing Python or reloading a running ComfyUI process;
    the next Gemma handoff uses the saved file contents.
    """
    try:
        source = GEMMA4_PROMPTS_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise Gemma4ObservationError(
            f"Could not read editable Gemma 4 prompt file {GEMMA4_PROMPTS_PATH}: {error}"
        ) from error
    sections = {
        match.group(1): match.group(2).strip()
        for match in _PROMPT_SECTION.finditer(source)
    }
    required = ("SYSTEM", "OBSERVATION")
    missing = [section for section in required if not sections.get(section)]
    if missing:
        raise Gemma4ObservationError(
            f"Gemma 4 prompt file {GEMMA4_PROMPTS_PATH} is missing non-empty "
            f"section(s): {', '.join(missing)}"
        )
    return sections


def _minimax_prompt_reference(mode: str) -> str:
    """Return the compact reviewed H3 working reference used at runtime.

    The complete MiniMax skill and mode guides remain vendored as reviewed
    runtime dependencies.  Injecting all of that material into every Gemma
    request consumed most of the 16K context without helping a chunk-local
    director.  This summary preserves the rules the director actually needs;
    ``mode`` is still validated so an unsupported integration cannot silently
    receive the wrong prompt contract.
    """
    if mode not in MINIMAX_PROMPT_GUIDES:
        raise Gemma4ObservationError(f"Unknown MiniMax prompt mode for Gemma: {mode!r}")
    try:
        summary = MINIMAX_PROMPT_SUMMARY_PATH.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise Gemma4ObservationError(
            f"Could not read MiniMax H3 prompt-working summary {MINIMAX_PROMPT_SUMMARY_PATH}: {error}"
        ) from error
    if not summary:
        raise Gemma4ObservationError(
            f"MiniMax H3 prompt-working summary {MINIMAX_PROMPT_SUMMARY_PATH} is empty"
        )
    return (
        "Reviewed MiniMax H3 prompt-writing working summary follows. The sampler's local frame, "
        "marker, and chunk contracts override any general full-video example.\n\n"
        + summary
    )


def _render_gemma_prompt(template: str, values: dict[str, str]) -> str:
    """Substitute only explicit ``{{lowercase_name}}`` template fields."""
    missing = sorted({match.group(1) for match in _PROMPT_PLACEHOLDER.finditer(template)} - values.keys())
    if missing:
        raise Gemma4ObservationError(
            "Gemma 4 prompt template contains unknown placeholder(s): " + ", ".join(missing)
        )
    for name, value in values.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def _model_paths() -> tuple[Path, Path]:
    model_dir = Path(folder_paths.models_dir) / GEMMA4_MODEL_DIRECTORY
    return model_dir / GEMMA4_MODEL_FILENAME, model_dir / GEMMA4_MMPROJ_FILENAME


def _ensure_model_files() -> tuple[Path, Path]:
    model_path, mmproj_path = _model_paths()
    if model_path.is_file() and mmproj_path.is_file():
        return model_path, mmproj_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise Gemma4DependencyError(
            "Gemma 4 continuity requires huggingface-hub. Install this custom node's requirements.txt."
        ) from error

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "HR Endless Sampler is downloading the Gemma 4 continuity model to %s. "
        "This one-time download is about 7.2 GB.",
        model_path.parent,
    )
    paths = []
    for filename in (GEMMA4_MODEL_FILENAME, GEMMA4_MMPROJ_FILENAME):
        # local_dir writes the actual file directly into our persistent model
        # folder; it does not create a second full file in Hugging Face's normal
        # global cache.
        paths.append(
            Path(
                hf_hub_download(
                    repo_id=GEMMA4_REPOSITORY,
                    filename=filename,
                    local_dir=model_path.parent,
                )
            )
        )
    return paths[0], paths[1]


def _ensure_mtp_model_file() -> Path:
    """Return the small Gemma QAT assistant head, downloading it once."""
    model_path, _ = _model_paths()
    mtp_path = model_path.parent / GEMMA4_MTP_FILENAME
    if mtp_path.is_file():
        return mtp_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise Gemma4DependencyError(
            "Gemma 4 MTP requires huggingface-hub. Install this custom node's requirements.txt."
        ) from error

    mtp_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "HR Endless Sampler is downloading the Gemma 4 MTP assistant to %s. "
        "This one-time download is about 465 MB.",
        mtp_path.parent,
    )
    return Path(
        hf_hub_download(
            repo_id=GEMMA4_MTP_REPOSITORY,
            filename=GEMMA4_MTP_FILENAME,
            local_dir=mtp_path.parent,
        )
    )


def _gemma4_mtmd_handler_type(base_handler):
    """Add only Endless's append-only turn helper to the current MTMD handler.

    JamePeng 0.3.49 already exposes Gemma 4's complete MTMD initializer,
    including visual-token and batch budgets. Inherit it unchanged so its
    private multimodal state stays compatible with the installed runtime.
    """

    class Gemma4MTMDChatHandler(base_handler):
        def append_user_chat_completion(
            self,
            *,
            llama: Any,
            content: str | Sequence[dict[str, Any]],
            temperature: float = 0.0,
            top_p: float = 1.0,
            top_k: int = GEMMA4_TOP_K,
            max_tokens: int = 1024,
            response_format: dict[str, Any] | None = None,
            reasoning_budget: int = GEMMA4_REASONING_BUDGET,
            reasoning_start: str = GEMMA4_REASONING_START,
            reasoning_end: str = GEMMA4_REASONING_END,
            reasoning_budget_message: str | None = GEMMA4_REASONING_BUDGET_MESSAGE,
            reasoning_start_in_prompt: bool = False,
        ) -> dict[str, Any]:
            """Append one user turn to an existing Gemma MTMD conversation.

            ``Gemma4ChatHandler.__call__`` always resets Llama and clears the
            KV cache.  That is correct for a new request but wasteful (and
            semantically wrong) for a correction to the answer it just
            produced.  This narrow companion mirrors its media tokenisation
            path while preserving the evaluated conversation and appending the
            closing-assistant/new-user chat-template suffix only.

            A synthetic assistant marker lets the model's own Jinja template
            tell us the exact inter-turn bytes for this GGUF.  We discard the
            synthetic prefix and retain the closing marker before the new user
            turn, rather than hard-coding Gemma special-token spelling.
            """
            import ctypes
            from llama_cpp.llama_chat_format import _grammar_for_response_format

            self._init_mtmd_context(llama)
            assert self.mtmd_ctx is not None

            if isinstance(content, str):
                user_content: list[dict[str, Any]] = [{"type": "text", "text": content}]
            else:
                user_content = list(content)
            prefix_anchor = "__HR_ENDLESS_SAMPLER_CACHED_PREFIX_TURN__"
            anchor = "__HR_ENDLESS_SAMPLER_CACHED_ASSISTANT_TURN__"
            messages = [
                # Keep the rendered template in the common user/model/user
                # alternation even for a template that rejects an assistant as
                # its first non-system message. Only the text after the second
                # synthetic turn is evaluated below.
                {"role": "user", "content": prefix_anchor},
                {"role": "assistant", "content": anchor},
                {"role": "user", "content": user_content},
            ]
            # A repair is always a text-only user turn. Re-evaluating media
            # would defeat the cached conversation and requires a separate
            # physical-media checkpoint protocol.
            media_items = self._get_media_items(messages)
            if media_items:
                raise ValueError("Gemma append-only repair accepts text content only")
            text = self._render_and_replace_media(messages=messages, media_items=media_items)
            if text.count(anchor) != 1:
                raise ValueError("Could not derive an append-only Gemma chat-template turn")
            # Keep the template's exact assistant closing sequence, then the
            # new user turn and model generation prompt.  The rendered prefix
            # contains a synthetic assistant turn which is not evaluated.
            suffix = text.split(anchor, 1)[1]
            if not suffix:
                raise ValueError("Gemma chat template produced an empty append suffix")

            chunks = None
            try:
                # The native tokenizer keeps Gemma's special-token rules.
                # ``llama.n_tokens`` is nonzero, so it omits BOS naturally.
                chunks = self._mtmd_tokenize(llama=llama, text=suffix, bitmaps=[])
                try:
                    for index in range(self._mtmd_cpp.mtmd_input_chunks_size(chunks)):
                        chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, index)
                        if chunk is None:
                            continue
                        chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                        if self._is_text_chunk(chunk_type):
                            n_tokens_out = ctypes.c_size_t()
                            tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                                chunk, ctypes.byref(n_tokens_out)
                            )
                            if tokens_ptr and n_tokens_out.value:
                                tokens = [tokens_ptr[token_index] for token_index in range(n_tokens_out.value)]
                                if llama.n_tokens + len(tokens) > llama.n_ctx():
                                    raise ValueError(
                                        f"Appended prompt exceeds n_ctx: {llama.n_tokens + len(tokens)} > {llama.n_ctx()}"
                                    )
                                llama.eval(tokens)
                        else:
                            raise ValueError("Gemma append-only repair unexpectedly rendered media")
                finally:
                    self._mtmd_cpp.mtmd_input_chunks_free(chunks)
                    chunks = None
            finally:
                if chunks is not None:
                    self._mtmd_cpp.mtmd_input_chunks_free(chunks)

            grammar = None
            if response_format is not None and response_format.get("type") == "json_object":
                grammar = _grammar_for_response_format(response_format)
            completion = llama.create_completion(
                prompt=llama.input_ids[:llama.n_tokens].tolist(),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                grammar=grammar,
                stop=[self.GEMMA4_EOS_TOKEN, self.GEMMA4_EOT_TOKEN, self.GEMMA4_STR_TOKEN],
                reasoning_budget=reasoning_budget,
                reasoning_start=reasoning_start,
                reasoning_end=reasoning_end,
                reasoning_budget_message=reasoning_budget_message,
                reasoning_start_in_prompt=reasoning_start_in_prompt,
            )
            return {
                "choices": [{"message": {"content": completion["choices"][0]["text"]}}],
            }

    Gemma4MTMDChatHandler.__name__ = "Gemma4MTMDChatHandler"
    return Gemma4MTMDChatHandler


def _load_runtime():
    try:
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_multimodal import Gemma4ChatHandler
        import llama_cpp.mtmd_cpp
    except ImportError as error:
        raise Gemma4DependencyError(
            "Gemma 4 continuity requires JamePeng llama-cpp-python==0.3.49 with CUDA support. "
            "Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        ) from error

    version = getattr(llama_cpp, "__version__", "unknown")
    if version.split("+", 1)[0] != GEMMA4_REQUIRED_VERSION:
        raise Gemma4DependencyError(
            f"Gemma 4 continuity requires JamePeng llama-cpp-python=={GEMMA4_REQUIRED_VERSION} with MTMD vision support; "
            f"found {version}. Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        )
    return Llama, _gemma4_mtmd_handler_type(Gemma4ChatHandler)


def _create_runtime_llm(
    Llama: Any,
    *,
    model_path: Path,
    handler: Any,
    debug: bool,
    gemma4_mtp: bool = False,
    seed: int = 0,
) -> Any:
    """Create either the original runtime or a target born as native MTP."""
    real_runtime = str(getattr(Llama, "__module__", "")).startswith("llama_cpp")
    llama_kwargs = {
        "chat_handler": handler,
        "n_gpu_layers": -1,
        "n_ctx": GEMMA4_CONTEXT_TOKENS,
        "n_batch": GEMMA4_BATCH_SIZE,
        "n_ubatch": GEMMA4_BATCH_SIZE,
        "flash_attn": True,
        # The old Python path left these at llama.cpp's F16 defaults. Q8_0 is
        # intentionally less aggressive than Q4_0, while halving K/V cache
        # storage and matching the user's working native server setup.
        "type_k": GEMMA4_KV_CACHE_Q8_0,
        "type_v": GEMMA4_KV_CACHE_Q8_0,
        # Native server uses its default (no --swa-full); make that explicit
        # instead of forcing the full-size SWA allocation seen in the worker
        # diagnostics.
        "swa_full": False,
        # The same per-render seed reaches both the ordinary decoder and the
        # native-MTP target constructor. Worker retries reuse the request, so
        # a MTP fallback starts from the identical sampling seed too.
        "seed": int(seed) & 0x7fffffff,
        # Sampler debug controls our captures, validation warnings, progress,
        # and MTP summary.  It must not enable llama.cpp's native trace stream:
        # native verbosity prints for every one-token MTP assistant decode and
        # every hybrid-state restore, serialising the speculative hot loop on
        # stderr and repeatedly disturbing CUDA-graph execution.
        "verbose": False,
    }
    if gemma4_mtp and real_runtime:
        # JamePeng's native MTP engine owns target verification, recurrent
        # checkpoints, and rollback. Do not layer the retired local adapter
        # on top of it: only one speculative owner may control the target KV.
        mtp_path = _ensure_mtp_model_file()
        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType

        llama_kwargs["speculative"] = SpecConfig(
            spec_type=SpeculativeType.DRAFT_MTP,
            draft_model_path=str(mtp_path),
            draft_n_max=4,
            draft_p_min=0.0,
            # Match llama.cpp's working ``ngl=999`` setup: MTP's small
            # assistant must be fully offloaded too. ``-1`` means the fork's
            # heuristic "auto" mode here, which can leave it CPU-bound.
            draft_n_gpu_layers="all",
            draft_type_k=GEMMA4_KV_CACHE_Q8_0,
            draft_type_v=GEMMA4_KV_CACHE_Q8_0,
            draft_backend_sampling=True,
        )
        llm = Llama(model_path=str(model_path), **llama_kwargs)
        print(
            "HR Endless Sampler Gemma 4 decoding mode: native draft-mtp "
            "(JamePeng SpecConfig, spec-draft-n-max=4; operation-local non-MTP retry enabled).",
            file=sys.stderr,
            flush=True,
        )
    else:
        llm = Llama(model_path=str(model_path), **llama_kwargs)
        if real_runtime:
            print(
                "HR Endless Sampler Gemma 4 decoding mode: original non-MTP.",
                file=sys.stderr,
                flush=True,
            )
    _install_worker_token_progress(llm)
    return llm


def _install_worker_token_progress(llm: Any) -> None:
    """Emit live output-token throughput for the parent progress display.

    Prompt evaluation and model loading are intentionally excluded. The rate
    begins with the first generated token, matching llama.cpp's decode-speed
    interpretation rather than blending prompt prefill into tokens/second.
    """
    original_generate = getattr(llm, "generate", None)
    if not callable(original_generate):
        # Unit-test runtimes may expose only create_chat_completion.
        return
    generation_number = 0

    def tracked_generate(*args: Any, **kwargs: Any):
        nonlocal generation_number
        generation_number += 1
        current_generation = generation_number
        iterator = original_generate(*args, **kwargs)
        generated = 0
        first_token_at = None
        last_emit_at = None
        last_emitted_count = 0
        send_value = None
        first_iteration = True
        live_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        _begin_live_gemma_generation(current_generation)
        try:
            while True:
                try:
                    token = next(iterator) if first_iteration else iterator.send(send_value)
                except StopIteration:
                    return
                first_iteration = False
                # llama.cpp yields token ids. Decode their original byte pieces
                # incrementally so UTF-8 characters remain intact in the live log.
                token_bytes = llm.detokenize([int(token)], special=True)
                if isinstance(token_bytes, bytes):
                    _append_live_gemma_text(live_decoder.decode(token_bytes, final=False))
                else:
                    _append_live_gemma_text(str(token_bytes))
                now = time.perf_counter()
                generated += 1
                if first_token_at is None:
                    first_token_at = now
                elapsed = now - first_token_at
                if generated >= 2 and (
                    last_emit_at is None or now - last_emit_at >= 0.5
                ):
                    rate = (generated - 1) / max(elapsed, 1e-9)
                    print(
                        _WORKER_PROGRESS_PREFIX
                        + json.dumps(
                            {
                                "generation": current_generation,
                                "tokens": generated,
                                "tokens_per_second": rate,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    last_emit_at = now
                    last_emitted_count = generated
                send_value = yield token
        finally:
            _append_live_gemma_text(live_decoder.decode(b"", final=True))
            _append_live_gemma_text("\n\n")
            if first_token_at is not None and generated >= 2 and generated != last_emitted_count:
                now = time.perf_counter()
                rate = (generated - 1) / max(now - first_token_at, 1e-9)
                print(
                    _WORKER_PROGRESS_PREFIX
                    + json.dumps(
                        {
                            "generation": current_generation,
                            "tokens": generated,
                            "tokens_per_second": rate,
                            "complete": True,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    llm.generate = tracked_generate


def _image_data_url(frame: torch.Tensor) -> str:
    image = frame.detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise Gemma4ObservationError(f"Gemma observation expected HWC RGB frames, got {tuple(image.shape)}")
    pixels = image[..., :3].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def _extract_json_object(content: str) -> tuple[dict[str, Any], str]:
    """Decode the response's outer JSON object, never a nested truncation.

    Gemma may prefix its answer with a short channel marker or Markdown fence,
    so the first opening brace need not be byte zero. When a completed Gemma
    thought channel exists, the visible answer begins after its final close;
    ignore illustrative JSON inside that thought. Once the visible root brace
    appears, reject an incomplete object rather than accepting a later nested
    ``last_seen_character_state`` entry as the whole response.
    """
    decoder = json.JSONDecoder()
    visible_content = content
    thought_end = content.rfind(GEMMA4_REASONING_END)
    if thought_end >= 0 and "{" in content[thought_end + len(GEMMA4_REASONING_END):]:
        visible_content = content[thought_end + len(GEMMA4_REASONING_END):]
    match = re.search(r"\{", visible_content)
    if match is not None:
        try:
            value, end = decoder.raw_decode(visible_content[match.start():])
        except json.JSONDecodeError as error:
            raise Gemma4ObservationError(
                "Gemma 4 returned an incomplete or malformed top-level JSON object"
            ) from error
        if isinstance(value, dict):
            return value, visible_content[match.start():match.start() + end]
    raise Gemma4ObservationError("Gemma 4 did not return a JSON object")


def _frame_timestamp(frame: int, fps: float) -> str:
    milliseconds = round(frame * 1000 / fps)
    minutes, remainder = divmod(milliseconds, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _shot_context(shots: Sequence[dict[str, Any]], fps: float, include_target: bool) -> str:
    blocks = []
    for shot in shots:
        start = int(shot["shot_start"])
        end = int(shot["shot_end"])
        lines = [
            f"Source Shot {int(shot['shot_number'])}: global frames {start}-{end - 1} inclusive "
            f"({_frame_timestamp(start, fps)}-{_frame_timestamp(end, fps)} on the full-video timeline)."
        ]
        if include_target:
            target_start = int(shot["target_start"])
            target_end = int(shot["target_end"])
            lines.append(
                f"This chunk must generate global frames {target_start}-{target_end - 1} of this shot."
            )
            required_marker = shot.get("required_marker")
            if required_marker:
                lines.append(
                    f"Required H3 marker using this shot's global label and the chunk-local clock: {required_marker}"
                )
            else:
                lines.append(
                    "This shot was already in progress before this physical chunk; begin it as unmarked continuation prose."
                )
        else:
            covered_start = int(shot["covered_start"])
            covered_end = int(shot["covered_end"])
            lines.append(
                f"The previous chunk contains global frames {covered_start}-{covered_end - 1} of this shot."
            )
        lines.append("Complete original source-shot description (authoritative intent, not a word-by-word output template):")
        lines.append(str(shot["source_body"]).strip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "none"


def _current_chunk_shot_timeline(shots: Sequence[dict[str, Any]], current: dict[str, Any]) -> str:
    """Summarize exact source-shot positions on the physical chunk timeline."""
    sampled_start = int(current["sampled_start"])
    lines = []
    for shot in shots:
        source_start = int(shot["shot_start"])
        target_start = int(shot["target_start"])
        target_end = int(shot["target_end"])
        source_local = source_start - sampled_start
        target_local_start = target_start - sampled_start
        target_local_end = target_end - sampled_start - 1
        if source_local < 0:
            source_position = (
                f"starts before this physical chunk at global frame {source_start} "
                f"(physical local frame {source_local})"
            )
        else:
            source_position = (
                f"starts at global frame {source_start} "
                f"(physical local frame {source_local})"
            )
        lines.append(
            f"- global [Shot {int(shot['shot_number'])}]: {source_position}; "
            f"this chunk must author its global frames {target_start}-{target_end - 1} "
            f"(physical local frames {target_local_start}-{target_local_end})."
        )
    return "\n".join(lines) if lines else "none"


def _slice_portion_name(shot: dict[str, Any]) -> str:
    """Describe how the retained chunk interval intersects one source shot."""
    has_prior = int(shot["target_start"]) > int(shot["shot_start"])
    has_later = int(shot["target_end"]) < int(shot["shot_end"])
    if has_prior and has_later:
        portion = "the continuing middle portion of"
    elif has_prior:
        portion = "the ending portion of"
    elif has_later:
        portion = "the opening portion of"
    else:
        portion = "all of"
    return f"{portion} global [Shot {int(shot['shot_number'])}]"


def _slice_portion_kind(shot: dict[str, Any]) -> str:
    has_prior = int(shot["target_start"]) > int(shot["shot_start"])
    has_later = int(shot["target_end"]) < int(shot["shot_end"])
    if has_prior and has_later:
        return "middle"
    if has_prior:
        return "ending"
    if has_later:
        return "opening"
    return "complete"


def _current_shot_timing_contract(shots: Sequence[dict[str, Any]], fps: float) -> str:
    """Expose source-relative duration and coverage so Gemma need not infer it."""
    lines = [
        "Before writing, allocate the source events across each complete source-shot duration below. "
        "Do not state an action's final outcome before this slice reaches the part of the shot where it belongs.",
    ]
    for shot in shots:
        shot_start = int(shot["shot_start"])
        shot_end = int(shot["shot_end"])
        target_start = int(shot["target_start"])
        target_end = int(shot["target_end"])
        shot_frames = shot_end - shot_start
        target_frames = target_end - target_start
        relative_start = target_start - shot_start
        relative_end = target_end - shot_start - 1
        start_percent = 100.0 * relative_start / shot_frames
        end_percent = 100.0 * (relative_end + 1) / shot_frames
        lines.append(
            f"- global [Shot {int(shot['shot_number'])}]: full source shot "
            f"global frames {shot_start}-{shot_end - 1} ({shot_frames} frames, {shot_frames / fps:.3f} s). "
            f"This chunk owns its {_slice_portion_kind(shot)} {target_frames} frames: "
            f"source-relative frames {relative_start}-{relative_end} ({start_percent:.1f}%-{end_percent:.1f}% of the shot)."
        )
    lines.append(
        "In timing_plan, state the concrete events to cover now and the concrete later events to defer for every global source shot. "
        "end_state must describe only the visible state reachable at this slice's final retained frame."
    )
    return "\n".join(lines)


def _joined_phrases(parts: Sequence[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _chunk_generation_request(shots: Sequence[dict[str, Any]], current: dict[str, Any]) -> str:
    """Build the front-loaded request that tells Gemma exactly what it must write."""
    sampled_start = int(current["sampled_start"])
    sampled_end = int(current["sampled_end"])
    output_start = int(current["output_start"])
    output_end = int(current["output_end"])
    physical_frames = sampled_end - sampled_start
    output_local_start = output_start - sampled_start
    output_local_end = output_end - sampled_start - 1
    portions = _joined_phrases([
        _slice_portion_name(shot)
        for shot in shots
    ])
    request = (
        f"Write one complete chunk-local detailed_description for {portions}. "
        f"All requested action belongs to physical global timeslice {sampled_start}-{sampled_end - 1} inclusive, "
        f"physical chunk-local timeslice 0-{physical_frames - 1} ({physical_frames} frames). "
        f"Author the retained output interval global frames {output_start}-{output_end - 1}, "
        f"physical local frames {output_local_start}-{output_local_end}; do not restage completed opening "
        f"conditioning frames outside that retained interval."
    )
    establishment = _first_frame_establishment(shots, current)
    if establishment:
        request += (
            " This is the first generated frame of the production, not a continuation. Immediately after the "
            "required opening [Shot N] marker, copy this authoritative source establishment exactly: "
            f"`{establishment}` Do not replace it with `The camera remains static in the established framing.`"
        )
    return request


def _first_frame_establishment(shots: Sequence[dict[str, Any]], current: dict[str, Any]) -> str:
    """Return an explicit source opening that must establish global frame zero."""
    if int(current["output_start"]) != 0 or not shots:
        return ""
    first = shots[0]
    if int(first["shot_start"]) != 0 or int(first["target_start"]) != 0:
        return ""
    body = str(first.get("source_body") or "").strip()
    opening = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()
    if not re.search(r"(?i)(?:exact\s+first\s+frame|first\s+frame.*<Picture\s+\d+>|<Picture\s+\d+>.*camera)", opening):
        return ""
    return opening


def _required_local_markers(shots: Sequence[dict[str, Any]]) -> str:
    """Render global shot labels with sampler-calculated local timecodes.

    The complete source prompt necessarily contains its original full-video
    timecodes. This copy-only block preserves every source shot number while
    making the sampler-calculated chunk-local clock visually distinct, so
    Gemma need not translate shot identity or infer which time H3 expects.
    """
    marked_shots = [shot for shot in shots if shot.get("required_marker")]
    lines = [
        "IMMUTABLE H3 SHOT MARKERS — GLOBAL SHOT LABELS, CHUNK-LOCAL TIMES",
        "Every [Shot N] keeps N from the original full-video prompt. Only each At timecode is recalculated on H3's physical chunk-local clock, whose zero is the first sampled frame.",
        "For detailed_description, write every quoted token exactly once and in this order; do not copy the leading hyphen or quotes.",
        "Original full-video timestamps elsewhere are planning context only and are forbidden as H3 markers unless identical to a quoted token below.",
    ]
    if marked_shots:
        lines.extend(f'- exact token: "{str(shot["required_marker"])}"' for shot in marked_shots)
    else:
        lines.append(
            "There is no H3 shot marker in this physical chunk: it begins inside an already established source shot. "
            "Begin detailed_description as plain continuation prose; do not add [Shot 1] or any other [Shot N] marker."
        )
    lines.extend(("", _global_marker_transition_map(shots)))
    return "\n".join(lines)


def _global_marker_transition_map(shots: Sequence[dict[str, Any]]) -> str:
    """Explain global shot identity and the sampler-owned local cut clock.

    Literal tokens alone are insufficient when a physical chunk begins in the
    middle of a source shot. Keep Python authoritative about the tokens while
    stating which global shot continues unmarked and which global shot each
    subsequent chunk-local timestamp introduces.
    """
    if not shots:
        return "SEMANTIC SHOT-TRANSITION MAP\n- This chunk contains no source shots."

    lines = ["SHOT-TRANSITION MAP — GLOBAL LABELS ON THE CHUNK-LOCAL CLOCK"]
    first = shots[0]
    first_global = int(first["shot_number"])
    first_marker = first.get("required_marker")
    if first_marker:
        lines.append(
            f"- Global Source Shot {first_global} begins at the physical chunk opening. Put "
            f"{str(first_marker)!r} immediately before its first action; its Shot number remains {first_global}."
        )
    else:
        lines.append(
            f"- This chunk time-slice begins inside global Source Shot {first_global}. Continue and/or finish that "
            "same global shot as plain unmarked prose. Do not restart it and do not "
            "put a [Shot] marker before its remaining action."
        )

    for shot_index, shot in enumerate(shots[1:], 1):
        marker = str(shot.get("required_marker") or "").strip()
        if not marker:
            continue
        previous_global = int(shots[shot_index - 1]["shot_number"])
        current_global = int(shot["shot_number"])
        match = _SHOT_MARKER.fullmatch(marker)
        if match and match.group(2) is not None:
            position = f"{match.group(2)}:{match.group(3)}.{match.group(4)}"
        else:
            position = "the supplied local position"
        lines.append(
            f"- The chunk time-slice encompasses the end of global Source Shot {previous_global}. Next, global "
            f"Source Shot {current_global} begins at chunk-local position {position}. Put the "
            f"exact token {marker!r} immediately before the first action of global Source Shot {current_global}, "
            f"never before remaining action from global Source Shot {previous_global}."
        )
    lines.append(
        "- Emit no other [Shot] marker. Preserve global shot numbers, but never copy their original full-video "
        "timecodes into this chunk-local detailed_description."
    )
    return "\n".join(lines)


def _marker_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return only errors that a focused shot-marker rewrite can repair."""
    return tuple(warning for warning in warnings if "marker" in warning.lower())


def _character_state_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return H3 retention defects that must never silently reach H3."""
    return tuple(
        warning for warning in warnings
        if "retention_analysis" in warning.lower()
    )


def _contract_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return contract findings that require a model-authored retry."""
    return tuple(
        warning for warning in warnings
        if (
            "marker" in warning.lower()
            or "mandatory coverage" in warning.lower()
            or "dialogue speaker form" in warning.lower()
            or "last-seen character state" in warning.lower()
            or "retention_analysis" in warning.lower()
            or "character continuity" in warning.lower()
            or "first-frame establishment" in warning.lower()
        )
    )


def _character_subject_mappings(request: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Read the immutable rendered name table supplied by preproduction."""
    table = str(request.get("character_name_table") or "")
    return tuple(
        (match.group(1).strip(), match.group(2).strip())
        for match in re.finditer(
            r"(?im)^\s*-\s*(.+?)\s*->\s*(<Subject\s+\d+>)\s*$",
            table,
        )
    )


def _last_seen_character_state_contract(request: dict[str, Any]) -> str:
    mappings = _character_subject_mappings(request)
    if not mappings:
        return (
            "No immutable named-character mappings exist. Return "
            '"last_seen_character_state": [].'
        )
    expected = "\n".join(f"- {name} -> {subject}" for name, subject in mappings)
    previous = request.get("previous_last_seen_character_state")
    previous_text = json.dumps(previous or [], ensure_ascii=False, indent=2)
    return (
        "Return exactly one entry for every immutable character below, in this order, and no other entries:\n"
        + expected
        + "\nEach entry must use this exact shape:\n"
        '{"character_name":"Tila","subject":"<Subject 2>",'
        '"last_seen_global_frame":208,"last_seen_source_shot":4,'
        '"environment":"inside the ancient temple",'
        '"pose_and_position":"mounted on the tiger, seated behind Heman",'
        '"state_and_action":"alert and leaning forward",'
        '"spatial_relationships":"behind Heman and on the tiger saddle"}\n'
        "Use null frame/shot values and the phrase 'not yet observed in generated video' in every text field "
        "until a character has actually appeared in attached rendered evidence. Update an entry only from the "
        "chronological stills attached to this request. If the character is absent from those stills, copy the "
        "previous entry exactly; never replace a known state with an inference from the current unrendered plan.\n"
        "Previous persistent table:\n"
        + previous_text
    )


def _validate_last_seen_character_state(value: dict[str, Any], request: dict[str, Any]) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    """Validate Gemma's persistent observed-state table without inventing it."""
    mappings = _character_subject_mappings(request)
    supplied = value.get("last_seen_character_state")
    if not mappings:
        if supplied in (None, []):
            return (), []
        return (), ["Gemma 4 last-seen character state must be an empty array when no character mappings exist"]
    if not isinstance(supplied, list):
        return (), ["Gemma 4 last-seen character state must be an array"]

    warnings: list[str] = []
    by_name: dict[str, dict[str, Any]] = {}
    for item in supplied:
        if not isinstance(item, dict):
            warnings.append("Gemma 4 last-seen character state entries must be objects")
            continue
        name = item.get("character_name")
        if not isinstance(name, str) or not name.strip():
            warnings.append("Gemma 4 last-seen character state entry has no character_name")
            continue
        key = name.strip().casefold()
        if key in by_name:
            warnings.append(f"Gemma 4 last-seen character state duplicates {name.strip()!r}")
            continue
        by_name[key] = item

    expected_keys = {name.casefold() for name, _subject in mappings}
    for key, item in by_name.items():
        if key not in expected_keys:
            warnings.append(
                f"Gemma 4 last-seen character state includes unknown character {item.get('character_name')!r}"
            )

    normalized: list[dict[str, Any]] = []
    text_fields = ("environment", "pose_and_position", "state_and_action", "spatial_relationships")
    for name, subject in mappings:
        item = by_name.get(name.casefold())
        if item is None:
            warnings.append(f"Gemma 4 last-seen character state omits {name!r}")
            continue
        if str(item.get("subject") or "").strip().casefold() != subject.casefold():
            warnings.append(
                f"Gemma 4 last-seen character state maps {name!r} to {item.get('subject')!r}; expected {subject}"
            )
        frames: dict[str, int | None] = {}
        for field in ("last_seen_global_frame", "last_seen_source_shot"):
            field_value = item.get(field)
            if field_value is not None and (isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0):
                warnings.append(f"Gemma 4 last-seen character state {name!r} has invalid {field}")
                field_value = None
            frames[field] = field_value
        texts: dict[str, str] = {}
        for field in text_fields:
            field_value = item.get(field)
            if not isinstance(field_value, str) or not field_value.strip():
                warnings.append(f"Gemma 4 last-seen character state {name!r} has no usable {field}")
                field_value = "unknown"
            texts[field] = field_value.strip()
        normalized.append({
            "character_name": name,
            "subject": subject,
            **frames,
            **texts,
        })
    return tuple(normalized), warnings


def _normalized_prompt_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _request_voice_profiles(request: dict[str, Any]) -> dict[str, str]:
    """Read immutable S# delivery phrases from the raw production bible."""
    raw = request.get("production_bible")
    if not isinstance(raw, str) or not raw.strip().startswith("{"):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    result: dict[str, str] = {}
    for item in value.get("speaker_voice_profiles", ()) if isinstance(value, dict) else ():
        if not isinstance(item, dict):
            continue
        speaker_id = str(item.get("speaker_id") or "").strip().upper().strip("()")
        profile = item.get("voice_profile")
        if re.fullmatch(r"S\d+", speaker_id) and isinstance(profile, str) and profile.strip():
            result[speaker_id] = profile.strip()
    return result


def _mandatory_coverage_warnings(value: dict[str, Any], request: dict[str, Any], description: str) -> list[str]:
    """Check Gemma's own attestation of the current planned beat intersections.

    We intentionally validate only the presence/ownership contract. We do not
    synthesize or substitute H3 prose. A failed contract gets one complete
    Gemma rewrite in the same way a wrong local cut marker does.
    """
    mandatory = request.get("mandatory_coverage", ())
    if not mandatory:
        return []
    warnings: list[str] = []
    required = {
        str(item.get("id")): item
        for item in mandatory
        if isinstance(item, dict) and item.get("id")
    }
    if not required:
        return []
    voice_profiles = _request_voice_profiles(request)
    supplied = value.get("coverage")
    if not isinstance(supplied, list):
        return ["Gemma 4 mandatory coverage requires a coverage JSON array"]

    records: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(supplied, 1):
        if not isinstance(item, dict):
            warnings.append(f"Gemma 4 mandatory coverage entry {index} is not an object")
            continue
        beat_id = item.get("id")
        if not isinstance(beat_id, str) or not beat_id:
            warnings.append(f"Gemma 4 mandatory coverage entry {index} has no usable id")
            continue
        if beat_id in records:
            warnings.append(f"Gemma 4 mandatory coverage repeats beat {beat_id}")
            continue
        records[beat_id] = item

    for beat_id, expected in required.items():
        record = records.get(beat_id)
        if record is None:
            warnings.append(f"Gemma 4 mandatory coverage omits current beat {beat_id}")
            continue
        status = record.get("status")
        if status not in {"begins", "continues", "completes"}:
            warnings.append(
                f"Gemma 4 mandatory coverage {beat_id} has status {status!r}; "
                "it must be begins, continues, or completes, never deferred"
            )
        evidence = record.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            warnings.append(f"Gemma 4 mandatory coverage {beat_id} has no detailed_description evidence")
        elif _normalized_prompt_text(evidence) not in _normalized_prompt_text(description):
            warnings.append(
                f"Gemma 4 mandatory coverage {beat_id} evidence is not present in detailed_description"
            )

        if expected.get("kind") == "overlay" and expected.get("overlay_type") == "dialogue":
            expected_dialogues = [_dialogue_text(item) for item in _DIALOGUE.findall(str(expected.get("action", "")))]
            actual_dialogues = [_dialogue_text(item) for item in _DIALOGUE.findall(description)]
            for dialogue in expected_dialogues:
                if dialogue and dialogue not in actual_dialogues:
                    warnings.append(
                        f"Gemma 4 mandatory coverage {beat_id} requires its exact assigned dialogue segment in detailed_description"
                    )
                    continue
                action = str(expected.get("action", ""))
                source_match = next(
                    (match for match in _DIALOGUE.finditer(action) if _dialogue_text(match.group(1)) == dialogue),
                    None,
                )
                speaker_id = "" if source_match is None else _dialogue_speaker_id(action, source_match.start())
                voice_profile = voice_profiles.get(speaker_id)
                for match in _DIALOGUE.finditer(description):
                    if _dialogue_text(match.group(1)) != dialogue:
                        continue
                    prefix = description[:match.start()]
                    previous_dialogue_end = prefix.rfind("</d>")
                    local_prefix = prefix[previous_dialogue_end + 4:] if previous_dialogue_end >= 0 else prefix
                    normalized_prefix = _normalized_prompt_text(local_prefix)
                    if voice_profile and _normalized_prompt_text(voice_profile) not in normalized_prefix:
                        warnings.append(
                            f"Gemma 4 mandatory coverage {beat_id} must repeat immutable {speaker_id} voice profile "
                            f"{voice_profile!r} outside its <d> block"
                        )
                    if expected.get("dialogue_continuation") and "continues uninterrupted" not in normalized_prefix:
                        warnings.append(
                            f"Gemma 4 mandatory coverage {beat_id} must state that {speaker_id or 'the speaker'} "
                            "continues uninterrupted"
                        )
                    break

    for beat_id in records.keys() - required.keys():
        warnings.append(f"Gemma 4 mandatory coverage includes unknown beat {beat_id}")
    return warnings


def _chunk_contract_correction_request(request: dict[str, Any], warnings: Sequence[str]) -> str:
    """Ask Gemma itself for one complete schema/contract correction, never a patch."""
    shots = request["target_shots"]
    mandatory = request.get("mandatory_coverage", ())
    required_coverage = "\n".join(
        f"- {item['id']} ({item['kind']}{'/' + str(item['overlay_type']) if item.get('overlay_type') else ''}): "
        f"source-relative frames {item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}; "
        f"{item['action']}"
        for item in mandatory
        if isinstance(item, dict) and item.get("id")
    ) or "- No current-slice coverage entries are required."
    current_dialogue = {
        _dialogue_text(dialogue)
        for source in (
            *(str(item.get("action", "")) for item in mandatory if isinstance(item, dict)),
            *(str(shot.get("source_body", "")) for shot in shots if isinstance(shot, dict)),
        )
        for dialogue in _DIALOGUE.findall(source)
        if _dialogue_text(dialogue)
    }
    speaker_forms = "\n".join(
        f"- {subject} ({speaker_id}) must introduce <d>{dialogue}</d>"
        for dialogue, subject, speaker_id in _mapped_dialogue_speaker_requirements(request)
        if dialogue in current_dialogue
    ) or "- No mapped dialogue speaker form is required in this slice."
    voice_profiles = _request_voice_profiles(request)
    voice_profile_text = "\n".join(
        f"- {speaker_id}: repeat verbatim outside <d>: {profile}"
        for speaker_id, profile in sorted(voice_profiles.items())
    ) or "- No immutable voice profile is available."
    return (
        "CHUNK CONTRACT CORRECTION REQUIRED\n"
        "Your immediately preceding JSON was missing a required field or violated the H3 shot-marker, persistent-state, "
        "dialogue, or mandatory current-slice coverage contract. "
        "Return one complete replacement JSON object "
        "with all eight required fields, not an explanation and not a textual patch. Keep the same current-frame-slice "
        "creative intent and continuity reasoning, but rewrite detailed_description and coverage so every current beat "
        "is explicitly started or continued now. A current beat may never be marked deferred. For dialogue coverage, "
        "include only the exact assigned dialogue-segment <d>...</d> line in detailed_description now. Never expand it "
        "back to the complete source utterance or repeat words assigned to a previous chunk. A mapped visual speaker "
        "must use the immediate "
        "official form <Subject N> (Sx) before that line, not Name (<Subject N>) (Sx). Use evidence copied exactly "
        "from your rewritten detailed_description. For a continuation segment, state that the speaker `continues "
        "uninterrupted`. Never add an At timecode to dialogue.\n\nMapped dialogue speaker form:\n"
        + speaker_forms
        + "\n\nImmutable speaker voice profiles:\n"
        + voice_profile_text
        + "\n\nH3 marker contract (global shot labels, chunk-local timecodes):\n"
        f"{_required_local_markers(shots)}\n\n"
        "Persistent last-seen character state contract:\n"
        f"{_last_seen_character_state_contract(request)}\n\n"
        "H3-facing retention_analysis contract:\n"
        f"{request.get('planned_character_continuity') or 'No planned characters apply to this slice.'}\n"
        "Return a short retention_analysis value covering every participating mapped character's actual entry "
        "location and physical state. Do not mention Gemma, last-seen memory, tables, plans, or future transitions. "
        "Also include every participating character in detailed_description and describe how current action moves "
        "from the latest rendered state toward that character's expected slice-exit state.\n\n"
        "Mandatory current-slice coverage:\n"
        + required_coverage
        + "\n\nThe original full-video timestamps must not be used as markers inside this physical chunk. Preserve every "
        "global shot number exactly, use only the supplied chunk-local timecodes, and do not add, remove, or move cuts.\n\n"
        "Detected validation errors in the preceding JSON:\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )


def _chunk_contract_followup_request(warnings: Sequence[str]) -> str:
    """Compact later repair turn; the complete contract is already in KV."""
    return (
        "CHUNK JSON REPAIR STILL REQUIRED\n"
        "The complete chunk contract is in the immediately preceding user turn. Do not repeat or explain it. "
        "Return one complete JSON object now with exactly these eight fields: confidence, analysis, timing_plan, "
        "end_state, retention_analysis, last_seen_character_state, coverage, and detailed_description. "
        "retention_analysis must be a short H3-facing string covering every participating character. "
        "detailed_description must be "
        "a non-empty JSON string containing the complete H3-facing prompt; coverage and "
        "last_seen_character_state must be arrays. Correct these remaining errors:\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )


def _marker_contract_followup_request(
    request: dict[str, Any], warnings: Sequence[str]
) -> str:
    """Reinforce a still-invalid marker mapping without changing Gemma's prose."""
    return (
        "H3 SHOT-MARKER REPAIR STILL REQUIRED\n"
        "Python rejected the marker structure in your immediately preceding JSON. Return one complete replacement "
        "JSON object with the same eight fields. Preserve the intended actions, dialogue, continuity, retention, coverage, and "
        "observed character state, but rewrite detailed_description so its global [Shot N] labels and chunk-local "
        "At timecodes obey this exact semantic "
        "transition map:\n\n"
        f"{_required_local_markers(request['target_shots'])}\n\n"
        "The prose before the first supplied marker belongs only to the already-continuing global source shot. "
        "The exact marker introduces the next global source shot shown in the map; it must not restart or relabel "
        "the preceding shot. Delete every invented marker and every marker carrying an original full-video timecode.\n\n"
        "Detected marker errors in the preceding JSON:\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )


def _dialogue_text(value: str) -> str:
    return re.sub(r"\s+", " ", _DIALOGUE_CONTROL.sub("", value)).strip()


def _mapped_dialogue_speaker_requirements(request: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """Find source dialogue whose named speaker has an explicit Subject mapping.

    MiniMax full-reference prompts distinguish a visible referenced subject from
    its stable speaker ID with the immediate form ``<Subject N> (Sx)``. The
    human-name binding rule is intentionally not applied inside that speaker
    form: ``Name (<Subject N>) (Sx)`` obscures the official token pattern.
    """
    table = str(request.get("character_name_table") or "")
    mappings = [
        (match.group(1).strip(), match.group(2).strip())
        for match in re.finditer(
            r"(?mi)^\s*-\s*(.+?)\s*->\s*(<Subject\s+\d+>)\s*$",
            table,
        )
    ]
    if not mappings:
        return ()
    source = str(request.get("original_prompt") or "")
    requirements: list[tuple[str, str, str]] = []
    for dialogue_match in _DIALOGUE.finditer(source):
        dialogue = _dialogue_text(dialogue_match.group(1))
        if not dialogue:
            continue
        prefix = source[max(0, dialogue_match.start() - 320):dialogue_match.start()]
        for name, subject in mappings:
            speaker_match = re.search(
                rf"(?is)\b{re.escape(name)}\b\s*\(\s*(S\d+)\s*\)[^<]*$",
                prefix,
            )
            if speaker_match is not None:
                requirement = (dialogue, subject, speaker_match.group(1))
                if requirement not in requirements:
                    requirements.append(requirement)
                break
    # A later physical chunk receives only its immutable dialogue fragment.
    # Carry the source speaker binding onto that fragment so the official H3
    # speaker form remains validated without requiring the full utterance.
    for item in request.get("mandatory_coverage", ()):
        if not isinstance(item, dict) or item.get("overlay_type") != "dialogue":
            continue
        segment_matches = _DIALOGUE.findall(str(item.get("action") or ""))
        if len(segment_matches) != 1:
            continue
        segment_dialogue = _dialogue_text(segment_matches[0])
        segment_language, segment_spoken = _dialogue_language_and_spoken_text(segment_matches[0])
        for full_dialogue, subject, speaker_id in tuple(requirements):
            full_language, full_spoken = _dialogue_language_and_spoken_text(full_dialogue)
            if segment_language == full_language and segment_spoken and segment_spoken in full_spoken:
                requirement = (segment_dialogue, subject, speaker_id)
                if requirement not in requirements:
                    requirements.append(requirement)
                break
    return tuple(requirements)


def _dialogue_is_source_fragment(output_dialogue: str, source_dialogue: str) -> bool:
    """Return whether one chunk-owned dialogue fragment is exact source text."""
    output_language, output_spoken = _dialogue_language_and_spoken_text(output_dialogue)
    source_language, source_spoken = _dialogue_language_and_spoken_text(source_dialogue)
    return (
        bool(output_spoken)
        and output_language == source_language
        and output_spoken in source_spoken
    )


def _dialogue_speaker_form_warnings(request: dict[str, Any], description: str) -> list[str]:
    """Require MiniMax's unambiguous Subject-plus-speaker-ID form when known."""
    warnings: list[str] = []
    for dialogue, subject, speaker_id in _mapped_dialogue_speaker_requirements(request):
        matching_dialogue = [
            match
            for match in _DIALOGUE.finditer(description)
            if _dialogue_text(match.group(1)) == dialogue
        ]
        # A missing dialogue line is reported by the mandatory overlay check;
        # this check only rejects a malformed speaker prefix for a line that
        # Gemma actually did include.
        if not matching_dialogue:
            continue
        subject_pattern = re.escape(subject)
        speaker_pattern = re.escape(speaker_id)
        valid = False
        for match in matching_dialogue:
            prefix = description[max(0, match.start() - 360):match.start()]
            # ``Heman (<Subject 1>) (S1)`` superficially contains the desired
            # token sequence but makes Subject 1 an appositive of the name,
            # rather than MiniMax's actual speaker token.  Reject that exact
            # nesting before accepting a genuine immediate speaker form.
            nested_name_form = re.compile(
                rf"(?is)\b[^\s<>()]+\s*\(\s*{subject_pattern}\s*\)\s*"
                rf"\(\s*{speaker_pattern}\s*\)\s*(?:says|shouts|replies|asks|whispers)?\s*[,;:]?\s*$"
            )
            if nested_name_form.search(prefix):
                continue
            if re.search(
                rf"(?is){subject_pattern}\s*\(\s*{speaker_pattern}\s*\)[^<]*$",
                prefix,
            ):
                valid = True
                break
        if not valid:
            warnings.append(
                f"Gemma 4 dialogue speaker form for {dialogue!r} must use "
                f"{subject} ({speaker_id}) immediately before its <d> line"
            )
    return warnings


def _validate_chunk_prompt(value: dict[str, Any], request: dict[str, Any], raw_json: str,
                           system_prompt: str = "", observation_prompt: str = "") -> GemmaChunkPrompt:
    """Return Gemma's usable description and report, never rewrite, structural defects.

    Marker, dialogue, and section-shape checks are diagnostics for prompt
    iteration. They must not replace Gemma's text with the sampler's static
    planner output. H3 receives the returned description except that legacy
    Gemma-only ``[end state]`` metadata is removed before encoding. Only the
    absence of a usable description remains a hard failure.
    """
    warnings: list[str] = []
    confidence = value.get("confidence", "unknown")
    if confidence not in ("high", "medium", "low", "unknown"):
        warnings.append(f"Gemma 4 returned unsupported confidence {confidence!r}; recorded as 'unknown'")
        confidence = "unknown"
    analysis = value.get("analysis", "")
    if not isinstance(analysis, str):
        analysis = ""
    timing_plan = value.get("timing_plan", "")
    if not isinstance(timing_plan, str) or not timing_plan.strip():
        warnings.append("Gemma 4 response has no usable timing_plan string")
        timing_plan = ""
    else:
        timing_plan = timing_plan.strip()
    end_state = value.get("end_state", "")
    if not isinstance(end_state, str):
        warnings.append("Gemma 4 response has no usable end_state string")
        end_state = ""
    else:
        end_state = end_state.strip()
    retention_analysis = value.get("retention_analysis", "")
    if not isinstance(retention_analysis, str):
        warnings.append("Gemma 4 retention_analysis must be a string")
        retention_analysis = ""
    else:
        retention_analysis = retention_analysis.strip()
    expected_characters = tuple(
        item for item in request.get("current_character_subjects", ())
        if isinstance(item, dict)
    )
    if expected_characters and not retention_analysis:
        warnings.append("Gemma 4 retention_analysis is empty despite planned characters in this slice")
    if retention_analysis:
        if len(retention_analysis) > 1600:
            warnings.append("Gemma 4 retention_analysis is too long; keep it short and physical")
        if re.search(
            r"(?i)\b(?:Gemma|last[- ]seen|continuity state relevant|bookkeeping|preproduction table)\b",
            retention_analysis,
        ):
            warnings.append("Gemma 4 retention_analysis exposes internal Gemma/bookkeeping language to H3")
        if re.search(r"(?i)\bretention_analysis\s*:", retention_analysis):
            warnings.append("Gemma 4 retention_analysis must contain only its value, not the field label")
        for character in expected_characters:
            name = str(character.get("character_name") or "").strip()
            subject = str(character.get("subject") or "").strip()
            if subject and re.search(re.escape(subject), retention_analysis, re.IGNORECASE) is None:
                warnings.append(
                    f"Gemma 4 retention_analysis omits planned character {name or subject} ({subject})"
                )
    last_seen_character_state, state_warnings = _validate_last_seen_character_state(value, request)
    warnings.extend(state_warnings)
    description = value.get("detailed_description")
    if not isinstance(description, str):
        raise Gemma4ObservationError("Gemma 4 response has no detailed_description string")
    description = description.strip()
    if not description:
        raise Gemma4ObservationError("Gemma 4 returned an empty detailed_description")
    legacy_end_state = _LEGACY_END_STATE.search(description)
    if legacy_end_state is not None:
        legacy_value = legacy_end_state.group(1).strip()
        description = description[:legacy_end_state.start()].rstrip()
        if not end_state and legacy_value:
            end_state = legacy_value
            warnings.append(
                "Gemma 4 put [end state] inside detailed_description; extracted it into Gemma-only end_state"
            )
        else:
            warnings.append(
                "Gemma 4 put [end state] inside detailed_description; removed it because H3 receives only detailed_description"
            )
    if not description:
        raise Gemma4ObservationError("Gemma 4 detailed_description contains only Gemma-only end-state metadata")
    for character in expected_characters:
        name = str(character.get("character_name") or "").strip()
        subject = str(character.get("subject") or "").strip()
        if subject and re.search(re.escape(subject), description, re.IGNORECASE) is None:
            warnings.append(
                f"Gemma 4 character continuity detailed_description omits planned character "
                f"{name or subject} ({subject}) and cannot describe that character's current-to-expected transition"
            )
    if not end_state:
        warnings.append("Gemma 4 response has no usable end_state after legacy extraction")
    if len(description) > 12000:
        warnings.append("Gemma 4 detailed_description is unexpectedly long")
    if re.search(r"\b(?:detailed_description|overall_soundscape|non_diegetic_music|subject_definitions|summary|retention_analysis)\s*:", description, re.IGNORECASE):
        warnings.append("Gemma 4 returned a structured field label inside detailed_description")

    establishment = _first_frame_establishment(request["target_shots"], request["current_chunk"])
    if establishment and establishment.lower() not in description.lower():
        warnings.append(
            "Gemma 4 first-frame establishment must copy the authoritative source opening exactly after the "
            f"opening shot marker: {establishment!r}; do not substitute continuation wording about an already "
            "established framing"
        )

    expected = [shot for shot in request["target_shots"] if shot.get("required_marker")]
    markers = list(_SHOT_MARKER.finditer(description))
    actual_marker_sequence = [marker.group(0) for marker in markers]
    required_marker_sequence = [str(shot["required_marker"]) for shot in expected]
    if len(markers) != len(expected):
        warnings.append(
            f"Gemma 4 returned {len(markers)} shot markers; this chunk requires {len(expected)}"
        )
    if [item.lower() for item in actual_marker_sequence] != [item.lower() for item in required_marker_sequence]:
        warnings.append(
            f"Gemma 4 shot marker sequence is {actual_marker_sequence!r}; required exact sequence is "
            f"{required_marker_sequence!r}"
        )
    for marker, shot in zip(markers, expected):
        actual_marker = marker.group(0)
        required_marker = str(shot["required_marker"])
        if actual_marker.lower() != required_marker.lower():
            warnings.append(
                f"Gemma 4 returned marker {actual_marker!r}; required marker is {required_marker!r}"
            )

    source_dialogue = [_dialogue_text(item) for item in _DIALOGUE.findall(str(request["original_prompt"]))]
    for output_dialogue in _DIALOGUE.findall(description):
        text = _dialogue_text(output_dialogue)
        if not text or not any(_dialogue_is_source_fragment(text, source) for source in source_dialogue):
            warnings.append("Gemma 4 modified or invented dialogue instead of preserving source words")
    warnings.extend(_dialogue_speaker_form_warnings(request, description))
    warnings.extend(_mandatory_coverage_warnings(value, request, description))

    return GemmaChunkPrompt(
        confidence=confidence,
        analysis=analysis.strip(),
        detailed_description=description,
        raw_json=raw_json,
        timing_plan=timing_plan,
        end_state=end_state,
        retention_analysis=retention_analysis,
        last_seen_character_state=last_seen_character_state,
        system_prompt=system_prompt,
        observation_prompt=observation_prompt,
        validation_warnings=tuple(dict.fromkeys(warnings)),
    )


def _chunk_prompt_payload(result: GemmaChunkPrompt) -> dict[str, Any]:
    return {
        "confidence": result.confidence,
        "analysis": result.analysis,
        "detailed_description": result.detailed_description,
        "raw_json": result.raw_json,
        "timing_plan": result.timing_plan,
        "end_state": result.end_state,
        "retention_analysis": result.retention_analysis,
        "last_seen_character_state": list(result.last_seen_character_state),
        "system_prompt": result.system_prompt,
        "observation_prompt": result.observation_prompt,
        "validation_warnings": list(result.validation_warnings),
        "attempts": [
            {
                "kind": attempt.kind,
                "raw_json": attempt.raw_json,
                "validation_warnings": list(attempt.validation_warnings),
                "correction_prompt": attempt.correction_prompt,
            }
            for attempt in result.attempts
        ],
    }


def _chunk_prompt_from_payload(value: dict[str, Any]) -> GemmaChunkPrompt:
    return GemmaChunkPrompt(
        confidence=str(value["confidence"]),
        analysis=str(value["analysis"]),
        detailed_description=str(value["detailed_description"]),
        raw_json=str(value["raw_json"]),
        timing_plan=str(value.get("timing_plan", "")),
        end_state=str(value.get("end_state", "")),
        retention_analysis=str(value.get("retention_analysis", "")),
        last_seen_character_state=tuple(
            dict(item) for item in value.get("last_seen_character_state", ()) if isinstance(item, dict)
        ),
        system_prompt=str(value.get("system_prompt", "")),
        observation_prompt=str(value.get("observation_prompt", "")),
        validation_warnings=tuple(str(item) for item in value.get("validation_warnings", ())),
        attempts=tuple(
            GemmaPromptAttempt(
                kind=str(item.get("kind", "unknown")),
                raw_json=str(item.get("raw_json", "")),
                validation_warnings=tuple(str(warning) for warning in item.get("validation_warnings", ())),
                correction_prompt=str(item.get("correction_prompt", "")),
            )
            for item in value.get("attempts", ())
            if isinstance(item, dict)
        ),
    )


def _timing_plan_payload(result: GemmaShotTimingPlan) -> dict[str, Any]:
    return {
        "confidence": result.confidence,
        "analysis": result.analysis,
        "character_name_table": [
            {
                "character_name": entry.character_name,
                "subject": entry.subject,
            }
            for entry in result.character_name_table
        ],
        "shots": [
            {
                "source_shot": shot.source_shot,
                "shot_start_frame": shot.shot_start_frame,
                "shot_end_frame": shot.shot_end_frame,
                "light_change": shot.light_change,
                "visual_beats": [
                    {
                        "start_frame": beat.start_frame,
                        "end_frame": beat.end_frame,
                        "action": beat.action,
                    }
                    for beat in shot.visual_beats
                ],
                "overlays": [
                    {
                        "start_frame": overlay.start_frame,
                        "end_frame": overlay.end_frame,
                        "type": overlay.overlay_type,
                        "content": overlay.content,
                        "continues_source_dialogue": overlay.continues_source_dialogue,
                        "dialogue_segments": [
                            {
                                "start_frame": segment.start_frame,
                                "end_frame": segment.end_frame,
                                "content": segment.content,
                            }
                            for segment in overlay.dialogue_segments
                        ],
                    }
                    for overlay in shot.overlays
                ],
                "continuity_slices": [
                    {
                        "start_frame": continuity.start_frame,
                        "end_frame": continuity.end_frame,
                        "characters": [
                            {
                                "character_name": character.character_name,
                                "subject": character.subject,
                                "entry_state": character.entry_state,
                                "expected_exit_state": character.expected_exit_state,
                            }
                            for character in continuity.characters
                        ],
                    }
                    for continuity in shot.continuity_slices
                ],
            }
            for shot in result.shots
        ],
        "raw_json": result.raw_json,
        "production_bible_json": result.production_bible_json,
        "shot_planning_prompts": list(result.shot_planning_prompts),
        "system_prompt": result.system_prompt,
        "planning_prompt": result.planning_prompt,
        "validation_warnings": list(result.validation_warnings),
        "attempts": [
            {
                "kind": attempt.kind,
                "raw_json": attempt.raw_json,
                "validation_warnings": list(attempt.validation_warnings),
                "correction_prompt": attempt.correction_prompt,
            }
            for attempt in result.attempts
        ],
    }


def _timing_plan_from_payload(value: dict[str, Any]) -> GemmaShotTimingPlan:
    character_name_table = tuple(
        GemmaCharacterSubject(
            character_name=str(item["character_name"]),
            subject=str(item["subject"]),
        )
        for item in value.get("character_name_table", ())
        if isinstance(item, dict)
    )
    shots: list[GemmaShotTimingShot] = []
    for item in value.get("shots", ()):
        if not isinstance(item, dict):
            continue
        visual_beats = tuple(
            GemmaShotTimingBeat(
                start_frame=int(beat["start_frame"]),
                end_frame=int(beat["end_frame"]),
                action=str(beat["action"]),
            )
            for beat in item.get("visual_beats", ())
            if isinstance(beat, dict)
        )
        overlays = tuple(
            GemmaShotTimingOverlay(
                start_frame=int(overlay["start_frame"]),
                end_frame=int(overlay["end_frame"]),
                overlay_type=str(overlay["type"]),
                content=str(overlay["content"]),
                dialogue_segments=tuple(
                    GemmaDialogueSegment(
                        start_frame=int(segment["start_frame"]),
                        end_frame=int(segment["end_frame"]),
                        content=str(segment["content"]),
                    )
                    for segment in overlay.get("dialogue_segments", ())
                    if isinstance(segment, dict)
                ),
                continues_source_dialogue=bool(overlay.get("continues_source_dialogue", False)),
            )
            for overlay in item.get("overlays", ())
            if isinstance(overlay, dict)
        )
        continuity_slices = tuple(
            GemmaShotContinuitySlice(
                start_frame=int(continuity["start_frame"]),
                end_frame=int(continuity["end_frame"]),
                characters=tuple(
                    GemmaCharacterContinuity(
                        character_name=str(character["character_name"]),
                        subject=str(character["subject"]),
                        entry_state=str(character["entry_state"]),
                        expected_exit_state=str(character["expected_exit_state"]),
                    )
                    for character in continuity.get("characters", ())
                    if isinstance(character, dict)
                ),
            )
            for continuity in item.get("continuity_slices", ())
            if isinstance(continuity, dict)
        )
        shots.append(
            GemmaShotTimingShot(
                source_shot=int(item["source_shot"]),
                shot_start_frame=int(item["shot_start_frame"]),
                shot_end_frame=int(item["shot_end_frame"]),
                visual_beats=visual_beats,
                overlays=overlays,
                continuity_slices=continuity_slices,
                # Old interrupted-render checkpoints have no lighting
                # decision. Treat them conservatively: do not alter their
                # completed output until Gemma rebuilds the timing plan.
                light_change=bool(item.get("light_change", True)),
            )
        )
    return GemmaShotTimingPlan(
        confidence=str(value.get("confidence", "unknown")),
        analysis=str(value.get("analysis", "")),
        shots=tuple(shots),
        character_name_table=character_name_table,
        raw_json=str(value.get("raw_json", "")),
        production_bible_json=str(value.get("production_bible_json", "")),
        shot_planning_prompts=tuple(str(item) for item in value.get("shot_planning_prompts", ())),
        system_prompt=str(value.get("system_prompt", "")),
        planning_prompt=str(value.get("planning_prompt", "")),
        validation_warnings=tuple(str(item) for item in value.get("validation_warnings", ())),
        attempts=tuple(
            GemmaPromptAttempt(
                kind=str(item.get("kind", "unknown")),
                raw_json=str(item.get("raw_json", "")),
                validation_warnings=tuple(str(warning) for warning in item.get("validation_warnings", ())),
                correction_prompt=str(item.get("correction_prompt", "")),
            )
            for item in value.get("attempts", ())
            if isinstance(item, dict)
        ),
    )


def _timing_plan_validation_error(message: str, raw_json: str) -> Gemma4ObservationError:
    return Gemma4ObservationError(f"Gemma 4 shot timing plan is invalid: {message}", raw_json=raw_json)


def _validate_character_name_table(value: dict[str, Any], raw_json: str) -> tuple[GemmaCharacterSubject, ...]:
    """Accept only explicit, stable name-to-existing-subject declarations."""
    raw_table = value.get("character_name_table")
    if not isinstance(raw_table, list):
        raise _timing_plan_validation_error("response field 'character_name_table' must be an array", raw_json)
    table: list[GemmaCharacterSubject] = []
    seen_names: set[str] = set()
    for item in raw_table:
        if not isinstance(item, dict):
            raise _timing_plan_validation_error("every character_name_table entry must be an object", raw_json)
        name = item.get("character_name")
        subject = item.get("subject")
        if not isinstance(name, str) or not name.strip():
            raise _timing_plan_validation_error("character_name_table entries need a non-empty character_name", raw_json)
        if not isinstance(subject, str) or not _SUBJECT_REFERENCE.fullmatch(subject.strip()):
            raise _timing_plan_validation_error(
                "character_name_table entries need an existing '<Subject N>' subject label", raw_json
            )
        normalized_name = name.strip().casefold()
        if normalized_name in seen_names:
            raise _timing_plan_validation_error(
                f"character_name_table repeats character name {name.strip()!r}", raw_json
            )
        seen_names.add(normalized_name)
        table.append(GemmaCharacterSubject(name.strip(), subject.strip()))
    return tuple(table)


def _declared_character_subject_mappings(prompt: str) -> tuple[GemmaCharacterSubject, ...]:
    """Extract unambiguous ``Name is <Subject N>`` declarations from source text."""
    result: list[GemmaCharacterSubject] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"(?mi)^\s*([A-Za-z][A-Za-z0-9_' -]{0,80}?)\s+is\s+(<Subject\s+\d+>)\s*\.?\s*$",
        prompt,
    ):
        name = match.group(1).strip()
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(GemmaCharacterSubject(name, match.group(2).strip()))
    return tuple(result)


def _declared_speaker_ids(prompt: str) -> tuple[str, ...]:
    """Return explicit stable speaker IDs that introduce source dialogue."""
    result: list[str] = []
    seen: set[str] = set()
    for match in _DIALOGUE.finditer(prompt):
        prefix = prompt[max(0, match.start() - 400):match.start()]
        speaker = re.search(r"\(\s*(S\d+)\s*\)[^<]*$", prefix, re.IGNORECASE | re.DOTALL)
        if speaker is None:
            continue
        speaker_id = speaker.group(1).upper()
        if speaker_id not in seen:
            seen.add(speaker_id)
            result.append(speaker_id)
    return tuple(result)


def _validate_speaker_voice_profiles(value: dict[str, Any], prompt: str, raw_json: str) -> None:
    """Require one immutable delivery profile for every explicitly numbered voice."""
    raw_profiles = value.get("speaker_voice_profiles")
    if not isinstance(raw_profiles, list):
        raise _timing_plan_validation_error(
            "response field 'speaker_voice_profiles' must be an array",
            raw_json,
        )
    supplied: set[str] = set()
    for item in raw_profiles:
        if not isinstance(item, dict):
            raise _timing_plan_validation_error(
                "every speaker_voice_profiles entry must be an object",
                raw_json,
            )
        speaker_id = str(item.get("speaker_id") or "").strip().upper().strip("()")
        source = item.get("source")
        profile = item.get("voice_profile")
        if not re.fullmatch(r"S\d+", speaker_id):
            raise _timing_plan_validation_error(
                "speaker_voice_profiles entries need a valid S# speaker_id",
                raw_json,
            )
        if speaker_id in supplied:
            raise _timing_plan_validation_error(
                f"speaker_voice_profiles repeats {speaker_id}",
                raw_json,
            )
        if not isinstance(source, str) or not source.strip():
            raise _timing_plan_validation_error(
                f"speaker_voice_profiles {speaker_id} needs a non-empty source",
                raw_json,
            )
        if not isinstance(profile, str) or not profile.strip():
            raise _timing_plan_validation_error(
                f"speaker_voice_profiles {speaker_id} needs a non-empty voice_profile",
                raw_json,
            )
        supplied.add(speaker_id)
    missing = set(_declared_speaker_ids(prompt)) - supplied
    if missing:
        raise _timing_plan_validation_error(
            "speaker_voice_profiles omits explicit source speaker(s): " + ", ".join(sorted(missing)),
            raw_json,
        )


def _validate_production_bible(
    value: dict[str, Any], request: dict[str, Any], raw_json: str,
) -> tuple[str, str, tuple[GemmaCharacterSubject, ...]]:
    """Validate the immutable global production pass without timing any shot."""
    confidence = value.get("confidence", "unknown")
    if confidence not in {"high", "medium", "low", "unknown"}:
        confidence = "unknown"
    analysis = value.get("analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        raise _timing_plan_validation_error("global production bible needs a concise analysis string", raw_json)
    table = _validate_character_name_table(value, raw_json)
    _validate_speaker_voice_profiles(
        value,
        str(request.get("original_prompt") or ""),
        raw_json,
    )
    known = {item.character_name.casefold(): item for item in table}
    declared = _declared_character_subject_mappings(str(request.get("original_prompt") or ""))
    declared_by_name = {item.character_name.casefold(): item for item in declared}
    missing_declarations = set(declared_by_name) - set(known)
    wrong_declarations = {
        key for key in set(declared_by_name) & set(known)
        if declared_by_name[key].subject.casefold() != known[key].subject.casefold()
    }
    if missing_declarations or wrong_declarations:
        missing_text = ", ".join(
            f"{declared_by_name[key].character_name} -> {declared_by_name[key].subject}"
            for key in sorted(missing_declarations | wrong_declarations)
        )
        raise _timing_plan_validation_error(
            f"global character_name_table omits or changes explicit source mapping(s): {missing_text}", raw_json
        )
    expected_shots = list(request.get("source_shots", ()))
    supplied_shots = value.get("shots")
    if not isinstance(supplied_shots, list) or len(supplied_shots) != len(expected_shots):
        count = len(supplied_shots) if isinstance(supplied_shots, list) else "non-array"
        raise _timing_plan_validation_error(
            f"global production bible has {count} shot records; expected {len(expected_shots)}", raw_json
        )
    for expected, supplied in zip(expected_shots, supplied_shots, strict=True):
        if not isinstance(supplied, dict):
            raise _timing_plan_validation_error("every global shot record must be an object", raw_json)
        shot_number = int(expected["shot_number"])
        try:
            supplied_number = int(supplied["source_shot"])
        except (KeyError, TypeError, ValueError) as error:
            raise _timing_plan_validation_error(
                f"global Source Shot {shot_number} record needs its integer source_shot", raw_json
            ) from error
        if supplied_number != shot_number:
            raise _timing_plan_validation_error(
                f"global production bible expected Source Shot {shot_number}, received {supplied_number}", raw_json
            )
        for field in ("shot_intent", "environment", "camera_and_cut"):
            text = supplied.get(field)
            if not isinstance(text, str) or not text.strip():
                raise _timing_plan_validation_error(
                    f"global Source Shot {shot_number} needs a non-empty {field}", raw_json
                )
        raw_characters = supplied.get("characters")
        if not isinstance(raw_characters, list):
            raise _timing_plan_validation_error(
                f"global Source Shot {shot_number} characters must be an array", raw_json
            )
        seen: set[str] = set()
        for character in raw_characters:
            if not isinstance(character, dict):
                raise _timing_plan_validation_error(
                    f"global Source Shot {shot_number} has a non-object character state", raw_json
                )
            name = character.get("character_name")
            if not isinstance(name, str) or name.strip().casefold() not in known:
                raise _timing_plan_validation_error(
                    f"global Source Shot {shot_number} character_name must use the immutable table", raw_json
                )
            table_entry = known[name.strip().casefold()]
            key = table_entry.character_name.casefold()
            if key in seen:
                raise _timing_plan_validation_error(
                    f"global Source Shot {shot_number} repeats {table_entry.character_name!r}", raw_json
                )
            seen.add(key)
            if str(character.get("subject") or "").strip().casefold() != table_entry.subject.casefold():
                raise _timing_plan_validation_error(
                    f"global Source Shot {shot_number} maps {table_entry.character_name!r} to the wrong subject",
                    raw_json,
                )
            for field in ("opening_state", "closing_state"):
                text = character.get(field)
                if not isinstance(text, str) or not text.strip():
                    raise _timing_plan_validation_error(
                        f"global Source Shot {shot_number} needs {field} for {table_entry.character_name}", raw_json
                    )
        source_body = str(expected.get("source_body") or "")
        explicitly_named = {
            item.character_name.casefold()
            for item in table
            if re.search(rf"(?i)(?<!\w){re.escape(item.character_name)}(?!\w)", source_body)
        }
        missing = explicitly_named - seen
        if missing:
            names = ", ".join(known[key].character_name for key in sorted(missing))
            raise _timing_plan_validation_error(
                f"global Source Shot {shot_number} omits explicitly participating character(s): {names}", raw_json
            )
    return str(confidence), analysis.strip(), table


def _expected_continuity_intervals(
    source_shot: dict[str, Any],
    request: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return exact source-relative pieces owned by physical output chunks."""
    shot_start = int(source_shot["shot_start"])
    shot_end = int(source_shot["shot_end"])
    intervals: list[tuple[int, int]] = []
    for chunk in request.get("chunks", ()):
        start = max(shot_start, int(chunk["output_start"]))
        end = min(shot_end, int(chunk["output_end"]))
        if start < end:
            intervals.append((start - shot_start, end - shot_start))
    return tuple(intervals)


def _dialogue_language_and_spoken_text(value: str) -> tuple[str, str]:
    """Split normalized ``<d>`` content into its language tag and spoken text."""
    normalized = _dialogue_text(value)
    match = re.match(r"^(\[[^\]\r\n]+\])(?:\s+|$)(.*)$", normalized, re.DOTALL)
    if match is None:
        return "", normalized
    return match.group(1), match.group(2).strip()


def _dialogue_segment_intervals(
    start_frame: int,
    end_frame: int,
    expected_shot: dict[str, Any],
    request: dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Intersect one utterance with non-overlapping retained chunk ownership."""
    return tuple(
        (max(start_frame, owned_start), min(end_frame, owned_end))
        for owned_start, owned_end in _expected_continuity_intervals(expected_shot, request)
        if max(start_frame, owned_start) < min(end_frame, owned_end)
    )


def _validate_dialogue_segments(
    raw_overlay: dict[str, Any],
    *,
    shot_number: int,
    overlay_index: int,
    start_frame: int,
    end_frame: int,
    full_content: str,
    expected_shot: dict[str, Any],
    request: dict[str, Any],
    raw_json: str,
) -> tuple[GemmaDialogueSegment, ...]:
    """Validate a word-exact, retained-slice partition of one utterance.

    Gemma chooses natural phrase boundaries during preproduction. Python owns
    only the immutable physical ownership intervals and verifies that the
    resulting pieces neither repeat nor lose any source dialogue tokens.
    """
    full_matches = _DIALOGUE.findall(full_content)
    if len(full_matches) != 1:
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} dialogue overlay {overlay_index} content must contain exactly one <d>...</d> line",
            raw_json,
        )
    full_language, full_spoken = _dialogue_language_and_spoken_text(full_matches[0])
    if not full_spoken:
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} dialogue overlay {overlay_index} has no spoken words",
            raw_json,
        )
    expected_intervals = _dialogue_segment_intervals(
        start_frame,
        end_frame,
        expected_shot,
        request,
    )
    supplied_segments = raw_overlay.get("dialogue_segments")
    if not isinstance(supplied_segments, list):
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} dialogue overlay {overlay_index} dialogue_segments must be an array",
            raw_json,
        )
    if len(supplied_segments) != len(expected_intervals):
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} dialogue overlay {overlay_index} needs {len(expected_intervals)} "
            f"dialogue_segments matching retained chunk ownership; received {len(supplied_segments)}",
            raw_json,
        )

    normalized: list[GemmaDialogueSegment] = []
    spoken_parts: list[str] = []
    for segment_index, (raw_segment, expected_interval) in enumerate(
        zip(supplied_segments, expected_intervals, strict=True), 1
    ):
        if not isinstance(raw_segment, dict):
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} must be an object",
                raw_json,
            )
        try:
            segment_start = int(raw_segment["start_frame"])
            segment_end = int(raw_segment["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} needs integer frames",
                raw_json,
            ) from error
        if (segment_start, segment_end) != expected_interval:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} must cover "
                f"source-relative half-open interval {expected_interval[0]}-{expected_interval[1]}; "
                f"received {segment_start}-{segment_end}",
                raw_json,
            )
        segment_content = raw_segment.get("content")
        if not isinstance(segment_content, str) or not segment_content.strip():
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} needs content",
                raw_json,
            )
        segment_matches = _DIALOGUE.findall(segment_content)
        if len(segment_matches) != 1:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} must contain exactly one <d>...</d> line",
                raw_json,
            )
        segment_language, segment_spoken = _dialogue_language_and_spoken_text(segment_matches[0])
        if segment_language != full_language or not segment_spoken:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} segment {segment_index} must preserve "
                f"the {full_language or 'source'} language tag and contain spoken words",
                raw_json,
            )
        spoken_parts.append(segment_spoken)
        normalized.append(GemmaDialogueSegment(segment_start, segment_end, segment_content.strip()))

    if " ".join(spoken_parts).split() != full_spoken.split():
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} dialogue overlay {overlay_index} dialogue_segments must concatenate to "
            "every original dialogue word and punctuation token exactly once, in order",
            raw_json,
        )
    return tuple(normalized)


def _dialogue_speaker_id(source: str, dialogue_start: int) -> str:
    """Return the stable ``S#`` immediately introducing one dialogue tag."""
    prefix = source[max(0, dialogue_start - 400):dialogue_start]
    match = re.search(r"\(\s*(S\d+)\s*\)[^<]*$", prefix, re.IGNORECASE | re.DOTALL)
    return match.group(1).upper() if match is not None else ""


def _dialogue_token_stream(source: str) -> list[tuple[str, str, str]]:
    """Represent dialogue as speaker/language/token triples in source order."""
    result: list[tuple[str, str, str]] = []
    for match in _DIALOGUE.finditer(source):
        language, spoken = _dialogue_language_and_spoken_text(match.group(1))
        speaker = _dialogue_speaker_id(source, match.start())
        result.extend((speaker, language, token) for token in spoken.split())
    return result


def _validate_shot_dialogue_reconstruction(
    overlays: Sequence[GemmaShotTimingOverlay],
    expected_shot: dict[str, Any],
    raw_json: str,
) -> tuple[GemmaShotTimingOverlay, ...]:
    """Require chronological overlay fragments to reconstruct source speech.

    One source ``<d>`` block may be many minutes long. Gemma is free to split
    it into multiple time-owned overlays, but those overlays collectively must
    preserve speaker, language, words, punctuation, and order exactly once.
    """
    shot_number = int(expected_shot["shot_number"])
    expected_stream = _dialogue_token_stream(str(expected_shot.get("source_body") or ""))
    indexed_dialogue = [
        (index, overlay)
        for index, overlay in enumerate(overlays)
        if overlay.overlay_type == "dialogue"
    ]
    chronological = [
        overlay
        for _index, overlay in sorted(
            indexed_dialogue,
            key=lambda item: (item[1].start_frame, item[1].end_frame, item[0]),
        )
    ]
    actual_stream: list[tuple[str, str, str]] = []
    for overlay in chronological:
        actual_stream.extend(_dialogue_token_stream(overlay.content))
    def token_matches(actual: tuple[str, str, str], expected: tuple[str, str, str]) -> bool:
        actual_speaker, actual_language, actual_token = actual
        expected_speaker, expected_language, expected_token = expected
        return (
            actual_language == expected_language
            and actual_token == expected_token
            and (not expected_speaker or actual_speaker == expected_speaker)
        )

    streams_match = len(actual_stream) == len(expected_stream) and all(
        token_matches(actual, expected)
        for actual, expected in zip(actual_stream, expected_stream, strict=True)
    )
    if not streams_match:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_stream, expected_stream, strict=False),
                )
                if not token_matches(actual, expected)
            ),
            min(len(actual_stream), len(expected_stream)),
        )
        expected_near = expected_stream[mismatch:mismatch + 5]
        actual_near = actual_stream[mismatch:mismatch + 5]
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} chronological dialogue overlay contents must reconstruct every source "
            f"<d> speaker/language/word/punctuation token exactly once; first mismatch at token {mismatch + 1} "
            f"(expected {expected_near!r}, received {actual_near!r})",
            raw_json,
        )
    source_dialogue_starts: set[int] = set()
    source_offset = 0
    source_body = str(expected_shot.get("source_body") or "")
    for match in _DIALOGUE.finditer(source_body):
        source_dialogue_starts.add(source_offset)
        _language, spoken = _dialogue_language_and_spoken_text(match.group(1))
        source_offset += len(spoken.split())

    normalized = list(overlays)
    actual_offset = 0
    for original_index, overlay in sorted(
        indexed_dialogue,
        key=lambda item: (item[1].start_frame, item[1].end_frame, item[0]),
    ):
        normalized[original_index] = replace(
            overlay,
            continues_source_dialogue=actual_offset not in source_dialogue_starts,
        )
        actual_offset += len(_dialogue_token_stream(overlay.content))
    return tuple(normalized)


def _validate_dialogue_speaking_density(
    overlays: Sequence[GemmaShotTimingOverlay],
    expected_shot: dict[str, Any],
    request: dict[str, Any],
    raw_json: str,
) -> None:
    """Reject dialogue schedules too dense for H3 to deliver naturally.

    Gemma owns phrase boundaries and dramatic pacing, but a hard upper guard
    prevents a correction pass from satisfying structural frame coverage by
    cramming a long speech into the first few chunks and padding the remaining
    source shot with silence.
    """
    try:
        fps = float(request.get("fps", 24.0))
    except (TypeError, ValueError):
        fps = 24.0
    if fps <= 0:
        fps = 24.0
    shot_number = int(expected_shot["shot_number"])
    for overlay_index, overlay in enumerate(overlays, 1):
        if overlay.overlay_type != "dialogue":
            continue
        word_count = len(_dialogue_token_stream(overlay.content))
        duration_seconds = (overlay.end_frame - overlay.start_frame) / fps
        spoken_text = " ".join(
            _dialogue_language_and_spoken_text(item)[1]
            for item in _DIALOGUE.findall(overlay.content)
        )
        ellipsis_count = len(re.findall(r"(?:\.{3,}|…+)", spoken_text))
        pause_seconds = ellipsis_count * GEMMA4_DIALOGUE_ELLIPSIS_SECONDS
        delivery_seconds = max(0.0, duration_seconds - pause_seconds)
        capacity = (
            math.ceil(delivery_seconds * GEMMA4_MAX_DIALOGUE_WORDS_PER_SECOND)
            + GEMMA4_DIALOGUE_BURST_WORD_ALLOWANCE
        )
        if word_count > capacity:
            rate = word_count / delivery_seconds if delivery_seconds > 0 else math.inf
            pause_detail = (
                f" after reserving {pause_seconds:.3f}s for {ellipsis_count} ellipsis pause(s)"
                if ellipsis_count else ""
            )
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} dialogue overlay {overlay_index} compresses {word_count} spoken "
                f"words into {duration_seconds:.3f}s{pause_detail} ({rate:.2f} words/s); the permissive maximum is "
                f"{GEMMA4_MAX_DIALOGUE_WORDS_PER_SECOND:g} words/s plus a "
                f"{GEMMA4_DIALOGUE_BURST_WORD_ALLOWANCE}-word short-phrase allowance. Extend this dialogue "
                "across more of the source shot and split its exact words among chronological overlays/segments",
                raw_json,
            )


def _validate_continuity_slices(
    supplied: dict[str, Any],
    expected_shot: dict[str, Any],
    request: dict[str, Any],
    character_name_table: tuple[GemmaCharacterSubject, ...],
    raw_json: str,
) -> tuple[GemmaShotContinuitySlice, ...]:
    """Validate Gemma-owned per-slice entry/exit character state planning."""
    shot_number = int(expected_shot["shot_number"])
    expected_intervals = _expected_continuity_intervals(expected_shot, request)
    raw_slices = supplied.get("continuity_slices")
    if not isinstance(raw_slices, list):
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} continuity_slices must be an array", raw_json
        )
    if len(raw_slices) != len(expected_intervals):
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} needs {len(expected_intervals)} continuity_slices matching its physical "
            f"output ownership; received {len(raw_slices)}", raw_json
        )
    known = {item.character_name.casefold(): item for item in character_name_table}
    source_body = str(expected_shot.get("source_body") or "")
    explicitly_named = {
        item.character_name.casefold()
        for item in character_name_table
        if re.search(rf"(?i)(?<!\w){re.escape(item.character_name)}(?!\w)", source_body)
    }
    normalized: list[GemmaShotContinuitySlice] = []
    seen_across_shot: set[str] = set()
    for index, (raw_slice, expected_interval) in enumerate(
        zip(raw_slices, expected_intervals, strict=True), 1
    ):
        if not isinstance(raw_slice, dict):
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} continuity slice {index} must be an object", raw_json
            )
        try:
            start = int(raw_slice["start_frame"])
            end = int(raw_slice["end_frame"])
        except (KeyError, TypeError, ValueError) as error:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} continuity slice {index} needs integer start_frame and end_frame",
                raw_json,
            ) from error
        if (start, end) != expected_interval:
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} continuity slice {index} must cover source-relative half-open "
                f"interval {expected_interval[0]}-{expected_interval[1]}; received {start}-{end}", raw_json
            )
        raw_characters = raw_slice.get("characters")
        if not isinstance(raw_characters, list):
            raise _timing_plan_validation_error(
                f"Source Shot {shot_number} continuity slice {index} characters must be an array", raw_json
            )
        characters: list[GemmaCharacterContinuity] = []
        seen: set[str] = set()
        for raw_character in raw_characters:
            if not isinstance(raw_character, dict):
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} has a non-object character", raw_json
                )
            name = raw_character.get("character_name")
            if not isinstance(name, str) or name.strip().casefold() not in known:
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} character_name must use the immutable table",
                    raw_json,
                )
            table_entry = known[name.strip().casefold()]
            key = table_entry.character_name.casefold()
            if key in seen:
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} repeats {table_entry.character_name!r}",
                    raw_json,
                )
            seen.add(key)
            seen_across_shot.add(key)
            subject = str(raw_character.get("subject") or "").strip()
            if subject.casefold() != table_entry.subject.casefold():
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} maps {table_entry.character_name!r} "
                    f"to {subject!r}; expected {table_entry.subject}", raw_json
                )
            entry_state = raw_character.get("entry_state")
            exit_state = raw_character.get("expected_exit_state")
            if not isinstance(entry_state, str) or not entry_state.strip():
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} needs a concise entry_state for "
                    f"{table_entry.character_name}", raw_json
                )
            if not isinstance(exit_state, str) or not exit_state.strip():
                raise _timing_plan_validation_error(
                    f"Source Shot {shot_number} continuity slice {index} needs a concise expected_exit_state for "
                    f"{table_entry.character_name}", raw_json
                )
            characters.append(
                GemmaCharacterContinuity(
                    table_entry.character_name,
                    table_entry.subject,
                    entry_state.strip(),
                    exit_state.strip(),
                )
            )
        normalized.append(GemmaShotContinuitySlice(start, end, tuple(characters)))
    # A character named anywhere in the source shot must be tracked somewhere
    # in that shot's continuity plan, but need not be invented in every physical
    # slice.  For example, Tiamat is legitimately still hidden before emerging
    # in Shot 7.  Requiring her in the opening slice was a false-positive render
    # stopper; requiring her at least once still catches a genuinely incomplete
    # continuity plan.
    missing = explicitly_named - seen_across_shot
    if missing:
        names = ", ".join(known[key].character_name for key in sorted(missing))
        raise _timing_plan_validation_error(
            f"Source Shot {shot_number} continuity_slices omit explicitly named character(s) entirely: {names}",
            raw_json,
        )
    return tuple(normalized)


def _validate_timing_plan(value: dict[str, Any], request: dict[str, Any], raw_json: str,
                          system_prompt: str = "", planning_prompt: str = "") -> GemmaShotTimingPlan:
    """Validate a complete Gemma-authored action schedule without rewriting it.

    Unlike warning-only H3 description checks, a malformed static schedule has
    no truthful fallback: its ownership math would be invented by sampler code.
    A caller may ask Gemma for one complete corrected JSON object, otherwise
    sampling stops before Chunk 1.
    """
    confidence = value.get("confidence", "unknown")
    if confidence not in ("high", "medium", "low", "unknown"):
        confidence = "unknown"
    analysis = value.get("analysis", "")
    if not isinstance(analysis, str):
        analysis = ""
    character_name_table = _validate_character_name_table(value, raw_json)
    source_shots = list(request.get("source_shots", ()))
    raw_shots = value.get("shots")
    if not isinstance(raw_shots, list):
        raise _timing_plan_validation_error("response field 'shots' must be an array", raw_json)
    if len(raw_shots) != len(source_shots):
        raise _timing_plan_validation_error(
            f"response has {len(raw_shots)} shot schedules; request requires {len(source_shots)}", raw_json
        )

    schedules: list[GemmaShotTimingShot] = []
    for expected, supplied in zip(source_shots, raw_shots, strict=True):
        if not isinstance(supplied, dict):
            raise _timing_plan_validation_error("every shot schedule must be an object", raw_json)
        try:
            source_shot = int(supplied["source_shot"])
        except (KeyError, TypeError, ValueError) as error:
            raise _timing_plan_validation_error(
                "every shot schedule needs an integer source_shot", raw_json
            ) from error
        expected_number = int(expected["shot_number"])
        expected_start = int(expected["shot_start"])
        expected_end = int(expected["shot_end"])
        if source_shot != expected_number:
            raise _timing_plan_validation_error(
                f"shot schedule must identify Source Shot {expected_number}", raw_json
            )
        raw_visual_beats = supplied.get("visual_beats")
        if not isinstance(raw_visual_beats, list) or not raw_visual_beats:
            raise _timing_plan_validation_error(
                f"Source Shot {expected_number} needs one or more visual_beats objects", raw_json
            )
        light_change = supplied.get("light_change")
        if not isinstance(light_change, bool):
            raise _timing_plan_validation_error(
                f"Source Shot {expected_number} light_change must be true or false", raw_json
            )
        previous_end = 0
        visual_beats: list[GemmaShotTimingBeat] = []
        duration = expected_end - expected_start
        for beat_index, raw_beat in enumerate(raw_visual_beats):
            if not isinstance(raw_beat, dict):
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} has a non-object visual_beats entry", raw_json
                )
            try:
                start = int(raw_beat["start_frame"])
                end = int(raw_beat["end_frame"])
            except (KeyError, TypeError, ValueError) as error:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} visual_beats need integer start_frame and end_frame", raw_json
                ) from error
            action = raw_beat.get("action")
            if not isinstance(action, str) or not action.strip():
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} visual beat {start}-{end} needs a non-empty action", raw_json
                )
            # Gemma consistently describes intermediate beat boundaries as
            # half-open (the next start equals the prior end), but can echo the
            # final *last frame index* from the human-readable source listing
            # instead of the schema's exclusive endpoint.  It is the same
            # final action, not a sampler-authored fallback, so normalize just
            # that unambiguous one-frame spelling at the known shot endpoint.
            if beat_index == len(raw_visual_beats) - 1 and end == duration - 1:
                end = duration
            if start != previous_end or end <= start or end > duration:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} visual_beats must be contiguous, non-empty source-relative "
                    f"intervals from 0 through {duration}; got {start}-{end} after {previous_end}", raw_json
                )
            visual_beats.append(GemmaShotTimingBeat(start, end, action.strip()))
            previous_end = end
        if previous_end != duration:
            raise _timing_plan_validation_error(
                f"Source Shot {expected_number} visual_beats end at {previous_end}, but its duration is {duration}", raw_json
            )
        raw_overlays = supplied.get("overlays", [])
        if not isinstance(raw_overlays, list):
            raise _timing_plan_validation_error(
                f"Source Shot {expected_number} overlays must be an array", raw_json
            )
        overlays: list[GemmaShotTimingOverlay] = []
        for overlay_index, raw_overlay in enumerate(raw_overlays, 1):
            if not isinstance(raw_overlay, dict):
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} overlay {overlay_index} must be an object", raw_json
                )
            try:
                start = int(raw_overlay["start_frame"])
                end = int(raw_overlay["end_frame"])
            except (KeyError, TypeError, ValueError) as error:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} overlay {overlay_index} needs integer start_frame and end_frame", raw_json
                ) from error
            overlay_type = raw_overlay.get("type")
            content = raw_overlay.get("content")
            if overlay_type not in {"dialogue", "sound", "action"}:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} overlay {overlay_index} type must be dialogue, sound, or action", raw_json
                )
            if not isinstance(content, str) or not content.strip():
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} overlay {overlay_index} needs non-empty content", raw_json
                )
            if start < 0 or end <= start or end > duration:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} overlay {overlay_index} must fit source-relative frames 0-{duration}", raw_json
                )
            dialogue_segments = ()
            if overlay_type == "dialogue":
                dialogue_segments = _validate_dialogue_segments(
                    raw_overlay,
                    shot_number=expected_number,
                    overlay_index=overlay_index,
                    start_frame=start,
                    end_frame=end,
                    full_content=content.strip(),
                    expected_shot=expected,
                    request=request,
                    raw_json=raw_json,
                )
            elif "dialogue_segments" in raw_overlay:
                raise _timing_plan_validation_error(
                    f"Source Shot {expected_number} non-dialogue overlay {overlay_index} must not include dialogue_segments",
                    raw_json,
                )
            overlays.append(
                GemmaShotTimingOverlay(start, end, overlay_type, content.strip(), dialogue_segments)
            )
        overlays = list(_validate_shot_dialogue_reconstruction(overlays, expected, raw_json))
        _validate_dialogue_speaking_density(overlays, expected, request, raw_json)
        continuity_slices = _validate_continuity_slices(
            supplied,
            expected,
            request,
            character_name_table,
            raw_json,
        )
        # Global source-shot boundaries are immutable sampler facts, not
        # generated timing content.  Do not require Gemma to echo a redundant
        # inclusive/exclusive endpoint field: that ambiguity cost a full
        # preproduction retry despite an otherwise valid action schedule.
        schedules.append(
            GemmaShotTimingShot(
                source_shot=source_shot,
                shot_start_frame=expected_start,
                shot_end_frame=expected_end,
                visual_beats=tuple(visual_beats),
                overlays=tuple(overlays),
                continuity_slices=continuity_slices,
                light_change=light_change,
            )
        )
    return GemmaShotTimingPlan(
        confidence=confidence,
        analysis=analysis.strip(),
        shots=tuple(schedules),
        character_name_table=character_name_table,
        raw_json=raw_json,
        system_prompt=system_prompt,
        planning_prompt=planning_prompt,
    )


def _validate_single_shot_plan(
    value: dict[str, Any],
    expected_shot: dict[str, Any],
    request: dict[str, Any],
    character_name_table: tuple[GemmaCharacterSubject, ...],
    raw_json: str,
) -> tuple[str, str, GemmaShotTimingShot]:
    """Validate one independently generated source-shot schedule."""
    synthetic_value = dict(value)
    synthetic_value["character_name_table"] = [
        {"character_name": item.character_name, "subject": item.subject}
        for item in character_name_table
    ]
    synthetic_value["shots"] = [{
        key: synthetic_value[key]
        for key in ("source_shot", "light_change", "visual_beats", "overlays", "continuity_slices")
        if key in synthetic_value
    }]
    synthetic_request = dict(request)
    synthetic_request["source_shots"] = [expected_shot]
    validated = _validate_timing_plan(synthetic_value, synthetic_request, raw_json)
    return validated.confidence, validated.analysis, validated.shots[0]


def _production_bible_correction_request(error: Gemma4ObservationError) -> str:
    return (
        "GLOBAL PREPRODUCTION CORRECTION REQUIRED\n"
        "Return one complete replacement global-production-bible JSON object, not a patch or explanation. "
        "Keep all supplied source shots in exact order. Include every explicit character-name-to-<Subject N> "
        "mapping, one immutable speaker_voice_profiles entry for every actual S# vocal source, and each shot's "
        "shot_intent, environment, camera_and_cut, and complete participating-character "
        "opening/closing states. Do not include frame-level timing. Correct this error:\n- "
        + str(error)
    )


def _single_shot_correction_request(
    expected_shot: dict[str, Any], request: dict[str, Any], error: Gemma4ObservationError,
) -> str:
    shot_number = int(expected_shot["shot_number"])
    intervals = ", ".join(
        f"[{start},{end})" for start, end in _expected_continuity_intervals(expected_shot, request)
    ) or "none"
    duration = int(expected_shot["shot_end"]) - int(expected_shot["shot_start"])
    return (
        f"SOURCE SHOT {shot_number} PLAN CORRECTION REQUIRED\n"
        "Return one complete replacement JSON object for this source shot only, not a patch, explanation, global "
        "bible, or another shot. Preserve the immutable global production. Required root fields are confidence, "
        "analysis, source_shot, light_change, visual_beats, overlays, and continuity_slices. light_change must be a boolean. visual_beats must contiguously cover "
        f"source-relative [0,{duration}); continuity_slices must be exactly {intervals}. Every dialogue overlay must "
        "contain dialogue_segments matching each retained ownership intersection. Split only at natural word boundaries, "
        "repeat the language tag in each segment, and make its segment words concatenate to that overlay's content "
        "exactly once. One source speech may span multiple chronological overlays, but all dialogue overlay contents "
        "together must reconstruct every source dialogue token exactly once. Correct this error:\n- "
        + str(error)
    )


def _timing_plan_correction_request(request: dict[str, Any], error: Gemma4ObservationError) -> str:
    """Give Gemma one precise opportunity to repair its complete schedule."""
    expected_lines = []
    for shot in request.get("source_shots", ()):
        intervals = ", ".join(
            f"[{start},{end})"
            for start, end in _expected_continuity_intervals(shot, request)
        ) or "none"
        expected_lines.append(
            f"- Source Shot {int(shot['shot_number'])}: global frames {int(shot['shot_start'])}-{int(shot['shot_end']) - 1}; "
            f"visual_beats must exactly and contiguously cover source-relative frames "
            f"0-{int(shot['shot_end']) - int(shot['shot_start']) - 1}; continuity_slices must exactly be: {intervals}."
        )
    expected = "\n".join(expected_lines)
    return (
        "TIMING-PLAN CORRECTION REQUIRED\n"
        "Your immediately preceding JSON does not form a complete usable schedule. Return one complete replacement "
        "JSON object, not an explanation or patch. Preserve your intended action timing, but use this exact schema "
        "and make every shot's visual_beats contiguous, non-empty, source-relative half-open intervals "
        "[start_frame, end_frame). Overlays may overlap those visual intervals:\n"
        '{"confidence":"high|medium|low", "analysis":"...", '
        '"character_name_table":[{"character_name":"Heman", "subject":"<Subject 1>"}], '
        '"shots":[{"source_shot":1, "light_change":false, '
        '"visual_beats":[{"start_frame":0, "end_frame":34, "action":"..."}], '
        '"overlays":[{"start_frame":4, "end_frame":20, "type":"dialogue|sound|action", "content":"...", '
        '"dialogue_segments":[{"start_frame":4,"end_frame":20,"content":"speaker says: <d>[English] exact assigned words</d>"}]}], '
        '"continuity_slices":[{"start_frame":0,"end_frame":34,"characters":['
        '{"character_name":"Heman","subject":"<Subject 1>","entry_state":"...",'
        '"expected_exit_state":"..."}]}]}]}\n\n'
        "`character_name_table` must be an array. Preserve only explicit name-to-<Subject N> mappings "
        "from the original prompt; use [] when there are none. Every continuity_slices interval must match the "
        "physical output ownership exactly and include every mapped character physically participating there.\n\n"
        "Required source-shot coverage:\n"
        + expected
        + "\n\nDetected error:\n- "
        + str(error)
    )


def _render_observation_messages(
    request: dict[str, Any],
) -> tuple[str, str]:
    """Render the exact system and user text sent beside chronological stills."""
    fps = float(request["fps"])
    current = request["current_chunk"]
    previous = request.get("previous_chunk")
    frame_numbers = [int(item) for item in request.get("observation_frame_numbers", ())]
    previous_last_seen_character_state = json.dumps(
        request.get("previous_last_seen_character_state") or [],
        ensure_ascii=False,
        indent=2,
    )
    if previous is None:
        previous_context = (
            "There is no previous generated chunk. No chronological observation stills are attached. "
            "Plan this first chunk directly from the original intent and its exact target frame slice."
        )
        previous_shots = "none"
        frame_manifest = "none"
        previous_gemma_description = "No previous Gemma-directed detailed_description exists for this first chunk."
        previous_gemma_timing_plan = "No previous Gemma timing plan exists for this first chunk."
        previous_gemma_end_state = "No previous Gemma end state exists for this first chunk."
    else:
        previous_context = (
            f"The immediately previous generated chunk sampled global frames "
            f"{int(previous['sampled_start'])}-{int(previous['sampled_end']) - 1} and retained output frames "
            f"{int(previous['output_start'])}-{int(previous['output_end']) - 1}."
        )
        previous_shots = _shot_context(request.get("previous_shots", ()), fps, include_target=False)
        frame_manifest = "\n".join(
            f"- attached image {index + 1}: exact global frame {frame_number}"
            for index, frame_number in enumerate(frame_numbers)
        ) or "none"
        previous_gemma_description = str(request.get("previous_gemma_description") or (
            "The previous chunk has no recorded Gemma-directed detailed_description. "
            "Use the attached stills and source-shot coverage alone."
        ))
        previous_gemma_timing_plan = str(request.get("previous_gemma_timing_plan") or (
            "The previous chunk has no recorded Gemma timing plan. Use the attached stills and source-shot coverage alone."
        ))
        previous_gemma_end_state = str(request.get("previous_gemma_end_state") or (
            "The previous chunk has no recorded Gemma end state. Use the latest attached still as the current state."
        ))
    templates = _gemma_prompt_templates()
    use_preproduction_cache = bool(request.get("preproduction_cache"))
    template_name = "CACHED_OBSERVATION" if use_preproduction_cache else "OBSERVATION"
    try:
        observation_template = templates[template_name]
    except KeyError as error:
        raise Gemma4ObservationError(
            f"Gemma 4 prompt file {GEMMA4_PROMPTS_PATH} is missing {template_name!r} "
            "for the requested directing mode"
        ) from error
    target_shots = request["target_shots"]
    message = _render_gemma_prompt(
        observation_template,
        {
            "chunk_number": str(int(request["chunk_number"])),
            "chunk_count": str(int(request["chunk_count"])),
            "fps": f"{fps:g}",
            "sampled_start": str(int(current["sampled_start"])),
            "sampled_end": str(int(current["sampled_end"]) - 1),
            "output_start": str(int(current["output_start"])),
            "output_end": str(int(current["output_end"]) - 1),
            "output_frames": str(int(current["output_end"]) - int(current["output_start"])),
            "output_seconds": f"{(int(current['output_end']) - int(current['output_start'])) / fps:.3f}",
            "conditioning_context": str(request["conditioning_context"]),
            "current_shot_timeline": _current_chunk_shot_timeline(target_shots, current),
            "current_shot_timing_contract": _current_shot_timing_contract(target_shots, fps),
            "production_bible": str(request.get("production_bible") or (
                "No separate global production bible is available."
            )),
            "preproduction_timing_plan": str(request.get("preproduction_timing_plan") or (
                "No preproduction timing schedule is available. Allocate the complete source-shot intent "
                "carefully from the timing contract and the rendered evidence."
            )),
            "preproduction_current_slice": str(request.get("preproduction_current_slice") or (
                "No mandatory current-slice beat coverage is available. Use the immutable schedule already "
                "provided in preproduction memory and the exact target frame contract."
            )),
            "character_name_table": str(request.get("character_name_table") or (
                "No explicit named-character-to-subject mapping was found in the original prompt."
            )),
            "planned_character_continuity": str(request.get("planned_character_continuity") or (
                "No planned character-continuity state is available for this slice."
            )),
            "required_local_markers": _required_local_markers(target_shots),
            "previous_context": previous_context,
            "previous_shots": previous_shots,
            "frame_manifest": frame_manifest,
            "previous_gemma_description": previous_gemma_description,
            "previous_gemma_timing_plan": previous_gemma_timing_plan,
            "previous_gemma_end_state": previous_gemma_end_state,
            "previous_last_seen_character_state": previous_last_seen_character_state,
            "last_seen_character_state_contract": _last_seen_character_state_contract(request),
            "target_shots": _shot_context(target_shots, fps, include_target=True),
            "chunk_generation_request": _chunk_generation_request(target_shots, current),
            "original_prompt": str(request["original_prompt"]),
        },
    )
    system = templates["SYSTEM"] + "\n\n" + _minimax_prompt_reference(str(request["prompt_mode"]))
    return system, message


def _preproduction_source_shots(shots: Sequence[dict[str, Any]], fps: float) -> str:
    """Render every complete source shot for the one-time timing-planning pass."""
    blocks: list[str] = []
    for shot in shots:
        start = int(shot["shot_start"])
        end = int(shot["shot_end"])
        duration = end - start
        blocks.append(
            "\n".join((
                f"Source Shot {int(shot['shot_number'])}: global frames {start}-{end - 1} inclusive "
                f"({_frame_timestamp(start, fps)}-{_frame_timestamp(end, fps)}; {duration} frames, {duration / fps:.3f} s).",
                "Complete original source-shot description (authoritative intent):",
                str(shot["source_body"]).strip(),
            ))
        )
    return "\n\n".join(blocks) if blocks else "none"


def _preproduction_chunk_map(chunks: Sequence[dict[str, Any]], shots: Sequence[dict[str, Any]]) -> str:
    """Expose the output ownership geometry that the static plan must serve."""
    lines: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        sampled_start = int(chunk["sampled_start"])
        sampled_end = int(chunk["sampled_end"])
        output_start = int(chunk["output_start"])
        output_end = int(chunk["output_end"])
        portions: list[str] = []
        for shot in shots:
            start = max(output_start, int(shot["shot_start"]))
            end = min(output_end, int(shot["shot_end"]))
            if start >= end:
                continue
            relative_start = start - int(shot["shot_start"])
            relative_end = end - int(shot["shot_start"])
            portions.append(
                f"Source Shot {int(shot['shot_number'])} source-relative half-open interval "
                f"[{relative_start},{relative_end}) (inclusive frames {relative_start}-{relative_end - 1})"
            )
        ownership = "; ".join(portions) if portions else "no retained source-shot frames"
        lines.append(
            f"- Chunk {index}: sampled global frames {sampled_start}-{sampled_end - 1}; "
            f"retains global frames {output_start}-{output_end - 1}; owns {ownership}."
        )
    return "\n".join(lines) if lines else "none"


def _render_timing_plan_messages(request: dict[str, Any]) -> tuple[str, str]:
    """Render the immutable global-production request sent before shot planning."""
    templates = _gemma_prompt_templates()
    try:
        planning_system = templates["PREPRODUCTION_SYSTEM"]
        planning_template = templates["PREPRODUCTION"]
    except KeyError as error:
        raise Gemma4ObservationError(
            f"Gemma 4 prompt file {GEMMA4_PROMPTS_PATH} is missing {error.args[0]!r} for shot timing preproduction"
        ) from error
    fps = float(request["fps"])
    source_shots = request.get("source_shots", ())
    message = _render_gemma_prompt(
        planning_template,
        {
            "fps": f"{fps:g}",
            "chunk_count": str(int(request["chunk_count"])),
            "source_shots": _preproduction_source_shots(source_shots, fps),
            "physical_chunk_map": _preproduction_chunk_map(request.get("chunks", ()), source_shots),
            "original_prompt": str(request["original_prompt"]),
        },
    )
    system = planning_system + "\n\n" + _minimax_prompt_reference(str(request["prompt_mode"]))
    return system, message


def _render_single_shot_plan_messages(
    request: dict[str, Any], production_bible_json: str, shot: dict[str, Any],
) -> tuple[str, str]:
    """Render an independently retryable preproduction request for one shot."""
    templates = _gemma_prompt_templates()
    try:
        system_template = templates["PREPRODUCTION_SHOT_SYSTEM"]
        prompt_template = templates["PREPRODUCTION_SHOT"]
    except KeyError as error:
        raise Gemma4ObservationError(
            f"Gemma 4 prompt file {GEMMA4_PROMPTS_PATH} is missing {error.args[0]!r} "
            "for independent source-shot planning"
        ) from error
    fps = float(request["fps"])
    intervals = _expected_continuity_intervals(shot, request)
    ownership = "\n".join(
        f"- source-relative half-open [{start},{end}) (inclusive frames {start}-{end - 1})"
        for start, end in intervals
    ) or "- no retained physical output interval"
    message = _render_gemma_prompt(
        prompt_template,
        {
            "shot_number": str(int(shot["shot_number"])),
            "fps": f"{fps:g}",
            "production_bible": production_bible_json,
            "source_shot": _preproduction_source_shots([shot], fps),
            "shot_ownership": ownership,
            "original_prompt": str(request["original_prompt"]),
        },
    )
    system = system_template + "\n\n" + _minimax_prompt_reference(str(request["prompt_mode"]))
    return system, message


def _render_preproduction_memory_messages(
    request: dict[str, Any], timing_plan: GemmaShotTimingPlan,
) -> tuple[str, str]:
    """Render the clean static conversation saved immediately before Chunk 1."""
    templates = _gemma_prompt_templates()
    try:
        memory_template = templates["PREPRODUCTION_MEMORY"]
    except KeyError as error:
        raise Gemma4ObservationError(
            f"Gemma 4 prompt file {GEMMA4_PROMPTS_PATH} is missing 'PREPRODUCTION_MEMORY' "
            "for the clean preproduction KV cache"
        ) from error
    fps = float(request["fps"])
    source_shots = request.get("source_shots", ())
    message = _render_gemma_prompt(
        memory_template,
        {
            "character_name_table": timing_plan.character_name_table_text(),
            "production_bible": timing_plan.production_bible_text(),
            "source_shots": _preproduction_source_shots(source_shots, fps),
            "physical_chunk_map": _preproduction_chunk_map(request.get("chunks", ()), source_shots),
            "original_prompt": str(request["original_prompt"]),
        },
    )
    system = templates["SYSTEM"] + "\n\n" + _minimax_prompt_reference(str(request["prompt_mode"]))
    return system, message


def _capture_data_url(data_url: str, destination: Path) -> None:
    prefix, separator, encoded = data_url.partition(",")
    if separator != "," or ";base64" not in prefix:
        raise Gemma4ObservationError("Gemma capture expected a base64 image data URL")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise Gemma4ObservationError("Gemma capture received malformed base64 image data") from error
    destination.write_bytes(payload)


def _capture_last_observation_images(destination: Path, chunk_number: int,
                                     frame_numbers: Sequence[int], image_urls: Sequence[str]) -> None:
    """Save the exact JPEG payloads passed to Gemma under stable frame-aware names."""
    for frame_number, image_url in zip(frame_numbers, image_urls, strict=True):
        filename = f"chunk_{chunk_number:03d}_source_frame_{int(frame_number):06d}.jpg"
        _capture_data_url(str(image_url), destination / filename)


def _capture_observation_request(capture_root: Path, sequence: int, request: dict[str, Any]) -> Path:
    """Persist one exact worker request plus human-readable prompt/image files."""
    chunk_number = int(request["chunk_number"])
    capture_dir = capture_root / f"prompt_{sequence:03d}_chunk_{chunk_number:03d}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    request_snapshot = json.loads(json.dumps(request, ensure_ascii=False))
    (capture_dir / "request.json").write_text(
        json.dumps(request_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    system_prompt, observation_prompt = _render_observation_messages(request_snapshot)
    (capture_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (capture_dir / "observation_prompt.txt").write_text(observation_prompt + "\n", encoding="utf-8")
    image_files = []
    frame_numbers = request_snapshot.get("observation_frame_numbers", ())
    for index, image_url in enumerate(request_snapshot["image_urls"]):
        frame_number = int(frame_numbers[index])
        filename = f"frame_{frame_number:06d}.jpg"
        _capture_data_url(str(image_url), capture_dir / filename)
        image_files.append(filename)
    manifest = {
        "format": "hr-endless-sampler-gemma4-capture-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_number": chunk_number,
        "image_files": image_files,
        "replay_uses": "request.json images and the repository's current gemma4_prompts.txt",
    }
    (capture_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return capture_dir


def _write_capture_result(capture_dir: Path, result: GemmaChunkPrompt) -> None:
    (capture_dir / "response.json").write_text(
        json.dumps(_chunk_prompt_payload(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_capture_error(capture_dir: Path, error: BaseException) -> None:
    raw_json = getattr(error, "raw_json", "")
    (capture_dir / "error.json").write_text(
        json.dumps(
            {
                "error_type": type(error).__name__,
                "message": str(error),
                **({"raw_json": raw_json} if raw_json else {}),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def load_gemma_capture(capture_dir: str | Path, debug: bool = False) -> dict[str, Any]:
    """Load an exact captured request for replay with the current prompt file."""
    path = Path(capture_dir)
    try:
        request = json.loads((path / "request.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Gemma4ObservationError(f"Could not load Gemma capture {path}: {error}") from error
    if "target_shots" not in request or "current_chunk" not in request:
        raise Gemma4ObservationError(
            f"Gemma capture {path} uses the retired action-ledger request format; "
            "capture a new run with the free chunk-prompt director"
        )
    request["debug"] = bool(debug)
    return request


def replay_gemma_capture(capture_dir: str | Path, debug: bool = False) -> GemmaChunkPrompt:
    """Run a saved chunk-director request through the current editable prompts."""
    return _observe_in_worker(load_gemma_capture(capture_dir, debug=debug))


def _json_format_repair_request(error: Gemma4ObservationError) -> str:
    """Ask Gemma to replace a thought/invalid reply without resending context."""
    return (
        "Your immediately preceding reply was unusable because it was not a complete JSON object "
        f"({error}). Return a complete replacement now. Output JSON only: begin with `{{`, end with `}}`, "
        "and include every field required by the original request. Do not emit analysis-channel text, "
        "Markdown fences, commentary, or an explanation."
    )


def _gemma_response_text(choice: dict[str, Any]) -> str:
    """Read either normal or Gemma-4 thought-channel completion content.

    Recent llama.cpp chat conversion separates Gemma's thought-channel output
    into ``reasoning_content``.  A malformed model reply can still contain a
    complete JSON object there, and discarding it turns a usable answer into a
    needless multi-pass retry.  The JSON validator remains the authority: an
    ordinary non-JSON thought is rejected exactly as before.
    """
    for field in ("content", "reasoning_content"):
        text = choice.get(field)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _gemma_chat_json(
    llm: Any,
    messages: Sequence[dict[str, Any]],
    *,
    handler: Any = None,
    max_tokens: int = GEMMA4_CHUNK_RESPONSE_TOKENS,
    mtp_active: bool = False,
) -> tuple[dict[str, Any], str]:
    """Run a fast JSON request, repairing malformed output as a chat turn.

    llama.cpp grammar is deliberately *not* the primary JSON recovery path.
    It is expensive for Gemma's vocabulary and, crucially, it cannot correct
    a model that has just selected an empty private-thought response.  When
    the MTMD handler supports append-only chat, a short model-authored repair
    continues the existing KV conversation instead of re-evaluating images or
    the full request.  Keep the old grammar retry only for minimal/mock
    runtimes with no append API, and as a final guard after two chat repairs.
    """

    def complete(response_format: dict[str, Any] | None = None, *, stage: str) -> str:
        kwargs: dict[str, Any] = {
            "messages": list(messages),
            "temperature": GEMMA4_TEMPERATURE,
            "top_p": GEMMA4_TOP_P,
            "top_k": GEMMA4_TOP_K,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs.update(_gemma_reasoning_budget_kwargs())
        response = llm.create_chat_completion(**kwargs)
        _append_raw_gemma_response(stage, response)
        choice = response["choices"][0]["message"]
        return _gemma_response_text(choice)

    text = complete(stage="initial unconstrained response")
    try:
        return _extract_json_object(text)
    except Gemma4ObservationError as error:
        if mtp_active:
            raise Gemma4MTPOutputError(
                "Gemma 4 MTP returned no complete JSON object; retry this operation with the original decoder",
                raw_json=text,
            ) from error
        append = getattr(handler, "append_user_chat_completion", None)
        if callable(append):
            latest_error = error
            for repair_index in range(1, GEMMA4_JSON_FORMAT_REPAIR_LIMIT + 1):
                logging.warning(
                    "HR Endless Sampler Gemma 4 returned malformed instructed JSON; "
                    "requesting a compact append-only JSON replacement (repair %d/%d): %s",
                    repair_index,
                    GEMMA4_JSON_FORMAT_REPAIR_LIMIT,
                    latest_error,
                )
                response = append(
                    llama=llm,
                    content=_json_format_repair_request(latest_error),
                    temperature=GEMMA4_TEMPERATURE,
                    top_p=GEMMA4_TOP_P,
                top_k=GEMMA4_TOP_K,
                max_tokens=max_tokens,
                **_gemma_reasoning_budget_kwargs(),
            )
                _append_raw_gemma_response(
                    f"append-only JSON format repair {repair_index}/{GEMMA4_JSON_FORMAT_REPAIR_LIMIT}",
                    response,
                )
                candidate = _gemma_response_text(response["choices"][0]["message"])
                try:
                    return _extract_json_object(candidate)
                except Gemma4ObservationError as repair_error:
                    latest_error = repair_error
            logging.warning(
                "HR Endless Sampler Gemma 4 append-only JSON replacements were unusable; "
                "using llama.cpp's grammar only as a final fallback: %s",
                latest_error,
            )
            response = append(
                llama=llm,
                content=_json_format_repair_request(latest_error),
                temperature=GEMMA4_TEMPERATURE,
                top_p=GEMMA4_TOP_P,
                top_k=GEMMA4_TOP_K,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **_gemma_reasoning_budget_kwargs(),
            )
            _append_raw_gemma_response("final grammar-constrained JSON fallback", response)
            candidate = _gemma_response_text(response["choices"][0]["message"])
            return _extract_json_object(candidate)

        logging.warning(
            "HR Endless Sampler Gemma 4 returned malformed instructed JSON; "
            "the runtime has no append-only chat API, so retrying with llama.cpp's slower JSON grammar: %s",
            error,
        )
        return _extract_json_object(
            complete(
                {"type": "json_object"},
                stage="compatibility grammar-constrained JSON fallback",
            )
        )


def _gemma_append_chat_json(handler: Any, llm: Any, content: str | Sequence[dict[str, Any]], *,
                             max_tokens: int = GEMMA4_CHUNK_RESPONSE_TOKENS,
                             mtp_active: bool = False) -> tuple[dict[str, Any], str]:
    """Ask the next user turn, preserving KV through JSON-format recovery."""
    append = getattr(handler, "append_user_chat_completion", None)
    if not callable(append):
        raise Gemma4ObservationError("Gemma runtime does not support append-only chat turns")

    completion_sequence = 0

    def complete(
        next_content: str | Sequence[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        *,
        stage: str,
    ) -> str:
        nonlocal completion_sequence
        completion_sequence += 1
        response = append(
            llama=llm,
            content=next_content,
            temperature=GEMMA4_TEMPERATURE,
            top_p=GEMMA4_TOP_P,
            top_k=GEMMA4_TOP_K,
            max_tokens=max_tokens,
            response_format=response_format,
            **_gemma_reasoning_budget_kwargs(),
        )
        _append_raw_gemma_response(f"{stage} (append call {completion_sequence})", response)
        choice = response["choices"][0]["message"]
        return _gemma_response_text(choice)

    text = complete(content, stage="initial appended response")
    try:
        return _extract_json_object(text)
    except Gemma4ObservationError as error:
        if mtp_active:
            raise Gemma4MTPOutputError(
                "Gemma 4 MTP returned no complete JSON object; retry this operation with the original decoder",
                raw_json=text,
            ) from error
        latest_error = error
        for repair_index in range(1, GEMMA4_JSON_FORMAT_REPAIR_LIMIT + 1):
            logging.warning(
                "HR Endless Sampler Gemma 4 returned malformed instructed JSON in an appended turn; "
                "requesting a compact JSON replacement in the same chat (repair %d/%d): %s",
                repair_index,
                GEMMA4_JSON_FORMAT_REPAIR_LIMIT,
                latest_error,
            )
            candidate = complete(
                _json_format_repair_request(latest_error),
                stage=f"append-only JSON format repair {repair_index}/{GEMMA4_JSON_FORMAT_REPAIR_LIMIT}",
            )
            try:
                return _extract_json_object(candidate)
            except Gemma4ObservationError as repair_error:
                latest_error = repair_error
        logging.warning(
            "HR Endless Sampler Gemma 4 append-only JSON replacements were unusable; "
            "using llama.cpp's grammar only as a final fallback: %s",
            latest_error,
        )
        return _extract_json_object(
            complete(
                _json_format_repair_request(latest_error),
                {"type": "json_object"},
                stage="final grammar-constrained JSON fallback",
            )
        )


def _write_preproduction_cache_state(llm: Any, spec: object, *, system_prompt: str,
                                     memory_prompt: str) -> int:
    """Persist only the clean directorial KV/input state, not the giant logits array.

    ``Llama.save_state`` also serializes one float logit row per already
    evaluated token.  A preproduction conversation can therefore turn a
    roughly 5 GiB KV snapshot into a much larger tens-of-GiB file even though
    append-only turns do not need historical logits.  The next turn always
    evaluates new suffix tokens before sampling, so persisting the native
    llama state plus token history is sufficient and materially cheaper.
    """
    import ctypes
    import hashlib
    import llama_cpp

    state_path, manifest_path = _preproduction_cache_paths(spec)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_size = int(llama_cpp.llama_state_get_size(llm._ctx.ctx))
    state_buffer = (ctypes.c_uint8 * state_size)()
    copied = int(llama_cpp.llama_state_get_data(llm._ctx.ctx, state_buffer, state_size))
    if copied <= 0 or copied > state_size:
        raise Gemma4ObservationError("llama.cpp could not export the Gemma preproduction KV state")
    metadata = {
        "format": _PREPRODUCTION_CACHE_FORMAT,
        "n_ctx": int(llm.n_ctx()),
        "n_tokens": int(llm.n_tokens),
        "input_ids": [int(token) for token in llm.input_ids[:llm.n_tokens]],
        "seed": int(getattr(llm, "_seed", 0)),
        "llama_state_size": copied,
        "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "memory_sha256": hashlib.sha256(memory_prompt.encode("utf-8")).hexdigest(),
    }
    temporary = state_path.with_suffix(".bin.partial")
    try:
        with temporary.open("wb") as state_file:
            state_file.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            state_file.write(b"\n")
            state_file.write(memoryview(state_buffer)[:copied])
        temporary.replace(state_path)
        manifest = {
            "format": _PREPRODUCTION_CACHE_FORMAT,
            "n_ctx": metadata["n_ctx"],
            "n_tokens": metadata["n_tokens"],
            "state_bytes": copied,
            "system_sha256": metadata["system_sha256"],
            "memory_sha256": metadata["memory_sha256"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_temporary = manifest_path.with_suffix(".json.partial")
        manifest_temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_temporary.replace(manifest_path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return copied


def _populate_preproduction_cache_state(
    llm: Any,
    request: dict[str, Any],
    timing_plan: GemmaShotTimingPlan,
) -> int:
    """Build and export the clean pre-Chunk-1 directorial conversation.

    This consumes the already validated timing plan rather than asking Gemma
    to plan it again. Both a fresh preproduction worker and a debug-replay
    worker use this path, keeping their restored chunk contexts equivalent.
    """
    cache_spec = request.get("preproduction_cache")
    if not cache_spec:
        raise Gemma4ObservationError("Gemma preproduction KV cache was not requested")
    memory_system, memory_prompt = _render_preproduction_memory_messages(request, timing_plan)
    # The state is deliberately before the first physical chunk, not a
    # rolling all-chunk history.
    _gemma_chat_json(
        llm,
        [
            {"role": "system", "content": memory_system},
            {"role": "user", "content": memory_prompt},
        ],
        max_tokens=64,
    )
    return _write_preproduction_cache_state(
        llm,
        cache_spec,
        system_prompt=memory_system,
        memory_prompt=memory_prompt,
    )


def _restore_preproduction_cache_state(llm: Any, spec: object) -> int:
    """Restore a clean preproduction snapshot into a newly isolated worker."""
    import ctypes
    import llama_cpp

    state_path, manifest_path = _preproduction_cache_paths(spec)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != _PREPRODUCTION_CACHE_FORMAT:
            raise ValueError("wrong cache format")
        with state_path.open("rb") as state_file:
            metadata = json.loads(state_file.readline().decode("utf-8"))
            if metadata.get("format") != _PREPRODUCTION_CACHE_FORMAT:
                raise ValueError("wrong state format")
            n_ctx = int(metadata["n_ctx"])
            n_tokens = int(metadata["n_tokens"])
            input_ids = metadata["input_ids"]
            state_size = int(metadata["llama_state_size"])
            if n_ctx != int(llm.n_ctx()):
                raise ValueError(f"cache n_ctx {n_ctx} does not match worker n_ctx {llm.n_ctx()}")
            if n_tokens <= 0 or n_tokens >= n_ctx or len(input_ids) != n_tokens:
                raise ValueError("invalid cached token history")
            state_buffer = (ctypes.c_uint8 * state_size)()
            copied = state_file.readinto(state_buffer)
            if copied != state_size or state_file.read(1):
                raise ValueError("truncated or overlong cached llama state")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise Gemma4ObservationError(f"could not restore Gemma preproduction KV cache: {error}") from error

    llm.reset()
    llm._ctx.kv_cache_clear()
    restored = int(llama_cpp.llama_state_set_data(llm._ctx.ctx, state_buffer, state_size))
    if restored != state_size:
        raise Gemma4ObservationError("llama.cpp could not restore the Gemma preproduction KV state")
    llm.input_ids[:n_tokens] = input_ids
    llm.n_tokens = n_tokens
    llm._requires_eval = True
    llm._seed = int(metadata.get("seed", 0))
    return state_size


def _observe_in_process(
    request: dict[str, Any],
    image_urls: Sequence[str],
    debug: bool,
) -> GemmaChunkPrompt:
    """Run one observation inside the disposable worker process."""
    Llama, Gemma4ChatHandler = _load_runtime()
    model_path, mmproj_path = _ensure_model_files()
    system_prompt, message = _render_observation_messages(request)
    # Gemma 4's official modality order is images before user text. Preserve
    # chronological order within the image sequence.
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    content.append({"type": "text", "text": message})

    handler = None
    llm = None
    response = None
    try:
        # Keep llama.cpp/MTMD native timing traces separate from the sampler's
        # useful debug mode.  The latter is preserved by our own diagnostics.
        handler = Gemma4ChatHandler(mmproj_path=str(mmproj_path), verbose=False, use_gpu=True, enable_thinking=True, image_min_tokens=GEMMA4_IMAGE_MIN_TOKENS, image_max_tokens=GEMMA4_IMAGE_MAX_TOKENS, batch_max_tokens=GEMMA4_BATCH_SIZE)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            seed=int(request.get("gemma4_seed", 0)),
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        latest_raw_json = ""
        if request.get("preproduction_cache"):
            try:
                restored_bytes = _restore_preproduction_cache_state(llm, request["preproduction_cache"])
                logging.info(
                    "HR Endless Sampler Gemma 4 restored clean preproduction KV context (%0.2f GiB) for Chunk %d.",
                    restored_bytes / (1024 ** 3),
                    int(request["chunk_number"]),
                )
                payload, raw_json = _gemma_append_chat_json(
                    handler,
                    llm,
                    content,
                    mtp_active=bool(request.get("gemma4_mtp", False)),
                )
            except (Gemma4ObservationError, OSError, RuntimeError, ValueError) as cache_error:
                # A cache is a speed-up, never a new dependency for a render.
                # Render the ordinary fully self-contained request when its
                # state cannot be restored by this disposable worker.
                logging.warning(
                    "HR Endless Sampler Gemma 4 could not use the clean preproduction KV cache for Chunk %d; "
                    "falling back to the full self-contained request: %s",
                    int(request["chunk_number"]),
                    cache_error,
                )
                request.pop("preproduction_cache", None)
                request.pop("preproduction_current_slice", None)
                system_prompt, message = _render_observation_messages(request)
                content[-1] = {"type": "text", "text": message}
                initial_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
                payload, raw_json = _gemma_chat_json(
                    llm,
                    initial_messages,
                    handler=handler,
                    mtp_active=bool(request.get("gemma4_mtp", False)),
                )
        else:
            payload, raw_json = _gemma_chat_json(
                llm,
                initial_messages,
                handler=handler,
                mtp_active=bool(request.get("gemma4_mtp", False)),
            )
        latest_raw_json = raw_json
        attempts: list[GemmaPromptAttempt] = []
        response_repairs = 0
        contract_correction_used = False
        structural_repairs = 0
        pending_correction_prompt = ""

        def request_replacement(correction_prompt: str) -> tuple[dict[str, Any], str]:
            if callable(getattr(handler, "append_user_chat_completion", None)):
                # This is a genuine next chat turn. Its user text is small and
                # the observation images plus preceding answer are already in
                # KV, so schema repair does not re-encode the complete request.
                return _gemma_append_chat_json(
                    handler,
                    llm,
                    correction_prompt,
                    mtp_active=bool(request.get("gemma4_mtp", False)),
                )
            # Compatibility path for the deliberately minimal mocked runtime
            # used by unit tests and unexpectedly old llama-cpp installs.
            correction_messages = [
                *initial_messages,
                {"role": "assistant", "content": latest_raw_json},
                {"role": "user", "content": correction_prompt},
            ]
            return _gemma_chat_json(
                llm,
                correction_messages,
                handler=handler,
                mtp_active=bool(request.get("gemma4_mtp", False)),
            )

        while True:
            try:
                candidate = _validate_chunk_prompt(
                    payload,
                    request,
                    latest_raw_json,
                    system_prompt=system_prompt,
                    observation_prompt=message,
                )
            except Gemma4ObservationError as error:
                attempts.append(
                    GemmaPromptAttempt(
                        kind=(
                            "initial invalid response"
                            if not attempts
                            else "invalid schema-repair response"
                        ),
                        raw_json=latest_raw_json,
                        validation_warnings=(str(error),),
                        correction_prompt=pending_correction_prompt,
                    )
                )
                if response_repairs >= GEMMA4_RESPONSE_REPAIR_LIMIT:
                    raise Gemma4ObservationError(
                        f"{error} after {response_repairs} model-authored repair attempts",
                        raw_json=latest_raw_json,
                    ) from error
                response_repairs += 1
                pending_correction_prompt = (
                    _chunk_contract_correction_request(request, (str(error),))
                    if response_repairs == 1
                    else _chunk_contract_followup_request((str(error),))
                )
                logging.warning(
                    "HR Endless Sampler Gemma 4 returned an unusable chunk response: %s. "
                    "Requesting a complete model-authored replacement in the same conversation "
                    "(repair %d/%d).",
                    error,
                    response_repairs,
                    GEMMA4_RESPONSE_REPAIR_LIMIT,
                )
                payload, latest_raw_json = request_replacement(pending_correction_prompt)
                continue

            attempts.append(
                GemmaPromptAttempt(
                    kind=(
                        "initial response"
                        if len(attempts) == 0
                        else "chunk-contract correction response"
                    ),
                    raw_json=candidate.raw_json,
                    validation_warnings=candidate.validation_warnings,
                    correction_prompt=pending_correction_prompt,
                )
            )
            contract_warnings = _contract_validation_warnings(candidate.validation_warnings)
            marker_warnings = _marker_validation_warnings(contract_warnings)
            character_state_warnings = _character_state_validation_warnings(contract_warnings)
            if not contract_warnings:
                return replace(candidate, attempts=tuple(attempts))

            if not contract_correction_used:
                # Keep the established policy of one creative contract rewrite.
                # Structural marker errors are different: Python owns their
                # exact sequence and may reinforce them again below without
                # substituting algorithmic H3 prose.
                contract_correction_used = True
                structural_repairs = 1 if (marker_warnings or character_state_warnings) else 0
                pending_correction_prompt = _chunk_contract_correction_request(request, contract_warnings)
            elif marker_warnings:
                if structural_repairs >= GEMMA4_RESPONSE_REPAIR_LIMIT:
                    raise Gemma4ObservationError(
                        "Gemma 4 still violates the sampler-owned H3 shot-marker contract after "
                        f"{structural_repairs} model-authored repairs: " + "; ".join(marker_warnings),
                        raw_json=candidate.raw_json,
                    )
                structural_repairs += 1
                pending_correction_prompt = _marker_contract_followup_request(request, marker_warnings)
                logging.warning(
                    "HR Endless Sampler Gemma 4 still returned invalid H3 shot markers. "
                    "Requesting focused model-authored marker repair %d/%d in the same conversation:\n- %s",
                    structural_repairs,
                    GEMMA4_RESPONSE_REPAIR_LIMIT,
                    "\n- ".join(marker_warnings),
                )
            elif character_state_warnings:
                if structural_repairs >= GEMMA4_RESPONSE_REPAIR_LIMIT:
                    raise Gemma4ObservationError(
                        "Gemma 4 still violates the character continuity contract after "
                        f"{structural_repairs} model-authored repairs: "
                        + "; ".join(character_state_warnings),
                        raw_json=candidate.raw_json,
                    )
                structural_repairs += 1
                pending_correction_prompt = _chunk_contract_followup_request(character_state_warnings)
                logging.warning(
                    "HR Endless Sampler Gemma 4 still returned invalid character continuity state. "
                    "Requesting model-authored repair %d/%d in the same conversation:\n- %s",
                    structural_repairs,
                    GEMMA4_RESPONSE_REPAIR_LIMIT,
                    "\n- ".join(character_state_warnings),
                )
            else:
                # Non-marker creative findings stay visible after the single
                # established rewrite; never replace Gemma prose algorithmically.
                return replace(candidate, attempts=tuple(attempts))
            payload, latest_raw_json = request_replacement(pending_correction_prompt)
    finally:
        if llm is not None:
            llm.close()
        # MTMD's context is registered on Llama's ExitStack and is closed by
        # llm.close(). Drop all Python owners too; worker exit below is the
        # authoritative cleanup for llama.cpp/ggml's non-PyTorch CUDA state.
        llm = None
        handler = None
        response = None
        content.clear()
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _materialize_preproduction_cache_in_process(
    request: dict[str, Any], timing_plan: GemmaShotTimingPlan, debug: bool,
) -> int:
    """Create a clean KV snapshot from an already-restored replay plan.

    A debug replay avoids re-running the creative planner, but it must rebuild
    this static directorial conversation after the render-local cache reset.
    Otherwise every replayed chunk falls back to a full self-contained prompt.
    """
    Llama, Gemma4ChatHandler = _load_runtime()
    model_path, mmproj_path = _ensure_model_files()
    handler = None
    llm = None
    try:
        handler = Gemma4ChatHandler(mmproj_path=str(mmproj_path), verbose=False, use_gpu=True, enable_thinking=True, image_min_tokens=GEMMA4_IMAGE_MIN_TOKENS, image_max_tokens=GEMMA4_IMAGE_MAX_TOKENS, batch_max_tokens=GEMMA4_BATCH_SIZE)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            seed=int(request.get("gemma4_seed", 0)),
        )
        state_bytes = _populate_preproduction_cache_state(llm, request, timing_plan)
        logging.info(
            "HR Endless Sampler Gemma 4 saved replay clean preproduction KV context to %s (%0.2f GiB).",
            _preproduction_cache_paths(request["preproduction_cache"])[0],
            state_bytes / (1024 ** 3),
        )
        return state_bytes
    finally:
        if llm is not None:
            llm.close()
        llm = None
        handler = None
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _plan_timing_in_process(request: dict[str, Any], debug: bool) -> GemmaShotTimingPlan:
    """Build a global production bible, then independently plan each shot."""
    Llama, Gemma4ChatHandler = _load_runtime()
    model_path, mmproj_path = _ensure_model_files()
    system_prompt, message = _render_timing_plan_messages(request)
    handler = None
    llm = None
    try:
        # Keep the official Gemma multimodal chat handler even though this pass
        # has no images.  It supplies the same model-specific conversation
        # formatting as the later image-and-text requests.
        handler = Gemma4ChatHandler(mmproj_path=str(mmproj_path), verbose=False, use_gpu=True, enable_thinking=True, image_min_tokens=GEMMA4_IMAGE_MIN_TOKENS, image_max_tokens=GEMMA4_IMAGE_MAX_TOKENS, batch_max_tokens=GEMMA4_BATCH_SIZE)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            seed=int(request.get("gemma4_seed", 0)),
        )
        global_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        logging.info(
            "HR Endless Sampler Gemma 4 preproduction phase 1: building the immutable global production bible."
        )
        payload, raw_json = _gemma_chat_json(
            llm,
            global_messages,
            handler=handler,
            max_tokens=GEMMA4_GLOBAL_PREPRODUCTION_RESPONSE_TOKENS,
            mtp_active=bool(request.get("gemma4_mtp", False)),
        )
        current_payload = payload
        current_raw_json = raw_json
        correction_prompt = ""
        attempts: list[GemmaPromptAttempt] = []
        global_result = None
        for repair_index in range(GEMMA4_RESPONSE_REPAIR_LIMIT + 1):
            try:
                confidence, analysis, character_name_table = _validate_production_bible(
                    current_payload,
                    request,
                    current_raw_json,
                )
            except Gemma4ObservationError as error:
                attempts.append(
                    GemmaPromptAttempt(
                        kind=("global production-bible response" if repair_index == 0 else
                              f"global production-bible correction {repair_index}"),
                        raw_json=current_raw_json,
                        validation_warnings=(str(error),),
                        correction_prompt=correction_prompt,
                    )
                )
                if repair_index >= GEMMA4_RESPONSE_REPAIR_LIMIT:
                    raise Gemma4ObservationError(str(error), raw_json=current_raw_json) from error
                correction_prompt = _production_bible_correction_request(error)
                logging.warning(
                    "HR Endless Sampler Gemma 4 global production bible failed validation; "
                    "requesting global correction %d/%d: %s",
                    repair_index + 1,
                    GEMMA4_RESPONSE_REPAIR_LIMIT,
                    error,
                )
                if callable(getattr(handler, "append_user_chat_completion", None)):
                    current_payload, current_raw_json = _gemma_append_chat_json(
                        handler,
                        llm,
                        correction_prompt,
                        max_tokens=GEMMA4_GLOBAL_PREPRODUCTION_RESPONSE_TOKENS,
                        mtp_active=bool(request.get("gemma4_mtp", False)),
                    )
                else:
                    correction_messages = [
                        *global_messages,
                        {"role": "assistant", "content": current_raw_json},
                        {"role": "user", "content": correction_prompt},
                    ]
                    current_payload, current_raw_json = _gemma_chat_json(
                        llm,
                        correction_messages,
                        handler=handler,
                        max_tokens=GEMMA4_GLOBAL_PREPRODUCTION_RESPONSE_TOKENS,
                        mtp_active=bool(request.get("gemma4_mtp", False)),
                    )
                continue
            attempts.append(
                GemmaPromptAttempt(
                    kind=("global production-bible response" if repair_index == 0 else
                          f"global production-bible correction {repair_index}"),
                    raw_json=current_raw_json,
                    correction_prompt=correction_prompt,
                )
            )
            global_result = (
                confidence,
                analysis,
                character_name_table,
                json.dumps(current_payload, ensure_ascii=False, indent=2),
            )
            break

        if global_result is None:  # Defensive: the loop either validates or raises.
            raise Gemma4ObservationError(
                "Gemma 4 global-production correction loop ended without a usable result",
                raw_json=current_raw_json,
            )

        confidence, analysis, character_name_table, production_bible_json = global_result
        finalized_shots: list[GemmaShotTimingShot] = []
        shot_planning_prompts: list[str] = []
        assembled_raw: dict[str, Any] = {
            "global_production_bible": current_payload,
            "shot_plans": [],
        }
        source_shots = tuple(request.get("source_shots", ()))
        for shot_index, expected_shot in enumerate(source_shots, 1):
            shot_number = int(expected_shot["shot_number"])
            logging.info(
                "HR Endless Sampler Gemma 4 preproduction phase 2: planning Source Shot %d independently (%d/%d).",
                shot_number,
                shot_index,
                len(source_shots),
            )
            shot_system, shot_message = _render_single_shot_plan_messages(
                request,
                production_bible_json,
                expected_shot,
            )
            shot_planning_prompts.append(shot_message)
            shot_messages = [
                {"role": "system", "content": shot_system},
                {"role": "user", "content": shot_message},
            ]
            shot_payload, shot_raw_json = _gemma_chat_json(
                llm,
                shot_messages,
                handler=handler,
                max_tokens=GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS,
                mtp_active=bool(request.get("gemma4_mtp", False)),
            )
            shot_correction_prompt = ""
            finalized_shot = None
            for repair_index in range(GEMMA4_RESPONSE_REPAIR_LIMIT + 1):
                try:
                    _shot_confidence, _shot_analysis, validated_shot = _validate_single_shot_plan(
                        shot_payload,
                        expected_shot,
                        request,
                        character_name_table,
                        shot_raw_json,
                    )
                except Gemma4ObservationError as error:
                    attempts.append(
                        GemmaPromptAttempt(
                            kind=(f"Source Shot {shot_number} response" if repair_index == 0 else
                                  f"Source Shot {shot_number} correction {repair_index}"),
                            raw_json=shot_raw_json,
                            validation_warnings=(str(error),),
                            correction_prompt=shot_correction_prompt,
                        )
                    )
                    if repair_index >= GEMMA4_RESPONSE_REPAIR_LIMIT:
                        raise Gemma4ObservationError(str(error), raw_json=shot_raw_json) from error
                    shot_correction_prompt = _single_shot_correction_request(
                        expected_shot, request, error,
                    )
                    logging.warning(
                        "HR Endless Sampler Gemma 4 Source Shot %d plan failed validation; "
                        "retrying only Source Shot %d (%d/%d): %s",
                        shot_number,
                        shot_number,
                        repair_index + 1,
                        GEMMA4_RESPONSE_REPAIR_LIMIT,
                        error,
                    )
                    if callable(getattr(handler, "append_user_chat_completion", None)):
                        shot_payload, shot_raw_json = _gemma_append_chat_json(
                            handler,
                            llm,
                            shot_correction_prompt,
                            max_tokens=GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS,
                            mtp_active=bool(request.get("gemma4_mtp", False)),
                        )
                    else:
                        shot_payload, shot_raw_json = _gemma_chat_json(
                            llm,
                            [
                                *shot_messages,
                                {"role": "assistant", "content": shot_raw_json},
                                {"role": "user", "content": shot_correction_prompt},
                            ],
                            handler=handler,
                            max_tokens=GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS,
                            mtp_active=bool(request.get("gemma4_mtp", False)),
                        )
                    continue
                attempts.append(
                    GemmaPromptAttempt(
                        kind=(f"Source Shot {shot_number} response" if repair_index == 0 else
                              f"Source Shot {shot_number} correction {repair_index}"),
                        raw_json=shot_raw_json,
                        correction_prompt=shot_correction_prompt,
                    )
                )
                finalized_shot = validated_shot
                assembled_raw["shot_plans"].append(shot_payload)
                break
            if finalized_shot is None:
                raise Gemma4ObservationError(
                    f"Gemma 4 Source Shot {shot_number} correction loop ended without a usable result",
                    raw_json=shot_raw_json,
                )
            finalized_shots.append(finalized_shot)

        result = GemmaShotTimingPlan(
            confidence=confidence,
            analysis=analysis,
            shots=tuple(finalized_shots),
            character_name_table=character_name_table,
            raw_json=json.dumps(assembled_raw, ensure_ascii=False, indent=2),
            production_bible_json=production_bible_json,
            shot_planning_prompts=tuple(shot_planning_prompts),
            system_prompt=(
                "=== GLOBAL PREPRODUCTION SYSTEM ===\n"
                + system_prompt
                + "\n\n=== INDEPENDENT SOURCE-SHOT SYSTEM ===\n"
                + shot_system
            ),
            planning_prompt=(
                "=== GLOBAL PRODUCTION-BIBLE REQUEST ===\n"
                + message
                + "\n\n"
                + "\n\n".join(
                    f"=== SOURCE SHOT {int(shot['shot_number'])} PREPRODUCTION REQUEST ===\n{prompt}"
                    for shot, prompt in zip(
                        request.get("source_shots", ()), shot_planning_prompts, strict=True
                    )
                )
            ),
            attempts=tuple(attempts),
        )

        if request.get("preproduction_cache"):
            try:
                state_bytes = _populate_preproduction_cache_state(llm, request, result)
                logging.info(
                    "HR Endless Sampler Gemma 4 saved clean preproduction KV context to %s (%0.2f GiB).",
                    _preproduction_cache_paths(request["preproduction_cache"])[0],
                    state_bytes / (1024 ** 3),
                )
            except (Gemma4ObservationError, OSError, RuntimeError, ValueError) as cache_error:
                # The cache toggle is intentionally opportunistic. A state
                # export failure must not invalidate a fully usable timing
                # plan or prevent H3 sampling.
                logging.warning(
                    "HR Endless Sampler Gemma 4 could not create the clean preproduction KV cache; "
                    "each chunk will use its ordinary self-contained request: %s",
                    cache_error,
                )
        return result
    finally:
        if llm is not None:
            llm.close()
        llm = None
        handler = None
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # The parent always decodes the private worker protocol as UTF-8. Force
    # the child to use the same encoding even on Windows, where an inherited
    # console locale can otherwise emit CP-1252 bytes such as 0x93.
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    comfy_root = str(Path(folder_paths.__file__).resolve().parent)
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        comfy_root if not current_pythonpath else comfy_root + os.pathsep + current_pythonpath
    )
    return environment


def _start_worker_process() -> subprocess.Popen:
    """Start one isolated Gemma worker with a stable UTF-8 text protocol."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_worker_environment(),
    )


def _stream_worker_output(
    process: subprocess.Popen,
    request: dict[str, Any],
    progress_callback: Any = None,
) -> str:
    """Send one request and consume progress records while the worker runs."""
    if process.stdin is None or process.stdout is None:
        raise Gemma4ObservationError("Gemma 4 worker pipes were not created")
    output: list[str] = []
    stop_watcher = threading.Event()
    cancel_requested = threading.Event()

    def watch_for_comfy_cancel() -> None:
        """Kill a blocking Gemma subprocess when ComfyUI interrupts the job."""
        poll = getattr(process, "poll", None)
        while not stop_watcher.wait(0.05):
            if callable(poll) and poll() is not None:
                return
            if comfy.model_management.processing_interrupted():
                cancel_requested.set()
                if not callable(poll) or poll() is None:
                    process.kill()
                return

    # Reading a pipe line-by-line can block while llama.cpp loads a model or
    # computes a token. A tiny watcher keeps cancellation responsive during
    # those periods instead of waiting for Gemma's next progress line.
    watcher = threading.Thread(
        target=watch_for_comfy_cancel,
        name="hr_endless_sampler_gemma_cancel",
        daemon=True,
    )
    watcher.start()
    try:
        process.stdin.write(json.dumps(request, ensure_ascii=False))
        process.stdin.close()
        for line in process.stdout:
            if line.startswith(_WORKER_PROGRESS_PREFIX):
                try:
                    progress = json.loads(line[len(_WORKER_PROGRESS_PREFIX):])
                    tokens = int(progress["tokens"])
                    tokens_per_second = float(progress["tokens_per_second"])
                    generation = int(progress.get("generation", 1))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # A cosmetic throughput record must never invalidate the
                    # actual model response following it.
                    continue
                if callable(progress_callback):
                    progress_callback(tokens, tokens_per_second, generation)
                continue
            output.append(line)
        process.wait()
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        stop_watcher.set()
        watcher.join(timeout=1.0)
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
    if cancel_requested.is_set():
        # Consume/reset ComfyUI's interrupt flag and raise its canonical
        # cancellation exception. This is intentionally not a Gemma retry.
        comfy.model_management.throw_exception_if_processing_interrupted()
    return "".join(output)


def _observe_in_worker(request: dict[str, Any], progress_callback: Any = None) -> GemmaChunkPrompt:
    process = _start_worker_process()
    try:
        stdout = _stream_worker_output(process, request, progress_callback)
    finally:
        request.clear()

    result_line = next(
        (line[len(_WORKER_RESULT_PREFIX):] for line in reversed(stdout.splitlines())
         if line.startswith(_WORKER_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise Gemma4WorkerExitError(
            f"Gemma 4 worker exited with status {process.returncode} without returning a result",
            returncode=process.returncode,
        )
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as error:
        raise Gemma4ObservationError("Gemma 4 worker returned malformed result JSON") from error
    if not result.get("ok"):
        message = str(result.get("message") or "unknown worker failure")
        raw_json = str(result.get("raw_json") or "")
        if result.get("error_type") == "Gemma4DependencyError":
            raise Gemma4DependencyError(message)
        if result.get("error_type") in {"Gemma4MTPError", "Gemma4MTPOutputError"}:
            worker_error_type = str(result["error_type"])
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type=worker_error_type,
                raw_json=raw_json,
            )
        raise Gemma4ObservationError(message, raw_json=raw_json)
    if process.returncode != 0:
        raise Gemma4WorkerExitError(
            f"Gemma 4 worker exited with status {process.returncode}",
            returncode=process.returncode,
        )
    return _chunk_prompt_from_payload(result["chunk_prompt"])


def _plan_timing_in_worker(request: dict[str, Any], progress_callback: Any = None) -> GemmaShotTimingPlan:
    """Run one isolated preproduction worker and decode its validated schedule."""
    request = json.loads(json.dumps(request, ensure_ascii=False))
    request["operation"] = "timing_plan"
    process = _start_worker_process()
    stdout = _stream_worker_output(process, request, progress_callback)

    result_line = next(
        (line[len(_WORKER_RESULT_PREFIX):] for line in reversed(stdout.splitlines())
         if line.startswith(_WORKER_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise Gemma4WorkerExitError(
            f"Gemma 4 worker exited with status {process.returncode} without returning a result",
            returncode=process.returncode,
        )
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as error:
        raise Gemma4ObservationError("Gemma 4 worker returned malformed result JSON") from error
    if not result.get("ok"):
        message = str(result.get("message") or "unknown worker failure")
        raw_json = str(result.get("raw_json") or "")
        if result.get("error_type") == "Gemma4DependencyError":
            raise Gemma4DependencyError(message)
        if result.get("error_type") in {"Gemma4MTPError", "Gemma4MTPOutputError"}:
            worker_error_type = str(result["error_type"])
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type=worker_error_type,
                raw_json=raw_json,
            )
        raise Gemma4ObservationError(message, raw_json=raw_json)
    if process.returncode != 0:
        raise Gemma4WorkerExitError(
            f"Gemma 4 worker exited with status {process.returncode}",
            returncode=process.returncode,
        )
    try:
        return _timing_plan_from_payload(result["timing_plan"])
    except (KeyError, TypeError, ValueError) as error:
        raise Gemma4ObservationError("Gemma 4 worker returned malformed timing-plan JSON") from error


def _materialize_preproduction_cache_in_worker(
    request: dict[str, Any], timing_plan: GemmaShotTimingPlan,
    progress_callback: Any = None,
) -> int:
    """Build a fresh clean cache without rerunning the replayed timing plan."""
    worker_request = json.loads(json.dumps(request, ensure_ascii=False))
    worker_request["operation"] = "preproduction_cache"
    worker_request["timing_plan"] = _timing_plan_payload(timing_plan)
    process = _start_worker_process()
    stdout = _stream_worker_output(process, worker_request, progress_callback)

    result_line = next(
        (line[len(_WORKER_RESULT_PREFIX):] for line in reversed(stdout.splitlines())
         if line.startswith(_WORKER_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise Gemma4WorkerExitError(
            f"Gemma 4 cache worker exited with status {process.returncode} without returning a result",
            returncode=process.returncode,
        )
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as error:
        raise Gemma4ObservationError("Gemma 4 cache worker returned malformed result JSON") from error
    if not result.get("ok"):
        message = str(result.get("message") or "unknown worker failure")
        if result.get("error_type") == "Gemma4DependencyError":
            raise Gemma4DependencyError(message)
        if result.get("error_type") in {"Gemma4MTPError", "Gemma4MTPOutputError"}:
            worker_error_type = str(result["error_type"])
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type=worker_error_type,
            )
        raise Gemma4ObservationError(message)
    if process.returncode != 0:
        raise Gemma4WorkerExitError(
            f"Gemma 4 cache worker exited with status {process.returncode}",
            returncode=process.returncode,
        )
    try:
        state_bytes = int(result["cache_state_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise Gemma4ObservationError("Gemma 4 cache worker returned invalid cache state size") from error
    if state_bytes <= 0:
        raise Gemma4ObservationError("Gemma 4 cache worker did not produce a cache state")
    return state_bytes


def _clear_previous_debug_captures():
    """Delete only disposable sampler directories from earlier runs."""
    root = tempfile.gettempdir()
    removed = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.startswith(GEMMA4_OWNED_TEMP_PREFIX):
                    continue
                # The prefix is owned by this node. Never follow a symlink
                # during cleanup, even when it happens to use that prefix.
                if entry.is_symlink():
                    os.unlink(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    continue
                removed += 1
    except OSError as error:
        logging.warning("HR Endless Sampler could not clear old Gemma debug captures in %s: %s", root, error)
        return
    if removed:
        logging.info("HR Endless Sampler removed %d old Gemma debug capture directories.", removed)


class Gemma4ContinuityDirector:
    """Preproduction timing planner plus one-shot local prompt director."""

    def __init__(self, debug: bool = False, gemma4_mtp: bool = False, seed: int = 0,
                 capture_directory: str | Path | None = None,
                 observation_image_directory: str | Path | None = None):
        self.debug = debug
        self.gemma4_mtp = bool(gemma4_mtp)
        self.seed = int(seed) & 0x7fffffff
        self._capture_sequence = 0
        self.last_system_prompt = ""
        self.last_observation_prompt = ""
        self.last_timing_system_prompt = ""
        self.last_timing_planning_prompt = ""
        self.capture_directory = None
        self.observation_image_directory = None
        if observation_image_directory is not None:
            self.observation_image_directory = Path(observation_image_directory)
            self.observation_image_directory.mkdir(parents=True, exist_ok=True)
        # Automatic captures are useful only for the active render. Clear an
        # earlier run before this director starts, even when this render's
        # Debug switch is off and therefore creates no replacement capture.
        if capture_directory is None:
            _clear_previous_debug_captures()
        if debug:
            if capture_directory is None:
                capture_directory = tempfile.mkdtemp(prefix=GEMMA4_DEBUG_CAPTURE_PREFIX)
            self.capture_directory = Path(capture_directory)
            self.capture_directory.mkdir(parents=True, exist_ok=True)
            logging.info("HR Endless Sampler Gemma capture directory: %s", self.capture_directory)

    def _run_worker_with_mtp_fallback(
        self,
        operation: str,
        request: dict[str, Any],
        worker: Any,
    ) -> Any:
        """Retry disposable-worker crashes without sacrificing the render.

        Native llama.cpp aborts cannot be caught inside the child process. The
        worker is disposable, however, so the parent preserves a JSON copy of
        the exact request and repeats the same operation in fresh processes.
        After an MTP failure all operation-local retries use the original
        non-MTP decoder. This changes only each retry's MTP flag: cache and
        request fields are retained, and the next independent Gemma operation
        still attempts MTP normally.
        """
        preserved_request = json.loads(json.dumps(request, ensure_ascii=False))
        attempted_mtp = bool(preserved_request.get("gemma4_mtp", False))
        use_mtp = attempted_mtp
        retries_used = 0

        while True:
            attempt_request = json.loads(json.dumps(preserved_request, ensure_ascii=False))
            attempt_request["gemma4_mtp"] = use_mtp
            try:
                return worker(attempt_request)
            except Gemma4WorkerExitError as error:
                if retries_used >= GEMMA4_WORKER_RETRY_LIMIT:
                    logging.error(
                        "HR Endless Sampler Gemma 4 worker failed during %s after %d fresh "
                        "worker retries; the render cannot continue: %s",
                        operation,
                        retries_used,
                        error,
                    )
                    raise
                retries_used += 1
                failed_with_mtp = use_mtp
                use_mtp = False
                status = error.worker_error_type or (
                    f"status {error.returncode}"
                    if error.returncode is not None
                    else type(error).__name__
                )
                logging.warning(
                    "HR Endless Sampler Gemma 4 %s worker failed during %s (%s): %s. "
                    "Retrying the exact operation in a fresh original non-MTP worker "
                    "(retry %d/%d); the next independent Gemma operation will try MTP "
                    "according to the sampler toggle again.",
                    "native MTP" if failed_with_mtp else "non-MTP retry",
                    operation,
                    status,
                    error,
                    retries_used,
                    GEMMA4_WORKER_RETRY_LIMIT,
                )
            except Gemma4ObservationError as error:
                # A long-running ComfyUI parent can briefly coexist with a
                # worker launched from a newer on-disk gemma4.py. Older parent
                # decoding may preserve the precise MTP failure message while
                # losing its newly introduced exception subtype. Treat only
                # this exact MTP transport failure as retryable; creative and
                # schema ObservationErrors must remain visible and must never
                # silently switch decoders.
                if not (
                    use_mtp
                    and str(error).startswith(
                        "Gemma 4 MTP returned no complete JSON object"
                    )
                ):
                    raise
                if retries_used >= GEMMA4_WORKER_RETRY_LIMIT:
                    logging.error(
                        "HR Endless Sampler Gemma 4 MTP output failed during %s after %d "
                        "fresh worker retries; the render cannot continue: %s",
                        operation,
                        retries_used,
                        error,
                    )
                    raise
                retries_used += 1
                use_mtp = False
                logging.warning(
                    "HR Endless Sampler Gemma 4 native MTP worker returned no complete JSON "
                    "during %s: %s. Retrying the exact operation in a fresh original "
                    "non-MTP worker (retry %d/%d); the next independent Gemma operation "
                    "will try MTP according to the sampler toggle again.",
                    operation,
                    error,
                    retries_used,
                    GEMMA4_WORKER_RETRY_LIMIT,
                )

    def plan_timing(self, request: dict[str, Any], progress_callback: Any = None) -> GemmaShotTimingPlan:
        """Create the immutable Gemma action schedule before any H3 chunk runs."""
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if not request.get("source_shots"):
            raise Gemma4ObservationError("Gemma 4 needs source shots for timing preproduction")
        self.last_timing_system_prompt, self.last_timing_planning_prompt = _render_timing_plan_messages(request)
        request["debug"] = self.debug
        request["gemma4_mtp"] = self.gemma4_mtp
        request["gemma4_seed"] = self.seed
        result = self._run_worker_with_mtp_fallback(
            "shot-timing preproduction",
            request,
            (
                (lambda payload: _plan_timing_in_worker(payload))
                if progress_callback is None
                else (lambda payload: _plan_timing_in_worker(payload, progress_callback))
            ),
        )
        self.last_timing_system_prompt = result.system_prompt or self.last_timing_system_prompt
        self.last_timing_planning_prompt = result.planning_prompt or self.last_timing_planning_prompt
        corrected_attempts = [attempt for attempt in result.attempts if attempt.validation_warnings]
        if corrected_attempts:
            logging.warning(
                "HR Endless Sampler Gemma 4 preproduction needed %d model-authored correction(s); "
                "only the affected global or source-shot plan was retried.",
                len(corrected_attempts),
            )
        return result

    def materialize_preproduction_cache(
        self, request: dict[str, Any], timing_plan: GemmaShotTimingPlan,
        progress_callback: Any = None,
    ) -> int:
        """Recreate the clean pre-Chunk-1 KV state from a replayed plan."""
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if not request.get("preproduction_cache"):
            raise Gemma4ObservationError("Gemma preproduction cache is not configured for this replay")
        request["debug"] = self.debug
        request["gemma4_mtp"] = self.gemma4_mtp
        request["gemma4_seed"] = self.seed
        return self._run_worker_with_mtp_fallback(
            "clean preproduction-cache creation",
            request,
            (
                (lambda payload: _materialize_preproduction_cache_in_worker(payload, timing_plan))
                if progress_callback is None
                else (
                    lambda payload: _materialize_preproduction_cache_in_worker(
                        payload, timing_plan, progress_callback
                    )
                )
            ),
        )

    def direct(
        self,
        request: dict[str, Any],
        frames: torch.Tensor | None = None,
        progress_callback: Any = None,
    ) -> GemmaChunkPrompt:
        request = json.loads(json.dumps(request, ensure_ascii=False))
        chunk_number = int(request["chunk_number"])
        # Retain the parent-side rendering too, so a worker/dependency failure
        # can still leave the exact intended request in the last-run transcript.
        self.last_system_prompt, self.last_observation_prompt = _render_observation_messages(request)
        frame_numbers = [int(item) for item in request.get("observation_frame_numbers", ())]
        if frames is None:
            if frame_numbers:
                raise Gemma4ObservationError("Gemma request has frame numbers but no decoded observation frames")
            image_urls = []
        else:
            if frames.ndim != 4:
                raise Gemma4ObservationError("Gemma 4 observation frames must be an NHWC image batch")
            if frames.shape[0] != len(frame_numbers):
                raise Gemma4ObservationError("Gemma observation frame numbers do not match the decoded image count")
            image_urls = [_image_data_url(frame) for frame in frames]
        if not request.get("target_shots"):
            raise Gemma4ObservationError("Gemma 4 needs at least one source shot for the current chunk")
        request["image_urls"] = image_urls
        request["debug"] = self.debug
        request["gemma4_mtp"] = self.gemma4_mtp
        request["gemma4_seed"] = self.seed
        if self.observation_image_directory is not None and image_urls:
            try:
                _capture_last_observation_images(
                    self.observation_image_directory,
                    chunk_number,
                    frame_numbers,
                    image_urls,
                )
            except (OSError, Gemma4ObservationError, ValueError) as error:
                logging.warning(
                    "HR Endless Sampler could not save last-run Gemma images to %s: %s",
                    self.observation_image_directory,
                    error,
                )
        capture_dir = None
        if self.capture_directory is not None:
            self._capture_sequence += 1
            capture_dir = _capture_observation_request(
                self.capture_directory,
                self._capture_sequence,
                request,
            )
        try:
            result = self._run_worker_with_mtp_fallback(
                f"Chunk {chunk_number} prompt directing",
                request,
                (
                    (lambda payload: _observe_in_worker(payload))
                    if progress_callback is None
                    else (lambda payload: _observe_in_worker(payload, progress_callback))
                ),
            )
        except BaseException as error:
            if capture_dir is not None:
                _write_capture_error(capture_dir, error)
            raise
        if capture_dir is not None:
            _write_capture_result(capture_dir, result)
            logging.info("HR Endless Sampler saved Gemma fixture: %s", capture_dir)
        if len(result.attempts) > 1:
            initial_contract_warnings = _contract_validation_warnings(result.attempts[0].validation_warnings)
            if initial_contract_warnings:
                logging.warning(
                    "HR Endless Sampler Gemma 4 initial response for chunk %d violated "
                    "the H3 global-label/local-time marker or current-slice coverage contract; requested one Gemma-generated correction and will use "
                    "that response:\n- %s",
                    chunk_number,
                    "\n- ".join(initial_contract_warnings),
                )
        self.last_system_prompt = result.system_prompt or self.last_system_prompt
        self.last_observation_prompt = result.observation_prompt or self.last_observation_prompt
        if result.validation_warnings:
            logging.warning(
                "HR Endless Sampler Gemma 4 response for chunk %d has validation warning(s); "
                "using Gemma's detailed_description unchanged:\n- %s",
                chunk_number,
                "\n- ".join(result.validation_warnings),
            )
        return result


def _worker_main() -> int:
    try:
        request = json.load(sys.stdin)
        _configure_raw_output_log(request)
        if request.get("operation") == "timing_plan":
            timing_plan = _plan_timing_in_process(
                request=request,
                debug=bool(request.get("debug", False)),
            )
            result = {"ok": True, "timing_plan": _timing_plan_payload(timing_plan)}
        elif request.get("operation") == "preproduction_cache":
            timing_plan = _timing_plan_from_payload(request["timing_plan"])
            state_bytes = _materialize_preproduction_cache_in_process(
                request=request,
                timing_plan=timing_plan,
                debug=bool(request.get("debug", False)),
            )
            result = {"ok": True, "cache_state_bytes": state_bytes}
        else:
            chunk_prompt = _observe_in_process(
                request=request,
                image_urls=[str(item) for item in request["image_urls"]],
                debug=bool(request["debug"]),
            )
            result = {"ok": True, "chunk_prompt": _chunk_prompt_payload(chunk_prompt)}
    except Exception as error:
        result = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        raw_json = getattr(error, "raw_json", "")
        if raw_json:
            result["raw_json"] = raw_json
    print(_WORKER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        raise SystemExit(_worker_main())
    raise SystemExit("gemma4.py is an internal worker; use it through the sampler node")
