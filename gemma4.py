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
import gc
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

import comfy.model_management
import folder_paths


GEMMA4_REPOSITORY = "google/gemma-4-12B-it-qat-q4_0-gguf"
GEMMA4_MODEL_FILENAME = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA4_MMPROJ_FILENAME = "mmproj-gemma-4-12b-it-qat-q4_0.gguf"
GEMMA4_MTP_REPOSITORY = "Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF"
GEMMA4_MTP_FILENAME = "gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"
GEMMA4_MODEL_DIRECTORY = "llama_cpp/gemma-4-12b-it-qat-q4_0"
GEMMA4_REQUIRED_VERSION = "0.3.35"
GEMMA4_IMAGE_MIN_TOKENS = 70
GEMMA4_IMAGE_MAX_TOKENS = 1120
GEMMA4_BATCH_SIZE = GEMMA4_IMAGE_MAX_TOKENS
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
                    # A 16K F16 Gemma KV snapshot is already around 5 GiB.
                    # Do not choose a tiny container /dev/shm and fail later
                    # when a normal temporary directory can hold it instead.
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
                    entry_start = shot.shot_start_frame + entry.start_frame
                    entry_end = shot.shot_start_frame + entry.end_frame
                    overlap_start = max(target_start, entry_start)
                    overlap_end = min(target_end, entry_end)
                    if overlap_start >= overlap_end:
                        continue
                    item: dict[str, Any] = {
                        "id": self._beat_identifier(shot.source_shot, kind, index),
                        "kind": "visual" if kind == "V" else "overlay",
                        "source_shot": shot.source_shot,
                        "source_start_frame": entry.start_frame,
                        "source_end_frame": entry.end_frame,
                        "overlap_start_frame": overlap_start - shot.shot_start_frame,
                        "overlap_end_frame": overlap_end - shot.shot_start_frame,
                        "action": entry.action if kind == "V" else entry.content,
                    }
                    if kind == "O":
                        item["overlay_type"] = entry.overlay_type
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
            else:
                lines.append("Concurrent overlays: none planned.")
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
                phase = "begins" if item["source_start_frame"] >= item["overlap_start_frame"] else "continues"
                descriptor = item["kind"]
                if item["kind"] == "overlay":
                    descriptor += f"/{item['overlay_type']}"
                required_lines.append(
                    f"- Required now [{item['id']}], {descriptor}, source-relative frames "
                    f"{item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}: "
                    f"this planned beat {phase} here — {item['action']}"
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
            matching = [item for item in self.mandatory_coverage((target,)) if item["source_shot"] == shot_number]
            if not matching:
                continue
            lines = [
                f"Source Shot {shot_number}: this chunk retains source-relative frames "
                f"{int(target['target_start']) - int(target['shot_start'])}-"
                f"{int(target['target_end']) - int(target['shot_start']) - 1}."
            ]
            for item in matching:
                phase = "begins" if item["source_start_frame"] >= item["overlap_start_frame"] else "continues"
                descriptor = item["kind"]
                if item["kind"] == "overlay":
                    descriptor += f"/{item['overlay_type']}"
                lines.append(
                    f"- Required now [{item['id']}], {descriptor}, source-relative frames "
                    f"{item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}: "
                    f"this planned beat {phase} here — {item['action']}"
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


def is_official_gemma4_pair(model_path: Path, mmproj_path: Path) -> bool:
    expected_model, expected_mmproj = _model_paths()
    return model_path.resolve() == expected_model.resolve() and mmproj_path.resolve() == expected_mmproj.resolve()


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


def _model_files_for_request(request: dict[str, Any]) -> tuple[Path, Path]:
    model_value = request.get("director_model_path")
    mmproj_value = request.get("director_mmproj_path")
    if model_value is None and mmproj_value is None:
        return _ensure_model_files()
    if not model_value or not mmproj_value:
        raise Gemma4DependencyError("A local director requires both model and mmproj files")
    models_root = Path(folder_paths.models_dir).resolve()
    model_path = Path(model_value).resolve()
    mmproj_path = Path(mmproj_value).resolve()
    for path in (model_path, mmproj_path):
        if not path.is_relative_to(models_root) or path.suffix.casefold() != ".gguf" or not path.is_file():
            raise Gemma4DependencyError(f"Invalid local director file: {path}")
    return model_path, mmproj_path


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


def _gemma4_mtmd_handler_type(base_handler, llama_cpp_module, suppress_output):
    """Expose Gemma 4's visual-token budget missing from the pinned handler.

    llama-cpp-python 0.3.35 binds ``image_min_tokens`` and
    ``image_max_tokens`` in ``mtmd_context_params``, but its public
    ``MTMDChatHandler`` constructor does not expose them. Keep this narrow
    subclass local to the disposable worker instead of modifying the installed
    package.
    """

    class Gemma4MTMDChatHandler(base_handler):
        def _init_mtmd_context(self, llama_model):
            self.verbose = llama_model.verbose
            if self.mtmd_ctx is not None:
                return

            with suppress_output(disable=self.verbose):
                ctx_params = self._mtmd_cpp.mtmd_context_params_default()
                ctx_params.use_gpu = self.use_gpu
                ctx_params.print_timings = self.verbose
                ctx_params.n_threads = llama_model.n_threads
                ctx_params.flash_attn_type = (
                    llama_cpp_module.LLAMA_FLASH_ATTN_TYPE_ENABLED
                    if (
                        llama_model.context_params.flash_attn_type
                        == llama_cpp_module.LLAMA_FLASH_ATTN_TYPE_ENABLED
                    )
                    else llama_cpp_module.LLAMA_FLASH_ATTN_TYPE_DISABLED
                )
                ctx_params.image_min_tokens = GEMMA4_IMAGE_MIN_TOKENS
                ctx_params.image_max_tokens = GEMMA4_IMAGE_MAX_TOKENS
                ctx_params.batch_max_tokens = max(
                    int(ctx_params.batch_max_tokens),
                    GEMMA4_IMAGE_MAX_TOKENS,
                )

                self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                    self.clip_model_path.encode(), llama_model.model, ctx_params
                )

                if self.mtmd_ctx is None:
                    raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")
                if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                    raise ValueError("Vision is not supported by this model")

                def mtmd_free():
                    with suppress_output(disable=self.verbose):
                        if self.mtmd_ctx is not None:
                            self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                            self.mtmd_ctx = None

                llama_model._stack.callback(mtmd_free)

        def append_user_chat_completion(
            self,
            *,
            llama: Any,
            content: str | Sequence[dict[str, Any]],
            temperature: float = 0.0,
            top_p: float = 1.0,
            max_tokens: int = 1024,
            response_format: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Append one user turn to an existing Gemma MTMD conversation.

            ``MTMDChatHandler.__call__`` always resets Llama and clears the
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
            import jinja2
            from llama_cpp.llama_chat_format import (
                ImmutableSandboxedEnvironment,
                Jinja2ChatFormatter,
                _grammar_for_response_format,
            )

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
            image_urls = self.get_image_urls(messages)
            media_marker = self._mtmd_cpp.mtmd_default_marker().decode("utf-8")
            template_env = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                extensions=[
                    Jinja2ChatFormatter.IgnoreGenerationTags,
                    jinja2.ext.loopcontrols,
                ],
            )
            template_env.filters["tojson"] = Jinja2ChatFormatter.tojson
            template = template_env.from_string(self._get_chat_template(llama))

            def raise_exception(message: str):
                raise ValueError(message)

            text = template.render(
                messages=self._get_template_messages(messages, media_marker),
                add_generation_prompt=True,
                eos_token=self._decode_token_piece(llama.detokenize([llama.token_eos()])),
                bos_token=self._decode_token_piece(llama.detokenize([llama.token_bos()])),
                raise_exception=raise_exception,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None,
                strftime_now=Jinja2ChatFormatter.strftime_now,
            )
            text = self._postprocess_template_text(text, image_urls, media_marker)
            if text.count(anchor) != 1:
                raise ValueError("Could not derive an append-only Gemma chat-template turn")
            # Keep the template's exact assistant closing sequence, then the
            # new user turn and model generation prompt.  The rendered prefix
            # contains a synthetic assistant turn which is not evaluated.
            suffix = text.split(anchor, 1)[1]
            if not suffix:
                raise ValueError("Gemma chat template produced an empty append suffix")

            bitmaps = []
            try:
                for image_url in image_urls:
                    bitmaps.append(self._create_bitmap_from_bytes(self.load_image(image_url)))

                input_text = self._mtmd_cpp.mtmd_input_text()
                input_bytes = suffix.encode("utf-8")
                input_text.text = input_bytes
                input_text.text_len = len(input_bytes)
                # Unlike a fresh full prompt, the suffix must not add a BOS
                # token: it continues the restored/pre-existing conversation.
                input_text.add_special = False
                input_text.parse_special = True
                chunks = self._mtmd_cpp.mtmd_input_chunks_init()
                if chunks is None:
                    raise ValueError("Failed to create append input chunks")
                try:
                    bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                    result = self._mtmd_cpp.mtmd_tokenize(
                        self.mtmd_ctx,
                        chunks,
                        ctypes.byref(input_text),
                        bitmap_array,
                        len(bitmaps),
                    )
                    if result != 0:
                        raise ValueError(f"Failed to tokenize appended input: error code {result}")
                    for index in range(self._mtmd_cpp.mtmd_input_chunks_size(chunks)):
                        chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, index)
                        if chunk is None:
                            continue
                        chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                        if chunk_type == self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
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
                        elif chunk_type in (
                            self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                            self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                        ):
                            chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                            if llama.n_tokens + chunk_n_tokens > llama.n_ctx():
                                raise ValueError(
                                    f"Appended prompt exceeds n_ctx: {llama.n_tokens + chunk_n_tokens} > {llama.n_ctx()}"
                                )
                            new_n_past = llama_cpp_module.llama_pos(0)
                            result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                                self.mtmd_ctx,
                                llama._ctx.ctx,
                                chunk,
                                llama_cpp_module.llama_pos(llama.n_tokens),
                                llama_cpp_module.llama_seq_id(0),
                                llama.n_batch,
                                False,
                                ctypes.byref(new_n_past),
                            )
                            if result != 0:
                                raise ValueError(f"Failed to evaluate appended media: error code {result}")
                            llama.n_tokens = new_n_past.value
                finally:
                    self._mtmd_cpp.mtmd_input_chunks_free(chunks)
            finally:
                for bitmap in bitmaps:
                    self._mtmd_cpp.mtmd_bitmap_free(bitmap)

            grammar = None
            if response_format is not None and response_format.get("type") == "json_object":
                grammar = _grammar_for_response_format(response_format)
            completion = llama.create_completion(
                prompt=llama.input_ids[:llama.n_tokens].tolist(),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                grammar=grammar,
            )
            return {
                "choices": [{"message": {"content": completion["choices"][0]["text"]}}],
            }

    Gemma4MTMDChatHandler.__name__ = "Gemma4MTMDChatHandler"
    return Gemma4MTMDChatHandler


def _load_runtime(backend="gemma4"):
    director_name = "Qwen3.5" if backend == "qwen3.5" else "Gemma 4"
    try:
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
        from llama_cpp._utils import suppress_stdout_stderr
        import llama_cpp.mtmd_cpp
    except ImportError as error:
        raise Gemma4DependencyError(
            f"{director_name} continuity requires llama-cpp-python==0.3.35 with CUDA support. "
            "Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        ) from error

    version = getattr(llama_cpp, "__version__", "unknown")
    if version != GEMMA4_REQUIRED_VERSION:
        raise Gemma4DependencyError(
            f"{director_name} continuity requires llama-cpp-python=={GEMMA4_REQUIRED_VERSION} with MTMD vision support; "
            f"found {version}. Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        )
    if backend == "qwen3.5":
        return Llama, MTMDChatHandler
    return Llama, _gemma4_mtmd_handler_type(
        MTMDChatHandler,
        llama_cpp,
        suppress_stdout_stderr,
    )


def _create_runtime_llm(
    Llama: Any,
    *,
    model_path: Path,
    handler: Any,
    debug: bool,
    gemma4_mtp: bool = False,
    n_ctx: int = 16384,
    n_batch: int = GEMMA4_BATCH_SIZE,
) -> Any:
    """Create either the original runtime or a target born as native MTP."""
    real_runtime = str(getattr(Llama, "__module__", "")).startswith("llama_cpp")
    llama_kwargs = {
        "chat_handler": handler,
        "n_gpu_layers": -1,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ubatch": n_batch,
        "flash_attn": True,
        # Sampler debug controls our captures, validation warnings, progress,
        # and MTP summary.  It must not enable llama.cpp's native trace stream:
        # native verbosity prints for every one-token MTP assistant decode and
        # every hybrid-state restore, serialising the speculative hot loop on
        # stderr and repeatedly disturbing CUDA-graph execution.
        "verbose": False,
    }
    if gemma4_mtp and real_runtime:
        # Download before allocating the target. Unlike the retired adapter,
        # failure is fatal: an enabled comparison toggle must never silently
        # run the ordinary decoder and report it as MTP.
        mtp_path = _ensure_mtp_model_file()
        from gemma4_mtp import create_native_mtp_llama

        llm = create_native_mtp_llama(
            Llama,
            model_path=model_path,
            draft_model_path=mtp_path,
            num_pred_tokens=4,
            **llama_kwargs,
        )
        print(
            "HR Endless Sampler Gemma 4 decoding mode: native draft-mtp "
            "(spec-draft-n-max=4, fast device checkpoints; operation-local non-MTP retry enabled).",
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
        try:
            while True:
                try:
                    token = next(iterator) if first_iteration else iterator.send(send_value)
                except StopIteration:
                    return
                first_iteration = False
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
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, end = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, content[match.start():match.start() + end]
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
                lines.append(f"Required local H3 marker: {required_marker}")
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
    for local_index, shot in enumerate(shots, 1):
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
            f"- local [Shot {local_index}] / source Shot {int(shot['shot_number'])}: {source_position}; "
            f"this chunk must author its global frames {target_start}-{target_end - 1} "
            f"(physical local frames {target_local_start}-{target_local_end})."
        )
    return "\n".join(lines) if lines else "none"


def _slice_portion_name(shot: dict[str, Any], local_index: int) -> str:
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
    return f"{portion} local [Shot {local_index}] (source Shot {int(shot['shot_number'])})"


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
    for local_index, shot in enumerate(shots, 1):
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
            f"- local [Shot {local_index}] / source Shot {int(shot['shot_number'])}: full source shot "
            f"global frames {shot_start}-{shot_end - 1} ({shot_frames} frames, {shot_frames / fps:.3f} s). "
            f"This chunk owns its {_slice_portion_kind(shot)} {target_frames} frames: "
            f"source-relative frames {relative_start}-{relative_end} ({start_percent:.1f}%-{end_percent:.1f}% of the shot)."
        )
    lines.append(
        "In timing_plan, state the concrete events to cover now and the concrete later events to defer for every local shot. "
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
        _slice_portion_name(shot, local_index)
        for local_index, shot in enumerate(shots, 1)
    ])
    return (
        f"Write one complete chunk-local detailed_description for {portions}. "
        f"All requested action belongs to physical global timeslice {sampled_start}-{sampled_end - 1} inclusive, "
        f"physical chunk-local timeslice 0-{physical_frames - 1} ({physical_frames} frames). "
        f"Author the retained output interval global frames {output_start}-{output_end - 1}, "
        f"physical local frames {output_local_start}-{output_local_end}; do not restage completed opening "
        f"conditioning frames outside that retained interval."
    )


def _required_local_markers(shots: Sequence[dict[str, Any]]) -> str:
    """Render literal H3-local marker tokens, separate from global source prose.

    The complete source prompt necessarily contains its original full-video
    timecodes.  This small copy-only block makes the sampler-calculated local
    tokens visually and semantically distinct, so Gemma need not infer which
    clock H3 expects for this physical window.
    """
    marked_shots = [shot for shot in shots if shot.get("required_marker")]
    lines = [
        "IMMUTABLE H3-LOCAL SHOT MARKERS — COPY THE QUOTED TOKEN EXACTLY",
        "These markers use H3's physical chunk-local clock, whose zero is the first sampled frame.",
        "For detailed_description, write every quoted token exactly once and in this order; do not copy the leading hyphen or quotes.",
        "Original source/global timestamps elsewhere are context only and are forbidden as H3 markers unless they are identical to the token below.",
    ]
    if marked_shots:
        lines.extend(f'- exact token: "{str(shot["required_marker"])}"' for shot in marked_shots)
    else:
        lines.append(
            "There is no H3 shot marker in this physical chunk: it begins inside an already established source shot. "
            "Begin detailed_description as plain continuation prose; do not add [Shot 1] or any other [Shot N] marker."
        )
    return "\n".join(lines)


def _marker_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return only errors that a focused local-marker rewrite can repair."""
    return tuple(warning for warning in warnings if "marker" in warning.lower())


def _contract_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return contract findings that require a model-authored retry."""
    return tuple(
        warning for warning in warnings
        if (
            "marker" in warning.lower()
            or "mandatory coverage" in warning.lower()
            or "dialogue speaker form" in warning.lower()
        )
    )


def _normalized_prompt_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


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
                        f"Gemma 4 mandatory coverage {beat_id} requires its exact dialogue in detailed_description"
                    )

    for beat_id in records.keys() - required.keys():
        warnings.append(f"Gemma 4 mandatory coverage includes unknown beat {beat_id}")
    return warnings


def _chunk_contract_correction_request(request: dict[str, Any], warnings: Sequence[str]) -> str:
    """Ask Gemma itself for one full marker/coverage correction, never a patch."""
    shots = request["target_shots"]
    mandatory = request.get("mandatory_coverage", ())
    required_coverage = "\n".join(
        f"- {item['id']} ({item['kind']}{'/' + str(item['overlay_type']) if item.get('overlay_type') else ''}): "
        f"source-relative frames {item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}; {item['action']}"
        for item in mandatory
        if isinstance(item, dict) and item.get("id")
    ) or "- No current-slice coverage entries are required."
    speaker_forms = "\n".join(
        f"- {subject} ({speaker_id}) must introduce <d>{dialogue}</d>"
        for dialogue, subject, speaker_id in _mapped_dialogue_speaker_requirements(request)
    ) or "- No mapped dialogue speaker form is required."
    return (
        "CHUNK CONTRACT CORRECTION REQUIRED\n"
        "Your immediately preceding JSON violated the H3 local-marker or mandatory current-slice coverage contract. "
        "Return one complete replacement JSON object "
        "with all six required fields, not an explanation and not a textual patch. Keep the same current-frame-slice "
        "creative intent and continuity reasoning, but rewrite detailed_description and coverage so every current beat "
        "is explicitly started or continued now. A current beat may never be marked deferred. For dialogue coverage, "
        "include the exact <d>...</d> line in detailed_description now. A mapped visual speaker must use the immediate "
        "official form <Subject N> (Sx) before that line, not Name (<Subject N>) (Sx). Use evidence copied exactly "
        "from your rewritten detailed_description.\n\nMapped dialogue speaker form:\n"
        + speaker_forms
        + "\n\nH3 local marker contract:\n"
        f"{_required_local_markers(shots)}\n\n"
        "Mandatory current-slice coverage:\n"
        + required_coverage
        + "\n\nThe source/global timestamps in the original prompt describe the full video and must not be used as markers "
        "inside this physical chunk. Do not add, remove, renumber, or move cuts.\n\n"
        "Detected validation errors in the preceding JSON:\n"
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
    return tuple(requirements)


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
    if not end_state:
        warnings.append("Gemma 4 response has no usable end_state after legacy extraction")
    if len(description) > 12000:
        warnings.append("Gemma 4 detailed_description is unexpectedly long")
    if re.search(r"\b(?:detailed_description|overall_soundscape|non_diegetic_music|subject_definitions|summary|retention_analysis)\s*:", description, re.IGNORECASE):
        warnings.append("Gemma 4 returned a structured field label inside detailed_description")

    expected = [shot for shot in request["target_shots"] if shot.get("required_marker")]
    markers = list(_SHOT_MARKER.finditer(description))
    if len(markers) != len(expected):
        warnings.append(
            f"Gemma 4 returned {len(markers)} shot markers; this chunk requires {len(expected)}"
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
        if not text or not any(text in source or source in text for source in source_dialogue):
            warnings.append("Gemma 4 modified or invented dialogue instead of preserving source words")
    warnings.extend(_dialogue_speaker_form_warnings(request, description))
    warnings.extend(_mandatory_coverage_warnings(value, request, description))

    return GemmaChunkPrompt(
        confidence,
        analysis.strip(),
        description,
        raw_json,
        timing_plan,
        end_state,
        system_prompt,
        observation_prompt,
        tuple(dict.fromkeys(warnings)),
    )


def _chunk_prompt_payload(result: GemmaChunkPrompt) -> dict[str, Any]:
    return {
        "confidence": result.confidence,
        "analysis": result.analysis,
        "detailed_description": result.detailed_description,
        "raw_json": result.raw_json,
        "timing_plan": result.timing_plan,
        "end_state": result.end_state,
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
                    }
                    for overlay in shot.overlays
                ],
            }
            for shot in result.shots
        ],
        "raw_json": result.raw_json,
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
            )
            for overlay in item.get("overlays", ())
            if isinstance(overlay, dict)
        )
        shots.append(
            GemmaShotTimingShot(
                source_shot=int(item["source_shot"]),
                shot_start_frame=int(item["shot_start_frame"]),
                shot_end_frame=int(item["shot_end_frame"]),
                visual_beats=visual_beats,
                overlays=overlays,
            )
        )
    return GemmaShotTimingPlan(
        confidence=str(value.get("confidence", "unknown")),
        analysis=str(value.get("analysis", "")),
        shots=tuple(shots),
        character_name_table=character_name_table,
        raw_json=str(value.get("raw_json", "")),
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
            overlays.append(GemmaShotTimingOverlay(start, end, overlay_type, content.strip()))
        # Global source-shot boundaries are immutable sampler facts, not
        # generated timing content.  Do not require Gemma to echo a redundant
        # inclusive/exclusive endpoint field: that ambiguity cost a full
        # preproduction retry despite an otherwise valid action schedule.
        schedules.append(
            GemmaShotTimingShot(
                source_shot,
                expected_start,
                expected_end,
                tuple(visual_beats),
                tuple(overlays),
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


def _timing_plan_correction_request(request: dict[str, Any], error: Gemma4ObservationError) -> str:
    """Give Gemma one precise opportunity to repair its complete schedule."""
    expected = "\n".join(
        f"- Source Shot {int(shot['shot_number'])}: global frames {int(shot['shot_start'])}-{int(shot['shot_end']) - 1}; "
        f"visual_beats must exactly and contiguously cover source-relative frames 0-{int(shot['shot_end']) - int(shot['shot_start']) - 1}."
        for shot in request.get("source_shots", ())
    )
    return (
        "TIMING-PLAN CORRECTION REQUIRED\n"
        "Your immediately preceding JSON does not form a complete usable schedule. Return one complete replacement "
        "JSON object, not an explanation or patch. Preserve your intended action timing, but use this exact schema "
        "and make every shot's visual_beats contiguous, non-empty, source-relative half-open intervals "
        "[start_frame, end_frame). Overlays may overlap those visual intervals:\n"
        '{"confidence":"high|medium|low", "analysis":"...", '
        '"character_name_table":[{"character_name":"Heman", "subject":"<Subject 1>"}], '
        '"shots":[{"source_shot":1, '
        '"visual_beats":[{"start_frame":0, "end_frame":34, "action":"..."}], '
        '"overlays":[{"start_frame":4, "end_frame":20, "type":"dialogue|sound|action", "content":"..."}]}]}\n\n'
        "`character_name_table` must be an array. Preserve only explicit name-to-<Subject N> mappings "
        "from the original prompt; use [] when there are none.\n\n"
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
            "required_local_markers": _required_local_markers(target_shots),
            "previous_context": previous_context,
            "previous_shots": previous_shots,
            "frame_manifest": frame_manifest,
            "previous_gemma_description": previous_gemma_description,
            "previous_gemma_timing_plan": previous_gemma_timing_plan,
            "previous_gemma_end_state": previous_gemma_end_state,
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
            relative_end = end - int(shot["shot_start"]) - 1
            portions.append(
                f"Source Shot {int(shot['shot_number'])} source-relative frames {relative_start}-{relative_end}"
            )
        ownership = "; ".join(portions) if portions else "no retained source-shot frames"
        lines.append(
            f"- Chunk {index}: sampled global frames {sampled_start}-{sampled_end - 1}; "
            f"retains global frames {output_start}-{output_end - 1}; owns {ownership}."
        )
    return "\n".join(lines) if lines else "none"


def _render_timing_plan_messages(request: dict[str, Any]) -> tuple[str, str]:
    """Render the text-only preproduction request sent before Chunk 1."""
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
            "source_shots": _preproduction_source_shots(source_shots, fps),
            "physical_chunk_map": _preproduction_chunk_map(request.get("chunks", ()), source_shots),
            "preproduction_memory": timing_plan.for_target_shots(source_shots, fps),
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


def _gemma_chat_json(llm: Any, messages: Sequence[dict[str, Any]], *, max_tokens: int = 1024) -> tuple[dict[str, Any], str]:
    """Run a fast instructed-JSON completion, constraining only recovery.

    llama.cpp's JSON grammar scans Gemma 4's very large vocabulary once per
    output token.  That CPU work dominated the MTP path and kept the GPU idle.
    Gemma is already given an explicit JSON contract, so use ordinary decoding
    first and validate the text afterward.  A malformed response gets one
    slower grammar-constrained retry from the same self-contained messages.
    """

    def complete(response_format: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "messages": list(messages),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = llm.create_chat_completion(**kwargs)
        choice = response["choices"][0]["message"]
        text = choice.get("content") or ""
        if not isinstance(text, str):
            raise Gemma4ObservationError("Gemma 4 returned no textual response")
        return text

    text = complete()
    try:
        return _extract_json_object(text)
    except Gemma4ObservationError as error:
        logging.warning(
            "HR Endless Sampler Gemma 4 returned malformed instructed JSON; "
            "retrying this response with llama.cpp's slower JSON grammar: %s",
            error,
        )
        return _extract_json_object(complete({"type": "json_object"}))


def _gemma_append_chat_json(handler: Any, llm: Any, content: str | Sequence[dict[str, Any]], *,
                             max_tokens: int = 1024) -> tuple[dict[str, Any], str]:
    """Ask the next user turn, constraining only malformed-JSON recovery."""
    append = getattr(handler, "append_user_chat_completion", None)
    if not callable(append):
        raise Gemma4ObservationError("Gemma runtime does not support append-only chat turns")

    def complete(next_content: str | Sequence[dict[str, Any]], response_format: dict[str, Any] | None = None) -> str:
        response = append(
            llama=llm,
            content=next_content,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        choice = response["choices"][0]["message"]
        text = choice.get("content") or ""
        if not isinstance(text, str):
            raise Gemma4ObservationError("Gemma 4 returned no textual response")
        return text

    text = complete(content)
    try:
        return _extract_json_object(text)
    except Gemma4ObservationError as error:
        logging.warning(
            "HR Endless Sampler Gemma 4 returned malformed instructed JSON in an appended turn; "
            "asking it to replace that response under the slower JSON grammar: %s",
            error,
        )
        repair = (
            "Your immediately preceding response was not a valid complete JSON object. "
            "Return the same answer again as one complete JSON object only, preserving all requested fields and content."
        )
        return _extract_json_object(complete(repair, {"type": "json_object"}))


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
    Llama, MTMDChatHandler = _load_runtime(request.get("director_backend", "gemma4"))
    model_path, mmproj_path = _model_files_for_request(request)
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
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=True)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            n_ctx=int(request.get("director_n_ctx", 16384)),
            n_batch=int(request.get("director_n_batch", GEMMA4_BATCH_SIZE)),
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
                payload, raw_json = _gemma_append_chat_json(handler, llm, content)
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
                payload, raw_json = _gemma_chat_json(llm, initial_messages)
        else:
            payload, raw_json = _gemma_chat_json(llm, initial_messages)
        latest_raw_json = raw_json
        try:
            initial = _validate_chunk_prompt(
                payload,
                request,
                raw_json,
                system_prompt=system_prompt,
                observation_prompt=message,
            )
            attempts = [
                GemmaPromptAttempt(
                    kind="initial response",
                    raw_json=initial.raw_json,
                    validation_warnings=initial.validation_warnings,
                )
            ]
            contract_warnings = _contract_validation_warnings(initial.validation_warnings)
            if not contract_warnings:
                return replace(initial, attempts=tuple(attempts))

            correction_prompt = _chunk_contract_correction_request(request, contract_warnings)
            if callable(getattr(handler, "append_user_chat_completion", None)):
                # This is a genuine second chat turn.  Its user text is tiny
                # and the initial response is already in KV, so the correction
                # does not re-encode the source prompt or observation images.
                corrected_payload, corrected_raw_json = _gemma_append_chat_json(
                    handler, llm, correction_prompt
                )
            else:
                # Preserve compatibility with a deliberately minimal mocked
                # runtime used by unit tests and with an unexpectedly older
                # llama-cpp install. Real supported workers take the path
                # above.
                correction_messages = [
                    *initial_messages,
                    {"role": "assistant", "content": initial.raw_json},
                    {"role": "user", "content": correction_prompt},
                ]
                corrected_payload, corrected_raw_json = _gemma_chat_json(llm, correction_messages)
            latest_raw_json = corrected_raw_json
            corrected = _validate_chunk_prompt(
                corrected_payload,
                request,
                corrected_raw_json,
                system_prompt=system_prompt,
                observation_prompt=message,
            )
            attempts.append(
                GemmaPromptAttempt(
                    kind="chunk-contract correction response",
                    raw_json=corrected.raw_json,
                    validation_warnings=corrected.validation_warnings,
                    correction_prompt=correction_prompt,
                )
            )
            return replace(corrected, attempts=tuple(attempts))
        except Gemma4ObservationError as error:
            raise Gemma4ObservationError(str(error), raw_json=latest_raw_json) from error
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
    Llama, MTMDChatHandler = _load_runtime(request.get("director_backend", "gemma4"))
    model_path, mmproj_path = _model_files_for_request(request)
    handler = None
    llm = None
    try:
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=True)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            n_ctx=int(request.get("director_n_ctx", 16384)),
            n_batch=int(request.get("director_n_batch", GEMMA4_BATCH_SIZE)),
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
    """Run the one-time text-only source-shot timing plan in the worker."""
    Llama, MTMDChatHandler = _load_runtime(request.get("director_backend", "gemma4"))
    model_path, mmproj_path = _model_files_for_request(request)
    system_prompt, message = _render_timing_plan_messages(request)
    handler = None
    llm = None
    try:
        # Keep the official Gemma multimodal chat handler even though this pass
        # has no images.  It supplies the same model-specific conversation
        # formatting as the later image-and-text requests.
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=True)
        llm = _create_runtime_llm(
            Llama,
            model_path=model_path,
            handler=handler,
            debug=debug,
            gemma4_mtp=bool(request.get("gemma4_mtp", False)),
            n_ctx=int(request.get("director_n_ctx", 16384)),
            n_batch=int(request.get("director_n_batch", GEMMA4_BATCH_SIZE)),
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        latest_raw_json = ""
        payload, raw_json = _gemma_chat_json(llm, initial_messages, max_tokens=2048)
        latest_raw_json = raw_json
        try:
            initial = _validate_timing_plan(
                payload,
                request,
                raw_json,
                system_prompt=system_prompt,
                planning_prompt=message,
            )
            attempts = [GemmaPromptAttempt(kind="initial response", raw_json=initial.raw_json)]
            result = replace(initial, attempts=tuple(attempts))
        except Gemma4ObservationError as error:
            correction_prompt = _timing_plan_correction_request(request, error)
            if callable(getattr(handler, "append_user_chat_completion", None)):
                corrected_payload, corrected_raw_json = _gemma_append_chat_json(
                    handler, llm, correction_prompt, max_tokens=2048
                )
            else:
                correction_messages = [
                    *initial_messages,
                    {"role": "assistant", "content": latest_raw_json},
                    {"role": "user", "content": correction_prompt},
                ]
                corrected_payload, corrected_raw_json = _gemma_chat_json(
                    llm, correction_messages, max_tokens=2048
                )
            latest_raw_json = corrected_raw_json
            try:
                corrected = _validate_timing_plan(
                    corrected_payload,
                    request,
                    corrected_raw_json,
                    system_prompt=system_prompt,
                    planning_prompt=message,
                )
            except Gemma4ObservationError as corrected_error:
                raise Gemma4ObservationError(str(corrected_error), raw_json=latest_raw_json) from corrected_error
            attempts = (
                GemmaPromptAttempt(
                    kind="initial response",
                    raw_json=raw_json,
                    validation_warnings=(str(error),),
                ),
                GemmaPromptAttempt(
                    kind="timing-plan correction response",
                    raw_json=corrected.raw_json,
                    correction_prompt=correction_prompt,
                ),
            )
            result = replace(corrected, attempts=attempts)

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
    comfy_root = str(Path(folder_paths.__file__).resolve().parent)
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        comfy_root if not current_pythonpath else comfy_root + os.pathsep + current_pythonpath
    )
    return environment


def _stream_worker_output(
    process: subprocess.Popen,
    request: dict[str, Any],
    progress_callback: Any = None,
) -> str:
    """Send one request and consume progress records while the worker runs."""
    if process.stdin is None or process.stdout is None:
        raise Gemma4ObservationError("Gemma 4 worker pipes were not created")
    output: list[str] = []
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
        process.kill()
        process.wait()
        raise
    return "".join(output)


def _observe_in_worker(request: dict[str, Any], progress_callback: Any = None) -> GemmaChunkPrompt:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_worker_environment(),
    )
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
        if result.get("error_type") == "Gemma4MTPError":
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type="Gemma4MTPError",
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
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_worker_environment(),
    )
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
        if result.get("error_type") == "Gemma4MTPError":
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type="Gemma4MTPError",
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
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_worker_environment(),
    )
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
        if result.get("error_type") == "Gemma4MTPError":
            raise Gemma4WorkerExitError(
                message,
                returncode=process.returncode,
                worker_error_type="Gemma4MTPError",
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


class Gemma4ContinuityDirector:
    """Preproduction timing planner plus one-shot local prompt director."""

    def __init__(self, debug: bool = False, gemma4_mtp: bool = False,
                 capture_directory: str | Path | None = None,
                 observation_image_directory: str | Path | None = None,
                 model_path: str | Path | None = None,
                 mmproj_path: str | Path | None = None):
        self.debug = debug
        self.gemma4_mtp = bool(gemma4_mtp)
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.mmproj_path = Path(mmproj_path).resolve() if mmproj_path is not None else None
        if (self.model_path is None) != (self.mmproj_path is None):
            raise Gemma4DependencyError("A local Gemma director requires both model and mmproj files")
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
        if debug:
            if capture_directory is None:
                capture_directory = tempfile.mkdtemp(prefix="hr-endless-sampler-gemma4-")
            self.capture_directory = Path(capture_directory)
            self.capture_directory.mkdir(parents=True, exist_ok=True)
            logging.info("HR Endless Sampler Gemma capture directory: %s", self.capture_directory)

    def _configure_request(self, request: dict[str, Any]) -> None:
        request["debug"] = self.debug
        request["gemma4_mtp"] = self.gemma4_mtp
        if self.model_path is not None:
            request["director_model_path"] = str(self.model_path)
            request["director_mmproj_path"] = str(self.mmproj_path)

    def _run_worker_with_mtp_fallback(
        self,
        operation: str,
        request: dict[str, Any],
        worker: Any,
    ) -> Any:
        """Retry a native-worker failure once with the original decoder.

        Native llama.cpp aborts cannot be caught inside the child process. The
        worker is disposable, however, so the parent can preserve a JSON copy
        of the request, observe the process exit, and run the same operation in
        a fresh non-MTP worker. This changes only the retry request; the next
        independent Gemma operation still attempts MTP normally.
        """
        attempted_mtp = bool(request.get("gemma4_mtp", False))
        retry_request = (
            json.loads(json.dumps(request, ensure_ascii=False))
            if attempted_mtp
            else None
        )
        try:
            return worker(request)
        except Gemma4WorkerExitError as error:
            if not attempted_mtp or retry_request is None:
                raise
            retry_request["gemma4_mtp"] = False
            status = (
                f"status {error.returncode}"
                if error.returncode is not None
                else error.worker_error_type or type(error).__name__
            )
            logging.warning(
                "HR Endless Sampler Gemma 4 native MTP worker failed during %s (%s): %s. "
                "Retrying this operation once with the original non-MTP decoder; "
                "the next Gemma operation will try native MTP again.",
                operation,
                status,
                error,
            )
            return worker(retry_request)

    def plan_timing(self, request: dict[str, Any], progress_callback: Any = None) -> GemmaShotTimingPlan:
        """Create the immutable Gemma action schedule before any H3 chunk runs."""
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if not request.get("source_shots"):
            raise Gemma4ObservationError("Gemma 4 needs source shots for timing preproduction")
        self.last_timing_system_prompt, self.last_timing_planning_prompt = _render_timing_plan_messages(request)
        self._configure_request(request)
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
        if len(result.attempts) > 1:
            logging.warning(
                "HR Endless Sampler Gemma 4 preproduction timing plan needed one model-authored correction; "
                "the corrected complete schedule will be used."
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
        self._configure_request(request)
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
        self._configure_request(request)
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
                    "the H3 local marker/current-slice coverage contract; requested one Gemma-generated correction and will use "
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
