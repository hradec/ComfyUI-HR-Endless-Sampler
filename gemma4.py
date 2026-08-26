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
import subprocess
import sys
import tempfile
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
GEMMA4_MODEL_DIRECTORY = "llama_cpp/gemma-4-12b-it-qat-q4_0"
GEMMA4_REQUIRED_VERSION = "0.3.35"
GEMMA4_IMAGE_MIN_TOKENS = 70
GEMMA4_IMAGE_MAX_TOKENS = 1120
GEMMA4_BATCH_SIZE = GEMMA4_IMAGE_MAX_TOKENS
GEMMA4_PROMPTS_PATH = Path(__file__).with_name("gemma4_prompts.txt")
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


class Gemma4DependencyError(RuntimeError):
    """Raised when the node cannot use its deliberately pinned local runtime."""


class Gemma4ObservationError(RuntimeError):
    """Raised for a malformed or unusable model observation."""

    def __init__(self, message: str, *, raw_json: str = ""):
        super().__init__(message)
        self.raw_json = raw_json


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
    """Load MiniMax's vendored prompt-writing skill and the mode-specific guide."""
    guide_path = MINIMAX_PROMPT_GUIDES.get(mode)
    if guide_path is None:
        raise Gemma4ObservationError(f"Unknown MiniMax prompt mode for Gemma: {mode!r}")
    try:
        skill = MINIMAX_PROMPT_SKILL_PATH.read_text(encoding="utf-8")
        guide = guide_path.read_text(encoding="utf-8")
    except OSError as error:
        raise Gemma4ObservationError(f"Could not read vendored MiniMax prompt documentation: {error}") from error
    return (
        "Official MiniMax H3 prompt-writing skill follows. Apply its prompt rules while obeying the "
        "chunk-local output contract above.\n\n"
        + skill.strip()
        + f"\n\nOfficial MiniMax H3 {mode} mode reference follows.\n\n"
        + guide.strip()
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
        "MiniMax H3 Unlimited is downloading the Gemma 4 continuity model to %s. "
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

    Gemma4MTMDChatHandler.__name__ = "Gemma4MTMDChatHandler"
    return Gemma4MTMDChatHandler


def _load_runtime():
    try:
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
        from llama_cpp._utils import suppress_stdout_stderr
        import llama_cpp.mtmd_cpp
    except ImportError as error:
        raise Gemma4DependencyError(
            "Gemma 4 continuity requires llama-cpp-python==0.3.35 with CUDA support. "
            "Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        ) from error

    version = getattr(llama_cpp, "__version__", "unknown")
    if version != GEMMA4_REQUIRED_VERSION:
        raise Gemma4DependencyError(
            f"Gemma 4 continuity requires llama-cpp-python=={GEMMA4_REQUIRED_VERSION} with MTMD vision support; "
            f"found {version}. Install this custom node's requirements.txt with ~/comfyui/tools/python.sh."
        )
    return Llama, _gemma4_mtmd_handler_type(
        MTMDChatHandler,
        llama_cpp,
        suppress_stdout_stderr,
    )


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
        '{"confidence":"high|medium|low", "analysis":"...", "shots":[{"source_shot":1, '
        '"visual_beats":[{"start_frame":0, "end_frame":34, "action":"..."}], '
        '"overlays":[{"start_frame":4, "end_frame":20, "type":"dialogue|sound|action", "content":"..."}]}]}\n\n'
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
    target_shots = request["target_shots"]
    message = _render_gemma_prompt(
        templates["OBSERVATION"],
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
        "format": "minimax-h3-gemma4-capture-v2",
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
    """Run one deterministic JSON completion and return its parsed/raw forms."""
    response = llm.create_chat_completion(
        messages=list(messages),
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    choice = response["choices"][0]["message"]
    text = choice.get("content") or ""
    if not isinstance(text, str):
        raise Gemma4ObservationError("Gemma 4 returned no textual response")
    return _extract_json_object(text)


def _observe_in_process(
    request: dict[str, Any],
    image_urls: Sequence[str],
    debug: bool,
) -> GemmaChunkPrompt:
    """Run one observation inside the disposable worker process."""
    Llama, MTMDChatHandler = _load_runtime()
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
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=debug, use_gpu=True)
        llm = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_gpu_layers=-1,
            n_ctx=16384,
            n_batch=GEMMA4_BATCH_SIZE,
            n_ubatch=GEMMA4_BATCH_SIZE,
            flash_attn=True,
            verbose=debug,
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        latest_raw_json = ""
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


def _plan_timing_in_process(request: dict[str, Any], debug: bool) -> GemmaShotTimingPlan:
    """Run the one-time text-only source-shot timing plan in the worker."""
    Llama, MTMDChatHandler = _load_runtime()
    model_path, mmproj_path = _ensure_model_files()
    system_prompt, message = _render_timing_plan_messages(request)
    handler = None
    llm = None
    try:
        # Keep the official Gemma multimodal chat handler even though this pass
        # has no images.  It supplies the same model-specific conversation
        # formatting as the later image-and-text requests.
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=debug, use_gpu=True)
        llm = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_gpu_layers=-1,
            n_ctx=16384,
            n_batch=GEMMA4_BATCH_SIZE,
            n_ubatch=GEMMA4_BATCH_SIZE,
            flash_attn=True,
            verbose=debug,
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
            return replace(initial, attempts=tuple(attempts))
        except Gemma4ObservationError as error:
            correction_prompt = _timing_plan_correction_request(request, error)
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
            return replace(corrected, attempts=attempts)
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


def _observe_in_worker(request: dict[str, Any]) -> GemmaChunkPrompt:
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
        stdout, _ = process.communicate(json.dumps(request, ensure_ascii=False))
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        request.clear()

    result_line = next(
        (line[len(_WORKER_RESULT_PREFIX):] for line in reversed(stdout.splitlines())
         if line.startswith(_WORKER_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise Gemma4ObservationError(
            f"Gemma 4 worker exited with status {process.returncode} without returning a result"
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
        raise Gemma4ObservationError(message, raw_json=raw_json)
    if process.returncode != 0:
        raise Gemma4ObservationError(f"Gemma 4 worker exited with status {process.returncode}")
    return _chunk_prompt_from_payload(result["chunk_prompt"])


def _plan_timing_in_worker(request: dict[str, Any]) -> GemmaShotTimingPlan:
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
    try:
        stdout, _ = process.communicate(json.dumps(request, ensure_ascii=False))
    except BaseException:
        process.kill()
        process.wait()
        raise

    result_line = next(
        (line[len(_WORKER_RESULT_PREFIX):] for line in reversed(stdout.splitlines())
         if line.startswith(_WORKER_RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise Gemma4ObservationError(
            f"Gemma 4 worker exited with status {process.returncode} without returning a result"
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
        raise Gemma4ObservationError(message, raw_json=raw_json)
    if process.returncode != 0:
        raise Gemma4ObservationError(f"Gemma 4 worker exited with status {process.returncode}")
    try:
        return _timing_plan_from_payload(result["timing_plan"])
    except (KeyError, TypeError, ValueError) as error:
        raise Gemma4ObservationError("Gemma 4 worker returned malformed timing-plan JSON") from error


class Gemma4ContinuityDirector:
    """Preproduction timing planner plus one-shot local prompt director."""

    def __init__(self, debug: bool = False, capture_directory: str | Path | None = None,
                 observation_image_directory: str | Path | None = None):
        self.debug = debug
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
                capture_directory = tempfile.mkdtemp(prefix="minimax-h3-gemma4-")
            self.capture_directory = Path(capture_directory)
            self.capture_directory.mkdir(parents=True, exist_ok=True)
            logging.info("SamplerCustomAdvanced-Unlimited Gemma capture directory: %s", self.capture_directory)

    def plan_timing(self, request: dict[str, Any]) -> GemmaShotTimingPlan:
        """Create the immutable Gemma action schedule before any H3 chunk runs."""
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if not request.get("source_shots"):
            raise Gemma4ObservationError("Gemma 4 needs source shots for timing preproduction")
        self.last_timing_system_prompt, self.last_timing_planning_prompt = _render_timing_plan_messages(request)
        request["debug"] = self.debug
        result = _plan_timing_in_worker(request)
        self.last_timing_system_prompt = result.system_prompt or self.last_timing_system_prompt
        self.last_timing_planning_prompt = result.planning_prompt or self.last_timing_planning_prompt
        if len(result.attempts) > 1:
            logging.warning(
                "SamplerCustomAdvanced-Unlimited Gemma 4 preproduction timing plan needed one model-authored correction; "
                "the corrected complete schedule will be used."
            )
        return result

    def direct(self, request: dict[str, Any], frames: torch.Tensor | None = None) -> GemmaChunkPrompt:
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
                    "SamplerCustomAdvanced-Unlimited could not save last-run Gemma images to %s: %s",
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
            result = _observe_in_worker(request)
        except BaseException as error:
            if capture_dir is not None:
                _write_capture_error(capture_dir, error)
            raise
        if capture_dir is not None:
            _write_capture_result(capture_dir, result)
            logging.info("SamplerCustomAdvanced-Unlimited saved Gemma fixture: %s", capture_dir)
        if len(result.attempts) > 1:
            initial_contract_warnings = _contract_validation_warnings(result.attempts[0].validation_warnings)
            if initial_contract_warnings:
                logging.warning(
                    "SamplerCustomAdvanced-Unlimited Gemma 4 initial response for chunk %d violated "
                    "the H3 local marker/current-slice coverage contract; requested one Gemma-generated correction and will use "
                    "that response:\n- %s",
                    chunk_number,
                    "\n- ".join(initial_contract_warnings),
                )
        self.last_system_prompt = result.system_prompt or self.last_system_prompt
        self.last_observation_prompt = result.observation_prompt or self.last_observation_prompt
        if result.validation_warnings:
            logging.warning(
                "SamplerCustomAdvanced-Unlimited Gemma 4 response for chunk %d has validation warning(s); "
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
