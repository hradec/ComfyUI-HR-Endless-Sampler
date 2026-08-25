"""Process-isolated Gemma 4 chunk-prompt director for MiniMax H3 Unlimited.

Gemma is intentionally short lived: one request loads the GGUF and multimodal
projector, studies the complete source intent plus chronological stills from
the completed H3 chunk, writes the complete H3 ``detailed_description`` for
the next physical chunk, and exits. Process exit is deliberate: llama.cpp's
CUDA backend owns allocations outside PyTorch and can retain a backend/context
high-water mark after ``Llama.close()``. Exiting the worker guarantees those
allocations are returned before H3/Qwen use the GPU again.
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
            lines.append(f"Required local H3 marker: {shot['required_marker']}")
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
    lines = [
        "IMMUTABLE H3-LOCAL SHOT MARKERS — COPY THE QUOTED TOKEN EXACTLY",
        "These markers use H3's physical chunk-local clock, whose zero is the first sampled frame.",
        "For detailed_description, write every quoted token exactly once and in this order; do not copy the leading hyphen or quotes.",
        "Original source/global timestamps elsewhere are context only and are forbidden as H3 markers unless they are identical to the token below.",
    ]
    lines.extend(f'- exact token: "{str(shot["required_marker"])}"' for shot in shots)
    return "\n".join(lines)


def _marker_validation_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    """Return only errors that a focused local-marker rewrite can repair."""
    return tuple(warning for warning in warnings if "marker" in warning.lower())


def _marker_correction_request(shots: Sequence[dict[str, Any]], warnings: Sequence[str]) -> str:
    """Ask Gemma itself for one complete corrected JSON response, never a patch."""
    return (
        "MARKER CORRECTION REQUIRED\n"
        "Your immediately preceding JSON used invalid H3 shot markers. Return one complete replacement JSON object "
        "with all five required fields, not an explanation and not a textual patch. Keep the same current-frame-slice "
        "creative intent, dialogue, and continuity reasoning, but rewrite detailed_description so its markers obey this "
        "literal contract:\n"
        f"{_required_local_markers(shots)}\n\n"
        "The source/global timestamps in the original prompt describe the full video and must not be used as markers "
        "inside this physical chunk. Do not add, remove, renumber, or move cuts.\n\n"
        "Detected validation errors in the preceding JSON:\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )


def _dialogue_text(value: str) -> str:
    return re.sub(r"\s+", " ", _DIALOGUE_CONTROL.sub("", value)).strip()


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

    expected = list(request["target_shots"])
    markers = list(_SHOT_MARKER.finditer(description))
    if len(markers) != len(expected):
        warnings.append(
            f"Gemma 4 returned {len(markers)} shot markers; this chunk requires {len(expected)}"
        )
    for index, (marker, shot) in enumerate(zip(markers, expected), 1):
        if int(marker.group(1)) != index:
            warnings.append("Gemma 4 chunk-local shot numbers must start at 1 and be sequential")
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


def _gemma_chat_json(llm: Any, messages: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Run one deterministic JSON completion and return its parsed/raw forms."""
    response = llm.create_chat_completion(
        messages=list(messages),
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
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
            marker_warnings = _marker_validation_warnings(initial.validation_warnings)
            if not marker_warnings:
                return replace(initial, attempts=tuple(attempts))

            correction_prompt = _marker_correction_request(request["target_shots"], marker_warnings)
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
                    kind="local-marker correction response",
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


class Gemma4ContinuityDirector:
    """One-shot local Gemma 4 chunk-prompt director with structural checks."""

    def __init__(self, debug: bool = False, capture_directory: str | Path | None = None,
                 observation_image_directory: str | Path | None = None):
        self.debug = debug
        self._capture_sequence = 0
        self.last_system_prompt = ""
        self.last_observation_prompt = ""
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
            initial_marker_warnings = _marker_validation_warnings(result.attempts[0].validation_warnings)
            if initial_marker_warnings:
                logging.warning(
                    "SamplerCustomAdvanced-Unlimited Gemma 4 initial response for chunk %d violated "
                    "the H3-local marker contract; requested one Gemma-generated correction and will use "
                    "that response:\n- %s",
                    chunk_number,
                    "\n- ".join(initial_marker_warnings),
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
