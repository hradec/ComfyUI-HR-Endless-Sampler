"""Process-isolated local Qwen3.5 multimodal director."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

try:
    from .director_errors import DirectorDependencyError, DirectorObservationError, DirectorWorkerError
    from .story_format import compile_h3_prompt, validate_storyboard_plan
except ImportError:  # Direct worker execution.
    from director_errors import DirectorDependencyError, DirectorObservationError, DirectorWorkerError
    from story_format import compile_h3_prompt, validate_storyboard_plan


QWEN35_CONTEXT_TOKENS = 65536
QWEN35_BATCH_SIZE = 256
QWEN35_CHUNK_RESPONSE_TOKENS = 8192
QWEN35_TIMING_RESPONSE_TOKENS = 32768
QWEN36_CONTEXT_TOKENS = 32768
QWEN36_CHUNK_RESPONSE_TOKENS = 4096
QWEN36_TIMING_RESPONSE_TOKENS = 8192
QWEN38_CONTEXT_TOKENS = 32768
QWEN38_CHUNK_RESPONSE_TOKENS = 4096
QWEN38_TIMING_RESPONSE_TOKENS = 8192
QWEN35_PROMPTS_PATH = Path(__file__).with_name("qwen35_prompts.txt")
_WORKER_RESULT_PREFIX = "MINIMAX_H3_QWEN35_RESULT="
_SECTION = re.compile(r"(?ms)^\[([A-Z][A-Z0-9_]*)\]\s*$\n?(.*?)(?=^\[[A-Z][A-Z0-9_]*\]\s*$|\Z)")
_PLACEHOLDER = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")


class Qwen35DependencyError(DirectorDependencyError):
    pass


class Qwen35ObservationError(DirectorObservationError):
    pass


@dataclass(frozen=True)
class QwenPromptAttempt:
    kind: str
    raw_json: str
    validation_warnings: tuple[str, ...] = ()
    correction_prompt: str = ""


@dataclass(frozen=True)
class QwenChunkPrompt:
    confidence: str
    analysis: str
    detailed_description: str
    raw_json: str
    timing_plan: str = ""
    end_state: str = ""
    last_seen_character_state: tuple[dict[str, Any], ...] = ()
    system_prompt: str = ""
    observation_prompt: str = ""
    validation_warnings: tuple[str, ...] = ()
    attempts: tuple[QwenPromptAttempt, ...] = ()


@dataclass(frozen=True)
class QwenShotTimingBeat:
    start_frame: int
    end_frame: int
    action: str


@dataclass(frozen=True)
class QwenShotTimingOverlay:
    start_frame: int
    end_frame: int
    overlay_type: str
    content: str


@dataclass(frozen=True)
class QwenShotTimingShot:
    source_shot: int
    shot_start_frame: int
    shot_end_frame: int
    visual_beats: tuple[QwenShotTimingBeat, ...]
    overlays: tuple[QwenShotTimingOverlay, ...] = ()


@dataclass(frozen=True)
class QwenCharacterSubject:
    character_name: str
    subject: str


@dataclass(frozen=True)
class QwenShotTimingPlan:
    confidence: str
    analysis: str
    shots: tuple[QwenShotTimingShot, ...]
    character_name_table: tuple[QwenCharacterSubject, ...]
    raw_json: str
    system_prompt: str = ""
    planning_prompt: str = ""
    validation_warnings: tuple[str, ...] = ()
    attempts: tuple[QwenPromptAttempt, ...] = ()

    def character_name_table_text(self) -> str:
        if not self.character_name_table:
            return "No explicit named-character-to-subject mapping was found in the original prompt."
        return "\n".join(f"- {item.character_name} -> {item.subject}" for item in self.character_name_table)

    @staticmethod
    def _id(shot: int, kind: str, index: int) -> str:
        return f"S{shot}.{kind}{index}"

    def mandatory_coverage(self, target_shots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        targets = {int(item["shot_number"]): item for item in target_shots}
        result = []
        for shot in self.shots:
            target = targets.get(shot.source_shot)
            if target is None or "target_start" not in target or "target_end" not in target:
                continue
            for kind, entries in (("V", shot.visual_beats), ("O", shot.overlays)):
                for index, entry in enumerate(entries, 1):
                    overlap_start = max(int(target["target_start"]), shot.shot_start_frame + entry.start_frame)
                    overlap_end = min(int(target["target_end"]), shot.shot_start_frame + entry.end_frame)
                    if overlap_start >= overlap_end:
                        continue
                    value = {
                        "id": self._id(shot.source_shot, kind, index),
                        "kind": "visual" if kind == "V" else "overlay",
                        "source_shot": shot.source_shot,
                        "source_start_frame": entry.start_frame,
                        "source_end_frame": entry.end_frame,
                        "overlap_start_frame": overlap_start - shot.shot_start_frame,
                        "overlap_end_frame": overlap_end - shot.shot_start_frame,
                        "action": entry.action if kind == "V" else entry.content,
                    }
                    if kind == "O":
                        value["overlay_type"] = entry.overlay_type
                    result.append(value)
        return result

    def for_target_shots(self, target_shots: Sequence[dict[str, Any]], fps: float) -> str:
        wanted = {int(item["shot_number"]) for item in target_shots}
        blocks = []
        for shot in self.shots:
            if shot.source_shot not in wanted:
                continue
            duration = shot.shot_end_frame - shot.shot_start_frame
            lines = [f"Source Shot {shot.source_shot}: {duration} frames ({duration / fps:.3f} s).", "Serial visual timeline:"]
            for index, beat in enumerate(shot.visual_beats, 1):
                lines.append(f"- [{self._id(shot.source_shot, 'V', index)}] source-relative frames {beat.start_frame}-{beat.end_frame - 1}: {beat.action}")
            if shot.overlays:
                lines.append("Concurrent overlays:")
                for index, overlay in enumerate(shot.overlays, 1):
                    lines.append(f"- [{self._id(shot.source_shot, 'O', index)}] {overlay.overlay_type} at source-relative frames {overlay.start_frame}-{overlay.end_frame - 1}: {overlay.content}")
            blocks.append("\n".join(lines))
        required = self.current_slice_coverage_text(target_shots)
        return required + "\n\nCOMPLETE RELEVANT PREPRODUCTION SCHEDULE\n" + "\n\n".join(blocks)

    def current_slice_coverage_text(self, target_shots: Sequence[dict[str, Any]]) -> str:
        items = self.mandatory_coverage(target_shots)
        if not items:
            return "No mandatory current-slice beat coverage is required."
        return "MANDATORY CURRENT-SLICE BEAT COVERAGE\n" + "\n".join(
            f"- Required now [{item['id']}], {item['kind']}, source-relative frames {item['overlap_start_frame']}-{item['overlap_end_frame'] - 1}: {item['action']}"
            for item in items
        )


def _templates() -> dict[str, str]:
    text = QWEN35_PROMPTS_PATH.read_text(encoding="utf-8")
    values = {name: body.strip() for name, body in _SECTION.findall(text)}
    required = {"TIMING_SYSTEM", "TIMING_USER", "CHUNK_SYSTEM", "CHUNK_USER"}
    if not required.issubset(values):
        raise Qwen35DependencyError(f"Qwen3.5 prompt file is missing sections: {', '.join(sorted(required - values.keys()))}")
    return values


def _render(template: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            raise Qwen35ObservationError(f"Qwen3.5 prompt value is missing: {name}")
        return str(values[name])
    return _PLACEHOLDER.sub(replace, template)


def _source_shots(shots: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for shot in shots:
        duration = int(shot["shot_end"]) - int(shot["shot_start"])
        marker = shot.get("required_marker")
        marker_line = f"Required H3 marker: {marker}\n" if marker else ""
        blocks.append(
            f"Source Shot {int(shot['shot_number'])}: duration {duration} frames; valid local interval [0,{duration}).\n"
            f"{marker_line}{str(shot['source_body']).strip()}"
        )
    return "\n\n".join(blocks)


def _timing_messages(request: dict[str, Any]) -> tuple[str, str]:
    templates = _templates()
    values = {
        "director_name": request.get("director_backend", "qwen3.5").replace("qwen", "Qwen"),
        "chunk_count": request["chunk_count"], "fps": request["fps"],
        "source_shots": _source_shots(request["source_shots"]), "original_prompt": request["original_prompt"],
    }
    return templates["TIMING_SYSTEM"], _render(templates["TIMING_USER"], values)


def _chunk_messages(request: dict[str, Any]) -> tuple[str, str]:
    templates = _templates()
    values = {
        "director_name": request.get("director_backend", "qwen3.5").replace("qwen", "Qwen"),
        "chunk_number": request["chunk_number"], "chunk_count": request["chunk_count"],
        "original_prompt": request["original_prompt"],
        "shot_context": _source_shots(request.get("target_shots", ())),
        "preproduction_timing_plan": request.get("preproduction_timing_plan", ""),
        "mandatory_coverage": json.dumps(request.get("mandatory_coverage", ()), ensure_ascii=False),
        "character_name_table": request.get("character_name_table", "none"),
        "conditioning_context": request.get("conditioning_context", "none"),
        "observation_frames": ", ".join(str(value) for value in request.get("observation_frame_numbers", ())) or "none",
        "previous_description": request.get("previous_gemma_description", "none") or "none",
        "previous_timing_plan": request.get("previous_gemma_timing_plan", "none") or "none",
        "previous_state": request.get("previous_gemma_end_state", request.get("previous_end_state", "none")) or "none",
        "previous_characters": json.dumps(request.get("previous_last_seen_character_state", ()), ensure_ascii=False),
    }
    return templates["CHUNK_SYSTEM"], _render(templates["CHUNK_USER"], values)


def _storyboard_messages(request: dict[str, Any]) -> tuple[str, str]:
    templates = _templates()
    count = int(request["image_count"])
    inventory = "\n".join(f"- <Picture {index}>: connected reference image {index}" for index in range(1, count + 1))
    values = {
        "director_name": request.get("director_backend", "qwen3.5").replace("qwen", "Qwen"),
        "story": str(request["story"]).strip(),
        "style": str(request.get("style", "Follow the story's natural visual style.")).strip(),
        "shot_density": str(request.get("shot_density", "medium")).strip(),
        "fps": request["fps"],
        "total_frames": request["total_frames"],
        "duration_seconds": request["duration_seconds"],
        "picture_inventory": inventory,
    }
    return templates["STORYBOARD_SYSTEM"], _render(templates["STORYBOARD_USER"], values)


def _storyboard_result(value: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    plan = validate_storyboard_plan(
        value,
        image_count=int(request["image_count"]),
        total_frames=int(request["total_frames"]),
    )
    fps = float(request["fps"])
    lines = []
    for shot in plan["shots"]:
        start = shot["start_frame"] / fps
        end = shot["end_frame"] / fps
        pictures = ", ".join(f"Picture {number}" for number in shot["pictures"]) or "无图片引用"
        lines.append(f"镜头 {shot['shot']}｜{start:.3f}-{end:.3f} 秒｜{pictures}\n{shot['description']}")
    return {
        "prompt": compile_h3_prompt(plan, fps=fps),
        "story_plan": plan,
        "shot_report": "\n\n".join(lines),
        "warnings": [],
    }


def _extract_json(text: str) -> tuple[dict[str, Any], str]:
    match = re.search(r"\{", text)
    if match is None:
        raise Qwen35ObservationError("Qwen3.5 did not return a JSON object", raw_json=text)
    try:
        value, end = json.JSONDecoder().raw_decode(text[match.start():])
    except json.JSONDecodeError as error:
        raise Qwen35ObservationError("Qwen3.5 returned malformed JSON", raw_json=text) from error
    if not isinstance(value, dict):
        raise Qwen35ObservationError("Qwen3.5 response root must be an object", raw_json=text)
    return value, text[match.start():match.start() + end]


def _source_shot_number(value: Any) -> int:
    if isinstance(value, bool):
        raise Qwen35ObservationError(f"Qwen3.5 returned an invalid source_shot: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"(?:Source\s+)?Shot\s+(\d+)", value.strip(), re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
        if value.strip().isdigit():
            return int(value.strip())
    raise Qwen35ObservationError(f"Qwen3.5 returned an invalid source_shot: {value!r}")


def _timing_plan(value: dict[str, Any], request: dict[str, Any], raw: str, system: str, prompt: str) -> QwenShotTimingPlan:
    supplied = value.get("shots")
    expected = request.get("source_shots", ())
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        actual = len(supplied) if isinstance(supplied, list) else type(supplied).__name__
        raise Qwen35ObservationError(
            f"Qwen3.5 timing plan returned {actual} shot entries; expected {len(expected)}",
            raw_json=raw,
        )
    shots = []
    for item, source in zip(supplied, expected):
        number = int(source["shot_number"])
        if not isinstance(item, dict):
            raise Qwen35ObservationError(f"Qwen3.5 timing plan must preserve Source Shot {number}", raw_json=raw)
        try:
            source_shot = _source_shot_number(item.get("source_shot"))
        except Qwen35ObservationError as error:
            raise Qwen35ObservationError(str(error), raw_json=raw) from error
        if source_shot != number:
            raise Qwen35ObservationError(f"Qwen3.5 timing plan must preserve Source Shot {number}; got {item.get('source_shot')!r}", raw_json=raw)
        duration = int(source["shot_end"]) - int(source["shot_start"])
        raw_beats = item.get("visual_beats")
        if not isinstance(raw_beats, list) or not raw_beats:
            raise Qwen35ObservationError(f"Qwen3.5 Source Shot {number} needs visual beats", raw_json=raw)
        beats = []
        previous = 0
        coordinate_offset = 0
        for index, beat in enumerate(raw_beats):
            if not isinstance(beat, dict):
                raise Qwen35ObservationError(f"Qwen3.5 Source Shot {number} visual beat {index + 1} is not an object", raw_json=raw)
            try:
                start = int(beat["start_frame"] if "start_frame" in beat else beat["frame_start"])
                if index == 0 and start == int(source["shot_start"]):
                    coordinate_offset = int(source["shot_start"])
                start -= coordinate_offset
                if index == len(raw_beats) - 1 and start == previous and start < duration:
                    end = duration
                elif "end_frame" in beat or "frame_end" in beat:
                    end_value = beat["end_frame"] if "end_frame" in beat else beat["frame_end"]
                    end = int(end_value) - coordinate_offset
                else:
                    next_beat = raw_beats[index + 1]
                    end = int(next_beat["start_frame"] if "start_frame" in next_beat else next_beat["frame_start"]) - coordinate_offset
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise Qwen35ObservationError(
                    f"Qwen3.5 Source Shot {number} visual beat {index + 1} needs integer start_frame and end_frame; returned keys: {', '.join(sorted(str(key) for key in beat))}",
                    raw_json=raw,
                ) from error
            if start != previous or end <= start or end > duration:
                raise Qwen35ObservationError(f"Qwen3.5 Source Shot {number} visual beats are not contiguous at {start}-{end} after {previous}", raw_json=raw)
            action = next(
                (
                    candidate.strip()
                    for name in ("action", "description", "visual_action", "content", "beat")
                    if isinstance(candidate := beat.get(name), str) and candidate.strip()
                ),
                "Continue the source shot action.",
            )
            beats.append(QwenShotTimingBeat(start, end, action))
            previous = end
        overlays = []
        for overlay in item.get("overlays", ()):
            if not isinstance(overlay, dict):
                continue
            kind = str(overlay.get("type", ""))
            content = str(overlay.get("content", "")).strip()
            if kind not in {"dialogue", "sound", "action"} or not content:
                continue
            try:
                start = max(0, int(overlay["start_frame"]) - coordinate_offset)
                end = min(duration, int(overlay["end_frame"]) - coordinate_offset)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if start < end:
                overlays.append(QwenShotTimingOverlay(start, end, kind, content))
        shots.append(QwenShotTimingShot(number, int(source["shot_start"]), int(source["shot_end"]), tuple(beats), tuple(overlays)))
    table = []
    seen = set()
    for item in value.get("character_name_table", ()):
        if not isinstance(item, dict):
            continue
        name, subject = str(item.get("character_name", "")).strip(), str(item.get("subject", "")).strip()
        key = (name.casefold(), subject.casefold())
        if name and re.fullmatch(r"<Subject\s+\d+>", subject, re.I) and key not in seen:
            seen.add(key)
            table.append(QwenCharacterSubject(name, subject))
    attempt = QwenPromptAttempt("initial response", raw)
    return QwenShotTimingPlan(str(value.get("confidence", "unknown")), str(value.get("analysis", "")).strip(), tuple(shots), tuple(table), raw, system, prompt, attempts=(attempt,))


def _chunk_prompt(value: dict[str, Any], raw: str, system: str, prompt: str, request: dict[str, Any] | None = None) -> QwenChunkPrompt:
    description = next(
        (
            candidate.strip()
            for name in ("detailed_description", "h3_prompt", "video_prompt", "chunk_prompt", "prompt", "description", "timing_plan")
            if isinstance(candidate := value.get(name), str) and candidate.strip()
        ),
        "",
    )
    if not description:
        keys = ", ".join(sorted(str(name) for name in value)) or "none"
        raise Qwen35ObservationError(
            f"Qwen response contains no usable H3 prompt text; returned keys: {keys}",
            raw_json=raw,
        )
    required_markers = [
        str(shot["required_marker"]).strip()
        for shot in (request or {}).get("target_shots", ())
        if shot.get("required_marker")
    ]
    missing_markers = [marker for marker in required_markers if marker not in description]
    if missing_markers:
        description = " ".join([*missing_markers, description])
    state = value.get("last_seen_character_state", ())
    if not isinstance(state, list):
        state = []
    return QwenChunkPrompt(
        str(value.get("confidence", "unknown")), str(value.get("analysis", "")).strip(), description.strip(), raw,
        str(value.get("timing_plan", "")), str(value.get("end_state", "")),
        tuple(dict(item) for item in state if isinstance(item, dict)), system, prompt,
        attempts=(QwenPromptAttempt("initial response", raw),),
    )


def _image_url(frame: torch.Tensor) -> str:
    image = frame.detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise Qwen35ObservationError(f"Qwen3.5 expected HWC RGB frames, got {tuple(image.shape)}")
    pixels = image[..., :3].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _gguf_mtp_layers(model_path: str | Path) -> int | None:
    """Return embedded NextN/MTP layers, zero when absent, or None when unreadable."""
    path = Path(model_path)
    if path.suffix.casefold() != ".gguf":
        return None
    fixed_types = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
        10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
    }
    try:
        with path.open("rb") as gguf:
            def read_exact(size: int) -> bytes:
                value = gguf.read(size)
                if len(value) != size:
                    raise EOFError
                return value

            def read_string() -> str:
                size = struct.unpack("<Q", read_exact(8))[0]
                return read_exact(size).decode("utf-8", errors="replace")

            def read_value(value_type: int, capture=False):
                if value_type in fixed_types:
                    fmt, size = fixed_types[value_type]
                    value = struct.unpack(fmt, read_exact(size))[0]
                    return value if capture else None
                if value_type == 8:
                    value = read_string()
                    return value if capture else None
                if value_type == 9:
                    element_type = struct.unpack("<I", read_exact(4))[0]
                    count = struct.unpack("<Q", read_exact(8))[0]
                    if element_type in fixed_types:
                        gguf.seek(fixed_types[element_type][1] * count, os.SEEK_CUR)
                    else:
                        for _ in range(count):
                            read_value(element_type)
                    return None
                raise ValueError(value_type)

            if read_exact(4) != b"GGUF" or struct.unpack("<I", read_exact(4))[0] not in (2, 3):
                return None
            tensor_count, metadata_count = struct.unpack("<QQ", read_exact(16))
            layers = 0
            for _ in range(metadata_count):
                key = read_string()
                value_type = struct.unpack("<I", read_exact(4))[0]
                capture = key.casefold().endswith(".nextn_predict_layers")
                value = read_value(value_type, capture)
                if capture and isinstance(value, (int, float)):
                    layers = max(layers, int(value))
            if layers <= 0:
                for _ in range(tensor_count):
                    name = read_string().casefold()
                    dimensions = struct.unpack("<I", read_exact(4))[0]
                    gguf.seek(dimensions * 8 + 4 + 8, os.SEEK_CUR)
                    if ".nextn." in name or ".mtp." in name:
                        layers = 1
            return layers
    except (OSError, EOFError, ValueError, struct.error):
        return None


def _qwen_family(model_path: str | Path) -> str:
    name = Path(model_path).name.casefold().replace("-", "").replace("_", "")
    if "qwen38" in name or "qwen3.8" in name:
        return "qwen3.8"
    if "qwen36" in name or "qwen3.6" in name:
        return "qwen3.6"
    return "qwen3.5"


def _adapt_qwen38_mtmd_template(chat_template: str) -> str:
    if not chat_template or "<|image_pad|>" not in chat_template:
        return chat_template
    pattern = r"\{\{-?\s*(['\"])<\|vision_start\|><\|image_pad\|><\|vision_end\|>\1\s*-?\}\}"
    replacement = (
        "{{- '<|vision_start|>' }}"
        "{%- if item.image_url is string %}{{- item.image_url }}"
        "{%- else %}{{- item.image_url.url }}{%- endif %}"
        "{{- '<|vision_end|>' }}"
    )
    adapted, count = re.subn(pattern, replacement, chat_template)
    if count == 0:
        raise Qwen35DependencyError("Qwen3.8 chat template contains an unsupported image_pad expression")
    return adapted


def _qwen38_text_handler(llm, formatter_class, handler_factory, reasoning_effort: str):
    template = (getattr(llm, "metadata", {}) or {}).get("tokenizer.chat_template")
    if not template:
        raise Qwen35DependencyError("Qwen3.8 GGUF is missing tokenizer.chat_template")
    model = getattr(llm, "_model", None)

    def token_text(token_id: int) -> str:
        if token_id == -1 or model is None or not hasattr(model, "token_get_text"):
            return ""
        return model.token_get_text(token_id)

    eos, bos, eot = llm.token_eos(), llm.token_bos(), llm.token_eot()
    formatter = formatter_class(
        template=template,
        eos_token=token_text(eos),
        bos_token=token_text(bos),
        stop_token_ids=[token for token in (eos, eot) if token != -1] or None,
    )

    def qwen38_formatter(*, messages, **kwargs):
        kwargs.update(enable_thinking=False, preserve_thinking=False, reasoning_effort=reasoning_effort)
        return formatter(messages=messages, **kwargs)

    return handler_factory(qwen38_formatter)


def _install_mtmd_physical_token_ledger(llm: Any) -> None:
    original_generate = getattr(llm, "generate", None)
    if not callable(original_generate):
        return

    def physical_generate(*args: Any, **kwargs: Any):
        n_tokens = int(getattr(llm, "n_tokens", 0))
        physical_tokens = llm.input_ids[:n_tokens].tolist() if n_tokens > 0 else []
        supplied_tokens = args[0] if args else kwargs.get("tokens")
        has_media_tokens = supplied_tokens is not None and any(int(token) < 0 for token in supplied_tokens)
        if has_media_tokens:
            final_text_token = next((int(token) for token in reversed(supplied_tokens) if int(token) >= 0), None)
            if final_text_token is None:
                raise Qwen35ObservationError("Qwen MTMD prompt has no final text token for speculative decoding")
            generation_tokens = [final_text_token]
            if args:
                args = (generation_tokens, *args[1:])
            else:
                kwargs["tokens"] = generation_tokens
            kwargs["reset"] = False
            supplied_tokens = generation_tokens
        elif supplied_tokens is not None and n_tokens > 0 and len(supplied_tokens) < n_tokens:
            if args:
                args = (physical_tokens, *args[1:])
            else:
                kwargs["tokens"] = physical_tokens
            supplied_tokens = physical_tokens
        if supplied_tokens is not None and list(supplied_tokens) == physical_tokens and n_tokens > 0:
            output_start = int(getattr(llm, "_last_eval_output_start", 0))
            output_count = int(getattr(llm, "_last_eval_output_count", 0))
            if not output_start <= n_tokens - 1 < output_start + output_count:
                llm._last_eval_output_start = n_tokens - 1
                llm._last_eval_output_count = 1
        return original_generate(*args, **kwargs)

    llm.generate = physical_generate


def _load_runtime():
    try:
        from llama_cpp import Llama, SpecConfig, SpeculativeType
        from llama_cpp.llama_chat_format import (
            Jinja2ChatFormatter,
            MTMDChatHandler,
            Qwen35ChatHandler,
            chat_formatter_to_chat_completion_handler,
        )
    except ImportError as error:
        raise Qwen35DependencyError("Qwen director requires llama-cpp-python 0.3.48+ with Qwen MTMD and MTP support") from error
    return Llama, MTMDChatHandler, Qwen35ChatHandler, Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler, SpecConfig, SpeculativeType


def _complete_qwen35(request: dict[str, Any]) -> dict[str, Any]:
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
    except ImportError as error:
        raise Qwen35DependencyError("Qwen3.5 requires llama-cpp-python with MTMD support") from error

    operation = request["operation"]
    if operation not in {"timing_plan", "chunk", "storyboard"}:
        raise Qwen35ObservationError(f"Unknown Qwen operation: {operation}")
    timing = operation == "timing_plan"
    storyboard = operation == "storyboard"
    handler = None if timing else MTMDChatHandler(
        clip_model_path=request["director_mmproj_path"], verbose=False, use_gpu=False,
    )
    llm = Llama(
        model_path=request["director_model_path"], chat_handler=handler, n_gpu_layers=-1,
        n_ctx=QWEN35_CONTEXT_TOKENS, n_batch=QWEN35_BATCH_SIZE, n_ubatch=QWEN35_BATCH_SIZE,
        flash_attn=True, type_k=8, type_v=8, swa_full=False, verbose=False,
    )
    try:
        if timing:
            system, prompt = _timing_messages(request)
        elif storyboard:
            system, prompt = _storyboard_messages(request)
        else:
            system, prompt = _chunk_messages(request)
        content: Any = prompt
        if not timing:
            content = [{"type": "image_url", "image_url": {"url": url}} for url in request.get("image_urls", ())]
            content.append({"type": "text", "text": prompt})
        response = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_object"}, temperature=0.7, top_p=0.9, top_k=40,
            max_tokens=QWEN35_TIMING_RESPONSE_TOKENS if timing else QWEN35_CHUNK_RESPONSE_TOKENS,
            reasoning_budget=0,
        )
        message = response["choices"][0]["message"]
        text = str(message.get("content") or message.get("reasoning_content") or "")
        value, raw = _extract_json(text)
        if storyboard:
            return {"storyboard": _storyboard_result(value, request)}
        result = _timing_plan(value, request, raw, system, prompt) if timing else _chunk_prompt(value, raw, system, prompt, request)
        return {"timing_plan" if timing else "chunk_prompt": _payload(result)}
    finally:
        llm.close()
        close_handler = getattr(handler, "close", None)
        if callable(close_handler):
            close_handler()


def _complete(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("director_backend", "qwen3.5") == "qwen3.5":
        return _complete_qwen35(request)

    Llama, MTMDChatHandler, Qwen35ChatHandler, Jinja2ChatFormatter, handler_factory, SpecConfig, SpeculativeType = _load_runtime()
    operation = request["operation"]
    if operation not in {"timing_plan", "chunk", "storyboard"}:
        raise Qwen35ObservationError(f"Unknown Qwen operation: {operation}")
    timing = operation == "timing_plan"
    storyboard = operation == "storyboard"
    family = _qwen_family(request["director_model_path"])
    handler = None if timing or family == "qwen3.8" else MTMDChatHandler(
        clip_model_path=request["director_mmproj_path"], verbose=False, use_gpu=False,
    )
    context_tokens = {
        "qwen3.5": QWEN35_CONTEXT_TOKENS,
        "qwen3.6": QWEN36_CONTEXT_TOKENS,
        "qwen3.8": QWEN38_CONTEXT_TOKENS,
    }[family]
    llama_kwargs = {
        "model_path": request["director_model_path"], "chat_handler": handler, "n_gpu_layers": -1,
        "n_ctx": context_tokens, "n_batch": QWEN35_BATCH_SIZE, "n_ubatch": QWEN35_BATCH_SIZE,
        "flash_attn": True, "type_k": 8, "type_v": 8, "swa_full": False, "verbose": False,
    }
    if family in {"qwen3.6", "qwen3.8"}:
        if request.get("director_cpu_moe", False):
            llama_kwargs["cpu_moe"] = True
        elif int(request.get("director_n_cpu_moe", 0)) > 0:
            llama_kwargs["n_cpu_moe"] = int(request["director_n_cpu_moe"])
    if family == "qwen3.8":
        logging.info(
            "HR Endless Sampler Qwen3.8 runtime: n_ctx=%d, n_gpu_layers=%d, cpu_moe=%s, n_cpu_moe=%d.",
            context_tokens,
            int(llama_kwargs["n_gpu_layers"]),
            bool(llama_kwargs.get("cpu_moe", False)),
            int(llama_kwargs.get("n_cpu_moe", 0)),
        )
    mtp_supported = family in {"qwen3.6", "qwen3.8"}
    mtp_layers = _gguf_mtp_layers(request["director_model_path"]) if mtp_supported and request.get("director_mtp", False) else 0
    if mtp_layers and mtp_layers > 0:
        llama_kwargs["speculative"] = SpecConfig(
            spec_type=SpeculativeType.DRAFT_MTP,
            draft_n_max=int(request.get("director_mtp_draft_tokens", 2)),
        )
    llm = Llama(**llama_kwargs)
    if not timing and "speculative" in llama_kwargs:
        _install_mtmd_physical_token_ledger(llm)
    try:
        if family == "qwen3.8":
            reasoning_effort = str(request.get("director_reasoning_effort", "xhigh"))
            if reasoning_effort not in {"xhigh", "medium", "low"}:
                raise Qwen35ObservationError(f"Unknown Qwen3.8 reasoning effort: {reasoning_effort}")
            if timing:
                handler = _qwen38_text_handler(llm, Jinja2ChatFormatter, handler_factory, reasoning_effort)
            else:
                template = (getattr(llm, "metadata", {}) or {}).get("tokenizer.chat_template")
                if not template:
                    raise Qwen35DependencyError("Qwen3.8 GGUF is missing tokenizer.chat_template")
                handler = Qwen35ChatHandler(
                    clip_model_path=request["director_mmproj_path"],
                    enable_thinking=False,
                    preserve_thinking=False,
                    extra_template_arguments={"reasoning_effort": reasoning_effort},
                    chat_template_override=_adapt_qwen38_mtmd_template(template),
                    verbose=False,
                    use_gpu=False,
                )
            llm.chat_handler = handler
        if timing:
            system, prompt = _timing_messages(request)
        elif storyboard:
            system, prompt = _storyboard_messages(request)
        else:
            system, prompt = _chunk_messages(request)
        content: Any = prompt
        if not timing:
            content = [{"type": "image_url", "image_url": {"url": url}} for url in request.get("image_urls", ())]
            content.append({"type": "text", "text": prompt})
        completion_kwargs = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "top_p": 0.8 if family == "qwen3.8" else 0.9,
            "top_k": 40,
            "max_tokens": {
                "qwen3.5": QWEN35_TIMING_RESPONSE_TOKENS if timing else QWEN35_CHUNK_RESPONSE_TOKENS,
                "qwen3.6": QWEN36_TIMING_RESPONSE_TOKENS if timing else QWEN36_CHUNK_RESPONSE_TOKENS,
                "qwen3.8": QWEN38_TIMING_RESPONSE_TOKENS if timing else QWEN38_CHUNK_RESPONSE_TOKENS,
            }[family],
            "reasoning_budget": 0,
        }
        if family == "qwen3.8":
            completion_kwargs["min_p"] = 0.0
        response = llm.create_chat_completion(**completion_kwargs)
        choice = response["choices"][0]
        message = choice["message"]
        text = str(message.get("content") or message.get("reasoning_content") or "")
        value, raw = _extract_json(text)
        if storyboard:
            result = _storyboard_result(value, request)
        else:
            result = _timing_plan(value, request, raw, system, prompt) if timing else _chunk_prompt(value, raw, system, prompt, request)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        stats = getattr(llm, "last_speculative_stats", None)
        return {
            "storyboard" if storyboard else ("timing_plan" if timing else "chunk_prompt"): result if storyboard else _payload(result),
            "generation": {
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "mtp_enabled": "speculative" in llama_kwargs,
                "mtp_stats": stats if isinstance(stats, dict) else None,
            },
        }
    finally:
        llm.close()
        close_handler = getattr(handler, "close", None)
        if callable(close_handler):
            close_handler()


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, QwenShotTimingPlan):
        return {
            "confidence": value.confidence, "analysis": value.analysis, "raw_json": value.raw_json,
            "system_prompt": value.system_prompt, "planning_prompt": value.planning_prompt,
            "character_name_table": [item.__dict__ for item in value.character_name_table],
            "shots": [{"source_shot": shot.source_shot, "shot_start_frame": shot.shot_start_frame, "shot_end_frame": shot.shot_end_frame,
                       "visual_beats": [item.__dict__ for item in shot.visual_beats],
                       "overlays": [{"start_frame": item.start_frame, "end_frame": item.end_frame, "overlay_type": item.overlay_type, "content": item.content} for item in shot.overlays]} for shot in value.shots],
        }
    return {name: getattr(value, name) for name in ("confidence", "analysis", "detailed_description", "raw_json", "timing_plan", "end_state", "last_seen_character_state", "system_prompt", "observation_prompt", "validation_warnings")}


def _from_payload(value: dict[str, Any], timing: bool):
    if not timing:
        return QwenChunkPrompt(**{**value, "last_seen_character_state": tuple(value.get("last_seen_character_state", ())), "validation_warnings": tuple(value.get("validation_warnings", ()))})
    shots = tuple(QwenShotTimingShot(
        int(shot["source_shot"]), int(shot["shot_start_frame"]), int(shot["shot_end_frame"]),
        tuple(QwenShotTimingBeat(**beat) for beat in shot["visual_beats"]),
        tuple(QwenShotTimingOverlay(**overlay) for overlay in shot.get("overlays", ())),
    ) for shot in value["shots"])
    table = tuple(QwenCharacterSubject(**item) for item in value.get("character_name_table", ()))
    return QwenShotTimingPlan(value["confidence"], value["analysis"], shots, table, value["raw_json"], value.get("system_prompt", ""), value.get("planning_prompt", ""))


def _run_worker_once(payload: dict[str, Any]) -> tuple[subprocess.CompletedProcess, dict[str, Any] | None]:
    process = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"], input=json.dumps(payload, ensure_ascii=False),
        text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    line = next((item[len(_WORKER_RESULT_PREFIX):] for item in reversed(process.stdout.splitlines()) if item.startswith(_WORKER_RESULT_PREFIX)), None)
    return process, json.loads(line) if line is not None else None


def _run_worker(request: dict[str, Any], timing: bool):
    payload = json.loads(json.dumps(request, ensure_ascii=False))
    payload["operation"] = "timing_plan" if timing else "chunk"
    process, value = _run_worker_once(payload)
    native_failure = value is None or value.get("error_type") not in {"Qwen35ObservationError", "Qwen35DependencyError"}
    if payload.get("director_mtp", False) and native_failure:
        payload["director_mtp"] = False
        process, value = _run_worker_once(payload)
    if value is None:
        raise DirectorWorkerError(f"Qwen worker exited with status {process.returncode} without a result", returncode=process.returncode)
    if not value.get("ok"):
        raise Qwen35ObservationError(str(value.get("message", "Qwen worker failed")), raw_json=str(value.get("raw_json", "")))
    generation = value.get("generation") if isinstance(value.get("generation"), dict) else {}
    stats = generation.get("mtp_stats") if isinstance(generation.get("mtp_stats"), dict) else {}
    logging.info(
        "HR Endless Sampler Qwen %s generation: finish=%s, prompt_tokens=%s, completion_tokens=%s, MTP=%s%s.",
        "timing" if timing else "chunk",
        generation.get("finish_reason", "unknown"),
        generation.get("prompt_tokens", "unknown"),
        generation.get("completion_tokens", "unknown"),
        "active" if generation.get("mtp_enabled") else "inactive",
        f", draft_acceptance={float(stats.get('draft_token_acceptance_rate', 0.0)):.1%}, decode={float(stats.get('decode_tokens_per_second', 0.0)):.2f} tok/s" if stats else "",
    )
    return _from_payload(value["timing_plan" if timing else "chunk_prompt"], timing)


def _run_storyboard_worker(request: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(request, ensure_ascii=False))
    payload["operation"] = "storyboard"
    process, value = _run_worker_once(payload)
    native_failure = value is None or value.get("error_type") not in {
        "Qwen35ObservationError", "Qwen35DependencyError", "ValueError"
    }
    if payload.get("director_mtp", False) and native_failure:
        payload["director_mtp"] = False
        process, value = _run_worker_once(payload)
    if value is None:
        raise DirectorWorkerError(
            f"Qwen storyboard worker exited with status {process.returncode} without a result",
            returncode=process.returncode,
        )
    if not value.get("ok"):
        raise Qwen35ObservationError(
            str(value.get("message", "Qwen storyboard worker failed")),
            raw_json=str(value.get("raw_json", "")),
        )
    return dict(value["storyboard"])


class Qwen35ContinuityDirector:
    def __init__(self, model_path: Path, mmproj_path: Path, debug=False, capture_directory=None, observation_image_directory=None,
                 mtp_enabled=True, mtp_draft_tokens=2, reasoning_effort="xhigh", cpu_moe=False, n_cpu_moe=0,
                 backend="qwen3.5"):
        self.model_path = Path(model_path).resolve()
        self.mmproj_path = Path(mmproj_path).resolve()
        self.backend = str(backend)
        if self.backend not in {"qwen3.5", "qwen3.6", "qwen3.8"}:
            raise ValueError(f"Unknown Qwen backend: {self.backend}")
        self.debug = bool(debug)
        self.mtp_enabled = bool(mtp_enabled) and self.backend in {"qwen3.6", "qwen3.8"}
        self.mtp_draft_tokens = int(mtp_draft_tokens)
        if self.mtp_draft_tokens < 1 or self.mtp_draft_tokens > 8:
            raise ValueError("Qwen MTP draft tokens must be between 1 and 8")
        self.cpu_moe = bool(cpu_moe)
        self.n_cpu_moe = int(n_cpu_moe)
        if self.n_cpu_moe < 0:
            raise ValueError("Qwen3.6 n_cpu_moe cannot be negative")
        self.reasoning_effort = str(reasoning_effort)
        if self.reasoning_effort not in {"xhigh", "medium", "low"}:
            raise ValueError(f"Unknown Qwen3.8 reasoning effort: {self.reasoning_effort}")
        self.capture_directory = Path(capture_directory) if capture_directory else None
        self.observation_image_directory = Path(observation_image_directory) if observation_image_directory else None
        self.last_system_prompt = self.last_observation_prompt = ""
        self.last_timing_system_prompt = self.last_timing_planning_prompt = ""

    def _configure_request(self, request: dict[str, Any]) -> None:
        context_tokens = {
            "qwen3.5": QWEN35_CONTEXT_TOKENS,
            "qwen3.6": QWEN36_CONTEXT_TOKENS,
            "qwen3.8": QWEN38_CONTEXT_TOKENS,
        }[self.backend]
        request.update(debug=self.debug, director_backend=self.backend, director_model_path=str(self.model_path),
                       director_mmproj_path=str(self.mmproj_path), director_n_ctx=context_tokens,
                       director_n_batch=QWEN35_BATCH_SIZE, gemma4_mtp=False,
                       director_mtp=self.mtp_enabled, director_mtp_draft_tokens=self.mtp_draft_tokens,
                       director_reasoning_effort=self.reasoning_effort,
                       director_cpu_moe=self.cpu_moe, director_n_cpu_moe=self.n_cpu_moe)

    def plan_timing(self, request: dict[str, Any], progress_callback: Any = None) -> QwenShotTimingPlan:
        request = json.loads(json.dumps(request, ensure_ascii=False))
        if not request.get("source_shots"):
            raise Qwen35ObservationError("Qwen3.5 needs source shots for timing preproduction")
        self.last_timing_system_prompt, self.last_timing_planning_prompt = _timing_messages(request)
        self._configure_request(request)
        result = _run_worker(request, True)
        self.last_timing_system_prompt = result.system_prompt or self.last_timing_system_prompt
        self.last_timing_planning_prompt = result.planning_prompt or self.last_timing_planning_prompt
        return result

    def direct(self, request: dict[str, Any], frames: torch.Tensor | None = None, progress_callback: Any = None) -> QwenChunkPrompt:
        request = json.loads(json.dumps(request, ensure_ascii=False))
        self.last_system_prompt, self.last_observation_prompt = _chunk_messages(request)
        frame_numbers = request.get("observation_frame_numbers", ())
        if frames is None:
            if frame_numbers:
                raise Qwen35ObservationError("Qwen3.5 has frame numbers but no observation frames")
            request["image_urls"] = []
        else:
            if frames.ndim != 4 or frames.shape[0] != len(frame_numbers):
                raise Qwen35ObservationError("Qwen3.5 observation frames must match the NHWC frame-number batch")
            request["image_urls"] = [_image_url(frame) for frame in frames]
        self._configure_request(request)
        result = _run_worker(request, False)
        self.last_system_prompt = result.system_prompt or self.last_system_prompt
        self.last_observation_prompt = result.observation_prompt or self.last_observation_prompt
        return result

    def plan_storyboard(self, story: str, frames: Sequence[torch.Tensor], *, duration_seconds: float, fps: float,
                        style: str = "cinematic realism", shot_density: str = "medium") -> dict[str, Any]:
        if not isinstance(story, str) or not story.strip():
            raise Qwen35ObservationError("Storyboard planning requires a non-empty story")
        frames = tuple(frames)
        if len(frames) < 1 or len(frames) > 9 or any(
            not isinstance(frame, torch.Tensor) or frame.ndim != 4 or frame.shape[0] != 1
            for frame in frames
        ):
            raise Qwen35ObservationError("Storyboard planning requires 1 to 9 single-image NHWC batches")
        total_frames = max(5, int(round(float(duration_seconds) * float(fps))))
        remainder = (total_frames - 5) % 17
        if remainder:
            total_frames += 17 - remainder
        request = {
            "story": story.strip(),
            "duration_seconds": float(duration_seconds),
            "fps": float(fps),
            "total_frames": total_frames,
            "image_count": len(frames),
            "style": str(style),
            "shot_density": str(shot_density),
            "image_urls": [_image_url(frame[0]) for frame in frames],
        }
        self._configure_request(request)
        return _run_storyboard_worker(request)

    def materialize_preproduction_cache(self, request, timing_plan, progress_callback=None):
        raise Qwen35ObservationError("Qwen3.5 does not support the Gemma preproduction KV cache")


def _worker_main() -> int:
    try:
        result = {"ok": True, **_complete(json.load(sys.stdin))}
    except Exception as error:
        result = {"ok": False, "error_type": type(error).__name__, "message": str(error), "raw_json": getattr(error, "raw_json", "")}
    print(_WORKER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        raise SystemExit(_worker_main())
    raise SystemExit("qwen35.py is an internal worker; use it through the sampler node")
