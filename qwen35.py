"""Process-isolated local Qwen3.5 multimodal director."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

try:
    from .director_errors import DirectorDependencyError, DirectorObservationError, DirectorWorkerError
except ImportError:  # Direct worker execution.
    from director_errors import DirectorDependencyError, DirectorObservationError, DirectorWorkerError


QWEN35_CONTEXT_TOKENS = 65536
QWEN35_BATCH_SIZE = 256
QWEN35_CHUNK_RESPONSE_TOKENS = 8192
QWEN35_TIMING_RESPONSE_TOKENS = 32768
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
        blocks.append(f"Source Shot {int(shot['shot_number'])}: duration {duration} frames; valid local interval [0,{duration}).\n{str(shot['source_body']).strip()}")
    return "\n\n".join(blocks)


def _timing_messages(request: dict[str, Any]) -> tuple[str, str]:
    templates = _templates()
    values = {
        "chunk_count": request["chunk_count"], "fps": request["fps"],
        "source_shots": _source_shots(request["source_shots"]), "original_prompt": request["original_prompt"],
    }
    return templates["TIMING_SYSTEM"], _render(templates["TIMING_USER"], values)


def _chunk_messages(request: dict[str, Any]) -> tuple[str, str]:
    templates = _templates()
    values = {
        "chunk_number": request["chunk_number"], "chunk_count": request["chunk_count"],
        "original_prompt": request["original_prompt"],
        "shot_context": _source_shots(request.get("target_shots", ())),
        "preproduction_timing_plan": request.get("preproduction_timing_plan", ""),
        "mandatory_coverage": json.dumps(request.get("mandatory_coverage", ()), ensure_ascii=False),
        "previous_state": request.get("previous_end_state", "none"),
    }
    return templates["CHUNK_SYSTEM"], _render(templates["CHUNK_USER"], values)


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
        for index, beat in enumerate(raw_beats):
            if not isinstance(beat, dict):
                raise Qwen35ObservationError(f"Qwen3.5 Source Shot {number} visual beat {index + 1} is not an object", raw_json=raw)
            try:
                start = int(beat["start_frame"])
                end = duration if index == len(raw_beats) - 1 and start == previous and start < duration else int(beat["end_frame"])
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
            start = max(0, int(overlay["start_frame"]))
            end = min(duration, int(overlay["end_frame"]))
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


def _chunk_prompt(value: dict[str, Any], raw: str, system: str, prompt: str) -> QwenChunkPrompt:
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
            f"Qwen3.5 response contains no usable H3 prompt text; returned keys: {keys}",
            raw_json=raw,
        )
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


def _load_runtime():
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
    except ImportError as error:
        raise Qwen35DependencyError("Qwen3.5 requires llama-cpp-python with MTMD support") from error
    return Llama, MTMDChatHandler


def _complete(request: dict[str, Any]) -> dict[str, Any]:
    Llama, MTMDChatHandler = _load_runtime()
    timing = request["operation"] == "timing_plan"
    handler = None if timing else MTMDChatHandler(clip_model_path=request["director_mmproj_path"], verbose=False, use_gpu=False)
    llm = Llama(
        model_path=request["director_model_path"], chat_handler=handler, n_gpu_layers=-1,
        n_ctx=QWEN35_CONTEXT_TOKENS, n_batch=QWEN35_BATCH_SIZE, n_ubatch=QWEN35_BATCH_SIZE,
        flash_attn=True, type_k=8, type_v=8, swa_full=False, verbose=False,
    )
    try:
        system, prompt = _timing_messages(request) if timing else _chunk_messages(request)
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
        result = _timing_plan(value, request, raw, system, prompt) if timing else _chunk_prompt(value, raw, system, prompt)
        return {"timing_plan" if timing else "chunk_prompt": _payload(result)}
    finally:
        llm.close()
        if handler is not None:
            handler.close()


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


def _run_worker(request: dict[str, Any], timing: bool):
    payload = json.loads(json.dumps(request, ensure_ascii=False))
    payload["operation"] = "timing_plan" if timing else "chunk"
    process = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"], input=json.dumps(payload, ensure_ascii=False),
        text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    line = next((item[len(_WORKER_RESULT_PREFIX):] for item in reversed(process.stdout.splitlines()) if item.startswith(_WORKER_RESULT_PREFIX)), None)
    if line is None:
        raise DirectorWorkerError(f"Qwen3.5 worker exited with status {process.returncode} without a result", returncode=process.returncode)
    value = json.loads(line)
    if not value.get("ok"):
        raise Qwen35ObservationError(str(value.get("message", "Qwen3.5 worker failed")), raw_json=str(value.get("raw_json", "")))
    return _from_payload(value["timing_plan" if timing else "chunk_prompt"], timing)


class Qwen35ContinuityDirector:
    def __init__(self, model_path: Path, mmproj_path: Path, debug=False, capture_directory=None, observation_image_directory=None):
        self.model_path = Path(model_path).resolve()
        self.mmproj_path = Path(mmproj_path).resolve()
        self.debug = bool(debug)
        self.capture_directory = Path(capture_directory) if capture_directory else None
        self.observation_image_directory = Path(observation_image_directory) if observation_image_directory else None
        self.last_system_prompt = self.last_observation_prompt = ""
        self.last_timing_system_prompt = self.last_timing_planning_prompt = ""

    def _configure_request(self, request: dict[str, Any]) -> None:
        request.update(debug=self.debug, director_backend="qwen3.5", director_model_path=str(self.model_path),
                       director_mmproj_path=str(self.mmproj_path), director_n_ctx=QWEN35_CONTEXT_TOKENS,
                       director_n_batch=QWEN35_BATCH_SIZE, gemma4_mtp=False)

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
