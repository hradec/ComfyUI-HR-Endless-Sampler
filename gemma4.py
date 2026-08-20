"""Process-isolated Gemma 4 continuity observation for MiniMax H3 Unlimited.

Gemma is intentionally short lived: a single observation loads the GGUF and
multimodal projector, observes sequential stills from the completed H3 chunk,
returns a constrained progress record plus H3-ready continuation prose, and
then exits its worker process.  Process exit is deliberate: llama.cpp's CUDA
backend owns allocations outside PyTorch and can retain a backend/context
high-water mark after ``Llama.close()``.  Exiting the worker guarantees those
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
from dataclasses import dataclass
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
GEMMA4_PROMPTS_PATH = Path(__file__).with_name("gemma4_prompts.txt")
_ACTION_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=;)\s+|\n+")
_PROMPT_SECTION = re.compile(r"(?ms)^\[([A-Z][A-Z0-9_]*)\]\s*$\n?(.*?)(?=^\[[A-Z][A-Z0-9_]*\]\s*$|\Z)")
_PROMPT_PLACEHOLDER = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")
_WORKER_RESULT_PREFIX = "MINIMAX_H3_GEMMA4_RESULT="


class Gemma4DependencyError(RuntimeError):
    """Raised when the node cannot use its deliberately pinned local runtime."""


class Gemma4ObservationError(RuntimeError):
    """Raised for a malformed or unusable model observation."""


@dataclass(frozen=True)
class GemmaAction:
    action_id: str
    text: str


@dataclass(frozen=True)
class GemmaObservation:
    completed_count: int
    in_progress_action_id: str | None
    confidence: str
    observation: str
    continuation_description: str
    raw_json: str


def action_ledger(shot_number: int, body: str) -> tuple[GemmaAction, ...]:
    """Build a deterministic, immutable action ledger from the source shot.

    This intentionally uses only the original source prompt.  A later Gemma
    reply never becomes the next ledger, so an invented LLM rewrite cannot
    accumulate from chunk to chunk.
    """
    actions = []
    for item in _ACTION_BREAK.split(body.strip()):
        text = re.sub(r"\s+", " ", item).strip()
        if text:
            actions.append(text)
    if not actions:
        actions = ["Continue the established shot and its ongoing action."]
    return tuple(
        GemmaAction(f"S{shot_number}.A{index + 1}", text)
        for index, text in enumerate(actions)
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


def _load_runtime():
    try:
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
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
    return Llama, MTMDChatHandler


def _image_data_url(frame: torch.Tensor) -> str:
    image = frame.detach().to(device="cpu", dtype=torch.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise Gemma4ObservationError(f"Gemma observation expected HWC RGB frames, got {tuple(image.shape)}")
    pixels = image[..., :3].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    encoded = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(encoded, format="JPEG", quality=88, optimize=True)
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


def _validate_observation(value: dict[str, Any], ledger: Sequence[GemmaAction], known_completed: int, raw_json: str) -> GemmaObservation:
    action_ids = [action.action_id for action in ledger]
    completed = value.get("completed_action_ids")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise Gemma4ObservationError("Gemma 4 response has no valid completed_action_ids list")
    if completed != action_ids[:len(completed)]:
        raise Gemma4ObservationError("Gemma 4 completed_action_ids are not an ordered source-ledger prefix")

    completed_count = max(known_completed, len(completed))
    in_progress = value.get("in_progress_action_id")
    if in_progress is not None and not isinstance(in_progress, str):
        raise Gemma4ObservationError("Gemma 4 in_progress_action_id must be a ledger ID or null")
    expected_in_progress = action_ids[completed_count] if completed_count < len(action_ids) else None
    if in_progress != expected_in_progress:
        raise Gemma4ObservationError(
            "Gemma 4 in_progress_action_id must be the first unfinished source-ledger action"
        )
    confidence = value.get("confidence", "unknown")
    if confidence not in ("high", "medium", "low", "unknown"):
        confidence = "unknown"
    observation = value.get("observation", "")
    if not isinstance(observation, str):
        observation = ""
    continuation_description = value.get("continuation_description")
    if not isinstance(continuation_description, str):
        raise Gemma4ObservationError("Gemma 4 response has no continuation_description string")
    continuation_description = re.sub(r"\s+", " ", continuation_description).strip()
    if not continuation_description:
        raise Gemma4ObservationError("Gemma 4 returned an empty continuation_description")
    if len(continuation_description) > 6000:
        raise Gemma4ObservationError("Gemma 4 continuation_description is unexpectedly long")
    return GemmaObservation(
        completed_count,
        in_progress,
        confidence,
        observation.strip(),
        continuation_description,
        raw_json,
    )


def _observation_payload(observation: GemmaObservation) -> dict[str, Any]:
    return {
        "completed_count": observation.completed_count,
        "in_progress_action_id": observation.in_progress_action_id,
        "confidence": observation.confidence,
        "observation": observation.observation,
        "continuation_description": observation.continuation_description,
        "raw_json": observation.raw_json,
    }


def _observation_from_payload(value: dict[str, Any]) -> GemmaObservation:
    return GemmaObservation(
        completed_count=int(value["completed_count"]),
        in_progress_action_id=value.get("in_progress_action_id"),
        confidence=str(value["confidence"]),
        observation=str(value["observation"]),
        continuation_description=str(value["continuation_description"]),
        raw_json=str(value["raw_json"]),
    )


def _observe_in_process(
    shot_number: int,
    shot_start: int,
    shot_end: int,
    fps: float,
    ledger: Sequence[GemmaAction],
    known_completed: int,
    image_urls: Sequence[str],
    continuation_frames: int,
    debug: bool,
) -> GemmaObservation:
    """Run one observation inside the disposable worker process."""
    Llama, MTMDChatHandler = _load_runtime()
    model_path, mmproj_path = _ensure_model_files()
    action_lines = "\n".join(f"- {action.action_id}: {action.text}" for action in ledger)
    completed_ids = [action.action_id for action in ledger[:known_completed]]
    templates = _gemma_prompt_templates()
    message = _render_gemma_prompt(
        templates["OBSERVATION"],
        {
            "shot_number": str(shot_number),
            "shot_start": str(shot_start),
            "shot_end": str(shot_end - 1),
            "fps": f"{fps:g}",
            "action_ledger": action_lines,
            "completed_ids": ", ".join(completed_ids) if completed_ids else "none",
            "continuation_frames": str(continuation_frames),
            "continuation_seconds": f"{continuation_frames / fps:.3f}",
        },
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": message}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    )

    handler = None
    llm = None
    response = None
    try:
        handler = MTMDChatHandler(clip_model_path=str(mmproj_path), verbose=debug, use_gpu=True)
        llm = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=512,
            flash_attn=True,
            verbose=debug,
        )
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": templates["SYSTEM"]},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        choice = response["choices"][0]["message"]
        text = choice.get("content") or ""
        if not isinstance(text, str):
            raise Gemma4ObservationError("Gemma 4 returned no textual response")
        payload, raw_json = _extract_json_object(text)
        return _validate_observation(payload, ledger, known_completed, raw_json)
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


def _observe_in_worker(request: dict[str, Any]) -> GemmaObservation:
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
        if result.get("error_type") == "Gemma4DependencyError":
            raise Gemma4DependencyError(message)
        raise Gemma4ObservationError(message)
    if process.returncode != 0:
        raise Gemma4ObservationError(f"Gemma 4 worker exited with status {process.returncode}")
    return _observation_from_payload(result["observation"])


class Gemma4ContinuityDirector:
    """One-shot local Gemma 4 visual observer with deterministic result checks."""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def observe(self, shot_number: int, shot_start: int, shot_end: int, fps: float,
                ledger: Sequence[GemmaAction], known_completed: int,
                frames: torch.Tensor, continuation_frames: int) -> GemmaObservation:
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise Gemma4ObservationError("Gemma 4 needs at least one decoded observation frame")
        if continuation_frames <= 0:
            raise Gemma4ObservationError("Gemma 4 needs a positive number of new continuation frames")
        image_urls = [_image_data_url(frame) for frame in frames]
        request = {
            "shot_number": shot_number,
            "shot_start": shot_start,
            "shot_end": shot_end,
            "fps": fps,
            "ledger": [
                {"action_id": action.action_id, "text": action.text}
                for action in ledger
            ],
            "known_completed": known_completed,
            "image_urls": image_urls,
            "continuation_frames": continuation_frames,
            "debug": self.debug,
        }
        return _observe_in_worker(request)


def _worker_main() -> int:
    try:
        request = json.load(sys.stdin)
        ledger = tuple(GemmaAction(str(item["action_id"]), str(item["text"])) for item in request["ledger"])
        observation = _observe_in_process(
            shot_number=int(request["shot_number"]),
            shot_start=int(request["shot_start"]),
            shot_end=int(request["shot_end"]),
            fps=float(request["fps"]),
            ledger=ledger,
            known_completed=int(request["known_completed"]),
            image_urls=[str(item) for item in request["image_urls"]],
            continuation_frames=int(request["continuation_frames"]),
            debug=bool(request["debug"]),
        )
        result = {"ok": True, "observation": _observation_payload(observation)}
    except Exception as error:
        result = {
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    print(_WORKER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        raise SystemExit(_worker_main())
    raise SystemExit("gemma4.py is an internal worker; use it through the sampler node")
