"""Read-only browser view of the last HR Endless Sampler replay cache."""

import hashlib
import json
from pathlib import Path

import torch

import comfy.nested_tensor
from aiohttp import web
from comfy_api.latest import io

from .nodes import REPLAY_CACHE_FORMAT, _LastRunReplayCache, _replay_cache_root, _replay_history_runs, _set_replay_control
from .video_io import HREndlessTimeline, normalize_timeline

RetakePlan = io.Custom("HR_RETAKE_PLAN")
RETAKE_MODES = {"video_only", "isolated_av", "continuous_av"}

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


def replay_cache_snapshot(run_id=None):
    try:
        root = _replay_cache_root(run_id)
    except ValueError as error:
        return {"available": False, "reason": str(error), "chunks": []}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "reason": "No last-run replay cache exists", "chunks": []}
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "reason": f"Could not read replay manifest: {error}", "chunks": []}
    chunks = []
    for entry in manifest.get("chunks", []):
        try:
            number = int(entry["chunk"])
            relative = Path(str(entry["metadata_path"]))
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or relative.is_absolute():
                raise ValueError("metadata path leaves replay cache")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            tensor_relative = Path(str(metadata.get("tensor_path", "")))
            tensor_path = (root / tensor_relative).resolve()
            tensor_exists = not tensor_relative.is_absolute() and root.resolve() in tensor_path.parents and tensor_path.is_file()
            images = []
            for value in metadata.get("observation_images", ()):
                image_relative = Path(str(value))
                image_path = (root / image_relative).resolve()
                if not image_relative.is_absolute() and root.resolve() in image_path.parents and image_path.is_file():
                    images.append(str(value))
            chunks.append({
                "chunk": number,
                "frame_start": metadata.get("frame_start"),
                "frame_end": metadata.get("frame_end"),
                "output_trim_frames": metadata.get("output_trim_frames", 0),
                "source_prompt": metadata.get("source_prompt", ""),
                "effective_h3_prompt": metadata.get("effective_h3_prompt", ""),
                "director_description": metadata.get("director_description", ""),
                "observation_images": images,
                "active_revision": metadata.get("active_revision", 0),
                "revisions": metadata.get("revisions", []),
                "complete": bool(tensor_exists and metadata.get("effective_h3_prompt")),
            })
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            chunks.append({"chunk": entry.get("chunk"), "complete": False, "error": str(error), "observation_images": []})
    identity_payload = {"format": manifest.get("format"), "fingerprint": manifest.get("fingerprint"),
                        "source_prompt_sha256": manifest.get("source_prompt_sha256"), "created": manifest.get("created")}
    return {
        "available": True,
        "run_id": manifest.get("run_id") if run_id is None else run_id,
        "runs": _replay_history_runs() if run_id is None else [],
        "cache_identity": hashlib.sha256(json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "format": manifest.get("format"),
        "supported_format": REPLAY_CACHE_FORMAT,
        "status": manifest.get("status", "unknown"),
        "created": manifest.get("created"),
        "completed_chunks": manifest.get("completed_chunks", 0),
        "fps": manifest.get("fingerprint", {}).get("fps", 24.0),
        "compatible": manifest.get("format") == REPLAY_CACHE_FORMAT,
        "chunks": sorted(chunks, key=lambda item: int(item.get("chunk") or 0)),
    }


def _asset_path(relative_value, run_id=None):
    try:
        root = _replay_cache_root(run_id).resolve()
    except ValueError:
        raise web.HTTPNotFound()
    relative = Path(str(relative_value))
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or not path.is_file():
        raise web.HTTPNotFound()
    return path


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_retake/runs")
    async def hr_endless_sampler_retake_runs(_request):
        return web.json_response({"runs": _replay_history_runs()}, headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_retake/cache")
    async def hr_endless_sampler_retake_cache(request):
        return web.json_response(replay_cache_snapshot(request.rel_url.query.get("run_id")),
                                 headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_retake/control")
    async def hr_endless_sampler_retake_control(request):
        try:
            payload = await request.json()
            action = str(payload.get("action", ""))
            running, _pending = _PROMPT_SERVER.prompt_queue.get_current_queue()
            if not running:
                raise ValueError("No generation is currently running")
            try:
                manifest = json.loads((_replay_cache_root() / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("The sampler has not started recording replay state") from error
            if manifest.get("status") != "recording":
                raise ValueError("The active replay is not recording")
            _set_replay_control(action)
            return web.json_response({"ok": True, "action": action}, headers={"Cache-Control": "no-store"})
        except (TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400,
                                     headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.post("/hr_endless_sampler_retake/activate")
    async def hr_endless_sampler_retake_activate(request):
        try:
            payload = await request.json()
            from .nodes import _LastRunReplayCache
            _LastRunReplayCache(payload.get("run_id")).activate_revision(int(payload["chunk"]), int(payload["revision"]))
            return web.json_response({"ok": True}, headers={"Cache-Control": "no-store"})
        except (KeyError, TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as error:
            return web.json_response({"ok": False, "error": str(error)}, status=400,
                                     headers={"Cache-Control": "no-store"})

    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_retake/asset")
    async def hr_endless_sampler_retake_asset(request):
        return web.FileResponse(_asset_path(request.rel_url.query.get("path", ""), request.rel_url.query.get("run_id")),
                                headers={"Cache-Control": "no-store"})


def build_retake_plan(retake_state):
    try:
        state = json.loads(retake_state or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid retake state JSON: {error.msg}") from error
    if not isinstance(state, dict):
        raise ValueError("Retake state must be a JSON object")
    run_id = state.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("Retake run_id must be text")
    snapshot = replay_cache_snapshot(run_id)
    if not snapshot.get("available") or not snapshot.get("compatible"):
        raise ValueError(snapshot.get("reason") or "The last-run replay cache is unavailable or incompatible")
    mode = str(state.get("mode", "video_only"))
    if mode not in RETAKE_MODES:
        raise ValueError(f"Unknown retake mode: {mode}")
    available = {int(chunk["chunk"]): chunk for chunk in snapshot["chunks"] if chunk.get("complete")}
    selected = state.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("Retake selected chunks must be a list")
    overrides = state.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("Retake prompt overrides must be an object")
    chunks = []
    for value in selected:
        if isinstance(value, bool):
            raise ValueError("Retake chunk numbers must be integers")
        number = int(value)
        if number not in available:
            raise ValueError(f"Chunk {number} is not available for retake")
        prompt = overrides.get(str(number), "")
        if not isinstance(prompt, str):
            raise ValueError(f"Chunk {number} prompt override must be text")
        chunks.append({"chunk": number, "prompt_override": prompt.strip(),
                       "original_h3_prompt": available[number]["effective_h3_prompt"]})
    if not chunks:
        raise ValueError("Select at least one complete chunk for retake")
    chunks.sort(key=lambda item: item["chunk"])
    return {"format": 1, "cache_identity": snapshot["cache_identity"], "run_id": run_id,
            "mode": mode, "chunks": chunks}


class HREndlessRetakeAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessRetakeAssemble",
            display_name="HR Endless Retake Assemble",
            category="model/sampling/custom",
            description="Assemble the active original/retake revision of every cached chunk without sampling.",
            outputs=[io.Latent.Output(display_name="output"), io.Latent.Output(display_name="denoised_output"),
                     HREndlessTimeline.Output(display_name="timeline")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls):
        snapshot = replay_cache_snapshot()
        if not snapshot.get("available") or not snapshot.get("compatible") or not snapshot.get("chunks"):
            raise ValueError(snapshot.get("reason") or "No compatible retake cache is available")
        cache = _LastRunReplayCache()
        states = [cache.load_active_chunk(chunk["chunk"]) for chunk in snapshot["chunks"]]
        if not all(chunk.get("complete") for chunk in snapshot["chunks"]):
            raise ValueError("Every chunk must be complete before assembly")
        output = dict(states[-1]["output_template"])
        denoised = dict(states[-1]["denoised_template"])
        output["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat([state["output_video"] for state in states], dim=2),
            torch.cat([state["output_audio"] for state in states], dim=-1),
        ))
        denoised["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat([state["denoised_video"] for state in states], dim=2),
            torch.cat([state["denoised_audio"] for state in states], dim=-1),
        ))
        chunks = [{"chunk": chunk["chunk"], "start": chunk["frame_start"] + chunk.get("output_trim_frames", 0),
                   "end": chunk["frame_end"] - 1, "active_revision": chunk.get("active_revision", 0)}
                  for chunk in snapshot["chunks"]]
        timeline = normalize_timeline({"fps": snapshot["fps"], "total_frames": chunks[-1]["end"] + 1,
                                       "chunks": chunks}, fps=snapshot["fps"], total_frames=chunks[-1]["end"] + 1)
        return io.NodeOutput(output, denoised, timeline)


class HREndlessSegmentRetakeDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSegmentRetakeDirector",
            display_name="HR Endless Segment Retake Director",
            category="model/sampling/custom",
            description="Select cached chunks, edit their H3 prompts, and create a validated retake plan.",
            inputs=[io.String.Input("retake_state", default='{"mode":"video_only","selected":[],"overrides":{}}', multiline=True)],
            outputs=[RetakePlan.Output(display_name="retake plan"), io.String.Output(display_name="plan JSON")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, retake_state):
        plan = build_retake_plan(retake_state)
        return io.NodeOutput(plan, json.dumps(plan, ensure_ascii=False, indent=2))
