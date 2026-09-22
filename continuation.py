"""Durable continuation checkpoints created from the last completed replay cache."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import torch
from comfy_api.latest import io

from .nodes import REPLAY_CACHE_FORMAT, _LastRunReplayCache, _replay_cpu_copy, _replay_load_tensor_file, _replay_write_json, _replay_write_tensor_file

ContinuationCheckpoint = io.Custom("HR_CONTINUATION_CHECKPOINT")
ContinuationPlan = io.Custom("HR_CONTINUATION_PLAN")
CONTINUATION_FORMAT = 1
_CHECKPOINT_ID = re.compile(r"^[a-f0-9]{32}$")


def _continuation_root() -> Path:
    try:
        import folder_paths
        root = Path(folder_paths.get_output_directory())
    except ImportError:
        root = Path.cwd() / "output"
    return root / "hr_endless_sampler" / "continuations"


def _checkpoint_root(checkpoint_id: str) -> Path:
    value = str(checkpoint_id)
    if _CHECKPOINT_ID.fullmatch(value) is None:
        raise ValueError("Invalid continuation checkpoint ID")
    return _continuation_root() / value


def _manifest_path(checkpoint_id: str) -> Path:
    return _checkpoint_root(checkpoint_id) / "manifest.json"


def _read_manifest(checkpoint_id: str) -> dict:
    path = _manifest_path(checkpoint_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read continuation checkpoint: {error}") from error
    if value.get("format") != CONTINUATION_FORMAT or value.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Continuation checkpoint manifest is incompatible")
    return value


def _index_entries() -> list[dict]:
    path = _continuation_root() / "index.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read continuation index: {error}") from error
    return [dict(item) for item in value.get("checkpoints", ()) if isinstance(item, dict)]


def list_checkpoints() -> list[dict]:
    result = []
    for entry in _index_entries():
        try:
            manifest = _read_manifest(str(entry["checkpoint_id"]))
            result.append({key: manifest.get(key) for key in (
                "checkpoint_id", "parent_checkpoint_id", "name", "created", "fps", "total_frames", "duration",
            )})
        except (KeyError, ValueError):
            continue
    return result


def latest_checkpoint() -> dict | None:
    checkpoints = list_checkpoints()
    return checkpoints[-1] if checkpoints else None


def _cache_identity(manifest: dict) -> str:
    payload = {"format": manifest.get("format"), "fingerprint": manifest.get("fingerprint"),
               "source_prompt_sha256": manifest.get("source_prompt_sha256"), "created": manifest.get("created")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def create_checkpoint_from_last_run(name: str = "", parent_checkpoint_id: str | None = None,
                                     reference_set=None, run_id: str | None = None) -> dict:
    run_id = (str(run_id).strip() or None) if run_id is not None else None
    cache = _LastRunReplayCache(run_id)
    try:
        replay_manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No readable last-run replay cache: {error}") from error
    if replay_manifest.get("format") != REPLAY_CACHE_FORMAT or replay_manifest.get("status") != "complete":
        raise ValueError("A complete current-format replay cache is required")
    entries = replay_manifest.get("chunks", ())
    if not entries:
        raise ValueError("The replay cache contains no completed chunks")
    last_number = max(int(item["chunk"]) for item in entries)
    states = [cache.load_active_chunk(number) for number in range(1, last_number + 1)]
    state = states[-1]
    required = {"sampled_video", "sampled_audio", "output_video", "output_audio", "denoised_video", "denoised_audio",
                "output_template", "denoised_template"}
    missing = required - set(state)
    if missing:
        raise ValueError("Final replay chunk is missing " + ", ".join(sorted(missing)))
    if parent_checkpoint_id is not None:
        _read_manifest(parent_checkpoint_id)
    checkpoint_id = uuid.uuid4().hex
    root = _checkpoint_root(checkpoint_id)
    root.mkdir(parents=True, exist_ok=False)
    state = dict(state)
    if reference_set is not None:
        state["reference_set"] = _replay_cpu_copy(reference_set)
    state.update({
        "base_output_video": torch.cat([item["output_video"] for item in states], dim=2),
        "base_output_audio": torch.cat([item["output_audio"] for item in states], dim=-1),
        "base_denoised_video": torch.cat([item["denoised_video"] for item in states], dim=2),
        "base_denoised_audio": torch.cat([item["denoised_audio"] for item in states], dim=-1),
    })
    state_path = root / "final_state.pt"
    _replay_write_tensor_file(state_path, state)
    fingerprint = replay_manifest.get("fingerprint", {})
    fps = float(fingerprint.get("fps", 24.0))
    total_frames = int(fingerprint.get("plan", [{}])[-1].get("frame_end", 0))
    manifest = {
        "format": CONTINUATION_FORMAT,
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "name": str(name).strip() or f"Continuation {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": "complete",
        "fps": fps,
        "total_frames": total_frames,
        "duration": total_frames / fps if fps > 0 else 0.0,
        "source_cache_identity": _cache_identity(replay_manifest),
        "source_run_id": replay_manifest.get("run_id") or run_id,
        "source_prompt_sha256": replay_manifest.get("source_prompt_sha256"),
        "fingerprint": fingerprint,
        "final_chunk": last_number,
        "state_path": "final_state.pt",
    }
    _replay_write_json(root / "manifest.json", manifest)
    entries = [item for item in _index_entries() if item.get("checkpoint_id") != checkpoint_id]
    entries.append({"checkpoint_id": checkpoint_id, "parent_checkpoint_id": parent_checkpoint_id, "created": manifest["created"]})
    _replay_write_json(_continuation_root() / "index.json", {"format": CONTINUATION_FORMAT, "checkpoints": entries})
    return manifest


def load_checkpoint(value: dict) -> tuple[dict, dict]:
    if not isinstance(value, dict) or value.get("type") != "HR_CONTINUATION_CHECKPOINT":
        raise ValueError("checkpoint must come from HR Endless Continuation Checkpoint")
    checkpoint_id = str(value.get("checkpoint_id", ""))
    manifest = _read_manifest(checkpoint_id)
    state_path = (_checkpoint_root(checkpoint_id) / str(manifest["state_path"])).resolve()
    root = _checkpoint_root(checkpoint_id).resolve()
    if root not in state_path.parents:
        raise ValueError("Continuation state path leaves its checkpoint")
    return manifest, _replay_load_tensor_file(state_path)


class HREndlessContinuationCheckpoint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessContinuationCheckpoint",
            display_name="HR Endless Continuation Checkpoint",
            category="model/sampling/custom",
            description="Freeze the last completed HR Endless Sampler result as an immutable continuation starting point.",
            inputs=[io.String.Input("name", default=""),
                    io.Custom("HR_MINIMAX_H3_REFERENCE_SET").Input("reference_set", optional=True),
                    io.String.Input("run_id", default="", optional=True)],
            outputs=[ContinuationCheckpoint.Output(display_name="checkpoint"), io.String.Output(display_name="checkpoint info")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, name="", reference_set=None, run_id=None):
        try:
            manifest = create_checkpoint_from_last_run(name, reference_set=reference_set, run_id=run_id)
        except ValueError as error:
            if str(error) != "A complete current-format replay cache is required":
                raise
            manifest = latest_checkpoint()
            if manifest is None:
                raise
            print(f"[HR Endless Sampler] Latest replay is incomplete; reusing continuation checkpoint {manifest['checkpoint_id']} ({manifest['name']}).", flush=True)
        value = {"type": "HR_CONTINUATION_CHECKPOINT", "version": 1, "checkpoint_id": manifest["checkpoint_id"]}
        return io.NodeOutput(value, json.dumps(manifest, ensure_ascii=False, indent=2))


class HREndlessContinuationAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessContinuationAssemble",
            display_name="HR Endless Continuation Assemble",
            category="model/sampling/custom",
            inputs=[ContinuationCheckpoint.Input("checkpoint"), io.Latent.Input("continuation_output"),
                    io.Latent.Input("continuation_denoised")],
            outputs=[io.Latent.Output(display_name="output"), io.Latent.Output(display_name="denoised_output")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, checkpoint, continuation_output, continuation_denoised):
        _manifest, state = load_checkpoint(checkpoint)
        output_streams = continuation_output["samples"].unbind()
        denoised_streams = continuation_denoised["samples"].unbind()
        if len(output_streams) != 2 or len(denoised_streams) != 2:
            raise ValueError("Continuation assembly requires MiniMax H3 nested AV latents")
        output = dict(continuation_output)
        denoised = dict(continuation_denoised)
        import comfy.nested_tensor
        output["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat((state["base_output_video"], output_streams[0].cpu()), dim=2),
            torch.cat((state["base_output_audio"], output_streams[1].cpu()), dim=-1),
        ))
        denoised["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat((state["base_denoised_video"], denoised_streams[0].cpu()), dim=2),
            torch.cat((state["base_denoised_audio"], denoised_streams[1].cpu()), dim=-1),
        ))
        return io.NodeOutput(output, denoised)


class HREndlessContinuationPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessContinuationPlan",
            display_name="HR Endless Continuation Plan",
            category="model/sampling/custom",
            inputs=[ContinuationCheckpoint.Input("checkpoint"), io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Combo.Input("audio_mode", options=["continue", "new_segment", "mute"], default="continue"),
                    io.Combo.Input("reference_policy", options=["inherit", "replace", "inherit_plus_replace"], default="replace"),
                    io.Custom("HR_MINIMAX_H3_REFERENCE_SET").Input("reference_set", optional=True)],
            outputs=[ContinuationPlan.Output(display_name="continuation plan"), io.String.Output(display_name="plan JSON")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, checkpoint, prompt, audio_mode="continue", reference_policy="replace", reference_set=None):
        manifest, state = load_checkpoint(checkpoint)
        if not str(prompt).strip():
            raise ValueError("Continuation prompt cannot be empty")
        inherited = state.get("reference_set")
        if reference_policy == "inherit":
            effective_references = inherited
        elif reference_policy == "replace":
            effective_references = reference_set
        else:
            if inherited is None:
                effective_references = reference_set
            elif reference_set is None:
                effective_references = inherited
            else:
                effective_references = dict(inherited)
                for key in ("images", "videos", "video_audios", "audios"):
                    effective_references[key] = tuple(inherited.get(key, ())) + tuple(reference_set.get(key, ()))
        plan = {"type": "HR_CONTINUATION_PLAN", "version": 1, "checkpoint_id": manifest["checkpoint_id"],
                "prompt": str(prompt).strip(), "audio_mode": audio_mode, "reference_policy": reference_policy,
                "reference_set": effective_references}
        summary = {key: value for key, value in plan.items() if key != "reference_set"}
        if effective_references is not None:
            summary["reference_set"] = {
                "images": len(effective_references.get("images", ())),
                "videos": sum(value is not None for value in effective_references.get("videos", ())),
                "video_audios": sum(value is not None for value in effective_references.get("video_audios", ())),
                "audios": len(effective_references.get("audios", ())),
                "ref_image_size": effective_references.get("ref_image_size", "match"),
                "ref_scale": effective_references.get("ref_scale", 1.0),
            }
        return io.NodeOutput(plan, json.dumps(summary, ensure_ascii=False, indent=2))
