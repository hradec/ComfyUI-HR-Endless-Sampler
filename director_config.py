"""Shared Qwen3.8 configuration for HR planning and chunk directing."""

from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from .director_backend import director_model_options, resolve_director_selection


HRDirectorConfig = io.Custom("HR_DIRECTOR_CONFIG")
CONFIG_VERSION = 1


def normalize_qwen38_config(value: Any) -> dict[str, Any]:
    """Validate one workflow-safe shared director configuration."""

    if not isinstance(value, dict):
        raise ValueError("HR director_config must be produced by HR Qwen3.8 Director Config")
    if int(value.get("version", -1)) != CONFIG_VERSION:
        raise ValueError("Unsupported HR director_config version")
    if value.get("backend") != "qwen3.8":
        raise ValueError("The shared HR director configuration currently supports Qwen3.8 only")
    draft_tokens = int(value.get("mtp_draft_tokens", 2))
    if draft_tokens < 1 or draft_tokens > 8:
        raise ValueError("Qwen3.8 MTP draft tokens must be between 1 and 8")
    reasoning = str(value.get("reasoning_effort", "medium"))
    if reasoning not in {"xhigh", "medium", "low"}:
        raise ValueError(f"Unknown Qwen3.8 reasoning effort: {reasoning}")
    n_cpu_moe = int(value.get("n_cpu_moe", 0))
    if n_cpu_moe < 0:
        raise ValueError("Qwen3.8 n_cpu_moe cannot be negative")
    return {
        "version": CONFIG_VERSION,
        "backend": "qwen3.8",
        "model": str(value.get("model", "auto")),
        "mmproj": str(value.get("mmproj", "auto")),
        "mtp": bool(value.get("mtp", True)),
        "mtp_draft_tokens": draft_tokens,
        "reasoning_effort": reasoning,
        "cpu_moe": bool(value.get("cpu_moe", False)),
        "n_cpu_moe": n_cpu_moe,
        "debug": bool(value.get("debug", False)),
    }


class HRQwen38DirectorConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRQwen38DirectorConfig",
            display_name="HR Qwen3.8 Director Config",
            category="model/sampling/custom",
            description=(
                "One Qwen3.8 model/runtime configuration shared by storyboard planning and HR chunk directing. "
                "Each operation still uses a disposable worker so the model does not remain beside H3 in VRAM."
            ),
            inputs=[
                io.Combo.Input("model", options=director_model_options(), default="auto",
                               tooltip="Local Qwen3.8 GGUF. The same file is used by Planner and Sampler."),
                io.Combo.Input("mmproj", options=director_model_options(projector=True), default="auto",
                               tooltip="Same-directory Qwen3.8 multimodal projector."),
                io.Boolean.Input("mtp", default=True),
                io.Int.Input("mtp_draft_tokens", default=2, min=1, max=8, step=1),
                io.Combo.Input("reasoning_effort", options=["xhigh", "medium", "low"], default="medium"),
                io.Boolean.Input("cpu_moe", default=False, advanced=True),
                io.Int.Input("n_cpu_moe", default=0, min=0, max=256, step=1, advanced=True),
                io.Boolean.Input("debug", default=False, advanced=True),
            ],
            outputs=[HRDirectorConfig.Output(display_name="director_config")],
        )

    @classmethod
    def execute(cls, model="auto", mmproj="auto", mtp=True, mtp_draft_tokens=2,
                reasoning_effort="medium", cpu_moe=False, n_cpu_moe=0, debug=False):
        selection = resolve_director_selection("qwen3.8", model, mmproj)
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("HR Qwen3.8 Director Config requires a local Qwen3.8 GGUF and same-directory mmproj")
        config = normalize_qwen38_config({
            "version": CONFIG_VERSION,
            "backend": "qwen3.8",
            "model": model,
            "mmproj": mmproj,
            "mtp": mtp,
            "mtp_draft_tokens": mtp_draft_tokens,
            "reasoning_effort": reasoning_effort,
            "cpu_moe": cpu_moe,
            "n_cpu_moe": n_cpu_moe,
            "debug": debug,
        })
        return io.NodeOutput(config)
