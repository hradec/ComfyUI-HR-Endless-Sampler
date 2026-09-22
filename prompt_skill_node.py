"""Shared-config H3 long-video prompt skill compiler node."""

from __future__ import annotations

import json

import comfy.model_management
from comfy_api.latest import io

from .director_backend import resolve_director_selection
from .director_config import HRDirectorConfig, normalize_qwen38_config
from .prompt_skill import CONTINUITY_MODES, build_prompt_skill_request
from .qwen35 import Qwen35ContinuityDirector
from .reference_set import HRReferenceSet, reference_images


HRH3EventLedger = io.Custom("HR_H3_EVENT_LEDGER")


class HRH3PromptSkillCompiler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRH3PromptSkillCompiler",
            display_name="HR H3 Prompt Skill Compiler",
            category="model/sampling/custom",
            description=(
                "Compile an ordinary story into a repetition-resistant MiniMax H3 prompt and event-owned shot plan. "
                "Uses exactly the Qwen3.5/3.6/3.8 backend selected by the connected HR Qwen Director Config."
            ),
            inputs=[
                io.String.Input("story", multiline=True, dynamic_prompts=True),
                io.Float.Input("duration_seconds", default=10.0, min=0.21, max=3600.0, step=0.001),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                HRReferenceSet.Input("reference_set"),
                HRDirectorConfig.Input("director_config"),
                io.String.Input("style", default="cinematic realism", advanced=True),
                io.Combo.Input("shot_density", options=["low", "medium", "high"], default="medium", advanced=True),
                io.Combo.Input("continuity_mode", options=list(CONTINUITY_MODES), default="balanced"),
                io.Combo.Input("prompt_lang", options=["zh", "en"], default="zh"),
            ],
            outputs=[
                io.String.Output(display_name="H3 prompt"),
                io.String.Output(display_name="structured shot plan"),
                HRH3EventLedger.Output(display_name="initial event ledger"),
                io.String.Output(display_name="event ledger JSON"),
                io.String.Output(display_name="validation report"),
                io.Int.Output(display_name="planned frames"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, story, duration_seconds, fps, reference_set, director_config,
                style="cinematic realism", shot_density="medium", continuity_mode="balanced", prompt_lang="zh"):
        images = reference_images(reference_set)
        if not images:
            raise ValueError("HR H3 Prompt Skill Compiler requires at least one identity/reference picture")
        config = normalize_qwen38_config(director_config)
        selection = resolve_director_selection(config["backend"], config["model"], config["mmproj"])
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("Prompt Skill Compiler requires a local matching Qwen model and mmproj")
        request = build_prompt_skill_request(
            story, duration_seconds=duration_seconds, fps=fps, image_count=len(images),
            style=style, shot_density=shot_density, continuity_mode=continuity_mode,
            prompt_lang=prompt_lang,
        )
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        director = Qwen35ContinuityDirector(
            selection.model_path, selection.mmproj_path,
            debug=config["debug"], mtp_enabled=config["mtp"],
            mtp_draft_tokens=config["mtp_draft_tokens"],
            reasoning_effort=config["reasoning_effort"],
            cpu_moe=config["cpu_moe"], n_cpu_moe=config["n_cpu_moe"],
            backend=config["backend"],
        )
        result = director.compile_prompt_skill(request, images)
        plan = result["shot_plan"]
        warnings = result.get("warnings", ())
        report = {
            "backend": config["backend"],
            "continuity_mode": continuity_mode,
            "subjects": len(plan.get("image_subjects", ())),
            "shots": len(plan.get("shots", ())),
            "events": sum(len(item.get("events", ())) for item in plan.get("shots", ())),
            "warnings": list(warnings),
        }
        return io.NodeOutput(
            str(result["prompt"]),
            json.dumps(plan, ensure_ascii=False, indent=2),
            result["initial_event_ledger"],
            json.dumps(result["initial_event_ledger"], ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
            int(result["planned_frames"]),
        )
