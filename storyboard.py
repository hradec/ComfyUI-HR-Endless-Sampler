"""HR-style local multimodal storyboard planner for MiniMax H3."""

from __future__ import annotations

import json


import comfy.model_management
from comfy_api.latest import io

from .director_backend import resolve_director_selection
from .director_config import HRDirectorConfig, normalize_qwen38_config
from .qwen35 import Qwen35ContinuityDirector
from .reference_set import HRReferenceSet, reference_images
from .story_format import planned_frame_count


class HRMiniMaxH3StoryboardPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3StoryboardPlanner",
            display_name="HR MiniMax H3 Storyboard Planner",
            category="model/sampling/custom",
            description=(
                "Uses the selected local Qwen multimodal director to turn ordered reference images, a story, "
                "and a target duration into one global MiniMax H3 prompt for HR Endless Sampler."
            ),
            inputs=[
                io.String.Input(
                    "story",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Story or screenplay. Original dialogue, lyrics, and visible text remain in their source language.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=10.0,
                    min=0.21,
                    max=3600.0,
                    step=0.001,
                    tooltip="Requested total duration. The planned frame count is aligned upward to MiniMax H3's 17k+5 grid.",
                ),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                HRReferenceSet.Input(
                    "reference_set",
                    tooltip="Shared H3 references. The planner currently analyzes its ordered pictures; conditioning and sampler also receive its video/audio references.",
                ),
                HRDirectorConfig.Input(
                    "director_config",
                    tooltip="Connect HR Qwen3.8 Director Config. Connect the same output to HR Endless Sampler.",
                ),
                io.String.Input(
                    "style",
                    default="cinematic realism",
                    multiline=False,
                    advanced=True,
                    tooltip="Visual style applied across the complete storyboard.",
                ),
                io.Combo.Input(
                    "shot_density",
                    options=["low", "medium", "high"],
                    default="medium",
                    advanced=True,
                    tooltip="Suggested cut density. The model still has to cover the complete target frame interval.",
                ),

            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="story_plan"),
                io.String.Output(display_name="shot_report"),
                io.String.Output(display_name="warnings"),
                io.Int.Output(display_name="planned_frames"),
            ],
        )

    @classmethod
    def execute(
        cls,
        story,
        duration_seconds,
        fps,
        reference_set,
        director_config,
        style="cinematic realism",
        shot_density="medium",
    ):
        if not isinstance(story, str) or not story.strip():
            raise ValueError("HR MiniMax H3 Storyboard Planner requires a non-empty story")
        images = reference_images(reference_set)
        if not images:
            raise ValueError("HR MiniMax H3 Storyboard Planner requires at least one picture in reference_set")
        config = normalize_qwen38_config(director_config)
        selection = resolve_director_selection("qwen3.8", config["model"], config["mmproj"])
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("Storyboard Planner requires a local Qwen GGUF model and same-family mmproj")

        # The planner is itself a GPU operation. Release any ComfyUI model owners
        # before starting its disposable llama.cpp worker so 12 GB systems do not
        # retain H3 beside the VLM.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        director = Qwen35ContinuityDirector(
            selection.model_path,
            selection.mmproj_path,
            debug=config["debug"],
            mtp_enabled=config["mtp"],
            mtp_draft_tokens=config["mtp_draft_tokens"],
            reasoning_effort=config["reasoning_effort"],
            cpu_moe=config["cpu_moe"],
            n_cpu_moe=config["n_cpu_moe"],
            backend="qwen3.8",
        )
        result = director.plan_storyboard(
            story,
            images,
            duration_seconds=float(duration_seconds),
            fps=float(fps),
            style=style,
            shot_density=shot_density,
        )
        planned_frames = planned_frame_count(duration_seconds, fps)
        plan = result.get("story_plan")
        if not isinstance(plan, dict) or int(plan.get("total_frames", -1)) != planned_frames:
            raise ValueError("Storyboard worker returned a plan with an unexpected total frame count")
        return io.NodeOutput(
            str(result["prompt"]),
            json.dumps(plan, ensure_ascii=False, indent=2),
            str(result.get("shot_report", "")),
            "\n".join(str(item) for item in result.get("warnings", ())),
            planned_frames,
        )
