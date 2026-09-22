"""Native JZL four-in-one storyboard MVP for HR MiniMax H3."""
from __future__ import annotations

import json
from typing import Any

import torch
import comfy.model_management
from comfy_api.latest import io

from .asr import transcribe_references
from .director_backend import resolve_director_selection
from .director_config import HRDirectorConfig, normalize_qwen38_config
from .presets.script import build_shot_prompt
from .qwen35 import Qwen35ContinuityDirector
from .reference_set import HRReferenceSet, normalize_reference_set, reference_images
from .story_format import dispatch_material_indices, dispatch_slot_names, normalize_slots, parse_material_slots, parse_story_blocks

JZL_STORYBOARD = io.Custom("HR_JZL_STORYBOARD")


def _uniform_frames(video: torch.Tensor, count: int) -> list[torch.Tensor]:
    if video.ndim != 4 or video.shape[0] < 1:
        return []
    indices = torch.linspace(0, video.shape[0] - 1, steps=min(count, video.shape[0])).round().long().tolist()
    return [video[index:index + 1] for index in dict.fromkeys(indices)]


def collect_observations(refs: dict[str, Any], *, max_frames: int = 23) -> tuple[list[torch.Tensor], str]:
    """Return all pictures and uniformly sampled reference-video frames."""
    frames = list(reference_images(refs))
    videos = [(index + 1, video) for index, video in enumerate(refs["videos"]) if video is not None]
    remaining = max(0, max_frames - len(frames))
    per_video = max(1, remaining // len(videos)) if videos else 0
    manifest = [f"Picture {index + 1}: reference image" for index in range(len(frames))]
    for index, video in videos:
        selected = _uniform_frames(video, per_video)
        frames.extend(selected)
        manifest.append(f"Video {index}: {len(selected)} uniformly sampled visual frames")
    return frames[:32], "\n".join(manifest)


def compile_jzl_outputs(text: str, *, expected_count: int | None = None) -> tuple[str, list[str], list[str], list[str], list[str]]:
    blocks = parse_story_blocks(text)
    if not blocks:
        raise ValueError("Qwen returned no [SHOT_START]...[SHOT_END] JZL story blocks")
    if expected_count is not None and len(blocks) != int(expected_count):
        raise ValueError(f"Qwen returned {len(blocks)} JZL segments; expected {int(expected_count)}")
    h3 = [block.h3_prompt for block in blocks]
    if any(not item.strip() for item in h3):
        raise ValueError("Every JZL story block must contain a non-empty ===H3_PROMPT=== section")
    return text.strip(), h3, [normalize_slots(b.scene_instruction) for b in blocks], [normalize_slots(b.video_instruction) for b in blocks], [normalize_slots(b.audio_instruction) for b in blocks]


class HRMiniMaxH3JZLStoryboard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3JZLStoryboard",
            display_name="HR MiniMax H3 JZL Storyboard",
            category="model/sampling/custom",
            description="Analyze images and uniformly sampled reference-video frames with local Qwen, transcribe reference audio locally, and emit native JZL four-in-one segments.",
            inputs=[
                io.String.Input("story", multiline=True, dynamic_prompts=True),
                io.Int.Input("segment_count", default=4, min=1, max=48, step=1),
                io.Int.Input("segment_duration", default=8, min=4, max=15, step=1),
                HRReferenceSet.Input("reference_set"),
                HRDirectorConfig.Input("director_config"),
                io.String.Input("whisper_model_path", default="", optional=True, advanced=True,
                                tooltip="Local faster-whisper model directory. Leave empty to omit ASR."),
                io.String.Input("language", default="", optional=True, advanced=True),
                io.String.Input("style", default="热血战斗", advanced=True),
                io.String.Input("image_materials", default="", multiline=True, optional=True, advanced=True,
                                tooltip="JZL slot declarations, for example: 角色A = 女主角（黑色短发）"),
                io.String.Input("video_materials", default="", multiline=True, optional=True, advanced=True),
                io.String.Input("audio_materials", default="", multiline=True, optional=True, advanced=True),
                io.String.Input("custom_rules", default="", multiline=True, optional=True, advanced=True),
            ],
            outputs=[
                JZL_STORYBOARD.Output(display_name="jzl_four_in_one"),
                io.String.Output(display_name="h3_segments"),
                io.String.Output(display_name="scene_segments"),
                io.String.Output(display_name="video_segments"),
                io.String.Output(display_name="audio_segments"),
                io.String.Output(display_name="transcripts"),
                HRReferenceSet.Output(display_name="reference_set"),
            ],
        )

    @classmethod
    def execute(cls, story, segment_count, segment_duration, reference_set, director_config,
                whisper_model_path="", language="", style="热血战斗", image_materials="", video_materials="",
                audio_materials="", custom_rules=""):
        if not str(story or "").strip():
            raise ValueError("JZL Storyboard requires a non-empty story")
        refs = normalize_reference_set(reference_set)
        frames, inventory = collect_observations(refs)
        if not frames:
            raise ValueError("JZL Storyboard requires at least one reference image or video frame")
        reference_audios = [audio for audio in (*refs["video_audios"], *refs["audios"]) if audio is not None]
        if reference_audios and not str(whisper_model_path or "").strip():
            raise ValueError("whisper_model_path is required to analyze connected reference audio")
        transcripts = transcribe_references(
            reference_audios, str(whisper_model_path or ""), language=str(language or ""),
        ) if reference_audios else []
        config = normalize_qwen38_config(director_config)
        selection = resolve_director_selection(config["backend"], config["model"], config["mmproj"])
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("JZL Storyboard requires a local Qwen GGUF model and matching mmproj")
        system = build_shot_prompt(
            str(story), segment_count_label=str(int(segment_count)), segment_duration=int(segment_duration),
            story_style=str(style or "热血战斗"), custom_rules=str(custom_rules or ""),
            ref_image_intro=str(image_materials or ""), ref_video_intro=str(video_materials or ""),
            ref_audio_intro=str(audio_materials or ""),
        )
        user = (
            "Return ONLY native JZL text, with one [SHOT_START]...[SHOT_END] block per requested segment. "
            "Each block MUST contain ===H3_PROMPT===, ===SCENE_INSTRUCTION===, ===VIDEO_INSTRUCTION===, "
            "and ===AUDIO_INSTRUCTION===. Use the supplied visual observations; do not invent reference identities. "
            "Keep dialogue, lyrics, and visible text in the story's original language.\n\n"
            f"Requested segments: {int(segment_count)}\nReference inventory:\n{inventory}\n"
            f"Local ASR transcripts: {json.dumps(transcripts, ensure_ascii=False)}\nStory:\n{story}"
        )
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        director = Qwen35ContinuityDirector(
            selection.model_path, selection.mmproj_path, debug=config["debug"], mtp_enabled=config["mtp"],
            mtp_draft_tokens=config["mtp_draft_tokens"], reasoning_effort=config["reasoning_effort"],
            cpu_moe=config["cpu_moe"], n_cpu_moe=config["n_cpu_moe"], backend=config["backend"],
        )
        raw = director.plan_jzl_storyboard(frames, system_prompt=system, user_prompt=user)
        jzl, h3, scene, video, audio = compile_jzl_outputs(raw, expected_count=int(segment_count))
        encoded = lambda value: json.dumps(value, ensure_ascii=False)
        storyboard = {
            "text": jzl,
            "image_slots": parse_material_slots(image_materials),
            "video_slots": parse_material_slots(video_materials),
            "audio_slots": parse_material_slots(audio_materials),
        }
        return io.NodeOutput(storyboard, encoded(h3), encoded(scene), encoded(video), encoded(audio), encoded(transcripts), refs)


class HRMiniMaxH3JZLSegmentDispatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3JZLSegmentDispatcher", display_name="HR MiniMax H3 JZL Segment Dispatcher",
            category="model/sampling/custom", description="Select one JZL segment and reorder its references to match the segment-local Picture, Video, and Audio labels.",
            inputs=[
                JZL_STORYBOARD.Input("jzl_four_in_one"), io.String.Input("h3_segments"),
                io.String.Input("scene_segments"), io.String.Input("video_segments"), io.String.Input("audio_segments"),
                HRReferenceSet.Input("reference_set"), io.Int.Input("segment_index", default=1, min=1, max=48, step=1),
            ],
            outputs=[io.String.Output(display_name="h3_prompt"), io.String.Output(display_name="scene_instruction"),
                     io.String.Output(display_name="video_instruction"), io.String.Output(display_name="audio_instruction"),
                     HRReferenceSet.Output(display_name="reference_set")],
        )

    @classmethod
    def execute(cls, jzl_four_in_one, h3_segments, scene_segments, video_segments, audio_segments, reference_set, segment_index=1):
        def select(value, name):
            try:
                items = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError as error:
                raise ValueError(f"{name} is not a JSON list") from error
            if not isinstance(items, list) or not 1 <= int(segment_index) <= len(items):
                raise ValueError(f"segment_index {segment_index} is outside {name} list")
            return str(items[int(segment_index) - 1])
        h3 = select(h3_segments, "h3_segments")
        scene = select(scene_segments, "scene_segments")
        video = select(video_segments, "video_segments")
        audio = select(audio_segments, "audio_segments")
        if not isinstance(jzl_four_in_one, dict):
            return io.NodeOutput(h3, scene, video, audio, reference_set)

        refs = normalize_reference_set(reference_set)
        image_indices = dispatch_material_indices(
            dispatch_slot_names(scene), tuple(jzl_four_in_one.get("image_slots", ())),
        )
        video_indices = dispatch_material_indices(
            dispatch_slot_names(video), tuple(jzl_four_in_one.get("video_slots", ())),
        )
        audio_indices = dispatch_material_indices(
            dispatch_slot_names(audio), tuple(jzl_four_in_one.get("audio_slots", ())),
        )
        connected_videos = tuple(
            (item, refs["video_audios"][index] if index < len(refs["video_audios"]) else None)
            for index, item in enumerate(refs["videos"]) if item is not None
        )
        if image_indices and max(image_indices) >= len(refs["images"]):
            raise ValueError("JZL image material declarations exceed the connected reference images")
        if video_indices and max(video_indices) >= len(connected_videos):
            raise ValueError("JZL video material declarations exceed the connected reference videos")
        if audio_indices and max(audio_indices) >= len(refs["audios"]):
            raise ValueError("JZL audio material declarations exceed the connected reference audios")

        selected_videos = tuple(connected_videos[index] for index in video_indices)
        selected_refs = normalize_reference_set({
            **refs,
            "images": tuple(refs["images"][index] for index in image_indices),
            "videos": tuple(item[0] for item in selected_videos),
            "video_audios": tuple(item[1] for item in selected_videos),
            "audios": tuple(refs["audios"][index] for index in audio_indices),
        })
        return io.NodeOutput(h3, scene, video, audio, selected_refs)
