"""Compatibility parser for JZL MiniMax H3 storyboard text.

This module contains adapted portions of ComfyUI-JZL-MiniMax-H3's
``story_nodes.py``. The original portions are Copyright (c) 2026 wjluoxiao and
are used under the MIT License. This adapted file is also distributed under the
repository's Apache-2.0 license.

Only pure text/JSON compatibility behavior lives here. ComfyUI nodes, global
asset pools, model loading, sampling, and filesystem side effects deliberately
do not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_STORY_BLOCK = re.compile(r"\[SHOT_START\](.*?)\[SHOT_END\]", re.DOTALL)
_SECTION = re.compile(
    r"(?ms)^===(H3_PROMPT|SCENE_INSTRUCTION|VIDEO_INSTRUCTION|AUDIO_INSTRUCTION)===\s*\n?(.*?)(?=^===(?:H3_PROMPT|SCENE_INSTRUCTION|VIDEO_INSTRUCTION|AUDIO_INSTRUCTION)===\s*$|\Z)"
)
_TYPE_PREFIX = {
    "场景": "场景",
    "角色": "角色",
    "道具": "道具",
    "视频": "视频",
    "音频": "音频",
    "scene": "场景",
    "character": "角色",
    "prop": "道具",
    "video": "视频",
    "audio": "音频",
}


@dataclass(frozen=True)
class StoryBlock:
    """One JZL ``[SHOT_START]`` block in its four-section representation."""

    h3_prompt: str
    scene_instruction: str = "{}"
    video_instruction: str = "{}"
    audio_instruction: str = "{}"
    source: str = ""


def parse_four_in_one(content: str) -> tuple[str, str, str, str]:
    """Return JZL's H3, scene, video, and audio sections.

    Missing dispatch sections retain JZL's historical ``{}`` default. A missing
    H3 section remains an empty string so callers can reject or explicitly use a
    passthrough policy.
    """

    values = {
        "H3_PROMPT": "",
        "SCENE_INSTRUCTION": "{}",
        "VIDEO_INSTRUCTION": "{}",
        "AUDIO_INSTRUCTION": "{}",
    }
    for name, body in _SECTION.findall(str(content or "").replace("\r\n", "\n").replace("\r", "\n")):
        values[name] = body.strip()
    return (
        values["H3_PROMPT"],
        values["SCENE_INSTRUCTION"],
        values["VIDEO_INSTRUCTION"],
        values["AUDIO_INSTRUCTION"],
    )


def parse_story_blocks(text: str) -> tuple[StoryBlock, ...]:
    """Parse every explicitly delimited JZL story block in source order."""

    blocks = []
    normalized = str(text or "").replace("\\n", "\n").replace("\\r", "\n")
    for match in _STORY_BLOCK.finditer(normalized):
        source = match.group(1).strip()
        h3, scene, video, audio = parse_four_in_one(source)
        blocks.append(StoryBlock(h3, scene, video, audio, source))
    return tuple(blocks)


def parse_slots(raw: Any) -> list[Any]:
    """Decode JZL dispatch slots from JSON, a one-item list, or a dict."""

    for _ in range(3):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                return []
        elif isinstance(raw, (list, tuple)):
            if not raw:
                return []
            raw = raw[0]
        elif isinstance(raw, dict):
            slots = raw.get("slots", [])
            return slots if isinstance(slots, list) else []
        else:
            return []
    return []


def parse_material_slots(materials: Any) -> tuple[str, ...]:
    """Return declared JZL slot names in their reference-input order."""

    slots = []
    for line in str(materials or "").splitlines():
        match = re.match(r"^(角色|场景|道具|视频|音频|分镜|音效|音乐|其他)\s*([A-Za-z])\s*[=＝:：]", line.strip())
        if match:
            slots.append(f"{match.group(1)}{match.group(2).upper()}")
    return tuple(slots)


def dispatch_slot_names(info_json: Any) -> tuple[str, ...]:
    """Return the material slot names requested by one JZL instruction."""

    names = []
    for slot in parse_slots(info_json):
        if not isinstance(slot, str) or ":" not in slot:
            continue
        name = slot.split(":", 1)[1].strip().rstrip(":：")
        if name and name not in names:
            names.append(name)
    return tuple(names)


def dispatch_material_indices(requested: Any, declared: Any) -> tuple[int, ...]:
    """Map requested slot names to declared material positions in request order."""

    declared_indices = {name: index for index, name in enumerate(declared)}
    indices = []
    for name in requested:
        if name not in declared_indices:
            raise ValueError(f"JZL slot {name!r} is not declared in the connected material list")
        index = declared_indices[name]
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def normalize_slots(info_json: Any) -> Any:
    """Apply JZL's compatibility repairs without changing valid input."""

    try:
        value = json.loads(info_json) if isinstance(info_json, str) else (info_json or {})
    except (TypeError, json.JSONDecodeError):
        return info_json
    if not isinstance(value, dict):
        return info_json
    slots = value.get("slots") or []
    if not isinstance(slots, list):
        return info_json

    changed = False
    normalized = []
    for slot in slots:
        if not isinstance(slot, str) or ":" not in slot:
            normalized.append(slot)
            continue
        kind, name = slot.split(":", 1)
        kind = kind.strip()
        name = name.strip().rstrip(":：")
        prefix = _TYPE_PREFIX.get(kind.lower() if kind.isascii() else kind)
        if prefix and re.fullmatch(r"[A-H]", name, re.IGNORECASE):
            fixed = f"{kind}:{prefix}{name}"
        else:
            fixed = f"{kind}:{name}"
        normalized.append(fixed)
        changed = changed or fixed != slot

    if not changed:
        return info_json
    result = dict(value)
    result["slots"] = normalized
    return json.dumps(result, ensure_ascii=False)


def planned_frame_count(duration_seconds: float, fps: float) -> int:
    """Return the smallest H3-native ``17k+5`` frame count covering duration."""

    duration = float(duration_seconds)
    frame_rate = float(fps)
    if duration <= 0 or frame_rate <= 0:
        raise ValueError("duration_seconds and fps must be positive")
    frames = max(5, int(round(duration * frame_rate)))
    remainder = (frames - 5) % 17
    return frames if remainder == 0 else frames + 17 - remainder


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error


def validate_storyboard_plan(value: Any, *, image_count: int, total_frames: int) -> dict[str, Any]:
    """Temporary compatibility for saved MVP workflows while native JZL nodes replace it."""

    if not isinstance(value, dict):
        raise ValueError("storyboard response must be a JSON object")
    subjects = value.get("image_subjects")
    shots = value.get("shots")
    if not isinstance(subjects, list) or not subjects or not isinstance(shots, list) or not shots:
        raise ValueError("storyboard response needs image_subjects and shots")
    normalized_subjects = []
    for index, item in enumerate(subjects, 1):
        if not isinstance(item, dict):
            raise ValueError(f"image_subjects[{index}] must be an object")
        picture_value = item.get("picture")
        if isinstance(picture_value, str):
            match = re.search(r"\d+", picture_value)
            picture_value = match.group() if match else picture_value
        picture = _positive_int(picture_value, f"image_subjects[{index}].picture")
        if picture < 1 or picture > image_count:
            raise ValueError(f"Picture {picture} is outside the {image_count} connected images")
        normalized_subjects.append({
            "picture": picture,
            "subject": picture,
            "name": str(item.get("name", "")).strip(),
            "observable_features": str(item.get("observable_features", item.get("description", ""))).strip(),
        })
    normalized_shots = []
    previous_end = 0
    for index, item in enumerate(shots, 1):
        start = _positive_int(item.get("start_frame"), f"shots[{index}].start_frame")
        end = _positive_int(item.get("end_frame"), f"shots[{index}].end_frame")
        if start != previous_end or end <= start or end > total_frames:
            raise ValueError(f"Shot {index} has invalid or non-contiguous frame interval [{start},{end})")
        pictures = []
        for raw in item.get("pictures", []):
            match = re.search(r"\d+", raw) if isinstance(raw, str) else None
            picture = _positive_int(match.group() if match else raw, f"shots[{index}].pictures")
            if picture < 1 or picture > image_count:
                raise ValueError(f"Picture {picture} is outside the {image_count} connected images")
            pictures.append(picture)
        normalized_shots.append({"shot": index, "start_frame": start, "end_frame": end, "pictures": pictures, "description": str(item.get("description", "")).strip()})
        previous_end = end
    if previous_end != total_frames:
        raise ValueError(f"storyboard shots must cover the complete target; ended at {previous_end}, expected {total_frames}")
    result = dict(value)
    result.update(image_subjects=normalized_subjects, shots=normalized_shots, total_frames=total_frames)
    return result


def compile_h3_prompt(plan: dict[str, Any], *, fps: float) -> str:
    subjects = [f"<Subject {item['subject']}> is {item['name']} from <Picture {item['picture']}>: {item['observable_features']}." for item in plan["image_subjects"]]
    shots = []
    for item in plan["shots"]:
        marker = f"[Shot {item['shot']}]"
        if item["shot"] > 1:
            milliseconds = round(item["start_frame"] * 1000 / fps)
            minutes, remainder = divmod(milliseconds, 60000)
            seconds, milliseconds = divmod(remainder, 1000)
            marker += f" At {minutes:02d}:{seconds:02d}.{milliseconds:03d},"
        shots.append(f"{marker} {item['description']}")
    return "\n\n".join((
        "subject_definitions:\n" + "\n".join(subjects),
        "summary:\n" + str(plan.get("summary", "")),
        "retention_analysis:\n" + str(plan.get("retention_analysis", "")),
        "detailed_description:\n" + "\n".join(shots),
        "overall_soundscape:\n" + str(plan.get("overall_soundscape", "")),
        "non_diegetic_music:\n" + str(plan.get("non_diegetic_music", "")),
    ))
