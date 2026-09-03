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
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error
    return result


def validate_storyboard_plan(value: Any, *, image_count: int, total_frames: int) -> dict[str, Any]:
    """Validate and normalize one model-authored storyboard plan."""

    if not isinstance(value, dict):
        raise ValueError("storyboard response must be a JSON object")
    image_count = _positive_int(image_count, "image_count")
    total_frames = _positive_int(total_frames, "total_frames")
    if image_count < 1 or image_count > 9:
        raise ValueError("image_count must be between 1 and 9")
    if total_frames < 5:
        raise ValueError("total_frames must be at least 5")

    subjects = value.get("image_subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("image_subjects must be a non-empty array")
    normalized_subjects = []
    seen_pictures = set()
    seen_subjects = set()
    for index, item in enumerate(subjects, 1):
        if not isinstance(item, dict):
            raise ValueError(f"image_subjects[{index}] must be an object")
        picture = _positive_int(item.get("picture"), f"image_subjects[{index}].picture")
        subject = _positive_int(item.get("subject", picture), f"image_subjects[{index}].subject")
        if picture < 1 or picture > image_count:
            raise ValueError(f"Picture {picture} is outside the {image_count} connected images")
        if picture in seen_pictures or subject in seen_subjects:
            raise ValueError("image subject picture and subject numbers must be unique")
        name = str(item.get("name", "")).strip()
        features = str(item.get("observable_features", item.get("description", ""))).strip()
        if not name or not features:
            raise ValueError(f"image_subjects[{index}] needs name and observable_features")
        seen_pictures.add(picture)
        seen_subjects.add(subject)
        normalized_subjects.append({"picture": picture, "subject": subject, "name": name, "observable_features": features})

    shots = value.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("shots must be a non-empty array")
    normalized_shots = []
    previous_end = 0
    for index, item in enumerate(shots, 1):
        if not isinstance(item, dict):
            raise ValueError(f"shots[{index}] must be an object")
        number = _positive_int(item.get("shot", index), f"shots[{index}].shot")
        start = _positive_int(item.get("start_frame"), f"shots[{index}].start_frame")
        end = _positive_int(item.get("end_frame"), f"shots[{index}].end_frame")
        if number != index:
            raise ValueError(f"storyboard shots must be sequential; expected Shot {index}, got {number}")
        if start != previous_end:
            raise ValueError(f"storyboard shots must be contiguous; Shot {index} starts at {start} after {previous_end}")
        if end <= start or end > total_frames:
            raise ValueError(f"Shot {index} has invalid frame interval [{start},{end})")
        raw_pictures = item.get("pictures", [])
        if not isinstance(raw_pictures, list):
            raise ValueError(f"shots[{index}].pictures must be an array")
        pictures = []
        for raw_picture in raw_pictures:
            picture = _positive_int(raw_picture, f"shots[{index}].pictures")
            if picture < 1 or picture > image_count:
                raise ValueError(f"Picture {picture} is outside the {image_count} connected images")
            if picture not in pictures:
                pictures.append(picture)
        description = str(item.get("description", item.get("visual_action", ""))).strip()
        if not description:
            raise ValueError(f"Shot {index} needs a description")
        normalized_shots.append({"shot": number, "start_frame": start, "end_frame": end, "pictures": pictures, "description": description})
        previous_end = end
    if previous_end != total_frames:
        raise ValueError(f"storyboard shots must cover the complete target; ended at {previous_end}, expected {total_frames}")

    result = dict(value)
    result["image_subjects"] = normalized_subjects
    result["shots"] = normalized_shots
    result["total_frames"] = total_frames
    return result


def frame_timestamp(frame: int, fps: float) -> str:
    """Format a global frame as the sampler's millisecond shot timestamp."""

    frame = _positive_int(frame, "frame")
    fps = float(fps)
    if frame < 0 or fps <= 0:
        raise ValueError("frame must be non-negative and fps must be positive")
    milliseconds = int(round(frame * 1000.0 / fps))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def compile_h3_prompt(plan: dict[str, Any], *, fps: float) -> str:
    """Compile validated storyboard data into a deterministic Ref2VA H3 prompt."""

    subjects = []
    for item in plan["image_subjects"]:
        subjects.append(
            f"<Subject {item['subject']}> is {item['name']} from <Picture {item['picture']}>: "
            f"{item['observable_features']}."
        )
    shot_lines = []
    for item in plan["shots"]:
        marker = f"[Shot {item['shot']}]"
        if item["shot"] > 1:
            marker += f" At {frame_timestamp(item['start_frame'], fps)},"
        shot_lines.append(f"{marker} {item['description']}")
    summary = str(plan.get("summary", "[reference generation] Generate the planned continuous story using the supplied picture references.")).strip()
    retention = str(plan.get("retention_analysis", "")).strip()
    if not retention:
        retention = "\n".join(
            f"<Subject {item['subject']}>: fully_preserved - {item['observable_features']}."
            for item in plan["image_subjects"]
        )
    soundscape = str(plan.get("overall_soundscape", "Use synchronized diegetic sound appropriate to the visible actions and preserve supplied dialogue exactly.")).strip()
    music = str(plan.get("non_diegetic_music", "No non-diegetic music unless explicitly required by the story.")).strip()
    return "\n\n".join((
        "subject_definitions:\n" + "\n".join(subjects),
        "summary:\n" + summary,
        "retention_analysis:\n" + retention,
        "detailed_description:\n" + "\n".join(shot_lines),
        "overall_soundscape:\n" + soundscape,
        "non_diegetic_music:\n" + music,
    ))
