"""Shared structured prompt skill contract for HR Endless Sampler."""

from __future__ import annotations

import re
from typing import Any

try:
    from .story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan
except ImportError:  # Direct worker execution.
    from story_format import compile_h3_prompt, planned_frame_count, validate_storyboard_plan


CONTINUITY_MODES = ("strict", "balanced", "cinematic")


def prompt_skill_messages(request: dict[str, Any]) -> tuple[str, str]:
    image_count = int(request.get("image_count", 0))
    total_frames = int(request["total_frames"])
    fps = float(request["fps"])
    continuity = str(request.get("continuity_mode", "balanced"))
    language = "Chinese" if request.get("prompt_lang", "zh") == "zh" else "English"
    inventory = "\n".join(f"- <Picture {i}>: connected identity/reference picture {i}" for i in range(1, image_count + 1)) or "- none"
    system = f"""You are an HR Endless Sampler prompt skill compiler for MiniMax H3.
Convert the user's ordinary story into a strict structured shot plan. Return exactly one JSON object, no markdown.
Write descriptions in {language}, while preserving dialogue, lyrics, visible text, H3 labels, and required field names exactly.

Hard rules:
- Map every named character to one stable <Subject N>; never assign one name to two subjects.
- Every event has one stable ID such as S2.V1 and may start in only one shot.
- Every shot has camera, start_state, events, end_state, forbidden_replays, audio, start_frame, and end_frame.
- Each event contains id, action, and phase (start, continue, or complete).
- A later shot must use already/completed language instead of restarting an earlier action.
- Adjacent shots must differ in at least one of camera side, scale, height, movement, or subject arrangement.
- Split compound actions connected by then, after, followed by, finally, or physical dependencies.
- Shots cover exactly [0,{total_frames}) with contiguous half-open frame intervals.
- Do not write [Shot N] markers or timestamps; the program owns them.
- Continuity mode is {continuity}: strict strongly forbids visual/action repetition; balanced permits necessary continuation; cinematic permits intentional match cuts only when declared.
"""
    user = f"""Story:
--- BEGIN STORY ---
{str(request['story']).strip()}
--- END STORY ---

Target: {total_frames} frames at {fps:g} fps.
Style: {request.get('style', 'cinematic realism')}.
Shot density: {request.get('shot_density', 'medium')}.
Connected pictures:
{inventory}

Return JSON with:
{{
  "image_subjects": [{{"picture":1,"name":"Name","observable_features":"visible identity details"}}],
  "summary": "complete story summary",
  "retention_analysis": "identity and continuity locks",
  "shots": [{{
    "start_frame": 0,
    "end_frame": {total_frames},
    "pictures": [1],
    "camera": "specific camera setup",
    "start_state": "visible opening state",
    "events": [{{"id":"S1.V1","action":"one observable event","phase":"start"}}],
    "end_state": "visible final state usable by the next shot",
    "forbidden_replays": ["completed event or old composition that must not return"],
    "audio": "synchronized sound",
    "description": "complete H3-ready shot prose using the camera, states, events, prohibitions, and audio"
  }}],
  "overall_soundscape": "global diegetic sound plan",
  "non_diegetic_music": "music plan or N/A",
  "warnings": []
}}
"""
    return system, user


def validate_prompt_skill_result(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    total_frames = int(request["total_frames"])
    image_count = int(request.get("image_count", 0))
    plan = validate_storyboard_plan(value, image_count=image_count, total_frames=total_frames)
    event_owner: dict[str, int] = {}
    previous_camera = None
    ledger_pending = []
    warnings = [str(item) for item in value.get("warnings", ())] if isinstance(value.get("warnings", ()), list) else []
    for index, (raw, normalized) in enumerate(zip(value["shots"], plan["shots"]), 1):
        for name in ("camera", "start_state", "end_state", "audio"):
            if not str(raw.get(name, "")).strip():
                raise ValueError(f"shots[{index}].{name} must be non-empty")
        events = raw.get("events")
        forbidden = raw.get("forbidden_replays")
        if not isinstance(events, list) or not events or not isinstance(forbidden, list):
            raise ValueError(f"shots[{index}] needs events and forbidden_replays arrays")
        normalized_events = []
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"shots[{index}].events items must be objects")
            event_id = str(event.get("id", "")).strip()
            action = str(event.get("action", "")).strip()
            phase = str(event.get("phase", "")).strip()
            if not re.fullmatch(r"S\d+\.[A-Z]\d+", event_id) or not action or phase not in {"start", "continue", "complete"}:
                raise ValueError(f"shots[{index}] has an invalid event")
            if phase == "start" and event_id in event_owner:
                raise ValueError(f"Event {event_id} starts in more than one shot")
            event_owner.setdefault(event_id, index)
            normalized_events.append({"id": event_id, "action": action, "phase": phase})
            ledger_pending.append({"id": event_id, "summary": action, "owner_shot": index})
        camera = str(raw["camera"]).strip()
        if previous_camera and camera.casefold() == previous_camera.casefold() and request.get("continuity_mode") == "strict":
            raise ValueError(f"Shots {index - 1} and {index} repeat the same camera in strict mode")
        previous_camera = camera
        normalized.update(
            camera=camera,
            start_state=str(raw["start_state"]).strip(),
            end_state=str(raw["end_state"]).strip(),
            events=normalized_events,
            forbidden_replays=[str(item).strip() for item in forbidden if str(item).strip()],
            audio=str(raw["audio"]).strip(),
        )
    plan["initial_event_ledger"] = {
        "completed": [], "active": [], "pending": ledger_pending, "forbidden": [],
    }
    plan["warnings"] = warnings
    return plan


def compile_prompt_skill(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    plan = validate_prompt_skill_result(value, request)
    return {
        "prompt": compile_h3_prompt(plan, fps=float(request["fps"])),
        "shot_plan": plan,
        "initial_event_ledger": plan["initial_event_ledger"],
        "warnings": plan["warnings"],
        "planned_frames": int(request["total_frames"]),
    }


def build_prompt_skill_request(story: str, *, duration_seconds: float, fps: float, image_count: int,
                               style: str, shot_density: str, continuity_mode: str, prompt_lang: str) -> dict[str, Any]:
    if not isinstance(story, str) or not story.strip():
        raise ValueError("Prompt Skill Compiler requires a non-empty story")
    if continuity_mode not in CONTINUITY_MODES:
        raise ValueError(f"Unknown continuity mode: {continuity_mode}")
    if prompt_lang not in {"zh", "en"}:
        raise ValueError(f"Unknown prompt language: {prompt_lang}")
    return {
        "story": story.strip(), "duration_seconds": float(duration_seconds), "fps": float(fps),
        "total_frames": planned_frame_count(duration_seconds, fps), "image_count": int(image_count),
        "style": str(style), "shot_density": str(shot_density),
        "continuity_mode": continuity_mode, "prompt_lang": prompt_lang,
    }
