"""Opt-in, real-GPU Gemma 4 integration test using the current user prompt.

This deliberately sits outside the mocked unit suite.  It performs the same
two disposable-worker operations as a live run: full pre-production planning
from ``prompt.txt`` (625 frames by default), then one real continuation-chunk
prompt using the saved stills from the latest Gemma capture under ``/tmp``.

It never downloads a model and is skipped unless explicitly enabled:

    HR_ENDLESS_SAMPLER_RUN_GEMMA4_LIVE_TEST=1 \\
    HR_ENDLESS_SAMPLER_GEMMA4_LIVE_MTP=1 \\
    ~/comfyui/tools/python.sh -m unittest discover -s tests -p 'test_gemma4_live.py' -v

Set ``HR_ENDLESS_SAMPLER_GEMMA4_LIVE_MTP=0`` for the original-decoder
baseline.  ``HR_ENDLESS_SAMPLER_GEMMA4_CAPTURE`` may select a specific saved
``request.json`` when a different captured chunk should be replayed.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy
import torch
from PIL import Image


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

gemma4 = importlib.import_module(PLUGIN_ROOT.name + ".gemma4")
nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")

DEFAULT_TOTAL_FRAMES = 625


def _live_test_enabled() -> bool:
    return os.environ.get("HR_ENDLESS_SAMPLER_RUN_GEMMA4_LIVE_TEST") == "1"


@unittest.skipUnless(
    _live_test_enabled(),
    "Set HR_ENDLESS_SAMPLER_RUN_GEMMA4_LIVE_TEST=1 to run the real Gemma 4 GPU integration test.",
)
class Gemma4LiveIntegrationTest(unittest.TestCase):
    """Actual prompt/captured-still reproduction, not a toy model request."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.prompt_path = PLUGIN_ROOT / "prompt.txt"
        if not cls.prompt_path.is_file():
            raise unittest.SkipTest(f"No user diagnostic prompt found at {cls.prompt_path}")
        cls.prompt = cls.prompt_path.read_text(encoding="utf-8")
        cls.total_frames = int(os.environ.get("HR_ENDLESS_SAMPLER_GEMMA4_TOTAL_FRAMES", DEFAULT_TOTAL_FRAMES))
        cls.fps = float(os.environ.get("HR_ENDLESS_SAMPLER_GEMMA4_FPS", "24"))
        model_path, projector_path = gemma4._model_paths()
        missing = [str(path) for path in (model_path, projector_path) if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Gemma 4 model files are not installed; this integration test does not download them: "
                + ", ".join(missing)
            )
        cls.capture_path = cls._find_capture()

    @staticmethod
    def _find_capture() -> Path:
        specified = os.environ.get("HR_ENDLESS_SAMPLER_GEMMA4_CAPTURE")
        candidates = [Path(specified)] if specified else sorted(
            Path("/tmp").glob("hr-endless-sampler-gemma4-*/prompt_*/request.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for request_path in candidates:
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                frame_numbers = [int(value) for value in request.get("observation_frame_numbers", ())]
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if frame_numbers and all(
                (request_path.parent / f"frame_{frame:06d}.jpg").is_file()
                for frame in frame_numbers
            ):
                return request_path
        raise unittest.SkipTest(
            "No saved Gemma capture with chronological JPG stills was found under /tmp. "
            "Run one sampler chunk with debug enabled first, or set HR_ENDLESS_SAMPLER_GEMMA4_CAPTURE."
        )

    @classmethod
    def _source_shots(cls) -> list[dict]:
        _markers, parsed_shots, _description_end = nodes._parse_prompt_shots(
            cls.prompt, cls.total_frames, cls.fps,
        )
        if not parsed_shots:
            raise AssertionError("prompt.txt has no detailed_description [Shot N] blocks")
        return [
            {
                "shot_number": shot_index + 1,
                "shot_start": shot_start,
                "shot_end": shot_end,
                "source_body": source_body,
            }
            for shot_index, shot_start, shot_end, source_body in parsed_shots
        ]

    @staticmethod
    def _physical_chunks(total_frames: int, capture: dict) -> list[dict]:
        """Recreate the capture's ordinary carried-prefix physical geometry."""
        output_frames = int(capture["current_chunk"]["sampled_end"]) - int(
            capture["current_chunk"]["sampled_start"]
        )
        carry_frames = int(capture["current_chunk"]["output_start"]) - int(
            capture["current_chunk"]["sampled_start"]
        )
        if output_frames <= 0 or carry_frames < 0:
            raise AssertionError("saved capture has invalid physical chunk geometry")
        chunks: list[dict] = []
        output_start = 0
        while output_start < total_frames:
            produced_frames = output_frames if not chunks else output_frames - carry_frames
            output_end = min(output_start + produced_frames, total_frames)
            chunks.append({
                "sampled_start": max(0, output_start - carry_frames),
                "sampled_end": output_end,
                "output_start": output_start,
                "output_end": output_end,
            })
            output_start = output_end
        return chunks

    @staticmethod
    def _load_observation_frames(request_path: Path, capture: dict) -> torch.Tensor:
        images: list[torch.Tensor] = []
        for frame_number in capture["observation_frame_numbers"]:
            image_path = request_path.parent / f"frame_{int(frame_number):06d}.jpg"
            with Image.open(image_path) as image:
                images.append(torch.from_numpy(numpy.asarray(image.convert("RGB")).copy()).float().div_(255.0))
        return torch.stack(images)

    @classmethod
    def _chunk_request(cls, plan: object, capture: dict) -> dict:
        source_by_number = {item["shot_number"]: item for item in cls._source_shots()}
        request = {key: value for key, value in capture.items() if key not in {
            "image_urls", "debug", "gemma4_mtp", "preproduction_timing_plan",
            "preproduction_current_slice", "mandatory_coverage", "character_name_table",
        }}
        request["original_prompt"] = cls.prompt
        request["target_shots"] = [
            {
                **source_by_number[int(shot["shot_number"])],
                "target_start": int(shot["target_start"]),
                "target_end": int(shot["target_end"]),
                "required_marker": shot.get("required_marker"),
            }
            for shot in capture["target_shots"]
        ]
        request["previous_shots"] = [
            {
                **source_by_number[int(shot["shot_number"])],
                "covered_start": int(shot["covered_start"]),
                "covered_end": int(shot["covered_end"]),
            }
            for shot in capture.get("previous_shots", ())
        ]
        request["character_name_table"] = plan.character_name_table_text()
        request["preproduction_timing_plan"] = plan.for_target_shots(request["target_shots"], cls.fps)
        request["preproduction_current_slice"] = plan.current_slice_coverage_text(request["target_shots"])
        request["mandatory_coverage"] = plan.mandatory_coverage(request["target_shots"])
        return request

    def test_preproduction_then_captured_chunk_prompt(self) -> None:
        """Plan the actual 625-frame prompt, then direct a saved real chunk."""
        capture = json.loads(self.capture_path.read_text(encoding="utf-8"))
        physical_chunks = self._physical_chunks(self.total_frames, capture)
        planning_request = {
            "chunk_count": len(physical_chunks),
            "fps": self.fps,
            "prompt_mode": str(capture.get("prompt_mode", "ref")),
            "source_shots": self._source_shots(),
            "chunks": physical_chunks,
            "original_prompt": self.prompt,
        }
        use_mtp = os.environ.get("HR_ENDLESS_SAMPLER_GEMMA4_LIVE_MTP", "1") != "0"
        director = gemma4.Gemma4ContinuityDirector(debug=True, gemma4_mtp=use_mtp)

        timing_plan = director.plan_timing(planning_request)
        self.assertEqual(len(timing_plan.shots), len(planning_request["source_shots"]))
        self.assertTrue(timing_plan.raw_json.strip().startswith("{"))

        request = self._chunk_request(timing_plan, capture)
        result = director.direct(request, self._load_observation_frames(self.capture_path, capture))
        self.assertTrue(result.raw_json.strip().startswith("{"))
        self.assertTrue(result.detailed_description.strip())

