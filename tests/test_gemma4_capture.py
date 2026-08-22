from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import gemma4  # noqa: E402


class GemmaCaptureTest(unittest.TestCase):
    @staticmethod
    def request():
        return {
            "chunk_number": 7,
            "chunk_count": 8,
            "fps": 24.0,
            "prompt_mode": "ref",
            "current_chunk": {
                "sampled_start": 204,
                "sampled_end": 243,
                "output_start": 209,
                "output_end": 243,
            },
            "previous_chunk": {
                "sampled_start": 170,
                "sampled_end": 209,
                "output_start": 175,
                "output_end": 209,
            },
            "previous_shots": [{
                "shot_number": 4,
                "shot_start": 193,
                "shot_end": 220,
                "covered_start": 193,
                "covered_end": 209,
                "source_body": "Heman dismounts the tiger and walks right.",
            }],
            "observation_frame_numbers": [170, 208],
            "target_shots": [
                {
                    "shot_number": 4,
                    "shot_start": 193,
                    "shot_end": 220,
                    "target_start": 209,
                    "target_end": 220,
                    "required_marker": "[Shot 1]",
                    "source_body": "Heman dismounts the tiger and walks right.",
                },
                {
                    "shot_number": 5,
                    "shot_start": 220,
                    "shot_end": 260,
                    "target_start": 220,
                    "target_end": 243,
                    "required_marker": "[Shot 2] At 00:00.667,",
                    "source_body": "Heman says: <d>[English] Stay back!</d>",
                },
            ],
            "conditioning_context": "a bounded 22-frame continuation reference as <Video 1>",
            "original_prompt": (
                "detailed_description: [Shot 4] At 00:08.042, Heman dismounts the tiger and walks right. "
                "[Shot 5] At 00:09.167, Heman says: <d>[English] Stay back!</d>"
            ),
        }

    def test_debug_capture_preserves_exact_request_images_prompts_and_response(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="Heman is already on the ground and moving right.",
            detailed_description=(
                "[Shot 1] Heman continues walking toward the right. "
                "[Shot 2] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
            raw_json='{"confidence":"high"}',
        )
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        frames[1, :, :, 0] = 1.0
        request = self.request()
        with tempfile.TemporaryDirectory() as temp_dir:
            director = gemma4.Gemma4ContinuityDirector(debug=True, capture_directory=temp_dir)
            captured_request = {}

            def fake_worker(request):
                captured_request.update(json.loads(json.dumps(request)))
                request.clear()
                return result

            with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker):
                actual = director.direct(request, frames)

            self.assertEqual(actual, result)
            capture_dir = next(Path(temp_dir).glob("prompt_*"))
            saved_request = json.loads((capture_dir / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_request, captured_request)
            self.assertEqual(len(list(capture_dir.glob("frame_*.jpg"))), 2)
            self.assertIn("Heman dismounts the tiger and walks right", (capture_dir / "observation_prompt.txt").read_text())
            system_prompt = (capture_dir / "system_prompt.txt").read_text()
            self.assertIn("not maintaining an algorithmic action ledger", system_prompt)
            self.assertIn("# H3 Prompt Writing", system_prompt)
            self.assertIn("# Full-Reference Mode Rewrite Output Format Guide", system_prompt)
            self.assertNotIn("# Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)", system_prompt)
            saved_response = json.loads((capture_dir / "response.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_response["detailed_description"], result.detailed_description)
            replay_request = gemma4.load_gemma_capture(capture_dir)
            self.assertEqual(replay_request["image_urls"], saved_request["image_urls"])

    def test_validates_exact_multi_shot_markers_and_dialogue(self):
        request = self.request()
        value = {
            "confidence": "high",
            "analysis": "The dismount is complete; finish walking, then cut.",
            "detailed_description": (
                "[Shot 1] Heman continues walking right. "
                "[Shot 2] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        result = gemma4._validate_chunk_prompt(value, request, json.dumps(value))
        self.assertEqual(result.detailed_description, value["detailed_description"])

        wrong_marker = dict(value)
        wrong_marker["detailed_description"] = value["detailed_description"].replace("00:00.667", "00:00.500")
        with self.assertRaises(gemma4.Gemma4ObservationError):
            gemma4._validate_chunk_prompt(wrong_marker, request, json.dumps(wrong_marker))

        changed_dialogue = dict(value)
        changed_dialogue["detailed_description"] = value["detailed_description"].replace("Stay back!", "Run away!")
        with self.assertRaises(gemma4.Gemma4ObservationError):
            gemma4._validate_chunk_prompt(changed_dialogue, request, json.dumps(changed_dialogue))


if __name__ == "__main__":
    unittest.main()
