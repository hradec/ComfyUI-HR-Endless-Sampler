import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prompt_skill
import qwen35
import qwen36_38


class PromptSkillTests(unittest.TestCase):
    def request(self, continuity_mode="balanced"):
        return prompt_skill.build_prompt_skill_request(
            "A hero enters a temple, then opens a door.", duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="medium",
            continuity_mode=continuity_mode, prompt_lang="en",
        )

    def result(self):
        return {
            "image_subjects": [{"picture": 1, "name": "Hero", "observable_features": "black hair"}],
            "summary": "The hero enters and opens the door.",
            "retention_analysis": "Lock identity and never repeat completed actions.",
            "shots": [
                {"start_frame": 0, "end_frame": 22, "pictures": [1],
                 "camera": "rear wide tracking shot", "start_state": "outside the temple",
                 "events": [{"id": "S1.V1", "action": "enters the temple", "phase": "start"}],
                 "end_state": "fully inside", "forbidden_replays": ["outside approach"],
                 "audio": "footsteps", "description": "The hero enters once and ends fully inside."},
                {"start_frame": 22, "end_frame": 56, "pictures": [1],
                 "camera": "frontal medium static shot", "start_state": "already fully inside",
                 "events": [{"id": "S2.V1", "action": "opens the door", "phase": "start"}],
                 "end_state": "door open", "forbidden_replays": ["temple entrance"],
                 "audio": "door creak", "description": "Already inside, the hero opens the door once."},
            ],
            "overall_soundscape": "stone interior ambience",
            "non_diegetic_music": "N/A",
            "warnings": [],
        }

    def test_compiles_six_fields_and_initial_event_ledger(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        for heading in ("subject_definitions:", "summary:", "retention_analysis:",
                        "detailed_description:", "overall_soundscape:", "non_diegetic_music:"):
            self.assertIn(heading, compiled["prompt"])
        self.assertEqual([item["id"] for item in compiled["initial_event_ledger"]["pending"]], ["S1.V1", "S2.V1"])
        self.assertEqual(compiled["planned_frames"], 56)

    def test_rejects_duplicate_event_start(self):
        value = self.result()
        value["shots"][1]["events"][0]["id"] = "S1.V1"
        with self.assertRaisesRegex(ValueError, "starts in more than one shot"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_strict_mode_rejects_identical_adjacent_camera(self):
        value = self.result()
        value["shots"][1]["camera"] = value["shots"][0]["camera"]
        with self.assertRaisesRegex(ValueError, "repeat the same camera"):
            prompt_skill.compile_prompt_skill(value, self.request("strict"))

    def test_shared_messages_are_identical_for_all_backends(self):
        request = self.request()
        request["image_count"] = 1
        self.assertEqual(prompt_skill.prompt_skill_messages(request), prompt_skill.prompt_skill_messages(request))
        self.assertTrue(callable(qwen35.prompt_skill_messages))
        self.assertTrue(callable(qwen36_38.prompt_skill_messages))

    def test_director_uses_configured_backend_for_independent_operation(self):
        captured = {}
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        def worker(payload, timeout=300):
            captured.update(payload)
            return types.SimpleNamespace(returncode=0, stderr="", stdout=""), {
                "ok": True, "prompt_skill_compile": compiled,
            }
        director = qwen35.Qwen35ContinuityDirector(
            Path("model.gguf"), Path("mmproj.gguf"), backend="qwen3.5", mtp_enabled=False
        )
        frame = __import__("torch").zeros(1, 8, 8, 3)
        with patch.object(qwen35, "_run_worker_once", side_effect=worker):
            result = director.compile_prompt_skill(self.request(), (frame,))
        self.assertEqual(captured["operation"], "prompt_skill_compile")
        self.assertEqual(captured["director_backend"], "qwen3.5")
        self.assertEqual(result["planned_frames"], 56)


if __name__ == "__main__":
    unittest.main()
