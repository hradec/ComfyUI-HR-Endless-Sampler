import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "qwen35_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package

errors_spec = importlib.util.spec_from_file_location(PACKAGE + ".director_errors", PLUGIN_ROOT / "director_errors.py")
errors = importlib.util.module_from_spec(errors_spec)
sys.modules[errors_spec.name] = errors
errors_spec.loader.exec_module(errors)

spec = importlib.util.spec_from_file_location(PACKAGE + ".qwen35", PLUGIN_ROOT / "qwen35.py")
qwen35 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qwen35
spec.loader.exec_module(qwen35)


class Qwen35Tests(unittest.TestCase):
    def request(self):
        return {
            "chunk_count": 2,
            "fps": 24,
            "original_prompt": "A named hero walks and speaks.",
            "source_shots": [{
                "shot_number": 1,
                "shot_start": 0,
                "shot_end": 68,
                "source_body": "The hero walks while speaking.",
            }],
        }

    def test_module_does_not_import_gemma_backend(self):
        self.assertNotIn("gemma4", qwen35.__dict__)
        self.assertTrue(all("gemma4" not in value for value in qwen35.Qwen35ContinuityDirector.__mro__[1].__module__.split()))

    def test_timing_prompt_has_only_local_duration_coordinates(self):
        _system, prompt = qwen35._timing_messages(self.request())
        self.assertIn("valid local interval [0,68)", prompt)
        self.assertNotIn("global frames", prompt)
        self.assertNotIn("physical chunk", prompt)

    def test_timing_parser_owns_final_endpoint_and_intersects_overlays(self):
        value = {
            "confidence": "high",
            "analysis": "schedule",
            "character_name_table": [],
            "shots": [{
                "source_shot": "Shot 1",
                "visual_beats": [
                    {"start_frame": 0, "end_frame": 50, "action": "walk"},
                    {"start_frame": 50, "end_frame": 76, "action": "stop"},
                ],
                "overlays": [
                    {"start_frame": 50, "end_frame": 76, "type": "dialogue", "content": "line"},
                ],
            }],
        }
        plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
        self.assertEqual(plan.shots[0].visual_beats[-1].end_frame, 68)
        self.assertEqual((plan.shots[0].overlays[0].start_frame, plan.shots[0].overlays[0].end_frame), (50, 68))

    def test_timing_parser_accepts_qwen_visual_beat_field_names(self):
        for field in ("action", "description", "visual_action", "content", "beat"):
            with self.subTest(field=field):
                value = {
                    "shots": [{
                        "source_shot": 1,
                        "visual_beats": [{"start_frame": 0, "end_frame": 68, field: "The hero walks."}],
                    }],
                }
                plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
                self.assertEqual(plan.shots[0].visual_beats[0].action, "The hero walks.")

    def test_timing_parser_keeps_a_valid_interval_when_action_text_is_missing(self):
        value = {
            "shots": [{
                "source_shot": 1,
                "visual_beats": [{"start_frame": 0, "end_frame": 68}],
            }],
        }
        plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
        self.assertEqual(plan.shots[0].visual_beats[0].action, "Continue the source shot action.")

    def test_chunk_parser_accepts_qwen_prompt_field_names(self):
        for field in ("detailed_description", "h3_prompt", "video_prompt", "chunk_prompt", "prompt", "description", "timing_plan"):
            with self.subTest(field=field):
                value = {"confidence": "high", field: "[Shot 1] Continue the action."}
                result = qwen35._chunk_prompt(value, json.dumps(value), "system", "prompt")
                self.assertEqual(result.detailed_description, "[Shot 1] Continue the action.")

    def test_chunk_parser_reports_actual_keys_when_prompt_is_missing(self):
        value = {"confidence": "low", "analysis": "empty"}
        with self.assertRaisesRegex(qwen35.Qwen35ObservationError, "analysis, confidence"):
            qwen35._chunk_prompt(value, json.dumps(value), "system", "prompt")

    def test_qwen_timing_has_independent_long_output_budget(self):
        self.assertEqual(qwen35.QWEN35_TIMING_RESPONSE_TOKENS, 32768)
        self.assertEqual(qwen35.QWEN35_CHUNK_RESPONSE_TOKENS, 8192)

    def test_qwen_configuration_disables_mtp(self):
        director = qwen35.Qwen35ContinuityDirector(Path("model.gguf"), Path("mmproj.gguf"))
        request = {}
        director._configure_request(request)
        self.assertEqual(request["director_n_ctx"], 65536)
        self.assertFalse(request["gemma4_mtp"])


if __name__ == "__main__":
    unittest.main()
