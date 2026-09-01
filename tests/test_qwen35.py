import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest

import torch
from pathlib import Path
from unittest.mock import patch


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

    def test_timing_parser_normalizes_global_frames_to_shot_local_frames(self):
        request = self.request()
        request["source_shots"][0].update(shot_start=68, shot_end=163)
        value = {
            "shots": [{
                "source_shot": 1,
                "visual_beats": [
                    {"start_frame": 68, "end_frame": 95, "action": "walk"},
                    {"start_frame": 95, "end_frame": 163, "action": "stop"},
                ],
                "overlays": [{"start_frame": 80, "end_frame": 100, "type": "sound", "content": "footsteps"}],
            }],
        }
        plan = qwen35._timing_plan(value, request, json.dumps(value), "system", "prompt")
        self.assertEqual(
            [(beat.start_frame, beat.end_frame) for beat in plan.shots[0].visual_beats],
            [(0, 27), (27, 95)],
        )
        self.assertEqual((plan.shots[0].overlays[0].start_frame, plan.shots[0].overlays[0].end_frame), (12, 32))

    def test_timing_parser_normalizes_qwen_frame_bound_field_names(self):
        value = {
            "shots": [{
                "source_shot": 1,
                "visual_beats": [{"frame_start": 0, "frame_end": 68, "action": "walk"}],
            }],
        }
        plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
        self.assertEqual(
            [(beat.start_frame, beat.end_frame, beat.action) for beat in plan.shots[0].visual_beats],
            [(0, 68, "walk")],
        )

    def test_timing_parser_infers_missing_end_frame(self):
        value = {
            "shots": [{
                "source_shot": 1,
                "visual_beats": [
                    {"start_frame": 0, "action": "walk"},
                    {"start_frame": 40, "action": "stop"},
                ],
            }],
        }
        plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
        self.assertEqual(
            [(beat.start_frame, beat.end_frame, beat.action) for beat in plan.shots[0].visual_beats],
            [(0, 40, "walk"), (40, 68, "stop")],
        )

    def test_timing_parser_ignores_overlay_without_frame_bounds(self):
        value = {
            "shots": [{
                "source_shot": 1,
                "visual_beats": [{"start_frame": 0, "end_frame": 68, "action": "walk"}],
                "overlays": [{"type": "dialogue", "content": "line"}],
            }],
        }
        plan = qwen35._timing_plan(value, self.request(), json.dumps(value), "system", "prompt")
        self.assertEqual(plan.shots[0].overlays, ())

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

    def test_chunk_prompt_includes_markers_and_continuity_context(self):
        request = {
            **self.request(),
            "chunk_number": 2,
            "target_shots": [{
                "shot_number": 1, "shot_start": 0, "shot_end": 68,
                "required_marker": "[Shot 1] At 00:00.208,",
                "source_body": "The hero walks while speaking.",
            }],
            "preproduction_timing_plan": "timing schedule",
            "mandatory_coverage": [{"id": "S1.V1"}],
            "character_name_table": "Hero -> <Subject 1>",
            "conditioning_context": "<Video 1> and <Audio 1>",
            "observation_frame_numbers": [40, 60],
            "previous_gemma_description": "previous prompt",
            "previous_gemma_timing_plan": "previous timing",
            "previous_gemma_end_state": "previous state",
            "previous_last_seen_character_state": [{"character": "Hero"}],
        }
        _system, prompt = qwen35._chunk_messages(request)
        for text in (
            "[Shot 1] At 00:00.208,", "Hero -> <Subject 1>", "<Video 1> and <Audio 1>",
            "40, 60", "previous prompt", "previous timing", "previous state", '"character": "Hero"',
        ):
            self.assertIn(text, prompt)

    def test_chunk_parser_accepts_qwen_prompt_field_names(self):
        for field in ("detailed_description", "h3_prompt", "video_prompt", "chunk_prompt", "prompt", "description", "timing_plan"):
            with self.subTest(field=field):
                value = {"confidence": "high", field: "[Shot 1] Continue the action."}
                result = qwen35._chunk_prompt(value, json.dumps(value), "system", "prompt")
                self.assertEqual(result.detailed_description, "[Shot 1] Continue the action.")

    def test_chunk_parser_restores_required_h3_marker_without_rewriting_model_text(self):
        value = {"confidence": "high", "detailed_description": "Continue the action."}
        request = {"target_shots": [{"required_marker": "[Shot 2] At 00:00.208,"}]}
        result = qwen35._chunk_prompt(value, json.dumps(value), "system", "prompt", request)
        self.assertEqual(result.detailed_description, "[Shot 2] At 00:00.208, Continue the action.")

    def test_chunk_parser_reports_actual_keys_when_prompt_is_missing(self):
        value = {"confidence": "low", "analysis": "empty"}
        with self.assertRaisesRegex(qwen35.Qwen35ObservationError, "analysis, confidence"):
            qwen35._chunk_prompt(value, json.dumps(value), "system", "prompt")

    def test_qwen_family_context_and_output_budgets(self):
        self.assertEqual(qwen35.QWEN35_CONTEXT_TOKENS, 65536)
        self.assertEqual(qwen35.QWEN35_TIMING_RESPONSE_TOKENS, 32768)
        self.assertEqual(qwen35.QWEN35_CHUNK_RESPONSE_TOKENS, 8192)
        self.assertEqual(qwen35.QWEN36_CONTEXT_TOKENS, 32768)
        self.assertEqual(qwen35.QWEN36_TIMING_RESPONSE_TOKENS, 8192)
        self.assertEqual(qwen35.QWEN36_CHUNK_RESPONSE_TOKENS, 4096)
        self.assertEqual(qwen35.QWEN38_CONTEXT_TOKENS, 32768)
        self.assertEqual(qwen35.QWEN38_TIMING_RESPONSE_TOKENS, 8192)
        self.assertEqual(qwen35.QWEN38_CHUNK_RESPONSE_TOKENS, 4096)

    def test_qwen38_configuration_enables_mtp(self):
        director = qwen35.Qwen35ContinuityDirector(
            Path("qwen3.8-model.gguf"), Path("mmproj-qwen3.8.gguf"), mtp_draft_tokens=3,
            reasoning_effort="medium", cpu_moe=True, n_cpu_moe=4, backend="qwen3.8",
        )
        request = {}
        director._configure_request(request)
        self.assertEqual(request["director_n_ctx"], 32768)
        self.assertFalse(request["gemma4_mtp"])
        self.assertTrue(request["director_mtp"])
        self.assertEqual(request["director_mtp_draft_tokens"], 3)
        self.assertEqual(request["director_reasoning_effort"], "medium")
        self.assertTrue(request["director_cpu_moe"])
        self.assertEqual(request["director_n_cpu_moe"], 4)

    def test_qwen38_runtime_passes_partial_cpu_moe_to_llama(self):
        captured = {}

        class FakeLlama:
            metadata = {"tokenizer.chat_template": "template"}

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def create_chat_completion(self, **kwargs):
                value = {
                    "shots": [{
                        "source_shot": 1,
                        "visual_beats": [{"start_frame": 0, "end_frame": 68, "action": "walk"}],
                    }],
                }
                return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], "usage": {}}

            def close(self):
                pass

        request = self.request()
        request.update(
            operation="timing_plan",
            director_backend="qwen3.8",
            director_model_path="qwen3.8-model.gguf",
            director_mmproj_path="mmproj-qwen3.8.gguf",
            director_cpu_moe=False,
            director_n_cpu_moe=8,
            director_mtp=False,
        )
        runtime = (FakeLlama, object, object, object, object, object, object)
        with patch.object(qwen35, "_load_runtime", return_value=runtime), patch.object(qwen35, "_qwen38_text_handler", return_value=None):
            qwen35._complete(request)
        self.assertEqual(captured["n_cpu_moe"], 8)
        self.assertNotIn("cpu_moe", captured)
        self.assertEqual(captured["n_ctx"], 32768)

    def test_qwen35_configuration_always_disables_mtp(self):
        director = qwen35.Qwen35ContinuityDirector(
            Path("qwen3.5-model.gguf"), Path("mmproj-qwen3.5.gguf"), mtp_enabled=True, backend="qwen3.5",
        )
        request = {}
        director._configure_request(request)
        self.assertEqual(request["director_n_ctx"], 65536)
        self.assertFalse(request["director_mtp"])

    def test_selected_qwen_backend_is_propagated_to_worker_request(self):
        director = qwen35.Qwen35ContinuityDirector(Path("qwen3.8.gguf"), Path("mmproj-qwen3.8.gguf"), backend="qwen3.8")
        request = {}
        director._configure_request(request)
        self.assertEqual(request["director_backend"], "qwen3.8")

    def test_qwen_family_detection_supports_new_series(self):
        self.assertEqual(qwen35._qwen_family("Qwen3.6-VL-35B.gguf"), "qwen3.6")
        self.assertEqual(qwen35._qwen_family("Qwen3.8-27B.gguf"), "qwen3.8")
        self.assertEqual(qwen35._qwen_family("Qwen3.5-9B.gguf"), "qwen3.5")

    def test_qwen38_mtmd_template_replaces_image_pad(self):
        template = "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        adapted = qwen35._adapt_qwen38_mtmd_template(template)
        self.assertNotIn("<|image_pad|>", adapted)
        self.assertIn("item.image_url", adapted)

    def test_visual_mtp_generation_uses_physical_mtmd_ledger(self):
        captured = {}

        class FakeLlama:
            n_tokens = 5
            input_ids = torch.tensor([10, 11, 12, 13, 14])
            _last_eval_output_start = 0
            _last_eval_output_count = 0

            def generate(self, tokens, reset=True):
                captured["tokens"] = tokens
                captured["reset"] = reset
                yield 15

        llm = FakeLlama()
        qwen35._install_mtmd_physical_token_ledger(llm)
        self.assertEqual(list(llm.generate([10, -123, 14])), [15])
        self.assertEqual(captured["tokens"], [14])
        self.assertFalse(captured["reset"])

    def test_native_mtp_failure_retries_timing_once_without_mtp(self):
        success = {
            "ok": True,
            "timing_plan": {
                "confidence": "high", "analysis": "ok", "raw_json": "{}",
                "character_name_table": [],
                "shots": [{
                    "source_shot": 1, "shot_start_frame": 0, "shot_end_frame": 68,
                    "visual_beats": [{"start_frame": 0, "end_frame": 68, "action": "walk"}],
                    "overlays": [],
                }],
            },
        }
        calls = []

        def worker(payload):
            calls.append(payload.copy())
            if len(calls) == 1:
                return types.SimpleNamespace(returncode=1), {"ok": False, "error_type": "RuntimeError", "message": "MTP failed"}
            return types.SimpleNamespace(returncode=0), success

        with patch.object(qwen35, "_run_worker_once", side_effect=worker):
            result = qwen35._run_worker({"director_mtp": True}, True)
        self.assertEqual(result.shots[0].source_shot, 1)
        self.assertTrue(calls[0]["director_mtp"])
        self.assertFalse(calls[1]["director_mtp"])

    def test_validation_failure_does_not_repeat_timing_without_mtp(self):
        failure = {"ok": False, "error_type": "Qwen35ObservationError", "message": "bad JSON", "raw_json": "bad"}
        with patch.object(qwen35, "_run_worker_once", return_value=(types.SimpleNamespace(returncode=1), failure)) as worker:
            with self.assertRaisesRegex(qwen35.Qwen35ObservationError, "bad JSON"):
                qwen35._run_worker({"director_mtp": True}, True)
        worker.assert_called_once()

    def test_gguf_mtp_detection_reads_nextn_metadata(self):
        key = b"qwen3.nextn_predict_layers"
        data = b"".join((
            b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, 1),
            struct.pack("<Q", len(key)), key, struct.pack("<I", 4), struct.pack("<I", 2),
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(data)
            self.assertEqual(qwen35._gguf_mtp_layers(path), 2)


if __name__ == "__main__":
    unittest.main()
