from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
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
            "previous_gemma_description": (
                "[Shot 1] Heman dismounts from the tiger, lands on the temple floor, "
                "and begins walking toward the right."
            ),
            "previous_gemma_timing_plan": (
                "[Shot 1]: the previous slice covered Heman landing; defer his full walk right."
            ),
            "previous_gemma_end_state": "Heman has landed and is beginning to walk right.",
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
            observation_prompt = (capture_dir / "observation_prompt.txt").read_text()
            self.assertIn("Heman dismounts the tiger and walks right", observation_prompt)
            self.assertIn(request["previous_gemma_description"], observation_prompt)
            self.assertIn("attached stills", observation_prompt)
            system_prompt = (capture_dir / "system_prompt.txt").read_text()
            self.assertIn("# H3 Prompt Writing", system_prompt)
            self.assertIn("# Full-Reference Mode Rewrite Output Format Guide", system_prompt)
            self.assertNotIn("# Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)", system_prompt)
            saved_response = json.loads((capture_dir / "response.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_response["detailed_description"], result.detailed_description)
            replay_request = gemma4.load_gemma_capture(capture_dir)
            self.assertEqual(replay_request["image_urls"], saved_request["image_urls"])

    def test_gemma4_handler_sets_full_visual_budget(self):
        initialized = {}

        class FakeMTMD:
            @staticmethod
            def mtmd_context_params_default():
                return SimpleNamespace(batch_max_tokens=1024)

            @staticmethod
            def mtmd_init_from_file(path, model, params):
                initialized.update(path=path, model=model, params=params)
                return object()

            @staticmethod
            def mtmd_support_vision(_context):
                return True

            @staticmethod
            def mtmd_free(_context):
                initialized["freed"] = True

        fake_mtmd = FakeMTMD()

        class FakeBaseHandler:
            def __init__(self, clip_model_path, verbose=True, use_gpu=True):
                self.clip_model_path = clip_model_path
                self.verbose = verbose
                self.use_gpu = use_gpu
                self.mtmd_ctx = None
                self._mtmd_cpp = fake_mtmd

        class FakeStack:
            def callback(self, callback):
                initialized["cleanup"] = callback

        fake_llama_cpp = SimpleNamespace(
            LLAMA_FLASH_ATTN_TYPE_ENABLED=1,
            LLAMA_FLASH_ATTN_TYPE_DISABLED=0,
        )

        @contextmanager
        def fake_suppress_output(disable):
            initialized["suppress_disabled"] = disable
            yield

        handler_type = gemma4._gemma4_mtmd_handler_type(
            FakeBaseHandler,
            fake_llama_cpp,
            fake_suppress_output,
        )
        handler = handler_type("mmproj.gguf", verbose=False, use_gpu=True)
        llama_model = SimpleNamespace(
            verbose=False,
            n_threads=8,
            context_params=SimpleNamespace(flash_attn_type=1),
            model=object(),
            _stack=FakeStack(),
        )
        handler._init_mtmd_context(llama_model)

        params = initialized["params"]
        self.assertEqual(params.image_min_tokens, 70)
        self.assertEqual(params.image_max_tokens, 1120)
        self.assertEqual(params.batch_max_tokens, 1120)
        self.assertTrue(params.use_gpu)
        initialized["cleanup"]()
        self.assertTrue(initialized["freed"])
        self.assertIsNone(handler.mtmd_ctx)

    def test_worker_places_chronological_images_before_text_and_uses_matching_batch(self):
        captured = {}

        class FakeHandler:
            def __init__(self, **kwargs):
                captured["handler_kwargs"] = kwargs

        class FakeLlama:
            def __init__(self, **kwargs):
                captured["llama_kwargs"] = kwargs

            def create_chat_completion(self, **kwargs):
                captured["completion_kwargs"] = kwargs
                captured["messages"] = json.loads(json.dumps(kwargs["messages"]))
                response = {
                    "confidence": "high",
                    "analysis": "The prior action is visible.",
                    "timing_plan": "[Shot 1]: continue walking; [Shot 2]: defer dialogue until the cut.",
                    "end_state": "Heman continues walking toward the right.",
                    "detailed_description": (
                        "[Shot 1] Heman continues walking toward the right. "
                        "[Shot 2] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
                    ),
                }
                return {"choices": [{"message": {"content": json.dumps(response)}}]}

            def close(self):
                captured["closed"] = True

        image_urls = ["data:image/jpeg;base64,first", "data:image/jpeg;base64,second"]
        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(self.request(), image_urls, debug=False)

        content = captured["messages"][1]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "image_url", "text"])
        self.assertEqual([item["image_url"]["url"] for item in content[:2]], image_urls)
        self.assertIn("chunk 7 of 8", content[-1]["text"])
        self.assertEqual(captured["llama_kwargs"]["n_batch"], 1120)
        self.assertEqual(captured["llama_kwargs"]["n_ubatch"], 1120)
        self.assertTrue(captured["closed"])
        self.assertEqual(result.confidence, "high")

    def test_previous_gemma_description_is_explicitly_linked_to_prior_stills(self):
        request = self.request()
        _system, observation = gemma4._render_observation_messages(request)

        self.assertIn(request["previous_gemma_description"], observation)
        self.assertIn(request["previous_gemma_timing_plan"], observation)
        self.assertIn(request["previous_gemma_end_state"], observation)
        self.assertIn("exact attached stills", observation)
        self.assertIn("latest rendered still is authoritative", gemma4._gemma_prompt_templates()["SYSTEM"])

        first_chunk = self.request()
        first_chunk["previous_chunk"] = None
        first_chunk["previous_shots"] = []
        first_chunk["observation_frame_numbers"] = []
        first_chunk["previous_gemma_description"] = None
        first_chunk["previous_gemma_timing_plan"] = None
        first_chunk["previous_gemma_end_state"] = None
        _system, first_observation = gemma4._render_observation_messages(first_chunk)
        self.assertIn("No previous Gemma-directed detailed_description exists", first_observation)
        self.assertIn("No previous Gemma timing plan exists", first_observation)
        self.assertIn("No previous Gemma end state exists", first_observation)

    def test_observation_front_loads_physical_shot_starts_and_slice_request(self):
        _system, observation = gemma4._render_observation_messages(self.request())

        self.assertIn("Local source-shot timeline and start frames:", observation)
        self.assertIn(
            "local [Shot 1] / source Shot 4: starts before this physical chunk at global frame 193 "
            "(physical local frame -11); this chunk must author its global frames 209-219 "
            "(physical local frames 5-15).",
            observation,
        )
        self.assertIn(
            "local [Shot 2] / source Shot 5: starts at global frame 220 "
            "(physical local frame 16); this chunk must author its global frames 220-242 "
            "(physical local frames 16-38).",
            observation,
        )
        self.assertIn(
            "Write one complete chunk-local detailed_description for the ending portion of local [Shot 1] "
            "(source Shot 4) and the opening portion of local [Shot 2] (source Shot 5).",
            observation,
        )
        self.assertIn(
            "physical global timeslice 204-242 inclusive, physical chunk-local timeslice 0-38 (39 frames).",
            observation,
        )
        self.assertIn(
            "source-relative frames 16-26 (59.3%-100.0% of the shot).",
            observation,
        )
        self.assertIn("IMMUTABLE H3-LOCAL SHOT MARKERS", observation)
        self.assertIn('exact token: "[Shot 2] At 00:00.667,"', observation)

    def test_retries_once_with_literal_local_markers_when_initial_json_uses_global_timecodes(self):
        captured = {"messages": []}
        wrong = {
            "confidence": "high",
            "analysis": "Continue walking, then cut to the warning.",
            "timing_plan": "[Shot 1]: walking; [Shot 2]: the warning after the cut.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "[Shot 1] Heman continues walking right. "
                "[Shot 2] At 00:09.167, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        corrected = dict(wrong)
        corrected["detailed_description"] = wrong["detailed_description"].replace("00:09.167", "00:00.667")

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **_kwargs):
                self.responses = [wrong, corrected]

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(self.request(), [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(result.validation_warnings, ())
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[0].raw_json, json.dumps(wrong))
        self.assertTrue(any("required marker" in warning for warning in result.attempts[0].validation_warnings))
        self.assertEqual(result.attempts[1].raw_json, json.dumps(corrected))
        self.assertIn("MARKER CORRECTION REQUIRED", result.attempts[1].correction_prompt)
        self.assertIn('[Shot 2] At 00:00.667,', result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        retry_messages = captured["messages"][1]
        self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "user"])
        self.assertEqual(retry_messages[2]["content"], json.dumps(wrong))
        self.assertIn("MARKER CORRECTION REQUIRED", retry_messages[3]["content"])
        self.assertTrue(captured["closed"])

    def test_reports_marker_and_dialogue_warnings_without_replacing_gemma_text(self):
        request = self.request()
        value = {
            "confidence": "high",
            "analysis": "The dismount is complete; finish walking, then cut.",
            "timing_plan": "[Shot 1]: walk right now; [Shot 2]: defer dialogue until the cut.",
            "end_state": "Heman is walking right.",
            "detailed_description": (
                "[Shot 1] Heman continues walking right. "
                "[Shot 2] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        result = gemma4._validate_chunk_prompt(
            value,
            request,
            json.dumps(value),
            system_prompt="exact system prompt",
            observation_prompt="exact chunk request",
        )
        self.assertEqual(result.detailed_description, value["detailed_description"])
        self.assertEqual(result.timing_plan, value["timing_plan"])
        self.assertEqual(result.end_state, value["end_state"])
        self.assertEqual(result.validation_warnings, ())
        self.assertEqual(result.system_prompt, "exact system prompt")
        self.assertEqual(result.observation_prompt, "exact chunk request")
        self.assertEqual(gemma4._chunk_prompt_from_payload(gemma4._chunk_prompt_payload(result)), result)

        wrong_marker = dict(value)
        wrong_marker["detailed_description"] = value["detailed_description"].replace("00:00.667", "00:00.500")
        marker_result = gemma4._validate_chunk_prompt(wrong_marker, request, json.dumps(wrong_marker))
        self.assertEqual(marker_result.detailed_description, wrong_marker["detailed_description"])
        self.assertTrue(any("required marker" in warning for warning in marker_result.validation_warnings))
        self.assertEqual(
            gemma4._chunk_prompt_from_payload(gemma4._chunk_prompt_payload(marker_result)),
            marker_result,
        )

        changed_dialogue = dict(value)
        changed_dialogue["detailed_description"] = value["detailed_description"].replace("Stay back!", "Run away!")
        dialogue_result = gemma4._validate_chunk_prompt(changed_dialogue, request, json.dumps(changed_dialogue))
        self.assertEqual(dialogue_result.detailed_description, changed_dialogue["detailed_description"])
        self.assertIn(
            "Gemma 4 modified or invented dialogue instead of preserving source words",
            dialogue_result.validation_warnings,
        )

        missing_marker = dict(value)
        missing_marker["detailed_description"] = missing_marker["detailed_description"].replace("[Shot 2]", "Shot 2")
        missing_result = gemma4._validate_chunk_prompt(missing_marker, request, json.dumps(missing_marker))
        self.assertEqual(missing_result.detailed_description, missing_marker["detailed_description"])
        self.assertIn(
            "Gemma 4 returned 1 shot markers; this chunk requires 2",
            missing_result.validation_warnings,
        )

        legacy_end_state = dict(value)
        legacy_end_state.pop("end_state")
        legacy_end_state["detailed_description"] += " [end state] Heman is walking right."
        legacy_result = gemma4._validate_chunk_prompt(
            legacy_end_state,
            request,
            json.dumps(legacy_end_state),
        )
        self.assertEqual(legacy_result.end_state, "Heman is walking right.")
        self.assertNotIn("[end state]", legacy_result.detailed_description.lower())
        self.assertIn(
            "Gemma 4 put [end state] inside detailed_description; extracted it into Gemma-only end_state",
            legacy_result.validation_warnings,
        )

    def test_last_run_observation_images_are_saved_when_debug_is_disabled(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="The prior action is visible.",
            detailed_description=(
                "[Shot 1] Heman continues walking toward the right. "
                "[Shot 2] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
            raw_json='{"confidence":"high"}',
        )
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        frames[1, :, :, 1] = 1.0
        with tempfile.TemporaryDirectory() as temp_dir:
            image_directory = Path(temp_dir) / "last_gemma_images"
            director = gemma4.Gemma4ContinuityDirector(
                debug=False,
                observation_image_directory=image_directory,
            )

            def fake_worker(request):
                request.clear()
                return result

            with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker):
                actual = director.direct(self.request(), frames)

            self.assertEqual(actual, result)
            self.assertEqual(
                sorted(path.name for path in image_directory.glob("*.jpg")),
                [
                    "chunk_007_source_frame_000170.jpg",
                    "chunk_007_source_frame_000208.jpg",
                ],
            )
            for image_file in image_directory.glob("*.jpg"):
                self.assertTrue(image_file.read_bytes().startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
