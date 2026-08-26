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
                    "required_marker": None,
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

    @staticmethod
    def timing_request():
        return {
            "chunk_count": 4,
            "fps": 24.0,
            "prompt_mode": "ref",
            "source_shots": [
                {
                    "shot_number": 1,
                    "shot_start": 0,
                    "shot_end": 68,
                    "source_body": "The tiger runs toward the temple, then enters it.",
                },
                {
                    "shot_number": 2,
                    "shot_start": 68,
                    "shot_end": 148,
                    "source_body": "Heman pulls the harness; the tiger skids to a stop and they inspect the room.",
                },
            ],
            "chunks": [
                {"sampled_start": 0, "sampled_end": 39, "output_start": 0, "output_end": 39},
                {"sampled_start": 34, "sampled_end": 73, "output_start": 39, "output_end": 73},
                {"sampled_start": 68, "sampled_end": 107, "output_start": 73, "output_end": 107},
                {"sampled_start": 102, "sampled_end": 148, "output_start": 107, "output_end": 148},
            ],
            "original_prompt": (
                "detailed_description: [Shot 1] The tiger runs toward the temple, then enters it. "
                "[Shot 2] At 00:02.833, Heman pulls the harness; the tiger skids to a stop and they inspect the room."
            ),
        }

    @staticmethod
    def timing_response():
        return {
            "confidence": "high",
            "analysis": "The run, entrance, braking, skid, stop, and inspection are distributed across both shots.",
            "character_name_table": [
                {"character_name": "Heman", "subject": "<Subject 1>"},
                {"character_name": "Tila", "subject": "<Subject 2>"},
            ],
            "shots": [
                {
                    "source_shot": 1,
                    "visual_beats": [
                        {"start_frame": 0, "end_frame": 39, "action": "Tiger runs through the jungle toward the temple."},
                        {"start_frame": 39, "end_frame": 68, "action": "Tiger reaches and enters the temple."},
                    ],
                    "overlays": [],
                },
                {
                    "source_shot": 2,
                    "visual_beats": [
                        {"start_frame": 0, "end_frame": 24, "action": "The heroes continue inside at speed."},
                        {"start_frame": 24, "end_frame": 48, "action": "Heman pulls the harness and the tiger begins a hard skid."},
                        {"start_frame": 48, "end_frame": 68, "action": "The tiger settles to a stop."},
                        {"start_frame": 68, "end_frame": 80, "action": "The riders inspect the temple room."},
                    ],
                    "overlays": [],
                },
            ],
        }

    def test_debug_capture_preserves_exact_request_images_prompts_and_response(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="Heman is already on the ground and moving right.",
            detailed_description=(
                "Heman continues walking toward the right. "
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
                        "Heman continues walking toward the right. "
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

    def test_preproduction_timing_plan_covers_each_source_shot_and_feeds_relevant_schedule(self):
        request = self.timing_request()
        _system, planning_prompt = gemma4._render_timing_plan_messages(request)
        self.assertIn("Source Shot 2: global frames 68-147", planning_prompt)
        self.assertIn("Chunk 3: sampled global frames 68-106; retains global frames 73-106", planning_prompt)

        plan = gemma4._validate_timing_plan(
            self.timing_response(), request, json.dumps(self.timing_response())
        )
        self.assertEqual([shot.source_shot for shot in plan.shots], [1, 2])
        self.assertEqual([(beat.start_frame, beat.end_frame) for beat in plan.shots[1].visual_beats], [(0, 24), (24, 48), (48, 68), (68, 80)])
        self.assertEqual(
            plan.character_name_table_text(),
            "- Heman -> <Subject 1>\n- Tila -> <Subject 2>",
        )
        self.assertEqual(gemma4._timing_plan_from_payload(gemma4._timing_plan_payload(plan)), plan)
        preproduction_log = plan.for_target_shots(request["source_shots"], 24.0)
        self.assertIn("Source Shot 1: immutable preproduction timing schedule", preproduction_log)
        self.assertIn("Source Shot 2: immutable preproduction timing schedule", preproduction_log)
        self.assertNotIn("MANDATORY CURRENT-SLICE BEAT COVERAGE", preproduction_log)

        relevant = plan.for_target_shots([
            {
                "shot_number": 2,
                "shot_start": 68,
                "shot_end": 148,
                "target_start": 73,
                "target_end": 107,
                "required_marker": "[Shot 1]",
                "source_body": request["source_shots"][1]["source_body"],
            },
        ], 24.0)
        self.assertIn("Source Shot 2: immutable preproduction timing schedule", relevant)
        self.assertNotIn("Source Shot 1:", relevant)
        self.assertIn("MANDATORY CURRENT-SLICE BEAT COVERAGE", relevant)
        self.assertIn(
            "Required now [S2.V1], visual, source-relative frames 5-23: this planned beat continues here",
            relevant,
        )
        self.assertIn(
            "Required now [S2.V2], visual, source-relative frames 24-38: this planned beat begins here",
            relevant,
        )
        self.assertIn("source-relative frames 24-47", relevant)

        chunk_request = self.request()
        chunk_request["preproduction_timing_plan"] = relevant
        chunk_request["character_name_table"] = plan.character_name_table_text()
        _system, observation = gemma4._render_observation_messages(chunk_request)
        self.assertIn("complete immutable preproduction timing schedule", observation)
        self.assertIn("MANDATORY CURRENT-SLICE BEAT COVERAGE", observation)
        self.assertIn("Source Shot 2: immutable preproduction timing schedule", observation)
        self.assertIn("Heman -> <Subject 1>", observation)

    def test_preproduction_allows_dialogue_overlay_and_exposes_it_as_current_coverage(self):
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [{
            "start_frame": 24,
            "end_frame": 48,
            "type": "dialogue",
            "content": "Heman says: <d>[English] Stay back!</d>",
        }]
        plan = gemma4._validate_timing_plan(response, self.timing_request(), json.dumps(response))
        target = [{
            "shot_number": 2,
            "shot_start": 68,
            "shot_end": 148,
            "target_start": 73,
            "target_end": 107,
        }]
        coverage = plan.mandatory_coverage(target)
        dialogue = next(item for item in coverage if item["id"] == "S2.O1")
        self.assertEqual(dialogue["kind"], "overlay")
        self.assertEqual(dialogue["overlay_type"], "dialogue")
        self.assertEqual((dialogue["overlap_start_frame"], dialogue["overlap_end_frame"]), (24, 39))
        rendered = plan.for_target_shots(target, 24.0)
        self.assertIn("[S2.O1] dialogue at source-relative frames 24-47", rendered)
        self.assertIn("Required now [S2.O1], overlay/dialogue", rendered)

    def test_preproduction_retries_one_complete_json_when_visual_beats_do_not_cover_a_shot(self):
        captured = {"messages": []}
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        invalid["shots"][1]["visual_beats"][-1]["end_frame"] = 78
        corrected = self.timing_response()

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **_kwargs):
                self.responses = [invalid, corrected]

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                captured.setdefault("max_tokens", []).append(kwargs["max_tokens"])
                return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._plan_timing_in_process(self.timing_request(), debug=False)

        self.assertEqual(len(result.attempts), 2)
        self.assertIn("TIMING-PLAN CORRECTION REQUIRED", result.attempts[1].correction_prompt)
        self.assertEqual(captured["max_tokens"], [2048, 2048])
        self.assertEqual([message["role"] for message in captured["messages"][0]], ["system", "user"])
        self.assertIsInstance(captured["messages"][0][1]["content"], str)
        self.assertTrue(captured["closed"])

    def test_preproduction_rejects_gapped_schedule_without_sampler_fallback(self):
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        invalid["shots"][0]["visual_beats"][1]["start_frame"] = 40
        with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "contiguous"):
            gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))

    def test_preproduction_uses_sampler_owned_global_boundaries_not_ambiguous_model_echoes(self):
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        # A model may echo a human-readable inclusive endpoint. It must not
        # invalidate otherwise complete source-relative action beats.
        response["shots"][1]["shot_start_frame"] = 68
        response["shots"][1]["shot_end_frame"] = 148
        plan = gemma4._validate_timing_plan(response, self.timing_request(), json.dumps(response))
        self.assertEqual((plan.shots[1].shot_start_frame, plan.shots[1].shot_end_frame), (68, 148))

    def test_preproduction_normalizes_only_the_final_inclusive_frame_spelling(self):
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][0]["visual_beats"][-1]["end_frame"] = 67
        plan = gemma4._validate_timing_plan(response, self.timing_request(), json.dumps(response))
        self.assertEqual(plan.shots[0].visual_beats[-1].end_frame, 68)

    def test_preproduction_rejects_invalid_or_duplicate_character_subject_table(self):
        invalid = self.timing_response()
        invalid["character_name_table"] = [
            {"character_name": "Heman", "subject": "<Subject 1>"},
            {"character_name": "Heman", "subject": "<Subject 2>"},
        ]
        with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "repeats character name"):
            gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))

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
        self.assertIn("begin it as unmarked continuation prose", observation)
        self.assertNotIn('exact token: "[Shot 1]"', observation)
        self.assertIn('exact token: "[Shot 2] At 00:00.667,"', observation)

    def test_retries_once_with_literal_local_markers_when_initial_json_uses_global_timecodes(self):
        captured = {"messages": []}
        wrong = {
            "confidence": "high",
            "analysis": "Continue walking, then cut to the warning.",
            "timing_plan": "[Shot 1]: walking; [Shot 2]: the warning after the cut.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "Heman continues walking right. "
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
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", result.attempts[1].correction_prompt)
        self.assertIn('[Shot 2] At 00:00.667,', result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        retry_messages = captured["messages"][1]
        self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "user"])
        self.assertEqual(retry_messages[2]["content"], json.dumps(wrong))
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", retry_messages[3]["content"])
        self.assertTrue(captured["closed"])

    def test_retries_with_a_gemma_authored_coverage_correction_when_current_dialogue_is_deferred(self):
        request = self.request()
        request["mandatory_coverage"] = [{
            "id": "S5.O1",
            "kind": "overlay",
            "overlay_type": "dialogue",
            "source_shot": 5,
            "source_start_frame": 0,
            "source_end_frame": 23,
            "overlap_start_frame": 0,
            "overlap_end_frame": 23,
            "action": "Heman says: <d>[English] Stay back!</d>",
        }]
        deferred = {
            "confidence": "high",
            "analysis": "The dialogue belongs after the ongoing action.",
            "timing_plan": "S5.O1 deferred until the next chunk.",
            "end_state": "Heman has not spoken yet.",
            "coverage": [{
                "id": "S5.O1",
                "status": "deferred",
                "evidence": "Heman continues walking right.",
            }],
            "detailed_description": "Heman continues walking right.",
        }
        corrected = {
            "confidence": "high",
            "analysis": "The dialogue begins during the current action.",
            "timing_plan": "S5.O1 begins now while Heman continues walking.",
            "end_state": "Heman is still walking after delivering the warning.",
            "coverage": [{
                "id": "S5.O1",
                "status": "begins",
                "evidence": "Heman says: <d>[English] Stay back!</d>",
            }],
            "detailed_description": "Heman says: <d>[English] Stay back!</d> while continuing to walk right.",
        }
        captured = {"messages": []}

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **_kwargs):
                self.responses = [deferred, corrected]

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(request, [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(len(result.attempts), 2)
        self.assertIn("status 'deferred'", "\n".join(result.attempts[0].validation_warnings))
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", result.attempts[1].correction_prompt)
        self.assertIn("S5.O1", result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        self.assertTrue(captured["closed"])

    def test_retries_with_official_subject_speaker_form_for_mapped_dialogue(self):
        request = self.request()
        request["character_name_table"] = "- Heman -> <Subject 1>"
        request["original_prompt"] = (
            "detailed_description: [Shot 4] Heman walks right. "
            "[Shot 5] At 00:09.167, Heman (S1) says: <d>[English] Stay back!</d>"
        )
        malformed = {
            "confidence": "high",
            "analysis": "The warning is delivered after the cut.",
            "timing_plan": "Heman gives the warning in the current slice.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 2] At 00:00.667, Heman (<Subject 1>) (S1) says, "
                "<d>[English] Stay back!</d>"
            ),
        }
        corrected = dict(malformed)
        corrected["detailed_description"] = malformed["detailed_description"].replace(
            "Heman (<Subject 1>) (S1)", "<Subject 1> (S1)"
        )
        captured = {"messages": []}

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **_kwargs):
                self.responses = [malformed, corrected]

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(request, [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(any(
            "dialogue speaker form" in warning
            for warning in result.attempts[0].validation_warnings
        ))
        self.assertIn("<Subject 1> (S1) must introduce", result.attempts[1].correction_prompt)
        self.assertIn("not Name (<Subject N>) (Sx)", result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        self.assertTrue(captured["closed"])

    def test_system_requires_continuity_prefix_for_non_cut_camera_motion(self):
        system = gemma4._gemma_prompt_templates()["SYSTEM"]
        planning_system = gemma4._gemma_prompt_templates()["PREPRODUCTION_SYSTEM"]

        self.assertIn("Every camera movement, follow, pan, zoom, track, shake, or reposition", system)
        self.assertIn("`In a continuous movement,`", system)
        self.assertIn("never turn it into an undocumented cut", system)
        self.assertIn("`In a continuous movement,`", planning_system)

    def test_reports_marker_and_dialogue_warnings_without_replacing_gemma_text(self):
        request = self.request()
        value = {
            "confidence": "high",
            "analysis": "The dismount is complete; finish walking, then cut.",
            "timing_plan": "[Shot 1]: walk right now; [Shot 2]: defer dialogue until the cut.",
            "end_state": "Heman is walking right.",
            "detailed_description": (
                "Heman continues walking right. "
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

        spurious_opening = dict(value)
        spurious_opening["detailed_description"] = (
            "[Shot 1] " + value["detailed_description"]
        )
        spurious_result = gemma4._validate_chunk_prompt(
            spurious_opening,
            request,
            json.dumps(spurious_opening),
        )
        self.assertEqual(spurious_result.detailed_description, spurious_opening["detailed_description"])
        self.assertIn(
            "Gemma 4 returned 2 shot markers; this chunk requires 1",
            spurious_result.validation_warnings,
        )

        mid_shot_only_request = dict(request)
        mid_shot_only_request["target_shots"] = [dict(request["target_shots"][0])]
        plain_mid_shot = dict(value)
        plain_mid_shot["detailed_description"] = "Heman continues walking right."
        plain_result = gemma4._validate_chunk_prompt(
            plain_mid_shot,
            mid_shot_only_request,
            json.dumps(plain_mid_shot),
        )
        self.assertEqual(plain_result.validation_warnings, ())

        marked_mid_shot = dict(plain_mid_shot)
        marked_mid_shot["detailed_description"] = "[Shot 1] Heman continues walking right."
        marked_result = gemma4._validate_chunk_prompt(
            marked_mid_shot,
            mid_shot_only_request,
            json.dumps(marked_mid_shot),
        )
        self.assertIn(
            "Gemma 4 returned 1 shot markers; this chunk requires 0",
            marked_result.validation_warnings,
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
            "Gemma 4 returned 0 shot markers; this chunk requires 1",
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
                "Heman continues walking toward the right. "
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
