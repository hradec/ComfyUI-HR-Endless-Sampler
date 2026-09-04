from __future__ import annotations

import json
import io
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import gemma4  # noqa: E402
import gemma4_mtp  # noqa: E402


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
                "[Shot 4] Heman dismounts from the tiger, lands on the temple floor, "
                "and begins walking toward the right."
            ),
            "previous_gemma_timing_plan": (
                "[Shot 4]: the previous slice covered Heman landing; defer his full walk right."
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
                    "required_marker": "[Shot 5] At 00:00.667,",
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
                "subject_definitions:\nHeman is <Subject 1>.\nTila is <Subject 2>.\n"
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
                    "light_change": False,
                    "visual_beats": [
                        {"start_frame": 0, "end_frame": 39, "action": "Tiger runs through the jungle toward the temple."},
                        {"start_frame": 39, "end_frame": 68, "action": "Tiger reaches and enters the temple."},
                    ],
                    "overlays": [],
                    "continuity_slices": [
                        {"start_frame": 0, "end_frame": 39, "characters": []},
                        {"start_frame": 39, "end_frame": 68, "characters": []},
                    ],
                },
                {
                    "source_shot": 2,
                    "light_change": False,
                    "visual_beats": [
                        {"start_frame": 0, "end_frame": 24, "action": "The heroes continue inside at speed."},
                        {"start_frame": 24, "end_frame": 48, "action": "Heman pulls the harness and the tiger begins a hard skid."},
                        {"start_frame": 48, "end_frame": 68, "action": "The tiger settles to a stop."},
                        {"start_frame": 68, "end_frame": 80, "action": "The riders inspect the temple room."},
                    ],
                    "overlays": [],
                    "continuity_slices": [
                        {
                            "start_frame": 0,
                            "end_frame": 5,
                            "characters": [{
                                "character_name": "Heman",
                                "subject": "<Subject 1>",
                                "entry_state": "mounted on the moving tiger inside the temple",
                                "expected_exit_state": "still mounted as the tiger continues forward",
                            }],
                        },
                        {
                            "start_frame": 5,
                            "end_frame": 39,
                            "characters": [{
                                "character_name": "Heman",
                                "subject": "<Subject 1>",
                                "entry_state": "mounted on the moving tiger inside the temple",
                                "expected_exit_state": "mounted and pulling the harness as the tiger skids",
                            }],
                        },
                        {
                            "start_frame": 39,
                            "end_frame": 80,
                            "characters": [{
                                "character_name": "Heman",
                                "subject": "<Subject 1>",
                                "entry_state": "mounted while the tiger is skidding inside the temple",
                                "expected_exit_state": "mounted on the stopped tiger and inspecting the room",
                            }],
                        },
                    ],
                },
            ],
        }

    @classmethod
    def production_bible_response(cls):
        timing = cls.timing_response()
        return {
            "confidence": "high",
            "analysis": "The heroes move from the jungle into the temple and stop together.",
            "character_name_table": timing["character_name_table"],
            "speaker_voice_profiles": [],
            "shots": [
                {
                    "source_shot": 1,
                    "shot_intent": "The mounted heroes reach and enter the temple.",
                    "environment": "Jungle approach changing into the temple entrance.",
                    "camera_and_cut": "Opening tracking shot with continuous progression toward the temple.",
                    "characters": [],
                },
                {
                    "source_shot": 2,
                    "shot_intent": "The tiger brakes and the heroes inspect the temple.",
                    "environment": "Interior temple room.",
                    "camera_and_cut": "The supplied Shot 2 cut establishes the interior action.",
                    "characters": [{
                        "character_name": "Heman",
                        "subject": "<Subject 1>",
                        "opening_state": "mounted on the moving tiger inside the temple",
                        "closing_state": "mounted on the stopped tiger and inspecting the room",
                    }],
                },
            ],
        }

    @classmethod
    def single_shot_response(cls, index):
        timing = cls.timing_response()
        return {
            "confidence": timing["confidence"],
            "analysis": timing["analysis"],
            **json.loads(json.dumps(timing["shots"][index])),
        }

    def test_worker_protocol_forces_utf8_and_tolerates_stray_invalid_bytes(self):
        fake_process = object()
        with patch.object(gemma4.subprocess, "Popen", return_value=fake_process) as popen:
            process = gemma4._start_worker_process()

        self.assertIs(process, fake_process)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [
            sys.executable,
            "-u",
            str(Path(gemma4.__file__).resolve()),
            "--worker",
        ])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(kwargs["env"]["PYTHONUTF8"], "1")
        self.assertIn(str(COMFY_ROOT), kwargs["env"]["PYTHONPATH"])

    def test_automatic_debug_capture_cleanup_keeps_only_non_owned_temp_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_capture = os.path.join(temporary, gemma4.GEMMA4_DEBUG_CAPTURE_PREFIX + "old")
            old_worker = os.path.join(temporary, gemma4.GEMMA4_OWNED_TEMP_PREFIX + "worker-old")
            foreign_directory = os.path.join(temporary, "unrelated-debug-data")
            os.mkdir(old_capture)
            os.mkdir(old_worker)
            os.mkdir(foreign_directory)
            with patch.object(gemma4.tempfile, "gettempdir", return_value=temporary):
                gemma4.Gemma4ContinuityDirector(debug=False)
            self.assertFalse(os.path.exists(old_capture))
            self.assertFalse(os.path.exists(old_worker))
            self.assertTrue(os.path.isdir(foreign_directory))

    def test_debug_capture_preserves_exact_request_images_prompts_and_response(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="Heman is already on the ground and moving right.",
            detailed_description=(
                "Heman continues walking toward the right. "
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
            raw_json='{"confidence":"high"}',
        )
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        frames[1, :, :, 0] = 1.0
        request = self.request()
        with tempfile.TemporaryDirectory() as temp_dir:
            director = gemma4.Gemma4ContinuityDirector(debug=True, seed=123456, capture_directory=temp_dir)
            captured_request = {}

            def fake_worker(request):
                captured_request.update(json.loads(json.dumps(request)))
                request.clear()
                return result

            with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker):
                actual = director.direct(request, frames)

            self.assertEqual(actual, result)
            self.assertEqual(captured_request["gemma4_seed"], 123456)
            capture_dir = next(Path(temp_dir).glob("prompt_*"))
            saved_request = json.loads((capture_dir / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_request, captured_request)
            self.assertEqual(len(list(capture_dir.glob("frame_*.jpg"))), 2)
            observation_prompt = (capture_dir / "observation_prompt.txt").read_text()
            self.assertIn("Heman dismounts the tiger and walks right", observation_prompt)
            self.assertIn(request["previous_gemma_description"], observation_prompt)
            self.assertIn("attached stills", observation_prompt)
            system_prompt = (capture_dir / "system_prompt.txt").read_text()
            self.assertIn("# MiniMax H3 prompt-writing working summary", system_prompt)
            self.assertNotIn("# H3 Prompt Writing", system_prompt)
            self.assertNotIn("# Full-Reference Mode Rewrite Output Format Guide", system_prompt)
            saved_response = json.loads((capture_dir / "response.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_response["detailed_description"], result.detailed_description)
            replay_request = gemma4.load_gemma_capture(capture_dir)
            self.assertEqual(replay_request["image_urls"], saved_request["image_urls"])

    def test_native_mtp_worker_abort_retries_only_failed_operation_without_mtp(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="The action continues.",
            detailed_description="Heman continues walking right.",
            raw_json='{"confidence":"high"}',
        )
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        calls = []

        def fake_worker(request):
            calls.append({
                "mtp": request.get("gemma4_mtp"),
                "has_cache": "preproduction_cache" in request,
                "has_cache_slice": "preproduction_current_slice" in request,
            })
            request.clear()
            if len(calls) == 1:
                raise gemma4.Gemma4WorkerExitError(
                    "worker aborted",
                    returncode=-6,
                )
            return result

        first_request = self.request()
        first_request["preproduction_cache"] = {"path": "/dev/shm/test-cache"}
        first_request["preproduction_current_slice"] = "cached slice"
        second_request = self.request()
        second_request["preproduction_cache"] = {"path": "/dev/shm/test-cache"}
        second_request["preproduction_current_slice"] = "cached slice"
        with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker), \
                self.assertLogs(level="WARNING") as warning_log:
            self.assertIs(director.direct(first_request, frames), result)
            self.assertIs(director.direct(second_request, frames), result)

        self.assertEqual(calls, [
            {"mtp": True, "has_cache": True, "has_cache_slice": True},
            {"mtp": False, "has_cache": True, "has_cache_slice": True},
            {"mtp": True, "has_cache": True, "has_cache_slice": True},
        ])
        self.assertTrue(director.gemma4_mtp)
        warning_text = "\n".join(warning_log.output)
        self.assertIn("status -6", warning_text)
        self.assertIn("retry 1/10", warning_text)
        self.assertIn("next independent Gemma operation will try MTP", warning_text)

    def test_mtp_empty_json_retries_only_failed_operation_without_mtp(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="The action continues.",
            detailed_description="Heman continues walking right.",
            raw_json='{"confidence":"high"}',
        )
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        calls = []

        def fake_worker(request):
            calls.append(request.get("gemma4_mtp"))
            if request.get("gemma4_mtp"):
                raise gemma4.Gemma4WorkerExitError(
                    "MTP returned no JSON",
                    returncode=1,
                    worker_error_type="Gemma4MTPOutputError",
                )
            return result

        with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker), \
                self.assertLogs(level="WARNING") as warning_log:
            self.assertIs(director.direct(self.request(), frames), result)

        self.assertEqual(calls, [True, False])
        self.assertTrue(director.gemma4_mtp)
        self.assertIn("MTP returned no JSON", "\n".join(warning_log.output))

    def test_mtp_empty_json_message_survives_parent_worker_version_skew(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="The action continues.",
            detailed_description="Heman continues walking right.",
            raw_json='{"confidence":"high"}',
        )
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        calls = []

        def fake_worker(request):
            calls.append(request.get("gemma4_mtp"))
            if request.get("gemma4_mtp"):
                # Simulate a stale parent that decoded the worker's new typed
                # error as its older generic observation-error class.
                raise gemma4.Gemma4ObservationError(
                    "Gemma 4 MTP returned no complete JSON object; retry this operation with the original decoder"
                )
            return result

        with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker), \
                self.assertLogs(level="WARNING") as warning_log:
            self.assertIs(director.direct(self.request(), frames), result)

        self.assertEqual(calls, [True, False])
        self.assertTrue(director.gemma4_mtp)
        self.assertIn("returned no complete JSON", "\n".join(warning_log.output))

    def test_worker_abort_retries_ten_times_with_fresh_preserved_requests(self):
        result = gemma4.GemmaChunkPrompt(
            confidence="high",
            analysis="The action continues.",
            detailed_description="Heman continues walking right.",
            raw_json='{"confidence":"high"}',
        )
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        calls = []

        def fake_worker(request):
            calls.append({
                "mtp": request.get("gemma4_mtp"),
                "cache": request.get("preproduction_cache"),
                "slice": request.get("preproduction_current_slice"),
            })
            request.clear()
            if len(calls) <= gemma4.GEMMA4_WORKER_RETRY_LIMIT:
                raise gemma4.Gemma4WorkerExitError("worker aborted", returncode=-6)
            return result

        request = self.request()
        request["preproduction_cache"] = {"path": "/dev/shm/test-cache"}
        request["preproduction_current_slice"] = "cached slice"
        with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker), \
                self.assertLogs(level="WARNING") as warning_log:
            self.assertIs(director.direct(request, frames), result)

        self.assertEqual(len(calls), gemma4.GEMMA4_WORKER_RETRY_LIMIT + 1)
        self.assertTrue(calls[0]["mtp"])
        self.assertTrue(all(not call["mtp"] for call in calls[1:]))
        self.assertTrue(all(call["cache"] == {"path": "/dev/shm/test-cache"} for call in calls))
        self.assertTrue(all(call["slice"] == "cached slice" for call in calls))
        self.assertIn("retry 10/10", "\n".join(warning_log.output))

    def test_worker_abort_after_ten_retries_raises_last_failure(self):
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=False)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        with patch.object(
            gemma4,
            "_observe_in_worker",
            side_effect=gemma4.Gemma4WorkerExitError("worker aborted", returncode=-6),
        ) as worker, self.assertLogs(level="WARNING"):
            with self.assertRaisesRegex(gemma4.Gemma4WorkerExitError, "worker aborted"):
                director.direct(self.request(), frames)

        self.assertEqual(worker.call_count, gemma4.GEMMA4_WORKER_RETRY_LIMIT + 1)

    def test_context_overflow_preserves_preproduction_cache(self):
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        calls = []

        def fake_worker(request):
            calls.append(dict(request))
            raise gemma4.Gemma4ObservationError(
                "Appended prompt exceeds n_ctx: 32813 > 32768"
            )

        request = self.request()
        request["preproduction_cache"] = {"path": "/dev/shm/test-cache"}
        request["preproduction_current_slice"] = "cached slice"
        with patch.object(gemma4, "_observe_in_worker", side_effect=fake_worker):
            with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "32813 > 32768"):
                director.direct(request, frames)

        self.assertEqual(len(calls), 1)
        self.assertIn("preproduction_cache", calls[0])
        self.assertIn("preproduction_current_slice", calls[0])

    def test_non_process_gemma_error_does_not_disable_mtp_or_retry(self):
        director = gemma4.Gemma4ContinuityDirector(gemma4_mtp=True)
        frames = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        with patch.object(
            gemma4,
            "_observe_in_worker",
            side_effect=gemma4.Gemma4ObservationError("invalid model response"),
        ) as worker:
            with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "invalid model response"):
                director.direct(self.request(), frames)

        self.assertEqual(worker.call_count, 1)
        self.assertTrue(director.gemma4_mtp)

    def test_runtime_direct_speaker_template_uses_documented_colon(self):
        templates = gemma4._gemma_prompt_templates()
        summary = gemma4._minimax_prompt_reference("ref")

        expected = "<Subject N> (Sx) says: <d>[Language] exact words</d>"
        legacy = "<Subject N> (Sx) says, <d>[Language] exact words</d>"
        self.assertIn(expected, templates["SYSTEM"])
        self.assertIn(expected, summary)
        self.assertNotIn(legacy, templates["SYSTEM"])

    def test_runtime_creates_native_mtp_target_when_enabled(self):
        class FakeRealLlama:
            __module__ = "llama_cpp.llama"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        target = Path("gemma.gguf")
        draft = Path("gemma-mtp.gguf")
        handler = object()
        with patch.object(gemma4, "_ensure_mtp_model_file", return_value=draft):
            llm = gemma4._create_runtime_llm(
                FakeRealLlama,
                model_path=target,
                handler=handler,
                debug=True,
                gemma4_mtp=True,
                seed=123456,
            )

        self.assertIsInstance(llm, FakeRealLlama)
        kwargs = llm.kwargs
        self.assertIs(kwargs["chat_handler"], handler)
        self.assertEqual(kwargs["n_ctx"], 32768)
        self.assertEqual(kwargs["type_k"], gemma4.GEMMA4_KV_CACHE_Q8_0)
        self.assertEqual(kwargs["type_v"], gemma4.GEMMA4_KV_CACHE_Q8_0)
        self.assertFalse(kwargs["swa_full"])
        self.assertFalse(kwargs["verbose"])
        self.assertEqual(kwargs["seed"], 123456)
        speculative = kwargs["speculative"]
        self.assertEqual(speculative.spec_type.name, "DRAFT_MTP")
        self.assertEqual(speculative.draft_model_path, str(draft))
        self.assertEqual(speculative.draft_n_max, 4)
        self.assertEqual(speculative.draft_p_min, 0.0)
        self.assertEqual(speculative.draft_n_gpu_layers, "all")
        self.assertTrue(speculative.draft_backend_sampling)
        self.assertEqual(speculative.draft_type_k, gemma4.GEMMA4_KV_CACHE_Q8_0)
        self.assertEqual(speculative.draft_type_v, gemma4.GEMMA4_KV_CACHE_Q8_0)

    def test_runtime_mtp_failure_is_not_silently_reported_as_mtp(self):
        class FakeRealLlama:
            __module__ = "llama_cpp.llama"

            def __init__(self, **_kwargs):
                pass

        with patch.object(gemma4, "_ensure_mtp_model_file", return_value=Path("gemma-mtp.gguf")), \
                patch("llama_cpp.llama_speculative.SpecConfig", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                gemma4._create_runtime_llm(
                    FakeRealLlama,
                    model_path=Path("gemma.gguf"),
                    handler=object(),
                    debug=False,
                    gemma4_mtp=True,
                )

    def test_runtime_false_uses_original_non_mtp_constructor(self):
        created = {}

        class FakeRealLlama:
            __module__ = "llama_cpp.llama"

            def __init__(self, **kwargs):
                created.update(kwargs)

        with patch.object(gemma4, "_ensure_mtp_model_file") as ensure:
            llm = gemma4._create_runtime_llm(
                FakeRealLlama,
                model_path=Path("gemma.gguf"),
                handler=object(),
                debug=False,
                gemma4_mtp=False,
                seed=123456,
            )

        self.assertIsInstance(llm, FakeRealLlama)
        self.assertNotIn("logits_all", created)
        self.assertEqual(created["n_ctx"], 32768)
        self.assertEqual(created["type_k"], gemma4.GEMMA4_KV_CACHE_Q8_0)
        self.assertEqual(created["type_v"], gemma4.GEMMA4_KV_CACHE_Q8_0)
        self.assertFalse(created["swa_full"])
        self.assertEqual(created["seed"], 123456)
        ensure.assert_not_called()

    def test_json_completion_uses_fast_unconstrained_path_when_gemma_returns_valid_json(self):
        calls = []

        class FakeLlama:
            def create_chat_completion(self, **kwargs):
                calls.append(kwargs)
                return {
                    "choices": [
                        {"message": {"content": '{"confidence":"high"}'}}
                    ]
                }

        payload, raw = gemma4._gemma_chat_json(
            FakeLlama(), [{"role": "user", "content": "Return JSON"}]
        )

        self.assertEqual(payload, {"confidence": "high"})
        self.assertEqual(raw, '{"confidence":"high"}')
        self.assertEqual(len(calls), 1)
        self.assertNotIn("response_format", calls[0])
        self.assertEqual(calls[0]["max_tokens"], gemma4.GEMMA4_CHUNK_RESPONSE_TOKENS)
        self.assertEqual(calls[0]["reasoning_budget"], 4096)
        self.assertEqual(calls[0]["reasoning_start"], "<|think|>")
        self.assertEqual(calls[0]["reasoning_end"], "<channel|>")
        self.assertEqual(
            calls[0]["reasoning_budget_message"],
            "\n...Wait, I have been thinking long enough. Let me start answering the user's question.\n",
        )
        self.assertTrue(calls[0]["reasoning_start_in_prompt"])

    def test_debug_raw_output_log_records_the_unparsed_llama_response(self):
        response = {
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": '{"partial":true',
                    "reasoning_content": "private thought text",
                },
            }],
            "usage": {"completion_tokens": 16384},
        }

        class FakeLlama:
            def create_chat_completion(self, **_kwargs):
                return response

        previous_path = gemma4._ACTIVE_RAW_OUTPUT_PATH
        previous_operation = gemma4._ACTIVE_RAW_OUTPUT_OPERATION
        previous_live_path = gemma4._ACTIVE_LIVE_OUTPUT_PATH
        previous_live_operation = gemma4._ACTIVE_LIVE_OUTPUT_OPERATION
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with patch.object(gemma4.tempfile, "gettempdir", return_value=temporary):
                    path = gemma4.reset_gemma4_raw_output_log(True)
                    gemma4._configure_raw_output_log({
                        "debug": True,
                        "operation": "timing_plan",
                    })
                    with self.assertRaises(gemma4.Gemma4ObservationError):
                        gemma4._gemma_chat_json(
                            FakeLlama(),
                            [{"role": "user", "content": "Return JSON"}],
                            mtp_active=True,
                            max_tokens=gemma4.GEMMA4_TIMING_RESPONSE_TOKENS,
                        )
                    text = path.read_text(encoding="utf-8")

                self.assertIn("timing_plan | initial unconstrained response", text)
                self.assertIn('"finish_reason": "length"', text)
                self.assertIn('{\\"partial\\":true', text)
                self.assertIn("private thought text", text)
                self.assertIn('"completion_tokens": 16384', text)
        finally:
            gemma4._ACTIVE_RAW_OUTPUT_PATH = previous_path
            gemma4._ACTIVE_RAW_OUTPUT_OPERATION = previous_operation
            gemma4._ACTIVE_LIVE_OUTPUT_PATH = previous_live_path
            gemma4._ACTIVE_LIVE_OUTPUT_OPERATION = previous_live_operation

    def test_live_output_log_receives_decoder_text_before_completion(self):
        """The live transcript must be readable while a generate iterator is active."""
        class FakeLlama:
            def generate(self, *_args, **_kwargs):
                yield 10
                yield 11

            def detokenize(self, tokens, special=False):
                return {10: b"live ", 11: b"Gemma"}[tokens[0]]

        previous_live_path = gemma4._ACTIVE_LIVE_OUTPUT_PATH
        previous_live_operation = gemma4._ACTIVE_LIVE_OUTPUT_OPERATION
        previous_raw_path = gemma4._ACTIVE_RAW_OUTPUT_PATH
        previous_raw_operation = gemma4._ACTIVE_RAW_OUTPUT_OPERATION
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with patch.object(gemma4.tempfile, "gettempdir", return_value=temporary):
                    path = gemma4.reset_gemma4_live_output_log()
                    gemma4._configure_raw_output_log({"operation": "timing_plan"})
                    llm = FakeLlama()
                    gemma4._install_worker_token_progress(llm)
                    self.assertEqual(list(llm.generate([1, 2, 3])), [10, 11])
                    text = path.read_text(encoding="utf-8")

                self.assertIn("timing_plan | decoder generation 1", text)
                self.assertIn("live Gemma", text)
        finally:
            gemma4._ACTIVE_LIVE_OUTPUT_PATH = previous_live_path
            gemma4._ACTIVE_LIVE_OUTPUT_OPERATION = previous_live_operation
            gemma4._ACTIVE_RAW_OUTPUT_PATH = previous_raw_path
            gemma4._ACTIVE_RAW_OUTPUT_OPERATION = previous_raw_operation

    def test_incomplete_root_json_never_salvages_a_nested_character_object(self):
        truncated = (
            '{"confidence":"high","last_seen_character_state":['
            '{"character_name":"Heman","subject":"<Subject 1>"}'
        )
        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "incomplete or malformed top-level JSON",
        ):
            gemma4._extract_json_object(truncated)

    def test_json_extraction_ignores_an_example_inside_a_completed_thought(self):
        content = (
            'I will return {"example":true but this is not complete.\n'
            '<channel|>{"confidence":"high"}'
        )
        payload, raw = gemma4._extract_json_object(content)
        self.assertEqual(payload, {"confidence": "high"})
        self.assertEqual(raw, '{"confidence":"high"}')

    def test_chunk_response_budget_leaves_room_for_thinking_and_final_json(self):
        self.assertEqual(gemma4.GEMMA4_CHUNK_RESPONSE_TOKENS, 8132)

    def test_reasoning_budget_matches_the_native_gemma4_template(self):
        kwargs = gemma4._gemma_reasoning_budget_kwargs()
        self.assertEqual(kwargs["reasoning_start"], "<|think|>")
        self.assertEqual(kwargs["reasoning_end"], "<channel|>")
        self.assertTrue(kwargs["reasoning_start_in_prompt"])

    def test_json_completion_accepts_valid_json_returned_as_gemma_reasoning_content(self):
        class FakeLlama:
            def create_chat_completion(self, **_kwargs):
                return {"choices": [{"message": {"content": "", "reasoning_content": '{"confidence":"high"}'}}]}

        payload, raw = gemma4._gemma_chat_json(
            FakeLlama(), [{"role": "user", "content": "Return JSON"}]
        )
        self.assertEqual(payload, {"confidence": "high"})
        self.assertEqual(raw, '{"confidence":"high"}')

    def test_json_completion_uses_grammar_only_to_recover_malformed_json(self):
        calls = []
        responses = iter(("not json", '{"confidence":"high"}'))

        class FakeLlama:
            def create_chat_completion(self, **kwargs):
                calls.append(kwargs)
                return {
                    "choices": [
                        {"message": {"content": next(responses)}}
                    ]
                }

        with self.assertLogs(level="WARNING") as captured:
            payload, _raw = gemma4._gemma_chat_json(
                FakeLlama(), [{"role": "user", "content": "Return JSON"}]
            )

        self.assertEqual(payload, {"confidence": "high"})
        self.assertEqual(len(calls), 2)
        self.assertNotIn("response_format", calls[0])
        self.assertEqual(calls[1]["response_format"], {"type": "json_object"})
        self.assertIn("malformed instructed JSON", "\n".join(captured.output))

    def test_json_completion_repairs_thought_only_reply_as_an_append_chat_turn_before_grammar(self):
        initial_calls = []
        append_calls = []

        class FakeLlama:
            def create_chat_completion(self, **kwargs):
                initial_calls.append(kwargs)
                return {"choices": [{"message": {"content": ""}}]}

        class FakeHandler:
            def append_user_chat_completion(self, **kwargs):
                append_calls.append(kwargs)
                return {"choices": [{"message": {"content": '{"confidence":"high"}'}}]}

        with self.assertLogs(level="WARNING") as captured:
            payload, raw = gemma4._gemma_chat_json(
                FakeLlama(),
                [{"role": "user", "content": "Return JSON"}],
                handler=FakeHandler(),
            )

        self.assertEqual(payload, {"confidence": "high"})
        self.assertEqual(raw, '{"confidence":"high"}')
        self.assertEqual(len(initial_calls), 1)
        self.assertEqual(len(append_calls), 1)
        self.assertNotIn("response_format", append_calls[0])
        self.assertEqual(append_calls[0]["reasoning_budget"], 4096)
        self.assertEqual(append_calls[0]["reasoning_start"], "<|think|>")
        self.assertEqual(append_calls[0]["reasoning_end"], "<channel|>")
        self.assertEqual(
            append_calls[0]["reasoning_budget_message"],
            "\n...Wait, I have been thinking long enough. Let me start answering the user's question.\n",
        )
        self.assertIn("append-only JSON replacement", "\n".join(captured.output))

    def test_mtp_empty_json_exits_before_slow_append_or_grammar_repairs(self):
        append_calls = []

        class FakeLlama:
            def create_chat_completion(self, **_kwargs):
                return {"choices": [{"message": {"content": ""}}]}

        class FakeHandler:
            def append_user_chat_completion(self, **kwargs):
                append_calls.append(kwargs)
                return {"choices": [{"message": {"content": '{"confidence":"high"}'}}]}

        with self.assertRaisesRegex(gemma4.Gemma4MTPOutputError, "original decoder"):
            gemma4._gemma_chat_json(
                FakeLlama(),
                [{"role": "user", "content": "Return JSON"}],
                handler=FakeHandler(),
                mtp_active=True,
            )

        self.assertEqual(append_calls, [])

    def test_native_mtp_factory_configures_target_before_construction(self):
        target_config = {}

        def ordinary_model_defaults():
            return SimpleNamespace(load_mtp=False)

        def ordinary_context_defaults():
            return SimpleNamespace(n_rs_seq=0, kv_unified=False)

        fake_low_level = SimpleNamespace(
            llama_model_default_params=ordinary_model_defaults,
            llama_context_default_params=ordinary_context_defaults,
        )
        fake_package = SimpleNamespace(llama_cpp=fake_low_level)

        class FakeTarget:
            def __init__(self, **kwargs):
                target_config["model_load_mtp"] = fake_low_level.llama_model_default_params().load_mtp
                target_config["n_rs_seq"] = fake_low_level.llama_context_default_params().n_rs_seq
                target_config["kv_unified"] = fake_low_level.llama_context_default_params().kv_unified
                target_config["kwargs"] = kwargs

            def close(self):
                target_config["closed"] = True

        owner = object()
        with patch.dict(sys.modules, {"llama_cpp": fake_package}), patch.object(
            gemma4_mtp, "attach_gemma4_mtp", return_value=owner
        ) as attach:
            target = gemma4_mtp.create_native_mtp_llama(
                FakeTarget,
                model_path="target.gguf",
                draft_model_path="assistant.gguf",
                num_pred_tokens=4,
                n_ctx=gemma4.GEMMA4_CONTEXT_TOKENS,
            )

        self.assertTrue(target_config["model_load_mtp"])
        self.assertEqual(target_config["n_rs_seq"], 0)
        self.assertTrue(target_config["kv_unified"])
        self.assertTrue(target_config["kwargs"]["logits_all"])
        self.assertIs(fake_low_level.llama_model_default_params, ordinary_model_defaults)
        self.assertIs(fake_low_level.llama_context_default_params, ordinary_context_defaults)
        self.assertIs(target._hr_endless_mtp, owner)
        attach.assert_called_once_with(target, "assistant.gguf", num_pred_tokens=4)

    def test_native_mtp_partial_rejection_restores_and_replays_accepted_prefix(self):
        owner = gemma4_mtp.Gemma4MTPDraft.__new__(gemma4_mtp.Gemma4MTPDraft)
        owner._proposal_base = 10
        owner._proposal_count = 4
        owner._proposal_tokens = [201, 202, 203, 204]
        owner._proposal_verified_end = 15
        owner._proposal_sampled = [201, 999]
        owner._checkpoint_data = object()
        owner._checkpoint_size = 37
        owner.accepted_tokens = 0
        owner.rollback_count = 0
        owner.replayed_tokens = 0
        owner.target_ctx = object()
        owner.target = SimpleNamespace(
            input_ids=torch.tensor(
                [*range(10), 101, 201, 202, 203, 204], dtype=torch.int32
            ).numpy(),
            n_tokens=12,
            _requires_eval=False,
        )
        state_calls = []

        def restore_state(ctx, data, size, seq_id, flags):
            state_calls.append((ctx, data, size, seq_id, flags))
            return size

        owner._llama_cpp = SimpleNamespace(
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY=1,
            llama_state_seq_set_data_ext=restore_state,
        )
        removals = []
        owner._original_kv_cache_seq_rm = (
            lambda seq_id, p0, p1: removals.append((seq_id, p0, p1)) or False
        )
        replayed = []

        def replay(tokens):
            replayed.extend(tokens)
            owner.target.n_tokens += len(tokens)

        owner._decode_target_tokens = replay

        handled = owner._kv_cache_seq_rm(None, -1, 12, -1)

        self.assertTrue(handled)
        self.assertEqual(removals, [(-1, 10, -1)])
        self.assertEqual(replayed, [101, 201])
        self.assertEqual(owner.target.n_tokens, 12)
        self.assertEqual(owner.accepted_tokens, 1)
        self.assertEqual(owner.rollback_count, 1)
        self.assertEqual(owner.replayed_tokens, 2)
        self.assertEqual(state_calls[0][2:], (37, 0, 1))
        self.assertIsNone(owner._proposal_base)

    def test_native_mtp_checkpoint_uses_reusable_reference_host_state(self):
        owner = gemma4_mtp.Gemma4MTPDraft.__new__(gemma4_mtp.Gemma4MTPDraft)
        owner.target_ctx = object()
        owner.checkpoint_seconds = 0.0
        owner._checkpoint_buffer = None
        owner._checkpoint_capacity = 0
        state_calls = []

        def checkpoint_size(ctx, seq_id, flags):
            state_calls.append(("size", ctx, seq_id, flags))
            return 16

        def checkpoint_data(ctx, data, size, seq_id, flags):
            state_calls.append(("data", ctx, size, seq_id, flags))
            return size

        owner._llama_cpp = SimpleNamespace(
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY=1,
            llama_state_seq_get_size_ext=checkpoint_size,
            llama_state_seq_get_data_ext=checkpoint_data,
        )

        owner._save_target_checkpoint(42)
        first_buffer = owner._checkpoint_data
        owner._clear_proposal()
        owner._save_target_checkpoint(43)

        self.assertEqual(state_calls[0][2:], (0, 1))
        self.assertEqual(state_calls[1][2:], (16, 0, 1))
        self.assertEqual(state_calls[2][2:], (0, 1))
        self.assertEqual(state_calls[3][2:], (16, 0, 1))
        self.assertIs(owner._checkpoint_data, first_buffer)
        self.assertEqual(owner._checkpoint_size, 16)
        self.assertEqual(owner._proposal_base, 43)

    def test_worker_generation_emits_decode_tokens_per_second(self):
        class FakeLlama:
            def generate(self, *_args, **_kwargs):
                yield 10
                yield 11
                yield 12

            def detokenize(self, tokens, special=False):
                return bytes([tokens[0]])

        llm = FakeLlama()
        gemma4._install_worker_token_progress(llm)
        with patch.object(gemma4.time, "perf_counter", side_effect=[1.0, 1.5, 2.0]), \
                patch("builtins.print") as printed:
            self.assertEqual(list(llm.generate([1, 2, 3])), [10, 11, 12])

        progress_lines = [
            call.args[0]
            for call in printed.call_args_list
            if call.args and call.args[0].startswith(gemma4._WORKER_PROGRESS_PREFIX)
        ]
        self.assertEqual(len(progress_lines), 2)
        payload = json.loads(progress_lines[-1][len(gemma4._WORKER_PROGRESS_PREFIX):])
        self.assertEqual(payload["tokens"], 3)
        self.assertAlmostEqual(payload["tokens_per_second"], 2.0)

    def test_parent_streams_worker_token_rate_without_mixing_it_into_result(self):
        class CapturingInput(io.StringIO):
            def close(self):
                self.closed_by_worker = True

        class FakeProcess:
            def __init__(self):
                self.stdin = CapturingInput()
                self.stdout = io.StringIO(
                    gemma4._WORKER_PROGRESS_PREFIX
                    + '{"generation":1,"tokens":64,"tokens_per_second":128.5}\n'
                    + gemma4._WORKER_RESULT_PREFIX
                    + '{"ok":true}\n'
                )
                self.returncode = None

            def wait(self):
                self.returncode = 0

            def kill(self):
                self.returncode = -9

        rates = []
        process = FakeProcess()
        output = gemma4._stream_worker_output(
            process,
            {"operation": "test"},
            lambda tokens, rate, generation: rates.append((tokens, rate, generation)),
        )

        self.assertEqual(rates, [(64, 128.5, 1)])
        self.assertEqual(output, gemma4._WORKER_RESULT_PREFIX + '{"ok":true}\n')
        self.assertEqual(json.loads(process.stdin.getvalue()), {"operation": "test"})

    def test_parent_kills_blocked_worker_when_comfyui_interrupts(self):
        class FakeProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.killed = False
                self.returncode = None
                self.released = threading.Event()

                process = self

                class BlockingOutput:
                    closed = False

                    def __iter__(self):
                        while not process.killed:
                            process.released.wait(0.01)
                        return iter(())

                    def close(self):
                        self.closed = True

                self.stdout = BlockingOutput()

            def poll(self):
                return -9 if self.killed else None

            def wait(self):
                self.returncode = -9 if self.killed else 0

            def kill(self):
                self.killed = True
                self.released.set()

        process = FakeProcess()
        with patch.object(
            gemma4.comfy.model_management,
            "processing_interrupted",
            return_value=True,
        ), patch.object(
            gemma4.comfy.model_management,
            "throw_exception_if_processing_interrupted",
            side_effect=gemma4.comfy.model_management.InterruptProcessingException,
        ):
            with self.assertRaises(gemma4.comfy.model_management.InterruptProcessingException):
                gemma4._stream_worker_output(process, {"operation": "test"})
        self.assertTrue(process.killed)

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
                        "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
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
        self.assertEqual(captured["handler_kwargs"]["mmproj_path"], "mmproj")
        self.assertEqual(captured["handler_kwargs"]["image_min_tokens"], 70)
        self.assertEqual(captured["handler_kwargs"]["image_max_tokens"], 1120)
        self.assertEqual(captured["handler_kwargs"]["batch_max_tokens"], 1120)
        self.assertTrue(captured["closed"])
        self.assertEqual(result.confidence, "high")

    def test_preproduction_timing_plan_covers_each_source_shot_and_feeds_relevant_schedule(self):
        request = self.timing_request()
        _system, planning_prompt = gemma4._render_timing_plan_messages(request)
        self.assertIn("Source Shot 2: global frames 68-147", planning_prompt)
        self.assertNotIn("Chunk 3: sampled global frames", planning_prompt)
        _shot_system, shot_prompt = gemma4._render_single_shot_plan_messages(
            request,
            json.dumps(self.production_bible_response()),
            request["source_shots"][1],
        )
        self.assertIn("source-relative half-open [5,39)", shot_prompt)
        self.assertIn("Independently plan Source Shot 2", shot_prompt)

        plan = gemma4._validate_timing_plan(
            self.timing_response(), request, json.dumps(self.timing_response())
        )
        self.assertEqual([shot.source_shot for shot in plan.shots], [1, 2])
        self.assertEqual([shot.light_change for shot in plan.shots], [False, False])
        self.assertEqual([(beat.start_frame, beat.end_frame) for beat in plan.shots[1].visual_beats], [(0, 24), (24, 48), (48, 68), (68, 80)])
        self.assertEqual(
            plan.character_name_table_text(),
            "- Heman -> <Subject 1>\n- Tila -> <Subject 2>",
        )
        self.assertEqual(
            [(item.start_frame, item.end_frame) for item in plan.shots[1].continuity_slices],
            [(0, 5), (5, 39), (39, 80)],
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
                "required_marker": "[Shot 2]",
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
        continuity = plan.continuity_for_target_shots([{
            "shot_number": 2,
            "shot_start": 68,
            "shot_end": 148,
            "target_start": 73,
            "target_end": 107,
        }])
        self.assertIn("Heman (<Subject 1>)", continuity)
        self.assertIn("planned entry: mounted on the moving tiger", continuity)
        self.assertEqual(
            plan.current_character_subjects([{
                "shot_number": 2,
                "shot_start": 68,
                "shot_end": 148,
                "target_start": 73,
                "target_end": 107,
            }]),
            (gemma4.GemmaCharacterSubject("Heman", "<Subject 1>"),),
        )

        chunk_request = self.request()
        chunk_request["preproduction_timing_plan"] = relevant
        chunk_request["character_name_table"] = plan.character_name_table_text()
        _system, observation = gemma4._render_observation_messages(chunk_request)
        self.assertIn("complete immutable preproduction timing schedule", observation)
        self.assertIn("MANDATORY CURRENT-SLICE BEAT COVERAGE", observation)
        self.assertIn("Source Shot 2: immutable preproduction timing schedule", observation)
        self.assertIn("Heman -> <Subject 1>", observation)
        self.assertIn("literal contiguous substring copied from your final detailed_description", observation)
        self.assertIn("exactly these eight fields", observation)

    def test_preproduction_requires_explicit_light_change_decision_per_shot(self):
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        invalid["shots"][0].pop("light_change")
        with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "Shot 1 light_change must be true or false"):
            gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))

    def test_clean_preproduction_memory_keeps_static_source_context_out_of_cached_chunks(self):
        timing_request = self.timing_request()
        plan = gemma4._validate_timing_plan(
            self.timing_response(), timing_request, json.dumps(self.timing_response())
        )
        bible = json.dumps(self.production_bible_response(), ensure_ascii=False, indent=2)
        plan = replace(plan, production_bible_json=bible)
        memory_system, memory = gemma4._render_preproduction_memory_messages(timing_request, plan)
        self.assertIn(timing_request["original_prompt"], memory)
        self.assertIn("Immutable global production bible", memory)
        self.assertIn("The mounted heroes reach and enter the temple", memory)
        self.assertNotIn("immutable preproduction timing schedule", memory)
        self.assertIn("# MiniMax H3 prompt-writing working summary", memory_system)

        request = self.request()
        request["preproduction_timing_plan"] = plan.for_target_shots(request["target_shots"], 24.0)
        request["production_bible"] = plan.production_bible_text()
        request["preproduction_current_slice"] = plan.current_slice_coverage_text(request["target_shots"])
        request["character_name_table"] = plan.character_name_table_text()
        request["preproduction_cache"] = {
            "format": "hr-endless-sampler-gemma4-preproduction-kv-v1",
            "state_path": "/tmp/unused-state",
            "manifest_path": "/tmp/unused-manifest",
        }
        _system, cached_observation = gemma4._render_observation_messages(request)
        self.assertIn("using the immutable preproduction memory", cached_observation)
        self.assertIn("Finalized preproduction plans for only the source shot", cached_observation)
        self.assertIn(request["preproduction_timing_plan"], cached_observation)
        self.assertIn("Mandatory current-slice portions", cached_observation)
        self.assertIn(request["previous_gemma_description"], cached_observation)
        self.assertIn("literal contiguous substring copied from your final detailed_description", cached_observation)
        self.assertIn("immediate `<Subject N> (Sx)` speaker form", cached_observation)
        self.assertIn("Start each source-shot segment with its authoritative camera-continuity sentence", cached_observation)
        self.assertNotIn(request["original_prompt"], cached_observation)
        self.assertNotIn(request["target_shots"][1]["source_body"], cached_observation)
        self.assertNotIn("COMPLETE RELEVANT PREPRODUCTION SCHEDULE", cached_observation)

    def test_replay_materializes_clean_cache_from_existing_timing_plan(self):
        request = self.timing_request()
        request["preproduction_cache"] = {
            "format": "hr-endless-sampler-gemma4-preproduction-kv-v1",
            "state_path": "/tmp/unused-state",
            "manifest_path": "/tmp/unused-manifest",
        }
        plan = gemma4._validate_timing_plan(
            self.timing_response(), request, json.dumps(self.timing_response())
        )
        captured = {}

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **kwargs):
                captured["llama_kwargs"] = kwargs

            def close(self):
                captured["closed"] = True

        def fake_populate(llm, actual_request, actual_plan):
            captured["llm"] = llm
            captured["request"] = actual_request
            captured["plan"] = actual_plan
            return 123

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4, "_populate_preproduction_cache_state", side_effect=fake_populate), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            state_bytes = gemma4._materialize_preproduction_cache_in_process(request, plan, debug=False)

        self.assertEqual(state_bytes, 123)
        self.assertIs(captured["plan"], plan)
        self.assertIs(captured["request"], request)
        self.assertEqual(captured["llama_kwargs"]["n_gpu_layers"], -1)
        self.assertTrue(captured["closed"])

    def test_preproduction_allows_dialogue_overlay_and_exposes_it_as_current_coverage(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        request["source_shots"][1]["source_body"] += (
            " Heman (S1) says: <d>[English] Stay back!</d>"
        )
        request["original_prompt"] += " Heman (S1) says: <d>[English] Stay back!</d>"
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [{
            "start_frame": 24,
            "end_frame": 48,
            "type": "dialogue",
            "content": "<Subject 1> (S1) says: <d>[English] Stay back!</d>",
            "dialogue_segments": [
                {
                    "start_frame": 24,
                    "end_frame": 39,
                    "content": "<Subject 1> (S1) says: <d>[English] Stay</d>",
                },
                {
                    "start_frame": 39,
                    "end_frame": 48,
                    "content": "<Subject 1> (S1) says: <d>[English] back!</d>",
                },
            ],
        }]
        plan = gemma4._validate_timing_plan(response, request, json.dumps(response))
        target = [{
            "shot_number": 2,
            "shot_start": 68,
            "shot_end": 148,
            "target_start": 73,
            "target_end": 107,
        }]
        coverage = plan.mandatory_coverage(target)
        dialogue = next(item for item in coverage if item["id"] == "S2.O1.D1")
        self.assertEqual(dialogue["kind"], "overlay")
        self.assertEqual(dialogue["overlay_type"], "dialogue")
        self.assertEqual((dialogue["overlap_start_frame"], dialogue["overlap_end_frame"]), (24, 39))
        self.assertEqual(
            dialogue["action"],
            "<Subject 1> (S1) says: <d>[English] Stay</d>",
        )
        self.assertFalse(dialogue["dialogue_continuation"])
        rendered = plan.for_target_shots(target, 24.0)
        self.assertIn("[S2.O1] dialogue at source-relative frames 24-47", rendered)
        self.assertIn("[S2.O1.D1] source-relative frames 24-38", rendered)
        self.assertIn("Required now [S2.O1.D1], overlay/dialogue", rendered)
        self.assertNotIn("<d>[English] Stay back!</d>\n\nCOMPLETE", rendered.split("Required now", 1)[1].split("COMPLETE", 1)[0])

        later_target = [{
            "shot_number": 2,
            "shot_start": 68,
            "shot_end": 148,
            "target_start": 107,
            "target_end": 148,
        }]
        later = next(item for item in plan.mandatory_coverage(later_target) if item["id"] == "S2.O1.D2")
        self.assertTrue(later["dialogue_continuation"])
        self.assertEqual(later["action"], "<Subject 1> (S1) says: <d>[English] back!</d>")
        self.assertIn("continues without restarting", plan.current_slice_coverage_text(later_target))

    def test_preproduction_rejects_repeated_words_across_dialogue_segments(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        request["source_shots"][1]["source_body"] += (
            " Heman (S1) says: <d>[English] Stay back now!</d>"
        )
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [{
            "start_frame": 24,
            "end_frame": 48,
            "type": "dialogue",
            "content": "<Subject 1> (S1) says: <d>[English] Stay back now!</d>",
            "dialogue_segments": [
                {
                    "start_frame": 24,
                    "end_frame": 39,
                    "content": "<Subject 1> (S1) says: <d>[English] Stay back</d>",
                },
                {
                    "start_frame": 39,
                    "end_frame": 48,
                    "content": "<Subject 1> (S1) says: <d>[English] back now!</d>",
                },
            ],
        }]
        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "every original dialogue word and punctuation token exactly once",
        ):
            gemma4._validate_timing_plan(response, request, json.dumps(response))

    def test_preproduction_allows_one_source_speech_to_span_multiple_dialogue_overlays(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        request["source_shots"][1]["source_body"] += (
            " Heman (S1) says: <d>[English] Stay back now, do not move!</d>"
        )
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [
            {
                "start_frame": 24,
                "end_frame": 39,
                "type": "dialogue",
                "content": "<Subject 1> (S1) says: <d>[English] Stay back now,</d>",
                "dialogue_segments": [{
                    "start_frame": 24,
                    "end_frame": 39,
                    "content": "<Subject 1> (S1) says: <d>[English] Stay back now,</d>",
                }],
            },
            {
                "start_frame": 39,
                "end_frame": 60,
                "type": "dialogue",
                "content": "<Subject 1> (S1) says: <d>[English] do not move!</d>",
                "dialogue_segments": [{
                    "start_frame": 39,
                    "end_frame": 60,
                    "content": "<Subject 1> (S1) says: <d>[English] do not move!</d>",
                }],
            },
        ]
        plan = gemma4._validate_timing_plan(response, request, json.dumps(response))
        first, second = plan.shots[1].overlays
        self.assertFalse(first.continues_source_dialogue)
        self.assertTrue(second.continues_source_dialogue)
        later_target = [{
            "shot_number": 2,
            "shot_start": 68,
            "shot_end": 148,
            "target_start": 107,
            "target_end": 148,
        }]
        later = next(item for item in plan.mandatory_coverage(later_target) if item["id"] == "S2.O2.D1")
        self.assertTrue(later["dialogue_continuation"])
        self.assertEqual(
            later["action"],
            "<Subject 1> (S1) says: <d>[English] do not move!</d>",
        )

    def test_preproduction_rejects_unrealistically_compressed_dialogue(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        words = (
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
            "sixteen seventeen eighteen nineteen twenty"
        )
        request["source_shots"][1]["source_body"] += (
            f" Heman (S1) says: <d>[English] {words}</d>"
        )
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [{
            "start_frame": 0,
            "end_frame": 80,
            "type": "dialogue",
            "content": f"<Subject 1> (S1) says: <d>[English] {words}</d>",
            "dialogue_segments": [
                {
                    "start_frame": 0,
                    "end_frame": 5,
                    "content": "<Subject 1> (S1) says: <d>[English] one two</d>",
                },
                {
                    "start_frame": 5,
                    "end_frame": 39,
                    "content": (
                        "<Subject 1> (S1) says: <d>[English] three four five six seven eight nine ten</d>"
                    ),
                },
                {
                    "start_frame": 39,
                    "end_frame": 80,
                    "content": (
                        "<Subject 1> (S1) says: <d>[English] eleven twelve thirteen fourteen fifteen sixteen "
                        "seventeen eighteen nineteen twenty</d>"
                    ),
                },
            ],
        }]

        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "compresses 20 spoken words into 3.333s",
        ):
            gemma4._validate_timing_plan(response, request, json.dumps(response))

    def test_preproduction_dialogue_density_reserves_ellipsis_pause_time(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        spoken = "one two three... four five six... seven eight nine ten eleven twelve"
        request["source_shots"][1]["source_body"] += (
            f" Heman (S1) says: <d>[English] {spoken}</d>"
        )
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [{
            "start_frame": 0,
            "end_frame": 48,
            "type": "dialogue",
            "content": f"<Subject 1> (S1) says: <d>[English] {spoken}</d>",
            "dialogue_segments": [
                {
                    "start_frame": 0,
                    "end_frame": 5,
                    "content": "<Subject 1> (S1) says: <d>[English] one two</d>",
                },
                {
                    "start_frame": 5,
                    "end_frame": 39,
                    "content": "<Subject 1> (S1) says: <d>[English] three... four five six...</d>",
                },
                {
                    "start_frame": 39,
                    "end_frame": 48,
                    "content": (
                        "<Subject 1> (S1) says: <d>[English] seven eight nine ten eleven twelve</d>"
                    ),
                },
            ],
        }]

        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "reserving 1.000s for 2 ellipsis pause",
        ):
            gemma4._validate_timing_plan(response, request, json.dumps(response))

    def test_preproduction_rejects_missing_or_repeated_words_between_dialogue_overlays(self):
        request = self.timing_request()
        request["source_shots"] = json.loads(json.dumps(request["source_shots"]))
        request["source_shots"][1]["source_body"] += (
            " Heman (S1) says: <d>[English] Stay back now!</d>"
        )
        response = self.timing_response()
        response["shots"] = json.loads(json.dumps(response["shots"]))
        response["shots"][1]["overlays"] = [
            {
                "start_frame": 24,
                "end_frame": 39,
                "type": "dialogue",
                "content": "<Subject 1> (S1) says: <d>[English] Stay back</d>",
                "dialogue_segments": [{
                    "start_frame": 24,
                    "end_frame": 39,
                    "content": "<Subject 1> (S1) says: <d>[English] Stay back</d>",
                }],
            },
            {
                "start_frame": 39,
                "end_frame": 48,
                "type": "dialogue",
                "content": "<Subject 1> (S1) says: <d>[English] back now!</d>",
                "dialogue_segments": [{
                    "start_frame": 39,
                    "end_frame": 48,
                    "content": "<Subject 1> (S1) says: <d>[English] back now!</d>",
                }],
            },
        ]
        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "chronological dialogue overlay contents must reconstruct every source",
        ):
            gemma4._validate_timing_plan(response, request, json.dumps(response))

    def test_chunk_validator_accepts_only_the_assigned_later_dialogue_segment(self):
        request = self.request()
        request["character_name_table"] = "- Heman -> <Subject 1>"
        request["original_prompt"] = request["original_prompt"].replace(
            "Heman says: <d>[English] Stay back!</d>",
            "Heman (S1) says: <d>[English] Stay back!</d>",
        )
        request["target_shots"][1]["source_body"] = (
            "Heman (S1) says: <d>[English] Stay back!</d>"
        )
        voice_profile = "a low-pitched, warm, slightly raspy voice with a measured speaking rate"
        request["production_bible"] = json.dumps({
            "speaker_voice_profiles": [{
                "speaker_id": "S1",
                "source": "<Subject 1>",
                "voice_profile": voice_profile,
            }],
        })
        request["mandatory_coverage"] = [{
            "id": "S5.O1.D2",
            "kind": "overlay",
            "overlay_type": "dialogue",
            "source_shot": 5,
            "source_start_frame": 12,
            "source_end_frame": 23,
            "overlap_start_frame": 12,
            "overlap_end_frame": 23,
            "dialogue_segment_index": 2,
            "dialogue_segment_count": 2,
            "dialogue_continuation": True,
            "action": "<Subject 1> (S1) says: <d>[English] back!</d>",
        }]
        value = {
            "confidence": "high",
            "analysis": "The warning is already in progress and only its final word remains.",
            "timing_plan": "Continue the second immutable dialogue segment without restarting.",
            "end_state": "Heman has finished the warning.",
            "retention_analysis": "",
            "last_seen_character_state": [],
            "coverage": [{
                "id": "S5.O1.D2",
                "status": "completes",
                "evidence": voice_profile,
            }],
            "detailed_description": (
                "The already-started warning carries through the real cut. "
                f"[Shot 5] At 00:00.667, <Subject 1> (S1), {voice_profile}, continues uninterrupted "
                "and says: <d>[English] back!</d>"
            ),
        }
        result = gemma4._validate_chunk_prompt(value, request, json.dumps(value))
        self.assertNotIn(
            "Gemma 4 modified or invented dialogue instead of preserving source words",
            result.validation_warnings,
        )
        self.assertFalse(any("dialogue speaker form" in item for item in result.validation_warnings))
        self.assertFalse(any("exact assigned dialogue segment" in item for item in result.validation_warnings))
        self.assertFalse(any("voice profile" in item for item in result.validation_warnings))
        self.assertFalse(any("continues uninterrupted" in item for item in result.validation_warnings))

        missing_voice = dict(value)
        missing_voice["detailed_description"] = missing_voice["detailed_description"].replace(
            f", {voice_profile}, continues uninterrupted",
            ", continues",
        )
        warned = gemma4._validate_chunk_prompt(missing_voice, request, json.dumps(missing_voice))
        self.assertTrue(any("voice profile" in item for item in warned.validation_warnings))
        self.assertTrue(any("continues uninterrupted" in item for item in warned.validation_warnings))

    def test_preproduction_retries_only_invalid_shot_when_visual_beats_do_not_cover_it(self):
        captured = {"messages": [], "append": []}
        invalid = self.single_shot_response(1)
        invalid["visual_beats"][-1]["end_frame"] = 78
        initial_responses = [
            self.production_bible_response(),
            self.single_shot_response(0),
            invalid,
        ]
        append_responses = [self.single_shot_response(1)]

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

            def append_user_chat_completion(self, **kwargs):
                captured["append"].append(kwargs["content"])
                return {"choices": [{"message": {"content": json.dumps(append_responses.pop(0))}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                captured.setdefault("max_tokens", []).append(kwargs["max_tokens"])
                return {"choices": [{"message": {"content": json.dumps(initial_responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._plan_timing_in_process(self.timing_request(), debug=False)

        self.assertEqual(len(result.attempts), 4)
        self.assertIn("SOURCE SHOT 2 PLAN CORRECTION REQUIRED", result.attempts[-1].correction_prompt)
        self.assertEqual(
            captured["max_tokens"],
            [
                gemma4.GEMMA4_GLOBAL_PREPRODUCTION_RESPONSE_TOKENS,
                gemma4.GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS,
                gemma4.GEMMA4_SHOT_PREPRODUCTION_RESPONSE_TOKENS,
            ],
        )
        self.assertEqual(len(captured["append"]), 1)
        self.assertEqual([shot.source_shot for shot in result.shots], [1, 2])
        self.assertTrue(captured["closed"])

    def test_preproduction_correction_schema_requires_character_name_table_array(self):
        captured = {"messages": []}
        invalid = self.production_bible_response()
        invalid["character_name_table"] = {"Heman": "<Subject 1>"}
        initial_responses = [invalid, self.single_shot_response(0), self.single_shot_response(1)]
        append_responses = [self.production_bible_response()]

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

            def append_user_chat_completion(self, **_kwargs):
                return {"choices": [{"message": {"content": json.dumps(append_responses.pop(0))}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                captured["messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(initial_responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._plan_timing_in_process(self.timing_request(), debug=False)

        self.assertEqual(len(result.attempts), 4)
        correction_prompt = result.attempts[1].correction_prompt
        self.assertIn("GLOBAL PREPRODUCTION CORRECTION REQUIRED", correction_prompt)
        self.assertIn("every explicit character-name-to-<Subject N>", correction_prompt)
        self.assertIn("response field 'character_name_table' must be an array", correction_prompt)
        self.assertEqual(result.character_name_table_text(), "- Heman -> <Subject 1>\n- Tila -> <Subject 2>")
        self.assertTrue(captured["closed"])

    def test_global_bible_requires_every_explicit_source_character_mapping(self):
        response = self.production_bible_response()
        response["character_name_table"] = response["character_name_table"][:1]
        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "omits or changes explicit source mapping.*Tila.*<Subject 2>",
        ):
            gemma4._validate_production_bible(
                response,
                self.timing_request(),
                json.dumps(response),
            )

    def test_global_bible_requires_immutable_profile_for_explicit_speaker(self):
        request = self.timing_request()
        request["original_prompt"] += (
            "\n<Subject 1> (S1) says: <d>[English] Stay back!</d>"
        )
        response = self.production_bible_response()
        with self.assertRaisesRegex(
            gemma4.Gemma4ObservationError,
            "speaker_voice_profiles omits explicit source speaker.*S1",
        ):
            gemma4._validate_production_bible(response, request, json.dumps(response))

        response["speaker_voice_profiles"] = [{
            "speaker_id": "S1",
            "source": "<Subject 1>",
            "voice_profile": "a low-pitched, warm, slightly raspy voice with a measured speaking rate",
        }]
        confidence, _analysis, _table = gemma4._validate_production_bible(
            response,
            request,
            json.dumps(response),
        )
        self.assertEqual(confidence, "high")

    def test_preproduction_correction_is_an_appended_chat_turn_when_runtime_supports_it(self):
        captured = {"initial_messages": [], "append_content": []}
        invalid = self.single_shot_response(1)
        invalid["visual_beats"][-1]["end_frame"] = 78
        initial_responses = [
            self.production_bible_response(),
            self.single_shot_response(0),
            invalid,
        ]
        append_responses = [self.single_shot_response(1)]

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

            def append_user_chat_completion(self, **kwargs):
                captured["append_content"].append(kwargs["content"])
                return {"choices": [{"message": {"content": json.dumps(append_responses.pop(0))}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                captured["initial_messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(initial_responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._plan_timing_in_process(self.timing_request(), debug=False)

        self.assertEqual(len(result.attempts), 4)
        self.assertEqual(len(captured["initial_messages"]), 3)
        self.assertEqual(len(captured["append_content"]), 1)
        self.assertIn("SOURCE SHOT 2 PLAN CORRECTION REQUIRED", captured["append_content"][0])
        self.assertTrue(captured["closed"])

    def test_preproduction_keeps_requesting_semantic_corrections_until_plan_is_valid(self):
        captured = {"append_content": []}
        invalid_one = self.single_shot_response(1)
        invalid_one["visual_beats"][-1]["end_frame"] = 78
        invalid_two = self.single_shot_response(1)
        invalid_two["continuity_slices"] = invalid_two["continuity_slices"][:-1]
        initial_responses = [
            self.production_bible_response(),
            self.single_shot_response(0),
            invalid_one,
        ]
        append_responses = [invalid_two, self.single_shot_response(1)]

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

            def append_user_chat_completion(self, **kwargs):
                captured["append_content"].append(kwargs["content"])
                return {"choices": [{"message": {"content": json.dumps(append_responses.pop(0))}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **_kwargs):
                return {"choices": [{"message": {"content": json.dumps(initial_responses.pop(0))}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._plan_timing_in_process(self.timing_request(), debug=False)

        self.assertEqual(len(result.attempts), 5)
        self.assertEqual(len(captured["append_content"]), 2)
        self.assertIn("visual_beats end at 78", captured["append_content"][0])
        self.assertIn("continuity_slices", captured["append_content"][1])
        self.assertTrue(captured["closed"])

    def test_preproduction_rejects_gapped_schedule_without_sampler_fallback(self):
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        invalid["shots"][0]["visual_beats"][1]["start_frame"] = 40
        with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "contiguous"):
            gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))

    def test_preproduction_allows_named_character_to_be_absent_from_one_slice(self):
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        invalid["shots"][1]["continuity_slices"][1]["characters"] = []
        plan = gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))
        self.assertEqual(plan.shots[1].continuity_slices[1].characters, ())

    def test_preproduction_rejects_continuity_plan_that_omits_a_named_character_entirely(self):
        invalid = self.timing_response()
        invalid["shots"] = json.loads(json.dumps(invalid["shots"]))
        for continuity_slice in invalid["shots"][1]["continuity_slices"]:
            continuity_slice["characters"] = []
        with self.assertRaisesRegex(gemma4.Gemma4ObservationError, "omit explicitly named character.*entirely.*Heman"):
            gemma4._validate_timing_plan(invalid, self.timing_request(), json.dumps(invalid))

    def test_chunk_retention_requires_every_planned_character_without_internal_language(self):
        request = self.request()
        request["character_name_table"] = "- Heman -> <Subject 1>\n- Tila -> <Subject 2>"
        request["current_character_subjects"] = [
            {"character_name": "Heman", "subject": "<Subject 1>"},
            {"character_name": "Tila", "subject": "<Subject 2>"},
        ]
        last_seen = []
        for name, subject in (("Heman", "<Subject 1>"), ("Tila", "<Subject 2>")):
            last_seen.append({
                "character_name": name,
                "subject": subject,
                "last_seen_global_frame": 208,
                "last_seen_source_shot": 4,
                "environment": "inside the temple",
                "pose_and_position": "mounted on the tiger",
                "state_and_action": "moving forward",
                "spatial_relationships": "beside the other rider",
            })
        payload = {
            "confidence": "high",
            "analysis": "Both riders remain mounted.",
            "timing_plan": "Continue the current action.",
            "end_state": "Both riders remain inside the temple.",
            "retention_analysis": (
                "Gemma last-seen continuity state relevant to this chunk: "
                "Tila (<Subject 2>) remains mounted inside the temple."
            ),
            "last_seen_character_state": last_seen,
            "coverage": [],
            "detailed_description": (
                "Heman (<Subject 1>) continues forward. "
                "[Shot 5] At 00:00.667, <Subject 1> says: <d>[English] Stay back!</d>"
            ),
        }
        result = gemma4._validate_chunk_prompt(payload, request, json.dumps(payload))
        contract = gemma4._contract_validation_warnings(result.validation_warnings)
        self.assertTrue(any("omits planned character Heman" in warning for warning in contract))
        self.assertTrue(any("bookkeeping language" in warning for warning in contract))

        payload["retention_analysis"] = (
            "Heman (<Subject 1>) is mounted at the front of the tiger inside the temple; "
            "Tila (<Subject 2>) is mounted behind him, leaning forward."
        )
        corrected = gemma4._validate_chunk_prompt(payload, request, json.dumps(payload))
        self.assertFalse(gemma4._character_state_validation_warnings(corrected.validation_warnings))

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
        request["character_name_table"] = "- Heman -> <Subject 1>"
        request["previous_last_seen_character_state"] = [{
            "character_name": "Heman",
            "subject": "<Subject 1>",
            "last_seen_global_frame": 208,
            "last_seen_source_shot": 4,
            "environment": "inside the ancient temple",
            "pose_and_position": "standing on the temple floor",
            "state_and_action": "walking right",
            "spatial_relationships": "away from the tiger",
        }]
        _system, observation = gemma4._render_observation_messages(request)

        self.assertIn(request["previous_gemma_description"], observation)
        self.assertIn(request["previous_gemma_timing_plan"], observation)
        self.assertIn(request["previous_gemma_end_state"], observation)
        self.assertIn('"last_seen_global_frame": 208', observation)
        self.assertIn("Return exactly one entry for every immutable character", observation)
        self.assertIn("exact attached stills", observation)
        self.assertIn("latest rendered still is authoritative", gemma4._gemma_prompt_templates()["SYSTEM"])

        first_chunk = self.request()
        first_chunk["previous_chunk"] = None
        first_chunk["previous_shots"] = []
        first_chunk["observation_frame_numbers"] = []
        first_chunk["previous_gemma_description"] = None
        first_chunk["previous_gemma_timing_plan"] = None
        first_chunk["previous_gemma_end_state"] = None
        first_chunk["previous_last_seen_character_state"] = None
        _system, first_observation = gemma4._render_observation_messages(first_chunk)
        self.assertIn("No previous Gemma-directed detailed_description exists", first_observation)
        self.assertIn("No previous Gemma timing plan exists", first_observation)
        self.assertIn("No previous Gemma end state exists", first_observation)

    def test_observation_front_loads_physical_shot_starts_and_slice_request(self):
        _system, observation = gemma4._render_observation_messages(self.request())

        self.assertIn("Global source-shot timeline and chunk-local start frames:", observation)
        self.assertIn(
            "global [Shot 4]: starts before this physical chunk at global frame 193 "
            "(physical local frame -11); this chunk must author its global frames 209-219 "
            "(physical local frames 5-15).",
            observation,
        )
        self.assertIn(
            "global [Shot 5]: starts at global frame 220 "
            "(physical local frame 16); this chunk must author its global frames 220-242 "
            "(physical local frames 16-38).",
            observation,
        )
        self.assertIn(
            "Write one complete chunk-local detailed_description for the ending portion of global [Shot 4] "
            "and the opening portion of global [Shot 5].",
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
        self.assertIn("IMMUTABLE H3 SHOT MARKERS — GLOBAL SHOT LABELS, CHUNK-LOCAL TIMES", observation)
        self.assertIn("begin it as unmarked continuation prose", observation)
        self.assertNotIn('exact token: "[Shot 1]"', observation)
        self.assertIn('exact token: "[Shot 5] At 00:00.667,"', observation)
        self.assertIn("The chunk time-slice encompasses the end of global Source Shot 4", observation)
        self.assertIn(
            "global Source Shot 5 begins at chunk-local position 00:00.667",
            observation,
        )
        self.assertIn(
            "never before remaining action from global Source Shot 4",
            observation,
        )

    def test_retries_once_with_global_labels_when_initial_json_uses_full_video_timecodes(self):
        captured = {"messages": []}
        wrong = {
            "confidence": "high",
            "analysis": "Continue walking, then cut to the warning.",
            "timing_plan": "[Shot 4]: walking; [Shot 5]: the warning after the cut.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:09.167, Heman says: <d>[English] Stay back!</d>"
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
        self.assertIn('[Shot 5] At 00:00.667,', result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        retry_messages = captured["messages"][1]
        self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "user"])
        self.assertEqual(retry_messages[2]["content"], json.dumps(wrong))
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", retry_messages[3]["content"])
        self.assertTrue(captured["closed"])

    def test_reinforces_global_labels_and_local_times_until_marker_sequence_is_valid(self):
        captured = {"messages": []}
        wrong = {
            "confidence": "high",
            "analysis": "Finish source Shot 4 and then begin source Shot 5.",
            "timing_plan": "Finish walking, then make the one required cut.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:00.667, Heman keeps walking. "
                "[Shot 5] At 00:09.167, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        still_wrong = dict(wrong)
        corrected = dict(wrong)
        corrected["detailed_description"] = (
            "Heman continues walking right. "
            "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
        )

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

        class FakeLlama:
            def __init__(self, **_kwargs):
                self.responses = [wrong, still_wrong, corrected]

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
        self.assertEqual(len(result.attempts), 3)
        reinforcement = result.attempts[2].correction_prompt
        self.assertIn("H3 SHOT-MARKER REPAIR STILL REQUIRED", reinforcement)
        self.assertIn("The chunk time-slice encompasses the end of global Source Shot 4", reinforcement)
        self.assertIn(
            "global Source Shot 5 begins at chunk-local position 00:00.667",
            reinforcement,
        )
        self.assertIn("Delete every invented marker", reinforcement)
        self.assertTrue(captured["closed"])

    def test_missing_detailed_description_is_repaired_in_same_operation(self):
        captured = {"messages": []}
        malformed = {
            "confidence": "high",
            "analysis": "I identified the correct continuation but omitted the output field.",
            "timing_plan": "Continue walking, then make the required cut.",
            "end_state": "Heman has delivered the warning.",
        }
        corrected = {
            **malformed,
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
        }

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
            result = gemma4._observe_in_process(self.request(), [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[0].kind, "initial invalid response")
        self.assertIn("no detailed_description string", result.attempts[0].validation_warnings[0])
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", result.attempts[1].correction_prompt)
        self.assertIn("no detailed_description string", result.attempts[1].correction_prompt)
        self.assertEqual(len(captured["messages"]), 2)
        self.assertTrue(captured["closed"])

    def test_missing_detailed_description_can_need_multiple_bounded_repairs(self):
        malformed = {
            "confidence": "high",
            "analysis": "The response is still missing the required output field.",
            "timing_plan": "Continue walking, then make the cut.",
            "end_state": "Heman has delivered the warning.",
        }
        corrected = {
            **malformed,
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        appended = []

        class FakeHandler:
            def __init__(self, **_kwargs):
                self.responses = [malformed, corrected]

            def append_user_chat_completion(self, **kwargs):
                appended.append(kwargs["content"])
                return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **_kwargs):
                return {"choices": [{"message": {"content": json.dumps(malformed)}}]}

            def close(self):
                pass

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(self.request(), [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(len(appended), 2)
        self.assertTrue(all("no detailed_description string" in prompt for prompt in appended))
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", appended[0])
        self.assertIn("CHUNK JSON REPAIR STILL REQUIRED", appended[1])
        self.assertNotIn("Persistent last-seen character state contract", appended[1])

    def test_chunk_contract_correction_is_an_appended_chat_turn_when_runtime_supports_it(self):
        captured = {"initial_messages": [], "append_content": []}
        wrong = {
            "confidence": "high",
            "analysis": "Continue walking, then cut to the warning.",
            "timing_plan": "[Shot 4]: walking; [Shot 5]: the warning after the cut.",
            "end_state": "Heman has delivered the warning.",
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:09.167, Heman says: <d>[English] Stay back!</d>"
            ),
        }
        corrected = dict(wrong)
        corrected["detailed_description"] = wrong["detailed_description"].replace("00:09.167", "00:00.667")

        class FakeHandler:
            def __init__(self, **_kwargs):
                pass

            def append_user_chat_completion(self, **kwargs):
                captured["append_content"].append(kwargs["content"])
                return {"choices": [{"message": {"content": json.dumps(corrected)}}]}

        class FakeLlama:
            def __init__(self, **_kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                captured["initial_messages"].append(json.loads(json.dumps(kwargs["messages"])))
                return {"choices": [{"message": {"content": json.dumps(wrong)}}]}

            def close(self):
                captured["closed"] = True

        with patch.object(gemma4, "_load_runtime", return_value=(FakeLlama, FakeHandler)), \
                patch.object(gemma4, "_ensure_model_files", return_value=(Path("model"), Path("mmproj"))), \
                patch.object(gemma4.comfy.model_management, "soft_empty_cache"):
            result = gemma4._observe_in_process(self.request(), [], debug=False)

        self.assertEqual(result.detailed_description, corrected["detailed_description"])
        self.assertEqual(len(captured["initial_messages"]), 1)
        self.assertEqual(len(captured["append_content"]), 1)
        self.assertIn("CHUNK CONTRACT CORRECTION REQUIRED", captured["append_content"][0])
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
            "detailed_description": (
                "Heman continues the already established walk. "
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d> while continuing to walk right."
            ),
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
                "[Shot 5] At 00:00.667, Heman (<Subject 1>) (S1) says, "
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

    def test_system_preserves_only_source_camera_motion_without_enhancement(self):
        system = gemma4._gemma_prompt_templates()["SYSTEM"]
        planning_system = gemma4._gemma_prompt_templates()["PREPRODUCTION_SHOT_SYSTEM"]

        self.assertIn("Do not embellish, expand, enhance, improve", system)
        self.assertIn("never add a camera move, camera angle, camera setup", system)
        self.assertIn("Write a camera movement only when it is explicitly required", system)
        self.assertIn("The camera remains static in the established framing.", system)
        self.assertIn("every source-shot segment", system)
        self.assertNotIn("You may enhance visual", system)
        self.assertIn("`In a continuous movement,`", system)
        self.assertIn("never turn it into an undocumented cut", system)
        self.assertIn("Maintain `last_seen_character_state`", system)
        self.assertIn("remains off-screen", system)
        self.assertIn("never substitute a referenced animal", system)
        self.assertIn("Do not add camera movement.", planning_system)
        self.assertIn("`In a continuous movement,`", planning_system)

    def test_reports_marker_and_dialogue_warnings_without_replacing_gemma_text(self):
        request = self.request()
        value = {
            "confidence": "high",
            "analysis": "The dismount is complete; finish walking, then cut.",
            "timing_plan": "[Shot 4]: walk right now; [Shot 5]: defer dialogue until the cut.",
            "end_state": "Heman is walking right.",
            "detailed_description": (
                "Heman continues walking right. "
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
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

        state_request = dict(request)
        state_request["character_name_table"] = "- Heman -> <Subject 1>"
        state_request["previous_last_seen_character_state"] = []
        state_value = dict(value)
        state_value["last_seen_character_state"] = [{
            "character_name": "Heman",
            "subject": "<Subject 1>",
            "last_seen_global_frame": 208,
            "last_seen_source_shot": 4,
            "environment": "inside the ancient temple",
            "pose_and_position": "standing on the temple floor",
            "state_and_action": "walking toward the right",
            "spatial_relationships": "away from the mounted riders",
        }]
        state_result = gemma4._validate_chunk_prompt(
            state_value,
            state_request,
            json.dumps(state_value),
        )
        self.assertEqual(state_result.last_seen_character_state[0]["character_name"], "Heman")
        self.assertEqual(state_result.last_seen_character_state[0]["last_seen_global_frame"], 208)
        self.assertFalse(any("last-seen character state" in warning for warning in state_result.validation_warnings))
        self.assertEqual(
            gemma4._chunk_prompt_from_payload(gemma4._chunk_prompt_payload(state_result)),
            state_result,
        )

        missing_state = gemma4._validate_chunk_prompt(value, state_request, json.dumps(value))
        self.assertIn(
            "Gemma 4 last-seen character state must be an array",
            missing_state.validation_warnings,
        )
        self.assertIn(
            "Persistent last-seen character state contract",
            gemma4._chunk_contract_correction_request(
                state_request,
                gemma4._contract_validation_warnings(missing_state.validation_warnings),
            ),
        )

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
        missing_marker["detailed_description"] = missing_marker["detailed_description"].replace("[Shot 5]", "Shot 5")
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
                "[Shot 5] At 00:00.667, Heman says: <d>[English] Stay back!</d>"
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
