from __future__ import annotations

import hashlib
import importlib
import inspect
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
sys.path.insert(0, str(PLUGIN_ROOT.parent))

nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")
preview = importlib.import_module(PLUGIN_ROOT.name + ".preview")


class _FakeVAE:
    def decode(self, _latent):
        return torch.zeros((39, 2, 2, 3), dtype=torch.float32)


class _IndexedFakeVAE:
    def decode(self, _latent):
        return torch.arange(39, dtype=torch.float32).reshape(39, 1, 1, 1).expand(39, 2, 2, 3)


class ChunkDirectorHelperTest(unittest.TestCase):
    def test_hr_endless_sampler_schema_hides_retired_experiments_and_puts_debug_last(self):
        schema = nodes.HREndlessSampler.define_schema()
        input_ids = [item.id for item in schema.inputs]

        self.assertEqual(schema.node_id, "HREndlessSampler")
        self.assertEqual(schema.display_name, "HR Endless Sampler")
        self.assertIn("video_continuation", input_ids)
        self.assertIn("director_backend", input_ids)
        self.assertIn("director_model", input_ids)
        self.assertIn("director_mmproj", input_ids)
        self.assertIn("cache_gemma_preproduction", input_ids)
        self.assertIn("gemma4_mtp", input_ids)
        self.assertNotIn("video_continuation_enable", input_ids)
        self.assertFalse({
            "context_keyframes_enable",
            "context_keyframes",
            "guide_overlap_enable",
            "guide_overlap",
            "qwen_full_history",
            "prompt_preview_only",
        } & set(input_ids))
        self.assertEqual(
            input_ids[-8:-3],
            ["cache_gemma_preproduction", "gemma4_mtp", "debug", "debug_stop_chunk", "debug_start_chunk"],
        )
        self.assertEqual(input_ids[-3:], ["director_backend", "director_model", "director_mmproj"])

        execute_params = inspect.signature(nodes.HREndlessSampler.execute).parameters
        self.assertNotIn("video_continuation_enable", execute_params)
        self.assertNotIn("context_keyframes_enable", execute_params)
        self.assertNotIn("guide_overlap_enable", execute_params)
        self.assertNotIn("qwen_full_history", execute_params)
        self.assertNotIn("prompt_preview_only", execute_params)
        self.assertIn("debug_start_chunk", execute_params)
        self.assertIn("cache_gemma_preproduction", execute_params)
        self.assertIn("gemma4_mtp", execute_params)
        parameter_ids = list(execute_params)
        self.assertLess(parameter_ids.index("debug_start_chunk"), parameter_ids.index("director_backend"))
        self.assertEqual(parameter_ids[-4:-1], ["director_backend", "director_model", "director_mmproj"])

    def test_last_run_replay_cache_keeps_exact_cpu_tensors_and_truncates_replayed_suffix(self):
        video = torch.arange(24, dtype=torch.float16).reshape(1, 1, 2, 3, 4)
        audio = torch.arange(16, dtype=torch.float16).reshape(1, 1, 2, 8)
        plan = [{
            "frame_start": 0,
            "frame_end": 5,
            "video_start": 0,
            "video_end": 2,
            "audio_start": 0,
            "audio_end": 8,
            "context_video_t": 0,
            "context_audio_t": 0,
            "output_trim_frames": 0,
            "synthetic_prefix": False,
        }]
        fingerprint = nodes._replay_fingerprint(
            video,
            audio,
            plan,
            fps=24.0,
            chunk_frames=5,
            context_keyframes=0,
            guide_overlap=0,
            video_continuation=0,
            ref2va=False,
        )
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root):
            cache = nodes._LastRunReplayCache()
            cache.create(
                fingerprint,
                "original prompt",
                {
                    "video": video,
                    "audio": audio,
                    "video_noise": video + 1,
                    "audio_noise": audio + 1,
                    "noise_seed": 123,
                },
            )
            cache.save_chunk(1, {
                "sampled_video": video,
                "sampled_audio": audio,
                "previous_frame_count": 5,
                "output_video": video,
                "output_audio": audio,
                "denoised_video": video,
                "denoised_audio": audio,
                "output_template": {"batch_index": torch.tensor([0])},
                "denoised_template": {},
                "gemma_description": "directed prompt",
                "gemma_timing_plan": "timing",
                "gemma_end_state": "end",
                "debug_prompt": "chunk debug",
                "prefix_video_noise": None,
                "prefix_audio_noise": None,
            })
            loaded, reason = cache.load_if_compatible(fingerprint)
            self.assertIsNone(reason)
            self.assertEqual(loaded["initial"]["noise_seed"], 123)
            self.assertEqual(loaded["initial"]["video"].device.type, "cpu")
            self.assertTrue(torch.equal(cache.load_chunk(1)["sampled_video"], video.cpu()))
            self.assertIsNone(cache.load_if_compatible({"different": True})[0])
            cache.truncate_from(1)
            self.assertFalse(cache.has_chunk(1))

    def test_rebuilt_replay_timing_plan_updates_the_source_prompt_hash(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes, "_timing_plan_payload", return_value={"plan": "rebuilt"}):
            cache = nodes._LastRunReplayCache()
            cache.create({"geometry": "stable"}, "old source prompt", {"video": torch.zeros(1)})
            cache.save_timing_plan(object(), source_prompt="edited source prompt")

            manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            timing = json.loads(cache.timing_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["source_prompt_sha256"],
                hashlib.sha256(b"edited source prompt").hexdigest(),
            )
            self.assertEqual(timing, {"plan": "rebuilt"})

    def test_preview_chunk_metadata_survives_a_server_state_restore(self):
        node_id = "preview-tooltip-test"
        with preview._PREVIEW_CACHE_LOCK:
            preview._PREVIEW_CACHE.pop(node_id, None)
        try:
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "reset",
                "chunk_ranges": [
                    {"chunk": 1, "start": 0, "end": 38},
                    {"chunk": 2, "start": 39, "end": 72},
                ],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "phase",
                "phase": "Gemma 4 is planning 4 source shots before H3 sampling",
                "chunk": 0,
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "chunk_metadata",
                "chunk": 0,
                "gemma_detailed_description": "  [Shot 1] The tiger runs.  ",
                "h3_render_seconds": 42.5,
                "gemma_seconds": 3.25,
                "gemma_preproduction_seconds": 2.0,
                "chunk_total_seconds": 48.75,
            })
            snapshot = preview._cached_snapshot(node_id)
            self.assertEqual(
                snapshot["reset"]["chunk_ranges"][0]["gemma_detailed_description"],
                "[Shot 1] The tiger runs.",
            )
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["h3_render_seconds"], 42.5)
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["gemma_seconds"], 3.25)
            self.assertEqual(
                snapshot["reset"]["chunk_ranges"][0]["gemma_preproduction_seconds"],
                2.0,
            )
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["chunk_total_seconds"], 48.75)
            self.assertEqual(
                snapshot["phase"]["phase"],
                "Gemma 4 is planning 4 source shots before H3 sampling",
            )
            self.assertNotIn("gemma_detailed_description", snapshot["reset"]["chunk_ranges"][1])
        finally:
            with preview._PREVIEW_CACHE_LOCK:
                preview._PREVIEW_CACHE.pop(node_id, None)

    def test_preparation_progress_reports_to_console_and_preview(self):
        phases = []

        class FakePreview:
            def set_phase(self, phase, *, chunk=None):
                phases.append((phase, chunk))

        with patch.object(nodes.logging, "info") as logged:
            with nodes._PreparationProgress(
                "Gemma 4 is planning source shots",
                FakePreview(),
                chunk=2,
                interval=60,
            ):
                pass

        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0][1], 2)
        self.assertIn("still working", phases[0][0])
        self.assertIn("complete", phases[1][0])
        self.assertEqual(logged.call_count, 2)

    def test_gemma_directing_preparation_uses_an_indeterminate_live_tqdm_bar(self):
        bars = []

        class FakeBar:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.n = 0
                self.disable = False
                self.refresh_count = 0
                self.closed = False

            def refresh(self):
                self.refresh_count += 1

            def close(self):
                self.closed = True

        def fake_tqdm(**kwargs):
            bar = FakeBar(**kwargs)
            bars.append(bar)
            return bar

        with patch.object(nodes, "tqdm", fake_tqdm), patch.object(nodes.logging, "info"):
            with nodes._PreparationProgress(
                "Chunk 4/10: Gemma 4 is directing the chunk prompt",
                interval=60,
                live_console_bar=True,
            ):
                pass

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].unit, "gemma")
        self.assertEqual(bars[0].total, 30)
        self.assertGreaterEqual(bars[0].refresh_count, 2)
        self.assertTrue(bars[0].closed)

    def test_gemma_progress_displays_live_decode_tokens_per_second(self):
        phases = []

        class FakePreview:
            def set_phase(self, phase, *, chunk=None):
                phases.append((phase, chunk))

        class FakeBar:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.n = 0
                self.disable = False
                self.postfix = ""

            def set_postfix_str(self, postfix, refresh=False):
                self.postfix = postfix

            def refresh(self):
                pass

            def close(self):
                pass

        with patch.object(nodes, "tqdm", lambda **kwargs: FakeBar(**kwargs)), \
                patch.object(nodes.logging, "info"):
            with nodes._PreparationProgress(
                "Chunk 2/8: Gemma 4 is directing the chunk prompt",
                FakePreview(),
                chunk=1,
                interval=60,
                live_console_bar=True,
            ) as progress:
                progress.update_token_progress(96, 127.45, 1)
                self.assertIn("96 tokens, 127.5 tokens/sec", progress._bar.postfix)

        self.assertTrue(any("127.5 tokens/sec" in phase for phase, _chunk in phases))

    def test_timing_preproduction_is_logged_before_chunk_transcripts(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes.logging, "info"):
            path = nodes._begin_last_gemma_prompt_log(39, 0, 0, 22, 24.0, 3)
            nodes._append_gemma_timing_plan(
                path,
                "Source Shot 1: immutable preproduction timing schedule",
                system_prompt="preproduction system",
                planning_prompt="preproduction request",
                gemma_response='{"shots": []}',
            )
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 1 ===",
                "first H3 prompt",
                observation_prompt="first Gemma request",
                gemma_response='{"detailed_description":"first response"}',
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("=== GEMMA PREPRODUCTION SYSTEM PROMPT ===", content)
            self.assertIn("=== GEMMA SHOT TIMING PREPRODUCTION ===", content)
            self.assertLess(content.index("preproduction request"), content.index("=== Chunk 1 ==="))
            self.assertLess(content.index("Source Shot 1: immutable"), content.index("=== Chunk 1 ==="))

    def test_last_gemma_prompt_log_is_replaced_then_flushed_per_chunk(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes.logging, "info"):
            path = nodes._begin_last_gemma_prompt_log(39, 0, 0, 22, 24.0, 3)
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 1 ===",
                "first H3 prompt",
                system_prompt="system instructions",
                observation_prompt="first Gemma request",
                gemma_response='{"detailed_description":"first response"}',
            )
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 2 ===",
                "second H3 prompt",
                observation_prompt="second Gemma request",
                gemma_response='{"detailed_description":"second response"}',
                validation_warnings=("Gemma 4 returned 2 shot markers; this chunk requires 3",),
            )

            self.assertEqual(path.name, nodes.GEMMA_PROMPT_LOG_FILENAME)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Configuration: chunk_frames=39, context_keyframes=0", content)
            self.assertEqual(content.count("=== GEMMA SYSTEM PROMPT ==="), 1)
            self.assertIn("=== GEMMA SYSTEM PROMPT ===\nsystem instructions", content)
            separator = "=" * 200
            self.assertIn(f"{separator}\n=== Chunk 1 ===", content)
            self.assertIn(f"{separator}\n=== Chunk 2 ===", content)
            self.assertLess(content.index("first Gemma request"), content.index("first response"))
            self.assertLess(content.index("first response"), content.index("first H3 prompt"))
            self.assertLess(content.index("second Gemma request"), content.index("second response"))
            self.assertLess(content.index("second response"), content.index("second H3 prompt"))
            self.assertIn("=== GEMMA VALIDATION WARNINGS ===", content)
            self.assertIn("Gemma 4 returned 2 shot markers; this chunk requires 3", content)

            replacement = nodes._begin_last_gemma_prompt_log(56, 5, 0, 0, 24.0, 2)
            replacement_content = replacement.read_text(encoding="utf-8")
            self.assertIn("Configuration: chunk_frames=56, context_keyframes=5", replacement_content)
            self.assertNotIn("first Gemma request", replacement_content)

            image_directory = nodes._reset_last_gemma_image_log()
            stale_image = image_directory / "stale.jpg"
            stale_image.write_bytes(b"stale")
            replacement_image_directory = nodes._reset_last_gemma_image_log()
            self.assertEqual(replacement_image_directory, image_directory)
            self.assertFalse(stale_image.exists())

    def test_observation_samples_only_retained_previous_output(self):
        frames, indices = nodes._decoded_video_frames(
            _FakeVAE(),
            object(),
            include_final=True,
            start_frame=5,
        )
        self.assertEqual(indices, [5, 17, 29, 38])
        self.assertEqual(frames.shape[0], 4)

    def test_observation_can_return_exact_full_resolution_final_frame(self):
        frames, indices, final_frame = nodes._decoded_video_frames(
            _IndexedFakeVAE(),
            object(),
            include_final=True,
            start_frame=5,
            return_final_frame=True,
        )
        self.assertEqual(indices, [5, 17, 29, 38])
        self.assertEqual(frames.shape[0], 4)
        self.assertEqual(final_frame.shape, (1, 2, 2, 3))
        self.assertTrue(torch.all(final_frame == 38))
        self.assertEqual(final_frame.device.type, "cpu")

    def test_target_shots_omit_opening_marker_for_mid_shot_and_keep_real_local_cut(self):
        shots = [
            (0, 0, 68, "Tiger approaches the temple."),
            (1, 68, 117, "Tiger enters the temple and stops."),
        ]
        records = nodes._gemma_shot_records(
            shots,
            range_start=50,
            range_end=89,
            sampled_start=50,
            fps=24.0,
            target=True,
        )
        self.assertEqual([record["source_body"] for record in records], [shot[3] for shot in shots])
        self.assertIsNone(records[0]["required_marker"])
        self.assertEqual(records[1]["required_marker"], "[Shot 2] At 00:00.750,")

    def test_target_shots_use_opening_marker_only_at_real_physical_shot_start(self):
        shots = [
            (0, 0, 68, "Tiger approaches the temple."),
            (1, 68, 117, "Tiger enters the temple and stops."),
        ]
        at_real_start = nodes._gemma_shot_records(
            shots,
            range_start=68,
            range_end=107,
            sampled_start=68,
            fps=24.0,
            target=True,
        )
        self.assertEqual(at_real_start[0]["required_marker"], "[Shot 1]")

        after_carried_prefix = nodes._gemma_shot_records(
            shots,
            range_start=68,
            range_end=107,
            sampled_start=63,
            fps=24.0,
            target=True,
        )
        self.assertEqual(after_carried_prefix[0]["required_marker"], "[Shot 2] At 00:00.208,")

    def test_prompt_preview_omits_shot_one_for_mid_shot_continuation(self):
        prompt = (
            "detailed_description: [Shot 1] The tiger runs through the jungle. "
            "[Shot 2] At 00:02.833, The tiger enters the temple."
        )
        rewritten = nodes._prompt_for_chunk(
            prompt,
            frame_start=34,
            frame_end=73,
            total_frames=73,
            fps=24.0,
            content_start=39,
            continuation=True,
        )
        description = rewritten.split("detailed_description:", 1)[1]
        self.assertNotIn("[Shot 1]", description)
        self.assertIn("[Shot 2] At 00:01.417,", description)

        boundary = nodes._prompt_for_chunk(
            prompt,
            frame_start=0,
            frame_end=39,
            total_frames=73,
            fps=24.0,
        )
        self.assertIn("[Shot 1]", boundary)

    def test_keyframes_use_truthful_physical_overlap_geometry(self):
        total_frames = 56
        plan = nodes._chunk_plan(
            nodes._video_steps(total_frames),
            nodes._audio_steps(total_frames),
            39,
            22,
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual((plan[1]["frame_start"], plan[1]["frame_end"]), (17, 56))
        self.assertEqual(plan[1]["output_trim_frames"], 22)
        self.assertEqual(plan[1]["context_video_t"], nodes._video_steps(22))
        self.assertNotIn("synthetic_prefix", plan[1])
        keyframes, guide, video, keyframe_t = nodes._continuation_controls(22, 0, 0, 39)
        self.assertEqual((keyframes, guide, video, keyframe_t), (22, 0, 0, nodes._video_steps(22)))

    def test_keyframe_overlap_must_be_smaller_than_chunk(self):
        with self.assertRaisesRegex(ValueError, "must be smaller"):
            nodes._continuation_controls(39, 0, 0, 39)

    def test_video_continuation_allows_equal_chunk_and_clamps_a_larger_request(self):
        self.assertEqual(
            nodes._continuation_controls(0, 0, 22, 22)[:3],
            (0, 0, 22),
        )
        self.assertEqual(
            nodes._continuation_controls(0, 0, 56, 22)[:3],
            (0, 0, 22),
        )

    def test_video1_boundary_keyframe_uses_complete_discarded_packing_prefix(self):
        plan = nodes._chunk_plan_without_overlap(
            nodes._video_steps(73),
            nodes._audio_steps(73),
            39,
        )
        previous = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1, 1)
        boundary_latent, boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 0, True)
        self.assertEqual(boundary_start, 0)
        self.assertEqual(boundary_latent.shape[2], nodes._video_steps(5))
        self.assertTrue(torch.equal(boundary_latent, previous[:, :, -2:]))
        self.assertNotEqual(boundary_latent.data_ptr(), previous.data_ptr())

        no_boundary, no_boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 22, True)
        self.assertIsNone(no_boundary)
        self.assertEqual(no_boundary_start, 0)
        no_boundary, no_boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 0, False)
        self.assertIsNone(no_boundary)
        self.assertEqual(no_boundary_start, 0)

        boundary_latent = torch.zeros((1, 24, 2, 2, 2), dtype=torch.float32)
        conds = nodes._conditioning_for_chunk(
            {"positive": [{"minimax_keyframes": []}]},
            34,
            73,
            (torch.zeros((1, 1, 1)), {}),
            video_context=boundary_latent,
            video_context_start=0,
        )
        keyframe = conds["positive"][0]["minimax_keyframes"][0]
        self.assertEqual(keyframe["resolved_frame_index"], 0)
        self.assertIs(keyframe["latent"], boundary_latent)

    def test_debug_redraw_leaves_sampling_steps_after_chunk_bar(self):
        refreshed = []

        class FakeBar:
            disable = False

            def __init__(self, unit):
                self.unit = unit

            def refresh(self):
                refreshed.append(self.unit)

        class FakeTqdm:
            _instances = (FakeBar("steps"), FakeBar("chunk"))

        with patch.object(nodes, "tqdm", FakeTqdm), patch.object(nodes, "_cli_tqdm", FakeTqdm):
            nodes._refresh_console_progress()

        self.assertEqual(refreshed, ["chunk", "steps"])

    def test_peak_time_interpolates_time_closer_to_peak_than_average(self):
        gib = 1024 ** 3
        with patch.object(nodes, "_memory_backend", return_value=None):
            timing = nodes._SamplerTiming(torch.device("cpu"))

        timing._snapshot_count = 4
        timing._snapshot_sum = 4 * 8 * gib
        timing.max_device_used = 14 * gib
        timing.device_total = 16 * gib
        timing._physical_samples = [
            (0.0, 8 * gib),
            (10.0, 14 * gib),
            (20.0, 14 * gib),
            (30.0, 8 * gib),
        ]

        physical = timing._physical_summary()

        self.assertEqual(physical["threshold"], 11 * gib)
        self.assertAlmostEqual(physical["peak_time"], 20.0)

    def test_non_debug_monitor_still_collects_report_samples(self):
        class FakeTiming:
            def __init__(self):
                self.observations = []

            def observe_memory(self, **kwargs):
                self.observations.append(kwargs)

        timing = FakeTiming()
        monitor = nodes._VRAMMonitor(timing, torch.device("cpu"), [], 1, debug=False)
        with patch.object(nodes, "_vram_report") as detailed_report:
            monitor.report("sampling", sample_group="dit")

        self.assertEqual(timing.observations, [{"sample_group": "dit", "chunk_index": 0}])
        detailed_report.assert_not_called()

    def test_always_on_report_puts_peak_time_directly_after_peak(self):
        gib = 1024 ** 3
        with patch.object(nodes, "_memory_backend", return_value=None):
            timing = nodes._SamplerTiming(torch.device("cpu"))
        timing.started -= 30.0
        timing._snapshot_count = 4
        timing._snapshot_sum = 4 * 8 * gib
        timing._dit_snapshot_count = 2
        timing._dit_snapshot_sum = 2 * 10 * gib
        timing._later_dit_snapshot_count = 1
        timing._later_dit_snapshot_sum = 10 * gib
        timing.max_device_used = 14 * gib
        timing.device_total = 16 * gib
        timing._physical_samples = [
            (0.0, 8 * gib),
            (10.0, 14 * gib),
            (20.0, 14 * gib),
            (30.0, 8 * gib),
        ]
        timing.chunk_seconds = {0: 15.0, 1: 15.0}
        timing.seconds["h3_sampling"] = 20.0
        timing.calls["h3_sampling"] = 2
        run = {
            "chunk_frames": 39,
            "context_keyframes": 5,
            "guide_overlap": 0,
            "video_continuation": 22,
            "width": 864,
            "height": 480,
            "sampling_steps": 20,
            "rendered_frames": 73,
            "full_frames": 73,
            "full_chunks": 2,
        }

        with patch.object(nodes.logging, "info") as log_info:
            timing.report("complete", 2, run)

        report = log_info.call_args.args[0]
        self.assertIn("HR Endless Sampler run report:", report)
        self.assertIn("Baseline from this run:", report)
        self.assertIn("VRAM baseline:", report)
        self.assertIn("  Peak Time: 20.00s (VRAM closer to Peak than Average; above 11.00 GiB)", report)
        self.assertIn("Time baseline:", report)
        self.assertIn("Breakdown:", report)
        peak_line = report.index("  Peak:")
        peak_time_line = report.index("  Peak Time:")
        torch_line = report.index("  PyTorch VRAM high-water:")
        self.assertLess(peak_line, peak_time_line)
        self.assertLess(peak_time_line, torch_line)


if __name__ == "__main__":
    unittest.main()
