from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")


class _FakeVAE:
    def decode(self, _latent):
        return torch.zeros((39, 2, 2, 3), dtype=torch.float32)


class _IndexedFakeVAE:
    def decode(self, _latent):
        return torch.arange(39, dtype=torch.float32).reshape(39, 1, 1, 1).expand(39, 2, 2, 3)


class ChunkDirectorHelperTest(unittest.TestCase):
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

    def test_target_shots_keep_complete_bodies_and_exact_local_cut(self):
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
        self.assertEqual(records[0]["required_marker"], "[Shot 1]")
        self.assertEqual(records[1]["required_marker"], "[Shot 2] At 00:00.750,")

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
        self.assertIn("SamplerCustomAdvanced-Unlimited run report:", report)
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
