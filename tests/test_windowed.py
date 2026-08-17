import importlib.util
import pathlib
import sys
import unittest

import torch

import comfy.cli_args
import comfy.conds
import comfy.utils


comfy.cli_args.args.cpu = True
comfy.utils.PROGRESS_BAR_ENABLED = False


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_unlimited_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)

from h3_unlimited_test.nodes import _audio_steps, _chunk_plan, _pixel_frames, _prompt_for_window
from h3_unlimited_test.windowed import MiniMaxH3WindowedContextHandler, WINDOW_INDEX_KEY, _pixel_start


class WindowedSamplingTest(unittest.TestCase):
    def test_planner_covers_both_modalities(self):
        for video_t in range(2, 208, 5):
            audio_t = _audio_steps(_pixel_frames(video_t))
            for chunk_frames in (22, 39, 56, 73, 124, 243):
                plan = _chunk_plan(video_t, audio_t, chunk_frames)
                video_coverage = [0] * video_t
                audio_coverage = [0] * audio_t
                for item in plan:
                    self.assertEqual(item["video_start"] % 5, 0)
                    for index in range(item["video_start"], item["video_end"]):
                        video_coverage[index] += 1
                    for index in range(item["audio_start"], item["audio_end"]):
                        audio_coverage[index] += 1
                self.assertTrue(all(count > 0 for count in video_coverage))
                self.assertTrue(all(count > 0 for count in audio_coverage))

    def test_window_fusion_and_global_positions(self):
        video = torch.randn(1, 24, 12, 4, 4)
        audio = torch.randn(1, 32, 2, 65)
        packed, shapes = comfy.utils.pack_latents((video, audio))
        plan = _chunk_plan(12, 65, 22)
        self.assertEqual(
            [(item["video_start"], item["video_end"], item["audio_start"], item["audio_end"]) for item in plan],
            [(0, 7, 0, 37), (5, 12, 28, 65)],
        )

        active_windows = []
        handler = MiniMaxH3WindowedContextHandler(plan, shapes, active_windows.append)
        self.assertEqual(handler.noise_shape(packed.shape), [1, 1, 5056])
        conds = [[{
            WINDOW_INDEX_KEY: index,
            "model_conds": {
                "c_crossattn": comfy.conds.CONDRegular(torch.zeros(1, 4, 8)),
                "minimax_payload": comfy.conds.CONDConstant({}),
                "latent_shapes": comfy.conds.CONDConstant(shapes),
            },
        } for index in range(2)]]
        seen = []

        def calculate(_model, selected, sub_x, _timestep, _model_options):
            condition = selected[0][0]
            layout = condition["model_conds"]["minimax_payload"].cond["layout"]
            seen.append((condition[WINDOW_INDEX_KEY], condition["model_conds"]["latent_shapes"].cond, layout))
            return [sub_x]

        sentinel = object()
        model_options = {"transformer_options": {"context_window": sentinel}}
        result = handler.execute(calculate, None, conds, packed, torch.ones(1), model_options)[0]
        handler.close()
        self.assertTrue(torch.allclose(result, packed))
        self.assertIs(model_options["transformer_options"]["context_window"], sentinel)
        self.assertEqual([item[0] for item in seen], [0, 1])
        self.assertEqual(active_windows, [0, 1])
        self.assertEqual(seen[1][1], [video[:, :, 5:12].shape, audio[..., 28:65].shape])

        layout = seen[1][2]
        video_start = next(start for start, _end, kind in layout.segments if kind == "video")
        audio_start = next(start for start, _end, kind in layout.segments if kind == "audio")
        self.assertAlmostEqual(
            float(layout.position_ids[video_start, 0]),
            4 + (5.0 / 3.0) * _pixel_start(5),
        )
        self.assertEqual(float(layout.position_ids[audio_start, 0]), 4 + 28)

    def test_cut_keeps_its_global_time(self):
        prompt = (
            "detailed_description: [Shot 1] The subject walks forward. "
            "[Shot 2] At 00:04.167, the camera cuts to a close-up.\n"
            "overall_soundscape: room tone"
        )
        rewritten = _prompt_for_window(prompt, 50, 150, 356, 24.0, continuation=True)
        self.assertIn("[Shot 2] At 00:04.167,", rewritten)
        self.assertIn("Continue the already established shot", rewritten)


if __name__ == "__main__":
    unittest.main()
