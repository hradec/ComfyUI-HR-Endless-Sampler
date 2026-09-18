import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import comfy.nested_tensor


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hr_video_bridge_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package
MODULE = importlib.import_module(PACKAGE + ".video_bridge")


class VideoBridgeExtractTests(unittest.TestCase):
    def frames(self, count):
        return torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1).expand(count, 4, 6, 3)

    def test_extracts_exact_a_tail_and_b_head(self):
        output = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(40), 24.0, 24.0)
        source, a_tail, b_head, *_rest = output.result
        self.assertEqual(tuple(a_tail.shape), (22, 4, 6, 3))
        self.assertEqual(tuple(b_head.shape), (22, 4, 6, 3))
        self.assertEqual(a_tail[0, 0, 0, 0].item(), 8)
        self.assertEqual(b_head[-1, 0, 0, 0].item(), 21)
        self.assertIs(MODULE.normalize_bridge_source(source), source)

    def test_rejects_short_video_and_fps_mismatch(self):
        with self.assertRaisesRegex(ValueError, "at least 22"):
            MODULE.HRVideoBridgeExtract.execute(self.frames(21), self.frames(22), 24.0, 24.0)
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 30.0)

    def test_resamples_selected_video_to_common_fps(self):
        output = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 30.0, "resample_b_to_a"
        )
        source = output.result[0]
        self.assertEqual(source["fps"], 24.0)
        self.assertEqual(source["source_a_frame_count"], 30)
        self.assertEqual(source["source_b_frame_count"], 24)

    def test_extracts_synchronized_audio_windows(self):
        waveform = torch.arange(48000, dtype=torch.float32).reshape(1, 1, 48000)
        audio = {"waveform": waveform, "sample_rate": 48000}
        output = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 24.0, audio_a=audio, audio_b=audio
        )
        a_audio, b_audio = output.result[3:5]
        expected = round(22 * 48000 / 24)
        self.assertEqual(a_audio["waveform"].shape[-1], expected)
        self.assertEqual(b_audio["waveform"].shape[-1], expected)
        self.assertEqual(a_audio["waveform"][0, 0, 0].item(), 48000 - expected)
        self.assertEqual(b_audio["waveform"][0, 0, -1].item(), expected - 1)

    def test_director_uses_only_qwen36_38_bridge_operation(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        config = {"version": 1, "backend": "qwen3.8", "model": "model.gguf", "mmproj": "mmproj.gguf",
                  "mtp": False, "mtp_draft_tokens": 2, "reasoning_effort": "medium",
                  "cpu_moe": False, "n_cpu_moe": 0, "debug": False}
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        selection = types.SimpleNamespace(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        response = {"ok": True, "video_bridge": {"version": 1, "transition_frames": 39,
                    "strategy": "match_cut", "analysis": {}, "constraints": ["identity"],
                    "h3_prompt": "subject_definitions: x", "risk_report": "low",
                    "reference_labels": {"pictures": [], "video_a": "<Video 2>", "video_b": "<Video 1>"}}}
        captured = {}
        def worker(request, timeout):
            captured.update(request)
            return types.SimpleNamespace(stderr="", stdout=""), response
        with patch.object(MODULE, "resolve_director_selection", return_value=selection), \
             patch.object(MODULE, "_run_worker_once", side_effect=worker), \
             patch.object(MODULE.comfy.model_management, "unload_all_models"), \
             patch.object(MODULE.comfy.model_management, "soft_empty_cache"):
            output = MODULE.HRVideoBridgeDirector.execute(source, refs, config)
        self.assertEqual(captured["operation"], "video_bridge")
        self.assertEqual(captured["director_backend"], "qwen3.8")
        self.assertEqual(len(captured["image_urls"]), 44)
        self.assertEqual(output.result[0]["type"], "HR_VIDEO_BRIDGE_PLAN")
        self.assertEqual(output.result[0]["reference_labels"]["video_a"], "<Video 2>")
        self.assertEqual(output.result[0]["reference_labels"]["video_b"], "<Video 1>")

    def test_director_rejects_qwen35_without_launching_worker(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        config = {"version": 1, "backend": "qwen3.5", "model": "auto", "mmproj": "auto",
                  "mtp": False, "mtp_draft_tokens": 2, "reasoning_effort": "medium",
                  "cpu_moe": False, "n_cpu_moe": 0, "debug": False}
        with self.assertRaisesRegex(ValueError, "only.*Qwen3.6/3.8"):
            MODULE.HRVideoBridgeDirector.execute(source, refs, config)

    def test_conditioning_preserves_b_reference_and_adds_a_start_anchor(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        plan = {"type": "HR_VIDEO_BRIDGE_PLAN", "version": 1, "transition_frames": 39,
                "h3_prompt": "subject_definitions: x", "source": source, "analysis": {}}
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        embedding = torch.ones(1, 2, 3)
        b_refs = [{"kind": "video", "latent": torch.zeros(1, 24, 7, 4, 6)}]
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, 12, 4, 6), torch.zeros(1, 32, 2, 65)
        ))}
        conditioned = types.SimpleNamespace(result=([[embedding, {"minimax_refs": b_refs}]], latent, refs))
        fake_node = types.SimpleNamespace(execute=lambda *args, **kwargs: conditioned)
        vae = types.SimpleNamespace(encode=lambda frames: torch.zeros(1, 24, 7, frames.shape[1] // 16, frames.shape[2] // 16))
        with patch.object(MODULE, "HRMiniMaxH3ReferenceConditioning", fake_node):
            output = MODULE.HRVideoBridgeConditioning.execute(None, vae, None, plan, refs, 96, 64, "mute")
        positive, result_latent, continuation, *_ = output.result
        self.assertIs(result_latent, latent)
        self.assertIs(positive[0][1]["minimax_refs"], b_refs)
        self.assertEqual(positive[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0)
        self.assertEqual(continuation["type"], "HR_H3_EXTERNAL_CONTINUATION")
        self.assertEqual(continuation["target_frames"], 39)

    def test_hard_cut_assembly_preserves_all_frames_and_audio_duration(self):
        audio = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        source = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 24.0, audio_a=audio, audio_b=audio
        ).result[0]
        bridge = self.frames(22)
        bridge_audio = {"waveform": torch.ones(1, 1, 44000), "sample_rate": 48000}
        output = MODULE.HRVideoBridgeAssemble.execute(
            source, bridge, bridge_audio, assemble_policy="hard_cut_debug", fade_seconds=0
        )
        frames, result_audio, timeline, report = output.result
        self.assertEqual(frames.shape[0], 82)
        self.assertEqual(timeline["total_frames"], 82)
        self.assertEqual(result_audio["waveform"].shape[-1], round(82 * 48000 / 24))
        self.assertEqual(len(timeline["segments"]), 3)
        self.assertEqual(len(timeline["seams"]), 2)
        self.assertEqual(__import__("json").loads(report)["output_frames"], 82)


if __name__ == "__main__":
    unittest.main()
