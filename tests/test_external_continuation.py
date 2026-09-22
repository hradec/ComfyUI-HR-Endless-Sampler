import importlib
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import comfy.nested_tensor

MODULE = importlib.import_module("custom_nodes.ComfyUI-MiniMax-H3-Sampler-Unlimited.external_continuation")
QWEN = importlib.import_module("custom_nodes.ComfyUI-MiniMax-H3-Sampler-Unlimited.qwen35")


class FakeVAE:
    def encode(self, images):
        return torch.zeros(1, 24, 7, images.shape[1] // 16, images.shape[2] // 16)


class FakeAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.zeros(1, 32, 2, max(1, waveform.shape[-1] // 256))


def latent():
    return {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 7, 4, 6), torch.zeros(1, 32, 2, 37),
    ))}


class ExternalContinuationTests(unittest.TestCase):
    def test_tail_uses_h3_aligned_final_frames(self):
        images = torch.arange(30).view(30, 1, 1, 1).expand(30, 2, 2, 3)
        self.assertEqual(MODULE._tail_frames(images, 22).shape[0], 22)
        self.assertEqual(MODULE._tail_frames(images[:21], 22).shape[0], 5)
        with self.assertRaisesRegex(ValueError, "at least 5"):
            MODULE._tail_frames(images[:4], 22)

    def test_apply_preserves_embedding_and_h3_metadata(self):
        embedding = torch.ones(1, 2, 3)
        refs = [{"kind": "image", "latent": torch.zeros(1)}]
        tags = {"token": [1]}
        positive = [[embedding, {"cross_attn": embedding, "minimax_refs": refs,
                                "minimax_token_tags": tags}]]
        context = {"type": "HR_H3_EXTERNAL_CONTINUATION_CONTEXT", "version": 1,
                   "tail_images": torch.zeros(22, 64, 96, 3), "source_audio_tail": None,
                   "prompt": "[Shot 1] Continue.", "analysis": {}}
        latent_in = latent()
        out = MODULE.HRMiniMaxH3VideoContinuationApply.execute(
            positive, latent_in, context, FakeVAE())
        result_positive, result_latent, continuation = out.result
        metadata = result_positive[0][1]
        self.assertIs(result_positive[0][0], embedding)
        self.assertIs(metadata["minimax_refs"], refs)
        self.assertIs(metadata["minimax_token_tags"], tags)
        self.assertIs(result_latent, latent_in)
        self.assertEqual(continuation["type"], "HR_H3_EXTERNAL_CONTINUATION")

    def test_analyzer_uses_qwen35_and_outputs_unencoded_context(self):
        result = types.SimpleNamespace(
            confidence="high", observed_end_state={"camera": "static"},
            transition_plan={"first_action": "step"}, h3_prompt="[Shot 1] Continue.")
        director = types.SimpleNamespace(external_video_continuation=lambda *args, **kwargs: result)
        selection = types.SimpleNamespace(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        with patch.object(MODULE, "resolve_director_selection", return_value=selection), \
             patch.object(MODULE, "Qwen35ContinuityDirector", return_value=director):
            output = MODULE.HRMiniMaxH3VideoContinuationAnalyzer.execute(
                torch.zeros(22, 64, 96, 3), prompt="continue", source_fps=24.0)
        prompt, context, analysis, tail = output.result
        self.assertEqual(prompt, result.h3_prompt)
        self.assertEqual(context["type"], "HR_H3_EXTERNAL_CONTINUATION_CONTEXT")
        self.assertEqual(context["tail_images"].shape, (22, 64, 96, 3))
        self.assertEqual(json.loads(analysis)["confidence"], "high")
        self.assertEqual(tail.shape, (22, 64, 96, 3))


if __name__ == "__main__":
    unittest.main()
