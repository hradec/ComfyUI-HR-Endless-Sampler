import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "continuation_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package

for name in ("director_errors", "gemma4", "preview", "qwen35", "director_config", "reference_set", "story_format", "storyboard", "video_io", "director_backend"):
    path = PLUGIN_ROOT / f"{name}.py"
    if path.is_file() and PACKAGE + "." + name not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

nodes_spec = importlib.util.spec_from_file_location(PACKAGE + ".nodes", PLUGIN_ROOT / "nodes.py")
nodes = importlib.util.module_from_spec(nodes_spec)
sys.modules[nodes_spec.name] = nodes
nodes_spec.loader.exec_module(nodes)

spec = importlib.util.spec_from_file_location(PACKAGE + ".continuation", PLUGIN_ROOT / "continuation.py")
continuation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = continuation
spec.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def test_checkpoint_requires_complete_replay(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(nodes.tempfile, "gettempdir", return_value=directory), \
             patch.object(continuation, "_continuation_root", return_value=Path(directory) / "archive"):
            cache = nodes._LastRunReplayCache()
            cache.create({"fps": 24.0, "plan": [{"frame_end": 22}]}, "prompt", {"video": torch.zeros(1)})
            with self.assertRaisesRegex(ValueError, "complete"):
                continuation.create_checkpoint_from_last_run()

    def test_checkpoint_is_durable_and_plan_reloads_it(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(nodes.tempfile, "gettempdir", return_value=directory), \
             patch.object(continuation, "_continuation_root", return_value=Path(directory) / "archive"):
            cache = nodes._LastRunReplayCache()
            cache.create({"fps": 24.0, "plan": [{"frame_end": 22}]}, "prompt", {"video": torch.zeros(1)})
            video = torch.zeros((1, 24, 7, 2, 2))
            audio = torch.zeros((1, 32, 2, 37))
            cache.save_chunk(1, {"sampled_video": video, "sampled_audio": audio,
                "output_video": video, "output_audio": audio,
                "denoised_video": video, "denoised_audio": audio,
                "output_template": {}, "denoised_template": {}},
                metadata={"frame_start": 0, "frame_end": 22, "effective_h3_prompt": "old"})
            cache.mark_complete(1)
            manifest = continuation.create_checkpoint_from_last_run("Episode A")
            checkpoint = {"type": "HR_CONTINUATION_CHECKPOINT", "version": 1,
                          "checkpoint_id": manifest["checkpoint_id"]}
            loaded_manifest, loaded_state = continuation.load_checkpoint(checkpoint)
            self.assertEqual(loaded_manifest["name"], "Episode A")
            self.assertTrue(torch.equal(loaded_state["sampled_audio"], audio))
            plan, plan_json = continuation.HREndlessContinuationPlan.execute(
                checkpoint, "new story", "continue", "replace").result
            self.assertEqual(plan["checkpoint_id"], manifest["checkpoint_id"])
            self.assertEqual(json.loads(plan_json)["prompt"], "new story")
            reference_set = {"version": 1, "images": (torch.zeros((2, 8, 8, 3)),),
                             "videos": (), "video_audios": (), "audios": (),
                             "ref_image_size": "match", "ref_scale": 1.0}
            plan, plan_json = continuation.HREndlessContinuationPlan.execute(
                checkpoint, "new story", "continue", "replace", reference_set).result
            self.assertIs(plan["reference_set"]["images"][0], reference_set["images"][0])
            self.assertEqual(json.loads(plan_json)["reference_set"]["images"], 1)
            self.assertEqual(continuation.list_checkpoints()[0]["checkpoint_id"], manifest["checkpoint_id"])
            cache._update_manifest(status="interrupted", completed_chunks=0)
            reused, _info = continuation.HREndlessContinuationCheckpoint.execute("ignored").result
            self.assertEqual(reused["checkpoint_id"], manifest["checkpoint_id"])
            cache.clear()

    def test_checkpoint_id_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            continuation._checkpoint_root("../outside")


if __name__ == "__main__":
    unittest.main()
