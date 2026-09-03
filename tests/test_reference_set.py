import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]

# reference_set has ComfyUI node definitions, but these tests exercise only its
# deterministic media-slot and protocol validation helpers.
comfy = types.ModuleType("comfy")
comfy.__path__ = []
for name in ("model_management", "nested_tensor", "utils"):
    module = types.ModuleType(f"comfy.{name}")
    setattr(comfy, name, module)
    sys.modules[module.__name__] = module
sys.modules["comfy"] = comfy
sys.modules.setdefault("node_helpers", types.ModuleType("node_helpers"))

class _Custom:
    def __init__(self, name):
        self.name = name

    def Input(self, *args, **kwargs):
        return (args, kwargs)

    def Output(self, *args, **kwargs):
        return (args, kwargs)


class _ComfyNode:
    pass


class _Dummy:
    def __getattr__(self, _name):
        return _Dummy()

    def __call__(self, *args, **kwargs):
        return (args, kwargs)


io = _Dummy()
io.Custom = _Custom
io.ComfyNode = _ComfyNode
latest = types.ModuleType("comfy_api.latest")
latest.io = io
comfy_api = types.ModuleType("comfy_api")
comfy_api.__path__ = []
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest

spec = importlib.util.spec_from_file_location("hr_reference_set_test", PLUGIN_ROOT / "reference_set.py")
reference_set = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference_set
spec.loader.exec_module(reference_set)


class ReferenceSetTests(unittest.TestCase):
    def test_indexed_preserves_empty_video_audio_slots(self):
        first_video = object()
        second_video = object()
        second_audio = object()
        videos = reference_set._indexed({"ref_video_0": first_video, "ref_video_1": second_video}, "ref_video_", 3)
        audios = reference_set._indexed({"ref_video_audio_1": second_audio}, "ref_video_audio_", 3)
        self.assertEqual(videos, (first_video, second_video, None))
        self.assertEqual(audios, (None, second_audio, None))

    def test_normalize_accepts_same_index_video_soundtrack(self):
        video = object()
        audio = object()
        result = reference_set.normalize_reference_set({
            "version": 1,
            "videos": (None, video, None),
            "video_audios": (None, audio, None),
        })
        self.assertIs(result["videos"][1], video)
        self.assertIs(result["video_audios"][1], audio)

    def test_normalize_rejects_orphan_video_soundtrack(self):
        with self.assertRaisesRegex(ValueError, "same-index"):
            reference_set.normalize_reference_set({
                "version": 1,
                "videos": (object(), None, None),
                "video_audios": (None, object(), None),
            })

    def test_reference_limits_and_scale_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            reference_set.normalize_reference_set({"version": 1, "images": tuple(object() for _ in range(10))})
        with self.assertRaisesRegex(ValueError, "ref_scale"):
            reference_set.normalize_reference_set({"version": 1, "ref_scale": 5.1})


if __name__ == "__main__":
    unittest.main()
