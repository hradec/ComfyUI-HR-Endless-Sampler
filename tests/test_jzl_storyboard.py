import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load the pure helpers without importing ComfyUI's package initializer.
pkg = types.ModuleType("jzl_test_pkg")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg

spec = importlib.util.spec_from_file_location(pkg.__name__ + ".story_format", ROOT / "story_format.py")
story_format = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = story_format
spec.loader.exec_module(story_format)

spec = importlib.util.spec_from_file_location("jzl_helpers", ROOT / "jzl_storyboard.py")
# The node module requires ComfyUI; test its format contract through story_format.


class JZLStoryboardTests(unittest.TestCase):
    def test_four_in_one_compiles_to_ordered_segment_lists(self):
        text = """[SHOT_START]
===H3_PROMPT===
[Shot 1] A person walks.
===SCENE_INSTRUCTION===
{"slots":["场景:场景A"]}
===VIDEO_INSTRUCTION===
{"slots":[]}
===AUDIO_INSTRUCTION===
{"slots":["音频:音频A"]}
[SHOT_END]
[SHOT_START]
===H3_PROMPT===
[Shot 1] The person stops.
===SCENE_INSTRUCTION===
{}
[SHOT_END]"""
        blocks = story_format.parse_story_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].video_instruction, '{"slots":[]}')
        self.assertEqual(blocks[1].audio_instruction, "{}")

    def test_jzl_requires_explicit_story_blocks(self):
        self.assertEqual(story_format.parse_story_blocks("===H3_PROMPT=== plain"), ())


if __name__ == "__main__":
    unittest.main()
