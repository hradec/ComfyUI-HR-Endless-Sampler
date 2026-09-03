import importlib.util
import json
import sys
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "hr_story_format_test"
spec = importlib.util.spec_from_file_location(MODULE_NAME, PLUGIN_ROOT / "story_format.py")
story_format = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = story_format
spec.loader.exec_module(story_format)


class StoryFormatTests(unittest.TestCase):
    def test_extracts_multiple_jzl_story_blocks_and_sections(self):
        text = """前置统计信息
[SHOT_START]
### Video_001
===H3_PROMPT===
subject_definitions: hero

detailed_description: first
===SCENE_INSTRUCTION===
{"shot": 1, "slots": ["角色:A"]}
===VIDEO_INSTRUCTION===
{"shot": 1, "slots": []}
===AUDIO_INSTRUCTION===
{"shot": 1, "slots": ["音频:B:"]}
[SHOT_END]
[SHOT_START]
===H3_PROMPT===
detailed_description: second
===SCENE_INSTRUCTION===
{"shot": 2, "slots": []}
[SHOT_END]
"""
        blocks = story_format.parse_story_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("detailed_description: first", blocks[0].h3_prompt)
        self.assertEqual(json.loads(blocks[0].scene_instruction)["slots"], ["角色:A"])
        self.assertEqual(blocks[1].video_instruction, "{}")
        self.assertEqual(blocks[1].audio_instruction, "{}")

    def test_parse_four_in_one_preserves_jzl_defaults(self):
        h3, scene, video, audio = story_format.parse_four_in_one(
            "===H3_PROMPT===\ndetailed_description: one"
        )
        self.assertEqual(h3, "detailed_description: one")
        self.assertEqual((scene, video, audio), ("{}", "{}", "{}"))

    def test_parse_slots_accepts_string_list_and_dict(self):
        expected = ["角色:角色A"]
        self.assertEqual(story_format.parse_slots('{"slots":["角色:角色A"]}'), expected)
        self.assertEqual(story_format.parse_slots(['{"slots":["角色:角色A"]}']), expected)
        self.assertEqual(story_format.parse_slots({"slots": expected}), expected)
        self.assertEqual(story_format.parse_slots("not json"), [])

    def test_normalize_slots_matches_jzl_compatibility_rules(self):
        source = json.dumps(
            {"shot": 1, "slots": ["角色:A", "道具:C", "音频:D:", "literal"]},
            ensure_ascii=False,
        )
        normalized = json.loads(story_format.normalize_slots(source))
        self.assertEqual(
            normalized["slots"],
            ["角色:角色A", "道具:道具C", "音频:音频D", "literal"],
        )

    def test_normalize_slots_returns_unparseable_input_unchanged(self):
        self.assertEqual(story_format.normalize_slots("not json"), "not json")

    def test_plain_text_without_story_markers_is_not_silently_a_block(self):
        self.assertEqual(story_format.parse_story_blocks("plain prompt"), ())

    def test_h3_frame_count_is_aligned_up_to_native_grid(self):
        self.assertEqual(story_format.planned_frame_count(5.0, 24.0), 124)
        self.assertEqual(story_format.planned_frame_count(8.0, 24.0), 192)

    def test_validate_storyboard_requires_complete_contiguous_coverage(self):
        value = {
            "image_subjects": [{"picture": 1, "subject": 1, "name": "女主角", "observable_features": "黑色短发"}],
            "shots": [
                {"shot": 1, "start_frame": 0, "end_frame": 60, "pictures": [1], "description": "她走进房间。"},
                {"shot": 2, "start_frame": 60, "end_frame": 124, "pictures": [1], "description": "她看向镜头。"},
            ],
        }
        plan = story_format.validate_storyboard_plan(value, image_count=1, total_frames=124)
        self.assertEqual(plan["shots"][-1]["end_frame"], 124)
        broken = json.loads(json.dumps(value, ensure_ascii=False))
        broken["shots"][1]["start_frame"] = 61
        with self.assertRaisesRegex(ValueError, "contiguous"):
            story_format.validate_storyboard_plan(broken, image_count=1, total_frames=124)

    def test_compile_h3_prompt_owns_markers_and_preserves_dialogue(self):
        plan = story_format.validate_storyboard_plan(
            {
                "image_subjects": [{"picture": 1, "subject": 1, "name": "女主角", "observable_features": "黑色短发、灰色风衣"}],
                "summary": "[reference generation] A continuous encounter.",
                "retention_analysis": "<Subject 1>: fully_preserved - 黑色短发、灰色风衣。",
                "shots": [
                    {"shot": 1, "start_frame": 0, "end_frame": 60, "pictures": [1], "description": "A woman enters and says: <d>[Chinese] 你终于来了。</d>"},
                    {"shot": 2, "start_frame": 60, "end_frame": 124, "pictures": [1], "description": "The camera cuts closer as she looks into the lens."},
                ],
                "overall_soundscape": "Rain and footsteps.",
                "non_diegetic_music": "None.",
            },
            image_count=1,
            total_frames=124,
        )
        prompt = story_format.compile_h3_prompt(plan, fps=24.0)
        self.assertIn("subject_definitions:", prompt)
        self.assertIn("[Shot 1] A woman enters", prompt)
        self.assertIn("[Shot 2] At 00:02.500,", prompt)
        self.assertIn("<d>[Chinese] 你终于来了。</d>", prompt)
        self.assertEqual(prompt.count("[Shot 1]"), 1)
        self.assertEqual(prompt.count("[Shot 2]"), 1)

    def test_validate_storyboard_rejects_unknown_picture(self):
        value = {
            "image_subjects": [{"picture": 1, "subject": 1, "name": "hero", "observable_features": "coat"}],
            "shots": [{"shot": 1, "start_frame": 0, "end_frame": 124, "pictures": [2], "description": "walk"}],
        }
        with self.assertRaisesRegex(ValueError, "Picture 2"):
            story_format.validate_storyboard_plan(value, image_count=1, total_frames=124)


if __name__ == "__main__":
    unittest.main()
