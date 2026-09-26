from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")


def _patcher(identifier="patch-a", strength=1.0, value=1.0):
    """Build the minimal patcher surface the model signature inspects.

    ComfyUI stores every applied patch under a random identifier, so the
    signature must ignore the key and describe the patch itself.
    """
    patch = {"blocks.0.attn.qkv_proj.weight": torch.full((4, 4), value)}
    return SimpleNamespace(patches={identifier: (strength, patch)}, model=SimpleNamespace())


def _fingerprint(prompt="a source prompt", noise_seed=7, sigmas=None, kv_cache_compression="none", model_patcher=None, toggles=None):
    """Build one key through the real builder so its contents stay covered."""
    return nodes._teacher_audio_fingerprint(
        {"plan": [{"frame_start": 0, "frame_end": 124}], "fps": 24.0, "video_shape": [1, 24, 2, 4, 4], "compute_dtype": "default"},
        prompt=prompt,
        noise_seed=noise_seed,
        sigmas=torch.tensor([1.0, 0.5, 0.0]) if sigmas is None else sigmas,
        kv_cache_compression=kv_cache_compression,
        model_patcher=_patcher() if model_patcher is None else model_patcher,
        toggles={"audio_first": True} if toggles is None else toggles,
    )


def _payload(chunk_count=2, ticks=4):
    """Build one small pass payload with exactly the recorded products."""
    milestones = [[torch.full((ticks, 3), float(index + step)) for step in range(2)] for index in range(chunk_count)]
    return {
        "milestones": milestones,
        "audio_latent": torch.ones(1, 32, ticks * chunk_count),
        "prompts": [f"chunk {index} prompt" for index in range(chunk_count)],
        "transcripts": [{"text": f"heard {index}", "words": [], "language": "en"} for index in range(chunk_count)],
        "observations": [(torch.zeros(8, dtype=torch.float32), 32000) for _ in range(chunk_count)],
    }


def _assert_same_tensors(case, restored, original):
    """Compare nested tensor containers elementwise."""
    if isinstance(original, torch.Tensor):
        case.assertTrue(torch.equal(restored, original))
        return
    if not isinstance(original, (list, tuple)):
        case.assertEqual(restored, original)
        return
    # The payload validator accepts only tuples for a decoded observation, so a
    # restored list would silently reject every reuse.
    case.assertEqual(type(restored), type(original))
    case.assertEqual(len(restored), len(original))
    for left, right in zip(restored, original):
        _assert_same_tensors(case, left, right)


class TeacherAudioCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        # The cache is bound to a temporary root, so a test run can never touch
        # or erase the sampler's real cache.
        self.cache = nodes._TeacherAudioCache()
        self.cache.root = Path(self.directory.name) / "teacher_audio"

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip_restores_every_recorded_product(self):
        """A compatible key returns the complete pass, not a partial one."""
        fingerprint = _fingerprint()
        payload = _payload()
        self.cache.save(fingerprint, "a source prompt", payload)

        loaded, reason = self.cache.load_if_compatible(fingerprint, 2)

        self.assertIsNone(reason)
        self.assertEqual(loaded["prompts"], payload["prompts"])
        self.assertEqual(loaded["transcripts"], payload["transcripts"])
        self.assertTrue(torch.equal(loaded["audio_latent"], payload["audio_latent"]))
        _assert_same_tensors(self, loaded["milestones"], payload["milestones"])
        _assert_same_tensors(self, loaded["observations"], payload["observations"])
        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunks"], 2)
        self.assertEqual(manifest["fingerprint"], fingerprint)

    def test_fingerprint_stays_json_safe_and_reacts_to_every_input(self):
        """The key is stored in JSON, so it must serialize and must separate."""
        baseline = _fingerprint()
        json.dumps(baseline)
        variations = {
            "prompt": _fingerprint(prompt="an edited source prompt"),
            "seed": _fingerprint(noise_seed=8),
            "sigmas": _fingerprint(sigmas=torch.tensor([1.0, 0.25, 0.0])),
            "kv codec": _fingerprint(kv_cache_compression="turboquant"),
            "lora values": _fingerprint(model_patcher=_patcher(value=2.0)),
            "lora strength": _fingerprint(model_patcher=_patcher(strength=0.5)),
            "toggles": _fingerprint(toggles={"audio_first": True, "equal_sub_chunks": True}),
        }
        for name, variation in variations.items():
            self.assertNotEqual(baseline, variation, name)
        self.assertEqual(baseline, _fingerprint())

    def test_model_signature_ignores_the_random_patch_identifier(self):
        """Two renders with the same LoRA must share one signature."""
        self.assertEqual(nodes._teacher_audio_model_signature(_patcher(identifier="patch-a")), nodes._teacher_audio_model_signature(_patcher(identifier="patch-z")))
        self.assertNotEqual(nodes._teacher_audio_model_signature(_patcher(value=1.0)), nodes._teacher_audio_model_signature(_patcher(value=2.0)))

    def test_toggle_signature_names_every_divergence_switch(self):
        """The key must cover the switches that change the teacher's audio."""
        toggles = nodes._teacher_audio_toggle_signature()
        self.assertEqual(set(toggles), {"audio_first", "transcript_retry", "condition_attends_current_av", "move_prompt_references", "equal_sub_chunks", "compress_kv", "gpu_decompress_kv", "renorm_teacher_audio"})
        self.assertTrue(all(isinstance(value, bool) for value in toggles.values()))

    def test_rejects_a_changed_key_an_obsolete_format_and_a_truncated_payload(self):
        """Nothing incompatible may reach sampling as if it were a match."""
        fingerprint = _fingerprint()
        self.cache.save(fingerprint, "a source prompt", _payload())

        changed, reason = self.cache.load_if_compatible(_fingerprint(prompt="an edited source prompt"), 2)
        self.assertIsNone(changed)
        self.assertEqual(reason, "prompt, seed, or a teacher-audio setting changed")

        short, reason = self.cache.load_if_compatible(fingerprint, 3)
        self.assertIsNone(short)
        self.assertEqual(reason, "teacher audio cache chunk count does not match this render")

        truncated = _payload()
        truncated.pop("observations")
        nodes._replay_write_tensor_file(self.cache.payload_path, truncated)
        damaged, reason = self.cache.load_if_compatible(fingerprint, 2)
        self.assertIsNone(damaged)
        self.assertEqual(reason, "teacher audio cache payload is missing observations")

        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["format"] = nodes.TEACHER_AUDIO_CACHE_FORMAT + 1
        nodes._replay_write_json(self.cache.manifest_path, manifest)
        self.assertEqual(self.cache.load_if_compatible(fingerprint, 2), (None, "teacher audio cache format is obsolete"))

    def test_missing_and_unreadable_entries_report_instead_of_raising(self):
        """A cache miss is a normal render, never an exception."""
        absent, reason = self.cache.load_if_compatible(_fingerprint(), 2)
        self.assertIsNone(absent)
        self.assertEqual(reason, "no teacher audio cache exists")

        self.cache.root.mkdir(parents=True, exist_ok=True)
        self.cache.manifest_path.write_text("{half written", encoding="utf-8")
        unreadable, reason = self.cache.load_if_compatible(_fingerprint(), 2)
        self.assertIsNone(unreadable)
        self.assertIn("could not load teacher audio cache", reason)

    def test_save_replaces_an_earlier_pass_and_clear_removes_only_it(self):
        """One render keeps one reusable pass, and clearing is total."""
        self.cache.save(_fingerprint(prompt="first"), "first", _payload())
        self.cache.save(_fingerprint(prompt="second"), "second", _payload(chunk_count=3, ticks=2))

        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunks"], 3)
        loaded, reason = self.cache.load_if_compatible(_fingerprint(prompt="second"), 3)
        self.assertIsNone(reason)
        self.assertEqual(loaded["prompts"], ["chunk 0 prompt", "chunk 1 prompt", "chunk 2 prompt"])

        self.cache.clear()
        self.assertFalse(self.cache.root.exists())


class TeacherAudioCacheWiringTest(unittest.TestCase):
    """Pin the sampler wiring the unit tests above cannot execute directly."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PLUGIN_ROOT / "nodes.py").read_text(encoding="utf-8")

    def test_audio_first_pass_reuses_the_cache_and_still_records_one(self):
        self.assertIn("and not teacher_audio_reused:", self.source)
        self.assertIn('teacher_audio_cache.load_if_compatible(teacher_audio_fingerprint, replay_sample_end)', self.source)
        self.assertIn("taomate_backend.audio_first_milestones = {index: milestones for index, milestones in enumerate(cached_teacher_audio[\"milestones\"])}", self.source)
        self.assertIn("taomate_backend.audio_first_ready = True", self.source)
        self.assertIn("teacher_audio_cache.save(teacher_audio_fingerprint, prompt, {", self.source)

    def test_cache_button_off_resets_the_stored_pass_as_the_render_starts(self):
        self.assertIn('if not _replay_cache_enabled():\n                    teacher_audio_cache.clear()', self.source)
        self.assertIn('_set_teacher_audio_cache_state("disabled"', self.source)

    def test_a_storage_failure_never_stops_the_render(self):
        self.assertIn("could not record reusable teacher audio; this render continues without a stored copy", self.source)

    def test_preview_status_reports_the_teacher_audio_state(self):
        self.assertIn('"audio_cache": _teacher_audio_cache_ui_status_unlocked(),', self.source)


if __name__ == "__main__":
    unittest.main()
