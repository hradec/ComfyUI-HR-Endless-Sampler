from __future__ import annotations

import ast
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


def _key(chunk_prompts=("chunk 0 prompt", "chunk 1 prompt"), chunk_ticks=(4, 4)):
    """Build one key through the real builder so its contents stay covered."""
    return nodes._teacher_audio_key(chunk_prompts, chunk_ticks)


def _payload(chunk_count=2, ticks=4):
    """Build one small pass payload with exactly the recorded products."""
    milestones = [[torch.full((ticks, 3), float(index + step)) for step in range(2)] for index in range(chunk_count)]
    return {
        "milestones": milestones,
        "audio_latent": torch.ones(1, 32, ticks * chunk_count),
        "prompts": [f"chunk {index} prompt" for index in range(chunk_count)],
        "transcripts": [{"text": f"heard {index}", "words": [], "language": "en"} for index in range(chunk_count)],
        # Observations are decoded [channels, samples] waveforms, as the teacher
        # publishes them.
        "observations": [(torch.zeros(2, 8, dtype=torch.float32), 32000) for _ in range(chunk_count)],
        # Per-pass teacher state a resume has to restore.
        "audio_time_origin": 129,
        "audio_anchor": (torch.ones(4), torch.full((4,), 2.0)),
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
        key = _key()
        payload = _payload()
        self.cache.save(key, payload)

        loaded, reason = self.cache.load_if_compatible(key, 2)

        self.assertIsNone(reason)
        self.assertEqual(loaded["prompts"], payload["prompts"])
        self.assertEqual(loaded["transcripts"], payload["transcripts"])
        self.assertTrue(torch.equal(loaded["audio_latent"], payload["audio_latent"]))
        _assert_same_tensors(self, loaded["milestones"], payload["milestones"])
        _assert_same_tensors(self, loaded["observations"], payload["observations"])
        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunks"], 2)
        self.assertEqual(manifest["key"], key)

    def test_key_covers_the_chunk_prompts_and_the_structure_only(self):
        """The owner's rule: keep the audio unless a chunk prompt or the chunking changes."""
        baseline = _key()
        json.dumps(baseline)
        # Editing one chunk's prompt is the case the rule names: the changed
        # chunk is identifiable, so a future partial reuse can start there.
        edited = _key(chunk_prompts=("chunk 0 prompt", "an edited chunk 1 prompt"))
        self.assertNotEqual(baseline, edited)
        self.assertEqual(nodes._teacher_audio_key_difference(baseline, edited), "Chunk 2's prompt changed")
        # chunk_frames changes the chunk count, their audio lengths and every
        # chunk prompt, which is what redraws the audio.
        self.assertNotEqual(baseline, _key(chunk_ticks=(4, 4, 4)))
        self.assertNotEqual(baseline, _key(chunk_ticks=(4, 5)))
        self.assertEqual(baseline, _key())
        self.assertEqual(sorted(baseline), ["chunk_prompts", "chunk_ticks"])

    def test_rejects_a_changed_key_an_obsolete_format_and_a_truncated_payload(self):
        """Nothing incompatible may reach sampling as if it were a match."""
        key = _key()
        self.cache.save(key, _payload())

        changed, reason = self.cache.load_if_compatible(_key(chunk_prompts=("chunk 0 prompt", "an edited chunk 1 prompt")), 2)
        self.assertIsNone(changed)
        self.assertEqual(reason, "Chunk 2's prompt changed")

        rechunked, reason = self.cache.load_if_compatible(_key(chunk_ticks=(4, 5)), 2)
        self.assertIsNone(rechunked)
        self.assertIn("Chunk 2 now spans 5 audio ticks", reason)

        regrouped, reason = self.cache.load_if_compatible(_key(chunk_ticks=(4, 4, 4)), 2)
        self.assertIsNone(regrouped)
        self.assertIn("plans 3 chunks, the cache has 2", reason)

        short, reason = self.cache.load_if_compatible(key, 3)
        self.assertIsNone(short)
        self.assertEqual(reason, "teacher audio cache chunk count does not match this render")

        truncated = _payload()
        truncated.pop("observations")
        nodes._replay_write_tensor_file(self.cache.payload_path, truncated)
        damaged, reason = self.cache.load_if_compatible(key, 2)
        self.assertIsNone(damaged)
        self.assertEqual(reason, "teacher audio cache payload is missing observations")

        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["format"] = nodes.TEACHER_AUDIO_CACHE_FORMAT + 1
        nodes._replay_write_json(self.cache.manifest_path, manifest)
        self.assertEqual(self.cache.load_if_compatible(key, 2), (None, "teacher audio cache format is obsolete"))

    def test_missing_and_unreadable_entries_report_instead_of_raising(self):
        """A cache miss is a normal render, never an exception."""
        absent, reason = self.cache.load_if_compatible(_key(), 2)
        self.assertIsNone(absent)
        self.assertEqual(reason, "no teacher audio cache exists")

        self.cache.root.mkdir(parents=True, exist_ok=True)
        self.cache.manifest_path.write_text("{half written", encoding="utf-8")
        unreadable, reason = self.cache.load_if_compatible(_key(), 2)
        self.assertIsNone(unreadable)
        self.assertIn("could not load teacher audio cache", reason)

    def test_save_replaces_an_earlier_pass_and_clear_removes_only_it(self):
        """One render keeps one reusable pass, and clearing is total."""
        self.cache.save(_key(chunk_prompts=("first", "first")), _payload())
        self.cache.save(_key(chunk_prompts=("second", "second", "second"), chunk_ticks=(2, 2, 2)), _payload(chunk_count=3, ticks=2))

        manifest = json.loads((self.cache.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunks"], 3)
        loaded, reason = self.cache.load_if_compatible(_key(chunk_prompts=("second", "second", "second"), chunk_ticks=(2, 2, 2)), 3)
        self.assertIsNone(reason)
        self.assertEqual(loaded["prompts"], ["chunk 0 prompt", "chunk 1 prompt", "chunk 2 prompt"])

        self.cache.clear()
        self.assertFalse(self.cache.root.exists())


    def _record_files(self, indices):
        """Create stand-in record files; the resume check reads their names only."""
        self.cache.kv_root.mkdir(parents=True, exist_ok=True)
        for index in indices:
            (self.cache.kv_root / ("chunk-%03d.pt" % index)).write_bytes(b"")

    def test_a_stopped_pass_keeps_its_identity_and_its_settled_prefix(self):
        """A render that is stopped must leave the newest, resumable entry behind."""
        # The status reader resolves the sampler's own temporary layout, so this
        # test binds the cache to the same place a render would use.
        self.cache.root = Path(self.directory.name) / nodes.GEMMA_PROMPT_LOG_DIRNAME / nodes.TEACHER_AUDIO_CACHE_DIRNAME
        key = _key(("first", "second", "third"), (4, 4, 4))
        self.cache.begin_pass(key, 3, 0)
        partial = _payload(chunk_count=2, ticks=4)
        partial.pop("audio_latent")
        self.cache.save_chunk_state(1, partial)

        manifest = json.loads(self.cache.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "recording")
        self.assertEqual(manifest["chunks"], 3)
        self.assertEqual(manifest["recorded"], 2)
        self.assertEqual(self.cache.stored_key(), key)
        self.assertEqual(self.cache.settled_chunk_count(), 2)
        # A stopped pass cannot be reused whole, and a refresh still restores it
        # because its settled chunks carry their prompts and decoded audio.
        self.assertIsNotNone(nodes._teacher_audio_payload_problem(partial, 3))
        partial["preview"] = {"fps": 24.0, "chunks": [
            {"index": 0, "start": 0, "end": 23, "prompt": "chunk 0 prompt", "subtitle": "one"},
            {"index": 1, "start": 24, "end": 47, "prompt": "chunk 1 prompt", "subtitle": "two"},
        ]}
        snapshot = preview.build_teacher_audio_preview_snapshot("7", partial, fps=24.0)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["reset"]["chunk_count"], 2)

        with patch.object(nodes.tempfile, "gettempdir", return_value=self.directory.name):
            status = nodes._teacher_audio_cache_ui_status_unlocked()
        self.assertTrue(status["interrupted"])
        self.assertEqual(status["recorded"], 2)

    def test_every_cache_call_matches_its_method_signature(self):
        """A wrong argument count in the resume path kills a live render at Chunk 1."""
        names = ("load_prefix", "load_available_kv_prefix", "kv_chunk_indices", "settled_chunk_count",
                 "deleted_chunk_indices", "begin_pass", "save_chunk_state", "delete_chunk", "discard_kv_chunk",
                 "load_kv_chunk", "save_kv_chunk", "stored_key", "load_if_compatible", "clear_kv_records")
        # Two classes share some of these names with different arities, so every
        # arity this module defines for a name counts as valid for that name.
        allowed = {}
        for name in names:
            arities = set()
            for value in vars(nodes).values():
                if isinstance(value, type) and value.__module__ == nodes.__name__:
                    method = getattr(value, name, None)
                    if callable(method):
                        arities.add(len(inspect.signature(method).parameters) - 1)
            allowed[name] = arities
        checked = 0
        for node in ast.walk(ast.parse((PLUGIN_ROOT / "nodes.py").read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.keywords:
                continue
            arities = allowed.get(node.func.attr)
            if not arities:
                continue
            checked += 1
            self.assertIn(
                len(node.args), arities,
                f"{node.func.attr}() is called with {len(node.args)} arguments at nodes.py:{node.lineno}, but the method takes {sorted(arities)}",
            )
        self.assertGreaterEqual(checked, 12)

    def test_the_stored_pass_keeps_its_creation_time_across_rewrites(self):
        """The manifest names the pass; only its progress changes while it runs."""
        self.cache.begin_pass(_key(), 2, 0)
        created = json.loads(self.cache.manifest_path.read_text(encoding="utf-8"))["created"]
        self.cache.save_chunk_state(1, _payload())
        self.cache.save(_key(), _payload())
        manifest = json.loads(self.cache.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["created"], created)
        self.assertEqual(manifest["status"], "complete")

    def test_a_refresh_keeps_the_whole_plan_in_the_timeline(self):
        """Chunks the pass has not reached yet keep their slot, without audio."""
        payload = _payload(chunk_count=2, ticks=4)
        payload.pop("audio_latent")
        payload["preview"] = {"fps": 24.0, "chunks": [
            {"index": 0, "start": 0, "end": 23, "prompt": "chunk 0 prompt", "subtitle": "one", "settled": True},
            {"index": 1, "start": 24, "end": 47, "prompt": "chunk 1 prompt", "subtitle": "two", "settled": True},
            {"index": 2, "start": 48, "end": 71, "prompt": "planned third", "subtitle": "", "settled": False},
        ]}

        snapshot = preview.build_teacher_audio_preview_snapshot("7", payload, fps=24.0)

        reset = snapshot["reset"]
        self.assertEqual(reset["chunk_count"], 3)
        self.assertEqual(reset["total_frames"], 72)
        self.assertEqual(reset["cached_chunk_count"], 2)
        self.assertEqual(reset["reused_chunk_numbers"], [1, 2])
        self.assertEqual([action["chunk"] for action in snapshot["chunks"] if action["action"] == "chunk_audio_first"], [0, 1])
        self.assertEqual(reset["chunk_ranges"][2]["h3_prompt"], "planned third")

    def test_a_refresh_keeps_every_teacher_audio_attempt(self):
        """Ctrl-hover still names the first take and its retries after a reload."""
        payload = _payload(chunk_count=2, ticks=4)
        payload.pop("audio_latent")
        payload["preview"] = {"fps": 24.0, "chunks": [
            {"index": 0, "start": 0, "end": 23, "prompt": "retry one", "subtitle": "one", "settled": True,
             "attempts": ["take one", "retry one"], "attempt_transcripts": ["heard one", "the retry spoke"],
             "subchunk_silent_numbers": [2], "subchunk_silent_prompt": "no dialogue here",
             "subchunk_transcripts": ["heard one", ""]},
            {"index": 1, "start": 24, "end": 47, "prompt": "second", "subtitle": "two", "settled": True},
        ]}

        snapshot = preview.build_teacher_audio_preview_snapshot("7", payload, fps=24.0)

        ranges = snapshot["reset"]["chunk_ranges"]
        self.assertEqual(ranges[0]["audio_teacher_attempts"], ["take one", "retry one"])
        self.assertEqual(ranges[0]["audio_teacher_attempt_transcripts"], ["heard one", "the retry spoke"])
        self.assertEqual(ranges[0]["audio_teacher_prompt"], "retry one")
        # Ctrl+Shift names the sub-chunk under the pointer, so a reload keeps them too.
        self.assertEqual((ranges[0]["subchunk_silent_numbers"], ranges[0]["subchunk_silent_prompt"]), ([2], "no dialogue here"))
        # The empty second slot stays, so a transcript cannot drift onto another sub-chunk.
        self.assertEqual(ranges[0]["subchunk_transcripts"], ["heard one", ""])
        # A chunk that was never retried carries no list, so the tooltip shows its
        # own prompt as the single attempt.
        self.assertNotIn("audio_teacher_attempts", ranges[1])
        self.assertNotIn("audio_teacher_attempt_transcripts", ranges[1])
        self.assertNotIn("subchunk_silent_numbers", ranges[1])
        self.assertNotIn("subchunk_transcripts", ranges[1])

    def test_a_fresh_pass_replaces_the_stopped_one(self):
        """A pass that starts from Chunk 1 must not leave the old pass behind."""
        self.cache.begin_pass(_key(), 2, 0)
        self.cache.save_chunk_state(1, _payload())
        self._record_files([0, 1])
        fresh = _key(("chunk 0 prompt", "an edited chunk 1 prompt"))
        self.cache.begin_pass(fresh, 2, 0)

        self.assertEqual(self.cache.stored_key(), fresh)
        self.assertEqual(self.cache.settled_chunk_count(), 0)
        self.assertEqual(self.cache.kv_chunk_indices(), [])
        self.assertFalse(self.cache.payload_path.is_file())

    def test_a_resume_keeps_the_prefix_and_drops_the_regenerated_rows(self):
        """The rows after the resume point belong to the take being replaced."""
        self._record_files([0, 1, 2, 3])
        key = _key(("a", "b", "c", "d"), (4, 4, 4, 4))
        self.cache.begin_pass(key, 4, 2)
        self.cache.save_chunk_state(1, _payload())

        self.assertEqual(self.cache.kv_chunk_indices(), [0, 1])
        self.assertEqual(self.cache.settled_chunk_count(), 2)
        self.assertEqual(self.cache.stored_key(), key)

    def test_the_button_choice_outlives_the_process(self):
        """A restart must not flip a render back to cache reuse."""
        original = nodes.REPLAY_CACHE_ENABLED
        self.addCleanup(setattr, nodes, "REPLAY_CACHE_ENABLED", original)
        with patch.object(nodes.tempfile, "gettempdir", return_value=self.directory.name):
            nodes._save_cache_policy(False)
            nodes.REPLAY_CACHE_ENABLED = True
            nodes._load_cache_policy()
            self.assertFalse(nodes.REPLAY_CACHE_ENABLED)
            nodes._save_cache_policy(True)
            nodes.REPLAY_CACHE_ENABLED = False
            nodes._load_cache_policy()
            self.assertTrue(nodes.REPLAY_CACHE_ENABLED)


class TeacherAudioCacheWiringTest(unittest.TestCase):
    """Pin the sampler wiring the unit tests above cannot execute directly."""

    @classmethod
    def setUpClass(cls):
        cls.source = (PLUGIN_ROOT / "nodes.py").read_text(encoding="utf-8")

    def test_audio_first_pass_reuses_the_cache_and_still_records_one(self):
        self.assertIn("and not teacher_audio_reused:", self.source)
        self.assertIn('teacher_audio_cache.load_if_compatible(teacher_audio_key, replay_sample_end)', self.source)
        self.assertIn("taomate_backend.audio_first_milestones = {index: milestones for index, milestones in enumerate(cached_teacher_audio[\"milestones\"])}", self.source)
        self.assertIn("taomate_backend.audio_first_ready = True", self.source)
        self.assertIn("teacher_audio_cache.save(teacher_audio_key, teacher_audio_payload(full_teacher_audio))", self.source)

    def test_the_key_is_the_chunk_prompts_and_the_chunk_structure(self):
        """Only the owner's chunk prompts and the chunk lengths may rebuild audio."""
        start = self.source.index("teacher_audio_key = _teacher_audio_key(")
        call = self.source[start:self.source.index("\n", start)]
        self.assertIn("teacher_audio_key_prompts", call)
        self.assertIn("teacher_audio_chunk_ticks", call)
        # The owner's own table wins; the planned prose is only the fallback.
        self.assertIn("teacher_audio_key_prompts = [str(text) for text in manual_prompts.prompts()[:replay_sample_end]]", self.source)
        self.assertIn("teacher_audio_key_prompts = [item[0] for item in planned_prompts[:replay_sample_end]]", self.source)
        self.assertIn('teacher_audio_chunk_ticks = [int(chunk["audio_end"]) - int(chunk["audio_start"]) for chunk in active_plan[:replay_sample_end]]', self.source)
        for ignored in ("noise_seed", "sigmas", "kv_cache_compression", "model_patcher", "toggles", "replay_fingerprint", "video_continuation"):
            self.assertNotIn(ignored, call)

    def test_cache_button_off_clears_every_last_cache_as_the_render_starts(self):
        """One rule: a non-cached render starts with no last cache at all."""
        self.assertIn("if not _replay_cache_enabled():\n            # A non-cached render starts clean", self.source)
        self.assertIn("_remove_replay_cache()\n            _remove_teacher_audio_cache()", self.source)
        self.assertIn('_set_teacher_audio_cache_state("disabled"', self.source)

    def test_an_incompatible_cache_warns_instead_of_failing_silently(self):
        """The button asked for the cache, so an unusable one must say so."""
        self.assertIn('_set_teacher_audio_cache_state("incompatible", teacher_audio_reason)', self.source)
        self.assertIn("cannot reuse the stored teacher audio and this render overwrites it", self.source)

    def test_a_storage_failure_never_stops_the_render(self):
        self.assertIn("could not record reusable teacher audio; this render continues without a stored copy", self.source)

    def test_preview_status_reports_the_teacher_audio_state(self):
        self.assertIn('"audio_cache": _teacher_audio_cache_ui_status_unlocked(),', self.source)

    def test_the_pass_identifies_itself_before_it_samples(self):
        """A stopped render must leave an owned, resumable pass, not orphaned rows."""
        self.assertIn("teacher_audio_cache.begin_pass(teacher_audio_key, replay_sample_end, teacher_audio_resume, keep_payload=teacher_audio_kept_tail is not None)", self.source)
        self.assertIn("teacher_audio_cache.save_chunk_state(audio_index, teacher_audio_payload())", self.source)
        self.assertIn("teacher_audio_note_settled_chunk(audio_index)", self.source)
        self.assertNotIn("teacher_audio_cache.clear_kv_records()", self.source)

    def test_a_taomate_render_drops_an_older_video_checkpoint(self):
        """TaoMate carries no video cache, so an older checkpoint can only mislead."""
        self.assertIn("elif use_taomate:\n            # A TaoMate render reads no video checkpoint", self.source)
        self.assertIn("deleted the video replay checkpoint of an earlier render", self.source)

    def test_the_stored_preview_describes_the_whole_plan(self):
        """A truncated snapshot would drop the chunks the pass has not reached."""
        self.assertIn('"settled": audio_index < stored,', self.source)
        self.assertIn("_teacher_audio_chunk_prompt(preview_chunk_ranges, prompts, audio_index)", self.source)
        builder = (PLUGIN_ROOT / "preview.py").read_text(encoding="utf-8")
        self.assertIn('if chunk.get("settled") is False:', builder)

    def test_the_stored_payload_keeps_the_approved_tail(self):
        """A stop right after the edited chunk must not lose the kept tail."""
        self.assertIn("if teacher_audio_kept_tail is not None and not teacher_audio_kept_tail_merged and settled > teacher_audio_resume:", self.source)
        self.assertIn('prompts.extend(teacher_audio_kept_tail["prompts"])', self.source)
        self.assertIn('full_teacher_audio = torch.cat([full_teacher_audio, *teacher_audio_kept_tail["latents"]], dim=-1)', self.source)

    def test_a_resumed_render_reports_how_much_it_kept(self):
        """The preview counts the kept chunks, not the plan's chunk count."""
        self.assertIn("kept_chunks = teacher_audio_resume + (len(teacher_audio_kept_tail[\"prompts\"]) if teacher_audio_kept_tail is not None else 0)", self.source)
        self.assertIn("_teacher_audio_resume_reason(teacher_audio_resume, replay_sample_end, teacher_audio_kept_tail is not None)", self.source)
        self.assertIn("generate_count=1 if teacher_audio_kept_tail is not None else None", self.source)

    def test_a_changed_first_chunk_keeps_the_later_chunks(self):
        """Chunk 1 has no prefix, but the finished pass still offers the rest."""
        self.assertIn("stored_pass, _stored_reason = teacher_audio_cache.load_complete_payload(replay_sample_end)", self.source)
        self.assertIn("teacher_audio_kept_tail = _teacher_audio_kept_tail(stored_pass, 0, replay_sample_end, teacher_audio_cache.kv_chunk_indices())", self.source)
        self.assertIn("elif teacher_audio_resume or teacher_audio_kept_tail is not None:", self.source)
        self.assertIn("keep_payload=teacher_audio_kept_tail is not None", self.source)

    def test_every_render_logs_its_cache_decision(self):
        """A silent rebuild cannot be diagnosed, so one line names the branch."""
        self.assertIn('"HR Endless Sampler teacher audio cache decision: %s (%s)%s.",', self.source)
        self.assertIn('_teacher_audio_cache_state_snapshot()["run_state"],', self.source)
        self.assertIn('", keeping %d chunks" % teacher_audio_resume if teacher_audio_resume else "",', self.source)

    def test_the_decision_line_has_a_reason_whatever_branch_ran(self):
        """The button being off skips the cache load, so that branch names its own reason.

        The decision line below reads the variable in every branch, and a render with
        the cache button off never reached the assignment, which crashed the render
        with an UnboundLocalError after the whole pass had run.
        """
        branch = self.source.index("if not _replay_cache_enabled():")
        decision = self.source.index('"HR Endless Sampler teacher audio cache decision: %s (%s)%s.",')
        body = self.source[branch:decision]
        name = body.index("teacher_audio_reason =")
        # The off branch binds the reason before the branch that loads the cache.
        self.assertLess(name, body.index("cached_teacher_audio, teacher_audio_reason ="))
        self.assertIn("the cache button is off, so this render records a fresh teacher audio pass", body)

    def test_the_cache_button_choice_is_remembered_and_reported(self):
        self.assertIn("_save_cache_policy(REPLAY_CACHE_ENABLED)", self.source)
        self.assertIn("_load_cache_policy()", self.source)
        self.assertIn('"kv_chunks": _TeacherAudioCache().kv_chunk_indices(),', self.source)

    def test_an_unusable_cache_is_deleted_rather_than_left_behind(self):
        """Button on and the cache cannot drive this render: it must not survive."""
        self.assertIn("deleted it and recording a fresh replacement", self.source)
        self.assertIn("cache button is on but the stored checkpoint cannot drive this render", self.source)
        # TaoMate has no compatible checkpoint at all, so the same rule applies
        # without waiting for a fingerprint comparison.
        self.assertIn("elif use_taomate:\n            # A TaoMate render reads no video checkpoint", self.source)

    def test_the_timeline_deletes_one_cached_teacher_audio_chunk(self):
        """The owner's ask: force a single chunk to be generated again."""
        self.assertIn("/hr_endless_sampler_preview/teacher_audio_chunk", self.source)
        self.assertIn("cache.delete_chunk(chunk_index)", self.source)
        script = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn("Delete teacher audio cached chunk", script)
        self.assertIn("/hr_endless_sampler_preview/teacher_audio_chunk?chunk=${chunkIndex}", script)
        self.assertIn("audioCacheChunkIndices", script)

    def test_a_refresh_rebuilds_the_last_render_from_the_stored_audio(self):
        """A refresh must restore the newest cache, frames or not."""
        payload = _payload(chunk_count=2, ticks=4)
        payload["preview"] = {"fps": 24.0, "chunks": [
            {"index": 0, "start": 0, "end": 23, "prompt": "chunk 0 prompt", "subtitle": "one"},
            {"index": 1, "start": 24, "end": 47, "prompt": "chunk 1 prompt", "subtitle": "two"},
        ]}
        snapshot = preview.build_teacher_audio_preview_snapshot("7", payload, fps=24.0)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["reset"]["chunk_count"], 2)
        self.assertEqual(snapshot["reset"]["total_frames"], 48)
        self.assertEqual(snapshot["reset"]["phase"], "Restored the last TaoMate render audio")
        self.assertEqual([chunk["chunk"] for chunk in snapshot["chunks"]], [0, 1])
        for chunk in snapshot["chunks"]:
            self.assertEqual(chunk["action"], "chunk_audio_first")
            self.assertEqual(chunk["audio_mime"], "audio/wav")
            self.assertIn("audio", chunk)
        self.assertEqual(snapshot["audio_chunks"], [0, 1])
        self.assertEqual(snapshot["complete"]["action"], "complete")

    def test_the_refresh_builder_refuses_a_payload_it_cannot_restore(self):
        """A missing preview block or missing audio must not produce a player."""
        self.assertIsNone(preview.build_teacher_audio_preview_snapshot("7", _payload(), fps=24.0))
        self.assertIsNone(preview.build_teacher_audio_preview_snapshot("7", {"preview": {"chunks": [{"index": 0, "start": 0, "end": 1}]}, "observations": []}, fps=24.0))
        self.assertIsNone(preview.build_teacher_audio_preview_snapshot("7", {"preview": {"chunks": []}, "observations": [(torch.zeros(2, 8), 32000)]}, fps=24.0))


if __name__ == "__main__":
    unittest.main()
