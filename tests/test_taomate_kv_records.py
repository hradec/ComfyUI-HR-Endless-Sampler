from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")
taomate = importlib.import_module(PLUGIN_ROOT.name + ".python.taomate")


AUDIO_TAG = 2


class TeacherKvRecordTest(unittest.TestCase):
    """The per-chunk teacher KV records must restore the rows bit for bit."""

    def _cache(self, mode):
        contract = taomate.KVContract(local_heads=2, head_dim=128, dtype=torch.bfloat16, device_type="cpu")
        cache = taomate.TeacherAudioKVCache(contract, mode)
        # The runtime hook reports the attention device; without a render there
        # is none, and history() needs it to materialise rows.
        cache.active_device = "cpu"
        return cache

    def _stage_chunk(self, cache, ticks, seed=0):
        """Stage one chunk's audio rows the way the teacher's commit does."""
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(ticks, 2, 128, generator=generator).to(torch.bfloat16)
        value = torch.randn(ticks, 2, 128, generator=generator).to(torch.bfloat16)
        tags = torch.full((ticks,), AUDIO_TAG, dtype=torch.long)
        mask = torch.ones(ticks, dtype=torch.bool)
        cache.begin_clean_commit(cache.committed_blocks)
        for layer in taomate.MAIN_LAYER_NAMES:
            cache.stage(layer, key, value, tags, mask)
        return cache.staged_snapshot()

    def test_round_trip_restores_every_codec_exactly(self):
        """All TurboQuant widths, int8, zstd frames and raw rows survive save/load."""
        for mode in ("turboquant4", "turboquant3", "turboquant2", "int8", "none", "zstd lossless"):
            with self.subTest(mode=mode):
                source = self._cache(mode)
                pairs, block_index, tags = self._stage_chunk(source, 6)
                source.commit()

                payload = {
                    "format": taomate.TEACHER_KV_RECORD_FORMAT,
                    "layers": taomate.serialize_stored_kv_chunk(pairs),
                    "block": block_index,
                    "tags": list(tags),
                    "ticks": len(tags),
                }
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "chunk.pt"
                    torch.save(payload, path)
                    # The sampler's own loader: the record must stay inside the
                    # weights_only allowlist, bytes for codec frames included.
                    loaded = torch.load(path, map_location="cpu", weights_only=True)

                restored = self._cache(mode)
                restored.prime_staged(taomate.deserialize_stored_kv_chunk(loaded["layers"]), loaded["block"], loaded["tags"])

                self.assertEqual(restored.committed_blocks, source.committed_blocks)
                self.assertEqual(restored.history_tokens, source.history_tokens)
                for layer in taomate.MAIN_LAYER_NAMES:
                    self.assertTrue(torch.equal(restored.history(layer).key, source.history(layer).key), layer)
                    self.assertTrue(torch.equal(restored.history(layer).value, source.history(layer).value), layer)

    def test_two_primed_chunks_continue_the_history_in_order(self):
        """Priming chunk by chunk must reproduce a two-chunk history."""
        mode = "turboquant4"
        source = self._cache(mode)
        records = []
        for index in range(2):
            pairs, block_index, tags = self._stage_chunk(source, 4 + index, seed=index + 1)
            records.append({"layers": taomate.serialize_stored_kv_chunk(pairs), "block": block_index, "tags": list(tags)})
            source.commit()

        restored = self._cache(mode)
        for record in records:
            restored.prime_staged(taomate.deserialize_stored_kv_chunk(record["layers"]), record["block"], record["tags"])

        self.assertEqual(restored.history_tokens, source.history_tokens)
        self.assertEqual(restored.history_audio_tokens, source.history_audio_tokens)
        for layer in taomate.MAIN_LAYER_NAMES:
            self.assertTrue(torch.equal(restored.history(layer).key, source.history(layer).key), layer)

    def test_legacy_turboquant_records_restore_as_four_bit(self):
        """Previously saved `turboquant` records keep their 4-bit interpretation."""
        cache = self._cache("turboquant4")
        pairs, _block, _tags = self._stage_chunk(cache, 2)
        payload = taomate.serialize_stored_kv(pairs["blocks.0.attn"])
        payload["mode"] = "turboquant"
        restored = taomate.deserialize_stored_kv(payload)
        self.assertEqual(restored.mode, "turboquant4")
        self.assertTrue(torch.equal(restored.tensor(0, "cpu"), pairs["blocks.0.attn"].tensor(0, "cpu")))

    def test_turboquant_width_storage_and_mixed_mode_resume(self):
        """Bit widths meet the expected packed sizes and resume into the chosen codec."""
        key = torch.randn(3, 2, 128).to(torch.bfloat16)
        value = torch.randn(3, 2, 128).to(torch.bfloat16)
        pair = taomate.AVKV(key, value)
        for mode, bits in (("turboquant4", 4), ("turboquant3", 3), ("turboquant2", 2)):
            indices = torch.randint(0, 1 << bits, (3, 2, 128), dtype=torch.uint8)
            packed = taomate._pack_turboquant_indices(indices, bits)
            unpacked = taomate._unpack_turboquant_indices(packed, bits, indices.shape)
            self.assertTrue(torch.equal(unpacked, indices.long()))
            compressed = taomate.QuantizedKV(pair, mode)
            vectors = key.shape[0] * key.shape[1]
            expected_bytes = vectors * 2 * (128 * bits // 8 + 4)
            self.assertEqual(compressed.stored_bytes, expected_bytes)
            self.assertTrue(torch.isfinite(compressed.tensor(0, "cpu")).all())

        self.assertEqual(self._cache(None).compression_mode, "turboquant3")

        old = taomate.QuantizedKV(pair, "turboquant4")
        new = taomate.QuantizedKV(pair, "turboquant3")
        combined = taomate.QuantizedKV.concat(old, new)
        self.assertEqual(combined.mode, "turboquant3")
        self.assertEqual(combined.shape[0], 6)

    def test_priming_rejects_an_incomplete_or_active_chunk(self):
        """A truncated record must fail loudly instead of building a short history."""
        cache = self._cache("turboquant")
        pairs, block_index, tags = self._stage_chunk(cache, 4)
        cache.commit()
        partial = dict(list(taomate.serialize_stored_kv_chunk(pairs).items())[:5])
        with self.assertRaises(RuntimeError):
            self._cache("turboquant").prime_staged(taomate.deserialize_stored_kv_chunk(partial), block_index, tags)
        cache.begin_clean_commit(cache.committed_blocks)
        with self.assertRaises(RuntimeError):
            cache.prime_staged({}, block_index, tags)


class TeacherAudioKvRecordTest(unittest.TestCase):
    """The pass cache stores and returns one KV record per chunk."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = nodes._TeacherAudioCache()
        self.cache.root = Path(self.directory.name) / "teacher_audio"
        contract = taomate.KVContract(local_heads=2, head_dim=128, dtype=torch.bfloat16, device_type="cpu")
        self.teacher = taomate.TeacherAudioKVCache(contract, "turboquant4")
        self.teacher.active_device = "cpu"

    def tearDown(self):
        self.directory.cleanup()

    def _commit_chunk(self, index, ticks, seed):
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(ticks, 2, 128, generator=generator).to(torch.bfloat16)
        value = torch.randn(ticks, 2, 128, generator=generator).to(torch.bfloat16)
        tags = torch.full((ticks,), AUDIO_TAG, dtype=torch.long)
        mask = torch.ones(ticks, dtype=torch.bool)
        self.teacher.begin_clean_commit(self.teacher.committed_blocks)
        for layer in taomate.MAIN_LAYER_NAMES:
            self.teacher.stage(layer, key, value, tags, mask)
        snapshot = self.teacher.staged_snapshot()
        self.teacher.commit()
        self.assertEqual(self.cache.save_kv_chunk(index, snapshot), ticks)

    def test_records_round_trip_and_rebuild_the_history(self):
        """Stored records for two chunks must prime a fresh teacher exactly."""
        self._commit_chunk(0, 5, seed=1)
        self._commit_chunk(1, 7, seed=2)
        self.assertEqual(self.cache.kv_chunk_indices(), [0, 1])

        prefix = self.cache.load_available_kv_prefix(2)
        self.assertEqual([record["ticks"] for record in prefix], [5, 7])
        rebuilt = taomate.TeacherAudioKVCache(self.teacher.contract, "turboquant")
        rebuilt.active_device = "cpu"
        for record in prefix:
            rebuilt.prime_staged(record["pairs"], record["block"], record["tags"])
        self.assertEqual(rebuilt.history_tokens, self.teacher.history_tokens)
        for layer in taomate.MAIN_LAYER_NAMES:
            self.assertTrue(torch.equal(rebuilt.history(layer).key, self.teacher.history(layer).key), layer)

    def test_a_gap_in_the_records_blocks_the_prefix(self):
        """A missing record means the KV cannot be rebuilt, so nothing is offered."""
        self._commit_chunk(0, 5, seed=1)
        self._commit_chunk(1, 5, seed=2)
        (self.cache.kv_root / "chunk-000.pt").unlink()
        self.assertEqual(self.cache.kv_chunk_indices(), [1])
        self.assertEqual(self.cache.load_available_kv_prefix(2), [])
        # Nothing before the resume point needs no records at all.
        self.assertEqual(self.cache.load_available_kv_prefix(0), [])

    def test_clearing_the_entry_removes_its_records(self):
        """The cache button must not leave orphaned attention state behind."""
        self._commit_chunk(0, 5, seed=1)
        self.assertTrue(self.cache.kv_root.is_dir())
        self.cache.clear()
        self.assertFalse(self.cache.root.exists())

    def test_resume_index_stops_at_the_first_edited_chunk(self):
        """An edited chunk prompt is the resume point, and only when supported."""
        ticks = [5, 7, 6]
        prompts = ("chunk one", "chunk two", "chunk three")
        self._commit_chunk(0, 5, seed=1)
        self._commit_chunk(1, 7, seed=2)
        self._commit_chunk(2, 6, seed=3)
        key = nodes._teacher_audio_key(prompts, ticks)
        payload = {
            "milestones": [[torch.zeros(3, 5), torch.zeros(3, 5)], [torch.zeros(3, 7), torch.zeros(3, 7)], [torch.zeros(3, 6), torch.zeros(3, 6)]],
            "audio_latent": torch.zeros(1, 32, 18),
            "prompts": list(prompts),
            "transcripts": [{}, {}, {}],
            "observations": [(torch.zeros(4), 32000)] * 3,
        }
        self.cache.save(key, payload)

        # An identical key is a full reuse, not a resume.
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 0)        # Editing the third prompt resumes at it, keeping two settled chunks.
        edited = nodes._teacher_audio_key(("chunk one", "chunk two", "chunk three!"), ticks)
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, edited, 3), 2)
        # A different chunk structure cannot keep any window.
        rechunked = nodes._teacher_audio_key(("chunk one", "chunk two"), [5, 7])
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, rechunked, 2), 0)
        # A missing record means the KV cannot be rebuilt, so nothing is kept.
        self.cache.discard_kv_chunk(0)
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, edited, 3), 0)

    def test_an_interrupted_pass_resumes_at_its_stop_point(self):
        """A stopped render continues where it stopped instead of from Chunk 1."""
        ticks = [5, 7, 6]
        prompts = ("chunk one", "chunk two", "chunk three")
        key = nodes._teacher_audio_key(prompts, ticks)
        # The pass names itself first, then records what it settles, then stops.
        self.cache.begin_pass(key, 3, 0)
        self._commit_chunk(0, 5, seed=1)
        self._commit_chunk(1, 7, seed=2)
        self.cache.save_chunk_state(1, {
            "milestones": [[torch.zeros(3, 5), torch.zeros(3, 5)], [torch.zeros(3, 7), torch.zeros(3, 7)]],
            "prompts": list(prompts[:2]),
            "transcripts": [{}, {}],
            "observations": [(torch.zeros(4), 32000)] * 2,
        })

        self.assertEqual(self.cache.settled_chunk_count(), 2)
        # The unfinished pass cannot be reused whole, so its chunks are kept as a
        # prefix and the teacher continues at Chunk 3.
        self.assertIsNotNone(nodes._teacher_audio_payload_problem(nodes._replay_load_tensor_file(self.cache.payload_path), 3))
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 2)
        # A delete inside the settled prefix moves the resume point back to it.
        self.cache.delete_chunk(0)
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 0)

    def test_rows_that_lag_the_payload_cap_the_resume_point(self):
        """A chunk's rows land with the next chunk's forward, so stop where they stop."""
        ticks = [5, 7, 6]
        prompts = ("chunk one", "chunk two", "chunk three")
        key = nodes._teacher_audio_key(prompts, ticks)
        self.cache.begin_pass(key, 3, 0)
        self._commit_chunk(0, 5, seed=1)
        self.cache.save_chunk_state(1, {
            "milestones": [[torch.zeros(3, 5), torch.zeros(3, 5)], [torch.zeros(3, 7), torch.zeros(3, 7)]],
            "prompts": list(prompts[:2]),
            "transcripts": [{}, {}],
            "observations": [(torch.zeros(4), 32000)] * 2,
        })

        # Two chunks are settled, but only Chunk 1 has rows on disk.
        self.assertEqual(self.cache.settled_chunk_count(), 2)
        self.assertEqual(self.cache.kv_chunk_indices(), [0])
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 1)

    def test_a_finished_pass_offers_the_chunks_after_the_edit(self):
        """Those chunks keep approved audio; only the edited one is generated."""
        ticks = [5, 7, 6]
        payload = {
            "prompts": ["one", "two", "three"],
            "transcripts": [{"text": "one"}, {"text": "two"}, {"text": "three"}],
            "observations": [(torch.zeros(4), 32000)] * 3,
            "milestones": [[torch.zeros(3, size), torch.zeros(3, size)] for size in ticks],
        }

        tail = nodes._teacher_audio_kept_tail(payload, 0, 3, [0, 1, 2])

        self.assertEqual(tail["prompts"], ["two", "three"])
        self.assertEqual(tail["transcripts"], [{"text": "two"}, {"text": "three"}])
        self.assertEqual(len(tail["observations"]), 2)
        self.assertEqual(len(tail["latents"]), 2)
        self.assertEqual([int(latent.shape[-1]) for latent in tail["latents"]], [7, 6])
        # The edited chunk is the last one, so there is nothing after it to keep.
        self.assertIsNone(nodes._teacher_audio_kept_tail(payload, 2, 3, [0, 1, 2]))

    def test_a_tail_without_its_attention_rows_is_not_kept(self):
        """Audio and its rows are cached together, so missing rows lose the tail."""
        payload = {
            "prompts": ["one", "two", "three"],
            "transcripts": [{"text": "one"}, {"text": "two"}, {"text": "three"}],
            "observations": [(torch.zeros(4), 32000)] * 3,
            "milestones": [[torch.zeros(3, size), torch.zeros(3, size)] for size in (5, 7, 6)],
        }

        # The pass regenerates Chunk 1, so only Chunks 2 and 3 need their rows.
        self.assertIsNone(nodes._teacher_audio_kept_tail(payload, 0, 3, [0]))
        self.assertIsNone(nodes._teacher_audio_kept_tail(payload, 0, 3, [0, 1]))
        self.assertIsNotNone(nodes._teacher_audio_kept_tail(payload, 0, 3, [0, 1, 2]))

    def test_an_unfinished_pass_offers_no_tail(self):
        """A hole in the tail would need rows that no longer exist."""
        payload = {
            "prompts": ["one", "two"],
            "transcripts": [{"text": "one"}, {"text": "two"}],
            "observations": [(torch.zeros(4), 32000)] * 2,
            "milestones": [[torch.zeros(3, 5), torch.zeros(3, 5)], [torch.zeros(3, 7), torch.zeros(3, 7)]],
        }
        self.assertIsNone(nodes._teacher_audio_kept_tail(payload, 0, 3, [0, 1, 2]))

    def test_the_resume_reason_names_both_sides(self):
        """The preview prints this sentence, so it has to describe the real work."""
        self.assertEqual(
            nodes._teacher_audio_resume_reason(3, 6, True),
            "kept teacher audio for Chunks 1-3 and Chunks 5-6; generating Chunk 4",
        )
        self.assertEqual(
            nodes._teacher_audio_resume_reason(3, 6, False),
            "kept teacher audio for Chunks 1-3; generating Chunks 4-6",
        )
        self.assertEqual(nodes._teacher_audio_resume_reason(0, 6, False), "generating Chunks 1-6")
        self.assertEqual(
            nodes._teacher_audio_resume_reason(4, 6, True),
            "kept teacher audio for Chunks 1-4 and Chunk 6; generating Chunk 5",
        )
        self.assertEqual(
            nodes._teacher_audio_resume_reason(1, 2, True),
            "kept teacher audio for Chunk 1; generating Chunk 2",
        )

    def test_an_edited_first_chunk_keeps_the_chunks_after_it(self):
        """Chunk 1 changing regenerates that chunk only, as any later edit does."""
        ticks = [5, 7, 6]
        prompts = ("chunk one", "chunk two", "chunk three")
        key = nodes._teacher_audio_key(prompts, ticks)
        self.cache.save(key, {
            "milestones": [[torch.zeros(3, size), torch.zeros(3, size)] for size in ticks],
            "audio_latent": torch.zeros(1, 32, 18),
            "prompts": list(prompts),
            "transcripts": [{}, {}, {}],
            "observations": [(torch.zeros(4), 32000)] * 3,
        })
        edited = nodes._teacher_audio_key(("edited one", "chunk two", "chunk three"), ticks)

        # No prefix exists before Chunk 1, so the resume point is 0...
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, edited, 3), 0)
        # ...and the finished stored pass still offers the chunks after it.
        stored, reason = self.cache.load_complete_payload(3)
        self.assertIsNone(reason)
        tail = nodes._teacher_audio_kept_tail(stored, 0, 3, [0, 1, 2])
        self.assertEqual(tail["prompts"], ["chunk two", "chunk three"])
        self.assertEqual(nodes._teacher_audio_resume_reason(0, 3, True), "kept teacher audio for Chunks 2-3; generating Chunk 1")

    def test_a_kept_tail_keeps_the_rows_of_its_chunks(self):
        """The kept audio stays next to its rows: only the generated chunk loses its own."""
        ticks = [5, 7]
        self.cache.save(nodes._teacher_audio_key(("one", "two"), ticks), {
            "milestones": [[torch.zeros(3, size), torch.zeros(3, size)] for size in ticks],
            "audio_latent": torch.zeros(1, 32, 12),
            "prompts": ["one", "two"],
            "transcripts": [{}, {}],
            "observations": [(torch.zeros(4), 32000)] * 2,
        })
        self.cache.kv_root.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            (self.cache.kv_root / ("chunk-%03d.pt" % index)).write_bytes(b"")

        # Chunk 1 is edited and Chunk 2 keeps its approved audio, so only Chunk 1's
        # row is dropped.
        self.cache.begin_pass(nodes._teacher_audio_key(("edited one", "two"), ticks), 2, 0, keep_payload=True)

        self.assertTrue(self.cache.payload_path.is_file())
        self.assertEqual(self.cache.kv_chunk_indices(), [1])
        self.assertEqual(self.cache.settled_chunk_count(), 0)
        # The pass generates that one chunk, so the stored pairs are whole again.
        self._commit_chunk(0, 5, seed=9)
        self.assertEqual(self.cache.kv_chunk_indices(), [0, 1])
        # A later edit of Chunk 2 drops that row and keeps Chunk 1's.
        self.cache.begin_pass(nodes._teacher_audio_key(("edited one", "edited two"), ticks), 2, 1, keep_payload=True)
        self.assertEqual(self.cache.kv_chunk_indices(), [0])
        # Without a kept tail the same call drops the payload, as before.
        self.cache.begin_pass(nodes._teacher_audio_key(("edited one", "two"), ticks), 2, 0)
        self.assertFalse(self.cache.payload_path.is_file())
        self.assertEqual(self.cache.kv_chunk_indices(), [])

    def test_deleting_one_stored_chunk_forces_a_new_take_from_it(self):
        """The timeline delete must beat a complete pass's whole-pass reuse."""
        ticks = [5, 7, 6]
        prompts = ("chunk one", "chunk two", "chunk three")
        self._commit_chunk(0, 5, seed=1)
        self._commit_chunk(1, 7, seed=2)
        self._commit_chunk(2, 6, seed=3)
        key = nodes._teacher_audio_key(prompts, ticks)
        self.cache.save(key, {
            "milestones": [[torch.zeros(3, size), torch.zeros(3, size)] for size in ticks],
            "audio_latent": torch.zeros(1, 32, 18),
            "prompts": list(prompts),
            "transcripts": [{}, {}, {}],
            "observations": [(torch.zeros(4), 32000)] * 3,
        })
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 0)

        self.cache.delete_chunk(1)

        loaded, reason = self.cache.load_if_compatible(key, 3)
        self.assertIsNone(loaded)
        self.assertIn("Chunk 2 of the stored teacher audio was deleted", reason)
        self.assertEqual(self.cache.kv_chunk_indices(), [0, 2])
        self.assertEqual(nodes._teacher_audio_resume_index(self.cache, key, 3), 1)

    def test_prefix_loading_requires_every_settled_chunk(self):
        """The prefix is offered only when its chunks still describe this render."""
        ticks = [5, 7]
        prompts = ("chunk one", "chunk two")
        key = nodes._teacher_audio_key(prompts, ticks)
        payload = {
            "milestones": [[torch.zeros(3, 5), torch.zeros(3, 5)], [torch.zeros(3, 7), torch.zeros(3, 7)]],
            "audio_latent": torch.zeros(1, 32, 12),
            "prompts": list(prompts),
            "transcripts": [{}, {}],
            "observations": [(torch.zeros(4), 32000)] * 2,
        }
        self.cache.save(key, payload)
        loaded, reason = self.cache.load_prefix(2, ticks)
        self.assertIsNone(reason)
        self.assertEqual(len(loaded["milestones"]), 2)
        # A shorter payload cannot settle the requested prefix.
        short, reason = self.cache.load_prefix(3, ticks + [4])
        self.assertIsNone(short)
        self.assertIn("shorter than the resume point", reason)
        # Moved audio windows invalidate the stored clean audio.
        moved, reason = self.cache.load_prefix(1, [6, 7])
        self.assertIsNone(moved)
        self.assertIn("Chunk 1 now spans 6 audio ticks", reason)
        # Nothing to resume needs no payload at all.
        self.assertEqual(self.cache.load_prefix(0, ticks), (None, "nothing to resume"))


if __name__ == "__main__":
    unittest.main()
