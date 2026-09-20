from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")
preview = importlib.import_module(PLUGIN_ROOT.name + ".preview")


class _FakeVAE:
    def decode(self, _latent):
        return torch.zeros((39, 2, 2, 3), dtype=torch.float32)


class _IndexedFakeVAE:
    def decode(self, _latent):
        return torch.arange(39, dtype=torch.float32).reshape(39, 1, 1, 1).expand(39, 2, 2, 3)


class _FakeAudioVAE:
    audio_sample_rate = 8000

    def decode(self, _latent):
        return torch.ones((1, 12000, 2), dtype=torch.float32)


class ChunkDirectorHelperTest(unittest.TestCase):
    def test_manual_prompt_edits_preserve_old_and_new_cache_compatibility(self):
        """Ignore editor text but still reject changed render geometry."""
        module = importlib.import_module(PLUGIN_ROOT.name + ".python.preproduction")
        before = module.LegacyChunkPrompts("old text").fingerprint()
        after = module.LegacyChunkPrompts("new text").fingerprint()
        self.assertEqual(before, after)
        wanted = {"chunk_frames": 124, "pre_production": after}
        for provider in (before, dict(before, text="old cached text")):
            manifest = {"format": nodes.REPLAY_CACHE_FORMAT, "fingerprint": {"chunk_frames": 124, "pre_production": provider}}
            root = Mock()
            root.__truediv__ = Mock(return_value=Mock(read_text=Mock(return_value=json.dumps(manifest))))
            with patch.object(nodes, "_replay_cache_root", return_value=root), patch.object(nodes, "_replay_load_tensor_file", return_value={"video": "saved"}):
                cache = nodes._LastRunReplayCache()
                loaded, reason = cache.load_if_compatible(wanted)
                self.assertIsNone(reason)
                self.assertEqual(loaded["initial"]["video"], "saved")
                self.assertIsNone(cache.load_if_compatible(dict(wanted, chunk_frames=192))[0])

    def test_manual_chunk_prompts_round_trip_and_validation(self):
        """Separators are removed without rewriting authored H3 text."""
        module = importlib.import_module(PLUGIN_ROOT.name + ".python.preproduction")
        prompts = ["summary: My opening.\n\ndetailed_description: [Shot 1] Wide camera.", "summary: My continuation.\n\ndetailed_description: [Shot 1] Keep my exact wording."]
        provider = module.LegacyChunkPrompts(module.LegacyChunkPrompts.format(prompts))
        self.assertEqual(provider.prompts(), prompts)
        self.assertEqual(provider.get_chunk_prompt(2), prompts[1])
        with self.assertRaises(ValueError):
            provider.get_chunk_prompt(3)
        for invalid in ("", "No delimiter", module.LegacyChunkPrompts.format([""]), module.LegacyChunkPrompts.format(prompts).replace("Chunk 2", "Chunk 3")):
            with self.assertRaises(ValueError):
                module.LegacyChunkPrompts(invalid).prompts()

    def test_manual_prompt_bake_uses_native_layout_and_summary(self):
        """The button's backend emits a usable provider without diffusion."""
        module = importlib.import_module(PLUGIN_ROOT.name + ".python.preproduction")
        import server
        prompt = "summary: My opening.\n\ndetailed_description: [Shot 1] The camera is a closeup. She smiles."
        with patch.object(server.PromptServer, "instance", Mock(), create=True) as instance:
            module.HREndlessLegacyPromptBake.execute(73, "[]", prompt, 24, 39, 5, nodes.VIDEO_CONTINUATION_METHOD_MASKED_AV, "42", "request")
        event, payload = instance.send_sync.call_args.args
        self.assertEqual(event, "hr_endless_legacy_prompts")
        self.assertEqual(payload["request_id"], "request")
        provider = module.LegacyChunkPrompts(payload["text"])
        plan = nodes._chunk_plan_without_overlap(nodes._video_steps(73), nodes._audio_steps(73), 39, 5)
        self.assertEqual(len(provider.prompts()), len(plan))
        self.assertIn("summary: My opening.", provider.get_chunk_prompt(1))
        self.assertIn("keyframe completion", provider.get_chunk_prompt(2))
        self.assertNotIn("--8<--", provider.get_chunk_prompt(2))
        self.assertEqual(module.HREndlessLegacyChunkPrompts.define_schema().node_id, "HREndlessLegacyChunkPrompts")
        self.assertTrue(module.HREndlessLegacyPromptBake.define_schema().is_output_node)

    def test_visual_condition_noise_setting_on_both_branches_without_mutating_source(self):
        """Every chunk propagates the configured visual noise, including CFG negative."""
        original = {"positive": [{"minimax_visual_cond_noise_aug": 0.999}], "negative": [{}]}
        updated = nodes._conditioning_for_chunk(original, 0, 39, (torch.zeros(1, 1, 1), {}))
        for branch in ("positive", "negative"):
            self.assertEqual(updated[branch][0]["minimax_visual_cond_noise_aug"], nodes.minimax_visual_cond_noise_aug)
        self.assertEqual(original["positive"][0]["minimax_visual_cond_noise_aug"], 0.999)
        self.assertNotIn("minimax_visual_cond_noise_aug", original["negative"][0])

    def test_dialogue_language_codes_cover_h3_stable_languages(self):
        """Use espeak variants for MiniMax H3's stable dialogue languages."""
        expected = {"Arabic": "ar", "Chinese": "cmn", "English": "en-us", "French": "fr-fr", "German": "de", "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt-br", "Russian": "ru", "Spanish": "es"}
        self.assertEqual({language: nodes._dialogue_language_code(language) for language in expected}, expected)

    def test_audiosr_preview_copies_leave_original_audio_for_the_final_pass(self):
        """Preview processing must not feed enhanced samples into the final pass."""
        originals = [torch.full((1, 2, 320), 0.1), torch.full((1, 2, 320), 0.2)]
        seen = []

        def enhance(audio, **options):
            """Record the source and return visibly different 48 kHz preview data."""
            seen.append(audio["waveform"].clone())
            count = round(audio["waveform"].shape[-1] * 48000 / audio["sample_rate"])
            return {"waveform": torch.full((1, 2, count), 0.9), "sample_rate": 48000}

        with patch.object(nodes, "fix_audio", side_effect=enhance), patch.object(nodes.comfy.model_management, "unload_all_models"), patch.object(nodes.comfy.model_management, "soft_empty_cache"):
            for original in originals:
                waveform, rate, overlap = nodes._enhance_decoded_audio(original, 32000, enabled=True)
                self.assertEqual(rate, 48000)
                self.assertEqual(waveform.shape[-1], 480)
                self.assertIsNone(overlap)
            complete_original = torch.cat(originals, dim=-1)
            waveform, rate, _overlap = nodes._enhance_decoded_audio(complete_original, 32000, enabled=True)
            self.assertTrue(torch.equal(seen[-1], complete_original))
            self.assertEqual(waveform.shape[-1], 960)
            self.assertTrue(torch.equal(originals[0], torch.full((1, 2, 320), 0.1)))
            self.assertTrue(torch.equal(originals[1], torch.full((1, 2, 320), 0.2)))

    def test_chunk_summary_keeps_opening_verbatim_and_replaces_later_reference_roles(self):
        """Old first-frame/video/audio tasks must not leak into later summaries."""
        original = "subject_definitions:\nAlice.\nsummary: [keyframe completion + audio reference]\n<Picture 1> is first frame. Use <Video 1> and <Audio 1>.\n\ndetailed_description: [Shot 1] Alice speaks."
        rewritten = "subject_definitions:\nAlice.\nsummary: A director replacement.\n\ndetailed_description: Alice continues."
        first = nodes._chunk_summary_prompt(rewritten, original, False)
        self.assertIn(original[original.index("summary:"):original.index("detailed_description:")], first)
        self.assertIn("detailed_description: Alice continues.", first)
        for current in (original, rewritten):
            later = nodes._chunk_summary_prompt(current, original, True, picture_label="<Picture 2>", video_label="<Video 2>", audio_label="<Audio 2>")
            summary = later[later.index("summary:"):later.index("detailed_description:")]
            self.assertIn("[video continuation + audio reference + keyframe completion]", summary)
            self.assertIn("<Picture 2> serves as first frame of target video.", summary)
            self.assertIn("the previous video's audio", summary)
            self.assertNotIn("chunk", summary.lower())
            for old in ("<Picture 1>", "<Video 1>", "<Audio 1>", "A director replacement"):
                self.assertNotIn(old, summary)
            masked = nodes._chunk_summary_prompt(current, original, True, audio_label="<Audio 2>", boundary_keyframe=True)
            summary = masked[masked.index("summary:"):masked.index("detailed_description:")]
            self.assertNotIn("<Picture", summary)
            self.assertNotIn("video continuation", summary)
        self.assertEqual(nodes._chunk_summary_prompt("detailed_description: Plain.", "detailed_description: Plain.", False), "detailed_description: Plain.")

    def test_local_planner_keeps_the_complete_source_description_for_first_video(self):
        """The first video keeps visual prose while later videos slice action."""
        prompt = (
            "summary: original task\n\n"
            "detailed_description: [Shot 1] Initial source prose remains complete. "
            "A second source action remains present.\n\n"
            "overall_soundscape: quiet room"
        )
        plan = [
            {"frame_start": 0, "frame_end": 48, "output_trim_frames": 0},
            {"frame_start": 48, "frame_end": 96, "output_trim_frames": 0},
        ]
        with patch.object(nodes, "_legacy_shot_body_for_range", return_value="SLICED CONTINUATION") as slicer:
            planned = nodes._planned_chunk_prompts(prompt, plan, plan, 24.0, 0, False, False, True, 1, 1, legacy=True)

        self.assertIn("Initial source prose remains complete.", planned[0][0])
        self.assertIn("A second source action remains present.", planned[0][0])
        self.assertNotIn("SLICED CONTINUATION", planned[0][0])
        self.assertIn("SLICED CONTINUATION", planned[1][0])
        self.assertEqual(slicer.call_count, 1)

    def test_local_opening_body_keeps_visual_prose_but_times_dialogue(self):
        """Chunk 1 cannot repeat dialogue that belongs to a later video."""
        body = "Complete opening visual prose. <Subject 1> (S1) says: <d>[English] one two three four.</d>"
        words = list(nodes.re.finditer(r"\S+", "one two three four."))
        with patch.object(nodes, "_dialogue_word_weights", return_value=tuple(zip(words, (1.0, 1.0, 1.0, 1.0)))):
            opening = nodes._legacy_opening_body_with_timed_dialogue(body, 0, 4, 0, 2, 1)

        self.assertIn("Complete opening visual prose.", opening)
        self.assertIn("one two", opening)
        self.assertNotIn("three four.", opening)

    def test_first_legacy_chunk_keeps_the_source_static_camera_sentence(self):
        prompt = (
            "summary: Opening.\n\n"
            "detailed_description: [Shot 1] The camera stays static at the same position from start to end of the shot. "
            "<Subject 1> (S1) says: <d>[English] One two three four.</d>"
        )
        plan = [
            {"frame_start": 0, "frame_end": 24, "output_trim_frames": 0},
            {"frame_start": 24, "frame_end": 48, "output_trim_frames": 0},
        ]
        planned = nodes._planned_chunk_prompts(prompt, plan, plan, 24, 0, False, False, False, 1, 1, legacy=True)

        self.assertIn("The camera stays static at the same position from start to end of the shot.", planned[0][0])
        self.assertIn("The camera stays static in the stablished frame.", planned[1][0])

    def test_continuation_drops_complete_opening_picture_instruction_before_slicing(self):
        """An opening setup cannot be cut into a dangling dialogue prefix."""
        setup = "Starts with <Picture 1> as the exact first frame for the target video, and the camera stays static at <Picture 1> stablished position for the entire duration of the shot."
        body = setup + " She smiles warmly."
        first = nodes._legacy_opening_body_with_timed_dialogue(body, 0, 240, 0, 120)
        later = nodes._legacy_shot_body_for_range(body, 0, 240, 120, 240)
        self.assertIn(setup, first)
        self.assertNotIn("<Picture 1>", later)
        self.assertIn("She smiles warmly.", later)
        long_sentence = "The subject " + "very " * 40 + "slowly turns around."
        sliced = nodes._legacy_shot_body_for_range(long_sentence, 0, 240, 120, 240)
        self.assertEqual(sliced.strip(), long_sentence)

    def test_camera_establishment_is_exclusive_per_source_shot(self):
        """A new shot establishes its camera even when it starts mid-chunk."""
        prompt = "detailed_description: [Shot 1] The camera is a frontal closeup. She smiles. [Shot 2] At 00:03.000, The camera is a side view. He waves."
        plan = [{"frame_start": start, "frame_end": start + 48, "output_trim_frames": 0} for start in (0, 48, 96)]
        planned = nodes._planned_chunk_prompts(prompt, plan, plan, 24, 0, False, False, False, 1, 1, legacy=True)
        automatic = "The camera stays static in the stablished frame."
        self.assertIn("frontal closeup", planned[0][0])
        self.assertNotIn(automatic, planned[0][0])
        self.assertNotIn("frontal closeup", planned[1][0])
        self.assertIn("side view", planned[1][0])
        self.assertEqual(planned[1][0].count(automatic), 1)
        self.assertNotIn("side view", planned[2][0])
        self.assertEqual(planned[2][0].count(automatic), 1)

    def test_video1_last_frame_is_an_ordinary_picture_reference(self):
        """Qwen and H3 receive the same final frame without a target keyframe."""
        frames = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 32, 64, 3) / 4.0
        vae = Mock()
        vae.encode.return_value = torch.ones((1, 24, 1, 2, 4))
        item, ref = nodes._last_frame_picture_reference(vae, frames, 64, 32)
        self.assertTrue(torch.equal(item["data"], frames[-1:]))
        self.assertIs(vae.encode.call_args.args[0], item["data"])
        self.assertEqual((ref["kind"], ref["latent_h"], ref["latent_w"]), ("image", 2, 4))
        original = {"positive": [{"cross_attn": None, "minimax_refs": [{"kind": "image"}]}]}
        clip = Mock()
        nodes._prompt_tokens(clip, "summary: test", frames[:1], original["positive"], 64, 32, True, [item])
        refs = clip.tokenize.call_args.kwargs["minimax_ref_items"]
        self.assertEqual([entry["type"] for entry in refs], ["image", "image"])
        self.assertIs(refs[-1], item)
        linear_ref = {"kind": "image", "latent": torch.full((1,), 0.25)}
        conds = nodes._conditioning_for_chunk(original, 39, 78, (torch.ones((1, 2, 3)), {}), video_refs=[ref], replacement_refs=[linear_ref])
        self.assertIs(conds["positive"][0]["minimax_refs"][0], linear_ref)
        self.assertIs(conds["positive"][0]["minimax_refs"][-1], ref)
        self.assertNotIn("minimax_keyframes", conds["positive"][0])
        self.assertEqual(len(original["positive"][0]["minimax_refs"]), 1)

    def test_video1_first_frame_summary_preserves_tasks_and_is_idempotent(self):
        """The exact picture role survives authored summaries and absent summaries."""
        for summary in ("summary: [video continuation + audio reference] Continue the scene.\n\n", "summary: Continue the scene.\n\n", ""):
            prompt = "subject_definitions:\n<Subject 1> is Alice.\n\n" + summary + "detailed_description: Alice talks.\noverall_soundscape: Wind."
            result = nodes._first_frame_picture_prompt(prompt, "<Picture 3>")
            self.assertIn("keyframe completion", result)
            self.assertEqual(result.count("<Picture 3> serves as first frame of target video."), 1)
            self.assertIn("detailed_description: Alice talks.\noverall_soundscape: Wind.", result)
            self.assertEqual(nodes._first_frame_picture_prompt(result, "<Picture 3>"), result)
            if "video continuation" in summary:
                self.assertIn("[video continuation + audio reference + keyframe completion]", result)

    def test_legacy_static_camera_checks_both_slices(self):
        """Motion in either description prevents the static-camera prefix."""
        still = "summary: Camera zooms in the source.\ndetailed_description: [Shot 1] She speaks."
        sentence = "The camera stays static in the stablished frame."
        result = nodes._legacy_static_camera_prompt(still)
        self.assertIn("[Shot 1] " + sentence + " She speaks.", result)
        self.assertEqual(nodes._legacy_static_camera_prompt(result), result)
        moving = "detailed_description: The camera slowly zooms out."
        self.assertEqual(nodes._legacy_static_camera_prompt(still, moving), still)
        self.assertEqual(nodes._legacy_static_camera_prompt(moving, still), moving)
        self.assertIn(sentence, nodes._legacy_static_camera_prompt("detailed_description: No camera movement. She runs."))

    def test_legacy_static_camera_replaces_the_repeated_full_shot_instruction(self):
        """A sliced static shot has one local camera instruction."""
        prompt = "detailed_description: The camera stays static at the same position from start to end of the shot. She speaks."
        result = nodes._legacy_static_camera_prompt(prompt)
        self.assertNotIn("at the same position from start to end", result)
        self.assertEqual(result.count("The camera stays static in the stablished frame."), 1)

    def test_legacy_dialogue_keeps_tags_and_continues_without_repeated_words(self):
        """Whole words have one owner; every emitted utterance closes its tags."""
        body = "Camera stays static. <Subject 1> (S1) says: <d>[English] One two three, four five six seven eight.</d>"
        fragments = [nodes._legacy_shot_body_for_range(body, 0, 240, start, start + 24, 24) for start in range(0, 240, 24)]
        words = []
        for fragment in fragments:
            self.assertEqual(fragment.count("<d>"), fragment.count("</d>"))
            if "<d>" in fragment:
                self.assertIn("<Subject 1> (S1) says: <d>[English]", fragment)
                words.extend(fragment.split("[English] ", 1)[1].split("</d>", 1)[0].split())
        self.assertEqual(words, "One two three, four five six seven eight.".split())
        self.assertNotEqual(fragments[0], fragments[1])

    def test_legacy_dialogue_normalizes_internal_line_breaks(self):
        """H3 receives one continuous spoken line per dialogue fragment."""
        body = "<Subject 1> (S1) says: <d>[English] one two.\n\nthree four.</d>"
        fragment = nodes._legacy_shot_body_for_range(body, 0, 240, 0, 240, 24)
        spoken = fragment.split("[English] ", 1)[1].split("</d>", 1)[0]
        self.assertEqual(spoken, "one two. three four.")

    def test_legacy_dialogue_uses_the_global_clock_without_restarting_later_chunks(self):
        """Later physical slices retain only their source-clock words and no opening silence."""
        body = "<Subject 1> starts the video in silence for 00:01.00, then (S1) says: <d>[English] One two three four five six seven eight nine ten.</d>"
        first = nodes._legacy_shot_body_for_range(body, 0, 240, 0, 48, 24)
        later = nodes._legacy_shot_body_for_range(body, 0, 240, 48, 96, 24)
        self.assertNotIn("already in progress", first)
        self.assertNotIn("already in progress", later)
        self.assertNotIn("start the video in silence", later.lower())
        self.assertNotIn("One two", later)

    def test_legacy_dialogue_honors_explicit_opening_silence_on_the_global_clock(self):
        """A source timing instruction delays speech once, instead of every chunk."""
        body = "<Subject 1> starts the video in silence for 00:01.00, then (S1) says: <d>[English] One two three four.</d>"
        silent = nodes._legacy_shot_body_for_range(body, 0, 240, 0, 24, 24)
        speaking = nodes._legacy_shot_body_for_range(body, 0, 240, 24, 72, 24)
        self.assertNotIn("<d>", silent)
        self.assertIn("<d>[English] One", speaking)
        self.assertNotIn("start the video in silence", speaking.lower())

    def test_phoneme_timed_dialogue_contract_is_word_exact_and_chunk_owned(self):
        """The preproduction verifier receives the same deterministic split as legacy sampling."""
        body = "<Subject 1> (S1) says: <d>[English] One two three four five six seven eight nine ten.</d>"
        chunks = [{"output_start": 0, "output_end": 120}, {"output_start": 120, "output_end": 240}]
        segments = nodes._deterministic_dialogue_segments(body, 0, 240, chunks, 24)
        words = [match.group(1).split("]", 1)[1].strip() for segment in segments for match in nodes.DIALOGUE_BLOCK.finditer(segment["content"])]
        self.assertEqual(" ".join(words).split(), "One two three four five six seven eight nine ten.".split())
        self.assertEqual([segment["chunk"] for segment in segments], [1, 2])
        source_shots, request = nodes._gemma_preproduction_request("detailed_description: " + body, [(0, 0, 240, body)], [{"frame_start": 0, "frame_end": 120, "output_trim_frames": 0}, {"frame_start": 120, "frame_end": 240, "output_trim_frames": 0}], 24, False)
        self.assertEqual(source_shots[0]["deterministic_dialogue_segments"], segments)
        self.assertEqual([(item["output_start"], item["output_end"]) for item in request["chunks"]], [(0, 120), (120, 240)])

    def test_dialogue_word_crossing_a_boundary_belongs_to_its_end_chunk(self):
        """A word may start in a prefix, but its complete sound belongs later."""
        body = "<Subject 1> (S1) says: <d>[English] one two.</d>"
        words = list(nodes.re.finditer(r"\S+", "one two."))
        with patch.object(nodes, "_dialogue_word_weights", return_value=tuple(zip(words, (13.0, 11.0)))):
            _visual, first = nodes._legacy_dialogue_for_range(body, 0, 240, 0, 120, 1)
            _visual, second = nodes._legacy_dialogue_for_range(body, 0, 240, 120, 240, 1)
        self.assertEqual(first, [])
        self.assertIn("one two.", second[-1])

    def test_dialogue_prompt_includes_words_spoken_during_the_carried_prefix(self):
        """The resampled boundary receives its prior chunk's tail dialogue."""
        body = "<Subject 1> (S1) says: <d>[English] one two three four.</d>"
        words = list(nodes.re.finditer(r"\S+", "one two three four."))
        ranges = ((0, 20), (20, 40))
        with patch.object(nodes, "_dialogue_word_weights", return_value=tuple(zip(words, (10.0, 10.0, 10.0, 10.0)))):
            _visual, first = nodes._legacy_dialogue_for_range(body, 0, 40, 0, 20, 1, ranges)
            _visual, second = nodes._legacy_dialogue_for_range(body, 0, 40, 20, 40, 1, ranges, 10)
        self.assertIn("one two", first[-1])
        self.assertEqual(second[0], "The dialogue is already in progress at this video opening.")
        self.assertIn("two three four.", second[-1])

    def test_taomate_sentence_allocation_and_no_prefix_repetition(self):
        """Whole sentences survive unequal ranges and physical decoding halos."""
        speech = "one two three. four five. six seven eight."
        body = "<Subject 1> (S1) says: <d>[English] " + speech + "</d>"
        words = list(nodes.re.finditer(r"\S+", speech))
        ranges = ((0, 5), (5, 10), (10, 15))
        parts = []
        with patch.object(nodes, "_dialogue_word_weights", return_value=tuple((word, 1.0) for word in words)):
            for start, end in ranges:
                _visual, lines = nodes._legacy_dialogue_for_range(body, 0, 15, start, end, 1, ranges, max(0, start - 2), sentence_chunks=True)
                parts.append(" ".join(nodes.re.findall(r"<d>\[English\] (.*?)</d>", " ".join(lines))))
        self.assertEqual(parts, ["one two three.", "four five.", "six seven eight."])
        self.assertEqual(" ".join(parts), speech)

    def test_taomate_sentence_split_threshold_and_short_final_chunk(self):
        """Only sentences strictly above 1.5 nominal chunks allow internal cuts."""
        import importlib
        allocate = importlib.import_module(nodes.__package__ + ".python.dialogue_timing").sentence_word_owners
        self.assertEqual(allocate([[1] * 15, [1] * 5], [10, 10]), [0] * 15 + [1] * 5)
        owners = allocate([[1] * 16, [1] * 4], [10, 10])
        self.assertEqual(set(owners[:16]), {0, 1})
        self.assertEqual(owners, sorted(owners))
        self.assertEqual(len(owners), 20)
        # The short final group cannot force a short sentence to split.
        owners = allocate([[1] * 4, [1] * 4], [5, 5, 0.5])
        self.assertEqual(len(set(owners[:4])), 1)
        self.assertEqual(len(set(owners[4:])), 1)

    def test_dialogue_uses_every_output_chunk_in_its_source_shot(self):
        """Shot duration controls speaking speed without silent dialogue chunks."""
        body = "<Subject 1> (S1) says: <d>[English] one two three four five six.</d>"
        chunks = [{"output_start": 0, "output_end": 80}, {"output_start": 80, "output_end": 160}, {"output_start": 160, "output_end": 240}]
        segments = nodes._deterministic_dialogue_segments(body, 0, 240, chunks, 24)
        self.assertEqual([segment["chunk"] for segment in segments], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "every dialogue chunk"):
            nodes._deterministic_dialogue_segments("<Subject 1> (S1) says: <d>[English] one two.</d>", 0, 240, chunks, 24)

    def test_legacy_dialogue_rejects_malformed_or_impossible_speech(self):
        """Do not silently drop unparseable speech or overflow its shot."""
        with self.assertRaisesRegex(ValueError, "complete"):
            nodes._legacy_shot_body_for_range("<Subject 1> (S1) says: <d>[English] Hello", 0, 240, 0, 24)
        with self.assertRaisesRegex(ValueError, "estimates"):
            nodes._legacy_shot_body_for_range("<Subject 1> (S1) says: <d>[English] This sentence cannot fit.</d>", 0, 5, 0, 5)

    def test_preproduction_node_is_lazy_and_creates_fresh_director_sessions(self):
        """Graph execution configures Gemma; only sampler calls create sessions."""
        module = importlib.import_module(PLUGIN_ROOT.name + ".python.preproduction")
        with patch.object(module, "Gemma4ContinuityDirector", side_effect=lambda **kwargs: SimpleNamespace(settings=kwargs)) as factory:
            provider = module.HREndlessGemmaPreProduction.execute(production_think_budget=0, chunk_think_budget=1024).result[0]
            factory.assert_not_called()
            context = {"prompt": "source", "fps": 24, "latent": object()}
            first = provider.create_session(render_context=context, seed=42)
            second = provider.create_session(render_context=context, seed=42)
        self.assertIsNot(first, second)
        self.assertIs(first.render_context, context)
        self.assertEqual(first.settings["production_think_budget"], 0)
        self.assertEqual(first.settings["chunk_think_budget"], 1024)
        self.assertEqual(first.settings["seed"], 42)
        self.assertNotEqual(provider.fingerprint(), module.GemmaPreProduction().fingerprint())
        self.assertEqual([item.id for item in module.HREndlessGemmaPreProduction.define_schema().inputs], ["cache_gemma_preproduction", "gemma4_mtp", "production_think_budget", "chunk_think_budget"])

    def test_legacy_prompt_slicing_selects_the_proportional_action(self):
        """Without a director, restore the old deterministic word-count method."""
        body = "Tiger runs. Tiger stops."
        self.assertEqual(nodes._legacy_shot_body_for_range(body, 0, 100, 0, 50).strip(), "Tiger runs.")
        self.assertEqual(nodes._legacy_shot_body_for_range(body, 0, 100, 50, 100).strip(), "Tiger stops.")
        self.assertEqual(nodes._legacy_shot_body_for_range(body, 0, 100, 0, 100), body)

    def test_sampler_compute_precision_is_local_and_default_is_unchanged(self):
        """FP32 reaches sampling through a clone without changing the upstream guider."""
        model = Mock(load_device=torch.device("cpu"), model_options={"original": True})
        model.clone.return_value = Mock(model_options={"cloned": True})
        guider = SimpleNamespace(model_patcher=model, model_options=model.model_options, cfg=4.0, original_conds={})
        latent = {"samples": torch.zeros((1, 4, 1, 1))}
        # Exercise the real execute entry point while skipping GPU sampling and cache notifications.
        with patch.object(nodes, "_set_pytorch_memory_fraction"), patch.object(nodes, "_replay_cache_activity"), patch.object(nodes.SamplerCustomAdvanced, "execute", return_value=(latent, latent)) as sample:
            nodes.HREndlessSampler.execute(None, guider, None, None, latent, None, "", compute_precision="fp32 (full precision but more VRAM needed)")
            local_guider = sample.call_args.args[1]
            self.assertIsNot(local_guider, guider)
            self.assertIs(local_guider.model_patcher, model.clone.return_value)
            self.assertIs(local_guider.model_options, local_guider.model_patcher.model_options)
            self.assertEqual(local_guider.cfg, 4.0)
            local_guider.model_patcher.set_model_compute_dtype.assert_called_once_with(torch.float32)
            self.assertIs(guider.model_patcher, model)
            self.assertIs(guider.model_options, model.model_options)
            model.set_model_compute_dtype.assert_not_called()
            nodes.HREndlessSampler.execute(None, guider, None, None, latent, None, "")
            self.assertIs(sample.call_args.args[1], guider)
            model.clone.assert_called_once()
            with self.assertRaises(ValueError):
                nodes.HREndlessSampler.execute(None, guider, None, None, latent, None, "", compute_dtype="invalid")

    def test_cached_preview_reports_encoding_progress(self):
        """A dormant restore reports start and completion for its frame group."""
        updates = []
        chunk = {"index": 0, "frames": torch.zeros((1, 2, 2, 3)), "output_start": 0}
        snapshot = preview.build_cached_final_preview_snapshot("restore-progress-test", [{"chunk": 1, "start": 0, "end": 0}], [], [chunk], fps=24, progress_callback=lambda done, total: updates.append((done, total)))
        self.assertEqual(updates, [(0, 1), (1, 1)])
        self.assertEqual(len(snapshot["chunks"]), 1)

    def test_allocator_setting_is_not_a_workflow_input(self):
        """Old hidden allocator values must not participate in validation."""
        schema = nodes.HREndlessSampler.define_schema()
        self.assertNotIn("pytorch_memory_fraction", [item.id for item in schema.inputs])

    def test_gemma_summary_replaces_global_summary_after_reference_injection(self):
        """The final summary is the authored chunk paragraph, exactly once."""
        prompt = "subject_definitions:\n<Subject 1> is Alice.\n\nsummary: Old summary.\nOld second line.\n\nretention_analysis:\nKeep identity.\n\ndetailed_description: Old action.\noverall_soundscape: Wind."
        summary = "[video continuation + audio reference] Continue <Video 2>, using <Audio 3> for voice continuity."
        result = nodes._prompt_with_gemma_description(prompt, "Alice talks.", continuation_video_label="<Video 2>", continuation_audio_label="<Audio 3>", summary=summary)
        self.assertEqual(result.count("summary:"), 1)
        self.assertIn("summary: " + summary, result)
        self.assertNotIn("Old second line", result)
        self.assertIn("<Audio 3> is", result)
        self.assertIn("retention_analysis:", result)

    def test_static_grade_deduplication_ignores_case_punctuation_and_spacing(self):
        """Equivalent wording is kept once, without confusing picture IDs."""
        shots = [(0, 0, 100, "Using <Picture 1> as the exact first frame, the camera stays static.")]
        camera = "The camera remains static in the established framing. "
        for instruction in (
            "Use the lighting and color grading from <Picture 1>.",
            "USE THE LIGHTING, AND COLOR-GRADING FROM <PICTURE   1>!",
            "use\n the lighting and\tcolor grading from <Picture1>",
        ):
            description = camera + instruction
            self.assertEqual(nodes._reinforce_static_shot_grade(description, shots, 39), description)
        description = camera + "Use the lighting and color grading from <Picture 10>."
        self.assertIn("Use the lighting and color grading from <Picture 1>.", nodes._reinforce_static_shot_grade(description, shots, 39))

    def test_prefix_noise_reuses_the_previous_full_sequence_tail(self):
        """Prefix and retained noise form one contiguous full-sequence slice."""
        video = torch.arange(20).reshape(1, 1, 20, 1, 1)
        audio = torch.arange(40).reshape(1, 1, 1, 40)
        prefix_video, prefix_audio = nodes._full_noise_prefix(video, audio, 7, 16, 2, 4)
        self.assertTrue(torch.equal(torch.cat((prefix_video, video[:, :, 7:12]), dim=2), video[:, :, 5:12]))
        self.assertTrue(torch.equal(torch.cat((prefix_audio, audio[..., 16:24]), dim=-1), audio[..., 12:24]))
        next_video, _ = nodes._full_noise_prefix(video, audio, 12, 24, 2, 4)
        self.assertFalse(torch.equal(prefix_video, next_video))
        with self.assertRaises(ValueError):
            nodes._full_noise_prefix(video, audio, 1, 16, 2, 4)

    def test_independent_chunk_noise_uses_one_based_seed_multiples(self):
        self.assertFalse(nodes.TOGGLE_SINGLE_NOISE)
        self.assertEqual([nodes._per_chunk_noise_seed(17, index) for index in range(3)], [17, 34, 51])
        self.assertEqual([nodes._per_chunk_noise_seed(0, index) for index in range(3)], [0, 0, 0])

    def test_color_mode_disabled_preserves_pixels_and_visible_prefix_is_corrected(self):
        """Disabled grading is identity; zero trim grades the first visible frame."""
        frames = torch.full((8, 2, 2, 3), 0.2)
        reference = torch.full((1, 2, 2, 3), 0.3)
        unchanged, transform, count = nodes._correct_decoded_chunk_color(reference, frames, 0, enabled=False)
        self.assertIs(unchanged, frames)
        self.assertIsNone(transform)
        self.assertEqual(count, 0)
        corrected, transform, count = nodes._correct_decoded_chunk_color(reference, frames, 0)
        self.assertEqual(count, 8)
        self.assertFalse(torch.equal(corrected[0], frames[0]))
        self.assertTrue(torch.equal(corrected[0], corrected[5]))

    def test_static_continuation_uses_its_source_first_picture_grade(self):
        """Use the actual shot anchor only after that shot has started."""
        shots = [(0, 0, 100, "Using <Picture 4> as the exact first frame, the camera stays static.")]
        description = "The camera remains static in the established framing. She walks."
        result = nodes._reinforce_static_shot_grade(description, shots, 39)
        self.assertIn("Use the lighting and color grading from <Picture 4>.", result)
        self.assertNotIn("first frame", result)
        self.assertEqual(nodes._reinforce_static_shot_grade(description, shots, 0), description)
        self.assertEqual(nodes._reinforce_static_shot_grade("The camera pans.", shots, 39), "The camera pans.")

    def test_no_trim_preview_retains_packed_prefixes_sequentially(self):
        plan = [
            {"frame_start": 0, "frame_end": 243, "output_trim_frames": 0},
            {"frame_start": 238, "frame_end": 481, "output_trim_frames": 5},
            {"frame_start": 476, "frame_end": 719, "output_trim_frames": 5},
        ]
        self.assertEqual(nodes._preview_ranges_for_plan(plan, keep_physical_prefix=True), [
            {"chunk": 1, "start": 0, "end": 242},
            {"chunk": 2, "start": 243, "end": 485},
            {"chunk": 3, "start": 486, "end": 728},
        ])
        marked = nodes._preview_ranges_for_plan(plan, show_replaced_tail=True)
        self.assertEqual(marked[0]["replacement_overlap_frames"], 5)
        self.assertEqual(marked[1]["replacement_overlap_frames"], 5)
        self.assertNotIn("replacement_overlap_frames", marked[2])

    def test_dormant_preview_restores_finalized_cpu_media_from_replay_cache(self):
        """A browser refresh must not require a model execution to show cache."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "last_run_replay"
            fingerprint = {
                "fps": 24.0,
                "linear_color_compute": True,
                "plan": [{"frame_start": 0, "frame_end": 2, "output_trim_frames": 0}],
            }
            state = {
                "decoded_video_frames": torch.zeros((2, 2, 2, 3)),
                "corrected_video_frames": torch.full((2, 2, 2, 3), 0.25),
                "decoded_preview_start": 0,
                "decoded_preview_end": 1,
                "decoded_preview_offset": 0,
                "gemma_description": "cached prompt",
                "chunk_total_seconds": 3.0,
            }
            with patch.object(nodes, "_replay_cache_root", return_value=root), \
                    patch.object(nodes, "REPLAY_CACHE_ENABLED", True), \
                    patch.object(nodes, "_REPLAY_CACHE_ACTIVE_RUNS", 0), \
                    patch.object(nodes, "load_replay_preview_proxy", return_value=(state["corrected_video_frames"], None, None)):
                cache = nodes._LastRunReplayCache()
                cache.create(fingerprint, "source", {"video": torch.zeros(1)})
                cache.save_chunk(1, state)
                # A cache written before the sRGB marker held compute RGB in
                # the browser proxy. Verify refresh repairs that legacy data.
                legacy = cache.load_chunk(1)
                legacy.pop("preview_proxy_color_space")
                nodes._replay_write_tensor_file(cache.chunk_path(1), legacy)
                with patch.object(nodes, "build_cached_final_preview_snapshot", return_value={"reset": {}}) as build:
                    snapshot = nodes._cached_replay_preview_snapshot(
                        "141",
                        max_resolution=0,
                        quality=75,
                        fps=30.0,
                    )

        self.assertEqual(snapshot, {"reset": {}})
        self.assertEqual(build.call_args.kwargs["fps"], 24.0)
        self.assertEqual(build.call_args.args[1][0]["gemma_detailed_description"], "cached prompt")
        self.assertTrue(torch.allclose(
            build.call_args.args[3][0]["frames"],
            nodes._convert_image_transfer(state["corrected_video_frames"], nodes._inverse_gamma_compute_to_srgb),
        ))

    def test_preview_reuses_live_cache_status_for_timeline_underlines(self):
        """The cache button's manifest count must also drive its timeline marks."""
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn("if (cacheStatus?.has_cache)", source)
        self.assertIn("cachedChunkIndices = new Set(cachedChunks", source)
        self.assertIn("cachedChunkCount = cachedChunkIndices.size;", source)
        self.assertIn("offset += spans[index];\n                    const end", source)
        self.assertIn("if (!cachedChunkIndices.has(index)) continue;", source)
        self.assertIn("renderTransport();\n                } catch (error)", source)

    def test_preview_marks_replaced_tail_and_preloads_the_next_chunk_audio(self):
        """The timeline and audio transport expose native continuation ownership."""
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn("replacement_overlap_frames", source)
        self.assertIn("renderOverlapReplacementLines", source)
        self.assertIn("const color = chunkColors[(index + 1) % chunkColors.length];", source)
        self.assertIn('const alpha = available(index + 1) ? "" : "35";', source)
        self.assertIn("function preloadAudio(group)", source)
        self.assertIn("audioStandbyPlayer", source)
        self.assertIn("const shotNumber = Number(shot?.shot) || 1;", source)

    def test_preview_inverse_gamma_toggle_applies_only_finalized_decodes(self):
        """The browser-only display curve must never alter latent step previews."""
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn('linearDisplayButton.textContent = "L";', source)
        self.assertIn('filter.setAttribute("color-interpolation-filters", "sRGB");', source)
        self.assertIn('const enabled = inverseGammaDisplay && finalized;', source)
        self.assertIn('Boolean(group.finalized),', source)
        self.assertIn('\n                        false,\n                    );', source)

    def test_preview_cache_context_menu_targets_the_clicked_chunk(self):
        """Global reconnect remains available beside a local cache action."""
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn('root.addEventListener("contextmenu"', source)
        self.assertIn("chunkTooltip.hide();", source)
        self.assertIn('reconnectButton.textContent = "Reconnect"', source)
        self.assertIn("await restoreServerState(0, true);", source)
        self.assertIn('divider.style.cssText = "height:1px', source)
        self.assertIn("showPreviewContextMenu(chunkIndex, event);", source)
        self.assertIn("replay_cache_chunk?chunk=${chunkNumber}", source)
        self.assertIn("Delete cached chunk", source)

    def test_preview_names_the_exact_reused_chunks_in_cache_green(self):
        """The on-image cache notice must report the actual reused chunk list."""
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn("color:#69b76f", source)
        self.assertIn("reusing cached chunks ${chunks}", source)
        self.assertIn("data.reused_chunk_numbers", source)

    def test_connected_save_video_prefix_comes_from_this_sampler_timeline(self):
        class DynamicPrompt:
            @staticmethod
            def get_original_prompt():
                return {
                    "141": {"class_type": "HREndlessSampler", "inputs": {}},
                    "144": {
                        "class_type": "HREndlessSamplerSaveVideo",
                        "inputs": {
                            "timeline": ["141", 3],
                            "filename_prefix": "movies/my_workflow",
                        },
                    },
                    "145": {
                        "class_type": "HREndlessSamplerSaveVideo",
                        "inputs": {
                            "timeline": ["999", 3],
                            "filename_prefix": "wrong_sampler",
                        },
                    },
                }

        self.assertEqual(
            nodes._connected_save_video_prefix(DynamicPrompt(), "141"),
            "movies/my_workflow",
        )

    def test_connected_save_video_prefix_resolves_a_string_primitive(self):
        class DynamicPrompt:
            original_prompt = {
                "12": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "film/scene"}},
                "20": {
                    "class_type": "HREndlessSamplerSaveVideo",
                    "inputs": {"timeline": ["7", 3], "filename_prefix": ["12", 0]},
                },
            }

        self.assertEqual(nodes._connected_save_video_prefix(DynamicPrompt(), 7), "film/scene")

    def test_only_last_dialogue_terminal_punctuation_is_normalized_for_h3(self):
        cases = (
            (
                "Tila says: <d>[English] Wait...</d> Heman says: <d>[English] Run...</d>",
                "Tila says: <d>[English] Wait...</d> Heman says: <d>[English] Run.</d>",
            ),
            (
                "Heman says: <d>[English] Keep moving,</d>",
                "Heman says: <d>[English] Keep moving.</d>",
            ),
            (
                "Heman says: <d>[English] I understand…</d>",
                "Heman says: <d>[English] I understand.</d>",
            ),
            (
                "Heman says: <d>[English] Are you sure?</d>",
                "Heman says: <d>[English] Are you sure?</d>",
            ),
            (
                "Heman says: <d>[English] Stop!</d>",
                "Heman says: <d>[English] Stop!</d>",
            ),
            (
                "Heman says: <d>[English] Keep moving</d>",
                "Heman says: <d>[English] Keep moving</d>",
            ),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(
                    nodes._normalize_last_dialogue_terminal_punctuation(source),
                    expected,
                )

    def test_replay_preview_uses_full_vae_frames_and_decoded_audio(self):
        decoded, retained, waveform, sample_rate, overlap = nodes._decode_replay_preview_media(
            _IndexedFakeVAE(),
            _FakeAudioVAE(),
            torch.zeros((1, 24, 11, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 16), dtype=torch.float32),
            output_trim_frames=5,
            context_audio_t=2,
            output_frames=22,
            fps=24.0,
        )

        self.assertEqual(decoded.shape[0], 39)
        self.assertEqual(retained.shape[0], 22)
        self.assertEqual(retained[0, 0, 0, 0].item(), 5.0)
        self.assertEqual(retained[-1, 0, 0, 0].item(), 26.0)
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(waveform.shape, (1, 2, round(22 * 8000 / 24.0)))
        self.assertIsNone(overlap)

    def test_replay_masked_audio_is_split_into_retroactive_overlap_and_retained_output(self):
        _, _, waveform, sample_rate, overlap = nodes._decode_replay_preview_media(
            _IndexedFakeVAE(),
            _FakeAudioVAE(),
            torch.zeros((1, 24, 11, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 16), dtype=torch.float32),
            output_trim_frames=5,
            context_audio_t=8,
            output_frames=22,
            fps=24.0,
            masked_audio_overlap_frames=5,
        )

        self.assertEqual(sample_rate, 8000)
        self.assertEqual(overlap.shape[-1], round(5 * 8000 / 24.0))
        self.assertEqual(waveform.shape[-1], round(22 * 8000 / 24.0))

    def test_replay_native_audio_tail_uses_exact_audio_latent_ticks(self):
        _, _, waveform, sample_rate, overlap = nodes._decode_replay_preview_media(
            _IndexedFakeVAE(),
            _FakeAudioVAE(),
            torch.zeros((1, 24, 11, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 16), dtype=torch.float32),
            output_trim_frames=5,
            context_audio_t=8,
            output_frames=22,
            fps=24.0,
            replace_audio_tail_ticks=True,
        )
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(overlap.shape[-1], round(8 * 8000 / nodes.AUDIO_LATENT_FPS))
        self.assertEqual(waveform.shape[-1], round(22 * 8000 / 24.0))

    def test_masked_av_prefix_inspection_keeps_the_locked_prefix(self):
        decoded, retained, waveform, sample_rate, overlap = nodes._decode_replay_preview_media(
            _IndexedFakeVAE(),
            _FakeAudioVAE(),
            torch.zeros((1, 24, 11, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1, 16), dtype=torch.float32),
            output_trim_frames=5,
            context_audio_t=0,
            output_frames=27,
            fps=24.0,
            masked_audio_overlap_frames=0,
            keep_masked_av_prefix=True,
        )

        self.assertEqual(sample_rate, 8000)
        self.assertIsNone(overlap)
        self.assertEqual(decoded.shape[0], 39)
        self.assertEqual(retained.shape[0], 32)
        self.assertEqual(waveform.shape[-1], round(32 * 8000 / 24.0))

    def test_preview_execution_ids_survive_wrapper_and_server_recreation(self):
        """Fresh wrappers and restarted counters cannot reuse an older render ID."""
        first = preview._AccumulatedPreviewWrapper("restart", 0, 80, 24, 1, "none")
        second = preview._AccumulatedPreviewWrapper("restart", 0, 80, 24, 1, "none")
        with patch.object(preview, "_send"), patch.object(preview, "_PREVIEW_EXECUTION_ID", 0), patch.object(preview.time, "time_ns", return_value=1800000000000000000):
            a = first.begin(1)
            b = first.begin(1)
            c = second.begin(1)
            self.assertLess(a, b)
            self.assertLess(b, c)
            # Simulate a new process with its counter reset and a later clock.
            preview._PREVIEW_EXECUTION_ID = 0
            with patch.object(preview.time, "time_ns", return_value=1800000001000000000):
                d = second.begin(1)
            self.assertLess(c, d)
            self.assertLess(d, 2 ** 53)

    def test_preview_audio_overlap_replaces_prior_groups_and_updates_cached_final_audio(self):
        wrapper = preview._AccumulatedPreviewWrapper(
            node_id="preview-audio-overlap-test",
            max_resolution=0,
            quality=80,
            fps=24.0,
            frame_stride=1,
            tiny_vae="none",
        )
        payloads = []
        with patch.object(preview, "_send", side_effect=lambda payload: payloads.append(payload)):
            ranges = [
                {"chunk": 1, "start": 0, "end": 3},
                {"chunk": 2, "start": 4, "end": 5},
                {"chunk": 3, "start": 6, "end": 8},
            ]
            execution_id = wrapper.begin(ranges, reusing_cached_chunks=True, cached_chunk_count=2, reused_chunk_numbers=(1, 3))
            wrapper.final_audio = {
                0: torch.zeros((1, 1, 4), dtype=torch.float32),
                1: torch.ones((1, 1, 2), dtype=torch.float32),
            }
            wrapper.final_audio_rates = {0: 8000, 1: 8000}
            replacement = torch.arange(10, 15, dtype=torch.float32).reshape(1, 1, 5)
            wrapper.replace_audio_tail(execution_id, 2, replacement, 8000)

        self.assertEqual(wrapper.final_audio[0].flatten().tolist(), [0.0, 10.0, 11.0, 12.0])
        self.assertEqual(wrapper.final_audio[1].flatten().tolist(), [13.0, 14.0])
        updates = [payload for payload in payloads if payload.get("action") == "chunk_audio_update"]
        self.assertEqual([payload["chunk"] for payload in updates], [0, 1])
        reset = next(payload for payload in payloads if payload.get("action") == "reset")
        self.assertTrue(reset["reusing_cached_chunks"])
        self.assertEqual(reset["cached_chunk_count"], 2)
        self.assertEqual(reset["reused_chunk_numbers"], [1, 3])

    def test_preview_shift_tooltip_receives_exact_h3_prompt(self):
        wrapper = preview._AccumulatedPreviewWrapper(
            node_id="preview-h3-prompt-test",
            max_resolution=0,
            quality=80,
            fps=24.0,
            frame_stride=1,
            tiny_vae="none",
        )
        payloads = []
        with patch.object(preview, "_send", side_effect=lambda payload: payloads.append(payload)):
            execution_id = wrapper.begin([{"chunk": 1, "start": 0, "end": 4}])
            wrapper.set_chunk(execution_id, 0, 0, 4, 0, 4, 0, "short description", None, "complete H3 prompt")

        metadata = next(payload for payload in payloads if payload.get("action") == "chunk_metadata")
        self.assertEqual(metadata["h3_prompt"], "complete H3 prompt")
        source = (PLUGIN_ROOT / "web" / "unlimited_preview.js").read_text(encoding="utf-8")
        self.assertIn('const showFullPrompt = Boolean(event.shiftKey || shiftPromptVisible);', source)
        self.assertIn('showFullPrompt ? "Full prompt sent to H3:"', source)

        node_id = "preview-audio-update-cache-test"
        with preview._PREVIEW_CACHE_LOCK:
            preview._PREVIEW_CACHE.pop(node_id, None)
        try:
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "reset",
                "chunk_ranges": [{"chunk": 1, "start": 0, "end": 3}],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "chunk_final",
                "chunk": 0,
                "frames": ["frame0"],
                "audio": "old-wav",
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "chunk_audio_update",
                "chunk": 0,
                "audio": "new-wav",
                "audio_sample_rate": 8000,
            })
            cached = preview._cached_snapshot(node_id)["chunks"][0]
            self.assertEqual(cached["frames"], ["frame0"])
            self.assertEqual(cached["audio"], "new-wav")
            self.assertEqual(cached["audio_sample_rate"], 8000)
        finally:
            with preview._PREVIEW_CACHE_LOCK:
                preview._PREVIEW_CACHE.pop(node_id, None)

    def test_gemma_preproduction_uses_full_plan_when_render_is_debug_limited(self):
        full_plan = [
            {
                "frame_start": start,
                "frame_end": end,
                "output_trim_frames": 0 if index == 0 else 39,
            }
            for index, (start, end) in enumerate(
                (
                    (0, 243),
                    (204, 447),
                    (408, 651),
                    (612, 855),
                    (816, 1059),
                    (1020, 1263),
                    (1224, 1450),
                )
            )
        ]
        debug_limited_plan = full_plan[:3]
        shots = [(0, 0, 1450, "A sixty-second monologue.")]

        source_shots, request = nodes._gemma_preproduction_request(
            "detailed_description: [Shot 1] A sixty-second monologue.",
            shots,
            full_plan,
            24.0,
            True,
        )

        self.assertEqual(len(debug_limited_plan), 3)
        self.assertEqual(request["chunk_count"], 7)
        self.assertEqual(len(request["chunks"]), 7)
        self.assertEqual(request["chunks"][-1]["output_end"], 1450)
        self.assertEqual(request["chunks"][1]["output_start"], 243)
        self.assertEqual(source_shots[0]["shot_end"], 1450)

    def test_hr_endless_sampler_schema_hides_retired_experiments_and_puts_debug_last(self):
        schema = nodes.HREndlessSampler.define_schema()
        input_ids = [item.id for item in schema.inputs]

        self.assertEqual(schema.node_id, "HREndlessSampler")
        self.assertEqual(schema.display_name, "HR Endless Sampler")
        self.assertIn("video_continuation", input_ids)
        self.assertIn("audio_vae", input_ids)
        self.assertEqual(input_ids[input_ids.index("video_continuation") + 1], "video_continuation_method")
        self.assertEqual(input_ids[input_ids.index("video_continuation_method") + 1], "video_continuation_res")
        video_continuation_input = next(item for item in schema.inputs if item.id == "video_continuation")
        self.assertEqual(video_continuation_input.default, 22)
        continuation_method_input = next(item for item in schema.inputs if item.id == "video_continuation_method")
        self.assertEqual(continuation_method_input.default, nodes.VIDEO_CONTINUATION_METHOD_VIDEO1)
        self.assertEqual(tuple(continuation_method_input.options), nodes.VIDEO_CONTINUATION_METHODS)
        self.assertIn("pre_production", input_ids)
        linear_color_input = next(item for item in schema.inputs if item.id == "linear_color_compute")
        self.assertFalse(linear_color_input.default)
        audio_feather_input = next(item for item in schema.inputs if item.id == "audio_feathered_overlap")
        self.assertFalse(audio_feather_input.default)
        self.assertNotIn("cache_gemma_preproduction", input_ids)
        self.assertNotIn("gemma4_mtp", input_ids)
        self.assertFalse(nodes.ENABLE_GEMMA4_MTP)
        self.assertNotIn("pytorch_memory_fraction", input_ids)
        self.assertNotIn("video_continuation_enable", input_ids)
        self.assertFalse({
            "context_keyframes_enable",
            "context_keyframes",
            "guide_overlap_enable",
            "guide_overlap",
            "qwen_full_history",
            "prompt_preview_only",
        } & set(input_ids))
        self.assertEqual(
            input_ids[-4:],
            [
                "pre_production",
                "debug",
                "debug_stop_chunk",
                "debug_start_chunk",
            ],
        )

        execute_params = inspect.signature(nodes.HREndlessSampler.execute).parameters
        self.assertNotIn("video_continuation_enable", execute_params)
        self.assertNotIn("context_keyframes_enable", execute_params)
        self.assertNotIn("guide_overlap_enable", execute_params)
        self.assertNotIn("qwen_full_history", execute_params)
        self.assertNotIn("prompt_preview_only", execute_params)
        self.assertIn("debug_start_chunk", execute_params)
        self.assertNotIn("cache_gemma_preproduction", execute_params)
        self.assertNotIn("gemma4_mtp", execute_params)
        self.assertIsNone(execute_params["pre_production"].default)
        self.assertFalse(execute_params["audio_feathered_overlap"].default)
        self.assertNotIn("pytorch_memory_fraction", execute_params)
        self.assertEqual(execute_params["video_continuation"].default, 22)
        self.assertEqual(
            execute_params["video_continuation_method"].default,
            nodes.VIDEO_CONTINUATION_METHOD_VIDEO1,
        )
        self.assertEqual(
            nodes.VIDEO_CONTINUATION_RESOLUTIONS,
            (
                "full",
                "0.98mp (1344x768 native)",
                "0.90mp (1280x736)",
                "0.80mp (1216x672)",
                "0.70mp (1152x640)",
                "0.60mp (1056x608)",
                "0.50mp (960x544)",
                "0.40mp (864x480)",
                "0.30mp (736x416)",
                "0.20mp (608x352)",
                "0.10mp (448x256)",
            ),
        )

    def test_pytorch_memory_fraction_sets_explicit_cuda_allocator_limit(self):
        properties = type("DeviceProperties", (), {"total_memory": 16 * 1024 ** 3})()
        with patch.object(nodes.torch.cuda, "is_available", return_value=True), \
                patch.object(nodes.torch.cuda, "set_per_process_memory_fraction") as setter, \
                patch.object(nodes.torch.cuda, "get_device_properties", return_value=properties), \
                patch.object(nodes.torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"):
            result = nodes._set_pytorch_memory_fraction(0.85, torch.device("cuda:0"))

        setter.assert_called_once_with(0.85, device=torch.device("cuda:0"))
        self.assertEqual(result["fraction"], 0.85)
        self.assertEqual(result["limit_bytes"], int(16 * 1024 ** 3 * 0.85))
        self.assertEqual(result["backend"], "cudaMallocAsync")

    def test_pytorch_memory_fraction_skips_non_cuda_runtime(self):
        with patch.object(nodes.torch.cuda, "is_available", return_value=False), \
                patch.object(nodes.torch.cuda, "set_per_process_memory_fraction") as setter:
            self.assertIsNone(nodes._set_pytorch_memory_fraction(0.85, torch.device("cpu")))
        setter.assert_not_called()

    def test_last_run_replay_cache_keeps_exact_latents_and_moves_decodes_to_proxy(self):
        video = torch.arange(24, dtype=torch.float16).reshape(1, 1, 2, 3, 4)
        audio = torch.arange(16, dtype=torch.float16).reshape(1, 1, 2, 8)
        plan = [{
            "frame_start": 0,
            "frame_end": 5,
            "video_start": 0,
            "video_end": 2,
            "audio_start": 0,
            "audio_end": 8,
            "context_video_t": 0,
            "context_audio_t": 0,
            "output_trim_frames": 0,
            "synthetic_prefix": False,
        }]
        fingerprint = nodes._replay_fingerprint(
            video,
            audio,
            plan,
            fps=24.0,
            chunk_frames=5,
            context_keyframes=0,
            guide_overlap=0,
            video_continuation=0,
            video_continuation_res="full",
            ref2va=False,
        )
        self.assertEqual(
            fingerprint["video_continuation_method"],
            nodes.VIDEO_CONTINUATION_METHOD_VIDEO1,
        )
        decoded_frames = torch.arange(60, dtype=torch.float32).reshape(5, 2, 2, 3)
        corrected_frames = decoded_frames + 0.25
        decoded_audio = torch.arange(100, dtype=torch.float32).reshape(1, 2, 50)
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes, "save_replay_preview_proxy") as save_proxy:
            cache = nodes._LastRunReplayCache()
            cache.create(
                fingerprint,
                "original prompt",
                {
                    "video": video,
                    "audio": audio,
                    "video_noise": video + 1,
                    "audio_noise": audio + 1,
                    "noise_seed": 123,
                },
            )
            cache.save_chunk(1, {
                "sampled_video": video,
                "sampled_audio": audio,
                "previous_frame_count": 5,
                "output_video": video,
                "output_audio": audio,
                "denoised_video": video,
                "denoised_audio": audio,
                "output_template": {"batch_index": torch.tensor([0])},
                "denoised_template": {},
                "gemma_description": "directed prompt",
                "gemma_timing_plan": "timing",
                "gemma_end_state": "end",
                "debug_prompt": "chunk debug",
                "prefix_video_noise": None,
                "prefix_audio_noise": None,
                "decoded_video_frames": decoded_frames,
                "corrected_video_frames": corrected_frames,
                "decoded_preview_start": 0,
                "decoded_preview_end": 4,
                "decoded_preview_offset": 0,
                "decoded_preview_audio": decoded_audio,
                "decoded_preview_audio_rate": 48000,
                "decoded_preview_overlap_audio": None,
            }, fps=24.0)
            loaded, reason = cache.load_if_compatible(fingerprint)
            self.assertIsNone(reason)
            self.assertEqual(loaded["initial"]["noise_seed"], 123)
            self.assertEqual(loaded["initial"]["video"].device.type, "cpu")
            cached_chunk = cache.load_chunk(1)
            self.assertTrue(torch.equal(cached_chunk["sampled_video"], video.cpu()))
            self.assertNotIn("decoded_video_frames", cached_chunk)
            self.assertNotIn("corrected_video_frames", cached_chunk)
            self.assertNotIn("decoded_preview_audio", cached_chunk)
            self.assertTrue(torch.equal(save_proxy.call_args.args[1], corrected_frames))
            self.assertTrue(torch.equal(save_proxy.call_args.kwargs["audio_waveform"], decoded_audio))
            self.assertIsNone(cache.load_if_compatible({"different": True})[0])
            cache.truncate_from(1)
            self.assertFalse(cache.has_chunk(1))

    def test_replay_cache_lifecycle_selects_only_interrupted_runs_for_automatic_resume(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root):
            cache = nodes._LastRunReplayCache()
            cache.create({"geometry": "stable"}, "original prompt", {"video": torch.zeros(1)})
            loaded, reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertIsNone(reason)
            self.assertEqual(
                cache.automatic_resume_chunk(loaded["manifest"], 3),
                1,
            )

            cache.save_chunk(1, {"sampled_video": torch.zeros(1)})
            loaded, _reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertEqual(loaded["manifest"]["status"], "recording")
            self.assertEqual(
                cache.automatic_resume_chunk(loaded["manifest"], 3),
                2,
            )

            cache.mark_debug_stop(1)
            loaded, _reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertIsNone(cache.automatic_resume_chunk(loaded["manifest"], 3))

            cache.mark_interrupted(1)
            loaded, _reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertEqual(
                cache.automatic_resume_chunk(loaded["manifest"], 3),
                2,
            )

            # Sampling can finish before final decoding, grading, or output
            # assembly crashes. One-past-end restores every cached chunk and
            # enters finalization without running H3 again.
            cache.mark_interrupted(3)
            loaded, _reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertEqual(
                cache.automatic_resume_chunk(loaded["manifest"], 3),
                4,
            )

            cache.mark_complete(3)
            loaded, _reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertIsNone(cache.automatic_resume_chunk(loaded["manifest"], 3))

    def test_deleting_cached_chunk_preserves_later_replay_entries(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root):
            cache = nodes._LastRunReplayCache()
            cache.create({"geometry": "stable"}, "prompt", {"video": torch.zeros(1)})
            for chunk_number in range(1, 4):
                cache.save_chunk(chunk_number, {"sampled_video": torch.zeros(1)})
            cache.mark_complete(3)

            cache.delete_chunk(2)

            loaded, reason = cache.load_if_compatible({"geometry": "stable"})
            self.assertIsNone(reason)
            self.assertTrue(cache.has_chunk(1))
            self.assertFalse(cache.has_chunk(2))
            self.assertTrue(cache.has_chunk(3))
            self.assertEqual(loaded["manifest"]["status"], "interrupted")
            self.assertEqual(loaded["manifest"]["completed_chunks"], 1)
            self.assertEqual(cache.automatic_resume_chunk(loaded["manifest"], 3), 2)
            self.assertEqual(nodes._replay_cache_ui_status()["cached_chunks"], [1, 3])

    def test_preview_cache_toggle_status_preserves_interrupted_cache_visibility(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root):
            cache = nodes._LastRunReplayCache()
            original_enabled = nodes.REPLAY_CACHE_ENABLED
            nodes.REPLAY_CACHE_ENABLED = True
            self.assertEqual(
                nodes._replay_cache_ui_status(),
                {
                    "has_cache": False,
                    "enabled": True,
                    "active": False,
                    "status": "",
                    "completed_chunks": 0,
                    "cached_chunks": [],
                },
            )

            cache.create({"geometry": "stable"}, "prompt", {"video": torch.zeros(1)})
            self.assertFalse(nodes._replay_cache_ui_status()["has_cache"])
            cache.mark_interrupted(0)
            status = nodes._replay_cache_ui_status()
            self.assertTrue(status["has_cache"])
            self.assertTrue(status["enabled"])

            cache.create({"geometry": "stable"}, "prompt", {"video": torch.zeros(1)})
            cache.save_chunk(1, {"sampled_video": torch.zeros(1)})
            self.assertTrue(nodes._replay_cache_ui_status()["has_cache"])

            nodes._replay_cache_activity(True)
            try:
                status = nodes._replay_cache_ui_status()
                self.assertTrue(status["has_cache"])
                self.assertTrue(status["active"])
            finally:
                nodes._replay_cache_activity(False)

            cache.mark_interrupted(1)
            nodes._replay_cache_activity(True)
            try:
                status = nodes._replay_cache_ui_status()
                self.assertTrue(status["active"])
                self.assertTrue(status["has_cache"])
            finally:
                nodes._replay_cache_activity(False)

            nodes.REPLAY_CACHE_ENABLED = False
            self.assertFalse(nodes._replay_cache_ui_status()["enabled"])
            self.assertFalse(nodes._replay_cache_enabled())
            nodes.REPLAY_CACHE_ENABLED = original_enabled
            cache.clear()
            self.assertFalse(nodes._replay_cache_ui_status()["has_cache"])

    def test_rebuilt_replay_timing_plan_updates_the_source_prompt_hash(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes, "_timing_plan_payload", return_value={"plan": "rebuilt"}):
            cache = nodes._LastRunReplayCache()
            cache.create({"geometry": "stable"}, "old source prompt", {"video": torch.zeros(1)})
            cache.save_timing_plan(object(), source_prompt="edited source prompt")

            manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            timing = json.loads(cache.timing_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["source_prompt_sha256"],
                hashlib.sha256(b"edited source prompt").hexdigest(),
            )
            self.assertEqual(timing, {"plan": "rebuilt"})

    def test_preview_chunk_metadata_survives_a_server_state_restore(self):
        node_id = "preview-tooltip-test"
        with preview._PREVIEW_CACHE_LOCK:
            preview._PREVIEW_CACHE.pop(node_id, None)
        try:
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "reset",
                "chunk_ranges": [
                    {"chunk": 1, "start": 0, "end": 38},
                    {"chunk": 2, "start": 39, "end": 72},
                ],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "phase",
                "phase": "Gemma 4 is planning 4 source shots before H3 sampling",
                "chunk": 0,
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 1,
                "action": "chunk_metadata",
                "chunk": 0,
                "gemma_detailed_description": "  [Shot 1] The tiger runs.  ",
                "gemma_retention_analysis": "  Gemma last-seen state for Tila.  ",
                "h3_render_seconds": 42.5,
                "gemma_seconds": 3.25,
                "gemma_preproduction_seconds": 2.0,
                "chunk_total_seconds": 48.75,
            })
            snapshot = preview._cached_snapshot(node_id)
            self.assertEqual(
                snapshot["reset"]["chunk_ranges"][0]["gemma_detailed_description"],
                "[Shot 1] The tiger runs.",
            )
            self.assertEqual(
                snapshot["reset"]["chunk_ranges"][0]["gemma_retention_analysis"],
                "Gemma last-seen state for Tila.",
            )
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["h3_render_seconds"], 42.5)
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["gemma_seconds"], 3.25)
            self.assertEqual(
                snapshot["reset"]["chunk_ranges"][0]["gemma_preproduction_seconds"],
                2.0,
            )
            self.assertEqual(snapshot["reset"]["chunk_ranges"][0]["chunk_total_seconds"], 48.75)
            self.assertEqual(
                snapshot["phase"]["phase"],
                "Gemma 4 is planning 4 source shots before H3 sampling",
            )
            self.assertNotIn("gemma_detailed_description", snapshot["reset"]["chunk_ranges"][1])
        finally:
            with preview._PREVIEW_CACHE_LOCK:
                preview._PREVIEW_CACHE.pop(node_id, None)

    def test_replay_chunk_latent_is_rebuilt_as_a_playable_preview_group(self):
        class LatentFormat:
            latent_rgb_factors = [[0.0, 0.0, 0.0] for _index in range(24)]
            latent_rgb_factors_bias = None
            latent_rgb_factors_reshape = None

        wrapper = preview._AccumulatedPreviewWrapper(
            "replay-preview-test",
            max_resolution=0,
            quality=80,
            fps=24.0,
            frame_stride=1,
            tiny_vae="none",
        )
        payloads = []
        with patch.object(preview, "_send", side_effect=lambda payload: payloads.append(payload)):
            execution_id = wrapper.begin([
                {"chunk": 1, "start": 0, "end": 38},
                {"chunk": 2, "start": 39, "end": 72},
            ])
            wrapper.restore_chunk(
                execution_id,
                index=0,
                video=torch.zeros((1, 24, 12, 2, 2), dtype=torch.float32),
                sampled_start=0,
                sampled_end=38,
                output_start=0,
                output_end=38,
                trim_steps=0,
                latent_format=LatentFormat(),
                gemma_detailed_description="[Shot 1] The tiger runs.",
                gemma_retention_analysis="Preserve the established jungle lighting.",
            )

        restored = payloads[-1]
        self.assertEqual(restored["action"], "chunk")
        self.assertEqual(restored["chunk"], 0)
        self.assertEqual(restored["step"], 0)
        self.assertEqual(restored["frame_numbers"][0], 0)
        self.assertEqual(len(restored["frames"]), 12)
        self.assertEqual(restored["gemma_detailed_description"], "[Shot 1] The tiger runs.")
        self.assertEqual(
            restored["gemma_retention_analysis"],
            "Preserve the established jungle lighting.",
        )

    def test_final_vae_chunk_replaces_latent_preview_with_every_pixel_frame_and_audio(self):
        wrapper = preview._AccumulatedPreviewWrapper(
            "final-preview-test",
            max_resolution=0,
            quality=80,
            fps=24.0,
            frame_stride=4,
            tiny_vae="none",
        )
        payloads = []
        with patch.object(preview, "_send", side_effect=lambda payload: payloads.append(payload)):
            execution_id = wrapper.begin([{"chunk": 1, "start": 10, "end": 13}])
            wrapper.finalize_chunk(
                execution_id,
                0,
                torch.linspace(0, 1, 4 * 2 * 2 * 3, dtype=torch.float32).reshape(4, 2, 2, 3),
                10,
                13,
                audio_waveform=torch.zeros((1, 2, 1000), dtype=torch.float32),
                audio_sample_rate=8000,
                gemma_detailed_description="The final directed prompt.",
            )

        finalized = payloads[-1]
        self.assertEqual(finalized["action"], "chunk_final")
        self.assertTrue(finalized["finalized"])
        # Final VAE publishing deliberately ignores the live latent stride so
        # browser arrows advance one real output frame at a time.
        self.assertEqual(finalized["frame_numbers"], [10, 11, 12, 13])
        self.assertEqual(len(finalized["frames"]), 4)
        self.assertEqual(finalized["previewer"], "Full VAE (final)")
        self.assertEqual(finalized["audio_mime"], "audio/wav")
        self.assertTrue(base64.b64decode(finalized["audio"]).startswith(b"RIFF"))

    def test_final_chunk_payload_replaces_cached_latent_group_for_browser_refresh(self):
        node_id = "final-preview-cache-test"
        with preview._PREVIEW_CACHE_LOCK:
            preview._PREVIEW_CACHE.pop(node_id, None)
        try:
            preview._cache_payload({
                "node_id": node_id,
                "execution": 3,
                "action": "reset",
                "chunk_ranges": [{"chunk": 1, "start": 0, "end": 1}],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 3,
                "action": "chunk",
                "chunk": 0,
                "frames": ["latent"],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 3,
                "action": "chunk_final",
                "chunk": 0,
                "frames": ["frame0", "frame1"],
                "frame_numbers": [0, 1],
                "audio": "wav",
            })
            snapshot = preview._cached_snapshot(node_id)
            self.assertEqual(snapshot["chunks"][0]["action"], "chunk_final")
            self.assertEqual(snapshot["chunks"][0]["frames"], ["frame0", "frame1"])
            self.assertEqual(snapshot["chunks"][0]["audio"], "wav")
        finally:
            with preview._PREVIEW_CACHE_LOCK:
                preview._PREVIEW_CACHE.pop(node_id, None)

    def test_late_latent_payload_cannot_replace_cached_final_chunk(self):
        node_id = "final-preview-late-latent-test"
        with preview._PREVIEW_CACHE_LOCK:
            preview._PREVIEW_CACHE.pop(node_id, None)
        try:
            preview._cache_payload({
                "node_id": node_id,
                "execution": 4,
                "action": "reset",
                "chunk_ranges": [{"chunk": 1, "start": 0, "end": 1}],
            })
            preview._cache_payload({
                "node_id": node_id,
                "execution": 4,
                "action": "chunk_final",
                "chunk": 0,
                "frames": ["frame0", "frame1"],
                "frame_numbers": [0, 1],
                "audio": "wav",
            })
            # Simulate a live-preview websocket/encoder payload arriving after
            # the browser has begun restoring the completed server snapshot.
            preview._cache_payload({
                "node_id": node_id,
                "execution": 4,
                "action": "chunk",
                "chunk": 0,
                "frames": ["latent"],
            })
            snapshot = preview._cached_snapshot(node_id)
            self.assertEqual(snapshot["chunks"][0]["action"], "chunk_final")
            self.assertEqual(snapshot["chunks"][0]["frames"], ["frame0", "frame1"])
            self.assertEqual(snapshot["chunks"][0]["audio"], "wav")
        finally:
            with preview._PREVIEW_CACHE_LOCK:
                preview._PREVIEW_CACHE.pop(node_id, None)

    def test_audio_preview_decode_trims_latent_prefix_and_exact_video_duration(self):
        class FakeAudioVAE:
            audio_sample_rate = 8000

            def decode(self, _latent):
                # VAE output layout is [B,S,C].
                return torch.ones((1, 2000, 2), dtype=torch.float32)

        waveform, sample_rate = nodes._decode_audio_preview(
            FakeAudioVAE(),
            torch.zeros((1, 32, 2, 10), dtype=torch.float32),
            trim_latent_steps=2,
            output_frames=2,
            fps=20,
        )
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(tuple(waveform.shape), (1, 2, 800))

    def test_preparation_progress_reports_to_console_and_preview(self):
        phases = []

        class FakePreview:
            def set_phase(self, phase, *, chunk=None):
                phases.append((phase, chunk))

        with patch.object(nodes.logging, "info") as logged:
            with nodes._PreparationProgress(
                "Gemma 4 is planning source shots",
                FakePreview(),
                chunk=2,
                interval=60,
            ):
                pass

        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0][1], 2)
        self.assertIn("still working", phases[0][0])
        self.assertIn("complete", phases[1][0])
        self.assertEqual(logged.call_count, 2)

    def test_gemma_directing_preparation_uses_an_indeterminate_live_tqdm_bar(self):
        bars = []

        class FakeBar:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.n = 0
                self.disable = False
                self.refresh_count = 0
                self.closed = False

            def refresh(self):
                self.refresh_count += 1

            def close(self):
                self.closed = True

        def fake_tqdm(**kwargs):
            bar = FakeBar(**kwargs)
            bars.append(bar)
            return bar

        with patch.object(nodes, "tqdm", fake_tqdm), patch.object(nodes.logging, "info"):
            with nodes._PreparationProgress(
                "Chunk 4/10: directing chunk",
                interval=60,
                live_console_bar=True,
            ):
                pass

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].unit, "gemma")
        self.assertEqual(bars[0].total, 30)
        self.assertGreaterEqual(bars[0].refresh_count, 2)
        self.assertTrue(bars[0].closed)

    def test_gemma_progress_displays_live_decode_tokens_per_second(self):
        phases = []

        class FakePreview:
            def set_phase(self, phase, *, chunk=None):
                phases.append((phase, chunk))

        class FakeBar:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.n = 0
                self.disable = False
                self.postfix = ""

            def set_postfix_str(self, postfix, refresh=False):
                self.postfix = postfix

            def refresh(self):
                pass

            def close(self):
                pass

        with patch.object(nodes, "tqdm", lambda **kwargs: FakeBar(**kwargs)), \
                patch.object(nodes.logging, "info"):
            with nodes._PreparationProgress(
                "Chunk 2/8: directing chunk",
                FakePreview(),
                chunk=1,
                interval=60,
                live_console_bar=True,
            ) as progress:
                progress.update_token_progress(96, 127.45, 1)
                self.assertIn("96 tokens, 127.5 tokens/sec", progress._bar.postfix)

        self.assertTrue(any("127.5 tokens/sec" in phase for phase, _chunk in phases))

    def test_timing_preproduction_is_logged_before_chunk_transcripts(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes.logging, "info"):
            path = nodes._begin_last_gemma_prompt_log(
                39, 0, 0, 22, nodes.VIDEO_CONTINUATION_METHOD_VIDEO1, "full", 24.0, 3
            )
            nodes._append_gemma_timing_plan(
                path,
                "Source Shot 1: immutable preproduction timing schedule",
                system_prompt="preproduction system",
                planning_prompt="preproduction request",
                gemma_response='{"shots": []}',
            )
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 1 ===",
                "first H3 prompt",
                observation_prompt="first Gemma request",
                gemma_response='{"detailed_description":"first response"}',
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("=== GEMMA PREPRODUCTION SYSTEM PROMPT ===", content)
            self.assertIn("=== GEMMA SHOT TIMING PREPRODUCTION ===", content)
            self.assertLess(content.index("preproduction request"), content.index("=== Chunk 1 ==="))
            self.assertLess(content.index("Source Shot 1: immutable"), content.index("=== Chunk 1 ==="))

    def test_last_gemma_prompt_log_is_replaced_then_flushed_per_chunk(self):
        with tempfile.TemporaryDirectory() as temp_root, \
                patch.object(nodes.tempfile, "gettempdir", return_value=temp_root), \
                patch.object(nodes.logging, "info"):
            path = nodes._begin_last_gemma_prompt_log(
                39, 0, 0, 22, nodes.VIDEO_CONTINUATION_METHOD_VIDEO1, "full", 24.0, 3
            )
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 1 ===",
                "first H3 prompt",
                system_prompt="system instructions",
                observation_prompt="first Gemma request",
                gemma_response='{"detailed_description":"first response"}',
            )
            nodes._append_last_gemma_prompt(
                path,
                "=== Chunk 2 ===",
                "second H3 prompt",
                observation_prompt="second Gemma request",
                gemma_response='{"detailed_description":"second response"}',
                validation_warnings=("Gemma 4 returned 2 shot markers; this chunk requires 3",),
            )

            self.assertEqual(path.name, nodes.GEMMA_PROMPT_LOG_FILENAME)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Configuration: chunk_frames=39, context_keyframes=0", content)
            self.assertEqual(content.count("=== GEMMA SYSTEM PROMPT ==="), 1)
            self.assertIn("=== GEMMA SYSTEM PROMPT ===\nsystem instructions", content)
            separator = "=" * 200
            self.assertIn(f"{separator}\n=== Chunk 1 ===", content)
            self.assertIn(f"{separator}\n=== Chunk 2 ===", content)
            self.assertLess(content.index("first Gemma request"), content.index("first response"))
            self.assertLess(content.index("first response"), content.index("first H3 prompt"))
            self.assertLess(content.index("second Gemma request"), content.index("second response"))
            self.assertLess(content.index("second response"), content.index("second H3 prompt"))
            self.assertIn("=== GEMMA VALIDATION WARNINGS ===", content)
            self.assertIn("Gemma 4 returned 2 shot markers; this chunk requires 3", content)

            replacement = nodes._begin_last_gemma_prompt_log(
                56, 5, 0, 0, nodes.VIDEO_CONTINUATION_METHOD_VIDEO1, "full", 24.0, 2
            )
            replacement_content = replacement.read_text(encoding="utf-8")
            self.assertIn("Configuration: chunk_frames=56, context_keyframes=5", replacement_content)
            self.assertNotIn("first Gemma request", replacement_content)

            image_directory = nodes._reset_last_gemma_image_log()
            stale_image = image_directory / "stale.jpg"
            stale_image.write_bytes(b"stale")
            replacement_image_directory = nodes._reset_last_gemma_image_log()
            self.assertEqual(replacement_image_directory, image_directory)
            self.assertFalse(stale_image.exists())

    def test_observation_samples_only_retained_previous_output(self):
        frames, indices = nodes._decoded_video_frames(
            _FakeVAE(),
            object(),
            include_final=True,
            start_frame=5,
        )
        self.assertEqual(indices, [5, 17, 29, 38])
        self.assertEqual(frames.shape[0], 4)

    def test_generated_video_observations_match_stock_h3_reference_video_canvas(self):
        self.assertEqual(nodes._reference_video_canvas(1920, 1088), (1344, 768))
        self.assertEqual(nodes._reference_video_canvas(2048, 1152), (1344, 768))
        self.assertEqual(nodes._reference_video_canvas(640, 352), (640, 352))
        self.assertEqual(nodes._reference_video_canvas(1088, 1920), (768, 1344))

    def test_continuation_reference_is_resized_in_pixels_and_vae_encoded(self):
        class RecordingVAE:
            def __init__(self):
                self.encoded_shape = None

            def encode(self, frames):
                self.encoded_shape = tuple(frames.shape)
                return torch.zeros((1, 24, 7, frames.shape[1] // 16, frames.shape[2] // 16))

        frames = torch.zeros((22, 64, 128, 3), dtype=torch.float32)
        vae = RecordingVAE()
        with patch.dict(nodes.VIDEO_CONTINUATION_CANVASES, {"test": (64, 32)}):
            latent, canvas = nodes._encode_resized_continuation_reference(vae, frames, "test")

        self.assertEqual(canvas, (64, 32))
        self.assertEqual(vae.encoded_shape, (22, 32, 64, 3))
        self.assertEqual(tuple(latent.shape), (1, 24, 7, 2, 4))

    def test_full_continuation_reference_keeps_original_latent_without_reencoding(self):
        class FailingVAE:
            def encode(self, _frames):
                raise AssertionError("full must not re-encode Video1")

        frames = torch.zeros((22, 64, 128, 3), dtype=torch.float32)
        latent, canvas = nodes._encode_resized_continuation_reference(FailingVAE(), frames, "full")
        self.assertIsNone(latent)
        self.assertEqual(canvas, (128, 64))

    def test_continuation_payload_reports_raw_bytes_and_attention_rows(self):
        reference = torch.zeros((1, 24, 7, 34, 60), dtype=torch.bfloat16)
        audio = torch.zeros((1, 32, 2, 37), dtype=torch.bfloat16)
        boundary = torch.zeros((1, 24, 2, 68, 120), dtype=torch.bfloat16)
        target_video = torch.zeros((1, 24, 17, 68, 120), dtype=torch.bfloat16)
        target_audio = torch.zeros((1, 32, 2, 93), dtype=torch.bfloat16)
        full_reference = torch.zeros((1, 24, 7, 68, 120), dtype=torch.bfloat16)

        metrics = nodes._continuation_payload_metrics(
            reference,
            audio,
            boundary,
            target_video,
            target_audio,
            full_reference,
        )

        self.assertEqual(metrics["video_bytes"], 685440)
        self.assertEqual(metrics["audio_bytes"], 4736)
        self.assertEqual(metrics["boundary_bytes"], 783360)
        self.assertEqual(metrics["total_bytes"], 1473536)
        self.assertEqual(metrics["video_rows"], 3570)
        self.assertEqual(metrics["audio_rows"], 74)
        self.assertEqual(metrics["boundary_rows"], 4080)
        self.assertEqual(metrics["total_rows"], 7724)
        self.assertEqual(metrics["target_rows"], 34866)
        self.assertEqual(metrics["full_reference_rows"], 14280)

    def test_observation_can_return_exact_full_resolution_final_frame(self):
        frames, indices, final_frame = nodes._decoded_video_frames(
            _IndexedFakeVAE(),
            object(),
            include_final=True,
            start_frame=5,
            return_final_frame=True,
        )
        self.assertEqual(indices, [5, 17, 29, 38])
        self.assertEqual(frames.shape[0], 4)
        self.assertEqual(final_frame.shape, (1, 2, 2, 3))
        self.assertTrue(torch.all(final_frame == 38))
        self.assertEqual(final_frame.device.type, "cpu")

    def test_target_shots_omit_opening_marker_for_mid_shot_and_keep_real_local_cut(self):
        shots = [
            (0, 0, 68, "Tiger approaches the temple."),
            (1, 68, 117, "Tiger enters the temple and stops."),
        ]
        records = nodes._gemma_shot_records(
            shots,
            range_start=50,
            range_end=89,
            sampled_start=50,
            fps=24.0,
            target=True,
        )
        self.assertEqual([record["source_body"] for record in records], [shot[3] for shot in shots])
        self.assertIsNone(records[0]["required_marker"])
        self.assertEqual(records[1]["required_marker"], "[Shot 2] At 00:00.750,")

    def test_target_shots_use_opening_marker_only_at_real_physical_shot_start(self):
        shots = [
            (0, 0, 68, "Tiger approaches the temple."),
            (1, 68, 117, "Tiger enters the temple and stops."),
        ]
        at_real_start = nodes._gemma_shot_records(
            shots,
            range_start=68,
            range_end=107,
            sampled_start=68,
            fps=24.0,
            target=True,
        )
        self.assertEqual(at_real_start[0]["required_marker"], "[Shot 2]")

        after_carried_prefix = nodes._gemma_shot_records(
            shots,
            range_start=68,
            range_end=107,
            sampled_start=63,
            fps=24.0,
            target=True,
        )
        self.assertEqual(after_carried_prefix[0]["required_marker"], "[Shot 2] At 00:00.208,")

    def test_target_shots_omit_cut_that_is_complete_inside_discarded_prefix(self):
        shots = [
            (4, 259, 359, "Heman warns the others."),
            (5, 359, 414, "Cut to Tila reacting inside the temple."),
        ]
        records = nodes._gemma_shot_records(
            shots,
            range_start=362,
            range_end=413,
            sampled_start=357,
            fps=24.0,
            target=True,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["shot_number"], 6)
        self.assertIsNone(records[0]["required_marker"])

    def test_prompt_preview_omits_shot_one_for_mid_shot_continuation(self):
        prompt = (
            "detailed_description: [Shot 1] The tiger runs through the jungle. "
            "[Shot 2] At 00:02.833, The tiger enters the temple."
        )
        rewritten = nodes._prompt_for_chunk(
            prompt,
            frame_start=34,
            frame_end=73,
            total_frames=73,
            fps=24.0,
            content_start=39,
            continuation=True,
        )
        description = rewritten.split("detailed_description:", 1)[1]
        self.assertNotIn("[Shot 1]", description)
        self.assertIn("[Shot 2] At 00:01.417,", description)

        boundary = nodes._prompt_for_chunk(
            prompt,
            frame_start=0,
            frame_end=39,
            total_frames=73,
            fps=24.0,
        )
        self.assertIn("[Shot 1]", boundary)

    def test_chunk_prompt_preserves_only_global_retention_and_ignores_dynamic_values(self):
        prompt = (
            "subject_definitions:\nTila is <Subject 2>.\nHeman is <Subject 1>.\n\n"
            "retention_analysis:\nKeep identities consistent.\n\n"
            "detailed_description: [Shot 1] Original description.\n\n"
            "overall_soundscape: temple ambience"
        )
        retention_value = (
            "Tila (<Subject 2>) is mounted behind Heman on the tiger inside the temple, leaning forward; "
            "Heman (<Subject 1>) is mounted in front of Tila, holding the harness as the tiger moves."
        )

        rewritten = nodes._prompt_with_gemma_description(
            prompt,
            "In a closeup, Tila (<Subject 2>) reacts to an off-screen roar.",
            continuation_video_label="<Video 1>",
            continuation_audio_label="<Audio 1>",
            retention_analysis=retention_value,
        )

        self.assertFalse(nodes.INCLUDE_PER_CHUNK_RETENTION_ANALYSIS)
        self.assertEqual(rewritten.count("retention_analysis:"), 1)
        self.assertIn("retention_analysis:\nKeep identities consistent.", rewritten)
        self.assertIn(
            "<Audio 1>: reference - its audible continuity guides the new audio without copying the original signal.",
            rewritten,
        )
        self.assertNotIn(retention_value, rewritten)
        self.assertNotIn("fully_preserved", rewritten)
        self.assertNotIn("continuation starting point for this chunk", rewritten)
        # Video continuation remains declared in its established sections. The
        # full Audio reference is intentionally retained even when dynamic
        # Gemma character-state retention prose is disabled.
        self.assertIn("<Video 1>", rewritten)
        self.assertIn("<Audio 1>", rewritten)
        self.assertIn(
            "<Audio 1> is the previous video's full audio reference for seamless continuation in this video.",
            rewritten,
        )
        self.assertEqual(nodes._chunk_retention_analysis(retention_value), retention_value)

    def test_audio_only_reference_names_previous_speakers_and_languages(self):
        prompt = (
            "subject_definitions:\nTila is <Subject 2>.\n\n"
            "retention_analysis:\nKeep Tila consistent.\n\n"
            "detailed_description: [Shot 1] Tila continues.\n\n"
            "overall_soundscape: temple ambience"
        )
        previous = (
            "<Subject 2> (S1) says: <d>[English] Stay close!</d> "
            "<Subject 1> (S2) replies: <d>[Spanish] Vamos.</d>"
        )
        speakers = nodes._audio_reference_speakers(previous)
        rewritten = nodes._prompt_with_gemma_description(
            prompt,
            "Tila continues walking.",
            continuation_audio_label="<Audio 1>",
            audio_speakers=speakers,
        )

        self.assertEqual(speakers, (
            ("<Subject 2>", "S1", ("English",)),
            ("<Subject 1>", "S2", ("Spanish",)),
        ))
        self.assertIn("voice-timbre reference for <Subject 2> (S1) and <Subject 1> (S2)", rewritten)
        self.assertIn("containing spoken English and Spanish vocal layers", rewritten)
        self.assertIn("vocal timbres guide the dialogue delivery of <Subject 2> and <Subject 1>", rewritten)
        self.assertNotIn("<Video 1>", rewritten)

    def test_masked_av_native_boundary_uses_selected_continuation_duration(self):
        """The synthetic native boundary follows the selected overlap duration."""
        self.assertFalse(nodes.ENABLE_MASKED_AV_OVERLAP)
        plan = nodes._chunk_plan_without_overlap(
            nodes._video_steps(73),
            nodes._audio_steps(73),
            39,
            22,
        )
        boundary, start = nodes._video_continuation_boundary_guide(
            torch.zeros((1, 24, 12, 1, 1)), plan[1], 0, True,
        )
        self.assertEqual(start, 0)
        self.assertEqual(plan[1]["output_trim_frames"], 22)
        self.assertEqual(plan[1]["context_video_t"], nodes._video_steps(22))
        self.assertEqual(plan[1]["context_audio_t"], nodes._audio_steps(22))
        self.assertEqual(boundary.shape[2], nodes._video_steps(22))
        context = nodes._gemma_conditioning_context(
            True, 0, 0, 22, None, "<Audio 1>", False,
            nodes.VIDEO_CONTINUATION_METHOD_MASKED_AV, False,
        )
        self.assertIn("22-frame video/audio boundary", context)
        self.assertIn("discarded temporal packing prefix", context)
        self.assertIn("without a separate Audio reference", context)

    def test_whole_previous_chunk_audio_uses_a_standalone_h3_audio_reference(self):
        audio = torch.zeros((1, 32, 2, 207), dtype=torch.float32)
        block = nodes._audio_ref_block(audio)
        self.assertEqual(block["kind"], "audio")
        self.assertEqual(block["ref_audio_t"], 207)
        self.assertIs(block["audio_latent"], audio)

    def test_keyframes_use_truthful_physical_overlap_geometry(self):
        total_frames = 56
        plan = nodes._chunk_plan(
            nodes._video_steps(total_frames),
            nodes._audio_steps(total_frames),
            39,
            22,
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual((plan[1]["frame_start"], plan[1]["frame_end"]), (17, 56))
        self.assertEqual(plan[1]["output_trim_frames"], 22)
        self.assertEqual(plan[1]["context_video_t"], nodes._video_steps(22))
        self.assertNotIn("synthetic_prefix", plan[1])
        keyframes, guide, video, keyframe_t = nodes._continuation_controls(22, 0, 0, 39)
        self.assertEqual((keyframes, guide, video, keyframe_t), (22, 0, 0, nodes._video_steps(22)))

    def test_keyframe_overlap_must_be_smaller_than_chunk(self):
        with self.assertRaisesRegex(ValueError, "must be smaller"):
            nodes._continuation_controls(39, 0, 0, 39)

    def test_video_continuation_allows_equal_chunk_and_clamps_a_larger_request(self):
        self.assertEqual(
            nodes._continuation_controls(0, 0, 22, 22)[:3],
            (0, 0, 22),
        )

    def test_masked_av_continuation_uses_requested_overlap_inside_chunk_capacity(self):
        plan = nodes._chunk_plan(
            nodes._video_steps(107),
            nodes._audio_steps(107),
            56,
            22,
        )
        self.assertEqual(plan[0]["output_trim_frames"], 0)
        self.assertEqual(plan[1]["output_trim_frames"], 22)
        self.assertEqual(plan[1]["context_video_t"], nodes._video_steps(22))
        self.assertEqual(plan[1]["frame_end"] - plan[1]["frame_start"], 56)
        self.assertEqual(
            plan[1]["frame_end"] - (plan[1]["frame_start"] + plan[1]["output_trim_frames"]),
            34,
        )
        self.assertNotIn("synthetic_prefix", plan[1])

    def test_masked_av_target_copies_tail_and_feathers_only_audio_endpoint(self):
        chunk_video = torch.full((1, 2, 7, 2, 2), -1.0)
        chunk_audio = torch.full((1, 3, 2, 20), -1.0)
        previous_video = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1, 1).expand(1, 2, 10, 2, 2)
        previous_audio = torch.arange(30, dtype=torch.float32).reshape(1, 1, 1, 30).expand(1, 3, 2, 30)

        target_video, target_audio, nested_mask = nodes._masked_av_overlap_target(
            chunk_video,
            chunk_audio,
            previous_video,
            previous_audio,
            context_video_t=2,
            context_audio_t=10,
            feather_ticks=8,
        )
        video_mask, audio_mask = nested_mask.unbind()
        self.assertTrue(torch.equal(target_video[:, :, :2], previous_video[:, :, -2:]))
        self.assertTrue(torch.equal(target_audio[..., :10], previous_audio[..., -10:]))
        self.assertEqual(torch.count_nonzero(video_mask[:, :, :2]).item(), 0)
        self.assertEqual(video_mask[:, :, 2:].min().item(), 1.0)
        self.assertEqual(torch.count_nonzero(audio_mask[..., :2]).item(), 0)
        expected = torch.tensor([
            0.0380602, 0.1464466, 0.3086583, 0.5,
            0.6913417, 0.8535534, 0.9619398, 1.0,
        ])
        self.assertTrue(torch.allclose(audio_mask[0, 0, 0, 2:10], expected, atol=1e-5))
        self.assertEqual(audio_mask[..., 10:].min().item(), 1.0)

    def test_masked_av_default_audio_overlap_is_fully_frozen(self):
        chunk_video = torch.zeros((1, 2, 7, 2, 2), dtype=torch.float32)
        chunk_audio = torch.zeros((1, 3, 2, 20), dtype=torch.float32)
        previous_video = torch.ones((1, 2, 10, 2, 2), dtype=torch.float32)
        previous_audio = torch.ones((1, 3, 2, 30), dtype=torch.float32)

        _video, _audio, nested_mask = nodes._masked_av_overlap_target(
            chunk_video, chunk_audio, previous_video, previous_audio, 2, 10
        )
        _video_mask, audio_mask = nested_mask.unbind()
        self.assertEqual(nodes.MASKED_AV_AUDIO_FEATHER_TICKS, 0)
        self.assertEqual(torch.count_nonzero(audio_mask[..., :10]).item(), 0)
        self.assertEqual(audio_mask[..., 10:].min().item(), 1.0)

    def test_native_audio_boundary_seeds_only_audio_and_linearly_releases_it(self):
        chunk_video = torch.zeros((1, 2, 7, 2, 2), dtype=torch.float32)
        chunk_audio = torch.full((1, 3, 2, 20), -1.0)
        previous_audio = torch.arange(30, dtype=torch.float32).reshape(1, 1, 1, 30).expand(1, 3, 2, 30)
        target_audio, nested_mask = nodes._native_audio_boundary_target(
            chunk_video, chunk_audio, previous_audio, 10,
        )
        video_mask, audio_mask = nested_mask.unbind()
        positions = torch.linspace(0.0, 1.0, 10)
        expected = positions
        self.assertTrue(torch.equal(target_audio[..., :10], previous_audio[..., -10:]))
        self.assertTrue(torch.equal(target_audio[..., 10:], chunk_audio[..., 10:]))
        self.assertEqual(video_mask.min().item(), 1.0)
        self.assertTrue(torch.allclose(audio_mask[0, 0, 0, :10], expected))
        self.assertEqual(audio_mask[..., 10:].min().item(), 1.0)

    def test_native_audio_boundary_releases_only_final_word_ticks(self):
        chunk_video = torch.zeros((1, 2, 7, 2, 2), dtype=torch.float32)
        chunk_audio = torch.full((1, 3, 2, 20), -1.0)
        previous_audio = torch.arange(30, dtype=torch.float32).reshape(1, 1, 1, 30).expand(1, 3, 2, 30)

        target_audio, nested_mask = nodes._native_audio_boundary_target(chunk_video, chunk_audio, previous_audio, 10, 3)
        _video_mask, audio_mask = nested_mask.unbind()

        self.assertTrue(torch.equal(target_audio[..., :10], previous_audio[..., -10:]))
        self.assertEqual(torch.count_nonzero(audio_mask[..., :7]).item(), 0)
        self.assertTrue(torch.allclose(audio_mask[0, 0, 0, 7:10], torch.tensor([0.0, 0.5, 1.0])))
        self.assertEqual(audio_mask[..., 10:].min().item(), 1.0)

    def test_native_av_boundary_copies_tail_and_releases_both_streams(self):
        """The 39-frame prefix uses temporal weights and leaves the suffix free."""
        video = torch.zeros((1, 2, 17, 2, 2))
        audio = torch.zeros((1, 3, 2, 100))
        tail = torch.ones((1, 2, 12, 2, 2))
        audio_tail = torch.ones((1, 3, 2, 65))
        out_video, out_audio, masks = nodes._native_audio_boundary_target(video, audio, audio_tail, 65, 20, previous_video=tail)
        vm, am = masks.unbind()
        self.assertTrue(torch.equal(out_video[:, :, :12], tail))
        self.assertTrue(torch.equal(out_audio[..., :65], audio_tail))
        self.assertEqual(out_video[:, :, 12:].count_nonzero().item(), 0)
        self.assertEqual(video.count_nonzero().item(), 0)
        self.assertEqual(vm[0, 0, 0, 0, 0].item(), 0)
        self.assertEqual(vm[0, 0, 11, 0, 0].item(), 1)
        self.assertEqual(am[0, 0, 0, 64].item(), 1)
        self.assertEqual(vm[:, :, 12:].min().item(), 1)
        self.assertEqual(am[..., 65:].min().item(), 1)
        position = ((34.0 / 38.0 * 64.0) - 45.0) / 19.0
        self.assertAlmostEqual(vm[0, 0, 10, 0, 0].item(), position, places=5)

    def test_last_dialogue_word_feather_ticks_uses_only_terminal_word(self):
        with patch.object(nodes, "_dialogue_word_weights", return_value=((None, 0.165),)):
            ticks = nodes._last_dialogue_word_feather_ticks("<d>[English] Earlier words. The</d>", 66)

        self.assertEqual(ticks, 7)
        self.assertEqual(nodes._last_dialogue_word_feather_ticks("plain narrative", 66), 0)

    def test_masked_av_replaces_frozen_video_prefix_with_corrected_reencode(self):
        chunk_video = torch.zeros((1, 24, 12, 2, 3), dtype=torch.float32)
        corrected_reencode = torch.full((1, 24, 12, 2, 3), 0.75, dtype=torch.float32)

        updated = nodes._replace_masked_av_video_prefix(chunk_video, corrected_reencode, 12)

        self.assertTrue(torch.equal(updated, corrected_reencode))
        self.assertTrue(torch.equal(chunk_video, torch.zeros_like(chunk_video)))
        with self.assertRaisesRegex(ValueError, "shape"):
            nodes._replace_masked_av_video_prefix(chunk_video, corrected_reencode[:, :, :-1], 12)

    def test_masked_av_boundary_guides_encode_every_overlap_image_separately(self):
        class DummyVAE:
            def __init__(self):
                self.encoded = []

            def encode(self, frames):
                self.encoded.append(frames.clone())
                return torch.full((1, 24, 1, 2, 2), float(len(self.encoded)))

        vae = DummyVAE()
        decoded = torch.linspace(0.0, 0.9, 30 * 3 * 2 * 3, dtype=torch.float32).reshape(30, 3, 2, 3)
        guides = nodes._masked_av_boundary_guides(vae, decoded, 5)

        self.assertEqual([item["resolved_frame_index"] for item in guides], [0, 1, 2, 3, 4])
        self.assertEqual(len(vae.encoded), 5)
        # This helper faithfully encodes the caller-selected decoded frames.
        self.assertTrue(torch.equal(vae.encoded[0], decoded[25:26]))
        self.assertTrue(torch.equal(vae.encoded[1], decoded[26:27]))
        self.assertTrue(torch.equal(vae.encoded[2], decoded[27:28]))
        self.assertTrue(torch.equal(vae.encoded[3], decoded[28:29]))
        self.assertTrue(torch.equal(vae.encoded[4], decoded[29:30]))
        self.assertTrue(all(tuple(item["latent"].shape) == (1, 24, 1, 2, 2) for item in guides))

    def test_h3_context_frames_uses_corrected_tail_when_experiment_enabled(self):
        raw = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        corrected = torch.ones((2, 2, 2, 3), dtype=torch.float32)
        original = nodes.USE_COLOR_CORRECTED_H3_CONTEXT
        try:
            nodes.USE_COLOR_CORRECTED_H3_CONTEXT = True
            selected, label = nodes._h3_context_frames(raw, corrected)
            self.assertIs(selected, corrected)
            self.assertEqual(label, "color-corrected")
            nodes.USE_COLOR_CORRECTED_H3_CONTEXT = False
            selected, label = nodes._h3_context_frames(raw, corrected)
            self.assertIs(selected, raw)
            self.assertEqual(label, "raw")
        finally:
            nodes.USE_COLOR_CORRECTED_H3_CONTEXT = original

    def test_conditioning_appends_sparse_masked_av_boundary_guides(self):
        prompt = (torch.ones((1, 2, 3)), {})
        sparse = (
            {"resolved_frame_index": 0, "latent": torch.ones((1, 24, 1, 2, 2))},
            {"resolved_frame_index": 10, "latent": torch.full((1, 24, 1, 2, 2), 2.0)},
            {"resolved_frame_index": 21, "latent": torch.full((1, 24, 1, 2, 2), 3.0)},
        )
        conds = nodes._conditioning_for_chunk(
            {"positive": [{"cross_attn": None}]},
            0,
            56,
            prompt,
            video_contexts=sparse,
        )
        keyframes = conds["positive"][0]["minimax_keyframes"]
        self.assertEqual([item["resolved_frame_index"] for item in keyframes], [0, 10, 21])

    def test_masked_audio_assembly_replaces_accumulated_tail_across_parts(self):
        parts = [
            torch.tensor([[[[0.0, 1.0, 2.0]]]]),
            torch.tensor([[[[3.0, 4.0]]]]),
        ]
        replacement = torch.tensor([[[[20.0, 30.0, 40.0, 50.0]]]])
        nodes._replace_stream_tail(parts, replacement)
        self.assertTrue(
            torch.equal(torch.cat(parts, dim=-1), torch.tensor([[[[0.0, 20.0, 30.0, 40.0, 50.0]]]]))
        )

    def test_decoded_frame_before_tail_uses_the_last_frame_that_will_remain(self):
        parts = [
            torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1),
            torch.arange(3, 7, dtype=torch.float32).reshape(4, 1, 1, 1),
        ]
        reference = nodes._decoded_frame_before_tail(parts, 3)
        self.assertEqual(reference[:, 0, 0, 0].tolist(), [3.0])
        self.assertIsNone(nodes._decoded_frame_before_tail(parts, 7))

    def test_decoded_tail_replacement_keeps_duration_and_uses_new_chunk_prefix(self):
        parts = [
            torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1),
            torch.arange(3, 7, dtype=torch.float32).reshape(4, 1, 1, 1),
        ]
        replacement = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float32).reshape(3, 1, 1, 1)
        nodes._replace_decoded_frame_tail(parts, replacement)
        self.assertEqual(torch.cat(parts, dim=0)[:, 0, 0, 0].tolist(), [0.0, 1.0, 2.0, 3.0, 20.0, 30.0, 40.0])

    def test_masked_av_gemma_context_never_claims_video1_reference(self):
        context = nodes._gemma_conditioning_context(
            True,
            0,
            0,
            22,
            "<Video 1>",
            "<Audio 1>",
            False,
            nodes.VIDEO_CONTINUATION_METHOD_MASKED_AV,
            True,
        )
        self.assertIn("22-frame physical video/audio overlap", context)
        self.assertIn("not a Video/Audio reference", context)
        self.assertNotIn("<Video 1>", context)

    def test_masked_av_planned_prompt_does_not_inject_video_continuation_summary(self):
        prompt = (
            "summary: original task\n\n"
            "detailed_description: [Shot 1] A tiger runs through the jungle.\n\n"
            "overall_soundscape: jungle"
        )
        plan = nodes._chunk_plan(
            nodes._video_steps(73),
            nodes._audio_steps(73),
            56,
            22,
        )
        planned = nodes._planned_chunk_prompts(
            prompt,
            plan,
            plan,
            24.0,
            22,
            False,
            False,
            True,
            1,
            1,
        )
        second_prompt = planned[1][0]
        self.assertIn("summary: original task", second_prompt)
        self.assertNotIn("[video continuation]", second_prompt)
        self.assertNotIn("<Video 1>", second_prompt)
        self.assertNotIn("<Audio 1>", second_prompt)
        self.assertEqual(
            nodes._continuation_controls(0, 0, 56, 22)[:3],
            (0, 0, 22),
        )

    def test_video1_boundary_keyframe_uses_complete_discarded_packing_prefix(self):
        plan = nodes._chunk_plan_without_overlap(
            nodes._video_steps(73),
            nodes._audio_steps(73),
            39,
        )
        previous = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1, 1)
        boundary_latent, boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 0, True)
        self.assertEqual(boundary_start, 0)
        self.assertEqual(boundary_latent.shape[2], nodes._video_steps(5))
        self.assertTrue(torch.equal(boundary_latent, previous[:, :, -2:]))
        self.assertNotEqual(boundary_latent.data_ptr(), previous.data_ptr())

        no_boundary, no_boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 22, True)
        self.assertIsNone(no_boundary)
        self.assertEqual(no_boundary_start, 0)
        no_boundary, no_boundary_start = nodes._video_continuation_boundary_guide(previous, plan[1], 0, False)
        self.assertIsNone(no_boundary)
        self.assertEqual(no_boundary_start, 0)

        boundary_latent = torch.zeros((1, 24, 2, 2, 2), dtype=torch.float32)
        conds = nodes._conditioning_for_chunk(
            {"positive": [{"minimax_keyframes": []}]},
            34,
            73,
            (torch.zeros((1, 1, 1)), {}),
            video_context=boundary_latent,
            video_context_start=0,
        )
        keyframe = conds["positive"][0]["minimax_keyframes"][0]
        self.assertEqual(keyframe["resolved_frame_index"], 0)
        self.assertIs(keyframe["latent"], boundary_latent)

        audio_boundary = torch.zeros((1, 32, 2, nodes._audio_steps(5)), dtype=torch.float32)
        conds = nodes._conditioning_for_chunk(
            {"positive": [{"minimax_keyframes": []}]},
            34,
            73,
            (torch.zeros((1, 1, 1)), {}),
            audio_context=audio_boundary,
            audio_end_frame=audio_boundary.shape[-1] / nodes.FRAME_RESCALE,
        )
        audio_keyframe = conds["positive"][0]["minimax_keyframes"][0]
        self.assertEqual(nodes._audio_steps(5), 8)
        self.assertEqual(audio_keyframe["resolved_frame_index"], 0)
        self.assertIs(audio_keyframe["audio_latent"], audio_boundary)

    def test_debug_memory_preflight_uses_at_most_three_real_sigma_steps(self):
        sigmas = torch.arange(21, dtype=torch.float32)
        probe, steps = nodes._debug_preflight_sigmas(sigmas)

        self.assertEqual(steps, 3)
        self.assertTrue(torch.equal(probe, sigmas[:4]))

        short_probe, short_steps = nodes._debug_preflight_sigmas(sigmas[:3])
        self.assertEqual(short_steps, 2)
        self.assertTrue(torch.equal(short_probe, sigmas[:3]))

    def test_debug_memory_preflight_payload_matches_continuation_geometry(self):
        video = torch.zeros((1, 24, nodes._video_steps(56), 68, 120))
        audio = torch.zeros((1, 32, nodes._audio_steps(56)))

        payload = nodes._debug_preflight_continuation_payload(
            video,
            audio,
            22,
            "0.30mp (736x416)",
            1920,
            1088,
        )

        self.assertEqual(
            payload["reference_video"].shape,
            (1, 24, nodes._video_steps(22), 416 // 16, 736 // 16),
        )
        self.assertEqual(payload["full_reference_video"].shape, (1, 24, nodes._video_steps(22), 68, 120))
        self.assertEqual(payload["reference_audio"].shape[-1], nodes._audio_steps(22))
        self.assertEqual(payload["boundary_video"].shape[2], nodes._video_steps(5))
        self.assertEqual(len(payload["qwen_items"]), 2)
        self.assertEqual(payload["qwen_items"][1]["data"].shape[0], 2)

        full_payload = nodes._debug_preflight_continuation_payload(
            video,
            audio,
            22,
            "full",
            1920,
            1088,
        )
        self.assertIs(full_payload["full_reference_video"], full_payload["reference_video"])

    def test_debug_memory_preflight_cleanup_forces_model_and_allocator_release(self):
        patcher_a = object()
        patcher_b = object()
        backend = unittest.mock.MagicMock()
        with patch.object(nodes, "_memory_backend", return_value=backend), \
                patch.object(nodes.comfy.model_management, "unload_model_and_clones") as unload, \
                patch.object(nodes.comfy.model_management, "soft_empty_cache") as empty, \
                patch.object(nodes.gc, "collect") as collect:
            nodes._release_debug_preflight(torch.device("cuda:0"), patcher_a, None, patcher_b)

        self.assertEqual(unload.call_args_list, [unittest.mock.call(patcher_a), unittest.mock.call(patcher_b)])
        empty.assert_called_once_with(force=True)
        backend.synchronize.assert_called_once_with(torch.device("cuda:0"))
        backend.empty_cache.assert_called_once_with()
        self.assertEqual(collect.call_count, 2)

    def test_color_diagnostics_measure_chunk_tone_and_adjacent_boundary_shift(self):
        first_frames = torch.empty((4, 18, 32, 3), dtype=torch.float32)
        first_frames[..., 0] = 0.2
        first_frames[..., 1] = 0.4
        first_frames[..., 2] = 0.6
        second_frames = first_frames + 0.1

        first = nodes._video_color_diagnostics(first_frames)
        second = nodes._video_color_diagnostics(second_frames)
        boundary = nodes._boundary_color_diagnostics(first, second)

        self.assertEqual(first["frame_count"], 4)
        self.assertEqual(first["sampled_frame_count"], 4)
        for actual, expected in zip(first["rgb_mean"], (0.2, 0.4, 0.6)):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(first["luma_std"], 0.0, places=6)
        self.assertGreater(second["luma_mean"], first["luma_mean"])
        self.assertAlmostEqual(boundary["luma_mean_delta"], 0.1, places=5)
        self.assertEqual(len(first["luma_histogram"]), 32)
        self.assertIn("Chunk 1", nodes._format_color_diagnostics(1, first))
        self.assertIn("Chunk 1->2", nodes._format_boundary_color_diagnostics(1, 2, boundary))

    def test_fixed_output_color_correction_preserves_raw_prefix_and_matches_boundary_statistics(self):
        torch.manual_seed(7)
        base = torch.rand((1, 18, 32, 3), dtype=torch.float32) * 0.65 + 0.08
        reference = base.expand(5, -1, -1, -1).clone()
        darker = nodes._linear_rgb_to_srgb(nodes._srgb_to_linear_rgb(reference) * 0.8)
        protected_prefix = reference.clone()
        physical_chunk = torch.cat((protected_prefix, darker), dim=0)

        corrected, transform, corrected_frames = nodes._correct_decoded_chunk_color(
            reference,
            physical_chunk,
            5,
            5,
        )

        self.assertEqual(corrected_frames, 5)
        self.assertIsNotNone(transform)
        self.assertTrue(torch.equal(corrected[:5], protected_prefix))
        raw_stats = nodes._display_color_statistics(darker[:1])
        corrected_stats = nodes._display_color_statistics(corrected[5:6])
        reference_stats = nodes._display_color_statistics(reference[:1])
        self.assertLess(
            abs(float(corrected_stats["luma_median"] - reference_stats["luma_median"])),
            abs(float(raw_stats["luma_median"] - reference_stats["luma_median"])),
        )

    def test_float_color_path_preserves_negative_and_hdr_decoded_pixels(self):
        frames = torch.tensor([[[[-0.10, -0.10, -0.10], [1.20, 1.20, 1.20]]]], dtype=torch.float32)
        linear = nodes._srgb_to_linear_rgb(frames)
        restored = nodes._linear_rgb_to_srgb(linear)
        self.assertTrue(torch.allclose(restored, frames, atol=1e-5, rtol=1e-5))

        corrected = nodes._apply_display_tone_curve(
            frames,
            [-0.20, 0.00, 0.50, 1.00, 1.20],
            [-0.20, 0.00, 0.50, 1.00, 1.20],
        )
        self.assertTrue(torch.allclose(corrected, frames, atol=1e-5, rtol=1e-5))

    def test_inverse_gamma_compute_transfer_round_trips_rgb_and_preserves_alpha(self):
        frames = torch.tensor([[[[0.25, 0.50, 0.75, 0.30]]]], dtype=torch.float32)
        linear = nodes._convert_image_transfer(frames, nodes._srgb_to_inverse_gamma_compute_rgb)
        restored = nodes._convert_image_transfer(linear, nodes._inverse_gamma_compute_to_srgb)

        self.assertTrue(torch.allclose(restored, frames, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(float(linear[..., 3].item()), 0.30, places=6)

    def test_linear_ref2va_images_are_reencoded_from_original_pixels(self):
        images = torch.full((1, 32, 48, 3), 0.5, dtype=torch.float32)
        original_latent = torch.zeros((1, 24, 1, 2, 3), dtype=torch.float32)
        vae = Mock()
        vae.encode.return_value = torch.ones_like(original_latent)

        original_ref = {"kind": "image", "latent_h": 2, "latent_w": 3, "latent": original_latent}
        refs = nodes._linear_ref2va_image_refs([original_ref], images, vae)

        self.assertTrue(torch.equal(refs[0]["latent"], torch.ones_like(original_latent)))
        self.assertIsNot(refs[0], original_ref)
        encoded_image = vae.encode.call_args.args[0]
        self.assertEqual(tuple(encoded_image.shape[1:3]), (32, 48))
        self.assertAlmostEqual(float(encoded_image[..., :3].mean()), float(nodes._srgb_to_inverse_gamma_compute_rgb(images).mean()), places=2)

    def test_sampler_h3_decode_restores_finalizer_without_clipping_float_pixels(self):
        class FakeStage:
            pixel_mean = torch.zeros((1, 3, 1, 1, 1))
            pixel_std = torch.ones((1, 3, 1, 1, 1))

            @staticmethod
            def _finalize_pixels(part):
                return part.clamp(0, 1)

        class FakeVAE:
            def __init__(self):
                self.first_stage_model = FakeStage()

            def decode(self, _latent):
                part = torch.tensor([[[[[-1.0]]], [[[2.0]]], [[[0.5]]]]])
                return self.first_stage_model._finalize_pixels(part).movedim(1, -1)

        vae = FakeVAE()
        original = vae.first_stage_model._finalize_pixels
        decoded = nodes._decode_video_frames(vae, torch.empty(0))
        self.assertEqual(tuple(decoded.shape), (1, 1, 1, 3))
        self.assertEqual(decoded[0, 0, 0, 0].item(), -1.0)
        self.assertEqual(decoded[0, 0, 0, 1].item(), 2.0)
        self.assertIs(vae.first_stage_model._finalize_pixels, original)

    def test_final_stable_light_shot_grade_uses_first_chunk_without_touching_it(self):
        images = torch.full((4, 4, 4, 3), 0.6, dtype=torch.float32)
        images[2:] = 0.4
        timing_plan = SimpleNamespace(shots=(SimpleNamespace(source_shot=1, light_change=False),))
        corrected, reports = nodes._final_shot_color_correction(
            images,
            [(0, 0, 4, "stable lighting")],
            timing_plan,
            [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        )
        self.assertTrue(torch.equal(corrected[:2], images[:2]))
        self.assertGreater(float(corrected[2:].mean()), float(images[2:].mean()))
        self.assertIn("applied", reports[0][1])

    def test_final_all_chunks_grade_uses_mkl_without_gemma_preproduction(self):
        images = torch.full((4, 4, 4, 3), 0.6, dtype=torch.float32)
        images[2:] = 0.4
        corrected, reports = nodes._final_shot_color_correction(
            images,
            [(0, 0, 4, "stable lighting")],
            None,
            [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        )
        self.assertTrue(torch.equal(corrected[:2], images[:2]))
        self.assertGreater(float(corrected[2:].mean()), float(images[2:].mean()))
        self.assertIn("applied", reports[0][1])

    def test_ready_entire_shot_grade_uses_the_first_frame_and_grades_chunk_one(self):
        frames = torch.full((3, 4, 4, 3), 0.6, dtype=torch.float32)
        frames[1:] = 0.4
        corrected = nodes._correct_ready_entire_shot_frames(
            frames, 0, [(0, 0, 3, "stable lighting")], None, {},
        )

        self.assertTrue(torch.equal(corrected[:1], frames[:1]))
        self.assertGreater(float(corrected[1:].mean()), float(frames[1:].mean()))

    def test_final_shot_grade_skips_prompted_lighting_change(self):
        images = torch.full((4, 2, 2, 3), 0.4, dtype=torch.float32)
        images[2:] = 0.7
        timing_plan = SimpleNamespace(shots=(SimpleNamespace(source_shot=1, light_change=True),))
        corrected, reports = nodes._final_shot_color_correction(
            images,
            [(0, 0, 4, "sunrise brightens")],
            timing_plan,
            [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        )
        self.assertTrue(torch.equal(corrected, images))
        self.assertIn("permits lighting change", reports[0][1])

    def test_fixed_output_color_correction_does_not_introduce_a_temporal_gain_curve(self):
        torch.manual_seed(9)
        base = torch.rand((1, 18, 32, 3), dtype=torch.float32) * 0.45 + 0.10
        previous = base.expand(5, -1, -1, -1).clone()
        retained = torch.cat([
            nodes._linear_rgb_to_srgb(nodes._srgb_to_linear_rgb(base) * (1.0 - index * 0.05))
            for index in range(5)
        ], dim=0)
        physical_chunk = torch.cat((previous, retained), dim=0)

        corrected, transform, corrected_frames = nodes._correct_decoded_chunk_color(
            previous,
            physical_chunk,
            5,
            5,
        )

        self.assertEqual(corrected_frames, 5)
        self.assertIsNotNone(transform)
        self.assertTrue(torch.equal(corrected[:5], previous))
        from color_matcher import ColorMatcher
        expected = torch.stack([
            torch.from_numpy(ColorMatcher().transfer(src=frame.numpy(), ref=previous[-1].numpy(), method="mkl"))
            for frame in retained
        ]).to(torch.float32).clamp_(0, 1)
        self.assertEqual(transform["method"], "mkl")
        self.assertTrue(torch.allclose(corrected[5:], expected, atol=2e-5, rtol=2e-5))

    def test_adaptive_overlap_color_correction_is_identity_without_overlap(self):
        frames = torch.full((3, 4, 4, 3), 0.25, dtype=torch.float32)
        corrected, transform, corrected_frames = nodes._correct_decoded_chunk_color(None, frames, 0)
        self.assertIs(corrected, frames)
        self.assertIsNone(transform)
        self.assertEqual(corrected_frames, 0)

    def test_decoded_frame_tail_spans_parts_without_joining_the_full_render(self):
        parts = [
            torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1),
            torch.arange(4, 7, dtype=torch.float32).reshape(3, 1, 1, 1),
        ]
        tail = nodes._decoded_frame_tail(parts, 5)
        self.assertEqual(tail[:, 0, 0, 0].tolist(), [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_same_shot_color_correction_stops_at_next_cut_and_skips_cut_boundaries(self):
        shots = [
            (0, 0, 100, "first"),
            (1, 100, 200, "second"),
        ]
        self.assertEqual(nodes._same_shot_correction_frames(shots, 50, 150), 50)
        self.assertEqual(nodes._same_shot_correction_frames(shots, 100, 150), 0)
        self.assertEqual(nodes._same_shot_correction_frames([], 50, 150), 100)

    def test_debug_redraw_leaves_sampling_steps_after_chunk_bar(self):
        refreshed = []

        class FakeBar:
            disable = False

            def __init__(self, unit):
                self.unit = unit

            def refresh(self):
                refreshed.append(self.unit)

        class FakeTqdm:
            _instances = (FakeBar("steps"), FakeBar("chunk"))

        with patch.object(nodes, "tqdm", FakeTqdm), patch.object(nodes, "_cli_tqdm", FakeTqdm):
            nodes._refresh_console_progress()

        self.assertEqual(refreshed, ["chunk", "steps"])

    def test_peak_time_interpolates_time_closer_to_peak_than_average(self):
        gib = 1024 ** 3
        with patch.object(nodes, "_memory_backend", return_value=None):
            timing = nodes._SamplerTiming(torch.device("cpu"))

        timing._snapshot_count = 4
        timing._snapshot_sum = 4 * 8 * gib
        timing.max_device_used = 14 * gib
        timing.device_total = 16 * gib
        timing._physical_samples = [
            (0.0, 8 * gib),
            (10.0, 14 * gib),
            (20.0, 14 * gib),
            (30.0, 8 * gib),
        ]

        physical = timing._physical_summary()

        self.assertEqual(physical["threshold"], 11 * gib)
        self.assertAlmostEqual(physical["peak_time"], 20.0)

    def test_sampler_timing_has_final_output_shot_color_bucket(self):
        with patch.object(nodes, "_memory_backend", return_value=None):
            timing = nodes._SamplerTiming(torch.device("cpu"))

        self.assertEqual(timing.seconds["output_shot_color"], 0.0)
        self.assertEqual(timing.calls["output_shot_color"], 0)

    def test_non_debug_monitor_still_collects_report_samples(self):
        class FakeTiming:
            def __init__(self):
                self.observations = []

            def observe_memory(self, **kwargs):
                self.observations.append(kwargs)

        timing = FakeTiming()
        monitor = nodes._VRAMMonitor(timing, torch.device("cpu"), [], 1, debug=False)
        with patch.object(nodes, "_vram_report") as detailed_report:
            monitor.report("sampling", sample_group="dit")

        self.assertEqual(timing.observations, [{"sample_group": "dit", "chunk_index": 0}])
        detailed_report.assert_not_called()

    def test_always_on_report_puts_peak_time_directly_after_peak(self):
        gib = 1024 ** 3
        with patch.object(nodes, "_memory_backend", return_value=None):
            timing = nodes._SamplerTiming(torch.device("cpu"))
        timing.started -= 30.0
        timing._snapshot_count = 4
        timing._snapshot_sum = 4 * 8 * gib
        timing._dit_snapshot_count = 2
        timing._dit_snapshot_sum = 2 * 10 * gib
        timing._later_dit_snapshot_count = 1
        timing._later_dit_snapshot_sum = 10 * gib
        timing.max_device_used = 14 * gib
        timing.device_total = 16 * gib
        timing._physical_samples = [
            (0.0, 8 * gib),
            (10.0, 14 * gib),
            (20.0, 14 * gib),
            (30.0, 8 * gib),
        ]
        timing.chunk_seconds = {0: 15.0, 1: 15.0}
        timing.seconds["h3_sampling"] = 20.0
        timing.calls["h3_sampling"] = 2
        run = {
            "chunk_frames": 39,
            "context_keyframes": 5,
            "guide_overlap": 0,
            "video_continuation": 22,
            "width": 864,
            "height": 480,
            "sampling_steps": 20,
            "rendered_frames": 73,
            "full_frames": 73,
            "full_chunks": 2,
        }
        first_color = nodes._video_color_diagnostics(torch.full((2, 8, 8, 3), 0.25))
        second_color = nodes._video_color_diagnostics(torch.full((2, 8, 8, 3), 0.35))
        second_color["boundary_from_previous"] = nodes._boundary_color_diagnostics(
            first_color,
            second_color,
        )
        run["color_diagnostics"] = {0: first_color, 1: second_color}

        with patch.object(nodes.logging, "info") as log_info:
            timing.report("complete", 2, run)

        report = log_info.call_args.args[0]
        self.assertIn("HR Endless Sampler run report:", report)
        self.assertIn("Baseline from this run:", report)
        self.assertIn("VRAM baseline:", report)
        self.assertIn("  Peak Time: 20.00s (VRAM closer to Peak than Average; above 11.00 GiB)", report)
        self.assertIn("Time baseline:", report)
        self.assertIn("Breakdown:", report)
        self.assertIn("Decoded color/contrast baseline:", report)
        self.assertIn("Chunk 1 -> Chunk 2 whole-chunk drift:", report)
        peak_line = report.index("  Peak:")
        peak_time_line = report.index("  Peak Time:")
        torch_line = report.index("  PyTorch VRAM high-water:")
        self.assertLess(peak_line, peak_time_line)
        self.assertLess(peak_time_line, torch_line)


if __name__ == "__main__":
    unittest.main()
