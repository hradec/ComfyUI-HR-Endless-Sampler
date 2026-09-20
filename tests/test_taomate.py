"""CPU checks for the ComfyUI adapter and unmodified upstream streaming core."""

import importlib
import os
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
sys.path.insert(0, os.path.dirname(ROOT))
module = importlib.import_module(os.path.basename(ROOT) + ".python.taomate")


class TaoMateTest(unittest.TestCase):
    """Check timing, state boundaries and real H3 forward integration."""

    def test_chunk_frames_controls_groups_without_enlarging_phases(self):
        """Different prompt durations preserve the global AV clock and bounded KV phases."""
        state = module.TaoMateStreaming
        video_t = 142
        audio_t = round(state.frames(video_t) * module.h3.FRAME_RESCALE)
        for frames in (22, 39, 56, 124, 240, 243, 481):
            requests = state.request_plan(video_t, audio_t, frames)
            video_cursor = audio_cursor = 0
            maximum = frames - (frames - 5) % 17
            for index, request in enumerate(requests):
                self.assertEqual(request["video_start"], video_cursor)
                self.assertEqual(request["audio_start"], audio_cursor)
                self.assertLessEqual(request["frame_end"] - request["frame_start"], maximum)
                self.assertEqual(request["context_video_t"], 2 if index else 0)
                for phase in request["phases"]:
                    self.assertEqual(phase["video_start"], video_cursor)
                    self.assertEqual(phase["audio_start"], audio_cursor)
                    self.assertLessEqual(phase["video_end"] - video_cursor, 12 if video_cursor == 0 else 10)
                    video_cursor, audio_cursor = phase["video_end"], phase["audio_end"]
                    self.assertEqual(audio_cursor, round(state.frames(video_cursor) * module.h3.FRAME_RESCALE))
            self.assertEqual((video_cursor, audio_cursor), (video_t, audio_t))
        self.assertEqual([len(r["phases"]) for r in state.request_plan(video_t, audio_t, 243)], [8, 8])
        for invalid in (0, 21, 124.5, True):
            with self.assertRaises(ValueError):
                state.request_plan(video_t, audio_t, invalid)

    def setUp(self):
        """Keep existing raw-storage checks explicit; test compression separately."""
        setting = patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV", False)
        setting.start()
        self.addCleanup(setting.stop)
        gpu_setting = patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV", False)
        gpu_setting.start()
        self.addCleanup(gpu_setting.stop)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for nvCOMP round trip")
    def test_gpu_zstd_restores_exact_kv_on_current_stream(self):
        """Real nvCOMP decode handles multiple blocks, tail bytes and both dtypes."""
        with patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV", True), patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV", True):
            for dtype, integer in ((torch.bfloat16, torch.int16), (torch.float32, torch.int32)):
                count = 32 * 1024 ** 2 // torch.empty((), dtype=dtype).element_size() + 128
                values = torch.arange(count, dtype=torch.int32).to(integer).view(dtype).reshape(-1, 1, 2)
                packed = module.CompressedKV(module.AVKV(values, values.flip(0)))
                cache = module.CPUStreamingCache(module.KVContract(local_heads=1, head_dim=2, dtype=dtype, device_type="cpu"))
                cache.active_device = torch.device("cuda:0")
                cache._history["blocks.0.attn"] = packed
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    # GPU history must never call the CPU decompression properties.
                    with patch.object(module.CompressedKV, "key", new_callable=unittest.mock.PropertyMock, side_effect=AssertionError("CPU key decode")), patch.object(module.CompressedKV, "value", new_callable=unittest.mock.PropertyMock, side_effect=AssertionError("CPU value decode")):
                        restored = cache.history("blocks.0.attn")
                    self.assertTrue(torch.equal(restored.key.view(integer).cpu(), values.view(integer)))
                    self.assertTrue(torch.equal(restored.value.view(integer).cpu(), values.flip(0).view(integer)))
                self.assertTrue(torch.equal(packed.key.view(integer), values.view(integer)))

    def test_compressed_kv_exact_bits_and_rollover(self):
        """Compression preserves BF16/FP32 bits and the upstream retention contract."""
        for dtype, integer in ((torch.bfloat16, torch.int16), (torch.float32, torch.int32)):
            # Include arbitrary float bit patterns, including non-finite values.
            bits = torch.arange(-32768, 32768, dtype=integer)
            values = bits.view(dtype).reshape(-1, 1, 2)
            packed = module.CompressedKV(module.AVKV(values, values.flip(0)))
            self.assertTrue(torch.equal(packed.key.view(integer), values.view(integer)))
            self.assertTrue(torch.equal(packed.value.view(integer), values.flip(0).view(integer)))
        with patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV", True):
            self.test_commit_evicts_before_append_and_matches_upstream()
            self._run_native_phases(True)

    def test_commit_evicts_before_append_and_matches_upstream(self):
        """Across rollovers, append sees only anchor plus one old recent."""
        contract = module.KVContract(local_heads=1, head_dim=2, dtype=torch.float32, device_type="cpu")
        actual = module.CPUStreamingCache(contract)
        expected = module.CleanAVKVCache(contract)
        original_commit = module.CleanAVKVCache.commit
        statuses = []
        actual.on_status = statuses.append

        def inspect_append(cache):
            """Verify eviction happened before upstream builds combined tensors."""
            if cache is actual:
                self.assertLessEqual(len(cache._commit_token_counts), 2)
                if cache.committed_blocks >= 2:
                    self.assertEqual(set(cache._commit_token_tags[0]), {0})
            original_commit(cache)

        for block in range(7):
            for cache in (actual, expected):
                cache.begin_clean_commit(block)
                tags = torch.tensor([0] * (block % 3 + 1) + [2, 2])
                values = torch.arange(len(tags) * 2, dtype=torch.float32).reshape(-1, 1, 2) + block * 100
                for name in module.MAIN_LAYER_NAMES:
                    cache.stage(name, values, values + 1, tags, torch.ones_like(tags, dtype=torch.bool))
                with patch.object(module.CleanAVKVCache, "commit", inspect_append):
                    cache.commit()
                before = cache._history["blocks.0.attn"]
                cache.retain_sink_and_recent_commits()
                if cache is actual:
                    self.assertIs(before, cache._history["blocks.0.attn"])
            self.assertEqual(actual._commit_token_tags, expected._commit_token_tags)
            for name in module.MAIN_LAYER_NAMES:
                self.assertTrue(torch.equal(actual._history[name].key, expected._history[name].key))
                self.assertTrue(torch.equal(actual._history[name].value, expected._history[name].value))
            if block == 3:
                actual.drop_audio_history()
                expected.drop_audio_history()
        self.assertIn("KV cache: evicting outgoing history", statuses)
        self.assertIn("KV cache: committing new history", statuses)
        self.assertIn("KV cache: retaining history, layer 50/50", statuses)
        if module.TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV:
            self.assertIn("KV cache: committing/compressing layer 50/50", statuses)

    def test_cpu_retention_matches_upstream_without_holding_all_old_layers(self):
        """Release replaced layers immediately while preserving exact upstream KV."""
        contract = module.KVContract(local_heads=1, head_dim=2, dtype=torch.float32, device_type="cpu")
        actual = module.CPUStreamingCache(contract)
        expected = module.CleanAVKVCache(contract)
        for cache in (actual, expected):
            cache._commit_token_counts = [2] * 4
            cache._commit_token_tags = [(0, 2)] * 4
            for index in range(50):
                values = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2) + index
                cache._history["blocks.%d.attn" % index] = module.AVKV(values.clone(), values.clone())
        old_first = weakref.ref(actual._history["blocks.0.attn"].key)
        validate = module._validate_av_pair

        def check(pair, contract, *, layer_name):
            """The second replacement must not retain the first layer's old storage."""
            if layer_name == "blocks.1.attn":
                self.assertIsNone(old_first())
            validate(pair, contract, layer_name=layer_name)

        with patch.object(module, "_validate_av_pair", side_effect=check):
            actual.retain_sink_and_recent_commits()
        expected.retain_sink_and_recent_commits()
        self.assertEqual(actual._commit_token_tags, expected._commit_token_tags)
        for name, pair in actual._history.items():
            self.assertTrue(torch.equal(pair.key, expected._history[name].key))
            self.assertTrue(torch.equal(pair.value, expected._history[name].value))

    def test_continuous_video_decode_joins_new_media_once(self):
        """Final decoding preserves global temporal phase without duplicated halos."""
        segments = [torch.arange(37).reshape(1, 1, 37, 1, 1), torch.arange(37, 72).reshape(1, 1, 35, 1, 1)]
        calls = []

        def decode(latent):
            """Expose global token order and H3 frame coverage at the decoder boundary."""
            calls.append(latent.clone())
            spans = torch.tensor([module.TaoMateStreaming.frames(i + 1) - module.TaoMateStreaming.frames(i) for i in range(72)])
            return latent.flatten().repeat_interleave(spans).reshape(-1, 1, 1, 1)

        frames = module.TaoMateStreaming.decode_video_timeline(decode, segments, 243)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0].flatten(), torch.arange(72)))
        self.assertEqual(frames.dtype, torch.float32)
        self.assertEqual(frames.shape[0], 243)
        self.assertEqual(frames[124].item(), 37)
        with self.assertRaisesRegex(ValueError, "expected 244"):
            module.TaoMateStreaming.decode_video_timeline(decode, segments, 244)

    def test_condition_attention_toggle_preserves_media_history_routing(self):
        """Condition rows gain current AV access but never direct historical KV."""
        cache = module.CPUStreamingCache(module.KVContract(local_heads=1, head_dim=4, dtype=torch.float32, device_type="cpu"))
        hook = module.ComfyStreamingHook(cache)
        query = torch.zeros(3, 1, 4)
        values = torch.tensor([1., 3., 8.]).reshape(3, 1, 1).expand(3, 1, 4)
        args = dict(attention=SimpleNamespace(softmax_scale=.5), layer_name="blocks.0.attn", query=query, key=query, value=values, token_tags=torch.tensor([1, 0, 2]), commit_mask=torch.tensor([False, True, True]), cu_seqlens_host=(0, 3))
        for history in (False, True):
            if history:
                cache._history["blocks.0.attn"] = module.AVKV(torch.zeros(1, 1, 4), torch.full((1, 1, 4), 20.))
            results = []
            for enabled in (False, True):
                hook.activate(module.HookMode.NOISY)
                with patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV", enabled):
                    results.append(hook(**args))
                hook.deactivate()
            self.assertTrue(torch.allclose(results[0][0], torch.ones(1, 4)))
            self.assertTrue(torch.allclose(results[1][0], torch.full((1, 4), 4.)))
            self.assertTrue(torch.equal(results[0][1:], results[1][1:]))
            self.assertTrue(torch.allclose(results[1][1:], torch.full((2, 1, 4), 8. if history else 4.)))

    def test_live_preview_accumulates_phases_with_global_frame_positions(self):
        """Later denoising updates replace their own phase, not the whole group."""
        preview = importlib.import_module(os.path.basename(ROOT) + ".preview")
        wrapper = preview._AccumulatedPreviewWrapper("test", 128, 75, 24, 1, "none")
        parts = module.TaoMateStreaming.plan(37, 207)
        count = 0
        for phase in parts:
            start = module.TaoMateStreaming.frames(phase["video_start"])
            indices, durations, numbers = preview._frame_selection(phase["video_end"] - phase["video_start"], 0, 1, 24, start, phase["video_start"])
            self.assertEqual(numbers[0], start)
            self.assertLessEqual(abs(sum(durations) - (phase["frame_end"] - start) * 1000 / 24), 1)
            part = {"chunk": 0, "subchunk_start": start, "output_start": start, "output_end": phase["frame_end"] - 1, "frames": [str(number) for number in numbers], "frame_numbers": numbers, "frame_durations_ms": durations}
            merged = wrapper.merge_subchunk_preview(part)
            count += len(indices)
            self.assertEqual(len(merged["frames"]), count)
            self.assertEqual(merged["output_start"], 0)
            self.assertEqual(merged["output_end"], phase["frame_end"] - 1)
            updated = dict(part, frames=["updated"] * len(indices))
            replaced = wrapper.merge_subchunk_preview(updated)
            self.assertEqual(len(replaced["frames"]), count)
            self.assertEqual(replaced["frame_numbers"], merged["frame_numbers"])
            self.assertEqual(replaced["frames"][:-len(indices)], merged["frames"][:-len(indices)])
        self.assertEqual(merged["output_end"], 123)
        self.assertEqual(len(merged["frame_numbers"]), len(set(merged["frame_numbers"])))

    def test_taomate_requests_lazy_sampler_and_sigmas(self):
        """ComfyUI must evaluate connected sampler/scheduler nodes before execute."""
        nodes = importlib.import_module(os.path.basename(ROOT) + ".nodes")
        settings = {"video_continuation_method": nodes.VIDEO_CONTINUATION_METHOD_TAOMATE}
        missing = nodes.HREndlessSampler.check_lazy_status(noise=object(), clip=object(), **settings)
        self.assertEqual(missing, ["sampler", "sigmas"])
        ready = nodes.HREndlessSampler.check_lazy_status(noise=object(), clip=object(), sampler=object(), sigmas=torch.tensor([1., 0.]), **settings)
        self.assertEqual(ready, [])

    def test_first_step_diagnostic_is_observational_and_saves_exact_reference(self):
        """Capture once per branch without modifying inputs or consuming RNG."""
        diagnostics = importlib.import_module(os.path.basename(ROOT) + ".python.h3_diagnostics")
        video = torch.zeros(1, 24, 2, 2, 2)
        audio = torch.zeros(1, 32, 2, 8)
        picture = torch.ones(1, 24, 1, 2, 2, dtype=torch.bfloat16)
        context = torch.zeros(1, 4, 96)
        payload = {"refs": [{"kind": "image", "latent_h": 2, "latent_w": 2}], "cond_video_latents": [picture]}
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics.tempfile, "mkdtemp", return_value=directory):
            capture = diagnostics.H3FirstStepDiagnostic("test", {"sigmas": torch.tensor([1., 0.]), "noise": [video, audio]})
            result = object()
            executor = lambda *args, **kwargs: result
            self.assertIs(capture.forward(executor, [video, audio], torch.tensor([1000.]), context, {"cond_or_uncond": [0]}, minimax_payload=payload), result)
            capture.capture([video, audio], torch.tensor([500.]), context, {"cond_or_uncond": [0]}, payload)
            saved = torch.load(os.path.join(directory, "model-branch-0.pt"), weights_only=True)
            self.assertTrue(torch.equal(saved["payload"]["cond_video_latents"][0], picture))
            self.assertEqual(saved["timestep"].item(), 1000.)
            self.assertTrue(any(segment[2] == "ref_img" for segment in saved["segments"]))
            capture.routing((0,), torch.tensor([1, 0, 2]), torch.tensor([False, True, True]), 0)
            route = torch.load(os.path.join(directory, "routing-branch-0.pt"), weights_only=True)
            self.assertEqual(route["condition_mask"].tolist(), [True, False, False])
            self.assertEqual(route["history_tokens"], 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(video.any())
        self.assertTrue(torch.all(picture == 1))

    def test_audio_publication_decodes_joined_latents_without_group_normalization(self):
        """A temporal decoder must see across group boundaries and keep gain."""
        calls = []

        class AudioVAE:
            """Emulate a decoder whose samples depend on neighboring latents."""

            audio_sample_rate = 32000

            def decode(self, latent):
                """Use a small temporal filter that exposes independent-decode seams."""
                calls.append(latent.clone())
                return torch.nn.functional.avg_pool1d(latent, 3, stride=1, padding=1).movedim(1, -1)

        segments = [torch.full((1, 2, 5), 2.0), torch.full((1, 2, 4), 4.0)]
        waveform, rate = module.TaoMateStreaming.decode_audio_timeline(AudioVAE(), segments)
        expected = torch.nn.functional.avg_pool1d(torch.cat(segments, dim=-1), 3, stride=1, padding=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(rate, 32000)
        self.assertTrue(torch.equal(waveform, expected))
        self.assertGreater(float(waveform.max()), 1.0)

    def test_group_audio_boundaries_match_upstream_across_rounding_cycle(self):
        """Preserve upstream's alternating 198/199-tick continuation lengths."""
        groups = module.TaoMateStreaming.request_plan(142, round(module.TaoMateStreaming.frames(142) * module.h3.FRAME_RESCALE))
        for index, group in enumerate(groups):
            upstream = module.direct_5s_plan()
            if index:
                upstream = module.canonical_continuation_plan(upstream, request_index=index)
            for actual, expected in zip(group["phases"], upstream.phases):
                self.assertEqual(actual["audio_start"] - group["audio_start"], expected.audio_latent_start)
                self.assertEqual(actual["audio_end"] - group["audio_start"], expected.audio_latent_stop)

    def test_audio_preview_replacement_uses_absolute_boundaries(self):
        """Final preview audio slices reconstruct one decode without sample gaps."""
        preview = importlib.import_module(os.path.basename(ROOT) + ".preview")
        owner = preview._AccumulatedPreviewWrapper
        wrapper = SimpleNamespace(execution_id="test", node_id="1", final_audio={}, final_audio_rates={})
        waveform = torch.arange(101, dtype=torch.float32).reshape(1, 1, -1)
        ranges = [{"start": 0, "end": 4}, {"start": 5, "end": 9}, {"start": 10, "end": 14}]
        with patch.object(preview, "_send") as send, patch.object(preview, "_encode_audio_wav", return_value="wav"):
            owner.replace_audio_timeline(wrapper, "test", waveform, 101, ranges, 15)
            self.assertEqual(send.call_count, 3)
            self.assertTrue(torch.equal(torch.cat(list(wrapper.final_audio.values()), dim=-1), waveform))
            owner.replace_audio_timeline(wrapper, "old", waveform, 101, ranges, 15)
            self.assertEqual(send.call_count, 3)

    def test_geometry_conserves_full_audio_video_timeline(self):
        """Transport halos never duplicate generated frames or audio ticks."""
        for tokens in (2, 7, 12, 37, 72, 427):
            frames = module.TaoMateStreaming.frames(tokens)
            audio = round(frames * module.h3.FRAME_RESCALE)
            plan = module.TaoMateStreaming.plan(tokens, audio)
            self.assertEqual(sum(item["video_end"] - item["video_start"] for item in plan), tokens)
            self.assertEqual(sum(item["audio_end"] - item["audio_start"] for item in plan), audio)
            self.assertEqual(sum(item["frame_end"] - item["frame_start"] - item["output_trim_frames"] for item in plan), frames)
        plan = module.TaoMateStreaming.plan(72, 405)
        self.assertEqual([item["frame_end"] - item["frame_start"] - item["output_trim_frames"] for item in plan], [39, 34, 34, 17, 34, 34, 34, 17])

    def test_request_prompts_cover_four_phases(self):
        """Two nominal five-second prompts cover eight phases without lost media."""
        state = module.TaoMateStreaming()
        requests = state.request_plan(72, 405)
        self.assertEqual(len(requests), 2)
        self.assertEqual([len(item["phases"]) for item in requests], [4, 4])
        self.assertEqual([item["frame_end"] - item["frame_start"] - item["output_trim_frames"] for item in requests], [124, 119])
        guider = object()
        seen = []
        for request in requests:
            vt, at = request["context_video_t"], request["context_audio_t"]
            video = torch.zeros(1, 24, vt + request["video_end"] - request["video_start"], 2, 2)
            audio = torch.zeros(1, 32, 2, at + request["audio_end"] - request["audio_start"])
            latent = {"samples": module.comfy.nested_tensor.NestedTensor((video, audio))}
            origins = []

            def sample(noise, current_guider, sampler, sigmas, target):
                """Record identical conditioning and fixed text position across phases."""
                self.assertIs(current_guider, guider)
                streams = target["samples"].unbind()
                layout = state.forward(lambda *args, **kwargs: kwargs["minimax_payload"]["layout"], streams, torch.ones(1), torch.zeros(1, 4, 96), {})
                origins.append(layout.position_ids[0, 0].item())
                seen.append((streams[0].shape[2], streams[1].shape[-1]))
                return target, target

            completed = []
            result, _ = state.execute_chunk(sample, lambda seed, value: value, 1, latent["samples"], guider, None, latent, request, on_subchunk=completed.append)
            self.assertEqual(completed, [phase["frame_end"] - state.frames(request["video_start"]) for phase in request["phases"]])
            self.assertEqual(len(set(origins)), 1)
            self.assertEqual(result["samples"].unbind()[0].shape, video.shape)
            self.assertEqual(result["samples"].unbind()[1].shape, audio.shape)
        self.assertEqual(sum(v for v, a in seen), 72)
        self.assertEqual(sum(a for v, a in seen), 405)
        self.assertEqual(len(state.request_plan(42, round(state.frames(42) * module.h3.FRAME_RESCALE))[-1]["phases"]), 1)

    def test_global_positions_survive_prompt_length_changes(self):
        """Each target token keeps the upstream canonical global media position."""
        state = module.TaoMateStreaming()
        plan = state.plan(22, 122)
        executor = lambda *args, **kwargs: kwargs["minimax_payload"]["layout"]
        for index, item in enumerate(plan):
            state.begin_chunk(item)
            video = torch.zeros(1, 24, item["video_end"] - item["video_start"], 2, 2)
            audio = torch.zeros(1, 32, 2, item["audio_end"] - item["audio_start"])
            layout = state.forward(executor, [video, audio], torch.ones(1), torch.zeros(1, 4 + index, 96), {})
            video_begin = next(start for start, end, kind in layout.segments if kind == "video")
            expected = 4 + float(module.video_temporal_position(item["video_start"]))
            self.assertAlmostEqual(layout.position_ids[video_begin, 0].item(), expected)

    def test_prompt_reference_position_toggle(self):
        """Freeze conditioning across groups without changing media or keyframe timing."""
        executor = lambda *args, **kwargs: kwargs["minimax_payload"]["layout"]
        payload = {"refs": [{"kind": "image", "latent_h": 2, "latent_w": 2}, {"kind": "image", "latent_h": 2, "latent_w": 2}], "keyframes": [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 2, 2)}]}
        layouts = {}
        for enabled in (False, True):
            state = module.TaoMateStreaming()
            for start in (0, 37, 72):
                state.video_start = state.request_video_start = start
                state.audio_start = round(state.frames(start) * 40 / 24)
                with patch.object(module, "TOGGLE_TAOMATE_DIVERGENCY_MOVE_PROMPT_REFERENCES", enabled):
                    layout = state.forward(executor, [torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 8)], torch.ones(1), torch.zeros(1, 4, 96), {}, minimax_payload=payload)
                layouts[enabled, start] = layout.position_ids.clone()
                for begin, end, kind in layout.segments:
                    rows = layout.position_ids[begin:end]
                    if kind in ("text", "ref_img"):
                        expected = layouts[enabled, 0][begin:end].clone()
                        if enabled:
                            expected[:, 0] += float(module.video_temporal_position(start))
                        self.assertTrue(torch.allclose(rows, expected))
                    elif enabled:
                        self.assertTrue(torch.equal(rows, layouts[False, start][begin:end]))

    def test_renormalization_matches_upstream_statistics(self):
        """Normalize video features only; preserve the packed audio exactly."""
        state = module.TaoMateStreaming()
        state.video_shape = (1, 24, 12, 2, 2)
        torch.manual_seed(8)
        first = torch.randn(1, 1, 1152 + 100)
        self.assertTrue(torch.equal(state.renormalize(first), first))
        next_sample = first * 2 + 3
        corrected = state.renormalize(next_sample)
        self.assertTrue(torch.allclose(corrected[..., :1152], first[..., :1152], atol=1e-5))
        self.assertTrue(torch.equal(corrected[..., 1152:], next_sample[..., 1152:]))

    def test_native_h3_two_phase_denoise_and_clean_cache(self):
        """Exercise native H3 blocks, RoPE, SDPA and all 50 upstream KV commits."""
        self._run_native_phases(False)

    def test_native_references_keyframes_and_cfg_histories(self):
        """Conditioning survives real H3 forwards with independent CFG history."""
        self._run_native_phases(True)

    def _run_native_phases(self, conditioned):
        """Exercise all layers with or without AV conditioning and CFG."""
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = module.h3.MiniMaxH3Model(hidden_size=96, num_layers=50, token_refiner_num_layers=0, num_attention_heads=1, attention_head_dim=96, ffn_hidden_size=128, text_dim=96, timestep_input_dim=8, time_embed_hidden_size=32, time_embed_dim=16, dtype=torch.float32, device="cpu", operations=torch.nn)
        model.rope.inv_freq.fill_(0.01)
        state = module.TaoMateStreaming()
        for index, block in enumerate(model.blocks):
            block.attn.forward = state.attention_forward(block.attn, index)
        plan = state.plan(22, 122)
        sigmas = torch.tensor(module.select_time_shift_sigmas(shift_scale=12, state_indices=module.DISTILLED_STATE_INDICES))
        for item in plan:
            state.begin_chunk(item)
            video_shape = (1, 24, item["video_end"] - item["video_start"], 2, 2)
            audio_shape = (1, 32, 2, item["audio_end"] - item["audio_start"])
            noise = torch.randn(1, 1, torch.tensor(video_shape).prod().item() + torch.tensor(audio_shape).prod().item())
            context = torch.randn(1, 4, 96)
            calls = []

            def predict(packed, sigma, **kwargs):
                """Use the native model's velocity as a flow x0 prediction."""
                streams = module.comfy.utils.unpack_latents(packed, [video_shape, audio_shape])
                payload = {}
                if conditioned:
                    image = torch.randn(1, 24, 1, 2, 2)
                    sound = torch.randn(1, 32, 2, 2)
                    payload = {"keyframes": [{"resolved_frame_index": 0, "latent": image, "audio_latent": sound}], "refs": [{"kind": "image", "latent_h": 2, "latent_w": 2}, {"kind": "video_audio", "latent_t": 1, "latent_h": 2, "latent_w": 2, "ref_audio_t": 2}, {"kind": "audio", "ref_audio_t": 2}], "cond_video_latents": [image, image, image], "cond_audio_latents": [sound, sound, sound]}
                    # Negative and positive branches may have identical target shapes.
                    state.forward(model._forward, streams, sigma * 1000, context, {"cond_or_uncond": [1]})
                velocity = state.forward(model._forward, streams, sigma * 1000, context, {"cond_or_uncond": [0]}, minimax_payload=payload)
                if conditioned:
                    self.assertIn("ref_img", [kind for begin, end, kind in state.layout.segments])
                    self.assertIn("cond_audio", [kind for begin, end, kind in state.layout.segments])
                packed_velocity, _ = module.comfy.utils.pack_latents(velocity)
                calls.append(float(sigma[0]))
                return packed - sigma.reshape(1, 1, 1) * packed_velocity

            with torch.no_grad():
                result = state.sample(predict, noise, sigmas, disable=True)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[-1], 0)
            self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(state.caches), 2 if conditioned else 1)
        for cache, hook in state.caches.values():
            self.assertEqual(cache.committed_blocks, 2)
            self.assertEqual(cache.history_tokens, sum(part["video_end"] - part["video_start"] + 2 * (part["audio_end"] - part["audio_start"]) for part in plan))
        self.assertEqual(state.cache.committed_blocks, 2)
        self.assertEqual(len(state.cache._history), 50)
        self.assertTrue(all(pair.key.device.type == "cpu" for pair in state.cache._history.values()))
        state.close()
        self.assertEqual(state.cache.history_tokens, 0)

    def test_audio_teacher_uses_no_video_and_rolls_clean_tail(self):
        """Nine actual H3 forwards capture milestones and retain a frozen tail."""
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        teacher_module = importlib.import_module(os.path.basename(ROOT) + ".python.taomate_audio_teacher")
        model = module.h3.MiniMaxH3Model(hidden_size=96, num_layers=2, token_refiner_num_layers=0, num_attention_heads=1, attention_head_dim=96, ffn_hidden_size=128, text_dim=96, timestep_input_dim=8, time_embed_hidden_size=32, time_embed_dim=16, dtype=torch.float32, device="cpu", operations=torch.nn)
        model.rope.inv_freq.fill_(0.01)
        patcher = SimpleNamespace(load_device=torch.device("cpu"), model=SimpleNamespace(get_dtype_inference=lambda: torch.float32), model_options={}, pre_run=lambda: None, cleanup=lambda: None, get_model_object=lambda name: model)
        patcher.clone = lambda: patcher
        model.requires_grad_(False)
        teacher = teacher_module.Base10AudioTeacher(patcher)
        conditioning = {"cross_attn": torch.randn(1, 4, 96)}
        observed = []
        original = teacher.forward

        def forward(**kwargs):
            """Inspect the frozen reference throughout all nine updates."""
            rows = kwargs["audio_x"][0].index_select(0, teacher.branch.audio_pos_dev)
            observed.append(rows[:teacher.branch.audio_target_start].clone())
            return original(**kwargs)

        teacher.forward = forward
        with patch.object(module.h3, "optimized_attention", module.comfy.ldm.modules.attention.attention_pytorch), patch.object(teacher_module.comfy.model_management, "load_models_gpu"), patch.object(model.video_patch_proj, "forward", side_effect=AssertionError("video projection executed")), patch.object(model.final_layer.video_out, "forward", side_effect=AssertionError("video head executed")):
            first = teacher.generate(conditioning, torch.randn(1, 32, 2, 65), 2, 2)
            self.assertEqual(len(observed), 9)
            self.assertEqual(len(first), 3)
            self.assertTrue(all(value.shape == (1, 32, 2, 65) for value in first))
            observed.clear()
            requested = torch.tensor([1., .98, .96, .92, .85, .70, 0.])
            second = teacher.generate(conditioning, torch.randn(1, 32, 2, 57), 2, 2, requested, 8., 2.)
        self.assertEqual(len(observed), teacher.receipt["executed_forwards"])
        self.assertEqual(len(second), 6)
        expected = module.h3.pack_audio(first[-1][..., -40:])
        self.assertTrue(all(torch.equal(rows, expected) for rows in observed))
        self.assertEqual(teacher.receipt["reference_ticks"], 40)
        self.assertTrue(torch.equal(teacher.previous_clean, second[-1]))
        teacher.close()

    def test_teacher_schedule_captures_all_six_supplied_endpoints(self):
        """Extra video steps receive actual teacher states at their exact sigmas."""
        teacher_module = importlib.import_module(os.path.basename(ROOT) + ".python.taomate_audio_teacher")
        requested = torch.tensor([1., .98, .96, .92, .85, .70, 0.])
        original = requested.clone()
        video, audio, captures = teacher_module.Base10AudioTeacher.guidance_schedule(requested, 8., 2.)
        self.assertEqual([video[i] for i in captures], requested.tolist()[1:])
        self.assertEqual(len(captures), 6)
        self.assertTrue(torch.equal(requested, original))
        self.assertEqual(len(video), len(audio))
        for sigma, audio_sigma in zip(video, audio):
            self.assertAlmostEqual(audio_sigma, .25 * sigma / (1. - .75 * sigma))

    def test_teacher_injection_uses_native_audio_carry(self):
        """Each next forward and clean commit receive the teacher milestone."""
        state = module.TaoMateStreaming()
        state.video_shape = (1, 24, 2, 2, 2)
        state.phase_audio_start, state.phase_audio_end = 0, 8
        state.audio_scale = 4.0
        state.teacher_milestones = [torch.full((1, 32, 2, 8), value) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
        state.renormalize = lambda value: value
        sigmas = torch.tensor([1.0, 0.98, 0.96, 0.92, 0.85, 0.70, 0.0])
        seen = []

        def predict(value, sigma, **kwargs):
            """Return a deliberately different audio prediction to expose injection."""
            seen.append(value.clone())
            return torch.zeros_like(value)

        result = state.sample(predict, torch.ones(1, 1, 192 + 512), sigmas, disable=True)
        self.assertEqual(len(seen), 7)  # Six denoising calls plus clean KV capture.
        for step in range(6):
            expected = state.teacher_milestones[step].reshape(1, 1, -1) * (4.0 - 3.0 * float(sigmas[step + 1]))
            self.assertTrue(torch.equal(seen[step + 1][..., 192:], expected))
        self.assertTrue(torch.equal(result[..., 192:] / 4, state.teacher_milestones[-1].reshape(1, 1, -1)))

    def test_selected_sampler_runs_once_and_preserves_options(self):
        """Do not restart the selected multistep solver or discard its options."""
        state = module.TaoMateStreaming()
        state.renormalize = lambda value: value
        calls = []
        sigmas = torch.tensor([1., .8, .5, 0.])

        def solver(model, x, supplied, extra_args=None, callback=None, disable=None, custom=None):
            """Emulate a solver that owns its complete integration history."""
            calls.append((supplied, custom))
            return x + 3

        selected = module.comfy.samplers.KSAMPLER(solver, extra_options={"custom": 7}, inpaint_options={"random": True})
        wrapped = state.wrap_sampler(selected)
        self.assertIs(selected.sampler_function, solver)
        self.assertEqual(wrapped.extra_options, selected.extra_options)
        self.assertEqual(wrapped.inpaint_options, selected.inpaint_options)
        result = wrapped.sampler_function(lambda x, sigma: x, torch.zeros(1, 1, 4), sigmas, **wrapped.extra_options)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], sigmas)
        self.assertEqual(calls[0][1], 7)
        self.assertTrue(torch.all(result == 3))

    def test_heun_internal_stage_receives_teacher_audio(self):
        """A real non-Euler solver keeps its extra evaluations and exact final audio."""
        from comfy.k_diffusion.sampling import sample_heun
        state = module.TaoMateStreaming()
        state.video_shape = (1, 24, 2, 2, 2)
        state.phase_audio_start, state.phase_audio_end = 0, 8
        state.audio_scale = 4.
        state.teacher_milestones = [torch.full((1, 32, 2, 8), .2), torch.full((1, 32, 2, 8), .6)]
        state.renormalize = lambda value: value
        observed = []

        def predict(value, sigma, **kwargs):
            """Record every Heun stage's actual conditioned audio."""
            observed.append((float(sigma[0]), value[..., 192:].clone()))
            return torch.zeros_like(value)

        sigmas = torch.tensor([1., .5, 0.])
        result = state.sample(predict, torch.ones(1, 1, 704), sigmas, disable=True, sampler_function=sample_heun)
        self.assertEqual([sigma for sigma, audio in observed], [1., .5, .5, 0.])
        self.assertTrue(torch.equal(observed[1][1], torch.full((1, 1, 512), .5)))
        self.assertTrue(torch.equal(result[..., 192:], torch.full((1, 1, 512), 2.4)))

    def test_phase_preparation_initializes_guidance_shape_before_solver(self):
        """Exercise preparation through wrapped Heun without prefilling video_shape."""
        from comfy.k_diffusion.sampling import sample_heun
        state = module.TaoMateStreaming()
        self.assertIsNone(state.video_shape)
        state.renormalize = lambda value: value
        clean = torch.full((1, 32, 2, 207), .25)
        state.audio_teacher = SimpleNamespace(generate=lambda *args: [clean] * 6, receipt={})
        sampling = SimpleNamespace(audio_scale=4., shift=12., audio_shift=3.)
        guider = SimpleNamespace(original_conds={"positive": [{}]}, model_patcher=SimpleNamespace(get_model_object=lambda name: sampling))
        video = torch.zeros(1, 24, 37, 2, 2)
        latent = {"samples": module.comfy.nested_tensor.NestedTensor((video, torch.zeros_like(clean)))}
        sigmas = torch.tensor([1., .98, .96, .92, .85, .70, 0.])
        shapes = []

        def execute(noise, guider, sampler, schedule, target):
            """Pack inputs as ComfyUI does, then invoke the actual wrapped solver."""
            streams = target["samples"].unbind()
            shapes.append(tuple(streams[0].shape))
            packed, layout = module.comfy.utils.pack_latents(streams)
            result = sampler.sampler_function(lambda x, sigma: torch.zeros_like(x), torch.ones_like(packed), schedule, disable=True)
            generated = module.comfy.utils.unpack_latents(result, layout)
            target["samples"] = module.comfy.nested_tensor.NestedTensor((generated[0], generated[1] / 4.))
            return target, dict(target)

        output, _ = state.execute_chunk(execute, lambda seed, value: value, 1, latent["samples"], guider, sigmas, latent, state.request_plan(37, 207)[0], sampler=module.comfy.samplers.KSAMPLER(sample_heun))
        self.assertEqual([shape[2] for shape in shapes], [12, 10, 10, 5])
        self.assertEqual(state.video_shape, shapes[-1])
        self.assertTrue(torch.equal(output["samples"].unbind()[1], clean))

    def test_published_audio_must_equal_teacher(self):
        """Reject altered published audio rather than reporting a false match."""
        state = module.TaoMateStreaming()
        request = state.request_plan(37, 207)[0]
        clean = torch.randn(1, 32, 2, 207)
        state.audio_teacher = SimpleNamespace(generate=lambda *args: [clean, clean, clean], receipt={})
        guider = SimpleNamespace(original_conds={"positive": [{}]}, model_patcher=SimpleNamespace(get_model_object=lambda name: SimpleNamespace(audio_scale=4.0, shift=12.0, audio_shift=3.0)))
        video = torch.zeros(1, 24, 37, 2, 2)
        latent = {"samples": module.comfy.nested_tensor.NestedTensor((video, torch.zeros_like(clean)))}

        def sample(noise, guider, sampler, sigmas, target):
            """Return the corresponding teacher phase as the native sampler would."""
            current_video, audio = target["samples"].unbind()
            audio = clean[..., state.phase_audio_start:state.phase_audio_end].clone()
            target["samples"] = module.comfy.nested_tensor.NestedTensor((current_video, audio))
            return target, target

        result, _ = state.execute_chunk(sample, lambda seed, value: value, 1, latent["samples"], guider, None, latent, request)
        self.assertTrue(torch.equal(result["samples"].unbind()[1], clean))
        self.assertTrue(state.audio_teacher.receipt["published_clean_audio_exact_match"])

        def corrupt(*args):
            """Simulate an incorrect scale or output transformation."""
            result, other = sample(*args)
            result["samples"].unbind()[1].add_(1)
            return result, other

        with self.assertRaisesRegex(RuntimeError, "differs from the teacher"):
            state.execute_chunk(corrupt, lambda seed, value: value, 1, latent["samples"], guider, None, latent, request)

    def test_reuse_preparation_only_while_model_is_resident(self):
        """Skip redundant loading, but respect eviction, controls and memory growth."""
        state = module.TaoMateStreaming()
        state.reuse_preparation = True
        model = SimpleNamespace(model=object())
        calls = []
        current = {"positive": [{}]}

        def prepare(model, shape, conds, **kwargs):
            """Stand in for native loading and retain the supplied conditioning."""
            calls.append(shape)
            return model.model, conds, []

        with patch.object(module.comfy.model_management, "current_loaded_models", [SimpleNamespace(model=model)]), patch.object(module.comfy.sampler_helpers, "estimate_memory", side_effect=lambda model, shape, conds: (shape[-1], shape[-1])):
            state.prepare_sampling(prepare, model, (1, 100), current)
            next_conds = {"positive": [{"new_phase": True}]}
            result = state.prepare_sampling(prepare, model, (1, 80), next_conds)
            self.assertIs(result[1], next_conds)
            self.assertEqual(len(calls), 1)
            state.prepare_sampling(prepare, model, (1, 120), current)
            self.assertEqual(len(calls), 2)
            state.prepare_sampling(prepare, model, (1, 80), {"positive": [{"control": object()}]})
            self.assertEqual(len(calls), 3)
            state.prepare_sampling(prepare, model, (1, 80), current)
            self.assertEqual(len(calls), 4)
            module.comfy.model_management.current_loaded_models.clear()
            state.prepare_sampling(prepare, model, (1, 80), current)
            self.assertEqual(len(calls), 5)

    def test_decode_halo_never_enters_inference(self):
        """Historical AV is restored for decoding, without being sampled again."""
        state = module.TaoMateStreaming()
        chunk = state.plan(22, 122)[1]
        vt, at = chunk["context_video_t"], chunk["context_audio_t"]
        video = torch.ones(1, 24, vt + 10, 2, 2)
        audio = torch.ones(1, 32, 2, at + 57)
        video[:, :, vt:] = 0
        audio[..., at:] = 0
        latent = {"samples": module.comfy.nested_tensor.NestedTensor((video, audio))}
        noise = module.comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.ones_like(audio)))

        def sample(noise, guider, sampler, sigmas, target):
            """Assert that only new, empty media reaches the sampler."""
            streams = target["samples"].unbind()
            self.assertEqual(streams[0].shape[2], 10)
            self.assertEqual(streams[1].shape[-1], 57)
            self.assertTrue(all(torch.count_nonzero(stream) == 0 for stream in streams))
            target["samples"] = module.comfy.nested_tensor.NestedTensor(tuple(stream + 2 for stream in streams))
            return target, target

        result, _ = state.execute_chunk(sample, lambda seed, value: value, 1, noise, None, None, latent, chunk)
        video, audio = result["samples"].unbind()
        self.assertTrue(torch.all(video[:, :, :vt] == 1))
        self.assertTrue(torch.all(audio[..., :at] == 1))
        self.assertTrue(torch.all(video[:, :, vt:] == 2))
        self.assertTrue(torch.all(audio[..., at:] == 2))

    def test_debug_timing_reports_each_subchunk(self):
        """Debug timing exposes the host sampler boundary without changing samples."""
        state = module.TaoMateStreaming()
        chunk = state.plan(2, round(state.frames(2) * module.h3.FRAME_RESCALE))[0]
        video = torch.zeros(1, 24, 2, 2, 2)
        audio = torch.zeros(1, 32, 2, round(state.frames(2) * module.h3.FRAME_RESCALE))
        latent = {"samples": module.comfy.nested_tensor.NestedTensor((video, audio))}

        def sample(noise, guider, sampler, sigmas, target):
            """Return the supplied target, isolating timing publication."""
            return target, target

        with self.assertLogs(level="INFO") as output:
            state.execute_chunk(sample, lambda seed, value: value, 1, latent["samples"], None, None, latent, chunk, debug_timing=True)
        self.assertTrue(any("TaoMate profile group 1 sub-chunk 1/1" in line and "Comfy sampler total" in line for line in output.output))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU KV codecs require CUDA")
    def test_gpu_lossy_kv_codecs_restore_h3_shapes(self):
        """INT8 and TurboQuant keep compact CPU rows and reconstruct finite H3 KV."""
        pair = module.AVKV(torch.randn(11, 2, 128, device="cuda", dtype=torch.bfloat16), torch.randn(11, 2, 128, device="cuda", dtype=torch.bfloat16))
        for mode in ("int8", "turboquant"):
            compressed = module.QuantizedKV(pair, mode)
            self.assertLess(compressed.stored_bytes, compressed.raw_bytes)
            restored_key = compressed.tensor(0, "cuda")
            restored_value = compressed.tensor(1, "cuda")
            self.assertEqual(restored_key.shape, pair.key.shape)
            self.assertEqual(restored_value.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(restored_key).all())
            self.assertTrue(torch.isfinite(restored_value).all())
            similarity = torch.nn.functional.cosine_similarity(restored_key.float().reshape(-1, 128), pair.key.float().reshape(-1, 128), dim=-1).mean()
            self.assertGreater(similarity, 0.95)
            self.assertEqual(module.QuantizedKV.concat(compressed, compressed).shape[0], 22)
            self.assertEqual(compressed.select(torch.tensor([1, 7])).shape[0], 2)


if __name__ == "__main__":
    unittest.main()
