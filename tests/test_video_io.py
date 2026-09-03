from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT.parent))

video_io = importlib.import_module(PLUGIN_ROOT.name + ".video_io")
nodes = importlib.import_module(PLUGIN_ROOT.name + ".nodes")


class FinishedVideoIOTest(unittest.TestCase):
    def test_sampler_exposes_timeline_as_an_additive_fourth_output(self):
        schema = nodes.HREndlessSampler.define_schema()
        self.assertEqual(len(schema.outputs), 4)
        self.assertEqual(schema.outputs[-1].io_type, "HRENDLESS_TIMELINE")

    def test_load_video_preserves_existing_outputs_and_adds_media_components_and_geometry(self):
        schema = video_io.HREndlessSamplerLoadVideo.define_schema()
        self.assertEqual(
            [item.display_name for item in schema.outputs],
            [
                "timeline",
                "filename",
                "fps",
                "video",
                "images",
                "audio",
                "frame_count",
                "width",
                "height",
            ],
        )
        self.assertEqual(
            [item.io_type for item in schema.outputs],
            ["HRENDLESS_TIMELINE", "STRING", "FLOAT", "VIDEO", "IMAGE", "AUDIO", "INT", "INT", "INT"],
        )

    def test_load_video_execute_returns_decoded_components_and_dimensions(self):
        images = torch.zeros((7, 18, 32, 3), dtype=torch.float32)
        audio = {"waveform": torch.zeros((1, 2, 100)), "sample_rate": 48_000}
        video = object()
        timeline = video_io.normalize_timeline(None, fps=24, total_frames=7)

        def fake_load(_path, _fps, *, decoded):
            decoded.update({"video": video, "images": images, "audio": audio})
            return timeline, "/tmp/render.mp4", 24.0, {"timeline": timeline}

        with patch.object(video_io, "_load_video_payload", side_effect=fake_load):
            result = video_io.HREndlessSamplerLoadVideo.execute("render.mp4")

        self.assertEqual(result.result[:3], (timeline, "/tmp/render.mp4", 24.0))
        self.assertIs(result.result[3], video)
        self.assertIs(result.result[4], images)
        self.assertIs(result.result[5], audio)
        self.assertEqual(result.result[6:], (7, 32, 18))

    def test_timeline_normalization_keeps_prompt_metadata_and_repairs_missing_ranges(self):
        timeline = video_io.normalize_timeline(
            {
                "fps": 24,
                "total_frames": 10,
                "render_total_seconds": 95.25,
                "chunks": [
                    {
                        "chunk": 7,
                        "start": -2,
                        "end": 4,
                        "gemma_detailed_description": "  Continue the action.  ",
                        "h3_render_seconds": 42.5,
                        "gemma_seconds": 3.25,
                        "gemma_preproduction_seconds": 2.0,
                        "chunk_total_seconds": 48.75,
                    },
                    {"chunk": 8, "start": 9, "end": 3},
                ],
                "shots": [{"shot": 3, "start": 4, "end": 20, "source_end": 24}],
            },
            fps=30,
            total_frames=10,
        )
        self.assertEqual(timeline["fps"], 24.0)
        self.assertEqual(timeline["chunks"], [{
            "chunk": 7,
            "start": 0,
            "end": 4,
            "gemma_detailed_description": "Continue the action.",
            "h3_render_seconds": 42.5,
            "gemma_seconds": 3.25,
            "gemma_preproduction_seconds": 2.0,
            "chunk_total_seconds": 48.75,
        }])
        self.assertEqual(timeline["render_total_seconds"], 95.25)
        self.assertEqual(timeline["shots"], [{"shot": 3, "start": 4, "end": 9, "source_end": 24}])

    def test_float_exr_round_trip_preserves_values_outside_zero_one_without_clamp(self):
        image = torch.tensor(
            [[[-1.25, 0.0, 1.75], [0.25, 0.5, 0.75]], [[2.5, -0.5, 0.125], [0.2, 1.0, 3.0]]],
            dtype=torch.float32,
        )
        encoded = video_io._tensor_to_exr_bytes(image, "float", "none")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.exr"
            path.write_bytes(encoded)
            with av.open(str(path)) as container:
                decoded = next(container.decode(video=0)).to_ndarray(format="gbrpf32le")
        self.assertTrue(torch.allclose(torch.from_numpy(decoded), image, atol=0.0, rtol=0.0))

    def test_exr_sequence_writes_only_the_first_frame_embedded_timeline_and_a_sidecar_manifest(self):
        images = torch.linspace(-0.5, 1.5, 2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)
        timeline = video_io.normalize_timeline(None, fps=24, total_frames=2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(video_io.folder_paths, "get_output_directory", return_value=directory):
            primary, paths, _subfolder = video_io._save_exr_sequence(
                images,
                "sequence/test",
                "float",
                "zip16",
                timeline,
            )
            sidecar = video_io._write_sidecar(
                primary,
                timeline,
                {"kind": "exr_sequence", "frames": [path.name for path in paths]},
            )
            sidecar_timeline, media = video_io._read_sidecar(primary)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar_timeline, timeline)
            self.assertEqual(media["frames"], [path.name for path in paths])
            self.assertEqual(video_io._read_embedded_timeline(primary), timeline)

    def test_video_combine_adapter_passes_vhs_pixel_format_crf_audio_and_timeline_metadata(self):
        captured = {}

        class FakeCombine:
            def combine_video(self, **kwargs):
                captured.update(kwargs)
                return {"result": ((True, ["/tmp/metadata.png", "/tmp/render.mp4"]),)}

        class FakeVHS:
            VideoCombine = FakeCombine

        images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        audio = {"waveform": torch.zeros((1, 2, 32), dtype=torch.float32), "sample_rate": 48_000}
        timeline = video_io.normalize_timeline(None, fps=24, total_frames=2)
        with patch.object(video_io, "_vhs_nodes", return_value=FakeVHS):
            path = video_io._save_vhs_video(images, "render", "video/h264-mp4", 24, "yuv420p10le", 17, timeline, audio=audio)
        self.assertEqual(path, Path("/tmp/render.mp4"))
        self.assertEqual(captured["pix_fmt"], "yuv420p10le")
        self.assertEqual(captured["crf"], 17)
        self.assertIs(captured["audio"], audio)
        self.assertEqual(captured["extra_pnginfo"][video_io.TIMELINE_TAG], timeline)

    def test_vhs_discovery_reuses_the_submodule_loaded_by_comfyui(self):
        fake_vhs = types.SimpleNamespace(get_video_formats=lambda: ([], {}), VideoCombine=object)
        loaded_name = "/custom_nodes/ComfyUI-VideoHelperSuite.videohelpersuite.nodes"
        with patch.object(video_io.importlib, "import_module", side_effect=ModuleNotFoundError("standalone import unavailable")), \
                patch.dict(video_io.sys.modules, {loaded_name: fake_vhs}):
            self.assertIs(video_io._vhs_nodes(), fake_vhs)

    def test_native_h264_save_uses_comfy_video_api_without_vhs(self):
        captured = {}

        class FakeVideo:
            def save_to(self, path, **kwargs):
                captured["path"] = path
                captured.update(kwargs)

        def make_video(components, bit_depth):
            captured["components"] = components
            captured["bit_depth"] = bit_depth
            return FakeVideo()

        images = torch.zeros((2, 4, 6, 3), dtype=torch.float32)
        audio = {"waveform": torch.zeros((1, 2, 32)), "sample_rate": 48_000}
        timeline = video_io.normalize_timeline(None, fps=24, total_frames=2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(video_io.folder_paths, "get_output_directory", return_value=directory), \
                patch.object(video_io.folder_paths, "get_save_image_path", return_value=(directory, "render", 7, "", "render")), \
                patch.object(video_io.InputImpl, "VideoFromComponents", side_effect=make_video), \
                patch.object(video_io, "_vhs_nodes", side_effect=AssertionError("native H.264 must not touch VHS")):
            path = video_io._save_native_h264(
                images, "render", 24, "yuv420p10le", 17, timeline, audio=audio,
            )

        self.assertEqual(path, Path(directory) / "render_00007_.mp4")
        self.assertEqual(captured["bit_depth"], 10)
        self.assertIs(captured["components"].audio, audio)
        self.assertEqual(float(captured["components"].frame_rate), 24.0)
        self.assertEqual(captured["crf"], 17.0)
        self.assertEqual(captured["metadata"], {video_io.TIMELINE_TAG: timeline})

    def test_output_browser_lists_folders_videos_and_one_entry_per_exr_sequence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            video_io.folder_paths, "get_output_directory", return_value=directory,
        ):
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "movie.mp4").write_bytes(b"video")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            frames = [root / f"render_00001_{index:06}.exr" for index in range(3)]
            for frame in frames:
                frame.write_bytes(b"exr")
            timeline = video_io.normalize_timeline(None, fps=24, total_frames=3)
            video_io._write_sidecar(
                frames[0], timeline, {"kind": "exr_sequence", "frames": [item.name for item in frames]},
            )
            (root / "standalone.exr").write_bytes(b"single")

            listing = video_io._output_browser_listing("")

        by_kind = {}
        for entry in listing["entries"]:
            by_kind.setdefault(entry["kind"], []).append(entry)
        self.assertEqual([item["name"] for item in by_kind["directory"]], ["nested"])
        self.assertEqual([item["name"] for item in by_kind["video"]], ["movie.mp4"])
        self.assertEqual(len(by_kind["sequence"]), 1)
        self.assertEqual(by_kind["sequence"][0]["frames"], 3)
        self.assertEqual([item["name"] for item in by_kind["exr"]], ["standalone.exr"])

    def test_output_browser_rejects_paths_outside_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            video_io.folder_paths, "get_output_directory", return_value=directory,
        ):
            with self.assertRaisesRegex(ValueError, "leaves ComfyUI's output directory"):
                video_io._output_browser_directory("../elsewhere")

    def test_uploaded_video_filename_is_sanitized_and_rejects_non_video(self):
        self.assertEqual(video_io._safe_upload_filename("../../My clip (1).mp4"), "My clip (1).mp4")
        with self.assertRaisesRegex(ValueError, "supported video"):
            video_io._safe_upload_filename("payload.py")

    def test_matching_output_uses_save_prefix_or_strips_the_loaded_counter_suffix(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            video_io.folder_paths, "get_output_directory", return_value=directory,
        ):
            folder = Path(directory) / "video"
            folder.mkdir()
            older = folder / "scene_2026_00001_.mp4"
            newer = folder / "scene_2026_00002_.mp4"
            unrelated = folder / "different_00003_.mp4"
            for path in (older, newer, unrelated):
                path.write_bytes(b"video")
            older.touch()
            newer.touch()
            unrelated.touch()
            older_time = 1_700_000_000
            newer_time = older_time + 100
            import os
            os.utime(older, (older_time, older_time))
            os.utime(newer, (newer_time, newer_time))

            from_load = video_io._matching_output_listing(
                "video/scene_2026_00002_.mp4", strip_counter=True,
            )
            from_save = video_io._matching_output_listing("video/scene_2026")

        expected = [newer.name, older.name]
        self.assertEqual(from_load["prefix"], "video/scene_2026")
        self.assertEqual([item["name"] for item in from_load["entries"]], expected)
        self.assertEqual([item["name"] for item in from_save["entries"]], expected)

    def test_exr_proxy_uses_native_h264_without_vhs(self):
        images = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        timeline = video_io.normalize_timeline(None, fps=24, total_frames=2)
        with patch.object(video_io, "_save_native_h264", return_value=Path("/tmp/proxy.mp4")) as native, \
                patch.object(video_io, "_save_vhs_video", side_effect=AssertionError("proxy must not touch VHS")):
            result = video_io._make_proxy(images, 24, timeline)
        self.assertEqual(result, Path("/tmp/proxy.mp4"))
        self.assertFalse(native.call_args.kwargs["save_output"])

    def test_load_payload_returns_player_state_without_workflow_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            video_io.folder_paths, "get_output_directory", return_value=directory,
        ):
            path = Path(directory) / "movie.mp4"
            path.write_bytes(b"placeholder")
            timeline = video_io.normalize_timeline(
                {"chunks": [{"chunk": 1, "start": 0, "end": 11}]}, fps=24, total_frames=12,
            )
            video_io._write_sidecar(path, timeline, {"kind": "video", "filename": path.name})
            with patch.object(video_io, "_probe_video", return_value=(24.0, 12, 64, 32)):
                restored, filename, fps, state = video_io._load_video_payload("movie.mp4")

        self.assertEqual(restored, timeline)
        self.assertEqual(filename, str(path))
        self.assertEqual(fps, 24.0)
        self.assertEqual(state["timeline"], timeline)
        self.assertIn("filename=movie.mp4", state["media_url"])

    def test_save_node_exposes_an_optional_audio_input(self):
        schema = video_io.HREndlessSamplerSaveVideo.define_schema()
        audio = next(item for item in schema.inputs if item.id == "audio")
        self.assertTrue(audio.optional)

    def test_save_prompt_repairs_a_stale_upstream_output_slot_by_type(self):
        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={
            "Sampler": types.SimpleNamespace(RETURN_TYPES=("LATENT", "STRING", "HRENDLESS_TIMELINE")),
        })
        prompt = {
            "prompt": {
                "1": {"class_type": "Sampler", "inputs": {}},
                "2595": {"class_type": "HREndlessSamplerSaveVideo", "inputs": {"timeline": ["1", 4]}},
            },
        }
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            repaired = video_io._repair_save_video_prompt_links(prompt)
        self.assertEqual(repaired["prompt"]["2595"]["inputs"]["timeline"], ["1", 2])

    def test_prompt_repairs_transitional_qwen_widget_order(self):
        inputs = {
            "cache_gemma_preproduction": 2,
            "gemma4_mtp": "xhigh",
            "pytorch_memory_fraction": False,
            "debug": 0,
            "debug_stop_chunk": False,
            "debug_start_chunk": True,
            "director_backend": 0.85,
            "director_model": True,
            "director_mmproj": 0,
            "director_mtp_draft_tokens": 0,
            "director_reasoning_effort": "qwen3.8",
            "director_cpu_moe": "LLM/GGUF/qwen3.8-27B/model.gguf",
            "director_n_cpu_moe": "LLM/GGUF/qwen3.8-27B/mmproj-model.gguf",
        }
        prompt = {"prompt": {"2599": {"class_type": "HREndlessSampler", "inputs": inputs}}}
        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            video_io._repair_save_video_prompt_links(prompt)
        self.assertEqual(inputs["director_backend"], "qwen3.8")
        self.assertEqual(inputs["director_model"], "LLM/GGUF/qwen3.8-27B/model.gguf")
        self.assertEqual(inputs["director_mmproj"], "LLM/GGUF/qwen3.8-27B/mmproj-model.gguf")
        self.assertEqual(inputs["director_mtp_draft_tokens"], 2)
        self.assertEqual(inputs["director_reasoning_effort"], "xhigh")
        self.assertEqual(inputs["pytorch_memory_fraction"], 0.85)
        self.assertTrue(inputs["debug"])

    def test_prompt_restores_zero_memory_fraction_before_validation(self):
        inputs = {"pytorch_memory_fraction": 0.0}
        prompt = {"prompt": {"2599": {"class_type": "HREndlessSampler", "inputs": inputs}}}
        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            video_io._repair_save_video_prompt_links(prompt)
        self.assertEqual(inputs["pytorch_memory_fraction"], 0.85)

    def test_save_prompt_keeps_valid_v3_timeline_output(self):
        class V3Sampler:
            @classmethod
            def define_schema(cls):
                return nodes.HREndlessSampler.define_schema()

        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"HREndlessSampler": V3Sampler})
        prompt = {
            "prompt": {
                "2576": {"class_type": "HREndlessSampler", "inputs": {}},
                "2595": {"class_type": "HREndlessSamplerSaveVideo", "inputs": {"timeline": ["2576", 3]}},
            },
        }
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            repaired = video_io._repair_save_video_prompt_links(prompt)
        self.assertEqual(repaired["prompt"]["2595"]["inputs"]["timeline"], ["2576", 3])

    def test_save_prompt_removes_an_unrecoverable_stale_output_slot(self):
        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={
            "Decoder": types.SimpleNamespace(RETURN_TYPES=("LATENT",)),
        })
        prompt = {
            "prompt": {
                "1": {"class_type": "Decoder", "inputs": {}},
                "2595": {"class_type": "HREndlessSamplerSaveVideo", "inputs": {"images": ["1", 3]}},
            },
        }
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            repaired = video_io._repair_save_video_prompt_links(prompt)
        self.assertNotIn("images", repaired["prompt"]["2595"]["inputs"])

    def test_exr_audio_sidecar_round_trip_keeps_float_audio(self):
        audio = {
            "waveform": torch.tensor([[[0.0, -0.25, 0.5, 1.0], [0.25, 0.0, -0.5, 0.75]]]),
            "sample_rate": 48_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = video_io._write_wav_audio(audio, Path(directory) / "render.exr")
            restored = video_io._decode_audio_file(path)
        self.assertEqual(restored["sample_rate"], 48_000)
        self.assertEqual(tuple(restored["waveform"].shape), (1, 2, 4))
        self.assertTrue(torch.allclose(restored["waveform"], audio["waveform"], atol=1e-5, rtol=1e-5))

    def test_raw_h3_decode_flattens_a_video_batch_and_restores_the_vae_finalizer(self):
        class FakeSamples:
            is_nested = True

            @staticmethod
            def unbind():
                return torch.zeros((1, 24, 2, 1, 1)), torch.zeros((1, 32, 2, 1))

        class FakeStage:
            pixel_mean = torch.zeros((1, 3, 1, 1, 1))
            pixel_std = torch.ones((1, 3, 1, 1, 1))

            @staticmethod
            def _finalize_pixels(part):
                return part.clamp(0, 1)

        class FakeVAE:
            def __init__(self):
                self.first_stage_model = FakeStage()

            def decode(self, _video):
                # The test uses the currently installed method, which makes a
                # clamp regression visible without loading the real H3 VAE.
                part = torch.tensor([[[[[-1.0]]], [[[2.0]]], [[[0.5]]]]])
                output = self.first_stage_model._finalize_pixels(part)
                return output.movedim(1, -1)

        vae = FakeVAE()
        original = vae.first_stage_model._finalize_pixels
        decoded = video_io.HREndlessSamplerSaveVideo._raw_h3_decode({"samples": FakeSamples()}, vae)
        self.assertEqual(tuple(decoded.shape), (1, 1, 1, 3))
        self.assertEqual(decoded[0, 0, 0, 0].item(), -1.0)
        self.assertEqual(decoded[0, 0, 0, 1].item(), 2.0)
        self.assertIs(vae.first_stage_model._finalize_pixels, original)
