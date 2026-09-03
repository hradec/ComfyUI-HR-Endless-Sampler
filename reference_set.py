"""Shared MiniMax H3 reference set and Ref2VA conditioning nodes.

The reference encoding follows ComfyUI's MiniMax H3 Ref2VA data layout. The
``ref_scale`` area multiplier is adapted from ComfyUI-JZL-MiniMax-H3, Copyright
(c) 2026 wjluoxiao, under the MIT License.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import io


HRReferenceSet = io.Custom("HR_MINIMAX_H3_REFERENCE_SET")
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def _ordered(values: Any, prefix: str, limit: int) -> tuple[Any, ...]:
    if not values:
        return ()
    if not isinstance(values, dict):
        raise ValueError(f"{prefix} references must come from the HR Autogrow input")

    def index(item):
        match = re.search(r"_(\d+)$", item[0])
        return int(match.group(1)) if match else -1

    result = tuple(value for name, value in sorted(values.items(), key=index) if name.startswith(prefix) and value is not None)
    if len(result) > limit:
        raise ValueError(f"MiniMax H3 supports at most {limit} {prefix} references")
    return result


def _indexed(values: Any, prefix: str, limit: int) -> tuple[Any | None, ...]:
    """Preserve fixed media slots so video N stays paired with video-audio N."""

    result = [None] * limit
    if not values:
        return tuple(result)
    if not isinstance(values, dict):
        raise ValueError(f"{prefix} references must come from the HR Autogrow input")
    for name, value in values.items():
        if not name.startswith(prefix) or value is None:
            continue
        match = re.search(r"_(\d+)$", name)
        if match is None:
            raise ValueError(f"Invalid {prefix} slot name: {name}")
        index = int(match.group(1))
        if index < 0 or index >= limit:
            raise ValueError(f"MiniMax H3 supports {prefix} slots 0 through {limit - 1}")
        result[index] = value
    return tuple(result)


def normalize_reference_set(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or int(value.get("version", -1)) != 1:
        raise ValueError("reference_set must be produced by HR MiniMax H3 Reference Set")
    result = {
        "version": 1,
        "images": tuple(value.get("images", ())),
        "videos": tuple(value.get("videos", ())),
        "video_audios": tuple(value.get("video_audios", ())),
        "audios": tuple(value.get("audios", ())),
        "ref_image_size": str(value.get("ref_image_size", "match")),
        "ref_scale": float(value.get("ref_scale", 1.0)),
    }
    if len(result["images"]) > 9 or len(result["videos"]) > 3 or len(result["video_audios"]) > 3 or len(result["audios"]) > 3:
        raise ValueError("reference_set exceeds MiniMax H3 reference limits")
    if any(audio is not None and (index >= len(result["videos"]) or result["videos"][index] is None)
           for index, audio in enumerate(result["video_audios"])):
        raise ValueError("Each video soundtrack must have a same-index reference video")
    if result["ref_image_size"] not in {"match", "max"}:
        raise ValueError("ref_image_size must be match or max")
    if result["ref_scale"] < 1.0 or result["ref_scale"] > 5.0:
        raise ValueError("ref_scale must be between 1.0 and 5.0")
    return result


def reference_images(value: Any) -> tuple[torch.Tensor, ...]:
    refs = normalize_reference_set(value)
    images = []
    for image in refs["images"]:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("Every reference image must be a non-empty NHWC IMAGE batch")
        images.append(image[:1])
    return tuple(images)


def reference_presentation_items(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Build stock-H3 tokenizer media items without encoding new reference latents."""

    refs = normalize_reference_set(value)
    items = []
    for image in reference_images(refs):
        source_height, source_width = image.shape[1:3]
        if refs["ref_image_size"] == "match":
            scale = min(1.0, math.sqrt(refs["ref_scale"] * width * height / (source_width * source_height)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
        target_width = max(CANVAS_MULTIPLE, round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_height = max(CANVAS_MULTIPLE, round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        items.append({"type": "image", "data": _resize(image, target_width, target_height, "disabled")})
    for index, video in enumerate(refs["videos"]):
        if video is None:
            continue
        if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] < 5:
            raise ValueError("MiniMax H3 reference videos must be NHWC IMAGE batches with at least 5 frames")
        source_height, source_width = video.shape[1:3]
        target_width, target_height = _video_canvas(source_width, source_height)
        if source_width * source_height < target_width * target_height:
            target_width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            target_height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video, target_width, target_height, "disabled")
        if index < len(refs["video_audios"]):
            items.append({"type": "audio"})
        sample_indices = list(range(0, frames.shape[0], FPS // 2))
        items.append({"type": "video", "data": frames[sample_indices], "timestamps": [i / 2.0 for i in range(len(sample_indices))]})
    items.extend({"type": "audio"} for _audio in refs["audios"])
    return items


def align_frame_count(length: int) -> int:
    frames = max(5, int(length))
    remainder = (frames - 5) % 17
    return frames if remainder == 0 else frames + 17 - remainder


def video_latent_steps(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _empty_av_latent(width: int, height: int, length: int):
    frame_count = align_frame_count(length)
    video = torch.zeros(
        [1, 24, video_latent_steps(frame_count), height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )
    audio = torch.zeros(
        [1, 32, 2, round(frame_count * AUDIO_LATENT_FPS / FPS)],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _resize(image, width: int, height: int, crop: str):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _video_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    nominal_width, nominal_height = ((BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE) if ratio >= 1 else (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio))
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _encode_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != vae_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    return latent, latent.shape[-1]


class HRMiniMaxH3ReferenceSet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3ReferenceSet",
            display_name="HR MiniMax H3 Reference Set",
            category="model/sampling/custom",
            description="Pack ordered MiniMax H3 image, video, video-audio, and audio references into one reusable connection.",
            inputs=[
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Float.Input("ref_scale", default=1.0, min=1.0, max=5.0, step=0.1,
                               tooltip="Image area multiplier in match mode. 1.0 preserves stock H3 behavior."),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[HRReferenceSet.Output(display_name="reference_set")],
        )

    @classmethod
    def execute(cls, ref_image_size="match", ref_scale=1.0, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None):
        return io.NodeOutput(normalize_reference_set({
            "version": 1,
            "images": _ordered(ref_images, "ref_image_", 9),
            "videos": _indexed(ref_videos, "ref_video_", 3),
            "video_audios": _indexed(ref_video_audios, "ref_video_audio_", 3),
            "audios": _ordered(ref_audios, "ref_audio_", 3),
            "ref_image_size": ref_image_size,
            "ref_scale": ref_scale,
        }))


class HRMiniMaxH3ReferenceConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3ReferenceConditioning",
            display_name="HR MiniMax H3 Reference Conditioning",
            category="model/sampling/custom",
            description="Encode one HR Reference Set into MiniMax H3 Ref2VA conditioning and a long AV latent.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=36000, step=17),
                HRReferenceSet.Input("reference_set"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                HRReferenceSet.Output(display_name="reference_set"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, reference_set):
        refs = normalize_reference_set(reference_set)
        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items = []
        ref_blocks = []

        for image in reference_images(refs):
            source_height, source_width = image.shape[1:3]
            if refs["ref_image_size"] == "match":
                scale = min(1.0, math.sqrt(refs["ref_scale"] * width * height / (source_width * source_height)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
            target_width = max(CANVAS_MULTIPLE, round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            target_height = max(CANVAS_MULTIPLE, round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(image, target_width, target_height, "disabled")
            encoded = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": target_height // 16, "latent_w": target_width // 16, "latent": encoded})

        for index, video in enumerate(refs["videos"]):
            if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] < 5:
                raise ValueError("MiniMax H3 reference videos must be NHWC IMAGE batches with at least 5 frames")
            source_height, source_width = video.shape[1:3]
            target_width, target_height = _video_canvas(source_width, source_height)
            if source_width * source_height < target_width * target_height:
                target_width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                target_height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video[:frame_count], target_width, target_height, "disabled")
            usable = frames.shape[0]
            while usable >= 5 and usable % 17 != 5:
                usable -= 1
            if usable < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 aligned frames")
            frames = frames[:usable]
            encoded = vae.encode(frames)
            soundtrack = refs["video_audios"][index] if index < len(refs["video_audios"]) else None
            audio_latent, audio_steps = (None, 0) if soundtrack is None else _encode_audio(audio_vae, soundtrack)
            if soundtrack is not None:
                ref_items.append({"type": "audio"})
            sample_indices = list(range(0, frames.shape[0], FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_indices], "timestamps": [i / 2.0 for i in range(len(sample_indices))]})
            ref_blocks.append({
                "kind": "video_audio" if soundtrack is not None else "video",
                "latent_t": encoded.shape[2],
                "latent_h": target_height // 16,
                "latent_w": target_width // 16,
                "ref_audio_t": audio_steps,
                "latent": encoded,
                "audio_latent": audio_latent,
            })

        for audio in refs["audios"]:
            audio_latent, audio_steps = _encode_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": audio_steps, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
        return io.NodeOutput(conditioning, latent, refs)
