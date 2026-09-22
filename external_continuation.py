"""Qwen3.5 analysis and MiniMax H3 continuation application nodes.

Analysis is deliberately independent of H3 conditioning and VAEs.  The apply
node receives the already-tokenized MiniMax conditioning and only adds the
continuation keyframe, so H3 references and token metadata cannot be lost.
"""
from __future__ import annotations

import json

import torch
from comfy_api.latest import io

from .director_backend import director_model_options, resolve_director_selection
from .qwen35 import Qwen35ContinuityDirector
from .reference_set import _encode_audio


ExternalH3Context = io.Custom("HR_H3_EXTERNAL_CONTINUATION_CONTEXT")
ExternalH3Continuation = io.Custom("HR_H3_EXTERNAL_CONTINUATION")


def _tail_frames(images: torch.Tensor, requested: int = 22) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("source_images must be an NHWC IMAGE batch")
    if images.shape[0] < 5:
        raise ValueError("MiniMax H3 external continuation requires at least 5 source video frames")
    available = min(int(requested), int(images.shape[0]))
    aligned = 5 + ((available - 5) // 17) * 17
    return images[-aligned:, ..., :3].contiguous()


def _observation_indices(frame_count: int, count: int = 8) -> list[int]:
    count = min(max(1, int(count)), int(frame_count))
    if count == 1:
        return [frame_count - 1]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def _audio_tail(audio, frame_count: int, fps: float):
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("source_audio must be ComfyUI AUDIO data")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("source_audio waveform must have shape [1, channels, samples]")
    sample_rate = int(audio.get("sample_rate", 0))
    if sample_rate <= 0 or waveform.shape[-1] < 1:
        raise ValueError("source_audio must have a positive sample rate and samples")
    samples = max(1, round(float(frame_count) * sample_rate / float(fps)))
    return {"waveform": waveform[..., -samples:].contiguous(), "sample_rate": sample_rate}


def _autogrow_images(values):
    if not values:
        return ()
    if not isinstance(values, dict):
        raise ValueError("reference_images must come from the Autogrow input")
    items = []
    for name, value in sorted(values.items()):
        if name.startswith("reference_image_") and value is not None:
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError("reference images must be NHWC IMAGE batches")
            items.extend(value[index:index + 1] for index in range(value.shape[0]))
    return tuple(items)


class HRMiniMaxH3VideoContinuationAnalyzer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3VideoContinuationAnalyzer",
            display_name="HR MiniMax H3 Continuation Analyzer",
            category="model/sampling/custom",
            description="Use Qwen3.5 to analyze a video's tail and produce an H3 prompt and continuation context.",
            inputs=[
                io.Image.Input("source_images"),
                io.Audio.Input("source_audio", optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Float.Input("source_fps", default=24.0, min=1.0, max=240.0, step=0.001),
                io.Combo.Input("qwen_model", options=director_model_options(), default="auto"),
                io.Combo.Input("qwen_mmproj", options=director_model_options(projector=True), default="auto"),
                io.Autogrow.Input("reference_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("reference_image"), prefix="reference_image_", min=0, max=9)),
                io.Image.Input("reference_image_batch", optional=True),
                io.Combo.Input("audio_mode", options=["continue", "mute"], default="continue"),
            ],
            outputs=[
                io.String.Output(display_name="H3 prompt"),
                ExternalH3Context.Output(display_name="H3 continuation context"),
                io.String.Output(display_name="analysis JSON"),
                io.Image.Output(display_name="tail preview"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, source_images, source_audio=None, prompt="", source_fps=24.0,
                qwen_model="auto", qwen_mmproj="auto", reference_images=None,
                reference_image_batch=None, audio_mode="continue"):
        if not str(prompt or "").strip():
            raise ValueError("A new continuation prompt is required")
        if audio_mode not in {"continue", "mute"}:
            raise ValueError(f"Unknown external continuation audio mode: {audio_mode}")
        selection = resolve_director_selection("qwen3.5", qwen_model, qwen_mmproj)
        if selection.model_path is None or selection.mmproj_path is None:
            raise ValueError("A matching local Qwen3.5 model and mmproj are required")
        tail = _tail_frames(source_images)
        audio_tail = None if source_audio is None else _audio_tail(source_audio, int(tail.shape[0]), float(source_fps))
        refs = list(_autogrow_images(reference_images))
        if reference_image_batch is not None:
            if not isinstance(reference_image_batch, torch.Tensor) or reference_image_batch.ndim != 4:
                raise ValueError("reference_image_batch must be an NHWC IMAGE batch")
            refs.extend(reference_image_batch[index:index + 1] for index in range(reference_image_batch.shape[0]))
        if len(refs) > 9:
            raise ValueError("MiniMax H3 supports at most 9 LLM reference images")
        observation_frames = tail
        if refs:
            refs = [torch.nn.functional.interpolate(image.movedim(-1, 1), size=tail.shape[1:3], mode="bilinear", align_corners=False).movedim(1, -1) for image in refs]
            observation_frames = torch.cat((tail, *refs), dim=0)
        indices = list(range(int(tail.shape[0])))
        director = Qwen35ContinuityDirector(selection.model_path, selection.mmproj_path, mtp_enabled=False, backend="qwen3.5")
        result = director.external_video_continuation(
            str(prompt).strip(), observation_frames,
            source={"fps": float(source_fps), "source_frames": int(source_images.shape[0]),
                    "tail_frames": int(tail.shape[0]), "width": int(tail.shape[2]), "height": int(tail.shape[1]),
                    "observation_indices": indices, "tail_observation_count": len(indices),
                    "reference_image_count": len(refs), "source_audio": source_audio is not None,
                    "audio_mode": audio_mode},
            reference_summary=(f"{len(refs)} optional LLM-only reference images; source tail comes first"),
        )
        analysis = {"confidence": result.confidence, "observed_end_state": result.observed_end_state,
                    "transition_plan": result.transition_plan,
                    "source": {"fps": float(source_fps), "source_frames": int(source_images.shape[0]),
                               "tail_frames": int(tail.shape[0]), "source_audio": source_audio is not None,
                               "audio_mode": audio_mode}}
        context = {"type": "HR_H3_EXTERNAL_CONTINUATION_CONTEXT", "version": 1,
                   "tail_images": tail, "source_audio_tail": audio_tail, "prompt": result.h3_prompt,
                   "analysis": analysis, "source_fps": float(source_fps), "source_frames": int(source_images.shape[0]),
                   "tail_frames": int(tail.shape[0]), "audio_mode": audio_mode}
        return io.NodeOutput(result.h3_prompt, context,
                             json.dumps(analysis, ensure_ascii=False, indent=2), tail)


class HRMiniMaxH3VideoContinuationApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HRMiniMaxH3VideoContinuationApply", display_name="HR MiniMax H3 Continuation Apply",
            category="model/sampling/custom",
            description="Apply an analyzed tail to existing MiniMax H3 conditioning without re-tokenizing it.",
            inputs=[io.Conditioning.Input("positive"), io.Latent.Input("latent"), ExternalH3Context.Input("context"),
                    io.Vae.Input("vae"), io.Vae.Input("audio_vae", optional=True)],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent"),
                     ExternalH3Continuation.Output(display_name="external_continuation")], is_experimental=True)

    @classmethod
    def execute(cls, positive, latent, context, vae, audio_vae=None):
        if not isinstance(context, dict) or context.get("type") != "HR_H3_EXTERNAL_CONTINUATION_CONTEXT":
            raise ValueError("context must come from HR MiniMax H3 Continuation Analyzer")
        tail = context.get("tail_images")
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if not isinstance(tail, torch.Tensor) or samples is None or not samples.is_nested:
            raise ValueError("context and latent must contain MiniMax H3 continuation data")
        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
            raise ValueError("latent must be the MiniMax H3 AV latent from the existing conditioning node")
        target_video = streams[0]
        width, height = int(target_video.shape[-1]) * 16, int(target_video.shape[-2]) * 16
        resized = torch.nn.functional.interpolate(tail.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False).movedim(1, -1)
        keyframe = {"resolved_frame_index": 0, "latent": vae.encode(resized)}
        audio_context = None
        audio_tail = context.get("source_audio_tail")
        if audio_tail is not None and context.get("audio_mode", "continue") == "continue":
            if audio_vae is None:
                raise ValueError("audio_vae is required when the analyzer supplied source audio")
            audio_context, _audio_steps = _encode_audio(audio_vae, audio_tail)
            keyframe["audio_latent"] = audio_context
        updated_positive = []
        for embedding, metadata in positive:
            updated = metadata.copy()
            keyframes = [dict(item) for item in updated.get("minimax_keyframes", ())
                         if not (item.get("resolved_frame_index") == 0 and item.get("latent") is not None)]
            updated["minimax_keyframes"] = [*keyframes, keyframe]
            updated_positive.append([embedding, updated])
        continuation = {"type": "HR_H3_EXTERNAL_CONTINUATION", "version": 1,
                        "video_context": keyframe["latent"], "video_context_start": 0,
                        "audio_context": audio_context, "audio_context_start": 0,
                        "audio_mode": context.get("audio_mode", "continue"), "tail_images": resized,
                        "source_audio_tail": audio_tail, "prompt": context["prompt"],
                        "analysis": context.get("analysis", {}), "target_frames": 5 + max(0, (int(target_video.shape[2]) - 2) // 5) * 17,
                        "source_fps": context.get("source_fps"), "source_frames": context.get("source_frames"),
                        "tail_frames": int(resized.shape[0])}
        return io.NodeOutput(updated_positive, latent, continuation)


# Deliberately no compatibility alias: the old node combined two incompatible responsibilities.
NODE_CLASS_MAPPINGS = {
    "HRMiniMaxH3VideoContinuationAnalyzer": HRMiniMaxH3VideoContinuationAnalyzer,
    "HRMiniMaxH3VideoContinuationApply": HRMiniMaxH3VideoContinuationApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HRMiniMaxH3VideoContinuationAnalyzer": "HR MiniMax H3 Continuation Analyzer",
    "HRMiniMaxH3VideoContinuationApply": "HR MiniMax H3 Continuation Apply",
}
