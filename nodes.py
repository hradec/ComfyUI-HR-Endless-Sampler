import logging
import math
import re
import uuid

import comfy.model_patcher
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .preview import begin_preview_execution
from .windowed import MiniMaxH3WindowedContextHandler, WINDOW_INDEX_KEY, add_prepare_sampling_wrapper


AUDIO_LATENT_FPS = 40
VIDEO_FPS = 24
CONTEXT_VIDEO_STEPS = 2
CANVAS_MULTIPLE = 32
DESCRIPTION_FIELD = re.compile(r"(?:integrated_multimodal_description|detailed_description)\s*:", re.IGNORECASE)
SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
PICTURE_LABEL = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)
SHOT_OPENING = re.compile(r"^(\s*)(?:the camera|the shot)\s+(?:cuts|transitions|changes|switches)\s+to\s+", re.IGNORECASE)
SHOT_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+|\n\s*\n+")
SHOT_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t))


def _audio_steps(frames):
    return round(frames * AUDIO_LATENT_FPS / VIDEO_FPS)


def _timestamp_frame(minutes, seconds, milliseconds, fps):
    return round((int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0) * fps)


def _frame_timestamp(frame, fps):
    total_milliseconds = round(frame / fps * 1000.0)
    minutes, milliseconds = divmod(total_milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _drop_picture_anchors(prompt):
    field = DESCRIPTION_FIELD.search(prompt)
    if field is None:
        return PICTURE_LABEL.sub("the established subject and scene", prompt)
    prefix = "\n".join(line for line in prompt[:field.start()].splitlines() if "picture" not in line.lower())
    if prefix:
        prefix += "\n"
    return prefix + PICTURE_LABEL.sub("the established subject and scene", prompt[field.start():])


def _shot_body_for_range(body, shot_start, shot_end, frame_start, frame_end):
    if frame_start <= shot_start and frame_end >= shot_end:
        return body

    units = []
    for sentence in SHOT_BREAK.split(body):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = SHOT_CLAUSE_BREAK.split(sentence) if len(re.findall(r"\w+", sentence)) > 32 else [sentence]
        for clause in clauses:
            words = clause.split()
            units.extend(" ".join(words[index:index + 24]) for index in range(0, len(words), 24))
    weights = [max(1, len(re.findall(r"\w+", unit))) for unit in units]
    total_weight = sum(weights)
    start_weight = total_weight * max(frame_start, shot_start) - total_weight * shot_start
    end_weight = total_weight * min(frame_end, shot_end) - total_weight * shot_start
    duration = shot_end - shot_start
    start_weight /= duration
    end_weight /= duration

    selected = []
    offset = 0
    for unit, weight in zip(units, weights):
        if start_weight <= offset < end_weight:
            selected.append(unit)
        offset += weight
    return " " + " ".join(selected) if selected else " Continue the established shot and its ongoing action."


def _prompt_for_window(prompt, frame_start, frame_end, total_frames, fps, content_start=None, continuation=False, drop_picture_anchors=False):
    content_start = frame_start if content_start is None else content_start
    if drop_picture_anchors:
        prompt = _drop_picture_anchors(prompt)
    field = DESCRIPTION_FIELD.search(prompt)
    description_start = field.end() if field is not None else 0
    description_end_match = DESCRIPTION_END.search(prompt, description_start)
    description_end = description_end_match.start() if description_end_match is not None else len(prompt)
    markers = list(SHOT_MARKER.finditer(prompt, description_start, description_end))
    if not markers:
        return prompt

    shot_starts = []
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != index + 1:
            raise ValueError("MiniMax shot numbers must start at 1 and increase sequentially")
        if marker.group(2) is None:
            if index:
                raise ValueError("MiniMax shot markers after the opening shot must use 'At MM:SS.mmm,'")
            shot_starts.append(0)
        else:
            if not index:
                raise ValueError("MiniMax [Shot 1] must not have a timestamp")
            shot_starts.append(_timestamp_frame(marker.group(2), marker.group(3), marker.group(4), fps))
    if any(right <= left for left, right in zip(shot_starts, shot_starts[1:])):
        raise ValueError("MiniMax shot timestamps must be strictly increasing")

    selected = []
    for index, marker in enumerate(markers):
        shot_end = shot_starts[index + 1] if index + 1 < len(markers) else total_frames
        if shot_starts[index] < frame_end and shot_end > content_start:
            segment_end = markers[index + 1].start() if index + 1 < len(markers) else description_end
            body = _shot_body_for_range(prompt[marker.end():segment_end], shot_starts[index], shot_end, content_start, frame_end)
            if body is not None:
                selected.append((marker, body, shot_starts[index]))
    if not selected:
        raise ValueError(f"No prompt shots overlap frames {content_start} through {frame_end - 1}")

    rewritten = []
    for index, (marker, body, shot_start) in enumerate(selected):
        marker_text = f"[Shot {index + 1}]"
        if index:
            marker_text += f" At {_frame_timestamp(shot_start, fps)},"
        elif continuation and shot_start < content_start:
            body = SHOT_OPENING.sub(r"\1the continuing shot shows ", body, count=1)
            marker_text += " Continue the already established shot at this point; do not restart or replay earlier actions."
        rewritten.append(marker_text + body.rstrip() + " ")
    return prompt[:markers[0].start()] + "".join(rewritten) + prompt[description_end:]


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _reference_image(image, width, height):
    source_height, source_width = image.shape[1:3]
    scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    target_width = max(CANVAS_MULTIPLE, round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(image, target_width, target_height, "disabled")


def _prompt_tokens(clip, prompt, images, positive, width, height, continuation):
    refs = positive[0].get("minimax_refs") if positive else None
    image_list = [] if images is None else [images[index:index + 1] for index in range(images.shape[0])]
    if refs:
        ref_items = []
        image_index = 0
        for ref in refs:
            kind = ref["kind"]
            if kind == "image":
                if image_index >= len(image_list):
                    raise ValueError("SamplerCustomAdvanced-Unlimited needs every Ref2VA reference image in the images input")
                ref_items.append({"type": "image", "data": _reference_image(image_list[image_index], width, height)})
                image_index += 1
            elif kind == "audio":
                ref_items.append({"type": "audio"})
            elif kind in ("video", "video_audio"):
                raise ValueError("SamplerCustomAdvanced-Unlimited cannot rebuild video Ref2VA conditioning from an images input")
        if image_index != len(image_list):
            raise ValueError("SamplerCustomAdvanced-Unlimited received more images than the Ref2VA conditioning uses")
        return clip.tokenize(prompt, minimax_ref_items=ref_items)

    prompt_images = []
    for index, image in enumerate(() if continuation else image_list):
        prompt_images.append(_resize(image, width, height, "disabled" if index == 0 else "center"))
    return clip.tokenize(prompt, images=prompt_images)


def _encode_prompt(clip, prompt, images, positive, width, height, continuation):
    conditioning = clip.encode_from_tokens_scheduled(_prompt_tokens(clip, prompt, images, positive, width, height, continuation))
    if len(conditioning) != 1:
        raise ValueError("SamplerCustomAdvanced-Unlimited expects one MiniMax H3 conditioning segment")
    return conditioning[0]


def _chunk_plan(video_t, audio_t, chunk_frames):
    max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
    max_chunk_t = ((max_chunk_frames - 5) // 17) * 5 + CONTEXT_VIDEO_STEPS

    if video_t < CONTEXT_VIDEO_STEPS or (video_t - CONTEXT_VIDEO_STEPS) % 5:
        raise ValueError("SamplerCustomAdvanced-Unlimited expects a MiniMax H3 video latent on the 17k+5 frame grid")

    total_frames = _pixel_frames(video_t)
    if audio_t != _audio_steps(total_frames):
        raise ValueError("SamplerCustomAdvanced-Unlimited expects a MiniMax H3 audio latent matching the video duration")

    plan = []
    video_end = 0
    audio_end = 0
    output_frames = 0
    remaining = video_t
    while remaining:
        if not plan:
            chunk_t = min(max_chunk_t, remaining)
            video_start = 0
            new_video_t = chunk_t
            chunk_frame_count = _pixel_frames(chunk_t)
        else:
            new_video_t = min(max_chunk_t - CONTEXT_VIDEO_STEPS, remaining)
            chunk_t = new_video_t + CONTEXT_VIDEO_STEPS
            video_start = video_end - CONTEXT_VIDEO_STEPS
            chunk_frame_count = _pixel_frames(chunk_t)

        output_frames += chunk_frame_count if not plan else chunk_frame_count - 5
        next_audio_end = _audio_steps(output_frames)
        chunk_audio_t = _audio_steps(chunk_frame_count)
        new_audio_t = next_audio_end - audio_end
        context_audio_t = 0 if not plan else chunk_audio_t - new_audio_t
        audio_start = 0 if not plan else audio_end - context_audio_t

        plan.append({
            "video_start": video_start,
            "video_end": video_start + chunk_t,
            "audio_start": audio_start,
            "audio_end": next_audio_end,
            "context_audio_t": context_audio_t,
            "frame_start": 0 if not plan else output_frames - chunk_frame_count,
            "frame_end": output_frames,
        })
        video_end += new_video_t
        audio_end = next_audio_end
        remaining -= new_video_t

    return plan


def _conditioning_for_window(original_conds, encoded_prompt, window_index, video_shape, audio_shape):
    conds = {name: [item.copy() for item in values] for name, values in original_conds.items()}
    positive = conds.get("positive")
    if positive is None:
        raise ValueError("SamplerCustomAdvanced-Unlimited requires a standard guider with positive conditioning")

    cross_attn, prompt_metadata = encoded_prompt
    for cond in positive:
        cond["cross_attn"] = cross_attn
        token_tags = prompt_metadata.get("minimax_token_tags")
        if token_tags is not None:
            cond["minimax_token_tags"] = token_tags
        else:
            cond.pop("minimax_token_tags", None)
        cond["latent_shapes"] = [video_shape, audio_shape]
        cond[WINDOW_INDEX_KEY] = window_index
        cond["uuid"] = uuid.uuid4()
    return conds


class MiniMaxH3SamplerCustomAdvancedUnlimited(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerCustomAdvanced-Unlimited",
            display_name="SamplerCustomAdvanced-Unlimited",
            category="model/sampling/custom",
            description="Samples one long MiniMax H3 AV latent trajectory through overlapping temporal model windows. Replace SamplerCustomAdvanced and set the largest window that fits in VRAM.",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Clip.Input("clip", tooltip="The same MiniMax H3 CLIP used to encode the original conditioning."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="MiniMax prompt using [Shot 1] and [Shot N] At MM:SS.mmm, markers."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001,
                               tooltip="FPS used to convert absolute prompt timestamps to global frame positions."),
                io.Int.Input("chunk_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Maximum H3 frames evaluated by the model at once. Values are snapped down to the 17k+5 frame grid."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log every window prompt to the console and return them through chunk_prompts."),
                io.Image.Input("images", optional=True,
                               tooltip="Original H3 conditioning images as a batch: first frame, then optional last frame; or all image-only Ref2VA references in order."),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
                io.String.Output(display_name="chunk_prompts"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0, chunk_frames=124, debug=False, images=None):
        samples = latent_image["samples"]
        if not samples.is_nested:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24 or streams[1].ndim != 4 or streams[1].shape[1] != 32:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        video, audio = streams
        plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames)
        if len(plan) > 1 and "noise_mask" in latent_image:
            raise ValueError("SamplerCustomAdvanced-Unlimited does not support denoise masks with multiple windows")

        original_conds = guider.original_conds
        original_model_options = guider.model_options
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError("SamplerCustomAdvanced-Unlimited requires a standard guider with positive conditioning")
        ref2va = bool(positive[0].get("minimax_refs"))
        total_frames = plan[-1]["frame_end"]
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        window_conds = {name: [item.copy() for item in values] for name, values in original_conds.items()}
        window_conds["positive"] = []
        debug_prompts = []
        for index, chunk in enumerate(plan):
            continuation = index > 0
            window_prompt = _prompt_for_window(
                prompt,
                chunk["frame_start"],
                chunk["frame_end"],
                total_frames,
                fps,
                continuation=continuation,
                drop_picture_anchors=continuation and not ref2va,
            )
            if debug:
                debug_prompt = (
                    f"=== Window {index + 1}: frames {chunk['frame_start']}-{chunk['frame_end'] - 1} ===\n"
                    f"{window_prompt}"
                )
                debug_prompts.append(debug_prompt)
                logging.info("SamplerCustomAdvanced-Unlimited debug:\n%s", debug_prompt)
            encoded_prompt = _encode_prompt(clip, window_prompt, images, positive, width, height, continuation)
            window_shapes = [video[:, :, chunk["video_start"]:chunk["video_end"]].shape,
                             audio[..., chunk["audio_start"]:chunk["audio_end"]].shape]
            encoded_conds = _conditioning_for_window(
                original_conds, encoded_prompt, index, window_shapes[0], window_shapes[1]
            )
            window_conds["positive"].extend(encoded_conds["positive"])

        preview_execution = begin_preview_execution(guider.model_patcher, len(plan))
        context_handler = None
        try:
            if len(plan) > 1:
                guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
                context_handler = MiniMaxH3WindowedContextHandler(
                    plan,
                    [video.shape, audio.shape],
                    preview_execution.set_window if preview_execution is not None else None,
                )
                guider.model_options["context_handler"] = context_handler
                add_prepare_sampling_wrapper(guider.model_options)
            guider.original_conds = window_conds
            if preview_execution is not None:
                preview_execution.set_chunk(0, 0, total_frames - 1, 0, total_frames - 1, 0)
                if len(plan) == 1:
                    preview_execution.set_window(0)
            sampled, denoised = super().execute(noise, guider, sampler, sigmas, latent_image)
        finally:
            guider.original_conds = original_conds
            guider.model_options = original_model_options
            if preview_execution is not None:
                preview_execution.clear_chunk()
                preview_execution.close()
            if context_handler is not None:
                context_handler.close()

        return io.NodeOutput(sampled, denoised, "\n\n".join(debug_prompts))

    sample = execute
