import logging
import math
import re

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from tqdm.auto import tqdm

from .preview import begin_preview_execution


AUDIO_LATENT_FPS = 40
VIDEO_FPS = 24
MIN_VIDEO_STEPS = 2
CANVAS_MULTIPLE = 32
QWEN_VIDEO_MAX_PIXELS = 512 * 512
VRAM_DEBUG_WRAPPER_KEY = "minimax_h3_unlimited_vram_debug"
DETAILED_DESCRIPTION_FIELD = re.compile(r"detailed_description\s*:", re.IGNORECASE)
INTEGRATED_DESCRIPTION_FIELD = re.compile(r"integrated_multimodal_description\s*:", re.IGNORECASE)
SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
SUBJECT_DEFINITIONS_FIELD = re.compile(r"(?im)^\s*subject_definitions\s*:\s*$")
SUMMARY_FIELD = re.compile(r"(?im)^(\s*summary\s*:\s*)(.*)$")
RETENTION_FIELD = re.compile(r"(?im)^\s*retention_analysis\s*:\s*$")
PICTURE_LABEL = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)
SHOT_OPENING = re.compile(r"^(\s*)(?:(?:the\s+)?camera|the\s+shot)\s+(?:cuts?|transitions?|changes?|switches?)\s+to\s+", re.IGNORECASE)
SHOT_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+|\n\s*\n+")
SHOT_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def _description_field(prompt, start=0):
    return DETAILED_DESCRIPTION_FIELD.search(prompt, start) or INTEGRATED_DESCRIPTION_FIELD.search(prompt, start)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t))


def _video_steps(frames):
    return ((frames - 5) // 17) * 5 + MIN_VIDEO_STEPS


def _audio_steps(frames):
    return round(frames * AUDIO_LATENT_FPS / VIDEO_FPS)


def _context_video_steps(context_frames, max_chunk_frames):
    if context_frames < 5 or (context_frames - 5) % 17:
        raise ValueError("context_frames must use MiniMax H3's 17k+5 frame grid: 5, 22, 39, 56, ...")
    if context_frames >= max_chunk_frames:
        raise ValueError(f"context_frames ({context_frames}) must be smaller than the effective chunk size ({max_chunk_frames})")
    return _video_steps(context_frames)


def _timestamp_frame(minutes, seconds, milliseconds, fps):
    return round((int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0) * fps)


def _frame_timestamp(frame, fps):
    total_milliseconds = round(frame / fps * 1000.0)
    minutes, milliseconds = divmod(total_milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _drop_picture_anchors(prompt):
    field = _description_field(prompt)
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
        if offset < end_weight and offset + weight > start_weight:
            selected.append(unit)
        offset += weight
    return " " + " ".join(selected) if selected else " Continue the established shot and its ongoing action."


def _video_continuation_prompt(prompt, video_label):
    source_line = f"{video_label} is the completed ending of the video immediately before this chunk."
    subject = SUBJECT_DEFINITIONS_FIELD.search(prompt)
    if subject is not None:
        next_section = RETENTION_FIELD.search(prompt, subject.end()) or _description_field(prompt, subject.end())
        insert_at = next_section.start() if next_section is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + "\n" + source_line + "\n\n" + prompt[insert_at:].lstrip()
    else:
        field = _description_field(prompt)
        insert_at = field.start() if field is not None else 0
        prompt = prompt[:insert_at] + f"subject_definitions:\n{source_line}\n\n" + prompt[insert_at:]

    summary = SUMMARY_FIELD.search(prompt)
    summary_text = f"[video continuation] Continue the target video directly from the end of {video_label}."
    if summary is not None:
        existing = summary.group(2).strip()
        task = re.match(r"\[([^]]+)\]\s*(.*)", existing)
        if task is not None:
            types = [value.strip() for value in task.group(1).split("+")]
            if "video continuation" not in [value.lower() for value in types]:
                types.insert(0, "video continuation")
            existing = f"[{' + '.join(types)}] {task.group(2).strip()}".rstrip()
            replacement = summary.group(1) + existing + f" Continue the target video directly from the end of {video_label}."
        else:
            replacement = summary.group(1) + summary_text + (" " + existing if existing else "")
        prompt = prompt[:summary.start()] + replacement + prompt[summary.end():]
    else:
        retention = RETENTION_FIELD.search(prompt)
        field = _description_field(prompt)
        insert_at = retention.start() if retention is not None else field.start() if field is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + f"\n\nsummary: {summary_text}\n\n" + prompt[insert_at:].lstrip()

    retention_line = f"{video_label} (appears in [Shot 1]): fully_preserved - its ending is used as the continuation starting point for this chunk."
    retention = RETENTION_FIELD.search(prompt)
    if retention is not None:
        field = _description_field(prompt, retention.end())
        insert_at = field.start() if field is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + "\n" + retention_line + "\n\n" + prompt[insert_at:].lstrip()
    else:
        field = _description_field(prompt)
        insert_at = field.start() if field is not None else len(prompt)
        prompt = prompt[:insert_at].rstrip() + f"\n\nretention_analysis:\n{retention_line}\n\n" + prompt[insert_at:].lstrip()
    return prompt


def _prompt_for_chunk(prompt, frame_start, frame_end, total_frames, fps, content_start=None, continuation=False,
                      drop_picture_anchors=False, continuation_video_label=None, has_opening_frames=True):
    content_start = frame_start if content_start is None else content_start
    if drop_picture_anchors:
        prompt = _drop_picture_anchors(prompt)
    field = _description_field(prompt)
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
            marker_text += f" At {_frame_timestamp(shot_start - frame_start, fps)},"
        elif continuation and shot_start < content_start:
            body = SHOT_OPENING.sub(r"\1the continuing shot shows ", body, count=1)
            if continuation_video_label is not None:
                marker_text += f" Continue seamlessly from the end of {continuation_video_label} at this point in the shot; do not restart or replay earlier actions."
            elif has_opening_frames:
                marker_text += " Continue seamlessly from the provided opening frames at this point in the shot; do not restart or replay earlier actions."
            else:
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


def _prompt_tokens(clip, prompt, images, positive, width, height, continuation, video_items=()):
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
        ref_items.extend(video_items)
        return clip.tokenize(prompt, minimax_ref_items=ref_items)

    if video_items:
        raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")

    prompt_images = []
    for index, image in enumerate(() if continuation else image_list):
        prompt_images.append(_resize(image, width, height, "disabled" if index == 0 else "center"))
    return clip.tokenize(prompt, images=prompt_images)


def _encode_prompt(clip, prompt, images, positive, width, height, continuation, video_items=()):
    conditioning = clip.encode_from_tokens_scheduled(_prompt_tokens(clip, prompt, images, positive, width, height, continuation, video_items))
    if len(conditioning) != 1:
        raise ValueError("SamplerCustomAdvanced-Unlimited expects one MiniMax H3 conditioning segment")
    return conditioning[0]


def _chunk_plan(video_t, audio_t, chunk_frames, context_frames=5):
    max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
    max_chunk_t = _video_steps(max_chunk_frames)
    context_video_t = _context_video_steps(context_frames, max_chunk_frames)

    if video_t < MIN_VIDEO_STEPS or (video_t - MIN_VIDEO_STEPS) % 5:
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
            new_video_t = min(max_chunk_t - context_video_t, remaining)
            chunk_t = new_video_t + context_video_t
            video_start = video_end - context_video_t
            chunk_frame_count = _pixel_frames(chunk_t)

        output_frames += chunk_frame_count if not plan else chunk_frame_count - context_frames
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
            "context_video_t": 0 if not plan else context_video_t,
            "context_audio_t": context_audio_t,
            "frame_start": 0 if not plan else output_frames - chunk_frame_count,
            "frame_end": output_frames,
        })
        video_end += new_video_t
        audio_end = next_audio_end
        remaining -= new_video_t

    return plan


def _chunk_plan_without_overlap(video_t, audio_t, chunk_frames):
    plan = _chunk_plan(video_t, audio_t, chunk_frames, 5)
    for index in range(1, len(plan)):
        chunk = plan[index].copy()
        chunk["video_start"] += chunk["context_video_t"]
        chunk["audio_start"] += chunk["context_audio_t"]
        chunk["synthetic_prefix"] = True
        plan[index] = chunk
    return plan


def _conditioning_for_chunk(original_conds, frame_start, frame_end, encoded_prompt, video_context=None,
                            audio_context=None, audio_end_frame=5.0, video_refs=()):
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
        if video_refs:
            cond["minimax_refs"] = [*cond.get("minimax_refs", ()), *video_refs]
        keyframes = []
        for keyframe in cond.get("minimax_keyframes", ()):
            position = keyframe["resolved_frame_index"]
            if frame_start <= position < frame_end:
                local_keyframe = keyframe.copy()
                local_keyframe["resolved_frame_index"] = position - frame_start
                keyframes.append(local_keyframe)

        if video_context is not None:
            keyframes.append({"resolved_frame_index": 0, "latent": video_context})
        if audio_context is not None:
            audio_start = audio_end_frame - audio_context.shape[-1] / FRAME_RESCALE
            keyframes.append({"resolved_frame_index": audio_start, "audio_latent": audio_context})
        if keyframes:
            cond["minimax_keyframes"] = keyframes
        else:
            cond.pop("minimax_keyframes", None)
    return conds


def _decoded_video_item(vae, latent):
    frames = vae.decode(latent)
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    sample_indices = list(range(0, frames.shape[0], VIDEO_FPS // 2))
    qwen_frames = frames[sample_indices]
    height, width = qwen_frames.shape[1:3]
    if height * width > QWEN_VIDEO_MAX_PIXELS:
        scale = math.sqrt(QWEN_VIDEO_MAX_PIXELS / (height * width))
        target_width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        qwen_frames = _resize(qwen_frames, target_width, target_height, "disabled")
    return {
        "type": "video",
        "data": qwen_frames,
        "timestamps": [index / 2.0 for index in range(len(sample_indices))],
    }


def _video_ref_block(latent):
    return {
        "kind": "video",
        "latent_t": latent.shape[2],
        "latent_h": latent.shape[3],
        "latent_w": latent.shape[4],
        "ref_audio_t": 0,
        "latent": latent,
    }


def _tensor_bytes(value, device):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.device == device else 0
    if getattr(value, "is_nested", False):
        return sum(_tensor_bytes(item, device) for item in value.unbind())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, device) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, device) for item in value)
    return 0


def _memory_backend(device):
    if device.type == "cuda":
        return torch.cuda
    if device.type in ("xpu", "npu", "mlu"):
        return getattr(torch, device.type)
    return None


def _vram_report(stage, device, components=(), tensors=None):
    mib = 1024 ** 2
    total = comfy.model_management.get_total_memory(device)
    comfy_free, torch_cache_free = comfy.model_management.get_free_memory(device, torch_free_too=True)
    lines = [f"SamplerCustomAdvanced-Unlimited VRAM [{stage}] on {device}:"]
    backend = _memory_backend(device)
    if backend is not None:
        stats = backend.memory_stats(device)
        if device.type == "cuda":
            physical_free, physical_total = backend.mem_get_info(device)
        else:
            physical_total = total
            physical_free = comfy_free - torch_cache_free
        active = stats.get("active_bytes.all.current", 0)
        allocated = stats.get("allocated_bytes.all.current", active)
        reserved = stats.get("reserved_bytes.all.current", 0)
        peak_active = stats.get("active_bytes.all.peak", 0)
        peak_reserved = stats.get("reserved_bytes.all.peak", 0)
        lines.append(
            f"  device: {physical_total / mib:.1f} MiB total, {(physical_total - physical_free) / mib:.1f} MiB used by all processes, "
            f"{physical_free / mib:.1f} MiB physically free"
        )
        lines.append(
            f"  torch: {allocated / mib:.1f} MiB allocated, {active / mib:.1f} MiB active, {reserved / mib:.1f} MiB reserved, "
            f"{max(0, reserved - active) / mib:.1f} MiB cached/inactive"
        )
        lines.append(f"  peak: {peak_active / mib:.1f} MiB active, {peak_reserved / mib:.1f} MiB reserved")
    else:
        lines.append(f"  device: {total / mib:.1f} MiB total")
    lines.append(f"  ComfyUI usable free: {comfy_free / mib:.1f} MiB ({torch_cache_free / mib:.1f} MiB in the torch cache)")

    component_parts = []
    for name, patcher in components:
        if patcher is not None:
            component_parts.append(
                f"{name}={patcher.loaded_size() / mib:.1f} MiB loaded "
                f"({'dynamic' if patcher.is_dynamic() else 'standard'}, {patcher.load_device}, {len(patcher.patches)} patch keys)"
            )
    if component_parts:
        lines.append("  known models: " + "; ".join(component_parts))

    resident_parts = []
    for patcher in comfy.model_management.loaded_models():
        resident_parts.append(
            f"{patcher.model.__class__.__name__}={patcher.loaded_size() / mib:.1f} MiB/{len(patcher.patches)} patches"
        )
    lines.append("  ComfyUI model registry: " + ("; ".join(resident_parts) if resident_parts else "empty"))

    if tensors:
        tensor_parts = []
        for name, value in tensors.items():
            size = _tensor_bytes(value, device)
            if size:
                tensor_parts.append(f"{name}={size / mib:.1f} MiB")
        lines.append("  visible GPU tensor payloads: " + ("; ".join(tensor_parts) if tensor_parts else "none"))
    logging.info("\n".join(lines))


class _VRAMDebugMonitor:
    def __init__(self, device, components, chunk_count):
        self.device = device
        self.components = components
        self.chunk_count = chunk_count
        self.chunk = 0
        self.call = 0

    def set_chunk(self, index):
        self.chunk = index
        self.call = 0
        backend = _memory_backend(self.device)
        if self.device.type == "cuda" and backend is not None:
            backend.reset_peak_memory_stats(self.device)

    def report(self, stage, tensors=None):
        _vram_report(stage, self.device, self.components, tensors)

    def __call__(self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        self.call += 1
        label = f"chunk {self.chunk + 1}/{self.chunk_count} DiT evaluation {self.call}"
        tensors = {"model input": x, "cross attention": c_crossattn, "model conditions": kwargs}
        self.report(label + " before", tensors)
        try:
            result = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        except Exception:
            self.report(label + " FAILED", tensors)
            if self.device.type == "cuda":
                logging.info("SamplerCustomAdvanced-Unlimited CUDA allocator after failure:\n%s", torch.cuda.memory_summary(self.device, abbreviated=True))
            raise
        self.report(label + " after", {"model output": result})
        return result


class _FixedNoise:
    def __init__(self, seed, samples):
        self.seed = seed
        self.samples = samples

    def generate_noise(self, _latent):
        return self.samples


class _ChunkProgress:
    def __init__(self, count):
        self.count = count
        self.bar = None

    def start(self, index):
        if self.bar is None:
            self.bar = tqdm(
                total=self.count,
                desc=f"Chunk {index + 1}/{self.count}",
                unit="chunk",
                leave=False,
                position=0,
                dynamic_ncols=True,
                disable=not comfy.utils.PROGRESS_BAR_ENABLED,
            )
        self.bar.n = index
        self.bar.set_description_str(f"Chunk {index + 1}/{self.count}")
        self.bar.refresh()

    def finish(self, index):
        self.bar.n = index + 1
        self.bar.refresh()

    def close(self):
        if self.bar is not None:
            self.bar.close()


class MiniMaxH3SamplerCustomAdvancedUnlimited(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerCustomAdvanced-Unlimited",
            display_name="SamplerCustomAdvanced-Unlimited",
            category="model/sampling/custom",
            description="Samples a long MiniMax H3 AV latent as continuation-guided temporal chunks. Replace SamplerCustomAdvanced and set the largest chunk that fits in VRAM.",
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
                               tooltip="FPS used to convert absolute prompt timestamps to chunk-local frame positions."),
                io.Int.Input("chunk_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Maximum H3 frames sampled at once. Values are snapped down to the 17k+5 frame grid."),
                io.Int.Input("context_frames", default=5, min=5, max=3600, step=17,
                             tooltip="Bounded completed-frame context for guide_overlap and video_continuation. Valid values are 5, 22, 39, 56, ... and must be smaller than chunk_frames."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Log every chunk prompt and detailed VRAM snapshots to the console, and return the prompts through chunk_prompts."),
                io.Int.Input("debug_stop_chunk", default=0, min=0, max=10000, step=1,
                             tooltip="Stop after this 1-based chunk number and return the partial result. 0 samples every chunk."),
                io.Image.Input("images", optional=True,
                               tooltip="Original H3 conditioning images as a batch: first frame, then optional last frame; or all image-only Ref2VA references in order."),
                io.Combo.Input("guide_overlap", options=["context_frames", "5 frames", "off"], default="context_frames",
                               tooltip="Guide + overlap strength. context_frames uses the configured tail; 5 frames uses H3's minimum tail; off carries no previous frames and is intended for native video_continuation tests."),
                io.Boolean.Input("video_continuation", default=False,
                                 tooltip="Experimental: expose the bounded previous context as a native Ref2VA <Video N> and add [video continuation] to later chunk prompts. Requires vae."),
                io.Boolean.Input("qwen_full_history", default=False,
                                 tooltip="Experimental: show Qwen 2 FPS frames decoded from all completed output before each chunk. Does not add a DiT video reference or rewrite the prompt. Requires vae."),
                io.Vae.Input("vae", optional=True,
                             tooltip="MiniMax H3 video VAE. Required only by video_continuation or qwen_full_history."),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
                io.String.Output(display_name="chunk_prompts"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0, chunk_frames=124, debug=False,
                debug_stop_chunk=0, images=None, context_frames=5, guide_overlap="context_frames", video_continuation=False,
                qwen_full_history=False, vae=None):
        samples = latent_image["samples"]
        if not samples.is_nested:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24 or streams[1].ndim != 4 or streams[1].shape[1] != 32:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        video, audio = streams
        if guide_overlap is True:
            guide_overlap = "context_frames"
        elif guide_overlap is False:
            guide_overlap = "5 frames"
        if guide_overlap not in ("context_frames", "5 frames", "off"):
            raise ValueError("guide_overlap must be context_frames, 5 frames, or off")
        overlap_frames = context_frames if guide_overlap == "context_frames" else 5
        max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
        if video_continuation and guide_overlap == "off":
            _context_video_steps(context_frames, max_chunk_frames)
        if guide_overlap == "off":
            plan = _chunk_plan_without_overlap(video.shape[2], audio.shape[-1], chunk_frames)
        else:
            plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames, overlap_frames)
        if len(plan) > 1 and "noise_mask" in latent_image:
            raise ValueError("SamplerCustomAdvanced-Unlimited does not support denoise masks when chunking")
        if debug_stop_chunk > len(plan):
            raise ValueError(f"debug_stop_chunk is {debug_stop_chunk}, but this latent has only {len(plan)} chunks")
        active_plan = plan if debug_stop_chunk == 0 else plan[:debug_stop_chunk]

        fixed_latent = latent_image.copy()
        fixed_latent["samples"] = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            samples,
            latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"),
        )
        full_noise = noise.generate_noise(fixed_latent)
        if not full_noise.is_nested or len(full_noise.unbind()) != 2:
            raise ValueError("SamplerCustomAdvanced-Unlimited expected nested video and audio noise")
        video_noise, audio_noise = full_noise.unbind()

        original_conds = guider.original_conds
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError("SamplerCustomAdvanced-Unlimited requires a standard guider with positive conditioning")
        ref2va = bool(positive[0].get("minimax_refs"))
        if len(plan) > 1 and (video_continuation or qwen_full_history):
            if vae is None:
                raise ValueError("video_continuation and qwen_full_history require a MiniMax H3 video VAE")
            if not ref2va:
                raise ValueError("Experimental video conditioning requires positive conditioning from MiniMax H3 Reference to Video")
        original_refs = positive[0].get("minimax_refs", ())
        video_number = 1 + sum(ref["kind"] in ("video", "video_audio") for ref in original_refs)
        total_frames = plan[-1]["frame_end"]
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        output_video = []
        output_audio = []
        denoised_video = []
        denoised_audio = []
        previous_video = None
        previous_audio = None
        previous_frame_count = None
        output_template = None
        denoised_template = None
        debug_prompts = []
        preview_execution = begin_preview_execution(guider.model_patcher, len(active_plan))
        chunk_progress = _ChunkProgress(len(active_plan))
        vram_monitor = None
        if debug:
            components = [
                ("MiniMax H3 DiT", guider.model_patcher),
                ("Qwen/CLIP", clip.patcher),
                ("H3 video VAE", vae.patcher if vae is not None else None),
            ]
            vram_monitor = _VRAMDebugMonitor(guider.model_patcher.load_device, components, len(active_plan))
            guider.model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, VRAM_DEBUG_WRAPPER_KEY)
            guider.model_patcher.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                VRAM_DEBUG_WRAPPER_KEY,
                vram_monitor,
            )
            vram_monitor.report("execution prepared", {"full latent": samples, "full noise": full_noise})

        try:
            for index, chunk in enumerate(active_plan):
                if vram_monitor is not None:
                    vram_monitor.set_chunk(index)
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} start",
                        {
                            "full latent": samples,
                            "full noise": full_noise,
                            "completed output": (output_video, output_audio),
                            "completed denoised output": (denoised_video, denoised_audio),
                            "previous chunk": (previous_video, previous_audio),
                        },
                    )
                vs, ve = chunk["video_start"], chunk["video_end"]
                aus, aue = chunk["audio_start"], chunk["audio_end"]
                context_video_t = chunk["context_video_t"]
                context_audio_t = chunk["context_audio_t"]

                chunk_video = video[:, :, vs:ve]
                chunk_audio = audio[..., aus:aue]
                chunk_video_noise = video_noise[:, :, vs:ve]
                chunk_audio_noise = audio_noise[..., aus:aue]
                if chunk.get("synthetic_prefix"):
                    prefix_video = video.new_zeros((*video.shape[:2], context_video_t, *video.shape[3:]))
                    prefix_audio = audio.new_zeros((*audio.shape[:-1], context_audio_t))
                    prefix_latent = fixed_latent.copy()
                    prefix_latent["samples"] = comfy.nested_tensor.NestedTensor((prefix_video, prefix_audio))
                    prefix_noise = noise.generate_noise(prefix_latent)
                    if not prefix_noise.is_nested or len(prefix_noise.unbind()) != 2:
                        raise ValueError("SamplerCustomAdvanced-Unlimited expected nested video and audio prefix noise")
                    prefix_video_noise, prefix_audio_noise = prefix_noise.unbind()
                    chunk_video = torch.cat((prefix_video, chunk_video), dim=2)
                    chunk_audio = torch.cat((prefix_audio, chunk_audio), dim=-1)
                    chunk_video_noise = torch.cat((prefix_video_noise, chunk_video_noise), dim=2)
                    chunk_audio_noise = torch.cat((prefix_audio_noise, chunk_audio_noise), dim=-1)

                chunk_latent = fixed_latent.copy()
                chunk_latent["samples"] = comfy.nested_tensor.NestedTensor((chunk_video, chunk_audio))
                chunk_noise = comfy.nested_tensor.NestedTensor((chunk_video_noise, chunk_audio_noise))

                guide_enabled = guide_overlap != "off"
                video_context = None if previous_video is None or not guide_enabled else previous_video[:, :, -context_video_t:].clone()
                audio_context = None if previous_audio is None or not guide_enabled else previous_audio[..., -context_audio_t:].clone()
                audio_end_frame = float(overlap_frames)
                if audio_context is not None:
                    overhang = previous_audio.shape[-1] - FRAME_RESCALE * previous_frame_count
                    audio_end_frame += overhang / FRAME_RESCALE
                continuation = index > 0
                content_start = chunk["frame_start"] + (overlap_frames if continuation else 0)
                video_items = []
                video_refs = []
                continuation_video_label = None
                chunk_prompt_source = prompt
                if continuation and video_continuation:
                    reference_latent = previous_video[:, :, -_video_steps(context_frames):].clone()
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} before continuation VAE decode",
                            {"continuation latent": reference_latent},
                        )
                    video_items.append(_decoded_video_item(vae, reference_latent))
                    video_refs.append(_video_ref_block(reference_latent))
                    continuation_video_label = f"<Video {video_number}>"
                    chunk_prompt_source = _video_continuation_prompt(chunk_prompt_source, continuation_video_label)
                if continuation and qwen_full_history:
                    history_latent = torch.cat(output_video, dim=2)
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} before history VAE decode",
                            {"history latent": history_latent},
                        )
                    video_items.append(_decoded_video_item(vae, history_latent))
                    del history_latent
                if debug and video_items:
                    presentations = ", ".join(
                        f"{item['data'].shape[0]} frames at {item['data'].shape[2]}x{item['data'].shape[1]}"
                        for item in video_items
                    )
                    logging.info(
                        "SamplerCustomAdvanced-Unlimited chunk %d/%d Qwen video presentation: %s",
                        index + 1,
                        len(active_plan),
                        presentations,
                    )
                chunk_prompt = _prompt_for_chunk(
                    chunk_prompt_source,
                    chunk["frame_start"],
                    chunk["frame_end"],
                    total_frames,
                    fps,
                    content_start=content_start,
                    continuation=continuation,
                    drop_picture_anchors=continuation and not ref2va,
                    continuation_video_label=continuation_video_label,
                    has_opening_frames=guide_enabled,
                )
                if debug:
                    debug_prompt = (
                        f"=== Chunk {index + 1}: sampled frames {chunk['frame_start']}-{chunk['frame_end'] - 1}; "
                        f"output frames {content_start}-{chunk['frame_end'] - 1} ===\n{chunk_prompt}"
                    )
                    debug_prompts.append(debug_prompt)
                    logging.info("SamplerCustomAdvanced-Unlimited debug:\n%s", debug_prompt)
                if vram_monitor is not None:
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} before Qwen encode",
                        {
                            "chunk latent": chunk_latent,
                            "chunk noise": chunk_noise,
                            "Qwen video frames": video_items,
                            "DiT video references": video_refs,
                        },
                    )
                encoded_prompt = _encode_prompt(clip, chunk_prompt, images, positive, width, height, continuation, video_items)
                del video_items
                if vram_monitor is not None:
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} after Qwen encode",
                        {"encoded prompt": encoded_prompt, "DiT video references": video_refs},
                    )
                comfy.model_management.unload_model_and_clones(clip.patcher)
                if vae is not None:
                    comfy.model_management.unload_model_and_clones(vae.patcher)
                if debug:
                    logging.info(
                        "SamplerCustomAdvanced-Unlimited released Qwen%s before chunk %d/%d",
                        " and the H3 video VAE" if vae is not None else "",
                        index + 1,
                        len(active_plan),
                    )
                    vram_monitor.report(
                        f"chunk {index + 1}/{len(active_plan)} after Qwen/VAE release",
                        {"encoded prompt": encoded_prompt, "DiT video references": video_refs},
                    )
                guider.original_conds = _conditioning_for_chunk(
                    original_conds,
                    chunk["frame_start"],
                    chunk["frame_end"],
                    encoded_prompt,
                    video_context,
                    audio_context,
                    audio_end_frame,
                    video_refs,
                )

                chunk_seed = (noise.seed + index) & 0xffffffffffffffff
                if preview_execution is not None:
                    preview_execution.set_chunk(
                        index,
                        chunk["frame_start"],
                        chunk["frame_end"] - 1,
                        content_start,
                        chunk["frame_end"] - 1,
                        context_video_t,
                    )
                try:
                    chunk_progress.start(index)
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} immediately before sampler",
                            {
                                "chunk latent": chunk_latent,
                                "chunk noise": chunk_noise,
                                "conditioning": guider.original_conds,
                                "completed output": (output_video, output_audio),
                                "completed denoised output": (denoised_video, denoised_audio),
                            },
                        )
                    sampled, denoised = super().execute(
                        _FixedNoise(chunk_seed, chunk_noise), guider, sampler, sigmas, chunk_latent
                    )
                    if vram_monitor is not None:
                        vram_monitor.report(
                            f"chunk {index + 1}/{len(active_plan)} sampler complete",
                            {"sampled": sampled, "denoised": denoised},
                        )
                finally:
                    if preview_execution is not None:
                        preview_execution.clear_chunk()
                output_template = sampled
                denoised_template = denoised
                previous_video, previous_audio = sampled["samples"].unbind()
                previous_frame_count = chunk["frame_end"] - chunk["frame_start"]
                denoised_chunk_video, denoised_chunk_audio = denoised["samples"].unbind()

                video_trim = context_video_t
                audio_trim = 0 if index == 0 else context_audio_t
                output_video.append(previous_video[:, :, video_trim:].clone())
                output_audio.append(previous_audio[..., audio_trim:].clone())
                denoised_video.append(denoised_chunk_video[:, :, video_trim:].clone())
                denoised_audio.append(denoised_chunk_audio[..., audio_trim:].clone())
                chunk_progress.finish(index)
        finally:
            guider.original_conds = original_conds
            if vram_monitor is not None:
                guider.model_patcher.remove_wrappers_with_key(
                    comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                    VRAM_DEBUG_WRAPPER_KEY,
                )
            if preview_execution is not None:
                preview_execution.close()
            chunk_progress.close()

        output_template["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat(output_video, dim=2),
            torch.cat(output_audio, dim=-1),
        ))
        denoised_template["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat(denoised_video, dim=2),
            torch.cat(denoised_audio, dim=-1),
        ))
        return io.NodeOutput(output_template, denoised_template, "\n\n".join(debug_prompts))

    sample = execute
