import math

import torch

import comfy.conds
import comfy.context_windows
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, PackedLayout
from tqdm.auto import tqdm


WINDOW_INDEX_KEY = "minimax_h3_unlimited_window"
PREPARE_WRAPPER_KEY = "minimax_h3_unlimited_prepare_sampling"


class _WindowProgress:
    def __init__(self, count):
        self.count = count
        self.started = False
        self.bar = tqdm(
            total=count,
            desc=f"Chunk 1/{count}",
            unit="chunk",
            leave=False,
            disable=not comfy.utils.PROGRESS_BAR_ENABLED,
        )

    def start(self, index):
        if index == 0 and self.started:
            self.bar.reset(total=self.count)
        self.started = True
        self.bar.n = index
        self.bar.set_description_str(f"Chunk {index + 1}/{self.count}")
        self.bar.refresh()

    def finish(self):
        self.bar.n = self.count
        self.bar.set_description_str(f"Chunk {self.count}/{self.count}")
        self.bar.refresh()

    def close(self):
        self.bar.close()


def _pixel_start(latent_index):
    cycles, remainder = divmod(latent_index, len(FRAME_PER_TOKEN))
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def _pyramid_weights(length, device, dtype):
    middle = (length + 1) // 2
    weights = [min(index + 1, length - index, middle) for index in range(length)]
    return torch.tensor(weights, device=device, dtype=dtype)


class MiniMaxH3WindowedContextHandler(comfy.context_windows.ContextHandlerABC):
    def __init__(self, plan, latent_shapes, window_callback=None):
        self.plan = plan
        self.latent_shapes = latent_shapes
        self.window_callback = window_callback
        self.layouts = {}
        self.progress = _WindowProgress(len(plan))
        self.dim = 2
        self.context_length = max(chunk["video_end"] - chunk["video_start"] for chunk in plan)
        self.freenoise = False

    def noise_shape(self, original_shape):
        video_per_step = self.latent_shapes[0][1] * math.prod(self.latent_shapes[0][3:])
        audio_per_step = math.prod(self.latent_shapes[1][1:-1])
        largest = max(
            video_per_step * (chunk["video_end"] - chunk["video_start"])
            + audio_per_step * (chunk["audio_end"] - chunk["audio_start"])
            for chunk in self.plan
        )
        return [original_shape[0], 1, largest]

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        return len(self.plan) > 1

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        return cond_in

    def _layout(self, model_conds, payload_cond, chunk_index, video_shape, audio_shape):
        cross_attn = model_conds["c_crossattn"].cond
        cache_key = (id(payload_cond), chunk_index, cross_attn.shape[1], video_shape, audio_shape)
        layout = self.layouts.get(cache_key)
        if layout is not None:
            return layout

        payload = payload_cond.cond
        layout = PackedLayout(
            cross_attn.shape[1],
            video_shape[2],
            (video_shape[3] + 1) // 2 * 2,
            (video_shape[4] + 1) // 2 * 2,
            audio_shape[-1],
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )
        chunk = self.plan[chunk_index]
        video_offset = FRAME_RESCALE * _pixel_start(chunk["video_start"])
        audio_offset = chunk["audio_start"]
        for start, end, kind in layout.segments:
            if kind == "video":
                layout.position_ids[start:end, 0] += video_offset
            elif kind == "audio":
                layout.position_ids[start:end, 0] += audio_offset
        self.layouts[cache_key] = layout
        return layout

    def _resize_condition(self, condition, chunk_index, video_shape, audio_shape):
        resized = condition.copy()
        model_conds = condition["model_conds"].copy()
        model_conds["latent_shapes"] = comfy.conds.CONDConstant([video_shape, audio_shape])
        payload_cond = model_conds.get("minimax_payload")
        if payload_cond is None or not isinstance(payload_cond.cond, dict):
            raise ValueError("SamplerCustomAdvanced-Unlimited requires MiniMax H3 conditioning")
        payload = payload_cond.cond.copy()
        payload["layout"] = self._layout(model_conds, payload_cond, chunk_index, video_shape, audio_shape)
        model_conds["minimax_payload"] = payload_cond._copy_with(payload)
        resized["model_conds"] = model_conds
        return resized

    def _window_conditions(self, conds, chunk_index, video_shape, audio_shape):
        resized = []
        for cond_list in conds:
            if cond_list is None:
                resized.append(None)
                continue
            has_windows = any(WINDOW_INDEX_KEY in condition for condition in cond_list)
            selected = [condition for condition in cond_list if condition.get(WINDOW_INDEX_KEY) == chunk_index] if has_windows else cond_list
            if not selected:
                raise ValueError(f"SamplerCustomAdvanced-Unlimited has no conditioning for window {chunk_index + 1}")
            resized.append([
                self._resize_condition(condition, chunk_index, video_shape, audio_shape)
                for condition in selected
            ])
        return resized

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        video, audio = comfy.utils.unpack_latents(x_in, self.latent_shapes)
        video_accum = [torch.zeros_like(video) for _ in conds]
        audio_accum = [torch.zeros_like(audio) for _ in conds]
        video_counts = torch.zeros(video.shape[2], device=video.device, dtype=video.dtype)
        audio_counts = torch.zeros(audio.shape[-1], device=audio.device, dtype=audio.dtype)
        transformer_options = model_options.setdefault("transformer_options", {})
        previous_window = transformer_options.get("context_window")

        try:
            for chunk_index, chunk in enumerate(self.plan):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if self.window_callback is not None:
                    self.window_callback(chunk_index)
                self.progress.start(chunk_index)
                vs, ve = chunk["video_start"], chunk["video_end"]
                aus, aue = chunk["audio_start"], chunk["audio_end"]
                video_slice = video[:, :, vs:ve]
                audio_slice = audio[..., aus:aue]
                sub_x, sub_shapes = comfy.utils.pack_latents((video_slice, audio_slice))
                window = comfy.context_windows.IndexListContextWindow(
                    list(range(vs, ve)), dim=2, total_frames=video.shape[2], context_overlap=2
                )
                window.modality_windows = {
                    1: comfy.context_windows.IndexListContextWindow(
                        list(range(aus, aue)), dim=3, total_frames=audio.shape[-1],
                        context_overlap=chunk["context_audio_t"],
                    )
                }
                transformer_options["context_window"] = window
                sub_conds = self._window_conditions(conds, chunk_index, sub_shapes[0], sub_shapes[1])
                sub_outputs = calc_cond_batch(model, sub_conds, sub_x, timestep, model_options)

                video_weights = _pyramid_weights(ve - vs, video.device, video.dtype).view(1, 1, -1, 1, 1)
                audio_weights = _pyramid_weights(aue - aus, audio.device, audio.dtype).view(1, 1, 1, -1)
                for output_index, output in enumerate(sub_outputs):
                    video_output, audio_output = comfy.utils.unpack_latents(output, sub_shapes)
                    video_accum[output_index][:, :, vs:ve].add_(video_output * video_weights)
                    audio_accum[output_index][..., aus:aue].add_(audio_output * audio_weights)
                video_counts[vs:ve].add_(video_weights.view(-1))
                audio_counts[aus:aue].add_(audio_weights.view(-1))
            self.progress.finish()
        finally:
            if previous_window is None:
                transformer_options.pop("context_window", None)
            else:
                transformer_options["context_window"] = previous_window

        video_counts = video_counts.view(1, 1, -1, 1, 1)
        audio_counts = audio_counts.view(1, 1, 1, -1)
        outputs = []
        for video_output, audio_output in zip(video_accum, audio_accum):
            packed, _ = comfy.utils.pack_latents((video_output / video_counts, audio_output / audio_counts))
            outputs.append(packed)
        return outputs

    def close(self):
        self.progress.close()


def _prepare_sampling_wrapper(executor, model, noise_shape, conds, *args, **kwargs):
    model_options = kwargs["model_options"]
    handler = model_options.get("context_handler")
    if isinstance(handler, MiniMaxH3WindowedContextHandler):
        noise_shape = handler.noise_shape(noise_shape)
    return executor(model, noise_shape, conds, *args, **kwargs)


def add_prepare_sampling_wrapper(model_options):
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        PREPARE_WRAPPER_KEY,
        _prepare_sampling_wrapper,
        model_options,
        is_model_options=True,
    )
