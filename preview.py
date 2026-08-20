import base64
import io as pyio
import logging
import math
import queue
import threading
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image, ImageOps

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
from aiohttp import web
from comfy.taesd.taesd import Block, Clamp, conv
from comfy_api.latest import io


try:
    from server import PromptServer
except ImportError:
    PromptServer = None


PREVIEW_WRAPPER_KEY = "minimax_h3_unlimited_preview"
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_PREVIEW_CACHE_LIMIT = 8
_PREVIEW_CACHE = OrderedDict()
_PREVIEW_CACHE_LOCK = threading.Lock()


def _cache_payload(payload):
    node_id = payload.get("node_id")
    execution = payload.get("execution")
    if node_id is None or execution is None:
        return
    node_id = str(node_id)
    action = payload.get("action")
    with _PREVIEW_CACHE_LOCK:
        state = _PREVIEW_CACHE.get(node_id)
        if action == "reset":
            state = {
                "execution": execution,
                "reset": payload.copy(),
                "sample_start": None,
                "progress": None,
                "complete": None,
                "chunks": {},
                "deltas": [],
                "step_times": [],
            }
            _PREVIEW_CACHE[node_id] = state
            _PREVIEW_CACHE.move_to_end(node_id)
            while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_LIMIT:
                del _PREVIEW_CACHE[next(iter(_PREVIEW_CACHE))]
            return
        if state is None or state["execution"] != execution:
            return
        if action == "sample_start":
            state["sample_start"] = payload.copy()
            state["progress"] = None
            state["complete"] = None
            state["deltas"] = []
            state["step_times"] = []
        elif action == "progress":
            state["progress"] = payload.copy()
            step = int(payload.get("step") or 0)
            if step > 0:
                for key, source in (("deltas", "delta"), ("step_times", "step_ms")):
                    values = state[key]
                    while len(values) < step:
                        values.append(None)
                    values[step - 1] = payload.get(source)
        elif action == "chunk":
            state["chunks"][int(payload["chunk"])] = payload.copy()
        elif action == "complete":
            state["complete"] = payload.copy()


def _cached_snapshot(node_id):
    with _PREVIEW_CACHE_LOCK:
        state = _PREVIEW_CACHE.get(str(node_id))
        if state is None and node_id:
            suffix = f":{node_id}"
            for cached_id in reversed(_PREVIEW_CACHE):
                if cached_id == str(node_id) or cached_id.endswith(suffix):
                    state = _PREVIEW_CACHE[cached_id]
                    break
        if state is None:
            return None
        return {
            "execution": state["execution"],
            "reset": state["reset"].copy(),
            "sample_start": None if state["sample_start"] is None else state["sample_start"].copy(),
            "progress": None if state["progress"] is None else state["progress"].copy(),
            "complete": None if state["complete"] is None else state["complete"].copy(),
            "chunks": [state["chunks"][index].copy() for index in sorted(state["chunks"])],
            "deltas": list(state["deltas"]),
            "step_times": list(state["step_times"]),
        }


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/minimax_h3_unlimited_preview/state")
    async def minimax_h3_unlimited_preview_state(request):
        snapshot = _cached_snapshot(request.rel_url.query.get("node_id", ""))
        return web.json_response(snapshot or {}, headers={"Cache-Control": "no-store"})


class _LatestEncoder:
    def __init__(self):
        self.tasks = queue.Queue(maxsize=1)
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="minimax_h3_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, task):
        try:
            self.tasks.put_nowait(task)
        except queue.Full:
            try:
                self.tasks.get_nowait()
            except queue.Empty:
                pass
            try:
                self.tasks.put_nowait(task)
            except queue.Full:
                pass

    def _run(self):
        while True:
            try:
                task = self.tasks.get(timeout=0.1)
            except queue.Empty:
                if self.stopping:
                    return
                continue
            try:
                task()
            except Exception:
                logging.exception("MiniMax H3 accumulated preview encoding failed")
            if self.stopping and self.tasks.empty():
                return

    def close(self):
        self.stopping = True
        self.thread.join(timeout=10.0)


def _build_tiny_decoder(state_dict):
    first_key = next(iter(state_dict))
    if not first_key.split(".", 1)[0].isdigit():
        prefix = first_key.split(".", 1)[0] + "."
        state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

    entries = {}
    for key, value in state_dict.items():
        index, separator, tail = key.partition(".")
        if not separator or not index.isdigit():
            raise ValueError(f"unsupported tiny VAE key: {key}")
        entries.setdefault(int(index), {})[tail] = value

    layers = []
    for index in range(max(entries) + 1):
        values = entries.get(index)
        if values is None:
            layers.append(Clamp() if index == 0 else nn.ReLU() if index == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in values:
            weight = values["conv.0.weight"]
            layers.append(Block(weight.shape[1], weight.shape[0], use_midblock_gn="pool.0.weight" in values))
        elif "weight" in values:
            weight = values["weight"]
            layers.append(conv(weight.shape[1], weight.shape[0], bias="bias" in values))
        else:
            raise ValueError(f"unsupported tiny VAE layer {index}")
    decoder = nn.Sequential(*layers)
    decoder.load_state_dict(state_dict)
    return decoder


class _TinyDecoder:
    def __init__(self, name):
        path = folder_paths.get_full_path("vae_approx", name)
        if path is None:
            raise ValueError(f"tiny VAE '{name}' was not found in models/vae_approx")
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        self.model = _build_tiny_decoder(state_dict)
        self.latent_channels = self.model[1].weight.shape[1]
        self.device = comfy.model_management.vae_device()
        self.dtype = comfy.model_management.vae_dtype(self.device, [torch.float16, torch.bfloat16])
        self.model = self.model.eval().to(device=self.device, dtype=self.dtype)
        if torch.device(self.device).type == "cuda":
            self.model.to(memory_format=torch.channels_last)

    def decode_frame(self, latent):
        decoded = self.model(latent.to(device=self.device, dtype=self.dtype))
        return decoded[0].movedim(0, -1).to(device="cpu", dtype=torch.float32)


def _packed_video(x0, latent_shapes):
    if getattr(x0, "is_nested", False):
        return x0.unbind()[0]
    if latent_shapes and x0.ndim == 3:
        target = latent_shapes[0]
        count = math.prod(int(size) for size in target[1:])
        return x0[:, :, :count].reshape([x0.shape[0]] + list(target)[1:])
    return x0


def _latent_signature(latent, limit=65536):
    flat = latent.detach().reshape(-1)
    stride = max(1, math.ceil(flat.numel() / limit))
    return flat[::stride][:limit].to(device="cpu", dtype=torch.float32)


def _resize_pil(image, max_resolution):
    if max_resolution > 0 and (image.width > max_resolution or image.height > max_resolution):
        return ImageOps.contain(image, (max_resolution, max_resolution), Image.Resampling.LANCZOS)
    return image


def _tensor_image(tensor, max_resolution):
    pixels = tensor.mul(255.0).clamp(0, 255).to(torch.uint8).numpy()
    return _resize_pil(Image.fromarray(pixels), max_resolution)


def _latent_rgb_frames(video, latent_format, indices, max_resolution):
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None or video.ndim != 5:
        return []
    reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
    if reshape is not None:
        video = reshape(video)
    bias = getattr(latent_format, "latent_rgb_factors_bias", None)
    factor_tensor = torch.tensor(factors, device=video.device, dtype=video.dtype).transpose(0, 1)
    bias_tensor = torch.tensor(bias, device=video.device, dtype=video.dtype) if bias is not None else None
    selected = video[0, :, indices].movedim(0, -1)
    rgb = torch.nn.functional.linear(selected, factor_tensor, bias=bias_tensor)
    rgb = rgb.add(1.0).mul(0.5).clamp(0, 1).to(device="cpu", dtype=torch.float32)
    return [_tensor_image(frame, max_resolution) for frame in rgb]


def _tiny_frames(video, decoder, indices, max_resolution):
    if decoder.latent_channels != video.shape[1]:
        raise ValueError(f"tiny VAE expects {decoder.latent_channels} latent channels, but MiniMax H3 uses {video.shape[1]}")
    return [_tensor_image(decoder.decode_frame(video[0, :, index].unsqueeze(0)), max_resolution) for index in indices]


def _frame_selection(video_t, trim_steps, stride, fps, output_start=0):
    indices = list(range(trim_steps, video_t, stride))
    durations = []
    frame_numbers = []
    preview_frames = 0
    for index in indices:
        frame_numbers.append(int(output_start) + preview_frames)
        span = sum(FRAME_PER_TOKEN[position % len(FRAME_PER_TOKEN)] for position in range(index, min(video_t, index + stride)))
        next_preview_frames = preview_frames + span
        durations.append(max(1, round(next_preview_frames * 1000.0 / fps) - round(preview_frames * 1000.0 / fps)))
        preview_frames = next_preview_frames
    return indices, durations, frame_numbers


def _encode_frame_group(frames, durations, quality):
    encoded = []
    for frame in frames:
        buffer = pyio.BytesIO()
        frame.save(buffer, format="WEBP", quality=quality, method=3)
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded, list(durations)


def _send(payload):
    _cache_payload(payload)
    prompt_server = None if PromptServer is None else getattr(PromptServer, "instance", None)
    if prompt_server is not None:
        try:
            prompt_server.send_sync("minimax_h3_unlimited_preview", payload, prompt_server.client_id)
        except Exception as error:
            logging.warning(f"MiniMax H3 accumulated preview could not send an update: {error}")


class _PreviewExecution:
    def __init__(self, wrappers, chunk_ranges, shot_ranges):
        self.items = [(wrapper, wrapper.begin(chunk_ranges, shot_ranges)) for wrapper in wrappers]

    def set_chunk(self, index, sampled_start, sampled_end, output_start, output_end, trim_steps):
        for wrapper, execution_id in self.items:
            wrapper.set_chunk(execution_id, index, sampled_start, sampled_end, output_start, output_end, trim_steps)

    def clear_chunk(self):
        for wrapper, execution_id in self.items:
            wrapper.clear_chunk(execution_id)

    def close(self):
        for wrapper, execution_id in self.items:
            wrapper.finish(execution_id)


def begin_preview_execution(model_patcher, chunk_ranges, shot_ranges=()):
    wrappers = model_patcher.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PREVIEW_WRAPPER_KEY)
    return _PreviewExecution(wrappers, chunk_ranges, shot_ranges) if wrappers else None


class _AccumulatedPreviewWrapper:
    def __init__(self, node_id, max_resolution, quality, fps, frame_stride, tiny_vae):
        self.node_id = str(node_id) if node_id is not None else None
        self.max_resolution = max_resolution
        self.quality = quality
        self.fps = fps
        self.frame_stride = frame_stride
        self.tiny_vae_name = tiny_vae
        self.execution_id = 0
        self.chunk_count = 0
        self.current_chunk = None
        self.decoder = None
        self.decoder_failed = False
        self.started_at = None

    def _elapsed_ms(self):
        return None if self.started_at is None else (time.perf_counter() - self.started_at) * 1000.0

    def begin(self, chunk_ranges, shot_ranges=()):
        self.execution_id += 1
        if isinstance(chunk_ranges, int):
            chunk_ranges = [{"chunk": index + 1} for index in range(chunk_ranges)]
        chunk_ranges = [dict(item) for item in chunk_ranges]
        shot_ranges = [dict(item) for item in shot_ranges]
        self.chunk_count = len(chunk_ranges)
        self.current_chunk = None
        self.decoder = None
        self.decoder_failed = False
        self.started_at = time.perf_counter()
        _send({
            "node_id": self.node_id,
            "action": "reset",
            "execution": self.execution_id,
            "chunk_count": self.chunk_count,
            "chunk_ranges": chunk_ranges,
            "shot_ranges": shot_ranges,
            "total_frames": max((int(item.get("end", -1)) for item in chunk_ranges), default=-1) + 1,
            "fps": self.fps,
            "elapsed_ms": 0.0,
        })
        return self.execution_id

    def set_chunk(self, execution_id, index, sampled_start, sampled_end, output_start, output_end, trim_steps):
        if execution_id == self.execution_id:
            self.current_chunk = {
                "index": index,
                "sampled_start": sampled_start,
                "sampled_end": sampled_end,
                "output_start": output_start,
                "output_end": output_end,
                "trim_steps": trim_steps,
            }

    def clear_chunk(self, execution_id):
        if execution_id == self.execution_id:
            self.current_chunk = None

    def finish(self, execution_id):
        if execution_id != self.execution_id:
            return
        _send({
            "node_id": self.node_id,
            "action": "complete",
            "execution": execution_id,
            "chunk_count": self.chunk_count,
            "elapsed_ms": self._elapsed_ms(),
        })
        self.current_chunk = None
        self.decoder = None
        self.started_at = None

    def _decoder(self):
        if self.tiny_vae_name == "none" or self.decoder_failed:
            return None
        if self.decoder is None:
            try:
                self.decoder = _TinyDecoder(self.tiny_vae_name)
                logging.info(f"MiniMax H3 preview is using tiny VAE '{self.tiny_vae_name}'.")
            except Exception as error:
                logging.warning(f"MiniMax H3 accumulated preview could not load '{self.tiny_vae_name}', using Latent2RGB: {error}")
                self.decoder_failed = True
        return self.decoder

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        chunk = self.current_chunk
        if chunk is None:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

        model_patcher = executor.class_obj.model_patcher
        latent_format = model_patcher.model.latent_format
        encoder = _LatestEncoder()
        original_callback = callback
        execution_id = self.execution_id
        chunk_index = chunk["index"]
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        decoder = self._decoder()
        previewer_name = f"Tiny VAE: {self.tiny_vae_name}" if decoder is not None else "Latent2RGB"
        if decoder is not None and latent_shapes and decoder.latent_channels != int(latent_shapes[0][1]):
            logging.warning(
                f"MiniMax H3 preview ignored '{self.tiny_vae_name}': it expects {decoder.latent_channels} "
                f"latent channels, but the video latent has {latent_shapes[0][1]}."
            )
            self.decoder = None
            self.decoder_failed = True
            decoder = None
            previewer_name = "Latent2RGB"

        initial_signature = None
        try:
            if sigmas is not None and len(sigmas) > 0:
                sigma = sigmas[0].to(noise.device) if hasattr(sigmas[0], "to") else sigmas[0]
                initial_signature = _latent_signature(_packed_video(noise * sigma, latent_shapes))
        except Exception as error:
            logging.warning(f"MiniMax H3 preview could not initialize the latent-change graph: {error}")
        timing = {"last_time": time.perf_counter(), "step_ms": [], "signature": initial_signature}
        _send({
            "node_id": self.node_id,
            "action": "sample_start",
            "execution": execution_id,
            "chunk": chunk_index,
            "chunk_count": self.chunk_count,
            "steps": max(0, len(sigmas_list) - 1),
            "sigmas": sigmas_list,
            "fps": self.fps,
            "previewer": previewer_name,
            "elapsed_ms": self._elapsed_ms(),
        })

        def preview_callback(step, x0, x, callback_total):
            nonlocal decoder, previewer_name
            try:
                video = _packed_video(x0, latent_shapes)
                if video.ndim == 5:
                    now = time.perf_counter()
                    step_ms = (now - timing["last_time"]) * 1000.0
                    timing["last_time"] = now
                    timing["step_ms"].append(step_ms)
                    if len(timing["step_ms"]) > 8:
                        timing["step_ms"].pop(0)
                    average_step_ms = sum(timing["step_ms"]) / len(timing["step_ms"])

                    signature = _latent_signature(video)
                    previous_signature = timing["signature"]
                    timing["signature"] = signature
                    delta = None
                    if previous_signature is not None and previous_signature.shape == signature.shape:
                        difference = signature - previous_signature
                        delta = (difference.norm() / max(1, difference.numel()) ** 0.5).item()

                    _send({
                        "node_id": self.node_id,
                        "action": "progress",
                        "execution": execution_id,
                        "chunk": chunk_index,
                        "chunk_count": self.chunk_count,
                        "step": step + 1,
                        "steps": callback_total,
                        "sigmas": sigmas_list,
                        "sigma": sigmas_list[step] if 0 <= step < len(sigmas_list) else None,
                        "delta": delta,
                        "step_ms": step_ms,
                        "avg_step_ms": average_step_ms,
                        "fps": self.fps,
                        "previewer": previewer_name,
                        "elapsed_ms": self._elapsed_ms(),
                    })

                    indices, durations, frame_numbers = _frame_selection(
                        video.shape[2],
                        chunk["trim_steps"],
                        self.frame_stride,
                        self.fps,
                        chunk["output_start"],
                    )
                    if decoder is not None:
                        try:
                            frames = _tiny_frames(video, decoder, indices, self.max_resolution)
                        except Exception as error:
                            logging.warning(f"MiniMax H3 tiny VAE preview failed, using Latent2RGB: {error}")
                            self.decoder = None
                            self.decoder_failed = True
                            decoder = None
                            previewer_name = "Latent2RGB (tiny VAE failed)"
                            frames = _latent_rgb_frames(video, latent_format, indices, self.max_resolution)
                    else:
                        frames = _latent_rgb_frames(video, latent_format, indices, self.max_resolution)
                    if frames:
                        payload = {
                            "node_id": self.node_id,
                            "action": "chunk",
                            "execution": execution_id,
                            "chunk": chunk_index,
                            "chunk_count": self.chunk_count,
                            "step": step + 1,
                            "steps": callback_total,
                            "sigmas": sigmas_list,
                            "sampled_start": chunk["sampled_start"],
                            "sampled_end": chunk["sampled_end"],
                            "output_start": chunk["output_start"],
                            "output_end": chunk["output_end"],
                            "frame_numbers": frame_numbers,
                            "duration_ms": sum(durations),
                            "width": frames[0].width,
                            "height": frames[0].height,
                            "fps": self.fps,
                            "previewer": previewer_name,
                            "elapsed_ms": self._elapsed_ms(),
                        }

                        def encode_and_send(frames=frames, durations=durations, payload=payload):
                            encoded, frame_durations = _encode_frame_group(frames, durations, self.quality)
                            if encoded:
                                payload["frames"] = encoded
                                payload["frame_durations_ms"] = frame_durations
                                _send(payload)

                        encoder.submit(encode_and_send)
            except Exception as error:
                logging.warning(f"MiniMax H3 accumulated preview failed for chunk {chunk_index + 1}: {error}")
            if original_callback is not None:
                original_callback(step, x0, x, callback_total)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, preview_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.close()


class MiniMaxH3UnlimitedPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3UnlimitedPreview",
            display_name="MiniMax H3 Unlimited Preview",
            category="model/sampling/custom",
            description="Accumulates live MiniMax H3 previews across SamplerCustomAdvanced-Unlimited chunks.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("max_resolution", default=0, min=0, max=8192, step=8,
                             tooltip="Maximum preview side in pixels. 0 keeps the decoder's native output resolution."),
                io.Int.Input("quality", default=75, min=30, max=100, step=1),
                io.Float.Input("fps", default=24.0, min=1.0, step=1.0,
                               tooltip="Preview playback FPS. The browser applies changes immediately while a preview is playing."),
                io.Int.Input("frame_stride", default=1, min=1, max=16, step=1,
                             tooltip="Preview every Nth H3 latent frame while preserving its playback duration."),
                io.Combo.Input("tiny_vae", options=["none"] + folder_paths.get_filename_list("vae_approx"), default="none",
                               tooltip="Optional compatible 24-channel decoder such as taeh3.safetensors. None uses Latent2RGB."),
            ],
            outputs=[io.Model.Output()],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, max_resolution, quality, fps, frame_stride, tiny_vae="none"):
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            PREVIEW_WRAPPER_KEY,
            _AccumulatedPreviewWrapper(cls.hidden.unique_id, max_resolution, quality, fps, frame_stride, tiny_vae),
        )
        return io.NodeOutput(patched)
