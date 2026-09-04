import base64
import io as pyio
import logging
import math
import queue
import threading
import time
import wave
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


PREVIEW_WRAPPER_KEY = "hr_endless_sampler_preview"
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
                "phase": None,
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
        elif action == "phase":
            state["phase"] = payload.copy()
        elif action in ("chunk", "chunk_final"):
            chunk_index = int(payload["chunk"])
            cached_chunk = state["chunks"].get(chunk_index)
            # Full-VAE frames and their decoded audio are authoritative for a
            # completed chunk.  The latent encoder and websocket/state restore
            # travel on independent asynchronous paths, so a delayed live
            # ``chunk`` payload can otherwise arrive after ``chunk_final`` and
            # downgrade a refreshed preview back to Latent2RGB/tiny-VAE frames.
            if action == "chunk" and cached_chunk is not None \
                    and cached_chunk.get("action") == "chunk_final":
                return
            state["chunks"][chunk_index] = payload.copy()
        elif action == "chunk_audio_update":
            chunk_index = int(payload["chunk"])
            cached_chunk = state["chunks"].get(chunk_index)
            if cached_chunk is None or cached_chunk.get("action") != "chunk_final":
                return
            audio = payload.get("audio")
            if isinstance(audio, str) and audio:
                cached_chunk["audio"] = audio
                cached_chunk["audio_mime"] = payload.get("audio_mime", "audio/wav")
                if payload.get("audio_sample_rate") is not None:
                    cached_chunk["audio_sample_rate"] = int(payload["audio_sample_rate"])
        elif action == "chunk_metadata":
            # Metadata is sent as soon as Gemma has directed a chunk, before
            # the first preview image is encoded. Keep it inside reset's
            # durable timeline payload so a browser refresh restores tooltip
            # text for both completed and currently sampling chunks.
            try:
                chunk_index = int(payload["chunk"])
            except (KeyError, TypeError, ValueError):
                return
            chunk_ranges = state["reset"].get("chunk_ranges", ())
            # Preview payloads index chunks from zero, while the human-facing
            # ``chunk`` number stored in each range starts at one.
            if 0 <= chunk_index < len(chunk_ranges):
                description = payload.get("gemma_detailed_description")
                if isinstance(description, str) and description.strip():
                    chunk_ranges[chunk_index]["gemma_detailed_description"] = description.strip()
                retention_analysis = payload.get("gemma_retention_analysis")
                if isinstance(retention_analysis, str) and retention_analysis.strip():
                    chunk_ranges[chunk_index]["gemma_retention_analysis"] = retention_analysis.strip()
                for key in (
                    "h3_render_seconds",
                    "gemma_seconds",
                    "gemma_preproduction_seconds",
                    "chunk_total_seconds",
                ):
                    value = payload.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                        chunk_ranges[chunk_index][key] = float(value)
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
            "phase": None if state["phase"] is None else state["phase"].copy(),
            "chunks": [state["chunks"][index].copy() for index in sorted(state["chunks"])],
            "deltas": list(state["deltas"]),
            "step_times": list(state["step_times"]),
        }


def build_cached_final_preview_snapshot(node_id, chunk_ranges, shot_ranges, chunks, *, fps,
                                        max_resolution=0, quality=75):
    """Encode finalized replay media into one browser-restorable snapshot.

    This deliberately consumes the sampler's CPU checkpoint media only.  It
    is used after a browser refresh when no model wrapper is executing, so it
    must not instantiate a Tiny VAE or reload any sampling model.
    """
    # Zero is intentionally below every live wrapper execution id. A later
    # sampler reset must always replace this dormant cache snapshot; a
    # nanosecond timestamp would exceed JavaScript's safe integer range.
    execution = 0
    resolved_fps = max(0.001, float(fps))
    resolved_resolution = max(0, int(max_resolution))
    resolved_quality = max(1, min(100, int(quality)))
    normalized_ranges = [dict(item) for item in chunk_ranges]
    reset = {
        "node_id": str(node_id),
        "action": "reset",
        "execution": execution,
        "chunk_count": len(normalized_ranges),
        "chunk_ranges": normalized_ranges,
        "shot_ranges": [dict(item) for item in shot_ranges],
        "reusing_cached_chunks": True,
        "cached_chunk_count": len(chunks),
        "total_frames": max((int(item.get("end", -1)) for item in normalized_ranges), default=-1) + 1,
        "fps": resolved_fps,
        "elapsed_ms": 0.0,
        "phase": "Restored finalized cached chunks",
    }
    _cache_payload(reset)

    for chunk in chunks:
        frames = chunk.get("frames")
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or not int(frames.shape[0]):
            continue
        index = int(chunk["index"])
        output_start = int(chunk["output_start"])
        output_end = output_start + int(frames.shape[0]) - 1
        images = [_tensor_image(frame, resolved_resolution) for frame in frames]
        encoded, durations = _encode_frame_group(
            images,
            _exact_frame_durations(int(frames.shape[0]), resolved_fps),
            resolved_quality,
        )
        payload = {
            "node_id": str(node_id),
            "action": "chunk_final",
            "execution": execution,
            "chunk": index,
            "chunk_count": len(normalized_ranges),
            "step": 0,
            "steps": 0,
            "output_start": output_start,
            "output_end": output_end,
            "frame_numbers": list(range(output_start, output_end + 1)),
            "duration_ms": sum(durations),
            "frame_durations_ms": durations,
            "width": images[0].width,
            "height": images[0].height,
            "fps": resolved_fps,
            "previewer": "Full VAE (cached)",
            "finalized": True,
            "elapsed_ms": 0.0,
            "frames": encoded,
        }
        waveform = chunk.get("audio")
        sample_rate = chunk.get("audio_sample_rate")
        if isinstance(waveform, torch.Tensor) and sample_rate is not None:
            payload["audio"] = _encode_audio_wav(waveform, int(sample_rate))
            payload["audio_mime"] = "audio/wav"
            payload["audio_sample_rate"] = int(sample_rate)
        for key in (
            "gemma_detailed_description",
            "gemma_retention_analysis",
            "h3_render_seconds",
            "gemma_seconds",
            "gemma_preproduction_seconds",
            "chunk_total_seconds",
        ):
            if key in chunk and chunk[key] is not None:
                payload[key] = chunk[key]
        _cache_payload(payload)

    _cache_payload({
        "node_id": str(node_id),
        "action": "complete",
        "execution": execution,
        "elapsed_ms": 0.0,
    })
    return _cached_snapshot(node_id)


_PROMPT_SERVER = None if PromptServer is None else getattr(PromptServer, "instance", None)
if _PROMPT_SERVER is not None:
    @_PROMPT_SERVER.routes.get("/hr_endless_sampler_preview/state")
    async def hr_endless_sampler_preview_state(request):
        snapshot = _cached_snapshot(request.rel_url.query.get("node_id", ""))
        return web.json_response(snapshot or {}, headers={"Cache-Control": "no-store"})


class _LatestEncoder:
    def __init__(self):
        self.tasks = queue.Queue(maxsize=1)
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="hr_endless_sampler_preview_encoder", daemon=True)
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
                logging.exception("HR Endless Sampler preview encoding failed")
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
        raise ValueError(f"tiny VAE expects {decoder.latent_channels} latent channels, but the active video latent uses {video.shape[1]}")
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


def _exact_frame_durations(frame_count, fps):
    """Return integer millisecond durations without accumulating rounding drift."""
    durations = []
    for index in range(max(0, int(frame_count))):
        durations.append(max(1, round((index + 1) * 1000.0 / fps) - round(index * 1000.0 / fps)))
    return durations


def _encode_audio_wav(waveform, sample_rate):
    """Encode one decoded ComfyUI waveform as browser-compatible PCM16 WAV."""
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("final preview audio must be a torch tensor")
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2 or not waveform.shape[0] or not waveform.shape[1]:
        raise ValueError("final preview audio must have shape [B,C,S] or [C,S]")
    channels = int(waveform.shape[0])
    pcm = waveform.detach().to(device="cpu", dtype=torch.float32)
    pcm = pcm.clamp(-1.0, 1.0).movedim(0, 1).mul(32767.0).round().to(torch.int16).contiguous()
    buffer = pyio.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm.numpy().tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _send(payload):
    _cache_payload(payload)
    prompt_server = None if PromptServer is None else getattr(PromptServer, "instance", None)
    if prompt_server is not None:
        try:
            prompt_server.send_sync("hr_endless_sampler_preview", payload, prompt_server.client_id)
        except Exception as error:
            logging.warning(f"HR Endless Sampler preview could not send an update: {error}")


class _PreviewExecution:
    def __init__(self, wrappers, chunk_ranges, shot_ranges, reusing_cached_chunks=False,
                 cached_chunk_count=0):
        self.items = [(
            wrapper,
            wrapper.begin(
                chunk_ranges,
                shot_ranges,
                reusing_cached_chunks=reusing_cached_chunks,
                cached_chunk_count=cached_chunk_count,
            ),
        ) for wrapper in wrappers]

    def set_chunk(self, index, sampled_start, sampled_end, output_start, output_end, trim_steps,
                  gemma_detailed_description=None, gemma_retention_analysis=None):
        for wrapper, execution_id in self.items:
            wrapper.set_chunk(
                execution_id,
                index,
                sampled_start,
                sampled_end,
                output_start,
                output_end,
                trim_steps,
                gemma_detailed_description,
                gemma_retention_analysis,
            )

    def set_phase(self, phase, *, chunk=None):
        for wrapper, execution_id in self.items:
            wrapper.set_phase(execution_id, phase, chunk=chunk)

    def restore_chunks(self, chunks, latent_format):
        """Rebuild completed replay chunks before new sampling resumes."""
        try:
            for restored_index, chunk in enumerate(chunks, 1):
                chunk_index = int(chunk.get("index", restored_index - 1))
                self.set_phase(
                    f"Restoring cached preview Chunk {restored_index}/{len(chunks)}",
                    chunk=chunk_index,
                )
                for wrapper, execution_id in self.items:
                    wrapper.restore_chunk(execution_id, latent_format=latent_format, **chunk)
        finally:
            for wrapper, execution_id in self.items:
                wrapper.release_restore_decoder(execution_id)

    def set_chunk_timing(self, index, *, h3_render_seconds, gemma_seconds,
                         gemma_preproduction_seconds, chunk_total_seconds):
        for wrapper, execution_id in self.items:
            wrapper.set_chunk_timing(
                execution_id,
                index,
                h3_render_seconds=h3_render_seconds,
                gemma_seconds=gemma_seconds,
                gemma_preproduction_seconds=gemma_preproduction_seconds,
                chunk_total_seconds=chunk_total_seconds,
            )

    def finalize_chunk(self, index, frames, output_start, output_end, *,
                       audio_waveform=None, audio_sample_rate=None,
                       gemma_detailed_description=None, gemma_retention_analysis=None):
        for wrapper, execution_id in self.items:
            wrapper.finalize_chunk(
                execution_id,
                index,
                frames,
                output_start,
                output_end,
                audio_waveform=audio_waveform,
                audio_sample_rate=audio_sample_rate,
                gemma_detailed_description=gemma_detailed_description,
                gemma_retention_analysis=gemma_retention_analysis,
            )

    def replace_audio_tail(self, index, waveform, sample_rate):
        """Retroactively install a generated overlap before ``index``."""
        for wrapper, execution_id in self.items:
            wrapper.replace_audio_tail(execution_id, index, waveform, sample_rate)

    def clear_chunk(self):
        for wrapper, execution_id in self.items:
            wrapper.clear_chunk(execution_id)

    def close(self):
        for wrapper, execution_id in self.items:
            wrapper.finish(execution_id)


def begin_preview_execution(model_patcher, chunk_ranges, shot_ranges=(), reusing_cached_chunks=False,
                            cached_chunk_count=0):
    wrappers = model_patcher.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PREVIEW_WRAPPER_KEY)
    return _PreviewExecution(
        wrappers,
        chunk_ranges,
        shot_ranges,
        reusing_cached_chunks=reusing_cached_chunks,
        cached_chunk_count=cached_chunk_count,
    ) if wrappers else None


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
        self.final_audio = {}
        self.final_audio_rates = {}

    def _elapsed_ms(self):
        return None if self.started_at is None else (time.perf_counter() - self.started_at) * 1000.0

    def begin(self, chunk_ranges, shot_ranges=(), reusing_cached_chunks=False,
              cached_chunk_count=0):
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
        self.final_audio = {}
        self.final_audio_rates = {}
        _send({
            "node_id": self.node_id,
            "action": "reset",
            "execution": self.execution_id,
            "chunk_count": self.chunk_count,
            "chunk_ranges": chunk_ranges,
            "shot_ranges": shot_ranges,
            "reusing_cached_chunks": bool(reusing_cached_chunks),
            "cached_chunk_count": max(0, min(int(cached_chunk_count), self.chunk_count)),
            "total_frames": max((int(item.get("end", -1)) for item in chunk_ranges), default=-1) + 1,
            "fps": self.fps,
            "elapsed_ms": 0.0,
            "phase": "Preparing sampler",
        })
        return self.execution_id

    def set_phase(self, execution_id, phase, *, chunk=None):
        if execution_id != self.execution_id:
            return
        payload = {
            "node_id": self.node_id,
            "action": "phase",
            "execution": execution_id,
            "phase": str(phase),
            "elapsed_ms": self._elapsed_ms(),
        }
        if chunk is not None:
            payload["chunk"] = int(chunk)
        _send(payload)

    def set_chunk(self, execution_id, index, sampled_start, sampled_end, output_start, output_end, trim_steps,
                  gemma_detailed_description=None, gemma_retention_analysis=None):
        if execution_id == self.execution_id:
            self.current_chunk = {
                "index": index,
                "sampled_start": sampled_start,
                "sampled_end": sampled_end,
                "output_start": output_start,
                "output_end": output_end,
                "trim_steps": trim_steps,
            }
            metadata = {
                "node_id": self.node_id,
                "action": "chunk_metadata",
                "execution": execution_id,
                "chunk": index,
            }
            if isinstance(gemma_detailed_description, str) and gemma_detailed_description.strip():
                description = gemma_detailed_description.strip()
                self.current_chunk["gemma_detailed_description"] = description
                metadata["gemma_detailed_description"] = description
            if isinstance(gemma_retention_analysis, str) and gemma_retention_analysis.strip():
                retention_analysis = gemma_retention_analysis.strip()
                self.current_chunk["gemma_retention_analysis"] = retention_analysis
                metadata["gemma_retention_analysis"] = retention_analysis
            if len(metadata) > 4:
                _send(metadata)

    def clear_chunk(self, execution_id):
        if execution_id == self.execution_id:
            self.current_chunk = None

    def restore_chunk(self, execution_id, *, index, video, sampled_start, sampled_end,
                      output_start, output_end, trim_steps, latent_format,
                      gemma_detailed_description=None, gemma_retention_analysis=None):
        """Publish a cached completed latent as an ordinary playable chunk."""
        if execution_id != self.execution_id:
            return
        try:
            if not isinstance(video, torch.Tensor) or video.ndim != 5:
                raise ValueError("cached replay video must be a [B,C,T,H,W] tensor")
            indices, durations, frame_numbers = _frame_selection(
                int(video.shape[2]),
                int(trim_steps),
                self.frame_stride,
                self.fps,
                output_start,
            )
            decoder = self._decoder()
            previewer_name = f"Tiny VAE: {self.tiny_vae_name} (replay restore)" if decoder is not None else "Latent2RGB (replay restore)"
            if decoder is not None and decoder.latent_channels != int(video.shape[1]):
                logging.warning(
                    "HR Endless Sampler preview ignored '%s' while restoring replay Chunk %d: "
                    "it expects %d latent channels, but the cached video has %d.",
                    self.tiny_vae_name,
                    index + 1,
                    decoder.latent_channels,
                    int(video.shape[1]),
                )
                self.decoder = None
                self.decoder_failed = True
                decoder = None
                previewer_name = "Latent2RGB (replay restore)"
            if decoder is not None:
                try:
                    frames = _tiny_frames(video, decoder, indices, self.max_resolution)
                except Exception as error:
                    logging.warning(
                        "HR Endless Sampler tiny VAE replay restore failed for Chunk %d; "
                        "using Latent2RGB: %s",
                        index + 1,
                        error,
                    )
                    self.decoder = None
                    self.decoder_failed = True
                    frames = _latent_rgb_frames(video, latent_format, indices, self.max_resolution)
                    previewer_name = "Latent2RGB (replay restore; tiny VAE failed)"
            else:
                frames = _latent_rgb_frames(video, latent_format, indices, self.max_resolution)
            if not frames:
                raise ValueError("the active latent format cannot decode cached replay preview frames")
            encoded, frame_durations = _encode_frame_group(frames, durations, self.quality)
            if not encoded:
                raise ValueError("cached replay preview encoded no frames")
            payload = {
                "node_id": self.node_id,
                "action": "chunk",
                "execution": execution_id,
                "chunk": int(index),
                "chunk_count": self.chunk_count,
                "step": 0,
                "steps": 0,
                "sampled_start": int(sampled_start),
                "sampled_end": int(sampled_end),
                "output_start": int(output_start),
                "output_end": int(output_end),
                "frame_numbers": frame_numbers,
                "duration_ms": sum(durations),
                "frame_durations_ms": frame_durations,
                "width": frames[0].width,
                "height": frames[0].height,
                "fps": self.fps,
                "previewer": previewer_name,
                "elapsed_ms": self._elapsed_ms(),
                "frames": encoded,
            }
            if isinstance(gemma_detailed_description, str) and gemma_detailed_description.strip():
                payload["gemma_detailed_description"] = gemma_detailed_description.strip()
            if isinstance(gemma_retention_analysis, str) and gemma_retention_analysis.strip():
                payload["gemma_retention_analysis"] = gemma_retention_analysis.strip()
            _send(payload)
        except Exception as error:
            logging.warning(
                "HR Endless Sampler could not restore cached preview Chunk %d; "
                "sampling will continue: %s",
                index + 1,
                error,
            )

    def release_restore_decoder(self, execution_id):
        if execution_id == self.execution_id:
            # A replay restore happens before Gemma/Qwen/H3 preparation. Do not
            # keep an unmanaged tiny-decoder CUDA allocation across that work;
            # live sampling can load it again when its first callback arrives.
            self.decoder = None

    def set_chunk_timing(self, execution_id, index, *, h3_render_seconds, gemma_seconds,
                         gemma_preproduction_seconds, chunk_total_seconds):
        if execution_id != self.execution_id:
            return
        _send({
            "node_id": self.node_id,
            "action": "chunk_metadata",
            "execution": execution_id,
            "chunk": index,
            "h3_render_seconds": float(h3_render_seconds),
            "gemma_seconds": float(gemma_seconds),
            "gemma_preproduction_seconds": float(gemma_preproduction_seconds),
            "chunk_total_seconds": float(chunk_total_seconds),
        })

    def finalize_chunk(self, execution_id, index, frames, output_start, output_end, *,
                       audio_waveform=None, audio_sample_rate=None,
                       gemma_detailed_description=None, gemma_retention_analysis=None):
        """Atomically replace a latent preview group with every final VAE frame."""
        if execution_id != self.execution_id:
            return
        try:
            if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or not frames.shape[0]:
                raise ValueError("final video VAE frames must have shape [T,H,W,C]")
            expected_frames = max(0, int(output_end) - int(output_start) + 1)
            if expected_frames and int(frames.shape[0]) != expected_frames:
                logging.warning(
                    "HR Endless Sampler final preview Chunk %d decoded %d frames for a %d-frame output range; "
                    "publishing the available intersection.",
                    int(index) + 1,
                    int(frames.shape[0]),
                    expected_frames,
                )
            frame_count = min(int(frames.shape[0]), expected_frames or int(frames.shape[0]))
            frames = frames[:frame_count].detach().to(device="cpu", dtype=torch.float32)
            durations = _exact_frame_durations(frame_count, self.fps)
            images = [_tensor_image(frame, self.max_resolution) for frame in frames]
            encoded, frame_durations = _encode_frame_group(images, durations, self.quality)
            if not encoded:
                raise ValueError("final video VAE preview encoded no frames")

            payload = {
                "node_id": self.node_id,
                "action": "chunk_final",
                "execution": execution_id,
                "chunk": int(index),
                "chunk_count": self.chunk_count,
                "step": 0,
                "steps": 0,
                "output_start": int(output_start),
                "output_end": int(output_start) + frame_count - 1,
                "frame_numbers": list(range(int(output_start), int(output_start) + frame_count)),
                "duration_ms": sum(durations),
                "frame_durations_ms": frame_durations,
                "width": images[0].width,
                "height": images[0].height,
                "fps": self.fps,
                "previewer": "Full VAE (final)",
                "finalized": True,
                "elapsed_ms": self._elapsed_ms(),
                "frames": encoded,
            }
            if audio_waveform is not None and audio_sample_rate is not None:
                stored_audio = audio_waveform.detach().to(device="cpu", dtype=torch.float32).clone()
                self.final_audio[int(index)] = stored_audio
                self.final_audio_rates[int(index)] = int(audio_sample_rate)
                payload["audio"] = _encode_audio_wav(audio_waveform, audio_sample_rate)
                payload["audio_mime"] = "audio/wav"
                payload["audio_sample_rate"] = int(audio_sample_rate)
            elif int(index) in self.final_audio:
                # A final output-only color pass replaces image frames after
                # audio has already been finalized. Re-publish that existing
                # chunk audio so the browser does not lose synchronization.
                stored_audio = self.final_audio[int(index)]
                stored_rate = self.final_audio_rates[int(index)]
                payload["audio"] = _encode_audio_wav(stored_audio, stored_rate)
                payload["audio_mime"] = "audio/wav"
                payload["audio_sample_rate"] = stored_rate
            if isinstance(gemma_detailed_description, str) and gemma_detailed_description.strip():
                payload["gemma_detailed_description"] = gemma_detailed_description.strip()
            if isinstance(gemma_retention_analysis, str) and gemma_retention_analysis.strip():
                payload["gemma_retention_analysis"] = gemma_retention_analysis.strip()
            _send(payload)
        except Exception as error:
            logging.warning(
                "HR Endless Sampler could not publish final VAE preview Chunk %d; "
                "keeping its last latent preview: %s",
                int(index) + 1,
                error,
            )

    def replace_audio_tail(self, execution_id, index, waveform, sample_rate):
        """Replace finalized preview audio immediately before one chunk.

        Masked AV sampling owns the generated overlap retroactively: it
        replaces the tail of previously accumulated audio and is then trimmed
        from the new chunk. Browser preview groups must mirror that ownership
        instead of dropping the overlap from both sides of the visible seam.
        """
        if execution_id != self.execution_id:
            return
        try:
            replacement = waveform.detach().to(device="cpu", dtype=torch.float32)
            if replacement.ndim != 3 or not replacement.shape[-1]:
                return
            sample_rate = int(sample_rate)
            remaining = int(replacement.shape[-1])
            available = sum(
                int(self.final_audio[chunk_index].shape[-1])
                for chunk_index in range(int(index))
                if chunk_index in self.final_audio
                and self.final_audio_rates.get(chunk_index) == sample_rate
            )
            if available < remaining:
                raise ValueError(
                    f"preview has only {available} prior audio samples for a {remaining}-sample overlap"
                )
            source_end = remaining
            changed = []
            for chunk_index in range(int(index) - 1, -1, -1):
                target = self.final_audio.get(chunk_index)
                if target is None:
                    continue
                if self.final_audio_rates.get(chunk_index) != sample_rate:
                    raise ValueError("preview audio sample rate changed between chunks")
                if target.shape[:-1] != replacement.shape[:-1]:
                    raise ValueError("preview audio channel layout changed between chunks")
                take = min(int(target.shape[-1]), remaining)
                source_start = source_end - take
                updated = target.clone()
                updated[..., -take:] = replacement[..., source_start:source_end]
                self.final_audio[chunk_index] = updated
                changed.append(chunk_index)
                remaining -= take
                source_end = source_start
                if remaining == 0:
                    break

            for chunk_index in reversed(changed):
                _send({
                    "node_id": self.node_id,
                    "action": "chunk_audio_update",
                    "execution": execution_id,
                    "chunk": chunk_index,
                    "audio": _encode_audio_wav(self.final_audio[chunk_index], sample_rate),
                    "audio_mime": "audio/wav",
                    "audio_sample_rate": sample_rate,
                })
        except Exception as error:
            logging.warning(
                "HR Endless Sampler could not update finalized preview audio before Chunk %d: %s",
                int(index) + 1,
                error,
            )

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
        self.final_audio = {}
        self.final_audio_rates = {}
        self.started_at = None

    def _decoder(self):
        if self.tiny_vae_name == "none" or self.decoder_failed:
            return None
        if self.decoder is None:
            try:
                self.decoder = _TinyDecoder(self.tiny_vae_name)
                logging.info(f"HR Endless Sampler preview is using tiny VAE '{self.tiny_vae_name}'.")
            except Exception as error:
                logging.warning(f"HR Endless Sampler preview could not load '{self.tiny_vae_name}', using Latent2RGB: {error}")
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
                f"HR Endless Sampler preview ignored '{self.tiny_vae_name}': it expects {decoder.latent_channels} "
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
            logging.warning(f"HR Endless Sampler preview could not initialize the latent-change graph: {error}")
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
                            logging.warning(f"HR Endless Sampler tiny VAE preview failed, using Latent2RGB: {error}")
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
                        if chunk.get("gemma_detailed_description"):
                            payload["gemma_detailed_description"] = chunk["gemma_detailed_description"]
                        if chunk.get("gemma_retention_analysis"):
                            payload["gemma_retention_analysis"] = chunk["gemma_retention_analysis"]

                        def encode_and_send(frames=frames, durations=durations, payload=payload):
                            encoded, frame_durations = _encode_frame_group(frames, durations, self.quality)
                            if encoded:
                                payload["frames"] = encoded
                                payload["frame_durations_ms"] = frame_durations
                                _send(payload)

                        encoder.submit(encode_and_send)
            except Exception as error:
                logging.warning(f"HR Endless Sampler preview failed for chunk {chunk_index + 1}: {error}")
            if original_callback is not None:
                original_callback(step, x0, x, callback_total)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, preview_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.close()


class HREndlessSamplerPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HREndlessSamplerPreview",
            display_name="HR Endless Sampler Preview",
            category="model/sampling/custom",
            description="Accumulates live previews across HR Endless Sampler chunks.",
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
