"""Experimental single-device TaoMate-style H3 streaming for ComfyUI.

Implements the public TaoMate-H3 phase geometry, condition-only text attention,
clean media KV commits and first-phase latent-statistics anchoring. Persistent
KV lives on CPU: the memory ceiling is one video sink plus two AV phases for
all layers; a future GPU cache can trade VRAM for reduced transfer overhead.
The caller owns prompt encoding, output decoding and the supplied TaoMate LoRA.
"""

import copy
import math
import logging
import os
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

import comfy.ldm.minimax.model as h3
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.samplers
import comfy.sampler_helpers
from comfy.k_diffusion.sampling import sample_euler
from .taomate_upstream.cache import CleanAVKVCache, KVContract, AVKV, VIDEO_TOKEN_TAG, MAIN_LAYER_NAMES, _validate_av_pair
from .taomate_upstream.attention_hook import H3StreamingAttentionHook, HookMode
from .taomate_upstream.geometry import direct_5s_plan, canonical_continuation_plan, video_temporal_position
from .taomate_upstream.denoise_schedule import select_time_shift_sigmas, DISTILLED_STATE_INDICES

# Experiment: False restores upstream condition-only attention.
TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV = True
# Experiment: True restores prompt/reference movement at five-second boundaries.
TOGGLE_TAOMATE_DIVERGENCY_MOVE_PROMPT_REFERENCES = False
# Deprecated test compatibility switch; node input selects the runtime codec.
TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV = True
# NVIDIA CUDA experiment: transfer compressed bytes, decode and unshuffle on GPU.
TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV = True
# Use all allowed CPU cores, respecting process affinity; 1 restores serial work.
TAOMATE_KV_CPU_WORKERS = max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
KV_CACHE_COMPRESSION_MODES = ("none", "zstd lossless", "int8", "turboquant")
TURBOQUANT_4BIT_CENTROIDS = (-2.7326368225, -2.0690571770, -1.6180378132, -1.2561836443, -0.9423401764, -0.6567588956, -0.3880170670, -0.1284042432, 0.1284042432, 0.3880170670, 0.6567588956, 0.9423401764, 1.2561836443, 1.6180378132, 2.0690571770, 2.7326368225)


def _profile_add(profile, name, started):
    """Accumulate wall and whole-process CPU time for an optional phase profile."""
    if profile is not None:
        wall, cpu = profile.get(name, (0.0, 0.0))
        profile[name] = (wall + time.perf_counter() - started[0], cpu + time.process_time() - started[1])


def _turboquant_rotation(device):
    """Return one deterministic 128-wide orthogonal rotation per CUDA device."""
    cache = getattr(_turboquant_rotation, "cache", {})
    key = str(device)
    if key not in cache:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8675309)
        matrix, upper = torch.linalg.qr(torch.randn(128, 128, generator=generator, dtype=torch.float32))
        signs = torch.sign(torch.diag(upper))
        signs[signs == 0] = 1
        cache[key] = (matrix * signs.unsqueeze(0)).to(device)
        _turboquant_rotation.cache = cache
    return cache[key]


def _turboquant_centroids(device):
    """Build TurboQuant's fixed Gaussian 4-bit codebook for 128-wide heads."""
    return torch.tensor(TURBOQUANT_4BIT_CENTROIDS, device=device, dtype=torch.float32).div(math.sqrt(128.0))


class CompressedKV:
    """Store exact shuffled bytes in bounded Blosc or GPU-compatible Zstd blocks."""

    def __init__(self, pair):
        """Compress one CPU layer; fixed-size blocks bound codec scratch memory."""
        import blosc2
        self.shape = tuple(pair.key.shape)
        self.dtype = pair.key.dtype
        self.raw_zstd = TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV
        self.raw_bytes = 2 * pair.key.numel() * pair.key.element_size()
        self.blocks = []
        for tensor in (pair.key, pair.value):
            raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1).numpy()
            if self.raw_zstd:
                def compress_block(start):
                    """Give each worker its own Zstd context and shuffle buffer."""
                    import zstandard
                    part = raw[start:start + 32 * 1024 ** 2]
                    return zstandard.ZstdCompressor(level=1).compress(part.reshape(-1, tensor.element_size()).T.copy())

                # Jobs reference the existing raw tensor; only active jobs allocate
                # shuffle/codec scratch. Results are the eventual compressed cache.
                with ThreadPoolExecutor(max_workers=TAOMATE_KV_CPU_WORKERS) as pool:
                    self.blocks.append(list(pool.map(compress_block, range(0, raw.size, 32 * 1024 ** 2))))
                continue
            blocks = []
            for start in range(0, raw.size, 32 * 1024 ** 2):
                part = raw[start:start + 32 * 1024 ** 2]
                # Standard Zstd frames let nvCOMP decode without parsing Blosc headers.
                blocks.append(blosc2.compress(part, typesize=tensor.element_size(), clevel=1, filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.ZSTD))
            self.blocks.append(blocks)
        self.stored_bytes = sum(len(block) for blocks in self.blocks for block in blocks)

    def tensor(self, index, device="cpu"):
        """Restore directly into one writable tensor, preserving every bit."""
        import blosc2
        device = torch.device(device)
        output = torch.empty(self.shape, dtype=self.dtype, device=device)
        destination = output.view(torch.uint8).reshape(-1)
        codec = None
        if device.type == "cuda":
            if not self.raw_zstd:
                raise RuntimeError("GPU KV decoding requires the raw Zstd storage format")
            from nvidia import nvcomp
            codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW, device_id=device.index, cuda_stream=torch.cuda.current_stream(device).cuda_stream)
        elif self.raw_zstd:
            target = destination.numpy()

            def decompress_block(block_index):
                """Restore disjoint byte ranges without queuing decompressed results."""
                import numpy as np
                import zstandard
                start = block_index * 32 * 1024 ** 2
                end = min(start + 32 * 1024 ** 2, target.size)
                raw = zstandard.ZstdDecompressor().decompress(self.blocks[index][block_index])
                if len(raw) != end - start:
                    raise RuntimeError("Compressed KV block length does not match its tensor shape")
                target[start:end].reshape(-1, output.element_size())[:] = np.frombuffer(raw, dtype=np.uint8).reshape(output.element_size(), -1).T

            with ThreadPoolExecutor(max_workers=TAOMATE_KV_CPU_WORKERS) as pool:
                list(pool.map(decompress_block, range(len(self.blocks[index]))))
            return output
        offset = 0
        for block in self.blocks[index]:
            if device.type == "cuda":
                # Decode and byte-unshuffle on the consuming PyTorch CUDA stream.
                decoded = codec.decode(block)
                raw = torch.from_dlpack(decoded).view(torch.uint8)
                length = raw.numel()
                destination[offset:offset + length].reshape(-1, output.element_size()).copy_(raw.reshape(output.element_size(), -1).T)
                # ponytail: synchronize each block to bound nvCOMP buffer lifetimes;
                # event-managed reusable buffers can later overlap these operations.
                torch.cuda.current_stream(device).synchronize()
            else:
                raw = blosc2.decompress(block)
                length = len(raw)
                memoryview(destination.numpy())[offset:offset + length] = raw
            offset += length
        if offset != destination.numel():
            raise RuntimeError("Compressed KV byte count does not match its tensor shape")
        return output

    @property
    def key(self):
        """Restore keys only when an attention layer needs them."""
        return self.tensor(0)

    @property
    def value(self):
        """Restore values only when an attention layer needs them."""
        return self.tensor(1)


class QuantizedKV:
    """GPU-quantized, CPU-resident KV vectors for one H3 attention layer."""

    def __init__(self, pair, mode):
        """Quantize BF16 keys and values before transferring their compact form to CPU."""
        if mode not in ("int8", "turboquant"):
            raise ValueError("unknown quantized KV mode %r" % mode)
        if pair.key.shape[-1] != 128:
            raise ValueError("%s KV quantization requires H3's 128-wide attention heads" % mode)
        self.shape = tuple(pair.key.shape)
        self.dtype = pair.key.dtype
        self.mode = mode
        self.raw_bytes = 2 * pair.key.numel() * pair.key.element_size()
        self.key_codes, self.key_scales = self._encode(pair.key)
        self.value_codes, self.value_scales = self._encode(pair.value)
        self.stored_bytes = sum(item.numel() * item.element_size() for item in (self.key_codes, self.key_scales, self.value_codes, self.value_scales))

    def _encode(self, tensor):
        """Encode one [tokens, heads, 128] GPU tensor without a raw CPU copy."""
        values = tensor.detach().float()
        if self.mode == "int8":
            scales = values.abs().amax(dim=-1, keepdim=True).div(127).clamp_min(1e-12)
            codes = torch.round(values.div(scales)).clamp(-127, 127).to(torch.int8)
            return codes.cpu(), scales.to(torch.float16).cpu()
        norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-12)
        rotated = values.div(norms).matmul(_turboquant_rotation(values.device).T)
        centroids = _turboquant_centroids(values.device)
        boundaries = (centroids[:-1] + centroids[1:]).mul(0.5)
        indices = torch.bucketize(rotated, boundaries).to(torch.uint8)
        codes = indices[..., 0::2] | (indices[..., 1::2] << 4)
        return codes.cpu(), norms.cpu()

    @classmethod
    def concat(cls, first, second):
        """Join same-codec token rows without reconstructing their BF16 values."""
        if first.mode != second.mode or first.dtype != second.dtype or first.shape[1:] != second.shape[1:]:
            raise ValueError("cannot concatenate incompatible quantized KV entries")
        result = object.__new__(cls)
        result.shape = (first.shape[0] + second.shape[0],) + first.shape[1:]
        result.dtype = first.dtype
        result.mode = first.mode
        result.raw_bytes = first.raw_bytes + second.raw_bytes
        result.key_codes = torch.cat((first.key_codes, second.key_codes), dim=0)
        result.key_scales = torch.cat((first.key_scales, second.key_scales), dim=0)
        result.value_codes = torch.cat((first.value_codes, second.value_codes), dim=0)
        result.value_scales = torch.cat((first.value_scales, second.value_scales), dim=0)
        result.stored_bytes = sum(item.numel() * item.element_size() for item in (result.key_codes, result.key_scales, result.value_codes, result.value_scales))
        return result

    def select(self, indices):
        """Retain selected token rows while keeping their compact representation."""
        result = object.__new__(self.__class__)
        result.shape = (indices.numel(),) + self.shape[1:]
        result.dtype = self.dtype
        result.mode = self.mode
        result.raw_bytes = 2 * math.prod(result.shape) * self.dtype.itemsize
        result.key_codes = self.key_codes.index_select(0, indices)
        result.key_scales = self.key_scales.index_select(0, indices)
        result.value_codes = self.value_codes.index_select(0, indices)
        result.value_scales = self.value_scales.index_select(0, indices)
        result.stored_bytes = sum(item.numel() * item.element_size() for item in (result.key_codes, result.key_scales, result.value_codes, result.value_scales))
        return result

    def tensor(self, index, device):
        """Restore one approximate KV tensor directly onto the attention device."""
        codes = (self.key_codes, self.value_codes)[index].to(device, non_blocking=True)
        scales = (self.key_scales, self.value_scales)[index].to(device, non_blocking=True)
        if self.mode == "int8":
            return codes.float().mul(scales.float()).to(self.dtype)
        indices = torch.empty(self.shape, dtype=torch.long, device=device)
        indices[..., 0::2] = codes.bitwise_and(15).long()
        indices[..., 1::2] = codes.bitwise_right_shift(4).long()
        centroids = _turboquant_centroids(device)
        restored = centroids[indices].matmul(_turboquant_rotation(device)).mul(scales.float())
        return restored.to(self.dtype)


class CPUStreamingCache(CleanAVKVCache):
    """Upstream cache policy with CPU storage and per-layer device transfers."""

    def __init__(self, contract, compression_mode=None):
        """Keep one render's selected KV representation on CPU."""
        super().__init__(contract)
        self.compression_mode = compression_mode or ("zstd lossless" if TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV else "none")
        if self.compression_mode not in KV_CACHE_COMPRESSION_MODES:
            raise ValueError("unknown KV cache compression mode %r" % self.compression_mode)

    @property
    def history_tokens(self):
        """Read token counts without decompressing any tensors."""
        return sum(self._commit_token_counts)

    def store_pair(self, pair):
        """Use the selected storage format for both staged and retained layers."""
        if self.compression_mode == "zstd lossless":
            return CompressedKV(pair)
        if self.compression_mode in ("int8", "turboquant"):
            return QuantizedKV(pair, self.compression_mode)
        return pair

    def storage_bytes(self):
        """Report retained physical bytes and equivalent uncompressed bytes."""
        stored = raw = 0
        for pair in self._history.values():
            size = pair.raw_bytes if isinstance(pair, (CompressedKV, QuantizedKV)) else sum(t.numel() * t.element_size() for t in (pair.key, pair.value))
            raw += size
            stored += pair.stored_bytes if isinstance(pair, (CompressedKV, QuantizedKV)) else size
        return stored, raw

    def history(self, layer_name):
        """Fetch only the current layer onto its execution device."""
        started = (time.perf_counter(), time.process_time())
        try:
            pair = super().history(layer_name)
            if isinstance(pair, QuantizedKV):
                return AVKV(pair.tensor(0, self.active_device), pair.tensor(1, self.active_device))
            if isinstance(pair, CompressedKV) and TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV and torch.device(self.active_device).type == "cuda":
                return AVKV(pair.tensor(0, self.active_device), pair.tensor(1, self.active_device))
            return None if pair is None else AVKV(pair.key.to(self.active_device), pair.value.to(self.active_device))
        finally:
            _profile_add(getattr(self, "profile", None), "KV restore", started)

    def stage(self, layer_name, key, value, token_tags, commit_mask):
        """Keep all persistent and staged KV in CPU RAM."""
        started = (time.perf_counter(), time.process_time())
        try:
            if self.compression_mode in ("int8", "turboquant"):
                self._stage_quantized(layer_name, key, value, token_tags, commit_mask)
                return
            super().stage(layer_name, key.cpu(), value.cpu(), token_tags.cpu(), commit_mask.cpu())
            self._staged[layer_name] = self.store_pair(self._staged[layer_name])
        finally:
            _profile_add(getattr(self, "profile", None), "KV stage/compress", started)

    def _stage_quantized(self, layer_name, key, value, token_tags, commit_mask):
        """Select media rows and quantize them while the source KV is still on GPU."""
        if not self.clean_commit_active:
            raise RuntimeError("begin_clean_commit() must be called before stage()")
        self._validate_layer_name(layer_name)
        if layer_name in self._staged or key.shape != value.shape or key.ndim != 3:
            raise RuntimeError("%s: invalid duplicate or rank-3 KV staging input" % layer_name)
        if tuple(key.shape[1:]) != (self.contract.local_heads, self.contract.head_dim) or key.dtype != self.contract.dtype:
            raise RuntimeError("%s: KV does not match the H3 cache contract" % layer_name)
        if self._staged_indices is None:
            indices = torch.nonzero(commit_mask.to(torch.bool), as_tuple=False).flatten()
            selected_tags = token_tags.index_select(0, indices)
            tags = tuple(int(tag) for tag in selected_tags.detach().cpu().tolist())
            if not indices.numel() or frozenset(tags) != frozenset((VIDEO_TOKEN_TAG, 2)):
                raise RuntimeError("clean commit did not contain both video and audio KV")
            self._staged_indices = indices
            self._staged_tags = tags
        indices = self._staged_indices
        pair = AVKV(key.index_select(0, indices).detach(), value.index_select(0, indices).detach())
        self._staged[layer_name] = self.store_pair(pair)

    def commit(self):
        """Evict expired history after the clean pass, before upstream concatenation."""
        started = (time.perf_counter(), time.process_time())
        try:
            self._commit()
        finally:
            _profile_add(getattr(self, "profile", None), "KV commit", started)

    def _commit(self):
        """Perform one clean-pass cache commit; timing belongs to the caller."""
        # Validate completeness before modifying usable history, as upstream does.
        if not self.clean_commit_active:
            raise RuntimeError("no clean KV commit is active")
        missing = [name for name in MAIN_LAYER_NAMES if name not in self._staged]
        if missing:
            raise RuntimeError("clean KV commit is missing %d transformer layers" % len(missing))
        count = len(self._commit_token_counts)
        if count >= 2:
            # The incoming commit becomes the newest recent; only one old recent
            # and the first video's anchor remain necessary after this clean pass.
            self.report_status("KV cache: evicting outgoing history")
            self._retain_commit_rows([(0, True), (count - 1, False)])
        self.report_status("KV cache: committing new history")
        if self.compression_mode == "none":
            super().commit()
            return
        # Copied upstream append, with compression at the per-layer storage boundary.
        # Staged KV stays compressed while other layers are being committed.
        for layer_name in MAIN_LAYER_NAMES:
            self.report_status("KV cache: committing/compressing layer %d/%d" % (int(layer_name.split(".")[1]) + 1, len(MAIN_LAYER_NAMES)))
            current = self._staged.pop(layer_name)
            previous = self._history.pop(layer_name, None)
            if previous is None:
                self._history[layer_name] = current
            elif isinstance(previous, QuantizedKV) and isinstance(current, QuantizedKV):
                self._history[layer_name] = QuantizedKV.concat(previous, current)
            else:
                combined = AVKV(torch.cat((previous.key, current.key), dim=0), torch.cat((previous.value, current.value), dim=0))
                _validate_av_pair(combined, self.contract, layer_name=layer_name)
                self._history[layer_name] = self.store_pair(combined)
                del combined
        self._commit_token_counts.append(len(self._staged_tags))
        self._commit_token_tags.append(self._staged_tags)
        self._block_index += 1
        self._clear_staging()

    def report_status(self, message):
        """Publish optional execution-local progress without changing cache contents."""
        callback = getattr(self, "on_status", None)
        if callback is not None:
            callback(message)

    def _retain_commit_rows(self, selection):
        """Copy upstream retention, releasing each old CPU layer as it is replaced."""
        started = (time.perf_counter(), time.process_time())
        try:
            self._retain_commit_rows_inner(selection)
        finally:
            _profile_add(getattr(self, "profile", None), "KV retention", started)

    def _retain_commit_rows_inner(self, selection):
        """Apply the upstream row selection while the outer method measures it."""
        offsets = [0]
        for count in self._commit_token_counts:
            offsets.append(offsets[-1] + count)

        selected_rows: list[int] = []
        selected_counts: list[int] = []
        selected_tags: list[tuple[int, ...]] = []
        for block_index, video_only in selection:
            tags = self._commit_token_tags[block_index]
            start = offsets[block_index]
            local_rows = [
                i for i, tag in enumerate(tags) if not video_only or tag == VIDEO_TOKEN_TAG
            ]
            if not local_rows:
                continue
            selected_rows.extend(start + row for row in local_rows)
            kept_tags = tuple(tags[row] for row in local_rows)
            selected_counts.append(len(kept_tags))
            selected_tags.append(kept_tags)

        # The normal post-commit retention is now already satisfied. Avoid
        # copying every layer again when its selected rows are unchanged.
        if selected_rows == list(range(offsets[-1])):
            self._commit_token_counts = selected_counts
            self._commit_token_tags = selected_tags
            return

        if not selected_rows:
            self._history.clear()
            self._commit_token_counts.clear()
            self._commit_token_tags.clear()
            return

        # Same upstream row selection; reuse the dictionary to avoid a second
        # complete 50-layer cache during CPU retention. Values change, not keys.
        next_history = self._history
        for layer_name, pair in self._history.items():
            self.report_status("KV cache: retaining history, layer %d/%d" % (int(layer_name.split(".")[1]) + 1, len(MAIN_LAYER_NAMES)))
            indices = torch.tensor(selected_rows, dtype=torch.long, device="cpu")
            if isinstance(pair, QuantizedKV):
                next_history[layer_name] = pair.select(indices)
            else:
                retained = AVKV(
                    key=pair.key.index_select(0, indices),
                    value=pair.value.index_select(0, indices),
                )
                _validate_av_pair(retained, self.contract, layer_name=layer_name)
                next_history[layer_name] = self.store_pair(retained)
                del retained
        self._history = next_history
        self._commit_token_counts = selected_counts
        self._commit_token_tags = selected_tags


class ComfyStreamingHook(H3StreamingAttentionHook):
    """Reuse upstream routing with an optional native-style conditioning experiment."""

    def __init__(self, cache):
        """Initialize the upstream hook without importing a Hopper-only kernel."""
        self.cache = cache
        self._flash_attention = self.sdpa
        self.kernel_calls = self.query_tokens = self.key_tokens = 0
        self.mode = HookMode.IDLE
        self._live_documents = None
        self._live_document_rows = {}

    def __call__(self, **kwargs):
        """Optionally let conditioning queries read current AV, never historical KV."""
        output = super().__call__(**kwargs)
        if not TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV:
            return output
        query, key, value = kwargs["query"], kwargs["key"], kwargs["value"]
        mask = kwargs["commit_mask"]
        points = kwargs["cu_seqlens_host"]
        # ponytail: recompute only condition rows to keep vendored routing intact.
        # This adds one attention call per live document; fuse only if profiling warrants it.
        for start, end in zip(points[:-1], points[1:]):
            if not bool(mask[start:end].any()):
                continue
            indices = torch.nonzero(~mask[start:end].bool(), as_tuple=False).flatten() + start
            conditioned = self.sdpa(query.index_select(0, indices).unsqueeze(0), key[start:end].unsqueeze(0), value[start:end].unsqueeze(0), float(kwargs["attention"].softmax_scale))
            output.index_copy_(0, indices, conditioned.squeeze(0))
            self.kernel_calls += 1
            self.query_tokens += indices.numel()
            self.key_tokens += end - start
        return output

    @staticmethod
    def sdpa(query, key, value, softmax_scale, causal=False, num_splits=1):
        """Translate upstream NHD attention arguments to PyTorch SDPA."""
        return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=softmax_scale, is_causal=causal).transpose(1, 2)

    def _validate_qkv(self, layer_name, query, key, value):
        """Validate active QKV on-device while cache storage remains on CPU."""
        contract = self.cache.contract
        self.cache.contract = replace(contract, device_type=query.device.type)
        try:
            super()._validate_qkv(layer_name, query, key, value)
        finally:
            self.cache.contract = contract
        self.cache.active_device = query.device



class TeacherGuidedModel:
    """Expose teacher audio at solver evaluations while forwarding model attributes."""

    def __init__(self, model, state, sigmas, initial):
        """Keep the selected solver's state intact across its complete schedule."""
        self.model = model
        self.state = state
        self.sigmas = sigmas.detach().float().cpu().tolist()
        self.offset = math.prod(state.video_shape)
        self.initial = initial[..., self.offset:].detach().cpu().clone()

    def __getattr__(self, name):
        """Preserve sampler access to the wrapped ComfyUI model interface."""
        return getattr(self.model, name)

    def audio_at(self, index, like):
        """Return a stored audio state on the native carried-sigma clock."""
        if index == 0:
            return self.initial.to(like)
        audio = self.state.teacher_milestones[index - 1][..., self.state.phase_audio_start:self.state.phase_audio_end].to(like)
        carry = self.state.audio_scale + (1.0 - self.state.audio_scale) * self.sigmas[index]
        return audio.reshape(1, 1, -1) * carry

    def __call__(self, x, sigma, **kwargs):
        """Guide internal solver stages by linear interpolation between teacher states."""
        value = float(sigma.flatten()[0])
        index = next((i for i in range(len(self.sigmas) - 1) if value >= self.sigmas[i + 1]), len(self.sigmas) - 2)
        high, low = self.sigmas[index:index + 2]
        start, end = self.audio_at(index, x), self.audio_at(index + 1, x)
        slope = (start - end) / (high - low)
        guided = x.clone()
        # Also permits solver churn above the first sigma through extrapolation.
        guided[..., self.offset:] = start if value == high else end if value == low else start + (value - high) * slope
        prediction = self.model(guided, sigma, **kwargs).clone()
        prediction[..., self.offset:] = x[..., self.offset:] - value * slope
        return prediction


class TaoMateStreaming:
    """One render's isolated attention cache, global clock and clean anchor."""

    @staticmethod
    def frames(tokens):
        """Count frames using H3's global 1,4,4,4,4 temporal phase."""
        return (tokens // 5) * 17 + sum(h3.FRAME_PER_TOKEN[:tokens % 5])

    @classmethod
    def plan(cls, video_t, audio_t, continuation_frames=39):
        """Use a configurable first phase, then its derived continuation phases."""
        if video_t < 2 or (video_t - 2) % 5 or audio_t != round(cls.frames(video_t) * h3.FRAME_RESCALE):
            raise ValueError("TaoMate-H3 needs a valid, synchronized H3 AV latent")
        if isinstance(continuation_frames, bool) or int(continuation_frames) != continuation_frames or continuation_frames < 22 or (continuation_frames - 5) % 17:
            raise ValueError("TaoMate video_continuation must use the H3 grid and be at least 22 frames: 22, 39, 56, ...")
        continuation_frames = int(continuation_frames)
        groups = (continuation_frames - 5) // 17
        first_count = 2 + 5 * groups
        continuation_count = 5 * groups
        plan = []
        start = 0
        while start < video_t:
            index = len(plan)
            # 39 produces the upstream 39/34/34/17 then 34/34/34/17 cadence.
            count = first_count if index == 0 else 5 if index % 4 == 3 else continuation_count
            end = min(start + count, video_t)
            first_frame, end_frame = cls.frames(start), cls.frames(end)
            audio_start, audio_end = round(first_frame * h3.FRAME_RESCALE), round(end_frame * h3.FRAME_RESCALE)
            prefix = 5 if index else 0
            context_audio = round((end_frame - first_frame + prefix) * h3.FRAME_RESCALE) - (audio_end - audio_start)
            plan.append({"video_start": start, "video_end": end, "audio_start": audio_start, "audio_end": audio_end, "frame_start": first_frame - prefix, "frame_end": end_frame, "context_video_t": 2 if index else 0, "context_audio_t": context_audio, "output_trim_frames": prefix, "synthetic_prefix": bool(index)})
            start = end
        return plan

    @classmethod
    def request_plan(cls, video_t, audio_t, chunk_frames=124, continuation_frames=39):
        """Group bounded upstream sub-chunks into user-sized prompt/audio requests."""
        if isinstance(chunk_frames, bool) or int(chunk_frames) != chunk_frames or chunk_frames < 22:
            raise ValueError("TaoMate chunk_frames must be an integer of at least 22")
        chunk_frames = int(chunk_frames)
        chunk_frames -= (chunk_frames - 5) % 17
        capacity = (chunk_frames - 5) // 17 * 5
        phases = cls.plan(video_t, audio_t, continuation_frames)
        requests = []
        phase_index = 0
        start = 0
        while start < video_t:
            end = min(video_t, start + capacity + (2 if start == 0 else 0))
            group = []
            cursor = start
            while cursor < end:
                original = phases[phase_index]
                stop = min(end, original["video_end"])
                phase = dict(original)
                phase.update(video_start=cursor, video_end=stop, audio_start=round(cls.frames(cursor) * h3.FRAME_RESCALE), audio_end=round(cls.frames(stop) * h3.FRAME_RESCALE), frame_end=cls.frames(stop))
                group.append(phase)
                cursor = stop
                if stop == original["video_end"]:
                    phase_index += 1
            request = dict(group[0])
            prefix = 5 if start else 0
            request.update(frame_start=cls.frames(start) - prefix, context_video_t=2 if start else 0, output_trim_frames=prefix, synthetic_prefix=bool(start))
            for key in ("video_end", "audio_end", "frame_end"):
                request[key] = group[-1][key]
            # Only new media enters sampling; the five-frame halo is for previews.
            request["context_audio_t"] = round((request["frame_end"] - request["frame_start"]) * h3.FRAME_RESCALE) - (request["audio_end"] - request["audio_start"])
            request["phases"] = group
            requests.append(request)
            start = end
        return requests

    def __init__(self, kv_cache_compression=None):
        """Start with no retained model or render state."""
        self.cache = None
        self.caches = {}
        self.branch = ()
        self.hook = None
        self.anchor = None
        self.origin = None
        self.layout = None
        self.video_start = 0
        self.request_video_start = None
        self.audio_start = 0
        self.clean = False
        self.commits = 0
        self.layer_count = 0
        self.video_shape = None
        self.audio_teacher = None
        self.teacher_milestones = None
        self.phase_audio_start = 0
        self.phase_audio_end = 0
        self.audio_scale = 1.0
        self.request_count = 0
        self.prepared_model = None
        self.reuse_preparation = False
        self.phase_profile = None
        self.kv_cache_seconds = {}
        self.kv_cache_peak_stored_bytes = 0
        self.kv_cache_peak_raw_bytes = 0
        self.kv_cache_compression = kv_cache_compression or ("zstd lossless" if TOGGLE_TAOMATE_DIVERGENCY_COMPRESS_KV else "none")
        if self.kv_cache_compression not in KV_CACHE_COMPRESSION_MODES:
            raise ValueError("unknown KV cache compression mode %r" % self.kv_cache_compression)

    @staticmethod
    def validate(guider):
        """Check the model interface used by the native H3 adapter."""
        model = guider.model_patcher.get_model_object("diffusion_model")
        if not isinstance(model, h3.MiniMaxH3Model):
            raise ValueError("TaoMate-H3 requires ComfyUI's native MiniMax H3 model")

    def patch_guider(self, guider):
        """Clone the patcher and install attention without editing ComfyUI files."""
        self.validate(guider)
        logging.info("TaoMate TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV=%s", TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV)
        logging.info("TaoMate TOGGLE_TAOMATE_DIVERGENCY_MOVE_PROMPT_REFERENCES=%s", TOGGLE_TAOMATE_DIVERGENCY_MOVE_PROMPT_REFERENCES)
        logging.info("TaoMate KV cache compression=%s", self.kv_cache_compression)
        logging.info("TaoMate TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV=%s; CPU codec workers=%d", TOGGLE_TAOMATE_DIVERGENCY_GPU_DECOMPRESS_KV, TAOMATE_KV_CPU_WORKERS)
        from .taomate_audio_teacher import Base10AudioTeacher
        self.audio_teacher = Base10AudioTeacher(guider.model_patcher)
        # Leave other preparation wrappers in control of their own lifecycle.
        self.reuse_preparation = not guider.model_patcher.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING)
        result = copy.copy(guider)
        result.model_patcher = guider.model_patcher.clone()
        result.model_options = result.model_patcher.model_options
        model = result.model_patcher.get_model_object("diffusion_model")
        self.layer_count = len(model.blocks)
        for index, block in enumerate(model.blocks):
            result.model_patcher.add_object_patch("diffusion_model.blocks.%d.attn.forward" % index, self.attention_forward(block.attn, index))
        result.model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "hr_taomate_streaming", self.forward)
        result.model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, "hr_taomate_preparation", self.prepare_sampling)
        return result

    def _record_kv_profile(self):
        """Add this phase's cache-only wall times to the render summary."""
        for name, values in (self.phase_profile or {}).items():
            if name.startswith("KV "):
                self.kv_cache_seconds[name] = self.kv_cache_seconds.get(name, 0.0) + values[0]

    def kv_cache_report(self):
        """Return retained cache bytes and accumulated cache operation time."""
        sizes = [cache.storage_bytes() for cache, hook in self.caches.values()]
        stored = sum(size[0] for size in sizes)
        raw = sum(size[1] for size in sizes)
        return {
            "compression": self.kv_cache_compression,
            "seconds": dict(self.kv_cache_seconds),
            "stored_bytes": stored,
            "raw_bytes": raw,
            "peak_stored_bytes": self.kv_cache_peak_stored_bytes,
            "peak_raw_bytes": self.kv_cache_peak_raw_bytes,
        }

    def prepare_sampling(self, executor, model, noise_shape, conds, model_options=None, force_full_load=False, force_offload=False):
        """Reuse resident weights between phases without repeating native loading."""
        # ponytail: only plain single-device H3; controls/hooks retain native setup.
        simple = self.reuse_preparation and not force_full_load and not force_offload
        simple = simple and not (model_options or {}).get("multigpu_clones")
        simple = simple and not any(cond.get("control") is not None or cond.get("hooks") is not None for values in conds.values() for cond in values)
        required = comfy.sampler_helpers.estimate_memory(model, noise_shape, conds) if simple else None
        resident = any(loaded.model is model for loaded in comfy.model_management.current_loaded_models)
        cached = self.prepared_model
        if simple and resident and cached is not None and cached[0] is model and all(now <= before for now, before in zip(required, cached[1])):
            # The current phase must keep its own conditioning and latent shapes.
            return model.model, conds, []
        result = executor(model, noise_shape, conds, model_options=model_options, force_full_load=force_full_load, force_offload=force_offload)
        self.prepared_model = (model, required) if simple and not result[2] else None
        return result

    def forward(self, executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        """Replace local RoPE positions with a single global media timeline."""
        video, audio = x[:2]
        if video.shape[0] != 1:
            raise ValueError("TaoMate-H3 supports one AV sample per render")
        payload = dict(minimax_payload or {})
        self.branch = tuple(transformer_options.get("cond_or_uncond", ()))
        text_t = context.shape[1]
        height, width = ((video.shape[-2] + 1) // 2 * 2, (video.shape[-1] + 1) // 2 * 2)
        signature = (text_t, video.shape[2], height, width, audio.shape[-1])
        # Rebuild from native conditioning; equal target shapes can have different references.
        self.layout = h3.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        positions = self.layout.position_ids
        target_begin = next(begin for begin, end, kind in self.layout.segments if kind == "audio")
        local_origin = float(positions[target_begin, 0])
        if self.origin is None:
            self.origin = local_origin
        # Text and reference positions stay fixed throughout a request.
        prompt_start = self.video_start if self.request_video_start is None else self.request_video_start
        positions[:, 0] += self.origin + float(video_temporal_position(prompt_start)) - local_origin
        for begin, end, kind in self.layout.segments:
            # Remove only the group offset from references/text; timed keyframes keep it.
            if not TOGGLE_TAOMATE_DIVERGENCY_MOVE_PROMPT_REFERENCES and kind in ("text", "ref_img", "ref_audio"):
                positions[begin:end, 0] -= float(video_temporal_position(prompt_start))
            if kind == "video":
                times = torch.tensor([float(video_temporal_position(self.video_start + index)) + self.origin for index in range(video.shape[2])], dtype=positions.dtype)
                positions[begin:end, 0] = times.repeat_interleave(height * width // 4)
            elif kind == "audio":
                positions[begin:end, 0] = (torch.arange(audio.shape[-1], dtype=positions.dtype) + self.audio_start + self.origin).repeat(2)
        payload["layout"] = self.layout
        self.video_shape = tuple(video.shape)
        diagnostic = getattr(self, "diagnostic", None)
        if diagnostic is not None:
            diagnostic.capture(x, timestep, context, transformer_options, payload)
        return executor(x, timestep, context, transformer_options, minimax_payload=payload, **kwargs)

    def attention_forward(self, attention, layer):
        """Build an execution-local H3 attention replacement for one layer."""
        def forward(x, rope_freqs=None, transformer_options=None):
            """Conditions attend conditions; targets also attend retained AV history."""
            if not torch.is_tensor(x):
                raise ValueError("TaoMate-H3 cannot consume a competing low-memory attention patch")
            count = x.shape[0]
            q, k, v = attention.qkv_proj(x).split(attention.heads * attention.head_dim, dim=-1)
            v = v.reshape(count, attention.heads, attention.head_dim)
            if rope_freqs is not None:
                q = q.reshape(1, count, attention.heads, attention.head_dim)
                k = k.reshape(1, count, attention.heads, attention.head_dim)
                qw = comfy.model_management.cast_to(attention.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attention.k_norm.weight, device=x.device)
                q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
                q, k = q[0], k[0]
            else:
                q = attention.q_norm(q.reshape(count, attention.heads, attention.head_dim))
                k = attention.k_norm(k.reshape(count, attention.heads, attention.head_dim))
            # CFG branches need independent history and clean commits.
            if self.branch not in self.caches:
                cache = CPUStreamingCache(KVContract(local_heads=attention.heads, head_dim=attention.head_dim, dtype=k.dtype, device_type="cpu"), self.kv_cache_compression)
                self.caches[self.branch] = (cache, ComfyStreamingHook(cache))
            self.cache, self.hook = self.caches[self.branch]
            self.cache.on_status = self.report_status
            self.cache.profile = self.phase_profile
            if self.clean:
                label = "uncompressed" if self.kv_cache_compression == "none" else self.kv_cache_compression
                self.report_status("H3 clean pass / staging %s KV: layer %d/%d" % (label, layer + 1, len(MAIN_LAYER_NAMES)))
            if self.clean and not self.cache.clean_commit_active:
                self.cache.begin_clean_commit(self.cache.committed_blocks)
            tags = torch.empty(count, dtype=torch.long, device=x.device)
            commit_mask = torch.zeros(count, dtype=torch.bool, device=x.device)
            for begin, end, kind in self.layout.segments:
                tags[begin:end] = {"video": 0, "audio": 2}.get(kind, 1)
                commit_mask[begin:end] = kind in ("video", "audio")
            diagnostic = getattr(self, "diagnostic", None)
            if diagnostic is not None and layer == 0:
                diagnostic.routing(self.branch, tags, commit_mask, self.cache.history_tokens, TOGGLE_TAOMATE_DIVERGENCY_CONDITION_ATTENDS_CURRENT_AV)
            self.hook.activate(HookMode.CLEAN_COMMIT if self.clean else HookMode.NOISY)
            try:
                output = self.hook(attention=type("AttentionScale", (), {"softmax_scale": attention.head_dim ** -0.5})(), layer_name="blocks.%d.attn" % layer, query=q, key=k, value=v, token_tags=tags, commit_mask=commit_mask, cu_seqlens_host=(0, count))
            finally:
                self.hook.deactivate()
            return attention.out_proj(output.reshape(count, -1))
        return forward

    def renormalize(self, packed):
        """Match patch-feature statistics to the first clean video phase."""
        shape = self.video_shape
        count = math.prod(shape)
        video = packed[..., :count].reshape(shape)
        # Match native H3's spatial padding and crop after restoring patch rows.
        padded = comfy.ldm.common_dit.pad_to_patch_size(video.float(), (1, 2, 2))
        rows = h3.patchify_video(padded)
        rows = self._renorm_clean_video_rows(rows)
        result = packed.clone()
        restored = h3.unpatchify_video(rows, shape[2], padded.shape[3] // 2, padded.shape[4] // 2)
        result[..., :count] = restored[..., :shape[3], :shape[4]].reshape(1, -1)
        return result

    def _renorm_clean_video_rows(self, rows):
        """Match generated blocks to persistent block-zero statistics."""

        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.anchor is None:
            self.anchor = (mean, std)
            return rows
        anchor_mean, anchor_std = self.anchor
        normalized = (current - mean).div(std).mul(anchor_std).add(anchor_mean)
        return normalized.to(dtype=rows.dtype)

    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=None, sampler_function=sample_euler, **sampler_options):
        """Follow supplied sigmas, renormalize, then capture clean attention history."""
        guided = model if self.teacher_milestones is None else TeacherGuidedModel(model, self, sigmas, x)
        # Invoke the selected solver once: splitting intervals would reset multistep history.
        started = (time.perf_counter(), time.process_time())
        result = sampler_function(guided, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **sampler_options)
        _profile_add(self.phase_profile, "denoise solver", started)
        if self.teacher_milestones is not None:
            result = result.clone()
            result[..., math.prod(self.video_shape):] = guided.audio_at(len(sigmas) - 1, result)
        self.report_status("TaoMate: matching clean video statistics")
        started = (time.perf_counter(), time.process_time())
        result = self.renormalize(result)
        _profile_add(self.phase_profile, "video renormalization", started)
        self.clean = True
        try:
            self.report_status("H3 clean pass: preparing KV cache")
            started = (time.perf_counter(), time.process_time())
            model(result, result.new_zeros([result.shape[0]]), **(extra_args or {}))
            _profile_add(self.phase_profile, "clean H3 pass", started)
            for cache, hook in self.caches.values():
                if cache.clean_commit_active:
                    cache.commit()
                    self.report_status("KV cache: checking anchor and recent history retention")
                    cache.retain_sink_and_recent_commits()
            sizes = [cache.storage_bytes() for cache, hook in self.caches.values()]
            stored, raw = sum(size[0] for size in sizes), sum(size[1] for size in sizes)
            self.kv_cache_peak_stored_bytes = max(self.kv_cache_peak_stored_bytes, stored)
            self.kv_cache_peak_raw_bytes = max(self.kv_cache_peak_raw_bytes, raw)
            logging.info("TaoMate retained CPU KV: %.3f GiB stored / %.3f GiB raw, %.1f%% saved across %d branches. Staging and model weights are additional.", stored / 1024 ** 3, raw / 1024 ** 3, 100 * (1 - stored / raw) if raw else 0, len(self.caches))
            self.report_status("KV cache ready: %.2f GiB stored / %.2f GiB raw" % (stored / 1024 ** 3, raw / 1024 ** 3))
        finally:
            self.clean = False
            for cache, hook in self.caches.values():
                if cache.clean_commit_active:
                    cache.rollback()
        return result

    def begin_chunk(self, chunk):
        """Start the next phase at its actual global AV position."""
        self.video_start = chunk["video_start"]
        self.audio_start = chunk["audio_start"]
        self.layout = None

    @staticmethod
    def decode_video_timeline(decode, segments, expected_frames):
        """Decode assembled new-media latents once, without group transport halos."""
        frames = decode(torch.cat(segments, dim=2)).detach().to(device="cpu", dtype=torch.float32)
        if int(frames.shape[0]) != int(expected_frames):
            raise ValueError("TaoMate continuous video decode returned %d frames; expected %d" % (frames.shape[0], expected_frames))
        return frames

    @staticmethod
    def decode_audio_timeline(audio_vae, segments):
        """Mirror upstream publication: join clean latents, then decode once."""
        # No per-group waveform normalization or trimming: both create new seams.
        latent = torch.cat(segments, dim=-1)
        waveform = audio_vae.decode(latent).movedim(-1, 1)
        sample_rate = int(getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000)))
        return waveform.detach().to(device="cpu", dtype=torch.float32), sample_rate

    def execute_chunk(self, execute_sampler, noise_factory, seed, noise, guider, sigmas, latent, chunk, on_subchunk=None, sampler=None, on_subchunk_start=None, on_status=None, debug_timing=False):
        """Sample only new media; prepend historical tokens for VAE decoding only."""
        self.on_status = on_status
        video, audio = latent["samples"].unbind()
        vn, an = noise.unbind()
        vt, at = chunk["context_video_t"], chunk["context_audio_t"]
        self.request_video_start = chunk["video_start"]
        if self.request_count and self.request_count % 12 == 0:
            for cache, hook in self.caches.values():
                cache.drop_audio_history()
        if self.audio_teacher is not None:
            self.report_status("TaoMate: generating teacher audio for this chunk")
            positive = guider.original_conds["positive"][0]
            sampling = guider.model_patcher.get_model_object("model_sampling")
            video_shift = float(sampling.shift)
            audio_shift = float(sampling.audio_shift if sampling.audio_shift is not None else sampling.shift)
            self.teacher_milestones = self.audio_teacher.generate(positive, an[..., at:], video.shape[-2], video.shape[-1], sigmas, video_shift, audio_shift)
            self.audio_scale = float(sampling.audio_scale)
        self.prepared_model = None
        output_video, output_audio = [], []
        # All phases share the same guider conditioning, encoded once per request.
        phases = chunk.get("phases", (chunk,))
        with tqdm(total=len(phases), desc="TaoMate group %d: video sub-chunks" % (self.request_count + 1), unit="sub-chunk", leave=True, disable=not comfy.utils.PROGRESS_BAR_ENABLED) as progress:
            for phase_number, phase in enumerate(phases, 1):
                logging.info("TaoMate group %d: video sub-chunk %d/%d", self.request_count + 1, phase_number, len(phases))
                phase_started = (time.perf_counter(), time.process_time())
                # Always retain cache timings for the final render report; debug
                # only controls the per-phase console detail.
                self.phase_profile = {}
                self.begin_chunk(phase)
                if on_subchunk_start is not None:
                    on_subchunk_start(phase)
                v0 = vt + phase["video_start"] - chunk["video_start"]
                v1 = vt + phase["video_end"] - chunk["video_start"]
                a0 = at + phase["audio_start"] - chunk["audio_start"]
                a1 = at + phase["audio_end"] - chunk["audio_start"]
                self.phase_audio_start = a0 - at
                self.phase_audio_end = a1 - at
                target = dict(latent)
                target["samples"] = comfy.nested_tensor.NestedTensor((video[:, :, v0:v1], audio[..., a0:a1]))
                # Guidance splits packed AV before the first model forward; set each phase's shape now.
                self.video_shape = tuple(video[:, :, v0:v1].shape)
                target_noise = comfy.nested_tensor.NestedTensor((vn[:, :, v0:v1], an[..., a0:a1]))
                selected = self.wrap_sampler(sampler) if sampler is not None else comfy.samplers.KSAMPLER(self.sample)
                sampler_started = (time.perf_counter(), time.process_time())
                result, _ = execute_sampler(noise_factory(seed, target_noise), guider, selected, sigmas, target)
                _profile_add(self.phase_profile, "Comfy sampler total", sampler_started)
                generated_video, generated_audio = result["samples"].unbind()
                if self.teacher_milestones is not None:
                    expected = self.teacher_milestones[-1][..., self.phase_audio_start:self.phase_audio_end].to(generated_audio)
                    # Remove only the known roundoff from native audio carry/un-carry.
                    roundtrip = (expected * self.audio_scale) * (1.0 / self.audio_scale)
                    if not torch.equal(generated_audio, expected) and not torch.equal(generated_audio, roundtrip):
                        raise RuntimeError("Published TaoMate audio differs from the teacher clean audio latent")
                    generated_audio = expected.clone()
                output_video.append(generated_video)
                output_audio.append(generated_audio)
                progress.update(1)
                if on_subchunk is not None:
                    callback_started = (time.perf_counter(), time.process_time())
                    on_subchunk(phase["frame_end"] - self.frames(chunk["video_start"]))
                    _profile_add(self.phase_profile, "preview publication", callback_started)
                self._record_kv_profile()
                if debug_timing:
                    total_wall = time.perf_counter() - phase_started[0]
                    total_cpu = time.process_time() - phase_started[1]
                    detail = "; ".join("%s %.3fs wall / %.3fs CPU" % (name, wall, cpu) for name, (wall, cpu) in self.phase_profile.items())
                    logging.info("TaoMate profile group %d sub-chunk %d/%d: total %.3fs wall / %.3fs CPU; %s", self.request_count + 1, phase_number, len(phases), total_wall, total_cpu, detail or "no measured operations")
        self.report_status("TaoMate: assembling sub-chunk video/audio latents")
        generated_video = torch.cat(output_video, dim=2)
        generated_audio = torch.cat(output_audio, dim=-1)
        if self.teacher_milestones is not None:
            expected = self.teacher_milestones[-1].to(generated_audio)
            if not torch.equal(generated_audio, expected):
                raise RuntimeError("Published TaoMate audio differs from the teacher clean audio latent")
            self.audio_teacher.receipt["published_clean_audio_exact_match"] = True
            logging.info("TaoMate teacher clean audio exact match: True")
            self.teacher_milestones = None
        result["samples"] = comfy.nested_tensor.NestedTensor((torch.cat((video[:, :, :vt].to(generated_video), generated_video), dim=2), torch.cat((audio[..., :at].to(generated_audio), generated_audio), dim=-1)))
        self.request_count += 1
        self.prepared_model = None
        return result, dict(result)

    def report_status(self, message):
        """Forward backend stages to the current preview execution."""
        callback = getattr(self, "on_status", None)
        if callback is not None:
            callback(message)

    def wrap_sampler(self, sampler):
        """Retain the input sampler and options, adding guidance and clean KV capture."""
        if not hasattr(sampler, "sampler_function"):
            raise TypeError("TaoMate requires a ComfyUI sampler exposing sampler_function; the supplied custom sampler cannot be wrapped.")
        selected = copy.copy(sampler)
        original = sampler.sampler_function

        def sample(*args, **kwargs):
            """Delegate the full schedule to the original solver without restarting it."""
            return self.sample(*args, sampler_function=original, **kwargs)

        selected.sampler_function = sample
        return selected

    def close(self):
        """Release potentially large CPU attention histories after any render exit."""
        for cache, hook in self.caches.values():
            cache.clear()
        self.caches.clear()
        self.prepared_model = None
        self.teacher_milestones = None
        if self.audio_teacher is not None:
            self.audio_teacher.close()
        self.anchor = None
        self.layout = None
