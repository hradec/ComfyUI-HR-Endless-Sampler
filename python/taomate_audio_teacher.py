"""ComfyUI base-H3 audio adapter for the copied TaoMate Base10 denoising loop."""

import logging
import time
from types import SimpleNamespace

import torch
import comfy.model_management
import comfy.model_prefetch
import comfy.ldm.minimax.model as h3

from .taomate_upstream.denoise import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .taomate_upstream.denoise_schedule import select_time_shift_sigmas
from .taomate_upstream.packed_sequence import minimax_h3_audio_only_packed_sequence
from .taomate_upstream.attention_hook import HookMode

# Experiment: anchor clean teacher-audio feature statistics to its first chunk.
TOGGLE_TAOMATE_DIVERGENCY_RENORM_TEACHER_AUDIO = False


class Base10AudioTeacher:
    """Generate request-wide audio states with the supplied LoRA and no video tokens."""

    def __init__(self, model_patcher, cache_factory=None, hook_factory=None):
        """Share the supplied model and LoRA; isolate streaming attention patches."""
        self.patcher = model_patcher.clone()
        self.cache_factory = cache_factory
        self.hook_factory = hook_factory
        self.cache = None
        self.hook = None
        self.clean_cache_pass = False
        self.audio_time_origin = None
        self.branch = None
        self.model = None
        self.options = None
        self.receipt = None
        self.audio_anchor = None

    def _renorm_clean_audio_rows(self, rows):
        """Match packed audio feature statistics to the first teacher chunk."""
        if not TOGGLE_TAOMATE_DIVERGENCY_RENORM_TEACHER_AUDIO:
            return rows
        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.audio_anchor is None:
            self.audio_anchor = (mean.cpu(), std.cpu())
            return rows
        # Keep only small feature statistics between chunks, never a GPU latent.
        anchor_mean, anchor_std = self.audio_anchor
        return ((current - mean).div(std).mul(anchor_std.to(current)).add(anchor_mean.to(current))).to(rows.dtype)

    def attention_forward(self, attention, layer):
        """Build the teacher's audio-only KV attention adapter for one H3 layer."""
        def forward(x, rope_freqs=None, transformer_options=None):
            """Let new audio attend text, its clean teacher history and current rows."""
            count = x.shape[0]
            q, k, v = attention.qkv_proj(x).split(attention.heads * attention.head_dim, dim=-1)
            v = v.reshape(count, attention.heads, attention.head_dim)
            q = q.reshape(1, count, attention.heads, attention.head_dim)
            k = k.reshape(1, count, attention.heads, attention.head_dim)
            qw = comfy.model_management.cast_to(attention.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attention.k_norm.weight, device=x.device)
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
            q, k = q[0], k[0]
            if self.cache is None:
                if self.cache_factory is None or self.hook_factory is None:
                    raise RuntimeError("teacher audio KV cache requires cache and attention-hook factories")
                self.cache = self.cache_factory(attention.heads, attention.head_dim, k.dtype)
                self.hook = self.hook_factory(self.cache)
            self.cache.on_status = None
            if self.clean_cache_pass and not self.cache.clean_commit_active:
                self.cache.begin_clean_commit(self.cache.committed_blocks)
            tags = self.branch.block_token_tags[:count]
            commit_mask = torch.zeros(count, dtype=torch.bool, device=x.device)
            commit_mask.index_fill_(0, self.branch.audio_target_seq_idx, True)
            self.hook.activate(HookMode.CLEAN_COMMIT if self.clean_cache_pass else HookMode.NOISY)
            try:
                output = self.hook(attention=type("AttentionScale", (), {"softmax_scale": attention.head_dim ** -0.5})(), layer_name="blocks.%d.attn" % layer, query=q, key=k, value=v, token_tags=tags, commit_mask=commit_mask, cu_seqlens_host=(0, count))
            finally:
                self.hook.deactivate()
            return attention.out_proj(output.reshape(count, -1))
        return forward

    def _install_cache_attention(self):
        """Temporarily replace attention only while this isolated teacher owns the model."""
        if self.cache_factory is None:
            return []
        originals = []
        for layer, block in enumerate(self.model.blocks):
            originals.append((block.attn, block.attn.forward))
            block.attn.forward = self.attention_forward(block.attn, layer)
        return originals

    @staticmethod
    def _restore_attention(originals):
        """Restore the shared model modules after one teacher operation."""
        for attention, forward in originals:
            attention.forward = forward

    def _commit_clean_audio_cache(self, audio_rows):
        """Capture target-only KV from one clean teacher-audio forward at sigma zero."""
        if self.cache is None:
            return
        clean_timestep = self.branch.prepare_timestep_plan(video_timesteps=[1.0], audio_timesteps=[1.0])[0]
        self.clean_cache_pass = True
        try:
            self.forward(**self.branch.forward_kwargs(video_rows=torch.empty(0, 96, device=audio_rows.device), audio_rows=audio_rows, step_timesteps=clean_timestep))
            if self.cache.clean_commit_active:
                self.cache.commit()
        finally:
            self.clean_cache_pass = False
            if self.cache.clean_commit_active:
                self.cache.rollback()

    def forward(self, **kwargs):
        """Run native H3 audio/text layers against upstream packed-row metadata."""
        model, branch = self.model, self.branch
        device = kwargs["audio_x"].device
        dtype = self.patcher.model.get_dtype_inference()
        # Upstream isolates padding in a separate attention sequence; omit it here.
        used = int(branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"][1])
        text = kwargs["prompt_embeds"]
        rows = kwargs["audio_x"][0].index_select(0, branch.audio_pos_dev)
        hidden = torch.cat((text, model.audio_patch_proj(rows).to(dtype)), dim=0)
        times = kwargs["unique_timesteps"]
        if model.use_adaln_curves:
            table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
            pos = times.clamp(0.0, 1.0) * (table.shape[0] - 1)
            start = pos.floor().long().clamp(max=table.shape[0] - 2)
            embedding = torch.lerp(table[start], table[start + 1], (pos - start).unsqueeze(1))
        else:
            embedding = model.time_embedder(times).to(dtype)
        # Use upstream modality/timestep assignments, including the clean tail.
        combined = kwargs["block_combined_indices"][:used].tolist()
        segments = []
        start = 0
        for end in range(1, used + 1):
            if end == used or combined[end] != combined[start]:
                segments.append((start, end, combined[start]))
                start = end
        rope = h3.rope_rotation_table(model.rope_freqs(branch.img_position_ids[0, :used], device), dtype)
        queue = comfy.model_prefetch.make_prefetch_queue(list(model.blocks), device, self.options)
        for block in model.blocks:
            comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, device, block)
            hidden = block(hidden, embedding, segments, rope, transformer_options=self.options)
        if queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
        # Only audio targets use the output head; no video projection/head executes.
        final = model.final_layer
        shift, scale = final.adaln_proj(embedding)
        indices = branch.audio_target_seq_idx
        time_rows = kwargs["inverse_indices"].index_select(0, indices)
        target = final.norm(hidden.index_select(0, indices))
        target = (target * (1.0 + scale[time_rows]) + shift[time_rows]).float()
        velocity = final.audio_out(target)
        output = torch.zeros_like(rows)
        output[branch.audio_target_slice] = velocity
        return rows.new_empty((0, 96)), output

    @torch.inference_mode()
    def generate(self, conditioning, noise, height, width, sigmas=None, video_shift=12.0, audio_shift=3.0, audio_time_start=0, progress_callback=None):
        """Capture exact student sigma endpoints using persistent clean audio KV."""
        video_sigmas, audio_sigmas, capture_steps = self.guidance_schedule(sigmas, video_shift, audio_shift)
        device = self.patcher.load_device
        logging.info("TaoMate audio teacher: %d audio-only forwards, %d guidance states, shifts video=%g audio=%g; supplied model and LoRA.", len(video_sigmas) - 1, len(capture_steps), video_shift, audio_shift)
        # The previous video phase can leave dynamic H3 weights resident on
        # the shared patcher. Drop those clones before loading the teacher so
        # its temporary dequantized weight buffers have maximum headroom.
        comfy.model_management.unload_model_and_clones(self.patcher)
        comfy.model_management.load_models_gpu([self.patcher])
        self.patcher.pre_run()
        originals = []
        try:
            self.model = self.patcher.get_model_object("diffusion_model")
            originals = self._install_cache_attention()
            self.options = self.patcher.model_options.get("transformer_options", {}).copy()
            dtype = self.patcher.model.get_dtype_inference()
            context = conditioning["cross_attn"].to(device=device, dtype=dtype)
            context = self.model.preprocess_text_embeds(context)[0]
            count, text_len = noise.shape[-1], context.shape[0]
            height, width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
            # Preserve H3's native first audio coordinate, then advance only by
            # the sampler's global audio timeline. Later prompt lengths cannot
            # move media relative to committed teacher KV.
            if self.audio_time_origin is None:
                self.audio_time_origin = text_len
            packed = minimax_h3_audio_only_packed_sequence(text_len=text_len, audio_t=count, latent_h=height, latent_w=width, time_start=audio_time_start, audio_time_origin=self.audio_time_origin)
            initial = h3.pack_audio(noise.float())
            tags = conditioning.get("minimax_token_tags")
            if tags is not None:
                packed["token_tags"][packed["text_pos"]] = tags.detach().cpu().view(-1)
            self.branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=context, token_tags=packed["token_tags"], device=device, parallel_context=SimpleNamespace(ulysses_world_size=1, ulysses_rank=0))
            captured = {}
            inference_started = time.perf_counter()

            def capture(step, video, audio):
                """Retain teacher states at the requested student step endpoints."""
                if step + 1 in capture_steps:
                    captured[step + 1] = h3.unpack_audio(audio[self.branch.audio_target_slice]).detach().float().cpu().clone()
                if progress_callback is not None:
                    self.average_step_ms = (time.perf_counter() - inference_started) * 1000.0 / (step + 1)
                    progress_callback(step + 1, len(video_sigmas) - 1)

            _, clean_audio_rows = minimax_h3_denoise_loop(model=self.forward, positive=self.branch, initial_video_rows=torch.empty(0, 96), initial_audio_rows=initial, sigmas_video=video_sigmas, sigmas_audio=audio_sigmas, device=device, on_step=capture)
            # Correct clean targets before both KV capture and clean guidance publication.
            target_slice = self.branch.audio_target_slice
            clean_audio_rows[target_slice] = self._renorm_clean_audio_rows(clean_audio_rows[target_slice])
            final_step = len(video_sigmas) - 1
            if final_step in captured:
                captured[final_step] = h3.unpack_audio(clean_audio_rows[target_slice]).detach().float().cpu().clone()
            milestones = [captured[number] for number in capture_steps]
            if not all(torch.isfinite(value).all() for value in milestones):
                raise RuntimeError("TaoMate Base10 teacher produced non-finite audio")
            self._commit_clean_audio_cache(clean_audio_rows)
            self.receipt = {"executed_forwards": len(video_sigmas) - 1, "states": capture_steps, "video_sigmas": video_sigmas, "audio_sigmas": audio_sigmas, "reference_ticks": 0, "audio_kv_ticks": 0 if self.cache is None else self.cache.history_tokens // 2, "audio_time_start": audio_time_start, "audio_time_end": audio_time_start + count, "published_clean_audio_exact_match": False}
            logging.info("TaoMate Base10 audio teacher prepared: %s", self.receipt)
            return milestones
        finally:
            self._restore_attention(originals)
            self.branch = None
            self.model = None
            self.options = None
            self.patcher.cleanup()

    @staticmethod
    def guidance_schedule(sigmas, video_shift, audio_shift):
        """Follow the video sampler step count using the matching audio shift."""
        if sigmas is None:
            video = select_time_shift_sigmas(num_steps=10, shift_scale=video_shift)
            base_audio = select_time_shift_sigmas(num_steps=10, shift_scale=audio_shift)
            return video, base_audio, [3, 6, 9]
        requested = sigmas.detach().float().cpu().tolist()
        if len(requested) < 2 or requested[0] != 1.0 or requested[-1] != 0.0 or any(not (a > b >= 0.0) for a, b in zip(requested, requested[1:])):
            raise ValueError("TaoMate audio guidance requires descending finite sigmas from 1 to 0 for noise initialization and clean KV capture.")
        # Reuse each video sigma exactly; only map its value to audio's shift.
        video = requested
        ratio = audio_shift / video_shift
        audio = [ratio * sigma / (1.0 + (ratio - 1.0) * sigma) for sigma in video]
        return video, audio, list(range(1, len(video)))

    def close(self):
        """Release request-local teacher KV."""
        self.audio_time_origin = None
        self.audio_anchor = None
        if self.cache is not None:
            self.cache.clear()
        self.cache = None
        self.hook = None
        self.branch = None
