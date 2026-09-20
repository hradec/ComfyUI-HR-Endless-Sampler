"""ComfyUI base-H3 audio adapter for the copied TaoMate Base10 denoising loop."""

import logging
from types import SimpleNamespace

import torch
import comfy.model_management
import comfy.model_prefetch
import comfy.ldm.minimax.model as h3

from .taomate_upstream.denoise import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .taomate_upstream.denoise_schedule import select_time_shift_sigmas
from .taomate_upstream.packed_sequence import minimax_h3_audio_only_packed_sequence, minimax_h3_audio_only_frozen_prefix_packed_sequence


class Base10AudioTeacher:
    """Generate request-wide audio states with the supplied LoRA and no video tokens."""

    def __init__(self, model_patcher):
        """Share the supplied model and LoRA; isolate streaming attention patches."""
        self.patcher = model_patcher.clone()
        self.previous_clean = None
        self.previous_count = None
        self.branch = None
        self.model = None
        self.options = None
        self.receipt = None

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
    def generate(self, conditioning, noise, height, width, sigmas=None, video_shift=12.0, audio_shift=3.0):
        """Capture exact student sigma endpoints, keeping the previous tail frozen."""
        video_sigmas, audio_sigmas, capture_steps = self.guidance_schedule(sigmas, video_shift, audio_shift)
        device = self.patcher.load_device
        logging.info("TaoMate audio teacher: %d audio-only forwards, %d guidance states, shifts video=%g audio=%g; supplied model and LoRA.", len(video_sigmas) - 1, len(capture_steps), video_shift, audio_shift)
        comfy.model_management.load_models_gpu([self.patcher])
        self.patcher.pre_run()
        try:
            self.model = self.patcher.get_model_object("diffusion_model")
            self.options = self.patcher.model_options.get("transformer_options", {}).copy()
            dtype = self.patcher.model.get_dtype_inference()
            context = conditioning["cross_attn"].to(device=device, dtype=dtype)
            context = self.model.preprocess_text_embeds(context)[0]
            count, text_len = noise.shape[-1], context.shape[0]
            height, width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
            if self.previous_clean is None:
                packed = minimax_h3_audio_only_packed_sequence(text_len=text_len, audio_t=count, latent_h=height, latent_w=width)
                initial = h3.pack_audio(noise.float())
            else:
                tail = self.previous_clean[..., -40:]
                tail_count = tail.shape[-1]
                packed = minimax_h3_audio_only_frozen_prefix_packed_sequence(text_len=text_len, ref_audio_t=tail_count, audio_t=count, latent_h=height, latent_w=width, reference_time_start=text_len + self.previous_count - tail_count, target_time_start=text_len + self.previous_count)
                initial = torch.cat((h3.pack_audio(tail), h3.pack_audio(noise.float())), dim=0)
            tags = conditioning.get("minimax_token_tags")
            if tags is not None:
                packed["token_tags"][packed["text_pos"]] = tags.detach().cpu().view(-1)
            self.branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=context, token_tags=packed["token_tags"], device=device, parallel_context=SimpleNamespace(ulysses_world_size=1, ulysses_rank=0))
            captured = {}

            def capture(step, video, audio):
                """Retain teacher states at the requested student step endpoints."""
                if step + 1 in capture_steps:
                    captured[step + 1] = h3.unpack_audio(audio[self.branch.audio_target_slice]).detach().float().cpu().clone()

            minimax_h3_denoise_loop(model=self.forward, positive=self.branch, initial_video_rows=torch.empty(0, 96), initial_audio_rows=initial, sigmas_video=video_sigmas, sigmas_audio=audio_sigmas, device=device, on_step=capture)
            milestones = [captured[number] for number in capture_steps]
            if not all(torch.isfinite(value).all() for value in milestones):
                raise RuntimeError("TaoMate Base10 teacher produced non-finite audio")
            self.receipt = {"executed_forwards": len(video_sigmas) - 1, "states": capture_steps, "video_sigmas": video_sigmas, "audio_sigmas": audio_sigmas, "reference_ticks": 0 if self.previous_clean is None else min(40, self.previous_count), "published_clean_audio_exact_match": False}
            self.previous_clean = milestones[-1]
            self.previous_count = count
            logging.info("TaoMate Base10 audio teacher prepared: %s", self.receipt)
            return milestones
        finally:
            self.branch = None
            self.model = None
            self.options = None
            self.patcher.cleanup()

    @staticmethod
    def guidance_schedule(sigmas, video_shift, audio_shift):
        """Add requested endpoints to the upstream teacher grid without interpolation."""
        video = select_time_shift_sigmas(num_steps=10, shift_scale=video_shift)
        base_audio = select_time_shift_sigmas(num_steps=10, shift_scale=audio_shift)
        if sigmas is None:
            return video, base_audio, [3, 6, 9]
        requested = sigmas.detach().float().cpu().tolist()
        if len(requested) < 2 or requested[0] != 1.0 or requested[-1] != 0.0 or any(not (a > b >= 0.0) for a, b in zip(requested, requested[1:])):
            raise ValueError("TaoMate audio guidance requires descending finite sigmas from 1 to 0 for noise initialization and clean KV capture.")
        # Keep the teacher's finer grid and insert exact student endpoints.
        video = sorted(set(video + requested), reverse=True)
        ratio = audio_shift / video_shift
        audio = [ratio * sigma / (1.0 + (ratio - 1.0) * sigma) for sigma in video]
        return video, audio, [video.index(sigma) for sigma in requested[1:]]

    def close(self):
        """Release request audio retained for teacher rollover."""
        self.previous_clean = None
        self.branch = None
