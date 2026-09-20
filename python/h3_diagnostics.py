"""First-step H3 reference diagnostics; CPU snapshots without changing sampling."""

import hashlib
import json
import logging
import os
import tempfile

import torch
import comfy.ldm.minimax.model as h3


class H3FirstStepDiagnostic:
    """Save one model input and routing snapshot per conditioning branch."""

    def __init__(self, mode, inputs):
        """Give every execution a unique directory so two modes remain comparable."""
        os.makedirs("/tmp/codex", exist_ok=True)
        self.directory = tempfile.mkdtemp(prefix="h3-first-step-", dir="/tmp/codex")
        self.mode = mode
        self.seen = set()
        self.routing_seen = set()
        self.save("sampler-input", dict(inputs, mode=mode))
        logging.info("H3 first-step diagnostic (%s): %s", mode, self.directory)

    @classmethod
    def cpu_copy(cls, value):
        """Keep tensors and simple metadata; avoid serializing models or callables."""
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {str(key): cls.cpu_copy(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.cpu_copy(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"object_type": type(value).__name__}

    @classmethod
    def summary(cls, value):
        """Hash exact tensor bytes for comparison without console tensor dumps."""
        if torch.is_tensor(value):
            raw = value.contiguous().reshape(-1).view(torch.uint8).numpy()
            return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(value, dict):
            return {key: cls.summary(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.summary(item) for item in value]
        return value

    def save(self, name, value):
        """Store exact CPU data and a small human-readable companion file."""
        value = self.cpu_copy(value)
        torch.save(value, os.path.join(self.directory, name + ".pt"))
        with open(os.path.join(self.directory, name + ".json"), "w") as handle:
            json.dump(self.summary(value), handle, indent=2)

    def capture(self, x, timestep, context, transformer_options, payload):
        """Capture effective layout after TaoMate position changes, before H3 runs."""
        branch = tuple(transformer_options.get("cond_or_uncond", ()))
        if branch in self.seen:
            return
        self.seen.add(branch)
        video, audio = x[:2]
        signature = (context.shape[1], video.shape[2], (video.shape[-2] + 1) // 2 * 2, (video.shape[-1] + 1) // 2 * 2, audio.shape[-1])
        layout = payload.get("layout")
        if layout is None or layout.signature != signature:
            layout = h3.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        name = "model-branch-" + ("-".join(str(item) for item in branch) or "default")
        self.save(name, {"x": x, "timestep": timestep, "context": context, "payload": {key: value for key, value in payload.items() if key != "layout"}, "segments": layout.segments, "position_ids": layout.position_ids, "branch": branch, "transformer_options": transformer_options, "cpu_rng_state": torch.get_rng_state()})
        logging.info("H3 diagnostic %s: %s; segments=%s; references=%d; keyframes=%d", self.mode, name, layout.segments, len(payload.get("refs", [])), len(payload.get("keyframes", [])))

    def forward(self, executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        """Observe the native Masked AV model input without changing its arguments."""
        self.capture(x, timestep, context, transformer_options, minimax_payload or {})
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    def routing(self, branch, tags, commit_mask, history_tokens, condition_attends_current_av=False):
        """Record actual first-layer streaming row masks without saving QKV tensors."""
        if branch in self.routing_seen:
            return
        self.routing_seen.add(branch)
        name = "routing-branch-" + ("-".join(str(item) for item in branch) or "default")
        condition_rule = "Condition queries attend all current rows, without historical KV" if condition_attends_current_av else "Condition queries attend condition rows only"
        self.save(name, {"token_tags": tags, "media_mask": commit_mask, "condition_mask": ~commit_mask, "history_tokens": history_tokens, "condition_attends_current_av": condition_attends_current_av, "rule": condition_rule + "; media queries attend all condition rows, cached history, and current media rows."})
