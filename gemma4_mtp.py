"""Single-sequence Gemma 4 MTP support for the disposable director worker.

This is a deliberately small adaptation of llama-cpp-python's reference MTP
provider from ``examples/server/server.py``.  The reference server also owns a
multi-request scheduler, sequence cloning, metrics, and several speculative
decoding modes.  HR Endless Sampler runs exactly one Gemma sequence in a
short-lived process, so only the linked target/draft context path is needed.

Reference implementation reviewed at:
https://github.com/abetlen/llama-cpp-python/blob/3691546f1c9e0c1bf93323dff02230bd959cf562/examples/server/server.py
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np


GEMMA4_MTP_DRAFT_TOKENS = 4


class Gemma4MTPError(RuntimeError):
    """Raised when the installed llama.cpp runtime cannot create the MTP path."""


class Gemma4MTPDraft:
    """A native Gemma MTP drafter attached to one high-level ``Llama`` object.

    llama-cpp-python's normal ``Llama.generate`` loop already knows how to
    target-verify a Python draft callback.  Native MTP additionally needs the
    hidden state produced by every target decode.  This adapter enables that
    output, feeds it to the linked MTP context, and hooks ``Llama.eval`` so the
    provider sees each physical target batch.

    The target remains the sole authority. Gemma 4 is a hybrid model, so a
    rejected draft cannot be repaired with llama-cpp-python's ordinary partial
    KV removal alone. Before every verification batch this adapter checkpoints
    the target's partial (SWA/recurrent) state. On rejection it restores that
    state, removes the attention suffix, and replays only the accepted target
    tokens. This mirrors llama.cpp's current speculative-simple rollback path.
    """

    def __init__(
        self,
        target: Any,
        draft_model_path: str | Path,
        *,
        num_pred_tokens: int = GEMMA4_MTP_DRAFT_TOKENS,
    ) -> None:
        try:
            from llama_cpp import llama_cpp, llama_cpp_ext
        except (ImportError, AttributeError) as error:
            raise Gemma4MTPError(
                "llama-cpp-python does not expose the native MTP bindings"
            ) from error

        required = (
            "LLAMA_CONTEXT_TYPE_MTP",
            "llama_model_n_layer_nextn",
            "llama_state_seq_get_size_ext",
            "llama_state_seq_get_data_ext",
            "llama_state_seq_set_data_ext",
            "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY",
            "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE",
        )
        missing = [name for name in required if not hasattr(llama_cpp, name)]
        extension_required = (
            "llama_get_ctx_other",
            "llama_get_embeddings_nextn",
            "llama_get_embeddings_nextn_ith",
            "llama_set_embeddings_nextn",
        )
        missing.extend(name for name in extension_required if not hasattr(llama_cpp_ext, name))
        if missing:
            raise Gemma4MTPError(
                "llama-cpp-python is missing native MTP symbol(s): " + ", ".join(missing)
            )
        if not hasattr(target, "_ctx") or not hasattr(target, "_model"):
            raise Gemma4MTPError("Gemma target is not a compatible llama-cpp-python Llama instance")

        self._llama_cpp = llama_cpp
        self._llama_cpp_ext = llama_cpp_ext
        self.target = target
        self.target_ctx = target._ctx.ctx
        self.num_pred_tokens = max(1, int(num_pred_tokens))
        self.n_embd = int(llama_cpp.llama_model_n_embd(target._model.model))
        self.n_vocab = int(target.n_vocab())

        self.model = None
        self.ctx = None
        self.batch = None
        self._batch_tokens = None
        self._sampler = None
        self._closed = False
        self._ready = False
        self._ready_pos = 0
        self._pending_h = np.zeros(self.n_embd, dtype=np.float32)
        self._verify_h = np.empty((0, self.n_embd), dtype=np.float32)
        self._verify_pos: list[int] = []
        self._proposal_base: int | None = None
        self._proposal_count = 0
        self._proposal_tokens: list[int] = []
        self._proposal_verified_end: int | None = None
        self._proposal_sampled: list[int] = []
        self._checkpoint_data: Any | None = None
        self._checkpoint_size = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self.proposal_batches = 0
        self.draft_decode_calls = 0
        self.draft_decode_seconds = 0.0
        self.target_verify_seconds = 0.0
        self.target_verify_calls = 0
        self.target_verify_rows = 0
        self.target_replay_seconds = 0.0
        self.target_replay_calls = 0
        self.target_replay_rows = 0
        self.target_sample_seconds = 0.0
        self.target_sample_calls = 0
        self.target_sync_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.rollback_count = 0
        self.replayed_tokens = 0

        self._original_eval = target.eval
        self._original_generate = target.generate
        self._original_sample = getattr(target, "sample", None)
        self._original_kv_cache_seq_rm = target._ctx.kv_cache_seq_rm
        self._original_draft_model = getattr(target, "draft_model", None)
        self._original_logits_all = bool(getattr(target, "_logits_all", False))

        try:
            self._load_draft_model(Path(draft_model_path))
            llama_cpp_ext.llama_set_embeddings_nextn(self.target_ctx, True, False)
            llama_cpp_ext.llama_set_embeddings_nextn(self.ctx, True, True)
            target._logits_all = True
            target.draft_model = self
            target.eval = types.MethodType(self._eval_target, target)
            target.generate = types.MethodType(self._generate_target, target)
            if callable(self._original_sample):
                target.sample = types.MethodType(self._sample_target, target)
            target._ctx.kv_cache_seq_rm = types.MethodType(
                self._kv_cache_seq_rm, target._ctx
            )
            # The callback was registered after Llama's target context/model,
            # so ExitStack closes the MTP context first.
            target._stack.callback(self.close)
        except BaseException:
            self.close()
            raise

    def _load_draft_model(self, path: Path) -> None:
        llama_cpp = self._llama_cpp
        if not path.is_file():
            raise Gemma4MTPError(f"Gemma MTP assistant model does not exist: {path}")

        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = 0x7FFFFFFF
        model_params.load_mtp = True
        target_model_params = getattr(self.target, "model_params", None)
        if target_model_params is not None:
            model_params.split_mode = target_model_params.split_mode
            model_params.main_gpu = target_model_params.main_gpu
            model_params.load_mode = target_model_params.load_mode

        self.model = llama_cpp.llama_model_load_from_file(
            str(path).encode("utf-8"), model_params
        )
        if self.model is None:
            raise Gemma4MTPError(f"llama.cpp could not load Gemma MTP assistant: {path}")

        draft_embd = int(llama_cpp.llama_model_n_embd_out(self.model))
        if draft_embd <= 0:
            draft_embd = int(llama_cpp.llama_model_n_embd(self.model))
        if draft_embd != self.n_embd:
            raise Gemma4MTPError(
                "Gemma MTP assistant embedding size does not match the target "
                f"({draft_embd} != {self.n_embd})"
            )

        target_params = self.target.context_params
        context_params = llama_cpp.llama_context_default_params()
        context_params.n_ctx = int(self.target.n_ctx())
        context_params.n_batch = int(self.target.n_batch)
        context_params.n_ubatch = int(target_params.n_ubatch)
        context_params.n_seq_max = 1
        context_params.n_rs_seq = 0
        context_params.n_outputs_max = 1
        context_params.n_outputs_max_per_seq = 1
        context_params.n_threads = int(target_params.n_threads)
        context_params.n_threads_batch = int(target_params.n_threads_batch)
        context_params.ctx_type = llama_cpp.LLAMA_CONTEXT_TYPE_MTP
        context_params.rope_scaling_type = target_params.rope_scaling_type
        context_params.pooling_type = target_params.pooling_type
        context_params.attention_type = target_params.attention_type
        context_params.flash_attn_type = target_params.flash_attn_type
        context_params.rope_freq_base = target_params.rope_freq_base
        context_params.rope_freq_scale = target_params.rope_freq_scale
        context_params.yarn_ext_factor = target_params.yarn_ext_factor
        context_params.yarn_attn_factor = target_params.yarn_attn_factor
        context_params.yarn_beta_fast = target_params.yarn_beta_fast
        context_params.yarn_beta_slow = target_params.yarn_beta_slow
        context_params.yarn_orig_ctx = target_params.yarn_orig_ctx
        context_params.type_k = target_params.type_k
        context_params.type_v = target_params.type_v
        context_params.embeddings = False
        context_params.offload_kqv = target_params.offload_kqv
        context_params.no_perf = target_params.no_perf
        context_params.op_offload = target_params.op_offload
        context_params.swa_full = target_params.swa_full
        context_params.kv_unified = target_params.kv_unified
        context_params.ctx_other = self.target_ctx

        self.ctx = llama_cpp.llama_init_from_model(self.model, context_params)
        if self.ctx is None:
            raise Gemma4MTPError("llama.cpp could not create the linked Gemma MTP context")
        linked = self._llama_cpp_ext.llama_get_ctx_other(self.ctx)
        if not linked or linked != self.target_ctx:
            raise Gemma4MTPError("Gemma MTP context did not link to the target context")

        n_batch = int(llama_cpp.llama_n_batch(self.ctx))
        if n_batch < self.num_pred_tokens + 1:
            raise Gemma4MTPError(
                f"Gemma MTP requires a batch of at least {self.num_pred_tokens + 1}; got {n_batch}"
            )
        self.batch = llama_cpp.llama_batch_init(n_batch, self.n_embd, 1)
        self._batch_tokens = (llama_cpp.llama_token * n_batch)()
        self.batch.token = self._batch_tokens
        self._batch_embeddings = np.ctypeslib.as_array(
            self.batch.embd, shape=(n_batch * self.n_embd,)
        )

        sampler_params = llama_cpp.llama_sampler_chain_default_params()
        sampler_params.no_perf = True
        self._sampler = llama_cpp.llama_sampler_chain_init(sampler_params)
        llama_cpp.llama_sampler_chain_add(
            self._sampler, llama_cpp.llama_sampler_init_greedy()
        )

    def _eval_target(self, target: Any, tokens: Sequence[int]) -> None:
        """Replacement for ``Llama.eval`` that also captures MTP hidden rows."""
        self._original_kv_cache_seq_rm(-1, target.n_tokens, -1)
        for index in range(0, len(tokens), target.n_batch):
            physical = tokens[index : min(len(tokens), index + target.n_batch)]
            n_past = int(target.n_tokens)
            n_tokens = len(physical)
            target._batch.set_batch(
                batch=physical,
                n_past=n_past,
                # Every proposed token needs target logits for verification.
                logits_all=True,
            )
            decode_started = time.perf_counter()
            target._ctx.decode(target._batch)
            decode_elapsed = time.perf_counter() - decode_started
            if self._proposal_base is not None:
                self.target_verify_seconds += decode_elapsed
                self.target_verify_calls += 1
                self.target_verify_rows += n_tokens
            self._process_target_batch(target._batch.batch)
            target.input_ids[n_past : n_past + n_tokens] = physical
            target.n_tokens += n_tokens
            target._requires_eval = False
        if self._proposal_base is not None:
            self._proposal_verified_end = int(target.n_tokens)

    def _process_target_batch(self, batch: Any) -> None:
        n_tokens = int(batch.n_tokens)
        if n_tokens <= 0:
            return
        embeddings = self._llama_cpp_ext.llama_get_embeddings_nextn(self.target_ctx)
        if not embeddings:
            raise Gemma4MTPError("llama.cpp returned no target NextN embeddings for Gemma MTP")
        rows = np.ctypeslib.as_array(
            embeddings, shape=(n_tokens * self.n_embd,)
        ).reshape(n_tokens, self.n_embd)
        positions = [int(batch.pos[row]) for row in range(n_tokens)]
        self._verify_h = rows.copy()
        self._verify_pos = positions
        self._pending_h[:] = rows[-1]
        self._ready = True
        self._ready_pos = positions[-1] + 1

    def _select_pending_for_target_length(self, target_length: int) -> None:
        if target_length <= 0:
            self._ready = False
            self._ready_pos = 0
            self._verify_h = np.empty((0, self.n_embd), dtype=np.float32)
            self._verify_pos = []
            return
        if self._ready and self._ready_pos == target_length:
            return
        try:
            row = self._verify_pos.index(target_length - 1)
        except ValueError:
            self._ready = False
            return
        self._pending_h[:] = self._verify_h[row]
        self._ready = True
        self._ready_pos = target_length

    def _save_target_checkpoint(self, target_length: int) -> None:
        checkpoint_started = time.perf_counter()
        # Keep the fast checkpoint on the device. This native path can abort
        # inside llama.cpp on some requests, so it is deliberately confined to
        # the disposable worker. The parent retries that exact operation with
        # MTP disabled without changing later operations.
        flags = (
            self._llama_cpp.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
            | self._llama_cpp.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
        )
        size = int(
            self._llama_cpp.llama_state_seq_get_size_ext(
                self.target_ctx, 0, flags
            )
        )
        if size <= 0:
            raise Gemma4MTPError(
                "llama.cpp returned an empty Gemma partial-state checkpoint"
            )
        checkpoint = (ctypes.c_uint8 * size)()
        copied = int(
            self._llama_cpp.llama_state_seq_get_data_ext(
                self.target_ctx, checkpoint, size, 0, flags
            )
        )
        if copied != size:
            raise Gemma4MTPError(
                "Gemma partial-state checkpoint size mismatch "
                f"(expected {size}, copied {copied})"
            )
        self._checkpoint_data = checkpoint
        self._checkpoint_size = size
        self._proposal_base = target_length
        self._proposal_verified_end = None
        self._proposal_sampled = []
        self.checkpoint_seconds += time.perf_counter() - checkpoint_started

    def _clear_proposal(self) -> None:
        self._proposal_base = None
        self._proposal_count = 0
        self._proposal_tokens = []
        self._proposal_verified_end = None
        self._proposal_sampled = []
        self._checkpoint_data = None
        self._checkpoint_size = 0

    def _record_previous_acceptance(self, accepted: int) -> None:
        if self._proposal_base is None:
            return
        self.accepted_tokens += max(0, min(self._proposal_count, accepted))

    def _restore_partial_proposal(self, target_length: int) -> None:
        """Restore a rejected target batch and replay its accepted prefix."""
        if self._proposal_base is None or self._checkpoint_data is None:
            return
        base = self._proposal_base
        accepted = max(0, min(self._proposal_count, target_length - base - 1))
        replay_end = base + 1 + accepted
        replay_tokens = self.target.input_ids[base:replay_end].astype(
            np.intc, copy=True
        )
        flags = (
            self._llama_cpp.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
            | self._llama_cpp.LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
        )
        restored = int(
            self._llama_cpp.llama_state_seq_set_data_ext(
                self.target_ctx,
                self._checkpoint_data,
                self._checkpoint_size,
                0,
                flags,
            )
        )
        if restored != self._checkpoint_size:
            raise Gemma4MTPError(
                "Gemma partial-state restore size mismatch "
                f"(expected {self._checkpoint_size}, restored {restored})"
            )

        # Restoring PARTIAL_ONLY repairs Gemma's SWA/recurrent state. This
        # second operation removes the speculative suffix from its ordinary
        # attention memory, just as llama.cpp's checkpoint path does.
        self._original_kv_cache_seq_rm(-1, base, -1)
        self.target.n_tokens = base
        self.target._requires_eval = True
        self._record_previous_acceptance(accepted)
        self._clear_proposal()
        if replay_tokens.size:
            self.rollback_count += 1
            self.replayed_tokens += int(replay_tokens.size)
            self._decode_target_tokens(replay_tokens.tolist())

    def _decode_target_tokens(self, tokens: Sequence[int]) -> None:
        """Decode known-good physical tokens without speculative bookkeeping."""
        target = self.target
        for index in range(0, len(tokens), target.n_batch):
            physical = tokens[index : min(len(tokens), index + target.n_batch)]
            n_past = int(target.n_tokens)
            n_tokens = len(physical)
            target._batch.set_batch(
                batch=physical,
                n_past=n_past,
                logits_all=True,
            )
            decode_started = time.perf_counter()
            target._ctx.decode(target._batch)
            self.target_replay_seconds += time.perf_counter() - decode_started
            self.target_replay_calls += 1
            self.target_replay_rows += n_tokens
            self._process_target_batch(target._batch.batch)
            target.input_ids[n_past : n_past + n_tokens] = physical
            target.n_tokens += n_tokens
            target._requires_eval = False

    def _finalize_full_proposal(self) -> None:
        if self._proposal_base is None:
            return
        accepted = self._proposal_count
        target_length = self._proposal_base + 1 + accepted
        self._record_previous_acceptance(accepted)
        self._select_pending_for_target_length(target_length)
        self._clear_proposal()

    def _kv_cache_seq_rm(
        self, _context: Any, seq_id: int, p0: int, p1: int
    ) -> bool:
        """Intercept llama-cpp-python's unsafe hybrid speculative truncate."""
        if (
            self._proposal_base is not None
            and self._proposal_verified_end is not None
            and seq_id < 1
            and p1 < 0
            and self._proposal_base < p0 < self._proposal_verified_end
        ):
            self._restore_partial_proposal(int(p0))
            return True
        return bool(self._original_kv_cache_seq_rm(seq_id, p0, p1))

    def _note_generated_token(self, token: int) -> None:
        if self._proposal_verified_end is not None:
            self._proposal_sampled.append(int(token))

    def _finish_generation(self) -> None:
        """Leave no unaccepted speculative state when a consumer stops early."""
        if self._proposal_base is None:
            return
        if self._proposal_verified_end is None:
            self._clear_proposal()
            return
        accepted = 0
        for actual, drafted in zip(self._proposal_sampled, self._proposal_tokens):
            if actual != drafted:
                break
            accepted += 1
        if accepted >= self._proposal_count and len(self._proposal_sampled) > accepted:
            self._finalize_full_proposal()
            return
        self._restore_partial_proposal(self._proposal_base + 1 + accepted)

    def _generate_target(self, _target: Any, *args: Any, **kwargs: Any):
        """Forward ``Generator.send`` while tracking early completion stops."""
        generator = self._original_generate(*args, **kwargs)
        send_value = None
        try:
            while True:
                try:
                    token = (
                        next(generator)
                        if send_value is None
                        else generator.send(send_value)
                    )
                except StopIteration:
                    return
                self._note_generated_token(int(token))
                send_value = yield token
        finally:
            try:
                generator.close()
            finally:
                self._finish_generation()

    def _sample_target(self, _target: Any, *args: Any, **kwargs: Any) -> int:
        """Split deferred CUDA synchronization from target sampler CPU work."""
        sync_started = time.perf_counter()
        self._llama_cpp.llama_synchronize(self.target_ctx)
        self.target_sync_seconds += time.perf_counter() - sync_started
        started = time.perf_counter()
        try:
            return int(self._original_sample(*args, **kwargs))
        finally:
            self.target_sample_seconds += time.perf_counter() - started
            self.target_sample_calls += 1

    def __call__(self, input_ids: np.ndarray, /, **_kwargs: Any) -> np.ndarray:
        target_length = int(input_ids.size) - 1
        if self._proposal_base is not None:
            accepted = target_length - self._proposal_base - 1
            if accepted >= self._proposal_count:
                self._finalize_full_proposal()
            elif self._proposal_verified_end is not None:
                self._restore_partial_proposal(target_length)
        if (
            self._closed
            or not self._ready
            or input_ids.size == 0
            or self._ready_pos != target_length
        ):
            return np.array([], dtype=np.intc)

        first_pos = target_length
        token = int(input_ids[-1])
        drafted: list[int] = []
        self._llama_cpp.llama_sampler_reset(self._sampler)

        for _ in range(self.num_pred_tokens):
            self.batch.n_tokens = 1
            self.batch.token[0] = token
            self.batch.pos[0] = first_pos
            self.batch.seq_id[0][0] = 0
            self.batch.n_seq_id[0] = 1
            self.batch.logits[0] = True
            self._batch_embeddings[: self.n_embd] = self._pending_h
            decode_started = time.perf_counter()
            result = int(self._llama_cpp.llama_decode(self.ctx, self.batch))
            self.draft_decode_seconds += time.perf_counter() - decode_started
            self.draft_decode_calls += 1
            if result != 0:
                break
            token = int(self._llama_cpp.llama_sampler_sample(self._sampler, self.ctx, 0))
            if token == self._llama_cpp.LLAMA_TOKEN_NULL:
                break
            drafted.append(token)
            if len(drafted) >= self.num_pred_tokens:
                break
            next_embedding = self._llama_cpp_ext.llama_get_embeddings_nextn_ith(self.ctx, 0)
            if not next_embedding:
                break
            self._pending_h[:] = np.ctypeslib.as_array(
                next_embedding, shape=(self.n_embd,)
            )

        if not drafted:
            return np.array([], dtype=np.intc)
        self.proposal_batches += 1
        self.proposed_tokens += len(drafted)
        self._proposal_count = len(drafted)
        self._proposal_tokens = list(drafted)
        # Save after drafting, matching llama.cpp: the linked Gemma assistant
        # has finished proposing, but the target has not verified the batch.
        self._save_target_checkpoint(target_length)
        return np.asarray(drafted, dtype=np.intc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        llama_cpp = getattr(self, "_llama_cpp", None)
        llama_cpp_ext = getattr(self, "_llama_cpp_ext", None)
        target = getattr(self, "target", None)
        if target is not None:
            self._finish_generation()
        if target is not None:
            target.eval = self._original_eval
            target.generate = self._original_generate
            if callable(self._original_sample):
                target.sample = self._original_sample
            target.draft_model = self._original_draft_model
            target._logits_all = self._original_logits_all
            target._ctx.kv_cache_seq_rm = self._original_kv_cache_seq_rm
        if llama_cpp_ext is not None and getattr(self, "target_ctx", None):
            try:
                llama_cpp_ext.llama_set_embeddings_nextn(self.target_ctx, False, False)
            except (AttributeError, RuntimeError):
                pass
        if llama_cpp is not None and self.batch is not None:
            self.batch.token = ctypes.POINTER(llama_cpp.llama_token)()
            llama_cpp.llama_batch_free(self.batch)
            self.batch = None
        if llama_cpp is not None and self._sampler is not None:
            llama_cpp.llama_sampler_free(self._sampler)
            self._sampler = None
        if llama_cpp is not None and self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None
        if llama_cpp is not None and self.model is not None:
            llama_cpp.llama_model_free(self.model)
            self.model = None
        if self.proposed_tokens:
            acceptance = 100.0 * self.accepted_tokens / self.proposed_tokens
            draft_rate = (
                self.proposed_tokens / self.draft_decode_seconds
                if self.draft_decode_seconds > 0.0
                else 0.0
            )
            # The worker's stderr is inherited by ComfyUI. Use it directly so
            # this comparison diagnostic cannot disappear behind the child
            # process logging level.
            print(
                "HR Endless Sampler Gemma 4 native MTP: "
                f"{self.accepted_tokens}/{self.proposed_tokens} draft tokens accepted "
                f"({acceptance:0.1f}%) across {self.proposal_batches} proposals; "
                f"assistant decode {self.draft_decode_seconds:0.3f}s / "
                f"{self.draft_decode_calls} calls ({draft_rate:0.1f} proposed tokens/sec); "
                f"target verify {self.target_verify_seconds:0.3f}s / "
                f"{self.target_verify_calls} calls / {self.target_verify_rows} rows; "
                f"target replay {self.target_replay_seconds:0.3f}s / "
                f"{self.target_replay_calls} calls / {self.target_replay_rows} rows; "
                f"target sync {self.target_sync_seconds:0.3f}s; "
                f"target sample {self.target_sample_seconds:0.3f}s / "
                f"{self.target_sample_calls} calls; "
                f"device checkpoints {self.checkpoint_seconds:0.3f}s, "
                f"{self.rollback_count} rollbacks / {self.replayed_tokens} replayed tokens.",
                file=sys.stderr,
                flush=True,
            )


def attach_gemma4_mtp(
    target: Any,
    draft_model_path: str | Path,
    *,
    num_pred_tokens: int = GEMMA4_MTP_DRAFT_TOKENS,
) -> Gemma4MTPDraft:
    """Attach native MTP to a live Gemma target and return its owner."""
    return Gemma4MTPDraft(
        target,
        draft_model_path,
        num_pred_tokens=num_pred_tokens,
    )


def create_native_mtp_llama(
    Llama: Any,
    *,
    model_path: str | Path,
    draft_model_path: str | Path,
    num_pred_tokens: int = GEMMA4_MTP_DRAFT_TOKENS,
    **llama_kwargs: Any,
) -> Any:
    """Create the target with llama.cpp's native MTP requirements from birth.

    llama-cpp-python 0.3.35 exposes the low-level fields used by native MTP,
    but its high-level ``Llama`` constructor does not expose ``load_mtp``.
    Intercept its default model parameters only for this constructor call so
    the target is loaded with the required NextN tensors. The target context
    deliberately keeps ``n_rs_seq=0``: current llama.cpp uses explicit partial
    state checkpoints for hybrid-model rollback, because Gemma does not support
    recurrent snapshot slots. This is done before either target object exists;
    attaching MTP tensors to an ordinary loaded model is not possible.
    """
    from llama_cpp import llama_cpp

    num_pred_tokens = max(1, int(num_pred_tokens))
    original_model_defaults = llama_cpp.llama_model_default_params
    original_context_defaults = llama_cpp.llama_context_default_params

    def mtp_model_defaults():
        params = original_model_defaults()
        params.load_mtp = True
        return params

    def mtp_context_defaults():
        params = original_context_defaults()
        params.n_rs_seq = 0
        # Match llama-cpp-python's advanced server default. The ordinary
        # high-level Llama constructor otherwise leaves this false.
        params.kv_unified = True
        return params

    llama_cpp.llama_model_default_params = mtp_model_defaults
    llama_cpp.llama_context_default_params = mtp_context_defaults
    try:
        target = Llama(
            model_path=str(model_path),
            logits_all=True,
            **llama_kwargs,
        )
    finally:
        llama_cpp.llama_model_default_params = original_model_defaults
        llama_cpp.llama_context_default_params = original_context_defaults

    try:
        owner = attach_gemma4_mtp(
            target,
            draft_model_path,
            num_pred_tokens=num_pred_tokens,
        )
        # Keep a public, inspectable owner for diagnostics and tests. The same
        # owner is also registered in Llama's ExitStack for ordered cleanup.
        target._hr_endless_mtp = owner
        return target
    except BaseException:
        target.close()
        raise
