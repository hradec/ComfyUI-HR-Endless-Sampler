"""Callable pre-production providers; the sampler owns when they are invoked.

A provider creates a fresh session per sampler execution. Sessions implement
plan_timing(request, progress_callback), direct(request, frames, progress_callback)
and materialize_preproduction_cache(request, plan, progress_callback), using the
existing structured plan/prompt results. Runtime tensors stay in render_context,
never in the JSON request sent to a disposable inference worker.
"""

from comfy_api.latest import io

from ..gemma4 import Gemma4ContinuityDirector


HRPreProduction = io.Custom("HR_PRE_PRODUCTION")
ENABLE_GEMMA4_MTP = False


class GemmaPreProduction:
    """Reusable node configuration, not a cached live model or chat session."""

    def __init__(self, cache_gemma_preproduction=False, gemma4_mtp=False, production_think_budget=2048, chunk_think_budget=2048):
        """Store validated settings independently from any sampler execution."""
        self.production_think_budget = int(production_think_budget)
        self.chunk_think_budget = int(chunk_think_budget)
        if min(self.production_think_budget, self.chunk_think_budget) < 0 or max(self.production_think_budget, self.chunk_think_budget) > 32768:
            raise ValueError("Thinking budgets must be between 0 and 32768")
        self.cache_gemma_preproduction = bool(cache_gemma_preproduction)
        self.gemma4_mtp = bool(gemma4_mtp and ENABLE_GEMMA4_MTP)

    def fingerprint(self):
        """Do not reuse rendered chunks across different director settings."""
        return {"provider": "gemma4", **vars(self)}

    def create_session(self, *, render_context, debug=False, seed=0, observation_image_directory=None):
        """Create the callable inference owner after the sampler knows its layout."""
        session = Gemma4ContinuityDirector(debug=debug, gemma4_mtp=self.gemma4_mtp, seed=seed, observation_image_directory=observation_image_directory, production_think_budget=self.production_think_budget, chunk_think_budget=self.chunk_think_budget)
        # Shared references, not tensor copies; the worker still receives only
        # its explicit request and selected chronological image data.
        session.render_context = render_context
        return session


class HREndlessGemmaPreProduction(io.ComfyNode):
    """Connect a Gemma director to the sampler without loading it upstream."""

    @classmethod
    def define_schema(cls):
        """Expose Gemma-only controls on the provider node."""
        return io.Schema(node_id="HREndlessGemmaPreProduction", display_name="HR Endless Pre-production — Gemma 4", category="sampling/custom_sampling", inputs=[io.Boolean.Input("cache_gemma_preproduction", default=False, tooltip="Restore clean pre-production KV memory for each chunk."), io.Boolean.Input("gemma4_mtp", default=False, tooltip="Experimental MTP; currently disabled in code due to runtime failures and poor performance."), io.Int.Input("production_think_budget", default=2048, min=0, max=32768, step=1, tooltip="Reasoning tokens per production turn; 0 disables thinking."), io.Int.Input("chunk_think_budget", default=2048, min=0, max=32768, step=1, tooltip="Reasoning tokens per chunk turn; 0 disables thinking.")], outputs=[HRPreProduction.Output(display_name="pre_production")])

    @classmethod
    def execute(cls, cache_gemma_preproduction=False, gemma4_mtp=False, production_think_budget=2048, chunk_think_budget=2048):
        """Return a director configuration; inference occurs on sampler calls."""
        return io.NodeOutput(GemmaPreProduction(cache_gemma_preproduction, gemma4_mtp, production_think_budget, chunk_think_budget))


class LegacyChunkPrompts:
    """Editable, numbered full H3 prompts with strict delimiter validation."""

    def __init__(self, text):
        """Keep text serializable; validate when rendering rather than creating the node."""
        self.text = text

    def fingerprint(self):
        """Allow prompt edits while retaining explicitly selected replay chunks."""
        return {"provider": "legacy-chunk-prompts"}

    def create_session(self, **kwargs):
        """Implement the common provider contract without starting a model."""
        return self

    def prompts(self):
        """Reject missing, duplicate, unordered or empty chunk sections."""
        import re
        parts = re.split(r"(?m)^--8<--\[ Chunk ([1-9][0-9]*) \]--8<--------8<---------8<--------[ \t]*\r?$", self.text)
        if parts[0].strip() or len(parts) < 3:
            raise ValueError("Generate chunk prompts first, or use the exact Chunk N delimiter.")
        result = []
        for index in range(1, len(parts), 2):
            if int(parts[index]) != len(result) + 1 or not parts[index + 1].strip():
                raise ValueError("Chunk prompts must be nonempty and numbered consecutively from Chunk 1.")
            result.append(parts[index + 1].strip())
        return result

    def get_chunk_prompt(self, number):
        """Return the full user-authored prompt without its separator."""
        prompts = self.prompts()
        if number < 1 or number > len(prompts):
            raise ValueError("No manual prompt for Chunk %d; regenerate prompts for the current duration." % number)
        return prompts[number - 1]

    @staticmethod
    def format(prompts):
        """Round-trip full prompts through the user-requested editor format."""
        return "\n\n".join("--8<--[ Chunk %d ]--8<--------8<---------8<--------\n%s" % (index + 1, prompt) for index, prompt in enumerate(prompts))


class HREndlessLegacyChunkPrompts(io.ComfyNode):
    """Prepare editable legacy prompts for a connected Endless sampler."""

    @classmethod
    def define_schema(cls):
        """Expose one persistent multiline editor and the common provider output."""
        return io.Schema(node_id="HREndlessLegacyChunkPrompts", display_name="HR Endless Pre-production — Legacy Chunk Prompts", category="sampling/custom_sampling", inputs=[io.String.Input("chunk_prompts", default="", multiline=True)], outputs=[HRPreProduction.Output(display_name="pre_production")])

    @classmethod
    def execute(cls, chunk_prompts=""):
        """Return the editor contents without loading any inference model."""
        return io.NodeOutput(LegacyChunkPrompts(chunk_prompts))


class HREndlessLegacyPromptBake(io.ComfyNode):
    """Internal output used by the editor's prompt-only queue action."""

    @classmethod
    def define_schema(cls):
        """Resolve real graph inputs instead of guessing upstream widget values."""
        return io.Schema(node_id="HREndlessLegacyPromptBake", category="sampling/custom_sampling", is_dev_only=True, is_output_node=True, inputs=[io.Int.Input("length", default=124, min=5), io.String.Input("reference_kinds", default="[]"), io.String.Input("prompt", force_input=True), io.Float.Input("fps", default=24.0), io.Int.Input("chunk_frames", default=124), io.Int.Input("video_continuation", default=22), io.String.Input("video_continuation_method"), io.String.Input("target_node"), io.String.Input("request_id")], outputs=[])

    @classmethod
    def execute(cls, length, reference_kinds, prompt, fps, chunk_frames, video_continuation, video_continuation_method, target_node, request_id):
        """Run the native planner and send complete editable prompts to the browser."""
        from .. import nodes
        from server import PromptServer
        import json
        kinds = json.loads(reference_kinds)
        if not isinstance(kinds, list) or any(kind not in ("image", "video", "video_audio", "audio") for kind in kinds):
            raise ValueError("Invalid H3 reference metadata")
        # Match the native empty H3 node's upward temporal-grid rounding.
        length = max(5, int(length))
        length += (5 - length) % 17
        video_t, audio_t = nodes._video_steps(length), nodes._audio_steps(length)
        maximum = chunk_frames - (chunk_frames - 5) % 17
        _, _, duration, _ = nodes._continuation_controls(0, 0, video_continuation, maximum)
        taomate = video_continuation_method == nodes.VIDEO_CONTINUATION_METHOD_TAOMATE
        masked = video_continuation_method == nodes.VIDEO_CONTINUATION_METHOD_MASKED_AV
        if masked and duration >= maximum:
            raise ValueError("video_continuation must be smaller than chunk_frames")
        physical = masked and nodes.ENABLE_MASKED_AV_OVERLAP
        plan = nodes._chunk_plan(video_t, audio_t, chunk_frames, duration) if physical else nodes._chunk_plan_without_overlap(video_t, audio_t, chunk_frames, duration if masked else 5)
        if taomate:
            from .taomate import TaoMateStreaming
            plan = TaoMateStreaming.request_plan(video_t, audio_t, chunk_frames)
        refs = [{"kind": kind} for kind in kinds]
        video_number = 1 + sum(ref["kind"] in ("video", "video_audio") for ref in refs)
        audio_number = 1 + sum(ref["kind"] in ("audio", "video_audio") for ref in refs)
        picture_number = 1 + sum(ref["kind"] == "image" for ref in refs)
        video_ref = duration > 0 and not masked and not taomate and nodes.INCLUDE_VIDEO1_REFERENCE
        audio_ref = duration > 0 and not masked and not taomate
        planned = nodes._planned_chunk_prompts(prompt, plan, plan, fps, duration if physical else 0, video_ref, audio_ref, bool(refs), video_number, audio_number, legacy=True, taomate=taomate)
        prompts = []
        for index, (local, _) in enumerate(planned):
            continuation = index > 0
            prompts.append(nodes._chunk_summary_prompt(local, prompt, continuation, picture_label="<Picture %d>" % picture_number if continuation and video_ref else None, video_label="<Video %d>" % video_number if continuation and video_ref else None, audio_label="<Audio %d>" % audio_number if continuation and audio_ref else None, boundary_keyframe=continuation and masked))
        PromptServer.instance.send_sync("hr_endless_legacy_prompts", {"node_id": target_node, "request_id": request_id, "text": LegacyChunkPrompts.format(prompts)})
        return io.NodeOutput()
