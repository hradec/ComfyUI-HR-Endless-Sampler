from .nodes import HREndlessSampler
from .python.selective_lora import HREndlessSamplerSelectiveLora
from .python.preproduction import HREndlessGemmaPreProduction, HREndlessLegacyChunkPrompts, HREndlessLegacyPromptBake
from .preview import HREndlessSamplerPreview
from .video_io import HREndlessSamplerLoadVideo, HREndlessSamplerSaveVideo, HREndlessSamplerVideoCompare

__version__ = "0.9.0"


NODE_CLASS_MAPPINGS = {
    "HREndlessSamplerSelectiveLora": HREndlessSamplerSelectiveLora,
    "HREndlessLegacyChunkPrompts": HREndlessLegacyChunkPrompts,
    "HREndlessLegacyPromptBake": HREndlessLegacyPromptBake,
    "HREndlessGemmaPreProduction": HREndlessGemmaPreProduction,
    "HREndlessSampler": HREndlessSampler,
    "HREndlessSamplerPreview": HREndlessSamplerPreview,
    "HREndlessSamplerSaveVideo": HREndlessSamplerSaveVideo,
    "HREndlessSamplerLoadVideo": HREndlessSamplerLoadVideo,
    "HREndlessSamplerVideoCompare": HREndlessSamplerVideoCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HREndlessSamplerSelectiveLora": "HR Endless Sampler Selective LoRA",
    "HREndlessLegacyChunkPrompts": "HR Endless Pre-production — Legacy Chunk Prompts",
    "HREndlessGemmaPreProduction": "HR Endless Pre-production — Gemma 4",
    "HREndlessSampler": "HR Endless Sampler",
    "HREndlessSamplerPreview": "HR Endless Sampler Preview",
    "HREndlessSamplerSaveVideo": "HR Endless Sampler Save Video",
    "HREndlessSamplerLoadVideo": "HR Endless Sampler Load Video",
    "HREndlessSamplerVideoCompare": "HR Endless Sampler Video Compare",
}

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
