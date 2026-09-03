from .director_config import HRQwen38DirectorConfig
from .nodes import HREndlessSampler
from .preview import HREndlessSamplerPreview
from .reference_set import HRMiniMaxH3ReferenceConditioning, HRMiniMaxH3ReferenceSet
from .storyboard import HRMiniMaxH3StoryboardPlanner
from .video_io import HREndlessSamplerLoadVideo, HREndlessSamplerSaveVideo

__version__ = "0.9.0"


NODE_CLASS_MAPPINGS = {
    "HREndlessSampler": HREndlessSampler,
    "HREndlessSamplerPreview": HREndlessSamplerPreview,
    "HREndlessSamplerSaveVideo": HREndlessSamplerSaveVideo,
    "HREndlessSamplerLoadVideo": HREndlessSamplerLoadVideo,
    "HRMiniMaxH3StoryboardPlanner": HRMiniMaxH3StoryboardPlanner,
    "HRQwen38DirectorConfig": HRQwen38DirectorConfig,
    "HRMiniMaxH3ReferenceSet": HRMiniMaxH3ReferenceSet,
    "HRMiniMaxH3ReferenceConditioning": HRMiniMaxH3ReferenceConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HREndlessSampler": "HR Endless Sampler",
    "HREndlessSamplerPreview": "HR Endless Sampler Preview",
    "HREndlessSamplerSaveVideo": "HR Endless Sampler Save Video",
    "HREndlessSamplerLoadVideo": "HR Endless Sampler Load Video",
    "HRMiniMaxH3StoryboardPlanner": "HR MiniMax H3 Storyboard Planner",
    "HRQwen38DirectorConfig": "HR Qwen3.8 Director Config",
    "HRMiniMaxH3ReferenceSet": "HR MiniMax H3 Reference Set",
    "HRMiniMaxH3ReferenceConditioning": "HR MiniMax H3 Reference Conditioning",
}

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
