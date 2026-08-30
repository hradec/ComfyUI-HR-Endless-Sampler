from .nodes import HREndlessSampler
from .preview import HREndlessSamplerPreview
from .video_io import HREndlessSamplerLoadVideo, HREndlessSamplerSaveVideo

__version__ = "0.9.0"


NODE_CLASS_MAPPINGS = {
    "HREndlessSampler": HREndlessSampler,
    "HREndlessSamplerPreview": HREndlessSamplerPreview,
    "HREndlessSamplerSaveVideo": HREndlessSamplerSaveVideo,
    "HREndlessSamplerLoadVideo": HREndlessSamplerLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HREndlessSampler": "HR Endless Sampler",
    "HREndlessSamplerPreview": "HR Endless Sampler Preview",
    "HREndlessSamplerSaveVideo": "HR Endless Sampler Save Video",
    "HREndlessSamplerLoadVideo": "HR Endless Sampler Load Video",
}

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
