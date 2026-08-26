from .nodes import HREndlessSampler
from .preview import HREndlessSamplerPreview

__version__ = "0.9.0"


NODE_CLASS_MAPPINGS = {
    "HREndlessSampler": HREndlessSampler,
    "HREndlessSamplerPreview": HREndlessSamplerPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HREndlessSampler": "HR Endless Sampler",
    "HREndlessSamplerPreview": "HR Endless Sampler Preview",
}

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
