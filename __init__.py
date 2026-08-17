from .nodes import MiniMaxH3SamplerCustomAdvancedUnlimited
from .preview import MiniMaxH3UnlimitedPreview


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvanced-Unlimited": MiniMaxH3SamplerCustomAdvancedUnlimited,
    "MiniMaxH3UnlimitedPreview": MiniMaxH3UnlimitedPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvanced-Unlimited": "SamplerCustomAdvanced-Unlimited",
    "MiniMaxH3UnlimitedPreview": "MiniMax H3 Unlimited Preview",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
