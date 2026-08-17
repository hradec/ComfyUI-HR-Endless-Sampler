from .nodes import MiniMaxH3SamplerCustomAdvancedUnlimited


NODE_CLASS_MAPPINGS = {
    "SamplerCustomAdvanced-Unlimited": MiniMaxH3SamplerCustomAdvancedUnlimited,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerCustomAdvanced-Unlimited": "SamplerCustomAdvanced-Unlimited",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
